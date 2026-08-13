from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch
import yaml

from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
    LingbotVLAConfig,
)
from lingbotvla.models.vla.lingbot_vla.tactile_vtla import (
    TactileTokenEncoder,
    TactileVTLAConfig,
    build_marker_region_mapping,
    continuous_sincos_1d,
    continuous_sincos_2d,
    load_vtla_checkpoint_state_dict,
)
from lingbotvla.tactile_contact import (
    CONTACT_HOLD,
    CONTACT_OFF,
    CONTACT_ON,
    CONTACT_UNKNOWN_HOLD,
    CONTACT_UNKNOWN_OFF,
    MarkerContactGate,
    build_marker_region_mapping_numpy,
    runtime_gate_config_from_mapping,
)
from tools.compute_marker_contact_stats import compute_contact_stats
from tools.generate_vtla_ablation_config import resolve_config


ROOT = Path(__file__).resolve().parents[1]


def _reference(num_sensors: int = 2) -> list[list[list[float]]]:
    grid = np.asarray(
        [[x, y] for y in np.linspace(0.1, 0.9, 6) for x in np.linspace(0.1, 0.9, 8)],
        dtype=np.float32,
    )
    references = [grid]
    if num_sensors > 1:
        # A different coordinate scale and marker order verifies per-sensor mapping.
        references.append((grid[::-1] * [2.0, 3.0] + [4.0, -2.0]).copy())
    return np.stack(references[:num_sensors]).tolist()


def _mapping(
    *,
    mode: str = "regional",
    num_sensors: int = 2,
    history_length: int = 8,
    temporal: str = "sincos",
    spatial: str = "sincos",
    sample_hz: float = 30.0,
    gate_mode: str = "none",
    gate_target: str = "marker_only",
) -> dict:
    gated = gate_mode != "none"
    return {
        "enabled": True,
        "num_sensors": num_sensors,
        "sensor_names": ["left", "right"][:num_sensors],
        "num_markers": 48,
        "use_rgb": True,
        "use_markers": True,
        "marker_history_length": history_length,
        "marker_sample_hz": sample_hz,
        "marker_input_features": 2,
        "marker_feature_mode": "displacement_history",
        "marker_hidden_dim": 32,
        "marker_mean": [0.0, 0.0],
        "marker_std": [1.0, 1.0],
        "marker_reference_xy": _reference(num_sensors),
        "marker_tokenization": {
            "mode": mode,
            "num_regions": {
                "global": 1,
                "regional": 4,
                "point_spatiotemporal": 48,
            }[mode],
            "region_layout": {
                "global": "1x1",
                "regional": "2x2",
                "point_spatiotemporal": "points",
            }[mode],
            "aggregation": "mean_max",
            "include_reference_xy": mode != "global",
            "point_hidden_dim": 16,
            "region_hidden_dim": 32,
        },
        "marker_position_encoding": {
            "temporal_type": temporal,
            "spatial_type": spatial,
            "use_real_time": True,
            "combination": "additive",
        },
        "marker_contact_gate": {
            "enabled": gated,
            "target": gate_target,
            "mode": gate_mode,
            "threshold_source": "explicit",
            "point_threshold": 0.4,
            "on_threshold": 0.5,
            "off_threshold": 0.2,
            "soft_threshold": 0.3,
            "topk_markers": 3,
            "min_active_markers": 2,
            "min_valid_markers_per_region": 2,
            "min_valid_markers_global": 24,
            "on_consecutive_frames": 2,
            "off_consecutive_frames": 2,
            "release_hold_frames": 3,
            "regional_soft_gate": True,
            "soft_gate_temperature": 0.1,
            "unknown_tracking_policy": "hold_previous",
            "max_unknown_hold_frames": 3,
        },
        "gate_tactile_rgb": gate_target == "marker_and_rgb" and gated,
        "rgb_keys": [f"rgb_{index}" for index in range(num_sensors)],
        "marker_displacement_keys": [f"disp_{index}" for index in range(num_sensors)],
        "marker_valid_mask_keys": [f"valid_{index}" for index in range(num_sensors)],
    }


def _inputs(settings: TactileVTLAConfig, batch: int = 2):
    shape = (
        batch,
        settings.num_sensors,
        settings.marker_history_length,
        settings.num_markers,
        2,
    )
    history = torch.full(shape, 0.7, dtype=torch.float32)
    valid = torch.ones(shape[:-1], dtype=torch.bool)
    history_valid = torch.ones(shape[:3], dtype=torch.bool)
    sensors = torch.ones(shape[:2], dtype=torch.bool)
    return history, valid, history_valid, sensors


@pytest.mark.parametrize(
    ("mode", "history_length", "tokens"),
    [
        ("global", 8, 8),
        ("regional", 8, 32),
        ("point_spatiotemporal", 4, 192),
    ],
)
def test_marker_tokenization_shapes(
    mode: str, history_length: int, tokens: int
):
    settings = TactileVTLAConfig.from_mapping(
        _mapping(mode=mode, history_length=history_length)
    )
    encoder = TactileTokenEncoder(settings, context_dim=17)
    inputs = _inputs(settings)
    output, mask = encoder.encode_markers(*inputs)
    assert output.shape == (2, 2, tokens, 17)
    assert mask.shape == (2, 2, tokens)
    assert output.flatten(1, 2).shape == (2, 2 * tokens, 17)


def test_point_spatiotemporal_tokens_preserve_point_mask_and_order():
    settings = TactileVTLAConfig.from_mapping(
        _mapping(
            mode="point_spatiotemporal",
            history_length=4,
            num_sensors=1,
            gate_mode="none",
        )
    )
    encoder = TactileTokenEncoder(settings, context_dim=16)
    history, valid, history_valid, sensors = _inputs(settings, batch=1)
    valid[0, 0, 1, 7] = False
    output = encoder.encode_markers(
        history,
        valid,
        history_valid,
        sensors,
        return_debug_info=True,
    )

    assert output.tokens.shape == (1, 1, 192, 16)
    assert output.token_mask.shape == (1, 1, 192)
    assert not output.token_mask[0, 0, 1 * 48 + 7]
    assert torch.count_nonzero(output.tokens[0, 0, 1 * 48 + 7]) == 0
    assert output.token_layout == {
        "mode": "point_spatiotemporal",
        "history_length": 4,
        "num_regions": 48,
        "order": "sensor-major,time-major,spatial-minor",
    }
    restored = TactileTokenEncoder(settings, context_dim=16)
    report = load_vtla_checkpoint_state_dict(restored, encoder.state_dict())
    assert not report["marker_modules_reinitialized"]
    assert report["forbidden_missing_keys"] == []


def test_region_mapping_covers_once_and_matches_deployment_numpy():
    reference = np.asarray(_reference(2), dtype=np.float32)
    torch_ids, torch_centers = build_marker_region_mapping(
        reference, num_regions=4, region_layout="2x2"
    )
    numpy_ids, numpy_centers = build_marker_region_mapping_numpy(
        reference, num_regions=4, region_layout="2x2"
    )
    np.testing.assert_array_equal(torch_ids.numpy(), numpy_ids)
    np.testing.assert_allclose(torch_centers.numpy(), numpy_centers)
    for sensor_ids in numpy_ids:
        covered = np.concatenate([np.flatnonzero(sensor_ids == region) for region in range(4)])
        assert sorted(covered.tolist()) == list(range(48))
        assert all(np.count_nonzero(sensor_ids == region) == 12 for region in range(4))
    assert not np.array_equal(numpy_ids[0], numpy_ids[1])


def test_regional_masks_cover_marker_region_history_sensor_and_contact():
    settings = TactileVTLAConfig.from_mapping(
        _mapping(gate_mode="hard_hysteresis_soft_region")
    )
    encoder = TactileTokenEncoder(settings, context_dim=16)
    history, valid, history_valid, sensors = _inputs(settings, batch=1)
    region_zero = encoder.marker_region_ids[0] == 0
    valid[0, 0, 2, region_zero] = False
    history_valid[0, 0, 3] = False
    sensors[0, 1] = False
    state = torch.tensor([[CONTACT_ON, CONTACT_ON]], dtype=torch.int8)
    _, mask = encoder.encode_markers(
        history, valid, history_valid, sensors, state
    )
    assert not mask[0, 0, 2 * 4 + 0]
    assert not mask[0, 0, 3 * 4 : 4 * 4].any()
    assert not mask[0, 1].any()

    off = torch.full((1, 2), CONTACT_OFF, dtype=torch.int8)
    _, off_mask = encoder.encode_markers(
        history, valid, history_valid, torch.ones_like(sensors), off
    )
    assert not off_mask.any()
    hold = torch.full((1, 2), CONTACT_HOLD, dtype=torch.int8)
    _, hold_mask = encoder.encode_markers(
        history, valid, history_valid, torch.ones_like(sensors), hold
    )
    assert hold_mask.any()


def test_single_invalid_marker_cannot_pollute_regional_pooling():
    settings = TactileVTLAConfig.from_mapping(_mapping(num_sensors=1))
    encoder = TactileTokenEncoder(settings, context_dim=16)
    history, valid, history_valid, sensors = _inputs(settings, batch=1)
    valid[..., 0] = False
    baseline = history.clone()
    baseline[..., 0, :] = 0.0
    corrupted = baseline.clone()
    corrupted[..., 0, :] = 1.0e6

    expected, expected_mask = encoder.encode_markers(
        baseline, valid, history_valid, sensors
    )
    actual, actual_mask = encoder.encode_markers(
        corrupted, valid, history_valid, sensors
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, expected_mask)


def test_marker_only_gate_never_changes_rgb_mask_and_control_can_change_it():
    marker_settings = TactileVTLAConfig.from_mapping(
        _mapping(gate_mode="hard", gate_target="marker_only")
    )
    marker_encoder = TactileTokenEncoder(
        marker_settings, context_dim=8, vision_output_dim=8
    )
    rgb = torch.randn(1, 2, 3, 8)
    available = torch.ones(1, 2, dtype=torch.bool)
    off = torch.zeros(1, 2, dtype=torch.int8)
    gated = marker_encoder.gate_rgb_sensor_mask(available, off)
    _, rgb_mask = marker_encoder.encode_rgb_embeddings(rgb, gated, gated)
    assert rgb_mask.all()

    control_settings = TactileVTLAConfig.from_mapping(
        _mapping(gate_mode="hard", gate_target="marker_and_rgb")
    )
    control = TactileTokenEncoder(control_settings, context_dim=8, vision_output_dim=8)
    assert not control.gate_rgb_sensor_mask(available, off).any()


@pytest.mark.parametrize("kind", ["none", "learned", "sincos"])
def test_position_encoding_modes_have_stable_shapes(kind: str):
    settings = TactileVTLAConfig.from_mapping(
        _mapping(temporal=kind, spatial=kind)
    )
    encoder = TactileTokenEncoder(settings, context_dim=15)
    temporal = encoder._temporal_position(torch.float32, torch.device("cpu"))
    spatial = encoder._spatial_position(torch.float32, torch.device("cpu"))
    assert temporal.shape == (8, 15)
    assert spatial.shape == (2, 4, 15)
    if kind == "learned":
        assert encoder.marker_temporal_embedding.weight.requires_grad
        assert encoder.marker_spatial_embedding.weight.requires_grad
    else:
        assert encoder.marker_temporal_embedding is None
        assert encoder.marker_spatial_embedding is None
    if kind == "sincos":
        clone = TactileTokenEncoder(settings, context_dim=15)
        torch.testing.assert_close(temporal, clone.marker_temporal_encoding)
        assert not torch.equal(temporal[0], temporal[-1])
        assert not torch.equal(spatial[0, 0], spatial[0, 1])


def test_continuous_sincos_handles_odd_dims_and_real_time_rate():
    one = continuous_sincos_1d(torch.tensor([-0.2, 0.0]), 1)
    two = continuous_sincos_2d(torch.tensor([[0.2, -0.3]]), 7)
    assert one.shape == (2, 1)
    assert two.shape == (1, 7)
    at_30 = TactileTokenEncoder(
        TactileVTLAConfig.from_mapping(_mapping(sample_hz=30.0)), context_dim=9
    ).marker_temporal_encoding
    at_60 = TactileTokenEncoder(
        TactileVTLAConfig.from_mapping(_mapping(sample_hz=60.0)), context_dim=9
    ).marker_temporal_encoding
    assert not torch.equal(at_30[:-1], at_60[:-1])
    torch.testing.assert_close(at_30[-1], at_60[-1])


def _runtime_gate(mode: str = "hard_hysteresis_soft_region"):
    mapping = _mapping(num_sensors=1, gate_mode=mode)
    settings = TactileVTLAConfig.from_mapping(mapping)
    runtime = runtime_gate_config_from_mapping(
        settings.marker_contact_gate.to_dict(), num_sensors=1, num_markers=48
    )
    region_ids, _ = build_marker_region_mapping_numpy(
        settings.marker_reference_xy, num_regions=4, region_layout="2x2"
    )
    return MarkerContactGate(runtime, region_ids[0]), runtime


def test_gate_rejects_noise_and_single_outlier_but_accepts_local_contact():
    gate, _ = _runtime_gate("hard")
    valid = np.ones(48, dtype=np.bool_)
    noise = np.full((48, 2), 0.05, dtype=np.float32)
    assert gate.step(noise, valid).state == CONTACT_OFF
    outlier = noise.copy()
    outlier[0] = [1.0, 0.0]
    assert gate.step(outlier, valid).state == CONTACT_OFF
    contact = noise.copy()
    contact[:3] = [0.8, 0.0]
    frame = gate.step(contact, valid)
    assert frame.state == CONTACT_ON
    assert np.all((frame.regional_soft_gates >= 0) & (frame.regional_soft_gates <= 1))


def test_gate_hysteresis_hold_unknown_limit_and_episode_parity():
    gate, runtime = _runtime_gate()
    valid = np.ones(48, dtype=np.bool_)
    unknown = np.zeros(48, dtype=np.bool_)
    contact = np.zeros((48, 2), dtype=np.float32)
    contact[:3] = [0.8, 0.0]
    quiet = np.zeros_like(contact)
    sequence = np.stack([contact, contact, quiet, quiet] + [quiet] * 7)
    masks = np.stack([valid] * 4 + [unknown] * 7)

    online = []
    for displacement, mask in zip(sequence, masks):
        online.append(gate.step(displacement, mask).state)
    assert online[:4] == [CONTACT_OFF, CONTACT_ON, CONTACT_ON, CONTACT_HOLD]
    assert online[4:7] == [CONTACT_UNKNOWN_HOLD] * runtime.max_unknown_hold_frames
    assert online[-1] == CONTACT_UNKNOWN_OFF

    offline_gate, _ = _runtime_gate()
    offline = offline_gate.run_episode(sequence, masks)
    np.testing.assert_array_equal(offline.states, np.asarray(online, dtype=np.int8))


def test_unknown_frame_breaks_consecutive_contact_evidence():
    gate, _ = _runtime_gate("hard_hysteresis")
    valid = np.ones(48, dtype=np.bool_)
    unknown = np.zeros(48, dtype=np.bool_)
    contact = np.zeros((48, 2), dtype=np.float32)
    contact[:3] = [0.8, 0.0]

    assert gate.step(contact, valid).state == CONTACT_OFF
    gate.step(contact, unknown)
    assert gate.step(contact, valid).state == CONTACT_OFF
    assert gate.step(contact, valid).state == CONTACT_ON


def test_global_active_count_gate_requires_two_large_valid_markers():
    mapping = _mapping(
        num_sensors=1,
        gate_mode="global_active_count_hysteresis_soft_region",
    )
    mapping["marker_contact_gate"]["min_active_markers"] = 2
    settings = TactileVTLAConfig.from_mapping(mapping)
    runtime = runtime_gate_config_from_mapping(
        settings.marker_contact_gate.to_dict(), num_sensors=1, num_markers=48
    )
    region_ids, _ = build_marker_region_mapping_numpy(
        settings.marker_reference_xy, num_regions=4, region_layout="2x2"
    )
    gate = MarkerContactGate(runtime, region_ids[0])
    valid = np.ones(48, dtype=np.bool_)

    one = np.zeros((48, 2), dtype=np.float32)
    one[0, 0] = 0.8
    assert gate.step(one, valid).state == CONTACT_OFF
    frame = gate.step(one, valid)
    assert frame.state == CONTACT_OFF
    assert frame.on_active_marker_count == 1

    two = one.copy()
    two[1, 1] = 0.8
    first = gate.step(two, valid)
    second = gate.step(two, valid)
    assert first.state == CONTACT_OFF
    assert second.state == CONTACT_ON
    assert second.on_active_marker_count == 2
    assert second.active_counts.sum() == 2

    between_thresholds = np.zeros_like(two)
    between_thresholds[:2, 0] = 0.3
    held = gate.step(between_thresholds, valid)
    assert held.state == CONTACT_ON
    assert held.on_active_marker_count == 0
    assert held.off_active_marker_count == 2

    quiet = np.zeros_like(two)
    assert gate.step(quiet, valid).state == CONTACT_ON
    assert gate.step(quiet, valid).state == CONTACT_HOLD


def test_runtime_gate_validation_rejects_bad_thresholds_and_temperature():
    settings = TactileVTLAConfig.from_mapping(
        _mapping(num_sensors=1, gate_mode="hard")
    )
    resolved = settings.marker_contact_gate.to_dict()
    invalid = deepcopy(resolved)
    invalid["off_threshold"] = invalid["on_threshold"]
    with pytest.raises(ValueError, match="less than"):
        runtime_gate_config_from_mapping(invalid, num_sensors=1, num_markers=48)
    invalid = deepcopy(resolved)
    invalid["soft_gate_temperature"] = 0
    with pytest.raises(ValueError, match="temperature"):
        runtime_gate_config_from_mapping(invalid, num_sensors=1, num_markers=48)


def test_calibration_threshold_rule_is_reproducible():
    displacement = np.zeros((5, 48, 2), dtype=np.float32)
    displacement[..., 0] = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05])[:, None]
    valid = np.ones((5, 48), dtype=np.bool_)
    region_ids = np.repeat(np.arange(4), 12)

    stats = compute_contact_stats(
        displacement,
        valid,
        region_ids,
        topk_markers=3,
        mad_multiplier=6.0,
    )

    assert stats["point_noise_median"] == pytest.approx([0.03] * 48)
    assert stats["point_noise_mad"] == pytest.approx([0.01] * 48)
    assert stats["suggested_point_threshold"] == pytest.approx([0.09] * 48)
    assert stats["suggested_global_off_threshold"] < stats["suggested_global_on_threshold"]


def test_regional_encoder_and_learned_positions_receive_gradients():
    settings = TactileVTLAConfig.from_mapping(
        _mapping(temporal="learned", spatial="learned")
    )
    encoder = TactileTokenEncoder(settings, context_dim=16)
    history, valid, history_valid, sensors = _inputs(settings)
    history.requires_grad_(True)
    tokens, _ = encoder.encode_markers(history, valid, history_valid, sensors)
    tokens.square().mean().backward()
    assert encoder.regional_marker_encoder.point_encoder[0].weight.grad is not None
    assert encoder.regional_marker_encoder.region_projection[0].weight.grad is not None
    assert encoder.marker_temporal_embedding.weight.grad is not None
    assert encoder.marker_spatial_embedding.weight.grad is not None


def test_minimal_flow_matching_loss_backpropagates_to_regional_encoder():
    settings = TactileVTLAConfig.from_mapping(_mapping())
    encoder = TactileTokenEncoder(settings, context_dim=16)
    vector_field = torch.nn.Linear(16, 8)
    history, valid, history_valid, sensors = _inputs(settings)
    tokens, token_mask = encoder.encode_markers(
        history, valid, history_valid, sensors
    )
    pooled = (tokens * token_mask.unsqueeze(-1)).sum(dim=(1, 2))
    pooled = pooled / token_mask.sum(dim=(1, 2), keepdim=False).clamp_min(1).unsqueeze(-1)
    predicted_velocity = vector_field(pooled)
    actions = torch.randn_like(predicted_velocity)
    noise = torch.randn_like(predicted_velocity)
    target_velocity = noise - actions
    torch.nn.functional.mse_loss(predicted_velocity, target_velocity).backward()

    assert encoder.regional_marker_encoder.point_encoder[0].weight.grad is not None
    assert vector_field.weight.grad is not None


def test_config_round_trip_and_checkpoint_policies_are_explicit():
    regional = TactileVTLAConfig.from_mapping(_mapping())
    assert TactileVTLAConfig.from_mapping(regional.to_dict()).to_dict() == regional.to_dict()

    global_settings = TactileVTLAConfig.from_mapping(
        _mapping(mode="global", temporal="learned", spatial="none")
    )
    source = TactileTokenEncoder(global_settings, context_dim=16)
    exact = TactileTokenEncoder(global_settings, context_dim=16)
    exact_report = load_vtla_checkpoint_state_dict(exact, source.state_dict())
    assert not exact_report["marker_modules_reinitialized"]
    assert exact_report["forbidden_missing_keys"] == []
    assert exact_report["intentionally_reinitialized_keys"]

    target = TactileTokenEncoder(regional, context_dim=16)
    migration = load_vtla_checkpoint_state_dict(target, source.state_dict())
    assert migration["marker_modules_reinitialized"]
    assert "global_marker_encoder_to_regional" in migration["config_incompatibilities"]
    assert "learned_temporal_to_fixed_or_none" in migration["config_incompatibilities"]
    assert any(
        key.endswith("marker_temporal_embedding.weight")
        for key in migration["intentionally_ignored_keys"]
    )
    assert migration["forbidden_missing_keys"] == []
    assert migration["unexpected_keys"] == []

    learned_regional = TactileTokenEncoder(
        TactileVTLAConfig.from_mapping(
            _mapping(temporal="learned", spatial="learned")
        ),
        context_dim=16,
    )
    learned_report = load_vtla_checkpoint_state_dict(
        learned_regional, target.state_dict()
    )
    assert "fixed_or_missing_temporal_to_learned" in learned_report[
        "config_incompatibilities"
    ]
    assert "fixed_or_missing_spatial_to_learned" in learned_report[
        "config_incompatibilities"
    ]

    bad = dict(source.state_dict())
    bad["sensor_side_embeddings.weight"] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="unsupported shape mismatches"):
        load_vtla_checkpoint_state_dict(exact, bad)


def test_every_yaml_ablation_resolves_without_source_changes():
    config_dir = ROOT / "configs/vla/tacthru_umi"
    base = yaml.safe_load(
        (config_dir / "tacthru_umi_vtla_rgb_marker.yaml").read_text()
    )
    presets = yaml.safe_load(
        (config_dir / "ablations/presets.yaml").read_text()
    )
    assert len(presets) == 14
    for name in presets:
        resolved = resolve_config(base, presets, name)
        settings = TactileVTLAConfig.from_mapping(resolved["train"]["tactile"])
        encoder = TactileTokenEncoder(settings, context_dim=16, vision_output_dim=16)
        assert resolved["train"]["output_dir"].endswith(name)
        assert settings.marker_tokens_per_sensor in {8, 32, 192}
        if not settings.use_markers:
            assert encoder.marker_encoder is None
            assert encoder.regional_marker_encoder is None
            assert encoder.point_spatiotemporal_marker_encoder is None


def test_huggingface_config_round_trip_embeds_resolved_tactile_values(tmp_path):
    config = LingbotVLAConfig(tactile=_mapping(num_sensors=1))
    config.save_pretrained(tmp_path)

    loaded = LingbotVLAConfig.from_pretrained(tmp_path)
    settings = TactileVTLAConfig.from_mapping(loaded.tactile)

    assert settings.marker_tokenization.mode == "regional"
    assert settings.marker_tokens_per_sensor == 32
    assert len(settings.marker_reference_xy[0]) == 48


def test_embedded_resolved_values_override_changed_provenance_files(tmp_path):
    stats_path = tmp_path / "marker_stats.json"
    stats_path.write_text(
        json.dumps({"marker_mean": [9.0, 9.0], "marker_std": [9.0, 9.0]})
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "sensor_names": ["left"],
                "reference_xy": _reference(1)[0][::-1],
            }
        )
    )
    contact_path = tmp_path / "contact.json"
    contact_path.write_text(
        json.dumps(
            {
                "sensor_names": ["left"],
                "suggested_point_threshold": [9.0] * 48,
                "suggested_global_on_threshold": 9.0,
                "suggested_global_off_threshold": 8.0,
            }
        )
    )
    mapping = _mapping(num_sensors=1, gate_mode="hard")
    mapping.update(
        marker_stats_path=str(stats_path),
        marker_mean=[1.0, 2.0],
        marker_std=[3.0, 4.0],
        marker_reference_xy_path=str(reference_path),
    )
    mapping["marker_contact_gate"].update(
        threshold_source="calibration",
        threshold_stats_path=str(contact_path),
    )

    settings = TactileVTLAConfig.from_mapping(mapping)

    assert settings.marker_mean == (1.0, 2.0)
    assert settings.marker_std == (3.0, 4.0)
    assert settings.marker_contact_gate.on_thresholds == (0.5,)
    assert settings.marker_contact_gate.off_thresholds == pytest.approx((0.2,))
    expected = TactileVTLAConfig.from_mapping(
        {**mapping, "marker_reference_xy_path": None}
    )
    assert settings.marker_reference_xy == expected.marker_reference_xy


def test_regional_encoder_state_round_trips_through_dcp(tmp_path):
    import torch.distributed.checkpoint as dcp

    settings = TactileVTLAConfig.from_mapping(_mapping(num_sensors=1))
    source = TactileTokenEncoder(settings, context_dim=16)
    target = TactileTokenEncoder(settings, context_dim=16)
    state = {"model": source.state_dict()}
    dcp.save_state_dict(
        state,
        dcp.FileSystemWriter(str(tmp_path)),
        no_dist=True,
    )
    restored = {"model": target.state_dict()}
    dcp.load_state_dict(
        restored,
        dcp.FileSystemReader(str(tmp_path)),
        no_dist=True,
    )
    target.load_state_dict(restored["model"])

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
