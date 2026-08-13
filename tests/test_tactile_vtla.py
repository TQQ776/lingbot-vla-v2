from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import einops
import pytest
import numpy as np
import torch
from torch import nn
import yaml

from lingbotvla.models.vla.lingbot_vla import tactile_vtla as _tactile_model

ROOT = Path(__file__).parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_tactile_data = _load_module(
    "_test_tactile_data_module",
    ROOT / "lingbotvla/data/vla_data/tactile.py",
)
_marker_stats = _load_module(
    "_test_tactile_marker_stats_module",
    ROOT / "tools/compute_tacthru_marker_stats.py",
)
_vtla_optim = _load_module(
    "_test_tactile_optimizer_module",
    ROOT / "lingbotvla/optim/vtla.py",
)

MarkerEncoder = _tactile_model.MarkerEncoder
TactileTokenEncoder = _tactile_model.TactileTokenEncoder
TactileVTLAConfig = _tactile_model.TactileVTLAConfig
concatenate_vtla_context = _tactile_model.concatenate_vtla_context
load_marker_statistics = _tactile_model.load_marker_statistics
load_vtla_checkpoint_state_dict = _tactile_model.load_vtla_checkpoint_state_dict
migrate_legacy_tactile_config = _tactile_model.migrate_legacy_tactile_config
validate_marker_displacement_history = _tactile_model.validate_marker_displacement_history
left_pad_marker_history = _tactile_data.left_pad_marker_history
prepare_tactile_sample = _tactile_data.prepare_tactile_sample


def _settings(**overrides) -> TactileVTLAConfig:
    values = {
        "enabled": True,
        "num_sensors": 2,
        "num_markers": 4,
        "use_rgb": True,
        "use_markers": True,
        "marker_history_length": 8,
        "marker_input_features": 2,
        "marker_feature_mode": "displacement_history",
        "marker_hidden_dim": 16,
        "marker_mean": [0.0, 0.0],
        "marker_std": [1.0, 1.0],
        "rgb_keys": ["rgb_l", "rgb_r"],
        "marker_displacement_keys": ["disp_l", "disp_r"],
        "marker_valid_mask_keys": ["valid_l", "valid_r"],
    }
    values.update(overrides)
    return TactileVTLAConfig.from_mapping(values)


def test_config_defaults_to_tactile_disabled_and_rejects_invalid_modes():
    config = TactileVTLAConfig.from_mapping(None)
    assert config.enabled is False

    with pytest.raises(ValueError, match="requires use_rgb or use_markers"):
        TactileVTLAConfig.from_mapping(
            {"enabled": True, "use_rgb": False, "use_markers": False}
        )
    with pytest.raises(ValueError, match="num_sensors"):
        TactileVTLAConfig.from_mapping({"enabled": True, "num_sensors": 0})
    with pytest.raises(ValueError, match="marker_input_features"):
        TactileVTLAConfig.from_mapping(
            {"enabled": True, "marker_input_features": 6}
        )


def test_saved_marker_stats_remain_portable_without_original_path(tmp_path):
    settings = _settings(
        marker_stats_path=str(tmp_path / "server-only-stats.json"),
        marker_mean=[1.0, 2.0],
        marker_std=[0.5, 0.6],
    )
    assert settings.marker_mean == (1.0, 2.0)
    assert settings.marker_std == pytest.approx((0.5, 0.6))


def test_legacy_four_channel_config_migration_is_explicit_and_global():
    legacy = {
        "enabled": True,
        "num_sensors": 1,
        "sensor_names": ["left"],
        "num_markers": 48,
        "use_rgb": True,
        "use_markers": True,
        "marker_input_features": 4,
        "marker_tokens_per_sensor": 1,
        "marker_flow_keys": ["observation.tactile.marker_flow_left"],
        "marker_positions_keys": ["observation.tactile.marker_positions_left"],
        "marker_reference_keys": ["observation.tactile.marker_reference_left"],
        "marker_valid_mask_keys": ["observation.tactile.marker_valid_left"],
        "marker_mean": [0.0, 0.0, 0.0, 0.0],
        "marker_std": [1.0, 1.0, 1.0, 1.0],
    }

    strict, report = migrate_legacy_tactile_config(legacy)
    assert strict == legacy
    assert report is None
    with pytest.raises(ValueError, match="marker_input_features"):
        TactileVTLAConfig.from_mapping(strict)

    migrated, report = migrate_legacy_tactile_config(
        legacy, allow_legacy_marker_reinit=True
    )
    settings = TactileVTLAConfig.from_mapping(migrated)
    assert report["requires_marker_module_reinitialization"] is True
    assert settings.marker_input_features == 2
    assert settings.marker_history_length == 8
    assert settings.marker_tokens_per_sensor == 8
    assert settings.marker_tokenization.mode == "global"
    assert settings.marker_position_encoding.temporal_type == "learned"
    assert settings.marker_displacement_keys == (
        "observation.tactile.marker_displacement_left",
    )
    assert TactileVTLAConfig.from_mapping(settings.to_dict()) == settings


def test_marker_history_uses_only_displacement_and_masks_invalid_markers():
    history = torch.tensor([[[[[2.0, 3.0], [4.0, 6.0]]]]])
    valid = torch.tensor([[[[True, False]]]])
    features, output_mask = validate_marker_displacement_history(history, valid)
    assert output_mask.equal(valid)
    assert features[0, 0, 0, 0].tolist() == [2.0, 3.0]
    assert torch.count_nonzero(features[0, 0, 0, 1]) == 0


def test_marker_statistics_reject_legacy_four_channel_file(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"marker_mean":[0,0,0,0],"marker_std":[1,1,1,1]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Expected marker statistics for \[dx, dy\]"):
        load_marker_statistics(path)


def test_marker_encoder_single_and_two_sensor_shapes():
    encoder = MarkerEncoder(
        num_markers=4,
        context_dim=8,
        hidden_dim=16,
    )
    single = encoder(
        torch.randn(2, 1, 8, 4, 2),
        torch.ones(2, 1, 8, 4, dtype=torch.bool),
    )
    dual = encoder(
        torch.randn(2, 2, 8, 4, 2),
        torch.ones(2, 2, 8, 4, dtype=torch.bool),
    )
    assert single.shape == (2, 1, 8, 8)
    assert single.flatten(1, 2).shape == (2, 8, 8)
    assert dual.shape == (2, 2, 8, 8)
    assert dual.flatten(1, 2).shape == (2, 16, 8)


def test_marker_encoder_masks_fully_missing_sensor_after_mlp_bias():
    encoder = MarkerEncoder(num_markers=4, context_dim=8, hidden_dim=16)
    features = torch.randn(2, 2, 8, 4, 2)
    valid = torch.ones(2, 2, 8, 4, dtype=torch.bool)
    valid[:, 1, 3] = False
    tokens = encoder(features, valid)
    assert tokens.shape == (2, 2, 8, 8)
    assert torch.count_nonzero(tokens[:, 1, 3]) == 0


def test_marker_temporal_and_sensor_order_is_sensor_major_time_minor():
    tactile = TactileTokenEncoder(
        _settings(use_rgb=False), context_dim=2, vision_output_dim=2
    )
    for parameter in tactile.marker_encoder.parameters():
        nn.init.zeros_(parameter)
    tactile.marker_modality_embedding.data.zero_()
    tactile.sensor_side_embeddings.weight.data.copy_(
        torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    )
    tactile.marker_temporal_embedding.weight.data.copy_(
        torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    )
    history = torch.zeros(1, 2, 8, 4, 2)
    tokens, mask = tactile.encode_markers(
        history,
        torch.ones(1, 2, 8, 4, dtype=torch.bool),
        torch.ones(1, 2, 8, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )
    flattened = tokens.flatten(1, 2)
    assert flattened[0, :, 0].tolist() == [*range(8), *range(100, 108)]
    assert mask.flatten(1, 2).all()


def test_marker_masks_are_applied_after_all_biased_embeddings():
    tactile = TactileTokenEncoder(_settings(use_rgb=False), context_dim=8)
    history = torch.randn(1, 2, 8, 4, 2)
    marker_valid = torch.ones(1, 2, 8, 4, dtype=torch.bool)
    marker_valid[:, 0, 2] = False
    history_valid = torch.ones(1, 2, 8, dtype=torch.bool)
    history_valid[:, 0, 3] = False
    sensor_valid = torch.tensor([[True, False]])
    tokens, mask = tactile.encode_markers(
        history, marker_valid, history_valid, sensor_valid
    )
    assert not mask[0, 0, 2]
    assert not mask[0, 0, 3]
    assert not mask[0, 1].any()
    assert torch.count_nonzero(tokens[0, 0, 2:4]) == 0
    assert torch.count_nonzero(tokens[0, 1]) == 0


def test_tactile_rgb_projection_shape_and_missing_sensor_mask():
    encoder = TactileTokenEncoder(
        _settings(use_markers=False),
        context_dim=8,
        vision_output_dim=6,
    )
    embeddings = torch.randn(2, 2, 5, 6)
    tokens, mask = encoder.encode_rgb_embeddings(
        embeddings,
        tactile_sensor_mask=torch.tensor([[True, False], [True, True]]),
    )
    assert tokens.shape == (2, 10, 8)
    assert mask.shape == (2, 10)
    assert not mask[0, 5:].any()
    assert torch.count_nonzero(tokens[0, 5:]) == 0


def test_vtla_context_and_mask_lengths_are_extended_on_token_axis():
    vl = torch.randn(2, 7, 8)
    vl_mask = torch.ones(2, 7, dtype=torch.bool)
    rgb = torch.randn(2, 10, 8)
    rgb_mask = torch.ones(2, 10, dtype=torch.bool)
    marker = torch.randn(2, 2, 8)
    marker_mask = torch.ones(2, 2, dtype=torch.bool)
    context, mask = concatenate_vtla_context(
        vl, vl_mask, [(rgb, rgb_mask), (marker, marker_mask)]
    )
    assert context.shape == (2, 19, 8)
    assert mask.shape == (2, 19)


def test_tactile_disabled_regression_adds_no_context_tokens():
    vl = torch.randn(2, 7, 8)
    vl_mask = torch.ones(2, 7, dtype=torch.bool)
    context, mask = concatenate_vtla_context(vl, vl_mask, [])
    assert torch.equal(context, vl)
    assert torch.equal(mask, vl_mask)


def test_backward_reaches_new_modules_and_action_but_not_frozen_vision():
    torch.manual_seed(0)
    vision = nn.Linear(3, 6, bias=False).requires_grad_(False)
    tactile = TactileTokenEncoder(
        _settings(),
        context_dim=8,
        vision_output_dim=6,
    )
    action_expert = nn.Linear(8, 4)

    rgb_input = torch.randn(2, 2, 5, 3)
    rgb_emb = vision(rgb_input)
    rgb_tokens, _ = tactile.encode_rgb_embeddings(
        rgb_emb,
        tactile_sensor_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    history = torch.randn(2, 2, 8, 4, 2)
    marker_tokens, _ = tactile.encode_markers(
        history,
        torch.ones(2, 2, 8, 4, dtype=torch.bool),
        torch.ones(2, 2, 8, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
    )
    prediction = action_expert(
        torch.cat([rgb_tokens, marker_tokens.flatten(1, 2)], dim=1).mean(1)
    )
    prediction.square().mean().backward()

    assert vision.weight.grad is None
    assert tactile.tactile_rgb_projection.weight.grad is not None
    assert tactile.marker_encoder.encoder[0].weight.grad is not None
    assert tactile.tactile_rgb_modality_embedding.grad is not None
    assert tactile.marker_modality_embedding.grad is not None
    assert tactile.sensor_side_embeddings.weight.grad is not None
    assert tactile.marker_temporal_embedding.weight.grad is not None
    assert action_expert.weight.grad is not None


def test_optimizer_groups_are_disjoint_and_exclude_frozen_parameters():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = nn.Linear(2, 2)
            self.qwen_expert = nn.Linear(2, 2)
            self.depth_head = nn.Linear(2, 2)
            self.frozen_visual = nn.Linear(2, 2).requires_grad_(False)

    model = TinyModel()
    groups = _vtla_optim.build_vtla_param_groups(
        model,
        new_modules_lr=1e-4,
        action_expert_lr=1e-5,
        base_lr=2e-5,
        weight_decay=1e-2,
    )
    by_name = {group["name"]: group for group in groups}
    assert by_name["tactile"]["lr"] == pytest.approx(1e-4)
    assert by_name["action_expert"]["lr"] == pytest.approx(1e-5)
    assert by_name["base_trainable"]["lr"] == pytest.approx(2e-5)
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identifiers) == len(set(identifiers))
    assert id(model.frozen_visual.weight) not in identifiers
    assert id(model.frozen_visual.bias) not in identifiers


class _FakeImageProcessor:
    merge_size = 2

    def __call__(self, image):
        # Preserve deterministic values while mimicking Qwen's flattened patch
        # representation and grid contract.
        value = image.to(torch.float32).mean().reshape(1, 1).expand(4, 6).clone()
        return {
            "pixel_values": value,
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


@pytest.mark.parametrize("available", [1, 2, 7, 8, 10])
def test_marker_history_left_padding_for_episode_prefixes(available):
    raw = torch.arange(available, dtype=torch.float32).view(available, 1, 1)
    output = left_pad_marker_history(
        raw,
        history_length=8,
        trailing_shape=(1, 1),
        name="marker",
        dtype=torch.float32,
    )
    source = list(range(max(0, available - 8), available))
    expected = [source[0]] * (8 - len(source)) + source
    assert output[:, 0, 0].tolist() == expected


def test_data_transform_builds_oldest_to_current_displacement_history():
    settings = _settings()
    displacement_l = torch.stack(
        [torch.full((4, 2), float(index)) for index in range(3)], dim=0
    )
    sample = {
        "disp_l": displacement_l,
        "valid_l": torch.ones(3, 4, dtype=torch.bool),
        "rgb_l": torch.ones(3, 8, 8, dtype=torch.uint8),
        "scene": torch.zeros(3, 8, 8, dtype=torch.uint8),
    }
    output = prepare_tactile_sample(
        sample,
        _FakeImageProcessor(),
        settings,
        fallback_image_keys=("scene",),
    )
    assert output["marker_displacement_history"].shape == (2, 8, 4, 2)
    assert output["marker_displacement_history"][0, :, 0, 0].tolist() == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0
    ]
    assert output["marker_history_valid_mask"][0].all()
    assert not output["marker_history_valid_mask"][1].any()
    assert output["tactile_rgb"].shape == (2, 4, 6)
    assert output["tactile_rgb_mask"].tolist() == [True, False]
    assert output["tactile_sensor_mask"].tolist() == [True, False]


def test_inference_data_accepts_aggregate_tactile_fields_and_sensor_mask():
    settings = _settings()
    history = torch.randn(2, 8, 4, 2)
    sample = {
        "marker_displacement_history": history,
        "marker_valid_mask": torch.ones(2, 8, 4, dtype=torch.bool),
        "marker_history_valid_mask": torch.ones(2, 8, dtype=torch.bool),
        "tactile_rgb": torch.ones(2, 3, 8, 8, dtype=torch.uint8),
        "tactile_sensor_mask": torch.tensor([True, False]),
        "scene": torch.zeros(3, 8, 8, dtype=torch.uint8),
    }
    output = prepare_tactile_sample(
        sample,
        _FakeImageProcessor(),
        settings,
        fallback_image_keys=("scene",),
    )
    assert torch.equal(output["marker_displacement_history"], history)
    assert not output["marker_valid_mask"][1].any()
    assert not output["marker_history_valid_mask"][1].any()
    assert output["tactile_rgb_mask"].tolist() == [True, False]
    assert output["tactile_sensor_mask"].tolist() == [True, False]


def test_marker_stats_use_selected_episodes_without_padding_or_velocity():
    marker = np.asarray(
        [
            [[-100.0, -200.0]],
            [[-90.0, -180.0]],
            [[10.0, 20.0]],
            [[14.0, 28.0]],
        ],
        dtype=np.float32,
    )
    mean, std, count, frame_count = _marker_stats.compute_stats(
        marker,
        np.asarray([2, 4], dtype=np.int64),
        episode_start=1,
        episode_end=2,
        chunk_frames=1,
        eps=1e-6,
    )
    np.testing.assert_allclose(mean, [12.0, 24.0])
    np.testing.assert_allclose(std, [2.0, 4.0])
    assert count == 2
    assert frame_count == 2


def test_model_training_and_sampling_interfaces_expose_all_tactile_fields():
    source = (
        Path(__file__).parents[1]
        / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    signatures = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                signatures[(node.name, child.name)] = {
                    argument.arg for argument in child.args.args
                }
    required = {
        "tactile_rgb",
        "tactile_rgb_grid_thw",
        "marker_displacement_history",
        "marker_valid_mask",
        "marker_history_valid_mask",
        "marker_contact_state",
        "tactile_sensor_mask",
        "tactile_rgb_mask",
    }
    assert required <= signatures[("FlowMatchingV2", "forward")]
    assert required <= signatures[("FlowMatchingV2", "sample_actions")]
    assert required <= signatures[("LingbotVlaV2Policy", "forward")]
    assert required <= signatures[("LingbotVlaV2Policy", "sample_actions")]


def test_lerobot_v3_shared_parquet_paths_are_deduplicated():
    source = (ROOT / "lingbotvla/data/vla_data/base_dataset.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LeRobotDataset"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_hf_dataset"
    )
    assignments = {
        target.id: value
        for statement in method.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        for value in [statement.value]
    }
    files_expression = assignments["files"]
    assert isinstance(files_expression, ast.Call)
    assert isinstance(files_expression.func, ast.Name)
    assert files_expression.func.id == "sorted"
    assert isinstance(files_expression.args[0], ast.SetComp)


def _load_prefix_harness():
    """Execute the real prefix method without importing the 6B model stack."""

    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    flow_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )
    method = next(
        node
        for node in flow_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_embed_prefix_vtla"
    )
    harness = ast.ClassDef(
        name="PrefixHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "Tensor": torch.Tensor,
        "einops": einops,
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        "torch": torch,
    }
    exec(compile(module, str(ROOT / "prefix_harness.py"), "exec"), namespace)
    return namespace["PrefixHarness"]


class _FakeQwenPrefixBackend:
    def __init__(self, hidden_dim: int, num_patches: int) -> None:
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.position_call = None
        self.qwenvl = SimpleNamespace(
            config=SimpleNamespace(
                image_token_id=10,
                vision_start_token_id=11,
                vision_end_token_id=12,
                text_config=SimpleNamespace(eos_token_id=13),
            )
        )

    def embed_image(self, images, image_grid_thw):
        num_views = images.shape[0]
        view_values = torch.arange(
            1,
            num_views + 1,
            dtype=images.dtype,
            device=images.device,
        ).view(num_views, 1, 1)
        embeddings = view_values.expand(
            num_views, self.num_patches, self.hidden_dim
        ).clone()
        return embeddings, [embeddings + 100]

    def embed_special_token(self, token_id, batch, count, device, dtype):
        return torch.full(
            (batch, count, 1, self.hidden_dim),
            float(token_id),
            dtype=dtype,
            device=device,
        )

    def embed_language_tokens(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1).expand(
            *tokens.shape, self.hidden_dim
        )

    def build_prefix_position_ids(
        self,
        input_ids,
        attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        self.position_call = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "image_grid_thw": image_grid_thw.clone(),
        }
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)


def test_full_vtla_prefix_path_preserves_order_masks_rope_and_deepstack():
    harness = _load_prefix_harness()()
    harness.tactile_settings = _settings()
    harness.tactile_encoder = TactileTokenEncoder(
        harness.tactile_settings,
        context_dim=8,
        vision_output_dim=8,
    )
    harness.qwenvl_with_expert = _FakeQwenPrefixBackend(8, 3)
    harness.config = SimpleNamespace(
        qwen3vl_use_vision_boundaries=True,
        vlm_causal=False,
    )
    harness.use_depth_align = False
    harness._tactile_debug_logged = False

    batch_size, scene_views, tactile_sensors = 2, 2, 2
    images = torch.zeros(batch_size, scene_views, 4, 6)
    tactile_rgb = torch.ones(batch_size, tactile_sensors, 4, 6)
    scene_grid = torch.ones(batch_size, scene_views, 3, dtype=torch.long)
    tactile_grid = torch.ones(
        batch_size, tactile_sensors, 3, dtype=torch.long
    )
    image_mask = torch.tensor([[True, False], [True, True]])
    sensor_mask = torch.tensor([[True, False], [True, True]])
    language = torch.tensor([[21, 22, 23, 24], [31, 32, 33, 34]])
    language_mask = torch.ones_like(language, dtype=torch.bool)
    marker_history = torch.ones(batch_size, tactile_sensors, 8, 4, 2)
    marker_valid = torch.ones(
        batch_size, tactile_sensors, 8, 4, dtype=torch.bool
    )
    marker_history_valid = torch.ones(
        batch_size, tactile_sensors, 8, dtype=torch.bool
    )

    (
        context,
        pad_mask,
        attention_mask,
        position_ids,
        visual_mask,
        deepstack,
    ) = harness._embed_prefix_vtla(
        images=images,
        img_masks=image_mask,
        lang_tokens=language,
        lang_masks=language_mask,
        image_grid_thw=scene_grid,
        tactile_rgb=tactile_rgb,
        tactile_rgb_grid_thw=tactile_grid,
        marker_displacement_history=marker_history,
        marker_valid_mask=marker_valid,
        marker_history_valid_mask=marker_history_valid,
        marker_contact_state=None,
        tactile_sensor_mask=sensor_mask,
        tactile_rgb_mask=sensor_mask,
    )

    scene_length = scene_views * (3 + 2)
    language_length = language.shape[1]
    rgb_length = tactile_sensors * (3 + 2)
    marker_length = tactile_sensors * 8
    expected_length = scene_length + language_length + rgb_length + marker_length
    assert context.shape == (batch_size, expected_length, 8)
    assert pad_mask.shape == (batch_size, expected_length)
    assert attention_mask.shape == (batch_size, expected_length)
    assert position_ids.shape == (3, batch_size, expected_length)
    assert visual_mask.shape == (batch_size, expected_length)

    # Prefix order is scene, language, tactile RGB, marker.  A missing tactile
    # sensor retains its five-token slot, but both embeddings and mask are zero.
    rgb_start = scene_length + language_length
    missing_rgb = slice(rgb_start + 5, rgb_start + 10)
    assert not pad_mask[0, missing_rgb].any()
    assert torch.count_nonzero(context[0, missing_rgb]) == 0
    marker_start = rgb_start + rgb_length
    assert not pad_mask[0, marker_start + 8 : marker_start + 16].any()
    assert torch.count_nonzero(context[0, marker_start + 8 : marker_start + 16]) == 0

    position_call = harness.qwenvl_with_expert.position_call
    assert position_call is not None
    assert position_call["input_ids"][0, scene_length:rgb_start].equal(language[0])
    assert position_call["input_ids"][0, marker_start:].tolist() == [13] * 16
    expected_valid_views = int(image_mask.sum() + sensor_mask.sum())
    assert position_call["image_grid_thw"].shape == (expected_valid_views, 3)
    expected_visual_tokens = expected_valid_views * 3
    assert int(visual_mask.sum()) == expected_visual_tokens
    assert deepstack[0].shape == (expected_visual_tokens, 8)

    slow_result = harness._embed_prefix_vtla(
        images=images,
        img_masks=image_mask,
        lang_tokens=language,
        lang_masks=language_mask,
        image_grid_thw=scene_grid,
        tactile_rgb=tactile_rgb,
        tactile_rgb_grid_thw=tactile_grid,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=sensor_mask,
        tactile_rgb_mask=sensor_mask,
        _slow_only=True,
    )
    slow_length = scene_length + language_length + rgb_length
    torch.testing.assert_close(slow_result[0], context[:, :slow_length])
    torch.testing.assert_close(slow_result[1], pad_mask[:, :slow_length])
    torch.testing.assert_close(slow_result[2], attention_mask[:, :slow_length])
    torch.testing.assert_close(slow_result[3], position_ids[:, :, :slow_length])
    torch.testing.assert_close(slow_result[4], visual_mask[:, :slow_length])
    assert slow_result[0].shape[1] == slow_length


def test_flow_matching_and_euler_math_remain_the_original_expressions():
    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    flow_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )

    def method_source(name: str) -> str:
        method = next(
            node
            for node in flow_class.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return "\n".join(
            ast.unparse(statement) for statement in method.body
        )

    training = method_source("forward")
    sampling = method_source("sample_actions")
    assert "noise = torch.randn(actions.shape, device=device, dtype=dtype)" in training
    assert "time = self.sample_time(actions.size(0), device).to(dtype)" in training
    assert "x_t = time_expanded * noise + (1 - time_expanded) * actions" in training
    assert "u_t = noise - actions" in training
    assert "F.mse_loss(u_t, v_t, reduction='none')" in training
    assert "dt = torch.tensor(-1.0 / self.config.num_steps" in sampling
    assert "x_t += dt * v_t" in sampling
    assert "time += dt" in sampling


def test_checkpoint_compatibility_is_limited_to_tactile_namespace():
    method = TactileVTLAConfig.from_mapping(
        {"enabled": True, "num_sensors": 1, "use_rgb": False}
    )
    assert method.enabled
    model_source = (
        Path(__file__).parents[1]
        / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    assert 'tactile_prefix = "model.tactile_encoder."' in model_source
    assert "loaded_parameter_names" in model_source


def test_legacy_marker_checkpoint_requires_opt_in_and_reinitializes_only_marker_input():
    settings = _settings(use_rgb=False)
    target = TactileTokenEncoder(settings, context_dim=8)
    old_state = {name: value.clone() for name, value in target.state_dict().items()}
    weight_key = "marker_encoder.encoder.0.weight"
    old_state[weight_key] = torch.randn(
        old_state[weight_key].shape[0], old_state[weight_key].shape[1] * 2
    )
    old_state["marker_encoder.marker_mean"] = torch.zeros(4)
    old_state["marker_encoder.marker_std"] = torch.ones(4)
    old_state.pop("marker_temporal_embedding.weight")
    old_state["marker_modality_embedding"].fill_(3.0)

    with pytest.raises(RuntimeError, match="ALLOW_LEGACY_MARKER_REINIT"):
        load_vtla_checkpoint_state_dict(target, old_state)

    report = load_vtla_checkpoint_state_dict(
        target, old_state, allow_legacy_marker_reinit=True
    )
    assert report["legacy_marker_reinitialized"] is True
    assert report["unexpected_keys"] == []
    assert torch.all(target.marker_modality_embedding == 3.0)
    assert target.marker_encoder.encoder[0].weight.shape[1] == 4 * 2
    assert target.marker_encoder.marker_mean.shape == (2,)
    assert target.marker_temporal_embedding.weight.shape == (8, 8)


def test_missing_direct_tactile_parameter_uses_leaf_initializer():
    source = (ROOT / "lingbotvla/models/module_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_parameter"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"nn": nn}
    exec(compile(module, str(ROOT / "module_utils_harness.py"), "exec"), namespace)

    class DirectParameter(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Parameter(torch.full((2,), float("nan")))
            self.initialized = []

        def _init_missing_parameter(self, name):
            self.initialized.append(name)
            with torch.no_grad():
                self._parameters[name].zero_()

    class RootModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = DirectParameter()

        @staticmethod
        def _init_weights(module):
            return None

    root = RootModule()
    namespace["_init_parameter"](root, "tactile_encoder.embedding")
    assert root.tactile_encoder.initialized == ["embedding"]
    assert torch.equal(root.tactile_encoder.embedding, torch.zeros(2))


@pytest.mark.parametrize(
    ("name", "enabled", "use_rgb", "use_markers"),
    [
        ("tacthru_umi_vtla_base.yaml", False, True, True),
        ("tacthru_umi_vtla_rgb.yaml", True, True, False),
        ("tacthru_umi_vtla_marker.yaml", True, False, True),
        ("tacthru_umi_vtla_rgb_marker.yaml", True, True, True),
    ],
)
def test_ablation_configs_validate(name, enabled, use_rgb, use_markers):
    path = ROOT / "configs/vla/tacthru_umi" / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = TactileVTLAConfig.from_mapping(payload["train"]["tactile"])
    assert settings.enabled is enabled
    assert settings.use_rgb is use_rgb
    assert settings.use_markers is use_markers
    expected_dataset = (
        "insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_history8_v2"
        if enabled and use_markers
        else "insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_v1"
    )
    assert payload["data"]["train_path"].endswith(expected_dataset)
    if enabled:
        assert payload["train"]["optimizer"] == "adamw"
        assert payload["train"]["freeze_vlm"] is True
    if enabled and use_markers:
        assert settings.marker_history_length == 8
        assert settings.marker_input_features == 2
        assert settings.marker_feature_mode == "displacement_history"
        assert settings.marker_position_encoding.temporal_type in {
            "none",
            "learned",
            "sincos",
        }
