import hashlib
import json
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
import torch

from deploy.tacthru_umi_v2.http_server import (
    LingBotV2Backend,
    _checkpoint_robot_config_name,
    _tactile_experiment_contract_sha256,
    _validate_deployment_contract,
    create_http_server,
)
from deploy.tacthru_umi_v2.protocol import Observation, PROTOCOL_VERSION_V2
from deploy.tacthru_umi_v2.realman_client import LingBotV2HttpClient


class _Policy:
    def __init__(self) -> None:
        self.inputs = []

    def reset(self, robo_name):
        pass

    def infer(self, observation):
        self.inputs.append(observation)
        action = np.zeros((50, 8), dtype=np.float32)
        action[:, 6] = 1.0
        action[:, 7] = 0.004
        return {"action": action}


def _backend(*, allow_missing: bool = False, allow_ablation: bool = False):
    policy = _Policy()
    contract = {
        "combined_sha256": "c" * 64,
        "tactile": {
            "checkpoint_modalities": {
                "wrist_rgb": True,
                "tactile_rgb": True,
                "tactile_marker": True,
            },
            "history_steps": 2,
            "history_stride": 1,
            "max_timestamp_skew_s": 0.1,
            "marker_count": 48,
            "marker_dim": 2,
            "marker_normalization": "image_size_xy",
            "marker_normalization_size_xy": [640, 480],
            "missing_policy": "mask",
        },
    }
    backend = LingBotV2Backend(
        policy,
        checkpoint=Path("/tmp/fake/hf_ckpt"),
        norm_stats=Path("/tmp/fake/norm.json"),
        robot_config_path=Path("/tmp/fake/tacthru_umi_v2.yaml"),
        chunk_size=50,
        use_compile=False,
        dtype="bf16",
        contract=contract,
        allow_missing_tactile=allow_missing,
        allow_tactile_ablation=allow_ablation,
    )
    return backend, policy


def _observation(*, include_rgb=True, include_marker=True, force_marker=False, dry_run=True):
    now = time.time()
    return Observation(
        instruction="Insert the Ethernet cable.",
        state=np.asarray([0, 0, 0, 0, 0, 0, 1, 0.004], dtype=np.float32),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        protocol_version=PROTOCOL_VERSION_V2,
        contract_sha256="c" * 64,
        timestamp=now,
        wrist_timestamp=now,
        tactile_rgb_history=(np.zeros((2, 224, 224, 3), dtype=np.uint8) if include_rgb else None),
        tactile_rgb_timestamps=(np.full(2, now, dtype=np.float64) if include_rgb else None),
        tactile_rgb_history_mask=(np.ones(2, dtype=np.bool_) if include_rgb else None),
        marker_flow=(np.zeros((2, 48, 2), dtype=np.float32) if include_marker else None),
        marker_valid_mask=(np.ones((2, 48), dtype=np.bool_) if include_marker else None),
        marker_timestamps=(np.full(2, now, dtype=np.float64) if include_marker else None),
        marker_history_mask=(np.ones(2, dtype=np.bool_) if include_marker else None),
        force_mask_tactile_marker=force_marker,
        metadata={"dry_run": dry_run, "episode_reset": True},
    )


def _serve(backend):
    server = create_http_server(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_tactile_health_and_v2_request_map_fixed_model_batch_keys() -> None:
    backend, policy = _backend()
    server, thread = _serve(backend)
    host, port = server.server_address
    client = LingBotV2HttpClient(f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100)
    try:
        health = client.health()
        assert health["protocol_version"] == 1
        assert health["protocol_versions_supported"] == [1, 2]
        assert health["preferred_protocol_version"] == 2
        assert health["tactile_marker_spec"]["normalization_size_xy"] == [640, 480]
        response = client.predict(_observation(), expected_steps=50)
        assert response.protocol_version == 2
        assert set(policy.inputs[-1]) == {
            "observation.state",
            "observation.images.camera_wrist_left",
            "task",
            "tactile_rgb_history",
            "tactile_rgb_history_mask",
            "tactile_marker_flow",
            "tactile_marker_valid_mask",
            "tactile_marker_history_mask",
            "tactile_history_timestamps",
        }
        assert policy.inputs[-1]["tactile_rgb_history_mask"].all()
        assert policy.inputs[-1]["tactile_marker_valid_mask"].all()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_missing_tactile_and_contract_mismatch_are_explicit_http_400() -> None:
    backend, _ = _backend()
    server, thread = _serve(backend)
    host, port = server.server_address
    client = LingBotV2HttpClient(f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100)
    try:
        with pytest.raises(RuntimeError, match="Checkpoint requires tactile_marker"):
            client.predict(_observation(include_marker=False), expected_steps=50)
        mismatch = _observation()
        mismatch = Observation(**{**mismatch.__dict__, "contract_sha256": "d" * 64})
        with pytest.raises(RuntimeError, match="contract mismatch"):
            client.predict(mismatch, expected_steps=50)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_force_mask_only_works_in_explicit_dry_run_and_clears_masks() -> None:
    backend, policy = _backend(allow_ablation=True)
    response = backend.predict(_observation(force_marker=True, dry_run=True))
    assert response["metadata"]["tactile"]["force_mask"]["tactile_marker"] is True
    assert not policy.inputs[-1]["tactile_marker_valid_mask"].any()
    assert not policy.inputs[-1]["tactile_marker_history_mask"].any()

    with pytest.raises(ValueError, match="forbidden for execution"):
        backend.predict(_observation(force_marker=True, dry_run=False))


def test_missing_tactile_ablation_uses_fixed_zero_tensors_only_in_dry_run() -> None:
    backend, policy = _backend(allow_missing=True)
    backend.predict(_observation(include_marker=False, dry_run=True))
    assert policy.inputs[-1]["tactile_marker_flow"].shape == (2, 48, 2)
    assert not policy.inputs[-1]["tactile_marker_valid_mask"].any()
    assert not policy.inputs[-1]["tactile_marker_history_mask"].any()

    with pytest.raises(ValueError, match="Checkpoint requires tactile_marker"):
        backend.predict(_observation(include_marker=False, dry_run=False))


def _make_tactile_checkpoint_contract(tmp_path: Path) -> dict:
    project = tmp_path / "project"
    checkpoint = project / "output/run/checkpoints/global_step_1/hf_ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    norm = project / "assets/norm_stats/tactile.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}\n", encoding="utf-8")
    training = {
        "model": {"model_path": "/tmp/model"},
        "data": {
            "data_name": "tacthru_umi_v2_tactile",
            "cameras": ["camera_wrist_left"],
            "train_path": str(project / "data" / "lerobot"),
            "norm_stats_file": "assets/norm_stats/tactile.json",
            "tactile_rgb_key": "observation.images.tactile_left",
            "tactile_marker_key": "observation.tactile.marker_flow_left",
            "tactile_marker_valid_key": "observation.tactile.marker_valid_left",
        },
        "train": {
            "chunk_size": 50,
            "action_dim": 55,
            "max_action_dim": 55,
            "max_state_dim": 55,
            "tactile_rgb_enabled": True,
            "tactile_marker_enabled": False,
            "tactile_train_stage": "full",
            "tactile_experiment_contract_sha256": "",
            "tactile_dataset_manifest_sha256": "",
            "tactile_rgb_backbone_sha256": "b" * 64,
            "tactile_params": {
                "history_steps": 4,
                "history_stride": 2,
                "history_frequency_hz": 30.0,
                "max_timestamp_skew_s": 0.05,
                "marker_count": 48,
                "marker_dim": 2,
                "marker_normalization": "image_size_xy",
                "marker_normalization_size_xy": [640, 480],
                "missing_policy": "mask",
            },
        },
    }
    config_path = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    config_path.write_text(yaml.safe_dump(training), encoding="utf-8")
    dataset_manifest = project / "data" / "tacthru_umi_v2_conversion.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_text('{"source_episodes": 1}\n', encoding="utf-8")
    experiment = {
        "schema_version": 1,
        "dataset_tactile_mode": "rgb-marker",
        "train_tactile_mode": "rgb",
        "tactile_train_stage": "full",
        "source": "/tmp/source.zarr.zip",
        "lerobot_dataset": str(project / "data" / "lerobot"),
        "norm_stats": str(norm),
        "initial_model": "/tmp/model",
        "base_model_assets": "/tmp/model",
        "train_overrides": [],
        "config_sha256": "a" * 64,
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
        "norm_stats_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "rgb_backbone_sha256": "b" * 64,
        "git_head": "c" * 40,
    }
    dataset_manifest_snapshot = config_path.parent / "tactile_dataset_manifest.json"
    dataset_manifest_snapshot.write_bytes(dataset_manifest.read_bytes())
    experiment["dataset_manifest_snapshot"] = str(dataset_manifest_snapshot)
    experiment["contract_sha256"] = _tactile_experiment_contract_sha256(experiment)
    (config_path.parent / "tactile_experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    training["train"]["tactile_experiment_contract_sha256"] = experiment["contract_sha256"]
    training["train"]["tactile_dataset_manifest_sha256"] = experiment["dataset_manifest_sha256"]
    config_path.write_text(yaml.safe_dump(training), encoding="utf-8")
    robot_path = project / "configs/robot_configs/tacthru_umi_v2_tactile.yaml"
    robot_path.parent.mkdir(parents=True)
    robot_path.write_text(
        yaml.safe_dump(
            {
                "states": [
                    {"observation.state.end.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 7}}]}},
                    {"observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 7, "end": 8}}]}},
                ],
                "actions": [
                    {"action.end.position": {"origin_keys": [{"action": {"start": 0, "end": 7}}], "subtract_state": True, "relative_type": "quaternion_local"}},
                    {"action.effector.position": {"origin_keys": [{"action": {"start": 7, "end": 8}}], "subtract_state": False}},
                ],
                "images": ["observation.images.camera_wrist_left"],
                "norm_stats": "assets/norm_stats/tactile.json",
            }
        ),
        encoding="utf-8",
    )

    assert _checkpoint_robot_config_name(checkpoint) == "tacthru_umi_v2_tactile"
    contract = _validate_deployment_contract(
        project_root=project,
        checkpoint=checkpoint,
        norm_stats=norm,
        robot_config_path=robot_path,
    )
    assert contract["tactile"]["checkpoint_modalities"]["tactile_rgb"] is True
    assert contract["tactile"]["checkpoint_modalities"]["tactile_marker"] is False
    assert contract["tactile"]["history_steps"] == 4
    assert contract["tactile"]["history_stride"] == 2
    assert contract["tactile"]["history_frequency_hz"] == pytest.approx(30.0)
    assert contract["tactile"]["marker_normalization_size_xy"] == [640, 480]
    return {
        "project": project,
        "checkpoint": checkpoint,
        "norm": norm,
        "robot_path": robot_path,
        "config_path": config_path,
        "experiment_path": config_path.parent / "tactile_experiment.json",
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_snapshot": dataset_manifest_snapshot,
        "contract": contract,
    }


def test_checkpoint_training_config_is_authoritative_for_tactile_contract(tmp_path: Path) -> None:
    case = _make_tactile_checkpoint_contract(tmp_path)
    assert case["contract"]["tactile"]["experiment_contract_sha256"]


def test_tactile_experiment_self_hash_tampering_is_rejected(tmp_path: Path) -> None:
    case = _make_tactile_checkpoint_contract(tmp_path)
    experiment = json.loads(case["experiment_path"].read_text(encoding="utf-8"))
    experiment["train_tactile_mode"] = "marker"
    case["experiment_path"].write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="tactile experiment self-hash mismatch"):
        _validate_deployment_contract(
            project_root=case["project"],
            checkpoint=case["checkpoint"],
            norm_stats=case["norm"],
            robot_config_path=case["robot_path"],
        )


@pytest.mark.parametrize(
    ("training_key", "error_name"),
    [
        ("tactile_dataset_manifest_sha256", "training tactile dataset manifest SHA256 mismatch"),
        ("tactile_rgb_backbone_sha256", "training tactile RGB backbone SHA256 mismatch"),
    ],
)
def test_training_yaml_hash_mismatch_with_experiment_contract_is_rejected(
    tmp_path: Path,
    training_key: str,
    error_name: str,
) -> None:
    case = _make_tactile_checkpoint_contract(tmp_path)
    training = yaml.safe_load(case["config_path"].read_text(encoding="utf-8"))
    training["train"][training_key] = "d" * 64
    case["config_path"].write_text(yaml.safe_dump(training), encoding="utf-8")

    with pytest.raises(RuntimeError, match=error_name):
        _validate_deployment_contract(
            project_root=case["project"],
            checkpoint=case["checkpoint"],
            norm_stats=case["norm"],
            robot_config_path=case["robot_path"],
        )


@pytest.mark.parametrize(
    ("mutate", "error_name"),
    [
        (
            lambda training: training["train"].__setitem__(
                "tactile_train_stage", "adapters"
            ),
            "tactile experiment/checkpoint training stage mismatch",
        ),
        (
            lambda training: training["train"].__setitem__(
                "tactile_rgb_enabled", False
            ),
            "tactile experiment/checkpoint modality mode mismatch",
        ),
        (
            lambda training: training["data"].__setitem__(
                "train_path", "/tmp/other-lerobot-dataset"
            ),
            "tactile experiment/checkpoint dataset path mismatch",
        ),
    ],
)
def test_training_yaml_semantics_must_match_tactile_experiment(
    tmp_path: Path,
    mutate,
    error_name: str,
) -> None:
    case = _make_tactile_checkpoint_contract(tmp_path)
    training = yaml.safe_load(case["config_path"].read_text(encoding="utf-8"))
    mutate(training)
    case["config_path"].write_text(yaml.safe_dump(training), encoding="utf-8")

    with pytest.raises(RuntimeError, match=error_name):
        _validate_deployment_contract(
            project_root=case["project"],
            checkpoint=case["checkpoint"],
            norm_stats=case["norm"],
            robot_config_path=case["robot_path"],
        )


def test_dataset_manifest_snapshot_content_tampering_is_rejected(tmp_path: Path) -> None:
    case = _make_tactile_checkpoint_contract(tmp_path)
    case["dataset_manifest_snapshot"].write_text(
        '{"source_episodes": 999}\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="dataset manifest snapshot SHA256 mismatch"):
        _validate_deployment_contract(
            project_root=case["project"],
            checkpoint=case["checkpoint"],
            norm_stats=case["norm"],
            robot_config_path=case["robot_path"],
        )


def test_from_checkpoint_propagates_missing_and_ablation_runtime_flags(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    from deploy.tacthru_umi_v2 import http_server as server_module

    checkpoint = tmp_path / "hf_ckpt"
    norm = tmp_path / "norm.json"
    checkpoint.mkdir()
    norm.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(server_module, "_checkpoint_robot_config_name", lambda _path: "tacthru_umi_v2_tactile")
    monkeypatch.setattr(server_module, "_validate_runtime_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        server_module,
        "_validate_deployment_contract",
        lambda **kwargs: {
            "combined_sha256": "e" * 64,
            "robot_default_norm_overridden": False,
            "tactile": {
                "checkpoint_modalities": {
                    "wrist_rgb": True,
                    "tactile_rgb": True,
                    "tactile_marker": True,
                }
            },
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class FakePolicyServer:
        def __init__(self, *args, **kwargs):
            self.config = types.SimpleNamespace(chunk_size=50)

        def reset(self, robo_name):
            self.robo_name = robo_name

    fake_policy_module = types.ModuleType("deploy.lingbot_vla_v2_policy")
    fake_policy_module.LingbotVLAv2Server = FakePolicyServer
    monkeypatch.setitem(sys.modules, "deploy.lingbot_vla_v2_policy", fake_policy_module)

    backend = LingBotV2Backend.from_checkpoint(
        project_root=tmp_path,
        checkpoint=checkpoint,
        norm_stats=norm,
        qwen_path=None,
        use_compile=False,
        dtype="bf16",
        inference_lock_timeout_s=1.0,
        allow_missing_tactile=True,
        allow_tactile_ablation=True,
    )

    assert backend.allow_missing_tactile is True
    assert backend.allow_tactile_ablation is True
    assert backend.health()["allow_missing_tactile"] is True
    assert backend.health()["allow_tactile_ablation"] is True
