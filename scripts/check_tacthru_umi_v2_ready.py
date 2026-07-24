#!/usr/bin/env python3
"""Validate artifacts needed for TacThru UMI post-training on LingBot-VLA v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingbotvla.tactile.schema import (
    TACTILE_MARKER_FLOW_KEY,
    TACTILE_MARKER_VALID_KEY,
    TACTILE_RGB_KEY,
    TACTILE_SCHEMA_VERSION,
    TACTILE_TIMESTAMP_KEY,
)


CHUNK_SIZE = 50
EXPECTED_SOURCE_FPS = 30
EXPECTED_DATASET_FPS = 30
EXPECTED_FUTURE_EFFECTIVE_FPS = EXPECTED_DATASET_FPS / (CHUNK_SIZE - 1)
REQUIRED_VIDEO_KEYS = {
    "observation.images.camera_wrist_left",
}
UNEXPECTED_VIDEO_KEYS = {
    "observation.images.camera_top",
    "observation.images.camera_wrist_right",
}
NORM_DIMS = {
    "observation.state.end.position": 7,
    "observation.state.effector.position": 1,
    "action.end.position": 7,
    "action.effector.position": 1,
}


def _tactile_modalities(mode: str) -> set[str]:
    return {
        "none": set(),
        "rgb": {"rgb"},
        "marker": {"marker"},
        "rgb-marker": {"rgb", "marker"},
    }[mode]


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def ok(self, name: str, detail: str) -> None:
        self.checks.append({"status": "ok", "name": name, "detail": detail})

    def warn(self, name: str, detail: str) -> None:
        self.checks.append({"status": "warning", "name": name, "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.checks.append({"status": "failed", "name": name, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(item["status"] == "failed" for item in self.checks)

    def display(self, as_json: bool) -> None:
        if as_json:
            print(json.dumps({"ready": not self.failed, "checks": self.checks}, ensure_ascii=False, indent=2))
            return
        symbols = {"ok": "[OK]", "warning": "[WARN]", "failed": "[FAIL]"}
        for item in self.checks:
            print(f"{symbols[item['status']]} {item['name']}: {item['detail']}")
        print("READY" if not self.failed else "NOT READY")


def check_python(report: Report) -> None:
    try:
        import lerobot
        import scipy
        import torch
        import transformers
        import zarr

        report.ok(
            "Python environment",
            f"python={sys.version.split()[0]}, torch={torch.__version__}, "
            f"transformers={transformers.__version__}, "
            f"lerobot={getattr(lerobot, '__version__', 'unknown')}, "
            f"scipy={scipy.__version__}, zarr={zarr.__version__}",
        )
    except Exception as exc:
        report.fail("Python environment", repr(exc))


def check_model(report: Report, path: Path) -> None:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    if not config.is_file():
        report.fail("LingBot-v2 config", f"missing {config}")
    else:
        report.ok("LingBot-v2 config", str(config))

    shards: list[Path] = []
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            shards = sorted({path / name for name in payload["weight_map"].values()})
        except Exception as exc:
            report.fail("LingBot-v2 weight index", repr(exc))
            return
    else:
        shards = sorted(path.glob("*.safetensors"))

    if not shards:
        report.fail("LingBot-v2 weights", f"no safetensors shards in {path}")
        return
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        report.fail("LingBot-v2 weights", f"missing shards: {missing}")
        return

    total_size = sum(shard.stat().st_size for shard in shards)
    if total_size < 25_000_000_000:
        report.fail("LingBot-v2 weights", f"incomplete total size: {total_size:,} bytes")
        return
    try:
        from safetensors import safe_open

        tensor_count = 0
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                tensor_count += len(handle.keys())
        report.ok(
            "LingBot-v2 weights",
            f"{len(shards)} shard(s), {total_size:,} bytes, {tensor_count} tensors",
        )
    except Exception as exc:
        report.fail("LingBot-v2 weights", f"cannot read safetensors headers: {exc}")


def check_qwen(report: Report, path: Path) -> None:
    required = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        report.fail("Qwen3 processor assets", f"missing {missing} in {path}")
        return
    try:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        report.ok("Qwen3 processor assets", f"offline load succeeds from {path}")
    except Exception as exc:
        report.fail("Qwen3 processor assets", repr(exc))
        return

    index = path / "model.safetensors.index.json"
    if not index.is_file():
        report.fail(
            "Qwen3 standalone weights",
            f"missing official shard index: {index}",
        )
        return
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_files = sorted({path / name for name in payload["weight_map"].values()})
    except Exception as exc:
        report.fail("Qwen3 standalone weights", f"invalid weight index: {exc}")
        return

    if not weight_files:
        report.fail(
            "Qwen3 standalone weights",
            "missing; the official v2 README requires the complete Qwen3-VL checkpoint",
        )
        return
    missing_weights = [str(weight) for weight in weight_files if not weight.is_file()]
    if missing_weights:
        report.fail("Qwen3 standalone weights", f"missing shards: {missing_weights}")
        return
    try:
        from safetensors import safe_open

        tensor_count = 0
        for weight in weight_files:
            with safe_open(weight, framework="pt", device="cpu") as handle:
                tensor_count += len(handle.keys())
        total_size = sum(weight.stat().st_size for weight in weight_files)
        report.ok(
            "Qwen3 standalone weights",
            f"{len(weight_files)} complete shard(s), {total_size:,} bytes, {tensor_count} tensors",
        )
    except Exception as exc:
        report.fail("Qwen3 standalone weights", f"cannot read safetensors headers: {exc}")


def check_teacher_file(report: Report, name: str, path: Path, minimum_size: int) -> None:
    if not path.is_file():
        report.fail(name, f"missing {path}")
        return
    size = path.stat().st_size
    if size < minimum_size:
        report.fail(name, f"file appears incomplete: {size:,} bytes at {path}")
    else:
        report.ok(name, f"{size:,} bytes at {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_tactile_config(report: Report, path: Path, tactile_mode: str) -> None:
    """Validate tactile contract and only require DINO weights for RGB models."""

    modalities = _tactile_modalities(tactile_mode)
    if not modalities:
        report.ok("Tactile architecture config", "disabled for wrist-only architecture")
        return
    if not path.is_file():
        report.fail("Tactile architecture config", f"missing {path}")
        return
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = payload.get("train", {}).get("tactile_params", {})
        data = payload.get("data", {})
        errors = []
        expected_keys = {
            "tactile_rgb_key": TACTILE_RGB_KEY,
            "tactile_marker_key": TACTILE_MARKER_FLOW_KEY,
            "tactile_marker_valid_key": TACTILE_MARKER_VALID_KEY,
            "tactile_timestamp_key": TACTILE_TIMESTAMP_KEY,
        }
        for key, expected in expected_keys.items():
            if data.get(key) != expected:
                errors.append(f"data.{key}={data.get(key)!r}, expected {expected!r}")
        if int(params.get("history_steps", 0)) < 1:
            errors.append("history_steps must be >= 1")
        if float(params.get("history_frequency_hz", 0.0)) <= 0:
            errors.append("history_frequency_hz must be positive")
        if "marker" in modalities:
            if int(params.get("marker_count", 0)) != 48 or int(params.get("marker_dim", 0)) != 2:
                errors.append("marker contract must be [48,2]")
            if params.get("marker_normalization") != "image_size_xy":
                errors.append("marker_normalization must be image_size_xy")
            if list(params.get("marker_normalization_size_xy", [])) != [640, 480]:
                errors.append("marker_normalization_size_xy must be [640,480]")
        if errors:
            report.fail("Tactile architecture config", "; ".join(errors))
            return
        report.ok(
            "Tactile architecture config",
            f"mode={tactile_mode}, history={params['history_steps']} at "
            f"{params['history_frequency_hz']}Hz",
        )

        if "rgb" not in modalities:
            report.ok("Tactile RGB backbone", "not required by marker-only architecture")
            return
        if params.get("rgb_backbone") != "dinov2_vits14":
            report.fail(
                "Tactile RGB backbone",
                f"unsupported rgb_backbone={params.get('rgb_backbone')!r}",
            )
            return
        if int(params.get("rgb_input_size", 224)) != 224:
            report.fail("Tactile RGB backbone", "rgb_input_size must be 224")
            return
        backbone_value = params.get("rgb_backbone_path")
        if not backbone_value:
            report.fail("Tactile RGB backbone", "rgb_backbone_path is empty")
            return
        root = Path(__file__).resolve().parents[1]
        backbone_path = Path(backbone_value).expanduser()
        if not backbone_path.is_absolute():
            backbone_path = root / backbone_path
        if not backbone_path.is_file() or backbone_path.stat().st_size <= 0:
            report.fail("Tactile RGB backbone", f"missing or empty {backbone_path}")
            return
        report.ok(
            "Tactile RGB backbone",
            f"dinov2_vits14, input=224, bytes={backbone_path.stat().st_size:,}, "
            f"sha256={_sha256_file(backbone_path)}",
        )
    except Exception as exc:
        report.fail("Tactile architecture config", repr(exc))


def _episode_length(episode: dict) -> int:
    return int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])


def check_dataset(
    report: Report,
    path: Path,
    expected_source_episodes: int | None,
    tactile_mode: str = "none",
) -> tuple[int | None, int | None]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id=path.name, root=path)
        tactile_rgb_enabled = tactile_mode in {"rgb", "rgb-marker"}
        tactile_marker_enabled = tactile_mode in {"marker", "rgb-marker"}
        required_video_keys = set(REQUIRED_VIDEO_KEYS)
        if tactile_rgb_enabled:
            required_video_keys.add(TACTILE_RGB_KEY)
        features = set(meta.features)
        missing = required_video_keys - features
        if missing:
            report.fail("LeRobot images", f"missing {sorted(missing)}")
        else:
            report.ok(
                "LeRobot images",
                "wrist RGB" + (" + tactile RGB" if tactile_rgb_enabled else " only"),
            )
        unexpected_candidates = set(UNEXPECTED_VIDEO_KEYS)
        if not tactile_rgb_enabled:
            unexpected_candidates.add(TACTILE_RGB_KEY)
        unexpected = unexpected_candidates & features
        if unexpected:
            report.fail(
                "Disabled visual inputs",
                f"unexpected converted image features: {sorted(unexpected)}",
            )
        else:
            report.ok("Disabled visual inputs", "unrequested visual slots are absent")

        tactile_shape_errors = []
        if tactile_marker_enabled:
            marker_shape = tuple(meta.features.get(TACTILE_MARKER_FLOW_KEY, {}).get("shape", ()))
            valid_shape = tuple(meta.features.get(TACTILE_MARKER_VALID_KEY, {}).get("shape", ()))
            if marker_shape != (48, 2):
                tactile_shape_errors.append(
                    f"{TACTILE_MARKER_FLOW_KEY}={marker_shape}, expected (48,2)"
                )
            if valid_shape != (48,):
                tactile_shape_errors.append(
                    f"{TACTILE_MARKER_VALID_KEY}={valid_shape}, expected (48,)"
                )
        if tactile_rgb_enabled or tactile_marker_enabled:
            timestamp_shape = tuple(meta.features.get(TACTILE_TIMESTAMP_KEY, {}).get("shape", ()))
            if timestamp_shape != (1,):
                tactile_shape_errors.append(
                    f"{TACTILE_TIMESTAMP_KEY}={timestamp_shape}, expected (1,)"
                )
        if tactile_shape_errors:
            report.fail("Canonical tactile features", "; ".join(tactile_shape_errors))
        elif tactile_rgb_enabled or tactile_marker_enabled:
            report.ok("Canonical tactile features", f"mode={tactile_mode}, schema={TACTILE_SCHEMA_VERSION}")

        shape_errors = []
        for key in ("observation.state", "action"):
            actual = tuple(meta.features.get(key, {}).get("shape", ()))
            if actual != (8,):
                shape_errors.append(f"{key}={actual}, expected (8,)")
        if shape_errors:
            report.fail("Canonical raw state/action", "; ".join(shape_errors))
        else:
            report.ok("Canonical raw state/action", "xyz + quaternion_xyzw + gripper = 8D")

        if not math.isclose(
            float(meta.fps),
            EXPECTED_DATASET_FPS,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            report.fail(
                "Dataset FPS",
                f"{meta.fps}; expected native {EXPECTED_DATASET_FPS} Hz",
            )
        else:
            report.ok("Dataset FPS", f"native {EXPECTED_DATASET_FPS} Hz")

        lengths = [_episode_length(ep) for ep in meta.episodes]
        valid_anchors = sum(max(0, length - (CHUNK_SIZE - 1)) for length in lengths)
        report.ok(
            "Episode metadata",
            f"episodes={meta.total_episodes}, frames={meta.total_frames}, valid anchors={valid_anchors}",
        )
        short_episodes = [index for index, length in enumerate(lengths) if length < CHUNK_SIZE]
        if short_episodes:
            report.warn(
                "Short native episodes",
                f"{len(short_episodes)} episode(s) contribute zero complete 50-step anchors",
            )

        if expected_source_episodes is not None and meta.total_episodes != expected_source_episodes:
            report.fail(
                "Output episode count",
                f"output episodes={meta.total_episodes}, expected={expected_source_episodes}",
            )
        elif expected_source_episodes is not None:
            report.ok(
                "Output episode count",
                f"{expected_source_episodes} (one output episode per source episode)",
            )

        manifest_path = path / "tacthru_umi_v2_conversion.json"
        source_episodes = None
        if not manifest_path.is_file():
            report.fail("Conversion manifest", f"missing {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_episodes = int(manifest.get("source_episodes", -1))
            output_episodes = int(manifest.get("output_episodes", -1))
            converted_episodes = int(
                manifest.get("converted_episodes", manifest.get("output_episodes", -1))
            )
            manifest_errors = []
            expected_manifest_values = {
                "converter": "tacthru_umi_v2_native_30hz",
                "converter_version": (
                    5 if (tactile_rgb_enabled or tactile_marker_enabled) else 4
                ),
                "source_fps": EXPECTED_SOURCE_FPS,
                "output_fps": EXPECTED_DATASET_FPS,
                "action_chunk_size": CHUNK_SIZE,
                "source_frames": meta.total_frames,
                "output_frames": meta.total_frames,
                "frames": meta.total_frames,
                "valid_anchors": valid_anchors,
                "episode_mapping": "one_source_episode_to_one_output_episode",
                "frame_sampling": "contiguous_stride_1",
                "frames_dropped": 0,
                "frames_duplicated": 0,
                "representation": "xyz_quaternion_xyzw_gripper",
                "state_action_representation": "xyz3+quaternion_xyzw4+gripper1",
                "quaternion_canonicalization": "w_nonnegative",
            }
            for key, expected in expected_manifest_values.items():
                if manifest.get(key) != expected:
                    manifest_errors.append(
                        f"{key}={manifest.get(key)!r}, expected {expected!r}"
                    )
            if converted_episodes != meta.total_episodes:
                manifest_errors.append(
                    f"converted_episodes={converted_episodes}, metadata={meta.total_episodes}"
                )
            if output_episodes != meta.total_episodes:
                manifest_errors.append(
                    f"output_episodes={output_episodes}, metadata={meta.total_episodes}"
                )
            if source_episodes != meta.total_episodes:
                manifest_errors.append(
                    f"source_episodes={source_episodes}, output metadata={meta.total_episodes}"
                )
            if manifest.get("output_episode_lengths") != lengths:
                manifest_errors.append("output_episode_lengths do not match metadata")
            expected_image_features = ["observation.images.camera_wrist_left"]
            if tactile_rgb_enabled:
                expected_image_features.append(TACTILE_RGB_KEY)
            if manifest.get("image_features") != expected_image_features:
                manifest_errors.append(
                    f"image_features={manifest.get('image_features')!r}, "
                    f"expected {expected_image_features!r}"
                )

            expected_visual_policy = (
                "wrist_plus_tactile_rgb" if tactile_rgb_enabled else "wrist_rgb_only"
            )
            if manifest.get("visual_input_policy") != expected_visual_policy:
                manifest_errors.append(
                    f"visual_input_policy={manifest.get('visual_input_policy')!r}, "
                    f"expected {expected_visual_policy!r}"
                )
            if manifest.get("wrist_rgb_source_key") != "camera0_rgb":
                manifest_errors.append(
                    f"wrist_rgb_source_key={manifest.get('wrist_rgb_source_key')!r}, "
                    "expected 'camera0_rgb'"
                )

            tactile_info = manifest.get("tactile_inputs", {})
            tactile_enabled = tactile_rgb_enabled or tactile_marker_enabled
            if tactile_info.get("enabled") is not tactile_enabled:
                manifest_errors.append(
                    f"tactile_inputs.enabled={tactile_info.get('enabled')!r}, expected {tactile_enabled}"
                )
            if tactile_info.get("copied_to_lerobot") is not tactile_enabled:
                manifest_errors.append(
                    "tactile_inputs.copied_to_lerobot does not match requested mode"
                )
            if tactile_enabled:
                if tactile_info.get("used_for_training") is not True:
                    manifest_errors.append(
                        "tactile_inputs.used_for_training must be true for tactile mode"
                    )
                if tactile_info.get("schema_version") != TACTILE_SCHEMA_VERSION:
                    manifest_errors.append("tactile_inputs.schema_version mismatch")
                if tactile_info.get("rgb_enabled") is not tactile_rgb_enabled:
                    manifest_errors.append("tactile_inputs.rgb_enabled mismatch")
                if tactile_info.get("marker_enabled") is not tactile_marker_enabled:
                    manifest_errors.append("tactile_inputs.marker_enabled mismatch")
                if tactile_marker_enabled:
                    normalization = tactile_info.get("marker_normalization", {})
                    if normalization.get("target") != "image_size_xy":
                        manifest_errors.append(
                            "tactile marker target normalization must be image_size_xy"
                        )
                    if (
                        normalization.get("image_width"),
                        normalization.get("image_height"),
                    ) != (640, 480):
                        manifest_errors.append(
                            "tactile marker normalization size must be [640,480]"
                        )
            elif tactile_info.get("used_for_training") is not False:
                manifest_errors.append("tactile_inputs.used_for_training must be false")

            if manifest_errors:
                report.fail("Conversion manifest", "; ".join(manifest_errors))
            else:
                report.ok(
                    "Conversion manifest",
                    f"source episodes={source_episodes}, output episodes={converted_episodes}, "
                    "native contiguous 30 Hz",
                )
                if tactile_enabled and not tactile_info.get(
                    "timestamp_is_independently_measured", False
                ):
                    report.warn(
                        "Tactile timestamp provenance",
                        "timestamps are derived from the synchronized episode frame index; "
                        "independent wrist/tactile skew cannot be recovered from this legacy Zarr",
                    )
            if expected_source_episodes is not None and source_episodes != expected_source_episodes:
                report.fail(
                    "Source episode count",
                    f"source episodes={source_episodes}, expected={expected_source_episodes}",
                )
            elif expected_source_episodes is not None:
                report.ok("Source episode count", str(expected_source_episodes))

        video_count = 0
        first_video_paths: dict[str, Path] = {}
        for episode_index in range(meta.total_episodes):
            for key in sorted(required_video_keys):
                relpath = meta.get_video_file_path(episode_index, key)
                video_path = relpath if relpath.is_absolute() else path / relpath
                if not video_path.is_file():
                    report.fail("Video files", f"missing {video_path}")
                    return valid_anchors, source_episodes
                first_video_paths.setdefault(key, video_path)
                video_count += 1
        report.ok("Video files", f"all {video_count} required encoded videos are present")
        try:
            import av

            decoded = []
            for key, video_path in first_video_paths.items():
                with av.open(str(video_path)) as container:
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError(f"no decodable frame in {video_path}")
                    decoded.append(f"{key}={frame.width}x{frame.height}")
            report.ok("Video decoding", ", ".join(decoded))
        except Exception as exc:
            report.fail("Video decoding", repr(exc))
        return valid_anchors, source_episodes
    except Exception as exc:
        report.fail("LeRobot dataset", repr(exc))
        return None, None


def check_norm(
    report: Report,
    path: Path,
    expected_count: int | None,
    tactile_marker_enabled: bool = False,
) -> None:
    if not path.is_file():
        report.fail("Normalization stats", f"missing {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload["norm_stats"]
        errors = []
        scale_warnings = []
        norm_dims = dict(NORM_DIMS)
        if tactile_marker_enabled:
            norm_dims[TACTILE_MARKER_FLOW_KEY] = 2
        for key, dim in norm_dims.items():
            entry = stats.get(key)
            if entry is None:
                errors.append(f"missing {key}")
                continue
            for field in ("mean", "std", "q01", "q99", "min", "max"):
                values = entry.get(field, [])
                if len(values) != dim:
                    errors.append(f"{key}.{field} has dim {len(values)}, expected {dim}")
                elif not all(math.isfinite(float(value)) for value in values):
                    errors.append(f"{key}.{field} contains NaN/Inf")
            if any(len(entry.get(field, [])) != dim for field in ("mean", "std", "q01", "q99", "min", "max")):
                continue

            mean = np.asarray(entry["mean"], dtype=np.float64)
            std = np.asarray(entry["std"], dtype=np.float64)
            q01 = np.asarray(entry["q01"], dtype=np.float64)
            q99 = np.asarray(entry["q99"], dtype=np.float64)
            min_value = np.asarray(entry["min"], dtype=np.float64)
            max_value = np.asarray(entry["max"], dtype=np.float64)

            negative_std = np.flatnonzero(std < 0)
            if negative_std.size:
                errors.append(f"{key}.std is negative at dims {negative_std.tolist()}")

            varying = max_value > min_value
            invalid_dynamic_std = np.flatnonzero(varying & (std <= 0))
            if invalid_dynamic_std.size:
                errors.append(
                    f"{key}.std is non-positive for varying dims "
                    f"{invalid_dynamic_std.tolist()}"
                )

            standardized_extent = np.maximum(
                np.abs((q01 - mean) / (std + 1e-6)),
                np.abs((q99 - mean) / (std + 1e-6)),
            )
            extreme_dims = np.flatnonzero(standardized_extent > 100.0)
            if extreme_dims.size:
                scale_warnings.append(
                    f"{key} has |standardized q01/q99| > 100 at dims "
                    f"{extreme_dims.tolist()}"
                )
        if expected_count is not None and payload.get("count") != expected_count:
            errors.append(f"count={payload.get('count')}, expected valid anchors={expected_count}")
        if errors:
            report.fail("Normalization stats", "; ".join(errors))
        else:
            report.ok("Normalization stats", f"canonical EEF/gripper stats at {path}")
            if scale_warnings:
                report.warn("Normalization scale", "; ".join(scale_warnings))
    except Exception as exc:
        report.fail("Normalization stats", repr(exc))


def check_dataset_transform(
    report: Report,
    dataset_path: Path,
    expected_length: int | None,
    tactile_mode: str = "none",
    dataset_tactile_mode: str | None = None,
) -> None:
    """Run the exact v2 robot mapping and quaternion-relative transform."""

    try:
        from types import SimpleNamespace

        import torch

        from lingbotvla.data.vla_data.base_dataset import VLADataset

        root = Path(__file__).resolve().parents[1]
        tactile_rgb_enabled = tactile_mode in {"rgb", "rgb-marker"}
        tactile_marker_enabled = tactile_mode in {"marker", "rgb-marker"}
        dataset_tactile_mode = dataset_tactile_mode or tactile_mode
        dataset_has_tactile = bool(_tactile_modalities(dataset_tactile_mode))
        data_config = SimpleNamespace(
            joints=[
                "{'arm.position': 14}",
                "{'end.position': 14}",
                "{'effector.position': 2}",
                "{'waist.position': 4}",
                "{'head.position': 2}",
                "{'base.position': 3}",
                "{'hand.position': 12}",
            ],
            cameras=["camera_wrist_left"],
            norm_type=[
                "{'end.position': 'meanstd'}",
                "{'effector.position': 'meanstd'}",
            ],
            norm_stats_file=None,
            tactile_rgb_key=TACTILE_RGB_KEY,
            tactile_marker_key=TACTILE_MARKER_FLOW_KEY,
            tactile_marker_valid_key=TACTILE_MARKER_VALID_KEY,
            tactile_timestamp_key=TACTILE_TIMESTAMP_KEY,
        )
        dataset = VLADataset(
            repo_id=str(dataset_path),
            data_name=(
                "tacthru_umi_v2_tactile"
                if dataset_has_tactile
                else "tacthru_umi_v2"
            ),
            dataset_config=data_config,
            robot_config_root=str(root / "configs/robot_configs"),
            config=None,
            processor=None,
            chunk_size=CHUNK_SIZE,
            do_nomalize=False,
            return_item=True,
            disabled_image_features=True,
            use_future_image=True,
            tactile_rgb_enabled=tactile_rgb_enabled,
            tactile_marker_enabled=tactile_marker_enabled,
            tactile_params={"history_steps": 4, "history_stride": 1},
        )
        if expected_length is not None and len(dataset) != expected_length:
            raise AssertionError(f"dataset length={len(dataset)}, expected valid anchors={expected_length}")
        if len(dataset) == 0:
            raise ValueError("dataset has no complete 50-step native UMI chunk")

        delta_timestamps = dataset.get_delta_timestamps()
        action_features = dataset.feature_transform.org_features["actions"]
        expected_action_offsets = [index / EXPECTED_DATASET_FPS for index in range(CHUNK_SIZE)]
        for feature in action_features:
            actual_offsets = delta_timestamps.get(feature)
            if actual_offsets is None or len(actual_offsets) != CHUNK_SIZE:
                raise AssertionError(
                    f"{feature} has invalid delta timestamps: {actual_offsets!r}"
                )
            for actual, expected in zip(actual_offsets, expected_action_offsets):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
                    raise AssertionError(
                        f"{feature} is not sampled contiguously at 30 Hz: "
                        f"actual={actual_offsets!r}"
                    )

        sample = dataset[0]
        expected_shapes = {
            "observation.state.end.position": (7,),
            "observation.state.effector.position": (1,),
            "action.end.position": (CHUNK_SIZE, 7),
            "action.effector.position": (CHUNK_SIZE, 1),
            "action_is_pad": (CHUNK_SIZE,),
        }
        errors = []
        for key, shape in expected_shapes.items():
            actual = tuple(sample[key].shape)
            if actual != shape:
                errors.append(f"{key}={actual}, expected {shape}")
        if bool(torch.as_tensor(sample["action_is_pad"]).any()):
            errors.append("filtered sample still contains padded actions")
        if tactile_marker_enabled:
            marker = torch.as_tensor(sample.get(TACTILE_MARKER_FLOW_KEY))
            marker_valid = torch.as_tensor(sample.get(TACTILE_MARKER_VALID_KEY))
            marker_pad = torch.as_tensor(sample.get(f"{TACTILE_MARKER_FLOW_KEY}_is_pad"))
            if tuple(marker.shape) != (4, 48, 2):
                errors.append(f"marker history={tuple(marker.shape)}, expected (4,48,2)")
            if tuple(marker_valid.shape) != (4, 48):
                errors.append(
                    f"marker valid history={tuple(marker_valid.shape)}, expected (4,48)"
                )
            if tuple(marker_pad.shape) != (4,):
                errors.append(f"marker temporal pad={tuple(marker_pad.shape)}, expected (4,)")
        future_effective_fps = sample.get("future_video_effective_fps")
        if future_effective_fps is None:
            errors.append("future_video_effective_fps is missing")
        elif not math.isclose(
            float(torch.as_tensor(future_effective_fps).item()),
            EXPECTED_FUTURE_EFFECTIVE_FPS,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            errors.append(
                "future_video_effective_fps="
                f"{float(torch.as_tensor(future_effective_fps).item())}, "
                f"expected {EXPECTED_DATASET_FPS}/49={EXPECTED_FUTURE_EFFECTIVE_FPS}"
            )
        end_action = torch.as_tensor(sample["action.end.position"])
        if not bool(torch.isfinite(end_action).all()):
            errors.append("relative EEF action contains NaN/Inf")
        if end_action.shape == (CHUNK_SIZE, 7):
            torch.testing.assert_close(end_action[0, :3], torch.zeros(3), atol=1e-5, rtol=0)
            torch.testing.assert_close(
                end_action[0, 3:],
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
                atol=1e-5,
                rtol=0,
            )
        if errors:
            report.fail("v2 training transform", "; ".join(errors))
        else:
            report.ok(
                "v2 training transform",
                f"samples={len(dataset)}, EEF=7D quaternion_local, gripper=1D, "
                f"chunk=50 contiguous at 30 Hz, future effective fps="
                f"{EXPECTED_FUTURE_EFFECTIVE_FPS:.9f}, dataset_mode={dataset_tactile_mode}, "
                f"train_mode={tactile_mode}",
            )
    except Exception as exc:
        report.fail("v2 training transform", repr(exc))


def check_relative_math(report: Report) -> None:
    try:
        import torch

        from lingbotvla.data.vla_data.ee_pose_transform import relative_pose_quaternion

        current = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        future = torch.stack(
            [current, torch.tensor([2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])]
        )
        relative = relative_pose_quaternion(future, current, relative_type="quaternion_local")
        torch.testing.assert_close(relative[0, :3], torch.zeros(3))
        torch.testing.assert_close(relative[0, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(relative[1, :3], torch.tensor([1.0, 0.0, 0.0]))
        report.ok("UMI SE(3) math", "v2 quaternion_local self-test passed")
    except Exception as exc:
        report.fail("UMI SE(3) math", repr(exc))


def check_accelerator(report: Report, require_gpu: bool, require_flash: bool) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            report.ok("CUDA", f"{torch.cuda.device_count()} GPU(s): {names}")
        elif require_gpu:
            report.fail("CUDA", "no GPU is visible; use the B200 instance before training")
        else:
            report.warn("CUDA", "no GPU visible; CPU-side preparation can continue")
    except Exception as exc:
        report.fail("CUDA", repr(exc))

    try:
        import flash_attn

        report.ok("flash-attn", getattr(flash_attn, "__version__", "installed"))
    except Exception as exc:
        if require_flash:
            report.fail("flash-attn", repr(exc))
        else:
            report.warn("flash-attn", "not importable yet")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    model = root / "models/lingbot-vla-v2-6b"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--norm", type=Path)
    parser.add_argument("--model", type=Path, default=model)
    parser.add_argument("--tokenizer", type=Path, default=root / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--moge", type=Path, default=root / "models/moge-2-vitb-normal/model.pt")
    parser.add_argument("--morgbd", type=Path, default=model / "depth/model.pt")
    parser.add_argument("--dino-checkpoint", type=Path, default=model / "dino_video/teacher_step_10000.pth")
    parser.add_argument("--dino-config", type=Path, default=model / "dino_video/config.yaml")
    parser.add_argument("--expected-source-episodes", type=int, default=100)
    parser.add_argument(
        "--tactile-mode",
        choices=("none", "rgb", "marker", "rgb-marker"),
        default="none",
        help="Training architecture ablation (encoder branches enabled for this run)",
    )
    parser.add_argument(
        "--dataset-tactile-mode",
        choices=("none", "rgb", "marker", "rgb-marker"),
        default=None,
        help=(
            "Actual converter schema to validate. Leave unset to use --tactile-mode; "
            "for a reusable superset dataset choose rgb-marker."
        ),
    )
    parser.add_argument(
        "--tactile-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/vla/tacthru_umi/tacthru_umi_tactile.yaml",
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-flash", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    dataset_tactile_mode = args.dataset_tactile_mode or args.tactile_mode
    train_modalities = _tactile_modalities(args.tactile_mode)
    dataset_modalities = _tactile_modalities(dataset_tactile_mode)
    if not train_modalities.issubset(dataset_modalities):
        report.fail(
            "Tactile dataset/architecture contract",
            f"train mode={args.tactile_mode} requires {sorted(train_modalities)}, "
            f"but dataset mode={dataset_tactile_mode} provides {sorted(dataset_modalities)}",
        )
    else:
        report.ok(
            "Tactile dataset/architecture contract",
            f"dataset={dataset_tactile_mode}, train={args.tactile_mode}",
        )
    check_tactile_config(report, args.tactile_config.expanduser().resolve(), args.tactile_mode)
    check_python(report)
    check_model(report, args.model.expanduser().resolve())
    check_qwen(report, args.tokenizer.expanduser().resolve())
    check_teacher_file(report, "MoGe teacher", args.moge.expanduser().resolve(), 400_000_000)
    check_teacher_file(report, "LingBot-Depth teacher", args.morgbd.expanduser().resolve(), 1_300_000_000)
    check_teacher_file(report, "DINO-Video teacher", args.dino_checkpoint.expanduser().resolve(), 1_390_000_000)
    check_teacher_file(report, "DINO-Video config", args.dino_config.expanduser().resolve(), 100)
    check_relative_math(report)

    valid_anchors = None
    if args.dataset is not None:
        valid_anchors, _ = check_dataset(
            report,
            args.dataset.expanduser().resolve(),
            args.expected_source_episodes,
            dataset_tactile_mode,
        )
        check_dataset_transform(
            report,
            args.dataset.expanduser().resolve(),
            valid_anchors,
            args.tactile_mode,
            dataset_tactile_mode,
        )
    if args.norm is not None:
        check_norm(
            report,
            args.norm.expanduser().resolve(),
            valid_anchors,
            tactile_marker_enabled="marker" in dataset_modalities,
        )

    check_accelerator(report, args.require_gpu, args.require_flash)
    report.display(args.json)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
