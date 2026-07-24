#!/usr/bin/env python3
"""Convert TacThru UMI Zarr data to a LingBot-VLA v2 LeRobot dataset.

The conversion preserves the source dataset's native 30 Hz timeline and
episode boundaries: one TacThru episode becomes one LeRobot episode, and every
source frame is written exactly once in its original order.  State and action
are stored as the same absolute 8D pose
``xyz + quaternion_xyzw + gripper``; the v2 robot config is responsible for
converting future absolute poses to local relative actions.

The default command remains the original wrist-RGB-only conversion.  Tactile
RGB and marker flow are copied only when the corresponding explicit
``--include-tactile-*`` flags are supplied, allowing a versioned superset
dataset to support fair ablations without changing the wrist-only baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingbotvla.tactile.schema import TACTILE_SCHEMA_VERSION, get_tactile_keys
from lingbotvla.tactile.transforms import marker_flow_to_image_size_xy, sanitize_marker_flow


SOURCE_FPS = 30
DATASET_FPS = SOURCE_FPS
ACTION_CHUNK_SIZE = 50
CONVERTER_VERSION = 4
TACTILE_CONVERTER_VERSION = 5
MANIFEST_NAME = "tacthru_umi_v2_conversion.json"
ROBOT_TYPE = "realman_tacthru_umi_v2"

REQUIRED_KEYS = (
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
)

IMAGE_FEATURES = (
    "observation.images.camera_wrist_left",
)
TACTILE_TIMESTAMP_KEY = "observation.tactile.timestamp"
EXCLUDED_TACTILE_SOURCE_KEYS = (
    "tacthru_l_rgb",
    "tacthru_r_rgb",
    "tacthru_l_marker",
    "tacthru_r_marker",
)

STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "gripper_width",
)


def source_signature(path: Path) -> dict[str, Any]:
    """Return a stable-enough signature for safe converted-dataset reuse."""

    stat = path.stat()
    if path.is_file():
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        file_count = 1
    else:
        size = 0
        mtime_ns = int(stat.st_mtime_ns)
        file_count = 0
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            child_stat = child.stat()
            size += int(child_stat.st_size)
            mtime_ns = max(mtime_ns, int(child_stat.st_mtime_ns))
            file_count += 1

    return {
        "source": str(path),
        "source_type": "directory" if path.is_dir() else "file",
        "source_size": size,
        "source_mtime_ns": mtime_ns,
        "source_file_count": file_count,
    }


@contextmanager
def open_zarr_group(path: Path) -> Iterator[Any]:
    """Open either a directory-backed Zarr group or a ``.zarr.zip`` file."""

    if path.is_dir():
        yield zarr.open_group(str(path), mode="r")
        return

    zip_store_cls = getattr(zarr, "ZipStore", None)
    if zip_store_cls is None:
        zip_store_cls = zarr.storage.ZipStore
    store = zip_store_cls(str(path), mode="r")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        store.close()


def _validate_image_array(array: Any, key: str) -> tuple[int, int, int]:
    shape = tuple(array.shape)
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"{key} must have shape (N, H, W, 3), got {shape}")
    if np.dtype(array.dtype) != np.dtype(np.uint8):
        raise ValueError(f"{key} must have dtype uint8, got {array.dtype}")
    return tuple(int(value) for value in shape[1:])


def _marker_axis_stats(marker_flow: np.ndarray) -> dict[str, Any]:
    """Return finite-value QA statistics for normalized marker flow."""

    flow, valid = sanitize_marker_flow(marker_flow)
    axis_values = [flow[..., axis][valid] for axis in range(2)]
    names = ("x", "y")
    result: dict[str, Any] = {
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "invalid_points": int(valid.size - np.count_nonzero(valid)),
    }
    for name, values in zip(names, axis_values):
        if values.size == 0:
            result[name] = None
            continue
        result[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q01": float(np.quantile(values, 0.01)),
            "q50": float(np.quantile(values, 0.50)),
            "q99": float(np.quantile(values, 0.99)),
        }
    return result


def inspect_source(
    root: Any,
    *,
    include_tactile_rgb: bool = False,
    include_tactile_marker: bool = False,
    tactile_side: str = "left",
) -> dict[str, Any]:
    """Validate the TacThru replay-buffer schema and return source metadata."""

    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError("Expected Zarr groups data/* and meta/episode_ends")

    data = root["data"]
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing required TacThru fields: {missing}")

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or len(episode_ends) == 0:
        raise ValueError(f"Invalid episode_ends shape: {episode_ends.shape}")

    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    if np.any(episode_lengths <= 0):
        raise ValueError("episode_ends must be strictly increasing")
    total_frames = int(episode_ends[-1])
    for key in REQUIRED_KEYS:
        if len(data[key]) != total_frames:
            raise ValueError(f"{key} has {len(data[key])} frames, expected {total_frames}")

    wrist_shape = _validate_image_array(data["camera0_rgb"], "camera0_rgb")
    if tuple(data["robot0_eef_pos"].shape[1:]) != (3,):
        raise ValueError("robot0_eef_pos must have shape (N, 3)")
    if tuple(data["robot0_eef_rot_axis_angle"].shape[1:]) != (3,):
        raise ValueError("robot0_eef_rot_axis_angle must have shape (N, 3)")
    if tuple(data["robot0_gripper_width"].shape[1:]) != (1,):
        raise ValueError("robot0_gripper_width must have shape (N, 1)")

    for key in (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    ):
        values = np.asarray(data[key][:])
        if not np.all(np.isfinite(values)):
            bad_count = int(values.size - np.count_nonzero(np.isfinite(values)))
            raise ValueError(f"{key} contains {bad_count} non-finite values")

    tactile_keys = get_tactile_keys(tactile_side)
    tactile_rgb_shape = None
    marker_shape = None
    marker_stats = None
    target_marker_stats = None
    if include_tactile_rgb:
        if tactile_keys.source_rgb not in data:
            raise ValueError(
                f"Missing requested tactile RGB field: {tactile_keys.source_rgb}"
            )
        if len(data[tactile_keys.source_rgb]) != total_frames:
            raise ValueError(
                f"{tactile_keys.source_rgb} has {len(data[tactile_keys.source_rgb])} "
                f"frames, expected {total_frames}"
            )
        tactile_rgb_shape = _validate_image_array(
            data[tactile_keys.source_rgb], tactile_keys.source_rgb
        )
    if include_tactile_marker:
        if tactile_keys.source_marker_flow not in data:
            raise ValueError(
                f"Missing requested tactile marker field: {tactile_keys.source_marker_flow}"
            )
        marker_array = data[tactile_keys.source_marker_flow]
        marker_shape = tuple(int(value) for value in marker_array.shape[1:])
        if len(marker_array) != total_frames or len(marker_shape) != 2 or marker_shape[-1] != 2:
            raise ValueError(
                f"{tactile_keys.source_marker_flow} must have shape (N, M, 2) "
                f"with N={total_frames}, got {tuple(marker_array.shape)}"
            )

    return {
        "source_episodes": int(len(episode_ends)),
        "source_frames": total_frames,
        "episode_ends": episode_ends,
        "episode_lengths": episode_lengths.astype(np.int64, copy=False),
        "wrist_image_shape": wrist_shape,
        "tactile_keys": tactile_keys,
        "tactile_rgb_image_shape": tactile_rgb_shape,
        "tactile_marker_shape": marker_shape,
        "tactile_marker_stats": marker_stats,
        "tactile_target_marker_stats": target_marker_stats,
        "excluded_tactile_source_keys": [
            key for key in EXCLUDED_TACTILE_SOURCE_KEYS if key in data
        ],
    }


def make_episode_plan(
    episode_ends: np.ndarray,
    num_source_episodes: int,
) -> list[dict[str, int]]:
    """Map each source episode to one contiguous output episode."""

    plan: list[dict[str, int]] = []
    source_start = 0
    for source_episode_index in range(num_source_episodes):
        source_end = int(episode_ends[source_episode_index])
        length = source_end - source_start
        if length <= 0:
            raise ValueError(f"Source episode {source_episode_index} has no frames")
        plan.append(
            {
                "source_episode_index": source_episode_index,
                "source_start": source_start,
                "source_end": source_end,
                "length": length,
            }
        )
        source_start = source_end

    return plan


def rotvec_to_canonical_quaternion_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to normalized canonical quaternions in xyzw order."""

    rotvec = np.asarray(rotvec, dtype=np.float64)
    if rotvec.shape[-1:] != (3,):
        raise ValueError(f"Expected rotation vectors with last dimension 3, got {rotvec.shape}")
    quaternion = Rotation.from_rotvec(rotvec).as_quat()
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion[quaternion[..., 3] < 0] *= -1.0
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Rotation conversion produced a zero-norm quaternion")
    quaternion /= norms
    return quaternion.astype(np.float32)


def build_pose8(position: np.ndarray, rotvec: np.ndarray, gripper_width: np.ndarray) -> np.ndarray:
    """Build finite float32 ``xyz + quaternion_xyzw + gripper`` rows."""

    position = np.asarray(position, dtype=np.float32)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    gripper_width = np.asarray(gripper_width, dtype=np.float32)
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"Expected position shape (N, 3), got {position.shape}")
    if rotvec.shape != position.shape:
        raise ValueError(f"Rotation-vector shape {rotvec.shape} does not match {position.shape}")
    if gripper_width.shape != (len(position), 1):
        raise ValueError(f"Expected gripper shape ({len(position)}, 1), got {gripper_width.shape}")

    pose = np.concatenate([position, rotvec_to_canonical_quaternion_xyzw(rotvec), gripper_width], axis=-1).astype(
        np.float32, copy=False
    )
    if pose.shape != (len(position), 8):
        raise RuntimeError(f"Internal pose shape error: {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("Converted pose contains non-finite values")
    return pose


def build_features(
    wrist_shape: tuple[int, int, int],
    *,
    include_tactile_rgb: bool = False,
    include_tactile_marker: bool = False,
    tactile_side: str = "left",
    tactile_rgb_shape: tuple[int, int, int] | None = None,
    tactile_marker_shape: tuple[int, int] | None = None,
) -> dict:
    """Build the LeRobot v3 feature declaration."""

    wrist_height, wrist_width, _ = wrist_shape
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(STATE_NAMES),
        },
        IMAGE_FEATURES[0]: {
            "dtype": "video",
            "shape": (3, wrist_height, wrist_width),
            "names": ["channels", "height", "width"],
        },
    }
    tactile_keys = get_tactile_keys(tactile_side)
    if include_tactile_rgb:
        if tactile_rgb_shape is None:
            raise ValueError("tactile_rgb_shape is required when tactile RGB is enabled")
        tactile_height, tactile_width, tactile_channels = tactile_rgb_shape
        if tactile_channels != 3:
            raise ValueError(f"tactile RGB must have 3 channels, got {tactile_rgb_shape}")
        features[tactile_keys.rgb] = {
            "dtype": "video",
            "shape": (3, tactile_height, tactile_width),
            "names": ["channels", "height", "width"],
        }
    if include_tactile_marker:
        if tactile_marker_shape is None or tactile_marker_shape[-1] != 2:
            raise ValueError(
                "tactile_marker_shape=(marker_count, 2) is required when marker is enabled"
            )
        marker_count = int(tactile_marker_shape[0])
        features[tactile_keys.marker_flow] = {
            "dtype": "float32",
            "shape": (marker_count, 2),
            "names": None,
        }
        features[tactile_keys.marker_valid] = {
            "dtype": "bool",
            "shape": (marker_count,),
            "names": None,
        }
    if include_tactile_rgb or include_tactile_marker:
        features[tactile_keys.timestamp] = {
            "dtype": "float64",
            "shape": (1,),
            "names": ["seconds"],
        }
    return features


def validate_output(
    output: Path,
    expected_episode_lengths: list[int],
    expected_frames: int,
    *,
    include_tactile_rgb: bool = False,
    include_tactile_marker: bool = False,
    tactile_side: str = "left",
) -> dict[str, Any]:
    """Validate counts, feature schema, and all referenced video files."""

    meta = LeRobotDatasetMetadata(repo_id=output.name, root=output)
    errors: list[str] = []
    expected_episodes = len(expected_episode_lengths)

    if meta.total_episodes != expected_episodes:
        errors.append(f"episodes={meta.total_episodes}, expected {expected_episodes}")
    if meta.total_frames != expected_frames:
        errors.append(f"frames={meta.total_frames}, expected {expected_frames}")
    if meta.fps != DATASET_FPS:
        errors.append(f"fps={meta.fps}, expected {DATASET_FPS}")

    tactile_keys = get_tactile_keys(tactile_side)
    required_features = set(IMAGE_FEATURES + ("observation.state", "action"))
    video_features = list(IMAGE_FEATURES)
    if include_tactile_rgb:
        required_features.add(tactile_keys.rgb)
        video_features.append(tactile_keys.rgb)
    if include_tactile_marker:
        required_features.update((tactile_keys.marker_flow, tactile_keys.marker_valid))
    if include_tactile_rgb or include_tactile_marker:
        required_features.add(tactile_keys.timestamp)
    missing_features = required_features - set(meta.features)
    if missing_features:
        errors.append(f"missing features: {sorted(missing_features)}")

    unexpected_image_features = {
        "observation.images.camera_top",
        "observation.images.camera_wrist_right",
    } & set(meta.features)
    if unexpected_image_features:
        errors.append(
            "unexpected non-wrist visual features: "
            f"{sorted(unexpected_image_features)}"
        )

    for key in ("observation.state", "action"):
        feature = meta.features.get(key)
        if feature is not None and tuple(feature.get("shape", ())) != (8,):
            errors.append(f"{key} shape={feature.get('shape')}, expected [8]")

    if len(getattr(meta, "episodes", [])) == expected_episodes:
        actual_lengths = [int(episode["length"]) for episode in meta.episodes]
        if actual_lengths != expected_episode_lengths:
            errors.append(
                f"episode lengths do not match source episode plan: actual={actual_lengths}, "
                f"expected={expected_episode_lengths}"
            )

    missing_video_paths: list[str] = []
    if not missing_features:
        for episode_index in range(meta.total_episodes):
            for video_key in video_features:
                video_path = output / meta.get_video_file_path(episode_index, video_key)
                if not video_path.is_file():
                    missing_video_paths.append(str(video_path))
    if missing_video_paths:
        preview = missing_video_paths[:5]
        errors.append(f"missing {len(missing_video_paths)} referenced video files; first paths: {preview}")

    if errors:
        raise RuntimeError("Converted LeRobot dataset failed validation: " + "; ".join(errors))

    return {
        "output_episodes": meta.total_episodes,
        "output_frames": meta.total_frames,
        "output_fps": meta.fps,
        "output_episode_lengths": expected_episode_lengths,
        "features": sorted(meta.features),
    }


def _requested_manifest(
    *,
    signature: dict[str, Any],
    task: str,
    repo_id: str,
    source_episodes: int,
    source_frames: int,
    plan: list[dict[str, int]],
    wrist_shape: tuple[int, int, int],
    excluded_tactile_source_keys: list[str],
    include_tactile_rgb: bool = False,
    include_tactile_marker: bool = False,
    tactile_side: str = "left",
    tactile_rgb_shape: tuple[int, int, int] | None = None,
    tactile_marker_shape: tuple[int, int] | None = None,
    tactile_marker_stats: dict[str, Any] | None = None,
    tactile_target_marker_stats: dict[str, Any] | None = None,
    marker_input_space: str = "normalized",
    marker_image_width: int = 640,
    marker_image_height: int = 480,
) -> dict[str, Any]:
    episode_lengths = [item["length"] for item in plan]
    output_frames = int(sum(episode_lengths))
    valid_anchors = int(sum(max(0, episode_length - ACTION_CHUNK_SIZE + 1) for episode_length in episode_lengths))
    tactile_enabled = bool(include_tactile_rgb or include_tactile_marker)
    tactile_keys = get_tactile_keys(tactile_side)
    image_features = list(IMAGE_FEATURES)
    if include_tactile_rgb:
        image_features.append(tactile_keys.rgb)
    visual_input_policy = "wrist_rgb_only" if not include_tactile_rgb else "wrist_plus_tactile_rgb"
    tactile_source_keys = []
    if include_tactile_rgb:
        tactile_source_keys.append(tactile_keys.source_rgb)
    if include_tactile_marker:
        tactile_source_keys.append(tactile_keys.source_marker_flow)
    if tactile_enabled:
        source_normalization = {
            "normalized": "image_size_xy",
            "pixel": "pixel_delta_xy",
            "legacy_400_normalized": "legacy_uniform_400",
        }[marker_input_space]
        tactile_inputs = {
            "source_keys_present": sorted(excluded_tactile_source_keys),
            "enabled": True,
            "copied_to_lerobot": True,
            "used_for_training": True,
            "schema_version": TACTILE_SCHEMA_VERSION,
            "side": tactile_side,
            "rgb_enabled": bool(include_tactile_rgb),
            "marker_enabled": bool(include_tactile_marker),
            "requested_source_keys": tactile_source_keys,
            "rgb_key": tactile_keys.rgb if include_tactile_rgb else None,
            "rgb_shape_hwc": list(tactile_rgb_shape) if tactile_rgb_shape else None,
            "marker_flow_key": tactile_keys.marker_flow if include_tactile_marker else None,
            "marker_valid_key": tactile_keys.marker_valid if include_tactile_marker else None,
            "marker_shape": list(tactile_marker_shape) if tactile_marker_shape else None,
            "timestamp_key": tactile_keys.timestamp,
            "timestamp_source": "episode_frame_index/output_fps",
            "timestamp_is_independently_measured": False,
            "marker_input_space": marker_input_space if include_tactile_marker else None,
            "marker_normalization": (
                {
                    "source": source_normalization,
                    "target": "image_size_xy",
                    "version": 1,
                    "formula": "[dx / width * 2, dy / height * 2]",
                    "coordinate_space": "marker_tracker_frame",
                    "image_width": int(marker_image_width),
                    "image_height": int(marker_image_height),
                    "applied_by_converter": marker_input_space != "normalized",
                    "correction": (
                        "none_already_normalized"
                        if marker_input_space == "normalized"
                        else (
                            "normalize_pixel_delta"
                            if marker_input_space == "pixel"
                            else "rescale_legacy_400_to_image_size_xy"
                        )
                    ),
                }
                if include_tactile_marker
                else None
            ),
            "source_marker_stats": tactile_marker_stats if include_tactile_marker else None,
            "target_marker_stats": (
                tactile_target_marker_stats if include_tactile_marker else None
            ),
            "reason": "explicit tactile superset conversion requested",
        }
    else:
        # Keep the v4 wrist-only manifest schema unchanged so existing
        # converted datasets can still be safely reused.
        tactile_inputs = {
            "source_keys_present": sorted(excluded_tactile_source_keys),
            "enabled": False,
            "copied_to_lerobot": False,
            "used_for_training": False,
            "reason": "wrist-camera-only baseline requested; tactile RGB and marker flow are excluded",
        }
    return {
        "converter": "tacthru_umi_v2_native_30hz",
        "converter_version": (
            TACTILE_CONVERTER_VERSION if tactile_enabled else CONVERTER_VERSION
        ),
        **signature,
        "task": task,
        "repo_id": repo_id,
        "robot_type": ROBOT_TYPE,
        "source_fps": SOURCE_FPS,
        "output_fps": DATASET_FPS,
        "episode_mapping": "one_source_episode_to_one_output_episode",
        "frame_sampling": "contiguous_stride_1",
        "frames_dropped": 0,
        "frames_duplicated": 0,
        "source_episodes": source_episodes,
        "source_frames": source_frames,
        "output_episodes": len(plan),
        "converted_episodes": len(plan),
        "output_frames": output_frames,
        "frames": output_frames,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "valid_anchors": valid_anchors,
        "output_episode_lengths": episode_lengths,
        "representation": "xyz_quaternion_xyzw_gripper",
        "state_action_representation": "xyz3+quaternion_xyzw4+gripper1",
        "quaternion_canonicalization": "w_nonnegative",
        "visual_input_policy": visual_input_policy,
        "image_features": image_features,
        "wrist_rgb_source_key": "camera0_rgb",
        "wrist_rgb_image_shape_hwc": list(wrist_shape),
        "tactile_inputs": tactile_inputs,
    }


def _format_manifest_mismatches(existing: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    return [
        f"{key}: existing={existing.get(key)!r}, requested={value!r}"
        for key, value in requested.items()
        if existing.get(key) != value
    ]


def convert(args: argparse.Namespace) -> None:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(f"Source and output paths must not overlap: source={source}, output={output}")

    include_tactile_rgb = bool(getattr(args, "include_tactile_rgb", False))
    include_tactile_marker = bool(getattr(args, "include_tactile_marker", False))
    tactile_side = str(getattr(args, "tactile_side", "left"))
    marker_input_space = str(getattr(args, "marker_input_space", "normalized"))
    marker_image_width = int(getattr(args, "marker_image_width", 640))
    marker_image_height = int(getattr(args, "marker_image_height", 480))
    if marker_input_space not in {"normalized", "pixel", "legacy_400_normalized"}:
        raise ValueError(f"Unsupported marker_input_space: {marker_input_space!r}")
    if marker_image_width <= 0 or marker_image_height <= 0:
        raise ValueError("marker image width/height must be positive")
    tactile_keys = get_tactile_keys(tactile_side)

    with open_zarr_group(source) as root:
        source_info = inspect_source(
            root,
            include_tactile_rgb=include_tactile_rgb,
            include_tactile_marker=include_tactile_marker,
            tactile_side=tactile_side,
        )
        available_source_episodes = source_info["source_episodes"]
        num_source_episodes = available_source_episodes
        if args.max_source_episodes is not None:
            num_source_episodes = min(num_source_episodes, args.max_source_episodes)

        episode_ends = source_info["episode_ends"]
        source_frames = int(episode_ends[num_source_episodes - 1])
        if include_tactile_marker:
            source_marker = np.asarray(
                root["data"][tactile_keys.source_marker_flow][:source_frames],
                dtype=np.float32,
            )
            source_info["tactile_marker_stats"] = _marker_axis_stats(source_marker)
            target_marker = marker_flow_to_image_size_xy(
                source_marker,
                input_space=marker_input_space,
                image_width=marker_image_width,
                image_height=marker_image_height,
            )
            source_info["tactile_target_marker_stats"] = _marker_axis_stats(
                target_marker
            )
        plan = make_episode_plan(episode_ends, num_source_episodes)
        output_episode_lengths = [item["length"] for item in plan]
        output_frames = int(sum(output_episode_lengths))
        if output_frames != source_frames:
            raise RuntimeError(
                f"Conversion lost or duplicated frames: output={output_frames}, source={source_frames}"
            )

        signature = source_signature(source)
        repo_id = args.repo_id or output.name
        requested_manifest = _requested_manifest(
            signature=signature,
            task=args.task,
            repo_id=repo_id,
            source_episodes=num_source_episodes,
            source_frames=source_frames,
            plan=plan,
            wrist_shape=source_info["wrist_image_shape"],
            excluded_tactile_source_keys=source_info["excluded_tactile_source_keys"],
            include_tactile_rgb=include_tactile_rgb,
            include_tactile_marker=include_tactile_marker,
            tactile_side=tactile_side,
            tactile_rgb_shape=source_info["tactile_rgb_image_shape"],
            tactile_marker_shape=source_info["tactile_marker_shape"],
            tactile_marker_stats=source_info["tactile_marker_stats"],
            tactile_target_marker_stats=source_info["tactile_target_marker_stats"],
            marker_input_space=marker_input_space,
            marker_image_width=marker_image_width,
            marker_image_height=marker_image_height,
        )

        print(
            json.dumps(
                {
                    "check_only": bool(args.check_only),
                    "available_source_episodes": available_source_episodes,
                    **requested_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.check_only:
            return

        if output.exists():
            if not args.overwrite:
                manifest_path = output / MANIFEST_NAME
                if (output / "meta/info.json").is_file() and manifest_path.is_file():
                    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    mismatches = _format_manifest_mismatches(existing_manifest, requested_manifest)
                    if not mismatches:
                        result = validate_output(
                            output,
                            output_episode_lengths,
                            output_frames,
                            include_tactile_rgb=include_tactile_rgb,
                            include_tactile_marker=include_tactile_marker,
                            tactile_side=tactile_side,
                        )
                        print(
                            json.dumps(
                                {"reused": True, "output": str(output), **result},
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        return
                    mismatch_text = "; ".join(mismatches)
                else:
                    mismatch_text = f"missing meta/info.json or {MANIFEST_NAME}"
                raise FileExistsError(
                    f"Existing converted dataset does not match this request: {mismatch_text}. "
                    "Use --overwrite to rebuild it."
                )
            shutil.rmtree(output)

        output.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            fps=DATASET_FPS,
            robot_type=ROBOT_TYPE,
            features=build_features(
                source_info["wrist_image_shape"],
                include_tactile_rgb=include_tactile_rgb,
                include_tactile_marker=include_tactile_marker,
                tactile_side=tactile_side,
                tactile_rgb_shape=source_info["tactile_rgb_image_shape"],
                tactile_marker_shape=source_info["tactile_marker_shape"],
            ),
            use_videos=True,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,
        )

        data = root["data"]
        for source_episode in tqdm(plan, desc="Converting native 30 Hz episodes"):
            source_slice = slice(
                source_episode["source_start"],
                source_episode["source_end"],
            )
            positions = np.asarray(data["robot0_eef_pos"][source_slice], dtype=np.float32)
            rotvecs = np.asarray(data["robot0_eef_rot_axis_angle"][source_slice], dtype=np.float64)
            gripper = np.asarray(data["robot0_gripper_width"][source_slice], dtype=np.float32)
            poses = build_pose8(positions, rotvecs, gripper)
            wrist_images = np.asarray(data["camera0_rgb"][source_slice], dtype=np.uint8)
            tactile_images = (
                np.asarray(data[tactile_keys.source_rgb][source_slice], dtype=np.uint8)
                if include_tactile_rgb
                else None
            )
            marker_flow = None
            marker_valid = None
            if include_tactile_marker:
                marker_flow = np.asarray(
                    data[tactile_keys.source_marker_flow][source_slice], dtype=np.float32
                )
                marker_flow = marker_flow_to_image_size_xy(
                    marker_flow,
                    input_space=marker_input_space,
                    image_width=marker_image_width,
                    image_height=marker_image_height,
                )
                marker_flow, marker_valid = sanitize_marker_flow(marker_flow)

            expected_length = source_episode["length"]
            actual_lengths = {
                "pose": len(poses),
                "wrist": len(wrist_images),
            }
            if tactile_images is not None:
                actual_lengths["tactile_rgb"] = len(tactile_images)
            if marker_flow is not None:
                actual_lengths["marker"] = len(marker_flow)
            if any(length != expected_length for length in actual_lengths.values()):
                raise RuntimeError(
                    "Contiguous slice length mismatch for "
                    f"source episode {source_episode['source_episode_index']}: "
                    f"actual={actual_lengths}, "
                    f"expected={expected_length}"
                )

            for frame_offset in range(expected_length):
                state = poses[frame_offset]
                frame = {
                    "observation.state": state,
                    "action": state.copy(),
                    IMAGE_FEATURES[0]: wrist_images[frame_offset],
                    "task": args.task,
                }
                if tactile_images is not None:
                    frame[tactile_keys.rgb] = tactile_images[frame_offset]
                if marker_flow is not None and marker_valid is not None:
                    frame[tactile_keys.marker_flow] = marker_flow[frame_offset]
                    frame[tactile_keys.marker_valid] = marker_valid[frame_offset]
                if include_tactile_rgb or include_tactile_marker:
                    frame[tactile_keys.timestamp] = np.asarray(
                        [frame_offset / SOURCE_FPS], dtype=np.float64
                    )
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=not args.serial_video_encoding)

        dataset.finalize()

    result = validate_output(
        output,
        output_episode_lengths,
        output_frames,
        include_tactile_rgb=include_tactile_rgb,
        include_tactile_marker=include_tactile_marker,
        tactile_side=tactile_side,
    )
    manifest = {
        **requested_manifest,
        "output": str(output),
        **result,
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="TacThru .zarr directory or .zarr.zip file")
    parser.add_argument("output", type=Path, help="Output local LeRobot v3 directory")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id (defaults to output name)")
    parser.add_argument("--task", default="Pull the tissue", help="Natural-language task prompt")
    parser.add_argument(
        "--max-source-episodes",
        type=int,
        default=None,
        help="Convert only a prefix of source episodes for testing",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--serial-video-encoding", action="store_true")
    parser.add_argument(
        "--include-tactile-rgb",
        action="store_true",
        help="Copy the selected TacThru tactile RGB stream into the superset dataset",
    )
    parser.add_argument(
        "--include-tactile-marker",
        action="store_true",
        help="Copy marker flow plus a per-marker validity mask into the superset dataset",
    )
    parser.add_argument(
        "--tactile-side",
        choices=("left", "right"),
        default="left",
        help="TacThru tactile side selected by the explicit include flags",
    )
    parser.add_argument(
        "--marker-input-space",
        choices=("normalized", "pixel", "legacy_400_normalized"),
        default="normalized",
        help=(
            "Coordinate convention of the source marker array. Existing TacThru Zarr files "
            "already contain normalized flow; use pixel only for raw dx/dy sources."
        ),
    )
    parser.add_argument("--marker-image-width", type=int, default=640)
    parser.add_argument("--marker-image-height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate source and print the conversion plan")
    args = parser.parse_args()

    if args.max_source_episodes is not None and args.max_source_episodes <= 0:
        parser.error("--max-source-episodes must be positive")
    if args.image_writer_threads < 0:
        parser.error("--image-writer-threads must be non-negative")
    if args.batch_encoding_size <= 0:
        parser.error("--batch-encoding-size must be positive")
    if args.marker_image_width <= 0 or args.marker_image_height <= 0:
        parser.error("--marker-image-width/--marker-image-height must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
