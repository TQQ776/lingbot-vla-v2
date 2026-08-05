#!/usr/bin/env python3
"""Convert TacThru UMI Zarr data to a LingBot-VLA v2 LeRobot dataset.

The conversion preserves the source dataset's native 30 Hz timeline and
episode boundaries: one TacThru episode becomes one LeRobot episode, and every
source frame is written exactly once in its original order.  State and action
are stored as the same absolute 8D pose
``xyz + quaternion_xyzw + gripper``; the v2 robot config is responsible for
converting future absolute poses to local relative actions.

The default remains wrist-RGB-only.  ``--include-tactile`` additionally writes
TacThru RGB and canonical marker position/reference/valid fields for the
minimal VTLA input path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SOURCE_FPS = 30
DATASET_FPS = SOURCE_FPS
ACTION_CHUNK_SIZE = 50
CONVERTER_VERSION = 5
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
TACTILE_RGB_FEATURE = "observation.images.tactile_left"
MARKER_POSITIONS_FEATURE = "observation.tactile.marker_positions_left"
MARKER_REFERENCE_FEATURE = "observation.tactile.marker_reference_left"
MARKER_VALID_FEATURE = "observation.tactile.marker_valid_left"
EXCLUDED_TACTILE_SOURCE_KEYS = (
    "tacthru_l_rgb",
    "tacthru_r_rgb",
    "tacthru_l_marker",
    "tacthru_r_marker",
)


def _lerobot_dataset_classes():
    """Import LeRobot only when output creation or validation needs it."""

    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Full conversion requires the training environment's lerobot package; "
            "--check-only can run without it"
        ) from exc
    return LeRobotDataset, LeRobotDatasetMetadata

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


def inspect_source(root: Any) -> dict[str, Any]:
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

    return {
        "source_episodes": int(len(episode_ends)),
        "source_frames": total_frames,
        "episode_ends": episode_ends,
        "episode_lengths": episode_lengths.astype(np.int64, copy=False),
        "wrist_image_shape": wrist_shape,
        "excluded_tactile_source_keys": [
            key for key in EXCLUDED_TACTILE_SOURCE_KEYS if key in data
        ],
    }


def inspect_tactile_source(root: Any) -> dict[str, Any]:
    """Validate the left TacThru RGB/marker-flow arrays used by VTLA."""

    data = root["data"]
    required = ("tacthru_l_rgb", "tacthru_l_marker")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"--include-tactile requires source fields: {missing}")
    rgb_shape = _validate_image_array(data["tacthru_l_rgb"], "tacthru_l_rgb")
    marker_shape = tuple(int(value) for value in data["tacthru_l_marker"].shape)
    if len(marker_shape) != 3 or marker_shape[-1] != 2:
        raise ValueError(
            f"tacthru_l_marker must have shape (N,M,2), got {marker_shape}"
        )
    total_frames = int(root["meta/episode_ends"][-1])
    if len(data["tacthru_l_rgb"]) != total_frames or marker_shape[0] != total_frames:
        raise ValueError("Left tactile RGB/marker timelines must match episode_ends")
    return {
        "tactile_rgb_shape": rgb_shape,
        "num_markers": marker_shape[1],
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
    tactile_info: dict[str, Any] | None = None,
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
    if tactile_info is not None:
        tactile_height, tactile_width, _ = tactile_info["tactile_rgb_shape"]
        num_markers = int(tactile_info["num_markers"])
        features.update(
            {
                TACTILE_RGB_FEATURE: {
                    "dtype": "video",
                    "shape": (3, tactile_height, tactile_width),
                    "names": ["channels", "height", "width"],
                },
                MARKER_POSITIONS_FEATURE: {
                    "dtype": "float32",
                    "shape": (num_markers, 2),
                    "names": None,
                },
                MARKER_REFERENCE_FEATURE: {
                    "dtype": "float32",
                    "shape": (num_markers, 2),
                    "names": None,
                },
                MARKER_VALID_FEATURE: {
                    "dtype": "bool",
                    "shape": (num_markers,),
                    "names": [f"marker_{index:03d}" for index in range(num_markers)],
                },
            }
        )
    return features


def validate_output(
    output: Path,
    expected_episode_lengths: list[int],
    expected_frames: int,
    include_tactile: bool = False,
) -> dict[str, Any]:
    """Validate counts, feature schema, and all referenced video files."""

    _, LeRobotDatasetMetadata = _lerobot_dataset_classes()
    meta = LeRobotDatasetMetadata(repo_id=output.name, root=output)
    errors: list[str] = []
    expected_episodes = len(expected_episode_lengths)

    if meta.total_episodes != expected_episodes:
        errors.append(f"episodes={meta.total_episodes}, expected {expected_episodes}")
    if meta.total_frames != expected_frames:
        errors.append(f"frames={meta.total_frames}, expected {expected_frames}")
    if meta.fps != DATASET_FPS:
        errors.append(f"fps={meta.fps}, expected {DATASET_FPS}")

    required_features = set(IMAGE_FEATURES + ("observation.state", "action"))
    video_features = list(IMAGE_FEATURES)
    if include_tactile:
        required_features.update(
            {
                TACTILE_RGB_FEATURE,
                MARKER_POSITIONS_FEATURE,
                MARKER_REFERENCE_FEATURE,
                MARKER_VALID_FEATURE,
            }
        )
        video_features.append(TACTILE_RGB_FEATURE)
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
    include_tactile: bool = False,
    tactile_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episode_lengths = [item["length"] for item in plan]
    output_frames = int(sum(episode_lengths))
    valid_anchors = int(sum(max(0, episode_length - ACTION_CHUNK_SIZE + 1) for episode_length in episode_lengths))
    return {
        "converter": "tacthru_umi_v2_native_30hz",
        "converter_version": CONVERTER_VERSION,
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
        "visual_input_policy": "wrist_plus_tactile_rgb" if include_tactile else "wrist_rgb_only",
        "image_features": list(IMAGE_FEATURES) + ([TACTILE_RGB_FEATURE] if include_tactile else []),
        "wrist_rgb_source_key": "camera0_rgb",
        "wrist_rgb_image_shape_hwc": list(wrist_shape),
        "tactile_inputs": {
            "source_keys_present": sorted(excluded_tactile_source_keys),
            "enabled": include_tactile,
            "copied_to_lerobot": include_tactile,
            "used_for_training": include_tactile,
            "rgb_feature": TACTILE_RGB_FEATURE if include_tactile else None,
            "marker_positions_feature": MARKER_POSITIONS_FEATURE if include_tactile else None,
            "marker_reference_feature": MARKER_REFERENCE_FEATURE if include_tactile else None,
            "marker_valid_feature": MARKER_VALID_FEATURE if include_tactile else None,
            "num_markers": None if tactile_info is None else tactile_info["num_markers"],
            "source_marker_representation": "normalized_marker_flow",
            "canonical_equivalence": (
                "position=flow, reference=0; previous position queried within episode"
                if include_tactile
                else None
            ),
        },
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

    with open_zarr_group(source) as root:
        source_info = inspect_source(root)
        tactile_info = inspect_tactile_source(root) if args.include_tactile else None
        available_source_episodes = source_info["source_episodes"]
        num_source_episodes = available_source_episodes
        if args.max_source_episodes is not None:
            num_source_episodes = min(num_source_episodes, args.max_source_episodes)

        episode_ends = source_info["episode_ends"]
        source_frames = int(episode_ends[num_source_episodes - 1])
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
            include_tactile=args.include_tactile,
            tactile_info=tactile_info,
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
                            include_tactile=args.include_tactile,
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
        LeRobotDataset, _ = _lerobot_dataset_classes()
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            fps=DATASET_FPS,
            robot_type=ROBOT_TYPE,
            features=build_features(source_info["wrist_image_shape"], tactile_info),
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
            tactile_images = None
            marker_flow = None
            if args.include_tactile:
                tactile_images = np.asarray(
                    data["tacthru_l_rgb"][source_slice], dtype=np.uint8
                )
                marker_flow = np.asarray(
                    data["tacthru_l_marker"][source_slice], dtype=np.float32
                )

            expected_length = source_episode["length"]
            lengths = [len(poses), len(wrist_images)]
            if tactile_images is not None:
                lengths.extend([len(tactile_images), len(marker_flow)])
            if any(length != expected_length for length in lengths):
                raise RuntimeError(
                    "Contiguous slice length mismatch for "
                    f"source episode {source_episode['source_episode_index']}: "
                    f"lengths={lengths}, "
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
                if args.include_tactile:
                    marker = marker_flow[frame_offset]
                    valid = np.isfinite(marker).all(axis=-1)
                    if not valid.all():
                        marker = np.where(valid[:, None], marker, 0.0).astype(np.float32)
                    frame.update(
                        {
                            TACTILE_RGB_FEATURE: tactile_images[frame_offset],
                            MARKER_POSITIONS_FEATURE: marker,
                            MARKER_REFERENCE_FEATURE: np.zeros_like(marker),
                            MARKER_VALID_FEATURE: valid.astype(np.bool_),
                        }
                    )
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=not args.serial_video_encoding)

        dataset.finalize()

    result = validate_output(
        output,
        output_episode_lengths,
        output_frames,
        include_tactile=args.include_tactile,
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
        "--include-tactile",
        action="store_true",
        help="Copy left TacThru RGB and canonical marker fields for VTLA",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate source and print the conversion plan")
    args = parser.parse_args()

    if args.max_source_episodes is not None and args.max_source_episodes <= 0:
        parser.error("--max-source-episodes must be positive")
    if args.image_writer_threads < 0:
        parser.error("--image-writer-threads must be non-negative")
    if args.batch_encoding_size <= 0:
        parser.error("--batch-encoding-size must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
