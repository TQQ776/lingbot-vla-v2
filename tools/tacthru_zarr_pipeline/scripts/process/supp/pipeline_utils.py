from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_world.vive_calibration import NEW_TCP_RELATIVE_FRAME, load_vive_tcp_calibration, tracker_relative_to_newtcp_relative


GRIPPER_MEASUREMENT_SOURCE = "synria_gloria_sdk"


def vive_tracker_to_tcp_tf(poses: np.ndarray) -> np.ndarray:
    """把 demo 内 tracker 相对轨迹转换为 new_tcp 相对轨迹。"""
    calibration = load_vive_tcp_calibration()
    return tracker_relative_to_newtcp_relative(poses, calibration.tracker_to_newtcp)


@dataclass(frozen=True)
class DemoSpec:
    demo_dir: Path
    raw_video_path: Path
    recording_dir: Path


def repo_root() -> Path:
    return REPO_ROOT


def resolve_session_dir(session_dir: str | Path) -> Path:
    return Path(session_dir).expanduser().resolve()


def resolve_demos_dir(path: str | Path) -> Path:
    path = resolve_session_dir(path)
    if path.name == "demos":
        return path
    return path / "demos"


def resolve_session_from_any(path: str | Path) -> Path:
    path = resolve_session_dir(path)
    if path.name == "demos":
        return path.parent
    return path


def sorted_recording_dirs(session_dir: str | Path) -> list[Path]:
    """查找新采集流程直接生成的 test-* 完整记录目录。"""
    session_dir = resolve_session_dir(session_dir)
    dirs = [path for path in session_dir.iterdir() if path.is_dir() and path.name.startswith("test-") and (path / "raw_video.mp4").is_file()]
    return sorted(dirs, key=_sensor_dir_sort_key)


def sorted_demo_dirs(session_dir: str | Path) -> list[Path]:
    demos_dir = resolve_demos_dir(session_dir)
    if not demos_dir.exists():
        return []
    return sorted([path for path in demos_dir.iterdir() if path.is_dir() and path.name.startswith("demo_")])


def ensure_demo_specs(session_dir: str | Path) -> list[DemoSpec]:
    session_dir = resolve_session_dir(session_dir)
    recording_dirs = sorted_recording_dirs(session_dir)
    if not recording_dirs:
        raise FileNotFoundError(f"No test-* recording directories containing raw_video.mp4 found under {session_dir}")
    return _build_demo_specs_from_recordings(session_dir, recording_dirs)


def _build_demo_specs_from_recordings(session_dir: Path, recording_dirs: list[Path]) -> list[DemoSpec]:
    demos_dir = session_dir / "demos"
    demos_dir.mkdir(parents=True, exist_ok=True)
    existing_demo_dirs = sorted_demo_dirs(session_dir)
    if existing_demo_dirs and len(existing_demo_dirs) != len(recording_dirs):
        raise ValueError(
            f"Found {len(existing_demo_dirs)} demo directories but {len(recording_dirs)} recording directories. "
            "Please remove or reconcile stale demo directories first."
        )

    specs: list[DemoSpec] = []
    for index, recording_dir in enumerate(recording_dirs):
        demo_dir = existing_demo_dirs[index] if existing_demo_dirs else demos_dir / f"demo_{index:03d}_{recording_dir.name}"
        demo_dir.mkdir(parents=True, exist_ok=True)
        specs.append(
            DemoSpec(
                demo_dir=demo_dir,
                raw_video_path=recording_dir / "raw_video.mp4",
                recording_dir=recording_dir,
            )
        )
    return specs


def tactile_config_names(gripper: str, swap_lr: bool) -> tuple[str, str]:
    if gripper not in {"a", "b", "m"}:
        raise ValueError(f"Unsupported gripper {gripper!r}")
    left_cfg = f"{gripper}l"
    right_cfg = f"{gripper}r"
    if swap_lr:
        return right_cfg, left_cfg
    return left_cfg, right_cfg


@lru_cache(maxsize=1)
def _sensor_display_name_candidates() -> dict[str, tuple[str, ...]]:
    sensor_dir = repo_root() / "cfg" / "sensor"
    display_name_to_stems: dict[str, list[str]] = {}
    for cfg_path in sorted(sensor_dir.glob("*.yaml")):
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        display_name = str(cfg.get("name", "")).strip()
        if not display_name:
            continue
        display_name_to_stems.setdefault(display_name, []).append(cfg_path.stem)
    return {name: tuple(stems) for name, stems in display_name_to_stems.items()}


def _resolve_sensor_cfg_name(name: str) -> tuple[str | None, tuple[str, ...]]:
    path = repo_root() / "cfg" / "sensor" / f"{name}.yaml"
    if path.is_file():
        return name, (name,)
    candidates = _sensor_display_name_candidates().get(name, ())
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def load_recorded_sensor_names(recording_dir: Path) -> list[str] | None:
    """读取采集端写入的 TacThru 语义信息。"""
    meta_path = recording_dir / "tactile_meta.json"
    if not meta_path.is_file():
        return None

    with meta_path.open("r") as f:
        payload = json.load(f)
    sensor_names = payload.get("sensor_names")
    if not isinstance(sensor_names, list) or not sensor_names:
        raise ValueError(f"Invalid tactile metadata in {meta_path}")
    if not all(isinstance(name, str) and name for name in sensor_names):
        raise ValueError(f"Invalid sensor_names in {meta_path}")
    return sensor_names


def resolve_tactile_cfg_names(recording_dir: Path, gripper: str, swap_lr: bool) -> tuple[str, str]:
    """优先使用采集元数据恢复左右语义；旧数据再回退到命令行约定。"""
    sensor_names = load_recorded_sensor_names(recording_dir)
    if sensor_names is None:
        return tactile_config_names(gripper=gripper, swap_lr=swap_lr)

    if len(sensor_names) == 2:
        resolved = []
        for name in sensor_names:
            cfg_name, candidates = _resolve_sensor_cfg_name(name)
            if cfg_name is not None:
                resolved.append(cfg_name)
                continue
            if candidates:
                return tactile_config_names(gripper=gripper, swap_lr=swap_lr)
            raise ValueError(f"Unknown tactile sensor name {name!r} in {recording_dir / 'tactile_meta.json'}")
        return resolved[0], resolved[1]
    if len(sensor_names) != 1:
        raise ValueError(f"Unsupported tactile sensor count in {recording_dir / 'tactile_meta.json'}: {len(sensor_names)}")

    first_name, candidates = _resolve_sensor_cfg_name(sensor_names[0])
    if first_name is None:
        if candidates:
            return tactile_config_names(gripper=gripper, swap_lr=swap_lr)
        raise ValueError(f"Unknown tactile sensor name {sensor_names[0]!r} in {recording_dir / 'tactile_meta.json'}")
    if first_name.endswith("l"):
        companion_name = f"{first_name[:-1]}r"
        if _sensor_cfg_exists(companion_name):
            return first_name, companion_name
        return first_name, first_name
    if first_name.endswith("r"):
        companion_name = f"{first_name[:-1]}l"
        if _sensor_cfg_exists(companion_name):
            return first_name, companion_name
        return first_name, first_name
    return first_name, first_name


def tactile_dataset_key_map(swap_lr: bool) -> dict[int, str]:
    if swap_lr:
        return {0: "tacthru_r", 1: "tacthru_l"}
    return {0: "tacthru_l", 1: "tacthru_r"}


def sensor_cfg_path(cfg_name: str) -> Path:
    path = repo_root() / "cfg" / "sensor" / f"{cfg_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sensor config: {path}")
    return path


def load_sensor_cfg(cfg_name: str) -> dict:
    with sensor_cfg_path(cfg_name).open("r") as f:
        return yaml.safe_load(f)


def _sensor_cfg_exists(cfg_name: str) -> bool:
    return (repo_root() / "cfg" / "sensor" / f"{cfg_name}.yaml").is_file()


def build_sensor_undistort_params(cfg_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_sensor_cfg(cfg_name)
    optics = cfg["optics"]
    cam_intr = np.asarray(optics["CameraMatrix"], dtype=np.float32)
    cam_dist = np.asarray(optics["DistCoeffs"], dtype=np.float32)
    crop = np.asarray(optics["CropCoords"], dtype=np.int32)
    return cam_intr, cam_dist, crop


def _apply_frame_flip(frame: np.ndarray, *, flip_vertical: bool = False, flip_horizontal: bool = False) -> np.ndarray:
    if flip_vertical and flip_horizontal:
        return cv2.flip(frame, -1)
    if flip_vertical:
        return cv2.flip(frame, 0)
    if flip_horizontal:
        return cv2.flip(frame, 1)
    return frame


def process_tactile_video(src_path: Path, dst_path: Path, cfg_name: str, force: bool = False) -> None:
    if dst_path.exists() and not force:
        return

    cam_intr, cam_dist, crop = build_sensor_undistort_params(cfg_name)
    cfg = load_sensor_cfg(cfg_name)
    do_undistortion = bool(cfg.get("do_undistortion", True))
    flip_vertical = bool(cfg.get("flip_vertical", False))
    flip_horizontal = bool(cfg.get("flip_horizontal", False))

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open tactile video {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map_x = map_y = None
    if do_undistortion:
        intrinsics_new, _ = cv2.getOptimalNewCameraMatrix(cam_intr, cam_dist, (width, height), 1, (width, height))
        map_x, map_y = cv2.initUndistortRectifyMap(cam_intr, cam_dist, None, intrinsics_new, (width, height), cv2.CV_32FC1)

    out_h = int(crop[3] - crop[1])
    out_w = int(crop[2] - crop[0])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer for {dst_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if do_undistortion:
                frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            frame = frame[crop[1] : crop[3], crop[0] : crop[2]]
            frame = _apply_frame_flip(frame, flip_vertical=flip_vertical, flip_horizontal=flip_horizontal)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def tactile_output_shape(cfg_name: str) -> tuple[int, int]:
    _, _, crop = build_sensor_undistort_params(cfg_name)
    out_h = int(crop[3] - crop[1])
    out_w = int(crop[2] - crop[0])
    return out_h, out_w


def create_placeholder_tactile_artifacts(
    *,
    reference_video_path: Path,
    reference_txt_path: Path,
    dst_video_path: Path,
    dst_txt_path: Path,
    cfg_name: str,
    force: bool = False,
    gray_value: int = 127,
) -> None:
    if dst_video_path.exists() and dst_txt_path.exists() and not force:
        return

    with reference_txt_path.open("r") as f:
        timestamps = [line.strip() for line in f if line.strip()]
    if not timestamps:
        raise RuntimeError(f"Reference tactile timestamp file is empty: {reference_txt_path}")

    cap = cv2.VideoCapture(str(reference_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open reference tactile video: {reference_video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    out_h, out_w = tactile_output_shape(cfg_name)
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open placeholder video writer for {dst_video_path}")

    try:
        frame = np.full((out_h, out_w, 3), int(gray_value), dtype=np.uint8)
        for _ in timestamps:
            writer.write(frame)
    finally:
        writer.release()

    with dst_txt_path.open("w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")


def copy_if_needed(src_path: Path, dst_path: Path, force: bool = False) -> None:
    if dst_path.exists() and not force:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def tactile_reference_points(cfg_name: str, frame_shape_hw: tuple[int, int]) -> np.ndarray:
    cfg = load_sensor_cfg(cfg_name)
    ref_path = repo_root() / cfg["tracking"]["tracking_pts_path"]
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing tactile reference keypoints: {ref_path}")
    ref = np.asarray(np.load(ref_path), dtype=np.float32)
    height, width = frame_shape_hw
    return ref * np.array([width, height], dtype=np.float32)


def tactile_marker_count(cfg_name: str) -> int:
    cfg = load_sensor_cfg(cfg_name)
    tracking_cfg = cfg.get("tracking", {})
    grid_shape = tracking_cfg.get("grid_shape")
    if grid_shape is not None:
        if len(grid_shape) != 2:
            raise ValueError(f"Invalid grid_shape for {cfg_name}: {grid_shape!r}")
        return int(grid_shape[0]) * int(grid_shape[1])

    ref_path = repo_root() / tracking_cfg["tracking_pts_path"]
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing tactile reference keypoints: {ref_path}")
    ref = np.asarray(np.load(ref_path), dtype=np.float32)
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"Invalid tactile reference keypoints at {ref_path}: expected (N, 2), got {ref.shape}")
    return int(ref.shape[0])


def write_vive_interp_csv(vive_pkl_path: Path, output_csv_path: Path, force: bool = False) -> None:
    if output_csv_path.exists() and not force:
        return
    data = pickle.load(vive_pkl_path.open("rb"))
    poses = np.asarray(data["poses"], dtype=np.float64)
    poses = vive_tracker_to_tcp_tf(poses)
    timestamps = np.asarray(data["ts"], dtype=np.float64)
    rotations = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    frame_idx = np.arange(len(timestamps), dtype=np.int64)

    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "timestamp": timestamps,
            "x": poses[:, 0, 3],
            "y": poses[:, 1, 3],
            "z": poses[:, 2, 3],
            "q_x": rotations[:, 0],
            "q_y": rotations[:, 1],
            "q_z": rotations[:, 2],
            "q_w": rotations[:, 3],
            "pose_frame": NEW_TCP_RELATIVE_FRAME,
            "is_lost": False,
        }
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)


def write_gripper_interp_csv(gripper_pkl_path: Path, output_csv_path: Path, force: bool = False) -> None:
    """读取 SDK 夹爪反馈，并将毫米宽度转换为数据集使用的米。"""
    if output_csv_path.exists() and not force:
        return

    with gripper_pkl_path.open("rb") as f:
        data = pickle.load(f)

    timestamps = np.asarray(data["ts"], dtype=np.float64).reshape(-1)
    width_mm = np.asarray(data["gripper_width_mm"], dtype=np.float64).reshape(-1)
    if len(timestamps) != len(width_mm) or len(timestamps) < 2:
        raise ValueError(f"Invalid gripper samples in {gripper_pkl_path}: timestamps={len(timestamps)}, widths={len(width_mm)}")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(width_mm)):
        raise ValueError(f"Non-finite gripper samples in {gripper_pkl_path}")
    if np.any(width_mm < 0):
        raise ValueError(f"Negative gripper width in {gripper_pkl_path}")

    # 反馈线程理论上按时间递增；排序并合并重复时间戳可避免插值器产生不确定结果。
    samples = pd.DataFrame({"timestamp": timestamps, "gripper_width_m": width_mm / 1000.0})
    samples = samples.sort_values("timestamp").groupby("timestamp", as_index=False).mean()
    if len(samples) < 2 or np.any(np.diff(samples["timestamp"].to_numpy()) <= 0):
        raise ValueError(f"Gripper timestamps are not usable for interpolation: {gripper_pkl_path}")

    samples.insert(0, "sample_idx", np.arange(len(samples), dtype=np.int64))
    samples["measurement_source"] = GRIPPER_MEASUREMENT_SOURCE
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_csv_path, index=False)


def _sensor_dir_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)$", path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)
