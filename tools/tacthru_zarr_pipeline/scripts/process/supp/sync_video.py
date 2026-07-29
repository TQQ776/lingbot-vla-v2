import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from supp.pipeline_utils import GRIPPER_MEASUREMENT_SOURCE


def load_timestamps(file_path):
    """读取逐行或 Python 列表格式的时间戳，不修改原始采集文件。"""
    text = Path(file_path).read_text()
    normalized = text.replace("[", " ").replace("]", " ").replace(",", " ")
    timestamps = np.fromstring(normalized, sep=" ", dtype=np.float64)
    if len(timestamps) == 0 or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"Invalid or empty timestamp file: {file_path}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Timestamps must be strictly increasing: {file_path}")
    return timestamps


def calculate_bounds(*time_arrays):
    # Calculate the lower and upper bounds for synchronization
    if not time_arrays:
        raise ValueError("At least one timestamp array is required.")
    lower_bound = max(time_array[0] for time_array in time_arrays)
    upper_bound = min(time_array[-1] for time_array in time_arrays)
    if upper_bound <= lower_bound:
        raise ValueError(f"Input streams have no overlapping time range: [{lower_bound}, {upper_bound}]")
    return lower_bound, upper_bound


def get_target_fps(video_path):
    # Get the frame rate of the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def infer_target_fps(video_path, timestamps, mismatch_rel_tol: float = 0.15):
    header_fps = float(get_target_fps(video_path))
    timestamps = np.asarray(timestamps, dtype=np.float64)
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if len(diffs) == 0:
        if header_fps <= 0:
            raise ValueError(f"Could not infer target fps for {video_path}: invalid video header fps and insufficient timestamps")
        return header_fps

    ts_fps = float(1.0 / np.median(diffs))
    if header_fps <= 0:
        print(f"Video header fps invalid for {video_path}; using timestamp-implied fps={ts_fps:.3f}")
        return ts_fps

    rel_mismatch = abs(ts_fps - header_fps) / max(ts_fps, header_fps, 1e-6)
    if rel_mismatch > mismatch_rel_tol:
        print(f"Video header fps={header_fps:.3f} disagrees with timestamp-implied fps={ts_fps:.3f} for {video_path}; using timestamp-implied fps")
        return ts_fps

    return header_fps


def generate_target_timestamps(lower_bound, upper_bound, target_fps):
    # Generate synchronized timestamps
    frame_dt = 1.0 / float(target_fps)
    num_frames = int(np.floor((upper_bound - lower_bound) / frame_dt))
    return lower_bound + np.arange(num_frames, dtype=np.float64) * frame_dt


def compute_closest_indices(time_array: np.ndarray, target_times: np.ndarray, latency: float = 0.0):
    """
    Compute for each target_time the index in time_array that is closest.
    Uses np.searchsorted to avoid full pairwise distance matrix.
    """
    if len(time_array) == 0:
        return np.zeros(len(target_times), dtype=int)
    # insertion positions
    pos = np.searchsorted(time_array, target_times - latency, side="left")
    # candidate indices: pos and pos-1 (clamped)
    left = np.clip(pos - 1, 0, len(time_array) - 1)
    right = np.clip(pos, 0, len(time_array) - 1)
    # pick closer of left/right
    dist_left = np.abs(time_array[left] - (target_times - latency))
    dist_right = np.abs(time_array[right] - (target_times - latency))
    choose_right = dist_right < dist_left
    indices = left
    indices[choose_right] = right[choose_right]
    return indices.astype(int)


def write_synced_video(video_path: str, target_indices: np.ndarray, output_path: str, target_fps: float, cam_modif: tuple = None):
    if len(target_indices) == 0:
        raise ValueError(f"No target indices for {output_path}")

    target_indices = np.asarray(target_indices, dtype=np.int64)
    if np.any(np.diff(target_indices) < 0):
        raise ValueError(f"Target indices must be sorted for {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    try:
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cam_modif is not None:
            (map_x, map_y, cam_crop) = cam_modif
            out_height = cam_crop[3] - cam_crop[1]
            out_width = cam_crop[2] - cam_crop[0]
        else:
            out_height = src_height
            out_width = src_width

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, float(target_fps), (out_width, out_height))
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter open failed: {output_path}")

        try:
            target_ptr = 0
            target_count = len(target_indices)
            frame_number = 0
            last_needed_idx = int(target_indices[-1])

            while target_ptr < target_count and frame_number <= last_needed_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number == int(target_indices[target_ptr]):
                    if cam_modif is not None:
                        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
                        frame = frame[cam_crop[1] : cam_crop[3], cam_crop[0] : cam_crop[2]]

                    while target_ptr < target_count and frame_number == int(target_indices[target_ptr]):
                        out.write(frame)
                        target_ptr += 1

                frame_number += 1

            if target_ptr != target_count:
                raise RuntimeError(f"Failed to write all synchronized frames for {output_path}: wrote {target_ptr} / {target_count}")
        finally:
            out.release()
    finally:
        cap.release()


def sync_vive_csv(vive_csv_path: str, target_timestamps: np.ndarray, output_csv_path: str, latency: float = 0.0):
    df = pd.read_csv(vive_csv_path)
    src_ts = df["timestamp"].values + latency
    aligned_rows = []
    for new_idx, tgt in enumerate(target_timestamps):
        src_idx = int(np.abs(src_ts - tgt).argmin())
        row = df.iloc[src_idx].copy()
        row["frame_idx"] = new_idx
        row["timestamp"] = tgt
        aligned_rows.append(row)
    out_df = pd.DataFrame(aligned_rows, columns=df.columns)
    out_df.to_csv(output_csv_path, index=False)


def sync_gripper_csv(gripper_csv_path: str, target_timestamps: np.ndarray, output_csv_path: str):
    """将夹爪反馈线性插值到视频帧时间轴，输出单位保持为米。"""
    df = pd.read_csv(gripper_csv_path)
    required_columns = {"timestamp", "gripper_width_m"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing gripper CSV columns {sorted(missing)} in {gripper_csv_path}")

    src_ts = df["timestamp"].to_numpy(dtype=np.float64)
    src_width = df["gripper_width_m"].to_numpy(dtype=np.float64)
    if len(src_ts) < 2 or np.any(np.diff(src_ts) <= 0):
        raise ValueError(f"Gripper timestamps must be strictly increasing: {gripper_csv_path}")
    if target_timestamps[0] < src_ts[0] or target_timestamps[-1] > src_ts[-1]:
        raise ValueError("Target timestamps exceed the gripper feedback range; synchronization bounds are inconsistent.")

    width = np.interp(target_timestamps, src_ts, src_width)
    out_df = pd.DataFrame(
        {
            "frame_idx": np.arange(len(target_timestamps), dtype=np.int64),
            "timestamp": target_timestamps,
            "gripper_width_m": width,
            "measurement_source": GRIPPER_MEASUREMENT_SOURCE,
        }
    )
    out_df.to_csv(output_csv_path, index=False)


def synchronize_videos(input_folder, latency: float = -0.15, vive_latency: float = 0.0, do_undistort: bool = False):
    if do_undistort:
        raise ValueError("Tactile undistortion must happen before synchronization.")

    output_folder = input_folder
    camera_video = os.path.join(input_folder, "raw_video.mp4")
    time_camera = load_timestamps(os.path.join(input_folder, "raw_video.txt"))
    tactile_streams = []
    for sensor_idx in (0, 1):
        video_path = os.path.join(input_folder, f"TacThru-{sensor_idx}.avi")
        timestamp_path = os.path.join(input_folder, f"TacThru-{sensor_idx}.txt")
        if os.path.isfile(video_path) and os.path.isfile(timestamp_path):
            tactile_streams.append(
                {
                    "idx": sensor_idx,
                    "video": video_path,
                    "timestamps": load_timestamps(timestamp_path),
                }
            )
    if not tactile_streams:
        raise FileNotFoundError(f"No tactile streams found under {input_folder}")

    gripper_csv = os.path.join(input_folder, "gripper_interp.csv")
    gripper_df = pd.read_csv(gripper_csv)
    time_gripper = gripper_df["timestamp"].to_numpy(dtype=np.float64)

    lower_bound, upper_bound = calculate_bounds(time_camera, *(stream["timestamps"] for stream in tactile_streams), time_gripper)
    target_fps = infer_target_fps(camera_video, time_camera)
    target_timestamps = generate_target_timestamps(lower_bound, upper_bound, target_fps)

    # Compute the indices (in each source time array) that correspond to each target timestamp.
    camera_indices_for_targets = compute_closest_indices(time_camera, target_timestamps, latency=0.0)
    tactile_indices_for_targets = {
        stream["idx"]: compute_closest_indices(stream["timestamps"], target_timestamps, latency=latency)
        for stream in tactile_streams
    }

    print(f"Camera: output frames={len(camera_indices_for_targets)}, unique source frames={len(np.unique(camera_indices_for_targets))}")
    for stream in tactile_streams:
        indices = tactile_indices_for_targets[stream["idx"]]
        print(f"TacThru-{stream['idx']}: output frames={len(indices)}, unique source frames={len(np.unique(indices))}")

    write_synced_video(camera_video, camera_indices_for_targets, os.path.join(output_folder, "camera_synced.mp4"), target_fps)
    for stream in tactile_streams:
        sensor_idx = stream["idx"]
        write_synced_video(
            stream["video"],
            tactile_indices_for_targets[sensor_idx],
            os.path.join(output_folder, f"TacThru-{sensor_idx}_synced.mp4"),
            target_fps,
        )

    vive_csv = os.path.join(input_folder, "vive_interp.csv")
    sync_vive_csv(
        vive_csv_path=vive_csv,
        target_timestamps=target_timestamps,
        output_csv_path=os.path.join(output_folder, "vive_synced.csv"),
        latency=vive_latency,
    )
    sync_gripper_csv(
        gripper_csv_path=gripper_csv,
        target_timestamps=target_timestamps,
        output_csv_path=os.path.join(output_folder, "gripper_synced.csv"),
    )

    lower_bound = lower_bound - time_camera[0]
    upper_bound = upper_bound - time_camera[0]

    return lower_bound, upper_bound
