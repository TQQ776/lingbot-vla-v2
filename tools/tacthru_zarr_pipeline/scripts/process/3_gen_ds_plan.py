from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import click
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_util import mat_to_pose
from real_world.vive_calibration import NEW_TCP_RELATIVE_FRAME
from supp.pipeline_utils import GRIPPER_MEASUREMENT_SOURCE, resolve_session_from_any, sorted_demo_dirs


def get_bool_segments(bool_seq: np.ndarray) -> tuple[list[slice], np.ndarray]:
    bool_seq = np.asarray(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = []
    segment_type = []
    for start, end in zip(segment_bounds[:-1], segment_bounds[1:], strict=True):
        segments.append(slice(start, end))
        segment_type.append(bool_seq[start])
    return segments, np.asarray(segment_type, dtype=bool)


def _video_info(video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, fps


def _align_lengths(
    vive_df: pd.DataFrame,
    gripper_df: pd.DataFrame,
    frame_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = min(len(vive_df), len(gripper_df), frame_count)
    if common <= 0:
        raise RuntimeError("No aligned frames available.")
    length_delta = max(
        abs(len(vive_df) - common),
        abs(len(gripper_df) - common),
        abs(frame_count - common),
    )
    if length_delta > 1:
        raise RuntimeError(f"Length mismatch too large: vive_synced={len(vive_df)}, gripper_synced={len(gripper_df)}, video_frames={frame_count}")
    return vive_df.iloc[:common].copy(), gripper_df.iloc[:common].copy()


@click.command(help="Generate the dataset plan for the mono-manual TacThru UMI pipeline.")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("-ml", "--min-episode-length", type=int, default=24, show_default=True)
@click.option("--force", is_flag=True, default=False)
def main(
    input_path: Path,
    output: Path | None,
    min_episode_length: int,
    force: bool,
) -> None:
    session_dir = resolve_session_from_any(input_path)
    demos_dir = session_dir / "demos"
    output = output.expanduser().resolve() if output is not None else session_dir / "dataset_plan_vive.pkl"
    if output.exists() and not force:
        click.echo(f"[plan] overwriting {output}")

    all_plans: list[dict] = []
    used_frames = 0
    available_frames = 0

    for demo_dir in sorted_demo_dirs(session_dir):
        video_path = demo_dir / "camera_synced.mp4"
        vive_csv_path = demo_dir / "vive_synced.csv"
        gripper_csv_path = demo_dir / "gripper_synced.csv"
        if not (video_path.is_file() and vive_csv_path.is_file() and gripper_csv_path.is_file()):
            click.echo(f"[plan] skip {demo_dir.name}: missing synced inputs")
            continue

        frame_count, fps = _video_info(video_path)
        vive_df = pd.read_csv(vive_csv_path)
        gripper_df = pd.read_csv(gripper_csv_path)
        if "pose_frame" not in vive_df.columns:
            raise RuntimeError(f"{vive_csv_path} has no pose_frame column. This is an old camera-frame CSV; rerun the sync stage with --force.")
        pose_frames = set(vive_df["pose_frame"].dropna().astype(str).unique())
        if pose_frames != {NEW_TCP_RELATIVE_FRAME}:
            raise RuntimeError(
                f"{vive_csv_path} uses pose_frame={sorted(pose_frames)}, expected only {NEW_TCP_RELATIVE_FRAME!r}. Rerun the sync stage with --force."
            )
        if "is_lost" not in vive_df.columns:
            vive_df["is_lost"] = False
        if "gripper_width_m" not in gripper_df.columns:
            raise RuntimeError(f"{gripper_csv_path} has no gripper_width_m column. Rerun the sync stage with --force.")
        measurement_sources = set(gripper_df.get("measurement_source", pd.Series(dtype=str)).dropna().astype(str).unique())
        if measurement_sources != {GRIPPER_MEASUREMENT_SOURCE}:
            raise RuntimeError(
                f"{gripper_csv_path} uses measurement_source={sorted(measurement_sources)}, "
                f"expected only {GRIPPER_MEASUREMENT_SOURCE!r}. Rerun the sync stage with --force."
            )

        vive_df, gripper_df = _align_lengths(vive_df, gripper_df, frame_count)
        timestamps = vive_df["timestamp"].to_numpy(dtype=np.float64)
        gripper_timestamps = gripper_df["timestamp"].to_numpy(dtype=np.float64)
        if not np.allclose(timestamps, gripper_timestamps, rtol=0.0, atol=1e-6):
            max_delta = float(np.max(np.abs(timestamps - gripper_timestamps)))
            raise RuntimeError(f"Vive/gripper timestamps are not aligned in {demo_dir}: max delta={max_delta:.9f}s")
        gripper_width = gripper_df["gripper_width_m"].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(gripper_width)) or np.any(gripper_width < 0):
            raise RuntimeError(f"Invalid SDK gripper widths in {gripper_csv_path}")
        available_frames += len(vive_df)

        is_tracked = ~vive_df["is_lost"].astype(bool).to_numpy()
        if is_tracked.sum() < max(min_episode_length, 2):
            click.echo(f"[plan] skip {demo_dir.name}: insufficient valid Vive frames")
            continue

        tcp_pos = vive_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        tcp_rot = Rotation.from_quat(vive_df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float64))
        tcp_pose_mat = np.zeros((len(vive_df), 4, 4), dtype=np.float64)
        tcp_pose_mat[:, 3, 3] = 1.0
        tcp_pose_mat[:, :3, 3] = tcp_pos
        tcp_pose_mat[:, :3, :3] = tcp_rot.as_matrix()
        tcp_pose = mat_to_pose(tcp_pose_mat)

        is_step_valid = is_tracked.copy()
        first_valid = int(np.nonzero(is_step_valid)[0][0])
        last_valid = int(np.nonzero(is_step_valid)[0][-1])
        demo_start_pose = tcp_pose[first_valid]
        demo_end_pose = tcp_pose[last_valid]

        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for segment_slice, is_valid in zip(segment_slices, segment_type, strict=True):
            if not is_valid:
                continue
            if (segment_slice.stop - segment_slice.start) < min_episode_length:
                is_step_valid[segment_slice] = False

        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for segment_slice, is_valid in zip(segment_slices, segment_type, strict=True):
            if not is_valid:
                continue
            start = segment_slice.start
            end = segment_slice.stop
            used_frames += end - start
            all_plans.append(
                {
                    "episode_timestamps": timestamps[start:end],
                    "grippers": [
                        {
                            "tcp_pose": tcp_pose[start:end].astype(np.float32),
                            "gripper_width": gripper_width[start:end].astype(np.float32),
                            "demo_start_pose": np.asarray(demo_start_pose, dtype=np.float32),
                            "demo_end_pose": np.asarray(demo_end_pose, dtype=np.float32),
                        }
                    ],
                    "cameras": [
                        {
                            "video_path": str(video_path.relative_to(demos_dir)),
                            "video_start_end": (int(start), int(end)),
                            "fps": float(fps),
                        }
                    ],
                }
            )

        click.echo(f"[plan] {demo_dir.name}: {len(vive_df)} synced frames @ {fps:.3f} fps")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(all_plans, f)

    used_ratio = 0.0 if available_frames == 0 else used_frames / available_frames
    click.echo(f"[plan] wrote {len(all_plans)} episodes to {output} ({used_ratio:.1%} usable frames)")


if __name__ == "__main__":
    main()
