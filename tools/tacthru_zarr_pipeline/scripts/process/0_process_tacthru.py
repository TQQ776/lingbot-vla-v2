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

from supp.pipeline_utils import (
    copy_if_needed,
    ensure_demo_specs,
    process_tactile_video,
    resolve_tactile_cfg_names,
    tactile_marker_count,
    tactile_reference_points,
)
from utils.proc_utils import KeypointsKFProcessor


def _prepare_demo(
    demo_dir: Path,
    recording_dir: Path,
    raw_video_path: Path,
    cfg_names: tuple[str, str],
    force: bool,
) -> None:
    click.echo(f"[prepare] {demo_dir.name}")
    copy_if_needed(raw_video_path, demo_dir / "raw_video.mp4", force=force)
    copy_if_needed(recording_dir / "raw_video.txt", demo_dir / "raw_video.txt", force=force)
    copy_if_needed(recording_dir / "vive.pkl", demo_dir / "vive.pkl", force=force)
    copy_if_needed(recording_dir / "gripper.pkl", demo_dir / "gripper.pkl", force=force)
    if (recording_dir / "tactile_meta.json").is_file():
        copy_if_needed(recording_dir / "tactile_meta.json", demo_dir / "tactile_meta.json", force=force)

    processed_streams = 0
    for sensor_idx, cfg_name in enumerate(cfg_names):
        src_video = recording_dir / f"TacThru-{sensor_idx}.avi"
        src_txt = recording_dir / f"TacThru-{sensor_idx}.txt"
        dst_video = demo_dir / src_video.name
        dst_txt = demo_dir / src_txt.name
        placeholder_flag = demo_dir / f"TacThru-{sensor_idx}.placeholder"

        if src_video.is_file() and src_txt.is_file():
            process_tactile_video(src_path=src_video, dst_path=dst_video, cfg_name=cfg_name, force=force)
            copy_if_needed(src_txt, dst_txt, force=force)
            if placeholder_flag.exists():
                placeholder_flag.unlink()
            processed_streams += 1
            continue

        for stale_path in (dst_video, dst_txt, placeholder_flag):
            if stale_path.exists():
                stale_path.unlink()
        click.echo(f"  skip missing TacThru-{sensor_idx}: no recorded tactile stream")

    if processed_streams == 0:
        raise FileNotFoundError(f"No tactile streams found in {recording_dir}: expected at least one TacThru-*.avi/.txt pair")


def _should_regenerate_kpts(kpts_path: Path, force: bool) -> bool:
    if force or (not kpts_path.is_file()):
        return True
    try:
        with kpts_path.open("rb") as f:
            data = pickle.load(f)
        return not all(key in data for key in ("marker", "marker_ref", "marker_flow"))
    except Exception:
        return True


def _track_tactile_video(video_path: Path, cfg_name: str, force: bool) -> None:
    kpts_path = video_path.with_name(f"{video_path.stem}_kpts.pkl")
    if not _should_regenerate_kpts(kpts_path=kpts_path, force=force):
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open synced tactile video {video_path}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Synced tactile video is empty: {video_path}")

    frame_shape_hw = frame.shape[:2]
    ref_marker_pos = tactile_reference_points(cfg_name=cfg_name, frame_shape_hw=frame_shape_hw)
    marker_count = int(ref_marker_pos.shape[0])
    detector = KeypointsKFProcessor(ref_marker_pos)
    detector.reset()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    labels = {"marker": [], "marker_ref": [], "marker_flow": []}
    frame_shape_wh = np.array([frame_shape_hw[1], frame_shape_hw[0]], dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector(frame)
        marker = np.asarray(result["marker"], dtype=np.float32)
        marker_ref = np.asarray(result["marker_ref"], dtype=np.float32)
        marker = marker / frame_shape_wh * 2.0 - 1.0
        marker_ref = marker_ref / frame_shape_wh * 2.0 - 1.0
        labels["marker"].append(marker)
        labels["marker_ref"].append(marker_ref)
        labels["marker_flow"].append(marker - marker_ref)

    cap.release()

    labels = {key: np.asarray(value, dtype=np.float32) if value else np.zeros((0, marker_count, 2), dtype=np.float32) for key, value in labels.items()}
    with kpts_path.open("wb") as f:
        pickle.dump(labels, f)


def _write_placeholder_kpts(video_path: Path, cfg_name: str) -> None:
    kpts_path = video_path.with_name(f"{video_path.stem}_kpts.pkl")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open placeholder synced tactile video {video_path}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Placeholder synced tactile video is empty: {video_path}")

    frame_shape_hw = frame.shape[:2]
    frame_shape_wh = np.array([frame_shape_hw[1], frame_shape_hw[0]], dtype=np.float32)
    ref_marker_pos = tactile_reference_points(cfg_name=cfg_name, frame_shape_hw=frame_shape_hw)
    marker_ref = ref_marker_pos / frame_shape_wh * 2.0 - 1.0
    marker_ref = marker_ref.astype(np.float32)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        frame_count = 1

    expected_marker_count = tactile_marker_count(cfg_name)
    if marker_ref.shape[0] != expected_marker_count:
        raise RuntimeError(
            f"Marker reference count mismatch for {cfg_name}: inferred {marker_ref.shape[0]} points from video, "
            f"but config expects {expected_marker_count}"
        )

    marker = np.repeat(marker_ref[None, ...], frame_count, axis=0)
    marker_ref_stack = np.repeat(marker_ref[None, ...], frame_count, axis=0)
    marker_flow = np.zeros_like(marker, dtype=np.float32)

    with kpts_path.open("wb") as f:
        pickle.dump({"marker": marker, "marker_ref": marker_ref_stack, "marker_flow": marker_flow}, f)


@click.command(help="Prepare raw tactile videos into demo dirs or track markers from synced tactile videos.")
@click.argument("session_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--mode", type=click.Choice(["prepare", "track", "all"]), default="all", show_default=True)
@click.option("--gripper", type=click.Choice(["a", "b", "m"]), required=True)
@click.option("--swap-lr/--no-swap-lr", default=False)
@click.option("--force", is_flag=True, default=False)
def main(session_dir: Path, mode: str, gripper: str, swap_lr: bool, force: bool) -> None:
    demo_specs = ensure_demo_specs(session_dir)

    if mode in {"prepare", "all"}:
        for spec in demo_specs:
            cfg_names = resolve_tactile_cfg_names(recording_dir=spec.recording_dir, gripper=gripper, swap_lr=swap_lr)
            _prepare_demo(
                demo_dir=spec.demo_dir,
                recording_dir=spec.recording_dir,
                raw_video_path=spec.raw_video_path,
                cfg_names=cfg_names,
                force=force,
            )

    if mode in {"track", "all"}:
        allow_missing_synced = mode == "all"
        for spec in demo_specs:
            cfg_names = resolve_tactile_cfg_names(recording_dir=spec.recording_dir, gripper=gripper, swap_lr=swap_lr)
            click.echo(f"[track] {spec.demo_dir.name}")
            for sensor_idx, cfg_name in enumerate(cfg_names):
                synced_video = spec.demo_dir / f"TacThru-{sensor_idx}_synced.mp4"
                if not synced_video.is_file():
                    source_video = spec.demo_dir / f"TacThru-{sensor_idx}.avi"
                    if allow_missing_synced or not source_video.is_file():
                        click.echo(f"  skip {synced_video.name}: sync output missing")
                        continue
                    raise FileNotFoundError(f"Missing synced tactile video: {synced_video}")
                placeholder_flag = spec.demo_dir / f"TacThru-{sensor_idx}.placeholder"
                if placeholder_flag.is_file():
                    click.echo(f"  write zero-flow markers for {synced_video.name}")
                    _write_placeholder_kpts(video_path=synced_video, cfg_name=cfg_name)
                    continue
                _track_tactile_video(video_path=synced_video, cfg_name=cfg_name, force=force)


if __name__ == "__main__":
    main()
