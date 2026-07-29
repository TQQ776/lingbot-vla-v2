from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np


# 校验tacthru视频和vive轨迹的长度是否大致匹配，检查录制过程中是否有明显的丢帧或时间偏移问题。
@dataclass
class StreamStats:
    name: str
    frame_count: int
    start_ts: float
    end_ts: float
    duration: float
    fps_est: float
    video_frame_count: int | None = None
    video_fps: float | None = None


def _sorted_record_dirs(session_dir: Path) -> list[Path]:
    return sorted([path for path in session_dir.iterdir() if path.is_dir() and path.name.startswith("test-")], key=lambda p: p.name)


def _load_timestamp_txt(path: Path) -> np.ndarray:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"Timestamp file is empty: {path}")
    return np.asarray(values, dtype=np.float64)


def _load_vive_ts(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)
    if "ts" not in data:
        raise KeyError(f"vive.pkl missing 'ts': {path}")
    ts = np.asarray(data["ts"], dtype=np.float64).reshape(-1)
    if len(ts) == 0:
        raise RuntimeError(f"vive timestamps are empty: {path}")
    return ts


def _video_info(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frame_count, fps


def _build_stats(name: str, ts: np.ndarray, video_path: Path | None = None) -> StreamStats:
    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    fps_est = float((len(ts) - 1) / duration) if duration > 1e-9 and len(ts) > 1 else 0.0
    video_frame_count = None
    video_fps = None
    if video_path is not None and video_path.is_file():
        video_frame_count, video_fps = _video_info(video_path)
    return StreamStats(
        name=name,
        frame_count=int(len(ts)),
        start_ts=float(ts[0]),
        end_ts=float(ts[-1]),
        duration=duration,
        fps_est=fps_est,
        video_frame_count=video_frame_count,
        video_fps=video_fps,
    )


def _compare_streams(
    tactile: StreamStats,
    vive: StreamStats,
    *,
    warn_duration_diff: float,
    error_duration_diff: float,
    warn_frame_diff: int,
    error_frame_diff: int,
    warn_start_end_offset: float,
    error_start_end_offset: float,
) -> tuple[str, list[str]]:
    messages: list[str] = []
    level = "OK"

    duration_diff = tactile.duration - vive.duration
    frame_diff = tactile.frame_count - vive.frame_count
    start_offset = tactile.start_ts - vive.start_ts
    end_offset = tactile.end_ts - vive.end_ts

    def bump(new_level: str) -> None:
        nonlocal level
        order = {"OK": 0, "WARN": 1, "ERROR": 2}
        if order[new_level] > order[level]:
            level = new_level

    if abs(duration_diff) >= error_duration_diff:
        bump("ERROR")
        messages.append(f"duration_diff={duration_diff:+.3f}s")
    elif abs(duration_diff) >= warn_duration_diff:
        bump("WARN")
        messages.append(f"duration_diff={duration_diff:+.3f}s")

    if abs(frame_diff) >= error_frame_diff:
        bump("ERROR")
        messages.append(f"frame_diff={frame_diff:+d}")
    elif abs(frame_diff) >= warn_frame_diff:
        bump("WARN")
        messages.append(f"frame_diff={frame_diff:+d}")

    max_offset = max(abs(start_offset), abs(end_offset))
    if max_offset >= error_start_end_offset:
        bump("ERROR")
        messages.append(f"start/end_offset=({start_offset:+.3f}s,{end_offset:+.3f}s)")
    elif max_offset >= warn_start_end_offset:
        bump("WARN")
        messages.append(f"start/end_offset=({start_offset:+.3f}s,{end_offset:+.3f}s)")

    return level, messages


def _fmt_stats(stats: StreamStats) -> str:
    video_info = ""
    if stats.video_frame_count is not None:
        video_info = f", video_frames={stats.video_frame_count}, video_fps={stats.video_fps:.2f}"
    return (
        f"frames={stats.frame_count}, dur={stats.duration:.3f}s, fps_est={stats.fps_est:.2f}, "
        f"start={stats.start_ts:.6f}, end={stats.end_ts:.6f}{video_info}"
    )


@click.command(help="Check whether tactile recording length and Vive trajectory length are roughly aligned for a recorded session.")
@click.argument("session_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--warn-duration-diff", type=float, default=0.20, show_default=True, help="Warn if |tactile_duration - vive_duration| exceeds this many seconds.")
@click.option("--error-duration-diff", type=float, default=0.50, show_default=True, help="Error if |tactile_duration - vive_duration| exceeds this many seconds.")
@click.option("--warn-frame-diff", type=int, default=5, show_default=True, help="Warn if |tactile_frames - vive_frames| exceeds this many frames.")
@click.option("--error-frame-diff", type=int, default=15, show_default=True, help="Error if |tactile_frames - vive_frames| exceeds this many frames.")
@click.option("--warn-start-end-offset", type=float, default=0.20, show_default=True, help="Warn if tactile/vive start or end timestamps differ by this many seconds.")
@click.option("--error-start-end-offset", type=float, default=0.50, show_default=True, help="Error if tactile/vive start or end timestamps differ by this many seconds.")
def main(
    session_dir: Path,
    warn_duration_diff: float,
    error_duration_diff: float,
    warn_frame_diff: int,
    error_frame_diff: int,
    warn_start_end_offset: float,
    error_start_end_offset: float,
) -> None:
    session_dir = session_dir.expanduser().resolve()
    record_dirs = _sorted_record_dirs(session_dir)
    if not record_dirs:
        raise click.ClickException(f"No test-* directories found under {session_dir}")

    total = 0
    warn_count = 0
    error_count = 0

    for record_dir in record_dirs:
        vive_path = record_dir / "vive.pkl"
        tactile_txt_paths = sorted(record_dir.glob("TacThru-*.txt"))
        tactile_video_paths = {path.stem: path for path in record_dir.glob("TacThru-*.avi")}

        if not vive_path.is_file():
            click.echo(f"[ERROR] {record_dir.name}: missing vive.pkl")
            error_count += 1
            total += 1
            continue
        if not tactile_txt_paths:
            click.echo(f"[ERROR] {record_dir.name}: missing TacThru-*.txt")
            error_count += 1
            total += 1
            continue

        vive_stats = _build_stats("vive", _load_vive_ts(vive_path))

        for tactile_txt in tactile_txt_paths:
            total += 1
            stem = tactile_txt.stem
            tactile_stats = _build_stats(
                stem,
                _load_timestamp_txt(tactile_txt),
                video_path=tactile_video_paths.get(stem),
            )
            level, messages = _compare_streams(
                tactile_stats,
                vive_stats,
                warn_duration_diff=warn_duration_diff,
                error_duration_diff=error_duration_diff,
                warn_frame_diff=warn_frame_diff,
                error_frame_diff=error_frame_diff,
                warn_start_end_offset=warn_start_end_offset,
                error_start_end_offset=error_start_end_offset,
            )

            if level == "WARN":
                warn_count += 1
            elif level == "ERROR":
                error_count += 1

            click.echo(f"[{level}] {record_dir.name} | {stem}")
            click.echo(f"  tactile: {_fmt_stats(tactile_stats)}")
            click.echo(f"  vive:    {_fmt_stats(vive_stats)}")

            duration_diff = tactile_stats.duration - vive_stats.duration
            frame_diff = tactile_stats.frame_count - vive_stats.frame_count
            start_offset = tactile_stats.start_ts - vive_stats.start_ts
            end_offset = tactile_stats.end_ts - vive_stats.end_ts
            click.echo(
                "  diff:    "
                f"duration={duration_diff:+.3f}s, frames={frame_diff:+d}, "
                f"start_offset={start_offset:+.3f}s, end_offset={end_offset:+.3f}s"
            )
            if messages:
                click.echo(f"  note:    {', '.join(messages)}")

    click.echo("")
    click.echo(
        f"Summary: checked={total}, ok={total - warn_count - error_count}, warn={warn_count}, error={error_count}"
    )


if __name__ == "__main__":
    main()
