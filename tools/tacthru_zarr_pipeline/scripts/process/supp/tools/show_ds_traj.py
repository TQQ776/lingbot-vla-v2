from __future__ import annotations

import contextlib
import math
import pickle
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
except Exception:  # pragma: no cover - optional legacy dependency
    register_codecs = None


DEFAULT_RGB_KEYS = ("camera0_rgb", "tacthru_l_rgb", "tacthru_r_rgb")
MARKER_KEYS = {"tacthru_l_rgb": "tacthru_l_marker", "tacthru_r_rgb": "tacthru_r_marker"}
REF_FILENAMES = {
    "tacthru_l_rgb": ("tacthru_l_marker_ref.npy", "tacthru_l_ref_kpts.npy"),
    "tacthru_r_rgb": ("tacthru_r_marker_ref.npy", "tacthru_r_ref_kpts.npy"),
}
TRAJ_COLORS = {
    "x": (70, 180, 255),
    "y": (110, 220, 110),
    "z": (255, 140, 80),
}
DEFAULT_DATASET_FPS = 30.0


def _available_rgb_keys(data_group: zarr.Group) -> tuple[str, ...]:
    rgb_keys = tuple(key for key in DEFAULT_RGB_KEYS if key in data_group)
    if "camera0_rgb" not in rgb_keys:
        raise click.ClickException("Dataset is missing required RGB array: camera0_rgb")
    return rgb_keys


def _maybe_register_codecs() -> None:
    if register_codecs is not None:
        try:
            register_codecs()
        except Exception:
            pass


def _open_zarr_root(dataset_path: Path) -> tuple[contextlib.ExitStack, zarr.Group]:
    stack = contextlib.ExitStack()
    if dataset_path.is_dir():
        store = zarr.DirectoryStore(str(dataset_path))
    else:
        store = stack.enter_context(zarr.ZipStore(str(dataset_path), mode="r"))
    root = zarr.group(store=store)
    return stack, root


def _candidate_task_dirs(dataset_path: Path) -> list[Path]:
    stem = dataset_path.name
    for suffix in (".zarr.zip", ".zip", ".zarr"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidates: list[Path] = [dataset_path.parent]
    stem_parts = stem.split("-")
    if stem_parts:
        candidates.append(dataset_path.parent / stem_parts[0])
    for sibling in dataset_path.parent.iterdir():
        if sibling.is_dir() and sibling not in candidates:
            if stem.startswith(sibling.name) or sibling.name.startswith(stem_parts[0]):
                candidates.append(sibling)
    return candidates


def _load_marker_ref(dataset_path: Path, explicit_path: Path | None, rgb_key: str) -> np.ndarray | None:
    ref_path = explicit_path
    if ref_path is None:
        for task_dir in _candidate_task_dirs(dataset_path):
            for filename in REF_FILENAMES[rgb_key]:
                candidate = task_dir / filename
                if candidate.is_file():
                    ref_path = candidate
                    break
            if ref_path is not None:
                break

    if ref_path is None:
        return None

    ref = np.asarray(np.load(ref_path.expanduser().resolve()), dtype=np.float32)
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise click.ClickException(f"Invalid marker reference array at {ref_path}: expected (N, 2), got {ref.shape}")

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    if 0.0 <= ref_min and ref_max <= 1.0:
        return ref * 2.0 - 1.0
    if -1.1 <= ref_min and ref_max <= 1.1:
        return ref
    raise click.ClickException(
        f"Unsupported marker reference coordinate range at {ref_path}: min={ref_min:.4f}, max={ref_max:.4f}. Expected either [0, 1] or [-1, 1]."
    )


def _infer_dataset_fps(dataset_path: Path) -> float:
    for task_dir in _candidate_task_dirs(dataset_path):
        plan_path = task_dir / "dataset_plan_vive.pkl"
        if not plan_path.is_file():
            continue
        try:
            import pickle

            with plan_path.open("rb") as f:
                plan = pickle.load(f)
            fps_values: list[float] = []
            for episode in plan:
                for camera in episode.get("cameras", []):
                    fps = camera.get("fps")
                    if fps is not None and float(fps) > 0:
                        fps_values.append(float(fps))
            if fps_values:
                return float(np.median(np.asarray(fps_values, dtype=np.float64)))
        except Exception:
            continue
    return DEFAULT_DATASET_FPS


def _load_episode_timestamps(dataset_path: Path, episode_ends: np.ndarray) -> list[np.ndarray] | None:
    expected_lengths = np.diff(np.concatenate([[0], np.asarray(episode_ends, dtype=np.int64)]))
    for task_dir in _candidate_task_dirs(dataset_path):
        plan_path = task_dir / "dataset_plan_vive.pkl"
        if not plan_path.is_file():
            continue
        try:
            with plan_path.open("rb") as f:
                plan = pickle.load(f)
        except Exception:
            continue

        if len(plan) != len(expected_lengths):
            continue

        episode_timestamps: list[np.ndarray] = []
        valid = True
        for episode, expected_length in zip(plan, expected_lengths, strict=True):
            ts = np.asarray(episode.get("episode_timestamps", []), dtype=np.float64)
            if len(ts) != int(expected_length):
                valid = False
                break
            episode_timestamps.append(ts)

        if valid:
            return episode_timestamps
    return None


def _draw_arrows(
    image: np.ndarray,
    keypoints_start: np.ndarray,
    keypoints_end: np.ndarray,
    line_scale: float = 3.0,
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
    tip_length: float = 0.25,
) -> np.ndarray:
    keypoints_start = np.asarray(keypoints_start, dtype=np.float32)
    keypoints_end = np.asarray(keypoints_end, dtype=np.float32)
    keypoints_diff = (keypoints_end - keypoints_start) * float(line_scale)
    viz_keypoints_end = keypoints_start + keypoints_diff

    viz_keypoints_start = np.round(keypoints_start).astype(np.int32)
    viz_keypoints_end = np.round(viz_keypoints_end).astype(np.int32)

    image_draw = image.copy()
    for keypoint_start, keypoint_end in zip(viz_keypoints_start, viz_keypoints_end):
        image_draw = cv2.arrowedLine(
            image_draw,
            tuple(keypoint_start),
            tuple(keypoint_end),
            color=color,
            thickness=thickness,
            line_type=line_type,
            tipLength=tip_length,
        )
    return image_draw


def _overlay_marker_flow(
    image_rgb: np.ndarray,
    marker_flow: np.ndarray | None,
    marker_ref_canonical: np.ndarray | None,
    arrow_scale: float,
) -> np.ndarray:
    image_bgr = np.asarray(image_rgb, dtype=np.uint8)[..., ::-1].copy()
    if marker_flow is None or marker_ref_canonical is None:
        return image_bgr

    height, width = image_bgr.shape[:2]
    scale_xy = np.array([width, height], dtype=np.float32)
    marker_ref_canonical = np.asarray(marker_ref_canonical, dtype=np.float32)
    marker_flow = np.asarray(marker_flow, dtype=np.float32)
    common = min(len(marker_ref_canonical), len(marker_flow))
    if common <= 0:
        return image_bgr
    marker_ref_canonical = marker_ref_canonical[:common]
    marker_flow = marker_flow[:common]
    marker_ref_px = (marker_ref_canonical + 1.0) * 0.5 * scale_xy
    marker_end_px = (marker_ref_canonical + marker_flow + 1.0) * 0.5 * scale_xy
    return _draw_arrows(image=image_bgr, keypoints_start=marker_ref_px, keypoints_end=marker_end_px, line_scale=arrow_scale)


def _annotate_panel(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (24, 24, 24), thickness=-1)
    cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _annotate_camera_timestamp(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    h, w = image.shape[:2]
    y0 = max(0, h - 28)
    cv2.rectangle(image, (0, y0), (w, h), (24, 24, 24), thickness=-1)
    cv2.putText(image, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * (height / float(image.shape[0])))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _make_plot_panel(width: int, height: int, title: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    rect = (54, 34, width - 24, height - 40)
    cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), thickness=-1)
    cv2.rectangle(panel, (rect[0], rect[1]), (rect[2], rect[3]), (210, 210, 210), thickness=1)
    cv2.putText(panel, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return panel, rect


def _expand_bounds(vmin: float, vmax: float, pad_ratio: float = 0.08) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if math.isclose(vmin, vmax):
        center = 0.5 * (vmin + vmax)
        return center - 0.5, center + 0.5
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


def _project_xy(points: np.ndarray, rect: tuple[int, int, int, int], xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    px = (points[:, 0] - xlim[0]) / max(xlim[1] - xlim[0], 1e-8)
    py = (points[:, 1] - ylim[0]) / max(ylim[1] - ylim[0], 1e-8)
    px = x0 + px * (x1 - x0)
    py = y1 - py * (y1 - y0)
    return np.round(np.stack([px, py], axis=-1)).astype(np.int32)


def _draw_axes_grid(panel: np.ndarray, rect: tuple[int, int, int, int], x_ticks: int = 4, y_ticks: int = 4) -> None:
    x0, y0, x1, y1 = rect
    for idx in range(1, x_ticks):
        x = int(round(x0 + (x1 - x0) * idx / x_ticks))
        cv2.line(panel, (x, y0), (x, y1), (235, 235, 235), 1, cv2.LINE_AA)
    for idx in range(1, y_ticks):
        y = int(round(y0 + (y1 - y0) * idx / y_ticks))
        cv2.line(panel, (x0, y), (x1, y), (235, 235, 235), 1, cv2.LINE_AA)


def _draw_xy_plot(
    title: str,
    points: np.ndarray,
    current_local_idx: int,
    labels: tuple[str, str],
    colors: tuple[int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    panel, rect = _make_plot_panel(width=width, height=height, title=title)
    _draw_axes_grid(panel, rect)

    pts = np.asarray(points, dtype=np.float32)
    xlim = _expand_bounds(float(np.min(pts[:, 0])), float(np.max(pts[:, 0])))
    ylim = _expand_bounds(float(np.min(pts[:, 1])), float(np.max(pts[:, 1])))
    pts_px = _project_xy(pts, rect, xlim=xlim, ylim=ylim)

    cv2.polylines(panel, [pts_px], isClosed=False, color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
    history_px = pts_px[: current_local_idx + 1]
    if len(history_px) >= 2:
        cv2.polylines(panel, [history_px], isClosed=False, color=colors, thickness=2, lineType=cv2.LINE_AA)
    if len(history_px) >= 1:
        current_pt = tuple(history_px[-1])
        cv2.circle(panel, current_pt, 4, (32, 32, 220), thickness=-1, lineType=cv2.LINE_AA)

    cv2.putText(panel, labels[0], (rect[2] - 18, rect[3] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(panel, labels[1], (12, rect[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{xlim[0]:.3f}", (rect[0] - 10, rect[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{xlim[1]:.3f}", (rect[2] - 48, rect[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{ylim[0]:.3f}", (4, rect[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{ylim[1]:.3f}", (4, rect[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    return panel


def _draw_series_plot(
    title: str,
    series: np.ndarray,
    current_local_idx: int,
    labels: tuple[str, ...],
    colors: tuple[tuple[int, int, int], ...],
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    panel, rect = _make_plot_panel(width=width, height=height, title=title)
    _draw_axes_grid(panel, rect)

    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]

    x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    vmin, vmax = _expand_bounds(float(np.min(values)), float(np.max(values)))
    x_px = rect[0] + x * (rect[2] - rect[0])

    for dim in range(values.shape[1]):
        y_norm = (values[:, dim] - vmin) / max(vmax - vmin, 1e-8)
        y_px = rect[3] - y_norm * (rect[3] - rect[1])
        pts = np.round(np.stack([x_px, y_px], axis=-1)).astype(np.int32)
        cv2.polylines(panel, [pts], isClosed=False, color=(215, 215, 215), thickness=1, lineType=cv2.LINE_AA)
        hist_pts = pts[: current_local_idx + 1]
        if len(hist_pts) >= 2:
            cv2.polylines(panel, [hist_pts], isClosed=False, color=colors[dim], thickness=2, lineType=cv2.LINE_AA)
        if len(hist_pts) >= 1:
            cv2.circle(panel, tuple(hist_pts[-1]), 3, colors[dim], thickness=-1, lineType=cv2.LINE_AA)

    current_x = int(round(rect[0] + (rect[2] - rect[0]) * current_local_idx / max(len(values) - 1, 1)))
    cv2.line(panel, (current_x, rect[1]), (current_x, rect[3]), (90, 90, 90), 1, cv2.LINE_AA)

    legend_x = 14
    for label, color in zip(labels, colors):
        cv2.line(panel, (legend_x, height - 14), (legend_x + 16, height - 14), color, 2, cv2.LINE_AA)
        cv2.putText(panel, label, (legend_x + 22, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1, cv2.LINE_AA)
        legend_x += 80

    cv2.putText(panel, f"{vmax:.3f}", (4, rect[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{vmin:.3f}", (4, rect[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, "episode time", (rect[2] - 94, rect[3] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1, cv2.LINE_AA)
    return panel


def _episode_bounds(episode_ends: np.ndarray, frame_idx: int) -> tuple[int, int, int]:
    episode_idx = int(np.searchsorted(episode_ends, frame_idx, side="right"))
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return episode_idx, start, end


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    mins, ms_rem = divmod(total_ms, 60_000)
    secs, millis = divmod(ms_rem, 1000)
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


def _compose_canvas(
    data_group: zarr.Group,
    rgb_keys: tuple[str, ...],
    frame_idx: int,
    episode_ends: np.ndarray,
    left_marker_ref: np.ndarray | None,
    right_marker_ref: np.ndarray | None,
    arrow_scale: float,
    dataset_fps: float,
    episode_timestamps: list[np.ndarray] | None,
) -> np.ndarray:
    marker_refs = {"tacthru_l_rgb": left_marker_ref, "tacthru_r_rgb": right_marker_ref}
    episode_idx, episode_start, episode_end = _episode_bounds(episode_ends=episode_ends, frame_idx=frame_idx)
    local_idx = frame_idx - episode_start

    video_panels: list[np.ndarray] = []
    for rgb_key in rgb_keys:
        image_rgb = np.asarray(data_group[rgb_key][frame_idx], dtype=np.uint8)
        marker_flow = None
        if rgb_key in MARKER_KEYS and MARKER_KEYS[rgb_key] in data_group:
            marker_flow = np.asarray(data_group[MARKER_KEYS[rgb_key]][frame_idx], dtype=np.float32)
        image_bgr = _overlay_marker_flow(
            image_rgb=image_rgb,
            marker_flow=marker_flow,
            marker_ref_canonical=marker_refs.get(rgb_key),
            arrow_scale=arrow_scale,
        )
        image_bgr = _annotate_panel(image_bgr, rgb_key)
        if rgb_key == "camera0_rgb":
            if episode_timestamps is not None:
                curr_episode_ts = episode_timestamps[episode_idx]
                episode_time = float(curr_episode_ts[local_idx] - curr_episode_ts[0])
                global_time = episode_time
                for prev_idx in range(episode_idx):
                    prev_episode_ts = episode_timestamps[prev_idx]
                    if len(prev_episode_ts) > 0:
                        global_time += float(prev_episode_ts[-1] - prev_episode_ts[0])
            else:
                global_time = frame_idx / dataset_fps
                episode_time = local_idx / dataset_fps
            image_bgr = _annotate_camera_timestamp(
                image_bgr,
                f"t={_format_seconds(global_time)} | ep={_format_seconds(episode_time)}",
            )
        video_panels.append(image_bgr)

    target_height = max(panel.shape[0] for panel in video_panels)
    video_panels = [_resize_to_height(panel, target_height) for panel in video_panels]
    video_canvas = np.hstack(video_panels)

    pos = np.asarray(data_group["robot0_eef_pos"][episode_start:episode_end], dtype=np.float32)
    rot = np.asarray(data_group["robot0_eef_rot_axis_angle"][episode_start:episode_end], dtype=np.float32)
    grip = np.asarray(data_group["robot0_gripper_width"][episode_start:episode_end], dtype=np.float32).reshape(-1)

    plot_width = video_canvas.shape[1] // 2
    plot_height = max(210, int(round(video_canvas.shape[0] * 0.92)))
    xy_panel = _draw_xy_plot(
        title="EEF XY (m)",
        points=pos[:, [0, 1]],
        current_local_idx=local_idx,
        labels=("x", "y"),
        colors=(80, 140, 255),
        size=(plot_width, plot_height),
    )
    xz_panel = _draw_xy_plot(
        title="EEF XZ (m)",
        points=pos[:, [0, 2]],
        current_local_idx=local_idx,
        labels=("x", "z"),
        colors=(255, 160, 90),
        size=(plot_width, plot_height),
    )
    rot_panel = _draw_series_plot(
        title="Axis-Angle Rotation (rad)",
        series=rot,
        current_local_idx=local_idx,
        labels=("rx", "ry", "rz"),
        colors=(TRAJ_COLORS["x"], TRAJ_COLORS["y"], TRAJ_COLORS["z"]),
        size=(plot_width, plot_height),
    )
    grip_panel = _draw_series_plot(
        title="Gripper Width (m)",
        series=grip,
        current_local_idx=local_idx,
        labels=("width",),
        colors=((180, 90, 255),),
        size=(plot_width, plot_height),
    )
    bottom_canvas = np.vstack([np.hstack([xy_panel, xz_panel]), np.hstack([rot_panel, grip_panel])])

    if bottom_canvas.shape[1] != video_canvas.shape[1]:
        bottom_canvas = cv2.resize(bottom_canvas, (video_canvas.shape[1], bottom_canvas.shape[0]), interpolation=cv2.INTER_LINEAR)

    canvas = np.vstack([video_canvas, bottom_canvas])
    status = (
        f"frame {frame_idx + 1}/{data_group[rgb_keys[0]].shape[0]} | "
        f"episode {episode_idx + 1}/{len(episode_ends)} | "
        f"episode_frame {local_idx + 1}/{episode_end - episode_start}"
    )
    cv2.rectangle(canvas, (0, canvas.shape[0] - 30), (canvas.shape[1], canvas.shape[0]), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, status, (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _create_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc_candidates = ("mp4v", "avc1", "H264")
    for fourcc_name in fourcc_candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    raise click.ClickException(f"Failed to open video writer for {path}")


@click.command(help="Visualize a TacThru UMI dataset with RGB/tactile streams and robot trajectory plots.")
@click.argument("dataset_path", type=click.Path(path_type=Path, exists=True))
@click.option("--left-marker-ref", type=click.Path(path_type=Path), default=None, help="Override the left tactile marker reference .npy.")
@click.option("--right-marker-ref", type=click.Path(path_type=Path), default=None, help="Override the right tactile marker reference .npy.")
@click.option("--fps", type=float, default=20.0, show_default=True, help="Playback or export fps.")
@click.option("--start", type=int, default=0, show_default=True, help="Starting frame index.")
@click.option("--stride", type=int, default=1, show_default=True, help="Frame increment while playing/exporting.")
@click.option("--max-frames", type=int, default=None, help="Optional limit on displayed/exported frames.")
@click.option("--scale", type=float, default=1.0, show_default=True, help="Scale the final rendered image.")
@click.option("--arrow-scale", type=float, default=3.0, show_default=True, help="Arrow length multiplier over the marker displacement.")
@click.option(
    "--dataset-fps",
    type=float,
    default=None,
    help="Time base used for the camera timestamp overlay. Defaults to the median fps from dataset_plan_vive.pkl when available.",
)
@click.option("--window-name", type=str, default="show_ds_traj", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Export the visualization to a video file instead of opening a window.")
def main(
    dataset_path: Path,
    left_marker_ref: Path | None,
    right_marker_ref: Path | None,
    fps: float,
    start: int,
    stride: int,
    max_frames: int | None,
    scale: float,
    arrow_scale: float,
    dataset_fps: float | None,
    window_name: str,
    output: Path | None,
) -> None:
    _maybe_register_codecs()

    dataset_path = dataset_path.expanduser().resolve()
    if fps <= 0:
        raise click.ClickException("--fps must be positive.")
    if stride <= 0:
        raise click.ClickException("--stride must be positive.")
    if scale <= 0:
        raise click.ClickException("--scale must be positive.")
    if dataset_fps is not None and dataset_fps <= 0:
        raise click.ClickException("--dataset-fps must be positive.")

    stack, root = _open_zarr_root(dataset_path)
    with stack:
        if "data" not in root:
            raise click.ClickException(f"Dataset is missing /data: {dataset_path}")
        data_group = root["data"]
        meta_group = root["meta"] if "meta" in root else None

        rgb_keys = _available_rgb_keys(data_group)
        for key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"):
            if key not in data_group:
                raise click.ClickException(f"Dataset is missing required trajectory array: {key}")

        episode_ends = (
            np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
            if meta_group is not None and "episode_ends" in meta_group
            else np.array([data_group[rgb_keys[0]].shape[0]], dtype=np.int64)
        )

        left_ref = (
            _load_marker_ref(dataset_path=dataset_path, explicit_path=left_marker_ref, rgb_key="tacthru_l_rgb")
            if "tacthru_l_rgb" in rgb_keys
            else None
        )
        right_ref = (
            _load_marker_ref(dataset_path=dataset_path, explicit_path=right_marker_ref, rgb_key="tacthru_r_rgb")
            if "tacthru_r_rgb" in rgb_keys
            else None
        )
        resolved_dataset_fps = float(dataset_fps) if dataset_fps is not None else _infer_dataset_fps(dataset_path)
        episode_timestamps = _load_episode_timestamps(dataset_path=dataset_path, episode_ends=episode_ends)

        total_frames = int(data_group[rgb_keys[0]].shape[0])
        if total_frames <= 0:
            raise click.ClickException(f"Dataset is empty: {dataset_path}")

        frame_idx = min(max(0, int(start)), total_frames - 1)
        shown_frames = 0

        first_canvas = _compose_canvas(
            data_group=data_group,
            rgb_keys=rgb_keys,
            frame_idx=frame_idx,
            episode_ends=episode_ends,
            left_marker_ref=left_ref,
            right_marker_ref=right_ref,
            arrow_scale=arrow_scale,
            dataset_fps=resolved_dataset_fps,
            episode_timestamps=episode_timestamps,
        )
        if scale != 1.0:
            first_canvas = cv2.resize(first_canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if output is not None:
            output = output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = _create_video_writer(output, fps=fps, frame_size=(first_canvas.shape[1], first_canvas.shape[0]))
            try:
                curr_frame_idx = frame_idx
                while True:
                    canvas = _compose_canvas(
                        data_group=data_group,
                        rgb_keys=rgb_keys,
                        frame_idx=curr_frame_idx,
                        episode_ends=episode_ends,
                        left_marker_ref=left_ref,
                        right_marker_ref=right_ref,
                        arrow_scale=arrow_scale,
                        dataset_fps=resolved_dataset_fps,
                        episode_timestamps=episode_timestamps,
                    )
                    if scale != 1.0:
                        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    writer.write(canvas)
                    shown_frames += 1

                    if max_frames is not None and shown_frames >= max_frames:
                        break
                    curr_frame_idx += stride
                    if curr_frame_idx >= total_frames:
                        break
                print(f"Exported {shown_frames} frame(s) to {output}")
            finally:
                writer.release()
            return

        print(f"Viewing {dataset_path}")
        if episode_timestamps is not None:
            print("Timestamp time base: exact episode_timestamps from dataset_plan_vive.pkl")
        else:
            print(f"Timestamp time base: fallback {resolved_dataset_fps:.3f} fps")
        print("Controls: q/ESC quit, space pause, j prev, k next, , back 10, . forward 10")
        paused = False
        delay_ms = max(1, int(round(1000.0 / fps)))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        curr_frame_idx = frame_idx
        while True:
            canvas = _compose_canvas(
                data_group=data_group,
                rgb_keys=rgb_keys,
                frame_idx=curr_frame_idx,
                episode_ends=episode_ends,
                left_marker_ref=left_ref,
                right_marker_ref=right_ref,
                arrow_scale=arrow_scale,
                dataset_fps=resolved_dataset_fps,
                episode_timestamps=episode_timestamps,
            )
            if scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, canvas)
            shown_frames += 1

            if max_frames is not None and shown_frames >= max_frames:
                break

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("j"):
                curr_frame_idx = max(0, curr_frame_idx - stride)
                paused = True
                continue
            if key == ord("k"):
                curr_frame_idx = min(total_frames - 1, curr_frame_idx + stride)
                paused = True
                continue
            if key == ord(","):
                curr_frame_idx = max(0, curr_frame_idx - 10 * stride)
                paused = True
                continue
            if key == ord("."):
                curr_frame_idx = min(total_frames - 1, curr_frame_idx + 10 * stride)
                paused = True
                continue

            if not paused:
                curr_frame_idx += stride
                if curr_frame_idx >= total_frames:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
