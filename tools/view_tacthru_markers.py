#!/usr/bin/env python3
"""Display live TacThru RGB frames with marker displacement overlays."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TACTHRU_REPO = PROJECT_ROOT.parent / "tacthru"
WINDOW_NAME = "TacThru Marker Viewer"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show the live TacThru camera and tracked marker displacement. "
            "This viewer does not record data or send robot/gripper commands."
        )
    )
    parser.add_argument(
        "--tacthru-repo",
        type=Path,
        default=DEFAULT_TACTHRU_REPO,
        help="TacThru repository root (default: sibling tacthru repository)",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=None,
        help="Sensor YAML (default: TACTHRU_REPO/cfg/sensor/ml.yaml)",
    )
    parser.add_argument(
        "--arrow-scale",
        type=_positive_float,
        default=6.0,
        help="Display-only marker arrow amplification (default: 6)",
    )
    parser.add_argument(
        "--window-scale",
        type=_positive_float,
        default=1.5,
        help="Initial OpenCV window scale (default: 1.5)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=3.0,
        help="Seconds to wait for the first/new camera frame (default: 3)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Track markers without opening a GUI window (for diagnostics)",
    )
    parser.add_argument(
        "--max-frames",
        type=_non_negative_int,
        default=0,
        help="Exit after this many fresh frames; 0 means unlimited",
    )
    return parser.parse_args()


def _resolve_sensor_config(tacthru_repo: Path, sensor_config: Path | None) -> Path:
    if sensor_config is None:
        sensor_config = tacthru_repo / "cfg/sensor/ml.yaml"
    return sensor_config.expanduser().resolve()


def _validate_camera_device(sensor_config: Path) -> None:
    from omegaconf import OmegaConf

    sensor_cfg = OmegaConf.load(sensor_config)
    camera_value = str(sensor_cfg.cam_path)
    if not camera_value.startswith("/dev/"):
        return
    camera_path = Path(camera_value)
    if camera_path.exists():
        return
    by_id_root = Path("/dev/v4l/by-id")
    available = sorted(str(path) for path in by_id_root.glob("*-video-index0"))
    available_text = "\n  ".join(available) if available else "(none)"
    raise FileNotFoundError(
        f"Configured TacThru camera is not connected: {camera_path}\n"
        f"Available video-index0 devices:\n  {available_text}"
    )


def reconstruct_current_pixels(
    reference_pixels: np.ndarray,
    normalized_displacement: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    reference_pixels = np.asarray(reference_pixels, dtype=np.float32)
    normalized_displacement = np.asarray(normalized_displacement, dtype=np.float32)
    if normalized_displacement.shape != reference_pixels.shape:
        raise ValueError(
            "Marker displacement/reference shape mismatch: "
            f"{normalized_displacement.shape} != {reference_pixels.shape}"
        )
    scale_xy = np.asarray([image_width, image_height], dtype=np.float32)
    return reference_pixels + 0.5 * normalized_displacement * scale_xy


def _draw_marker_overlay(
    *,
    tactile_rgb: np.ndarray,
    reference_pixels: np.ndarray,
    normalized_displacement: np.ndarray,
    valid_mask: np.ndarray,
    arrow_scale: float,
    capture_fps: float,
    detected_count: int | None,
) -> np.ndarray:
    image_rgb = np.asarray(tactile_rgb, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    displacement = np.asarray(normalized_displacement, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (len(reference_pixels),):
        raise ValueError(
            f"Marker valid mask must have shape {(len(reference_pixels),)}, got {valid.shape}"
        )

    current_pixels = reconstruct_current_pixels(
        reference_pixels,
        displacement,
        image_width=width,
        image_height=height,
    )
    amplified_pixels = reference_pixels + arrow_scale * (current_pixels - reference_pixels)

    for reference, arrow_end, is_valid in zip(reference_pixels, amplified_pixels, valid):
        if not is_valid:
            continue
        cv2.arrowedLine(
            image,
            tuple(np.rint(reference).astype(int)),
            tuple(np.rint(arrow_end).astype(int)),
            (0, 205, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.22,
        )

    for reference, current, is_valid in zip(reference_pixels, current_pixels, valid):
        reference_point = tuple(np.rint(reference).astype(int))
        if not is_valid:
            cv2.drawMarker(
                image,
                reference_point,
                (160, 160, 160),
                cv2.MARKER_TILTED_CROSS,
                9,
                1,
                cv2.LINE_AA,
            )
            continue
        cv2.circle(image, reference_point, 6, (30, 220, 70), 2, cv2.LINE_AA)
        cv2.circle(
            image,
            tuple(np.rint(current).astype(int)),
            4,
            (30, 30, 235),
            -1,
            cv2.LINE_AA,
        )

    header_height = 78
    canvas = np.full((height + header_height, width, 3), (20, 20, 20), dtype=np.uint8)
    canvas[header_height:] = image
    valid_norms = np.linalg.norm(displacement[valid], axis=-1)
    mean_norm = float(valid_norms.mean()) if len(valid_norms) else 0.0
    max_norm = float(valid_norms.max()) if len(valid_norms) else 0.0
    detected_text = "n/a" if detected_count is None else str(detected_count)
    cv2.putText(
        canvas,
        f"TacThru live marker view | FPS {capture_fps:4.1f}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"mean|d|={mean_norm:.5f}  max|d|={max_norm:.5f}  "
            f"finite={int(valid.sum())}/{len(valid)}  candidates={detected_text}"
        ),
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    legend_y = 69
    cv2.circle(canvas, (16, legend_y), 4, (30, 220, 70), -1, cv2.LINE_AA)
    cv2.putText(canvas, "ref", (25, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(canvas, (73, legend_y), 4, (30, 30, 235), -1, cv2.LINE_AA)
    cv2.putText(canvas, "current", (82, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (154, legend_y), (183, legend_y), (0, 205, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        canvas,
        f"displacement x{arrow_scale:g}",
        (190, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    tacthru_repo = args.tacthru_repo.expanduser().resolve()
    sensor_config = _resolve_sensor_config(tacthru_repo, args.sensor_config)
    if not tacthru_repo.is_dir():
        raise NotADirectoryError(f"TacThru repository not found: {tacthru_repo}")
    if not sensor_config.is_file():
        raise FileNotFoundError(f"Sensor config not found: {sensor_config}")
    _validate_camera_device(sensor_config)

    repo_string = str(tacthru_repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from multiprocessing.managers import SharedMemoryManager

    from omegaconf import OmegaConf
    from real_world.sensor_utils import TacThruClient

    sensor_cfg = OmegaConf.load(sensor_config)
    tracking_path = Path(str(sensor_cfg.tracking.tracking_pts_path)).expanduser()
    if not tracking_path.is_absolute():
        sensor_cfg.tracking.tracking_pts_path = str(
            (tacthru_repo / tracking_path).resolve()
        )

    print("[tacthru-viewer] view-only mode: recording is disabled; no robot or gripper command is sent")
    print(f"[tacthru-viewer] sensor config: {sensor_config}")
    if not args.headless:
        print("[tacthru-viewer] press q or Esc in the window to exit")

    frame_count = 0
    last_timestamp: float | None = None
    last_arrival: float | None = None
    fps_ema = 0.0
    window_created = False
    manager = SharedMemoryManager()
    sensor = None
    try:
        manager.start()
        sensor = TacThruClient(
            manager,
            None,
            sensor_cfg,
            "cuda:0",
            False,
            str(sensor_cfg.name),
            do_tracking=True,
        )
        sensor.start(wait=True)
        deadline = time.monotonic() + args.timeout
        while int(sensor.ring_buffer.count) < 1:
            if not sensor.is_alive():
                raise RuntimeError("TacThru sensor process exited before producing a frame")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the first TacThru frame")
            time.sleep(0.01)

        while args.max_frames == 0 or frame_count < args.max_frames:
            if not sensor.is_alive():
                raise RuntimeError("TacThru sensor process stopped unexpectedly")
            sensor_data = sensor.get(k=1)
            capture_timestamp = float(np.asarray(sensor_data["timestamp"])[-1])
            if last_timestamp is not None and capture_timestamp <= last_timestamp:
                time.sleep(0.002)
                continue
            last_timestamp = capture_timestamp

            now = time.monotonic()
            if last_arrival is not None and now > last_arrival:
                instantaneous_fps = 1.0 / (now - last_arrival)
                fps_ema = instantaneous_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * instantaneous_fps
            last_arrival = now

            tactile_rgb = np.asarray(sensor_data["rgb"][-1], dtype=np.uint8)
            marker = np.asarray(sensor_data["marker"][-1], dtype=np.float32)
            reference_pixels = np.asarray(
                sensor_data["marker_ref"][-1], dtype=np.float32
            )
            valid = np.isfinite(marker).all(axis=-1)
            valid &= np.isfinite(reference_pixels).all(axis=-1)
            height, width = tactile_rgb.shape[:2]
            scale_xy = np.asarray([width, height], dtype=np.float32)
            displacement = np.where(
                valid[:, None],
                (marker - reference_pixels) / scale_xy * 2.0,
                0.0,
            ).astype(np.float32, copy=False)
            detected_array = np.asarray(sensor_data.get("n_all_kpts", []))
            detected_count = (
                int(detected_array.reshape(-1)[-1]) if detected_array.size else None
            )
            canvas = _draw_marker_overlay(
                tactile_rgb=tactile_rgb,
                reference_pixels=reference_pixels,
                normalized_displacement=displacement,
                valid_mask=valid,
                arrow_scale=args.arrow_scale,
                capture_fps=fps_ema,
                detected_count=detected_count,
            )
            frame_count += 1

            if args.headless:
                if frame_count == 1 or frame_count % 30 == 0:
                    valid_norms = np.linalg.norm(displacement[valid], axis=-1)
                    mean_norm = float(valid_norms.mean()) if len(valid_norms) else 0.0
                    print(
                        f"[tacthru-viewer] frame={frame_count} fps={fps_ema:.1f} "
                        f"mean|d|={mean_norm:.5f} finite={int(valid.sum())}/{len(valid)} "
                        f"candidates={detected_count}"
                    )
                continue

            if not window_created:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    WINDOW_NAME,
                    int(canvas.shape[1] * args.window_scale),
                    int(canvas.shape[0] * args.window_scale),
                )
                window_created = True
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if sensor is not None:
            sensor.stop(wait=False)
            sensor.join(timeout=3.0)
            if sensor.is_alive():
                sensor.terminate()
                sensor.join(timeout=1.0)
        manager.shutdown()
        if window_created:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)

    print(f"[tacthru-viewer] stopped after {frame_count} fresh frames")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(f"[tacthru-viewer] {error}", file=sys.stderr)
        raise SystemExit(2) from None
