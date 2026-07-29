from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_v4l2ctl(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["v4l2-ctl", *args]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"v4l2-ctl exited with code {result.returncode}"
        raise RuntimeError(f"{' '.join(cmd)} failed: {detail}")
    return result


def _load_camera_cfg(camera_cfg: Path) -> dict:
    with camera_cfg.open("r") as f:
        return yaml.safe_load(f)


def _resolve_device(camera_cfg: Path | None, camera_dev: str | None) -> tuple[str, list[int] | None]:
    if camera_dev is not None:
        return os.path.realpath(camera_dev), [1280, 720]
    if camera_cfg is None:
        raise ValueError("camera_cfg is required when camera_dev is not provided")
    cfg = _load_camera_cfg(camera_cfg)
    return os.path.realpath(str(cfg["dev_video_path"])), cfg.get("resolution")


def _frame_brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _capture_brightness_stats(
    camera_dev: str,
    resolution: list[int] | None,
    warmup_frames: int,
    sample_frames: int,
) -> tuple[dict, np.ndarray | None]:
    cap = cv2.VideoCapture(camera_dev)
    if not cap.isOpened():
        return {
            "open_ok": False,
            "reads": 0,
            "ok_reads": 0,
            "brightness_min": float("nan"),
            "brightness_max": float("nan"),
            "brightness_mean": float("nan"),
        }, None

    try:
        if resolution and len(resolution) == 2:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))

        for _ in range(max(0, int(warmup_frames))):
            cap.read()

        brightness_values: list[float] = []
        best_frame = None
        best_brightness = -1.0
        ok_reads = 0

        for _ in range(max(1, int(sample_frames))):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            ok_reads += 1
            brightness = _frame_brightness(frame_bgr)
            brightness_values.append(brightness)
            if brightness > best_brightness:
                best_brightness = brightness
                best_frame = frame_bgr.copy()

        stats = {
            "open_ok": True,
            "reads": max(1, int(sample_frames)),
            "ok_reads": ok_reads,
            "brightness_min": float(min(brightness_values)) if brightness_values else float("nan"),
            "brightness_max": float(max(brightness_values)) if brightness_values else float("nan"),
            "brightness_mean": float(np.mean(brightness_values)) if brightness_values else float("nan"),
        }
        return stats, best_frame
    finally:
        cap.release()


def _save_preview(frame_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f"Failed to write preview image to {output_path}")


def _build_ctrl_assignments(
    set_ctrl: tuple[str, ...],
    brightness: int | None,
    contrast: int | None,
    saturation: int | None,
    gain: int | None,
    sharpness: int | None,
    backlight_compensation: int | None,
    white_balance_automatic: bool | None,
    white_balance_temperature: int | None,
    exposure_auto: int | None,
    exposure_absolute: int | None,
) -> list[str]:
    ctrl_pairs: list[str] = list(set_ctrl)

    def add(name: str, value) -> None:
        if value is not None:
            ctrl_pairs.append(f"{name}={value}")

    add("brightness", brightness)
    add("contrast", contrast)
    add("saturation", saturation)
    add("gain", gain)
    add("sharpness", sharpness)
    add("backlight_compensation", backlight_compensation)
    if white_balance_automatic is not None:
        add("white_balance_automatic", 1 if white_balance_automatic else 0)
    add("white_balance_temperature", white_balance_temperature)
    add("exposure_auto", exposure_auto)
    add("exposure_absolute", exposure_absolute)
    return ctrl_pairs


@click.command(help="List and control UVC/V4L2 camera parameters such as brightness, exposure, gain, and white balance.")
@click.option(
    "--camera-cfg", type=click.Path(path_type=Path, exists=True), default=REPO_ROOT / "cfg" / "camera" / "synria_c10.yaml", show_default=True
)
@click.option("--camera-dev", type=str, default=None, help="Optional explicit /dev/video* or /dev/v4l/by-id/* path. Overrides --camera-cfg.")
@click.option("--list-devices", is_flag=True, default=False, help="Print v4l2 devices and exit.")
@click.option("--list-controls", is_flag=True, default=False, help="Print controls for the selected device.")
@click.option("--all", "show_all", is_flag=True, default=False, help="Print full v4l2-ctl --all for the selected device.")
@click.option("--set-ctrl", multiple=True, help="Raw v4l2 control assignment, e.g. --set-ctrl brightness=96.")
@click.option("--brightness", type=int, default=None, help="Set brightness.")
@click.option("--contrast", type=int, default=None, help="Set contrast.")
@click.option("--saturation", type=int, default=None, help="Set saturation.")
@click.option("--gain", type=int, default=None, help="Set gain.")
@click.option("--sharpness", type=int, default=None, help="Set sharpness.")
@click.option("--backlight-compensation", type=int, default=None, help="Set backlight compensation.")
@click.option("--white-balance-automatic/--no-white-balance-automatic", default=None, help="Enable or disable automatic white balance.")
@click.option("--white-balance-temperature", type=int, default=None, help="Set white balance temperature.")
@click.option("--exposure-auto", type=int, default=None, help="Set exposure_auto control directly. Common UVC values include 1(manual), 3(auto).")
@click.option("--exposure-absolute", type=int, default=None, help="Set exposure_absolute control directly.")
@click.option("--capture-stats/--no-capture-stats", default=True, help="Capture a few frames and report brightness statistics.")
@click.option("--warmup-frames", type=int, default=12, show_default=True, help="Frames to discard before measuring brightness.")
@click.option("--sample-frames", type=int, default=24, show_default=True, help="Frames to sample for brightness stats.")
@click.option("--preview-output", type=click.Path(path_type=Path), default=None, help="Optional path to save the brightest captured frame.")
def main(
    camera_cfg: Path,
    camera_dev: str | None,
    list_devices: bool,
    list_controls: bool,
    show_all: bool,
    set_ctrl: tuple[str, ...],
    brightness: int | None,
    contrast: int | None,
    saturation: int | None,
    gain: int | None,
    sharpness: int | None,
    backlight_compensation: int | None,
    white_balance_automatic: bool | None,
    white_balance_temperature: int | None,
    exposure_auto: int | None,
    exposure_absolute: int | None,
    capture_stats: bool,
    warmup_frames: int,
    sample_frames: int,
    preview_output: Path | None,
) -> None:
    try:
        device, resolution = _resolve_device(camera_cfg=camera_cfg, camera_dev=camera_dev)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"device: {device}")
    if resolution is not None:
        click.echo(f"resolution: {resolution[0]}x{resolution[1]}")

    if list_devices:
        try:
            result = _run_v4l2ctl(["--list-devices"])
            click.echo(result.stdout.rstrip())
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        if not (list_controls or show_all or capture_stats or set_ctrl):
            return

    if list_controls:
        try:
            result = _run_v4l2ctl(["--device", device, "--list-ctrls-menus"])
            click.echo(result.stdout.rstrip())
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

    if show_all:
        try:
            result = _run_v4l2ctl(["--device", device, "--all"])
            click.echo(result.stdout.rstrip())
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

    ctrl_pairs = _build_ctrl_assignments(
        set_ctrl=set_ctrl,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        gain=gain,
        sharpness=sharpness,
        backlight_compensation=backlight_compensation,
        white_balance_automatic=white_balance_automatic,
        white_balance_temperature=white_balance_temperature,
        exposure_auto=exposure_auto,
        exposure_absolute=exposure_absolute,
    )
    if ctrl_pairs:
        ctrl_arg = ",".join(ctrl_pairs)
        click.echo(f"apply: v4l2-ctl --device {shlex.quote(device)} --set-ctrl {shlex.quote(ctrl_arg)}")
        try:
            result = _run_v4l2ctl(["--device", device, "--set-ctrl", ctrl_arg])
            if result.stdout.strip():
                click.echo(result.stdout.rstrip())
            if result.stderr.strip():
                click.echo(result.stderr.rstrip())
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

    if capture_stats:
        stats, preview = _capture_brightness_stats(
            camera_dev=device,
            resolution=resolution,
            warmup_frames=warmup_frames,
            sample_frames=sample_frames,
        )
        click.echo(
            "capture_stats "
            f"open_ok={stats['open_ok']} "
            f"ok_reads={stats['ok_reads']}/{stats['reads']} "
            f"brightness_mean={stats['brightness_mean']:.3f} "
            f"brightness_min={stats['brightness_min']:.3f} "
            f"brightness_max={stats['brightness_max']:.3f}"
        )
        if preview_output is not None and preview is not None:
            _save_preview(frame_bgr=preview, output_path=preview_output.expanduser().resolve())
            click.echo(f"preview_output: {preview_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
