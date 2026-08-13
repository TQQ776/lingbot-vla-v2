#!/usr/bin/env python3
"""Show the production VTLA marker contact gate on a live TacThru stream."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TACTHRU_REPO = PROJECT_ROOT.parent / "tacthru"
DEFAULT_VTLA_CONFIG = (
    PROJECT_ROOT
    / "configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml"
)
WINDOW_NAME = "VTLA TacThru Contact Gate"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


@dataclass(frozen=True)
class GateContext:
    settings: Any
    runtime: Any
    gate: Any
    region_ids: np.ndarray


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
            "Display the live TacThru stream and run the same marker contact gate "
            "used by the current VTLA configuration. No data or robot command is sent."
        )
    )
    parser.add_argument(
        "--tacthru-repo",
        type=Path,
        default=DEFAULT_TACTHRU_REPO,
        help="TacThru repository root",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=None,
        help="TacThru sensor YAML (default: TACTHRU_REPO/cfg/sensor/ml.yaml)",
    )
    parser.add_argument(
        "--vtla-config",
        type=Path,
        default=DEFAULT_VTLA_CONFIG,
        help="VTLA training YAML whose train.tactile gate is evaluated",
    )
    parser.add_argument("--cam-path", default=None, help="Optional TacThru camera override")
    parser.add_argument("--arrow-scale", type=_positive_float, default=6.0)
    parser.add_argument("--window-scale", type=_positive_float, default=1.35)
    parser.add_argument("--timeout", type=_positive_float, default=4.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=_non_negative_int,
        default=0,
        help="0 means unlimited",
    )
    return parser.parse_args()


def _resolve_project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (PROJECT_ROOT / path, config_path.parent / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_gate_context(config_path: Path) -> GateContext:
    from lingbotvla.tactile_contact import (
        MarkerContactGate,
        build_marker_region_mapping_numpy,
        runtime_gate_config_from_mapping,
    )

    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    tactile = copy.deepcopy(payload.get("train", {}).get("tactile"))
    if not isinstance(tactile, Mapping):
        raise ValueError(f"{config_path} does not contain train.tactile")
    tactile = dict(tactile)
    reference_path = _resolve_project_path(
        tactile.get("marker_reference_xy_path", ""), config_path
    )
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    gate_mapping = dict(tactile.get("marker_contact_gate", {}))
    if gate_mapping.get("threshold_stats_path"):
        gate_mapping["threshold_stats_path"] = str(
            _resolve_project_path(gate_mapping["threshold_stats_path"], config_path)
        )
    num_sensors = int(tactile.get("num_sensors", 0))
    num_markers = int(tactile.get("num_markers", 0))
    sensor_names = tuple(str(item) for item in tactile.get("sensor_names", ()))
    if not bool(tactile.get("enabled")) or not bool(tactile.get("use_markers")):
        raise ValueError("The selected VTLA config does not enable marker tactile input")
    if num_sensors != 1:
        raise ValueError(
            f"This viewer currently supports one TacThru sensor, got {num_sensors}"
        )
    if len(sensor_names) != num_sensors or num_markers <= 0:
        raise ValueError("Invalid sensor_names/num_markers in the VTLA tactile config")

    module_name = "_vtla_marker_contact_config_lightweight"
    marker_contact_path = (
        PROJECT_ROOT
        / "lingbotvla/models/vla/lingbot_vla/marker_contact.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, marker_contact_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load marker contact config from {marker_contact_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    gate_config = module.MarkerContactGateConfig.from_mapping(
        gate_mapping,
        num_sensors=num_sensors,
        num_markers=num_markers,
        sensor_names=sensor_names,
    )
    resolved_gate = gate_config.to_dict()
    runtime = runtime_gate_config_from_mapping(
        resolved_gate,
        num_sensors=num_sensors,
        num_markers=num_markers,
    )
    if runtime.mode == "none":
        raise ValueError("The selected VTLA config has marker contact gating disabled")

    with reference_path.open("r", encoding="utf-8") as handle:
        reference_payload = json.load(handle)
    reference_xy = np.asarray(
        reference_payload.get("reference_xy", reference_payload), dtype=np.float32
    )
    if reference_xy.ndim == 2:
        reference_xy = reference_xy[None]
    expected_reference_shape = (num_sensors, num_markers, 2)
    if reference_xy.shape != expected_reference_shape:
        raise ValueError(
            f"Marker reference must have shape {expected_reference_shape}, got {reference_xy.shape}"
        )
    tokenization = dict(tactile.get("marker_tokenization", {}))
    if tokenization.get("mode", "global") == "regional":
        region_ids, _ = build_marker_region_mapping_numpy(
            reference_xy,
            num_regions=int(tokenization["num_regions"]),
            region_layout=str(tokenization["region_layout"]),
        )
    else:
        region_ids = np.zeros((num_sensors, num_markers), dtype=np.int64)
    gate = MarkerContactGate(runtime, region_ids[0], sensor_index=0)
    settings = SimpleNamespace(
        enabled=True,
        use_markers=True,
        num_sensors=num_sensors,
        num_markers=num_markers,
        sensor_names=sensor_names,
        marker_contact_gate=gate_config,
        marker_tokenization=SimpleNamespace(**tokenization),
        marker_reference_xy=reference_xy,
    )
    return GateContext(
        settings=settings,
        runtime=runtime,
        gate=gate,
        region_ids=np.asarray(region_ids[0], dtype=np.int64),
    )


def normalized_marker_displacement(
    marker: np.ndarray,
    reference: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the current deployment formula and finite-coordinate validity rule."""
    marker = np.asarray(marker, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if marker.shape != reference.shape or marker.ndim != 2 or marker.shape[-1] != 2:
        raise ValueError(
            f"marker/reference must have equal [N,2] shapes, got {marker.shape} and {reference.shape}"
        )
    valid = np.isfinite(marker).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    scale_xy = np.asarray([image_width, image_height], dtype=np.float32)
    displacement = (marker - reference) / scale_xy * 2.0
    displacement = np.where(valid[:, None], displacement, 0.0).astype(
        np.float32, copy=False
    )
    return displacement, valid


def gate_label(state: int, *, warmup_remaining: int) -> tuple[str, tuple[int, int, int]]:
    from lingbotvla.tactile_contact import contact_state_is_visible_numpy

    if warmup_remaining > 0:
        return f"模板预热中 ({warmup_remaining})", (255, 205, 70)
    visible = bool(contact_state_is_visible_numpy(np.asarray(state)).item())
    if visible:
        return "检测到触觉", (75, 220, 105)
    return "无触觉", (225, 225, 225)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _state_name(state: int) -> str:
    from lingbotvla.tactile_contact import (
        CONTACT_HOLD,
        CONTACT_OFF,
        CONTACT_ON,
        CONTACT_UNKNOWN_HOLD,
        CONTACT_UNKNOWN_OFF,
        CONTACT_UNKNOWN_ON,
    )

    names = {
        CONTACT_OFF: "off",
        CONTACT_ON: "on",
        CONTACT_HOLD: "hold",
        CONTACT_UNKNOWN_OFF: "unknown_off",
        CONTACT_UNKNOWN_ON: "unknown_on",
        CONTACT_UNKNOWN_HOLD: "unknown_hold",
    }
    return names.get(int(state), f"unknown({state})")


def _draw_region_boxes(
    image: np.ndarray,
    reference: np.ndarray,
    region_ids: np.ndarray,
    gate_frame: Any | None,
    on_threshold: float,
) -> None:
    for region_index in range(int(region_ids.max()) + 1):
        points = reference[region_ids == region_index]
        if not len(points):
            continue
        low = np.floor(points.min(axis=0) - 16).astype(int)
        high = np.ceil(points.max(axis=0) + 16).astype(int)
        active = False
        if gate_frame is not None and bool(gate_frame.region_valid_mask[region_index]):
            if getattr(gate_frame, "on_active_marker_count", None) is not None:
                active = int(gate_frame.active_counts[region_index]) > 0
            else:
                active = float(gate_frame.region_scores[region_index]) > on_threshold
        color = (50, 210, 80) if active else (130, 130, 130)
        cv2.rectangle(image, tuple(low), tuple(high), color, 1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"R{region_index}",
            (int(low[0]) + 4, int(low[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def render_gate_ui(
    *,
    rgb: np.ndarray,
    marker: np.ndarray,
    reference: np.ndarray,
    tracker_valid: np.ndarray,
    tracker_fallback: np.ndarray,
    vtla_valid: np.ndarray,
    region_ids: np.ndarray,
    gate_frame: Any | None,
    runtime: Any,
    warmup_remaining: int,
    candidate_count: int,
    fps: float,
    arrow_scale: float,
) -> np.ndarray:
    image = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    marker = np.asarray(marker, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    tracker_valid = np.asarray(tracker_valid, dtype=bool)
    tracker_fallback = np.asarray(tracker_fallback, dtype=bool)
    vtla_valid = np.asarray(vtla_valid, dtype=bool)
    if marker.shape != reference.shape or tracker_valid.shape != marker.shape[:-1]:
        raise ValueError("Marker UI arrays have inconsistent shapes")

    _draw_region_boxes(
        image,
        reference,
        region_ids,
        gate_frame,
        float(runtime.on_thresholds[0]),
    )
    for index, (ref, current) in enumerate(zip(reference, marker)):
        ref_point = tuple(np.rint(ref).astype(int))
        if np.isfinite(current).all():
            arrow_end = ref + arrow_scale * (current - ref)
            color = (
                (0, 205, 255)
                if tracker_valid[index] and tracker_fallback[index]
                else (30, 30, 235)
                if tracker_valid[index]
                else (145, 145, 145)
            )
            cv2.arrowedLine(
                image,
                ref_point,
                tuple(np.rint(arrow_end).astype(int)),
                color,
                1 if not tracker_valid[index] else 2,
                cv2.LINE_AA,
                tipLength=0.22,
            )
            cv2.circle(
                image,
                tuple(np.rint(current).astype(int)),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )
        cv2.circle(image, ref_point, 6, (30, 220, 70), 1, cv2.LINE_AA)

    header_height = 172
    canvas = np.full((image.shape[0] + header_height, image.shape[1], 3), 20, dtype=np.uint8)
    canvas[header_height:] = image
    state = 0 if gate_frame is None else int(gate_frame.state)
    label, label_color = gate_label(state, warmup_remaining=warmup_remaining)
    scores = np.zeros(int(region_ids.max()) + 1, dtype=np.float32)
    active_counts = np.zeros_like(scores, dtype=np.int64)
    soft = np.zeros_like(scores)
    tracking_unknown = False
    on_active_marker_count = 0
    off_active_marker_count = 0
    if gate_frame is not None:
        scores = np.asarray(gate_frame.region_scores)
        active_counts = np.asarray(gate_frame.active_counts)
        soft = np.asarray(gate_frame.regional_soft_gates)
        tracking_unknown = bool(gate_frame.tracking_unknown)
        on_active_marker_count = int(
            getattr(gate_frame, "on_active_marker_count", int(active_counts.sum()))
        )
        off_active_marker_count = int(
            getattr(gate_frame, "off_active_marker_count", on_active_marker_count)
        )

    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_canvas)
    draw = ImageDraw.Draw(pil_image)
    draw.text((12, 4), label, font=_font(34, bold=True), fill=label_color)
    draw.text(
        (365, 12),
        f"状态: {_state_name(state)}",
        font=_font(18, bold=True),
        fill=(255, 205, 70) if tracking_unknown else (235, 235, 235),
    )
    draw.text(
        (12, 54),
        (
            f"VTLA有效点 {int(vtla_valid.sum())}/{len(vtla_valid)}   "
            f"追踪实测 {int((tracker_valid & ~tracker_fallback).sum())}/{len(tracker_valid)}   "
            f"fallback {int(tracker_fallback.sum())}   候选 {candidate_count}   FPS {fps:.1f}"
        ),
        font=_font(15),
        fill=(230, 230, 230),
    )
    score_text = "   ".join(
        f"R{i}={score:.4f}({int(active_counts[i])})" for i, score in enumerate(scores)
    )
    draw.text(
        (12, 82),
        (
            f"全局大形变点 ON={on_active_marker_count}/{runtime.min_active_markers}  "
            f"OFF保持={off_active_marker_count}/{runtime.min_active_markers}   {score_text}"
        ),
        font=_font(13),
        fill=(220, 220, 220),
    )
    soft_text = "   ".join(f"R{i}={value:.2f}" for i, value in enumerate(soft))
    draw.text(
        (12, 108),
        (
            f"单点幅值阈值 ON={float(runtime.on_thresholds[0]):.5f}  "
            f"OFF={float(runtime.off_thresholds[0]):.5f}   软门控: {soft_text}"
        ),
        font=_font(14),
        fill=(210, 210, 210),
    )
    draw.text(
        (12, 137),
        "红=实测  黄=短时补点  灰=预测   q/Esc退出   r重置追踪和门控",
        font=_font(14),
        fill=(185, 185, 185),
    )
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _validate_camera(sensor_config: Path, cam_override: str | None) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(sensor_config)
    camera = str(cam_override or cfg.cam_path)
    if camera.startswith("/dev/") and not Path(camera).exists():
        available = sorted(str(path) for path in Path("/dev/v4l/by-id").glob("*-video-index0"))
        listing = "\n  ".join(available) if available else "(none)"
        raise FileNotFoundError(
            f"Configured TacThru camera is not connected: {camera}\n"
            f"Available video-index0 devices:\n  {listing}"
        )


def main() -> int:
    args = parse_args()
    tacthru_repo = args.tacthru_repo.expanduser().resolve()
    sensor_config = (
        args.sensor_config.expanduser().resolve()
        if args.sensor_config is not None
        else tacthru_repo / "cfg/sensor/ml.yaml"
    )
    vtla_config = args.vtla_config.expanduser().resolve()
    if not tacthru_repo.is_dir():
        raise NotADirectoryError(tacthru_repo)
    if not sensor_config.is_file():
        raise FileNotFoundError(sensor_config)
    if not vtla_config.is_file():
        raise FileNotFoundError(vtla_config)
    _validate_camera(sensor_config, args.cam_path)

    tacthru_string = str(tacthru_repo)
    if tacthru_string not in sys.path:
        sys.path.insert(0, tacthru_string)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from multiprocessing.managers import SharedMemoryManager

    from omegaconf import OmegaConf
    from real_world.sensor_utils import TacThruClient

    context = load_gate_context(vtla_config)
    sensor_cfg = OmegaConf.load(sensor_config)
    if args.cam_path is not None:
        sensor_cfg.cam_path = args.cam_path
    tracking_path = Path(str(sensor_cfg.tracking.tracking_pts_path)).expanduser()
    if not tracking_path.is_absolute():
        sensor_cfg.tracking.tracking_pts_path = str((tacthru_repo / tracking_path).resolve())

    print("[vtla-gate] view-only mode: no recording, robot, or gripper command")
    print(f"[vtla-gate] VTLA config: {vtla_config}")
    print(f"[vtla-gate] sensor config: {sensor_config}")
    print(
        f"[vtla-gate] mode={context.runtime.mode} "
        f"on={context.runtime.on_thresholds[0]:.6f} "
        f"off={context.runtime.off_thresholds[0]:.6f}"
    )

    manager = SharedMemoryManager()
    sensor = None
    window_created = False
    last_timestamp: float | None = None
    fps = 0.0
    processed_frames = 0
    latest_gate = None
    previous_warmup: int | None = None
    try:
        manager.start()
        sensor = TacThruClient(
            manager,
            None,
            sensor_cfg,
            "cuda:0",
            False,
            f"{sensor_cfg.name}-vtla-gate-view",
            do_tracking=True,
        )
        sensor.start(wait=True)
        deadline = time.monotonic() + args.timeout
        while int(sensor.ring_buffer.count) < 1:
            if not sensor.is_alive():
                raise RuntimeError("TacThru process exited before producing a frame")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the first TacThru frame")
            time.sleep(0.01)

        while args.max_frames == 0 or processed_frames < args.max_frames:
            if not sensor.is_alive():
                raise RuntimeError("TacThru process stopped unexpectedly")
            count = min(30, int(sensor.ring_buffer.count))
            data = sensor.get(k=count)
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
            fresh = (
                np.ones(len(timestamps), dtype=bool)
                if last_timestamp is None
                else timestamps > last_timestamp + 1e-9
            )
            indices = np.flatnonzero(fresh)
            if not len(indices):
                time.sleep(0.002)
                continue

            latest_index = int(indices[-1])
            latest_displacement = None
            latest_vtla_valid = None
            marker_valid_data = data.get("marker_valid")
            for index in indices:
                timestamp = float(timestamps[index])
                if last_timestamp is not None and timestamp > last_timestamp:
                    instant_fps = 1.0 / (timestamp - last_timestamp)
                    fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
                last_timestamp = timestamp
                rgb = np.asarray(data["rgb"][index], dtype=np.uint8)
                marker = np.asarray(data["marker"][index], dtype=np.float32)
                reference = np.asarray(data["marker_ref"][index], dtype=np.float32)
                latest_displacement, latest_vtla_valid = normalized_marker_displacement(
                    marker,
                    reference,
                    image_width=rgb.shape[1],
                    image_height=rgb.shape[0],
                )
                if marker_valid_data is not None:
                    current_tracker_valid = np.asarray(
                        marker_valid_data[index], dtype=bool
                    )
                    if current_tracker_valid.shape != latest_vtla_valid.shape:
                        raise ValueError(
                            "TacThru marker_valid shape does not match marker coordinates"
                        )
                    latest_vtla_valid &= current_tracker_valid
                    latest_displacement = np.where(
                        latest_vtla_valid[:, None], latest_displacement, 0.0
                    ).astype(np.float32, copy=False)
                warmup = int(np.asarray(data.get("tracking_warmup_remaining", 0))[index])
                if warmup > 0:
                    context.gate.reset()
                    latest_gate = None
                else:
                    if previous_warmup is not None and previous_warmup > 0:
                        context.gate.reset()
                    latest_gate = context.gate.step(latest_displacement, latest_vtla_valid)
                previous_warmup = warmup
                processed_frames += 1
                if args.max_frames and processed_frames >= args.max_frames:
                    latest_index = int(index)
                    break

            rgb = np.asarray(data["rgb"][latest_index], dtype=np.uint8)
            marker = np.asarray(data["marker"][latest_index], dtype=np.float32)
            reference = np.asarray(data["marker_ref"][latest_index], dtype=np.float32)
            if latest_displacement is None or latest_vtla_valid is None:
                raise RuntimeError("No fresh marker frame was processed")
            tracker_valid = (
                np.asarray(marker_valid_data[latest_index], dtype=bool)
                if marker_valid_data is not None
                else latest_vtla_valid
            )
            fallback_data = data.get("marker_fallback")
            tracker_fallback = (
                np.asarray(fallback_data[latest_index], dtype=bool)
                if fallback_data is not None
                else np.zeros_like(tracker_valid)
            )
            warmup = int(
                np.asarray(data.get("tracking_warmup_remaining", np.zeros(len(timestamps))))[
                    latest_index
                ]
            )
            candidate_data = np.asarray(data.get("n_all_kpts", np.zeros(len(timestamps))))
            candidate_count = int(candidate_data[latest_index])
            canvas = render_gate_ui(
                rgb=rgb,
                marker=marker,
                reference=reference,
                tracker_valid=tracker_valid,
                tracker_fallback=tracker_fallback,
                vtla_valid=latest_vtla_valid,
                region_ids=context.region_ids,
                gate_frame=latest_gate,
                runtime=context.runtime,
                warmup_remaining=warmup,
                candidate_count=candidate_count,
                fps=fps,
                arrow_scale=args.arrow_scale,
            )

            if args.headless:
                if processed_frames == len(indices) or processed_frames % 30 < len(indices):
                    state = 0 if latest_gate is None else int(latest_gate.state)
                    print(
                        f"[vtla-gate] frame={processed_frames} fps={fps:.1f} "
                        f"state={_state_name(state)} warmup={warmup} "
                        f"vtla_valid={int(latest_vtla_valid.sum())}/{len(latest_vtla_valid)}"
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
            if key == ord("r"):
                context.gate.reset()
                latest_gate = None
                previous_warmup = None
                sensor.reset()
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

    print(f"[vtla-gate] stopped after {processed_frames} fresh frames")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"[vtla-gate] {error}", file=sys.stderr)
        raise SystemExit(2) from None
