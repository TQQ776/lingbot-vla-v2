from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import select
import ssl
import sys
import termios
import threading
import time
import tty
import uuid
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml

from .protocol import (
    CAMERA_KEY,
    CONTROL_FREQUENCY_HZ,
    POSE_FRAME,
    POSE_SEMANTICS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_V1,
    PROTOCOL_VERSION_V2,
    SUPPORTED_PROTOCOL_VERSIONS,
    ROBOT_CONFIG,
    TACTILE_ROBOT_CONFIG,
    TACTILE_ENABLED,
    TACTILE_MARKER_COUNT,
    TACTILE_MARKER_DIM,
    ActionResponse,
    Observation,
    action_response_from_json,
    observation_to_json,
    validate_action_spec,
)
from .realman_runtime import RealmanConfig, RealmanEpisodeRuntime, SafetyViolation
from .transforms import validate_state8


DEFAULT_IMAGE_SIZE = 224
CAMERA_FRESHNESS_FRACTION = 0.80
RETRYABLE_HTTP_STATUSES = frozenset({408, 503})


class RetryableInferenceError(RuntimeError):
    """A prediction attempt failed transiently and may be retried with fresh sensors.

    The HTTP layer must never replay ``POST /predict`` itself: after an
    ambiguous timeout the server may already have processed the request.  The
    real-robot loop catches this exception only in non-streaming mode, drops
    the old observation/action, and captures a new image and robot state.
    """

    def __init__(self, message: str, *, timing: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.timing = dict(timing or {})


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    capture_timestamp: float
    receive_timestamp: float


@dataclass(frozen=True)
class TactileHistory:
    rgb: np.ndarray | None
    rgb_timestamps: np.ndarray | None
    rgb_history_mask: np.ndarray | None
    marker_flow: np.ndarray | None
    marker_valid_mask: np.ndarray | None
    marker_timestamps: np.ndarray | None
    marker_history_mask: np.ndarray | None
    debug: dict[str, Any]


def _estimate_camera_capture_timestamp(
    cap,
    *,
    receive_timestamp: float,
    read_started_timestamp: float,
    frame_period_s: float,
) -> float:
    """Map a live UVC/V4L2 frame timestamp onto the wall clock.

    For live V4L2 devices OpenCV exposes ``CAP_PROP_POS_MSEC`` on the same
    monotonic clock used by the kernel.  TacThru's native ``UvcCamera`` uses
    this mapping as well.  If a backend does not provide a usable timestamp,
    estimate capture at most one configured frame period before receipt rather
    than incorrectly treating the entire blocking ``read()`` as sensor age.
    """

    try:
        media_timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        monotonic_now = time.monotonic()
        wall_now = time.time()
        candidate = media_timestamp_s - monotonic_now + wall_now
        age_s = float(receive_timestamp) - candidate
        if np.isfinite(media_timestamp_s) and media_timestamp_s > 0.0:
            if not np.isfinite(candidate):
                raise RuntimeError("Wrist camera driver timestamp mapped to a non-finite wall time")
            if age_s < -0.02:
                raise RuntimeError(
                    "Wrist camera driver timestamp is unexpectedly in the future: "
                    f"offset={-age_s:.4f}s"
                )
            # Preserve valid old timestamps so downstream freshness checks can
            # reject stale frames. Falling back merely because a frame is more
            # than one second old would incorrectly make that frame look fresh.
            return min(candidate, float(receive_timestamp))
    except (AttributeError, TypeError, ValueError, cv2.error):
        pass

    read_duration_s = max(0.0, float(receive_timestamp) - float(read_started_timestamp))
    fallback_age_s = min(read_duration_s, max(0.0, float(frame_period_s)))
    return float(receive_timestamp) - fallback_age_s


class WristCamera:
    """Continuously capture the latest wrist frame, matching TacThru's camera pipeline."""

    def __init__(
        self,
        device: str,
        *,
        width: int,
        height: int,
        fps: float,
        output_size: int = DEFAULT_IMAGE_SIZE,
        buffer_size: int = 1,
        max_frame_age_s: float = 0.10,
        initial_frame_timeout_s: float = 2.0,
        frame_timeout_s: float = 0.25,
    ) -> None:
        self.output_size = int(output_size)
        self.expected_width = int(width)
        self.expected_height = int(height)
        fps = float(fps)
        self.frame_period_s = 1.0 / fps if np.isfinite(fps) and fps > 0.0 else 0.0
        self.max_frame_age_s = float(max_frame_age_s)
        self.initial_frame_timeout_s = float(initial_frame_timeout_s)
        self.frame_timeout_s = float(frame_timeout_s)
        for name, value in (
            ("max_frame_age_s", self.max_frame_age_s),
            ("initial_frame_timeout_s", self.initial_frame_timeout_s),
            ("frame_timeout_s", self.frame_timeout_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value, got {value}")

        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open wrist camera: {device}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._release_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._capture_released = False
        self._worker_error: BaseException | None = None
        self._latest_frame: CameraFrame | None = None
        self._latest_sequence = 0
        self._last_delivered_sequence = 0

    def capture(self) -> CameraFrame:
        self._start_worker()
        with self._condition:
            timeout_s = (
                self.initial_frame_timeout_s
                if self._last_delivered_sequence == 0
                else self.frame_timeout_s
            )
            deadline = time.monotonic() + timeout_s
            latest_age_s: float | None = None
            while True:
                if self._worker_error is not None:
                    raise RuntimeError("Wrist camera capture worker failed") from self._worker_error
                if self._closed:
                    raise RuntimeError("Wrist camera is closed")

                frame = self._latest_frame
                if frame is not None and self._latest_sequence > self._last_delivered_sequence:
                    latest_age_s = time.time() - frame.capture_timestamp
                    if -0.02 <= latest_age_s <= self.max_frame_age_s:
                        self._last_delivered_sequence = self._latest_sequence
                        return CameraFrame(
                            rgb=frame.rgb.copy(),
                            capture_timestamp=frame.capture_timestamp,
                            receive_timestamp=frame.receive_timestamp,
                        )

                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    age_detail = "unavailable" if latest_age_s is None else f"{latest_age_s:.4f}s"
                    raise RuntimeError(
                        "No fresh wrist camera frame arrived within "
                        f"{timeout_s:.3f}s (latest_age={age_detail}, "
                        f"max_age={self.max_frame_age_s:.4f}s)"
                    )
                self._condition.wait(timeout=remaining_s)

    def _start_worker(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Wrist camera is closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="lingbot-v2-wrist-camera",
                daemon=True,
            )
            self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                read_started_timestamp = time.time()
                ok = self.cap.grab()
                if not ok:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError("Failed to grab wrist camera frame")
                ok, frame_bgr = self.cap.retrieve()
                receive_timestamp = time.time()
                if not ok or frame_bgr is None:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError("Failed to retrieve wrist camera frame")
                capture_timestamp = _estimate_camera_capture_timestamp(
                    self.cap,
                    receive_timestamp=receive_timestamp,
                    read_started_timestamp=read_started_timestamp,
                    frame_period_s=self.frame_period_s,
                )
                if frame_bgr.shape[:2] != (self.expected_height, self.expected_width):
                    raise RuntimeError(
                        "Wrist camera negotiated an unexpected resolution: "
                        f"expected={(self.expected_height, self.expected_width)}, actual={frame_bgr.shape[:2]}"
                    )
                frame = CameraFrame(
                    rgb=center_crop_resize_rgb(frame_bgr, output_size=self.output_size),
                    capture_timestamp=capture_timestamp,
                    receive_timestamp=receive_timestamp,
                )
                with self._condition:
                    if self._stop_event.is_set():
                        break
                    self._latest_frame = frame
                    self._latest_sequence += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                if not self._stop_event.is_set():
                    self._worker_error = exc
                self._condition.notify_all()
        finally:
            self._release_capture_once()
            with self._condition:
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                self._stop_event.set()
            thread = self._thread
            self._condition.notify_all()
        if thread is None:
            self._release_capture_once()
            return
        thread.join(timeout=max(0.25, self.frame_timeout_s))
        if thread.is_alive():
            # Releasing the device is the last-resort way to unblock a stalled
            # V4L2 grab during shutdown. The worker remains daemonized so it
            # cannot keep the client process alive if a driver is wedged.
            self._release_capture_once()
            thread.join(timeout=1.0)

    def _release_capture_once(self) -> None:
        with self._release_lock:
            if self._capture_released:
                return
            self._capture_released = True
            self.cap.release()


class TacThruTactileSource:
    """Read synchronized RGB/marker history from TacThru's existing sensor process.

    Imports are deliberately delayed until this class is instantiated so the
    unchanged protocol-v1 client never imports CuPy, opens a tactile camera, or
    requires TacThru's tracking dependencies.
    """

    def __init__(
        self,
        *,
        tacthru_repo: Path,
        sensor_cfg_path: Path,
        history_steps: int,
        history_stride: int,
        include_rgb: bool,
        include_marker: bool,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.tacthru_repo = tacthru_repo.expanduser().resolve()
        self.sensor_cfg_path = sensor_cfg_path.expanduser().resolve()
        self.history_steps = int(history_steps)
        self.history_stride = int(history_stride)
        self.include_rgb = bool(include_rgb)
        self.include_marker = bool(include_marker)
        self.startup_timeout_s = float(startup_timeout_s)
        if self.history_steps <= 0 or self.history_stride <= 0:
            raise ValueError("Tactile history steps and stride must be positive")
        if not (self.include_rgb or self.include_marker):
            raise ValueError("TacThruTactileSource requires at least one tactile modality")
        if not self.sensor_cfg_path.is_file():
            raise FileNotFoundError(self.sensor_cfg_path)

        repo_text = str(self.tacthru_repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from omegaconf import OmegaConf
        from real_world.sensor_utils import TacThruClient

        with self.sensor_cfg_path.open("r", encoding="utf-8") as file:
            raw_cfg = yaml.safe_load(file) or {}
        tracking = raw_cfg.get("tracking") or {}
        tracking_path = tracking.get("tracking_pts_path")
        if tracking_path:
            value = Path(str(tracking_path)).expanduser()
            tracking["tracking_pts_path"] = str(
                value.resolve() if value.is_absolute() else (self.tacthru_repo / value).resolve()
            )
            raw_cfg["tracking"] = tracking
        sensor_cfg = OmegaConf.create(raw_cfg)

        self._manager = SharedMemoryManager()
        self._manager.start()
        try:
            self._sensor = TacThruClient(
                self._manager,
                None,
                sensor_cfg,
                "cuda:0",
                False,
                str(raw_cfg.get("name", "tacthru_l")),
                do_tracking=self.include_marker,
            )
            self._sensor.start(wait=False)
            if not self._sensor.ready_event.wait(self.startup_timeout_s):
                raise RuntimeError(
                    f"Timed out waiting {self.startup_timeout_s:.1f}s for TacThru tactile sensor"
                )
            if not self._sensor.is_alive():
                raise RuntimeError("TacThru tactile sensor process exited during startup")
        except BaseException:
            self.close()
            raise

    def capture(self) -> TactileHistory:
        sensor = getattr(self, "_sensor", None)
        if sensor is None or not sensor.is_alive():
            raise RuntimeError("TacThru tactile sensor is not running")
        needed = (self.history_steps - 1) * self.history_stride + 1
        deadline = time.monotonic() + max(0.5, min(self.startup_timeout_s, 3.0))
        while sensor.ring_buffer.count <= 0 and time.monotonic() < deadline:
            if not sensor.is_alive():
                raise RuntimeError("TacThru tactile sensor process exited before producing a frame")
            time.sleep(0.01)
        available = min(int(sensor.ring_buffer.count), int(sensor.ring_buffer.get_max_k), needed)
        if available <= 0:
            raise RuntimeError("TacThru tactile sensor has not produced any frames")
        data = sensor.get(k=available)
        return build_tactile_history(
            data,
            history_steps=self.history_steps,
            history_stride=self.history_stride,
            include_rgb=self.include_rgb,
            include_marker=self.include_marker,
        )

    def close(self) -> None:
        sensor = getattr(self, "_sensor", None)
        self._sensor = None
        if sensor is not None:
            try:
                sensor.stop(wait=False)
                sensor.join(timeout=3.0)
                if sensor.is_alive():
                    sensor.terminate()
                    sensor.join(timeout=1.0)
            except BaseException:
                if sensor.is_alive():
                    sensor.terminate()
                    sensor.join(timeout=1.0)
        manager = getattr(self, "_manager", None)
        self._manager = None
        if manager is not None:
            try:
                manager.shutdown()
            except BaseException:
                pass


def build_tactile_history(
    sensor_data: dict[str, Any],
    *,
    history_steps: int,
    history_stride: int,
    include_rgb: bool,
    include_marker: bool,
) -> TactileHistory:
    """Build chronological fixed-K tactile tensors with left-padding masks."""

    steps = int(history_steps)
    stride = int(history_stride)
    if steps <= 0 or stride <= 0:
        raise ValueError("history_steps and history_stride must be positive")
    timestamps_all = np.asarray(sensor_data.get("timestamp"), dtype=np.float64)
    if timestamps_all.ndim != 1 or len(timestamps_all) == 0:
        raise ValueError("TacThru sensor history must contain a non-empty timestamp vector")
    if not np.isfinite(timestamps_all).all() or np.any(timestamps_all <= 0.0):
        raise ValueError("TacThru timestamps must be positive finite Unix timestamps")
    newest = len(timestamps_all) - 1
    raw_indices = newest - np.arange(steps - 1, -1, -1, dtype=np.int64) * stride
    history_mask = raw_indices >= 0
    indices = np.maximum(raw_indices, 0)
    timestamps = np.ascontiguousarray(timestamps_all[indices])

    rgb = rgb_timestamps = rgb_mask = None
    if include_rgb:
        raw_rgb = np.asarray(sensor_data.get("rgb"))
        if raw_rgb.dtype != np.uint8 or raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3:
            raise ValueError(f"TacThru rgb must be uint8 [T,H,W,3], got {raw_rgb.dtype} {raw_rgb.shape}")
        if len(raw_rgb) != len(timestamps_all):
            raise ValueError("TacThru RGB/timestamp history lengths differ")
        rgb = np.stack(
            [
                cv2.resize(raw_rgb[index], (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                for index in indices
            ],
            axis=0,
        )
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        rgb_timestamps = timestamps.copy()
        rgb_mask = np.ascontiguousarray(history_mask, dtype=np.bool_)

    marker_flow = marker_valid = marker_timestamps = marker_mask = None
    marker_debug: dict[str, Any] = {}
    if include_marker:
        marker = np.asarray(sensor_data.get("marker"), dtype=np.float32)
        marker_ref = np.asarray(sensor_data.get("marker_ref"), dtype=np.float32)
        expected_shape = (len(timestamps_all), TACTILE_MARKER_COUNT, TACTILE_MARKER_DIM)
        if marker.shape != expected_shape or marker_ref.shape != expected_shape:
            raise ValueError(
                "TacThru marker/marker_ref must both have shape "
                f"{expected_shape}, got {marker.shape}/{marker_ref.shape}"
            )
        raw_rgb = np.asarray(sensor_data.get("rgb"))
        if raw_rgb.ndim != 4 or len(raw_rgb) != len(timestamps_all):
            raise ValueError("Marker normalization requires aligned TacThru RGB frames")
        height, width = raw_rgb.shape[1:3]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid TacThru frame dimensions: {width}x{height}")
        selected_marker = marker[indices]
        selected_ref = marker_ref[indices]
        marker_valid = np.isfinite(selected_marker).all(axis=-1) & np.isfinite(selected_ref).all(axis=-1)
        marker_valid &= history_mask[:, None]
        delta = selected_marker - selected_ref
        delta[~np.isfinite(delta)] = 0.0
        # Normalize in the tracker coordinate system before the independent
        # 224x224 network resize. ML48 tracking uses the actual 640x480 frame;
        # the legacy TacThru /400 transform is intentionally not reproduced.
        marker_flow = delta / np.asarray([width, height], dtype=np.float32) * 2.0
        marker_flow[~marker_valid] = 0.0
        marker_flow = np.ascontiguousarray(marker_flow, dtype=np.float32)
        marker_valid = np.ascontiguousarray(marker_valid, dtype=np.bool_)
        marker_timestamps = timestamps.copy()
        marker_mask = np.ascontiguousarray(history_mask, dtype=np.bool_)
        latest_valid = marker_valid[-1]
        magnitudes = np.linalg.norm(marker_flow[-1], axis=-1)
        marker_debug = {
            "marker_normalization": "image_size_xy",
            "marker_normalization_size_xy": [int(width), int(height)],
            "marker_raw_frame_width": int(width),
            "marker_raw_frame_height": int(height),
            "latest_valid_marker_count": int(latest_valid.sum()),
            "latest_marker_flow_max_norm": float(magnitudes[latest_valid].max()) if latest_valid.any() else None,
            "latest_marker_flow_mean_norm": float(magnitudes[latest_valid].mean()) if latest_valid.any() else None,
        }
        detected_counts = sensor_data.get("n_all_kpts")
        if detected_counts is not None:
            detected_counts = np.asarray(detected_counts, dtype=np.int64)
            if detected_counts.shape == timestamps_all.shape:
                selected_counts = np.maximum(detected_counts[indices], 0)
                marker_debug["detected_marker_count_history"] = selected_counts.astype(int).tolist()
                marker_debug["latest_detected_marker_count"] = int(selected_counts[-1])

    return TactileHistory(
        rgb=rgb,
        rgb_timestamps=rgb_timestamps,
        rgb_history_mask=rgb_mask,
        marker_flow=marker_flow,
        marker_valid_mask=marker_valid,
        marker_timestamps=marker_timestamps,
        marker_history_mask=marker_mask,
        debug={
            "history_steps": steps,
            "history_stride": stride,
            "available_sensor_frames": int(len(timestamps_all)),
            "history_mask": history_mask.astype(bool).tolist(),
            "latest_timestamp": float(timestamps[-1]),
            **marker_debug,
        },
    )


def validate_local_tactile_guard(
    history: TactileHistory,
    *,
    wrist_timestamp: float,
    robot_timestamp: float,
    now: float,
    marker_required: bool,
    max_tactile_age_s: float,
    max_tactile_skew_s: float,
    min_valid_markers: int,
    max_marker_flow_norm: float | None = None,
    max_marker_velocity_norm_s: float | None = None,
) -> dict[str, Any]:
    """Fail closed on stale/lost/extreme tactile input without relaxing arm limits."""

    timestamp_vectors = [
        value
        for value in (history.rgb_timestamps, history.marker_timestamps)
        if value is not None
    ]
    if not timestamp_vectors:
        raise SafetyViolation("local tactile guard received no tactile timestamps")
    latest_timestamp = min(float(value[-1]) for value in timestamp_vectors)
    age_s = float(now) - latest_timestamp
    if not np.isfinite(age_s) or age_s < -0.02:
        raise SafetyViolation(f"tactile timestamp is invalid or in the future: age={age_s:.4f}s")
    if age_s > float(max_tactile_age_s):
        raise SafetyViolation(
            f"tactile_stale: age {age_s:.4f}s exceeds limit {float(max_tactile_age_s):.4f}s"
        )
    skew_to_wrist_s = abs(latest_timestamp - float(wrist_timestamp))
    skew_to_robot_s = abs(latest_timestamp - float(robot_timestamp))
    maximum_skew_s = max(skew_to_wrist_s, skew_to_robot_s)
    if maximum_skew_s > float(max_tactile_skew_s):
        raise SafetyViolation(
            f"tactile_stale: tactile/wrist/state skew {maximum_skew_s:.4f}s exceeds "
            f"limit {float(max_tactile_skew_s):.4f}s"
        )

    debug: dict[str, Any] = {
        "enabled": True,
        "latest_timestamp": latest_timestamp,
        "age_s": age_s,
        "skew_to_wrist_s": skew_to_wrist_s,
        "skew_to_robot_s": skew_to_robot_s,
        "marker_required": bool(marker_required),
    }
    if marker_required:
        if history.marker_flow is None or history.marker_valid_mask is None:
            raise SafetyViolation("marker_tracking_lost: marker payload is missing")
        valid = np.asarray(history.marker_valid_mask[-1], dtype=np.bool_)
        valid_count = int(valid.sum())
        detected_count = history.debug.get("latest_detected_marker_count")
        if isinstance(detected_count, int):
            valid_count = min(valid_count, detected_count)
            debug["detected_marker_count"] = detected_count
        debug["valid_marker_count"] = valid_count
        if valid_count < int(min_valid_markers):
            raise SafetyViolation(
                f"marker_tracking_lost: valid markers {valid_count} below minimum {int(min_valid_markers)}"
            )
        magnitudes = np.linalg.norm(np.asarray(history.marker_flow[-1], dtype=np.float64), axis=-1)
        latest_max = float(magnitudes[valid].max())
        debug["marker_flow_max_norm"] = latest_max
        if max_marker_flow_norm is not None and latest_max > float(max_marker_flow_norm):
            raise SafetyViolation(
                f"contact_magnitude_high: marker flow {latest_max:.6f} exceeds "
                f"limit {float(max_marker_flow_norm):.6f}"
            )
        if max_marker_velocity_norm_s is not None and len(history.marker_flow) >= 2:
            previous_valid = np.asarray(history.marker_valid_mask[-2], dtype=np.bool_)
            common = valid & previous_valid
            dt = float(history.marker_timestamps[-1] - history.marker_timestamps[-2])
            if common.any() and dt > 0.0:
                velocity = (
                    np.asarray(history.marker_flow[-1], dtype=np.float64)
                    - np.asarray(history.marker_flow[-2], dtype=np.float64)
                ) / dt
                velocity_max = float(np.linalg.norm(velocity, axis=-1)[common].max())
                debug["marker_velocity_max_norm_s"] = velocity_max
                if velocity_max > float(max_marker_velocity_norm_s):
                    raise SafetyViolation(
                        f"slip_detected: marker velocity {velocity_max:.6f}/s exceeds "
                        f"limit {float(max_marker_velocity_norm_s):.6f}/s"
                    )
    return debug


class LingBotV2HttpClient:
    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float,
        jpeg_quality: int,
        api_key: str | None = None,
        keep_alive: bool = False,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.api_key = api_key
        self.keep_alive = bool(keep_alive)
        parsed = urlparse(self.server_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError(f"Unsupported LingBot V2 server URL: {server_url!r}")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("LingBot V2 server URL must not contain credentials, query, or fragment")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path.rstrip("/")
        self._ssl_context = ssl_context
        self._connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
        self._connection_lock = threading.RLock()
        self._closed = False
        self.last_request_timing: dict[str, Any] = {}

    def health(self) -> dict:
        total_started = time.perf_counter()
        data, transport_timing = self._request_bytes(
            "GET",
            "/health",
            body=None,
            headers=self._headers(),
            retry_safe=True,
        )
        parse_started = time.perf_counter()
        payload = json.loads(data.decode("utf-8"))
        timing = {
            "encode_s": 0.0,
            **transport_timing,
            "response_parse_s": time.perf_counter() - parse_started,
            "total_s": time.perf_counter() - total_started,
        }
        self.last_request_timing = timing
        return payload

    def predict(self, observation: Observation, *, expected_steps: int) -> ActionResponse:
        response, _ = self.predict_timed(observation, expected_steps=expected_steps)
        return response

    def predict_timed(
        self,
        observation: Observation,
        *,
        expected_steps: int,
    ) -> tuple[ActionResponse, dict[str, Any]]:
        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        body = observation_to_json(observation, jpeg_quality=self.jpeg_quality)
        encode_s = time.perf_counter() - encode_started
        try:
            data, transport_timing = self._request_bytes(
                "POST",
                "/predict",
                body=body,
                headers={**self._headers(), "Content-Type": "application/json"},
                retry_safe=False,
            )
        except RetryableInferenceError as exc:
            timing = {
                "encode_s": encode_s,
                **exc.timing,
                "total_s": time.perf_counter() - total_started,
            }
            exc.timing = timing
            self.last_request_timing = timing
            raise
        parse_started = time.perf_counter()
        response = action_response_from_json(
            data,
            expected_request_id=observation.request_id,
            expected_session_id=observation.session_id,
            expected_steps=expected_steps,
        )
        if response.protocol_version != observation.protocol_version:
            raise RuntimeError(
                "Inference response protocol version mismatch: "
                f"request=v{observation.protocol_version}, response=v{response.protocol_version}"
            )
        timing = {
            "encode_s": encode_s,
            **transport_timing,
            "response_parse_s": time.perf_counter() - parse_started,
            "total_s": time.perf_counter() - total_started,
        }
        self.last_request_timing = timing
        return response, timing

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_bytes(
        self,
        method: str,
        endpoint: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        retry_safe: bool,
    ) -> tuple[bytes, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("LingBot V2 HTTP client is closed")
        if not self.keep_alive:
            return self._legacy_request_bytes(method, endpoint, body=body, headers=headers)
        return self._persistent_request_bytes(
            method,
            endpoint,
            body=body,
            headers=headers,
            retry_safe=retry_safe,
        )

    def _legacy_request_bytes(
        self,
        method: str,
        endpoint: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
    ) -> tuple[bytes, dict[str, Any]]:
        request = Request(
            f"{self.server_url}{endpoint}",
            data=body,
            headers=headers,
            method=method,
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                data = response.read()
                return data, {
                    "transport": "legacy_urllib",
                    "connection_reused": False,
                    "reconnect_count": 0,
                    "request_response_s": time.perf_counter() - started,
                    "response_status": int(response.status),
                    "response_connection": response.headers.get("Connection"),
                    "server_timing_header": response.headers.get("Server-Timing"),
                    "server_timing_s": _parse_server_timing(response.headers.get("Server-Timing")),
                    "request_trace_id": response.headers.get("X-LingBot-Trace-Id"),
                }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_STATUSES:
                raise RetryableInferenceError(
                    f"LingBot V2 server HTTP {exc.code}: {detail}",
                    timing={
                        "transport": "legacy_urllib",
                        "connection_reused": False,
                        "reconnect_count": 0,
                        "request_response_s": time.perf_counter() - started,
                        "response_status": int(exc.code),
                        "response_connection": exc.headers.get("Connection"),
                        "server_timing_header": exc.headers.get("Server-Timing"),
                        "server_timing_s": _parse_server_timing(exc.headers.get("Server-Timing")),
                        "request_trace_id": exc.headers.get("X-LingBot-Trace-Id"),
                        "response_received": True,
                    },
                ) from exc
            raise RuntimeError(f"LingBot V2 server HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise RetryableInferenceError(
                f"Cannot reach LingBot V2 server at {self.server_url}: {exc}",
                timing={
                    "transport": "legacy_urllib",
                    "connection_reused": False,
                    "reconnect_count": 0,
                    "request_response_s": time.perf_counter() - started,
                    "response_received": False,
                },
            ) from exc

    def _persistent_request_bytes(
        self,
        method: str,
        endpoint: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        retry_safe: bool,
    ) -> tuple[bytes, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        max_attempts = 2 if retry_safe else 1
        with self._connection_lock:
            for attempt_index in range(max_attempts):
                attempt: dict[str, Any] = {"attempt": attempt_index + 1}
                attempts.append(attempt)
                try:
                    connection, reused, connect_s = self._ensure_connection()
                    attempt["connection_reused"] = reused
                    attempt["connect_s"] = connect_s
                    request_headers = {**headers, "Connection": "keep-alive"}
                    request_started = time.perf_counter()
                    connection.request(
                        method,
                        self._endpoint_path(endpoint),
                        body=body,
                        headers=request_headers,
                    )
                    attempt["request_write_s"] = time.perf_counter() - request_started
                    headers_started = time.perf_counter()
                    response = connection.getresponse()
                    attempt["response_headers_wait_s"] = time.perf_counter() - headers_started
                    read_started = time.perf_counter()
                    data = response.read()
                    attempt["response_read_s"] = time.perf_counter() - read_started
                    response_connection = response.getheader("Connection")
                    server_timing_header = response.getheader("Server-Timing")
                    trace_id = response.getheader("X-LingBot-Trace-Id")
                    will_close = bool(
                        response.will_close
                        or response.version < 11
                        or (response_connection or "").lower() == "close"
                    )
                    attempt["response_status"] = int(response.status)
                    attempt["response_connection"] = response_connection
                    attempt["server_will_close"] = will_close
                    if will_close:
                        self._drop_connection()
                    timing = {
                        "transport": "http_keep_alive",
                        "connection_reused": reused,
                        "reconnect_count": attempt_index,
                        "attempts": attempts,
                        "connect_s": sum(float(item.get("connect_s", 0.0)) for item in attempts),
                        "request_write_s": sum(float(item.get("request_write_s", 0.0)) for item in attempts),
                        "response_headers_wait_s": sum(
                            float(item.get("response_headers_wait_s", 0.0)) for item in attempts
                        ),
                        "response_read_s": sum(float(item.get("response_read_s", 0.0)) for item in attempts),
                        "response_status": int(response.status),
                        "response_connection": response_connection,
                        "server_will_close": will_close,
                        "server_timing_header": server_timing_header,
                        "server_timing_s": _parse_server_timing(server_timing_header),
                        "request_trace_id": trace_id,
                        "response_received": True,
                    }
                    if response.status >= 400:
                        detail = data.decode("utf-8", errors="replace")
                        if response.status in RETRYABLE_HTTP_STATUSES:
                            raise RetryableInferenceError(
                                f"LingBot V2 server HTTP {response.status}: {detail}",
                                timing=timing,
                            )
                        raise RuntimeError(f"LingBot V2 server HTTP {response.status}: {detail}")
                    return data, timing
                except RuntimeError:
                    raise
                except (http.client.HTTPException, OSError) as exc:
                    attempt["error"] = f"{type(exc).__name__}: {exc}"
                    self._drop_connection()
                    if attempt_index + 1 < max_attempts:
                        continue
                    raise RetryableInferenceError(
                        f"Cannot reach LingBot V2 server at {self.server_url}: {type(exc).__name__}: {exc}",
                        timing={
                            "transport": "http_keep_alive",
                            "connection_reused": bool(attempt.get("connection_reused", False)),
                            "reconnect_count": attempt_index,
                            "attempts": attempts,
                            "connect_s": sum(float(item.get("connect_s", 0.0)) for item in attempts),
                            "request_write_s": sum(
                                float(item.get("request_write_s", 0.0)) for item in attempts
                            ),
                            "response_headers_wait_s": sum(
                                float(item.get("response_headers_wait_s", 0.0)) for item in attempts
                            ),
                            "response_read_s": sum(
                                float(item.get("response_read_s", 0.0)) for item in attempts
                            ),
                            "response_received": False,
                        },
                    ) from exc
        raise RuntimeError("LingBot V2 persistent HTTP request failed without an exception")

    def _ensure_connection(
        self,
    ) -> tuple[http.client.HTTPConnection | http.client.HTTPSConnection, bool, float]:
        if self._connection is not None and self._connection.sock is not None:
            return self._connection, True, 0.0
        self._drop_connection()
        if self._scheme == "https":
            connection = http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self.timeout_s,
                context=self._ssl_context or ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout_s)
        connect_started = time.perf_counter()
        connection.connect()
        connect_s = time.perf_counter() - connect_started
        self._connection = connection
        return connection, False, connect_s

    def _endpoint_path(self, endpoint: str) -> str:
        return f"{self._base_path}{endpoint}" or "/"

    def _drop_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def reset_connection(self) -> None:
        """Discard the current HTTP connection without closing the client.

        ``health`` is intentionally sent before opening hardware and waiting
        for the operator's execution confirmation.  A keep-alive server may
        close that idle socket while the operator is deciding; explicitly
        dropping it here makes the first ``/predict`` establish a fresh
        connection instead of mistaking a stale socket for a reusable one.
        This method never retries or replays a prediction request.
        """

        with self._connection_lock:
            if self._closed:
                return
            self._drop_connection()

    def close(self) -> None:
        with self._connection_lock:
            if self._closed:
                return
            self._closed = True
            self._drop_connection()

    def __enter__(self) -> "LingBotV2HttpClient":
        return self

    def __exit__(self, exc_type, exc, traceback_obj) -> None:
        self.close()


def _parse_server_timing(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    parsed: dict[str, float] = {}
    for item in value.split(","):
        fields = [part.strip() for part in item.split(";") if part.strip()]
        if not fields:
            continue
        name = fields[0].replace("-", "_")
        duration_ms: float | None = None
        for field in fields[1:]:
            if not field.startswith("dur="):
                continue
            try:
                duration_ms = float(field[4:])
            except ValueError:
                duration_ms = None
            break
        if duration_ms is not None and np.isfinite(duration_ms) and duration_ms >= 0.0:
            parsed[f"{name}_s"] = duration_ms / 1000.0
    return parsed


def center_crop_resize_rgb(frame_bgr: np.ndarray, *, output_size: int) -> np.ndarray:
    """Reproduce dataset generation: center crop to target aspect, resize, BGR->RGB."""

    frame = np.asarray(frame_bgr)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected uint8 BGR frame (H,W,3), got {frame.dtype} {frame.shape}")
    input_height, input_width = frame.shape[:2]
    output_width = output_height = int(output_size)
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")

    crop_height = input_height
    crop_width = round(input_height / output_height * output_width)
    if crop_width > input_width:
        crop_width = input_width
        crop_height = round(input_width / output_width * output_height)
    x0 = (input_width - crop_width) // 2
    y0 = (input_height - crop_height) // 2
    cropped_rgb = frame[y0 : y0 + crop_height, x0 : x0 + crop_width, ::-1]
    resized_rgb = cv2.resize(cropped_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized_rgb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LingBot-VLA V2 TacThru UMI Realman client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="Validate the inference-server contract")
    _add_http_args(health, default_timeout=10.0)

    synthetic = subparsers.add_parser("synthetic", help="Send one request without opening camera or robot hardware")
    _add_http_args(synthetic, default_timeout=120.0)
    synthetic.add_argument("--instruction", default="Pull the tissue")
    synthetic.add_argument("--image", type=Path, default=None, help="Optional local image; black 224x224 is used otherwise")
    synthetic.add_argument("--state", default="0,0,0,0,0,0,1,0.045")
    synthetic.add_argument("--control-frequency", type=float, default=30.0)
    _add_tactile_args(synthetic, hardware=False)

    run = subparsers.add_parser("run", help="Capture wrist RGB/state, infer, and optionally control Realman")
    _add_http_args(run, default_timeout=120.0)
    run.add_argument("--instruction", default="Pull the tissue")
    run.add_argument("--control-frequency", type=float, default=30.0)
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--rate-hz", type=float, default=1.0)
    run.add_argument("--max-roundtrip-s", type=float, default=30.0)
    run.add_argument(
        "--max-consecutive-roundtrip-rejects",
        type=int,
        default=0,
        help=(
            "In non-streaming mode, discard stale/transiently failed predictions and recapture fresh "
            "sensors until this many consecutive rejects occur. 0 preserves fail-fast behavior."
        ),
    )
    run.add_argument("--execute", action="store_true", help="Enable real actuator commands after a mandatory Space confirmation")
    run.add_argument("--stream-replan", action="store_true", help="Do not wait for the dispatched short chunk to finish")
    run.add_argument("--preview", action="store_true", help="Show the exact 224x224 RGB frame sent to the server")
    run.add_argument("--wait-for-space", action="store_true", help="Also gate a dry-run on Space; execute mode is always gated")
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--log-jsonl", type=Path, default=None)
    run.add_argument("--tacthru-repo", type=Path, default=Path("/mnt/models/VTLA-RDT/tacthru"))
    run.add_argument("--camera-cfg", type=Path, default=Path("cfg/camera/synria_c10.yaml"))
    run.add_argument("--robot-cfg", type=Path, default=Path("cfg/robot/realman.yaml"))
    run.add_argument("--gripper-cfg", type=Path, default=Path("cfg/gripper/synria_gloria.yaml"))
    _add_tactile_args(run, hardware=True)
    run.add_argument("--disable-gripper", action="store_true")
    run.add_argument("--realman-ip", default=None)
    run.add_argument("--realman-port", type=int, default=None)
    run.add_argument("--assumed-gripper-width-m", type=float, default=0.045)
    run.add_argument("--gripper-min-width-m", type=float, default=0.0)
    run.add_argument("--gripper-max-width-m", type=float, default=None)
    run.add_argument("--gripper-command-torque", type=int, default=None)
    run.add_argument("--gripper-command-speed", type=int, default=None)
    run.add_argument(
        "--gripper-startup-width-m",
        type=float,
        default=None,
        help=(
            "Move Gloria directly to this width during initialization, require fresh hardware "
            "feedback within tolerance, then ask for a second Space confirmation before inference."
        ),
    )
    run.add_argument("--gripper-startup-tolerance-m", type=float, default=0.005)
    run.add_argument("--gripper-startup-timeout-s", type=float, default=3.0)
    run.add_argument(
        "--gripper-action-select",
        choices=["last", "first", "min", "median", "threshold"],
        default=None,
        help=(
            "Select the commanded width from the executable action window. "
            "threshold uses a non-latching binary policy: min prediction at or below "
            "--gripper-hold-closed-below-m closes to --gripper-hold-closed-target-m; "
            "otherwise it opens to gripper initial_width_mm."
        ),
    )
    run.add_argument(
        "--gripper-hold-closed-below-m",
        type=float,
        default=None,
        help="Inclusive threshold in metres for threshold mode; also the latch threshold for non-threshold modes.",
    )
    run.add_argument(
        "--gripper-hold-closed-target-m",
        type=float,
        default=None,
        help="Closed target width in metres.",
    )
    run.add_argument(
        "--gripper-open-lookahead-start-step",
        type=int,
        default=None,
        help=(
            "Enable threshold-mode release lookahead at this full action-chunk index. Motion still uses "
            "--exec-start-step/--exec-end-step; omit this and the consecutive count to preserve old behavior."
        ),
    )
    run.add_argument(
        "--gripper-open-lookahead-consecutive-steps",
        type=int,
        default=None,
        help=(
            "Open when any run of this many predictions at/after the lookahead start are all above the "
            "gripper threshold."
        ),
    )
    run.add_argument("--exec-start-step", type=int, default=2)
    run.add_argument("--exec-end-step", type=int, default=8)
    run.add_argument("--no-preserve-exec-window-length", action="store_true")
    run.add_argument("--robot-action-latency", type=float, default=0.1)
    run.add_argument("--max-pos-speed", type=float, default=0.04)
    run.add_argument("--max-rot-speed", type=float, default=0.08)
    run.add_argument("--max-target-delta-m", type=float, default=0.12)
    run.add_argument("--max-target-rotation-rad", type=float, default=1.2)
    run.add_argument("--max-step-delta-m", type=float, default=0.04)
    run.add_argument("--max-step-rotation-rad", type=float, default=0.5)
    run.add_argument("--max-observation-drift-m", type=float, default=0.05)
    run.add_argument("--max-observation-drift-rotation-rad", type=float, default=0.35)
    run.add_argument("--max-robot-state-age-s", type=float, default=0.25)
    run.add_argument("--max-gripper-state-age-s", type=float, default=0.50)
    run.add_argument("--max-sensor-skew-s", type=float, default=0.10)
    run.add_argument("--max-scheduled-duration-s", type=float, default=3.0)
    run.add_argument("--verify-position-tolerance-m", type=float, default=0.03)
    run.add_argument("--verify-rotation-tolerance-rad", type=float, default=0.35)
    run.add_argument("--verify-gripper-tolerance-m", type=float, default=0.01)
    run.add_argument("--verification-timeout-s", type=float, default=1.0)
    run.add_argument(
        "--gripper-exit-policy",
        choices=["prompt", "immediate"],
        default="prompt",
        help="Before shutdown, prompt before Gloria torque is disabled (execute mode only).",
    )
    run.add_argument("--workspace-min-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    run.add_argument("--workspace-max-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    return parser


def _add_tactile_args(parser: argparse.ArgumentParser, *, hardware: bool) -> None:
    parser.add_argument(
        "--protocol-version",
        choices=["auto", "1", "2"],
        default="auto",
        help="auto keeps wrist-only requests on v1 and selects v2 when tactile/ablation is requested.",
    )
    parser.add_argument(
        "--tactile-mode",
        choices=["off", "rgb", "marker", "rgb-marker"],
        default="off",
    )
    parser.add_argument("--tactile-temporal-horizon", type=int, default=None)
    parser.add_argument("--tactile-history-stride", type=int, default=None)
    parser.add_argument("--allow-missing-tactile", action="store_true")
    parser.add_argument("--force-mask-tactile-rgb", action="store_true")
    parser.add_argument("--force-mask-marker", action="store_true")
    if hardware:
        parser.add_argument("--tactile-sensor-cfg", type=Path, default=Path("cfg/sensor/ml.yaml"))
        parser.add_argument(
            "--local-tactile-guard",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Fail closed on stale/lost/extreme tactile input; this guard never relaxes arm limits.",
        )
        parser.add_argument("--max-tactile-age-s", type=float, default=0.15)
        parser.add_argument("--max-tactile-skew-s", type=float, default=0.10)
        parser.add_argument("--min-valid-markers", type=int, default=40)
        parser.add_argument("--max-marker-flow-norm", type=float, default=None)
        parser.add_argument("--max-marker-velocity-norm-s", type=float, default=None)
    else:
        parser.add_argument("--tactile-image", type=Path, default=None)
        parser.add_argument(
            "--marker-flow",
            type=Path,
            default=None,
            help="Optional .npy marker flow with shape [48,2] or [K,48,2]; zeros are used otherwise.",
        )


def _add_http_args(parser: argparse.ArgumentParser, *, default_timeout: float) -> None:
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--timeout", type=float, default=default_timeout)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--api-key", default=os.environ.get("LINGBOT_V2_API_KEY"))
    parser.add_argument(
        "--http-keep-alive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one HTTP/1.1 connection; --no-http-keep-alive restores urllib Connection: close.",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    client = LingBotV2HttpClient(
        args.server_url,
        timeout_s=args.timeout,
        jpeg_quality=args.jpeg_quality,
        api_key=args.api_key,
        keep_alive=args.http_keep_alive,
    )
    try:
        health = client.health()
        chunk_size = validate_server_health(health)
        if args.command in {"synthetic", "run"}:
            _resolve_tactile_request_args(args, health)
        if hasattr(args, "control_frequency") and not np.isclose(
            float(args.control_frequency), float(health["control_frequency_hz"]), atol=1e-6
        ):
            raise RuntimeError(
                f"This deployment must run at {health['control_frequency_hz']:g}Hz, "
                f"got --control-frequency {args.control_frequency:g}"
            )
        if args.command == "health":
            print(json.dumps(health, indent=2, ensure_ascii=False))
        elif args.command == "synthetic":
            run_synthetic(args, client, health, chunk_size)
        elif args.command == "run":
            run_realman(args, client, health, chunk_size)
        else:
            raise RuntimeError(f"Unhandled command: {args.command}")
    finally:
        client.close()


def validate_server_health(health: dict) -> int:
    if not isinstance(health, dict) or health.get("ok") is not True or health.get("ready") is not True:
        raise RuntimeError(f"Server is not ready: {health!r}")
    expected = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "pose_frame": POSE_FRAME,
        "pose_semantics": POSE_SEMANTICS,
        "camera_key": CAMERA_KEY,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
    }
    for key, value in expected.items():
        if health.get(key) != value:
            raise RuntimeError(f"Server contract mismatch for {key}: expected {value!r}, got {health.get(key)!r}")
    if health.get("robot_config") not in {ROBOT_CONFIG, TACTILE_ROBOT_CONFIG}:
        raise RuntimeError(
            f"Server contract mismatch for robot_config: got {health.get('robot_config')!r}"
        )
    chunk_size = int(health.get("chunk_size", 0))
    if chunk_size <= 0:
        raise RuntimeError(f"Invalid server chunk_size: {chunk_size}")
    validate_action_spec(health.get("action_spec"), expected_steps=chunk_size)
    image_shape = health.get("image_shape_hwc")
    if image_shape != [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3]:
        raise RuntimeError(f"Expected server image_shape_hwc=[224,224,3], got {image_shape!r}")
    supported = health.get("protocol_versions_supported", [health.get("protocol_version")])
    if not isinstance(supported, list) or PROTOCOL_VERSION_V1 not in supported:
        raise RuntimeError(f"Server must retain protocol v1 compatibility, got {supported!r}")
    return chunk_size


def _resolve_tactile_request_args(args: argparse.Namespace, health: dict) -> None:
    requested = _tactile_mode_flags(args.tactile_mode)
    force_mask = {
        "tactile_rgb": bool(args.force_mask_tactile_rgb),
        "tactile_marker": bool(args.force_mask_marker),
    }
    modalities = health.get("checkpoint_modalities")
    if not isinstance(modalities, dict):
        legacy_enabled = bool(health.get("tactile_enabled", TACTILE_ENABLED))
        modalities = {
            "wrist_rgb": True,
            "tactile_rgb": legacy_enabled,
            "tactile_marker": False,
        }
    checkpoint = {
        "tactile_rgb": bool(modalities.get("tactile_rgb", False)),
        "tactile_marker": bool(modalities.get("tactile_marker", False)),
    }
    supported = health.get("protocol_versions_supported", [health.get("protocol_version")])
    explicit_version = None if args.protocol_version == "auto" else int(args.protocol_version)
    needs_v2 = any(requested.values()) or any(force_mask.values()) or any(checkpoint.values())
    protocol_version = explicit_version or (PROTOCOL_VERSION_V2 if needs_v2 else PROTOCOL_VERSION_V1)
    if protocol_version not in supported:
        raise RuntimeError(
            f"Server does not support requested protocol v{protocol_version}; supported={supported!r}"
        )
    if protocol_version == PROTOCOL_VERSION_V1 and needs_v2:
        raise RuntimeError("Protocol v1 cannot carry tactile, missing-modality, or force-mask requests")

    execute = bool(getattr(args, "execute", False))
    safe_ablation_context = args.command == "synthetic" or not execute
    if execute and args.allow_missing_tactile:
        raise RuntimeError("--allow-missing-tactile is forbidden with --execute")
    if execute and any(force_mask.values()):
        raise RuntimeError("Tactile force-mask is allowed only for synthetic/offline/dry-run evaluation")
    if any(force_mask.values()) and not safe_ablation_context:
        raise RuntimeError("Tactile force-mask is forbidden for execution")
    if any(force_mask.values()) and health.get("allow_tactile_ablation") is not True:
        raise RuntimeError("Server health does not permit tactile ablation")
    if args.allow_missing_tactile and health.get("allow_missing_tactile") is not True:
        raise RuntimeError("Server health does not permit missing tactile input")

    for modality in ("tactile_rgb", "tactile_marker"):
        if requested[modality] and not checkpoint[modality]:
            raise RuntimeError(f"Checkpoint does not support requested modality {modality}")
        if force_mask[modality] and not checkpoint[modality]:
            raise RuntimeError(f"Cannot force-mask absent checkpoint modality {modality}")
        if checkpoint[modality] and not requested[modality]:
            if execute or not args.allow_missing_tactile:
                raise RuntimeError(
                    f"Checkpoint requires {modality}; select a matching --tactile-mode. "
                    "Missing tactile is available only for explicitly enabled dry-run/synthetic ablation."
                )

    horizon = int(health.get("tactile_temporal_horizon", 1))
    stride = int(health.get("tactile_history_stride", 1))
    if args.tactile_temporal_horizon is not None and int(args.tactile_temporal_horizon) != horizon:
        raise RuntimeError(
            f"Tactile horizon mismatch: checkpoint={horizon}, CLI={args.tactile_temporal_horizon}"
        )
    if args.tactile_history_stride is not None and int(args.tactile_history_stride) != stride:
        raise RuntimeError(
            f"Tactile history stride mismatch: checkpoint={stride}, CLI={args.tactile_history_stride}"
        )
    if horizon <= 0 or stride <= 0:
        raise RuntimeError(f"Invalid tactile history contract: horizon={horizon}, stride={stride}")
    if execute and any(checkpoint.values()) and not bool(getattr(args, "local_tactile_guard", False)):
        raise RuntimeError("Tactile checkpoint execution requires --local-tactile-guard")

    contract_sha256 = health.get("tactile_contract_sha256")
    if any(checkpoint.values()):
        if not isinstance(contract_sha256, str) or len(contract_sha256) != 64:
            raise RuntimeError("Tactile checkpoint health is missing tactile_contract_sha256")
    args._tactile_requested = requested
    args._checkpoint_tactile = checkpoint
    args._protocol_version = protocol_version
    args._tactile_horizon = horizon
    args._tactile_stride = stride
    args._tactile_contract_sha256 = contract_sha256


def _tactile_mode_flags(mode: str) -> dict[str, bool]:
    mapping = {
        "off": {"tactile_rgb": False, "tactile_marker": False},
        "rgb": {"tactile_rgb": True, "tactile_marker": False},
        "marker": {"tactile_rgb": False, "tactile_marker": True},
        "rgb-marker": {"tactile_rgb": True, "tactile_marker": True},
    }
    try:
        return dict(mapping[mode])
    except KeyError as exc:
        raise ValueError(f"Unsupported tactile mode: {mode!r}") from exc


def run_synthetic(
    args: argparse.Namespace,
    client: LingBotV2HttpClient,
    health: dict,
    chunk_size: int,
) -> None:
    state = _parse_state(args.state)
    if args.image is None:
        image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
    else:
        bgr = cv2.imread(str(args.image.expanduser().resolve()), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        image = center_crop_resize_rgb(bgr, output_size=DEFAULT_IMAGE_SIZE)
    now = time.time()
    requested = args._tactile_requested
    horizon = int(args._tactile_horizon)
    tactile_rgb = tactile_rgb_timestamps = tactile_rgb_mask = None
    if requested["tactile_rgb"]:
        if args.tactile_image is None:
            tactile_image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            tactile_bgr = cv2.imread(str(args.tactile_image.expanduser().resolve()), cv2.IMREAD_COLOR)
            if tactile_bgr is None:
                raise FileNotFoundError(args.tactile_image)
            tactile_image = cv2.resize(
                tactile_bgr[..., ::-1],
                (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
        tactile_rgb = np.repeat(tactile_image[None, ...], horizon, axis=0)
        tactile_rgb_timestamps = np.full((horizon,), now, dtype=np.float64)
        tactile_rgb_mask = np.ones((horizon,), dtype=np.bool_)

    marker_flow = marker_valid = marker_timestamps = marker_mask = None
    if requested["tactile_marker"]:
        if args.marker_flow is None:
            marker_flow = np.zeros(
                (horizon, TACTILE_MARKER_COUNT, TACTILE_MARKER_DIM), dtype=np.float32
            )
        else:
            marker_flow = np.asarray(
                np.load(args.marker_flow.expanduser().resolve(), allow_pickle=False), dtype=np.float32
            )
            if marker_flow.shape == (TACTILE_MARKER_COUNT, TACTILE_MARKER_DIM):
                marker_flow = np.repeat(marker_flow[None, ...], horizon, axis=0)
            if marker_flow.shape != (horizon, TACTILE_MARKER_COUNT, TACTILE_MARKER_DIM):
                raise ValueError(
                    "--marker-flow must have shape [48,2] or "
                    f"[{horizon},48,2], got {marker_flow.shape}"
                )
        marker_valid = np.ones((horizon, TACTILE_MARKER_COUNT), dtype=np.bool_)
        marker_timestamps = np.full((horizon,), now, dtype=np.float64)
        marker_mask = np.ones((horizon,), dtype=np.bool_)

    observation = Observation(
        instruction=args.instruction,
        state=state,
        wrist_rgb=image,
        control_frequency_hz=args.control_frequency,
        timestamp=now,
        metadata={"synthetic": True, "episode_reset": True, "dry_run": True, "execute": False},
        protocol_version=args._protocol_version,
        contract_sha256=args._tactile_contract_sha256 if args._protocol_version == PROTOCOL_VERSION_V2 else None,
        wrist_timestamp=now if args._protocol_version == PROTOCOL_VERSION_V2 else None,
        tactile_rgb_history=tactile_rgb,
        tactile_rgb_timestamps=tactile_rgb_timestamps,
        tactile_rgb_history_mask=tactile_rgb_mask,
        marker_flow=marker_flow,
        marker_valid_mask=marker_valid,
        marker_timestamps=marker_timestamps,
        marker_history_mask=marker_mask,
        force_mask_tactile_rgb=args.force_mask_tactile_rgb,
        force_mask_tactile_marker=args.force_mask_marker,
    )
    response, timing = client.predict_timed(observation, expected_steps=chunk_size)
    print(
        json.dumps(
            {
                "request_id": response.request_id,
                "roundtrip_s": timing["total_s"],
                "latency": timing,
                "action_shape": list(response.action_chunk.shape),
                "first_action": response.action_chunk[0].astype(float).tolist(),
                "metadata": response.metadata,
                "protocol_version": response.protocol_version,
                "server_checkpoint_modalities": health.get("checkpoint_modalities"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_realman(
    args: argparse.Namespace,
    client: LingBotV2HttpClient,
    health: dict,
    chunk_size: int,
) -> None:
    if not hasattr(args, "_tactile_requested"):
        requested = _tactile_mode_flags(getattr(args, "tactile_mode", "off"))
        if any(requested.values()):
            raise RuntimeError("Tactile run must validate the server health contract before opening hardware")
        args._tactile_requested = requested
        args._checkpoint_tactile = {"tactile_rgb": False, "tactile_marker": False}
        args._protocol_version = PROTOCOL_VERSION_V1
        args._tactile_horizon = 1
        args._tactile_stride = 1
        args._tactile_contract_sha256 = None
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.max_consecutive_roundtrip_rejects < 0:
        raise ValueError("--max-consecutive-roundtrip-rejects must be non-negative")
    for name in (
        "timeout",
        "control_frequency",
        "rate_hz",
        "max_roundtrip_s",
        "max_sensor_skew_s",
        "verify_position_tolerance_m",
        "verify_rotation_tolerance_rad",
        "verify_gripper_tolerance_m",
        "verification_timeout_s",
        "gripper_startup_tolerance_m",
        "gripper_startup_timeout_s",
    ):
        _require_positive_cli(getattr(args, name), f"--{name.replace('_', '-')}")
    if any(args._tactile_requested.values()):
        for name in ("max_tactile_age_s", "max_tactile_skew_s"):
            _require_positive_cli(getattr(args, name), f"--{name.replace('_', '-')}")
        if args.min_valid_markers < 0 or args.min_valid_markers > TACTILE_MARKER_COUNT:
            raise ValueError(
                f"--min-valid-markers must be in [0,{TACTILE_MARKER_COUNT}], "
                f"got {args.min_valid_markers}"
            )
        for name in ("max_marker_flow_norm", "max_marker_velocity_norm_s"):
            value = getattr(args, name)
            if value is not None:
                _require_positive_cli(value, f"--{name.replace('_', '-')}")
    if args.execute and (args.workspace_min_xyz is None or args.workspace_max_xyz is None):
        raise RuntimeError(
            "Real execution requires explicit --workspace-min-xyz X Y Z and "
            "--workspace-max-xyz X Y Z bounds in the Realman base frame"
        )
    if args.disable_gripper and args.gripper_startup_width_m is not None:
        raise RuntimeError("--gripper-startup-width-m cannot be used with --disable-gripper")
    lookahead_start = args.gripper_open_lookahead_start_step
    lookahead_count = args.gripper_open_lookahead_consecutive_steps
    if (lookahead_start is None) != (lookahead_count is None):
        raise ValueError(
            "--gripper-open-lookahead-start-step and "
            "--gripper-open-lookahead-consecutive-steps must be supplied together"
        )
    if lookahead_start is not None and (lookahead_start < 0 or lookahead_count <= 0):
        raise ValueError("Gripper lookahead start must be non-negative and consecutive steps must be positive")
    if lookahead_start is not None and lookahead_start + lookahead_count > chunk_size:
        raise ValueError(
            "Gripper lookahead window exceeds the server action chunk: "
            f"start={lookahead_start}, consecutive={lookahead_count}, chunk_size={chunk_size}"
        )
    if lookahead_start is not None and args.gripper_action_select != "threshold":
        raise ValueError("Gripper open lookahead requires --gripper-action-select threshold")
    if args.max_consecutive_roundtrip_rejects > 0 and args.stream_replan:
        raise RuntimeError(
            "Recoverable inference rejection cannot be combined with --stream-replan because a prior "
            "trajectory may still be active"
        )
    if args.max_consecutive_roundtrip_rejects > 0 and health.get("requests_are_stateless") is not True:
        raise RuntimeError(
            "Recoverable inference rejection requires server health requests_are_stateless=true"
        )
    # ``main`` performs a health check before opening the camera and waiting
    # for the Space safety gate.  Do not carry that potentially idle
    # keep-alive socket into the first prediction: the server may have closed
    # it during operator setup, and /predict must never be replayed after an
    # ambiguous disconnect.
    client.reset_connection()
    parsed_server_url = urlparse(args.server_url)
    if args.execute and parsed_server_url.scheme != "https" and parsed_server_url.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError(
            "Real execution over plaintext HTTP is allowed only through a localhost SSH tunnel. "
            "Use --server-url http://127.0.0.1:<port> or terminate TLS with HTTPS."
        )
    tacthru_repo = args.tacthru_repo.expanduser().resolve()
    camera_cfg = _resolve_repo_path(tacthru_repo, args.camera_cfg)
    robot_cfg = _resolve_repo_path(tacthru_repo, args.robot_cfg)
    gripper_cfg = _resolve_repo_path(tacthru_repo, args.gripper_cfg)
    tactile_cfg = _resolve_repo_path(tacthru_repo, args.tactile_sensor_cfg)
    tactile_source: TacThruTactileSource | None = None
    camera: WristCamera | None = None
    runtime: RealmanEpisodeRuntime | None = None
    try:
        if any(args._tactile_requested.values()):
            try:
                tactile_source = _open_tactile_source(
                    tacthru_repo=tacthru_repo,
                    sensor_cfg_path=tactile_cfg,
                    history_steps=args._tactile_horizon,
                    history_stride=args._tactile_stride,
                    include_rgb=args._tactile_requested["tactile_rgb"],
                    include_marker=args._tactile_requested["tactile_marker"],
                )
            except BaseException as exc:
                if args.allow_missing_tactile and not args.execute:
                    print(
                        f"[lingbot-v2-client] tactile source unavailable in allowed dry-run ablation: {exc}",
                        flush=True,
                    )
                    tactile_source = None
                else:
                    raise
        camera = _open_wrist_camera(
            camera_cfg,
            max_frame_age_s=args.max_sensor_skew_s * CAMERA_FRESHNESS_FRACTION,
        )
        runtime = RealmanEpisodeRuntime(
            RealmanConfig(
            tacthru_repo=tacthru_repo,
            robot_cfg=robot_cfg,
            gripper_cfg=gripper_cfg,
            robot_ip=args.realman_ip,
            robot_port=args.realman_port,
            enable_gripper=not args.disable_gripper,
            assumed_gripper_width_m=args.assumed_gripper_width_m,
            gripper_min_width_m=args.gripper_min_width_m,
            gripper_max_width_m=args.gripper_max_width_m,
            gripper_command_torque=args.gripper_command_torque,
            gripper_command_speed=args.gripper_command_speed,
            gripper_startup_width_m=args.gripper_startup_width_m,
            gripper_startup_tolerance_m=args.gripper_startup_tolerance_m,
            gripper_startup_timeout_s=args.gripper_startup_timeout_s,
            gripper_action_select=args.gripper_action_select,
            gripper_hold_closed_below_m=args.gripper_hold_closed_below_m,
            gripper_hold_closed_target_m=args.gripper_hold_closed_target_m,
            gripper_open_lookahead_start_step=args.gripper_open_lookahead_start_step,
            gripper_open_lookahead_consecutive_steps=args.gripper_open_lookahead_consecutive_steps,
            exec_start_step=args.exec_start_step,
            exec_end_step=args.exec_end_step,
            preserve_exec_window_length=not args.no_preserve_exec_window_length,
            robot_action_latency_s=args.robot_action_latency,
            max_pos_speed_m_s=args.max_pos_speed,
            max_rot_speed_rad_s=args.max_rot_speed,
            max_target_delta_m=args.max_target_delta_m,
            max_target_rotation_rad=args.max_target_rotation_rad,
            max_step_delta_m=args.max_step_delta_m,
            max_step_rotation_rad=args.max_step_rotation_rad,
            max_observation_drift_m=args.max_observation_drift_m,
            max_observation_drift_rotation_rad=args.max_observation_drift_rotation_rad,
            max_robot_state_age_s=args.max_robot_state_age_s,
            max_gripper_state_age_s=args.max_gripper_state_age_s,
            max_scheduled_duration_s=args.max_scheduled_duration_s,
            workspace_min_xyz_m=tuple(args.workspace_min_xyz) if args.workspace_min_xyz else None,
            workspace_max_xyz_m=tuple(args.workspace_max_xyz) if args.workspace_max_xyz else None,
            )
        )
    except BaseException:
        if camera is not None:
            camera.close()
        if tactile_source is not None:
            tactile_source.close()
        raise
    output_dir, log_path = _prepare_logging(
        args,
        resolved_paths={
            "tacthru_repo": tacthru_repo,
            "camera_cfg": camera_cfg,
            "robot_cfg": robot_cfg,
            "gripper_cfg": gripper_cfg,
            "tactile_sensor_cfg": tactile_cfg,
        },
        health=health,
    )
    session_id = uuid.uuid4().hex
    error: str | None = None
    try:
        runtime.start()
        if args.execute:
            if args.gripper_open_lookahead_start_step is not None:
                print(
                    "\n[SAFETY] Gripper release lookahead is enabled: a consecutive above-threshold "
                    "run in the full future action chunk may open Gloria even though the arm still "
                    "executes only the configured short window. Review a dry-run before relying on it.\n",
                    flush=True,
                )
            print(
                "\n[SAFETY] --execute was supplied. Space confirmation will enable the arm trajectory "
                "and may initialize/move the gripper. Keep an emergency stop within reach.\n",
                flush=True,
            )
            _wait_for_space(camera if args.preview else None, execute=True)
            startup_verification = runtime.enable_actuation()
            if startup_verification is not None:
                _append_log(
                    log_path,
                    {
                        "event": "gripper_startup_verified",
                        "timestamp": time.time(),
                        **startup_verification,
                    },
                )
                print(
                    "[SAFETY] Gloria reached the startup width. Confirm the held object is secure; "
                    "no inference request has been sent yet.",
                    flush=True,
                )
                _wait_for_space(
                    camera if args.preview else None,
                    execute=True,
                    label="CONFIRM OBJECT IS SECURE AND START INFERENCE",
                )
        else:
            if args.wait_for_space:
                _wait_for_space(camera if args.preview else None, execute=False)
            runtime.reset_episode_start()

        period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
        active_trajectory_until_s = 0.0
        step_index = 0
        request_attempt = 0
        consecutive_inference_rejects = 0
        while step_index < args.steps:
            request_attempt += 1
            loop_start = time.time()
            camera_frame = camera.capture()
            snapshot = runtime.read_policy_state()
            sensor_skew_s = abs(camera_frame.capture_timestamp - snapshot.timestamp)
            if sensor_skew_s > args.max_sensor_skew_s:
                raise SafetyViolation(
                    f"Wrist image/robot state skew {sensor_skew_s:.4f}s exceeds "
                    f"--max-sensor-skew-s {args.max_sensor_skew_s:.4f}s"
                )
            tactile_history: TactileHistory | None = None
            tactile_guard_debug: dict[str, Any] | None = None
            tactile_capture_error: str | None = None
            tactile_capture_s = 0.0
            tactile_guard_s = 0.0
            if tactile_source is not None:
                tactile_capture_started = time.perf_counter()
                try:
                    tactile_history = tactile_source.capture()
                except BaseException as exc:
                    tactile_capture_error = f"{type(exc).__name__}: {exc}"
                    if args.execute or not args.allow_missing_tactile:
                        raise SafetyViolation(
                            f"tactile sensor capture failed; no action will be planned: {tactile_capture_error}"
                        ) from exc
                finally:
                    tactile_capture_s = time.perf_counter() - tactile_capture_started
            if tactile_history is not None and args.local_tactile_guard:
                tactile_guard_started = time.perf_counter()
                tactile_guard_debug = validate_local_tactile_guard(
                    tactile_history,
                    wrist_timestamp=camera_frame.capture_timestamp,
                    robot_timestamp=snapshot.timestamp,
                    now=time.time(),
                    marker_required=args._tactile_requested["tactile_marker"],
                    max_tactile_age_s=args.max_tactile_age_s,
                    max_tactile_skew_s=min(
                        args.max_tactile_skew_s,
                        float(health.get("tactile_max_timestamp_skew_s", args.max_tactile_skew_s)),
                    ),
                    min_valid_markers=args.min_valid_markers,
                    max_marker_flow_norm=args.max_marker_flow_norm,
                    max_marker_velocity_norm_s=args.max_marker_velocity_norm_s,
                )
                tactile_guard_s = time.perf_counter() - tactile_guard_started
            observation_timestamps = [camera_frame.capture_timestamp, snapshot.timestamp]
            if tactile_history is not None:
                if tactile_history.rgb_timestamps is not None:
                    observation_timestamps.append(float(tactile_history.rgb_timestamps[-1]))
                if tactile_history.marker_timestamps is not None:
                    observation_timestamps.append(float(tactile_history.marker_timestamps[-1]))
            observation_timestamp = min(observation_timestamps)
            observation = Observation(
                instruction=args.instruction,
                state=snapshot.state,
                wrist_rgb=camera_frame.rgb,
                control_frequency_hz=args.control_frequency,
                session_id=session_id,
                timestamp=observation_timestamp,
                protocol_version=args._protocol_version,
                contract_sha256=(
                    args._tactile_contract_sha256
                    if args._protocol_version == PROTOCOL_VERSION_V2
                    else None
                ),
                wrist_timestamp=(
                    camera_frame.capture_timestamp
                    if args._protocol_version == PROTOCOL_VERSION_V2
                    else None
                ),
                tactile_rgb_history=(tactile_history.rgb if tactile_history is not None else None),
                tactile_rgb_timestamps=(
                    tactile_history.rgb_timestamps if tactile_history is not None else None
                ),
                tactile_rgb_history_mask=(
                    tactile_history.rgb_history_mask if tactile_history is not None else None
                ),
                marker_flow=(tactile_history.marker_flow if tactile_history is not None else None),
                marker_valid_mask=(
                    tactile_history.marker_valid_mask if tactile_history is not None else None
                ),
                marker_timestamps=(
                    tactile_history.marker_timestamps if tactile_history is not None else None
                ),
                marker_history_mask=(
                    tactile_history.marker_history_mask if tactile_history is not None else None
                ),
                force_mask_tactile_rgb=args.force_mask_tactile_rgb,
                force_mask_tactile_marker=args.force_mask_marker,
                metadata={
                    # Only the first request asks for an explicit reset. If an
                    # ambiguous first request never reached the server, the
                    # retry still resets because this new session_id differs
                    # from the server's active session. Repeating
                    # episode_reset after a response was lost could otherwise
                    # reset a stateful policy twice.
                    "episode_reset": request_attempt == 1,
                    "client_step": step_index,
                    "client_request_attempt": request_attempt,
                    "pose_frame": POSE_FRAME,
                    "dry_run": not args.execute,
                    "execute": bool(args.execute),
                    "camera_capture_timestamp": camera_frame.capture_timestamp,
                    "camera_receive_timestamp": camera_frame.receive_timestamp,
                    "robot_state_timestamp": snapshot.timestamp,
                    "sensor_skew_s": sensor_skew_s,
                    "tactile_mode": args.tactile_mode,
                    "tactile_capture_error": tactile_capture_error,
                    "tactile": tactile_history.debug if tactile_history is not None else None,
                    "local_tactile_guard": tactile_guard_debug,
                    "tactile_capture_s": tactile_capture_s,
                    "tactile_guard_s": tactile_guard_s,
                },
            )
            if args.preview:
                _show_preview(camera_frame.rgb, f"step={step_index} sending")
            request_started = time.perf_counter()
            try:
                response, latency = client.predict_timed(observation, expected_steps=chunk_size)
            except RetryableInferenceError as exc:
                roundtrip_s = time.perf_counter() - request_started
                consecutive_inference_rejects += 1
                will_retry = bool(
                    args.max_consecutive_roundtrip_rejects > 0
                    and consecutive_inference_rejects < args.max_consecutive_roundtrip_rejects
                )
                latency = dict(exc.timing)
                latency["outer_roundtrip_s"] = roundtrip_s
                client.reset_connection()
                _append_log(
                    log_path,
                    {
                        "event": "inference_transport_rejected",
                        "timestamp": time.time(),
                        "step": step_index,
                        "request_attempt": request_attempt,
                        "request_id": observation.request_id,
                        "session_id": session_id,
                        "execute": bool(args.execute),
                        "roundtrip_s": roundtrip_s,
                        "max_roundtrip_s": args.max_roundtrip_s,
                        "response_received": bool(latency.get("response_received", False)),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "latency": latency,
                        "consecutive_inference_rejects": consecutive_inference_rejects,
                        "max_consecutive_roundtrip_rejects": args.max_consecutive_roundtrip_rejects,
                        "will_retry_with_fresh_observation": will_retry,
                        "image": {
                            **_image_debug(camera_frame.rgb),
                            "capture_timestamp": camera_frame.capture_timestamp,
                            "receive_timestamp": camera_frame.receive_timestamp,
                            "sensor_skew_s": sensor_skew_s,
                        },
                        "tactile": _tactile_debug_record(
                            tactile_history,
                            capture_error=tactile_capture_error,
                            guard=tactile_guard_debug,
                        ),
                        "state": snapshot.state.astype(float).tolist(),
                        "state_debug": snapshot.debug,
                    },
                )
                if will_retry:
                    print(
                        "[lingbot-v2-client] discarded transiently failed inference "
                        f"for step={step_index} ({consecutive_inference_rejects}/"
                        f"{args.max_consecutive_roundtrip_rejects}); no action was sent, "
                        "holding position/gripper and recapturing fresh sensors",
                        flush=True,
                    )
                    continue
                raise SafetyViolation(
                    "Inference transport failed after "
                    f"{consecutive_inference_rejects} consecutive reject(s): {exc}"
                ) from exc
            roundtrip_s = time.perf_counter() - request_started
            latency = dict(latency)
            latency["outer_roundtrip_s"] = roundtrip_s
            if roundtrip_s > args.max_roundtrip_s:
                consecutive_inference_rejects += 1
                will_retry = bool(
                    args.max_consecutive_roundtrip_rejects > 0
                    and consecutive_inference_rejects < args.max_consecutive_roundtrip_rejects
                )
                server_inference_s = response.metadata.get("inference_time_s")
                non_inference_s = None
                if isinstance(server_inference_s, (int, float)) and np.isfinite(server_inference_s):
                    non_inference_s = roundtrip_s - float(server_inference_s)
                rejected_gripper_width_m = response.action_chunk[:, 7].astype(float)
                client.reset_connection()
                _append_log(
                    log_path,
                    {
                        "event": "roundtrip_rejected",
                        "timestamp": time.time(),
                        "step": step_index,
                        "request_attempt": request_attempt,
                        "request_id": observation.request_id,
                        "session_id": session_id,
                        "execute": bool(args.execute),
                        "roundtrip_s": roundtrip_s,
                        "max_roundtrip_s": args.max_roundtrip_s,
                        "roundtrip_overhead_s": non_inference_s,
                        "response_received": True,
                        "response_metadata": response.metadata,
                        "latency": latency,
                        "action_shape": list(response.action_chunk.shape),
                        "rejected_action_gripper_width_m": rejected_gripper_width_m.tolist(),
                        "rejected_action_gripper_min_m": float(rejected_gripper_width_m.min()),
                        "rejected_action_gripper_max_m": float(rejected_gripper_width_m.max()),
                        "consecutive_inference_rejects": consecutive_inference_rejects,
                        "max_consecutive_roundtrip_rejects": args.max_consecutive_roundtrip_rejects,
                        "will_retry_with_fresh_observation": will_retry,
                        "image": {
                            **_image_debug(camera_frame.rgb),
                            "capture_timestamp": camera_frame.capture_timestamp,
                            "receive_timestamp": camera_frame.receive_timestamp,
                            "sensor_skew_s": sensor_skew_s,
                        },
                        "tactile": _tactile_debug_record(
                            tactile_history,
                            capture_error=tactile_capture_error,
                            guard=tactile_guard_debug,
                        ),
                        "state": snapshot.state.astype(float).tolist(),
                        "state_debug": snapshot.debug,
                    },
                )
                detail = ""
                if isinstance(server_inference_s, (int, float)) and np.isfinite(server_inference_s):
                    detail = (
                        f" (server_inference={float(server_inference_s):.3f}s, "
                        f"non_inference={float(non_inference_s):.3f}s)"
                    )
                if will_retry:
                    print(
                        f"[lingbot-v2-client] discarded stale inference for step={step_index}: "
                        f"roundtrip={roundtrip_s:.3f}s > {args.max_roundtrip_s:.3f}s "
                        f"({consecutive_inference_rejects}/{args.max_consecutive_roundtrip_rejects}); "
                        "no action was sent, holding position/gripper and recapturing fresh sensors",
                        flush=True,
                    )
                    continue
                raise SafetyViolation(
                    f"Inference roundtrip {roundtrip_s:.3f}s exceeds "
                    f"--max-roundtrip-s {args.max_roundtrip_s:.3f}s{detail}; "
                    f"consecutive rejects={consecutive_inference_rejects}"
                )
            compensate_inference_latency = bool(
                args.execute
                and args.stream_replan
                and observation_timestamp < active_trajectory_until_s
            )
            plan = runtime.plan_action_chunk(
                response.action_chunk,
                observation_state=snapshot.state,
                observation_timestamp=observation_timestamp,
                control_frequency_hz=args.control_frequency,
                compensate_inference_latency=compensate_inference_latency,
            )
            execution = runtime.execute_plan(plan) if args.execute else None
            if (
                args.execute
                and args.stream_replan
                and execution
                and execution.get("dispatched")
                and len(plan.timestamps)
            ):
                active_trajectory_until_s = float(plan.timestamps[-1])
            verification = None
            if args.execute and not args.stream_replan and len(plan.timestamps):
                wait_until = float(plan.timestamps[-1]) + 0.05
            else:
                wait_until = loop_start + period
            _wait_until(
                wait_until,
                preview_rgb=camera_frame.rgb if args.preview else None,
                status=f"step={step_index} trajectory dispatched",
            )
            if args.execute and not args.stream_replan and execution and execution.get("dispatched"):
                verification = runtime.verify_plan_completion(
                    plan,
                    gripper_command_width_m=execution.get("gripper_command_width_m"),
                    position_tolerance_m=args.verify_position_tolerance_m,
                    rotation_tolerance_rad=args.verify_rotation_tolerance_rad,
                    gripper_tolerance_m=args.verify_gripper_tolerance_m,
                    timeout_s=args.verification_timeout_s,
                )
            record = {
                "event": "step",
                "timestamp": time.time(),
                "step": step_index,
                "request_attempt": request_attempt,
                "request_id": observation.request_id,
                "session_id": session_id,
                "execute": bool(args.execute),
                "roundtrip_s": roundtrip_s,
                "latency": latency,
                "image": {
                    **_image_debug(camera_frame.rgb),
                    "capture_timestamp": camera_frame.capture_timestamp,
                    "receive_timestamp": camera_frame.receive_timestamp,
                    "sensor_skew_s": sensor_skew_s,
                },
                "tactile": _tactile_debug_record(
                    tactile_history,
                    capture_error=tactile_capture_error,
                    guard=tactile_guard_debug,
                ),
                "state": snapshot.state.astype(float).tolist(),
                "state_debug": snapshot.debug,
                "response_metadata": response.metadata,
                "action_shape": list(response.action_chunk.shape),
                "plan": plan.debug,
                "execution": execution,
                "verification": verification,
                "recovered_after_consecutive_inference_rejects": consecutive_inference_rejects,
            }
            _append_log(log_path, record)
            print(
                f"[lingbot-v2-client] step={step_index} roundtrip={roundtrip_s:.3f}s "
                f"selected={plan.selected_indices.tolist()} execute={args.execute}",
                flush=True,
            )
            if args.preview:
                _show_preview(camera_frame.rgb, f"step={step_index} done")
            consecutive_inference_rejects = 0
            step_index += 1
    except KeyboardInterrupt as exc:
        error = repr(exc)
        if runtime.actuation_enabled:
            runtime.abort_motion()
        _append_log(log_path, {"event": "run_abort", "timestamp": time.time(), "error": error})
        raise
    except Exception as exc:
        error = repr(exc)
        if runtime.actuation_enabled:
            runtime.abort_motion()
        _append_log(log_path, {"event": "run_error", "timestamp": time.time(), "error": error})
        raise
    finally:
        _append_log(log_path, {"event": "run_end", "timestamp": time.time(), "error": error})
        if (
            args.execute
            and runtime.actuation_enabled
            and runtime.gripper is not None
            and args.gripper_exit_policy == "prompt"
        ):
            print(
                "[SAFETY] The client is about to stop Gloria and disable gripper torque. "
                "Secure/remove any held object first.",
                flush=True,
            )
            try:
                _wait_for_space(
                    camera if args.preview else None,
                    execute=False,
                    label="DISABLE GRIPPER TORQUE AND CLOSE",
                    allow_cancel=False,
                )
            except BaseException as prompt_error:
                print(
                    f"[lingbot-v2-client] shutdown confirmation interrupted ({prompt_error!r}); closing anyway",
                    flush=True,
                )
        if camera is not None:
            camera.close()
        if tactile_source is not None:
            tactile_source.close()
        if runtime is not None:
            runtime.close()
        if args.preview:
            cv2.destroyAllWindows()
        if output_dir is not None:
            print(f"[lingbot-v2-client] logs: {output_dir}", flush=True)


def _open_wrist_camera(path: Path, *, max_frame_age_s: float) -> WristCamera:
    with path.open("r") as file:
        cfg = yaml.safe_load(file) or {}
    device = str(Path(str(cfg["dev_video_path"])).expanduser().resolve(strict=False))
    width, height = [int(value) for value in cfg.get("resolution", [1280, 960])]
    fps = float(cfg.get("capture_fps") or cfg.get("record_fps") or 30.0)
    buffer_size = int(cfg.get("cap_buffer_size", 1) or 1)
    return WristCamera(
        device,
        width=width,
        height=height,
        fps=fps,
        output_size=DEFAULT_IMAGE_SIZE,
        buffer_size=buffer_size,
        max_frame_age_s=max_frame_age_s,
    )


def _open_tactile_source(
    *,
    tacthru_repo: Path,
    sensor_cfg_path: Path,
    history_steps: int,
    history_stride: int,
    include_rgb: bool,
    include_marker: bool,
) -> TacThruTactileSource:
    return TacThruTactileSource(
        tacthru_repo=tacthru_repo,
        sensor_cfg_path=sensor_cfg_path,
        history_steps=history_steps,
        history_stride=history_stride,
        include_rgb=include_rgb,
        include_marker=include_marker,
    )


def _wait_for_space(
    camera: WristCamera | None,
    *,
    execute: bool,
    label: str | None = None,
    allow_cancel: bool = True,
) -> None:
    mode = label or ("ENABLE EXECUTION" if execute else "START DRY-RUN")
    print(f"[lingbot-v2-client] Ready. Press Space to {mode}; q/Esc/Ctrl-C cancels.", flush=True)
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old_settings = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)
    if fd is None and camera is None:
        raise RuntimeError("Space confirmation requires an interactive terminal or --preview window")
    try:
        while True:
            if camera is not None:
                frame = camera.capture()
                bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
                cancel_text = " | Q/ESC: cancel" if allow_cancel else ""
                cv2.putText(bgr, f"SPACE: {mode}{cancel_text}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow("LingBot V2 TacThru UMI", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    return
                if allow_cancel and key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
            if fd is not None:
                ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                if ready:
                    key = sys.stdin.read(1)
                    if key == " ":
                        return
                    if allow_cancel and key in ("q", "Q", "\x1b", "\x03"):
                        raise KeyboardInterrupt
            elif camera is None:
                time.sleep(0.02)
    finally:
        if old_settings is not None and fd is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _show_preview(rgb: np.ndarray, status: str) -> None:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    cv2.putText(bgr, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("LingBot V2 TacThru UMI", bgr)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        raise KeyboardInterrupt


def _wait_until(deadline: float, *, preview_rgb: np.ndarray | None, status: str) -> None:
    while time.time() < deadline:
        if preview_rgb is not None:
            _show_preview(preview_rgb, status + " | q/Esc: software stop request")
        time.sleep(min(0.02, max(0.0, deadline - time.time())))


def _require_positive_cli(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value}")


def _prepare_logging(
    args: argparse.Namespace,
    *,
    resolved_paths: dict[str, Path],
    health: dict,
) -> tuple[Path | None, Path | None]:
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    log_path = args.log_jsonl.expanduser().resolve() if args.log_jsonl else None
    if output_dir is None and log_path is None:
        mode = "execute" if args.execute else "dryrun"
        output_dir = Path(__file__).resolve().parents[2] / "logs" / "realman" / time.strftime("%Y.%m.%d") / (
            time.strftime("%H.%M.%S") + f"_tacthru_umi_v2_{mode}"
        )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_path or output_dir / "events.jsonl"
        safe_args = dict(vars(args))
        safe_args["api_key"] = "<set>" if args.api_key else None
        _write_json(
            output_dir / "config.json",
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "args": safe_args,
                "resolved_paths": resolved_paths,
                "server_health": health,
            },
        )
        _write_json(output_dir / "health.json", health)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, log_path


def _parse_state(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=np.float32)
    return validate_state8(values)


def _resolve_repo_path(repo: Path, path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def _image_debug(image: np.ndarray) -> dict:
    value = np.asarray(image, dtype=np.uint8)
    return {
        "shape": list(value.shape),
        "sha1": hashlib.sha1(value.tobytes()).hexdigest(),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


def _tactile_debug_record(
    history: TactileHistory | None,
    *,
    capture_error: str | None,
    guard: dict[str, Any] | None,
) -> dict[str, Any]:
    if history is None:
        return {"present": False, "capture_error": capture_error, "guard": guard}
    record: dict[str, Any] = {
        "present": True,
        "capture_error": capture_error,
        "guard": guard,
        "debug": history.debug,
    }
    if history.rgb is not None:
        record["rgb"] = {
            "shape": list(history.rgb.shape),
            "history_mask": history.rgb_history_mask.astype(bool).tolist(),
            "timestamps": history.rgb_timestamps.astype(float).tolist(),
            "latest": _image_debug(history.rgb[-1]),
        }
    if history.marker_flow is not None:
        valid = np.asarray(history.marker_valid_mask, dtype=np.bool_)
        record["marker"] = {
            "shape": list(history.marker_flow.shape),
            "history_mask": history.marker_history_mask.astype(bool).tolist(),
            "timestamps": history.marker_timestamps.astype(float).tolist(),
            "valid_count": valid.sum(axis=1).astype(int).tolist(),
            "flow_min": float(history.marker_flow.min()),
            "flow_max": float(history.marker_flow.max()),
            "flow_mean": float(history.marker_flow.mean()),
        }
    return record


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _append_log(path: Path | None, record: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(record), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
