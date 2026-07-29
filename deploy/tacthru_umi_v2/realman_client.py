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
    ROBOT_CONFIG,
    TACTILE_ENABLED,
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


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    capture_timestamp: float
    receive_timestamp: float


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
        data, transport_timing = self._request_bytes(
            "POST",
            "/predict",
            body=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            retry_safe=False,
        )
        parse_started = time.perf_counter()
        response = action_response_from_json(
            data,
            expected_request_id=observation.request_id,
            expected_session_id=observation.session_id,
            expected_steps=expected_steps,
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
            raise RuntimeError(f"LingBot V2 server HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach LingBot V2 server at {self.server_url}: {exc}") from exc

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
                    if response.status >= 400:
                        detail = data.decode("utf-8", errors="replace")
                        raise RuntimeError(f"LingBot V2 server HTTP {response.status}: {detail}")
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
                    }
                    return data, timing
                except RuntimeError:
                    raise
                except (http.client.HTTPException, OSError) as exc:
                    attempt["error"] = f"{type(exc).__name__}: {exc}"
                    self._drop_connection()
                    if attempt_index + 1 < max_attempts:
                        continue
                    raise RuntimeError(
                        f"Cannot reach LingBot V2 server at {self.server_url}: {type(exc).__name__}: {exc}"
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

    run = subparsers.add_parser("run", help="Capture wrist RGB/state, infer, and optionally control Realman")
    _add_http_args(run, default_timeout=120.0)
    run.add_argument("--instruction", default="Pull the tissue")
    run.add_argument("--control-frequency", type=float, default=30.0)
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--rate-hz", type=float, default=1.0)
    run.add_argument("--max-roundtrip-s", type=float, default=30.0)
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
    run.add_argument("--disable-gripper", action="store_true")
    run.add_argument("--realman-ip", default=None)
    run.add_argument("--realman-port", type=int, default=None)
    run.add_argument("--assumed-gripper-width-m", type=float, default=0.045)
    run.add_argument("--gripper-min-width-m", type=float, default=0.0)
    run.add_argument("--gripper-max-width-m", type=float, default=None)
    run.add_argument("--gripper-command-torque", type=int, default=None)
    run.add_argument("--gripper-command-speed", type=int, default=None)
    run.add_argument(
        "--gripper-action-select",
        choices=["last", "first", "min", "median", "threshold"],
        default=None,
        help=(
            "Select the commanded width from the executable action window. "
            "threshold uses a non-latching binary policy: min prediction below "
            "--gripper-hold-closed-below-m closes to --gripper-hold-closed-target-m; "
            "otherwise it opens to gripper initial_width_mm."
        ),
    )
    run.add_argument(
        "--gripper-hold-closed-below-m",
        type=float,
        default=None,
        help="Threshold in metres for threshold mode; also the latch threshold for non-threshold modes.",
    )
    run.add_argument(
        "--gripper-hold-closed-target-m",
        type=float,
        default=None,
        help="Closed target width in metres.",
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
            run_synthetic(args, client, chunk_size)
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
        "robot_config": ROBOT_CONFIG,
        "pose_frame": POSE_FRAME,
        "pose_semantics": POSE_SEMANTICS,
        "camera_key": CAMERA_KEY,
        "tactile_enabled": TACTILE_ENABLED,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
    }
    for key, value in expected.items():
        if health.get(key) != value:
            raise RuntimeError(f"Server contract mismatch for {key}: expected {value!r}, got {health.get(key)!r}")
    chunk_size = int(health.get("chunk_size", 0))
    if chunk_size <= 0:
        raise RuntimeError(f"Invalid server chunk_size: {chunk_size}")
    validate_action_spec(health.get("action_spec"), expected_steps=chunk_size)
    image_shape = health.get("image_shape_hwc")
    if image_shape != [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3]:
        raise RuntimeError(f"Expected server image_shape_hwc=[224,224,3], got {image_shape!r}")
    return chunk_size


def run_synthetic(args: argparse.Namespace, client: LingBotV2HttpClient, chunk_size: int) -> None:
    state = _parse_state(args.state)
    if args.image is None:
        image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
    else:
        bgr = cv2.imread(str(args.image.expanduser().resolve()), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        image = center_crop_resize_rgb(bgr, output_size=DEFAULT_IMAGE_SIZE)
    observation = Observation(
        instruction=args.instruction,
        state=state,
        wrist_rgb=image,
        control_frequency_hz=args.control_frequency,
        metadata={"synthetic": True, "episode_reset": True},
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
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    for name in (
        "control_frequency",
        "rate_hz",
        "max_roundtrip_s",
        "max_sensor_skew_s",
        "verify_position_tolerance_m",
        "verify_rotation_tolerance_rad",
        "verify_gripper_tolerance_m",
        "verification_timeout_s",
    ):
        _require_positive_cli(getattr(args, name), f"--{name.replace('_', '-')}")
    if args.execute and (args.workspace_min_xyz is None or args.workspace_max_xyz is None):
        raise RuntimeError(
            "Real execution requires explicit --workspace-min-xyz X Y Z and "
            "--workspace-max-xyz X Y Z bounds in the Realman base frame"
        )
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
            gripper_action_select=args.gripper_action_select,
            gripper_hold_closed_below_m=args.gripper_hold_closed_below_m,
            gripper_hold_closed_target_m=args.gripper_hold_closed_target_m,
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
    output_dir, log_path = _prepare_logging(
        args,
        resolved_paths={
            "tacthru_repo": tacthru_repo,
            "camera_cfg": camera_cfg,
            "robot_cfg": robot_cfg,
            "gripper_cfg": gripper_cfg,
        },
        health=health,
    )
    session_id = uuid.uuid4().hex
    error: str | None = None
    try:
        runtime.start()
        if args.execute:
            print(
                "\n[SAFETY] --execute was supplied. Space confirmation will enable the arm trajectory "
                "and may initialize/move the gripper. Keep an emergency stop within reach.\n",
                flush=True,
            )
            _wait_for_space(camera if args.preview else None, execute=True)
            runtime.enable_actuation()
        else:
            if args.wait_for_space:
                _wait_for_space(camera if args.preview else None, execute=False)
            runtime.reset_episode_start()

        period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
        active_trajectory_until_s = 0.0
        for step_index in range(args.steps):
            loop_start = time.time()
            camera_frame = camera.capture()
            snapshot = runtime.read_policy_state()
            sensor_skew_s = abs(camera_frame.capture_timestamp - snapshot.timestamp)
            if sensor_skew_s > args.max_sensor_skew_s:
                raise SafetyViolation(
                    f"Wrist image/robot state skew {sensor_skew_s:.4f}s exceeds "
                    f"--max-sensor-skew-s {args.max_sensor_skew_s:.4f}s"
                )
            observation_timestamp = min(camera_frame.capture_timestamp, snapshot.timestamp)
            observation = Observation(
                instruction=args.instruction,
                state=snapshot.state,
                wrist_rgb=camera_frame.rgb,
                control_frequency_hz=args.control_frequency,
                session_id=session_id,
                timestamp=observation_timestamp,
                metadata={
                    "episode_reset": step_index == 0,
                    "client_step": step_index,
                    "pose_frame": POSE_FRAME,
                    "dry_run": not args.execute,
                    "camera_capture_timestamp": camera_frame.capture_timestamp,
                    "camera_receive_timestamp": camera_frame.receive_timestamp,
                    "robot_state_timestamp": snapshot.timestamp,
                    "sensor_skew_s": sensor_skew_s,
                },
            )
            if args.preview:
                _show_preview(camera_frame.rgb, f"step={step_index} sending")
            request_started = time.perf_counter()
            response, latency = client.predict_timed(observation, expected_steps=chunk_size)
            roundtrip_s = time.perf_counter() - request_started
            latency = dict(latency)
            latency["outer_roundtrip_s"] = roundtrip_s
            if roundtrip_s > args.max_roundtrip_s:
                server_inference_s = response.metadata.get("inference_time_s")
                non_inference_s = None
                if isinstance(server_inference_s, (int, float)) and np.isfinite(server_inference_s):
                    non_inference_s = roundtrip_s - float(server_inference_s)
                _append_log(
                    log_path,
                    {
                        "event": "roundtrip_rejected",
                        "timestamp": time.time(),
                        "step": step_index,
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
                        "image": {
                            **_image_debug(camera_frame.rgb),
                            "capture_timestamp": camera_frame.capture_timestamp,
                            "receive_timestamp": camera_frame.receive_timestamp,
                            "sensor_skew_s": sensor_skew_s,
                        },
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
                raise SafetyViolation(
                    f"Inference roundtrip {roundtrip_s:.3f}s exceeds "
                    f"--max-roundtrip-s {args.max_roundtrip_s:.3f}s{detail}"
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
                "state": snapshot.state.astype(float).tolist(),
                "state_debug": snapshot.debug,
                "response_metadata": response.metadata,
                "action_shape": list(response.action_chunk.shape),
                "plan": plan.debug,
                "execution": execution,
                "verification": verification,
            }
            _append_log(log_path, record)
            print(
                f"[lingbot-v2-client] step={step_index} roundtrip={roundtrip_s:.3f}s "
                f"selected={plan.selected_indices.tolist()} execute={args.execute}",
                flush=True,
            )
            if args.preview:
                _show_preview(camera_frame.rgb, f"step={step_index} done")
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
        camera.close()
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
