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
    TACTILE_MARKER_COUNT,
    TACTILE_MARKER_HISTORY_LENGTH,
    TACTILE_MARKER_SAMPLE_HZ,
    TACTILE_SENSOR_COUNT,
    ActionResponse,
    FastPredictRequest,
    Observation,
    SlowActionPlanRequest,
    SlowContextRequest,
    TactileRefineRequest,
    action_response_from_json,
    fast_predict_to_json,
    observation_to_json,
    slow_action_plan_to_json,
    slow_context_to_json,
    tactile_refine_to_json,
    validate_action_spec,
)
from .realman_runtime import RealmanConfig, RealmanEpisodeRuntime, SafetyViolation
from .tactile_source import TactileFrame, TacThruSource
from .transforms import validate_action_chunk, validate_state8


DEFAULT_IMAGE_SIZE = 224
CAMERA_FRESHNESS_FRACTION = 0.80
PREVIEW_WINDOW_NAME = "LingBot V2 TacThru UMI"
PREVIEW_ARROW_SCALE = 6.0
_PREVIEW_WINDOW_CREATED = False


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

    def refresh_slow_context(self, request: SlowContextRequest) -> dict[str, Any]:
        body = slow_context_to_json(request, jpeg_quality=self.jpeg_quality)
        data, timing = self._request_bytes(
            "POST",
            "/context/refresh",
            body=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            retry_safe=False,
        )
        payload = json.loads(data.decode("utf-8"))
        if payload.get("request_id") != request.request_id:
            raise ValueError("Slow context response request_id mismatch")
        if payload.get("session_id") != request.session_id:
            raise ValueError("Slow context response session_id mismatch")
        self.last_request_timing = timing
        return payload

    def predict_fast_timed(
        self,
        request: FastPredictRequest,
        *,
        expected_steps: int,
    ) -> tuple[ActionResponse, dict[str, Any]]:
        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        body = fast_predict_to_json(request)
        encode_s = time.perf_counter() - encode_started
        network_started = time.perf_counter()
        data, transport_timing = self._request_bytes(
            "POST",
            "/predict_fast",
            body=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            retry_safe=False,
        )
        request_response_ms = (time.perf_counter() - network_started) * 1000.0
        response = action_response_from_json(
            data,
            expected_request_id=request.request_id,
            expected_session_id=request.session_id,
            expected_steps=expected_steps,
        )
        server_total_ms = float(response.metadata.get("server_total_ms", 0.0))
        network_ms = max(0.0, request_response_ms - server_total_ms)
        timing = {
            "http_encode_ms": encode_s * 1000.0,
            **transport_timing,
            "network_ms": network_ms,
            "total_s": time.perf_counter() - total_started,
        }
        self.last_request_timing = timing
        return response, timing

    def build_slow_action_plan(self, request: SlowActionPlanRequest) -> dict[str, Any]:
        body = slow_action_plan_to_json(request)
        data, timing = self._request_bytes(
            "POST",
            "/action/plan",
            body=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            retry_safe=False,
        )
        payload = json.loads(data.decode("utf-8"))
        if payload.get("request_id") != request.request_id:
            raise ValueError("Slow action plan response request_id mismatch")
        if payload.get("session_id") != request.session_id:
            raise ValueError("Slow action plan response session_id mismatch")
        self.last_request_timing = timing
        return payload

    def refine_action_timed(
        self,
        request: TactileRefineRequest,
        *,
        expected_steps: int,
    ) -> tuple[ActionResponse, dict[str, Any]]:
        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        body = tactile_refine_to_json(request)
        encode_s = time.perf_counter() - encode_started
        data, transport_timing = self._request_bytes(
            "POST",
            "/action/refine",
            body=body,
            headers={**self._headers(), "Content-Type": "application/json"},
            retry_safe=False,
        )
        response = action_response_from_json(
            data,
            expected_request_id=request.request_id,
            expected_session_id=request.session_id,
            expected_steps=expected_steps,
        )
        timing = {
            "http_encode_ms": encode_s * 1000.0,
            **transport_timing,
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

    def reset_connection(self) -> None:
        """Drop an idle keep-alive socket without closing the client."""

        with self._connection_lock:
            if self._closed:
                raise RuntimeError("LingBot V2 HTTP client is closed")
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


class LingBotV2InProcessClient:
    """Direct policy adapter that bypasses HTTP, JPEG and JSON serialization."""

    def __init__(self, backend, *, warmup_result: dict[str, Any] | None = None) -> None:
        self.backend = backend
        self.warmup_result = dict(warmup_result or {})
        self.last_request_timing: dict[str, Any] = {}
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        *,
        project_root: Path,
        checkpoint: Path,
        norm_stats: Path,
        qwen_path: Path | None,
        use_compile: bool,
        dtype: str,
        inference_lock_timeout_s: float,
        warmup: bool,
    ) -> "LingBotV2InProcessClient":
        # Keep model-only imports out of the existing lightweight HTTP client path.
        from .http_server import LingBotV2Backend

        backend = LingBotV2Backend.from_checkpoint(
            project_root=project_root,
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            qwen_path=qwen_path,
            use_compile=use_compile,
            dtype=dtype,
            inference_lock_timeout_s=inference_lock_timeout_s,
        )
        warmup_result = backend.warmup() if warmup else None
        return cls(backend, warmup_result=warmup_result)

    def health(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        payload = dict(self.backend.health())
        payload["backend"] = "lingbot-vla-v2-tacthru-umi-inprocess"
        payload["transport"] = {
            "mode": "inprocess",
            "http": False,
            "image_encoding": "raw_rgb_uint8",
            "serialization": "none",
        }
        if self.warmup_result:
            payload["warmup"] = dict(self.warmup_result)
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
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        total_started = time.perf_counter()
        backend_started = time.perf_counter()
        response = self.backend.predict_response(observation)
        backend_call_s = time.perf_counter() - backend_started
        if response.request_id != observation.request_id:
            raise ValueError(
                "Response request_id mismatch: "
                f"expected {observation.request_id!r}, got {response.request_id!r}"
            )
        if response.session_id != observation.session_id:
            raise ValueError(
                "Response session_id mismatch: "
                f"expected {observation.session_id!r}, got {response.session_id!r}"
            )
        actions = validate_action_chunk(response.action_chunk, expected_steps=expected_steps)
        metadata = dict(response.metadata)
        metadata["backend"] = "lingbot-vla-v2-tacthru-umi-inprocess"
        metadata["transport"] = "inprocess_direct"
        response = ActionResponse(
            action_chunk=actions,
            request_id=response.request_id,
            session_id=response.session_id,
            metadata=metadata,
            server_timestamp=response.server_timestamp,
        )
        timing = {
            "transport": "inprocess",
            "encode_s": 0.0,
            "request_response_s": backend_call_s,
            "response_parse_s": 0.0,
            "server_timing_s": metadata.get("server_timing_s", {}),
            "total_s": time.perf_counter() - total_started,
        }
        self.last_request_timing = timing
        return response, timing

    def refresh_slow_context(self, request: SlowContextRequest) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        return self.backend.refresh_slow_context(request)

    def predict_fast_timed(
        self,
        request: FastPredictRequest,
        *,
        expected_steps: int,
    ) -> tuple[ActionResponse, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        started = time.perf_counter()
        payload = self.backend.predict_fast(request)
        response = action_response_from_json(
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            expected_request_id=request.request_id,
            expected_session_id=request.session_id,
            expected_steps=expected_steps,
        )
        timing = {
            "transport": "inprocess",
            "http_encode_ms": 0.0,
            "network_ms": 0.0,
            "total_s": time.perf_counter() - started,
        }
        self.last_request_timing = timing
        return response, timing

    def build_slow_action_plan(self, request: SlowActionPlanRequest) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        return self.backend.build_slow_action_plan(request)

    def refine_action_timed(
        self,
        request: TactileRefineRequest,
        *,
        expected_steps: int,
    ) -> tuple[ActionResponse, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("LingBot V2 in-process client is closed")
        started = time.perf_counter()
        payload = self.backend.refine_action_with_tactile(request)
        response = action_response_from_json(
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            expected_request_id=request.request_id,
            expected_session_id=request.session_id,
            expected_steps=expected_steps,
        )
        timing = {
            "transport": "inprocess",
            "http_encode_ms": 0.0,
            "network_ms": 0.0,
            "total_s": time.perf_counter() - started,
        }
        self.last_request_timing = timing
        return response, timing

    def reset_connection(self) -> None:
        """Match the HTTP client's recovery interface; there is no connection to reset."""

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "LingBotV2InProcessClient":
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

    run = subparsers.add_parser(
        "run",
        help="Capture wrist RGB/state and checkpoint-required TacThru inputs, then infer/control Realman",
    )
    _add_http_args(run, default_timeout=120.0)
    run.add_argument("--instruction", default="Pull the tissue")
    run.add_argument("--control-frequency", type=float, default=30.0)
    run.add_argument("--steps", type=int, default=1)
    run.add_argument("--rate-hz", type=float, default=1.0)
    run.add_argument("--max-roundtrip-s", type=float, default=30.0)
    run.add_argument("--execute", action="store_true", help="Enable real actuator commands after a mandatory Space confirmation")
    run.add_argument("--stream-replan", action="store_true", help="Do not wait for the dispatched short chunk to finish")
    run.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Open a local GUI with the wrist RGB and the exact TacThru RGB/marker "
            "observation sent for inference"
        ),
    )
    run.add_argument("--wait-for-space", action="store_true", help="Also gate a dry-run on Space; execute mode is always gated")
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--log-jsonl", type=Path, default=None)
    run.add_argument("--tacthru-repo", type=Path, default=Path("/mnt/models/VTLA-RDT/tacthru"))
    run.add_argument("--camera-cfg", type=Path, default=Path("cfg/camera/synria_c10.yaml"))
    run.add_argument("--tactile-sensor-cfg", type=Path, default=Path("cfg/sensor/ml.yaml"))
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
        "--gripper-startup-width-m",
        type=float,
        default=None,
        help="Enable two-stage execution: first Space grips to this width; second Space starts inference.",
    )
    run.add_argument("--gripper-startup-tolerance-m", type=float, default=0.005)
    run.add_argument("--gripper-startup-timeout-s", type=float, default=3.0)
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
    run.add_argument("--exec-end-step", type=int, default=None)
    run.add_argument(
        "--slow-fast-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use server-side RGB slow context plus marker-only fast replanning when supported.",
    )
    run.add_argument(
        "--slow-refresh-every",
        type=int,
        default=0,
        help="Refresh RGB slow context every N replans; 0 means episode start only.",
    )
    run.add_argument(
        "--fixed-exec-window",
        action="store_true",
        help="Always select [exec-start-step, exec-end-step), without online-delay index shifting.",
    )
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
    run.add_argument("--max-tactile-skew-s", type=float, default=0.20)
    run.add_argument("--max-tactile-age-s", type=float, default=0.25)
    run.add_argument("--min-valid-markers", type=int, default=40)
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
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_default = os.environ.get("LINGBOT_V2_CHECKPOINT")
    norm_stats_default = os.environ.get(
        "LINGBOT_V2_NORM_STATS",
        "assets/norm_stats/tacthru_umi_v2.json",
    )
    qwen_default = os.environ.get("QWEN3VL_PATH", "models/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--transport",
        choices=["http", "inprocess"],
        default="http",
        help="HTTP preserves the existing split deployment; inprocess loads the policy here with no serialization.",
    )
    parser.add_argument("--server-url", default=os.environ.get("LINGBOT_V2_SERVER_URL"))
    parser.add_argument("--timeout", type=float, default=default_timeout)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--api-key", default=os.environ.get("LINGBOT_V2_API_KEY"))
    parser.add_argument(
        "--http-keep-alive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one HTTP/1.1 connection; --no-http-keep-alive restores urllib Connection: close.",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(checkpoint_default) if checkpoint_default else None,
        help="In-process only: final .../global_step_N/hf_ckpt directory.",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path(norm_stats_default),
        help="In-process only: normalization statistics used by the checkpoint.",
    )
    parser.add_argument(
        "--qwen-path",
        type=Path,
        default=Path(qwen_default),
        help="In-process only: local Qwen3-VL model directory.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--use-compile", action="store_true", help="In-process only: enable torch.compile.")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In-process only: warm up once before cameras and robot hardware are opened.",
    )
    parser.add_argument("--inference-lock-timeout", type=float, default=5.0)


def _build_inference_client(
    args: argparse.Namespace,
) -> LingBotV2HttpClient | LingBotV2InProcessClient:
    if args.transport == "http":
        if not args.server_url:
            raise ValueError("--server-url is required when --transport=http")
        return LingBotV2HttpClient(
            args.server_url,
            timeout_s=args.timeout,
            jpeg_quality=args.jpeg_quality,
            api_key=args.api_key,
            keep_alive=args.http_keep_alive,
        )

    project_root = args.project_root.expanduser().resolve()
    if args.checkpoint is None:
        raise ValueError(
            "--checkpoint or LINGBOT_V2_CHECKPOINT is required when --transport=inprocess"
        )
    checkpoint = _resolve_repo_path(project_root, args.checkpoint)
    norm_stats = _resolve_repo_path(project_root, args.norm_stats)
    qwen_path = _resolve_repo_path(project_root, args.qwen_path)
    print(
        "[lingbot-v2-client] loading in-process policy; HTTP/JPEG/JSON transport is disabled",
        flush=True,
    )
    client = LingBotV2InProcessClient.from_checkpoint(
        project_root=project_root,
        checkpoint=checkpoint,
        norm_stats=norm_stats,
        qwen_path=qwen_path,
        use_compile=args.use_compile,
        dtype=args.dtype,
        inference_lock_timeout_s=args.inference_lock_timeout,
        warmup=args.warmup,
    )
    print(
        f"[lingbot-v2-client] in-process policy ready warmup={args.warmup}",
        flush=True,
    )
    return client


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    client = _build_inference_client(args)
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
            run_synthetic(args, client, chunk_size, health)
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
    _tactile_contract_from_health(health)
    return chunk_size


def _tactile_contract_from_health(health: dict) -> dict[str, Any]:
    value = health.get("tactile")
    if not isinstance(value, dict):
        raise RuntimeError(f"Server health is missing the tactile contract: {value!r}")
    enabled = value.get("enabled")
    use_rgb = value.get("use_rgb")
    use_markers = value.get("use_markers")
    for name, setting in (
        ("enabled", enabled),
        ("use_rgb", use_rgb),
        ("use_markers", use_markers),
    ):
        if not isinstance(setting, bool):
            raise RuntimeError(f"Server tactile.{name} must be boolean, got {setting!r}")
    expected_enabled = bool(health.get("tactile_enabled"))
    if enabled != expected_enabled:
        raise RuntimeError(
            "Server tactile health is inconsistent: "
            f"tactile_enabled={expected_enabled}, tactile.enabled={enabled}"
        )
    num_sensors = int(value.get("num_sensors", -1))
    num_markers = int(value.get("num_markers", -1))
    marker_history_length = int(value.get("marker_history_length", 0))
    marker_sample_hz = float(value.get("marker_sample_hz", 0.0))
    if enabled:
        if num_sensors != TACTILE_SENSOR_COUNT or num_markers != TACTILE_MARKER_COUNT:
            raise RuntimeError(
                "Unsupported tactile shape: "
                f"sensors={num_sensors}, markers={num_markers}"
            )
        if not (use_rgb or use_markers):
            raise RuntimeError("Enabled tactile checkpoint has no active modality")
        if use_markers:
            if marker_history_length <= 0:
                raise RuntimeError(
                    "Marker history length must be positive: "
                    f"{marker_history_length}"
                )
            if not np.isclose(
                marker_sample_hz,
                TACTILE_MARKER_SAMPLE_HZ,
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Unsupported marker sample rate: {marker_sample_hz}"
                )
    elif any(
        (
            num_sensors != 0,
            num_markers != 0,
            use_rgb,
            use_markers,
            marker_history_length != 0,
            marker_sample_hz != 0.0,
        )
    ):
        raise RuntimeError(f"Disabled tactile checkpoint has inconsistent contract: {value}")
    return {
        "enabled": enabled,
        "num_sensors": num_sensors,
        "num_markers": num_markers,
        "use_rgb": use_rgb,
        "use_markers": use_markers,
        "marker_history_length": marker_history_length,
        "marker_sample_hz": marker_sample_hz,
        "marker_tokenization": value.get("marker_tokenization", {}),
        "marker_position_encoding": value.get("marker_position_encoding", {}),
        "marker_contact_gate": value.get("marker_contact_gate", {}),
        "marker_ablation": value.get("marker_ablation", {}),
        "marker_reference_xy_path": value.get("marker_reference_xy_path"),
        "marker_reference_xy": value.get("marker_reference_xy", []),
    }


def _synthetic_tactile_inputs(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    if not contract["enabled"]:
        return {}
    result: dict[str, np.ndarray] = {
        "tactile_sensor_mask": np.ones((TACTILE_SENSOR_COUNT,), dtype=np.bool_),
    }
    if contract["use_rgb"]:
        result["tactile_rgb"] = np.zeros(
            (TACTILE_SENSOR_COUNT, 480, 640, 3), dtype=np.uint8
        )
    if contract["use_markers"]:
        marker_shape = (
            TACTILE_SENSOR_COUNT,
            int(contract["marker_history_length"]),
            TACTILE_MARKER_COUNT,
            2,
        )
        result.update(
            marker_displacement_history=np.zeros(marker_shape, dtype=np.float32),
            marker_valid_mask=np.ones(marker_shape[:-1], dtype=np.bool_),
            marker_history_valid_mask=np.ones(marker_shape[:2], dtype=np.bool_),
        )
        if contract.get("marker_contact_gate", {}).get("mode", "none") != "none":
            result["marker_contact_state"] = np.zeros(
                (TACTILE_SENSOR_COUNT,), dtype=np.int8
            )
    return result


def run_synthetic(
    args: argparse.Namespace,
    client: LingBotV2HttpClient | LingBotV2InProcessClient,
    chunk_size: int,
    health: dict,
) -> None:
    state = _parse_state(args.state)
    if args.image is None:
        image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
    else:
        bgr = cv2.imread(str(args.image.expanduser().resolve()), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        image = center_crop_resize_rgb(bgr, output_size=DEFAULT_IMAGE_SIZE)
    tactile = _synthetic_tactile_inputs(_tactile_contract_from_health(health))
    observation = Observation(
        instruction=args.instruction,
        state=state,
        wrist_rgb=image,
        **tactile,
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
    client: LingBotV2HttpClient | LingBotV2InProcessClient,
    health: dict,
    chunk_size: int,
) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.slow_refresh_every < 0:
        raise ValueError("--slow-refresh-every must be non-negative")
    for name in (
        "control_frequency",
        "rate_hz",
        "max_roundtrip_s",
        "max_sensor_skew_s",
        "max_tactile_skew_s",
        "max_tactile_age_s",
        "gripper_startup_tolerance_m",
        "gripper_startup_timeout_s",
        "verify_position_tolerance_m",
        "verify_rotation_tolerance_rad",
        "verify_gripper_tolerance_m",
        "verification_timeout_s",
    ):
        _require_positive_cli(getattr(args, name), f"--{name.replace('_', '-')}")
    tactile_contract = _tactile_contract_from_health(health)
    cache_health = health.get("slow_fast_cache", {})
    use_slow_fast = bool(
        args.slow_fast_cache
        and isinstance(cache_health, dict)
        and cache_health.get("supported") is True
    )
    cascaded_health = health.get("cascaded_tactile_flow", {})
    use_cascaded = bool(
        use_slow_fast
        and isinstance(cascaded_health, dict)
        and cascaded_health.get("supported") is True
    )
    if args.slow_fast_cache and not use_slow_fast:
        print(
            "[lingbot-v2-client] slow/fast cache is unavailable; falling back to /predict",
            flush=True,
        )
    effective_exec_end = args.exec_end_step
    if effective_exec_end is None:
        effective_exec_end = 4 if use_slow_fast else 8
    exec_window_size = int(effective_exec_end) - int(args.exec_start_step)
    if exec_window_size <= 0:
        raise ValueError("Execution window must contain at least one action")
    if use_cascaded:
        print(
            "[lingbot-v2-client] cascaded tactile flow enabled: "
            "one immutable slow plan will be refined across advancing action windows",
            flush=True,
        )
    if not 0 <= args.min_valid_markers <= TACTILE_MARKER_COUNT:
        raise ValueError(
            f"--min-valid-markers must be in [0,{TACTILE_MARKER_COUNT}], "
            f"got {args.min_valid_markers}"
        )
    if args.gripper_startup_width_m is not None:
        if args.disable_gripper:
            raise RuntimeError("--gripper-startup-width-m cannot be used with --disable-gripper")
        if (
            not np.isfinite(args.gripper_startup_width_m)
            or args.gripper_startup_width_m < args.gripper_min_width_m
        ):
            raise ValueError("--gripper-startup-width-m must be finite and at least --gripper-min-width-m")
    if args.execute and (args.workspace_min_xyz is None or args.workspace_max_xyz is None):
        raise RuntimeError(
            "Real execution requires explicit --workspace-min-xyz X Y Z and "
            "--workspace-max-xyz X Y Z bounds in the Realman base frame"
        )
    if args.execute and args.transport == "http":
        parsed_server_url = urlparse(args.server_url)
        if parsed_server_url.scheme != "https" and parsed_server_url.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise RuntimeError(
                "Real execution over plaintext HTTP is allowed only through a localhost SSH tunnel. "
                "Use --server-url http://127.0.0.1:<port> or terminate TLS with HTTPS."
            )
    if args.preview:
        _require_preview_gui()
    reset_connection = getattr(client, "reset_connection", None)
    if callable(reset_connection):
        # The health request runs before camera/gripper setup and operator
        # confirmation. Discard that socket so the first prediction cannot
        # reuse a server-side keep-alive connection that expired meanwhile.
        reset_connection()
    tacthru_repo = args.tacthru_repo.expanduser().resolve()
    camera_cfg = _resolve_repo_path(tacthru_repo, args.camera_cfg)
    tactile_sensor_cfg = _resolve_repo_path(tacthru_repo, args.tactile_sensor_cfg)
    robot_cfg = _resolve_repo_path(tacthru_repo, args.robot_cfg)
    gripper_cfg = _resolve_repo_path(tacthru_repo, args.gripper_cfg)
    camera = _open_wrist_camera(
        camera_cfg,
        max_frame_age_s=args.max_sensor_skew_s * CAMERA_FRESHNESS_FRACTION,
    )
    tactile_source: TacThruSource | None = None
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
            gripper_initialize_on_start=(
                False if args.gripper_startup_width_m is not None else None
            ),
            gripper_episode_start_width_m=args.gripper_startup_width_m,
            gripper_action_select=args.gripper_action_select,
            gripper_hold_closed_below_m=args.gripper_hold_closed_below_m,
            gripper_hold_closed_target_m=args.gripper_hold_closed_target_m,
            exec_start_step=args.exec_start_step,
            exec_end_step=effective_exec_end,
            fixed_exec_window=args.fixed_exec_window,
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
            "tactile_sensor_cfg": tactile_sensor_cfg,
            "robot_cfg": robot_cfg,
            "gripper_cfg": gripper_cfg,
        },
        health=health,
    )
    session_id = uuid.uuid4().hex
    context_version: int | None = None
    slow_plan_version: int | None = None
    action_offset = int(args.exec_start_step)
    error: str | None = None
    try:
        if tactile_contract["enabled"]:
            tactile_source = TacThruSource(
                tacthru_repo=tacthru_repo,
                sensor_cfg_path=tactile_sensor_cfg,
                use_rgb=tactile_contract["use_rgb"],
                use_markers=tactile_contract["use_markers"],
                tactile_config=tactile_contract,
            )
            tactile_source.start()
            print(
                "[lingbot-v2-client] TacThru VTLA input ready: "
                f"rgb={tactile_contract['use_rgb']} markers={tactile_contract['use_markers']}",
                flush=True,
            )
        runtime.start()
        if args.execute:
            startup_result = _prepare_real_execution(
                args,
                runtime,
                camera if args.preview else None,
            )
            if startup_result is not None:
                _append_log(
                    log_path,
                    {"event": "gripper_startup", "timestamp": time.time(), **startup_result},
                )
        else:
            if args.wait_for_space:
                _wait_for_space(camera if args.preview else None, execute=False)
            runtime.reset_episode_start()

        period = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
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
            tactile_frame = None
            tactile_inputs: dict[str, np.ndarray] = {}
            tactile_skew_s = None
            tactile_age_s = None
            if tactile_source is not None:
                tactile_frame = tactile_source.capture(episode_reset=step_index == 0)
                tactile_age_s = time.time() - tactile_frame.capture_timestamp
                if tactile_age_s < 0.0 or tactile_age_s > args.max_tactile_age_s:
                    raise SafetyViolation(
                        f"TacThru frame age {tactile_age_s:.4f}s is outside "
                        f"[0,{args.max_tactile_age_s:.4f}]s"
                    )
                tactile_skew_s = max(
                    abs(tactile_frame.capture_timestamp - camera_frame.capture_timestamp),
                    abs(tactile_frame.capture_timestamp - snapshot.timestamp),
                )
                if tactile_skew_s > args.max_tactile_skew_s:
                    raise SafetyViolation(
                        f"TacThru/wrist/robot skew {tactile_skew_s:.4f}s exceeds "
                        f"--max-tactile-skew-s {args.max_tactile_skew_s:.4f}s"
                    )
                if tactile_contract["use_markers"]:
                    valid_count = int(tactile_frame.marker_valid_mask[0, -1].sum())
                    if valid_count < args.min_valid_markers:
                        raise SafetyViolation(
                            f"TacThru valid markers {valid_count} below "
                            f"--min-valid-markers {args.min_valid_markers}"
                        )
                tactile_inputs = {
                    "tactile_rgb": tactile_frame.tactile_rgb,
                    "marker_displacement_history": tactile_frame.marker_displacement_history,
                    "marker_valid_mask": tactile_frame.marker_valid_mask,
                    "marker_history_valid_mask": tactile_frame.marker_history_valid_mask,
                    "marker_contact_state": tactile_frame.marker_contact_state,
                    "tactile_sensor_mask": tactile_frame.tactile_sensor_mask,
                }
                tactile_inputs = {
                    key: value for key, value in tactile_inputs.items() if value is not None
                }
            observation_timestamps = [camera_frame.capture_timestamp, snapshot.timestamp]
            if tactile_frame is not None:
                observation_timestamps.append(tactile_frame.capture_timestamp)
            observation_timestamp = min(observation_timestamps)
            observation = Observation(
                instruction=args.instruction,
                state=snapshot.state,
                wrist_rgb=camera_frame.rgb,
                **tactile_inputs,
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
                    "tactile_capture_timestamp": (
                        tactile_frame.capture_timestamp if tactile_frame is not None else None
                    ),
                    "tactile_receive_timestamp": (
                        tactile_frame.receive_timestamp if tactile_frame is not None else None
                    ),
                    "tactile_age_s": tactile_age_s,
                    "tactile_skew_s": tactile_skew_s,
                    "tactile": tactile_frame.debug if tactile_frame is not None else None,
                },
            )
            if args.preview:
                _show_preview(
                    camera_frame.rgb,
                    f"step={step_index} sending",
                    tactile_frame=tactile_frame,
                )
            if use_slow_fast:
                if tactile_frame is None or tactile_frame.tactile_rgb is None:
                    raise SafetyViolation("Slow/fast VTLA cache requires current tactile RGB")
                refresh_due = bool(
                    context_version is None
                    or (
                        args.slow_refresh_every > 0
                        and step_index > 0
                        and step_index % args.slow_refresh_every == 0
                    )
                )
                if refresh_due:
                    refresh = SlowContextRequest(
                        instruction=args.instruction,
                        wrist_rgb=camera_frame.rgb,
                        tactile_rgb=tactile_frame.tactile_rgb,
                        tactile_sensor_mask=tactile_frame.tactile_sensor_mask,
                        session_id=session_id,
                        scene_timestamp=camera_frame.capture_timestamp,
                        tactile_rgb_timestamp=tactile_frame.capture_timestamp,
                        metadata={"episode_reset": step_index == 0},
                    )
                    refresh_response = client.refresh_slow_context(refresh)
                    context_version = int(refresh_response["context_version"])
                    slow_plan_version = None
                    action_offset = int(args.exec_start_step)
                    _append_log(
                        log_path,
                        {
                            "event": "slow_context_refreshed",
                            "timestamp": time.time(),
                            "step": step_index,
                            **refresh_response,
                        },
                    )
                request_started = time.perf_counter()
                if use_cascaded:
                    if slow_plan_version is None or action_offset >= chunk_size:
                        action_offset = int(args.exec_start_step)
                        plan_response = client.build_slow_action_plan(
                            SlowActionPlanRequest(
                                state=snapshot.state,
                                session_id=session_id,
                                context_version=int(context_version),
                                action_offset=action_offset,
                                timestamp=snapshot.timestamp,
                                metadata={"client_step": step_index},
                            )
                        )
                        slow_plan_version = int(plan_response["plan_version"])
                        _append_log(
                            log_path,
                            {
                                "event": "slow_action_plan_built",
                                "timestamp": time.time(),
                                "step": step_index,
                                **plan_response,
                            },
                        )
                    refine_request = TactileRefineRequest(
                        state=snapshot.state,
                        marker_displacement_history=(
                            tactile_frame.marker_displacement_history
                        ),
                        marker_valid_mask=tactile_frame.marker_valid_mask,
                        marker_history_valid_mask=(
                            tactile_frame.marker_history_valid_mask
                        ),
                        marker_contact_state=tactile_frame.marker_contact_state,
                        tactile_sensor_mask=tactile_frame.tactile_sensor_mask,
                        session_id=session_id,
                        context_version=int(context_version),
                        plan_version=int(slow_plan_version),
                        action_offset=action_offset,
                        marker_timestamp=tactile_frame.capture_timestamp,
                        control_frequency_hz=args.control_frequency,
                        metadata={"client_step": step_index},
                    )
                    response, latency = client.refine_action_timed(
                        refine_request, expected_steps=chunk_size
                    )
                else:
                    fast_request = FastPredictRequest(
                        state=snapshot.state,
                        marker_displacement_history=(
                            tactile_frame.marker_displacement_history
                        ),
                        marker_valid_mask=tactile_frame.marker_valid_mask,
                        marker_history_valid_mask=(
                            tactile_frame.marker_history_valid_mask
                        ),
                        marker_contact_state=tactile_frame.marker_contact_state,
                        tactile_sensor_mask=tactile_frame.tactile_sensor_mask,
                        session_id=session_id,
                        context_version=context_version,
                        marker_timestamp=tactile_frame.capture_timestamp,
                        control_frequency_hz=args.control_frequency,
                        metadata={"client_step": step_index},
                    )
                    response, latency = client.predict_fast_timed(
                        fast_request, expected_steps=chunk_size
                    )
            else:
                request_started = time.perf_counter()
                response, latency = client.predict_timed(
                    observation, expected_steps=chunk_size
                )
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
                        "request_id": response.request_id,
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
            plan = runtime.plan_action_chunk(
                response.action_chunk,
                observation_state=snapshot.state,
                observation_timestamp=observation_timestamp,
                control_frequency_hz=args.control_frequency,
                exec_start_step=action_offset if use_cascaded else None,
                exec_end_step=(
                    min(action_offset + exec_window_size, chunk_size)
                    if use_cascaded
                    else None
                ),
            )
            if use_cascaded and len(plan.selected_indices):
                action_offset = int(plan.selected_indices[-1]) + 1
            execution_started = time.perf_counter()
            execution = runtime.execute_plan(plan) if args.execute else None
            verification = None
            if args.execute and not args.stream_replan and len(plan.timestamps):
                wait_until = float(plan.timestamps[-1]) + 0.05
            else:
                wait_until = loop_start + period
            _wait_until(
                wait_until,
                preview_rgb=camera_frame.rgb if args.preview else None,
                status=f"step={step_index} trajectory dispatched",
                preview_tactile_frame=tactile_frame,
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
            robot_execution_ms = (time.perf_counter() - execution_started) * 1000.0
            record = {
                "event": "step",
                "timestamp": time.time(),
                "step": step_index,
                "request_id": response.request_id,
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
                "tactile": tactile_frame.debug if tactile_frame is not None else None,
                "state": snapshot.state.astype(float).tolist(),
                "state_debug": snapshot.debug,
                "response_metadata": response.metadata,
                "action_shape": list(response.action_chunk.shape),
                "plan": plan.debug,
                "execution": execution,
                "robot_execution_ms": robot_execution_ms,
                "verification": verification,
            }
            _append_log(log_path, record)
            print(
                f"[lingbot-v2-client] step={step_index} roundtrip={roundtrip_s:.3f}s "
                f"selected={plan.selected_indices.tolist()} execute={args.execute}",
                flush=True,
            )
            if args.preview:
                _show_preview(
                    camera_frame.rgb,
                    f"step={step_index} done",
                    tactile_frame=tactile_frame,
                )
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
        if tactile_source is not None:
            tactile_source.close()
        camera.close()
        runtime.close()
        if args.preview:
            _close_preview()
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
                cancel_text = " | Q/ESC: cancel" if allow_cancel else ""
                key = _show_preview(
                    frame.rgb,
                    f"SPACE: {mode}{cancel_text}",
                    return_key=True,
                )
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


def _prepare_real_execution(
    args: argparse.Namespace,
    runtime: RealmanEpisodeRuntime,
    camera: WristCamera | None,
) -> dict[str, float] | None:
    startup_width_m = args.gripper_startup_width_m
    if startup_width_m is None:
        print(
            "\n[SAFETY] --execute was supplied. Space confirmation will enable the arm trajectory "
            "and may initialize/move the gripper. Keep an emergency stop within reach.\n",
            flush=True,
        )
        _wait_for_space(camera, execute=True)
        runtime.enable_actuation()
        return None

    print(
        "\n[SAFETY] Two-stage start is enabled. The first Space only closes Gloria; "
        "the arm remains disabled. The second Space enables arm execution and inference. "
        "Keep an emergency stop within reach.\n",
        flush=True,
    )
    _wait_for_space(
        camera,
        execute=False,
        label=f"CLOSE GRIPPER TO {startup_width_m * 1000.0:.1f} mm (ARM DISABLED)",
    )
    startup_result = runtime.prepare_gripper_for_episode(
        startup_width_m,
        tolerance_m=args.gripper_startup_tolerance_m,
        timeout_s=args.gripper_startup_timeout_s,
    )
    _wait_for_space(
        camera,
        execute=True,
        label="START INFERENCE AND ENABLE ARM EXECUTION",
    )
    runtime.enable_actuation()
    return startup_result


def _require_preview_gui() -> None:
    gui_backend = "unknown"
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            gui_backend = stripped.split(":", 1)[1].strip()
            break
    if gui_backend.upper() in {"", "NONE"}:
        raise RuntimeError(
            "--preview requires an OpenCV build with GUI support; the selected "
            f"Python reports GUI={gui_backend!r}. Use the TacThru .venv Python "
            "with QT5 support, or run without LINGBOT_V2_PREVIEW=1."
        )
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "--preview requires a graphical desktop session. Run the client from "
            "the 30.133 desktop terminal, or set DISPLAY and XAUTHORITY for that session."
        )


def _gate_preview_status(tactile_frame: TactileFrame | None) -> tuple[str, tuple[int, int, int]]:
    if tactile_frame is None or tactile_frame.marker_contact_state is None:
        return "GATE N/A", (170, 170, 170)
    states = np.asarray(tactile_frame.marker_contact_state).reshape(-1)
    if len(states) == 0:
        return "GATE N/A", (170, 170, 170)
    state = int(states[0])
    labels = {
        0: ("NO CONTACT", (70, 80, 235)),
        1: ("CONTACT", (65, 205, 80)),
        2: ("CONTACT HOLD", (40, 210, 235)),
        3: ("NO CONTACT / TRACKING UNKNOWN", (70, 145, 235)),
        4: ("CONTACT / TRACKING UNKNOWN", (70, 190, 235)),
        5: ("CONTACT HOLD / TRACKING UNKNOWN", (70, 190, 235)),
    }
    return labels.get(state, (f"GATE STATE {state}", (170, 170, 170)))


def _build_preview_canvas(
    wrist_rgb: np.ndarray,
    status: str,
    *,
    tactile_frame: TactileFrame | None = None,
    arrow_scale: float = PREVIEW_ARROW_SCALE,
) -> np.ndarray:
    wrist = np.asarray(wrist_rgb, dtype=np.uint8)
    if wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"Preview wrist RGB must have shape [H,W,3], got {wrist.shape}")
    if not np.isfinite(arrow_scale) or arrow_scale <= 0.0:
        raise ValueError(f"Preview arrow_scale must be positive and finite, got {arrow_scale}")

    panel_height = 480
    tactile_width = 640
    wrist_width = 480
    header_height = 64
    gap = 2
    tactile_panel = np.full(
        (panel_height, tactile_width, 3), (24, 24, 24), dtype=np.uint8
    )
    valid_count = 0
    marker_count = 0
    detected_count = None
    mean_displacement = 0.0
    max_displacement = 0.0

    tactile_rgb = None if tactile_frame is None else tactile_frame.tactile_rgb
    if tactile_rgb is not None:
        tactile = np.asarray(tactile_rgb, dtype=np.uint8)
        if tactile.ndim == 4 and tactile.shape[0] == 1:
            tactile = tactile[0]
        if tactile.ndim != 3 or tactile.shape[-1] != 3:
            raise ValueError(
                f"Preview TacThru RGB must have shape [1,H,W,3] or [H,W,3], got {tactile.shape}"
            )
        tactile_bgr = cv2.cvtColor(tactile, cv2.COLOR_RGB2BGR)

        reference_value = tactile_frame.marker_reference_pixels
        current_value = tactile_frame.marker_current_pixels
        valid_value = tactile_frame.marker_valid_mask
        displacement_value = tactile_frame.marker_displacement_history
        if (
            reference_value is not None
            and current_value is not None
            and valid_value is not None
            and displacement_value is not None
        ):
            reference = np.asarray(reference_value, dtype=np.float32)
            current = np.asarray(current_value, dtype=np.float32)
            valid = np.asarray(valid_value, dtype=np.bool_)
            displacement = np.asarray(displacement_value, dtype=np.float32)
            if reference.ndim == 3 and reference.shape[0] == 1:
                reference = reference[0]
            if current.ndim == 3 and current.shape[0] == 1:
                current = current[0]
            if valid.ndim == 3 and valid.shape[0] == 1:
                valid = valid[0, -1]
            if displacement.ndim == 4 and displacement.shape[0] == 1:
                displacement = displacement[0, -1]
            expected_points = (TACTILE_MARKER_COUNT, 2)
            if reference.shape != expected_points or current.shape != expected_points:
                raise ValueError(
                    "Preview marker coordinates must both have shape "
                    f"{expected_points}, got {reference.shape} and {current.shape}"
                )
            if valid.shape != (TACTILE_MARKER_COUNT,):
                raise ValueError(
                    "Preview marker validity must have shape "
                    f"{(TACTILE_MARKER_COUNT,)}, got {valid.shape}"
                )
            if displacement.shape != expected_points:
                raise ValueError(
                    "Preview marker displacement must have shape "
                    f"{expected_points}, got {displacement.shape}"
                )
            finite_reference = np.isfinite(reference).all(axis=-1)
            finite_current = np.isfinite(current).all(axis=-1)
            display_valid = valid & finite_reference & finite_current
            marker_count = int(len(valid))
            valid_count = int(display_valid.sum())
            finite_displacement = display_valid & np.isfinite(displacement).all(axis=-1)
            norms = np.linalg.norm(displacement[finite_displacement], axis=-1)
            if len(norms):
                mean_displacement = float(norms.mean())
                max_displacement = float(norms.max())

            for reference_xy, current_xy, reference_is_finite, point_is_valid in zip(
                reference, current, finite_reference, display_valid
            ):
                if not reference_is_finite:
                    continue
                reference_point = tuple(np.rint(reference_xy).astype(int))
                if not point_is_valid:
                    # Keep the point visible without implying a measured displacement.
                    cv2.circle(
                        tactile_bgr,
                        reference_point,
                        5,
                        (145, 145, 145),
                        1,
                        cv2.LINE_AA,
                    )
                    continue
                arrow_end = reference_xy + float(arrow_scale) * (current_xy - reference_xy)
                cv2.arrowedLine(
                    tactile_bgr,
                    reference_point,
                    tuple(np.rint(arrow_end).astype(int)),
                    (0, 205, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.22,
                )
                cv2.circle(
                    tactile_bgr,
                    reference_point,
                    6,
                    (30, 220, 70),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    tactile_bgr,
                    tuple(np.rint(current_xy).astype(int)),
                    4,
                    (30, 30, 235),
                    -1,
                    cv2.LINE_AA,
                )

        detected_count = tactile_frame.debug.get("detected_keypoint_count")
        tactile_panel = cv2.resize(
            tactile_bgr,
            (tactile_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        cv2.putText(
            tactile_panel,
            "TACTILE INPUT WAITING",
            (155, panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (150, 150, 150),
            2,
            cv2.LINE_AA,
        )

    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
    wrist_panel = cv2.resize(
        wrist_bgr,
        (wrist_width, panel_height),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.putText(
        tactile_panel,
        "TACTILE RGB + MARKERS",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        wrist_panel,
        "WRIST RGB (MODEL INPUT)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    canvas_width = tactile_width + gap + wrist_width
    canvas = np.full(
        (header_height + panel_height, canvas_width, 3),
        (20, 20, 20),
        dtype=np.uint8,
    )
    canvas[header_height:, :tactile_width] = tactile_panel
    canvas[header_height:, tactile_width + gap :] = wrist_panel
    gate_label, gate_color = _gate_preview_status(tactile_frame)
    cv2.putText(
        canvas,
        status,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    detected_text = "n/a" if detected_count is None else str(detected_count)
    cv2.putText(
        canvas,
        (
            f"{gate_label} | valid={valid_count}/{marker_count or TACTILE_MARKER_COUNT} "
            f"detected={detected_text} mean|d|={mean_displacement:.5f} "
            f"max|d|={max_displacement:.5f} arrows=x{arrow_scale:g}"
        ),
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        gate_color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def _show_preview(
    rgb: np.ndarray,
    status: str,
    *,
    tactile_frame: TactileFrame | None = None,
    return_key: bool = False,
) -> int | None:
    global _PREVIEW_WINDOW_CREATED
    canvas = _build_preview_canvas(rgb, status, tactile_frame=tactile_frame)
    if not _PREVIEW_WINDOW_CREATED:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, canvas.shape[1], canvas.shape[0])
        _PREVIEW_WINDOW_CREATED = True
    cv2.imshow(PREVIEW_WINDOW_NAME, canvas)
    key = cv2.waitKey(1) & 0xFF
    if not return_key and key in (ord("q"), ord("Q"), 27):
        raise KeyboardInterrupt
    return key if return_key else None


def _close_preview() -> None:
    global _PREVIEW_WINDOW_CREATED
    if not _PREVIEW_WINDOW_CREATED:
        return
    try:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)
        cv2.waitKey(1)
    except cv2.error as exc:
        print(
            f"[lingbot-v2-client] OpenCV preview cleanup skipped: {exc}",
            flush=True,
        )
    finally:
        _PREVIEW_WINDOW_CREATED = False


def _wait_until(
    deadline: float,
    *,
    preview_rgb: np.ndarray | None,
    status: str,
    preview_tactile_frame: TactileFrame | None = None,
) -> None:
    while time.time() < deadline:
        if preview_rgb is not None:
            _show_preview(
                preview_rgb,
                status + " | q/Esc: software stop request",
                tactile_frame=preview_tactile_frame,
            )
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
