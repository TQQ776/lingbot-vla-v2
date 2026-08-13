from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lingbotvla.tactile_contact import CONTACT_OFF
from lingbotvla.models.vla.lingbot_vla.tactile_vtla import (
    TactileVTLAConfig,
    migrate_legacy_tactile_config,
)

from .protocol import (
    CAMERA_KEY,
    CONTROL_FREQUENCY_HZ,
    MARKER_DISPLACEMENT_KEY,
    MARKER_VALID_KEY,
    POSE_FRAME,
    POSE_SEMANTICS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ROBOT_CONFIG,
    TACTILE_MARKER_COUNT,
    TACTILE_MARKER_HISTORY_LENGTH,
    TACTILE_MARKER_SAMPLE_HZ,
    TACTILE_RGB_KEY,
    TACTILE_SENSOR_COUNT,
    ActionResponse,
    Observation,
    action_response_to_payload,
    action_spec,
    observation_from_json,
)
from .transforms import ACTION_DIM, STATE_DIM, validate_action_chunk


class BackendBusy(RuntimeError):
    def __init__(self, message: str, *, timing_s: dict[str, float] | None = None) -> None:
        super().__init__(message)
        self.timing_s = dict(timing_s or {})


class LingBotV2Backend:
    """Serialized, stateless-request facade around ``LingbotVLAv2Server``."""

    def __init__(
        self,
        policy,
        *,
        checkpoint: Path,
        norm_stats: Path,
        robot_config_path: Path,
        chunk_size: int,
        use_compile: bool,
        dtype: str,
        contract: dict[str, Any] | None = None,
        inference_lock_timeout_s: float = 5.0,
    ) -> None:
        self.policy = policy
        self.checkpoint = checkpoint
        self.norm_stats = norm_stats
        self.robot_config_path = robot_config_path
        self.chunk_size = int(chunk_size)
        self.use_compile = bool(use_compile)
        self.dtype = dtype
        self.contract = dict(contract or {})
        self.tactile_contract = _tactile_contract_from_policy(policy)
        expected_tactile = self.contract.get("tactile")
        if expected_tactile is not None and expected_tactile != self.tactile_contract:
            raise RuntimeError(
                "Loaded policy tactile contract does not match its training config: "
                f"policy={self.tactile_contract}, training={expected_tactile}"
            )
        self.inference_lock_timeout_s = float(inference_lock_timeout_s)
        self._lock = threading.Lock()
        self._active_session_id: str | None = None
        self._request_count = 0
        self.started_at = time.time()

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
    ) -> "LingBotV2Backend":
        project_root = project_root.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        norm_stats = norm_stats.expanduser().resolve()
        robot_config_path = (project_root / "configs/robot_configs/tacthru_umi_v2.yaml").resolve()
        _validate_runtime_paths(project_root, checkpoint, norm_stats, robot_config_path, qwen_path)
        os.chdir(project_root)
        contract = _validate_deployment_contract(
            project_root=project_root,
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            robot_config_path=robot_config_path,
        )
        if contract["robot_default_norm_overridden"]:
            print(
                "[lingbot-v2-server] using checkpoint norm instead of the robot config default: "
                f"checkpoint={norm_stats}, robot_default={contract['robot_default_norm_stats']}",
                flush=True,
            )

        if qwen_path is not None:
            os.environ["QWEN3VL_PATH"] = str(qwen_path.expanduser().resolve())

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; LingBot V2 real-robot server requires a GPU")
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        policy = LingbotVLAv2Server(
            str(checkpoint),
            robot_norm_path=str(norm_stats),
            use_length=50,
            chunk_ret=True,
            use_bf16=dtype == "bf16",
            use_fp32=dtype == "fp32",
            use_compile=use_compile,
        )
        policy.reset(robo_name=ROBOT_CONFIG)
        chunk_size = int(getattr(policy.config, "chunk_size", 50))
        if chunk_size != 50:
            raise RuntimeError(f"Expected trained action chunk size 50, got {chunk_size}")
        return cls(
            policy,
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            robot_config_path=robot_config_path,
            chunk_size=chunk_size,
            use_compile=use_compile,
            dtype=dtype,
            contract=contract,
            inference_lock_timeout_s=inference_lock_timeout_s,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": True,
            "backend": "lingbot-vla-v2-tacthru-umi-http",
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "robot_config": ROBOT_CONFIG,
            "robot_config_file": self.robot_config_path.name,
            "checkpoint_id": f"{self.checkpoint.parent.name}/{self.checkpoint.name}",
            "norm_stats_file": self.norm_stats.name,
            "chunk_size": self.chunk_size,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "action_horizon_s": (self.chunk_size - 1) / CONTROL_FREQUENCY_HZ,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "pose_frame": POSE_FRAME,
            "pose_semantics": POSE_SEMANTICS,
            "camera_key": CAMERA_KEY,
            "image_shape_hwc": [224, 224, 3],
            "tactile_enabled": self.tactile_contract["enabled"],
            "tactile": self.tactile_contract,
            "action_spec": action_spec(chunk_size=self.chunk_size),
            "dtype": self.dtype,
            "use_compile": self.use_compile,
            "max_concurrent_inference": 1,
            "requests_are_stateless": True,
            "contract": self.contract,
            "request_count": self._request_count,
            "uptime_s": time.time() - self.started_at,
        }

    def predict(self, observation: Observation) -> dict[str, Any]:
        """Run inference and return the existing HTTP protocol payload."""

        response = self.predict_response(observation)
        return action_response_to_payload(
            action_chunk=response.action_chunk,
            request_id=response.request_id,
            session_id=response.session_id,
            expected_steps=self.chunk_size,
            metadata=response.metadata,
        )

    def predict_response(self, observation: Observation) -> ActionResponse:
        """Run inference without JSON/JPEG serialization for in-process callers."""

        if not np.isclose(observation.control_frequency_hz, CONTROL_FREQUENCY_HZ, atol=1e-6):
            raise ValueError(
                f"This checkpoint was trained at {CONTROL_FREQUENCY_HZ:g}Hz; "
                f"request asked for {observation.control_frequency_hz:g}Hz"
            )
        backend_started = time.perf_counter()
        lock_started = time.perf_counter()
        acquired = self._lock.acquire(timeout=self.inference_lock_timeout_s)
        lock_wait_s = time.perf_counter() - lock_started
        if not acquired:
            raise BackendBusy(
                "Inference backend is busy; retry after the current request finishes",
                timing_s={"backend_lock_wait_s": lock_wait_s},
            )
        try:
            inference_started = time.perf_counter()
            episode_reset = bool(observation.metadata.get("episode_reset"))
            session_changed = observation.session_id != self._active_session_id
            reset_started = time.perf_counter()
            if episode_reset or session_changed:
                self.policy.reset(robo_name=ROBOT_CONFIG)
                self._active_session_id = observation.session_id
            reset_s = time.perf_counter() - reset_started

            model_observation = {
                "observation.state": np.asarray(observation.state, dtype=np.float32),
                CAMERA_KEY: np.asarray(observation.wrist_rgb, dtype=np.uint8),
                "task": observation.instruction,
            }
            model_observation.update(
                _model_tactile_observation(observation, self.tactile_contract)
            )
            policy_started = time.perf_counter()
            result = self.policy.infer(model_observation)
            policy_infer_s = time.perf_counter() - policy_started
            postprocess_started = time.perf_counter()
            if not isinstance(result, dict) or "action" not in result:
                raise RuntimeError(f"Policy returned an unsupported result: {type(result)!r}")
            action = np.asarray(result["action"], dtype=np.float32)
            if action.ndim == 3 and action.shape[0] == 1:
                action = action[0]
            action = validate_action_chunk(action, expected_steps=self.chunk_size)
            postprocess_s = time.perf_counter() - postprocess_started
            inference_time_s = time.perf_counter() - inference_started
            server_timing_s = {
                "backend_lock_wait_s": lock_wait_s,
                "backend_reset_s": reset_s,
                "backend_policy_infer_s": policy_infer_s,
                "backend_postprocess_s": postprocess_s,
                "backend_inference_s": inference_time_s,
                "backend_total_s": time.perf_counter() - backend_started,
            }
            self._request_count += 1
            return ActionResponse(
                action_chunk=action,
                request_id=observation.request_id,
                session_id=observation.session_id,
                metadata={
                    "backend": "lingbot-vla-v2-tacthru-umi-http",
                    "checkpoint": str(self.checkpoint),
                    "norm_stats": str(self.norm_stats),
                    "instruction": observation.instruction,
                    "inference_time_s": inference_time_s,
                    "server_timing_s": server_timing_s,
                    "episode_reset": episode_reset,
                    "session_changed": session_changed,
                    "input_state": "episode-start-frame xyz+quaternion_xyzw+gripper_width_m",
                    "output_action": "episode-start-frame absolute xyz+quaternion_xyzw+gripper_width_m",
                    "contract_sha256": self.contract.get("combined_sha256"),
                },
                server_timestamp=time.time(),
            )
        finally:
            self._lock.release()

    def warmup(self, *, instruction: str = "Pull the tissue") -> dict[str, Any]:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("Warmup instruction must not be empty")
        tactile = _synthetic_tactile_observation(self.tactile_contract)
        observation = Observation(
            instruction=instruction,
            state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.045], dtype=np.float32),
            wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            **tactile,
            metadata={"synthetic": True, "episode_reset": True},
        )
        started = time.time()
        response = self.predict(observation)
        return {
            "warmup_s": time.time() - started,
            "instruction": instruction,
            "action_shape": list(np.asarray(response["action_chunk"]).shape),
        }


class LingBotV2HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        backend: LingBotV2Backend,
        api_key: str | None,
        max_body_bytes: int,
        request_timeout_s: float,
        max_request_threads: int,
        http_keep_alive: bool,
        keep_alive_idle_timeout_s: float,
        keep_alive_max_requests: int,
    ) -> None:
        if not _is_loopback_host(server_address[0]) and not api_key:
            raise ValueError("A non-loopback server address requires an API key")
        if (
            max_body_bytes <= 0
            or request_timeout_s <= 0.0
            or max_request_threads <= 0
            or keep_alive_idle_timeout_s <= 0.0
            or keep_alive_max_requests <= 0
        ):
            raise ValueError("Server limits must be positive")
        super().__init__(server_address, handler_class)
        self.backend = backend
        self.api_key = api_key
        self.max_body_bytes = int(max_body_bytes)
        self.request_timeout_s = float(request_timeout_s)
        self.http_keep_alive = bool(http_keep_alive)
        self.keep_alive_idle_timeout_s = float(keep_alive_idle_timeout_s)
        self.keep_alive_max_requests = int(keep_alive_max_requests)
        self._request_slots = threading.BoundedSemaphore(int(max_request_threads))
        self.accepted_connection_count = 0

    def get_request(self):
        request, client_address = super().get_request()
        self.accepted_connection_count += 1
        return request, client_address


class Handler(BaseHTTPRequestHandler):
    server_version = "LingBotV2TacThruHTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.request_timeout_s)
        self._responses_sent = 0

    def do_GET(self) -> None:
        trace_id = uuid.uuid4().hex
        if not self._acquire_request_slot(trace_id):
            return
        try:
            self.connection.settimeout(self.server.request_timeout_s)
            if self.path.rstrip("/") in ("/health", "/healthz"):
                if self.server.api_key and not self._authorized():
                    self._send_error_json(
                        "unauthorized",
                        status=401,
                        error_stage="authorization",
                        trace_id=trace_id,
                        force_close=True,
                    )
                    return
                health = self.server.backend.health()
                health["transport"] = {
                    "http_protocol": self.protocol_version,
                    "http_keep_alive_enabled": self.server.http_keep_alive,
                    "keep_alive_idle_timeout_s": self.server.keep_alive_idle_timeout_s,
                    "keep_alive_max_requests": self.server.keep_alive_max_requests,
                    "server_timing_version": 1,
                }
                self._send_json(health, trace_id=trace_id)
            else:
                self._send_error_json(
                    "not found",
                    status=404,
                    error_stage="routing",
                    trace_id=trace_id,
                )
        finally:
            self.server._request_slots.release()

    def do_POST(self) -> None:
        trace_id = uuid.uuid4().hex
        timing_s: dict[str, float] = {}
        if not self._acquire_request_slot(trace_id):
            return
        try:
            self.connection.settimeout(self.server.request_timeout_s)
            if self.path.rstrip("/") != "/predict":
                self._send_error_json(
                    "not found",
                    status=404,
                    error_stage="routing",
                    trace_id=trace_id,
                    force_close=True,
                )
                return
            if not self._authorized():
                self._send_error_json(
                    "unauthorized",
                    status=401,
                    error_stage="authorization",
                    trace_id=trace_id,
                    force_close=True,
                )
                return
            if self.headers.get("Transfer-Encoding"):
                self._send_error_json(
                    "Transfer-Encoding is not supported",
                    status=400,
                    error_stage="headers",
                    trace_id=trace_id,
                    force_close=True,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Content-Length must be an integer") from exc
            if length <= 0:
                raise ValueError("Request body is empty")
            if length > self.server.max_body_bytes:
                self._send_error_json(
                    f"request body exceeds {self.server.max_body_bytes} bytes",
                    status=413,
                    error_stage="headers",
                    trace_id=trace_id,
                    force_close=True,
                )
                return
            read_started = time.perf_counter()
            raw_body = self.rfile.read(length)
            timing_s["request_read_s"] = time.perf_counter() - read_started
            if len(raw_body) != length:
                raise ValueError(f"Request body ended early: expected {length} bytes, got {len(raw_body)}")
            decode_started = time.perf_counter()
            observation = observation_from_json(raw_body)
            timing_s["request_decode_s"] = time.perf_counter() - decode_started
            payload = self.server.backend.predict(observation)
            metadata = payload.setdefault("metadata", {})
            server_timing_s = metadata.setdefault("server_timing_s", {})
            server_timing_s.update(timing_s)
            metadata["request_trace_id"] = trace_id
            self._send_json(payload, trace_id=trace_id, timing_s=server_timing_s)
        except BackendBusy as exc:
            timing_s.update(exc.timing_s)
            self._send_error_json(
                f"BackendBusy: {exc}",
                status=503,
                error_stage="backend_lock",
                trace_id=trace_id,
                timing_s=timing_s,
            )
        except (socket.timeout, TimeoutError) as exc:
            self._send_error_json(
                f"RequestTimeout: {exc}",
                status=408,
                error_stage="request_io",
                trace_id=trace_id,
                timing_s=timing_s,
                force_close=True,
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send_error_json(
                f"{type(exc).__name__}: {exc}",
                status=400,
                error_stage="request_decode",
                trace_id=trace_id,
                timing_s=timing_s,
                force_close="request_read_s" not in timing_s,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_error_json(
                f"{type(exc).__name__}: {exc}",
                status=500,
                error_stage="backend_inference",
                trace_id=trace_id,
                timing_s=timing_s,
            )
        finally:
            self.server._request_slots.release()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.client_address[0]} {fmt % args}",
            flush=True,
        )

    def _authorized(self) -> bool:
        expected = self.server.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[len("Bearer ") :] if header.startswith("Bearer ") else ""
        return hmac.compare_digest(supplied, expected)

    def _acquire_request_slot(self, trace_id: str) -> bool:
        if self.server._request_slots.acquire(blocking=False):
            return True
        self._send_error_json(
            "server busy",
            status=503,
            error_stage="request_slot",
            trace_id=trace_id,
            force_close=True,
            extra_headers={"Retry-After": "1"},
        )
        return False

    def _send_error_json(
        self,
        message: str,
        *,
        status: int,
        error_stage: str,
        trace_id: str,
        timing_s: dict[str, float] | None = None,
        force_close: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._send_json(
            {
                "error": message,
                "error_stage": error_stage,
                "request_trace_id": trace_id,
                "timing_s": dict(timing_s or {}),
            },
            status=status,
            trace_id=trace_id,
            timing_s=timing_s,
            force_close=force_close,
            extra_headers=extra_headers,
        )

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: int = 200,
        trace_id: str | None = None,
        timing_s: dict[str, float] | None = None,
        force_close: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encode_started = time.perf_counter()
        data = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        response_encode_s = time.perf_counter() - encode_started
        response_timing_s = dict(timing_s or {})
        response_timing_s["response_encode_s"] = response_encode_s
        self._responses_sent += 1
        client_requested_close = self.headers.get("Connection", "").lower() == "close"
        keep_alive = bool(
            self.server.http_keep_alive
            and self.protocol_version == "HTTP/1.1"
            and not force_close
            and not client_requested_close
            and self._responses_sent < self.server.keep_alive_max_requests
        )
        self.close_connection = not keep_alive
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive" if keep_alive else "close")
        if keep_alive:
            self.send_header(
                "Keep-Alive",
                f"timeout={self.server.keep_alive_idle_timeout_s:g}, "
                f"max={self.server.keep_alive_max_requests - self._responses_sent}",
            )
        if trace_id:
            self.send_header("X-LingBot-Trace-Id", trace_id)
        server_timing = _format_server_timing(response_timing_s)
        if server_timing:
            self.send_header("Server-Timing", server_timing)
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()
        if keep_alive:
            self.connection.settimeout(self.server.keep_alive_idle_timeout_s)


def create_http_server(
    backend: LingBotV2Backend,
    *,
    host: str,
    port: int,
    api_key: str | None = None,
    max_body_bytes: int = 8 * 1024 * 1024,
    request_timeout_s: float = 30.0,
    max_request_threads: int = 8,
    http_keep_alive: bool = False,
    keep_alive_idle_timeout_s: float = 5.0,
    keep_alive_max_requests: int = 100,
) -> LingBotV2HTTPServer:
    protocol_version = "HTTP/1.1" if http_keep_alive else "HTTP/1.0"
    configured_handler = type(
        "ConfiguredLingBotV2Handler",
        (Handler,),
        {"protocol_version": protocol_version},
    )
    return LingBotV2HTTPServer(
        (host, int(port)),
        configured_handler,
        backend=backend,
        api_key=api_key,
        max_body_bytes=max_body_bytes,
        request_timeout_s=request_timeout_s,
        max_request_threads=max_request_threads,
        http_keep_alive=http_keep_alive,
        keep_alive_idle_timeout_s=keep_alive_idle_timeout_s,
        keep_alive_max_requests=keep_alive_max_requests,
    )


def _format_server_timing(timing_s: dict[str, float]) -> str:
    parts: list[str] = []
    for name, value in timing_s.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(seconds) or seconds < 0.0:
            continue
        metric_name = name.removesuffix("_s").replace("_", "-")
        parts.append(f"{metric_name};dur={seconds * 1000.0:.3f}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="LingBot-VLA V2 TacThru UMI HTTP inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Final .../global_step_N/hf_ckpt directory")
    parser.add_argument("--norm-stats", type=Path, default=Path("assets/norm_stats/tacthru_umi_v2.json"))
    parser.add_argument("--qwen-path", type=Path, default=Path("models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--warmup-instruction",
        default="Pull the tissue",
        help="Task instruction used by --warmup; match the real client instruction when using torch.compile.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LINGBOT_V2_API_KEY"))
    parser.add_argument("--max-body-mb", type=_positive_float, default=8.0)
    parser.add_argument("--inference-lock-timeout", type=_positive_float, default=5.0)
    parser.add_argument("--request-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--max-request-threads", type=_positive_int, default=8)
    parser.add_argument(
        "--http-keep-alive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse HTTP/1.1 connections; --no-http-keep-alive restores HTTP/1.0 Connection: close.",
    )
    parser.add_argument("--keep-alive-idle-timeout", type=_positive_float, default=5.0)
    parser.add_argument("--keep-alive-max-requests", type=_positive_int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _is_loopback_host(args.host) and not args.api_key:
        raise RuntimeError("A non-loopback --host requires LINGBOT_V2_API_KEY or --api-key")
    if not _is_loopback_host(args.host):
        print(
            "[lingbot-v2-server] WARNING: direct HTTP does not protect action integrity. "
            "Use an SSH tunnel or an HTTPS/mTLS reverse proxy for real execution.",
            flush=True,
        )
    project_root = args.project_root.expanduser().resolve()
    checkpoint = _resolve_project_path(project_root, args.checkpoint)
    norm_stats = _resolve_project_path(project_root, args.norm_stats)
    qwen_path = _resolve_project_path(project_root, args.qwen_path) if args.qwen_path is not None else None
    backend = LingBotV2Backend.from_checkpoint(
        project_root=project_root,
        checkpoint=checkpoint,
        norm_stats=norm_stats,
        qwen_path=qwen_path,
        use_compile=args.use_compile,
        dtype=args.dtype,
        inference_lock_timeout_s=args.inference_lock_timeout,
    )
    print(json.dumps(backend.health(), indent=2, ensure_ascii=False), flush=True)
    if args.warmup:
        print(
            json.dumps(
                backend.warmup(instruction=args.warmup_instruction),
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    server = create_http_server(
        backend,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        max_body_bytes=int(args.max_body_mb * 1024 * 1024),
        request_timeout_s=args.request_timeout,
        max_request_threads=args.max_request_threads,
        http_keep_alive=args.http_keep_alive,
        keep_alive_idle_timeout_s=args.keep_alive_idle_timeout,
        keep_alive_max_requests=args.keep_alive_max_requests,
    )
    print(
        f"[lingbot-v2-server] listening on http://{args.host}:{args.port} "
        f"keep_alive={args.http_keep_alive}",
        flush=True,
    )
    server.serve_forever()


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _resolve_training_project_path(project_root: Path, path: Path) -> Path:
    """Rebase an artifact path saved under a previous copy of this project."""

    value = path.expanduser()
    if value.is_absolute():
        matching_indices = [
            index for index, part in enumerate(value.parts) if part == project_root.name
        ]
        if matching_indices:
            relative_parts = value.parts[matching_indices[-1] + 1 :]
            return project_root.joinpath(*relative_parts).resolve()
    return _resolve_project_path(project_root, value)


def _validate_runtime_paths(
    project_root: Path,
    checkpoint: Path,
    norm_stats: Path,
    robot_config_path: Path,
    qwen_path: Path | None,
) -> None:
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)
    if not any(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors shards found in {checkpoint}")
    training_config = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    if not training_config.is_file():
        raise FileNotFoundError(f"Training config expected by official loader is missing: {training_config}")
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)
    if not robot_config_path.is_file():
        raise FileNotFoundError(robot_config_path)
    if qwen_path is not None and not qwen_path.is_dir():
        raise NotADirectoryError(qwen_path)


def _validate_deployment_contract(
    *,
    project_root: Path,
    checkpoint: Path,
    norm_stats: Path,
    robot_config_path: Path,
) -> dict[str, Any]:
    training_config_path = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    with training_config_path.open("r", encoding="utf-8") as file:
        training = yaml.safe_load(file) or {}
    with robot_config_path.open("r", encoding="utf-8") as file:
        robot = yaml.safe_load(file) or {}

    data = training.get("data") or {}
    train = training.get("train") or {}
    tactile_mapping, tactile_migration = migrate_legacy_tactile_config(
        train.get("tactile"),
        allow_legacy_marker_reinit=(
            os.environ.get("LINGBOT_V2_ALLOW_LEGACY_MARKER_REINIT", "0") == "1"
        ),
    )
    tactile = _tactile_contract_from_mapping(tactile_mapping)
    _require_equal(data.get("data_name"), ROBOT_CONFIG, "training data.data_name")
    _require_equal(data.get("cameras"), ["camera_wrist_left"], "training data.cameras")
    for key, expected in (
        ("chunk_size", 50),
        ("action_dim", 55),
        ("max_action_dim", 55),
        ("max_state_dim", 55),
    ):
        _require_equal(train.get(key), expected, f"training train.{key}")

    expected_norm_from_training = _resolve_training_project_path(
        project_root, Path(str(data.get("norm_stats_file")))
    )
    expected_norm_from_robot = _resolve_project_path(project_root, Path(str(robot.get("norm_stats"))))
    if expected_norm_from_training != norm_stats:
        raise RuntimeError(
            "Norm stats path mismatch: "
            f"server={norm_stats}, training={expected_norm_from_training}"
        )

    # LingbotVLAv2Server passes norm_stats_path explicitly to FeatureTransform,
    # so the robot YAML value is only a fallback for callers that omit it. A
    # single robot mapping can therefore serve checkpoints trained on distinct
    # datasets without mutating the shared YAML before every inference run.
    robot_default_norm_overridden = expected_norm_from_robot != norm_stats

    state_end = _mapping_entry(robot.get("states"), "observation.state.end.position")
    state_effector = _mapping_entry(robot.get("states"), "observation.state.effector.position")
    action_end = _mapping_entry(robot.get("actions"), "action.end.position")
    action_effector = _mapping_entry(robot.get("actions"), "action.effector.position")
    _require_equal(
        state_end,
        {"origin_keys": [{"observation.state": {"start": 0, "end": 7}}]},
        "robot state end mapping",
    )
    _require_equal(
        state_effector,
        {"origin_keys": [{"observation.state": {"start": 7, "end": 8}}]},
        "robot state effector mapping",
    )
    _require_equal(
        action_end,
        {
            "origin_keys": [{"action": {"start": 0, "end": 7}}],
            "subtract_state": True,
            "relative_type": "quaternion_local",
        },
        "robot action end mapping",
    )
    _require_equal(
        action_effector,
        {
            "origin_keys": [{"action": {"start": 7, "end": 8}}],
            "subtract_state": False,
        },
        "robot action effector mapping",
    )
    _require_equal(robot.get("images"), [CAMERA_KEY], "robot image mapping")

    index_path = checkpoint / "model.safetensors.index.json"
    hashes = {
        "training_config_sha256": _sha256(training_config_path),
        "robot_config_sha256": _sha256(robot_config_path),
        "norm_stats_sha256": _sha256(norm_stats),
        "checkpoint_index_sha256": _sha256(index_path),
    }
    combined = hashlib.sha256()
    for key in sorted(hashes):
        combined.update(key.encode("utf-8"))
        combined.update(hashes[key].encode("ascii"))
    return {
        "version": 1,
        **hashes,
        "combined_sha256": combined.hexdigest(),
        "robot_default_norm_stats": str(expected_norm_from_robot),
        "robot_default_norm_overridden": robot_default_norm_overridden,
        "tactile": tactile,
        "legacy_tactile_config_migration": tactile_migration,
    }


def _tactile_contract_from_policy(policy: Any) -> dict[str, Any]:
    config = getattr(policy, "config", None)
    tactile = getattr(config, "tactile", None) if config is not None else None
    return _tactile_contract_from_mapping(tactile)


def _tactile_contract_from_mapping(value: Any) -> dict[str, Any]:
    tactile = dict(value or {})
    settings = TactileVTLAConfig.from_mapping(tactile)
    resolved = settings.to_dict()
    enabled = settings.enabled
    if not enabled:
        return {
            "enabled": False,
            "num_sensors": 0,
            "num_markers": 0,
            "use_rgb": False,
            "use_markers": False,
            "marker_history_length": 0,
            "marker_sample_hz": 0.0,
        }
    contract = {
        "enabled": enabled,
        "num_sensors": settings.num_sensors,
        "num_markers": settings.num_markers,
        "use_rgb": settings.use_rgb if enabled else False,
        "use_markers": settings.use_markers if enabled else False,
        "marker_history_length": settings.marker_history_length,
        "marker_sample_hz": settings.marker_sample_hz,
    }
    _require_equal(contract["num_sensors"], TACTILE_SENSOR_COUNT, "tactile num_sensors")
    _require_equal(contract["num_markers"], TACTILE_MARKER_COUNT, "tactile num_markers")
    if not (contract["use_rgb"] or contract["use_markers"]):
        raise RuntimeError("Enabled tactile checkpoint must use RGB and/or markers")
    if contract["use_rgb"]:
        _require_equal(tactile.get("rgb_keys"), [TACTILE_RGB_KEY], "tactile rgb_keys")
    if contract["use_markers"]:
        if contract["marker_history_length"] <= 0:
            raise RuntimeError("tactile marker_history_length must be positive")
        _require_equal(
            contract["marker_sample_hz"],
            TACTILE_MARKER_SAMPLE_HZ,
            "tactile marker_sample_hz",
        )
        _require_equal(
            tactile.get("marker_feature_mode"),
            "displacement_history",
            "tactile marker_feature_mode",
        )
        _require_equal(
            tactile.get("marker_displacement_keys"),
            [MARKER_DISPLACEMENT_KEY],
            "tactile marker_displacement_keys",
        )
        _require_equal(
            tactile.get("marker_valid_mask_keys"),
            [MARKER_VALID_KEY],
            "tactile marker_valid_mask_keys",
        )
        contract.update(
            marker_tokenization=resolved["marker_tokenization"],
            marker_position_encoding=resolved["marker_position_encoding"],
            marker_contact_gate=resolved["marker_contact_gate"],
            marker_ablation=resolved["marker_ablation"],
            marker_reference_xy_path=resolved["marker_reference_xy_path"],
            marker_reference_xy=resolved["marker_reference_xy"],
            marker_tokens_per_sensor=settings.marker_tokens_per_sensor,
            gate_tactile_rgb=(
                settings.marker_contact_gate.mode != "none"
                and settings.marker_contact_gate.target == "marker_and_rgb"
            ),
        )
    return contract


def _model_tactile_observation(
    observation: Observation,
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    values = {
        "tactile_rgb": observation.tactile_rgb,
        "marker_displacement_history": observation.marker_displacement_history,
        "marker_valid_mask": observation.marker_valid_mask,
        "marker_history_valid_mask": observation.marker_history_valid_mask,
        "marker_contact_state": observation.marker_contact_state,
        "tactile_sensor_mask": observation.tactile_sensor_mask,
    }
    supplied = {name for name, value in values.items() if value is not None}
    if not contract["enabled"]:
        if supplied:
            raise ValueError(f"Checkpoint is tactile-disabled but request supplied {sorted(supplied)}")
        return {}

    required = {"tactile_sensor_mask"}
    if contract["use_rgb"]:
        required.add("tactile_rgb")
    if contract["use_markers"]:
        required.update(
            {
                "marker_displacement_history",
                "marker_valid_mask",
                "marker_history_valid_mask",
            }
        )
        if contract.get("marker_contact_gate", {}).get("mode", "none") != "none":
            required.add("marker_contact_state")
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - required)
    if missing or unexpected:
        raise ValueError(
            "Tactile request does not match checkpoint: "
            f"missing={missing}, unexpected={unexpected}, contract={contract}"
        )
    result: dict[str, np.ndarray] = {
        name: np.asarray(value) for name, value in values.items() if value is not None
    }
    if contract["use_markers"]:
        marker_shape = (
            contract["num_sensors"],
            contract["marker_history_length"],
            contract["num_markers"],
            2,
        )
        if result["marker_displacement_history"].shape != marker_shape:
            raise ValueError(
                "Tactile marker history does not match checkpoint: "
                f"expected={marker_shape}, "
                f"got={result['marker_displacement_history'].shape}"
            )
    return result


def _synthetic_tactile_observation(
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
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
            result["marker_contact_state"] = np.full(
                (TACTILE_SENSOR_COUNT,), CONTACT_OFF, dtype=np.int8
            )
    return result


def _mapping_entry(entries: Any, key: str) -> dict[str, Any]:
    if not isinstance(entries, list):
        raise RuntimeError(f"Robot mapping section for {key} must be a list")
    matches = [entry[key] for entry in entries if isinstance(entry, dict) and key in entry]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one robot mapping for {key}, got {len(matches)}")
    if len(entries) != 2:
        raise RuntimeError(f"Expected exactly two entries in mapping section containing {key}, got {len(entries)}")
    return matches[0]


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive finite value, got {value!r}")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return result


def _is_loopback_host(host: str) -> bool:
    return str(host).strip().lower() in {"127.0.0.1", "localhost", "::1"}


if __name__ == "__main__":
    main()
