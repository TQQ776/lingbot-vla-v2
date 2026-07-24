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
        robot_config_name: str = ROBOT_CONFIG,
        chunk_size: int,
        use_compile: bool,
        dtype: str,
        contract: dict[str, Any] | None = None,
        inference_lock_timeout_s: float = 5.0,
        allow_missing_tactile: bool = False,
        allow_tactile_ablation: bool = False,
    ) -> None:
        self.policy = policy
        self.checkpoint = checkpoint
        self.norm_stats = norm_stats
        self.robot_config_path = robot_config_path
        self.robot_config_name = str(robot_config_name)
        self.chunk_size = int(chunk_size)
        self.use_compile = bool(use_compile)
        self.dtype = dtype
        self.contract = dict(contract or {})
        self.inference_lock_timeout_s = float(inference_lock_timeout_s)
        self.allow_missing_tactile = bool(allow_missing_tactile)
        self.allow_tactile_ablation = bool(allow_tactile_ablation)
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
        allow_missing_tactile: bool = False,
        allow_tactile_ablation: bool = False,
    ) -> "LingBotV2Backend":
        project_root = project_root.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        norm_stats = norm_stats.expanduser().resolve()
        robot_config_name = _checkpoint_robot_config_name(checkpoint)
        robot_config_path = (project_root / f"configs/robot_configs/{robot_config_name}.yaml").resolve()
        _validate_runtime_paths(project_root, checkpoint, norm_stats, robot_config_path, qwen_path)
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

        os.chdir(project_root)
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
        policy.reset(robo_name=robot_config_name)
        chunk_size = int(getattr(policy.config, "chunk_size", 50))
        if chunk_size != 50:
            raise RuntimeError(f"Expected trained action chunk size 50, got {chunk_size}")
        return cls(
            policy,
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            robot_config_path=robot_config_path,
            robot_config_name=robot_config_name,
            chunk_size=chunk_size,
            use_compile=use_compile,
            dtype=dtype,
            contract=contract,
            inference_lock_timeout_s=inference_lock_timeout_s,
            allow_missing_tactile=allow_missing_tactile,
            allow_tactile_ablation=allow_tactile_ablation,
        )

    def health(self) -> dict[str, Any]:
        tactile = dict(self.contract.get("tactile") or {})
        checkpoint_modalities = dict(
            tactile.get(
                "checkpoint_modalities",
                {"wrist_rgb": True, "tactile_rgb": False, "tactile_marker": False},
            )
        )
        tactile_enabled = bool(
            checkpoint_modalities.get("tactile_rgb") or checkpoint_modalities.get("tactile_marker")
        )
        return {
            "ok": True,
            "ready": True,
            "backend": "lingbot-vla-v2-tacthru-umi-http",
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_versions_supported": list(SUPPORTED_PROTOCOL_VERSIONS),
            "preferred_protocol_version": PROTOCOL_VERSION_V2 if tactile_enabled else PROTOCOL_VERSION_V1,
            "robot_config": self.robot_config_name,
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
            "tactile_enabled": tactile_enabled,
            "checkpoint_modalities": checkpoint_modalities,
            "tactile_temporal_horizon": int(tactile.get("history_steps", 1)),
            "tactile_history_stride": int(tactile.get("history_stride", 1)),
            "tactile_history_frequency_hz": float(tactile.get("history_frequency_hz", 30.0)),
            "tactile_max_timestamp_skew_s": float(tactile.get("max_timestamp_skew_s", 0.05)),
            "tactile_marker_spec": {
                "count": int(tactile.get("marker_count", TACTILE_MARKER_COUNT)),
                "dim": int(tactile.get("marker_dim", TACTILE_MARKER_DIM)),
                "normalization": tactile.get("marker_normalization", "image_size_xy"),
                "normalization_size_xy": list(tactile.get("marker_normalization_size_xy", [640, 480])),
            },
            "tactile_missing_policy": tactile.get("missing_policy", "mask"),
            "allow_missing_tactile": self.allow_missing_tactile,
            "allow_tactile_ablation": self.allow_tactile_ablation,
            "tactile_contract_sha256": self.contract.get("combined_sha256"),
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
        if not np.isclose(observation.control_frequency_hz, CONTROL_FREQUENCY_HZ, atol=1e-6):
            raise ValueError(
                f"This checkpoint was trained at {CONTROL_FREQUENCY_HZ:g}Hz; "
                f"request asked for {observation.control_frequency_hz:g}Hz"
            )
        tactile_request = self._validate_tactile_request(observation)
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
                self.policy.reset(robo_name=self.robot_config_name)
                self._active_session_id = observation.session_id
            reset_s = time.perf_counter() - reset_started

            input_prepare_started = time.perf_counter()
            model_observation = {
                "observation.state": np.asarray(observation.state, dtype=np.float32),
                CAMERA_KEY: np.asarray(observation.wrist_rgb, dtype=np.uint8),
                "task": observation.instruction,
            }
            tactile = dict(self.contract.get("tactile") or {})
            history_steps = int(tactile.get("history_steps", 1))
            marker_count = int(tactile.get("marker_count", TACTILE_MARKER_COUNT))
            marker_dim = int(tactile.get("marker_dim", TACTILE_MARKER_DIM))
            if tactile_request["checkpoint_modalities"]["tactile_rgb"]:
                if observation.tactile_rgb_history is None:
                    rgb_history = np.zeros((history_steps, 224, 224, 3), dtype=np.uint8)
                    rgb_mask = np.zeros((history_steps,), dtype=np.bool_)
                else:
                    rgb_history = np.asarray(observation.tactile_rgb_history, dtype=np.uint8)
                    rgb_mask = np.asarray(observation.tactile_rgb_history_mask, dtype=np.bool_).copy()
                if observation.force_mask_tactile_rgb:
                    rgb_mask[:] = False
                model_observation["tactile_rgb_history"] = rgb_history
                model_observation["tactile_rgb_history_mask"] = rgb_mask
            if tactile_request["checkpoint_modalities"]["tactile_marker"]:
                if observation.marker_flow is None:
                    marker_flow = np.zeros(
                        (history_steps, marker_count, marker_dim), dtype=np.float32
                    )
                    marker_valid = np.zeros((history_steps, marker_count), dtype=np.bool_)
                    marker_history_mask = np.zeros((history_steps,), dtype=np.bool_)
                else:
                    marker_flow = np.asarray(observation.marker_flow, dtype=np.float32)
                    marker_valid = np.asarray(observation.marker_valid_mask, dtype=np.bool_).copy()
                    marker_history_mask = np.asarray(
                        observation.marker_history_mask, dtype=np.bool_
                    ).copy()
                if observation.force_mask_tactile_marker:
                    marker_valid[:] = False
                    marker_history_mask[:] = False
                model_observation["tactile_marker_flow"] = marker_flow
                model_observation["tactile_marker_valid_mask"] = marker_valid
                model_observation["tactile_marker_history_mask"] = marker_history_mask
            if observation.marker_timestamps is not None:
                model_observation["tactile_history_timestamps"] = np.asarray(
                    observation.marker_timestamps, dtype=np.float64
                )
            elif observation.tactile_rgb_timestamps is not None:
                model_observation["tactile_history_timestamps"] = np.asarray(
                    observation.tactile_rgb_timestamps, dtype=np.float64
                )
            input_prepare_s = time.perf_counter() - input_prepare_started
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
                "backend_input_prepare_s": input_prepare_s,
                "backend_policy_infer_s": policy_infer_s,
                "backend_postprocess_s": postprocess_s,
                "backend_inference_s": inference_time_s,
                "backend_total_s": time.perf_counter() - backend_started,
            }
            self._request_count += 1
            return action_response_to_payload(
                action_chunk=action,
                request_id=observation.request_id,
                session_id=observation.session_id,
                expected_steps=self.chunk_size,
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
                    "protocol_version": observation.protocol_version,
                    "tactile": tactile_request,
                },
                protocol_version=observation.protocol_version,
            )
        finally:
            self._lock.release()

    def _validate_tactile_request(self, observation: Observation) -> dict[str, Any]:
        tactile = dict(self.contract.get("tactile") or {})
        checkpoint_modalities = dict(
            tactile.get(
                "checkpoint_modalities",
                {"wrist_rgb": True, "tactile_rgb": False, "tactile_marker": False},
            )
        )
        checkpoint_modalities = {
            "wrist_rgb": True,
            "tactile_rgb": bool(checkpoint_modalities.get("tactile_rgb", False)),
            "tactile_marker": bool(checkpoint_modalities.get("tactile_marker", False)),
        }
        present = {
            "wrist_rgb": True,
            "tactile_rgb": observation.tactile_rgb_history is not None,
            "tactile_marker": observation.marker_flow is not None,
        }
        force_mask = {
            "tactile_rgb": bool(observation.force_mask_tactile_rgb),
            "tactile_marker": bool(observation.force_mask_tactile_marker),
        }
        safe_ablation_context = bool(
            observation.metadata.get("synthetic")
            or observation.metadata.get("offline_evaluation")
            or (
                observation.metadata.get("dry_run")
                and not observation.metadata.get("execute")
            )
        )
        tactile_checkpoint = bool(
            checkpoint_modalities["tactile_rgb"] or checkpoint_modalities["tactile_marker"]
        )
        if tactile_checkpoint and observation.protocol_version != PROTOCOL_VERSION_V2:
            raise ValueError("This checkpoint requires protocol v2 tactile observations")
        if observation.protocol_version == PROTOCOL_VERSION_V2:
            expected_contract = self.contract.get("combined_sha256")
            if tactile_checkpoint and observation.contract_sha256 is None:
                raise ValueError("Protocol v2 tactile request must include contract_sha256")
            if observation.contract_sha256 is not None and observation.contract_sha256 != expected_contract:
                raise ValueError(
                    "Tactile deployment contract mismatch: "
                    f"server={expected_contract}, request={observation.contract_sha256}"
                )

        for modality in ("tactile_rgb", "tactile_marker"):
            if present[modality] and not checkpoint_modalities[modality]:
                raise ValueError(f"Checkpoint does not contain the requested {modality} branch")
            if force_mask[modality] and not checkpoint_modalities[modality]:
                raise ValueError(f"Cannot force-mask absent checkpoint branch {modality}")
            if checkpoint_modalities[modality] and not present[modality]:
                if not (self.allow_missing_tactile and safe_ablation_context):
                    raise ValueError(
                        f"Checkpoint requires {modality}, but the request omitted it; "
                        "missing tactile is allowed only for explicitly enabled synthetic/offline/dry-run evaluation"
                    )
        if any(force_mask.values()):
            if not safe_ablation_context:
                raise ValueError("Tactile force-mask is forbidden for execution requests")
            if not self.allow_tactile_ablation:
                raise ValueError("Server was not started with --allow-tactile-ablation")

        history_steps = int(tactile.get("history_steps", 1))
        if present["tactile_rgb"] and len(observation.tactile_rgb_history) != history_steps:
            raise ValueError(
                f"tactile RGB history mismatch: checkpoint={history_steps}, "
                f"request={len(observation.tactile_rgb_history)}"
            )
        if present["tactile_marker"] and len(observation.marker_flow) != history_steps:
            raise ValueError(
                f"marker history mismatch: checkpoint={history_steps}, request={len(observation.marker_flow)}"
            )

        wrist_timestamp = float(observation.wrist_timestamp or observation.timestamp)
        max_skew_s = float(tactile.get("max_timestamp_skew_s", 0.05))
        timestamp_skew_s: dict[str, float] = {}
        if present["tactile_rgb"]:
            timestamp_skew_s["tactile_rgb_to_wrist"] = abs(
                float(observation.tactile_rgb_timestamps[-1]) - wrist_timestamp
            )
        if present["tactile_marker"]:
            timestamp_skew_s["tactile_marker_to_wrist"] = abs(
                float(observation.marker_timestamps[-1]) - wrist_timestamp
            )
        if timestamp_skew_s and max(timestamp_skew_s.values()) > max_skew_s:
            raise ValueError(
                f"Tactile/wrist timestamp skew {max(timestamp_skew_s.values()):.4f}s exceeds "
                f"checkpoint limit {max_skew_s:.4f}s"
            )
        effective = {
            modality: bool(present[modality] and not force_mask[modality])
            for modality in ("tactile_rgb", "tactile_marker")
        }
        return {
            "checkpoint_modalities": checkpoint_modalities,
            "present_modalities": present,
            "effective_modalities": {"wrist_rgb": True, **effective},
            "force_mask": force_mask,
            "safe_ablation_context": safe_ablation_context,
            "timestamp_skew_s": timestamp_skew_s,
            "history_steps": history_steps,
            "history_stride": int(tactile.get("history_stride", 1)),
            "marker_normalization": tactile.get("marker_normalization", "image_size_xy"),
        }

    def warmup(self) -> dict[str, Any]:
        tactile = dict(self.contract.get("tactile") or {})
        modalities = dict(tactile.get("checkpoint_modalities") or {})
        history_steps = int(tactile.get("history_steps", 1))
        timestamp = time.time()
        rgb_enabled = bool(modalities.get("tactile_rgb", False))
        marker_enabled = bool(modalities.get("tactile_marker", False))
        observation = Observation(
            instruction="Pull the tissue",
            state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.045], dtype=np.float32),
            wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            metadata={"synthetic": True, "episode_reset": True},
            protocol_version=PROTOCOL_VERSION_V2 if (rgb_enabled or marker_enabled) else PROTOCOL_VERSION_V1,
            contract_sha256=self.contract.get("combined_sha256") if (rgb_enabled or marker_enabled) else None,
            timestamp=timestamp,
            wrist_timestamp=timestamp if (rgb_enabled or marker_enabled) else None,
            tactile_rgb_history=(
                np.zeros((history_steps, 224, 224, 3), dtype=np.uint8) if rgb_enabled else None
            ),
            tactile_rgb_timestamps=(
                np.full((history_steps,), timestamp, dtype=np.float64) if rgb_enabled else None
            ),
            tactile_rgb_history_mask=(
                np.ones((history_steps,), dtype=np.bool_) if rgb_enabled else None
            ),
            marker_flow=(
                np.zeros(
                    (
                        history_steps,
                        int(tactile.get("marker_count", TACTILE_MARKER_COUNT)),
                        int(tactile.get("marker_dim", TACTILE_MARKER_DIM)),
                    ),
                    dtype=np.float32,
                )
                if marker_enabled
                else None
            ),
            marker_valid_mask=(
                np.ones(
                    (history_steps, int(tactile.get("marker_count", TACTILE_MARKER_COUNT))),
                    dtype=np.bool_,
                )
                if marker_enabled
                else None
            ),
            marker_timestamps=(
                np.full((history_steps,), timestamp, dtype=np.float64) if marker_enabled else None
            ),
            marker_history_mask=(
                np.ones((history_steps,), dtype=np.bool_) if marker_enabled else None
            ),
        )
        started = time.time()
        response = self.predict(observation)
        return {
            "warmup_s": time.time() - started,
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
    parser.add_argument("--api-key", default=os.environ.get("LINGBOT_V2_API_KEY"))
    parser.add_argument("--max-body-mb", type=_positive_float, default=8.0)
    parser.add_argument("--inference-lock-timeout", type=_positive_float, default=5.0)
    parser.add_argument(
        "--allow-missing-tactile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow required tactile branches to be absent only in synthetic/offline/dry-run requests.",
    )
    parser.add_argument(
        "--allow-tactile-ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow force-masking checkpoint tactile branches only in synthetic/offline/dry-run requests.",
    )
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
        allow_missing_tactile=args.allow_missing_tactile,
        allow_tactile_ablation=args.allow_tactile_ablation,
    )
    print(json.dumps(backend.health(), indent=2, ensure_ascii=False), flush=True)
    if args.warmup:
        print(json.dumps(backend.warmup(), indent=2, ensure_ascii=False), flush=True)
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
    model = training.get("model") or {}
    train = training.get("train") or {}
    data_name = data.get("data_name")
    if data_name not in {ROBOT_CONFIG, TACTILE_ROBOT_CONFIG}:
        raise RuntimeError(
            f"training data.data_name must be {ROBOT_CONFIG!r} or {TACTILE_ROBOT_CONFIG!r}, "
            f"got {data_name!r}"
        )
    _require_equal(robot_config_path.stem, data_name, "robot config/checkpoint data name")
    _require_equal(data.get("cameras"), ["camera_wrist_left"], "training data.cameras")
    for key, expected in (
        ("chunk_size", 50),
        ("action_dim", 55),
        ("max_action_dim", 55),
        ("max_state_dim", 55),
    ):
        _require_equal(train.get(key), expected, f"training train.{key}")

    tactile_rgb_enabled = _config_bool(
        train.get("tactile_rgb_enabled", False), "training train.tactile_rgb_enabled"
    )
    tactile_marker_enabled = _config_bool(
        train.get("tactile_marker_enabled", False), "training train.tactile_marker_enabled"
    )
    expected_tactile_mode = (
        "rgb-marker"
        if tactile_rgb_enabled and tactile_marker_enabled
        else "rgb"
        if tactile_rgb_enabled
        else "marker"
        if tactile_marker_enabled
        else "none"
    )
    tactile_train_stage = str(train.get("tactile_train_stage", "full"))
    if tactile_train_stage not in {"adapters", "expert", "full"}:
        raise RuntimeError(
            "training train.tactile_train_stage must be adapters, expert, or full, "
            f"got {tactile_train_stage!r}"
        )
    tactile_params = train.get("tactile_params") or {}
    if not isinstance(tactile_params, dict):
        raise RuntimeError("training train.tactile_params must be a mapping")
    history_steps = _positive_config_int(
        tactile_params.get(
            "history_steps",
            train.get("tactile_temporal_horizon", train.get("temporal_horizon", 1)),
        ),
        "training train.tactile_params.history_steps",
    )
    history_stride = _positive_config_int(
        tactile_params.get("history_stride", 1),
        "training train.tactile_params.history_stride",
    )
    history_frequency_hz = float(tactile_params.get("history_frequency_hz", 30.0))
    if not np.isfinite(history_frequency_hz) or history_frequency_hz <= 0.0:
        raise RuntimeError(
            "training train.tactile_params.history_frequency_hz must be a positive finite value"
        )
    marker_count = _positive_config_int(
        tactile_params.get("marker_count", TACTILE_MARKER_COUNT),
        "training train.tactile_params.marker_count",
    )
    marker_dim = _positive_config_int(
        tactile_params.get("marker_dim", TACTILE_MARKER_DIM),
        "training train.tactile_params.marker_dim",
    )
    if marker_count != TACTILE_MARKER_COUNT or marker_dim != TACTILE_MARKER_DIM:
        raise RuntimeError(
            "Deployment protocol currently requires marker shape "
            f"[{TACTILE_MARKER_COUNT},{TACTILE_MARKER_DIM}], got [{marker_count},{marker_dim}]"
        )
    marker_normalization = str(tactile_params.get("marker_normalization", "image_size_xy"))
    if marker_normalization != "image_size_xy":
        raise RuntimeError(
            "Unsupported marker normalization for online deployment: "
            f"expected 'image_size_xy', got {marker_normalization!r}"
        )
    marker_normalization_size_xy = tactile_params.get("marker_normalization_size_xy", [640, 480])
    if marker_normalization_size_xy != [640, 480]:
        raise RuntimeError(
            "TacThru ML48 online deployment requires marker_normalization_size_xy=[640, 480], "
            f"got {marker_normalization_size_xy!r}"
        )
    missing_policy = str(tactile_params.get("missing_policy", "mask"))
    if missing_policy != "mask":
        raise RuntimeError(
            f"Online optional tactile payload requires missing_policy='mask', got {missing_policy!r}"
        )
    max_timestamp_skew_s = float(tactile_params.get("max_timestamp_skew_s", 0.05))
    if not np.isfinite(max_timestamp_skew_s) or max_timestamp_skew_s <= 0.0:
        raise RuntimeError(
            "training train.tactile_params.max_timestamp_skew_s must be a positive finite value"
        )
    tactile_rgb_key = str(data.get("tactile_rgb_key", "observation.images.tactile_left"))
    tactile_marker_key = str(
        data.get("tactile_marker_key", "observation.tactile.marker_flow_left")
    )
    tactile_marker_valid_key = str(
        data.get("tactile_marker_valid_key", "observation.tactile.marker_valid_left")
    )
    if tactile_rgb_enabled and not tactile_rgb_key.strip():
        raise RuntimeError("Tactile RGB checkpoint is missing data.tactile_rgb_key")
    if tactile_marker_enabled and (
        not tactile_marker_key.strip() or not tactile_marker_valid_key.strip()
    ):
        raise RuntimeError("Tactile marker checkpoint is missing marker data keys")

    tactile_experiment = None
    tactile_experiment_path = None
    dataset_manifest_snapshot = None
    if data_name == TACTILE_ROBOT_CONFIG:
        tactile_experiment_path = training_config_path.parent / "tactile_experiment.json"
        if not tactile_experiment_path.is_file():
            raise FileNotFoundError(
                "Tactile checkpoint is missing its immutable experiment contract: "
                f"{tactile_experiment_path}"
            )
        tactile_experiment = json.loads(
            tactile_experiment_path.read_text(encoding="utf-8")
        )
        expected_experiment_contract = _tactile_experiment_contract_sha256(
            tactile_experiment
        )
        _require_equal(
            tactile_experiment.get("contract_sha256"),
            expected_experiment_contract,
            "tactile experiment self-hash",
        )
        _require_equal(
            tactile_experiment.get("schema_version"),
            1,
            "tactile experiment schema version",
        )
        _require_equal(
            tactile_experiment.get("dataset_tactile_mode"),
            "rgb-marker",
            "tactile experiment dataset superset mode",
        )
        _require_equal(
            tactile_experiment.get("train_tactile_mode"),
            expected_tactile_mode,
            "tactile experiment/checkpoint modality mode",
        )
        _require_equal(
            tactile_experiment.get("tactile_train_stage"),
            tactile_train_stage,
            "tactile experiment/checkpoint training stage",
        )
        experiment_dataset = tactile_experiment.get("lerobot_dataset")
        training_dataset = data.get("train_path")
        if not experiment_dataset or not training_dataset:
            raise RuntimeError(
                "Tactile experiment and training config must both record the LeRobot dataset path"
            )
        _require_equal(
            _resolve_project_path(project_root, Path(str(experiment_dataset))),
            _resolve_project_path(project_root, Path(str(training_dataset))),
            "tactile experiment/checkpoint dataset path",
        )
        experiment_initial_model = tactile_experiment.get("initial_model")
        training_initial_model = model.get("model_path")
        if not experiment_initial_model or not training_initial_model:
            raise RuntimeError(
                "Tactile experiment and training config must both record the initialization model path"
            )
        _require_equal(
            _resolve_project_path(project_root, Path(str(experiment_initial_model))),
            _resolve_project_path(project_root, Path(str(training_initial_model))),
            "tactile experiment/checkpoint initialization model path",
        )
        _require_equal(
            _resolve_project_path(
                project_root, Path(str(tactile_experiment.get("norm_stats")))
            ),
            norm_stats,
            "tactile experiment/server norm path",
        )
        _require_equal(
            train.get("tactile_experiment_contract_sha256"),
            expected_experiment_contract,
            "training tactile experiment contract",
        )
        dataset_manifest_sha256 = _require_sha256_value(
            tactile_experiment.get("dataset_manifest_sha256"),
            "tactile experiment dataset manifest SHA256",
        )
        _require_equal(
            train.get("tactile_dataset_manifest_sha256"),
            dataset_manifest_sha256,
            "training tactile dataset manifest SHA256",
        )
        dataset_manifest_snapshot = training_config_path.parent / "tactile_dataset_manifest.json"
        if not dataset_manifest_snapshot.is_file():
            raise FileNotFoundError(
                "Tactile checkpoint is missing its dataset manifest snapshot: "
                f"{dataset_manifest_snapshot}"
            )
        _require_equal(
            _resolve_project_path(
                project_root,
                Path(str(tactile_experiment.get("dataset_manifest_snapshot"))),
            ),
            dataset_manifest_snapshot,
            "tactile experiment dataset manifest snapshot path",
        )
        _require_equal(
            _sha256(dataset_manifest_snapshot),
            dataset_manifest_sha256,
            "tactile dataset manifest snapshot SHA256",
        )
        _require_equal(
            tactile_experiment.get("norm_stats_sha256"),
            _sha256(norm_stats),
            "tactile experiment norm SHA256",
        )
        rgb_backbone_sha256 = tactile_experiment.get("rgb_backbone_sha256")
        if tactile_rgb_enabled:
            rgb_backbone_sha256 = _require_sha256_value(
                rgb_backbone_sha256,
                "tactile RGB backbone SHA256",
            )
            _require_equal(
                train.get("tactile_rgb_backbone_sha256"),
                rgb_backbone_sha256,
                "training tactile RGB backbone SHA256",
            )
        elif train.get("tactile_rgb_backbone_sha256") not in (None, ""):
            raise RuntimeError(
                "Marker/wrist-only tactile checkpoint unexpectedly records an RGB backbone SHA256"
            )

    expected_norm_from_training = _resolve_project_path(project_root, Path(str(data.get("norm_stats_file"))))
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
    if tactile_experiment_path is not None:
        hashes["tactile_experiment_sha256"] = _sha256(tactile_experiment_path)
    if dataset_manifest_snapshot is not None:
        hashes["tactile_dataset_manifest_sha256"] = _sha256(
            dataset_manifest_snapshot
        )
    combined = hashlib.sha256()
    for key in sorted(hashes):
        combined.update(key.encode("utf-8"))
        combined.update(hashes[key].encode("ascii"))
    return {
        "version": 2 if (tactile_rgb_enabled or tactile_marker_enabled) else 1,
        **hashes,
        "combined_sha256": combined.hexdigest(),
        "robot_default_norm_stats": str(expected_norm_from_robot),
        "robot_default_norm_overridden": robot_default_norm_overridden,
        "tactile": {
            "input_schema_version": int(tactile_params.get("schema_version", 1)),
            "checkpoint_modalities": {
                "wrist_rgb": True,
                "tactile_rgb": tactile_rgb_enabled,
                "tactile_marker": tactile_marker_enabled,
            },
            "enabled_modalities": [
                name
                for name, enabled in (
                    ("wrist_rgb", True),
                    ("tactile_rgb", tactile_rgb_enabled),
                    ("tactile_marker", tactile_marker_enabled),
                )
                if enabled
            ],
            "history_steps": history_steps,
            "history_stride": history_stride,
            "history_frequency_hz": history_frequency_hz,
            "max_timestamp_skew_s": max_timestamp_skew_s,
            "marker_count": marker_count,
            "marker_dim": marker_dim,
            "marker_normalization": marker_normalization,
            "marker_normalization_size_xy": marker_normalization_size_xy,
            "missing_policy": missing_policy,
            "experiment_contract_sha256": (
                tactile_experiment.get("contract_sha256")
                if tactile_experiment is not None
                else None
            ),
            "dataset_manifest_sha256": (
                tactile_experiment.get("dataset_manifest_sha256")
                if tactile_experiment is not None
                else None
            ),
            "rgb_backbone_sha256": (
                tactile_experiment.get("rgb_backbone_sha256")
                if tactile_experiment is not None
                else None
            ),
            "data_keys": {
                "tactile_rgb": tactile_rgb_key,
                "tactile_marker": tactile_marker_key,
                "tactile_marker_valid": tactile_marker_valid_key,
            },
        },
    }


def _tactile_experiment_contract_sha256(payload: dict[str, Any]) -> str:
    contract_fields = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "dataset_tactile_mode",
            "train_tactile_mode",
            "tactile_train_stage",
            "source",
            "lerobot_dataset",
            "norm_stats",
            "initial_model",
            "base_model_assets",
            "train_overrides",
            "config_sha256",
            "dataset_manifest_sha256",
            "norm_stats_sha256",
            "rgb_backbone_sha256",
            "git_head",
        )
    }
    return hashlib.sha256(
        json.dumps(contract_fields, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a hexadecimal SHA256 string, got {value!r}")
    digest = value.strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"{name} must contain exactly 64 hexadecimal characters")
    return digest


def _mapping_entry(entries: Any, key: str) -> dict[str, Any]:
    if not isinstance(entries, list):
        raise RuntimeError(f"Robot mapping section for {key} must be a list")
    matches = [entry[key] for entry in entries if isinstance(entry, dict) and key in entry]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one robot mapping for {key}, got {len(matches)}")
    if len(entries) != 2:
        raise RuntimeError(f"Expected exactly two entries in mapping section containing {key}, got {len(entries)}")
    return matches[0]


def _checkpoint_robot_config_name(checkpoint: Path) -> str:
    training_config_path = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Training config expected by official loader is missing: {training_config_path}")
    with training_config_path.open("r", encoding="utf-8") as file:
        training = yaml.safe_load(file) or {}
    data_name = (training.get("data") or {}).get("data_name")
    if data_name not in {ROBOT_CONFIG, TACTILE_ROBOT_CONFIG}:
        raise RuntimeError(
            f"Unsupported checkpoint data.data_name {data_name!r}; expected "
            f"{ROBOT_CONFIG!r} or {TACTILE_ROBOT_CONFIG!r}"
        )
    return str(data_name)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _config_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{name} must be a boolean, got {value!r}")
    return value


def _positive_config_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0 or result != value:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    return result


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
