from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .transforms import ACTION_DIM, validate_action_chunk, validate_state8


PROTOCOL_NAME = "lingbot-vla-v2-tacthru-umi"
PROTOCOL_VERSION = 1
PROTOCOL_VERSION_V1 = 1
PROTOCOL_VERSION_V2 = 2
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2)
CAMERA_KEY = "observation.images.camera_wrist_left"
TACTILE_RGB_KEY = "observation.images.tactile_left"
TACTILE_MARKER_KEY = "observation.tactile.marker_flow_left"
TACTILE_MARKER_VALID_KEY = "observation.tactile.marker_valid_left"
ROBOT_CONFIG = "tacthru_umi_v2"
TACTILE_ROBOT_CONFIG = "tacthru_umi_v2_tactile"
POSE_FRAME = "episode_start_new_tcp"
POSE_SEMANTICS = "absolute_xyz_quaternion_xyzw_gripper_width_m"
IMAGE_COLOR_SPACE = "rgb"
IMAGE_ENCODING = "jpeg"
TACTILE_ENABLED = False
MAX_TACTILE_HISTORY = 16
TACTILE_MARKER_COUNT = 48
TACTILE_MARKER_DIM = 2
CONTROL_FREQUENCY_HZ = 30.0
MAX_INSTRUCTION_CHARS = 4096
MAX_IMAGE_PIXELS = 20_000_000
WIRE_IMAGE_HEIGHT = 224
WIRE_IMAGE_WIDTH = 224


@dataclass(frozen=True)
class Observation:
    instruction: str
    state: np.ndarray
    wrist_rgb: np.ndarray
    control_frequency_hz: float = 30.0
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION_V1
    contract_sha256: str | None = None
    wrist_timestamp: float | None = None
    tactile_rgb_history: np.ndarray | None = None
    tactile_rgb_timestamps: np.ndarray | None = None
    tactile_rgb_history_mask: np.ndarray | None = None
    marker_flow: np.ndarray | None = None
    marker_valid_mask: np.ndarray | None = None
    marker_timestamps: np.ndarray | None = None
    marker_history_mask: np.ndarray | None = None
    force_mask_tactile_rgb: bool = False
    force_mask_tactile_marker: bool = False


@dataclass(frozen=True)
class ActionResponse:
    action_chunk: np.ndarray
    request_id: str
    session_id: str
    metadata: dict[str, Any]
    server_timestamp: float | None = None
    protocol_version: int = PROTOCOL_VERSION_V1


def observation_to_payload(obs: Observation, *, jpeg_quality: int = 90) -> dict[str, Any]:
    instruction = _validate_instruction(obs.instruction)
    state = validate_state8(obs.state)
    frequency = _validate_frequency(obs.control_frequency_hz)
    session_id = _validate_identifier(obs.session_id, "session_id")
    request_id = _validate_identifier(obs.request_id, "request_id")
    protocol_version = _validate_protocol_version(obs.protocol_version)
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": protocol_version,
        "request_id": request_id,
        "session_id": session_id,
        "timestamp": _validate_timestamp(obs.timestamp),
        "instruction": instruction,
        "control_frequency_hz": frequency,
        "state": state.astype(float).tolist(),
        "images": {
            CAMERA_KEY: encode_rgb_image(obs.wrist_rgb, jpeg_quality=jpeg_quality),
        },
        "metadata": _validate_metadata(obs.metadata),
    }
    if protocol_version == PROTOCOL_VERSION_V1:
        _reject_v1_tactile_fields(obs)
        return payload

    payload["schema_version"] = PROTOCOL_VERSION_V2
    payload["contract_sha256"] = _validate_optional_sha256(obs.contract_sha256, "contract_sha256")
    payload["wrist_timestamp"] = _validate_timestamp(
        obs.timestamp if obs.wrist_timestamp is None else obs.wrist_timestamp
    )

    rgb_history, rgb_timestamps, rgb_history_mask = _validate_tactile_rgb_history(
        obs.tactile_rgb_history,
        obs.tactile_rgb_timestamps,
        obs.tactile_rgb_history_mask,
    )
    marker_flow, marker_valid_mask, marker_timestamps, marker_history_mask = _validate_marker_history(
        obs.marker_flow,
        obs.marker_valid_mask,
        obs.marker_timestamps,
        obs.marker_history_mask,
    )
    if rgb_history is not None:
        payload["tactile_rgb_history"] = [
            encode_rgb_image(image, jpeg_quality=jpeg_quality) for image in rgb_history
        ]
        payload["tactile_rgb_timestamps"] = rgb_timestamps.astype(float).tolist()
        payload["tactile_rgb_history_mask"] = rgb_history_mask.astype(bool).tolist()
    if marker_flow is not None:
        payload["marker_flow"] = marker_flow.astype(float).tolist()
        payload["marker_valid_mask"] = marker_valid_mask.astype(bool).tolist()
        payload["marker_timestamps"] = marker_timestamps.astype(float).tolist()
        payload["marker_history_mask"] = marker_history_mask.astype(bool).tolist()
    payload["modality_presence"] = {
        "wrist_rgb": True,
        "tactile_rgb": rgb_history is not None,
        "tactile_marker": marker_flow is not None,
    }
    payload["force_mask"] = {
        "tactile_rgb": _validate_bool(obs.force_mask_tactile_rgb, "force_mask_tactile_rgb"),
        "tactile_marker": _validate_bool(obs.force_mask_tactile_marker, "force_mask_tactile_marker"),
    }
    return payload


def observation_to_json(obs: Observation, *, jpeg_quality: int = 90) -> bytes:
    return _json_bytes(observation_to_payload(obs, jpeg_quality=jpeg_quality))


def observation_from_payload(payload: dict[str, Any]) -> Observation:
    protocol_version = _validate_protocol(payload)
    images = payload.get("images")
    if not isinstance(images, dict) or CAMERA_KEY not in images:
        raise ValueError(f"Request must contain images[{CAMERA_KEY!r}]")
    unexpected = set(images) - {CAMERA_KEY}
    if unexpected:
        raise ValueError(f"Unexpected image slots: {sorted(unexpected)}")
    metadata = _validate_metadata(payload.get("metadata"))
    if protocol_version == PROTOCOL_VERSION_V1:
        _reject_v1_payload_tactile_fields(payload)
        return Observation(
            instruction=_validate_instruction(payload.get("instruction")),
            state=validate_state8(np.asarray(payload.get("state"), dtype=np.float32)),
            wrist_rgb=decode_rgb_image(images[CAMERA_KEY]),
            control_frequency_hz=_validate_frequency(payload.get("control_frequency_hz")),
            session_id=_validate_identifier(payload.get("session_id"), "session_id"),
            request_id=_validate_identifier(payload.get("request_id"), "request_id"),
            timestamp=_validate_timestamp(payload.get("timestamp")),
            metadata=dict(metadata),
            protocol_version=PROTOCOL_VERSION_V1,
        )

    if payload.get("schema_version") != PROTOCOL_VERSION_V2:
        raise ValueError(
            f"protocol_version {PROTOCOL_VERSION_V2} requires schema_version "
            f"{PROTOCOL_VERSION_V2}, got {payload.get('schema_version')!r}"
        )
    rgb_payload = payload.get("tactile_rgb_history")
    rgb_history = None
    if rgb_payload is not None:
        if not isinstance(rgb_payload, list) or not rgb_payload:
            raise ValueError("tactile_rgb_history must be a non-empty JSON array")
        if len(rgb_payload) > MAX_TACTILE_HISTORY:
            raise ValueError(f"tactile_rgb_history exceeds maximum history {MAX_TACTILE_HISTORY}")
        rgb_history = np.stack([decode_rgb_image(item) for item in rgb_payload], axis=0)
    rgb_history, rgb_timestamps, rgb_history_mask = _validate_tactile_rgb_history(
        rgb_history,
        payload.get("tactile_rgb_timestamps"),
        payload.get("tactile_rgb_history_mask"),
    )
    marker_flow, marker_valid_mask, marker_timestamps, marker_history_mask = _validate_marker_history(
        payload.get("marker_flow"),
        payload.get("marker_valid_mask"),
        payload.get("marker_timestamps"),
        payload.get("marker_history_mask"),
    )
    expected_presence = {
        "wrist_rgb": True,
        "tactile_rgb": rgb_history is not None,
        "tactile_marker": marker_flow is not None,
    }
    if payload.get("modality_presence") != expected_presence:
        raise ValueError(
            "modality_presence does not match the supplied payload: "
            f"expected {expected_presence}, got {payload.get('modality_presence')!r}"
        )
    force_mask = payload.get("force_mask", {})
    if not isinstance(force_mask, dict):
        raise ValueError("force_mask must be a JSON object")
    return Observation(
        instruction=_validate_instruction(payload.get("instruction")),
        state=validate_state8(np.asarray(payload.get("state"), dtype=np.float32)),
        wrist_rgb=decode_rgb_image(images[CAMERA_KEY]),
        control_frequency_hz=_validate_frequency(payload.get("control_frequency_hz")),
        session_id=_validate_identifier(payload.get("session_id"), "session_id"),
        request_id=_validate_identifier(payload.get("request_id"), "request_id"),
        timestamp=_validate_timestamp(payload.get("timestamp")),
        metadata=dict(metadata),
        protocol_version=PROTOCOL_VERSION_V2,
        contract_sha256=_validate_optional_sha256(payload.get("contract_sha256"), "contract_sha256"),
        wrist_timestamp=_validate_timestamp(payload.get("wrist_timestamp")),
        tactile_rgb_history=rgb_history,
        tactile_rgb_timestamps=rgb_timestamps,
        tactile_rgb_history_mask=rgb_history_mask,
        marker_flow=marker_flow,
        marker_valid_mask=marker_valid_mask,
        marker_timestamps=marker_timestamps,
        marker_history_mask=marker_history_mask,
        force_mask_tactile_rgb=_validate_bool(force_mask.get("tactile_rgb", False), "force_mask.tactile_rgb"),
        force_mask_tactile_marker=_validate_bool(
            force_mask.get("tactile_marker", False), "force_mask.tactile_marker"
        ),
    )


def observation_from_json(data: bytes) -> Observation:
    return observation_from_payload(_strict_json_loads(data))


def action_response_to_payload(
    *,
    action_chunk: np.ndarray,
    request_id: str,
    session_id: str,
    metadata: dict[str, Any] | None = None,
    expected_steps: int | None = None,
    protocol_version: int = PROTOCOL_VERSION_V1,
) -> dict[str, Any]:
    actions = validate_action_chunk(action_chunk, expected_steps=expected_steps)
    version = _validate_protocol_version(protocol_version)
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": version,
        "request_id": _validate_identifier(request_id, "request_id"),
        "session_id": _validate_identifier(session_id, "session_id"),
        "server_timestamp": time.time(),
        "action_chunk": actions.astype(float).tolist(),
        "action_spec": action_spec(chunk_size=int(actions.shape[0])),
        "metadata": _validate_metadata(metadata),
    }
    if version == PROTOCOL_VERSION_V2:
        payload["schema_version"] = PROTOCOL_VERSION_V2
    return payload


def action_response_to_json(**kwargs: Any) -> bytes:
    return _json_bytes(action_response_to_payload(**kwargs))


def action_response_from_payload(
    payload: dict[str, Any],
    *,
    expected_request_id: str | None = None,
    expected_session_id: str | None = None,
    expected_steps: int | None = None,
) -> ActionResponse:
    protocol_version = _validate_protocol(payload)
    if protocol_version == PROTOCOL_VERSION_V2 and payload.get("schema_version") != PROTOCOL_VERSION_V2:
        raise ValueError(
            f"response protocol_version {PROTOCOL_VERSION_V2} requires schema_version {PROTOCOL_VERSION_V2}, "
            f"got {payload.get('schema_version')!r}"
        )
    request_id = _validate_identifier(payload.get("request_id"), "request_id")
    session_id = _validate_identifier(payload.get("session_id"), "session_id")
    if expected_request_id is not None and request_id != expected_request_id:
        raise ValueError(f"Response request_id mismatch: expected {expected_request_id!r}, got {request_id!r}")
    if expected_session_id is not None and session_id != expected_session_id:
        raise ValueError(f"Response session_id mismatch: expected {expected_session_id!r}, got {session_id!r}")

    spec = payload.get("action_spec")
    spec_steps = validate_action_spec(spec, expected_steps=expected_steps)
    actions = validate_action_chunk(
        np.asarray(payload.get("action_chunk"), dtype=np.float32),
        expected_steps=spec_steps,
    )
    metadata = _validate_metadata(payload.get("metadata"))
    return ActionResponse(
        action_chunk=actions,
        request_id=request_id,
        session_id=session_id,
        metadata=dict(metadata),
        server_timestamp=float(payload["server_timestamp"]) if payload.get("server_timestamp") is not None else None,
        protocol_version=protocol_version,
    )


def action_response_from_json(data: bytes, **kwargs: Any) -> ActionResponse:
    return action_response_from_payload(_strict_json_loads(data), **kwargs)


def action_spec(*, chunk_size: int) -> dict[str, Any]:
    return {
        "shape": [int(chunk_size), ACTION_DIM],
        "chunk_size": int(chunk_size),
        "pose_frame": POSE_FRAME,
        "pose_semantics": POSE_SEMANTICS,
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "gripper_semantics": "absolute_width",
        "gripper_unit": "m",
    }


def validate_action_spec(spec: Any, *, expected_steps: int | None = None) -> int:
    if not isinstance(spec, dict):
        raise ValueError("Response action_spec must be an object")
    required = {
        "pose_frame": POSE_FRAME,
        "pose_semantics": POSE_SEMANTICS,
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "gripper_semantics": "absolute_width",
        "gripper_unit": "m",
    }
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise ValueError(f"Unsupported action_spec.{key}: expected {expected!r}, got {spec.get(key)!r}")
    chunk_size = spec.get("chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"Response action_spec.chunk_size must be a positive integer, got {chunk_size!r}")
    if expected_steps is not None and chunk_size != int(expected_steps):
        raise ValueError(
            f"Response action_spec chunk_size mismatch: expected {expected_steps}, got {spec.get('chunk_size')}"
        )
    expected_shape = [chunk_size, ACTION_DIM]
    if spec.get("shape") != expected_shape:
        raise ValueError(f"Response action_spec shape mismatch: expected {expected_shape}, got {spec.get('shape')!r}")
    return chunk_size


def encode_rgb_image(image: np.ndarray, *, jpeg_quality: int) -> dict[str, Any]:
    rgb = _validate_rgb_image(image)
    quality = int(jpeg_quality)
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg_quality must be in [1, 100], got {quality}")
    ok, encoded = cv2.imencode(
        ".jpg",
        rgb[..., ::-1],
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return {
        "encoding": IMAGE_ENCODING,
        "color_space": IMAGE_COLOR_SPACE,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "data_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def decode_rgb_image(encoded: Any) -> np.ndarray:
    if not isinstance(encoded, dict):
        raise ValueError("Encoded image must be an object")
    if encoded.get("encoding") != IMAGE_ENCODING or encoded.get("color_space") != IMAGE_COLOR_SPACE:
        raise ValueError(
            f"Unsupported image encoding/color space: {encoded.get('encoding')!r}/{encoded.get('color_space')!r}"
        )
    declared_width = int(encoded.get("width", -1))
    declared_height = int(encoded.get("height", -1))
    if (declared_height, declared_width) != (WIRE_IMAGE_HEIGHT, WIRE_IMAGE_WIDTH):
        raise ValueError(
            f"Wire image must be {WIRE_IMAGE_WIDTH}x{WIRE_IMAGE_HEIGHT}, "
            f"got {declared_width}x{declared_height}"
        )
    try:
        raw = base64.b64decode(str(encoded["data_b64"]).encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 JPEG payload") from exc
    header_height, header_width = _jpeg_dimensions(raw)
    if (header_height, header_width) != (declared_height, declared_width):
        raise ValueError(
            "JPEG header size does not match declaration: "
            f"declared={(declared_height, declared_width)}, header={(header_height, header_width)}"
        )
    if header_height * header_width > MAX_IMAGE_PIXELS:
        raise ValueError(f"JPEG dimensions exceed pixel limit: {(header_height, header_width)}")
    array = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    _validate_rgb_image(rgb)
    if (declared_height, declared_width) != rgb.shape[:2]:
        raise ValueError(
            "Decoded image size does not match declaration: "
            f"declared={(declared_height, declared_width)}, actual={rgb.shape[:2]}"
        )
    return rgb


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB image with shape (H, W, 3), got {rgb.dtype} {rgb.shape}")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[0] * rgb.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"Invalid image dimensions: {rgb.shape}")
    if rgb.shape[:2] != (WIRE_IMAGE_HEIGHT, WIRE_IMAGE_WIDTH):
        raise ValueError(
            f"Wire RGB image must have shape ({WIRE_IMAGE_HEIGHT}, {WIRE_IMAGE_WIDTH}, 3), got {rgb.shape}"
        )
    return np.ascontiguousarray(rgb)


def _validate_protocol(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"Unsupported protocol: {payload.get('protocol')!r}")
    return _validate_protocol_version(payload.get("protocol_version"))


def _validate_protocol_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"protocol_version must be an integer, got {value!r}")
    version = int(value)
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(
            f"Unsupported protocol_version: expected one of {list(SUPPORTED_PROTOCOL_VERSIONS)}, got {version!r}"
        )
    return version


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a JSON boolean")
    return bool(value)


def _validate_optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a hexadecimal SHA256 string or null")
    digest = value.strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return digest


def _validate_tactile_rgb_history(
    history: Any,
    timestamps: Any,
    history_mask: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    supplied = (history is not None, timestamps is not None, history_mask is not None)
    if not any(supplied):
        return None, None, None
    if not all(supplied):
        raise ValueError(
            "tactile_rgb_history, tactile_rgb_timestamps, and tactile_rgb_history_mask "
            "must be supplied together"
        )
    rgb = np.asarray(history)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (
        WIRE_IMAGE_HEIGHT,
        WIRE_IMAGE_WIDTH,
        3,
    ):
        raise ValueError(
            "tactile_rgb_history must be uint8 with shape "
            f"[K,{WIRE_IMAGE_HEIGHT},{WIRE_IMAGE_WIDTH},3], got {rgb.dtype} {rgb.shape}"
        )
    steps = int(rgb.shape[0])
    if not 1 <= steps <= MAX_TACTILE_HISTORY:
        raise ValueError(f"tactile RGB history K must be in [1,{MAX_TACTILE_HISTORY}], got {steps}")
    ts = _validate_timestamp_vector(timestamps, steps, "tactile_rgb_timestamps")
    mask = _validate_mask_vector(history_mask, steps, "tactile_rgb_history_mask")
    return np.ascontiguousarray(rgb), ts, mask


def _validate_marker_history(
    marker_flow: Any,
    marker_valid_mask: Any,
    marker_timestamps: Any,
    marker_history_mask: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    supplied = (
        marker_flow is not None,
        marker_valid_mask is not None,
        marker_timestamps is not None,
        marker_history_mask is not None,
    )
    if not any(supplied):
        return None, None, None, None
    if not all(supplied):
        raise ValueError(
            "marker_flow, marker_valid_mask, marker_timestamps, and marker_history_mask "
            "must be supplied together"
        )
    flow = np.asarray(marker_flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[1:] != (TACTILE_MARKER_COUNT, TACTILE_MARKER_DIM):
        raise ValueError(
            "marker_flow must have shape "
            f"[K,{TACTILE_MARKER_COUNT},{TACTILE_MARKER_DIM}], got {flow.shape}"
        )
    steps = int(flow.shape[0])
    if not 1 <= steps <= MAX_TACTILE_HISTORY:
        raise ValueError(f"marker history K must be in [1,{MAX_TACTILE_HISTORY}], got {steps}")
    if not np.isfinite(flow).all():
        raise ValueError("marker_flow contains non-finite values; use zero plus marker_valid_mask=false")
    valid = np.asarray(marker_valid_mask)
    if valid.dtype != np.bool_ or valid.shape != (steps, TACTILE_MARKER_COUNT):
        raise ValueError(
            f"marker_valid_mask must be bool with shape [{steps},{TACTILE_MARKER_COUNT}], "
            f"got {valid.dtype} {valid.shape}"
        )
    ts = _validate_timestamp_vector(marker_timestamps, steps, "marker_timestamps")
    history_valid = _validate_mask_vector(marker_history_mask, steps, "marker_history_mask")
    if np.any(valid & ~history_valid[:, None]):
        raise ValueError("marker_valid_mask must be false for padded/invalid marker history steps")
    return np.ascontiguousarray(flow), np.ascontiguousarray(valid), ts, history_valid


def _validate_timestamp_vector(value: Any, steps: int, name: str) -> np.ndarray:
    timestamps = np.asarray(value, dtype=np.float64)
    if timestamps.shape != (steps,):
        raise ValueError(f"{name} must have shape [{steps}], got {timestamps.shape}")
    if not np.isfinite(timestamps).all() or np.any(timestamps <= 0.0):
        raise ValueError(f"{name} must contain positive finite Unix timestamps")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{name} must be chronological (non-decreasing)")
    return np.ascontiguousarray(timestamps)


def _validate_mask_vector(value: Any, steps: int, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (steps,):
        raise ValueError(f"{name} must be bool with shape [{steps}], got {mask.dtype} {mask.shape}")
    if not bool(mask[-1]):
        raise ValueError(f"{name} latest step must be valid")
    seen_valid = False
    for item in mask:
        if item:
            seen_valid = True
        elif seen_valid:
            raise ValueError(f"{name} may only contain left-padding false values")
    return np.ascontiguousarray(mask)


def _reject_v1_tactile_fields(obs: Observation) -> None:
    populated = {
        "contract_sha256": obs.contract_sha256,
        "wrist_timestamp": obs.wrist_timestamp,
        "tactile_rgb_history": obs.tactile_rgb_history,
        "tactile_rgb_timestamps": obs.tactile_rgb_timestamps,
        "tactile_rgb_history_mask": obs.tactile_rgb_history_mask,
        "marker_flow": obs.marker_flow,
        "marker_valid_mask": obs.marker_valid_mask,
        "marker_timestamps": obs.marker_timestamps,
        "marker_history_mask": obs.marker_history_mask,
    }
    names = [name for name, value in populated.items() if value is not None]
    if obs.force_mask_tactile_rgb or obs.force_mask_tactile_marker:
        names.append("force_mask")
    if names:
        raise ValueError(f"Protocol v1 does not support tactile fields: {sorted(names)}")


def _reject_v1_payload_tactile_fields(payload: dict[str, Any]) -> None:
    tactile_fields = {
        "schema_version",
        "contract_sha256",
        "wrist_timestamp",
        "tactile_rgb_history",
        "tactile_rgb_timestamps",
        "tactile_rgb_history_mask",
        "marker_flow",
        "marker_valid_mask",
        "marker_timestamps",
        "marker_history_mask",
        "modality_presence",
        "force_mask",
    }
    unexpected = sorted(tactile_fields.intersection(payload))
    if unexpected:
        raise ValueError(f"Protocol v1 request contains v2 tactile fields: {unexpected}")


def _validate_instruction(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("instruction must be a JSON string")
    instruction = value.strip()
    if not instruction:
        raise ValueError("instruction must be non-empty")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise ValueError(f"instruction exceeds {MAX_INSTRUCTION_CHARS} characters")
    return instruction


def _validate_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a JSON string")
    identifier = value.strip()
    if not identifier or len(identifier) > 128:
        raise ValueError(f"{name} must be a non-empty string with at most 128 characters")
    return identifier


def _validate_frequency(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("control_frequency_hz must be a number, not a boolean")
    frequency = float(value)
    if not np.isfinite(frequency) or frequency <= 0.0 or frequency > 240.0:
        raise ValueError(f"control_frequency_hz must be in (0, 240], got {frequency}")
    return frequency


def _validate_timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timestamp must be a number, not a boolean")
    timestamp = float(value)
    if not np.isfinite(timestamp) or timestamp <= 0.0:
        raise ValueError(f"timestamp must be a positive finite Unix timestamp, got {timestamp}")
    return timestamp


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    metadata = dict(value)
    for key in ("episode_reset", "synthetic", "offline_evaluation", "dry_run", "execute"):
        if key in metadata and not isinstance(metadata[key], bool):
            raise ValueError(f"metadata.{key} must be a JSON boolean")
    return metadata


def _strict_json_loads(data: bytes) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"Non-standard JSON constant is not allowed: {value}")

    payload = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    return payload


def _jpeg_dimensions(raw: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions before OpenCV allocates the decoded image."""

    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        raise ValueError("Invalid JPEG start marker")
    index = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while index < len(raw):
        while index < len(raw) and raw[index] != 0xFF:
            index += 1
        while index < len(raw) and raw[index] == 0xFF:
            index += 1
        if index >= len(raw):
            break
        marker = raw[index]
        index += 1
        if marker in {0xD8, 0xD9, 0x01, *range(0xD0, 0xD8)}:
            continue
        if index + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(raw):
            raise ValueError("Invalid JPEG segment length")
        if marker in sof_markers:
            if segment_length < 7:
                raise ValueError("Invalid JPEG SOF segment")
            height = int.from_bytes(raw[index + 3 : index + 5], "big")
            width = int.from_bytes(raw[index + 5 : index + 7], "big")
            if height <= 0 or width <= 0:
                raise ValueError("Invalid JPEG dimensions")
            return height, width
        index += segment_length
    raise ValueError("JPEG SOF marker not found")
