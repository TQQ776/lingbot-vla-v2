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
PROTOCOL_VERSION = 3
CAMERA_KEY = "observation.images.camera_wrist_left"
TACTILE_RGB_KEY = "observation.images.tactile_left"
MARKER_DISPLACEMENT_KEY = "observation.tactile.marker_displacement_left"
MARKER_VALID_KEY = "observation.tactile.marker_valid_left"
ROBOT_CONFIG = "tacthru_umi_v2"
POSE_FRAME = "episode_start_new_tcp"
POSE_SEMANTICS = "absolute_xyz_quaternion_xyzw_gripper_width_m"
IMAGE_COLOR_SPACE = "rgb"
IMAGE_ENCODING = "jpeg"
TACTILE_SENSOR_COUNT = 1
TACTILE_MARKER_COUNT = 48
TACTILE_MARKER_HISTORY_LENGTH = 8
TACTILE_MARKER_SAMPLE_HZ = 30.0
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
    tactile_rgb: np.ndarray | None = None
    marker_displacement_history: np.ndarray | None = None
    marker_valid_mask: np.ndarray | None = None
    marker_history_valid_mask: np.ndarray | None = None
    tactile_sensor_mask: np.ndarray | None = None
    control_frequency_hz: float = 30.0
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResponse:
    action_chunk: np.ndarray
    request_id: str
    session_id: str
    metadata: dict[str, Any]
    server_timestamp: float | None = None


def observation_to_payload(obs: Observation, *, jpeg_quality: int = 90) -> dict[str, Any]:
    instruction = _validate_instruction(obs.instruction)
    state = validate_state8(obs.state)
    frequency = _validate_frequency(obs.control_frequency_hz)
    session_id = _validate_identifier(obs.session_id, "session_id")
    request_id = _validate_identifier(obs.request_id, "request_id")
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "session_id": session_id,
        "timestamp": _validate_timestamp(obs.timestamp),
        "instruction": instruction,
        "control_frequency_hz": frequency,
        "state": state.astype(float).tolist(),
        "images": {
            CAMERA_KEY: encode_rgb_image(
                obs.wrist_rgb,
                jpeg_quality=jpeg_quality,
                expected_shape=(WIRE_IMAGE_HEIGHT, WIRE_IMAGE_WIDTH, 3),
            ),
        },
        "metadata": _validate_metadata(obs.metadata),
    }
    tactile = _tactile_to_payload(obs, jpeg_quality=jpeg_quality)
    if tactile is not None:
        payload["tactile"] = tactile
    return payload


def observation_to_json(obs: Observation, *, jpeg_quality: int = 90) -> bytes:
    return _json_bytes(observation_to_payload(obs, jpeg_quality=jpeg_quality))


def observation_from_payload(payload: dict[str, Any]) -> Observation:
    _validate_protocol(payload)
    images = payload.get("images")
    if not isinstance(images, dict) or CAMERA_KEY not in images:
        raise ValueError(f"Request must contain images[{CAMERA_KEY!r}]")
    unexpected = set(images) - {CAMERA_KEY}
    if unexpected:
        raise ValueError(f"Unexpected image slots: {sorted(unexpected)}")
    metadata = _validate_metadata(payload.get("metadata"))
    tactile = _tactile_from_payload(payload.get("tactile"))
    return Observation(
        instruction=_validate_instruction(payload.get("instruction")),
        state=validate_state8(np.asarray(payload.get("state"), dtype=np.float32)),
        wrist_rgb=decode_rgb_image(
            images[CAMERA_KEY],
            expected_shape=(WIRE_IMAGE_HEIGHT, WIRE_IMAGE_WIDTH, 3),
        ),
        **tactile,
        control_frequency_hz=_validate_frequency(payload.get("control_frequency_hz")),
        session_id=_validate_identifier(payload.get("session_id"), "session_id"),
        request_id=_validate_identifier(payload.get("request_id"), "request_id"),
        timestamp=_validate_timestamp(payload.get("timestamp")),
        metadata=dict(metadata),
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
) -> dict[str, Any]:
    actions = validate_action_chunk(action_chunk, expected_steps=expected_steps)
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": _validate_identifier(request_id, "request_id"),
        "session_id": _validate_identifier(session_id, "session_id"),
        "server_timestamp": time.time(),
        "action_chunk": actions.astype(float).tolist(),
        "action_spec": action_spec(chunk_size=int(actions.shape[0])),
        "metadata": _validate_metadata(metadata),
    }


def action_response_to_json(**kwargs: Any) -> bytes:
    return _json_bytes(action_response_to_payload(**kwargs))


def action_response_from_payload(
    payload: dict[str, Any],
    *,
    expected_request_id: str | None = None,
    expected_session_id: str | None = None,
    expected_steps: int | None = None,
) -> ActionResponse:
    _validate_protocol(payload)
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


def encode_rgb_image(
    image: np.ndarray,
    *,
    jpeg_quality: int,
    expected_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    rgb = _validate_rgb_image(image, expected_shape=expected_shape)
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


def decode_rgb_image(
    encoded: Any,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if not isinstance(encoded, dict):
        raise ValueError("Encoded image must be an object")
    if encoded.get("encoding") != IMAGE_ENCODING or encoded.get("color_space") != IMAGE_COLOR_SPACE:
        raise ValueError(
            f"Unsupported image encoding/color space: {encoded.get('encoding')!r}/{encoded.get('color_space')!r}"
        )
    declared_width = int(encoded.get("width", -1))
    declared_height = int(encoded.get("height", -1))
    if expected_shape is not None and (declared_height, declared_width, 3) != expected_shape:
        raise ValueError(
            f"Wire image must have shape {expected_shape}, "
            f"got {(declared_height, declared_width, 3)}"
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
    _validate_rgb_image(rgb, expected_shape=expected_shape)
    if (declared_height, declared_width) != rgb.shape[:2]:
        raise ValueError(
            "Decoded image size does not match declaration: "
            f"declared={(declared_height, declared_width)}, actual={rgb.shape[:2]}"
        )
    return rgb


def _tactile_to_payload(obs: Observation, *, jpeg_quality: int) -> dict[str, Any] | None:
    values = {
        "rgb": obs.tactile_rgb,
        "marker_displacement_history": obs.marker_displacement_history,
        "marker_valid_mask": obs.marker_valid_mask,
        "marker_history_valid_mask": obs.marker_history_valid_mask,
        "sensor_mask": obs.tactile_sensor_mask,
    }
    supplied = {name for name, value in values.items() if value is not None}
    if not supplied:
        return None
    if obs.tactile_sensor_mask is None:
        raise ValueError("Tactile input requires tactile_sensor_mask")

    sensor_mask = _validate_tactile_sensor_mask(obs.tactile_sensor_mask)
    payload: dict[str, Any] = {"sensor_mask": sensor_mask.astype(bool).tolist()}

    if obs.tactile_rgb is not None:
        rgb = np.asarray(obs.tactile_rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != TACTILE_SENSOR_COUNT or rgb.shape[-1] != 3:
            raise ValueError(
                "tactile_rgb must be uint8 with shape "
                f"[{TACTILE_SENSOR_COUNT},H,W,3], got {rgb.dtype} {rgb.shape}"
            )
        payload["rgb"] = [encode_rgb_image(frame, jpeg_quality=jpeg_quality) for frame in rgb]

    marker_names = {
        "marker_displacement_history": obs.marker_displacement_history,
        "marker_valid_mask": obs.marker_valid_mask,
        "marker_history_valid_mask": obs.marker_history_valid_mask,
    }
    marker_supplied = {name for name, value in marker_names.items() if value is not None}
    if marker_supplied and marker_supplied != set(marker_names):
        missing = sorted(set(marker_names) - marker_supplied)
        raise ValueError(f"Marker tactile input is incomplete; missing {missing}")
    if marker_supplied:
        marker_shape = (
            TACTILE_SENSOR_COUNT,
            TACTILE_MARKER_HISTORY_LENGTH,
            TACTILE_MARKER_COUNT,
            2,
        )
        history = np.asarray(obs.marker_displacement_history, dtype=np.float32)
        if history.shape != marker_shape or not np.all(np.isfinite(history)):
            raise ValueError(
                "marker_displacement_history must be finite float data with shape "
                f"{marker_shape}, got {history.shape}"
            )
        payload["marker_displacement_history"] = history.astype(float).tolist()
        valid = np.asarray(obs.marker_valid_mask)
        expected_valid_shape = marker_shape[:-1]
        if valid.dtype != np.bool_ or valid.shape != expected_valid_shape:
            raise ValueError(
                f"marker_valid_mask must be bool with shape {expected_valid_shape}, "
                f"got {valid.dtype} {valid.shape}"
            )
        payload["marker_valid_mask"] = valid.astype(bool).tolist()
        history_valid = np.asarray(obs.marker_history_valid_mask)
        expected_history_valid_shape = (
            TACTILE_SENSOR_COUNT,
            TACTILE_MARKER_HISTORY_LENGTH,
        )
        if (
            history_valid.dtype != np.bool_
            or history_valid.shape != expected_history_valid_shape
        ):
            raise ValueError(
                "marker_history_valid_mask must be bool with shape "
                f"{expected_history_valid_shape}, got "
                f"{history_valid.dtype} {history_valid.shape}"
            )
        payload["marker_history_valid_mask"] = history_valid.astype(bool).tolist()
        payload["marker_sample_hz"] = TACTILE_MARKER_SAMPLE_HZ
    if "rgb" not in payload and not marker_supplied:
        raise ValueError("Tactile input must include RGB and/or marker data")
    return payload


def _tactile_from_payload(value: Any) -> dict[str, np.ndarray | None]:
    empty = {
        "tactile_rgb": None,
        "marker_displacement_history": None,
        "marker_valid_mask": None,
        "marker_history_valid_mask": None,
        "tactile_sensor_mask": None,
    }
    if value is None:
        return empty
    if not isinstance(value, dict):
        raise ValueError("tactile must be a JSON object")
    allowed = {
        "rgb",
        "marker_displacement_history",
        "marker_valid_mask",
        "marker_history_valid_mask",
        "marker_sample_hz",
        "sensor_mask",
    }
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"Unexpected tactile fields: {sorted(unexpected)}")

    sensor_mask = _validate_tactile_sensor_mask(value.get("sensor_mask"))
    result = dict(empty)
    result["tactile_sensor_mask"] = sensor_mask

    rgb_payload = value.get("rgb")
    if rgb_payload is not None:
        if not isinstance(rgb_payload, list) or len(rgb_payload) != TACTILE_SENSOR_COUNT:
            raise ValueError(f"tactile.rgb must contain {TACTILE_SENSOR_COUNT} encoded image")
        result["tactile_rgb"] = np.stack(
            [decode_rgb_image(frame) for frame in rgb_payload], axis=0
        )

    marker_fields = (
        "marker_displacement_history",
        "marker_valid_mask",
        "marker_history_valid_mask",
    )
    marker_supplied = {name for name in marker_fields if value.get(name) is not None}
    if marker_supplied and marker_supplied != set(marker_fields):
        missing = sorted(set(marker_fields) - marker_supplied)
        raise ValueError(f"Marker tactile input is incomplete; missing {missing}")
    if marker_supplied:
        marker_shape = (
            TACTILE_SENSOR_COUNT,
            TACTILE_MARKER_HISTORY_LENGTH,
            TACTILE_MARKER_COUNT,
            2,
        )
        history = np.asarray(value["marker_displacement_history"], dtype=np.float32)
        if history.shape != marker_shape or not np.all(np.isfinite(history)):
            raise ValueError(
                "tactile.marker_displacement_history must be finite with shape "
                f"{marker_shape}, got {history.shape}"
            )
        result["marker_displacement_history"] = np.ascontiguousarray(history)
        valid = np.asarray(value["marker_valid_mask"])
        expected_valid_shape = marker_shape[:-1]
        if valid.dtype != np.bool_ or valid.shape != expected_valid_shape:
            raise ValueError(
                f"tactile.marker_valid_mask must be bool with shape {expected_valid_shape}, "
                f"got {valid.dtype} {valid.shape}"
            )
        result["marker_valid_mask"] = np.ascontiguousarray(valid)
        history_valid = np.asarray(value["marker_history_valid_mask"])
        expected_history_valid_shape = (
            TACTILE_SENSOR_COUNT,
            TACTILE_MARKER_HISTORY_LENGTH,
        )
        if (
            history_valid.dtype != np.bool_
            or history_valid.shape != expected_history_valid_shape
        ):
            raise ValueError(
                "tactile.marker_history_valid_mask must be bool with shape "
                f"{expected_history_valid_shape}, got "
                f"{history_valid.dtype} {history_valid.shape}"
            )
        result["marker_history_valid_mask"] = np.ascontiguousarray(history_valid)
        marker_sample_hz = float(value.get("marker_sample_hz", float("nan")))
        if not np.isclose(
            marker_sample_hz,
            TACTILE_MARKER_SAMPLE_HZ,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "tactile.marker_sample_hz must be "
                f"{TACTILE_MARKER_SAMPLE_HZ:g}, got {marker_sample_hz}"
            )
    if result["tactile_rgb"] is None and not marker_supplied:
        raise ValueError("tactile must include RGB and/or marker data")
    return result


def _validate_tactile_sensor_mask(value: Any) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (TACTILE_SENSOR_COUNT,):
        raise ValueError(
            f"tactile_sensor_mask must be bool with shape ({TACTILE_SENSOR_COUNT},), "
            f"got {mask.dtype} {mask.shape}"
        )
    return np.ascontiguousarray(mask)


def _validate_rgb_image(
    image: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB image with shape (H, W, 3), got {rgb.dtype} {rgb.shape}")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[0] * rgb.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"Invalid image dimensions: {rgb.shape}")
    if expected_shape is not None and tuple(rgb.shape) != expected_shape:
        raise ValueError(
            f"Wire RGB image must have shape {expected_shape}, got {rgb.shape}"
        )
    return np.ascontiguousarray(rgb)


def _validate_protocol(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"Unsupported protocol: {payload.get('protocol')!r}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol_version: expected {PROTOCOL_VERSION}, got {payload.get('protocol_version')!r}"
        )


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
    if "episode_reset" in metadata and not isinstance(metadata["episode_reset"], bool):
        raise ValueError("metadata.episode_reset must be a JSON boolean")
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
