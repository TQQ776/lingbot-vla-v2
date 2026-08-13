from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lingbotvla.tactile_contact import (
    CONTACT_OFF,
    MarkerContactGate,
    build_marker_region_mapping_numpy,
    runtime_gate_config_from_mapping,
)

from .protocol import (
    TACTILE_MARKER_COUNT,
    TACTILE_MARKER_HISTORY_LENGTH,
    TACTILE_MARKER_SAMPLE_HZ,
    TACTILE_SENSOR_COUNT,
)


@dataclass(frozen=True)
class TactileFrame:
    tactile_rgb: np.ndarray | None
    marker_displacement_history: np.ndarray | None
    marker_valid_mask: np.ndarray | None
    marker_history_valid_mask: np.ndarray | None
    marker_contact_state: np.ndarray | None
    marker_reference_pixels: np.ndarray | None
    marker_current_pixels: np.ndarray | None
    tactile_sensor_mask: np.ndarray
    capture_timestamp: float
    receive_timestamp: float
    debug: dict[str, Any]


def _normalized_marker_frames(
    sensor_data: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite normalized displacement frames and per-marker validity."""

    rgb_history = np.asarray(sensor_data.get("rgb"))
    marker = np.asarray(sensor_data.get("marker"), dtype=np.float32)
    reference = np.asarray(sensor_data.get("marker_ref"), dtype=np.float32)
    expected_shape = (len(rgb_history), TACTILE_MARKER_COUNT, 2)
    if marker.shape != expected_shape or reference.shape != expected_shape:
        raise ValueError(
            "TacThru marker and marker_ref must both have shape "
            f"{expected_shape}, got {marker.shape} and {reference.shape}"
        )
    image_height, image_width = rgb_history.shape[1:3]
    scale_xy = np.asarray([image_width, image_height], dtype=np.float32)
    valid = np.isfinite(marker).all(axis=-1)
    valid &= np.isfinite(reference).all(axis=-1)
    tracker_valid = sensor_data.get("marker_valid")
    if tracker_valid is not None:
        tracker_valid = np.asarray(tracker_valid, dtype=np.bool_)
        if tracker_valid.shape != expected_shape[:-1]:
            raise ValueError(
                "TacThru marker_valid must have shape "
                f"{expected_shape[:-1]}, got {tracker_valid.shape}"
            )
        valid &= tracker_valid
    displacement = (marker - reference) / scale_xy * 2.0
    displacement = np.where(
        valid[..., None], displacement, 0.0
    ).astype(np.float32, copy=False)
    return displacement, valid


def build_tactile_frame(
    sensor_data: Mapping[str, Any],
    *,
    use_rgb: bool,
    use_markers: bool,
    episode_reset: bool,
    marker_history_length: int = TACTILE_MARKER_HISTORY_LENGTH,
) -> TactileFrame:
    if not (use_rgb or use_markers):
        raise ValueError("At least one tactile modality must be enabled")

    timestamps = np.asarray(sensor_data.get("timestamp"), dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"TacThru timestamp must be a non-empty vector, got {timestamps.shape}")
    if not np.all(np.isfinite(timestamps)) or np.any(timestamps <= 0.0):
        raise ValueError("TacThru timestamps must be positive and finite")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("TacThru timestamps must be strictly increasing")

    rgb_history = np.asarray(sensor_data.get("rgb"))
    if (
        rgb_history.dtype != np.uint8
        or rgb_history.ndim != 4
        or rgb_history.shape[0] != len(timestamps)
        or rgb_history.shape[-1] != 3
    ):
        raise ValueError(
            "TacThru rgb must be uint8 [T,H,W,3] aligned with timestamps, "
            f"got {rgb_history.dtype} {rgb_history.shape}"
        )
    height, width = rgb_history.shape[1:3]
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid TacThru frame size {width}x{height}")

    tactile_rgb = None
    if use_rgb:
        tactile_rgb = np.ascontiguousarray(rgb_history[-1:])

    marker_displacement_history = None
    marker_valid_mask = None
    marker_history_valid_mask = None
    marker_reference_pixels = None
    marker_current_pixels = None
    valid_count = None
    if use_markers:
        flow, valid = _normalized_marker_frames(sensor_data)
        marker_pixels = np.asarray(sensor_data.get("marker"), dtype=np.float32)
        reference_pixels = np.asarray(sensor_data.get("marker_ref"), dtype=np.float32)
        if episode_reset:
            flow = flow[-1:]
            valid = valid[-1:]
        flow = flow[-marker_history_length:]
        valid = valid[-marker_history_length:]
        if len(flow) < marker_history_length:
            pad_count = marker_history_length - len(flow)
            flow = np.concatenate(
                [np.repeat(flow[:1], pad_count, axis=0), flow], axis=0
            )
            valid = np.concatenate(
                [np.repeat(valid[:1], pad_count, axis=0), valid], axis=0
            )

        marker_displacement_history = np.ascontiguousarray(
            flow[None], dtype=np.float32
        )
        marker_valid_mask = np.ascontiguousarray(valid[None], dtype=np.bool_)
        marker_history_valid_mask = np.ones(
            (TACTILE_SENSOR_COUNT, marker_history_length), dtype=np.bool_
        )
        marker_reference_pixels = np.ascontiguousarray(reference_pixels[-1:])
        marker_current_pixels = np.ascontiguousarray(marker_pixels[-1:])
        valid_count = int(valid[-1].sum())

    detected = sensor_data.get("n_all_kpts")
    detected_count = None
    if detected is not None:
        detected_array = np.asarray(detected).reshape(-1)
        if len(detected_array):
            detected_count = int(detected_array[-1])

    return TactileFrame(
        tactile_rgb=tactile_rgb,
        marker_displacement_history=marker_displacement_history,
        marker_valid_mask=marker_valid_mask,
        marker_history_valid_mask=marker_history_valid_mask,
        marker_contact_state=None,
        marker_reference_pixels=marker_reference_pixels,
        marker_current_pixels=marker_current_pixels,
        tactile_sensor_mask=np.ones((TACTILE_SENSOR_COUNT,), dtype=np.bool_),
        capture_timestamp=float(timestamps[-1]),
        receive_timestamp=time.time(),
        debug={
            "frame_shape_hwc": [int(height), int(width), 3],
            "history_frames": int(len(timestamps)),
            "marker_representation": "normalized_displacement",
            "marker_formula": "2 * (current_xy - reference_xy) / [image_width, image_height]",
            "marker_history_length": int(marker_history_length),
            "marker_sample_hz": float(TACTILE_MARKER_SAMPLE_HZ),
            "marker_order": "fixed",
            "valid_marker_count": valid_count,
            "detected_keypoint_count": detected_count,
        },
    )


class TacThruSource:
    def __init__(
        self,
        *,
        tacthru_repo: Path,
        sensor_cfg_path: Path,
        use_rgb: bool,
        use_markers: bool,
        tactile_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.tacthru_repo = tacthru_repo.expanduser().resolve()
        self.sensor_cfg_path = sensor_cfg_path.expanduser().resolve()
        self.use_rgb = bool(use_rgb)
        self.use_markers = bool(use_markers)
        self.marker_history_length = int(
            (tactile_config or {}).get(
                "marker_history_length", TACTILE_MARKER_HISTORY_LENGTH
            )
        )
        if self.marker_history_length <= 0:
            raise ValueError("marker_history_length must be positive")
        self._manager: SharedMemoryManager | None = None
        self._sensor = None
        self._marker_history: deque[np.ndarray] = deque(
            maxlen=self.marker_history_length
        )
        self._marker_valid_history: deque[np.ndarray] = deque(
            maxlen=self.marker_history_length
        )
        self._last_marker_timestamp: float | None = None
        self._contact_gates: list[MarkerContactGate] = []
        self._contact_state = np.full(
            (TACTILE_SENSOR_COUNT,), CONTACT_OFF, dtype=np.int8
        )
        if self.use_markers and tactile_config is not None:
            gate_mapping = dict(tactile_config.get("marker_contact_gate", {}))
            gate_settings = runtime_gate_config_from_mapping(
                gate_mapping,
                num_sensors=int(tactile_config["num_sensors"]),
                num_markers=int(tactile_config["num_markers"]),
            )
            if gate_settings.mode != "none":
                tokenization = dict(tactile_config.get("marker_tokenization", {}))
                if tokenization.get("mode", "global") == "regional":
                    region_ids, _ = build_marker_region_mapping_numpy(
                        tactile_config["marker_reference_xy"],
                        num_regions=int(tokenization["num_regions"]),
                        region_layout=str(tokenization["region_layout"]),
                    )
                else:
                    region_ids = np.zeros(
                        (
                            int(tactile_config["num_sensors"]),
                            int(tactile_config["num_markers"]),
                        ),
                        dtype=np.int64,
                    )
                self._contact_gates = [
                    MarkerContactGate(
                        gate_settings,
                        region_ids[sensor_index],
                        sensor_index=sensor_index,
                    )
                    for sensor_index in range(int(tactile_config["num_sensors"]))
                ]
        if not self.tacthru_repo.is_dir():
            raise NotADirectoryError(self.tacthru_repo)
        if not self.sensor_cfg_path.is_file():
            raise FileNotFoundError(self.sensor_cfg_path)
        if not (self.use_rgb or self.use_markers):
            raise ValueError("TacThruSource requires RGB and/or marker input")

    def start(self) -> None:
        if self._sensor is not None:
            raise RuntimeError("TacThruSource is already started")
        self._clear_marker_history()
        repo = str(self.tacthru_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from omegaconf import OmegaConf
        from real_world.sensor_utils import TacThruClient

        sensor_cfg = OmegaConf.load(self.sensor_cfg_path)
        tracking_path = Path(str(sensor_cfg.tracking.tracking_pts_path)).expanduser()
        if not tracking_path.is_absolute():
            sensor_cfg.tracking.tracking_pts_path = str(
                (self.tacthru_repo / tracking_path).resolve()
            )

        manager = SharedMemoryManager()
        manager.start()
        sensor = None
        try:
            sensor = TacThruClient(
                manager,
                None,
                sensor_cfg,
                "cuda:0",
                False,
                str(sensor_cfg.name),
                do_tracking=self.use_markers,
            )
            sensor.start(wait=True)
        except Exception:
            if sensor is not None and sensor.is_alive():
                sensor.terminate()
                sensor.join(timeout=1.0)
            manager.shutdown()
            raise
        self._manager = manager
        self._sensor = sensor

    def capture(self, *, episode_reset: bool, timeout_s: float = 2.0) -> TactileFrame:
        sensor = self._sensor
        if sensor is None:
            raise RuntimeError("TacThruSource is not started")
        deadline = time.monotonic() + float(timeout_s)
        while int(sensor.ring_buffer.count) < 1:
            if not sensor.is_alive():
                raise RuntimeError("TacThru sensor process exited before producing a frame")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a TacThru frame")
            time.sleep(0.01)
        frame_count = min(30, int(sensor.ring_buffer.count))
        data = sensor.get(k=frame_count)
        frame = build_tactile_frame(
            data,
            use_rgb=self.use_rgb,
            use_markers=self.use_markers,
            episode_reset=episode_reset,
            marker_history_length=self.marker_history_length,
        )
        if not self.use_markers:
            return frame

        timestamps = np.asarray(data["timestamp"], dtype=np.float64)
        history = np.asarray(frame.marker_displacement_history[0], dtype=np.float32)
        valid_history = np.asarray(frame.marker_valid_mask[0], dtype=np.bool_)
        if episode_reset:
            self._clear_marker_history()
            selected_history = history[-1:]
            selected_valid = valid_history[-1:]
            selected_timestamps = timestamps[-1:]
        else:
            if self._last_marker_timestamp is None:
                selected = np.ones(len(timestamps), dtype=np.bool_)
            else:
                selected = timestamps > self._last_marker_timestamp + 1e-9
            raw_history = _normalized_marker_frames(data)
            selected_history = raw_history[0][selected]
            selected_valid = raw_history[1][selected]
            selected_timestamps = timestamps[selected]

        for displacement, valid, timestamp in zip(
            selected_history, selected_valid, selected_timestamps
        ):
            if not self._marker_history:
                for _ in range(self.marker_history_length):
                    self._marker_history.append(displacement.copy())
                    self._marker_valid_history.append(valid.copy())
            else:
                self._marker_history.append(displacement.copy())
                self._marker_valid_history.append(valid.copy())
            self._last_marker_timestamp = float(timestamp)
            for sensor_index, gate in enumerate(self._contact_gates):
                self._contact_state[sensor_index] = gate.step(
                    displacement,
                    valid,
                ).state

        if not self._marker_history:
            raise RuntimeError("TacThru marker history is empty after capture")
        displacement_history = np.stack(tuple(self._marker_history), axis=0)
        valid_mask = np.stack(tuple(self._marker_valid_history), axis=0)
        return replace(
            frame,
            marker_displacement_history=np.ascontiguousarray(
                displacement_history[None], dtype=np.float32
            ),
            marker_valid_mask=np.ascontiguousarray(valid_mask[None], dtype=np.bool_),
            marker_history_valid_mask=np.ones(
                (TACTILE_SENSOR_COUNT, self.marker_history_length),
                dtype=np.bool_,
            ),
            marker_contact_state=(
                np.ascontiguousarray(self._contact_state, dtype=np.int8)
                if self._contact_gates
                else None
            ),
        )

    def _clear_marker_history(self) -> None:
        self._marker_history.clear()
        self._marker_valid_history.clear()
        self._last_marker_timestamp = None
        self._contact_state.fill(CONTACT_OFF)
        for gate in self._contact_gates:
            gate.reset()

    def close(self) -> None:
        sensor, self._sensor = self._sensor, None
        manager, self._manager = self._manager, None
        if sensor is not None:
            sensor.stop(wait=False)
            sensor.join(timeout=3.0)
            if sensor.is_alive():
                sensor.terminate()
                sensor.join(timeout=1.0)
        if manager is not None:
            manager.shutdown()
        self._clear_marker_history()

    def __enter__(self) -> "TacThruSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["TactileFrame", "TacThruSource", "build_tactile_frame"]
