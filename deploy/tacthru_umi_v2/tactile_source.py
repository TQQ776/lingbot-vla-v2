from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .protocol import TACTILE_MARKER_COUNT, TACTILE_SENSOR_COUNT


@dataclass(frozen=True)
class TactileFrame:
    tactile_rgb: np.ndarray | None
    marker_positions: np.ndarray | None
    marker_reference: np.ndarray | None
    previous_marker_positions: np.ndarray | None
    marker_valid_mask: np.ndarray | None
    tactile_sensor_mask: np.ndarray
    capture_timestamp: float
    receive_timestamp: float
    debug: dict[str, Any]


def build_tactile_frame(
    sensor_data: Mapping[str, Any],
    *,
    use_rgb: bool,
    use_markers: bool,
    episode_reset: bool,
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

    marker_positions = None
    marker_reference = None
    previous_marker_positions = None
    marker_valid_mask = None
    valid_count = None
    if use_markers:
        marker = np.asarray(sensor_data.get("marker"), dtype=np.float32)
        reference = np.asarray(sensor_data.get("marker_ref"), dtype=np.float32)
        expected_shape = (len(timestamps), TACTILE_MARKER_COUNT, 2)
        if marker.shape != expected_shape or reference.shape != expected_shape:
            raise ValueError(
                "TacThru marker and marker_ref must both have shape "
                f"{expected_shape}, got {marker.shape} and {reference.shape}"
            )

        scale_xy = np.asarray([width, height], dtype=np.float32)
        flow = (marker - reference) / scale_xy * 2.0
        current = flow[-1]
        previous = current if episode_reset or len(flow) == 1 else flow[-2]
        current_valid = np.isfinite(marker[-1]).all(axis=-1)
        current_valid &= np.isfinite(reference[-1]).all(axis=-1)
        if not episode_reset and len(flow) > 1:
            current_valid &= np.isfinite(marker[-2]).all(axis=-1)
            current_valid &= np.isfinite(reference[-2]).all(axis=-1)
        current = np.where(current_valid[:, None], current, 0.0)
        previous = np.where(current_valid[:, None], previous, 0.0)

        marker_positions = np.ascontiguousarray(current[None], dtype=np.float32)
        # Converted training data stores normalized flow as position and zero as reference.
        marker_reference = np.zeros_like(marker_positions)
        previous_marker_positions = np.ascontiguousarray(previous[None], dtype=np.float32)
        marker_valid_mask = np.ascontiguousarray(current_valid[None], dtype=np.bool_)
        valid_count = int(current_valid.sum())

    detected = sensor_data.get("n_all_kpts")
    detected_count = None
    if detected is not None:
        detected_array = np.asarray(detected).reshape(-1)
        if len(detected_array):
            detected_count = int(detected_array[-1])

    return TactileFrame(
        tactile_rgb=tactile_rgb,
        marker_positions=marker_positions,
        marker_reference=marker_reference,
        previous_marker_positions=previous_marker_positions,
        marker_valid_mask=marker_valid_mask,
        tactile_sensor_mask=np.ones((TACTILE_SENSOR_COUNT,), dtype=np.bool_),
        capture_timestamp=float(timestamps[-1]),
        receive_timestamp=time.time(),
        debug={
            "frame_shape_hwc": [int(height), int(width), 3],
            "history_frames": int(len(timestamps)),
            "marker_representation": "normalized_flow_2x_over_frame_size",
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
    ) -> None:
        self.tacthru_repo = tacthru_repo.expanduser().resolve()
        self.sensor_cfg_path = sensor_cfg_path.expanduser().resolve()
        self.use_rgb = bool(use_rgb)
        self.use_markers = bool(use_markers)
        self._manager: SharedMemoryManager | None = None
        self._sensor = None
        if not self.tacthru_repo.is_dir():
            raise NotADirectoryError(self.tacthru_repo)
        if not self.sensor_cfg_path.is_file():
            raise FileNotFoundError(self.sensor_cfg_path)
        if not (self.use_rgb or self.use_markers):
            raise ValueError("TacThruSource requires RGB and/or marker input")

    def start(self) -> None:
        if self._sensor is not None:
            raise RuntimeError("TacThruSource is already started")
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
        frame_count = min(2, int(sensor.ring_buffer.count))
        data = sensor.get(k=frame_count)
        return build_tactile_frame(
            data,
            use_rgb=self.use_rgb,
            use_markers=self.use_markers,
            episode_reset=episode_reset,
        )

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

    def __enter__(self) -> "TacThruSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["TactileFrame", "TacThruSource", "build_tactile_frame"]
