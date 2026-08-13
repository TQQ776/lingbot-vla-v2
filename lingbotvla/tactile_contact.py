"""NumPy-only TacThru contact state machine shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


CONTACT_OFF = 0
CONTACT_ON = 1
CONTACT_HOLD = 2
CONTACT_UNKNOWN_OFF = 3
CONTACT_UNKNOWN_ON = 4
CONTACT_UNKNOWN_HOLD = 5


def contact_state_is_visible_numpy(state: Any) -> np.ndarray:
    state = np.asarray(state)
    return np.isin(
        state,
        [CONTACT_ON, CONTACT_HOLD, CONTACT_UNKNOWN_ON, CONTACT_UNKNOWN_HOLD],
    )


def build_marker_region_mapping_numpy(
    reference_xy: Any,
    *,
    num_regions: int,
    region_layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_xy, dtype=np.float32)
    if reference.ndim == 2:
        reference = reference[None]
    if reference.ndim != 3 or reference.shape[-1] != 2 or not np.isfinite(reference).all():
        raise ValueError("marker reference coordinates must be finite [S,N,2]")
    low = reference.min(axis=1, keepdims=True)
    high = reference.max(axis=1, keepdims=True)
    if np.any(high <= low):
        raise ValueError("marker reference coordinates must span x and y")
    reference = 2.0 * (reference - low) / (high - low) - 1.0
    if num_regions == 1 and region_layout == "1x1":
        region_ids = np.zeros(reference.shape[:2], dtype=np.int64)
    elif num_regions == 4 and region_layout == "2x2":
        center = np.median(reference, axis=1, keepdims=True)
        right = reference[..., 0] >= center[..., 0]
        bottom = reference[..., 1] >= center[..., 1]
        region_ids = right.astype(np.int64) + 2 * bottom.astype(np.int64)
    else:
        raise ValueError("Only 1x1 global and 2x2 regional mappings are supported")
    centers = np.zeros((reference.shape[0], num_regions, 2), dtype=np.float32)
    for sensor_index in range(reference.shape[0]):
        for region_index in range(num_regions):
            mask = region_ids[sensor_index] == region_index
            if not mask.any():
                raise ValueError(f"Sensor {sensor_index} region {region_index} is empty")
            centers[sensor_index, region_index] = reference[sensor_index, mask].mean(0)
    return region_ids, centers


@dataclass(frozen=True)
class MarkerGateFrame:
    state: int
    region_scores: np.ndarray
    regional_soft_gates: np.ndarray
    region_valid_mask: np.ndarray
    active_counts: np.ndarray
    on_active_marker_count: int
    off_active_marker_count: int
    valid_marker_count: int
    tracking_unknown: bool


@dataclass(frozen=True)
class MarkerGateEpisodeOutput:
    states: np.ndarray
    region_scores: np.ndarray
    regional_soft_gates: np.ndarray
    region_valid_mask: np.ndarray
    active_counts: np.ndarray
    on_active_marker_count: np.ndarray
    off_active_marker_count: np.ndarray
    tracking_unknown: np.ndarray


def runtime_gate_config_from_mapping(
    value: Mapping[str, Any],
    *,
    num_sensors: int,
    num_markers: int,
) -> SimpleNamespace:
    """Parse the already-resolved gate mapping sent in server health."""

    values = dict(value)
    point = np.asarray(values.get("point_threshold", 0.0), dtype=np.float32)
    if point.ndim == 0:
        point = np.full((num_sensors, num_markers), float(point), dtype=np.float32)
    elif point.shape == (num_markers,):
        point = np.repeat(point[None], num_sensors, axis=0)
    if point.shape != (num_sensors, num_markers):
        raise ValueError("Resolved point_threshold must be scalar, [N], or [S,N]")

    def scalars(name: str, default: float = 0.0) -> tuple[float, ...]:
        raw = np.asarray(values.get(name, default), dtype=np.float32)
        if raw.ndim == 0:
            raw = np.repeat(raw[None], num_sensors)
        if raw.shape != (num_sensors,):
            raise ValueError(f"Resolved {name} must be scalar or [S]")
        return tuple(float(item) for item in raw)

    config = SimpleNamespace(
        mode=str(values.get("mode", "none")),
        topk_markers=int(values.get("topk_markers", 3)),
        min_active_markers=int(values.get("min_active_markers", 2)),
        min_valid_markers_per_region=int(values.get("min_valid_markers_per_region", 2)),
        min_valid_markers_global=int(values.get("min_valid_markers_global", 24)),
        point_thresholds=tuple(tuple(float(item) for item in row) for row in point),
        on_thresholds=scalars("on_threshold"),
        off_thresholds=scalars("off_threshold"),
        soft_thresholds=scalars("soft_threshold"),
        on_consecutive_frames=int(values.get("on_consecutive_frames", 2)),
        off_consecutive_frames=int(values.get("off_consecutive_frames", 3)),
        release_hold_frames=int(values.get("release_hold_frames", 3)),
        soft_gate_temperature=float(values.get("soft_gate_temperature", 0.1)),
        unknown_tracking_policy=str(values.get("unknown_tracking_policy", "hold_previous")),
        max_unknown_hold_frames=int(values.get("max_unknown_hold_frames", 3)),
    )
    supported_modes = {
        "none",
        "hard",
        "hard_hysteresis",
        "hard_hysteresis_hold",
        "hard_hysteresis_soft_region",
        "global_active_count_hysteresis_soft_region",
    }
    if config.mode not in supported_modes:
        raise ValueError(f"Unsupported marker contact gate mode: {config.mode}")
    positive_counts = (
        config.topk_markers,
        config.min_active_markers,
        config.min_valid_markers_per_region,
        config.min_valid_markers_global,
        config.on_consecutive_frames,
        config.off_consecutive_frames,
    )
    if any(value <= 0 for value in positive_counts):
        raise ValueError("Contact gate count fields must be positive")
    if config.release_hold_frames < 0 or config.max_unknown_hold_frames < 0:
        raise ValueError("Contact gate hold frame counts must be non-negative")
    if not np.isfinite(config.soft_gate_temperature) or config.soft_gate_temperature <= 0:
        raise ValueError("soft_gate_temperature must be positive and finite")
    if config.unknown_tracking_policy not in {
        "hold_previous",
        "force_off",
        "force_on",
        "validity_only",
    }:
        raise ValueError(
            f"Unsupported unknown_tracking_policy: {config.unknown_tracking_policy}"
        )
    threshold_values = np.asarray(
        [*config.on_thresholds, *config.off_thresholds, *config.soft_thresholds],
        dtype=np.float32,
    )
    if not np.isfinite(point).all() or not np.isfinite(threshold_values).all():
        raise ValueError("Contact gate thresholds must be finite")
    if config.mode != "none" and any(
        off >= on
        for off, on in zip(config.off_thresholds, config.on_thresholds)
    ):
        raise ValueError("marker contact off_threshold must be less than on_threshold")
    return config


class MarkerContactGate:
    """Deterministic per-sensor contact gate with hysteresis, HOLD, and UNKNOWN."""

    def __init__(self, config: Any, region_ids: Any, *, sensor_index: int = 0) -> None:
        self.config = config
        self.region_ids = np.asarray(region_ids, dtype=np.int64)
        if self.region_ids.ndim != 1 or not len(self.region_ids):
            raise ValueError("region_ids must be a non-empty [N] vector")
        self.num_regions = int(self.region_ids.max()) + 1
        if set(self.region_ids.tolist()) != set(range(self.num_regions)):
            raise ValueError("region_ids must use contiguous region indices")
        self.sensor_index = int(sensor_index)
        if self.sensor_index < 0 or self.sensor_index >= len(config.on_thresholds):
            raise ValueError(f"Invalid contact-gate sensor index: {self.sensor_index}")
        self.point_thresholds = np.asarray(
            config.point_thresholds[sensor_index], dtype=np.float32
        )
        self.on_threshold = float(config.on_thresholds[sensor_index])
        self.off_threshold = float(config.off_thresholds[sensor_index])
        self.soft_threshold = float(config.soft_thresholds[sensor_index])
        if self.point_thresholds.shape != self.region_ids.shape:
            raise ValueError("point thresholds and region mapping must have equal length")
        if config.soft_gate_temperature <= 0:
            raise ValueError("soft_gate_temperature must be positive")
        self.reset()

    def reset(self) -> None:
        self._state = CONTACT_OFF
        self._on_count = 0
        self._off_count = 0
        self._hold_remaining = 0
        self._unknown_count = 0

    def _measure(self, displacement: Any, valid_mask: Any) -> tuple[np.ndarray, ...]:
        displacement = np.asarray(displacement, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if displacement.shape != (len(self.region_ids), 2):
            raise ValueError(f"displacement must be [{len(self.region_ids)},2]")
        if valid.shape != (len(self.region_ids),):
            raise ValueError(f"valid_mask must be [{len(self.region_ids)}]")
        valid = valid & np.isfinite(displacement).all(axis=-1)
        amplitude = np.linalg.norm(
            np.where(valid[:, None], displacement, 0.0), axis=-1
        )
        scores = np.zeros(self.num_regions, dtype=np.float32)
        active = np.zeros(self.num_regions, dtype=np.int64)
        region_valid = np.zeros(self.num_regions, dtype=np.bool_)
        for region_index in range(self.num_regions):
            mask = valid & (self.region_ids == region_index)
            count = int(mask.sum())
            region_valid[region_index] = count >= self.config.min_valid_markers_per_region
            if count:
                values = amplitude[mask]
                k = min(self.config.topk_markers, count)
                scores[region_index] = np.partition(values, count - k)[-k:].mean()
                active[region_index] = int(
                    (amplitude[mask] > self.point_thresholds[mask]).sum()
                )
        soft = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    (scores - self.soft_threshold)
                    / self.config.soft_gate_temperature,
                    -60.0,
                    60.0,
                )
            )
        )
        return (
            scores,
            soft.astype(np.float32),
            region_valid,
            active,
            np.asarray(valid.sum()),
            amplitude,
            valid,
        )

    def _has_contact(
        self,
        scores: np.ndarray,
        active: np.ndarray,
        region_valid: np.ndarray,
        threshold: float,
    ) -> bool:
        return bool(
            np.any(
                (scores > threshold)
                & (active >= self.config.min_active_markers)
                & region_valid
            )
        )

    def _transition_off_evidence(self) -> None:
        self._on_count = 0
        if self._state == CONTACT_ON:
            self._off_count += 1
            if self._off_count >= self.config.off_consecutive_frames:
                if self.config.release_hold_frames > 0 and self.config.mode in {
                    "hard_hysteresis_hold",
                    "hard_hysteresis_soft_region",
                    "global_active_count_hysteresis_soft_region",
                }:
                    self._state = CONTACT_HOLD
                    self._hold_remaining = self.config.release_hold_frames
                else:
                    self._state = CONTACT_OFF
                self._off_count = 0
        elif self._state == CONTACT_HOLD:
            if self._hold_remaining > 1:
                self._hold_remaining -= 1
            else:
                self._state = CONTACT_OFF
                self._hold_remaining = 0

    def _state_code(self, unknown: bool) -> int:
        if not unknown:
            return self._state
        if self._state == CONTACT_ON:
            return CONTACT_UNKNOWN_ON
        if self._state == CONTACT_HOLD:
            return CONTACT_UNKNOWN_HOLD
        return CONTACT_UNKNOWN_OFF

    def _expire_unknown_hold(self) -> None:
        self._on_count = 0
        self._off_count = 0
        if self._state == CONTACT_ON:
            if self.config.release_hold_frames > 0 and self.config.mode in {
                "hard_hysteresis_hold",
                "hard_hysteresis_soft_region",
                "global_active_count_hysteresis_soft_region",
            }:
                self._state = CONTACT_HOLD
                self._hold_remaining = self.config.release_hold_frames
            else:
                self._state = CONTACT_OFF
        elif self._state == CONTACT_HOLD:
            if self._hold_remaining > 1:
                self._hold_remaining -= 1
            else:
                self._state = CONTACT_OFF
                self._hold_remaining = 0

    def step(self, displacement: Any, valid_mask: Any) -> MarkerGateFrame:
        scores, soft, region_valid, active, valid_count, amplitude, valid = self._measure(
            displacement, valid_mask
        )
        on_active_mask = valid & (amplitude > self.on_threshold)
        off_active_mask = valid & (amplitude > self.off_threshold)
        on_active_count = int(on_active_mask.sum())
        off_active_count = int(off_active_mask.sum())
        global_count_mode = (
            self.config.mode == "global_active_count_hysteresis_soft_region"
        )
        if global_count_mode:
            active = np.asarray(
                [
                    np.count_nonzero(on_active_mask & (self.region_ids == region_index))
                    for region_index in range(self.num_regions)
                ],
                dtype=np.int64,
            )
        unknown = int(valid_count) < self.config.min_valid_markers_global
        if self.config.mode == "none":
            self._state = CONTACT_ON
            unknown = False
        elif unknown:
            self._unknown_count += 1
            self._on_count = 0
            self._off_count = 0
            policy = self.config.unknown_tracking_policy
            if policy == "force_on":
                self._state = CONTACT_ON
            elif policy == "force_off":
                self._state = CONTACT_OFF
            elif policy == "validity_only":
                self._state = CONTACT_ON if int(valid_count) else CONTACT_OFF
            elif self._unknown_count > self.config.max_unknown_hold_frames:
                self._expire_unknown_hold()
        else:
            self._unknown_count = 0
            if global_count_mode:
                on_evidence = on_active_count >= self.config.min_active_markers
                off_contact = off_active_count >= self.config.min_active_markers
            else:
                on_evidence = self._has_contact(
                    scores, active, region_valid, self.on_threshold
                )
                off_contact = self._has_contact(
                    scores, active, region_valid, self.off_threshold
                )
            if self.config.mode == "hard":
                self._state = CONTACT_ON if on_evidence else CONTACT_OFF
            elif on_evidence:
                self._off_count = 0
                self._on_count += 1
                if self._state in {CONTACT_ON, CONTACT_HOLD} or (
                    self._on_count >= self.config.on_consecutive_frames
                ):
                    self._state = CONTACT_ON
                    self._on_count = 0
                    self._hold_remaining = 0
            elif not off_contact:
                self._transition_off_evidence()
            else:
                self._on_count = 0
                self._off_count = 0
        return MarkerGateFrame(
            state=self._state_code(unknown),
            region_scores=scores,
            regional_soft_gates=soft,
            region_valid_mask=region_valid,
            active_counts=active,
            on_active_marker_count=on_active_count,
            off_active_marker_count=off_active_count,
            valid_marker_count=int(valid_count),
            tracking_unknown=unknown,
        )

    def run_episode(self, displacement: Any, valid_mask: Any) -> MarkerGateEpisodeOutput:
        displacement = np.asarray(displacement, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if displacement.ndim != 3 or displacement.shape[-2:] != (len(self.region_ids), 2):
            raise ValueError("Episode displacement must be [T,N,2]")
        if valid.shape != displacement.shape[:-1]:
            raise ValueError("Episode valid mask must be [T,N]")
        self.reset()
        frames = [self.step(displacement[index], valid[index]) for index in range(len(displacement))]
        return MarkerGateEpisodeOutput(
            states=np.asarray([frame.state for frame in frames], dtype=np.int8),
            region_scores=np.stack([frame.region_scores for frame in frames]),
            regional_soft_gates=np.stack([frame.regional_soft_gates for frame in frames]),
            region_valid_mask=np.stack([frame.region_valid_mask for frame in frames]),
            active_counts=np.stack([frame.active_counts for frame in frames]),
            on_active_marker_count=np.asarray(
                [frame.on_active_marker_count for frame in frames], dtype=np.int64
            ),
            off_active_marker_count=np.asarray(
                [frame.off_active_marker_count for frame in frames], dtype=np.int64
            ),
            tracking_unknown=np.asarray(
                [frame.tracking_unknown for frame in frames], dtype=np.bool_
            ),
        )


__all__ = [
    "CONTACT_HOLD",
    "CONTACT_OFF",
    "CONTACT_ON",
    "CONTACT_UNKNOWN_HOLD",
    "CONTACT_UNKNOWN_OFF",
    "CONTACT_UNKNOWN_ON",
    "MarkerContactGate",
    "MarkerGateEpisodeOutput",
    "MarkerGateFrame",
    "build_marker_region_mapping_numpy",
    "contact_state_is_visible_numpy",
    "runtime_gate_config_from_mapping",
]
