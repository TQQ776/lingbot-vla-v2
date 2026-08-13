"""Shared contact-state logic for offline VTLA training and online deployment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch import Tensor

from lingbotvla.tactile_contact import (
    CONTACT_HOLD,
    CONTACT_OFF,
    CONTACT_ON,
    CONTACT_UNKNOWN_HOLD,
    CONTACT_UNKNOWN_OFF,
    CONTACT_UNKNOWN_ON,
    MarkerContactGate,
    MarkerGateEpisodeOutput,
    MarkerGateFrame,
)

CONTACT_STATE_NAMES = {
    CONTACT_OFF: "off",
    CONTACT_ON: "on",
    CONTACT_HOLD: "hold",
    CONTACT_UNKNOWN_OFF: "unknown_off",
    CONTACT_UNKNOWN_ON: "unknown_on",
    CONTACT_UNKNOWN_HOLD: "unknown_hold",
}


def contact_state_is_visible(state: Tensor) -> Tensor:
    """Return whether marker tokens should be visible for encoded state values."""

    return (state == CONTACT_ON) | (state == CONTACT_HOLD) | (
        state == CONTACT_UNKNOWN_ON
    ) | (state == CONTACT_UNKNOWN_HOLD)


def contact_state_is_unknown(state: Tensor) -> Tensor:
    return state >= CONTACT_UNKNOWN_OFF


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _per_sensor_scalars(
    value: Any,
    *,
    name: str,
    num_sensors: int,
) -> tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(_finite_float(item, name=name) for item in value)
        if len(values) != num_sensors:
            raise ValueError(f"{name} must contain {num_sensors} values")
        return values
    scalar = _finite_float(value, name=name)
    return (scalar,) * num_sensors


def _per_sensor_point_thresholds(
    value: Any,
    *,
    num_sensors: int,
    num_markers: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        scalar = _finite_float(value, name="point_threshold")
        return ((scalar,) * num_markers,) * num_sensors
    values = list(value)
    if len(values) == num_markers and not any(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in values
    ):
        row = tuple(_finite_float(item, name="point_threshold") for item in values)
        return (row,) * num_sensors
    if len(values) != num_sensors:
        raise ValueError(
            "point_threshold must be a scalar, [N], or [S,N] threshold array"
        )
    rows = []
    for row in values:
        if not isinstance(row, Sequence) or len(row) != num_markers:
            raise ValueError(f"Every point_threshold row must contain {num_markers} values")
        rows.append(tuple(_finite_float(item, name="point_threshold") for item in row))
    return tuple(rows)


def _calibration_values(
    payload: Mapping[str, Any],
    sensor_names: Sequence[str],
) -> tuple[Any, Any, Any]:
    sensors = payload.get("sensors")
    if isinstance(sensors, Mapping):
        point, on, off = [], [], []
        for name in sensor_names:
            if name not in sensors or not isinstance(sensors[name], Mapping):
                raise ValueError(f"Contact calibration is missing sensor {name!r}")
            entry = sensors[name]
            point.append(entry.get("suggested_point_threshold"))
            on.append(entry.get("suggested_global_on_threshold"))
            off.append(entry.get("suggested_global_off_threshold"))
        return point, on, off
    return (
        payload.get("suggested_point_threshold"),
        payload.get("suggested_global_on_threshold"),
        payload.get("suggested_global_off_threshold"),
    )


@dataclass(frozen=True)
class MarkerContactGateConfig:
    enabled: bool = False
    target: str = "marker_only"
    mode: str = "none"
    score_type: str = "topk_mean"
    topk_markers: int = 3
    min_active_markers: int = 2
    min_valid_markers_per_region: int = 2
    min_valid_markers_global: int = 24
    threshold_source: str = "explicit"
    threshold_stats_path: str | None = None
    point_thresholds: tuple[tuple[float, ...], ...] = ()
    on_thresholds: tuple[float, ...] = ()
    off_thresholds: tuple[float, ...] = ()
    soft_thresholds: tuple[float, ...] = ()
    mad_multiplier: float = 6.0
    on_consecutive_frames: int = 2
    off_consecutive_frames: int = 3
    release_hold_frames: int = 3
    regional_soft_gate: bool = True
    soft_gate_temperature: float = 0.1
    unknown_tracking_policy: str = "hold_previous"
    max_unknown_hold_frames: int = 3

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        num_sensors: int,
        num_markers: int,
        sensor_names: Sequence[str],
    ) -> "MarkerContactGateConfig":
        values = dict(value or {})
        enabled = _read_bool(values.get("enabled"), False)
        mode = str(values.get("mode", "hard" if enabled else "none"))
        if not enabled:
            mode = "none"
        modes = {
            "none",
            "hard",
            "hard_hysteresis",
            "hard_hysteresis_hold",
            "hard_hysteresis_soft_region",
            "global_active_count_hysteresis_soft_region",
        }
        if mode not in modes:
            raise ValueError(f"Unsupported marker contact gate mode: {mode}")
        target = str(values.get("target", "marker_only"))
        if target not in {"marker_only", "marker_and_rgb"}:
            raise ValueError("marker_contact_gate.target must be marker_only or marker_and_rgb")
        if str(values.get("score_type", "topk_mean")) != "topk_mean":
            raise ValueError("Only marker_contact_gate.score_type=topk_mean is supported")

        threshold_source = str(values.get("threshold_source", "explicit"))
        if threshold_source not in {"explicit", "calibration"}:
            raise ValueError("threshold_source must be explicit or calibration")
        stats_path = values.get("threshold_stats_path")
        point_value = values.get("point_threshold")
        on_value = values.get("on_threshold")
        off_value = values.get("off_threshold")
        if enabled and mode != "none" and threshold_source == "calibration":
            has_resolved_thresholds = all(
                item is not None for item in (point_value, on_value, off_value)
            )
            if has_resolved_thresholds:
                pass
            elif stats_path is not None and Path(stats_path).expanduser().is_file():
                with Path(stats_path).expanduser().open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                calibrated = _calibration_values(payload, sensor_names)
                point_value = calibrated[0]
                on_value = calibrated[1]
                off_value = calibrated[2]
            else:
                raise FileNotFoundError(
                    f"Contact calibration file does not exist: {Path(str(stats_path)).expanduser()}"
                )

        if enabled and mode != "none":
            if point_value is None or on_value is None or off_value is None:
                raise ValueError(
                    "Enabled contact gating requires point_threshold, on_threshold, and "
                    "off_threshold, directly or from calibration"
                )
            point_thresholds = _per_sensor_point_thresholds(
                point_value,
                num_sensors=num_sensors,
                num_markers=num_markers,
            )
            on_thresholds = _per_sensor_scalars(
                on_value, name="on_threshold", num_sensors=num_sensors
            )
            off_thresholds = _per_sensor_scalars(
                off_value, name="off_threshold", num_sensors=num_sensors
            )
            if any(off >= on for off, on in zip(off_thresholds, on_thresholds)):
                raise ValueError("marker contact off_threshold must be less than on_threshold")
        else:
            point_thresholds = ((0.0,) * num_markers,) * num_sensors
            on_thresholds = (0.0,) * num_sensors
            off_thresholds = (0.0,) * num_sensors

        soft_value = values.get("soft_threshold")
        soft_thresholds = (
            _per_sensor_scalars(
                soft_value, name="soft_threshold", num_sensors=num_sensors
            )
            if soft_value is not None
            else off_thresholds
        )
        positive_fields = {
            "topk_markers": int(values.get("topk_markers", 3)),
            "min_active_markers": int(values.get("min_active_markers", 2)),
            "min_valid_markers_per_region": int(
                values.get("min_valid_markers_per_region", 2)
            ),
            "min_valid_markers_global": int(values.get("min_valid_markers_global", 24)),
            "on_consecutive_frames": int(values.get("on_consecutive_frames", 2)),
            "off_consecutive_frames": int(values.get("off_consecutive_frames", 3)),
        }
        if any(item <= 0 for item in positive_fields.values()):
            raise ValueError(f"Contact gate count fields must be positive: {positive_fields}")
        release_hold_frames = int(values.get("release_hold_frames", 3))
        max_unknown = int(values.get("max_unknown_hold_frames", 3))
        if release_hold_frames < 0 or max_unknown < 0:
            raise ValueError("hold frame counts must be non-negative")
        temperature = _finite_float(
            values.get("soft_gate_temperature", 0.1),
            name="soft_gate_temperature",
        )
        if temperature <= 0:
            raise ValueError("soft_gate_temperature must be positive")
        unknown_policy = str(values.get("unknown_tracking_policy", "hold_previous"))
        if unknown_policy not in {"hold_previous", "force_off", "force_on", "validity_only"}:
            raise ValueError(f"Unsupported unknown_tracking_policy: {unknown_policy}")

        return cls(
            enabled=enabled,
            target=target,
            mode=mode,
            score_type="topk_mean",
            threshold_source=threshold_source,
            threshold_stats_path=str(stats_path) if stats_path is not None else None,
            point_thresholds=point_thresholds,
            on_thresholds=on_thresholds,
            off_thresholds=off_thresholds,
            soft_thresholds=soft_thresholds,
            mad_multiplier=_finite_float(
                values.get("mad_multiplier", 6.0), name="mad_multiplier"
            ),
            release_hold_frames=release_hold_frames,
            regional_soft_gate=_read_bool(values.get("regional_soft_gate"), True),
            soft_gate_temperature=temperature,
            unknown_tracking_policy=unknown_policy,
            max_unknown_hold_frames=max_unknown,
            **positive_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["point_threshold"] = [list(row) for row in self.point_thresholds]
        result["on_threshold"] = list(self.on_thresholds)
        result["off_threshold"] = list(self.off_thresholds)
        result["soft_threshold"] = list(self.soft_thresholds)
        result.pop("point_thresholds")
        result.pop("on_thresholds")
        result.pop("off_thresholds")
        result.pop("soft_thresholds")
        return result


__all__ = [
    "CONTACT_HOLD",
    "CONTACT_OFF",
    "CONTACT_ON",
    "CONTACT_STATE_NAMES",
    "CONTACT_UNKNOWN_HOLD",
    "CONTACT_UNKNOWN_OFF",
    "CONTACT_UNKNOWN_ON",
    "MarkerContactGate",
    "MarkerContactGateConfig",
    "MarkerGateEpisodeOutput",
    "MarkerGateFrame",
    "contact_state_is_unknown",
    "contact_state_is_visible",
]
