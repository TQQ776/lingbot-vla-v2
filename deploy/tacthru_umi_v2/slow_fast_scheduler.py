"""Validity policy for reusing cached three-stream VTLA Slow plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import numpy as np

from .transforms import GRIPPER_INDEX, POSITION_SLICE, QUATERNION_SLICE, validate_state8


@dataclass(frozen=True)
class SlowFastValiditySettings:
    reuse_max_age_s: float = 1.0
    action_refresh_max_age_s: float = 5.0
    reuse_max_position_drift_m: float = 0.005
    action_refresh_max_position_drift_m: float = 0.020
    reuse_max_rotation_drift_rad: float = 0.05
    action_refresh_max_rotation_drift_rad: float = 0.15
    reuse_max_gripper_drift_m: float = 0.003
    action_refresh_max_gripper_drift_m: float = 0.010
    reuse_max_executed_offset: int = 8
    action_refresh_max_executed_offset: int = 24

    @classmethod
    def from_env(cls) -> "SlowFastValiditySettings":
        defaults = cls()

        def floating(name: str, default: float) -> float:
            value = float(os.environ.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value, got {value}")
            return value

        def integer(name: str, default: int) -> int:
            value = int(os.environ.get(name, default))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            return value

        result = cls(
            reuse_max_age_s=floating(
                "LINGBOT_V2_VTLA_REUSE_MAX_AGE_S", defaults.reuse_max_age_s
            ),
            action_refresh_max_age_s=floating(
                "LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_AGE_S",
                defaults.action_refresh_max_age_s,
            ),
            reuse_max_position_drift_m=floating(
                "LINGBOT_V2_VTLA_REUSE_MAX_POSITION_DRIFT_M",
                defaults.reuse_max_position_drift_m,
            ),
            action_refresh_max_position_drift_m=floating(
                "LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_POSITION_DRIFT_M",
                defaults.action_refresh_max_position_drift_m,
            ),
            reuse_max_rotation_drift_rad=floating(
                "LINGBOT_V2_VTLA_REUSE_MAX_ROTATION_DRIFT_RAD",
                defaults.reuse_max_rotation_drift_rad,
            ),
            action_refresh_max_rotation_drift_rad=floating(
                "LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_ROTATION_DRIFT_RAD",
                defaults.action_refresh_max_rotation_drift_rad,
            ),
            reuse_max_gripper_drift_m=floating(
                "LINGBOT_V2_VTLA_REUSE_MAX_GRIPPER_DRIFT_M",
                defaults.reuse_max_gripper_drift_m,
            ),
            action_refresh_max_gripper_drift_m=floating(
                "LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_GRIPPER_DRIFT_M",
                defaults.action_refresh_max_gripper_drift_m,
            ),
            reuse_max_executed_offset=integer(
                "LINGBOT_V2_VTLA_REUSE_MAX_EXECUTED_OFFSET",
                defaults.reuse_max_executed_offset,
            ),
            action_refresh_max_executed_offset=integer(
                "LINGBOT_V2_VTLA_ACTION_REFRESH_MAX_EXECUTED_OFFSET",
                defaults.action_refresh_max_executed_offset,
            ),
        )
        for reuse_name, refresh_name in (
            ("reuse_max_age_s", "action_refresh_max_age_s"),
            ("reuse_max_position_drift_m", "action_refresh_max_position_drift_m"),
            ("reuse_max_rotation_drift_rad", "action_refresh_max_rotation_drift_rad"),
            ("reuse_max_gripper_drift_m", "action_refresh_max_gripper_drift_m"),
            ("reuse_max_executed_offset", "action_refresh_max_executed_offset"),
        ):
            if getattr(result, reuse_name) > getattr(result, refresh_name):
                raise ValueError(f"{reuse_name} must not exceed {refresh_name}")
        return result


@dataclass(frozen=True)
class SlowPlanRuntimeState:
    state: np.ndarray
    created_at_s: float
    prefix_created_at_s: float
    executed_offset: int
    instruction: str
    scene_version: int
    plan_version: int


@dataclass(frozen=True)
class SlowPlanDecision:
    level: str
    reason: str
    age_s: float
    prefix_age_s: float
    position_drift_m: float
    rotation_drift_rad: float
    gripper_drift_m: float
    executed_offset_delta: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "level": self.level,
            "reason": self.reason,
            "age_s": self.age_s,
            "prefix_age_s": self.prefix_age_s,
            "position_drift_m": self.position_drift_m,
            "rotation_drift_rad": self.rotation_drift_rad,
            "gripper_drift_m": self.gripper_drift_m,
            "executed_offset_delta": self.executed_offset_delta,
        }


def evaluate_slow_plan(
    plan: SlowPlanRuntimeState | None,
    *,
    current_state: np.ndarray,
    instruction: str,
    scene_version: int,
    executed_offset: int,
    settings: SlowFastValiditySettings,
    now_s: float | None = None,
) -> SlowPlanDecision:
    """Choose Prefix reuse, Action refresh, or a complete Slow rebuild."""

    if plan is None:
        return SlowPlanDecision(
            "rebuild", "missing_plan", 0.0, 0.0, 0.0, 0.0, 0.0, 0
        )
    current = validate_state8(current_state)
    previous = validate_state8(plan.state)
    current_time_s = time.monotonic() if now_s is None else now_s
    age_s = max(0.0, current_time_s - plan.created_at_s)
    prefix_age_s = max(0.0, current_time_s - plan.prefix_created_at_s)
    position_drift = float(
        np.linalg.norm(current[POSITION_SLICE] - previous[POSITION_SLICE])
    )
    rotation_drift = _quaternion_distance_rad(
        current[QUATERNION_SLICE], previous[QUATERNION_SLICE]
    )
    gripper_drift = abs(float(current[GRIPPER_INDEX] - previous[GRIPPER_INDEX]))
    offset_delta = max(0, int(executed_offset) - int(plan.executed_offset))

    metrics = (
        age_s,
        prefix_age_s,
        position_drift,
        rotation_drift,
        gripper_drift,
        offset_delta,
    )

    def decision(level: str, reason: str) -> SlowPlanDecision:
        return SlowPlanDecision(level, reason, *metrics)

    if instruction != plan.instruction:
        return decision("rebuild", "instruction_changed")
    if int(scene_version) != int(plan.scene_version):
        return decision("rebuild", "scene_version_changed")
    rebuild_checks = (
        (prefix_age_s > settings.action_refresh_max_age_s, "prefix_age"),
        (
            position_drift > settings.action_refresh_max_position_drift_m,
            "position_drift",
        ),
        (
            rotation_drift > settings.action_refresh_max_rotation_drift_rad,
            "rotation_drift",
        ),
        (
            gripper_drift > settings.action_refresh_max_gripper_drift_m,
            "gripper_drift",
        ),
        (
            offset_delta > settings.action_refresh_max_executed_offset,
            "executed_horizon",
        ),
    )
    for stale, reason in rebuild_checks:
        if stale:
            return decision("rebuild", reason)
    refresh_checks = (
        (age_s > settings.reuse_max_age_s, "plan_age"),
        (position_drift > settings.reuse_max_position_drift_m, "position_drift"),
        (rotation_drift > settings.reuse_max_rotation_drift_rad, "rotation_drift"),
        (gripper_drift > settings.reuse_max_gripper_drift_m, "gripper_drift"),
        (offset_delta > settings.reuse_max_executed_offset, "executed_horizon"),
    )
    for stale, reason in refresh_checks:
        if stale:
            return decision("refresh_action", reason)
    return decision("reuse", "within_reuse_limits")


def _quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)
