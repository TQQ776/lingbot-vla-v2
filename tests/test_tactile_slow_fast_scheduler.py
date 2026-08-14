from __future__ import annotations

import numpy as np

from deploy.tacthru_umi_v2.slow_fast_scheduler import (
    SlowFastValiditySettings,
    SlowPlanRuntimeState,
    evaluate_slow_plan,
)


def _state(*, x=0.0, rotation_z=0.0, gripper=0.004):
    half = rotation_z / 2.0
    return np.asarray(
        [x, 0.0, 0.0, 0.0, 0.0, np.sin(half), np.cos(half), gripper],
        dtype=np.float32,
    )


def _plan(**kwargs):
    values = {
        "state": _state(),
        "created_at_s": 10.0,
        "prefix_created_at_s": 10.0,
        "executed_offset": 0,
        "instruction": "Insert the Ethernet cable",
        "scene_version": 3,
        "plan_version": 1,
    }
    values.update(kwargs)
    return SlowPlanRuntimeState(**values)


def test_missing_or_changed_scene_requires_full_slow_rebuild():
    settings = SlowFastValiditySettings()
    missing = evaluate_slow_plan(
        None,
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=0,
        settings=settings,
        now_s=10.1,
    )
    changed = evaluate_slow_plan(
        _plan(),
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=4,
        executed_offset=0,
        settings=settings,
        now_s=10.1,
    )
    assert (missing.level, missing.reason) == ("rebuild", "missing_plan")
    assert (changed.level, changed.reason) == (
        "rebuild",
        "scene_version_changed",
    )


def test_validity_uses_separate_pose_rotation_gripper_and_horizon_limits():
    settings = SlowFastValiditySettings()
    reuse = evaluate_slow_plan(
        _plan(),
        current_state=_state(x=0.002, rotation_z=0.02, gripper=0.005),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=4,
        settings=settings,
        now_s=10.5,
    )
    refresh = evaluate_slow_plan(
        _plan(),
        current_state=_state(x=0.010),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=4,
        settings=settings,
        now_s=10.5,
    )
    rebuild = evaluate_slow_plan(
        _plan(),
        current_state=_state(rotation_z=0.20),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=4,
        settings=settings,
        now_s=10.5,
    )
    assert reuse.level == "reuse"
    assert (refresh.level, refresh.reason) == ("refresh_action", "position_drift")
    assert (rebuild.level, rebuild.reason) == ("rebuild", "rotation_drift")


def test_plan_age_and_executed_offset_have_two_refresh_tiers():
    settings = SlowFastValiditySettings()
    refresh = evaluate_slow_plan(
        _plan(),
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=9,
        settings=settings,
        now_s=10.5,
    )
    rebuild = evaluate_slow_plan(
        _plan(),
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=25,
        settings=settings,
        now_s=10.5,
    )
    age_rebuild = evaluate_slow_plan(
        _plan(),
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=0,
        settings=settings,
        now_s=15.1,
    )
    assert (refresh.level, refresh.reason) == (
        "refresh_action",
        "executed_horizon",
    )
    assert (rebuild.level, rebuild.reason) == ("rebuild", "executed_horizon")
    assert (age_rebuild.level, age_rebuild.reason) == ("rebuild", "prefix_age")


def test_action_refresh_does_not_reset_prefix_age():
    settings = SlowFastValiditySettings()
    refreshed_action = _plan(created_at_s=14.8, prefix_created_at_s=10.0)

    decision = evaluate_slow_plan(
        refreshed_action,
        current_state=_state(),
        instruction="Insert the Ethernet cable",
        scene_version=3,
        executed_offset=0,
        settings=settings,
        now_s=15.1,
    )

    assert decision.age_s < settings.reuse_max_age_s
    assert decision.prefix_age_s > settings.action_refresh_max_age_s
    assert (decision.level, decision.reason) == ("rebuild", "prefix_age")
