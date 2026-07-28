import time
from pathlib import Path

import numpy as np
import pytest

from deploy.tacthru_umi_v2.realman_runtime import (
    ActionPlan,
    RealmanConfig,
    RealmanEpisodeRuntime,
    SafetyViolation,
)
from deploy.tacthru_umi_v2.transforms import pose_mat_from_rotvec


class FakeRobot:
    def __init__(self, pose):
        self.pose = pose
        self.timestamp = None
        self.executions = []
        self.start_episode_calls = 0
        self.last_waypoint_debug = None

    def get(self):
        from scipy.spatial.transform import Rotation

        return {
            "timestamp": time.time() if self.timestamp is None else self.timestamp,
            "eef_pos": self.pose[:3, 3],
            "eef_rot_axis_angle": Rotation.from_matrix(self.pose[:3, :3]).as_rotvec(),
        }

    def start_episode(self):
        self.start_episode_calls += 1

    def execute_waypoints(self, *, umi_pose, gripper_width, timestamps):
        self.executions.append((umi_pose.copy(), gripper_width.copy(), timestamps.copy()))


class FakeGripper:
    def __init__(self):
        self.commands = []

    def goto_pos(self, width):
        self.commands.append(float(width))


class FakeStartupGripper:
    def __init__(self, *, width_m: float, feedback_valid: bool = True):
        self.width_m = float(width_m)
        self.feedback_valid = bool(feedback_valid)
        self.events = []

    def start(self, wait=False):
        self.events.append(("start", bool(wait)))

    def start_wait(self):
        self.events.append(("start_wait",))

    def start_episode(self):
        self.events.append(("start_episode",))

    def is_alive(self):
        return True

    def get_all_state(self):
        self.events.append(("feedback",))
        return {
            "gripper_timestamp": np.asarray([time.time()]),
            "gripper_width": np.asarray([self.width_m]),
            "gripper_feedback_valid": np.asarray([int(self.feedback_valid)], dtype=np.uint8),
        }


class FakeAdapterController:
    @staticmethod
    def get_eef_coll_points(poses, _gripper_width):
        points = poses[:, None, :3, 3].copy()
        points[..., 2] -= 0.10
        return points


class FakeAdapterRobot(FakeRobot):
    table_collision_height_threshold = 0.005

    def __init__(self, pose):
        super().__init__(pose)
        self.controller = FakeAdapterController()

    @staticmethod
    def umi_to_tcp_pose(poses):
        return np.asarray(poses).copy()


def make_runtime(**overrides) -> RealmanEpisodeRuntime:
    kwargs = dict(
        tacthru_repo=Path("/tmp/tacthru"),
        robot_cfg=Path("realman.yaml"),
        gripper_cfg=None,
        exec_start_step=2,
        exec_end_step=6,
        robot_action_latency_s=0.0,
        max_pos_speed_m_s=0.1,
        max_rot_speed_rad_s=1.0,
        max_target_delta_m=0.2,
        max_target_rotation_rad=1.0,
        max_step_delta_m=0.05,
        max_step_rotation_rad=0.5,
    )
    kwargs.update(overrides)
    runtime = RealmanEpisodeRuntime(RealmanConfig(**kwargs))
    runtime.base_start_pose = pose_mat_from_rotvec([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    runtime.robot = FakeRobot(runtime.base_start_pose.copy())
    return runtime


def make_actions() -> np.ndarray:
    actions = np.zeros((50, 8), dtype=np.float32)
    actions[:, 0] = np.arange(50) * 0.002
    actions[:, 6] = 1.0
    actions[:, 7] = 0.03
    return actions


def make_binary_gripper_runtime(**overrides) -> RealmanEpisodeRuntime:
    runtime = make_runtime(exec_end_step=3, **overrides)
    runtime._gripper_action_select = "threshold"
    runtime._gripper_hold_closed_below_m = 0.010
    runtime._gripper_hold_closed_target_m = 0.004
    runtime._gripper_initial_width_m = 0.045
    runtime._gripper_max_width_m = 0.050
    return runtime


def test_binary_gripper_closes_at_or_below_threshold_and_opens_only_above_it() -> None:
    runtime = make_binary_gripper_runtime()
    assert runtime._select_gripper_target(np.asarray([0.009]), update_hold=True) == pytest.approx(0.004)
    assert runtime._select_gripper_target(np.asarray([0.010]), update_hold=True) == pytest.approx(0.004)
    assert runtime._select_gripper_target(np.asarray([0.011]), update_hold=True) == pytest.approx(0.045)
    assert runtime._select_gripper_target(np.asarray([0.020]), update_hold=True) == pytest.approx(0.045)
    assert runtime._select_gripper_target(np.asarray([0.020, 0.010]), update_hold=True) == pytest.approx(0.004)
    assert runtime._held_gripper_width_m is None


def test_binary_gripper_lookahead_opens_on_consecutive_late_chunk_predictions() -> None:
    runtime = make_binary_gripper_runtime(
        gripper_open_lookahead_start_step=25,
        gripper_open_lookahead_consecutive_steps=5,
    )
    full_widths = np.full(50, 0.006, dtype=np.float64)
    full_widths[30:35] = 0.012
    decision = {}

    target = runtime._select_gripper_target(
        np.full(6, 0.006),
        update_hold=True,
        lookahead_widths=full_widths,
        decision_debug=decision,
    )

    assert target == pytest.approx(0.045)
    assert decision["selected_all_above_threshold"] is False
    assert decision["lookahead_triggered"] is True
    assert decision["lookahead_open_run_start_steps"] == [30]
    assert decision["decision"] == "open"


def test_binary_gripper_lookahead_ignores_early_or_too_short_runs() -> None:
    runtime = make_binary_gripper_runtime(
        gripper_open_lookahead_start_step=25,
        gripper_open_lookahead_consecutive_steps=5,
    )
    full_widths = np.full(50, 0.006, dtype=np.float64)
    full_widths[20:25] = 0.020
    full_widths[30:34] = 0.020
    decision = {}

    target = runtime._select_gripper_target(
        np.full(6, 0.006),
        update_hold=True,
        lookahead_widths=full_widths,
        decision_debug=decision,
    )

    assert target == pytest.approx(0.004)
    assert decision["lookahead_triggered"] is False
    assert decision["lookahead_open_run_start_steps"] == []


def test_binary_gripper_lookahead_is_disabled_by_default() -> None:
    runtime = make_binary_gripper_runtime()
    full_widths = np.full(50, 0.020, dtype=np.float64)
    decision = {}

    target = runtime._select_gripper_target(
        np.full(6, 0.006),
        update_hold=True,
        lookahead_widths=full_widths,
        decision_debug=decision,
    )

    assert target == pytest.approx(0.004)
    assert decision["lookahead_enabled"] is False


def test_binary_gripper_lookahead_requires_complete_configuration() -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        make_runtime(gripper_open_lookahead_start_step=25)


def test_gripper_startup_is_verified_before_episode_is_enabled(monkeypatch) -> None:
    runtime = make_runtime(
        gripper_startup_width_m=0.004,
        gripper_startup_tolerance_m=0.001,
        gripper_startup_timeout_s=0.1,
    )
    gripper = FakeStartupGripper(width_m=0.004)
    monkeypatch.setattr(runtime, "_load_gripper_cfg", lambda: {})
    monkeypatch.setattr(runtime, "_create_gripper", lambda _cfg: gripper)

    verification = runtime.enable_actuation()

    assert verification == pytest.approx(
        {"target_width_m": 0.004, "actual_width_m": 0.004, "error_m": 0.0}
    )
    assert runtime.actuation_enabled is True
    assert runtime.robot.start_episode_calls == 1
    assert gripper.events[:3] == [("start", False), ("start_wait",), ("feedback",)]
    assert gripper.events[-1] == ("start_episode",)


def test_invalid_gripper_feedback_blocks_episode_start(monkeypatch) -> None:
    runtime = make_runtime(
        gripper_startup_width_m=0.004,
        gripper_startup_tolerance_m=0.001,
        gripper_startup_timeout_s=0.03,
    )
    gripper = FakeStartupGripper(width_m=0.004, feedback_valid=False)
    monkeypatch.setattr(runtime, "_load_gripper_cfg", lambda: {})
    monkeypatch.setattr(runtime, "_create_gripper", lambda _cfg: gripper)

    with pytest.raises(SafetyViolation, match="valid hardware feedback"):
        runtime.enable_actuation()

    assert runtime.actuation_enabled is False
    assert runtime.robot.start_episode_calls == 0
    assert ("start_episode",) not in gripper.events


def test_startup_width_overrides_hardware_init_but_not_policy_open_target(monkeypatch) -> None:
    import sys
    import types

    captured = {}

    class StubController:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    gloria_controller = types.ModuleType("real_world.grippers.gloria_controller")
    gloria_controller.GloriaGripperController = StubController
    monkeypatch.setitem(sys.modules, "real_world.grippers.gloria_controller", gloria_controller)
    runtime = make_runtime(gripper_startup_width_m=0.004)
    runtime.shm_manager = object()
    gripper_cfg = {
        "driver": "gloria",
        "port": "/dev/null",
        "gripper_type": "50mm",
        "initial_width_mm": 45.0,
        "episode_start_width_mm": None,
    }

    runtime._configure_gripper_policy(gripper_cfg)
    runtime._create_gripper(gripper_cfg)

    assert captured["kwargs"]["initial_width_mm"] == pytest.approx(4.0)
    assert captured["kwargs"]["episode_start_width_mm"] == pytest.approx(4.0)
    assert runtime._gripper_initial_width_m == pytest.approx(0.045)


def test_binary_gripper_rejects_empty_or_nonfinite_predictions() -> None:
    runtime = make_binary_gripper_runtime()
    with pytest.raises(ValueError, match="empty prediction window"):
        runtime._select_gripper_target(np.asarray([]), update_hold=True)
    with pytest.raises(ValueError, match="no finite predicted widths"):
        runtime._select_gripper_target(np.asarray([np.nan, np.inf]), update_hold=True)


def test_binary_gripper_rejects_invalid_width_order() -> None:
    runtime = make_binary_gripper_runtime()
    runtime._gripper_hold_closed_target_m = 0.020
    with pytest.raises(ValueError, match="closed_width < threshold"):
        runtime._select_gripper_target(np.asarray([0.011]), update_hold=True)


def test_binary_gripper_execution_uses_actual_width_for_adapter_safety() -> None:
    runtime = make_binary_gripper_runtime()
    runtime._actuation_enabled = True
    runtime.gripper = FakeGripper()
    plan = ActionPlan(
        selected_indices=np.asarray([2]),
        selected_actions=make_actions()[2:3],
        target_base_poses=np.asarray([runtime.base_start_pose]),
        validated_controller_poses=np.asarray([runtime.base_start_pose]),
        gripper_width_m=np.asarray([0.020]),
        timestamps=np.asarray([time.time() + 1.0]),
        debug={"safe": True},
    )

    execution = runtime.execute_plan(plan)

    assert runtime.robot.executions[0][1] == pytest.approx([0.045])
    assert runtime.gripper.commands == pytest.approx([0.045])
    assert execution["gripper_command_width_m"] == pytest.approx(0.045)
    assert execution["gripper_held_width_m"] is None


def test_binary_gripper_plan_uses_actual_width_for_adapter_safety() -> None:
    runtime = make_binary_gripper_runtime(max_target_delta_m=0.2, max_step_delta_m=0.2)
    runtime.robot = FakeAdapterRobot(runtime.base_start_pose.copy())
    actions = make_actions()
    actions[:, 7] = 0.020
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.045], dtype=np.float32)

    plan = runtime.plan_action_chunk(
        actions,
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=100.0,
    )

    assert plan.debug["gripper_command_preview_m"] == pytest.approx(0.045)
    assert plan.debug["adapter_safety_gripper_width_m"] == pytest.approx([0.045])


def test_plan_keeps_short_motion_window_but_uses_full_chunk_for_gripper_lookahead() -> None:
    runtime = make_binary_gripper_runtime(
        max_target_delta_m=0.2,
        max_step_delta_m=0.2,
        gripper_open_lookahead_start_step=25,
        gripper_open_lookahead_consecutive_steps=5,
    )
    runtime.robot = FakeAdapterRobot(runtime.base_start_pose.copy())
    actions = make_actions()
    actions[:, 7] = 0.006
    actions[30:35, 7] = 0.020
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.006], dtype=np.float32)

    plan = runtime.plan_action_chunk(
        actions,
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=100.0,
    )

    assert plan.selected_indices.tolist() == [2]
    assert plan.target_base_poses[:, 0, 3] == pytest.approx([0.504])
    assert plan.gripper_width_m == pytest.approx([0.006])
    assert plan.full_gripper_width_m.shape == (50,)
    assert plan.debug["gripper_policy"]["lookahead_open_run_start_steps"] == [30]
    assert plan.debug["gripper_command_preview_m"] == pytest.approx(0.045)
    assert plan.debug["adapter_safety_gripper_width_m"] == pytest.approx([0.045])


def test_execution_repeats_full_chunk_lookahead_decision_before_dispatch() -> None:
    runtime = make_binary_gripper_runtime(
        gripper_open_lookahead_start_step=25,
        gripper_open_lookahead_consecutive_steps=5,
    )
    runtime._actuation_enabled = True
    runtime.gripper = FakeGripper()
    full_widths = np.full(50, 0.006, dtype=np.float64)
    full_widths[30:35] = 0.020
    plan = ActionPlan(
        selected_indices=np.asarray([2]),
        selected_actions=make_actions()[2:3],
        target_base_poses=np.asarray([runtime.base_start_pose]),
        validated_controller_poses=np.asarray([runtime.base_start_pose]),
        gripper_width_m=np.asarray([0.006]),
        timestamps=np.asarray([time.time() + 1.0]),
        debug={"safe": True},
        full_gripper_width_m=full_widths,
    )

    execution = runtime.execute_plan(plan)

    assert runtime.robot.executions[0][1] == pytest.approx([0.045])
    assert runtime.gripper.commands == pytest.approx([0.045])
    assert execution["gripper_command_width_m"] == pytest.approx(0.045)


def test_plan_uses_episode_start_frame_and_selected_short_window() -> None:
    runtime = make_runtime()
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    plan = runtime.plan_action_chunk(
        make_actions(),
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=100.0,
    )

    assert plan.selected_indices.tolist() == [2, 3, 4, 5]
    assert plan.target_base_poses[:, 0, 3] == pytest.approx([0.504, 0.506, 0.508, 0.510])
    assert plan.gripper_width_m == pytest.approx([0.03] * 4)
    assert np.all(np.diff(plan.timestamps) > 0)


def test_non_streaming_plan_does_not_skip_actions_during_inference_delay() -> None:
    runtime = make_runtime()
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    plan = runtime.plan_action_chunk(
        make_actions(),
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=101.2,
    )

    assert plan.selected_indices.tolist() == [2, 3, 4, 5]
    assert plan.debug["online_delay_s"] == pytest.approx(1.2)
    assert plan.debug["latency_compensation_enabled"] is False
    assert plan.debug["delay_steps"] == 0


def test_streaming_plan_can_compensate_when_an_older_trajectory_is_active() -> None:
    runtime = make_runtime()
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    plan = runtime.plan_action_chunk(
        make_actions(),
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        compensate_inference_latency=True,
        now=100.5,
    )

    assert plan.selected_indices.tolist() == [15, 16, 17, 18]
    assert plan.debug["latency_compensation_enabled"] is True
    assert plan.debug["delay_steps"] == 15


def test_first_waypoint_jump_is_reported_separately_from_consecutive_waypoints() -> None:
    runtime = make_runtime(max_step_delta_m=0.04, max_target_delta_m=0.2)
    actions = make_actions()
    actions[:, 0] = 0.06
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)

    with pytest.raises(SafetyViolation, match="first waypoint translation from current"):
        runtime.plan_action_chunk(
            actions,
            observation_state=observation_state,
            observation_timestamp=100.0,
            control_frequency_hz=30.0,
            now=100.0,
        )


def test_large_jump_is_rejected_before_any_robot_call() -> None:
    runtime = make_runtime(max_target_delta_m=0.05)
    actions = make_actions()
    actions[:, 0] = 0.3
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)

    with pytest.raises(SafetyViolation, match="target translation"):
        runtime.plan_action_chunk(
            actions,
            observation_state=observation_state,
            observation_timestamp=100.0,
            control_frequency_hz=30.0,
            now=100.0,
        )
    assert runtime.robot.executions == []


def test_dry_run_execute_plan_refuses_all_actuator_calls() -> None:
    runtime = make_runtime()
    runtime.gripper = FakeGripper()
    plan = ActionPlan(
        selected_indices=np.asarray([2]),
        selected_actions=make_actions()[2:3],
        target_base_poses=np.asarray([runtime.base_start_pose]),
        validated_controller_poses=np.asarray([runtime.base_start_pose]),
        gripper_width_m=np.asarray([0.03]),
        timestamps=np.asarray([time.time() + 1.0]),
        debug={"safe": True},
    )

    with pytest.raises(RuntimeError, match="Actuation is disabled"):
        runtime.execute_plan(plan)
    assert runtime.robot.executions == []
    assert runtime.gripper.commands == []


def test_repeated_pose_chunk_still_gets_strictly_increasing_timestamps() -> None:
    runtime = make_runtime(max_pos_speed_m_s=0.1, max_step_delta_m=0.051)
    actions = make_actions()
    actions[:, 0] = 0.05
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    plan = runtime.plan_action_chunk(
        actions,
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=100.0,
    )
    assert np.all(np.diff(plan.timestamps) > 0.0)
    assert float(np.min(np.diff(plan.timestamps))) >= 1.0 / 30.0 - 1e-9


def test_nan_safety_limit_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_target_delta_m"):
        make_runtime(max_target_delta_m=float("nan"))


def test_adapter_table_lift_is_checked_against_workspace_before_dispatch() -> None:
    runtime = make_runtime(
        workspace_min_xyz_m=(0.0, -1.0, -1.0),
        workspace_max_xyz_m=(1.0, 1.0, 0.05),
    )
    runtime.robot = FakeAdapterRobot(runtime.base_start_pose.copy())
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    with pytest.raises(SafetyViolation, match="workspace upper"):
        runtime.plan_action_chunk(
            make_actions(),
            observation_state=observation_state,
            observation_timestamp=100.0,
            control_frequency_hz=30.0,
            now=100.0,
        )


def test_stale_robot_ring_buffer_state_is_rejected() -> None:
    runtime = make_runtime(max_robot_state_age_s=0.05)
    runtime.robot.timestamp = time.time() - 1.0
    with pytest.raises(TimeoutError, match="fresh Realman state"):
        runtime._read_robot_state(timeout_s=0.01)
