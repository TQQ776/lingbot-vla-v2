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
    def __init__(self):
        self.width_m = 0.045
        self.commands = []
        self.start_episode_calls = 0

    def is_alive(self):
        return True

    def goto_pos(self, width):
        self.width_m = float(width)
        self.commands.append(self.width_m)

    def get_all_state(self):
        return {
            "gripper_width": np.asarray([self.width_m], dtype=np.float64),
            "gripper_timestamp": np.asarray([time.time()], dtype=np.float64),
        }

    def start_episode(self):
        self.start_episode_calls += 1


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


def test_fixed_exec_window_ignores_online_delay_index_shift() -> None:
    runtime = make_runtime(fixed_exec_window=True)
    observation_state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32)
    plan = runtime.plan_action_chunk(
        make_actions(),
        observation_state=observation_state,
        observation_timestamp=100.0,
        control_frequency_hz=30.0,
        now=101.0,
    )

    assert plan.debug["delay_steps"] == 30
    assert plan.debug["configured_exec_window"] == [2, 6]
    assert plan.debug["effective_exec_window"] == [2, 6]
    assert plan.debug["fixed_exec_window"] is True
    assert plan.selected_indices.tolist() == [2, 3, 4, 5]


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


def test_gripper_startup_is_verified_before_arm_actuation() -> None:
    runtime = make_runtime(
        gripper_initialize_on_start=False,
        gripper_episode_start_width_m=0.004,
    )
    gripper = FakeStartupGripper()
    runtime.gripper = gripper

    result = runtime.prepare_gripper_for_episode(
        0.004,
        tolerance_m=0.001,
        timeout_s=0.1,
    )

    assert result == {
        "target_width_m": pytest.approx(0.004),
        "actual_width_m": pytest.approx(0.004),
        "error_m": pytest.approx(0.0),
    }
    assert runtime.actuation_enabled is False
    assert runtime.robot.start_episode_calls == 0
    assert gripper.commands == [pytest.approx(0.004)]

    runtime.enable_actuation()
    assert runtime.actuation_enabled is True
    assert runtime.robot.start_episode_calls == 1
    assert gripper.start_episode_calls == 1


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
