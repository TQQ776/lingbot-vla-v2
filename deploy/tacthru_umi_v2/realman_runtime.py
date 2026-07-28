from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .transforms import (
    GRIPPER_INDEX,
    POSITION_SLICE,
    QUATERNION_SLICE,
    base_pose_from_episode_action,
    episode_state_from_base_poses,
    pose_mat_from_rotvec,
    pose_mat_from_xyz_quat,
    rotation_distance_rad,
    validate_action_chunk,
    validate_state8,
)


class SafetyViolation(RuntimeError):
    """Raised before any actuator call when a proposed chunk is unsafe."""


@dataclass(frozen=True)
class RealmanConfig:
    tacthru_repo: Path
    robot_cfg: Path
    gripper_cfg: Path | None
    robot_ip: str | None = None
    robot_port: int | None = None
    enable_gripper: bool = True
    assumed_gripper_width_m: float = 0.045
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float | None = None
    gripper_command_torque: int | None = None
    gripper_command_speed: int | None = None
    gripper_startup_width_m: float | None = None
    gripper_startup_tolerance_m: float = 0.005
    gripper_startup_timeout_s: float = 3.0
    gripper_action_select: str | None = None
    gripper_hold_closed_below_m: float | None = None
    gripper_hold_closed_target_m: float | None = None
    gripper_open_lookahead_start_step: int | None = None
    gripper_open_lookahead_consecutive_steps: int | None = None
    exec_start_step: int = 2
    exec_end_step: int = 8
    preserve_exec_window_length: bool = True
    robot_action_latency_s: float = 0.1
    max_pos_speed_m_s: float = 0.04
    max_rot_speed_rad_s: float = 0.08
    max_target_delta_m: float = 0.12
    max_target_rotation_rad: float = 1.2
    max_step_delta_m: float = 0.04
    max_step_rotation_rad: float = 0.5
    max_observation_drift_m: float = 0.05
    max_observation_drift_rotation_rad: float = 0.35
    max_robot_state_age_s: float = 0.25
    max_gripper_state_age_s: float = 0.50
    max_scheduled_duration_s: float = 3.0
    workspace_min_xyz_m: tuple[float, float, float] | None = None
    workspace_max_xyz_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PolicyStateSnapshot:
    state: np.ndarray
    base_pose: np.ndarray
    timestamp: float
    debug: dict[str, Any]


@dataclass(frozen=True)
class ActionPlan:
    selected_indices: np.ndarray
    selected_actions: np.ndarray
    target_base_poses: np.ndarray
    validated_controller_poses: np.ndarray
    gripper_width_m: np.ndarray
    timestamps: np.ndarray
    debug: dict[str, Any]
    full_gripper_width_m: np.ndarray | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.selected_indices) == 0


class RealmanEpisodeRuntime:
    """Thin TacThru Realman bridge for the V2 episode-relative pose contract.

    Starting the runtime sends no waypoint command: the arm controller is used
    for state polling, but ``start_episode`` and the Gloria gripper are not
    started until :meth:`enable_actuation` is called after explicit user
    confirmation. The Realman SDK still selects the configured run mode and
    ``new_tcp`` tool frame, so dry-run is motion-free rather than configuration-free.
    """

    def __init__(self, cfg: RealmanConfig) -> None:
        self.cfg = cfg
        self._validate_config()
        self.tacthru_repo = cfg.tacthru_repo.expanduser().resolve()
        self.shm_manager: SharedMemoryManager | None = None
        self.robot = None
        self.gripper = None
        self.base_start_pose: np.ndarray | None = None
        self._actuation_enabled = False
        self._gripper_max_width_m = float(cfg.gripper_max_width_m or 0.05)
        self._gripper_initial_width_m = float(cfg.assumed_gripper_width_m)
        self._last_gripper_width_m = float(cfg.assumed_gripper_width_m)
        self._held_gripper_width_m: float | None = None
        self._gripper_action_select = "last"
        self._gripper_hold_closed_below_m: float | None = None
        self._gripper_hold_closed_target_m: float | None = None
        self._warned_gripper_read_failure = False
        self._gripper_feedback_is_assumed = True

    @property
    def actuation_enabled(self) -> bool:
        return self._actuation_enabled

    def start(self) -> None:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Realman client must run in TacThru's Python 3.11 Realman environment")
        self._install_tacthru_path()
        robot_cfg = self._load_robot_cfg()
        gripper_cfg = self._load_gripper_cfg() if self.cfg.enable_gripper else None
        if self.cfg.gripper_max_width_m is None and gripper_cfg is not None:
            self._gripper_max_width_m = _infer_gripper_max_width_m(gripper_cfg)
        self._configure_gripper_policy(gripper_cfg)
        self._last_gripper_width_m = float(
            np.clip(self.cfg.assumed_gripper_width_m, self.cfg.gripper_min_width_m, self._gripper_max_width_m)
        )

        from real_world.robots.realman.adapter import RealmanAdapter

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.robot = RealmanAdapter(self.shm_manager, robot_cfg=robot_cfg, debug=False)
        self.robot.start(wait=False)
        self.robot.start_wait()
        self.base_start_pose, _ = self._read_current_pose_with_timestamp()
        print(
            "[lingbot-v2-client] Realman state polling is ready; no waypoint or gripper "
            "command has been sent (run mode/tool frame may have been selected)",
            flush=True,
        )

    def enable_actuation(self) -> dict[str, float] | None:
        if self.robot is None:
            raise RuntimeError("Runtime must be started before enabling actuation")
        if self._actuation_enabled:
            return None

        startup_verification = None
        if self.cfg.enable_gripper:
            gripper_cfg = self._load_gripper_cfg()
            self.gripper = self._create_gripper(gripper_cfg)
            startup_width = self.cfg.gripper_startup_width_m
            startup_detail = (
                f" to {startup_width * 1000.0:.1f}mm before inference"
                if startup_width is not None
                else ""
            )
            print(
                f"[lingbot-v2-client] enabling Gloria gripper{startup_detail}; initialization will move the gripper",
                flush=True,
            )
            self.gripper.start(wait=False)
            self.gripper.start_wait()
            if startup_width is not None:
                startup_verification = self._wait_for_gripper_target(
                    float(startup_width),
                    tolerance_m=self.cfg.gripper_startup_tolerance_m,
                    timeout_s=self.cfg.gripper_startup_timeout_s,
                )

        self._actuation_enabled = True
        self.reset_episode_start()
        return startup_verification

    def close(self) -> None:
        if self.gripper is not None:
            print(
                "[lingbot-v2-client] stopping Gloria now; the driver will disable servo torque",
                flush=True,
            )
        for component in (self.robot, self.gripper):
            if component is not None:
                try:
                    component.stop(wait=True)
                except Exception as exc:
                    print(
                        f"[lingbot-v2-client] warning: failed to stop {type(component).__name__}: {exc}",
                        flush=True,
                    )
        if self.shm_manager is not None:
            try:
                self.shm_manager.shutdown()
            except Exception as exc:
                print(f"[lingbot-v2-client] warning: failed to close shared memory: {exc}", flush=True)

    def reset_episode_start(self) -> None:
        if self.robot is None:
            raise RuntimeError("Runtime is not started")
        if self._actuation_enabled:
            self.robot.start_episode()
            if self.gripper is not None:
                self.gripper.start_episode()
        self.base_start_pose, _ = self._read_current_pose_with_timestamp()
        self._held_gripper_width_m = None
        print(
            f"[lingbot-v2-client] episode base_start_pos_m={self.base_start_pose[:3, 3].tolist()} "
            f"actuation_enabled={self._actuation_enabled}",
            flush=True,
        )

    def read_policy_state(self) -> PolicyStateSnapshot:
        if self.base_start_pose is None:
            raise RuntimeError("Episode start pose is not initialized")
        base_current, robot_timestamp = self._read_current_pose_with_timestamp()
        gripper_width = self._read_gripper_width()
        state = episode_state_from_base_poses(self.base_start_pose, base_current, gripper_width)
        now = time.time()
        return PolicyStateSnapshot(
            state=state,
            base_pose=base_current,
            timestamp=robot_timestamp,
            debug={
                "pose_frame": "episode_start_new_tcp",
                "base_start_pos_m": self.base_start_pose[:3, 3].astype(float).tolist(),
                "base_current_pos_m": base_current[:3, 3].astype(float).tolist(),
                "episode_current_xyz_m": state[POSITION_SLICE].astype(float).tolist(),
                "episode_current_quaternion_xyzw": state[QUATERNION_SLICE].astype(float).tolist(),
                "gripper_width_m": float(gripper_width),
                "gripper_width_source": "assumed" if self._gripper_feedback_is_assumed else "hardware_feedback",
                "robot_timestamp": float(robot_timestamp),
                "robot_state_age_s": float(now - robot_timestamp),
                "actuation_enabled": self._actuation_enabled,
            },
        )

    def plan_action_chunk(
        self,
        action_chunk: np.ndarray,
        *,
        observation_state: np.ndarray,
        observation_timestamp: float,
        control_frequency_hz: float,
        compensate_inference_latency: bool = False,
        now: float | None = None,
    ) -> ActionPlan:
        if self.base_start_pose is None:
            raise RuntimeError("Episode start pose is not initialized")
        if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0.0:
            raise ValueError(f"control_frequency_hz must be positive, got {control_frequency_hz}")
        _require_positive(self.cfg.max_pos_speed_m_s, "max_pos_speed_m_s")
        _require_positive(self.cfg.max_rot_speed_rad_s, "max_rot_speed_rad_s")

        actions = validate_action_chunk(
            action_chunk,
            min_gripper_width_m=self.cfg.gripper_min_width_m,
            max_gripper_width_m=self._gripper_max_width_m,
        )
        observation_state = validate_state8(
            observation_state,
            min_gripper_width_m=self.cfg.gripper_min_width_m,
            max_gripper_width_m=self._gripper_max_width_m,
        )
        now = time.time() if now is None else float(now)
        indices, timing_debug = self._select_action_indices(
            action_horizon=len(actions),
            observation_timestamp=float(observation_timestamp),
            control_frequency_hz=float(control_frequency_hz),
            now=now,
            compensate_inference_latency=bool(compensate_inference_latency),
        )
        selected = actions[indices]
        full_gripper = actions[:, GRIPPER_INDEX].astype(np.float64)
        if len(selected) == 0:
            return ActionPlan(
                selected_indices=indices,
                selected_actions=selected,
                target_base_poses=np.zeros((0, 4, 4), dtype=np.float64),
                validated_controller_poses=np.zeros((0, 4, 4), dtype=np.float64),
                gripper_width_m=np.zeros((0,), dtype=np.float64),
                timestamps=np.zeros((0,), dtype=np.float64),
                debug={**timing_debug, "num_selected": 0, "safe": True},
                full_gripper_width_m=full_gripper,
            )

        current_base = self._read_current_pose()
        observed_episode_pose = pose_mat_from_xyz_quat(
            observation_state[POSITION_SLICE], observation_state[QUATERNION_SLICE]
        )
        observed_base = self.base_start_pose @ observed_episode_pose
        drift_m = float(np.linalg.norm(current_base[:3, 3] - observed_base[:3, 3]))
        drift_rot = rotation_distance_rad(observed_base, current_base)
        if drift_m > self.cfg.max_observation_drift_m:
            raise SafetyViolation(
                f"Robot moved {drift_m:.4f}m after observation, exceeding "
                f"{self.cfg.max_observation_drift_m:.4f}m"
            )
        if drift_rot > self.cfg.max_observation_drift_rotation_rad:
            raise SafetyViolation(
                f"Robot rotated {drift_rot:.4f}rad after observation, exceeding "
                f"{self.cfg.max_observation_drift_rotation_rad:.4f}rad"
            )

        target_base = np.stack(
            [base_pose_from_episode_action(self.base_start_pose, action) for action in selected],
            axis=0,
        )
        gripper = selected[:, GRIPPER_INDEX].astype(np.float64)
        gripper_policy_debug: dict[str, Any] = {}
        gripper_preview = self._select_gripper_target(
            gripper,
            update_hold=False,
            lookahead_widths=full_gripper,
            decision_debug=gripper_policy_debug,
        )
        safety_gripper = np.full(gripper.shape, gripper_preview, dtype=np.float64)
        current_controller = self._umi_to_controller_pose(current_base)
        controller_targets, table_z_lift_m = self._preview_adapter_targets(target_base, safety_gripper)
        self._validate_workspace(controller_targets)
        target_delta = np.linalg.norm(
            controller_targets[:, :3, 3] - current_controller[:3, 3][None], axis=1
        )
        target_rotation = np.asarray(
            [rotation_distance_rad(current_controller, target) for target in controller_targets], dtype=np.float64
        )
        first_waypoint_delta = float(
            np.linalg.norm(controller_targets[0, :3, 3] - current_controller[:3, 3])
        )
        first_waypoint_rotation = float(rotation_distance_rad(current_controller, controller_targets[0]))
        consecutive_waypoint_delta = np.linalg.norm(
            np.diff(controller_targets[:, :3, 3], axis=0), axis=1
        )
        consecutive_waypoint_rotation = np.asarray(
            [
                rotation_distance_rad(start, target)
                for start, target in zip(controller_targets[:-1], controller_targets[1:])
            ],
            dtype=np.float64,
        )
        step_delta = np.concatenate(
            [np.asarray([first_waypoint_delta], dtype=np.float64), consecutive_waypoint_delta]
        )
        step_rotation = np.concatenate(
            [np.asarray([first_waypoint_rotation], dtype=np.float64), consecutive_waypoint_rotation]
        )
        _reject_above(target_delta, self.cfg.max_target_delta_m, "target translation from current")
        _reject_above(target_rotation, self.cfg.max_target_rotation_rad, "target rotation from current")
        _reject_above(
            np.asarray([first_waypoint_delta]),
            self.cfg.max_step_delta_m,
            "first waypoint translation from current",
        )
        _reject_above(
            np.asarray([first_waypoint_rotation]),
            self.cfg.max_step_rotation_rad,
            "first waypoint rotation from current",
        )
        if len(consecutive_waypoint_delta):
            _reject_above(
                consecutive_waypoint_delta,
                self.cfg.max_step_delta_m,
                "consecutive waypoint translation",
            )
        if len(consecutive_waypoint_rotation):
            _reject_above(
                consecutive_waypoint_rotation,
                self.cfg.max_step_rotation_rad,
                "consecutive waypoint rotation",
            )

        timestamps = self._build_timestamps(
            target_base=controller_targets,
            selected_indices=indices,
            observation_timestamp=float(observation_timestamp),
            control_frequency_hz=float(control_frequency_hz),
            now=now,
            current_base=current_controller,
        )
        scheduled_duration_s = float(timestamps[-1] - now)
        if scheduled_duration_s > self.cfg.max_scheduled_duration_s:
            raise SafetyViolation(
                f"Speed-limited plan duration {scheduled_duration_s:.3f}s exceeds "
                f"{self.cfg.max_scheduled_duration_s:.3f}s"
            )
        debug = {
            **timing_debug,
            "safe": True,
            "pose_frame": "episode_start_new_tcp",
            "action_semantics": "absolute_episode_pose_xyz_quaternion_xyzw_gripper_width_m",
            "num_selected": int(len(selected)),
            "selected_indices": indices.astype(int).tolist(),
            "selected_action_8d": selected.astype(float).tolist(),
            "base_start_pos_m": self.base_start_pose[:3, 3].astype(float).tolist(),
            "base_current_pos_m": current_base[:3, 3].astype(float).tolist(),
            "base_target_pos_m": target_base[:, :3, 3].astype(float).tolist(),
            "controller_target_pos_after_adapter_safety_m": controller_targets[:, :3, 3].astype(float).tolist(),
            "adapter_table_safety_z_lift_m": float(table_z_lift_m),
            "observation_drift_m": drift_m,
            "observation_drift_rotation_rad": drift_rot,
            "max_target_delta_m": float(target_delta.max()),
            "max_target_rotation_rad": float(target_rotation.max()),
            "first_waypoint_delta_m": first_waypoint_delta,
            "first_waypoint_rotation_rad": first_waypoint_rotation,
            "max_consecutive_waypoint_delta_m": float(consecutive_waypoint_delta.max())
            if len(consecutive_waypoint_delta)
            else 0.0,
            "max_consecutive_waypoint_rotation_rad": float(consecutive_waypoint_rotation.max())
            if len(consecutive_waypoint_rotation)
            else 0.0,
            "max_step_delta_m": float(step_delta.max()),
            "max_step_rotation_rad": float(step_rotation.max()),
            "gripper_width_m": gripper.astype(float).tolist(),
            "full_action_gripper_width_m": full_gripper.astype(float).tolist(),
            "gripper_policy": gripper_policy_debug,
            "gripper_command_preview_m": float(gripper_preview),
            "adapter_safety_gripper_width_m": safety_gripper.astype(float).tolist(),
            "timestamps": timestamps.astype(float).tolist(),
            "scheduled_duration_s": scheduled_duration_s,
            "limits": {
                "max_pos_speed_m_s": self.cfg.max_pos_speed_m_s,
                "max_rot_speed_rad_s": self.cfg.max_rot_speed_rad_s,
                "max_target_delta_m": self.cfg.max_target_delta_m,
                "max_target_rotation_rad": self.cfg.max_target_rotation_rad,
                "max_step_delta_m": self.cfg.max_step_delta_m,
                "max_step_rotation_rad": self.cfg.max_step_rotation_rad,
                "workspace_min_xyz_m": self.cfg.workspace_min_xyz_m,
                "workspace_max_xyz_m": self.cfg.workspace_max_xyz_m,
                "workspace_check_enabled": self.cfg.workspace_min_xyz_m is not None
                and self.cfg.workspace_max_xyz_m is not None,
                "max_scheduled_duration_s": self.cfg.max_scheduled_duration_s,
            },
        }
        return ActionPlan(
            selected_indices=indices,
            selected_actions=selected,
            target_base_poses=target_base,
            validated_controller_poses=controller_targets,
            gripper_width_m=gripper,
            timestamps=timestamps,
            debug=debug,
            full_gripper_width_m=full_gripper,
        )

    def execute_plan(self, plan: ActionPlan) -> dict[str, Any]:
        if not self._actuation_enabled:
            raise RuntimeError("Actuation is disabled; rerun with explicit --execute confirmation")
        if plan.is_empty:
            return {**plan.debug, "dispatched": False, "reason": "no future action remains in the window"}
        dispatch_time = time.time()
        if dispatch_time >= float(plan.timestamps[0]):
            raise SafetyViolation("First action waypoint expired before dispatch")
        if dispatch_time > float(plan.timestamps[-1]):
            raise SafetyViolation("Action plan expired before dispatch")

        gripper_preview = self._select_gripper_target(
            plan.gripper_width_m,
            update_hold=False,
            lookahead_widths=plan.full_gripper_width_m,
        )
        safety_gripper = np.full(plan.gripper_width_m.shape, gripper_preview, dtype=np.float64)
        controller_targets, _ = self._preview_adapter_targets(plan.target_base_poses, safety_gripper)
        if not np.allclose(controller_targets, plan.validated_controller_poses, atol=1e-8):
            raise SafetyViolation("Adapter safety transform changed after planning; refusing stale plan")

        self.robot.execute_waypoints(
            umi_pose=plan.target_base_poses.astype(np.float64),
            gripper_width=safety_gripper,
            timestamps=plan.timestamps.astype(np.float64),
        )
        gripper_command = self._select_gripper_target(
            plan.gripper_width_m,
            update_hold=True,
            lookahead_widths=plan.full_gripper_width_m,
        )
        if self.gripper is not None:
            self.gripper.goto_pos(float(gripper_command))
        adapter_debug = getattr(self.robot, "last_waypoint_debug", None)
        return {
            **plan.debug,
            "dispatched": True,
            "gripper_command_width_m": float(gripper_command),
            "gripper_held_width_m": self._held_gripper_width_m,
            "adapter_safety": adapter_debug,
        }

    def verify_plan_completion(
        self,
        plan: ActionPlan,
        *,
        gripper_command_width_m: float | None,
        position_tolerance_m: float = 0.03,
        rotation_tolerance_rad: float = 0.35,
        gripper_tolerance_m: float = 0.01,
        timeout_s: float = 1.0,
    ) -> dict[str, Any]:
        if plan.is_empty:
            return {"verified": True, "reason": "empty plan"}
        for name, value in (
            ("position_tolerance_m", position_tolerance_m),
            ("rotation_tolerance_rad", rotation_tolerance_rad),
            ("gripper_tolerance_m", gripper_tolerance_m),
            ("timeout_s", timeout_s),
        ):
            _require_positive(value, name)
        target = plan.validated_controller_poses[-1]
        deadline = time.monotonic() + timeout_s
        last_errors = (float("inf"), float("inf"), None)
        while True:
            current_umi = self._read_current_pose()
            current_controller = self._umi_to_controller_pose(current_umi)
            position_error = float(np.linalg.norm(current_controller[:3, 3] - target[:3, 3]))
            rotation_error = rotation_distance_rad(target, current_controller)
            gripper_error = None
            if self.gripper is not None and gripper_command_width_m is not None:
                actual_width = self._read_gripper_width()
                gripper_error = abs(actual_width - float(gripper_command_width_m))
            last_errors = (position_error, rotation_error, gripper_error)
            gripper_ok = gripper_error is None or gripper_error <= gripper_tolerance_m
            if (
                position_error <= position_tolerance_m
                and rotation_error <= rotation_tolerance_rad
                and gripper_ok
            ):
                return {
                    "verified": True,
                    "position_error_m": position_error,
                    "rotation_error_rad": rotation_error,
                    "gripper_error_m": gripper_error,
                }
            if time.monotonic() >= deadline:
                raise SafetyViolation(
                    "Post-dispatch verification timed out: "
                    f"position_error={last_errors[0]:.4f}m, "
                    f"rotation_error={last_errors[1]:.4f}rad, "
                    f"gripper_error={last_errors[2]!r}"
                )
            time.sleep(0.02)

    def abort_motion(self) -> None:
        """Request a software trajectory stop; this is not a physical emergency stop."""

        if self.robot is not None and self._actuation_enabled:
            self.robot.end_episode()

    def _select_action_indices(
        self,
        *,
        action_horizon: int,
        observation_timestamp: float,
        control_frequency_hz: float,
        now: float,
        compensate_inference_latency: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        dt = 1.0 / control_frequency_hz
        online_delay_s = max(0.0, now - observation_timestamp + self.cfg.robot_action_latency_s)
        compensated_delay_s = online_delay_s if compensate_inference_latency else 0.0
        delay_steps = max(0, math.ceil(compensated_delay_s / dt - 1e-9))
        configured_start = max(0, int(self.cfg.exec_start_step))
        configured_end = min(action_horizon, int(self.cfg.exec_end_step))
        if configured_end < configured_start:
            raise ValueError("exec_end_step must be >= exec_start_step")
        configured_count = configured_end - configured_start
        effective_start = min(action_horizon, max(configured_start, delay_steps))
        if self.cfg.preserve_exec_window_length:
            effective_end = min(action_horizon, effective_start + configured_count)
        else:
            effective_end = min(action_horizon, configured_end)
        effective_end = max(effective_start, effective_end)
        indices = np.arange(effective_start, effective_end, dtype=np.int64)
        return indices, {
            "control_frequency_hz": float(control_frequency_hz),
            "observation_timestamp": float(observation_timestamp),
            "plan_timestamp": float(now),
            "online_delay_s": float(online_delay_s),
            "latency_compensation_enabled": bool(compensate_inference_latency),
            "compensated_delay_s": float(compensated_delay_s),
            "delay_steps": int(delay_steps),
            "configured_exec_window": [configured_start, configured_end],
            "effective_exec_window": [effective_start, effective_end],
            "preserve_exec_window_length": bool(self.cfg.preserve_exec_window_length),
        }

    def _build_timestamps(
        self,
        *,
        target_base: np.ndarray,
        selected_indices: np.ndarray,
        observation_timestamp: float,
        control_frequency_hz: float,
        now: float,
        current_base: np.ndarray,
    ) -> np.ndarray:
        nominal = observation_timestamp + selected_indices.astype(np.float64) / control_frequency_hz
        timestamps = np.zeros_like(nominal)
        previous_pose = current_base
        previous_time = now
        earliest_dispatch = now + self.cfg.robot_action_latency_s
        for index, target in enumerate(target_base):
            position_duration = float(np.linalg.norm(target[:3, 3] - previous_pose[:3, 3])) / self.cfg.max_pos_speed_m_s
            rotation_duration = rotation_distance_rad(previous_pose, target) / self.cfg.max_rot_speed_rad_s
            earliest_for_speed = previous_time + max(
                position_duration,
                rotation_duration,
                1.0 / control_frequency_hz,
            )
            timestamps[index] = max(float(nominal[index]), earliest_dispatch, earliest_for_speed)
            previous_pose = target
            previous_time = float(timestamps[index])
        return timestamps

    def _validate_workspace(self, target_base: np.ndarray) -> None:
        positions = target_base[:, :3, 3]
        if self.cfg.workspace_min_xyz_m is not None and self.cfg.workspace_max_xyz_m is not None:
            lower = np.asarray(self.cfg.workspace_min_xyz_m, dtype=np.float64).reshape(3)
            upper = np.asarray(self.cfg.workspace_max_xyz_m, dtype=np.float64).reshape(3)
            if np.any(lower > upper):
                raise ValueError(f"workspace minimum exceeds maximum: min={lower.tolist()}, max={upper.tolist()}")
        if self.cfg.workspace_min_xyz_m is not None:
            lower = np.asarray(self.cfg.workspace_min_xyz_m, dtype=np.float64).reshape(3)
            if np.any(positions < lower[None]):
                raise SafetyViolation(
                    f"Target leaves workspace lower bound {lower.tolist()}: min={positions.min(axis=0).tolist()}"
                )
        if self.cfg.workspace_max_xyz_m is not None:
            upper = np.asarray(self.cfg.workspace_max_xyz_m, dtype=np.float64).reshape(3)
            if np.any(positions > upper[None]):
                raise SafetyViolation(
                    f"Target leaves workspace upper bound {upper.tolist()}: max={positions.max(axis=0).tolist()}"
                )

    def _read_current_pose(self) -> np.ndarray:
        pose, _ = self._read_current_pose_with_timestamp()
        return pose

    def _read_current_pose_with_timestamp(self) -> tuple[np.ndarray, float]:
        state = self._read_robot_state()
        pose = pose_mat_from_rotvec(state["eef_pos"], state["eef_rot_axis_angle"])
        timestamp = float(np.asarray(state["timestamp"]).reshape(-1)[-1])
        return pose, timestamp

    def _read_robot_state(self, *, timeout_s: float = 2.0) -> dict[str, Any]:
        controller = getattr(self.robot, "controller", None)
        if controller is not None and hasattr(controller, "is_alive") and not controller.is_alive():
            raise RuntimeError("Realman controller process is not alive")
        if hasattr(self.robot, "is_ready") and not self.robot.is_ready:
            raise RuntimeError("Realman adapter is not ready")
        deadline = time.monotonic() + timeout_s
        last_state: dict[str, Any] | None = None
        while True:
            last_state = self.robot.get()
            timestamp = float(np.asarray(last_state.get("timestamp", 0.0)).reshape(-1)[-1])
            position = np.asarray(last_state.get("eef_pos", np.zeros(3)), dtype=np.float64).reshape(-1)
            rotvec = np.asarray(last_state.get("eef_rot_axis_angle", np.zeros(3)), dtype=np.float64).reshape(-1)
            state_age_s = time.time() - timestamp
            if (
                timestamp > 0.0
                and position.shape == (3,)
                and rotvec.shape == (3,)
                and np.isfinite(position).all()
                and np.isfinite(rotvec).all()
                and -0.05 <= state_age_s <= self.cfg.max_robot_state_age_s
            ):
                return last_state
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for a fresh Realman state (max age "
                    f"{self.cfg.max_robot_state_age_s:.3f}s). Last state={last_state!r}"
                )
            time.sleep(0.02)

    def _read_gripper_width(
        self,
        *,
        require_hardware_feedback: bool = False,
        timeout_s: float = 1.0,
    ) -> float:
        if self.gripper is None:
            if require_hardware_feedback:
                raise RuntimeError("Gloria gripper is not running")
            return float(self._last_gripper_width_m)
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if hasattr(self.gripper, "is_alive") and not self.gripper.is_alive():
                    raise RuntimeError("Gloria gripper process is not alive")
                state = self.gripper.get_all_state() if hasattr(self.gripper, "get_all_state") else self.gripper.get()
                timestamp_key = "gripper_timestamp" if "gripper_timestamp" in state else "gripper_receive_timestamp"
                timestamp = float(np.asarray(state[timestamp_key]).reshape(-1)[-1])
                age_s = time.time() - timestamp
                if not -0.05 <= age_s <= self.cfg.max_gripper_state_age_s:
                    raise RuntimeError(
                        f"Gripper feedback age {age_s:.3f}s exceeds {self.cfg.max_gripper_state_age_s:.3f}s"
                    )
                if require_hardware_feedback:
                    valid = state.get("gripper_feedback_valid")
                    if valid is None or not bool(np.asarray(valid).reshape(-1)[-1]):
                        raise RuntimeError("Gloria state does not contain valid hardware position feedback")
                for key in ("gripper_width", "width", "width_m"):
                    if key in state:
                        width = float(np.asarray(state[key]).reshape(-1)[-1])
                        if np.isfinite(width):
                            self._last_gripper_width_m = width
                            self._gripper_feedback_is_assumed = False
                            return width
                raise RuntimeError(f"Gripper feedback has no width field: {sorted(state)}")
            except Exception as exc:
                last_error = exc
                time.sleep(0.02)
        if require_hardware_feedback or self._actuation_enabled:
            raise RuntimeError(f"No fresh Gloria gripper feedback: {last_error}") from last_error
        if not self._warned_gripper_read_failure:
            print(f"[lingbot-v2-client] warning: failed to read gripper feedback: {last_error}", flush=True)
            self._warned_gripper_read_failure = True
        return float(self._last_gripper_width_m)

    def _wait_for_gripper_target(
        self,
        target_width_m: float,
        *,
        tolerance_m: float,
        timeout_s: float,
    ) -> dict[str, float]:
        deadline = time.monotonic() + timeout_s
        last_width_m: float | None = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            remaining_s = max(0.01, deadline - time.monotonic())
            try:
                last_width_m = self._read_gripper_width(
                    require_hardware_feedback=True,
                    timeout_s=min(0.25, remaining_s),
                )
                error_m = abs(last_width_m - target_width_m)
                if error_m <= tolerance_m:
                    print(
                        "[lingbot-v2-client] Gloria startup verified from hardware feedback: "
                        f"target={target_width_m * 1000.0:.1f}mm, "
                        f"actual={last_width_m * 1000.0:.1f}mm",
                        flush=True,
                    )
                    return {
                        "target_width_m": float(target_width_m),
                        "actual_width_m": float(last_width_m),
                        "error_m": float(error_m),
                    }
            except Exception as exc:
                last_error = exc
            time.sleep(0.02)
        detail = f"last_width={last_width_m!r}m" if last_width_m is not None else f"last_error={last_error!r}"
        raise SafetyViolation(
            "Gloria did not reach the required pre-inference width from valid hardware feedback: "
            f"target={target_width_m:.4f}m, tolerance={tolerance_m:.4f}m, {detail}"
        )

    def _select_gripper_target(
        self,
        widths: np.ndarray,
        *,
        update_hold: bool,
        lookahead_widths: np.ndarray | None = None,
        decision_debug: dict[str, Any] | None = None,
    ) -> float:
        values = np.asarray(widths, dtype=np.float64).reshape(-1)
        mode = self._gripper_action_select
        if len(values) == 0 and mode == "threshold":
            raise ValueError("Threshold gripper policy received an empty prediction window")
        if len(values) == 0:
            return float(self._last_gripper_width_m)
        if mode == "threshold":
            threshold = self._gripper_hold_closed_below_m
            if threshold is None:
                raise ValueError("gripper_action_select=threshold requires hold_closed_below_m")
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                raise ValueError("Threshold gripper policy received no finite predicted widths")
            closed_width = (
                self.cfg.gripper_min_width_m
                if self._gripper_hold_closed_target_m is None
                else float(self._gripper_hold_closed_target_m)
            )
            threshold = float(threshold)
            open_width = float(self._gripper_initial_width_m)
            if not all(np.isfinite(value) for value in (closed_width, threshold, open_width)):
                raise ValueError("Threshold gripper widths must be finite")
            if not (
                self.cfg.gripper_min_width_m
                <= closed_width
                < threshold
                < open_width
                <= self._gripper_max_width_m
            ):
                raise ValueError(
                    "Threshold gripper policy requires "
                    "min_width <= closed_width < threshold < initial_width <= max_width; "
                    f"got min={self.cfg.gripper_min_width_m}, closed={closed_width}, "
                    f"threshold={threshold}, initial={open_width}, max={self._gripper_max_width_m}"
                )
            selected_min = float(np.min(finite_values))
            selected_max = float(np.max(finite_values))
            selected_triggered = selected_min > threshold
            lookahead_start = self.cfg.gripper_open_lookahead_start_step
            lookahead_count = self.cfg.gripper_open_lookahead_consecutive_steps
            lookahead_run_starts: list[int] = []
            lookahead_triggered = False
            if lookahead_start is not None:
                if lookahead_count is None:
                    raise ValueError(
                        "gripper_open_lookahead_consecutive_steps is required when lookahead is enabled"
                    )
                if lookahead_widths is None:
                    raise ValueError("Threshold gripper lookahead requires the full action chunk")
                full_values = np.asarray(lookahead_widths, dtype=np.float64).reshape(-1)
                if not np.isfinite(full_values).all():
                    raise ValueError("Threshold gripper lookahead received non-finite predicted widths")
                if lookahead_start + lookahead_count > len(full_values):
                    raise ValueError(
                        "Threshold gripper lookahead window does not fit the action chunk: "
                        f"start={lookahead_start}, consecutive={lookahead_count}, horizon={len(full_values)}"
                    )
                above_threshold = full_values[lookahead_start:] > threshold
                for offset in range(len(above_threshold) - lookahead_count + 1):
                    if bool(np.all(above_threshold[offset : offset + lookahead_count])):
                        lookahead_run_starts.append(int(lookahead_start + offset))
                lookahead_triggered = bool(lookahead_run_starts)
            should_open = bool(selected_triggered or lookahead_triggered)
            target = open_width if should_open else closed_width
            if decision_debug is not None:
                decision_debug.update(
                    {
                        "mode": "threshold",
                        "threshold_m": threshold,
                        "selected_min_m": selected_min,
                        "selected_max_m": selected_max,
                        "selected_all_above_threshold": bool(selected_triggered),
                        "lookahead_enabled": lookahead_start is not None,
                        "lookahead_start_step": lookahead_start,
                        "lookahead_consecutive_steps": lookahead_count,
                        "lookahead_open_run_start_steps": lookahead_run_starts,
                        "lookahead_triggered": bool(lookahead_triggered),
                        "decision": "open" if should_open else "closed",
                        "target_width_m": float(target),
                    }
                )
            return float(np.clip(target, self.cfg.gripper_min_width_m, self._gripper_max_width_m))
        elif mode in ("first", "earliest"):
            target = float(values[0])
        elif mode in ("min", "close", "tightest"):
            target = float(np.min(values))
        elif mode == "median":
            target = float(np.median(values))
        else:
            target = float(values[-1])

        target = float(np.clip(target, self.cfg.gripper_min_width_m, self._gripper_max_width_m))
        if self._gripper_hold_closed_below_m is not None:
            held = self._held_gripper_width_m
            if target <= self._gripper_hold_closed_below_m:
                candidate = target
                if self._gripper_hold_closed_target_m is not None:
                    candidate = min(candidate, self._gripper_hold_closed_target_m)
                held = candidate if held is None else min(held, candidate)
                if update_hold:
                    self._held_gripper_width_m = held
            if held is not None:
                target = min(target, held)
        return float(np.clip(target, self.cfg.gripper_min_width_m, self._gripper_max_width_m))

    def _umi_to_controller_pose(self, pose: np.ndarray) -> np.ndarray:
        if hasattr(self.robot, "umi_to_tcp_pose"):
            return np.asarray(self.robot.umi_to_tcp_pose(np.asarray(pose)[None])[0], dtype=np.float64)
        return np.asarray(pose, dtype=np.float64)

    def _preview_adapter_targets(
        self,
        target_base_poses: np.ndarray,
        gripper_width_m: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        targets = np.asarray(target_base_poses, dtype=np.float64)
        if not hasattr(self.robot, "umi_to_tcp_pose"):
            return targets.copy(), 0.0
        controller_targets = np.asarray(self.robot.umi_to_tcp_pose(targets), dtype=np.float64)
        controller = getattr(self.robot, "controller", None)
        if controller is None or not hasattr(controller, "get_eef_coll_points"):
            raise RuntimeError("Realman adapter cannot preview its table-collision safety transform")
        fingertip_points = controller.get_eef_coll_points(controller_targets, gripper_width_m)
        threshold = float(getattr(self.robot, "table_collision_height_threshold"))
        z_lift = max(threshold - float(np.min(fingertip_points[..., 2])), 0.0)
        controller_targets = controller_targets.copy()
        controller_targets[..., 2, 3] += z_lift
        return controller_targets, float(z_lift)

    def _install_tacthru_path(self) -> None:
        path = str(self.tacthru_repo)
        if path not in sys.path:
            sys.path.insert(0, path)

    def _resolve_tacthru_path(self, path: Path | str) -> Path:
        value = Path(path).expanduser()
        return value.resolve() if value.is_absolute() else (self.tacthru_repo / value).resolve()

    def _load_robot_cfg(self):
        cfg = OmegaConf.load(self._resolve_tacthru_path(self.cfg.robot_cfg))
        if self.cfg.robot_ip is not None:
            cfg.robot_ip = self.cfg.robot_ip
        if self.cfg.robot_port is not None:
            cfg.robot_port = int(self.cfg.robot_port)
        if str(cfg.get("tool_frame_name", "")) != "new_tcp":
            raise ValueError(
                f"TacThru UMI V2 requires robot.tool_frame_name='new_tcp', got {cfg.get('tool_frame_name')!r}"
            )
        expected_tool_pose = np.asarray([0.0, 0.0, 0.1301, 0.0, 0.0, 0.0], dtype=np.float64)
        actual_tool_pose = np.asarray(cfg.get("tool_frame_pose", []), dtype=np.float64)
        if actual_tool_pose.shape != (6,) or not np.allclose(actual_tool_pose, expected_tool_pose, atol=1e-6):
            raise ValueError(
                f"TacThru UMI V2 requires tool_frame_pose={expected_tool_pose.tolist()}, "
                f"got {actual_tool_pose.tolist()}"
            )
        configured_pos_speed = float(cfg.get("max_pos_speed", self.cfg.max_pos_speed_m_s))
        configured_rot_speed = float(cfg.get("max_rot_speed", self.cfg.max_rot_speed_rad_s))
        if self.cfg.max_pos_speed_m_s > configured_pos_speed + 1e-9:
            raise ValueError(
                f"Client max_pos_speed_m_s={self.cfg.max_pos_speed_m_s} exceeds robot config {configured_pos_speed}"
            )
        if self.cfg.max_rot_speed_rad_s > configured_rot_speed + 1e-9:
            raise ValueError(
                f"Client max_rot_speed_rad_s={self.cfg.max_rot_speed_rad_s} exceeds robot config {configured_rot_speed}"
            )
        urdf_path = Path(str(cfg.robot_cfg.kinematics.urdf_path))
        if not urdf_path.is_absolute():
            cfg.robot_cfg.kinematics.urdf_path = str((self.tacthru_repo / urdf_path).resolve())
        return cfg

    def _load_gripper_cfg(self):
        path = self.cfg.gripper_cfg or Path("cfg/gripper/synria_gloria.yaml")
        return OmegaConf.load(self._resolve_tacthru_path(path))

    def _configure_gripper_policy(self, gripper_cfg) -> None:
        self._gripper_initial_width_m = float(
            _cfg_get(gripper_cfg, "initial_width_mm", self.cfg.assumed_gripper_width_m * 1000.0)
        ) / 1000.0
        self._gripper_action_select = str(
            self.cfg.gripper_action_select
            if self.cfg.gripper_action_select is not None
            else _cfg_get(gripper_cfg, "action_select", "last")
        ).lower()
        self._gripper_hold_closed_below_m = _optional_float(
            self.cfg.gripper_hold_closed_below_m
            if self.cfg.gripper_hold_closed_below_m is not None
            else _cfg_get(gripper_cfg, "hold_closed_below_m", None)
        )
        self._gripper_hold_closed_target_m = _optional_float(
            self.cfg.gripper_hold_closed_target_m
            if self.cfg.gripper_hold_closed_target_m is not None
            else _cfg_get(gripper_cfg, "hold_closed_target_m", None)
        )
        if (
            self.cfg.gripper_open_lookahead_start_step is not None
            and self._gripper_action_select != "threshold"
        ):
            raise ValueError("Gripper open lookahead requires gripper_action_select=threshold")

    def _create_gripper(self, gripper_cfg):
        driver = str(gripper_cfg.get("driver", "gloria")).lower()
        if driver != "gloria":
            raise ValueError(f"Only the Gloria gripper is supported, got {driver!r}")
        from real_world.grippers.gloria_controller import GloriaGripperController

        startup_width_mm = (
            None
            if self.cfg.gripper_startup_width_m is None
            else float(self.cfg.gripper_startup_width_m) * 1000.0
        )
        return GloriaGripperController(
            self.shm_manager,
            _resolve_gripper_port(gripper_cfg),
            servo_id=int(gripper_cfg.get("servo_id", gripper_cfg.get("id", 1))),
            gripper_type=str(gripper_cfg.get("gripper_type", "50mm")),
            baudrate=int(gripper_cfg.get("baudrate", 1000000)),
            timeout=float(gripper_cfg.get("timeout", 0.1)),
            frequency=float(gripper_cfg.get("frequency", 20.0)),
            feedback_frequency=float(gripper_cfg.get("feedback_frequency", 15.0)),
            receive_latency=float(gripper_cfg.get("receive_latency", 0.0)),
            min_width_cmd_delta_mm=float(gripper_cfg.get("min_width_cmd_delta", 0.5)),
            command_torque=int(
                self.cfg.gripper_command_torque
                if self.cfg.gripper_command_torque is not None
                else gripper_cfg.get("command_torque", 0)
            ),
            command_speed=int(
                self.cfg.gripper_command_speed
                if self.cfg.gripper_command_speed is not None
                else gripper_cfg.get("command_speed", 2000)
            ),
            initial_width_mm=(
                startup_width_mm
                if startup_width_mm is not None
                else float(gripper_cfg.get("initial_width_mm", 45.0))
            ),
            episode_start_width_mm=(
                startup_width_mm
                if startup_width_mm is not None
                else gripper_cfg.get("episode_start_width_mm", None)
            ),
            episode_end_width_mm=gripper_cfg.get("episode_end_width_mm", None),
            initial_move_torque=int(gripper_cfg.get("initial_move_torque", 0)),
            initial_move_speed=int(gripper_cfg.get("initial_move_speed", 2000)),
            initial_settle_s=float(gripper_cfg.get("initial_settle_s", 0.5)),
            verbose=bool(gripper_cfg.get("verbose", False)),
            debug=False,
        )

    def _validate_config(self) -> None:
        positive_limits = {
            "max_pos_speed_m_s": self.cfg.max_pos_speed_m_s,
            "max_rot_speed_rad_s": self.cfg.max_rot_speed_rad_s,
            "max_target_delta_m": self.cfg.max_target_delta_m,
            "max_target_rotation_rad": self.cfg.max_target_rotation_rad,
            "max_step_delta_m": self.cfg.max_step_delta_m,
            "max_step_rotation_rad": self.cfg.max_step_rotation_rad,
            "max_observation_drift_m": self.cfg.max_observation_drift_m,
            "max_observation_drift_rotation_rad": self.cfg.max_observation_drift_rotation_rad,
            "max_robot_state_age_s": self.cfg.max_robot_state_age_s,
            "max_gripper_state_age_s": self.cfg.max_gripper_state_age_s,
            "max_scheduled_duration_s": self.cfg.max_scheduled_duration_s,
            "gripper_startup_tolerance_m": self.cfg.gripper_startup_tolerance_m,
            "gripper_startup_timeout_s": self.cfg.gripper_startup_timeout_s,
        }
        for name, value in positive_limits.items():
            _require_positive(value, name)
        if not np.isfinite(self.cfg.robot_action_latency_s) or self.cfg.robot_action_latency_s < 0.0:
            raise ValueError(f"robot_action_latency_s must be finite and non-negative, got {self.cfg.robot_action_latency_s}")
        if self.cfg.exec_start_step < 0 or self.cfg.exec_end_step < self.cfg.exec_start_step:
            raise ValueError("Execution window must satisfy 0 <= exec_start_step <= exec_end_step")
        lookahead_start = self.cfg.gripper_open_lookahead_start_step
        lookahead_count = self.cfg.gripper_open_lookahead_consecutive_steps
        if (lookahead_start is None) != (lookahead_count is None):
            raise ValueError(
                "gripper_open_lookahead_start_step and "
                "gripper_open_lookahead_consecutive_steps must be configured together"
            )
        if lookahead_start is not None:
            if isinstance(lookahead_start, bool) or int(lookahead_start) != lookahead_start or lookahead_start < 0:
                raise ValueError("gripper_open_lookahead_start_step must be a non-negative integer")
            if isinstance(lookahead_count, bool) or int(lookahead_count) != lookahead_count or lookahead_count <= 0:
                raise ValueError("gripper_open_lookahead_consecutive_steps must be a positive integer")
        gripper_max = self.cfg.gripper_max_width_m
        if gripper_max is not None and (not np.isfinite(gripper_max) or gripper_max <= self.cfg.gripper_min_width_m):
            raise ValueError("gripper_max_width_m must be finite and greater than gripper_min_width_m")
        if not np.isfinite(self.cfg.gripper_min_width_m) or self.cfg.gripper_min_width_m < 0.0:
            raise ValueError("gripper_min_width_m must be finite and non-negative")
        if not np.isfinite(self.cfg.assumed_gripper_width_m):
            raise ValueError("assumed_gripper_width_m must be finite")
        startup_width = self.cfg.gripper_startup_width_m
        if not self.cfg.enable_gripper and startup_width is not None:
            raise ValueError("gripper_startup_width_m requires enable_gripper=True")
        if startup_width is not None:
            effective_max = 0.05 if gripper_max is None else float(gripper_max)
            if (
                not np.isfinite(startup_width)
                or not self.cfg.gripper_min_width_m <= startup_width <= effective_max
            ):
                raise ValueError(
                    "gripper_startup_width_m must be finite and within the configured gripper range"
                )
        for name, value in (
            ("workspace_min_xyz_m", self.cfg.workspace_min_xyz_m),
            ("workspace_max_xyz_m", self.cfg.workspace_max_xyz_m),
        ):
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain three finite values")
        if self.cfg.workspace_min_xyz_m is not None and self.cfg.workspace_max_xyz_m is not None:
            if np.any(
                np.asarray(self.cfg.workspace_min_xyz_m, dtype=np.float64)
                > np.asarray(self.cfg.workspace_max_xyz_m, dtype=np.float64)
            ):
                raise ValueError("workspace_min_xyz_m must not exceed workspace_max_xyz_m")


def _reject_above(values: np.ndarray, limit: float, description: str) -> None:
    maximum = float(np.max(values))
    if maximum > float(limit):
        raise SafetyViolation(f"{description} {maximum:.4f} exceeds limit {float(limit):.4f}")


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _infer_gripper_max_width_m(gripper_cfg) -> float:
    gripper_type = str(gripper_cfg.get("gripper_type", "50mm")).lower()
    if "50" in gripper_type:
        return 0.05
    if "30" in gripper_type:
        return 0.03
    return float(gripper_cfg.get("max_width_m", gripper_cfg.get("max_q", 0.05)))


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return default


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _resolve_gripper_port(gripper_cfg) -> str:
    port = str(gripper_cfg.get("port", "auto"))
    if port != "auto":
        return port
    import glob

    for pattern in ("/dev/tacthru_gripper", "/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return port
