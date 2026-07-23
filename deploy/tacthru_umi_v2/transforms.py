from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


STATE_DIM = 8
ACTION_DIM = 8
POSITION_SLICE = slice(0, 3)
QUATERNION_SLICE = slice(3, 7)
GRIPPER_INDEX = 7


def normalize_quaternion_xyzw(
    quaternion: np.ndarray,
    *,
    norm_tolerance: float = 5e-2,
    canonicalize: bool = True,
) -> np.ndarray:
    """Validate and normalize quaternion(s) in SciPy/LingBot ``xyzw`` order."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1:] != (4,):
        raise ValueError(f"Expected quaternion last dimension 4 (xyzw), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Quaternion contains non-finite values")

    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("Quaternion norm is zero")
    if np.any(np.abs(norm - 1.0) > float(norm_tolerance)):
        raise ValueError(
            f"Quaternion norm differs from 1 by more than {norm_tolerance}: "
            f"range=({float(norm.min()):.6f}, {float(norm.max()):.6f})"
        )

    value = value / norm
    if canonicalize:
        value = np.where(value[..., 3:4] < 0.0, -value, value)
    return value.astype(np.float32)


def validate_state8(
    state: np.ndarray,
    *,
    min_gripper_width_m: float = 0.0,
    max_gripper_width_m: float = 0.20,
) -> np.ndarray:
    """Return a canonical finite 8D policy state.

    State is ``xyz + quaternion_xyzw + absolute_gripper_width_m`` in the
    coordinate frame fixed at the start of the current episode.
    """

    value = np.asarray(state, dtype=np.float32)
    if value.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("state contains non-finite values")

    value = value.copy()
    value[QUATERNION_SLICE] = normalize_quaternion_xyzw(value[QUATERNION_SLICE])
    gripper = float(value[GRIPPER_INDEX])
    if not float(min_gripper_width_m) <= gripper <= float(max_gripper_width_m):
        raise ValueError(
            f"state gripper width {gripper:.6f}m is outside "
            f"[{min_gripper_width_m:.6f}, {max_gripper_width_m:.6f}]m"
        )
    return value


def validate_action_chunk(
    action_chunk: np.ndarray,
    *,
    expected_steps: int | None = None,
    min_gripper_width_m: float = 0.0,
    max_gripper_width_m: float = 0.20,
) -> np.ndarray:
    """Validate a full V2 action chunk and canonicalize its quaternions."""

    value = np.asarray(action_chunk, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or value.shape[0] <= 0:
        raise ValueError(f"action_chunk must have shape (T, {ACTION_DIM}) with T>0, got {value.shape}")
    if expected_steps is not None and value.shape[0] != int(expected_steps):
        raise ValueError(f"Expected {expected_steps} action steps, got {value.shape[0]}")
    if not np.isfinite(value).all():
        raise ValueError("action_chunk contains non-finite values")

    value = value.copy()
    value[:, QUATERNION_SLICE] = normalize_quaternion_xyzw(value[:, QUATERNION_SLICE])
    gripper = value[:, GRIPPER_INDEX]
    if np.any(gripper < float(min_gripper_width_m)) or np.any(gripper > float(max_gripper_width_m)):
        raise ValueError(
            "action gripper width is outside "
            f"[{min_gripper_width_m:.6f}, {max_gripper_width_m:.6f}]m: "
            f"range=({float(gripper.min()):.6f}, {float(gripper.max()):.6f})"
        )
    return value


def pose_mat_from_xyz_quat(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = normalize_quaternion_xyzw(quaternion_xyzw).astype(np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    pose[:3, 3] = position
    return pose


def pose_mat_from_rotvec(position: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


def xyz_quat_from_pose_mat(pose: np.ndarray) -> np.ndarray:
    pose = validate_pose_mat(pose)
    quaternion = normalize_quaternion_xyzw(Rotation.from_matrix(pose[:3, :3]).as_quat())
    return np.concatenate([pose[:3, 3].astype(np.float32), quaternion], axis=0)


def state8_from_episode_pose(episode_pose: np.ndarray, gripper_width_m: float) -> np.ndarray:
    state = np.concatenate(
        [xyz_quat_from_pose_mat(episode_pose), np.asarray([gripper_width_m], dtype=np.float32)],
        axis=0,
    )
    return validate_state8(state)


def episode_state_from_base_poses(
    base_start_pose: np.ndarray,
    base_current_pose: np.ndarray,
    gripper_width_m: float,
) -> np.ndarray:
    """Map the current Realman pose into the frame used by the UMI dataset."""

    base_start_pose = validate_pose_mat(base_start_pose)
    base_current_pose = validate_pose_mat(base_current_pose)
    episode_pose = np.linalg.inv(base_start_pose) @ base_current_pose
    return state8_from_episode_pose(episode_pose, gripper_width_m)


def base_pose_from_episode_action(base_start_pose: np.ndarray, action8: np.ndarray) -> np.ndarray:
    """Convert one server action into an absolute Realman base-frame waypoint."""

    base_start_pose = validate_pose_mat(base_start_pose)
    action = validate_action_chunk(np.asarray(action8, dtype=np.float32).reshape(1, ACTION_DIM))[0]
    episode_target = pose_mat_from_xyz_quat(action[POSITION_SLICE], action[QUATERNION_SLICE])
    return base_start_pose @ episode_target


def validate_pose_mat(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("pose contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"pose has invalid homogeneous row: {value[3].tolist()}")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("pose rotation determinant is not +1")
    return value


def rotation_distance_rad(start_pose: np.ndarray, target_pose: np.ndarray) -> float:
    start = validate_pose_mat(start_pose)
    target = validate_pose_mat(target_pose)
    relative = Rotation.from_matrix(target[:3, :3]) * Rotation.from_matrix(start[:3, :3]).inv()
    return float(relative.magnitude())
