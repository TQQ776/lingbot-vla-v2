from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIVE_CALIBRATION_PATH = REPO_ROOT / "cfg" / "calibration" / "vive_new_tcp.json"
NEW_TCP_RELATIVE_FRAME = "new_tcp_relative"


@dataclass(frozen=True)
class ViveTcpCalibration:
    """保存 Vive tracker 到 new_tcp 的统一标定结果。"""

    tool_frame: str
    pose_frame: str
    tracker_to_newtcp: np.ndarray
    base_to_vive: np.ndarray
    source_path: Path


def validate_rigid_transform(matrix: np.ndarray, name: str, atol: float = 1e-6) -> np.ndarray:
    """校验矩阵是否为有限、正交且右手的 4x4 刚体变换。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(f"{name} rotation determinant must be 1, got {determinant}")
    return matrix


def invert_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    """计算单个 4x4 刚体变换的逆。"""
    matrix = validate_rigid_transform(matrix, "rigid_transform")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = matrix[:3, :3].T
    output[:3, 3] = -(output[:3, :3] @ matrix[:3, 3])
    return output


def load_vive_tcp_calibration(path: str | Path = DEFAULT_VIVE_CALIBRATION_PATH) -> ViveTcpCalibration:
    """读取并校验 Vive 到 new_tcp 的稳定标定配置。"""
    path = Path(path).expanduser().resolve()
    with path.open("r") as file:
        data = json.load(file)

    tool_frame = str(data["tool_frame"])
    pose_frame = str(data.get("pose_frame", NEW_TCP_RELATIVE_FRAME))
    if tool_frame != "new_tcp":
        raise ValueError(f"Expected tool_frame='new_tcp', got {tool_frame!r}")
    if pose_frame != NEW_TCP_RELATIVE_FRAME:
        raise ValueError(f"Expected pose_frame={NEW_TCP_RELATIVE_FRAME!r}, got {pose_frame!r}")

    tracker_to_newtcp = validate_rigid_transform(
        np.asarray(data["tracker_to_tcp_matrix"], dtype=np.float64),
        "tracker_to_tcp_matrix",
    )
    if "tcp_to_tracker_matrix" in data:
        newtcp_to_tracker = validate_rigid_transform(
            np.asarray(data["tcp_to_tracker_matrix"], dtype=np.float64),
            "tcp_to_tracker_matrix",
        )
        expected_inverse = invert_rigid_transform(tracker_to_newtcp)
        if not np.allclose(newtcp_to_tracker, expected_inverse, atol=1e-6):
            raise ValueError("tcp_to_tracker_matrix is not the inverse of tracker_to_tcp_matrix")
    base_to_vive = validate_rigid_transform(
        np.asarray(data["base_to_vive_matrix"], dtype=np.float64),
        "base_to_vive_matrix",
    )
    return ViveTcpCalibration(
        tool_frame=tool_frame,
        pose_frame=pose_frame,
        tracker_to_newtcp=tracker_to_newtcp,
        base_to_vive=base_to_vive,
        source_path=path,
    )


def tracker_relative_to_newtcp_relative(
    tracker_relative_poses: np.ndarray,
    tracker_to_newtcp: np.ndarray,
) -> np.ndarray:
    """把 T_tracker0_tracker_i 共轭变换为 T_newtcp0_newtcp_i。"""
    poses = np.asarray(tracker_relative_poses, dtype=np.float64)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"tracker_relative_poses must end with shape (4, 4), got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("tracker_relative_poses contains non-finite values")

    tracker_to_newtcp = validate_rigid_transform(tracker_to_newtcp, "tracker_to_newtcp")
    newtcp_to_tracker = invert_rigid_transform(tracker_to_newtcp)
    return newtcp_to_tracker @ poses @ tracker_to_newtcp


def vive_absolute_to_newtcp_absolute(
    vive_to_tracker_poses: np.ndarray,
    base_to_vive: np.ndarray,
    tracker_to_newtcp: np.ndarray,
) -> np.ndarray:
    """将 Vive 世界系中的 tracker 绝对位姿转换为机器人 Base 下的 new_tcp 位姿。"""
    poses = np.asarray(vive_to_tracker_poses, dtype=np.float64)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"vive_to_tracker_poses must end with shape (4, 4), got {poses.shape}")
    base_to_vive = validate_rigid_transform(base_to_vive, "base_to_vive")
    tracker_to_newtcp = validate_rigid_transform(tracker_to_newtcp, "tracker_to_newtcp")
    return base_to_vive @ poses @ tracker_to_newtcp
