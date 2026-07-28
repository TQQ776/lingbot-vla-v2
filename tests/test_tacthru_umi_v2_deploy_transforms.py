import numpy as np
import pytest
import cv2
from scipy.spatial.transform import Rotation

from deploy.tacthru_umi_v2.realman_client import build_parser, center_crop_resize_rgb
from deploy.tacthru_umi_v2.transforms import (
    base_pose_from_episode_action,
    episode_state_from_base_poses,
    pose_mat_from_rotvec,
    validate_action_chunk,
)


def test_episode_frame_roundtrip_matches_umi_training_semantics() -> None:
    base_start = pose_mat_from_rotvec([0.4, -0.1, 0.2], [0.0, 0.0, np.pi / 2])
    episode_current = pose_mat_from_rotvec([0.1, 0.0, 0.0], [0.0, 0.0, 0.2])
    base_current = base_start @ episode_current

    state = episode_state_from_base_poses(base_start, base_current, 0.04)

    assert state[:3] == pytest.approx([0.1, 0.0, 0.0], abs=1e-6)
    assert state[7] == pytest.approx(0.04)
    assert Rotation.from_quat(state[3:7]).magnitude() == pytest.approx(0.2, abs=1e-6)


def test_server_absolute_episode_action_is_multiplied_by_episode_start_not_current() -> None:
    base_start = pose_mat_from_rotvec([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    action = np.asarray([0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 1.0, 0.03], dtype=np.float32)

    target = base_pose_from_episode_action(base_start, action)

    assert target[:3, 3] == pytest.approx([0.55, 0.02, 0.0], abs=1e-6)
    assert action[7] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8,), dtype=np.float32),
        np.zeros((0, 8), dtype=np.float32),
        np.zeros((2, 7), dtype=np.float32),
        np.zeros((1, 2, 8), dtype=np.float32),
    ],
)
def test_action_chunk_shape_is_strict(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="action_chunk"):
        validate_action_chunk(bad)


def test_zero_quaternion_and_normalized_gripper_representation_are_rejected() -> None:
    actions = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        validate_action_chunk(actions)

    actions[:, 6] = 1.0
    actions[:, 7] = 0.8
    with pytest.raises(ValueError, match="gripper"):
        validate_action_chunk(actions, max_gripper_width_m=0.05)


def test_wrist_preprocess_matches_tacthru_crop_then_resize() -> None:
    frame_bgr = np.zeros((960, 1280, 3), dtype=np.uint8)
    frame_bgr[..., 0] = np.arange(1280, dtype=np.uint16)[None] % 256
    frame_bgr[..., 1] = np.arange(960, dtype=np.uint16)[:, None] % 256
    expected = cv2.resize(frame_bgr[:, 160:1120, ::-1], (224, 224), interpolation=cv2.INTER_AREA)
    actual = center_crop_resize_rgb(frame_bgr, output_size=224)
    assert np.array_equal(actual, expected)


def test_client_defaults_to_motion_free_mode_and_realman_rotation_limit() -> None:
    args = build_parser().parse_args(["run", "--server-url", "http://127.0.0.1:18081"])
    assert args.execute is False
    assert args.max_rot_speed == pytest.approx(0.08)
