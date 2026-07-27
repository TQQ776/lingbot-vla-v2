import threading
import time

import numpy as np
import pytest
import cv2
from scipy.spatial.transform import Rotation

from deploy.tacthru_umi_v2 import realman_client
from deploy.tacthru_umi_v2.realman_client import (
    WristCamera,
    _estimate_camera_capture_timestamp,
    build_parser,
    center_crop_resize_rgb,
)
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


class FakeTimestampCapture:
    def __init__(self, position_msec: float) -> None:
        self.position_msec = position_msec

    def get(self, prop: int) -> float:
        assert prop == cv2.CAP_PROP_POS_MSEC
        return self.position_msec


def test_uvc_timestamp_is_mapped_from_driver_monotonic_clock(monkeypatch) -> None:
    monkeypatch.setattr(realman_client.time, "monotonic", lambda: 100.04)
    monkeypatch.setattr(realman_client.time, "time", lambda: 1000.04)

    timestamp = _estimate_camera_capture_timestamp(
        FakeTimestampCapture(100_000.0),
        receive_timestamp=1000.04,
        read_started_timestamp=999.70,
        frame_period_s=1.0 / 30.0,
    )

    assert timestamp == pytest.approx(1000.0)


def test_invalid_driver_timestamp_falls_back_to_one_frame_period(monkeypatch) -> None:
    monkeypatch.setattr(realman_client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(realman_client.time, "time", lambda: 1000.0)

    timestamp = _estimate_camera_capture_timestamp(
        FakeTimestampCapture(0.0),
        receive_timestamp=1000.0,
        read_started_timestamp=999.6,
        frame_period_s=1.0 / 30.0,
    )

    assert timestamp == pytest.approx(1000.0 - 1.0 / 30.0)


def test_old_valid_driver_timestamp_is_not_disguised_as_fresh(monkeypatch) -> None:
    monkeypatch.setattr(realman_client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(realman_client.time, "time", lambda: 1000.0)

    timestamp = _estimate_camera_capture_timestamp(
        FakeTimestampCapture(98_500.0),
        receive_timestamp=1000.0,
        read_started_timestamp=999.99,
        frame_period_s=1.0 / 30.0,
    )

    assert timestamp == pytest.approx(998.5)


class FakeContinuousCapture:
    def __init__(self, *, frame_interval_s: float = 0.005, timestamp_offset_s: float = 0.0) -> None:
        self.frame_interval_s = frame_interval_s
        self.timestamp_offset_s = timestamp_offset_s
        self.sequence = 0
        self.position_msec = 0.0
        self.released = False
        self.release_calls = 0
        self._lock = threading.Lock()

    def isOpened(self) -> bool:
        return True

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def grab(self) -> bool:
        time.sleep(self.frame_interval_s)
        with self._lock:
            if self.released:
                return False
            self.sequence += 1
            self.position_msec = (time.monotonic() + self.timestamp_offset_s) * 1000.0
        return True

    def retrieve(self):
        with self._lock:
            if self.released:
                return False, None
            value = self.sequence % 255
        return True, np.full((4, 6, 3), value, dtype=np.uint8)

    def get(self, prop: int) -> float:
        assert prop == cv2.CAP_PROP_POS_MSEC
        with self._lock:
            return self.position_msec

    def release(self) -> None:
        with self._lock:
            self.release_calls += 1
            self.released = True


class FakeFailingCapture(FakeContinuousCapture):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_grab = threading.Event()

    def grab(self) -> bool:
        if self.sequence > 0:
            self.fail_next_grab.wait(timeout=1.0)
            return False
        return super().grab()


def _make_wrist_camera(monkeypatch, capture, **kwargs) -> WristCamera:
    monkeypatch.setattr(realman_client.cv2, "VideoCapture", lambda _device: capture)
    return WristCamera(
        "fake-camera",
        width=6,
        height=4,
        fps=200.0,
        output_size=2,
        max_frame_age_s=kwargs.pop("max_frame_age_s", 0.10),
        initial_frame_timeout_s=kwargs.pop("initial_frame_timeout_s", 0.50),
        frame_timeout_s=kwargs.pop("frame_timeout_s", 0.10),
        **kwargs,
    )


def test_wrist_camera_background_worker_returns_latest_frame(monkeypatch) -> None:
    capture = FakeContinuousCapture()
    camera = _make_wrist_camera(monkeypatch, capture)
    try:
        first = camera.capture()
        first_value = int(first.rgb[0, 0, 0])
        time.sleep(0.05)
        second = camera.capture()
        second_value = int(second.rgb[0, 0, 0])

        assert second_value >= first_value + 3
        assert time.time() - second.capture_timestamp < 0.10
    finally:
        camera.close()

    assert capture.released is True
    assert capture.release_calls == 1
    assert camera._thread is not None and not camera._thread.is_alive()


def test_wrist_camera_returns_causal_frame_aligned_to_tactile_timestamp(monkeypatch) -> None:
    capture = FakeContinuousCapture()
    camera = _make_wrist_camera(monkeypatch, capture, history_seconds=0.5)
    try:
        camera.start()
        time.sleep(0.05)
        with camera._condition:
            history = tuple(camera._frame_history)
        assert len(history) >= 4
        reference_index = len(history) - 3
        target_timestamp = history[reference_index].capture_timestamp + 0.001

        aligned = camera.capture_at_or_before(target_timestamp, max_skew_s=0.02)

        assert aligned.capture_timestamp == pytest.approx(
            history[reference_index].capture_timestamp
        )
        assert 0.0 <= target_timestamp - aligned.capture_timestamp <= 0.02
    finally:
        camera.close()


def test_wrist_camera_rejects_continuously_produced_old_frames(monkeypatch) -> None:
    capture = FakeContinuousCapture(timestamp_offset_s=-1.5)
    camera = _make_wrist_camera(
        monkeypatch,
        capture,
        initial_frame_timeout_s=0.05,
        frame_timeout_s=0.05,
    )
    try:
        with pytest.raises(RuntimeError, match="No fresh wrist camera frame"):
            camera.capture()
    finally:
        camera.close()


def test_wrist_camera_worker_error_is_propagated_and_device_is_released(monkeypatch) -> None:
    capture = FakeFailingCapture()
    camera = _make_wrist_camera(monkeypatch, capture)
    try:
        camera.capture()
        capture.fail_next_grab.set()
        with pytest.raises(RuntimeError, match="capture worker failed"):
            camera.capture()
    finally:
        camera.close()

    assert capture.released is True
    assert capture.release_calls == 1


def test_future_driver_timestamp_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(realman_client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(realman_client.time, "time", lambda: 1000.0)

    with pytest.raises(RuntimeError, match="unexpectedly in the future"):
        _estimate_camera_capture_timestamp(
            FakeTimestampCapture(101_000.0),
            receive_timestamp=1000.0,
            read_started_timestamp=999.99,
            frame_period_s=1.0 / 30.0,
        )


def test_client_defaults_to_motion_free_mode_and_realman_rotation_limit() -> None:
    args = build_parser().parse_args(["run", "--server-url", "http://127.0.0.1:18081"])
    assert args.execute is False
    assert args.max_rot_speed == pytest.approx(0.08)
