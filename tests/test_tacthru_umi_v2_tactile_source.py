from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.tacthru_umi_v2.tactile_source import TacThruSource, build_tactile_frame
from lingbotvla.tactile_contact import (
    CONTACT_OFF,
    CONTACT_ON,
    MarkerContactGate,
    runtime_gate_config_from_mapping,
)


def sensor_history() -> dict[str, np.ndarray]:
    reference = np.zeros((2, 48, 2), dtype=np.float32)
    previous = reference[0].copy()
    previous[:, 0] = 32.0
    previous[:, 1] = 24.0
    current = reference[1].copy()
    current[:, 0] = 64.0
    current[:, 1] = 48.0
    return {
        "timestamp": np.asarray([100.0, 100.0 + 1.0 / 30.0]),
        "rgb": np.stack(
            [
                np.full((480, 640, 3), 10, dtype=np.uint8),
                np.full((480, 640, 3), 20, dtype=np.uint8),
            ]
        ),
        "marker": np.stack([previous, current]),
        "marker_ref": reference,
        "n_all_kpts": np.asarray([47, 48]),
    }


def test_build_tactile_frame_matches_training_displacement_history() -> None:
    frame = build_tactile_frame(
        sensor_history(),
        use_rgb=True,
        use_markers=True,
        episode_reset=False,
    )

    assert frame.tactile_rgb.shape == (1, 480, 640, 3)
    assert frame.tactile_rgb.mean() == pytest.approx(20.0)
    assert frame.marker_displacement_history.shape == (1, 8, 48, 2)
    assert np.allclose(frame.marker_displacement_history[:, :7], 0.1)
    assert np.allclose(frame.marker_displacement_history[:, -1], 0.2)
    assert frame.marker_valid_mask.shape == (1, 8, 48)
    assert frame.marker_history_valid_mask.shape == (1, 8)
    assert frame.marker_valid_mask.all()
    assert frame.marker_history_valid_mask.all()
    assert frame.debug["marker_representation"] == "normalized_displacement"
    assert frame.debug["marker_history_length"] == 8
    assert frame.debug["marker_sample_hz"] == pytest.approx(30.0)
    assert frame.debug["detected_keypoint_count"] == 48


def test_episode_reset_replicates_current_frame_and_zeros_invalid_markers() -> None:
    history = sensor_history()
    history["marker"][-1, 3] = np.nan

    frame = build_tactile_frame(
        history,
        use_rgb=False,
        use_markers=True,
        episode_reset=True,
    )

    assert np.allclose(
        frame.marker_displacement_history,
        frame.marker_displacement_history[:, -1:],
    )
    assert not frame.marker_valid_mask[0, :, 3].any()
    assert np.count_nonzero(frame.marker_displacement_history[0, :, 3]) == 0


def test_build_tactile_frame_supports_rgb_only_checkpoint() -> None:
    frame = build_tactile_frame(
        sensor_history(),
        use_rgb=True,
        use_markers=False,
        episode_reset=False,
    )

    assert frame.tactile_rgb is not None
    assert frame.marker_displacement_history is None
    assert frame.marker_valid_mask is None
    assert frame.marker_history_valid_mask is None


def test_build_tactile_frame_rejects_reversed_history() -> None:
    history = sensor_history()
    history["timestamp"] = history["timestamp"][::-1]

    with pytest.raises(ValueError, match="strictly increasing"):
        build_tactile_frame(
            history,
            use_rgb=True,
            use_markers=True,
            episode_reset=False,
        )


class _FakeSensor:
    def __init__(self, data):
        self.data = data
        self.ring_buffer = SimpleNamespace(count=len(data["timestamp"]))

    def get(self, k):
        return {name: np.asarray(value)[-k:] for name, value in self.data.items()}


def _source_with_data(data) -> TacThruSource:
    source = object.__new__(TacThruSource)
    source.use_rgb = False
    source.use_markers = True
    source._sensor = _FakeSensor(data)
    source._marker_history = deque(maxlen=8)
    source._marker_valid_history = deque(maxlen=8)
    source._last_marker_timestamp = None
    source._contact_gates = []
    source._contact_state = np.full((1,), CONTACT_OFF, dtype=np.int8)
    return source


def test_live_source_rolls_only_new_timestamped_marker_frames() -> None:
    data = sensor_history()
    source = _source_with_data(data)
    first = source.capture(timeout_s=0.1, episode_reset=True)
    assert np.allclose(first.marker_displacement_history, 0.2)

    # Re-reading the same ring buffer does not append duplicate timestamps.
    duplicate = source.capture(timeout_s=0.1, episode_reset=False)
    assert np.array_equal(
        duplicate.marker_displacement_history,
        first.marker_displacement_history,
    )

    next_data = sensor_history()
    next_data["timestamp"] = np.asarray([data["timestamp"][-1] + 1.0 / 30.0])
    next_data["rgb"] = next_data["rgb"][-1:]
    next_data["marker"] = next_data["marker"][-1:] * 1.5
    next_data["marker_ref"] = next_data["marker_ref"][-1:]
    next_data["n_all_kpts"] = next_data["n_all_kpts"][-1:]
    source._sensor = _FakeSensor(next_data)
    rolled = source.capture(timeout_s=0.1, episode_reset=False)
    assert np.allclose(rolled.marker_displacement_history[0, :-1], 0.2)
    assert np.allclose(rolled.marker_displacement_history[0, -1], 0.3)


def test_live_source_episode_reset_discards_previous_history() -> None:
    source = _source_with_data(sensor_history())
    source.capture(timeout_s=0.1, episode_reset=False)
    reset_data = sensor_history()
    reset_data["timestamp"] += 1.0
    reset_data["marker"] *= 2.0
    source._sensor = _FakeSensor(reset_data)
    reset = source.capture(timeout_s=0.1, episode_reset=True)
    assert np.allclose(reset.marker_displacement_history, 0.4)


def test_live_source_emits_online_contact_state_from_each_new_frame() -> None:
    source = _source_with_data(sensor_history())
    config = runtime_gate_config_from_mapping(
        {
            "mode": "hard",
            "point_threshold": 0.05,
            "on_threshold": 0.08,
            "off_threshold": 0.04,
            "soft_threshold": 0.04,
            "min_active_markers": 2,
            "min_valid_markers_per_region": 2,
            "min_valid_markers_global": 24,
        },
        num_sensors=1,
        num_markers=48,
    )
    source._contact_gates = [
        MarkerContactGate(config, np.zeros(48, dtype=np.int64))
    ]

    frame = source.capture(timeout_s=0.1, episode_reset=True)

    assert frame.marker_contact_state.tolist() == [CONTACT_ON]
