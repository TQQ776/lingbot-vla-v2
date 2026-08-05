import numpy as np
import pytest

from deploy.tacthru_umi_v2.tactile_source import build_tactile_frame


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


def test_build_tactile_frame_matches_training_flow_and_previous_frame() -> None:
    frame = build_tactile_frame(
        sensor_history(),
        use_rgb=True,
        use_markers=True,
        episode_reset=False,
    )

    assert frame.tactile_rgb.shape == (1, 480, 640, 3)
    assert frame.tactile_rgb.mean() == pytest.approx(20.0)
    assert frame.marker_positions.shape == (1, 48, 2)
    assert np.allclose(frame.marker_positions, 0.2)
    assert np.allclose(frame.previous_marker_positions, 0.1)
    assert np.count_nonzero(frame.marker_reference) == 0
    assert frame.marker_valid_mask.all()
    assert frame.debug["marker_representation"] == "normalized_flow_2x_over_frame_size"
    assert frame.debug["detected_keypoint_count"] == 48


def test_episode_reset_sets_zero_velocity_and_invalid_markers_are_zeroed() -> None:
    history = sensor_history()
    history["marker"][-1, 3] = np.nan

    frame = build_tactile_frame(
        history,
        use_rgb=False,
        use_markers=True,
        episode_reset=True,
    )

    assert np.allclose(frame.previous_marker_positions, frame.marker_positions)
    assert frame.marker_valid_mask[0, 3] == np.bool_(False)
    assert np.count_nonzero(frame.marker_positions[0, 3]) == 0
    assert np.count_nonzero(frame.previous_marker_positions[0, 3]) == 0


def test_build_tactile_frame_supports_rgb_only_checkpoint() -> None:
    frame = build_tactile_frame(
        sensor_history(),
        use_rgb=True,
        use_markers=False,
        episode_reset=False,
    )

    assert frame.tactile_rgb is not None
    assert frame.marker_positions is None
    assert frame.marker_valid_mask is None


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
