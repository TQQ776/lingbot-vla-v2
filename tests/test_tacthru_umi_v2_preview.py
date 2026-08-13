import numpy as np
import pytest

from deploy.tacthru_umi_v2 import realman_client
from deploy.tacthru_umi_v2.tactile_source import TactileFrame


def _preview_frame(*, contact_state: int = 1) -> TactileFrame:
    rows, columns = np.meshgrid(
        np.linspace(55.0, 585.0, 8, dtype=np.float32),
        np.linspace(50.0, 430.0, 6, dtype=np.float32),
    )
    reference = np.stack([rows.reshape(-1), columns.reshape(-1)], axis=-1)
    current = reference + np.asarray([3.0, -2.0], dtype=np.float32)
    displacement = 2.0 * (current - reference) / np.asarray(
        [640.0, 480.0], dtype=np.float32
    )
    history = np.repeat(displacement[None, None], 8, axis=1)
    valid = np.ones((1, 8, 48), dtype=np.bool_)
    valid[:, :, 7] = False
    return TactileFrame(
        tactile_rgb=np.full((1, 480, 640, 3), 90, dtype=np.uint8),
        marker_displacement_history=history,
        marker_valid_mask=valid,
        marker_history_valid_mask=np.ones((1, 8), dtype=np.bool_),
        marker_contact_state=np.asarray([contact_state], dtype=np.int8),
        marker_reference_pixels=reference[None],
        marker_current_pixels=current[None],
        tactile_sensor_mask=np.ones((1,), dtype=np.bool_),
        capture_timestamp=1.0,
        receive_timestamp=1.0,
        debug={"detected_keypoint_count": 48},
    )


def test_combined_preview_draws_tactile_markers_without_mutating_inputs() -> None:
    wrist = np.full((224, 224, 3), 120, dtype=np.uint8)
    frame = _preview_frame()
    tactile_before = frame.tactile_rgb.copy()

    canvas = realman_client._build_preview_canvas(
        wrist,
        "step=2 sending",
        tactile_frame=frame,
    )

    assert canvas.shape == (544, 1122, 3)
    assert canvas.dtype == np.uint8
    assert np.array_equal(frame.tactile_rgb, tactile_before)
    assert np.any(canvas[64:, :640] != 90)
    assert realman_client._gate_preview_status(frame)[0] == "CONTACT"


def test_preview_without_tactile_keeps_stable_combined_layout() -> None:
    canvas = realman_client._build_preview_canvas(
        np.zeros((224, 224, 3), dtype=np.uint8),
        "waiting for operator",
    )

    assert canvas.shape == (544, 1122, 3)
    assert realman_client._gate_preview_status(None)[0] == "GATE N/A"


def test_preview_gui_preflight_rejects_headless_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        realman_client.cv2,
        "getBuildInformation",
        lambda: "OpenCV build information\n  GUI: NONE\n",
    )

    with pytest.raises(RuntimeError, match="GUI support"):
        realman_client._require_preview_gui()


def test_preview_gui_preflight_requires_desktop_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        realman_client.cv2,
        "getBuildInformation",
        lambda: "OpenCV build information\n  GUI: QT5\n",
    )
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    with pytest.raises(RuntimeError, match="graphical desktop session"):
        realman_client._require_preview_gui()
