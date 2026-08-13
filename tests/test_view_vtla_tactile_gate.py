from types import SimpleNamespace

import numpy as np

from lingbotvla.tactile_contact import CONTACT_OFF, CONTACT_ON
from tools.view_vtla_tactile_gate import (
    DEFAULT_TACTHRU_REPO,
    DEFAULT_VTLA_CONFIG,
    gate_label,
    load_gate_context,
    normalized_marker_displacement,
    render_gate_ui,
)


def test_default_viewer_uses_resolved_current_vtla_gate() -> None:
    context = load_gate_context(DEFAULT_VTLA_CONFIG)

    assert context.settings.num_sensors == 1
    assert context.settings.num_markers == 48
    assert context.runtime.mode == "global_active_count_hysteresis_soft_region"
    assert context.runtime.min_active_markers == 2
    assert np.isclose(context.runtime.on_thresholds[0], 0.010032787062227726)
    assert np.isclose(context.runtime.off_thresholds[0], 0.00651359673589468)
    assert context.region_ids.shape == (48,)
    assert set(context.region_ids.tolist()) == {0, 1, 2, 3}


def test_normalized_marker_displacement_matches_deployment_formula() -> None:
    reference = np.asarray([[64.0, 48.0], [128.0, 96.0]], dtype=np.float32)
    marker = reference + np.asarray([[32.0, 24.0], [0.0, -24.0]], dtype=np.float32)

    displacement, valid = normalized_marker_displacement(
        marker,
        reference,
        image_width=640,
        image_height=480,
    )

    np.testing.assert_allclose(displacement, [[0.1, 0.1], [0.0, -0.1]])
    np.testing.assert_array_equal(valid, [True, True])


def test_gate_label_distinguishes_warmup_off_and_visible_states() -> None:
    assert gate_label(CONTACT_OFF, warmup_remaining=12)[0].startswith("模板预热中")
    assert gate_label(CONTACT_OFF, warmup_remaining=0)[0] == "无触觉"
    assert gate_label(CONTACT_ON, warmup_remaining=0)[0] == "检测到触觉"


def test_render_gate_ui_produces_fixed_status_band() -> None:
    context = load_gate_context(DEFAULT_VTLA_CONFIG)
    rgb = np.full((480, 640, 3), 60, dtype=np.uint8)
    reference = np.load(DEFAULT_TACTHRU_REPO / "cfg/sensor/ml-ref_kpts.npy").astype(
        np.float32
    )
    reference *= np.asarray([640.0, 480.0], dtype=np.float32)
    marker = reference.copy()
    gate_frame = SimpleNamespace(
        state=CONTACT_OFF,
        region_scores=np.zeros(4, dtype=np.float32),
        regional_soft_gates=np.zeros(4, dtype=np.float32),
        region_valid_mask=np.ones(4, dtype=bool),
        active_counts=np.zeros(4, dtype=np.int64),
        tracking_unknown=False,
    )

    canvas = render_gate_ui(
        rgb=rgb,
        marker=marker,
        reference=reference,
        tracker_valid=np.ones(48, dtype=bool),
        tracker_fallback=np.zeros(48, dtype=bool),
        vtla_valid=np.ones(48, dtype=bool),
        region_ids=context.region_ids,
        gate_frame=gate_frame,
        runtime=context.runtime,
        warmup_remaining=0,
        candidate_count=48,
        fps=30.0,
        arrow_scale=6.0,
    )

    assert canvas.shape == (652, 640, 3)
    assert canvas.dtype == np.uint8
    assert np.count_nonzero(canvas[:172] != 20) > 0
