import time

import numpy as np
import pytest

from deploy.tacthru_umi_v2.realman_client import (
    TacThruTactileSource,
    TactileHistory,
    _resolve_tactile_request_args,
    build_parser,
    build_tactile_history,
    validate_local_tactile_guard,
)
from deploy.tacthru_umi_v2.realman_runtime import SafetyViolation


def _health(*, rgb: bool = True, marker: bool = True, ablation: bool = False) -> dict:
    return {
        "protocol_version": 1,
        "protocol_versions_supported": [1, 2],
        "checkpoint_modalities": {
            "wrist_rgb": True,
            "tactile_rgb": rgb,
            "tactile_marker": marker,
        },
        "tactile_enabled": rgb or marker,
        "tactile_temporal_horizon": 4,
        "tactile_history_stride": 1,
        "tactile_contract_sha256": "b" * 64,
        "allow_missing_tactile": ablation,
        "allow_tactile_ablation": ablation,
    }


def test_online_marker_normalization_uses_actual_640x480_tracking_coordinates() -> None:
    now = time.time()
    rgb = np.zeros((2, 480, 640, 3), dtype=np.uint8)
    marker_ref = np.zeros((2, 48, 2), dtype=np.float32)
    marker = marker_ref.copy()
    marker[..., 0] += 40.0
    marker[..., 1] += 20.0
    history = build_tactile_history(
        {
            "timestamp": np.asarray([now - 0.03, now], dtype=np.float64),
            "rgb": rgb,
            "marker": marker,
            "marker_ref": marker_ref,
        },
        history_steps=4,
        history_stride=1,
        include_rgb=True,
        include_marker=True,
    )

    assert history.rgb.shape == (4, 224, 224, 3)
    assert history.rgb_history_mask.tolist() == [False, False, True, True]
    assert history.marker_history_mask.tolist() == [False, False, True, True]
    assert history.marker_valid_mask[:2].sum() == 0
    assert history.marker_flow[-1, :, 0] == pytest.approx(40.0 / 640.0 * 2.0)
    assert history.marker_flow[-1, :, 1] == pytest.approx(20.0 / 480.0 * 2.0)
    assert history.debug["marker_normalization_size_xy"] == [640, 480]


def test_local_tactile_guard_fails_closed_on_stale_or_lost_markers() -> None:
    now = time.time()
    base = TactileHistory(
        rgb=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        rgb_timestamps=np.asarray([now - 0.04, now - 0.02]),
        rgb_history_mask=np.ones(2, dtype=np.bool_),
        marker_flow=np.zeros((2, 48, 2), dtype=np.float32),
        marker_valid_mask=np.ones((2, 48), dtype=np.bool_),
        marker_timestamps=np.asarray([now - 0.04, now - 0.02]),
        marker_history_mask=np.ones(2, dtype=np.bool_),
        debug={},
    )
    debug = validate_local_tactile_guard(
        base,
        wrist_timestamp=now - 0.02,
        robot_timestamp=now - 0.02,
        now=now,
        marker_required=True,
        max_tactile_age_s=0.1,
        max_tactile_skew_s=0.05,
        min_valid_markers=40,
    )
    assert debug["valid_marker_count"] == 48

    stale = TactileHistory(
        **{**base.__dict__, "rgb_timestamps": base.rgb_timestamps - 1.0, "marker_timestamps": base.marker_timestamps - 1.0}
    )
    with pytest.raises(SafetyViolation, match="tactile_stale"):
        validate_local_tactile_guard(
            stale,
            wrist_timestamp=now,
            robot_timestamp=now,
            now=now,
            marker_required=True,
            max_tactile_age_s=0.1,
            max_tactile_skew_s=0.05,
            min_valid_markers=40,
        )

    lost = TactileHistory(
        **{**base.__dict__, "marker_valid_mask": np.zeros((2, 48), dtype=np.bool_)}
    )
    with pytest.raises(SafetyViolation, match="marker_tracking_lost"):
        validate_local_tactile_guard(
            lost,
            wrist_timestamp=now - 0.02,
            robot_timestamp=now - 0.02,
            now=now,
            marker_required=True,
            max_tactile_age_s=0.1,
            max_tactile_skew_s=0.05,
            min_valid_markers=40,
        )


def test_local_tactile_guard_separates_receive_freshness_from_capture_alignment() -> None:
    now = time.time()
    latest_capture = now - 0.135
    rgb = np.zeros((2, 480, 640, 3), dtype=np.uint8)
    marker_ref = np.zeros((2, 48, 2), dtype=np.float32)
    history = build_tactile_history(
        {
            "timestamp": np.asarray([latest_capture - 1.0 / 30.0, latest_capture]),
            "rgb": rgb,
            "marker": marker_ref.copy(),
            "marker_ref": marker_ref,
        },
        history_steps=4,
        history_stride=1,
        include_rgb=False,
        include_marker=True,
        receive_latency_s=0.115,
    )

    debug = validate_local_tactile_guard(
        history,
        wrist_timestamp=latest_capture - 0.020,
        robot_timestamp=latest_capture - 0.030,
        now=now,
        marker_required=True,
        max_tactile_age_s=0.10,
        max_tactile_skew_s=0.05,
        min_valid_markers=40,
    )

    assert debug["capture_age_s"] == pytest.approx(0.135, abs=1e-4)
    assert debug["receive_age_s"] == pytest.approx(0.020, abs=1e-4)
    assert debug["age_s"] == pytest.approx(debug["receive_age_s"])
    assert debug["skew_to_wrist_s"] == pytest.approx(0.020, abs=1e-4)
    assert debug["skew_to_robot_s"] == pytest.approx(0.030, abs=1e-4)

    with pytest.raises(SafetyViolation, match=r"wrist=0\.135\d*s, robot=0\.135\d*s"):
        validate_local_tactile_guard(
            history,
            wrist_timestamp=now,
            robot_timestamp=now,
            now=now,
            marker_required=True,
            max_tactile_age_s=0.10,
            max_tactile_skew_s=0.05,
            min_valid_markers=40,
        )


def test_tactile_source_close_finalizes_sensor_before_manager_shutdown() -> None:
    events = []

    class ClosingSensor:
        def stop(self, wait=False):
            events.append(("stop", bool(wait)))

        def join(self, timeout=None):
            events.append(("join", timeout))

        def is_alive(self):
            return False

        def __del__(self):
            events.append(("del", True))

    class ClosingManager:
        def shutdown(self):
            events.append(("manager", True))

    source = TacThruTactileSource.__new__(TacThruTactileSource)
    source._sensor = ClosingSensor()
    source._manager = ClosingManager()

    source.close()

    event_names = [name for name, _ in events]
    assert event_names.index("stop") < event_names.index("del")
    assert event_names.index("del") < event_names.index("manager")
    assert source._sensor is None
    assert source._manager is None


def test_execute_requires_all_checkpoint_modalities_and_forbids_force_mask() -> None:
    missing = build_parser().parse_args(
        ["run", "--server-url", "http://127.0.0.1:18082", "--execute"]
    )
    with pytest.raises(RuntimeError, match="requires tactile_rgb"):
        _resolve_tactile_request_args(missing, _health())

    masked = build_parser().parse_args(
        [
            "run",
            "--server-url",
            "http://127.0.0.1:18082",
            "--execute",
            "--tactile-mode",
            "rgb-marker",
            "--force-mask-marker",
        ]
    )
    with pytest.raises(RuntimeError, match="force-mask"):
        _resolve_tactile_request_args(masked, _health(ablation=True))


def test_dry_run_ablation_is_explicit_and_contract_bound() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--server-url",
            "http://127.0.0.1:18082",
            "--tactile-mode",
            "rgb-marker",
            "--force-mask-marker",
        ]
    )
    _resolve_tactile_request_args(args, _health(ablation=True))
    assert args._protocol_version == 2
    assert args._tactile_horizon == 4
    assert args._tactile_contract_sha256 == "b" * 64
