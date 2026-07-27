import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.tacthru_umi_v2 import realman_client as client_module
from deploy.tacthru_umi_v2.protocol import ActionResponse
from deploy.tacthru_umi_v2.realman_client import (
    CameraFrame,
    RetryableInferenceError,
    TactileHistory,
    build_parser,
    run_realman,
)
from deploy.tacthru_umi_v2.realman_runtime import (
    HistoricalAlignmentNotReady,
    PolicyStateSnapshot,
    SafetyViolation,
)


def _actions(gripper_width_m: float) -> np.ndarray:
    result = np.zeros((50, 8), dtype=np.float32)
    result[:, 6] = 1.0
    result[:, 7] = float(gripper_width_m)
    return result


class FakeCamera:
    def __init__(self) -> None:
        self.capture_calls = 0
        self.aligned_capture_calls = []
        self.start_calls = 0
        self.closed = False

    def start(self) -> None:
        self.start_calls += 1

    def capture(self) -> CameraFrame:
        self.capture_calls += 1
        now = time.time()
        return CameraFrame(
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            capture_timestamp=now,
            receive_timestamp=now,
        )

    def capture_at_or_before(self, target_timestamp, *, max_skew_s):
        self.aligned_capture_calls.append((float(target_timestamp), float(max_skew_s)))
        return CameraFrame(
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            capture_timestamp=float(target_timestamp) - 0.01,
            receive_timestamp=time.time(),
        )

    def close(self) -> None:
        self.closed = True


class FakeRuntime:
    def __init__(self) -> None:
        self.actuation_enabled = False
        self.gripper = None
        self.read_calls = 0
        self.aligned_read_calls = []
        self.plan_calls = []
        self.execute_calls = 0
        self.verify_calls = 0
        self.abort_calls = 0
        self.closed = False

    def start(self) -> None:
        pass

    def reset_episode_start(self) -> None:
        pass

    def enable_actuation(self):
        self.actuation_enabled = True
        return None

    def read_policy_state(self) -> PolicyStateSnapshot:
        self.read_calls += 1
        return PolicyStateSnapshot(
            state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.004], dtype=np.float32),
            base_pose=np.eye(4, dtype=np.float64),
            timestamp=time.time(),
            debug={"read_call": self.read_calls},
        )

    def read_policy_state_at(self, target_timestamp, *, max_skew_s) -> PolicyStateSnapshot:
        self.aligned_read_calls.append((float(target_timestamp), float(max_skew_s)))
        timestamp = float(target_timestamp) - 0.02
        return PolicyStateSnapshot(
            state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.004], dtype=np.float32),
            base_pose=np.eye(4, dtype=np.float64),
            timestamp=timestamp,
            debug={"historical_alignment": True},
        )

    def plan_action_chunk(self, action_chunk, **_kwargs):
        self.plan_calls.append(np.asarray(action_chunk).copy())
        return SimpleNamespace(
            selected_indices=np.asarray([2], dtype=np.int64),
            timestamps=np.asarray([time.time() + 0.01], dtype=np.float64),
            debug={"safe": True},
        )

    def execute_plan(self, _plan):
        self.execute_calls += 1
        return {"dispatched": True, "gripper_command_width_m": 0.004}

    def verify_plan_completion(self, _plan, **_kwargs):
        self.verify_calls += 1
        return {"verified": True}

    def abort_motion(self) -> None:
        self.abort_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.observations = []
        self.reset_calls = 0

    def reset_connection(self) -> None:
        self.reset_calls += 1

    def predict_timed(self, observation, *, expected_steps):
        assert expected_steps == 50
        self.observations.append(observation)
        outcome = self.outcomes.pop(0)
        if outcome == "transport":
            raise RetryableInferenceError(
                "temporary tunnel failure",
                timing={"transport": "fake", "response_received": False},
            )
        if outcome == "fatal":
            raise RuntimeError("protocol/configuration failure")
        if outcome == "late":
            time.sleep(0.01)
            width = 0.049
        else:
            width = 0.006
        return (
            ActionResponse(
                action_chunk=_actions(width),
                request_id=observation.request_id,
                session_id=observation.session_id,
                metadata={"inference_time_s": 0.001},
            ),
            {"transport": "fake", "response_received": True},
        )


class FakeTactileSource:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = float(timestamp)
        self.capture_calls = 0
        self.closed = False

    def capture(self) -> TactileHistory:
        self.capture_calls += 1
        timestamps = np.asarray(
            [self.timestamp - 3.0 / 30.0, self.timestamp - 2.0 / 30.0, self.timestamp - 1.0 / 30.0, self.timestamp],
            dtype=np.float64,
        )
        return TactileHistory(
            rgb=None,
            rgb_timestamps=None,
            rgb_history_mask=None,
            marker_flow=np.zeros((4, 48, 2), dtype=np.float32),
            marker_valid_mask=np.ones((4, 48), dtype=np.bool_),
            marker_timestamps=timestamps,
            marker_history_mask=np.ones(4, dtype=np.bool_),
            debug={
                "estimated_latest_receive_timestamp": self.timestamp + 0.115,
                "latest_detected_marker_count": 48,
            },
        )

    def close(self) -> None:
        self.closed = True


def _args(tmp_path, *, steps=1, execute=False, max_rejects=3, stream_replan=False):
    argv = [
        "run",
        "--server-url",
        "http://127.0.0.1:18081",
        "--timeout",
        "5.0",
        "--steps",
        str(steps),
        "--rate-hz",
        "1000000",
        "--max-roundtrip-s",
        "0.005",
        "--max-consecutive-roundtrip-rejects",
        str(max_rejects),
        "--output-dir",
        str(tmp_path / "logs"),
        "--disable-gripper",
    ]
    if execute:
        argv.extend(
            [
                "--execute",
                "--workspace-min-xyz",
                "-1",
                "-1",
                "-1",
                "--workspace-max-xyz",
                "1",
                "1",
                "1",
            ]
        )
    if stream_replan:
        argv.append("--stream-replan")
    return build_parser().parse_args(argv)


def _install_hardware_fakes(monkeypatch, camera, runtime) -> None:
    monkeypatch.setattr(client_module, "_open_wrist_camera", lambda *_args, **_kwargs: camera)
    monkeypatch.setattr(client_module, "RealmanEpisodeRuntime", lambda _cfg: runtime)
    monkeypatch.setattr(client_module, "_wait_for_space", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(client_module, "_wait_until", lambda *_args, **_kwargs: None)


def _events(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stale_response_is_discarded_and_successful_steps_use_fresh_observations(
    tmp_path, monkeypatch
) -> None:
    camera = FakeCamera()
    runtime = FakeRuntime()
    client = FakeClient(["late", "success", "success"])
    _install_hardware_fakes(monkeypatch, camera, runtime)

    run_realman(
        _args(tmp_path, steps=2),
        client,
        {"requests_are_stateless": True},
        chunk_size=50,
    )

    assert camera.capture_calls == 3
    assert runtime.read_calls == 3
    assert len(runtime.plan_calls) == 2
    assert all(np.allclose(chunk[:, 7], 0.006) for chunk in runtime.plan_calls)
    assert len({obs.request_id for obs in client.observations}) == 3
    assert len({obs.session_id for obs in client.observations}) == 1
    assert [obs.metadata["episode_reset"] for obs in client.observations] == [True, False, False]
    assert [obs.metadata["client_step"] for obs in client.observations] == [0, 0, 1]

    events = _events(tmp_path)
    rejected = [event for event in events if event["event"] == "roundtrip_rejected"]
    completed = [event for event in events if event["event"] == "step"]
    assert len(rejected) == 1
    assert rejected[0]["will_retry_with_fresh_observation"] is True
    assert rejected[0]["rejected_action_gripper_min_m"] == pytest.approx(0.049)
    assert [event["step"] for event in completed] == [0, 1]


def test_tactile_run_uses_historical_wrist_and_robot_alignment(tmp_path, monkeypatch) -> None:
    target_timestamp = time.time() - 0.13
    tactile_source = FakeTactileSource(target_timestamp)
    camera = FakeCamera()
    runtime = FakeRuntime()
    client = FakeClient(["success"])
    _install_hardware_fakes(monkeypatch, camera, runtime)
    monkeypatch.setattr(
        client_module,
        "_open_tactile_source",
        lambda **_kwargs: tactile_source,
    )
    args = _args(tmp_path, steps=1)
    args.tactile_mode = "marker"
    args.local_tactile_guard = True
    args._tactile_requested = {"tactile_rgb": False, "tactile_marker": True}
    args._tactile_horizon = 4
    args._tactile_stride = 1
    args._protocol_version = 2
    args._tactile_contract_sha256 = "c" * 64

    run_realman(
        args,
        client,
        {
            "requests_are_stateless": True,
            "tactile_max_timestamp_skew_s": 0.05,
        },
        chunk_size=50,
    )

    assert camera.start_calls == 1
    assert camera.capture_calls == 0
    assert camera.aligned_capture_calls == pytest.approx([(target_timestamp, 0.05)])
    assert runtime.read_calls == 0
    assert runtime.aligned_read_calls == pytest.approx([(target_timestamp, 0.05)])
    assert tactile_source.capture_calls == 1
    observation = client.observations[0]
    assert observation.wrist_timestamp == pytest.approx(target_timestamp - 0.01)
    assert observation.marker_timestamps[-1] == pytest.approx(target_timestamp)
    assert observation.metadata["tactile_alignment_enabled"] is True
    assert observation.metadata["tactile_alignment_target_timestamp"] == pytest.approx(target_timestamp)
    assert observation.metadata["tactile_alignment_retry_count"] == 0


def test_tactile_run_warms_startup_history_before_first_request(tmp_path, monkeypatch) -> None:
    target_timestamp = time.time() - 0.13
    tactile_source = FakeTactileSource(target_timestamp)
    camera = FakeCamera()

    class StartupLagRuntime(FakeRuntime):
        def read_policy_state_at(self, target_timestamp, *, max_skew_s):
            if not self.aligned_read_calls:
                self.aligned_read_calls.append((float(target_timestamp), float(max_skew_s)))
                raise HistoricalAlignmentNotReady("Realman history starts after tactile target")
            return super().read_policy_state_at(
                target_timestamp,
                max_skew_s=max_skew_s,
            )

    runtime = StartupLagRuntime()
    client = FakeClient(["success"])
    _install_hardware_fakes(monkeypatch, camera, runtime)
    monkeypatch.setattr(
        client_module,
        "_open_tactile_source",
        lambda **_kwargs: tactile_source,
    )
    args = _args(tmp_path, steps=1)
    args.tactile_mode = "marker"
    args.local_tactile_guard = True
    args._tactile_requested = {"tactile_rgb": False, "tactile_marker": True}
    args._tactile_horizon = 4
    args._tactile_stride = 1
    args._protocol_version = 2
    args._tactile_contract_sha256 = "c" * 64

    run_realman(
        args,
        client,
        {
            "requests_are_stateless": True,
            "tactile_max_timestamp_skew_s": 0.05,
        },
        chunk_size=50,
    )

    assert tactile_source.capture_calls == 2
    assert len(camera.aligned_capture_calls) == 2
    assert len(runtime.aligned_read_calls) == 2
    assert len(client.observations) == 1
    observation = client.observations[0]
    assert observation.metadata["tactile_alignment_retry_count"] == 1
    assert observation.metadata["tactile_alignment_wait_s"] >= 0.0
    assert observation.metadata["tactile"]["historical_alignment"] is True


def test_success_resets_consecutive_rejection_budget(tmp_path, monkeypatch) -> None:
    camera = FakeCamera()
    runtime = FakeRuntime()
    client = FakeClient(["late", "success", "transport", "late", "success"])
    _install_hardware_fakes(monkeypatch, camera, runtime)

    run_realman(
        _args(tmp_path, steps=2),
        client,
        {"requests_are_stateless": True},
        chunk_size=50,
    )

    events = _events(tmp_path)
    rejected = [
        event
        for event in events
        if event["event"] in {"roundtrip_rejected", "inference_transport_rejected"}
    ]
    assert [event["consecutive_inference_rejects"] for event in rejected] == [1, 1, 2]
    assert len(runtime.plan_calls) == 2
    assert runtime.abort_calls == 0


def test_third_consecutive_rejection_aborts_without_planning_any_stale_action(
    tmp_path, monkeypatch
) -> None:
    camera = FakeCamera()
    runtime = FakeRuntime()
    client = FakeClient(["late", "transport", "late"])
    _install_hardware_fakes(monkeypatch, camera, runtime)

    with pytest.raises(SafetyViolation, match="consecutive rejects=3"):
        run_realman(
            _args(tmp_path, execute=True),
            client,
            {"requests_are_stateless": True},
            chunk_size=50,
        )

    assert runtime.plan_calls == []
    assert runtime.execute_calls == 0
    assert runtime.abort_calls == 1
    events = _events(tmp_path)
    rejected = [
        event
        for event in events
        if event["event"] in {"roundtrip_rejected", "inference_transport_rejected"}
    ]
    assert [event["will_retry_with_fresh_observation"] for event in rejected] == [True, True, False]
    assert any(event["event"] == "run_error" for event in events)


def test_fatal_prediction_error_is_not_retried(tmp_path, monkeypatch) -> None:
    camera = FakeCamera()
    runtime = FakeRuntime()
    client = FakeClient(["fatal", "success"])
    _install_hardware_fakes(monkeypatch, camera, runtime)

    with pytest.raises(RuntimeError, match="protocol/configuration"):
        run_realman(
            _args(tmp_path),
            client,
            {"requests_are_stateless": True},
            chunk_size=50,
        )

    assert len(client.observations) == 1
    assert runtime.plan_calls == []


def test_recovery_is_rejected_with_stream_replan_before_hardware_opens(tmp_path) -> None:
    client = FakeClient([])
    with pytest.raises(RuntimeError, match="cannot be combined with --stream-replan"):
        run_realman(
            _args(tmp_path, stream_replan=True),
            client,
            {"requests_are_stateless": True},
            chunk_size=50,
        )
    assert client.reset_calls == 0
