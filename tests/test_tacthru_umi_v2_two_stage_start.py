from types import SimpleNamespace

import pytest

from deploy.tacthru_umi_v2 import realman_client
from deploy.tacthru_umi_v2.realman_runtime import SafetyViolation


class FakeRuntime:
    def __init__(self, events):
        self.events = events

    def prepare_gripper_for_episode(self, width, *, tolerance_m, timeout_s):
        self.events.append(("prepare_gripper", width, tolerance_m, timeout_s))
        return {
            "target_width_m": width,
            "actual_width_m": width,
            "error_m": 0.0,
        }

    def enable_actuation(self):
        self.events.append(("enable_actuation",))


def test_two_stage_start_waits_again_after_verified_grip(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    runtime = FakeRuntime(events)
    args = SimpleNamespace(
        gripper_startup_width_m=0.004,
        gripper_startup_tolerance_m=0.001,
        gripper_startup_timeout_s=2.0,
    )

    def fake_wait(_camera, *, execute, label=None, allow_cancel=True):
        events.append(("space", execute, label, allow_cancel))

    monkeypatch.setattr(realman_client, "_wait_for_space", fake_wait)
    result = realman_client._prepare_real_execution(args, runtime, camera=None)

    assert result == {
        "target_width_m": pytest.approx(0.004),
        "actual_width_m": pytest.approx(0.004),
        "error_m": pytest.approx(0.0),
    }
    assert [event[0] for event in events] == [
        "space",
        "prepare_gripper",
        "space",
        "enable_actuation",
    ]
    assert events[0][1] is False
    assert "CLOSE GRIPPER" in events[0][2]
    assert events[2][1] is True
    assert events[2][2] == "START INFERENCE AND ENABLE ARM EXECUTION"


def test_two_stage_start_does_not_enable_after_grip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    runtime = FakeRuntime(events)
    args = SimpleNamespace(
        gripper_startup_width_m=0.004,
        gripper_startup_tolerance_m=0.001,
        gripper_startup_timeout_s=2.0,
    )

    def fake_wait(_camera, *, execute, label=None, allow_cancel=True):
        events.append(("space", execute, label, allow_cancel))

    def fail_prepare(*_args, **_kwargs):
        events.append(("prepare_failed",))
        raise SafetyViolation("startup grip failed")

    monkeypatch.setattr(realman_client, "_wait_for_space", fake_wait)
    runtime.prepare_gripper_for_episode = fail_prepare

    with pytest.raises(SafetyViolation, match="startup grip failed"):
        realman_client._prepare_real_execution(args, runtime, camera=None)

    assert [event[0] for event in events] == ["space", "prepare_failed"]
