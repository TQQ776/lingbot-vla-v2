from __future__ import annotations

import ast
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from deploy.tacthru_umi_v2.slow_fast_scheduler import SlowFastValiditySettings
from deploy.tacthru_umi_v2.slow_fast_scheduler import (
    SlowPlanRuntimeState,
    evaluate_slow_plan,
)


POLICY_PATH = Path(__file__).parents[1] / "deploy/lingbot_vla_v2_policy.py"


def _scheduler_harness():
    tree = ast.parse(POLICY_PATH.read_text(encoding="utf-8"))
    server = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LingbotVLAv2Server"
    )
    method = next(
        node
        for node in server.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_infer_cascaded_single"
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(
                    name="Harness",
                    bases=[],
                    keywords=[],
                    body=[method],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "np": np,
        "torch": torch,
        "time": time,
        "SlowPlanRuntimeState": SlowPlanRuntimeState,
        "evaluate_slow_plan": evaluate_slow_plan,
    }
    exec(compile(module, str(POLICY_PATH), "exec"), namespace)
    return namespace["Harness"]


class FakeCascadedVLA:
    def __init__(self):
        self.calls = {"slow": 0, "refresh": 0, "fast": 0}

    def _single_cascaded_model_inputs(self, observation, *, use_bf16):
        state = torch.zeros(1, 55)
        return {"state": state}, {
            "state": state,
            "marker_displacement_history": None,
            "marker_valid_mask": None,
            "marker_history_valid_mask": None,
            "marker_contact_state": None,
            "tactile_sensor_mask": None,
        }

    def build_cascaded_slow_plan(self, **kwargs):
        self.calls["slow"] += 1
        created_at_s = time.monotonic()
        return SimpleNamespace(
            created_at_s=created_at_s,
            prefix_created_at_s=created_at_s,
            plan_version=self.calls["slow"] + self.calls["refresh"],
            scene_version=kwargs["scene_version"],
        )

    def refresh_cascaded_action_plan(self, plan, **kwargs):
        self.calls["refresh"] += 1
        return SimpleNamespace(
            created_at_s=time.monotonic(),
            prefix_created_at_s=plan.prefix_created_at_s,
            plan_version=self.calls["slow"] + self.calls["refresh"],
            scene_version=plan.scene_version,
        )

    def refine_action_with_tactile(self, plan, **kwargs):
        self.calls["fast"] += 1
        return torch.zeros(1, 50, 55)


def _state(x):
    return np.asarray([x, 0, 0, 0, 0, 0, 1, 0.004], dtype=np.float32)


def _server():
    server = _scheduler_harness()()
    server.vla = FakeCascadedVLA()
    server.use_bf16 = False
    server.vtla_scheduler_mode = "auto"
    server.vtla_validity_settings = SlowFastValiditySettings()
    server._online_slow_plan = None
    server._online_plan_runtime = None
    server.last_inference_metadata = {}
    server._prepare_model_input = MethodType(lambda self, observation: {}, server)
    server._unapply_batched_actions = MethodType(
        lambda self, observations, actions: {
            "action": np.zeros((1, 50, 8), dtype=np.float32)
        },
        server,
    )
    return server


def _observation(x):
    return {
        "observation.state": _state(x),
        "task": "Insert the Ethernet cable",
        "_vtla_request": {
            "scene_version": 1,
            "executed_offset": 0,
            "vtla_mode": "auto",
        },
    }


def test_online_policy_routes_reuse_action_refresh_and_full_rebuild():
    server = _server()

    first = server._infer_cascaded_single(_observation(0.0))
    assert first["action"].shape == (1, 50, 8)
    assert server.last_inference_metadata["decision"]["level"] == "rebuild"

    server._infer_cascaded_single(_observation(0.002))
    assert server.last_inference_metadata["decision"]["level"] == "reuse"

    server._infer_cascaded_single(_observation(0.010))
    assert server.last_inference_metadata["decision"]["level"] == "refresh_action"

    server._infer_cascaded_single(_observation(0.040))
    assert server.last_inference_metadata["decision"]["level"] == "rebuild"
    assert server.vla.calls == {"slow": 2, "refresh": 1, "fast": 4}
