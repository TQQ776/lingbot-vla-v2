from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import time

import pytest
import torch
from torch import nn

from lingbotvla.models.vla.lingbot_vla.tactile_action_expert import (
    TactileActionExpert,
    TactileExpertConfig,
    TactileRefinementConfig,
    sinusoidal_position_embedding,
)
from lingbotvla.models.vla.lingbot_vla.tactile_vtla import (
    load_vtla_checkpoint_state_dict,
)
from lingbotvla.optim.vtla import build_vtla_param_groups


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"


def _load_cascaded_methods():
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )
    selected = {
        "_predict_tactile_velocity",
        "_tactile_slow_context_mask",
        "build_slow_action_plan",
        "refine_action_with_tactile",
    }
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected
    ]
    harness = ast.ClassDef(
        name="CascadedHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "time": time,
        "sinusoidal_time_embedding": __import__(
            "lingbotvla.models.vla.lingbot_vla.tactile_action_expert",
            fromlist=["sinusoidal_time_embedding"],
        ).sinusoidal_time_embedding,
        "sinusoidal_position_embedding": sinusoidal_position_embedding,
        "SlowActionPlan": SimpleNamespace,
        "VTLASlowCache": SimpleNamespace,
    }
    exec(compile(module, str(MODEL_PATH), "exec"), namespace)
    return namespace["CascadedHarness"]


CascadedHarness = _load_cascaded_methods()


class _MarkerEncoder(nn.Module):
    marker_dtype = torch.float32

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(2, hidden_size, bias=False)
        self.last_marker_diagnostics = {
            "contact_on_ratio": torch.tensor(1.0),
        }

    def encode_markers(
        self,
        history,
        marker_valid_mask,
        marker_history_valid_mask,
        tactile_sensor_mask,
        marker_contact_state,
    ):
        tokens = self.proj(history)
        batch, sensors, frames, markers = history.shape[:4]
        if marker_contact_state is None:
            visible = torch.ones(batch, sensors, dtype=torch.bool, device=history.device)
        else:
            visible = (marker_contact_state == 1) | (marker_contact_state == 2)
        mask = marker_valid_mask & marker_history_valid_mask[:, :, :, None]
        mask = mask & visible[:, :, None, None]
        if tactile_sensor_mask is not None:
            mask = mask & tactile_sensor_mask[:, :, None, None]
        tokens = tokens * mask.unsqueeze(-1)
        self.last_marker_diagnostics = {
            "contact_on_ratio": visible.float().mean().detach(),
        }
        return tokens.flatten(2, 3), mask.flatten(2, 3)


def _settings() -> TactileRefinementConfig:
    return TactileRefinementConfig.from_mapping(
        {
            "enabled": True,
            "mode": "cascaded_flow",
            "tau_split": 0.4,
            "expert": {
                "hidden_size": 16,
                "num_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
            },
            "inference": {
                "total_steps": 10,
                "slow_steps": 6,
                "tactile_steps": 4,
            },
        }
    )


def _slow_cache(batch: int = 1, hidden_size: int = 20):
    prefix_len = 7
    return SimpleNamespace(
        past_key_values={0: {"key_states": torch.zeros(batch, prefix_len, 1)}},
        pad_masks=torch.ones(batch, prefix_len, dtype=torch.bool),
        att_masks=torch.ones(batch, prefix_len, dtype=torch.bool),
        input_ids=torch.zeros(batch, prefix_len, dtype=torch.long),
        position_ids=torch.arange(prefix_len).view(1, 1, -1).expand(3, batch, -1),
        rope_grid_thw=torch.ones(1, 3, dtype=torch.long),
        prefix_len=prefix_len,
        version=5,
        final_hidden_states=torch.randn(batch, prefix_len, hidden_size),
        includes_task_queries=True,
    )


def _harness() -> SimpleNamespace:
    torch.manual_seed(7)
    settings = _settings()
    hidden = settings.expert.hidden_size
    harness = SimpleNamespace(
        tactile_refinement_settings=settings,
        config=SimpleNamespace(n_action_steps=50, max_action_dim=55),
        tactile_encoder=_MarkerEncoder(hidden_size=20),
        tactile_action_expert=TactileActionExpert(settings.expert),
        tactile_state_proj=nn.Linear(55, hidden),
        tactile_action_in_proj=nn.Linear(55, hidden),
        tactile_action_out_proj=nn.Linear(hidden, 55),
        tactile_context_proj=nn.Linear(20, hidden),
        tactile_plan_proj=nn.Linear(12, hidden),
        tactile_marker_proj=nn.Linear(20, hidden),
        tactile_time_mlp=nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        ),
        use_depth_align=False,
        slow_calls=[],
        tactile_calls=[],
        tactile_action_shapes=[],
        tactile_context_shapes=[],
        action_context_calls=[],
    )
    for name in (
        "_predict_tactile_velocity",
        "_tactile_slow_context_mask",
        "build_slow_action_plan",
        "refine_action_with_tactile",
    ):
        setattr(harness, name, MethodType(getattr(CascadedHarness, name), harness))

    def predict_velocity(
        self,
        state,
        pad,
        cache,
        x_t,
        timestep,
        *,
        return_action_hidden=False,
        **_,
    ):
        velocity = torch.ones_like(x_t)
        if return_action_hidden:
            self.action_context_calls.append(timestep.detach().clone())
            context = torch.ones(x_t.shape[0], x_t.shape[1], 12)
            return velocity, context.to(device=x_t.device, dtype=x_t.dtype)
        self.slow_calls.append(timestep.detach().clone())
        return velocity

    original_tactile = harness._predict_tactile_velocity

    def tactile_velocity(self, **kwargs):
        self.tactile_calls.append(kwargs["timestep"].detach().clone())
        self.tactile_action_shapes.append(tuple(kwargs["x_t"].shape))
        self.tactile_context_shapes.append(
            tuple(kwargs["action_context_tokens"].shape)
        )
        return original_tactile(**kwargs)

    harness.predict_velocity = MethodType(predict_velocity, harness)
    harness._predict_tactile_velocity = MethodType(tactile_velocity, harness)
    return harness


def _marker_inputs(value: float = 0.0, *, contact: int = 1):
    history = torch.full((1, 1, 4, 48, 2), value)
    return {
        "marker_displacement_history": history,
        "marker_valid_mask": torch.ones(1, 1, 4, 48, dtype=torch.bool),
        "marker_history_valid_mask": torch.ones(1, 1, 4, dtype=torch.bool),
        "marker_contact_state": torch.full((1, 1), contact, dtype=torch.int8),
        "tactile_sensor_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def _plan(harness, action_offset: int = 0):
    state = torch.zeros(1, 55)
    noise = torch.zeros(1, 50, 55)
    return harness.build_slow_action_plan(
        _slow_cache(), state, noise=noise, action_offset=action_offset
    )


def test_cascaded_intervals_cover_one_to_zero_without_gap_or_overlap():
    harness = _harness()
    plan = _plan(harness)
    harness.refine_action_with_tactile(
        plan, state=torch.zeros(1, 55), **_marker_inputs()
    )
    assert [round(float(value.item()), 7) for value in harness.slow_calls] == [
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
    ]
    assert [round(float(value.item()), 7) for value in harness.tactile_calls] == [
        0.4,
        0.3,
        0.2,
        0.1,
    ]
    assert plan.tau_split == 0.4
    assert [round(float(value.item()), 7) for value in harness.action_context_calls] == [
        0.4
    ]
    assert plan.action_context.shape == (1, 50, 12)


def test_refinement_does_not_call_slow_model_or_vit():
    harness = _harness()
    plan = _plan(harness)
    slow_call_count = len(harness.slow_calls)
    harness.refine_action_with_tactile(
        plan, state=torch.zeros(1, 55), **_marker_inputs(0.25)
    )
    assert len(harness.slow_calls) == slow_call_count


def test_marker_change_changes_refined_action():
    harness = _harness()
    plan = _plan(harness)
    state = torch.zeros(1, 55)
    output_a = harness.refine_action_with_tactile(
        plan, state=state, **_marker_inputs(0.1)
    )
    output_b = harness.refine_action_with_tactile(
        plan, state=state, **_marker_inputs(0.9)
    )
    assert not torch.allclose(output_a, output_b)


def test_gate_off_is_finite_and_has_valid_shape():
    harness = _harness()
    output = harness.refine_action_with_tactile(
        _plan(harness), state=torch.zeros(1, 55), **_marker_inputs(1.0, contact=0)
    )
    assert output.shape == (1, 50, 55)
    assert torch.isfinite(output).all()


def test_x_split_is_not_modified_in_place():
    harness = _harness()
    plan = _plan(harness)
    snapshot = plan.x_split.clone()
    harness.refine_action_with_tactile(
        plan, state=torch.zeros(1, 55), **_marker_inputs(0.5)
    )
    torch.testing.assert_close(plan.x_split, snapshot)


def test_repeated_refinement_restarts_from_same_x_split():
    harness = _harness()
    plan = _plan(harness)
    state = torch.zeros(1, 55)
    first = harness.refine_action_with_tactile(
        plan, state=state, **_marker_inputs(0.2)
    )
    second = harness.refine_action_with_tactile(
        plan, state=state, **_marker_inputs(0.2)
    )
    torch.testing.assert_close(first, second)


def test_action_offset_keeps_executed_prefix_fixed():
    harness = _harness()
    plan = _plan(harness, action_offset=2)
    output = harness.refine_action_with_tactile(
        plan, state=torch.zeros(1, 55), **_marker_inputs(0.7)
    )
    torch.testing.assert_close(output[:, :2], plan.x_split[:, :2])
    assert not torch.allclose(output[:, 2:], plan.x_split[:, 2:])
    assert harness.tactile_action_shapes == [(1, 50, 55)] * 4
    assert harness.tactile_context_shapes == [(1, 50, 12)] * 4


def test_action_offset_can_reuse_a_previous_legal_refined_prefix():
    harness = _harness()
    plan = _plan(harness, action_offset=2)
    legal_prefix = torch.full((1, 2, 55), 0.125)
    output = harness.refine_action_with_tactile(
        plan,
        state=torch.zeros(1, 55),
        fixed_action_prefix=legal_prefix,
        **_marker_inputs(0.7),
    )
    torch.testing.assert_close(output[:, :2], legal_prefix)


def test_tactile_expert_preserves_action_shape_and_backward():
    settings = _settings()
    expert = TactileActionExpert(settings.expert)
    batch, action_steps, hidden = 2, 50, settings.expert.hidden_size
    action = torch.randn(batch, action_steps + 1, hidden, requires_grad=True)
    marker = torch.randn(batch, 192, hidden, requires_grad=True)
    action_context = torch.randn(batch, action_steps, hidden, requires_grad=True)
    slow = torch.randn(batch, 11, hidden, requires_grad=True)
    output = expert(
        action,
        marker,
        torch.ones(batch, 192, dtype=torch.bool),
        action_context,
        torch.ones(batch, action_steps, dtype=torch.bool),
        slow,
        torch.ones(batch, 11, dtype=torch.bool),
    )
    assert output.shape == action.shape
    output.square().mean().backward()
    assert action.grad is not None
    assert marker.grad is not None
    assert action_context.grad is not None
    assert slow.grad is not None


def test_action_position_embedding_is_absolute_and_deterministic():
    positions = torch.arange(50)
    first = sinusoidal_position_embedding(positions, 16)
    second = sinusoidal_position_embedding(positions, 16)
    assert first.shape == (50, 16)
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first[0], first[1])
    assert not torch.allclose(first[2], first[12])


def test_configuration_rejects_gap_and_defaults_to_full_replan():
    default = TactileRefinementConfig.from_mapping(None)
    assert default.enabled is False
    assert default.mode == "full_replan"
    with pytest.raises(ValueError, match="no overlap or gap"):
        TactileRefinementConfig.from_mapping(
            {
                "enabled": True,
                "mode": "cascaded_flow",
                "tau_split": 0.4,
                "inference": {
                    "total_steps": 10,
                    "slow_steps": 7,
                    "tactile_steps": 3,
                },
            }
        )


def test_full_replan_source_and_cascaded_config_are_both_present():
    model_source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    config_source = (
        ROOT
        / "configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker_point192_history4.yaml"
    ).read_text(encoding="utf-8")
    assert "def sample_actions_fast(" in model_source
    assert "full_replan" in (
        ROOT / "lingbotvla/models/vla/lingbot_vla/tactile_action_expert.py"
    ).read_text(encoding="utf-8")
    assert "marker_history_length: 4" in config_source


def test_old_checkpoint_may_initialize_only_complete_refinement_namespace():
    class TinyCheckpointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.tactile_action_expert = nn.Linear(2, 2)
            self.tactile_state_proj = nn.Linear(2, 2)

    source = TinyCheckpointModel()
    old_state = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if not name.startswith("tactile_")
    }
    target = TinyCheckpointModel()
    report = load_vtla_checkpoint_state_dict(target, old_state)
    assert report["forbidden_missing_keys"] == []
    assert report["tactile_refinement_initialized_keys"]

    partial_state = dict(old_state)
    partial_state["tactile_action_expert.weight"] = (
        source.tactile_action_expert.weight.detach().clone()
    )
    with pytest.raises(RuntimeError, match="missing"):
        load_vtla_checkpoint_state_dict(TinyCheckpointModel(), partial_state)


def test_pre_plan_context_checkpoint_initializes_only_new_plan_attention():
    class UpgradeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_action_expert = TactileActionExpert(_settings().expert)
            self.tactile_plan_proj = nn.Linear(12, _settings().expert.hidden_size)

    source = UpgradeModel()
    old_state = {
        name: value.clone()
        for name, value in source.state_dict().items()
        if ".plan_norm." not in name
        and ".plan_attention." not in name
        and not name.startswith("tactile_plan_proj.")
    }
    report = load_vtla_checkpoint_state_dict(UpgradeModel(), old_state)
    assert report["forbidden_missing_keys"] == []
    initialized = report["tactile_refinement_upgrade_initialized_keys"]
    assert initialized
    assert all(
        ".plan_norm." in name
        or ".plan_attention." in name
        or name.startswith("tactile_plan_proj.")
        for name in initialized
    )


def test_refinement_parameters_use_new_module_optimizer_group():
    class TinyOptimizerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = nn.Linear(2, 2)
            self.tactile_action_expert = nn.Linear(2, 2)
            self.tactile_plan_proj = nn.Linear(2, 2)
            self.qwen_expert = nn.Linear(2, 2)

    model = TinyOptimizerModel()
    groups = build_vtla_param_groups(
        model,
        new_modules_lr=1e-4,
        action_expert_lr=1e-5,
        base_lr=2e-5,
        weight_decay=0.0,
    )
    by_name = {group["name"]: group for group in groups}
    tactile_ids = {id(parameter) for parameter in by_name["tactile"]["params"]}
    assert id(model.tactile_encoder.weight) in tactile_ids
    assert id(model.tactile_action_expert.weight) in tactile_ids
    assert id(model.tactile_plan_proj.weight) in tactile_ids
    assert id(model.qwen_expert.weight) not in tactile_ids


def test_cascaded_source_keeps_baseline_hidden_optional_and_marker_freeze_scoped():
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert (
        "final_hidden_states=slow_outputs[0] if include_task_queries else None"
        in source
    )
    assert "self.tactile_encoder.requires_grad_(False)" in source
    assert 'marker_parameter_names = {"marker_modality_embedding"}' in source
    assert "tactile_rgb_projection" not in source[
        source.index("marker_parameter_prefixes = (") :
        source.index("for module in (", source.index("marker_parameter_prefixes = ("))
    ]


def test_real_client_refreshes_slow_context_when_plan_is_exhausted():
    source = (
        ROOT / "deploy/tacthru_umi_v2/realman_client.py"
    ).read_text(encoding="utf-8")
    assert "plan_exhausted = bool(" in source
    assert "or plan_exhausted" in source
    assert 'refresh_reason = "slow_plan_exhausted"' in source
