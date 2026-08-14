from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
    LingbotVLAV2Config,
)
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
    LingbotVlaV2Policy,
    QwenvlWithExpertV2Model,
)
from lingbotvla.models.vla.lingbot_vla.tactile_action_expert import (
    CascadedSlowPlan,
    TactileExpertSettings,
    TactileRefinementConfig,
    build_tactile_expert,
    clone_kv_cache,
    copy_matching_action_weights,
)
from lingbotvla.models.vla.lingbot_vla.utils import (
    build_three_stream_attention_mask,
    make_att_2d_masks,
    our_eager_attention_forward,
)
from lingbotvla.optim.vtla import build_vtla_param_groups


ROOT = Path(__file__).parents[1]
MODEL_PATH = (
    ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
)
CONFIG_PATH = (
    ROOT
    / "configs/vla/tacthru_umi/"
    "tacthru_umi_vtla_rgb_marker_point192_history4_three_stream_mot.yaml"
)


def _flow_method(name: str):
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    flow = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )
    return next(
        node
        for node in flow.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _method_harness(*names: str):
    methods = [_flow_method(name) for name in names]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(
                    name="Harness",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "Tensor": torch.Tensor,
        "torch": torch,
        "clone_kv_cache": clone_kv_cache,
    }
    exec(compile(module, str(ROOT / "three_stream_harness.py"), "exec"), namespace)
    return namespace["Harness"]


def test_three_stream_config_and_split_schedule():
    config = TactileRefinementConfig.from_mapping({"enabled": True})
    assert config.expert.num_layers == 36
    assert config.tau_split == pytest.approx(0.6)
    assert config.inference.total_steps == 10
    assert config.inference.slow_steps == 4
    assert config.inference.tactile_steps == 6

    with pytest.raises(ValueError, match="tau_split"):
        TactileRefinementConfig.from_mapping(
            {"enabled": True, "tau_split": 0.5}
        )


def test_training_yaml_resolves_exact_first_version_contract():
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = LingbotVLAV2Config(**payload["model"], **payload["train"])
    assert config.tactile_refinement_enabled is True
    assert config.tactile_refinement["expert"]["num_layers"] == 36
    assert config.tactile["marker_history_length"] == 4
    assert config.tactile["marker_tokenization"]["num_regions"] == 48
    assert config.chunk_size == 50
    assert config.max_action_dim == 55


def test_three_stream_mask_has_exact_cross_stream_visibility():
    prefix_pad = torch.ones(1, 3, dtype=torch.bool)
    prefix_att = torch.ones_like(prefix_pad)
    action_pad = torch.ones(1, 2, dtype=torch.bool)
    action_att = torch.tensor([[True, False]])
    tactile_pad = torch.ones(1, 4, dtype=torch.bool)
    mask = build_three_stream_attention_mask(
        prefix_pad, prefix_att, action_pad, action_att, tactile_pad
    )[0]
    p = slice(0, 3)
    a = slice(3, 5)
    t = slice(5, 9)
    assert mask[p, p].any()
    assert not mask[p, a].any()
    assert not mask[p, t].any()
    assert mask[a, p].all()
    assert mask[a, a].any()
    assert not mask[a, t].any()
    assert mask[t, p].all()
    assert mask[t, a].all()
    assert mask[t, t].all()


def test_three_stream_mask_preserves_original_prefix_action_semantics():
    prefix_pad = torch.tensor([[True, True, False, True]])
    prefix_att = torch.tensor([[True, True, True, True]])
    action_pad = torch.tensor([[True, True, True]])
    action_att = torch.tensor([[True, True, False]])
    tactile_pad = torch.tensor([[True, False]])
    expected = make_att_2d_masks(
        torch.cat([prefix_pad, action_pad], dim=1),
        torch.cat([prefix_att, action_att], dim=1),
    )
    actual = build_three_stream_attention_mask(
        prefix_pad, prefix_att, action_pad, action_att, tactile_pad
    )
    assert torch.equal(actual[:, :7, :7], expected)


def test_tactile_sequence_layout_is_marker192_time1_action50():
    Harness = _method_harness("build_tactile_sequence")
    harness = Harness()
    harness.tactile_refinement_settings = SimpleNamespace(enabled=True)
    harness.config = SimpleNamespace(n_action_steps=50, max_action_dim=55)
    harness.tactile_settings = SimpleNamespace(
        num_sensors=1,
        marker_history_length=4,
        marker_tokenization=SimpleNamespace(num_regions=48),
    )
    harness.tactile_action_in_proj = nn.Linear(55, 8)

    class Time(nn.Module):
        def forward(self, timestep, *, dtype):
            return torch.zeros(timestep.shape[0], 1, 8, dtype=dtype)

    harness.tactile_time_embedder = Time()

    def encode(self, **kwargs):
        batch = kwargs["batch_size"]
        return (
            torch.zeros(batch, 192, 8),
            torch.ones(batch, 192, dtype=torch.bool),
            torch.ones(batch, dtype=torch.bool),
        )

    harness._encode_tactile_marker_tokens = MethodType(encode, harness)
    sequence, mask, active = harness.build_tactile_sequence(
        torch.zeros(2, 50, 55),
        torch.tensor([0.6, 0.2]),
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
    )
    assert sequence.shape == (2, 243, 8)
    assert mask.shape == (2, 243)
    assert active.all()


def test_three_stream_positions_continue_after_prefix_and_action():
    Harness = _method_harness(
        "_build_full_position_ids", "_build_three_stream_position_ids"
    )
    harness = Harness()
    prefix_positions = torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]])
    prefix_pad = torch.ones(1, 3, dtype=torch.bool)
    action_pad = torch.ones(1, 2, dtype=torch.bool)
    tactile_pad = torch.ones(1, 4, dtype=torch.bool)
    positions = harness._build_three_stream_position_ids(
        prefix_positions, prefix_pad, action_pad, tactile_pad
    )
    assert positions.shape == (3, 1, 9)
    assert positions[0, 0].tolist() == list(range(9))


def test_matching_initialization_copies_values_without_parameter_sharing():
    action = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
    tactile = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    with torch.no_grad():
        action[0].weight.fill_(2.0)
    copied = copy_matching_action_weights(action, tactile)
    assert "0.weight" in copied
    assert torch.equal(action[0].weight, tactile[0].weight)
    assert action[0].weight is not tactile[0].weight
    tactile[0].weight.data.zero_()
    assert not torch.equal(action[0].weight, tactile[0].weight)


def test_real_tactile_expert_has_36_independent_decoder_layers_and_qkv():
    settings = TactileExpertSettings(
        hidden_size=16,
        intermediate_size=24,
        num_layers=36,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    action = build_tactile_expert(
        settings, use_cache=True, eval_mode=False
    )
    tactile = build_tactile_expert(
        settings, use_cache=True, eval_mode=False
    )
    copy_matching_action_weights(action, tactile)
    assert len(action.model.layers) == len(tactile.model.layers) == 36
    for action_layer, tactile_layer in zip(
        action.model.layers, tactile.model.layers
    ):
        assert action_layer.self_attn.q_proj.weight is not tactile_layer.self_attn.q_proj.weight
        assert action_layer.self_attn.k_proj.weight is not tactile_layer.self_attn.k_proj.weight
        assert action_layer.self_attn.v_proj.weight is not tactile_layer.self_attn.v_proj.weight
        assert action_layer.self_attn.o_proj.weight is not tactile_layer.self_attn.o_proj.weight
        assert action_layer.input_layernorm.weight is not tactile_layer.input_layernorm.weight
        assert action_layer.post_attention_layernorm.weight is not tactile_layer.post_attention_layernorm.weight


def test_real_joint_forward_processes_all_three_streams_layer_by_layer():
    class VlmLayer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, *args, **kwargs):
            result = self.layer(*args, **kwargs)
            return result[0] if kwargs.get("output_atten") else result

    class VlmModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.layers = nn.ModuleList([VlmLayer(layer) for layer in model.layers])
            self.norm = model.norm

    settings = TactileExpertSettings(
        hidden_size=16,
        intermediate_size=24,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    prefix, action, tactile = [
        build_tactile_expert(settings, use_cache=True, eval_mode=False)
        for _ in range(3)
    ]
    model = object.__new__(QwenvlWithExpertV2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        qwen_expert_config=SimpleNamespace(num_hidden_layers=2),
        final_norm_adanorm=False,
        attention_implementation="eager",
    )
    model.qwenvl = SimpleNamespace(
        model=SimpleNamespace(language_model=VlmModel(prefix.model)),
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=2,
            )
        ),
    )
    model.qwen_expert = action
    model.tactile_expert = tactile
    model.attention_interface = our_eager_attention_forward
    model.apply_mrope = MethodType(lambda self, q, k, positions: (q, k), model)

    p = torch.randn(1, 2, 16, requires_grad=True)
    a = torch.randn(1, 3, 16, requires_grad=True)
    t = torch.randn(1, 4, 16, requires_grad=True)
    pp = torch.ones(1, 2, dtype=torch.bool)
    ap = torch.ones(1, 3, dtype=torch.bool)
    tp = torch.ones(1, 4, dtype=torch.bool)
    mask = build_three_stream_attention_mask(
        pp,
        torch.ones_like(pp),
        ap,
        torch.tensor([[True, True, False]]),
        tp,
    )
    outputs, _, _ = model.forward(
        attention_mask=mask,
        position_ids=torch.zeros(3, 1, 9, dtype=torch.long),
        inputs_embeds=[p, a, t],
        use_cache=False,
        fill_kv_cache=False,
    )
    assert [tuple(output.shape) for output in outputs] == [
        (1, 2, 16),
        (1, 3, 16),
        (1, 4, 16),
    ]


def test_slow_plan_is_frozen_and_cache_clone_does_not_replace_source():
    cache = {0: {"key_states": torch.ones(1), "value_states": torch.ones(1)}}
    cloned = clone_kv_cache(cache, clone_tensors=True)
    cloned[0]["key_states"].zero_()
    assert cache[0]["key_states"].item() == 1
    plan = CascadedSlowPlan(
        x_split=torch.zeros(1, 50, 55),
        tau_split=0.6,
        noise=torch.zeros(1, 50, 55),
        prefix_past_key_values=cache,
        past_key_values=cache,
        prefix_pad_masks=torch.ones(1, 2, dtype=torch.bool),
        action_pad_masks=torch.ones(1, 51, dtype=torch.bool),
        prefix_position_ids=torch.zeros(3, 1, 2, dtype=torch.long),
        prefix_len=2,
        action_len=51,
    )
    with pytest.raises(FrozenInstanceError):
        plan.tau_split = 0.5


def test_model_source_builds_independent_three_models_and_dynamic_spans():
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert "models.append(self.tactile_expert.model)" in source
    assert "tactile_num_layers == action_num_layers == num_layers" in source
    forward = ast.unparse(_flow_method("build_tactile_sequence"))
    assert "torch.cat([marker_tokens.to(hidden_dtype), time_token, action_tokens]" in forward
    joint_forward = ast.unparse(
        next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)
            and node.name == "QwenvlWithExpertV2Model"
        )
    )
    assert "end = start + hidden_states.shape[1]" in joint_forward


def test_slow_and_fast_tau_schedules_and_boundary_refresh_are_explicit():
    slow = ast.unparse(_flow_method("build_cascaded_slow_plan"))
    fast = ast.unparse(_flow_method("refine_action_with_tactile"))
    refresh = ast.unparse(_flow_method("_refresh_action_boundary_cache"))
    assert "1.0 + index * dt" in slow
    assert "x_t = x_t + dt * velocity" in slow
    assert "plan.tau_split + index * dt" in fast
    assert "x_t = plan.x_split.clone()" in fast
    assert "append_kv_cache=True" in refresh
    assert "state, x_split, boundary_time" in refresh


def test_fast_path_uses_only_tactile_stream_and_gate_off_action_fallback():
    velocity = ast.unparse(_flow_method("_predict_tactile_velocity"))
    refine = ast.unparse(_flow_method("refine_action_with_tactile"))
    assert "inputs_embeds=[None, None, tactile_embs]" in velocity
    assert "outputs[2][:, -self.config.n_action_steps:]" in velocity
    assert "_action_fallback_from_boundary" in refine
    assert "torch.where(contact_active[:, None, None], x_t, fallback)" in refine


def test_training_source_has_analytic_tau_sampling_and_boundary_exposure():
    training = ast.unparse(_flow_method("_forward_cascaded_training_streams"))
    assert "analytic_boundary = split * noise + (1.0 - split) * actions" in training
    assert "torch.rand(state.shape[0]" in training
    assert "_rollout_action_to_boundary" in training
    assert "tactile_time = torch.full" in training
    assert "_refresh_action_boundary_cache" in training


def test_optimizer_groups_all_new_tactile_parameters_and_respects_freezing():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = nn.Linear(2, 2)
            self.marker_to_tactile_proj = nn.Linear(2, 2)
            self.tactile_time_embedder = nn.Linear(2, 2)
            self.tactile_action_in_proj = nn.Linear(2, 2)
            self.tactile_action_out_proj = nn.Linear(2, 2)
            self.qwen_expert = nn.Linear(2, 2)
            self.base = nn.Linear(2, 2)

    model = Tiny()
    model.qwen_expert.requires_grad_(False)
    groups = build_vtla_param_groups(
        model,
        new_modules_lr=1e-4,
        action_expert_lr=1e-5,
        base_lr=5e-5,
        weight_decay=0.01,
    )
    by_name = {group["name"]: group for group in groups}
    expected_tactile = sum(
        module.weight.numel() + module.bias.numel()
        for module in (
            model.tactile_encoder,
            model.marker_to_tactile_proj,
            model.tactile_time_embedder,
            model.tactile_action_in_proj,
            model.tactile_action_out_proj,
        )
    )
    assert sum(p.numel() for p in by_name["tactile"]["params"]) == expected_tactile
    assert "action_expert" not in by_name


def test_fast_refinement_is_deterministic_marker_sensitive_and_does_not_run_slow():
    Harness = _method_harness("refine_action_with_tactile")
    harness = Harness()
    harness.tactile_refinement_settings = SimpleNamespace(
        tau_split=0.6,
        inference=SimpleNamespace(tactile_steps=6),
    )
    calls = {"tactile": 0, "fallback": 0}

    def tactile(self, **kwargs):
        calls["tactile"] += 1
        marker = kwargs["marker_displacement_history"]
        scale = marker.mean(dim=(1, 2, 3, 4))[:, None, None]
        return torch.ones_like(kwargs["x_t"]) * scale, torch.ones(
            marker.shape[0], dtype=torch.bool
        )

    def fallback(self, plan, *, state):
        calls["fallback"] += 1
        return torch.full_like(plan.x_split, 99.0)

    harness._predict_tactile_velocity = MethodType(tactile, harness)
    harness._action_fallback_from_boundary = MethodType(fallback, harness)
    plan = CascadedSlowPlan(
        x_split=torch.zeros(1, 50, 55),
        tau_split=0.6,
        noise=torch.zeros(1, 50, 55),
        prefix_past_key_values={},
        past_key_values={},
        prefix_pad_masks=torch.ones(1, 1, dtype=torch.bool),
        action_pad_masks=torch.ones(1, 51, dtype=torch.bool),
        prefix_position_ids=torch.zeros(3, 1, 1, dtype=torch.long),
        prefix_len=1,
        action_len=51,
    )
    state = torch.zeros(1, 55)
    marker_a = torch.ones(1, 1, 4, 48, 2)
    marker_b = marker_a * 2
    result_a1 = harness.refine_action_with_tactile(
        plan,
        state=state,
        marker_displacement_history=marker_a,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
    )
    result_a2 = harness.refine_action_with_tactile(
        plan,
        state=state,
        marker_displacement_history=marker_a,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
    )
    result_b = harness.refine_action_with_tactile(
        plan,
        state=state,
        marker_displacement_history=marker_b,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
    )
    assert torch.equal(result_a1, result_a2)
    assert not torch.equal(result_a1, result_b)
    assert calls == {"tactile": 18, "fallback": 0}
    assert torch.equal(plan.x_split, torch.zeros_like(plan.x_split))


def test_old_checkpoint_strict_false_has_only_expected_new_missing_keys():
    class Legacy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action = nn.Linear(2, 2)
            self.marker_encoder = nn.Linear(2, 2)

    class ThreeStream(Legacy):
        def __init__(self):
            super().__init__()
            self.tactile_expert = nn.Linear(2, 2)
            self.marker_to_tactile_proj = nn.Linear(2, 2)

    legacy = Legacy()
    target = ThreeStream()
    incompatible = target.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "tactile_expert.weight",
        "tactile_expert.bias",
        "marker_to_tactile_proj.weight",
        "marker_to_tactile_proj.bias",
    }
    assert torch.equal(target.action.weight, legacy.action.weight)
    assert torch.equal(target.marker_encoder.weight, legacy.marker_encoder.weight)


def test_real_checkpoint_hook_allows_only_absent_refinement_namespace():
    fake = SimpleNamespace(
        config=SimpleNamespace(
            tactile_enabled=True,
            tactile_refinement_enabled=True,
        )
    )
    names = {
        "model.qwenvl_with_expert.tactile_expert.model.layers.0.self_attn.q_proj.weight",
        "model.marker_to_tactile_proj.weight",
        "model.action_out_proj.weight",
    }
    allowed = LingbotVlaV2Policy.checkpoint_allowed_missing_parameters(
        fake,
        names,
        loaded_parameter_names={"model.tactile_encoder.marker_modality_embedding"},
    )
    assert allowed == names - {"model.action_out_proj.weight"}


def test_first_stage_gradient_reaches_only_tactile_modules():
    vlm = nn.Linear(3, 3)
    action = nn.Linear(3, 3)
    marker_to_tactile = nn.Linear(3, 3)
    tactile = nn.Linear(3, 3)
    vlm.requires_grad_(False)
    action.requires_grad_(False)
    context = vlm(torch.ones(2, 3)) + action(torch.ones(2, 3))
    prediction = tactile(marker_to_tactile(torch.ones(2, 3)) + context)
    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in vlm.parameters())
    assert all(parameter.grad is None for parameter in action.parameters())
    assert all(parameter.grad is not None for parameter in tactile.parameters())
    assert all(
        parameter.grad is not None for parameter in marker_to_tactile.parameters()
    )


def test_legacy_mode_remains_disabled_by_default():
    config = TactileRefinementConfig.from_mapping(None)
    assert config.enabled is False
    model_source = MODEL_PATH.read_text(encoding="utf-8")
    assert "and not getattr(" in model_source
