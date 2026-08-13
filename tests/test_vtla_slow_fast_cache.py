from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"


def _model_ast():
    return ast.parse(MODEL_PATH.read_text(encoding="utf-8"))


def _method_source(class_name: str, method_name: str) -> str:
    tree = _model_ast()
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.unparse(method)


def _load_qwen_cache_harness():
    tree = _model_ast()
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenvlWithExpertV2Model"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"handle_kv_cache", "_apply_deepstack", "forward"}
    ]
    harness = ast.ClassDef(
        name="QwenCacheHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {"torch": torch, "KVCache": dict}
    exec(compile(module, str(ROOT / "qwen_cache_harness.py"), "exec"), namespace)
    return namespace["QwenCacheHarness"]


def _causal_mask(length: int) -> torch.Tensor:
    return torch.ones(length, length, dtype=torch.bool).tril()


def _attention(q, k, v, mask):
    q = q.squeeze(2)
    k = k.squeeze(2)
    v = v.squeeze(2)
    scores = torch.einsum("bqd,bkd->bqk", q, k) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~mask.squeeze(0), float("-inf"))
    return torch.einsum("bqk,bkd->bqd", scores.softmax(-1), v)


class _HarnessLayer(nn.Module):
    def __init__(self, width: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.q = nn.Parameter(torch.randn(width, width, generator=generator), requires_grad=False)
        self.k = nn.Parameter(torch.randn(width, width, generator=generator), requires_grad=False)
        self.v = nn.Parameter(torch.randn(width, width, generator=generator), requires_grad=False)
        self.out = nn.Parameter(
            torch.randn(width, width, generator=generator), requires_grad=False
        )

    def forward(self, hidden, attention=None, start=None, end=None, compute_kqv=False, **_):
        if compute_kqv:
            return (
                (hidden @ self.q).unsqueeze(2),
                (hidden @ self.k).unsqueeze(2),
                (hidden @ self.v).unsqueeze(2),
            )
        return hidden + attention[:, start:end] @ self.out


class _HarnessStream(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([_HarnessLayer(width, 23), _HarnessLayer(width, 29)])
        self.norm = nn.Identity()


def _configured_harness():
    harness = _load_qwen_cache_harness()()
    width = 8
    harness.qwenvl = type("Qwen", (), {})()
    harness.qwenvl.config = type(
        "Config",
        (),
        {
            "text_config": type(
                "Text", (), {"num_hidden_layers": 2, "num_attention_heads": 1}
            )()
        },
    )()
    harness.qwenvl.model = type("Model", (), {})()
    harness.qwenvl.model.language_model = _HarnessStream(width)
    harness.qwen_expert = type("Expert", (), {})()
    harness.qwen_expert.model = _HarnessStream(width)
    harness.config = type(
        "Config",
        (),
        {
            "qwen_expert_config": type("ExpertConfig", (), {"num_hidden_layers": 2})(),
            "attention_implementation": "eager",
            "final_norm_adanorm": False,
        },
    )()
    harness.apply_mrope = lambda q, k, _: (q, k)
    harness.attention_interface = _attention
    return harness, width


def test_point_marker_contract_is_exactly_192_tokens():
    config = (
        ROOT
        / "configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker_point192_history4.yaml"
    ).read_text(encoding="utf-8")
    assert "marker_history_length: 4" in config
    assert "num_markers: 48" in config
    assert "mode: point_spatiotemporal" in config
    assert 4 * 48 == 192


def test_production_qwen_slow_continuation_matches_full_causal_output_and_kv():
    harness, width = _configured_harness()
    torch.manual_seed(31)
    slow = torch.randn(1, 8, width)
    dynamic = torch.randn(1, 6, width)
    full = torch.cat([slow, dynamic], dim=1)
    full_out, full_cache, _ = harness.forward(
        attention_mask=_causal_mask(14).unsqueeze(0),
        position_ids=torch.zeros(3, 1, 14, dtype=torch.long),
        inputs_embeds=[full, None],
        use_cache=True,
        fill_kv_cache=True,
    )

    _, slow_cache, _ = harness.forward(
        attention_mask=_causal_mask(8).unsqueeze(0),
        position_ids=torch.zeros(3, 1, 8, dtype=torch.long),
        inputs_embeds=[slow, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    slow_snapshot = {
        layer: {
            name: tensor.clone() for name, tensor in values.items()
        }
        for layer, values in slow_cache.items()
    }
    working = {
        layer: dict(values) for layer, values in slow_cache.items()
    }
    split_out, split_cache, _ = harness.forward(
        attention_mask=_causal_mask(14)[-6:, :].unsqueeze(0),
        position_ids=torch.zeros(3, 1, 6, dtype=torch.long),
        past_key_values=working,
        inputs_embeds=[dynamic, None],
        use_cache=True,
        fill_kv_cache=False,
        append_kv_cache=True,
    )

    torch.testing.assert_close(split_out[0], full_out[0][:, -6:], atol=1e-5, rtol=1e-5)
    for layer in full_cache:
        torch.testing.assert_close(
            split_cache[layer]["key_states"], full_cache[layer]["key_states"]
        )
        torch.testing.assert_close(
            split_cache[layer]["value_states"], full_cache[layer]["value_states"]
        )
        torch.testing.assert_close(
            slow_cache[layer]["key_states"], slow_snapshot[layer]["key_states"]
        )
        torch.testing.assert_close(
            slow_cache[layer]["value_states"], slow_snapshot[layer]["value_states"]
        )
        assert slow_cache[layer]["key_states"].shape[1] == 8


def test_repeated_fast_replans_restart_from_slow_cache_instead_of_growing_it():
    harness, width = _configured_harness()
    slow = torch.randn(1, 8, width)
    _, slow_cache, _ = harness.forward(
        attention_mask=_causal_mask(8).unsqueeze(0),
        position_ids=torch.zeros(3, 1, 8, dtype=torch.long),
        inputs_embeds=[slow, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    outputs = []
    for seed in (41, 43):
        dynamic = torch.randn(1, 6, width, generator=torch.Generator().manual_seed(seed))
        working = {layer: dict(values) for layer, values in slow_cache.items()}
        _, full_cache, _ = harness.forward(
            attention_mask=_causal_mask(14)[-6:, :].unsqueeze(0),
            position_ids=torch.zeros(3, 1, 6, dtype=torch.long),
            past_key_values=working,
            inputs_embeds=[dynamic, None],
            use_cache=True,
            fill_kv_cache=False,
            append_kv_cache=True,
        )
        outputs.append(full_cache)
    assert all(values["key_states"].shape[1] == 8 for values in slow_cache.values())
    assert all(values["key_states"].shape[1] == 14 for values in outputs[0].values())
    assert all(values["key_states"].shape[1] == 14 for values in outputs[1].values())
    assert not torch.equal(outputs[0][0]["key_states"], outputs[1][0]["key_states"])


def test_full_position_ids_are_rebuilt_then_sliced_for_dynamic_tail():
    method = _method_source("FlowMatchingV2", "extend_slow_cache")
    assert "full_ids = torch.cat([slow_cache.input_ids, tail_input_ids]" in method
    assert "full_pad = torch.cat([slow_cache.pad_masks, tail_pad_masks]" in method
    assert "build_prefix_position_ids(full_ids, full_pad.long()" in method
    assert "position_ids=full_position_ids[:, :, -tail_length:]" in method


def test_dynamic_attention_is_full_causal_mask_with_only_tail_query_rows():
    method = _method_source("FlowMatchingV2", "extend_slow_cache")
    assert "full_attention = make_att_2d_masks(full_pad, full_att)" in method
    assert "continuation_attention = full_attention[:, -tail_length:, :]" in method


def test_action_state_noise_and_time_remain_suffix_only():
    prefix = _method_source("FlowMatchingV2", "_embed_prefix_vtla")
    fast = _method_source("FlowMatchingV2", "sample_actions_fast")
    velocity = _method_source("FlowMatchingV2", "predict_velocity")
    prefix_args = {
        argument.arg
        for argument in (
            *ast.parse(prefix).body[0].args.args,
            *ast.parse(prefix).body[0].args.kwonlyargs,
        )
    }
    assert "state" not in prefix_args
    assert "noise" not in prefix_args
    assert "timestep" not in prefix_args
    assert "self.predict_velocity" in fast
    assert "self.embed_suffix(state, x_t, timestep)" in velocity


def test_slow_cache_and_fast_profile_contracts_are_explicit():
    source = MODEL_PATH.read_text(encoding="utf-8")
    for key in (
        "scene_tacrgb_vit_ms",
        "slow_vlm_ms",
        "slow_cache_build_ms",
        "marker_mlp_ms",
        "marker_query_continuation_ms",
        "task_query_ms",
        "fm_sampling_ms",
        "fast_replan_total_ms",
    ):
        assert key in source
