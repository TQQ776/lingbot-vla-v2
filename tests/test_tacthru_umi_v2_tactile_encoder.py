from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lingbotvla/models/vla/lingbot_vla/tactile_encoder.py"
)
SPEC = importlib.util.spec_from_file_location("lingbot_tactile_encoder_test", MODULE_PATH)
TACTILE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TACTILE)


def _params(**overrides):
    params = {
        "history_steps": 2,
        "rgb_backbone": "native_patch_transformer",
        "rgb_backbone_path": None,
        "freeze_rgb_backbone": False,
        "rgb_input_size": 16,
        "rgb_backbone_pretrain_size": 16,
        "rgb_patch_size": 8,
        "rgb_encoder_dim": 16,
        "rgb_encoder_layers": 1,
        "rgb_encoder_heads": 4,
        "rgb_tokens": 3,
        "rgb_mean": [0.5, 0.5, 0.5],
        "rgb_std": [0.25, 0.25, 0.25],
        "marker_count": 4,
        "marker_dim": 2,
        "marker_encoder_dim": 16,
        "marker_encoder_layers": 1,
        "marker_encoder_heads": 4,
        "marker_tokens": 2,
        "marker_include_velocity": True,
        "rgb_dropout_prob": 0.0,
        "marker_dropout_prob": 0.0,
        "all_tactile_dropout_prob": 0.0,
        "gate_init": -4.0,
    }
    params.update(overrides)
    return params


def test_rgb_encoder_returns_fixed_tokens_and_masks_empty_sample():
    encoder = TACTILE.TactileRGBEncoder(_params(), prefix_hidden_dim=12)
    history = torch.rand(2, 2, 3, 16, 16)
    history_mask = torch.tensor([[True, True], [False, False]])

    tokens, mask = encoder(history, history_mask)

    assert tokens.shape == (2, 3, 12)
    assert mask.tolist() == [[True, True, True], [False, False, False]]
    assert torch.count_nonzero(tokens[1]) == 0


def test_marker_encoder_handles_partial_and_fully_invalid_history():
    encoder = TACTILE.TactileMarkerEncoder(_params(), prefix_hidden_dim=12)
    flow = torch.randn(2, 2, 4, 2)
    valid = torch.tensor(
        [
            [[True, True, False, False], [True, True, True, False]],
            [[False, False, False, False], [False, False, False, False]],
        ]
    )
    history_mask = torch.tensor([[True, True], [True, True]])

    tokens, mask = encoder(flow, valid, history_mask)

    assert tokens.shape == (2, 2, 12)
    assert mask.tolist() == [[True, True], [False, False]]
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[1]) == 0


def test_marker_velocity_uses_irregular_timestamps_and_safe_fallback():
    params = _params(history_frequency_hz=10.0)
    encoder = TACTILE.TactileMarkerEncoder(params, prefix_hidden_dim=12)
    flow = torch.zeros(1, 2, 4, 2)
    flow[:, 1, :, 0] = 1.0
    valid = torch.ones(1, 2, 4, dtype=torch.bool)

    irregular = encoder._build_flow_features(
        flow,
        valid,
        torch.tensor([[0.0, 0.5]]),
    )
    fallback = encoder._build_flow_features(flow, valid, None)
    invalid_time = encoder._build_flow_features(
        flow,
        valid,
        torch.tensor([[0.5, 0.5]]),
    )

    # 1 / 0.5 seconds versus the configured 10 Hz fallback.
    assert torch.allclose(irregular[:, 1, :, 2], torch.full((1, 4), 2.0))
    assert torch.allclose(fallback[:, 1, :, 2], torch.full((1, 4), 10.0))
    assert torch.count_nonzero(invalid_time[:, 1, :, 2:]) == 0


def test_rgb_history_shape_is_strict():
    encoder = TACTILE.TactileRGBEncoder(_params(), prefix_hidden_dim=12)
    bad = torch.rand(1, 1, 3, 16, 16)
    try:
        encoder(bad)
    except ValueError as exc:
        assert "Expected K=2" in str(exc)
    else:
        raise AssertionError("history length mismatch must fail")


def test_dinov2_uses_pretrain_grid_and_keeps_tactile_input_size(monkeypatch):
    captured = {}

    class FakeBackbone(torch.nn.Module):
        embed_dim = 384

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    def fake_dinov2_vits14(**kwargs):
        captured.update(kwargs)
        return FakeBackbone()

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("moge.model.dinov2.hub.backbones"):
            return SimpleNamespace(dinov2_vits14=fake_dinov2_vits14)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    params = _params(
        rgb_backbone="dinov2_vits14",
        rgb_backbone_path="/tmp/dinov2_vits14_pretrain.pth",
        rgb_input_size=224,
        rgb_backbone_pretrain_size=518,
        rgb_patch_size=14,
        rgb_encoder_dim=384,
        rgb_encoder_heads=6,
    )

    encoder = TACTILE.TactileRGBEncoder(params, prefix_hidden_dim=12)

    assert captured["img_size"] == 518
    assert encoder.input_size == 224
    assert encoder.backbone_pretrain_size == 518
