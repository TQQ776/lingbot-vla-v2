from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lingbotvla/models/vla/lingbot_vla/tactile_encoder.py"
)
SPEC = importlib.util.spec_from_file_location("lingbot_tactile_ablation_test", MODULE_PATH)
TACTILE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TACTILE)


def _params(**overrides):
    params = {
        "rgb_dropout_prob": 0.0,
        "marker_dropout_prob": 0.0,
        "all_tactile_dropout_prob": 0.0,
        "gate_init": 0.0,
    }
    params.update(overrides)
    return params


def test_force_masks_are_independent_per_modality_and_sample():
    fusion = TACTILE.GatedTactileFusion(8, True, True, _params())
    fusion.eval()
    rgb = torch.ones(2, 3, 8)
    marker = torch.ones(2, 2, 8)
    rgb_mask = torch.ones(2, 3, dtype=torch.bool)
    marker_mask = torch.ones(2, 2, dtype=torch.bool)
    present = torch.ones(2, dtype=torch.bool)

    _, mask, _ = fusion(
        rgb_tokens=rgb,
        rgb_token_mask=rgb_mask,
        rgb_present=present,
        marker_tokens=marker,
        marker_token_mask=marker_mask,
        marker_present=present,
        force_mask_rgb=torch.tensor([True, False]),
        force_mask_marker=torch.tensor([False, True]),
    )

    assert mask[:, :3].tolist() == [[False, False, False], [True, True, True]]
    assert mask[:, 3:].tolist() == [[True, True], [False, False]]


def test_modality_dropout_drops_the_whole_sample_window(monkeypatch):
    fusion = TACTILE.GatedTactileFusion(
        8,
        True,
        False,
        _params(rgb_dropout_prob=0.5),
    )
    fusion.train()

    def fake_rand(batch, device=None):
        assert batch == 2
        return torch.tensor([0.25, 0.75], device=device)

    monkeypatch.setattr(TACTILE.torch, "rand", fake_rand)
    _, mask, _ = fusion(
        rgb_tokens=torch.ones(2, 3, 8),
        rgb_token_mask=torch.ones(2, 3, dtype=torch.bool),
        rgb_present=torch.ones(2, dtype=torch.bool),
    )

    assert mask.tolist() == [[False, False, False], [True, True, True]]
