import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from lingbotvla.tactile.schema import history_offsets
from lingbotvla.tactile.transforms import (
    history_indices_with_repeat_first,
    marker_flow_to_image_size_xy,
    normalize_marker_flow,
    sanitize_marker_flow,
)


TRANSFORM_PATH = (
    Path(__file__).resolve().parents[1] / "lingbotvla/data/vla_data/transform.py"
)


def test_marker_normalization_uses_independent_xy_image_size() -> None:
    pixels = np.asarray([[[64.0, 48.0], [-32.0, -24.0]]], dtype=np.float32)
    normalized = normalize_marker_flow(pixels, image_width=640, image_height=480)
    np.testing.assert_allclose(
        normalized,
        [[[0.2, 0.2], [-0.1, -0.1]]],
        atol=1e-7,
    )


def test_already_normalized_marker_is_not_normalized_twice() -> None:
    source = torch.tensor([[[0.02, -0.03]]], dtype=torch.float32)
    output = marker_flow_to_image_size_xy(
        source,
        input_space="normalized",
        image_width=640,
        image_height=480,
    )
    torch.testing.assert_close(output, source)


def test_legacy_uniform_400_marker_is_corrected_to_image_size_xy() -> None:
    legacy = np.asarray([[[0.2, 0.2]]], dtype=np.float32)
    corrected = marker_flow_to_image_size_xy(
        legacy,
        input_space="legacy_400_normalized",
        image_width=640,
        image_height=480,
    )
    np.testing.assert_allclose(corrected, [[[0.125, 1.0 / 6.0]]], atol=1e-7)


def test_sanitize_marker_flow_zeroes_nonfinite_points() -> None:
    flow = np.asarray([[[0.1, 0.2], [np.nan, 0.3]]], dtype=np.float32)
    sanitized, valid = sanitize_marker_flow(flow)
    assert valid.tolist() == [[True, False]]
    np.testing.assert_array_equal(sanitized[0, 1], [0.0, 0.0])


def test_history_offsets_and_repeat_first_never_cross_episode() -> None:
    assert history_offsets(4, 2) == (-6, -4, -2, 0)
    indices, mask = history_indices_with_repeat_first(
        current_index=101,
        episode_start=100,
        history_steps=4,
        history_stride=1,
    )
    assert indices == (100, 100, 100, 101)
    assert mask == (False, False, True, True)
    assert min(indices) == 100


def test_prepare_tactile_histories_preserves_masks_and_output_contract(monkeypatch) -> None:
    # Load the leaf module directly so this unit test does not require the
    # optional training-only torchdata package imported by data.__init__.
    spec = importlib.util.spec_from_file_location("lingbot_tactile_data_transform", TRANSFORM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("einops", types.ModuleType("einops"))
    spec.loader.exec_module(module)
    prepare_tactile_marker_history = module.prepare_tactile_marker_history
    prepare_tactile_rgb_history = module.prepare_tactile_rgb_history

    rgb = torch.full((4, 3, 8, 8), 255, dtype=torch.uint8)
    rgb_history, rgb_mask, rgb_present = prepare_tactile_rgb_history(
        rgb,
        torch.tensor([True, True, False, False]),
        train=False,
    )
    assert rgb_history.dtype == torch.float32
    assert rgb_history.shape == (4, 3, 8, 8)
    assert rgb_mask.tolist() == [False, False, True, True]
    assert bool(rgb_present)
    assert float(rgb_history[:2].abs().max()) == 0.0
    assert float(rgb_history[2:].min()) == pytest.approx(1.0)

    marker = torch.ones(4, 48, 2)
    valid = torch.ones(4, 48, dtype=torch.bool)
    valid[-1, 3] = False
    marker_history, marker_valid, marker_mask, marker_present = (
        prepare_tactile_marker_history(
            marker,
            valid,
            torch.tensor([True, True, False, False]),
        )
    )
    assert marker_history.shape == (4, 48, 2)
    assert marker_valid.shape == (4, 48)
    assert marker_mask.tolist() == [False, False, True, True]
    assert not bool(marker_valid[-1, 3])
    assert bool(marker_present)
    torch.testing.assert_close(marker_history[-1, 3], torch.zeros(2))
