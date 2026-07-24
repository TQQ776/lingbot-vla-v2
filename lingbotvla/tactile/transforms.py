"""Pure tactile transforms shared by offline and online paths."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - converter-only environments may omit torch
    torch = None


def _validate_marker_shape(value: Any) -> None:
    if getattr(value, "ndim", 0) < 1 or int(value.shape[-1]) != 2:
        raise ValueError(
            "marker flow must have last dimension 2 for (dx, dy), "
            f"got shape {getattr(value, 'shape', None)}"
        )


def normalize_marker_flow(delta_xy: Any, image_width: int, image_height: int):
    """Normalize pixel marker displacement into image-relative ``[-2, 2]`` units.

    X and Y deliberately use different scales.  For an ML48 640x480 stream,
    this is ``dx / 640 * 2`` and ``dy / 480 * 2``.  The return type follows the
    input type (NumPy array or torch tensor).
    """

    image_width = int(image_width)
    image_height = int(image_height)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image dimensions must be positive, got {image_width}x{image_height}"
        )
    _validate_marker_shape(delta_xy)

    if torch is not None and isinstance(delta_xy, torch.Tensor):
        value = delta_xy.to(dtype=torch.float32)
        scale = value.new_tensor((2.0 / image_width, 2.0 / image_height))
        return value * scale

    value = np.asarray(delta_xy, dtype=np.float32)
    scale = np.asarray((2.0 / image_width, 2.0 / image_height), dtype=np.float32)
    return value * scale


def marker_flow_to_image_size_xy(
    marker_flow: Any,
    *,
    input_space: str,
    image_width: int,
    image_height: int,
    legacy_denominator: float = 400.0,
):
    """Convert supported marker conventions to canonical image-size XY units.

    ``normalized`` is already canonical and is returned unchanged. ``pixel``
    is normalized with the physical tracker width/height. The legacy real-env
    convention divided both axes by 400; it is corrected by multiplying X by
    ``400 / width`` and Y by ``400 / height``.
    """

    input_space = str(input_space)
    if input_space == "pixel":
        return normalize_marker_flow(marker_flow, image_width, image_height)
    _validate_marker_shape(marker_flow)
    if input_space == "normalized":
        if torch is not None and isinstance(marker_flow, torch.Tensor):
            return marker_flow.to(dtype=torch.float32)
        return np.asarray(marker_flow, dtype=np.float32)
    if input_space != "legacy_400_normalized":
        raise ValueError(
            "input_space must be normalized, pixel, or legacy_400_normalized, "
            f"got {input_space!r}"
        )
    if image_width <= 0 or image_height <= 0 or legacy_denominator <= 0:
        raise ValueError("marker dimensions and legacy_denominator must be positive")
    if torch is not None and isinstance(marker_flow, torch.Tensor):
        value = marker_flow.to(dtype=torch.float32)
        scale = value.new_tensor(
            (legacy_denominator / image_width, legacy_denominator / image_height)
        )
        return value * scale
    value = np.asarray(marker_flow, dtype=np.float32)
    scale = np.asarray(
        (legacy_denominator / image_width, legacy_denominator / image_height),
        dtype=np.float32,
    )
    return value * scale


def sanitize_marker_flow(marker_flow: Any, valid_mask: Any | None = None):
    """Replace invalid marker values by zero and return a per-marker validity mask."""

    _validate_marker_shape(marker_flow)
    if torch is not None and isinstance(marker_flow, torch.Tensor):
        flow = marker_flow.to(dtype=torch.float32)
        finite = torch.isfinite(flow).all(dim=-1)
        if valid_mask is not None:
            finite = finite & torch.as_tensor(valid_mask, device=flow.device, dtype=torch.bool)
        return torch.where(finite.unsqueeze(-1), flow, torch.zeros_like(flow)), finite

    flow = np.asarray(marker_flow, dtype=np.float32)
    finite = np.isfinite(flow).all(axis=-1)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=np.bool_)
    return np.where(finite[..., None], flow, 0.0).astype(np.float32, copy=False), finite


def history_indices_with_repeat_first(
    current_index: int,
    episode_start: int,
    history_steps: int,
    history_stride: int = 1,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Build causal indices without ever crossing an episode boundary.

    Missing history at an episode start repeats the first frame while returning
    ``False`` in the corresponding mask.  The current frame is always last.
    """

    from .schema import history_offsets

    current_index = int(current_index)
    episode_start = int(episode_start)
    if current_index < episode_start:
        raise ValueError(
            f"current_index {current_index} precedes episode_start {episode_start}"
        )
    indices = []
    mask = []
    for offset in history_offsets(history_steps, history_stride):
        requested = current_index + offset
        is_valid = requested >= episode_start
        indices.append(max(episode_start, requested))
        mask.append(is_valid)
    return tuple(indices), tuple(mask)


def sample_tactile_rgb_augmentation_params(reference):
    """Sample weak, spatially invariant tactile-RGB augmentation parameters."""

    if torch is None or not isinstance(reference, torch.Tensor):
        raise TypeError("tactile RGB augmentation expects a torch.Tensor")
    device = reference.device
    return {
        "brightness": 0.9 + torch.rand((), device=device) * 0.2,
        "contrast": 0.9 + torch.rand((), device=device) * 0.2,
        "noise_std": torch.rand((), device=device) * 0.01,
    }


def apply_tactile_rgb_augmentation(history, params):
    """Apply one weak color/noise transform to every frame in a tactile history.

    No crop, flip or rotation is performed because tactile pixel location has a
    fixed physical meaning.  Input may be uint8 [0,255] or floating point
    [0,1]; output is float32 [0,1].
    """

    if torch is None or not isinstance(history, torch.Tensor):
        raise TypeError("tactile RGB augmentation expects a torch.Tensor")
    if history.ndim not in {3, 4}:
        raise ValueError(
            f"expected tactile RGB (C,H,W) or (K,C,H,W), got {tuple(history.shape)}"
        )
    frames = history.unsqueeze(0) if history.ndim == 3 else history
    value = frames.to(dtype=torch.float32)
    if history.dtype == torch.uint8 or (value.numel() and float(value.max()) > 1.0):
        value = value / 255.0

    brightness = torch.as_tensor(params["brightness"], device=value.device, dtype=value.dtype)
    contrast = torch.as_tensor(params["contrast"], device=value.device, dtype=value.dtype)
    noise_std = torch.as_tensor(params.get("noise_std", 0.0), device=value.device, dtype=value.dtype)
    value = value * brightness
    mean = value.mean(dim=(-3, -2, -1), keepdim=True)
    value = (value - mean) * contrast + mean
    if bool(noise_std > 0):
        value = value + torch.randn_like(value) * noise_std
    value = value.clamp_(0.0, 1.0)
    return value.squeeze(0) if history.ndim == 3 else value
