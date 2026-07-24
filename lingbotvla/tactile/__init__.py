"""Shared tactile data contracts and transforms for LingBot-VLA.

This package intentionally has no dependency on the model or deployment
stack.  Dataset conversion, training and the real-robot client can therefore
share the exact same marker convention.
"""

from .schema import (
    TACTILE_MARKER_FLOW_KEY,
    TACTILE_MARKER_VALID_KEY,
    TACTILE_RGB_KEY,
    TACTILE_SCHEMA_VERSION,
    TACTILE_TIMESTAMP_KEY,
    TactileKeys,
    get_tactile_keys,
    history_offsets,
)
from .transforms import (
    apply_tactile_rgb_augmentation,
    history_indices_with_repeat_first,
    marker_flow_to_image_size_xy,
    normalize_marker_flow,
    sample_tactile_rgb_augmentation_params,
    sanitize_marker_flow,
)

__all__ = [
    "TACTILE_MARKER_FLOW_KEY",
    "TACTILE_MARKER_VALID_KEY",
    "TACTILE_RGB_KEY",
    "TACTILE_SCHEMA_VERSION",
    "TACTILE_TIMESTAMP_KEY",
    "TactileKeys",
    "apply_tactile_rgb_augmentation",
    "get_tactile_keys",
    "history_indices_with_repeat_first",
    "history_offsets",
    "marker_flow_to_image_size_xy",
    "normalize_marker_flow",
    "sample_tactile_rgb_augmentation_params",
    "sanitize_marker_flow",
]
