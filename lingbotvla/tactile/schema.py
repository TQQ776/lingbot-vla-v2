"""Canonical tactile field names and temporal-layout helpers."""

from __future__ import annotations

from dataclasses import dataclass


TACTILE_SCHEMA_VERSION = 1
TACTILE_RGB_KEY = "observation.images.tactile_left"
TACTILE_MARKER_FLOW_KEY = "observation.tactile.marker_flow_left"
TACTILE_MARKER_VALID_KEY = "observation.tactile.marker_valid_left"
TACTILE_TIMESTAMP_KEY = "observation.tactile.timestamp"


@dataclass(frozen=True)
class TactileKeys:
    """Source and canonical keys for one tactile side."""

    side: str
    source_rgb: str
    source_marker_flow: str
    rgb: str
    marker_flow: str
    marker_valid: str
    timestamp: str = TACTILE_TIMESTAMP_KEY


def get_tactile_keys(side: str = "left") -> TactileKeys:
    """Return the canonical contract for ``left`` or ``right`` tactile data."""

    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tactile side {side!r}; expected 'left' or 'right'")
    short_side = "l" if normalized_side == "left" else "r"
    return TactileKeys(
        side=normalized_side,
        source_rgb=f"tacthru_{short_side}_rgb",
        source_marker_flow=f"tacthru_{short_side}_marker",
        rgb=f"observation.images.tactile_{normalized_side}",
        marker_flow=f"observation.tactile.marker_flow_{normalized_side}",
        marker_valid=f"observation.tactile.marker_valid_{normalized_side}",
    )


def history_offsets(history_steps: int, history_stride: int = 1) -> tuple[int, ...]:
    """Return oldest-to-current integer offsets for a causal history window."""

    history_steps = int(history_steps)
    history_stride = int(history_stride)
    if history_steps < 1:
        raise ValueError(f"history_steps must be >= 1, got {history_steps}")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}")
    return tuple(
        -history_stride * step
        for step in range(history_steps - 1, -1, -1)
    )
