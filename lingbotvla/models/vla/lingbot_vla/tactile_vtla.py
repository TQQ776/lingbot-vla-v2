"""Minimal TacThru token encoders for LingBot-VLA v2.

This module intentionally contains no action or Flow Matching code.  It turns
TacThru RGB embeddings and marker coordinates into fixed-length prefix tokens
that can be appended to the existing vision-language prefix.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn

from lingbotvla.tactile_contact import build_marker_region_mapping_numpy

from .marker_contact import (
    MarkerContactGateConfig,
    contact_state_is_unknown,
    contact_state_is_visible,
)


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def migrate_legacy_tactile_config(
    config: Mapping[str, Any] | None,
    *,
    allow_legacy_marker_reinit: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve the pre-history marker contract for explicit deployment migration.

    The first TacThru checkpoint used one four-channel global marker token per
    sensor. Current deployment accepts an eight-frame ``[dx, dy]`` history.
    Converting the config is only safe together with the existing checkpoint
    loader's marker-only reinitialization, so the conversion remains opt-in.
    """

    values = dict(config or {})
    if not _read_bool(values.get("enabled"), False) or not _read_bool(
        values.get("use_markers"), True
    ):
        return values, None

    input_features = int(values.get("marker_input_features", 2))
    legacy_keys = {
        "marker_flow_keys",
        "marker_positions_keys",
        "marker_reference_keys",
    }
    is_legacy = input_features == 4 or bool(legacy_keys & values.keys())
    if not is_legacy or not allow_legacy_marker_reinit:
        return values, None

    removed_keys = sorted(legacy_keys & values.keys())
    for key in legacy_keys:
        values.pop(key, None)
    values.pop("marker_tokens_per_sensor", None)
    if len(values.get("marker_mean", ())) == 4:
        values.pop("marker_mean")
    if len(values.get("marker_std", ())) == 4:
        values.pop("marker_std")

    sensor_names = tuple(values.get("sensor_names") or ("left",))
    values.update(
        marker_input_features=2,
        marker_feature_mode="displacement_history",
        marker_history_length=8,
        marker_sample_hz=30.0,
        marker_tokenization={
            "mode": "global",
            "num_regions": 1,
            "region_layout": "1x1",
            "aggregation": "mean_max",
            "include_reference_xy": False,
        },
        marker_position_encoding={
            "temporal_type": "learned",
            "spatial_type": "none",
            "use_real_time": True,
            "combination": "additive",
        },
        marker_displacement_keys=[
            f"observation.tactile.marker_displacement_{name}"
            for name in sensor_names
        ],
        marker_valid_mask_keys=[
            f"observation.tactile.marker_valid_{name}" for name in sensor_names
        ],
    )
    report = {
        "kind": "legacy_four_channel_global_to_displacement_history",
        "marker_input_features": [input_features, 2],
        "marker_tokens_per_sensor": [
            config.get("marker_tokens_per_sensor") if config else None,
            8,
        ],
        "removed_keys": removed_keys,
        "requires_marker_module_reinitialization": True,
    }
    return values, report


def _as_float_tuple(value: Sequence[float], *, name: str, length: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    if not torch.isfinite(torch.tensor(result, dtype=torch.float32)).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def load_marker_statistics(path: str | Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load train-split marker statistics from a JSON file.

    Accepted files contain either top-level ``marker_mean``/``marker_std`` or
    a nested ``marker`` object with ``mean``/``std``.  Two values are required
    in ``[dx, dy]`` order.
    """

    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Marker statistics file does not exist: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    marker = payload.get("marker", {}) if isinstance(payload, Mapping) else {}
    mean = payload.get("marker_mean", marker.get("mean"))
    std = payload.get("marker_std", marker.get("std"))
    if mean is None or std is None:
        raise ValueError(
            f"{stats_path} must contain marker_mean/marker_std or marker.mean/marker.std"
        )
    if len(mean) != 2 or len(std) != 2:
        raise ValueError(
            "Expected marker statistics for [dx, dy], got "
            f"mean={len(mean)} channels and std={len(std)} channels. "
            "Please recompute tactile marker norm statistics."
        )
    mean_tuple = _as_float_tuple(mean, name="marker_mean", length=2)
    std_tuple = _as_float_tuple(std, name="marker_std", length=2)
    if any(value <= 0 for value in std_tuple):
        raise ValueError("marker_std values must be positive")
    return mean_tuple, std_tuple


@dataclass(frozen=True)
class MarkerTokenizationConfig:
    mode: str = "global"
    num_regions: int = 1
    region_layout: str = "1x1"
    aggregation: str = "mean_max"
    include_reference_xy: bool = False
    point_hidden_dim: int = 128
    region_hidden_dim: int = 512

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        legacy_hidden_dim: int,
    ) -> "MarkerTokenizationConfig":
        values = dict(value or {})
        mode = str(values.get("mode", "global"))
        modes = {"global", "regional", "point_spatiotemporal"}
        if mode not in modes:
            raise ValueError(
                "marker_tokenization.mode must be global, regional, or "
                "point_spatiotemporal"
            )
        default_regions = {"global": 1, "regional": 4}.get(mode, 48)
        num_regions = int(values.get("num_regions", default_regions))
        default_layout = {
            "global": "1x1",
            "regional": "2x2",
            "point_spatiotemporal": "points",
        }[mode]
        layout = str(values.get("region_layout", default_layout))
        if mode == "global" and (num_regions != 1 or layout != "1x1"):
            raise ValueError("Global marker tokenization requires num_regions=1 and region_layout=1x1")
        if mode == "regional" and (num_regions != 4 or layout != "2x2"):
            raise ValueError("Regional marker tokenization currently requires four 2x2 regions")
        if mode == "point_spatiotemporal" and layout != "points":
            raise ValueError(
                "Point spatiotemporal tokenization requires region_layout=points"
            )
        aggregation = str(values.get("aggregation", "mean_max"))
        if aggregation not in {"mean", "max", "mean_max"}:
            raise ValueError("marker aggregation must be mean, max, or mean_max")
        point_hidden_dim = int(values.get("point_hidden_dim", 128))
        region_hidden_dim = int(values.get("region_hidden_dim", legacy_hidden_dim))
        if point_hidden_dim <= 0 or region_hidden_dim <= 0:
            raise ValueError("Marker point/region hidden dimensions must be positive")
        return cls(
            mode=mode,
            num_regions=num_regions,
            region_layout=layout,
            aggregation=aggregation,
            include_reference_xy=_read_bool(
                values.get("include_reference_xy"),
                mode in {"regional", "point_spatiotemporal"},
            ),
            point_hidden_dim=point_hidden_dim,
            region_hidden_dim=region_hidden_dim,
        )


@dataclass(frozen=True)
class MarkerPositionEncodingConfig:
    temporal_type: str = "learned"
    spatial_type: str = "none"
    use_real_time: bool = True
    combination: str = "additive"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        legacy_temporal_embedding: bool,
    ) -> "MarkerPositionEncodingConfig":
        values = dict(value or {})
        temporal_type = str(
            values.get(
                "temporal_type",
                "learned" if legacy_temporal_embedding else "none",
            )
        )
        spatial_type = str(values.get("spatial_type", "none"))
        for name, setting in (
            ("temporal_type", temporal_type),
            ("spatial_type", spatial_type),
        ):
            if setting not in {"none", "learned", "sincos"}:
                raise ValueError(f"marker_position_encoding.{name}={setting!r} is invalid")
        combination = str(values.get("combination", "additive"))
        if combination != "additive":
            raise ValueError("Only additive marker position encoding is supported")
        return cls(
            temporal_type=temporal_type,
            spatial_type=spatial_type,
            use_real_time=_read_bool(values.get("use_real_time"), True),
            combination=combination,
        )


@dataclass(frozen=True)
class MarkerAblationConfig:
    zero_marker_content: bool = False
    shuffle_temporal_order: bool = False
    shuffle_region_order: bool = False
    disable_spatial_content: bool = False
    shuffle_seed: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MarkerAblationConfig":
        values = dict(value or {})
        return cls(
            zero_marker_content=_read_bool(values.get("zero_marker_content"), False),
            shuffle_temporal_order=_read_bool(
                values.get("shuffle_temporal_order"), False
            ),
            shuffle_region_order=_read_bool(values.get("shuffle_region_order"), False),
            disable_spatial_content=_read_bool(
                values.get("disable_spatial_content"), False
            ),
            shuffle_seed=int(values.get("shuffle_seed", 0)),
        )


def _normalize_reference_xy(reference_xy: Tensor) -> Tensor:
    if reference_xy.ndim != 3 or reference_xy.shape[-1] != 2:
        raise ValueError("marker reference coordinates must have shape [S,N,2]")
    if not torch.isfinite(reference_xy).all():
        raise ValueError("marker reference coordinates must be finite")
    low = reference_xy.amin(dim=1, keepdim=True)
    high = reference_xy.amax(dim=1, keepdim=True)
    span = high - low
    if (span <= 0).any():
        raise ValueError("marker reference coordinates must span both x and y")
    return (2.0 * (reference_xy - low) / span - 1.0).to(torch.float32)


def load_marker_reference_xy(
    path: str | Path,
    *,
    sensor_names: Sequence[str],
    num_markers: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    reference_path = Path(path).expanduser()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Marker reference file does not exist: {reference_path}")
    if reference_path.suffix.lower() == ".npy":
        import numpy as np

        raw = np.load(reference_path)
    else:
        with reference_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload_names = payload.get("sensor_names")
            raw = payload.get("reference_xy", payload.get("marker_reference_xy"))
            if payload_names is not None and list(payload_names) != list(sensor_names):
                raise ValueError(
                    f"Marker reference sensor order {payload_names} does not match {list(sensor_names)}"
                )
        else:
            raw = payload
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    expected = (len(sensor_names), num_markers, 2)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Marker reference must have shape {expected}, got {tuple(tensor.shape)}")
    tensor = _normalize_reference_xy(tensor)
    return tuple(
        tuple((float(point[0]), float(point[1])) for point in sensor)
        for sensor in tensor
    )


def build_marker_region_mapping(
    reference_xy: Tensor | Sequence[Sequence[Sequence[float]]],
    *,
    num_regions: int,
    region_layout: str,
) -> tuple[Tensor, Tensor]:
    """Return deterministic [S,N] region ids and [S,R,2] region centers."""

    reference = torch.as_tensor(reference_xy, dtype=torch.float32)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
    reference = _normalize_reference_xy(reference)
    region_ids, centers = build_marker_region_mapping_numpy(
        reference.detach().cpu().numpy(),
        num_regions=num_regions,
        region_layout=region_layout,
    )
    return torch.from_numpy(region_ids), torch.from_numpy(centers)


@dataclass(frozen=True)
class TactileVTLAConfig:
    """Validated, serializable configuration for the configurable VTLA extension."""

    enabled: bool = False
    num_sensors: int = 2
    num_markers: int = 48
    use_rgb: bool = True
    use_markers: bool = True
    share_vision_encoder: bool = True
    freeze_vision_encoder: bool = True
    marker_history_length: int = 8
    marker_sample_hz: float = 30.0
    marker_input_features: int = 2
    marker_feature_mode: str = "displacement_history"
    marker_temporal_embedding: bool = True
    marker_hidden_dim: int = 512
    marker_tokenization: MarkerTokenizationConfig = field(
        default_factory=MarkerTokenizationConfig
    )
    marker_position_encoding: MarkerPositionEncodingConfig = field(
        default_factory=MarkerPositionEncodingConfig
    )
    marker_contact_gate: MarkerContactGateConfig = field(
        default_factory=MarkerContactGateConfig
    )
    gate_tactile_rgb: bool = False
    marker_ablation: MarkerAblationConfig = field(default_factory=MarkerAblationConfig)
    marker_reference_xy_path: str | None = None
    marker_reference_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    add_modality_embedding: bool = True
    add_sensor_side_embedding: bool = True
    marker_stats_path: str | None = None
    marker_mean: tuple[float, ...] = (0.0, 0.0)
    marker_std: tuple[float, ...] = (1.0, 1.0)
    require_marker_stats: bool = False
    sensor_names: tuple[str, ...] = ("left", "right")
    rgb_keys: tuple[str, ...] = (
        "observation.images.tacthru_l_rgb",
        "observation.images.tacthru_r_rgb",
    )
    marker_displacement_keys: tuple[str, ...] = (
        "observation.tactile.marker_displacement_left",
        "observation.tactile.marker_displacement_right",
    )
    marker_valid_mask_keys: tuple[str, ...] = (
        "observation.tactile.marker_valid_left",
        "observation.tactile.marker_valid_right",
    )
    debug_shapes: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "TactileVTLAConfig":
        """Build and validate settings from the nested ``tactile`` mapping."""

        values = dict(config or {})
        enabled = _read_bool(values.get("enabled"), False)
        num_sensors = int(values.get("num_sensors", 2))
        num_markers = int(values.get("num_markers", 48))
        use_rgb = _read_bool(values.get("use_rgb"), True)
        use_markers = _read_bool(values.get("use_markers"), True)
        marker_history_length = int(values.get("marker_history_length", 8))
        marker_sample_hz = float(values.get("marker_sample_hz", 30.0))
        marker_input_features = int(values.get("marker_input_features", 2))
        marker_feature_mode = str(
            values.get("marker_feature_mode", "displacement_history")
        )
        marker_hidden_dim = int(values.get("marker_hidden_dim", 512))

        if num_sensors <= 0:
            raise ValueError("tactile.num_sensors must be positive")
        if num_markers <= 0:
            raise ValueError("tactile.num_markers must be positive")
        if marker_history_length <= 0:
            raise ValueError("tactile.marker_history_length must be positive")
        if marker_sample_hz <= 0:
            raise ValueError("tactile.marker_sample_hz must be positive")
        if enabled and not (use_rgb or use_markers):
            raise ValueError("tactile.enabled=true requires use_rgb or use_markers")
        if not _read_bool(values.get("share_vision_encoder"), True):
            raise ValueError("The minimal VTLA implementation requires share_vision_encoder=true")
        if marker_feature_mode != "displacement_history":
            raise ValueError(
                "The VTLA marker path requires "
                "marker_feature_mode='displacement_history'"
            )
        if marker_input_features != 2:
            raise ValueError(
                "marker_input_features must be 2 ([dx, dy]); explicit marker velocity "
                "is not supported by displacement_history mode"
            )
        if marker_hidden_dim <= 0:
            raise ValueError("tactile.marker_hidden_dim must be positive")

        def _keys(name: str, defaults: Sequence[str]) -> tuple[str, ...]:
            raw = values.get(name, defaults[:num_sensors])
            result = tuple(str(item) for item in raw)
            if len(result) != num_sensors:
                raise ValueError(
                    f"tactile.{name} must contain num_sensors={num_sensors} entries, "
                    f"got {len(result)}"
                )
            if len(set(result)) != len(result):
                raise ValueError(f"tactile.{name} entries must be unique")
            return result

        default_names = tuple("left" if i == 0 else "right" if i == 1 else f"sensor_{i}" for i in range(num_sensors))
        sensor_names = tuple(str(item) for item in values.get("sensor_names", default_names))
        if len(sensor_names) != num_sensors or len(set(sensor_names)) != num_sensors:
            raise ValueError("tactile.sensor_names must contain unique names for every sensor")

        stats_path = values.get("marker_stats_path")
        require_stats = _read_bool(values.get("require_marker_stats"), False)
        mean_value = values.get("marker_mean")
        std_value = values.get("marker_std")
        if mean_value is not None and std_value is not None:
            # Saved Hugging Face configs carry both the original provenance
            # path and resolved values, so inference remains portable even
            # when the training workspace is not mounted or the source changes.
            marker_mean = _as_float_tuple(
                mean_value, name="marker_mean", length=2
            )
            marker_std = _as_float_tuple(std_value, name="marker_std", length=2)
        elif stats_path is not None and Path(stats_path).expanduser().is_file():
            marker_mean, marker_std = load_marker_statistics(stats_path)
        elif stats_path is not None:
            raise FileNotFoundError(
                f"Marker statistics file does not exist: {Path(stats_path).expanduser()}"
            )
        else:
            if enabled and use_markers and require_stats and (mean_value is None or std_value is None):
                raise ValueError(
                    "Marker training requires tactile.marker_stats_path or explicit marker_mean/marker_std"
                )
            marker_mean = _as_float_tuple(
                mean_value or (0.0, 0.0),
                name="marker_mean",
                length=2,
            )
            marker_std = _as_float_tuple(
                std_value or (1.0, 1.0),
                name="marker_std",
                length=2,
            )
        if any(value <= 0 for value in marker_std):
            raise ValueError("marker_std values must be positive")

        tokenization = MarkerTokenizationConfig.from_mapping(
            values.get("marker_tokenization"),
            legacy_hidden_dim=marker_hidden_dim,
        )
        if (
            tokenization.mode == "point_spatiotemporal"
            and tokenization.num_regions != num_markers
        ):
            raise ValueError(
                "Point spatiotemporal tokenization requires num_regions to equal "
                f"num_markers={num_markers}"
            )
        legacy_token_count = values.get("marker_tokens_per_sensor")
        expected_token_count = marker_history_length * tokenization.num_regions
        if (
            legacy_token_count is not None
            and int(legacy_token_count) != expected_token_count
        ):
            raise ValueError(
                "marker_tokens_per_sensor is deprecated; the selected marker "
                "tokenization emits exactly marker_history_length * num_regions="
                f"{expected_token_count} tokens per sensor. Remove the old field."
            )
        legacy_temporal = _read_bool(values.get("marker_temporal_embedding"), True)
        position_encoding = MarkerPositionEncodingConfig.from_mapping(
            values.get("marker_position_encoding"),
            legacy_temporal_embedding=legacy_temporal,
        )
        reference_path = values.get("marker_reference_xy_path")
        reference_value = values.get("marker_reference_xy")
        has_reference_value = reference_value is not None and len(reference_value) > 0
        if has_reference_value:
            reference = _normalize_reference_xy(
                torch.as_tensor(reference_value, dtype=torch.float32)
            )
            if tuple(reference.shape) != (num_sensors, num_markers, 2):
                raise ValueError(
                    "marker_reference_xy must have shape "
                    f"[{num_sensors},{num_markers},2]"
                )
            marker_reference_xy = tuple(
                tuple((float(point[0]), float(point[1])) for point in sensor)
                for sensor in reference
            )
        elif reference_path is not None and Path(reference_path).expanduser().is_file():
            marker_reference_xy = load_marker_reference_xy(
                reference_path,
                sensor_names=sensor_names,
                num_markers=num_markers,
            )
        elif enabled and use_markers and tokenization.mode in {
            "regional",
            "point_spatiotemporal",
        }:
            raise ValueError(
                f"{tokenization.mode} marker tokenization requires "
                "marker_reference_xy_path or embedded marker_reference_xy"
            )
        else:
            marker_reference_xy = ()
        contact_gate = MarkerContactGateConfig.from_mapping(
            values.get("marker_contact_gate"),
            num_sensors=num_sensors,
            num_markers=num_markers,
            sensor_names=sensor_names,
        )
        marker_ablation = MarkerAblationConfig.from_mapping(
            values.get("marker_ablation")
        )
        gate_tactile_rgb = (
            contact_gate.mode != "none" and contact_gate.target == "marker_and_rgb"
        )
        if "gate_tactile_rgb" in values and _read_bool(
            values.get("gate_tactile_rgb"), False
        ) != gate_tactile_rgb:
            raise ValueError(
                "gate_tactile_rgb must agree with marker_contact_gate.target"
            )

        return cls(
            enabled=enabled,
            num_sensors=num_sensors,
            num_markers=num_markers,
            use_rgb=use_rgb,
            use_markers=use_markers,
            share_vision_encoder=True,
            freeze_vision_encoder=_read_bool(values.get("freeze_vision_encoder"), True),
            marker_history_length=marker_history_length,
            marker_sample_hz=marker_sample_hz,
            marker_input_features=marker_input_features,
            marker_feature_mode=marker_feature_mode,
            marker_temporal_embedding=position_encoding.temporal_type == "learned",
            marker_hidden_dim=marker_hidden_dim,
            marker_tokenization=tokenization,
            marker_position_encoding=position_encoding,
            marker_contact_gate=contact_gate,
            gate_tactile_rgb=gate_tactile_rgb,
            marker_ablation=marker_ablation,
            marker_reference_xy_path=(
                str(reference_path) if reference_path is not None else None
            ),
            marker_reference_xy=marker_reference_xy,
            add_modality_embedding=_read_bool(values.get("add_modality_embedding"), True),
            add_sensor_side_embedding=_read_bool(values.get("add_sensor_side_embedding"), True),
            marker_stats_path=str(stats_path) if stats_path is not None else None,
            marker_mean=marker_mean,
            marker_std=marker_std,
            require_marker_stats=require_stats,
            sensor_names=sensor_names,
            rgb_keys=_keys("rgb_keys", cls.rgb_keys),
            marker_displacement_keys=_keys(
                "marker_displacement_keys", cls.marker_displacement_keys
            ),
            marker_valid_mask_keys=_keys("marker_valid_mask_keys", cls.marker_valid_mask_keys),
            debug_shapes=_read_bool(values.get("debug_shapes"), False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-safe values for Hugging Face config serialization."""

        result = asdict(self)
        result["marker_tokenization"] = asdict(self.marker_tokenization)
        result["marker_position_encoding"] = asdict(self.marker_position_encoding)
        result["marker_contact_gate"] = self.marker_contact_gate.to_dict()
        result["marker_ablation"] = asdict(self.marker_ablation)

        def _lists(value: Any) -> Any:
            if isinstance(value, tuple):
                return [_lists(item) for item in value]
            if isinstance(value, dict):
                return {key: _lists(item) for key, item in value.items()}
            return value

        result = _lists(result)
        return result

    @property
    def marker_tokens_per_sensor(self) -> int:
        spatial_tokens = self.marker_tokenization.num_regions
        return self.marker_history_length * spatial_tokens


def validate_marker_displacement_history(
    marker_displacement_history: Tensor,
    marker_valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Validate normalized ``[dx, dy]`` history and mask invalid markers."""

    if marker_displacement_history.ndim != 5 or marker_displacement_history.shape[-1] != 2:
        raise ValueError(
            "marker_displacement_history must have shape [B,S,H,N,2], "
            f"got {tuple(marker_displacement_history.shape)}"
        )
    if not torch.is_floating_point(marker_displacement_history):
        raise ValueError("marker_displacement_history must be floating point")
    if not torch.isfinite(marker_displacement_history).all():
        raise ValueError("marker_displacement_history must be finite")

    expected_mask_shape = marker_displacement_history.shape[:-1]
    if marker_valid_mask is None:
        marker_valid_mask = torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=marker_displacement_history.device,
        )
    elif tuple(marker_valid_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            f"marker_valid_mask must have shape {tuple(expected_mask_shape)}, "
            f"got {tuple(marker_valid_mask.shape)}"
        )
    else:
        marker_valid_mask = marker_valid_mask.to(
            device=marker_displacement_history.device, dtype=torch.bool
        )

    features = marker_displacement_history * marker_valid_mask.unsqueeze(-1).to(
        dtype=marker_displacement_history.dtype
    )
    return features, marker_valid_mask


class MarkerEncoder(nn.Module):
    """Shared per-frame MLP that emits one token per marker history frame."""

    def __init__(
        self,
        num_markers: int,
        context_dim: int,
        input_features: int = 2,
        hidden_dim: int = 512,
        marker_mean: Sequence[float] = (0.0, 0.0),
        marker_std: Sequence[float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        if num_markers <= 0 or context_dim <= 0 or hidden_dim <= 0:
            raise ValueError("num_markers, context_dim, and hidden_dim must be positive")
        if input_features != 2:
            raise ValueError("MarkerEncoder input_features must be 2 ([dx, dy])")

        mean = _as_float_tuple(marker_mean, name="marker_mean", length=input_features)
        std = _as_float_tuple(marker_std, name="marker_std", length=input_features)
        if any(value <= 0 for value in std):
            raise ValueError("marker_std values must be positive")

        self.num_markers = int(num_markers)
        self.input_features = int(input_features)
        self.context_dim = int(context_dim)
        self.register_buffer("marker_mean", torch.tensor(mean, dtype=torch.float32), persistent=True)
        self.register_buffer("marker_std", torch.tensor(std, dtype=torch.float32), persistent=True)
        input_dim = self.num_markers * self.input_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        marker_features: Tensor,
        marker_valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode ``[B,S,H,N,2]`` into ``[B,S,H,D]`` marker tokens."""

        if marker_features.ndim != 5:
            raise ValueError("marker_features must have shape [B,S,H,N,F]")
        if marker_features.shape[-2:] != (self.num_markers, self.input_features):
            raise ValueError(
                f"Expected [...,{self.num_markers},{self.input_features}], "
                f"got {tuple(marker_features.shape)}"
            )
        expected_mask_shape = marker_features.shape[:-1]
        if marker_valid_mask is None:
            marker_valid_mask = torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=marker_features.device,
            )
        elif tuple(marker_valid_mask.shape) != tuple(expected_mask_shape):
            raise ValueError(
                f"marker_valid_mask must have shape {tuple(expected_mask_shape)}, "
                f"got {tuple(marker_valid_mask.shape)}"
            )
        marker_valid_mask = marker_valid_mask.to(device=marker_features.device, dtype=torch.bool)

        mean = self.marker_mean.to(device=marker_features.device, dtype=marker_features.dtype)
        std = self.marker_std.to(device=marker_features.device, dtype=marker_features.dtype)
        normalized = (marker_features - mean) / (std + 1e-6)
        normalized = normalized * marker_valid_mask.unsqueeze(-1).to(dtype=normalized.dtype)
        leading_shape = normalized.shape[:-2]
        flat = normalized.reshape(-1, self.num_markers * self.input_features)
        tokens = self.encoder(flat).reshape(*leading_shape, self.context_dim)
        sensor_valid = marker_valid_mask.any(dim=-1)
        tokens = tokens * sensor_valid.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens


class RegionalMarkerEncoder(nn.Module):
    """Shared point encoder and masked regional pooling for all sensors."""

    def __init__(
        self,
        *,
        reference_xy: Tensor,
        region_ids: Tensor,
        context_dim: int,
        point_hidden_dim: int,
        region_hidden_dim: int,
        aggregation: str,
        include_reference_xy: bool,
        marker_mean: Sequence[float],
        marker_std: Sequence[float],
        disable_spatial_content: bool,
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported regional marker aggregation: {aggregation}")
        reference_xy = torch.as_tensor(reference_xy, dtype=torch.float32)
        region_ids = torch.as_tensor(region_ids, dtype=torch.long)
        if reference_xy.ndim != 3 or reference_xy.shape[-1] != 2:
            raise ValueError("reference_xy must be [S,N,2]")
        if region_ids.shape != reference_xy.shape[:2]:
            raise ValueError("region_ids must be [S,N]")
        self.num_regions = int(region_ids.max().item()) + 1
        self.aggregation = aggregation
        self.include_reference_xy = bool(include_reference_xy)
        self.disable_spatial_content = bool(disable_spatial_content)
        self.register_buffer("reference_xy", reference_xy, persistent=True)
        self.register_buffer("region_ids", region_ids, persistent=True)
        membership = torch.nn.functional.one_hot(
            region_ids, num_classes=self.num_regions
        ).to(torch.bool)
        self.register_buffer("region_membership", membership, persistent=True)
        self.register_buffer(
            "marker_mean",
            torch.tensor(marker_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "marker_std",
            torch.tensor(marker_std, dtype=torch.float32),
            persistent=True,
        )
        input_dim = 4 if include_reference_xy else 2
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, point_hidden_dim),
            nn.LayerNorm(point_hidden_dim),
            nn.GELU(),
        )
        pooled_dim = point_hidden_dim * (2 if aggregation == "mean_max" else 1)
        self.region_projection = nn.Sequential(
            nn.Linear(pooled_dim, region_hidden_dim),
            nn.LayerNorm(region_hidden_dim),
            nn.GELU(),
            nn.Linear(region_hidden_dim, context_dim),
        )

    def forward(
        self,
        marker_features: Tensor,
        marker_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if marker_features.ndim != 5 or marker_features.shape[-1] != 2:
            raise ValueError("Regional marker features must be [B,S,H,N,2]")
        if marker_valid_mask.shape != marker_features.shape[:-1]:
            raise ValueError("Regional marker valid mask must be [B,S,H,N]")
        if (
            marker_features.shape[1] != self.reference_xy.shape[0]
            or marker_features.shape[3] != self.reference_xy.shape[1]
        ):
            raise ValueError("Regional marker input does not match reference coordinates")
        mean = self.marker_mean.to(marker_features)
        std = self.marker_std.to(marker_features)
        normalized = (marker_features - mean) / (std + 1e-6)
        normalized = normalized * marker_valid_mask.unsqueeze(-1).to(normalized.dtype)
        point_input = normalized
        if self.include_reference_xy:
            reference = self.reference_xy.to(marker_features)[None, :, None]
            reference = reference.expand(
                marker_features.shape[0],
                -1,
                marker_features.shape[2],
                -1,
                -1,
            )
            if self.disable_spatial_content:
                reference = torch.zeros_like(reference)
            point_input = torch.cat([normalized, reference], dim=-1)
        point_features = self.point_encoder(point_input)

        membership = self.region_membership.to(marker_valid_mask.device)[None, :, None]
        pool_mask = marker_valid_mask.unsqueeze(-1) & membership
        counts = pool_mask.sum(dim=3)
        region_valid = counts > 0
        expanded = point_features.unsqueeze(-2)
        mean_features = (
            expanded * pool_mask.unsqueeze(-1).to(point_features.dtype)
        ).sum(dim=3) / counts.clamp_min(1).unsqueeze(-1)
        max_features = expanded.expand(-1, -1, -1, -1, self.num_regions, -1)
        max_features = max_features.masked_fill(
            ~pool_mask.unsqueeze(-1),
            torch.finfo(point_features.dtype).min,
        ).amax(dim=3)
        max_features = torch.where(
            region_valid.unsqueeze(-1),
            max_features,
            torch.zeros_like(max_features),
        )
        if self.aggregation == "mean":
            pooled = mean_features
        elif self.aggregation == "max":
            pooled = max_features
        else:
            pooled = torch.cat([mean_features, max_features], dim=-1)
        tokens = self.region_projection(pooled)
        tokens = tokens * region_valid.unsqueeze(-1).to(tokens.dtype)
        return tokens, region_valid


class PointSpatiotemporalMarkerEncoder(nn.Module):
    """Encode every marker in every history frame without spatial pooling."""

    def __init__(
        self,
        *,
        reference_xy: Tensor,
        context_dim: int,
        point_hidden_dim: int,
        projection_hidden_dim: int,
        include_reference_xy: bool,
        marker_mean: Sequence[float],
        marker_std: Sequence[float],
        disable_spatial_content: bool,
    ) -> None:
        super().__init__()
        reference_xy = torch.as_tensor(reference_xy, dtype=torch.float32)
        if reference_xy.ndim != 3 or reference_xy.shape[-1] != 2:
            raise ValueError("reference_xy must be [S,N,2]")
        if point_hidden_dim <= 0 or projection_hidden_dim <= 0:
            raise ValueError("Point encoder hidden dimensions must be positive")
        self.include_reference_xy = bool(include_reference_xy)
        self.disable_spatial_content = bool(disable_spatial_content)
        self.register_buffer("reference_xy", reference_xy, persistent=True)
        self.register_buffer(
            "marker_mean",
            torch.tensor(marker_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "marker_std",
            torch.tensor(marker_std, dtype=torch.float32),
            persistent=True,
        )
        input_dim = 4 if include_reference_xy else 2
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, point_hidden_dim),
            nn.LayerNorm(point_hidden_dim),
            nn.GELU(),
            nn.Linear(point_hidden_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            nn.GELU(),
            nn.Linear(projection_hidden_dim, context_dim),
        )

    def forward(
        self,
        marker_features: Tensor,
        marker_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if marker_features.ndim != 5 or marker_features.shape[-1] != 2:
            raise ValueError(
                "Point spatiotemporal marker features must be [B,S,H,N,2]"
            )
        if marker_valid_mask.shape != marker_features.shape[:-1]:
            raise ValueError(
                "Point spatiotemporal marker valid mask must be [B,S,H,N]"
            )
        if (
            marker_features.shape[1] != self.reference_xy.shape[0]
            or marker_features.shape[3] != self.reference_xy.shape[1]
        ):
            raise ValueError(
                "Point spatiotemporal marker input does not match reference coordinates"
            )
        mean = self.marker_mean.to(marker_features)
        std = self.marker_std.to(marker_features)
        normalized = (marker_features - mean) / (std + 1e-6)
        normalized = normalized * marker_valid_mask.unsqueeze(-1).to(
            normalized.dtype
        )
        point_input = normalized
        if self.include_reference_xy:
            reference = self.reference_xy.to(marker_features)[None, :, None]
            reference = reference.expand(
                marker_features.shape[0],
                -1,
                marker_features.shape[2],
                -1,
                -1,
            )
            if self.disable_spatial_content:
                reference = torch.zeros_like(reference)
            point_input = torch.cat([normalized, reference], dim=-1)
        tokens = self.point_encoder(point_input)
        tokens = tokens * marker_valid_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, marker_valid_mask


def continuous_sincos_1d(values: Tensor, dim: int) -> Tensor:
    """Encode arbitrary continuous values, padding an unmatched final dimension."""

    if dim <= 0:
        raise ValueError("sincos dimension must be positive")
    values = torch.as_tensor(values, dtype=torch.float32)
    pairs = dim // 2
    output = torch.zeros(*values.shape, dim, dtype=torch.float32, device=values.device)
    if pairs == 0:
        return output
    exponent = torch.arange(pairs, dtype=torch.float32, device=values.device)
    exponent = exponent / max(1, pairs - 1)
    frequencies = torch.exp(-math.log(10000.0) * exponent)
    angles = values.unsqueeze(-1) * frequencies * (2.0 * math.pi)
    encoded = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    output[..., : 2 * pairs] = encoded
    return output


def continuous_sincos_2d(coordinates: Tensor, dim: int) -> Tensor:
    if coordinates.shape[-1] != 2:
        raise ValueError("2D sincos coordinates must end in [x,y]")
    x_dim = dim // 2
    y_dim = dim - x_dim
    return torch.cat(
        [
            continuous_sincos_1d(coordinates[..., 0], x_dim),
            continuous_sincos_1d(coordinates[..., 1], y_dim),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class MarkerEncodingOutput:
    tokens: Tensor
    token_mask: Tensor
    content_tokens: Tensor
    regional_scores: Tensor
    regional_soft_gates: Tensor
    region_valid_mask: Tensor
    global_contact_state: Tensor
    token_layout: Mapping[str, Any]


class TactileTokenEncoder(nn.Module):
    """Add modality/sensor identity and emit fixed-length tactile token blocks."""

    def __init__(
        self,
        settings: TactileVTLAConfig,
        context_dim: int,
        vision_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not settings.enabled:
            raise ValueError("TactileTokenEncoder requires tactile.enabled=true")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.settings = settings
        self.context_dim = int(context_dim)
        vision_output_dim = int(vision_output_dim or context_dim)

        self.marker_encoder: MarkerEncoder | None = None
        self.regional_marker_encoder: RegionalMarkerEncoder | None = None
        self.point_spatiotemporal_marker_encoder: (
            PointSpatiotemporalMarkerEncoder | None
        ) = None
        self.register_buffer("marker_reference_xy", torch.empty(0), persistent=True)
        self.register_buffer("marker_region_ids", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("marker_region_centers", torch.empty(0), persistent=True)
        if settings.use_markers:
            if settings.marker_reference_xy:
                reference = torch.tensor(settings.marker_reference_xy, dtype=torch.float32)
            else:
                reference = torch.zeros(
                    settings.num_sensors, settings.num_markers, 2, dtype=torch.float32
                )
                reference[..., 0] = torch.linspace(-1.0, 1.0, settings.num_markers)
                reference[..., 1] = torch.linspace(-1.0, 1.0, settings.num_markers)
            if settings.marker_tokenization.mode == "point_spatiotemporal":
                region_ids = torch.arange(
                    settings.num_markers, dtype=torch.long
                )[None].expand(settings.num_sensors, -1).clone()
                region_centers = reference.clone()
            else:
                region_ids, region_centers = build_marker_region_mapping(
                    reference,
                    num_regions=settings.marker_tokenization.num_regions,
                    region_layout=settings.marker_tokenization.region_layout,
                )
            self.marker_reference_xy = reference
            self.marker_region_ids = region_ids
            self.marker_region_centers = region_centers
            if settings.marker_tokenization.mode == "global":
                self.marker_encoder = MarkerEncoder(
                    num_markers=settings.num_markers,
                    context_dim=context_dim,
                    input_features=settings.marker_input_features,
                    hidden_dim=settings.marker_hidden_dim,
                    marker_mean=settings.marker_mean,
                    marker_std=settings.marker_std,
                )
            elif settings.marker_tokenization.mode == "regional":
                tokenization = settings.marker_tokenization
                self.regional_marker_encoder = RegionalMarkerEncoder(
                    reference_xy=reference,
                    region_ids=region_ids,
                    context_dim=context_dim,
                    point_hidden_dim=tokenization.point_hidden_dim,
                    region_hidden_dim=tokenization.region_hidden_dim,
                    aggregation=tokenization.aggregation,
                    include_reference_xy=tokenization.include_reference_xy,
                    marker_mean=settings.marker_mean,
                    marker_std=settings.marker_std,
                    disable_spatial_content=settings.marker_ablation.disable_spatial_content,
                )
            else:
                tokenization = settings.marker_tokenization
                self.point_spatiotemporal_marker_encoder = (
                    PointSpatiotemporalMarkerEncoder(
                        reference_xy=reference,
                        context_dim=context_dim,
                        point_hidden_dim=tokenization.point_hidden_dim,
                        projection_hidden_dim=tokenization.region_hidden_dim,
                        include_reference_xy=tokenization.include_reference_xy,
                        marker_mean=settings.marker_mean,
                        marker_std=settings.marker_std,
                        disable_spatial_content=(
                            settings.marker_ablation.disable_spatial_content
                        ),
                    )
                )

        if settings.use_rgb and vision_output_dim != context_dim:
            self.tactile_rgb_projection: nn.Module = nn.Linear(vision_output_dim, context_dim)
        else:
            self.tactile_rgb_projection = nn.Identity()

        if settings.add_modality_embedding:
            self.tactile_rgb_modality_embedding = nn.Parameter(torch.zeros(1, 1, context_dim))
            self.marker_modality_embedding = nn.Parameter(torch.zeros(1, 1, context_dim))
        else:
            self.register_parameter("tactile_rgb_modality_embedding", None)
            self.register_parameter("marker_modality_embedding", None)
        if settings.add_sensor_side_embedding:
            self.sensor_side_embeddings: nn.Module | None = nn.Embedding(
                settings.num_sensors,
                context_dim,
            )
        else:
            self.sensor_side_embeddings = None
        position = settings.marker_position_encoding
        if settings.use_markers and position.temporal_type == "learned":
            self.marker_temporal_embedding: nn.Module | None = nn.Embedding(
                settings.marker_history_length,
                context_dim,
            )
        else:
            self.marker_temporal_embedding = None

        temporal_values = torch.arange(settings.marker_history_length, dtype=torch.float32)
        if position.use_real_time:
            temporal_values = (
                temporal_values - float(settings.marker_history_length - 1)
            ) / float(settings.marker_sample_hz)
        if settings.use_markers and position.temporal_type == "sincos":
            temporal_encoding = continuous_sincos_1d(temporal_values, context_dim)
        else:
            temporal_encoding = torch.zeros(settings.marker_history_length, context_dim)
        self.register_buffer("marker_temporal_encoding", temporal_encoding, persistent=True)

        regions = settings.marker_tokenization.num_regions
        if settings.use_markers and position.spatial_type == "learned":
            self.marker_spatial_embedding: nn.Module | None = nn.Embedding(
                regions, context_dim
            )
        else:
            self.marker_spatial_embedding = None
        if settings.use_markers and position.spatial_type == "sincos":
            spatial_encoding = continuous_sincos_2d(
                self.marker_region_centers,
                context_dim,
            )
        else:
            spatial_encoding = torch.zeros(settings.num_sensors, regions, context_dim)
        self.register_buffer("marker_spatial_encoding", spatial_encoding, persistent=True)

        generator = torch.Generator().manual_seed(settings.marker_ablation.shuffle_seed)
        temporal_order = torch.arange(settings.marker_history_length)
        spatial_order = torch.arange(regions)
        if settings.marker_ablation.shuffle_temporal_order:
            temporal_order = torch.randperm(settings.marker_history_length, generator=generator)
        if settings.marker_ablation.shuffle_region_order:
            spatial_order = torch.randperm(regions, generator=generator)
        self.register_buffer("marker_temporal_content_order", temporal_order, persistent=True)
        self.register_buffer("marker_spatial_content_order", spatial_order, persistent=True)

        gate = settings.marker_contact_gate
        self.register_buffer(
            "marker_gate_point_thresholds",
            torch.tensor(gate.point_thresholds, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "marker_gate_soft_thresholds",
            torch.tensor(gate.soft_thresholds, dtype=torch.float32),
            persistent=True,
        )
        self.last_marker_diagnostics: dict[str, Tensor] = {}

    def _sensor_embedding(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.sensor_side_embeddings is None:
            return torch.zeros(
                self.settings.num_sensors,
                self.context_dim,
                dtype=dtype,
                device=device,
            )
        indices = torch.arange(self.settings.num_sensors, device=device)
        return self.sensor_side_embeddings(indices).to(dtype=dtype)

    def _init_missing_parameter(self, name: str) -> None:
        """Initialize direct parameters after an old checkpoint was dispatched."""

        if name not in {
            "tactile_rgb_modality_embedding",
            "marker_modality_embedding",
        }:
            raise ValueError(f"Unsupported direct tactile parameter: {name}")
        parameter = self._parameters.get(name)
        if parameter is not None:
            with torch.no_grad():
                parameter.zero_()

    @property
    def marker_dtype(self) -> torch.dtype:
        module: nn.Module | None = (
            self.marker_encoder
            or self.regional_marker_encoder
            or self.point_spatiotemporal_marker_encoder
        )
        if module is None:
            return torch.float32
        return next(module.parameters()).dtype

    @staticmethod
    def _sensor_mask(
        sensor_mask: Tensor | None,
        *,
        batch_size: int,
        num_sensors: int,
        device: torch.device,
    ) -> Tensor:
        if sensor_mask is None:
            return torch.ones(batch_size, num_sensors, dtype=torch.bool, device=device)
        if tuple(sensor_mask.shape) != (batch_size, num_sensors):
            raise ValueError(
                f"tactile_sensor_mask must be [{batch_size},{num_sensors}], "
                f"got {tuple(sensor_mask.shape)}"
            )
        return sensor_mask.to(device=device, dtype=torch.bool)

    def encode_markers(
        self,
        marker_displacement_history: Tensor,
        marker_valid_mask: Tensor | None,
        marker_history_valid_mask: Tensor | None,
        tactile_sensor_mask: Tensor | None,
        marker_contact_state: Tensor | None = None,
        *,
        return_debug_info: bool = False,
    ) -> tuple[Tensor, Tensor] | MarkerEncodingOutput:
        """Return marker tokens flattened in sensor/time/spatial order."""

        if (
            self.marker_encoder is None
            and self.regional_marker_encoder is None
            and self.point_spatiotemporal_marker_encoder is None
        ):
            raise RuntimeError("Marker encoding is disabled")
        if marker_displacement_history.ndim != 5:
            raise ValueError(
                "marker_displacement_history must have shape [B,S,H,N,2]"
            )
        batch_size, num_sensors, history_length, num_markers = (
            marker_displacement_history.shape[:4]
        )
        if num_sensors != self.settings.num_sensors:
            raise ValueError(
                f"Expected {self.settings.num_sensors} tactile sensors, got {num_sensors}"
            )
        if history_length != self.settings.marker_history_length:
            raise ValueError(
                f"Expected marker history length {self.settings.marker_history_length}, "
                f"got {history_length}"
            )
        if num_markers != self.settings.num_markers:
            raise ValueError(
                f"Expected {self.settings.num_markers} markers, got {num_markers}"
            )
        features, marker_mask = validate_marker_displacement_history(
            marker_displacement_history,
            marker_valid_mask,
        )
        sensor_mask = self._sensor_mask(
            tactile_sensor_mask,
            batch_size=batch_size,
            num_sensors=num_sensors,
            device=marker_displacement_history.device,
        )
        if marker_history_valid_mask is None:
            history_mask = torch.ones(
                batch_size,
                num_sensors,
                history_length,
                dtype=torch.bool,
                device=marker_displacement_history.device,
            )
        else:
            if tuple(marker_history_valid_mask.shape) != (
                batch_size,
                num_sensors,
                history_length,
            ):
                raise ValueError(
                    "marker_history_valid_mask must have shape "
                    f"[{batch_size},{num_sensors},{history_length}]"
                )
            history_mask = marker_history_valid_mask.to(
                device=marker_displacement_history.device,
                dtype=torch.bool,
            )
        if self.marker_encoder is not None:
            content = self.marker_encoder(features, marker_mask).unsqueeze(-2)
            region_valid = marker_mask.any(dim=-1, keepdim=True)
        elif self.regional_marker_encoder is not None:
            assert self.regional_marker_encoder is not None
            content, region_valid = self.regional_marker_encoder(features, marker_mask)
        else:
            assert self.point_spatiotemporal_marker_encoder is not None
            content, region_valid = self.point_spatiotemporal_marker_encoder(
                features, marker_mask
            )

        content = content[:, :, self.marker_temporal_content_order]
        region_valid = region_valid[:, :, self.marker_temporal_content_order]
        history_mask = history_mask[:, :, self.marker_temporal_content_order]
        content = content[:, :, :, self.marker_spatial_content_order]
        region_valid = region_valid[:, :, :, self.marker_spatial_content_order]

        scores, regional_soft_gates = self._regional_contact_scores(features, marker_mask)
        scores = scores[:, :, self.marker_temporal_content_order]
        regional_soft_gates = regional_soft_gates[:, :, self.marker_temporal_content_order]
        scores = scores[:, :, :, self.marker_spatial_content_order]
        regional_soft_gates = regional_soft_gates[:, :, :, self.marker_spatial_content_order]

        gate = self.settings.marker_contact_gate
        if gate.mode == "none":
            global_state = torch.full(
                (batch_size, num_sensors),
                1,
                dtype=torch.int8,
                device=features.device,
            )
        else:
            if marker_contact_state is None:
                raise ValueError(
                    "Stateful marker contact gating requires marker_contact_state [B,S] "
                    "from episode preprocessing or online TacThruSource"
                )
            if tuple(marker_contact_state.shape) != (batch_size, num_sensors):
                raise ValueError(
                    f"marker_contact_state must be [{batch_size},{num_sensors}]"
                )
            global_state = marker_contact_state.to(device=features.device, dtype=torch.int8)

        if gate.regional_soft_gate and gate.mode == "hard_hysteresis_soft_region":
            content = content * regional_soft_gates.unsqueeze(-1).to(content.dtype)
        if self.settings.marker_ablation.zero_marker_content:
            content = torch.zeros_like(content)

        tokens = content
        temporal = self._temporal_position(tokens.dtype, tokens.device)
        spatial = self._spatial_position(tokens.dtype, tokens.device)
        tokens = tokens + temporal[None, None, :, None, :]
        tokens = tokens + spatial[None, :, None, :, :]
        if self.marker_modality_embedding is not None:
            tokens = tokens + self.marker_modality_embedding.to(dtype=tokens.dtype)
        tokens = tokens + self._sensor_embedding(tokens.dtype, tokens.device)[
            None, :, None, None, :
        ]
        token_mask = (
            sensor_mask[:, :, None, None]
            & history_mask[:, :, :, None]
            & region_valid
            & contact_state_is_visible(global_state)[:, :, None, None]
        )
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        content = content * token_mask.unsqueeze(-1).to(dtype=content.dtype)
        tokens = tokens.flatten(2, 3)
        token_mask = token_mask.flatten(2, 3)
        rgb_ratio = self.last_marker_diagnostics.get("rgb_token_valid_ratio")
        self.last_marker_diagnostics = {
            "contact_on_ratio": (global_state == 1).float().mean().detach(),
            "contact_hold_ratio": ((global_state == 2) | (global_state == 5)).float().mean().detach(),
            "contact_off_ratio": ((global_state == 0) | (global_state == 3)).float().mean().detach(),
            "contact_unknown_ratio": contact_state_is_unknown(global_state).float().mean().detach(),
            "marker_token_valid_ratio": token_mask.float().mean().detach(),
            "regional_soft_gate_mean": regional_soft_gates.mean(dim=(0, 1, 2)).detach(),
            "regional_valid_marker_count": self._regional_valid_counts(marker_mask).float().mean(dim=(0, 1, 2)).detach(),
        }
        if rgb_ratio is not None:
            self.last_marker_diagnostics["rgb_token_valid_ratio"] = rgb_ratio
        output = MarkerEncodingOutput(
            tokens=tokens,
            token_mask=token_mask,
            content_tokens=content.flatten(2, 3),
            regional_scores=scores,
            regional_soft_gates=regional_soft_gates,
            region_valid_mask=region_valid,
            global_contact_state=global_state,
            token_layout={
                "mode": self.settings.marker_tokenization.mode,
                "history_length": history_length,
                "num_regions": self.settings.marker_tokenization.num_regions,
                "order": "sensor-major,time-major,spatial-minor",
            },
        )
        return output if return_debug_info else (output.tokens, output.token_mask)

    def _temporal_position(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.marker_temporal_embedding is not None:
            indices = torch.arange(self.settings.marker_history_length, device=device)
            return self.marker_temporal_embedding(indices).to(dtype=dtype)
        return self.marker_temporal_encoding.to(device=device, dtype=dtype)

    def _spatial_position(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.marker_spatial_embedding is not None:
            indices = torch.arange(
                self.settings.marker_tokenization.num_regions, device=device
            )
            value = self.marker_spatial_embedding(indices).to(dtype=dtype)
            return value[None].expand(self.settings.num_sensors, -1, -1)
        return self.marker_spatial_encoding.to(device=device, dtype=dtype)

    def _regional_valid_counts(self, marker_mask: Tensor) -> Tensor:
        membership = torch.nn.functional.one_hot(
            self.marker_region_ids,
            num_classes=self.settings.marker_tokenization.num_regions,
        ).to(device=marker_mask.device, dtype=torch.bool)
        return (marker_mask.unsqueeze(-1) & membership[None, :, None]).sum(dim=3)

    def _regional_contact_scores(self, raw: Tensor, marker_mask: Tensor) -> tuple[Tensor, Tensor]:
        amplitudes = torch.linalg.vector_norm(raw, dim=-1)
        membership = torch.nn.functional.one_hot(
            self.marker_region_ids,
            num_classes=self.settings.marker_tokenization.num_regions,
        ).to(device=raw.device, dtype=torch.bool)
        mask = marker_mask.unsqueeze(-1) & membership[None, :, None]
        expanded = amplitudes.unsqueeze(-1).expand_as(mask)
        negative = torch.finfo(amplitudes.dtype).min
        ranked = expanded.masked_fill(~mask, negative).transpose(3, 4)
        k = min(self.settings.marker_contact_gate.topk_markers, raw.shape[3])
        top = torch.topk(ranked, k=k, dim=-1).values
        top_valid = torch.isfinite(top) & (top != negative)
        scores = torch.where(top_valid, top, torch.zeros_like(top)).sum(-1)
        scores = scores / top_valid.sum(-1).clamp_min(1)
        soft_threshold = self.marker_gate_soft_thresholds.to(raw.device, raw.dtype)
        soft = torch.sigmoid(
            (scores - soft_threshold[None, :, None, None])
            / self.settings.marker_contact_gate.soft_gate_temperature
        )
        return scores, soft

    def gate_rgb_sensor_mask(
        self,
        rgb_sensor_mask: Tensor,
        marker_contact_state: Tensor | None,
    ) -> Tensor:
        """Apply the explicit negative-control gate; marker_only is unchanged."""

        gate = self.settings.marker_contact_gate
        if gate.mode == "none" or gate.target == "marker_only":
            return rgb_sensor_mask
        if marker_contact_state is None:
            raise ValueError("marker_and_rgb gate requires marker_contact_state")
        return rgb_sensor_mask & contact_state_is_visible(
            marker_contact_state.to(rgb_sensor_mask.device)
        )

    def encode_rgb_embeddings(
        self,
        rgb_embeddings: Tensor,
        tactile_sensor_mask: Tensor | None,
        tactile_rgb_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project shared-ViT embeddings and flatten sensors along token length.

        Args:
            rgb_embeddings: ``[B,S,P,D_vision]``.
            tactile_sensor_mask: optional ``[B,S]`` sensor availability.
            tactile_rgb_mask: optional ``[B,S]`` or ``[B,S,P]`` RGB validity.
        """

        if not self.settings.use_rgb:
            raise RuntimeError("Tactile RGB encoding is disabled")
        if rgb_embeddings.ndim != 4:
            raise ValueError("rgb_embeddings must have shape [B,S,P,D]")
        batch_size, num_sensors, num_patches, _ = rgb_embeddings.shape
        if num_sensors != self.settings.num_sensors:
            raise ValueError(
                f"Expected {self.settings.num_sensors} tactile sensors, got {num_sensors}"
            )
        sensor_mask = self._sensor_mask(
            tactile_sensor_mask,
            batch_size=batch_size,
            num_sensors=num_sensors,
            device=rgb_embeddings.device,
        )
        if tactile_rgb_mask is None:
            rgb_mask = sensor_mask.unsqueeze(-1).expand(-1, -1, num_patches)
        elif tuple(tactile_rgb_mask.shape) == (batch_size, num_sensors):
            rgb_mask = sensor_mask & tactile_rgb_mask.to(device=rgb_embeddings.device, dtype=torch.bool)
            rgb_mask = rgb_mask.unsqueeze(-1).expand(-1, -1, num_patches)
        elif tuple(tactile_rgb_mask.shape) == (batch_size, num_sensors, num_patches):
            rgb_mask = tactile_rgb_mask.to(device=rgb_embeddings.device, dtype=torch.bool)
            rgb_mask = rgb_mask & sensor_mask.unsqueeze(-1)
        else:
            raise ValueError("tactile_rgb_mask must have shape [B,S] or [B,S,P]")

        tokens = self.tactile_rgb_projection(rgb_embeddings)
        if self.tactile_rgb_modality_embedding is not None:
            tokens = tokens + self.tactile_rgb_modality_embedding.to(dtype=tokens.dtype)
        tokens = tokens + self._sensor_embedding(tokens.dtype, tokens.device)[None, :, None, :]
        tokens = tokens * rgb_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        self.last_marker_diagnostics["rgb_token_valid_ratio"] = (
            rgb_mask.float().mean().detach()
        )
        return tokens.flatten(1, 2), rgb_mask.flatten(1, 2)


_LEGACY_MARKER_REINITIALIZED_SUFFIXES = (
    "marker_encoder.encoder.0.weight",
    "marker_encoder.encoder.0.bias",
    "marker_encoder.marker_mean",
    "marker_encoder.marker_std",
    "marker_temporal_embedding.weight",
)

_CONFIG_DERIVED_MARKER_SUFFIXES = (
    "marker_reference_xy",
    "marker_region_ids",
    "marker_region_centers",
    "regional_marker_encoder.reference_xy",
    "regional_marker_encoder.region_ids",
    "regional_marker_encoder.region_membership",
    "point_spatiotemporal_marker_encoder.reference_xy",
    "marker_temporal_encoding",
    "marker_spatial_encoding",
    "marker_temporal_content_order",
    "marker_spatial_content_order",
    "marker_gate_point_thresholds",
    "marker_gate_soft_thresholds",
)

_TACTILE_REFINEMENT_PREFIXES = (
    "tactile_action_expert.",
    "tactile_state_proj.",
    "tactile_action_in_proj.",
    "tactile_action_out_proj.",
    "tactile_context_proj.",
    "tactile_plan_proj.",
    "tactile_marker_proj.",
    "tactile_time_mlp.",
)

_TACTILE_REFINEMENT_UPGRADE_FRAGMENTS = (
    ".plan_norm.",
    ".plan_attention.",
    "tactile_plan_proj.",
)


def load_vtla_checkpoint_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allow_legacy_marker_reinit: bool = False,
) -> dict[str, Any]:
    """Load a VTLA checkpoint with explicit marker-only migration policies.

    A one-token ``[dx,dy,vx,vy]`` checkpoint differs only in the first marker
    MLP weight and its four-channel normalization buffers, and has no temporal
    embedding.  With explicit opt-in those marker-only values are omitted so
    the new two-channel layer, buffers, and temporal embedding keep their
    configuration-derived initialization.  Every non-marker mismatch remains
    a hard error.
    """

    target = model.state_dict()
    supplied: MutableMapping[str, Tensor] = dict(state_dict)
    target_is_regional = any("regional_marker_encoder." in name for name in target)
    supplied_has_global = any("marker_encoder.encoder." in name for name in supplied)
    supplied_has_regional = any("regional_marker_encoder." in name for name in supplied)
    config_initialized = {
        name
        for name in target
        if any(name.endswith(suffix) for suffix in _CONFIG_DERIVED_MARKER_SUFFIXES)
    }
    refinement_target = {
        name
        for name in target
        if any(prefix in name for prefix in _TACTILE_REFINEMENT_PREFIXES)
    }
    supplied_has_refinement = any(
        any(prefix in name for prefix in _TACTILE_REFINEMENT_PREFIXES)
        for name in supplied
    )
    refinement_initialized = (
        refinement_target if refinement_target and not supplied_has_refinement else set()
    )
    refinement_upgrade_initialized = {
        name
        for name in refinement_target
        if name not in supplied
        and any(
            fragment in name for fragment in _TACTILE_REFINEMENT_UPGRADE_FRAGMENTS
        )
    }
    intentionally_ignored: list[str] = []
    reinitialized: list[str] = []
    config_incompatibilities: list[str] = []

    for name in list(supplied):
        if any(name.endswith(suffix) for suffix in _CONFIG_DERIVED_MARKER_SUFFIXES):
            supplied.pop(name)
            intentionally_ignored.append(name)

    if target_is_regional and supplied_has_global and not supplied_has_regional:
        config_incompatibilities.append("global_marker_encoder_to_regional")
        for name in list(supplied):
            if "marker_encoder." in name:
                supplied.pop(name)
                intentionally_ignored.append(name)
        reinitialized.extend(
            name for name in target if "regional_marker_encoder." in name
        )
        if any(name.endswith("marker_spatial_embedding.weight") for name in target):
            reinitialized.extend(
                name for name in target if name.endswith("marker_spatial_embedding.weight")
            )

    target_has_learned_temporal = any(
        name.endswith("marker_temporal_embedding.weight") for name in target
    )
    if not target_has_learned_temporal:
        for name in list(supplied):
            if name.endswith("marker_temporal_embedding.weight"):
                supplied.pop(name)
                intentionally_ignored.append(name)
                config_incompatibilities.append("learned_temporal_to_fixed_or_none")

    for suffix, incompatibility in (
        ("marker_temporal_embedding.weight", "fixed_or_missing_temporal_to_learned"),
        ("marker_spatial_embedding.weight", "fixed_or_missing_spatial_to_learned"),
    ):
        target_names = [name for name in target if name.endswith(suffix)]
        for name in target_names:
            if name not in supplied:
                reinitialized.append(name)
                config_incompatibilities.append(incompatibility)

    shape_mismatches = {
        name: (tuple(value.shape), tuple(target[name].shape))
        for name, value in supplied.items()
        if name in target and tuple(value.shape) != tuple(target[name].shape)
    }
    unexpected = sorted(set(supplied) - set(target))

    first_weight_names = [
        name
        for name in shape_mismatches
        if name.endswith("marker_encoder.encoder.0.weight")
    ]
    allowed_mismatch_suffixes = {
        "marker_encoder.encoder.0.weight",
        "marker_encoder.marker_mean",
        "marker_encoder.marker_std",
    }
    only_legacy_marker_mismatches = bool(first_weight_names) and all(
        any(name.endswith(suffix) for suffix in allowed_mismatch_suffixes)
        for name in shape_mismatches
    )
    for name in first_weight_names:
        old_shape, new_shape = shape_mismatches[name]
        only_legacy_marker_mismatches &= (
            len(old_shape) == 2
            and len(new_shape) == 2
            and old_shape[0] == new_shape[0]
            and old_shape[1] == new_shape[1] * 2
        )

    if shape_mismatches and not only_legacy_marker_mismatches:
        raise RuntimeError(
            "Checkpoint contains unsupported shape mismatches: "
            f"{shape_mismatches}"
        )
    if shape_mismatches and not allow_legacy_marker_reinit:
        raise RuntimeError(
            "Detected a legacy [dx,dy,vx,vy] marker checkpoint. The new "
            "8-frame [dx,dy] path requires marker-only reinitialization. "
            "Set LINGBOT_V2_ALLOW_LEGACY_MARKER_REINIT=1 to opt in; all "
            "non-marker weights will remain loaded."
        )

    if shape_mismatches:
        for name in target:
            if any(name.endswith(suffix) for suffix in _LEGACY_MARKER_REINITIALIZED_SUFFIXES):
                supplied.pop(name, None)
                reinitialized.append(name)

    incompatible = model.load_state_dict(supplied, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(set(unexpected) | set(incompatible.unexpected_keys))
    allowed_missing = (
        set(reinitialized)
        | config_initialized
        | refinement_initialized
        | refinement_upgrade_initialized
    )
    missing_config_initialized = set(missing) & config_initialized
    forbidden_missing = sorted(set(missing) - allowed_missing)
    if forbidden_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible after marker migration: "
            f"missing={forbidden_missing}, unexpected={unexpected}"
        )
    loaded_keys = sorted(set(supplied) & set(target))
    return {
        "legacy_marker_reinitialized": bool(shape_mismatches),
        "marker_modules_reinitialized": bool(reinitialized),
        "loaded_keys": loaded_keys,
        "intentionally_reinitialized_keys": sorted(
            set(reinitialized) | missing_config_initialized
        ),
        "reinitialized_keys": sorted(set(reinitialized)),
        "intentionally_ignored_keys": sorted(intentionally_ignored),
        "missing_keys": missing,
        "forbidden_missing_keys": forbidden_missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "config_incompatibilities": sorted(set(config_incompatibilities)),
        "tactile_refinement_initialized_keys": sorted(
            set(missing) & refinement_initialized
        ),
        "tactile_refinement_upgrade_initialized_keys": sorted(
            set(missing) & refinement_upgrade_initialized
        ),
    }


def concatenate_vtla_context(
    vision_language_context: Tensor,
    vision_language_mask: Tensor,
    tactile_blocks: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Concatenate tactile blocks on the token axis without changing hidden size."""

    if vision_language_context.ndim != 3 or vision_language_mask.ndim != 2:
        raise ValueError("vision-language context/mask must be [B,L,D] and [B,L]")
    if vision_language_context.shape[:2] != vision_language_mask.shape:
        raise ValueError("vision-language context and mask lengths differ")
    contexts = [vision_language_context]
    masks = [vision_language_mask]
    for tokens, mask in tactile_blocks:
        if tokens.ndim != 3 or mask.ndim != 2 or tokens.shape[:2] != mask.shape:
            raise ValueError("Each tactile token block must be [B,L,D] with mask [B,L]")
        if tokens.shape[0] != vision_language_context.shape[0]:
            raise ValueError("Tactile and vision-language batch sizes differ")
        if tokens.shape[-1] != vision_language_context.shape[-1]:
            raise ValueError("Tactile and vision-language hidden dimensions differ")
        contexts.append(tokens)
        masks.append(mask.to(dtype=vision_language_mask.dtype, device=vision_language_mask.device))
    return torch.cat(contexts, dim=1), torch.cat(masks, dim=1)


__all__ = [
    "MarkerAblationConfig",
    "MarkerContactGateConfig",
    "MarkerEncoder",
    "MarkerEncodingOutput",
    "MarkerPositionEncodingConfig",
    "MarkerTokenizationConfig",
    "PointSpatiotemporalMarkerEncoder",
    "RegionalMarkerEncoder",
    "TactileTokenEncoder",
    "TactileVTLAConfig",
    "build_marker_region_mapping",
    "concatenate_vtla_context",
    "continuous_sincos_1d",
    "continuous_sincos_2d",
    "load_marker_reference_xy",
    "migrate_legacy_tactile_config",
    "load_vtla_checkpoint_state_dict",
    "load_marker_statistics",
    "validate_marker_displacement_history",
]
