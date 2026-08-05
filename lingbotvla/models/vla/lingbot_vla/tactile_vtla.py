"""Minimal TacThru token encoders for LingBot-VLA v2.

This module intentionally contains no action or Flow Matching code.  It turns
TacThru RGB embeddings and marker coordinates into fixed-length prefix tokens
that can be appended to the existing vision-language prefix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
class TactileVTLAConfig:
    """Validated, serializable configuration for the minimal VTLA extension."""

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
        legacy_token_count = values.get("marker_tokens_per_sensor")
        if legacy_token_count is not None and int(legacy_token_count) != marker_history_length:
            raise ValueError(
                "marker_tokens_per_sensor is deprecated; displacement_history emits exactly "
                "marker_history_length tokens per sensor. Remove the old field and set "
                "marker_history_length explicitly."
            )
        if int(values.get("marker_hidden_dim", 512)) <= 0:
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
        if stats_path is not None and Path(stats_path).expanduser().is_file():
            marker_mean, marker_std = load_marker_statistics(stats_path)
        elif stats_path is not None and mean_value is not None and std_value is not None:
            # Saved Hugging Face configs carry both the original provenance
            # path and resolved values, so inference remains portable even
            # when the training workspace is not mounted.
            marker_mean = _as_float_tuple(
                mean_value, name="marker_mean", length=2
            )
            marker_std = _as_float_tuple(std_value, name="marker_std", length=2)
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
            marker_temporal_embedding=_read_bool(
                values.get("marker_temporal_embedding"), True
            ),
            marker_hidden_dim=int(values.get("marker_hidden_dim", 512)),
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
        for key, value in tuple(result.items()):
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


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

        if settings.use_markers:
            self.marker_encoder = MarkerEncoder(
                num_markers=settings.num_markers,
                context_dim=context_dim,
                input_features=settings.marker_input_features,
                hidden_dim=settings.marker_hidden_dim,
                marker_mean=settings.marker_mean,
                marker_std=settings.marker_std,
            )
        else:
            self.marker_encoder = None

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
        if settings.marker_temporal_embedding:
            self.marker_temporal_embedding: nn.Module | None = nn.Embedding(
                settings.marker_history_length,
                context_dim,
            )
        else:
            self.marker_temporal_embedding = None

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
    ) -> tuple[Tensor, Tensor]:
        """Return marker tokens/mask with shapes ``[B,S,H,D]`` and ``[B,S,H]``."""

        if self.marker_encoder is None:
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
        token_mask = (
            sensor_mask.unsqueeze(-1)
            & history_mask
            & marker_mask.any(dim=-1)
        )
        tokens = self.marker_encoder(features, marker_mask)
        if self.marker_modality_embedding is not None:
            tokens = tokens + self.marker_modality_embedding.to(dtype=tokens.dtype)
        tokens = tokens + self._sensor_embedding(tokens.dtype, tokens.device)[None, :, None, :]
        if self.marker_temporal_embedding is not None:
            temporal_indices = torch.arange(history_length, device=tokens.device)
            temporal = self.marker_temporal_embedding(temporal_indices).to(dtype=tokens.dtype)
            tokens = tokens + temporal[None, None, :, :]
        tokens = tokens * token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, token_mask

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
        return tokens.flatten(1, 2), rgb_mask.flatten(1, 2)


_LEGACY_MARKER_REINITIALIZED_SUFFIXES = (
    "marker_encoder.encoder.0.weight",
    "marker_encoder.encoder.0.bias",
    "marker_encoder.marker_mean",
    "marker_encoder.marker_std",
    "marker_temporal_embedding.weight",
)


def load_vtla_checkpoint_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allow_legacy_marker_reinit: bool = False,
) -> dict[str, Any]:
    """Load a VTLA checkpoint with a narrow, explicit legacy marker policy.

    A one-token ``[dx,dy,vx,vy]`` checkpoint differs only in the first marker
    MLP weight and its four-channel normalization buffers, and has no temporal
    embedding.  With explicit opt-in those marker-only values are omitted so
    the new two-channel layer, buffers, and temporal embedding keep their
    configuration-derived initialization.  Every non-marker mismatch remains
    a hard error.
    """

    target = model.state_dict()
    supplied: MutableMapping[str, Tensor] = dict(state_dict)
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

    reinitialized: list[str] = []
    if shape_mismatches:
        for name in target:
            if any(name.endswith(suffix) for suffix in _LEGACY_MARKER_REINITIALIZED_SUFFIXES):
                supplied.pop(name, None)
                reinitialized.append(name)

    incompatible = model.load_state_dict(supplied, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(set(unexpected) | set(incompatible.unexpected_keys))
    allowed_missing = set(reinitialized)
    unsupported_missing = sorted(set(missing) - allowed_missing)
    if unsupported_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible after marker migration: "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    return {
        "legacy_marker_reinitialized": bool(reinitialized),
        "reinitialized_keys": sorted(reinitialized),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
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
    "MarkerEncoder",
    "TactileTokenEncoder",
    "TactileVTLAConfig",
    "concatenate_vtla_context",
    "load_vtla_checkpoint_state_dict",
    "load_marker_statistics",
    "validate_marker_displacement_history",
]
