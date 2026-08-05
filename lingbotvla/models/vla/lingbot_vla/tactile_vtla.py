"""Minimal TacThru token encoders for LingBot-VLA v2.

This module intentionally contains no action or Flow Matching code.  It turns
TacThru RGB embeddings and marker coordinates into fixed-length prefix tokens
that can be appended to the existing vision-language prefix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    a nested ``marker`` object with ``mean``/``std``.  Four values are required
    in ``[dx, dy, vx, vy]`` order.
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
    mean_tuple = _as_float_tuple(mean, name="marker_mean", length=4)
    std_tuple = _as_float_tuple(std, name="marker_std", length=4)
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
    marker_input_features: int = 4
    marker_hidden_dim: int = 512
    marker_tokens_per_sensor: int = 1
    add_modality_embedding: bool = True
    add_sensor_side_embedding: bool = True
    marker_stats_path: str | None = None
    marker_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    marker_std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    require_marker_stats: bool = False
    sensor_names: tuple[str, ...] = ("left", "right")
    rgb_keys: tuple[str, ...] = (
        "observation.images.tacthru_l_rgb",
        "observation.images.tacthru_r_rgb",
    )
    marker_positions_keys: tuple[str, ...] = (
        "observation.tactile.marker_positions_left",
        "observation.tactile.marker_positions_right",
    )
    marker_reference_keys: tuple[str, ...] = (
        "observation.tactile.marker_reference_left",
        "observation.tactile.marker_reference_right",
    )
    marker_valid_mask_keys: tuple[str, ...] = (
        "observation.tactile.marker_valid_left",
        "observation.tactile.marker_valid_right",
    )
    marker_flow_keys: tuple[str, ...] = (
        "observation.tactile.marker_flow_left",
        "observation.tactile.marker_flow_right",
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
        marker_input_features = int(values.get("marker_input_features", 4))
        marker_tokens_per_sensor = int(values.get("marker_tokens_per_sensor", 1))

        if num_sensors <= 0:
            raise ValueError("tactile.num_sensors must be positive")
        if num_markers <= 0:
            raise ValueError("tactile.num_markers must be positive")
        if enabled and not (use_rgb or use_markers):
            raise ValueError("tactile.enabled=true requires use_rgb or use_markers")
        if not _read_bool(values.get("share_vision_encoder"), True):
            raise ValueError("The minimal VTLA implementation requires share_vision_encoder=true")
        if marker_input_features != 4:
            raise ValueError("marker_input_features must be 4 ([dx, dy, vx, vy])")
        if marker_tokens_per_sensor != 1:
            raise ValueError("The minimal MarkerEncoder supports exactly one token per sensor")
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
                mean_value, name="marker_mean", length=4
            )
            marker_std = _as_float_tuple(std_value, name="marker_std", length=4)
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
                mean_value or (0.0, 0.0, 0.0, 0.0),
                name="marker_mean",
                length=4,
            )
            marker_std = _as_float_tuple(
                std_value or (1.0, 1.0, 1.0, 1.0),
                name="marker_std",
                length=4,
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
            marker_input_features=marker_input_features,
            marker_hidden_dim=int(values.get("marker_hidden_dim", 512)),
            marker_tokens_per_sensor=marker_tokens_per_sensor,
            add_modality_embedding=_read_bool(values.get("add_modality_embedding"), True),
            add_sensor_side_embedding=_read_bool(values.get("add_sensor_side_embedding"), True),
            marker_stats_path=str(stats_path) if stats_path is not None else None,
            marker_mean=marker_mean,
            marker_std=marker_std,
            require_marker_stats=require_stats,
            sensor_names=sensor_names,
            rgb_keys=_keys("rgb_keys", cls.rgb_keys),
            marker_positions_keys=_keys("marker_positions_keys", cls.marker_positions_keys),
            marker_reference_keys=_keys("marker_reference_keys", cls.marker_reference_keys),
            marker_valid_mask_keys=_keys("marker_valid_mask_keys", cls.marker_valid_mask_keys),
            marker_flow_keys=_keys("marker_flow_keys", cls.marker_flow_keys),
            debug_shapes=_read_bool(values.get("debug_shapes"), False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-safe values for Hugging Face config serialization."""

        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


def marker_features_from_positions(
    marker_positions: Tensor,
    marker_reference: Tensor,
    previous_marker_positions: Tensor | None,
    marker_valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Construct ``[dx, dy, vx, vy]`` features and a bool marker mask.

    Inputs may be ``[B,N,2]`` or ``[B,S,N,2]``.  When a previous position is
    unavailable (the first frame of an episode), velocity is deterministically
    set to zero by reusing the current position.
    """

    if marker_positions.ndim not in (3, 4) or marker_positions.shape[-1] != 2:
        raise ValueError(
            "marker_positions must have shape [B,N,2] or [B,S,N,2], "
            f"got {tuple(marker_positions.shape)}"
        )
    if marker_reference.shape != marker_positions.shape:
        raise ValueError("marker_reference must have the same shape as marker_positions")
    if previous_marker_positions is None:
        previous_marker_positions = marker_positions
    if previous_marker_positions.shape != marker_positions.shape:
        raise ValueError("previous_marker_positions must have the same shape as marker_positions")
    if not torch.is_floating_point(marker_positions):
        raise ValueError("marker_positions must be floating point")
    if not (
        torch.isfinite(marker_positions).all()
        and torch.isfinite(marker_reference).all()
        and torch.isfinite(previous_marker_positions).all()
    ):
        raise ValueError("marker positions/reference/previous must be finite")

    expected_mask_shape = marker_positions.shape[:-1]
    if marker_valid_mask is None:
        marker_valid_mask = torch.ones(
            expected_mask_shape,
            dtype=torch.bool,
            device=marker_positions.device,
        )
    elif tuple(marker_valid_mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            f"marker_valid_mask must have shape {tuple(expected_mask_shape)}, "
            f"got {tuple(marker_valid_mask.shape)}"
        )
    else:
        marker_valid_mask = marker_valid_mask.to(device=marker_positions.device, dtype=torch.bool)

    displacement = marker_positions - marker_reference
    velocity = marker_positions - previous_marker_positions
    features = torch.cat([displacement, velocity], dim=-1)
    features = features * marker_valid_mask.unsqueeze(-1).to(dtype=features.dtype)
    return features, marker_valid_mask


class MarkerEncoder(nn.Module):
    """Shared MLP that emits one marker token per TacThru sensor."""

    def __init__(
        self,
        num_markers: int,
        context_dim: int,
        input_features: int = 4,
        hidden_dim: int = 512,
        marker_mean: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        marker_std: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if num_markers <= 0 or context_dim <= 0 or hidden_dim <= 0:
            raise ValueError("num_markers, context_dim, and hidden_dim must be positive")
        if input_features != 4:
            raise ValueError("MarkerEncoder input_features must be 4")

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
        """Encode ``[B,N,4]`` or ``[B,S,N,4]`` into marker tokens."""

        if marker_features.ndim not in (3, 4):
            raise ValueError("marker_features must have shape [B,N,F] or [B,S,N,F]")
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
        if marker_features.ndim == 3:
            return tokens.unsqueeze(1)
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
        marker_positions: Tensor,
        marker_reference: Tensor,
        previous_marker_positions: Tensor | None,
        marker_valid_mask: Tensor | None,
        tactile_sensor_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return marker tokens/mask with shapes ``[B,S,D]`` and ``[B,S]``."""

        if self.marker_encoder is None:
            raise RuntimeError("Marker encoding is disabled")
        if marker_positions.ndim != 4:
            raise ValueError("marker_positions must have shape [B,S,N,2]")
        batch_size, num_sensors = marker_positions.shape[:2]
        if num_sensors != self.settings.num_sensors:
            raise ValueError(
                f"Expected {self.settings.num_sensors} tactile sensors, got {num_sensors}"
            )
        features, marker_mask = marker_features_from_positions(
            marker_positions,
            marker_reference,
            previous_marker_positions,
            marker_valid_mask,
        )
        sensor_mask = self._sensor_mask(
            tactile_sensor_mask,
            batch_size=batch_size,
            num_sensors=num_sensors,
            device=marker_positions.device,
        )
        token_mask = sensor_mask & marker_mask.any(dim=-1)
        tokens = self.marker_encoder(features, marker_mask)
        if self.marker_modality_embedding is not None:
            tokens = tokens + self.marker_modality_embedding.to(dtype=tokens.dtype)
        tokens = tokens + self._sensor_embedding(tokens.dtype, tokens.device).unsqueeze(0)
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
    "load_marker_statistics",
    "marker_features_from_positions",
]
