"""Fast tactile expert for cascaded VTLA flow matching."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class TactileExpertConfig:
    hidden_size: int = 768
    num_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 2048

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileExpertConfig":
        values = dict(value or {})
        result = cls(
            hidden_size=int(values.get("hidden_size", 768)),
            num_layers=int(values.get("num_layers", 6)),
            num_attention_heads=int(values.get("num_attention_heads", 8)),
            num_key_value_heads=int(values.get("num_key_value_heads", 2)),
            intermediate_size=int(values.get("intermediate_size", 2048)),
        )
        if min(
            result.hidden_size,
            result.num_layers,
            result.num_attention_heads,
            result.num_key_value_heads,
            result.intermediate_size,
        ) <= 0:
            raise ValueError("tactile_refinement.expert dimensions must be positive")
        if result.hidden_size % result.num_attention_heads:
            raise ValueError("Tactile expert hidden_size must divide num_attention_heads")
        if result.num_attention_heads % result.num_key_value_heads:
            raise ValueError(
                "Tactile expert num_attention_heads must be divisible by "
                "num_key_value_heads"
            )
        return result


@dataclass(frozen=True)
class TactileRefinementLossConfig:
    slow_weight: float = 1.0
    tactile_weight: float = 1.0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "TactileRefinementLossConfig":
        values = dict(value or {})
        result = cls(
            slow_weight=float(values.get("slow_weight", 1.0)),
            tactile_weight=float(values.get("tactile_weight", 1.0)),
        )
        if result.slow_weight < 0 or result.tactile_weight < 0:
            raise ValueError("Cascaded Flow Matching loss weights must be non-negative")
        if result.slow_weight == 0 and result.tactile_weight == 0:
            raise ValueError("At least one cascaded Flow Matching loss must be enabled")
        return result


@dataclass(frozen=True)
class TactileRefinementTrainingConfig:
    freeze_vlm: bool = True
    freeze_action_expert: bool = True
    train_marker_encoder: bool = True
    train_tactile_expert: bool = True

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "TactileRefinementTrainingConfig":
        values = dict(value or {})
        return cls(
            freeze_vlm=_read_bool(values.get("freeze_vlm"), True),
            freeze_action_expert=_read_bool(values.get("freeze_action_expert"), True),
            train_marker_encoder=_read_bool(values.get("train_marker_encoder"), True),
            train_tactile_expert=_read_bool(values.get("train_tactile_expert"), True),
        )


@dataclass(frozen=True)
class TactileRefinementInferenceConfig:
    total_steps: int = 10
    slow_steps: int = 6
    tactile_steps: int = 4

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "TactileRefinementInferenceConfig":
        values = dict(value or {})
        result = cls(
            total_steps=int(values.get("total_steps", 10)),
            slow_steps=int(values.get("slow_steps", 6)),
            tactile_steps=int(values.get("tactile_steps", 4)),
        )
        if min(result.total_steps, result.slow_steps, result.tactile_steps) <= 0:
            raise ValueError("Cascaded inference step counts must be positive")
        if result.slow_steps + result.tactile_steps != result.total_steps:
            raise ValueError("slow_steps + tactile_steps must equal total_steps")
        return result


@dataclass(frozen=True)
class TactileRefinementConfig:
    enabled: bool = False
    mode: str = "full_replan"
    tau_split: float = 0.4
    expert: TactileExpertConfig = field(default_factory=TactileExpertConfig)
    loss: TactileRefinementLossConfig = field(
        default_factory=TactileRefinementLossConfig
    )
    training: TactileRefinementTrainingConfig = field(
        default_factory=TactileRefinementTrainingConfig
    )
    inference: TactileRefinementInferenceConfig = field(
        default_factory=TactileRefinementInferenceConfig
    )
    use_slow_context: bool = True

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "TactileRefinementConfig":
        values = dict(value or {})
        mode = str(values.get("mode", "full_replan"))
        if mode not in {"full_replan", "cascaded_flow"}:
            raise ValueError(
                "tactile_refinement.mode must be full_replan or cascaded_flow"
            )
        context = dict(values.get("context") or {})
        result = cls(
            enabled=_read_bool(values.get("enabled"), False),
            mode=mode,
            tau_split=float(values.get("tau_split", 0.4)),
            expert=TactileExpertConfig.from_mapping(values.get("expert")),
            loss=TactileRefinementLossConfig.from_mapping(values.get("loss")),
            training=TactileRefinementTrainingConfig.from_mapping(
                values.get("training")
            ),
            inference=TactileRefinementInferenceConfig.from_mapping(
                values.get("inference")
            ),
            use_slow_context=_read_bool(context.get("use_slow_context"), True),
        )
        if not 0.0 < result.tau_split < 1.0:
            raise ValueError("tactile_refinement.tau_split must be in (0, 1)")
        tactile_fraction = result.inference.tactile_steps / result.inference.total_steps
        if not math.isclose(tactile_fraction, result.tau_split, abs_tol=1e-7):
            raise ValueError(
                "tactile_steps / total_steps must equal tau_split so the two "
                "Flow Matching intervals have no overlap or gap"
            )
        if result.enabled and result.mode == "cascaded_flow" and not result.use_slow_context:
            raise ValueError("Cascaded tactile refinement requires slow context")
        return result

    @property
    def cascaded_enabled(self) -> bool:
        return self.enabled and self.mode == "cascaded_flow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sinusoidal_time_embedding(time: Tensor, dim: int) -> Tensor:
    """Return a deterministic embedding for normalized Flow Matching time."""

    if time.ndim != 1:
        raise ValueError("Flow Matching time must be [B]")
    half = dim // 2
    if half == 0:
        return time[:, None]
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half, device=time.device)
    )
    angles = time.float()[:, None] * frequencies[None] * (2.0 * math.pi)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class GroupedQueryAttention(nn.Module):
    """Compact grouped-query attention with a query-specific context mask."""

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.groups = num_heads // num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("Attention query/context must be [B,L,D]")
        if query.shape[0] != context.shape[0]:
            raise ValueError("Attention query/context batch sizes differ")
        batch, query_len = query.shape[:2]
        context_len = context.shape[1]
        q = self.q_proj(query).view(
            batch, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(context).view(
            batch, context_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(context).view(
            batch, context_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        if context_mask is None:
            mask = torch.ones(
                batch, 1, 1, context_len, dtype=torch.bool, device=query.device
            )
        else:
            if tuple(context_mask.shape) != (batch, context_len):
                raise ValueError("context_mask must be [B,L_context]")
            mask = context_mask[:, None, None, :].to(device=query.device, dtype=torch.bool)
        # A masked softmax with an explicit zero-context branch keeps Gate OFF finite.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        output = torch.matmul(weights, v.float()).to(dtype=query.dtype)
        output = output.transpose(1, 2).reshape(batch, query_len, self.hidden_size)
        return self.out_proj(output)


class TactileActionExpertLayer(nn.Module):
    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        kwargs = dict(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        self.self_norm = nn.LayerNorm(config.hidden_size)
        self.self_attention = GroupedQueryAttention(**kwargs)
        self.marker_norm = nn.LayerNorm(config.hidden_size)
        self.marker_attention = GroupedQueryAttention(**kwargs)
        self.context_norm = nn.LayerNorm(config.hidden_size)
        self.context_attention = GroupedQueryAttention(**kwargs)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(
        self,
        hidden: Tensor,
        marker_tokens: Tensor,
        marker_mask: Tensor,
        slow_context: Tensor,
        slow_context_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(hidden)
        hidden = hidden + self.self_attention(normalized, normalized)
        hidden = hidden + self.marker_attention(
            self.marker_norm(hidden), marker_tokens, marker_mask
        )
        hidden = hidden + self.context_attention(
            self.context_norm(hidden), slow_context, slow_context_mask
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


class TactileActionExpert(nn.Module):
    """A small action stream that reads Marker and reusable Slow context tokens."""

    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TactileActionExpertLayer(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden: Tensor,
        marker_tokens: Tensor,
        marker_mask: Tensor,
        slow_context: Tensor,
        slow_context_mask: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                marker_tokens,
                marker_mask,
                slow_context,
                slow_context_mask,
            )
        return self.final_norm(hidden)


__all__ = [
    "GroupedQueryAttention",
    "TactileActionExpert",
    "TactileExpertConfig",
    "TactileRefinementConfig",
    "TactileRefinementInferenceConfig",
    "TactileRefinementLossConfig",
    "TactileRefinementTrainingConfig",
    "sinusoidal_time_embedding",
]
