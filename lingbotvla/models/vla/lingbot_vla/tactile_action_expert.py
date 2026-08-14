"""Configuration and cache primitives for three-stream tactile refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor, nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .utils import create_sinusoidal_pos_embedding

if TYPE_CHECKING:
    from .qwen2_action_expert import Qwen2ForCausalLM


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class TactileExpertSettings:
    hidden_size: int = 768
    intermediate_size: int = 1536
    num_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    init_from_action_expert: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileExpertSettings":
        values = dict(value or {})
        result = cls(
            hidden_size=int(values.get("hidden_size", 768)),
            intermediate_size=int(values.get("intermediate_size", 1536)),
            num_layers=int(values.get("num_layers", 36)),
            num_attention_heads=int(values.get("num_attention_heads", 32)),
            num_key_value_heads=int(values.get("num_key_value_heads", 8)),
            head_dim=int(values.get("head_dim", 128)),
            init_from_action_expert=_as_bool(
                values.get("init_from_action_expert"), True
            ),
        )
        if min(
            result.hidden_size,
            result.intermediate_size,
            result.num_layers,
            result.num_attention_heads,
            result.num_key_value_heads,
            result.head_dim,
        ) <= 0:
            raise ValueError("All tactile expert dimensions must be positive")
        if result.num_attention_heads % result.num_key_value_heads != 0:
            raise ValueError(
                "tactile expert attention heads must be divisible by key/value heads"
            )
        return result


@dataclass(frozen=True)
class TactileSequenceSettings:
    include_marker_tokens: bool = True
    include_time_token: bool = True
    include_action_tokens: bool = True
    include_latest_state: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileSequenceSettings":
        values = dict(value or {})
        result = cls(
            include_marker_tokens=_as_bool(values.get("include_marker_tokens"), True),
            include_time_token=_as_bool(values.get("include_time_token"), True),
            include_action_tokens=_as_bool(values.get("include_action_tokens"), True),
            include_latest_state=_as_bool(values.get("include_latest_state"), False),
        )
        if not (
            result.include_marker_tokens
            and result.include_time_token
            and result.include_action_tokens
        ):
            raise ValueError(
                "three_stream_mot requires Marker, flow-time, and X_tau tokens"
            )
        if result.include_latest_state:
            raise ValueError("include_latest_state is not supported in the first version")
        return result


@dataclass(frozen=True)
class TactileInferenceSettings:
    total_steps: int = 10
    slow_steps: int = 4
    tactile_steps: int = 6
    gate_off_behavior: str = "action_fallback"
    clone_slow_cache_each_fast_tick: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileInferenceSettings":
        values = dict(value or {})
        result = cls(
            total_steps=int(values.get("total_steps", 10)),
            slow_steps=int(values.get("slow_steps", 4)),
            tactile_steps=int(values.get("tactile_steps", 6)),
            gate_off_behavior=str(
                values.get("gate_off_behavior", "action_fallback")
            ),
            clone_slow_cache_each_fast_tick=_as_bool(
                values.get("clone_slow_cache_each_fast_tick"), True
            ),
        )
        if result.total_steps <= 0 or result.slow_steps <= 0 or result.tactile_steps <= 0:
            raise ValueError("Cascaded inference step counts must be positive")
        if result.slow_steps + result.tactile_steps != result.total_steps:
            raise ValueError("slow_steps + tactile_steps must equal total_steps")
        if result.gate_off_behavior != "action_fallback":
            raise ValueError("Only gate_off_behavior=action_fallback is supported")
        if not result.clone_slow_cache_each_fast_tick:
            raise ValueError("Fast refinement requires immutable cloned slow caches")
        return result


@dataclass(frozen=True)
class TactileTrainingSettings:
    freeze_vlm: bool = True
    freeze_action_expert: bool = True
    train_marker_encoder: bool = True
    train_marker_to_tactile_proj: bool = True
    train_tactile_expert: bool = True
    train_tactile_flow_heads: bool = True
    tactile_loss_weight: float = 1.0
    rollout_boundary_exposure_prob: float = 0.25
    action_full_range_loss_weight: float = 0.0
    action_full_range_exposure_prob: float = 0.1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileTrainingSettings":
        values = dict(value or {})
        result = cls(
            freeze_vlm=_as_bool(values.get("freeze_vlm"), True),
            freeze_action_expert=_as_bool(values.get("freeze_action_expert"), True),
            train_marker_encoder=_as_bool(values.get("train_marker_encoder"), True),
            train_marker_to_tactile_proj=_as_bool(
                values.get("train_marker_to_tactile_proj"), True
            ),
            train_tactile_expert=_as_bool(values.get("train_tactile_expert"), True),
            train_tactile_flow_heads=_as_bool(
                values.get("train_tactile_flow_heads"), True
            ),
            tactile_loss_weight=float(values.get("tactile_loss_weight", 1.0)),
            rollout_boundary_exposure_prob=float(
                values.get("rollout_boundary_exposure_prob", 0.25)
            ),
            action_full_range_loss_weight=float(
                values.get("action_full_range_loss_weight", 0.0)
            ),
            action_full_range_exposure_prob=float(
                values.get("action_full_range_exposure_prob", 0.1)
            ),
        )
        for name in (
            "rollout_boundary_exposure_prob",
            "action_full_range_exposure_prob",
        ):
            if not 0.0 <= getattr(result, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if result.tactile_loss_weight <= 0 or result.action_full_range_loss_weight < 0:
            raise ValueError("Cascaded loss weights must be non-negative")
        if (
            not result.freeze_action_expert
            and result.action_full_range_loss_weight <= 0
        ):
            raise ValueError(
                "Trainable Action Expert requires action_full_range_loss_weight > 0 "
                "so gate-off fallback retains the full tau range"
            )
        return result


@dataclass(frozen=True)
class TactileRefinementConfig:
    enabled: bool = False
    architecture: str = "three_stream_mot"
    mode: str = "cascaded_flow"
    tau_split: float = 0.6
    expert: TactileExpertSettings = field(default_factory=TactileExpertSettings)
    sequence: TactileSequenceSettings = field(default_factory=TactileSequenceSettings)
    inference: TactileInferenceSettings = field(default_factory=TactileInferenceSettings)
    training: TactileTrainingSettings = field(default_factory=TactileTrainingSettings)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TactileRefinementConfig":
        values = dict(value or {})
        result = cls(
            enabled=_as_bool(values.get("enabled"), False),
            architecture=str(values.get("architecture", "three_stream_mot")),
            mode=str(values.get("mode", "cascaded_flow")),
            tau_split=float(values.get("tau_split", 0.6)),
            expert=TactileExpertSettings.from_mapping(values.get("expert")),
            sequence=TactileSequenceSettings.from_mapping(values.get("sequence")),
            inference=TactileInferenceSettings.from_mapping(values.get("inference")),
            training=TactileTrainingSettings.from_mapping(values.get("training")),
        )
        if result.architecture != "three_stream_mot":
            raise ValueError("tactile_refinement.architecture must be three_stream_mot")
        if result.mode != "cascaded_flow":
            raise ValueError("tactile_refinement.mode must be cascaded_flow")
        expected_split = 1.0 - (
            result.inference.slow_steps / result.inference.total_steps
        )
        if abs(result.tau_split - expected_split) > 1e-6:
            raise ValueError(
                "tau_split must equal 1 - slow_steps / total_steps "
                f"(expected {expected_split}, got {result.tau_split})"
            )
        if not 0.0 < result.tau_split < 1.0:
            raise ValueError("tau_split must be in (0,1)")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TactileTimeEmbedder(nn.Module):
    """Independent scalar flow-time embedding used by the tactile stream."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, timestep: Tensor, *, dtype: torch.dtype) -> Tensor:
        embedding = create_sinusoidal_pos_embedding(
            timestep.float(),
            self.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(dtype=dtype)
        return self.out_proj(torch.nn.functional.silu(self.in_proj(embedding)))[:, None]


@dataclass(frozen=True)
class CascadedSlowPlan:
    """Immutable boundary state and reusable P/A per-layer KV cache."""

    x_split: Tensor
    tau_split: float
    noise: Tensor
    prefix_past_key_values: dict[int, dict[str, Tensor]]
    past_key_values: dict[int, dict[str, Tensor]]
    prefix_pad_masks: Tensor
    action_pad_masks: Tensor
    prefix_position_ids: Tensor
    prefix_len: int
    action_len: int
    created_at_s: float = 0.0
    prefix_created_at_s: float = 0.0
    state_at_plan: Tensor | None = None
    executed_offset: int = 0
    plan_version: int = 0
    scene_version: int = 0


def clone_kv_cache(
    cache: dict[int, dict[str, Tensor]],
    *,
    clone_tensors: bool = False,
) -> dict[int, dict[str, Tensor]]:
    """Clone cache containers; optionally clone tensors for mutation tests."""

    return {
        layer: {
            name: value.clone() if clone_tensors else value
            for name, value in values.items()
        }
        for layer, values in cache.items()
    }


def build_tactile_expert(
    settings: TactileExpertSettings,
    *,
    use_cache: bool,
    eval_mode: bool,
) -> "Qwen2ForCausalLM":
    from .qwen2_action_expert import Qwen2ForCausalLM

    config = Qwen2Config(
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=settings.hidden_size,
        head_dim=settings.head_dim,
        initializer_range=0.02,
        intermediate_size=settings.intermediate_size,
        max_position_embeddings=32768,
        num_attention_heads=settings.num_attention_heads,
        num_hidden_layers=settings.num_layers,
        num_key_value_heads=settings.num_key_value_heads,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
        use_cache=use_cache,
        vocab_size=1,
    )
    # The decoder's built-in attention is bypassed by LingBot Joint Attention.
    # Keep construction independent of the optional flash_attn package.
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config, eval=eval_mode)


@torch.no_grad()
def copy_matching_action_weights(action_expert: nn.Module, tactile_expert: nn.Module) -> list[str]:
    """Copy shape-compatible decoder weights without sharing Parameter objects."""

    source = action_expert.state_dict()
    target = tactile_expert.state_dict()
    copied = []
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name].copy_(value)
            copied.append(name)
    return copied


__all__ = [
    "CascadedSlowPlan",
    "TactileExpertSettings",
    "TactileInferenceSettings",
    "TactileRefinementConfig",
    "TactileSequenceSettings",
    "TactileTimeEmbedder",
    "TactileTrainingSettings",
    "build_tactile_expert",
    "clone_kv_cache",
    "copy_matching_action_weights",
]
