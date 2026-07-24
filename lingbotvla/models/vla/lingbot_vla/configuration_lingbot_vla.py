# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from transformers import AutoConfig, PretrainedConfig


DEFAULT_TACTILE_PARAMS: Dict[str, Any] = {
    "schema_version": 1,
    "adapter_version": 1,
    "history_steps": 4,
    "history_stride": 1,
    "history_frequency_hz": 30.0,
    "max_timestamp_skew_s": 0.05,
    # Use the repository's self-contained DINOv2 ViT-S/14 implementation.  Its
    # pretrained weights must be supplied as a local file; no network fallback
    # is permitted during training or deployment.
    "rgb_backbone": "dinov2_vits14",
    "rgb_backbone_path": None,
    "freeze_rgb_backbone": True,
    "rgb_input_size": 224,
    "rgb_patch_size": 14,
    "rgb_encoder_dim": 384,
    "rgb_encoder_layers": 4,
    "rgb_encoder_heads": 6,
    "rgb_tokens": 8,
    "rgb_mean": [0.485, 0.456, 0.406],
    "rgb_std": [0.229, 0.224, 0.225],
    "marker_count": 48,
    "marker_dim": 2,
    "marker_encoder_dim": 256,
    "marker_encoder_layers": 2,
    "marker_encoder_heads": 8,
    "marker_tokens": 8,
    "marker_reference_mode": "index",
    "marker_include_velocity": True,
    "marker_normalization": "image_size_xy",
    "marker_normalization_size_xy": [640, 480],
    "fusion_type": "gated_prefix",
    "gate_init": -4.0,
    "teacher_attention_to_tactile": False,
    "missing_policy": "mask",
    "rgb_dropout_prob": 0.10,
    "marker_dropout_prob": 0.10,
    "all_tactile_dropout_prob": 0.05,
    "auxiliary_contact_loss_weight": 0.0,
    "auxiliary_slip_loss_weight": 0.0,
    "auxiliary_phase_loss_weight": 0.0,
}


def _normalize_tactile_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = deepcopy(DEFAULT_TACTILE_PARAMS)
    if params:
        normalized.update(deepcopy(params))

    positive_int_keys = (
        "history_steps",
        "history_stride",
        "rgb_input_size",
        "rgb_patch_size",
        "rgb_encoder_dim",
        "rgb_encoder_layers",
        "rgb_encoder_heads",
        "rgb_tokens",
        "marker_count",
        "marker_dim",
        "marker_encoder_dim",
        "marker_encoder_layers",
        "marker_encoder_heads",
        "marker_tokens",
    )
    for key in positive_int_keys:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"tactile_params.{key} must be a positive integer, got {value!r}.")

    if normalized["rgb_input_size"] % normalized["rgb_patch_size"] != 0:
        raise ValueError(
            "tactile_params.rgb_input_size must be divisible by rgb_patch_size "
            f"(got {normalized['rgb_input_size']} and {normalized['rgb_patch_size']})."
        )
    if normalized["rgb_encoder_dim"] % normalized["rgb_encoder_heads"] != 0:
        raise ValueError("tactile RGB encoder_dim must be divisible by encoder_heads.")
    if normalized["marker_encoder_dim"] % normalized["marker_encoder_heads"] != 0:
        raise ValueError("tactile marker encoder_dim must be divisible by encoder_heads.")

    if normalized["fusion_type"] != "gated_prefix":
        raise ValueError(
            "Only tactile_params.fusion_type='gated_prefix' is supported in V2, "
            f"got {normalized['fusion_type']!r}."
        )
    if normalized["missing_policy"] not in ("mask", "error"):
        raise ValueError("tactile_params.missing_policy must be 'mask' or 'error'.")
    if normalized["marker_normalization"] != "image_size_xy":
        raise ValueError(
            "Only marker_normalization='image_size_xy' is accepted; this prevents "
            "the historical /400 train/inference scale mismatch."
        )
    marker_size = normalized["marker_normalization_size_xy"]
    if (
        not isinstance(marker_size, (list, tuple))
        or len(marker_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in marker_size
        )
    ):
        raise ValueError(
            "tactile_params.marker_normalization_size_xy must be [width, height] "
            "with two positive values."
        )
    normalized["marker_normalization_size_xy"] = [
        int(marker_size[0]),
        int(marker_size[1]),
    ]
    if normalized["marker_reference_mode"] not in ("index", "auto"):
        raise ValueError("tactile_params.marker_reference_mode must be 'index' or 'auto'.")

    for key in ("rgb_dropout_prob", "marker_dropout_prob", "all_tactile_dropout_prob"):
        value = float(normalized[key])
        if not 0.0 <= value < 1.0:
            raise ValueError(f"tactile_params.{key} must be in [0, 1), got {value}.")
        normalized[key] = value

    for key in ("rgb_mean", "rgb_std"):
        value = normalized[key]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"tactile_params.{key} must contain three RGB values.")
        normalized[key] = [float(item) for item in value]
    if any(value <= 0 for value in normalized["rgb_std"]):
        raise ValueError("tactile_params.rgb_std values must be positive.")
    history_frequency_hz = normalized["history_frequency_hz"]
    if (
        not isinstance(history_frequency_hz, (int, float))
        or isinstance(history_frequency_hz, bool)
        or history_frequency_hz <= 0
    ):
        raise ValueError("tactile_params.history_frequency_hz must be positive.")
    normalized["history_frequency_hz"] = float(history_frequency_hz)

    for key in (
        "auxiliary_contact_loss_weight",
        "auxiliary_slip_loss_weight",
        "auxiliary_phase_loss_weight",
    ):
        if float(normalized[key]) != 0.0:
            raise ValueError(
                f"tactile_params.{key} is not implemented without reliable labels; keep it at 0.0."
            )
    return normalized

class LingbotVLAConfig(PretrainedConfig):
    """Configuration class for Lingbot-VLA.
    This is the configuration class to store the configuration of a [`Lingbot-VLA`].
    """

    model_type = "lingbotvla"
    is_composition = True

    def __init__(
        self,
        vlm_repo_id: Optional[str] = None,
        expert_vision_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,

        post_training: bool = False,
        adanorm_time: bool = False,
        split_gate_liner: bool = False,
        nosplit_gate_liner: bool = False,
        separate_time_proj: bool = False,
        final_norm_adanorm: bool = False,

        enable_expert_vision: bool = False,
        expert_vision_type: Optional[str] = None,
        freeze_vision_encoder: bool = False,

        incremental_training: bool = False,
        depth_incremental_training: bool = False,
        reinit_mismatched_weights: bool = False,

        action_dim: int = 14,
        max_action_dim: int = 14,
        max_state_dim: int = 14,
        chunk_size: int = 50,
        vlm_causal: bool = False,
        tokenizer_max_length: int = 48,
        loss_type: str = "fm",
        norm_qkv: bool = False,
        align_params: Optional[Dict[str, Any]] = None,
        use_compile: bool = False,

        use_moe: bool = False,
        token_moe_layers: Optional[list] = None,
        token_num_experts: int = 32,
        token_top_k: int = 1,
        token_moe_intermediate_size: int = 256,
        token_shared_intermediate_size: int = 256,
        bias_update_speed: float = 0.001,
        sequence_wise_loss_coeff: float = 0.001,
        sequence_wise_mode: str = "per_sequence",
        router_z_loss_coeff: float = 0.0,
        router_activation: str = "softmax",
        routed_scaling_factor: float = 1.0,
        use_shared_expert_gate: bool = True,
        moe_implementation: Optional[Literal[None, "eager", "fused"]] = None,
        split_fused_experts_from_decoder_fsdp: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 16,
        action_num_key_value_heads: int = 2,
        action_head_dim: int = 128,
        action_fp32: bool = False,
        use_qwen3_chat_template: bool = False,
        return_image_grid_thw: bool = False,
        qwen3vl_use_vision_boundaries: bool = False,
        precompute_grid_thw: bool = False,
        use_qwen3_fixed_grid_cache: bool = False,

        use_lm_head: bool = False,
        vocab_size: int = 0,
        vit_attn_implementation: str = "flash_attention_2",
        attention_implementation: str = "flex",

        train_expert_only: bool = False,
        train_state_proj: bool = True,

        # Dormant by default.  When both switches are false no tactile module is
        # instantiated, preserving the pre-tactile module graph and state_dict.
        tactile_rgb_enabled: bool = False,
        tactile_marker_enabled: bool = False,
        tactile_train_stage: Literal["adapters", "expert", "full"] = "full",
        tactile_params: Optional[Dict[str, Any]] = None,
        force_mask_tactile_rgb: bool = False,
        force_mask_tactile_marker: bool = False,

        **kwargs
    ):
        super().__init__()
        if moe_implementation is None:
            moe_implementation = kwargs.pop("_moe_implementation", None)
        self.architectures = ["LingbotVlaPolicy"]
        self.train_state_proj = train_state_proj
        self.train_expert_only = train_expert_only
        self.use_cache = False
        self.attention_implementation = attention_implementation
        self.num_steps = 10
        self.n_obs_steps = 1

        assert not (split_gate_liner and nosplit_gate_liner), \
            "split_gate_liner and nosplit_gate_liner can not be both True."

        self.vlm_repo_id = vlm_repo_id
        self.expert_vision_path = expert_vision_path
        self.tokenizer_path = tokenizer_path
        self.post_training = post_training
        self.adanorm_time = adanorm_time
        self.split_gate_liner = split_gate_liner
        self.nosplit_gate_liner = nosplit_gate_liner
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.incremental_training = incremental_training
        self.depth_incremental_training = depth_incremental_training
        self.reinit_mismatched_weights = reinit_mismatched_weights
        self.norm_qkv = norm_qkv
        self.use_compile = use_compile
        self.loss_type = loss_type
        self.separate_time_proj = separate_time_proj
        self.final_norm_adanorm = final_norm_adanorm
        self.freeze_vision_encoder = freeze_vision_encoder
        self.tokenizer_max_length = tokenizer_max_length
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.vlm_causal = vlm_causal
        self.align_params = align_params
        self.use_moe = use_moe
        if self.use_moe:
            self.token_moe_layers = token_moe_layers
            self.token_num_experts = token_num_experts
            self.token_top_k = token_top_k
            self.token_moe_intermediate_size = token_moe_intermediate_size
            self.token_shared_intermediate_size = token_shared_intermediate_size
        self.bias_update_speed = bias_update_speed
        self.sequence_wise_loss_coeff = sequence_wise_loss_coeff
        self.sequence_wise_mode = sequence_wise_mode
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_activation = router_activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert_gate = use_shared_expert_gate
        self.moe_implementation = moe_implementation
        if moe_implementation is not None:
            if moe_implementation not in ("eager", "fused"):
                raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
            self._moe_implementation = moe_implementation
        self.split_fused_experts_from_decoder_fsdp = split_fused_experts_from_decoder_fsdp
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        self.action_fp32 = action_fp32
        self.use_qwen3_chat_template = use_qwen3_chat_template
        self.return_image_grid_thw = return_image_grid_thw
        self.qwen3vl_use_vision_boundaries = qwen3vl_use_vision_boundaries
        self.precompute_grid_thw = precompute_grid_thw
        self.use_qwen3_fixed_grid_cache = use_qwen3_fixed_grid_cache
        self.use_lm_head = use_lm_head
        if vocab_size == 0:
            if vlm_repo_id and 'paligemma' in vlm_repo_id.lower():
                self.vocab_size = 257216
            elif vlm_repo_id and 'qwen' in vlm_repo_id.lower():
                self.vocab_size = 151936
            else:
                self.vocab_size = 257152
        else:
            self.vocab_size = vocab_size
        self.vit_attn_implementation = vit_attn_implementation
        self.tactile_rgb_enabled = bool(tactile_rgb_enabled)
        self.tactile_marker_enabled = bool(tactile_marker_enabled)
        if tactile_train_stage not in ("adapters", "expert", "full"):
            raise ValueError(
                "tactile_train_stage must be 'adapters', 'expert', or 'full', "
                f"got {tactile_train_stage!r}."
            )
        self.tactile_train_stage = tactile_train_stage
        if tactile_train_stage != "full" and not (
            self.tactile_rgb_enabled or self.tactile_marker_enabled
        ):
            raise ValueError(
                "tactile_train_stage adapters/expert requires at least one tactile branch."
            )
        self.tactile_params = _normalize_tactile_params(tactile_params)
        self.force_mask_tactile_rgb = bool(force_mask_tactile_rgb)
        self.force_mask_tactile_marker = bool(force_mask_tactile_marker)
        if self.force_mask_tactile_rgb and not self.tactile_rgb_enabled:
            raise ValueError("force_mask_tactile_rgb requires tactile_rgb_enabled=True.")
        if self.force_mask_tactile_marker and not self.tactile_marker_enabled:
            raise ValueError("force_mask_tactile_marker requires tactile_marker_enabled=True.")

class LingbotVLAV2Config(LingbotVLAConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("attention_implementation", "flex_cached")
        kwargs.setdefault("vit_attn_implementation", "flash_attention_2")
        kwargs.setdefault("action_num_attention_heads", 32)
        kwargs.setdefault("action_num_key_value_heads", 8)
        kwargs.setdefault("action_head_dim", 128)
        kwargs.setdefault("expert_hidden_size", 768)
        kwargs.setdefault("use_qwen3_chat_template", True)
        kwargs.setdefault("return_image_grid_thw", True)
        kwargs.setdefault("qwen3vl_use_vision_boundaries", True)
        kwargs.setdefault("use_qwen3_fixed_grid_cache", True)
        super().__init__(**kwargs)
        self.architectures = ["LingbotVlaV2Policy"]
        self.vlm_family = "qwen3_vl"


ConfigClass = [LingbotVLAConfig, LingbotVLAV2Config]
__all__ = [
    "DEFAULT_TACTILE_PARAMS",
    "LingbotVLAConfig",
    "LingbotVLAV2Config",
]
