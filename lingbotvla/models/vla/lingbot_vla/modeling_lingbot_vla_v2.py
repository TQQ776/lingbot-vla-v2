import time
from dataclasses import dataclass, field

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.cache_utils import Cache
from transformers.utils import logging

from .configuration_lingbot_vla import LingbotVLAV2Config
from .qwen3vl_in_vla import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    Qwen3VLPreTrainedModel,
    apply_rotary_pos_emb,
)
from .modeling_lingbot_vla import (
    AdaRMSNorm,
    FixAdaRMSNorm,
    replace_lnorm_with_adanorm,
    FlowMatching as FlowMatchingV1,
)
from .utils import (
    block_suffix_to_fv_,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    our_eager_attention_forward,
    prefix_query_segments,
    prefix_query_token_spans,
    sample_beta,
)
from .tactile_action_expert import (
    TactileActionExpert,
    TactileRefinementConfig,
    sinusoidal_position_embedding,
    sinusoidal_time_embedding,
)
from .tactile_vtla import TactileTokenEncoder, TactileVTLAConfig
from .flex_attention import build_block_mask, flex_attention_forward, flex_attention_with_block_mask
from lingbotvla.models.loader import LingBotVLAWeightLoader
from lingbotvla.ops.triton_moe_loss import triton_sequence_wise_balance_loss
from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
    Qwen2ForCausalLM,
    Qwen2TokenMoeBlock,
    Qwen2FusedExperts,
    FixQwen2RMSNorm,
)

try:
    from dinov3.hub.backbones import dinov3_vitb16
except Exception:
    dinov3_vitb16 = None


logger = logging.get_logger(__name__)


KVCache = dict[int, dict[str, Tensor]]


@dataclass(frozen=True)
class VTLASlowCache:
    """Immutable VLM cache for Scene -> Language -> tactile RGB."""

    past_key_values: KVCache
    pad_masks: Tensor
    att_masks: Tensor
    input_ids: Tensor
    position_ids: Tensor
    rope_grid_thw: Tensor
    prefix_len: int
    version: int = 0
    scene_timestamp: float | None = None
    tactile_rgb_timestamp: float | None = None
    profile_ms: dict[str, float] = field(default_factory=dict)
    final_hidden_states: Tensor | None = None
    includes_task_queries: bool = False


@dataclass(frozen=True)
class VTLAFastPrefix:
    """One-replan working prefix; it must never replace the slow cache."""

    past_key_values: KVCache
    pad_masks: Tensor
    position_ids: Tensor
    marker_token_count: int
    query_token_count: int
    profile_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SlowActionPlan:
    """Reusable intermediate action state at the cascaded FM split."""

    x_split: Tensor
    action_context: Tensor
    tau_split: float
    noise: Tensor
    state: Tensor
    slow_context: VTLASlowCache
    action_offset: int
    version: int
    profile_ms: dict[str, float] = field(default_factory=dict)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "flex_cached",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        freeze_vlm: bool = False,
        train_action_expert: bool = True,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        self.freeze_vlm = bool(freeze_vlm)
        self.train_action_expert = bool(train_action_expert)
        num_layers = 36

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.57.3",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        print(
            "=====Action Expert V2 init "
            f"{num_layers} Layers, hidden={expert_hidden_size}, "
            f"q_heads={action_num_attention_heads}, kv_heads={action_num_key_value_heads}, "
            f"head_dim={action_head_dim}.====="
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, eval=False):
        super().__init__(config=config)
        self.config = config
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
        if self.config.vocab_size not in (0, 257152):
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = "flash_attention_2"
        vlm_config.text_config._attn_implementation = "flash_attention_2"
        vlm_config.vision_config._attn_implementation = self.config.vit_attn_implementation
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = "flash_attention_2"
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        self.pos_embeds = None
        self.position_embeddings = None
        self.cu_seqlens = None
        self.visual_split_sizes = None
        self.visual_max_seqlen = None

        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if dinov3_vitb16 is None:
                raise ImportError("dinov3 is required when enable_expert_vision=True")
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )

        self.attention_interface = self.get_attention_interface()
        self.set_requires_grad()

    def _install_moe_blocks(self):
        if not getattr(self.config, "use_moe", False):
            return
        bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
        hidden_size = self.config.qwen_expert_config.hidden_size
        token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

        _moe_impl = getattr(self.config, "_moe_implementation", None)

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=getattr(self.config, "token_num_experts", 32),
                num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                norm_topk_prob=True,
                hidden_size=hidden_size,
                moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                shared_expert_intermediate_size=getattr(self.config, "token_shared_intermediate_size", 256),
                output_router_logits=False,
            )
            token_config.bias_update_speed = bias_update_speed
            token_config._moe_implementation = _moe_impl
            token_config.router_activation = getattr(self.config, "router_activation", "softmax")
            token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
            token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
            for idx in token_moe_layers:
                self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
            for params in self.qwenvl.visual.parameters():
                params.requires_grad = False
        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False
        if self.config.freeze_vlm:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False
        if not self.config.train_action_expert:
            self.qwen_expert.eval()
            for params in self.qwen_expert.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()
        if self.config.freeze_vlm:
            self.qwenvl.eval()
        if not self.config.train_action_expert:
            self.qwen_expert.eval()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        precompute_grid_thw = getattr(self.config, "precompute_grid_thw", False)
        if precompute_grid_thw and self.position_embeddings is None:
            (
                self.pos_embeds,
                self.position_embeddings,
                self.cu_seqlens,
                self.visual_split_sizes,
                self.visual_max_seqlen,
            ) = self.qwenvl.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
        image_embeds, deepstack_image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            pos_embeds=self.pos_embeds,
            position_embeddings=self.position_embeddings,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.visual_max_seqlen,
        )
        split_sizes = self.visual_split_sizes
        if split_sizes is None:
            split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.visual.spatial_merge_size**2).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes))
            for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [
            torch.stack(chunks, dim=0)
            for chunks in deepstack_chunks
        ]
        return image_embeds, deepstack_image_embeds

    def embed_image(self, image: torch.Tensor, image_grid_thw: torch.LongTensor):
        return self.get_image_features(
            image,
            image_grid_thw=image_grid_thw,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(self, input_ids, attention_mask,
                                   image_grid_thw=None, video_grid_thw=None):
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids

    def apply_mrope(self, query_states, key_states, position_ids):
        position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        append_kv_cache: bool = False,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache and append_kv_cache:
                raise ValueError("fill_kv_cache and append_kv_cache are mutually exclusive")
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                if layer_idx not in past_key_values:
                    raise ValueError(f"Missing cached VLM layer {layer_idx}")
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
                if append_kv_cache:
                    # The caller provides a fresh working dict. Cached tensor
                    # storage is shared, while the permanent slow dict remains
                    # structurally and numerically unchanged.
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
        return key_states, value_states, past_key_values

    def _apply_deepstack(self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds):
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = self.qwenvl.model.language_model._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )
        return hidden_states

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        append_kv_cache: bool = False,
        ada_cond: List[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers
        router_logits_list = []

        assert action_num_layers == num_layers, (
            "Action expert and VLM must have the same number of layers "
            f"(got action={action_num_layers}, vlm={num_layers})."
        )

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1:
                    q, k, v = models[i].layers[layer_idx](
                        hidden_states, compute_kqv=True, ada_cond=ada_cond
                    )
                else:
                    q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
                query_states.append(q.float())
                key_states.append(k.float())
                value_states.append(v.float())

            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            query_states, key_states = self.apply_mrope(query_states, key_states, position_ids)
            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                append_kv_cache=append_kv_cache,
            )
            if self.config.attention_implementation == "flex_cached":
                if layer_idx == 0:
                    _full_block_mask = build_block_mask(
                        attention_mask,
                        self.qwenvl.config.text_config.num_attention_heads,
                        query_states.shape[1],
                        key_states.shape[1],
                    )
                att_output = flex_attention_with_block_mask(
                    query_states, key_states, value_states, _full_block_mask, query_states.shape[1]
                )
            else:
                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                end = start + hidden_states.shape[1]
                if i == 1:
                    out_emb, router_logits = models[i].layers[layer_idx](
                        hidden_states,
                        att_output,
                        start,
                        end,
                        output_atten=True,
                        ada_cond=ada_cond,
                    )
                    if router_logits is not None:
                        router_logits_list.append(router_logits)
                else:
                    out_emb = models[i].layers[layer_idx](
                        hidden_states, att_output, start, end, output_atten=True
                    )
                    out_emb = self._apply_deepstack(out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds)
                outputs_embeds.append(out_emb)
                start = end
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "flex":
            print("=====Using Flex Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "flex_cached":
            print("=====Using Flex Cached (prebuilt BlockMask) Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "eager":
            print("=====Using Eager Attn=====")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class FlowMatchingV2(FlowMatchingV1):
    def __init__(self, config, eval):
        nn.Module.__init__(self)
        self.config = config
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
            action_num_attention_heads=getattr(self.config, "action_num_attention_heads", 32),
            action_num_key_value_heads=getattr(self.config, "action_num_key_value_heads", 8),
            action_head_dim=getattr(self.config, "action_head_dim", 128),
            freeze_vlm=getattr(self.config, "freeze_vlm", False),
            train_action_expert=getattr(self.config, "train_action_expert", True),
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "precompute_grid_thw",
            "vit_attn_implementation",
            "use_moe",
            "bias_update_speed",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
            "_moe_implementation",
        ]:
            if hasattr(config, name):
                setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(qwenvl_with_export_config.qwen_expert_config, "initializer_range", None)

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.tactile_settings = TactileVTLAConfig.from_mapping(
            getattr(self.config, "tactile", None)
        )
        if self.tactile_settings.enabled:
            vlm_hidden_size = int(
                self.qwenvl_with_expert.qwenvl.config.text_config.hidden_size
            )
            self.tactile_encoder: TactileTokenEncoder | None = TactileTokenEncoder(
                self.tactile_settings,
                context_dim=vlm_hidden_size,
                vision_output_dim=vlm_hidden_size,
            )
        else:
            self.tactile_encoder = None
        self._tactile_debug_logged = False

        self.config.align_params = getattr(self.config, "align_params", None) or {}
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False
            self.block_future_depth_to_action = False

        self.tactile_refinement_settings = TactileRefinementConfig.from_mapping(
            getattr(self.config, "tactile_refinement", None)
        )
        self.tactile_action_expert: TactileActionExpert | None = None
        self.tactile_state_proj: nn.Linear | None = None
        self.tactile_action_in_proj: nn.Linear | None = None
        self.tactile_action_out_proj: nn.Linear | None = None
        self.tactile_context_proj: nn.Linear | None = None
        self.tactile_plan_proj: nn.Linear | None = None
        self.tactile_marker_proj: nn.Linear | None = None
        self.tactile_time_mlp: nn.Sequential | None = None
        if self.tactile_refinement_settings.cascaded_enabled:
            self._require_cascaded_tactile_contract()
            expert_cfg = self.tactile_refinement_settings.expert
            self.tactile_action_expert = TactileActionExpert(expert_cfg)
            self.tactile_state_proj = nn.Linear(
                self.config.max_state_dim, expert_cfg.hidden_size
            )
            self.tactile_action_in_proj = nn.Linear(
                self.config.max_action_dim, expert_cfg.hidden_size
            )
            self.tactile_action_out_proj = nn.Linear(
                expert_cfg.hidden_size, self.config.max_action_dim
            )
            self.tactile_context_proj = nn.Linear(
                vlm_hidden_size, expert_cfg.hidden_size
            )
            self.tactile_plan_proj = nn.Linear(
                self.config.proj_width, expert_cfg.hidden_size
            )
            self.tactile_marker_proj = nn.Linear(
                vlm_hidden_size, expert_cfg.hidden_size
            )
            self.tactile_time_mlp = nn.Sequential(
                nn.Linear(expert_cfg.hidden_size, expert_cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(expert_cfg.hidden_size, expert_cfg.hidden_size),
            )

        self.set_requires_grad()

    def set_requires_grad(self):
        """Apply the existing state rule plus the VTLA action-expert freeze."""

        super().set_requires_grad()
        if not getattr(self.config, "train_action_expert", True):
            for module in (
                self.state_proj,
                self.action_in_proj,
                self.action_out_proj,
                self.action_time_mlp_in,
                self.action_time_mlp_out,
            ):
                module.requires_grad_(False)
        if self.tactile_refinement_settings.cascaded_enabled:
            training = self.tactile_refinement_settings.training
            if training.freeze_vlm:
                self.qwenvl_with_expert.qwenvl.requires_grad_(False)
            if training.freeze_action_expert:
                self.qwenvl_with_expert.qwen_expert.requires_grad_(False)
                for module in (
                    self.state_proj,
                    self.action_in_proj,
                    self.action_out_proj,
                    self.action_time_mlp_in,
                    self.action_time_mlp_out,
                ):
                    module.requires_grad_(False)
            if self.tactile_encoder is not None:
                # TacRGB and Marker share this container. The cascaded stage may
                # train the Marker encoder, but it must not silently unfreeze the
                # TacRGB projection or shared sensor embedding.
                self.tactile_encoder.requires_grad_(False)
                marker_parameter_prefixes = (
                    "marker_encoder.",
                    "regional_marker_encoder.",
                    "point_spatiotemporal_marker_encoder.",
                    "marker_temporal_embedding.",
                    "marker_spatial_embedding.",
                )
                marker_parameter_names = {"marker_modality_embedding"}
                for name, parameter in self.tactile_encoder.named_parameters():
                    if name in marker_parameter_names or name.startswith(
                        marker_parameter_prefixes
                    ):
                        parameter.requires_grad_(training.train_marker_encoder)
            for module in (
                self.tactile_action_expert,
                self.tactile_state_proj,
                self.tactile_action_in_proj,
                self.tactile_action_out_proj,
                self.tactile_context_proj,
                self.tactile_plan_proj,
                self.tactile_marker_proj,
                self.tactile_time_mlp,
            ):
                assert module is not None
                module.requires_grad_(training.train_tactile_expert)

    def _require_cascaded_tactile_contract(self) -> None:
        settings = self.tactile_settings
        if self.tactile_encoder is None or not settings.enabled:
            raise RuntimeError("cascaded_flow requires tactile encoding")
        if not settings.use_rgb or not settings.use_markers:
            raise RuntimeError("cascaded_flow requires TacThru RGB and markers")
        if (
            settings.num_sensors != 1
            or settings.marker_history_length != 4
            or settings.num_markers != 48
            or settings.marker_tokenization.mode != "point_spatiotemporal"
        ):
            raise RuntimeError(
                "cascaded_flow requires one sensor and 4x48 point-spatiotemporal tokens"
            )
        if settings.marker_contact_gate.target != "marker_only":
            raise RuntimeError("cascaded_flow requires gate target marker_only")
        if not getattr(self.config, "vlm_causal", False):
            raise RuntimeError("cascaded_flow requires vlm_causal=true")

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
        tactile_rgb=None,
        tactile_rgb_grid_thw=None,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
    ):
        if self.tactile_encoder is not None:
            return self._embed_prefix_vtla(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                image_grid_thw=image_grid_thw,
                tactile_rgb=tactile_rgb,
                tactile_rgb_grid_thw=tactile_rgb_grid_thw,
                marker_displacement_history=marker_displacement_history,
                marker_valid_mask=marker_valid_mask,
                marker_history_valid_mask=marker_history_valid_mask,
                marker_contact_state=marker_contact_state,
                tactile_sensor_mask=tactile_sensor_mask,
                tactile_rgb_mask=tactile_rgb_mask,
            )
        if image_grid_thw is None:
            raise ValueError("LingbotVlaV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(
            images,
            flat_grid_thw,
        )
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        deepstack_embs = [
            einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images)
            for x in deepstack_embs
        ]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        if getattr(self.config, "qwen3vl_use_vision_boundaries", True):
            start_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
            )
            end_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
            )
            img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
            image_token_len = num_patch + 2
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = torch.zeros_like(image_pad_masks)
            image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
            fake_image_ids[:, :, 0] = cfg.vision_start_token_id
            fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        else:
            img_chunks = img_emb
            image_token_len = num_patch
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = image_pad_masks
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)

        if self.use_depth_align and self.align_type == "query":
            def _get_align_tokens(tokens):
                tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_pad_masks = torch.ones(
                bsize,
                self.num_task_tokens,
                device=device,
                dtype=lang_masks.dtype
            )
            fake_align_ids = torch.full(
                (bsize, self.num_task_tokens),
                cfg.text_config.eos_token_id,
                dtype=torch.long,
                device=device
            )

            current_task = _get_align_tokens(self.depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_current_video_patch", False)
                and getattr(self, "use_current_shared_task_proj", False)
            ):
                current_video_task = _get_align_tokens(self.current_video_align_embs)
                current_task = self.current_shared_task_proj(
                    torch.cat([current_task, current_video_task], dim=-1)
                )
            align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
            parts = [img_emb]
            masks = [image_pad_masks]
            input_ids = [fake_image_ids]
            visual_masks = [visual_pos_masks]

            def _append(
                tokens,
                token_masks,
                token_ids,
                token_visual_masks=None,
            ):
                parts.append(tokens)
                masks.append(token_masks)
                input_ids.append(token_ids)
                if token_visual_masks is None:
                    token_visual_masks = torch.zeros_like(token_masks)
                visual_masks.append(token_visual_masks)

            future_align_embs = None
            if self.use_future_depth:
                future_task = _get_align_tokens(self.future_depth_align_embs)
                if (
                    getattr(self, "use_future_video", False)
                    and getattr(self, "use_future_video_patch", True)
                    and getattr(self, "future_video_share_future_depth_query", False)
                    and getattr(self, "use_shared_future_task_proj", False)
                ):
                    future_video_task = _get_align_tokens(self.future_video_align_embs)
                    future_task = self.future_shared_task_proj(
                        torch.cat([future_task, future_video_task], dim=-1)
                    )
                future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)

            if (
                not self.use_future_depth
                and getattr(self, "use_future_video", False)
                and getattr(self, "future_video_share_future_depth_query", False)
            ):
                raise ValueError(
                    "share_future_depth_query=True requires depth.use_future_depth=True."
                )

            for segment_name in prefix_query_segments(
                use_depth_align=True,
                use_future_depth=self.use_future_depth,
                use_future_video=getattr(self, "use_future_video", False),
                use_future_video_cls=getattr(self, "use_future_video_cls", False),
                use_future_video_patch=getattr(self, "use_future_video_patch", True),
                future_video_share_future_depth_query=getattr(
                    self,
                    "future_video_share_future_depth_query",
                    False,
                ),
            ):
                if segment_name == "language":
                    _append(
                        lang_emb,
                        lang_masks,
                        lang_tokens.to(device),
                    )
                elif segment_name == "current_depth":
                    _append(align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_video_cls":
                    future_video_cls_align_emb = self.future_video_cls_align_emb.weight.repeat(
                        img_emb.size(0), 1, 1
                    ).to(img_emb.device, img_emb.dtype)
                    cls_align_pad_masks = torch.ones(
                        bsize,
                        1,
                        device=device,
                        dtype=lang_masks.dtype,
                    )
                    fake_cls_align_ids = torch.full(
                        (bsize, 1),
                        cfg.text_config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _append(future_video_cls_align_emb, cls_align_pad_masks, fake_cls_align_ids)
                elif segment_name == "future_video":
                    future_video_align_embs = _get_align_tokens(self.future_video_align_embs).repeat(
                        img_emb.size(0), 1, 1
                    ).to(img_emb.device, img_emb.dtype)
                    _append(future_video_align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_depth":
                    _append(future_align_embs, align_pad_masks, fake_align_ids)
                else:
                    raise ValueError(f"Unsupported prefix query segment: {segment_name}")

            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)
        else:
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
            prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
            full_visual_pos_masks = torch.cat([visual_pos_masks, torch.zeros_like(lang_masks)], dim=1)

        if getattr(self.config, "vlm_causal", False):
            att_masks = torch.ones((bsize, embs.shape[1]), device=device, dtype=torch.bool)
        else:
            att_masks = torch.zeros((bsize, embs.shape[1]), device=device, dtype=torch.bool)

        flat_img_masks = einops.rearrange(img_masks, "b n -> (b n)")
        rope_grid_thw = flat_grid_thw[flat_img_masks]
        if rope_grid_thw.numel() == 0:
            rope_grid_thw = flat_grid_thw[:1]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )
        filtered_deepstack = []
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        for deepstack in deepstack_embs:
            filtered_deepstack.append(deepstack[img_visual_only])

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            filtered_deepstack,
        )
        return result

    def _embed_prefix_vtla(
        self,
        *,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw,
        tactile_rgb,
        tactile_rgb_grid_thw,
        marker_displacement_history,
        marker_valid_mask,
        marker_history_valid_mask,
        marker_contact_state,
        tactile_sensor_mask,
        tactile_rgb_mask,
        _slow_only: bool = False,
        _include_task_queries: bool = False,
        _return_cache_metadata: bool = False,
        _profile_ms: dict[str, float] | None = None,
    ):
        """Build the Qwen3-VL prefix with RGB and marker tactile tokens.

        Scene tokens keep their original order and values.  Tactile blocks are
        inserted immediately after language and before task-query tokens, so
        existing depth/future-query helpers can continue indexing from the
        prefix tail.  Tactile RGB uses normal Qwen image IDs/mRoPE; marker MLP
        tokens use legal text-style EOS IDs and therefore continuous 1D mRoPE.
        """

        if image_grid_thw is None:
            raise ValueError("LingbotVlaV2Policy requires image_grid_thw")
        if self.tactile_encoder is None:
            raise RuntimeError("VTLA prefix requested while tactile encoding is disabled")

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim not in (4, 5):
            raise ValueError(
                "images must be [B,N,L,D] Qwen patches or [B,N,C,H,W], "
                f"got {tuple(images.shape)}"
            )
        bsize, num_images = images.shape[:2]
        device = images.device
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)
        if tuple(img_masks.shape) != (bsize, num_images):
            raise ValueError("img_masks must have shape [B,N]")
        img_masks = img_masks.to(device=device, dtype=torch.bool)

        if image_grid_thw.ndim == 2:
            if image_grid_thw.shape[0] != bsize * num_images:
                raise ValueError("Flat image_grid_thw must contain B*N rows")
            scene_grid = image_grid_thw.reshape(bsize, num_images, 3)
        elif image_grid_thw.ndim == 3:
            scene_grid = image_grid_thw
        else:
            raise ValueError("image_grid_thw must be [B,N,3] or [B*N,3]")
        if tuple(scene_grid.shape) != (bsize, num_images, 3):
            raise ValueError("image_grid_thw does not match scene images")
        scene_grid = scene_grid.to(device=device, dtype=torch.long)

        sensor_count = self.tactile_settings.num_sensors
        if tactile_sensor_mask is None:
            sensor_mask = torch.ones(
                bsize,
                sensor_count,
                dtype=torch.bool,
                device=device,
            )
        else:
            if tactile_sensor_mask.ndim == 1 and bsize == 1:
                tactile_sensor_mask = tactile_sensor_mask.unsqueeze(0)
            if tuple(tactile_sensor_mask.shape) != (bsize, sensor_count):
                raise ValueError("tactile_sensor_mask must have shape [B,S]")
            sensor_mask = tactile_sensor_mask.to(device=device, dtype=torch.bool)

        rgb_sensor_mask = None
        all_images = images
        all_grid = scene_grid
        if self.tactile_settings.use_rgb:
            rgb_was_missing = tactile_rgb is None
            if tactile_rgb is None:
                # Reuse the already-processed scene tensor only as a shape
                # template.  The resulting zero images remain fixed in count
                # and are fully hidden by tactile_rgb_mask.
                tactile_rgb = torch.zeros_like(images[:, :1]).expand(
                    bsize, sensor_count, *images.shape[2:]
                ).clone()
                tactile_rgb_grid_thw = scene_grid[:, :1].expand(
                    bsize, sensor_count, 3
                ).clone()
            else:
                if tactile_rgb.ndim == images.ndim - 1 and bsize == 1:
                    tactile_rgb = tactile_rgb.unsqueeze(0)
                if tactile_rgb.ndim != images.ndim:
                    raise ValueError("tactile_rgb rank must match scene images")
                if tuple(tactile_rgb.shape[:2]) != (bsize, sensor_count):
                    raise ValueError("tactile_rgb must have shape [B,S,...]")
                if tactile_rgb.shape[2:] != images.shape[2:]:
                    raise ValueError(
                        "Shared Qwen vision encoding requires scene and tactile images "
                        "to use the same processed shape"
                    )
                if tactile_rgb_grid_thw is None:
                    raise ValueError("tactile_rgb_grid_thw is required with tactile_rgb")
                if tactile_rgb_grid_thw.ndim == 2:
                    tactile_rgb_grid_thw = tactile_rgb_grid_thw.reshape(
                        bsize, sensor_count, 3
                    )
                if tuple(tactile_rgb_grid_thw.shape) != (bsize, sensor_count, 3):
                    raise ValueError("tactile_rgb_grid_thw must have shape [B,S,3]")

            tactile_rgb = tactile_rgb.to(device=device, dtype=images.dtype)
            tactile_rgb_grid_thw = tactile_rgb_grid_thw.to(device=device, dtype=torch.long)
            if tactile_rgb_mask is None:
                rgb_sensor_mask = torch.full(
                    (bsize, sensor_count),
                    not rgb_was_missing,
                    dtype=torch.bool,
                    device=device,
                )
            else:
                if tactile_rgb_mask.ndim == 1 and bsize == 1:
                    tactile_rgb_mask = tactile_rgb_mask.unsqueeze(0)
                if tuple(tactile_rgb_mask.shape) != (bsize, sensor_count):
                    raise ValueError("tactile_rgb_mask must have shape [B,S]")
                rgb_sensor_mask = tactile_rgb_mask.to(device=device, dtype=torch.bool)
            rgb_sensor_mask &= sensor_mask
            rgb_sensor_mask = self.tactile_encoder.gate_rgb_sensor_mask(
                rgb_sensor_mask,
                marker_contact_state,
            )
            all_images = torch.cat([images, tactile_rgb], dim=1)
            all_grid = torch.cat([scene_grid, tactile_rgb_grid_thw], dim=1)

        flat_images = einops.rearrange(all_images, "b n ... -> (b n) ...")
        flat_grid = einops.rearrange(all_grid, "b n d -> (b n) d")
        if _profile_ms is not None:
            if flat_images.is_cuda:
                vit_start = torch.cuda.Event(enable_timing=True)
                vit_end = torch.cuda.Event(enable_timing=True)
                vit_start.record()
            else:
                vit_started = time.perf_counter()
        all_img_emb, all_deepstack = self.qwenvl_with_expert.embed_image(flat_images, flat_grid)
        if _profile_ms is not None:
            if flat_images.is_cuda:
                vit_end.record()
                vit_end.synchronize()
                _profile_ms["scene_tacrgb_vit_ms"] = float(vit_start.elapsed_time(vit_end))
            else:
                _profile_ms["scene_tacrgb_vit_ms"] = (
                    time.perf_counter() - vit_started
                ) * 1000.0
        embed_dtype = all_img_emb.dtype
        num_patch = all_img_emb.shape[1]
        total_views = all_images.shape[1]
        all_img_emb = einops.rearrange(
            all_img_emb,
            "(b n) l d -> b n l d",
            b=bsize,
            n=total_views,
        )
        all_deepstack = [
            einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=total_views)
            for x in all_deepstack
        ]
        scene_emb = all_img_emb[:, :num_images]

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id
        use_boundaries = getattr(self.config, "qwen3vl_use_vision_boundaries", True)

        if use_boundaries:
            scene_start = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_start_token_id,
                bsize,
                num_images,
                device,
                embed_dtype,
            )
            scene_end = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_end_token_id,
                bsize,
                num_images,
                device,
                embed_dtype,
            )
            scene_chunks = torch.cat([scene_start, scene_emb, scene_end], dim=2)
            scene_token_len = num_patch + 2
            scene_visual_masks = torch.zeros(
                bsize,
                num_images,
                scene_token_len,
                dtype=torch.bool,
                device=device,
            )
            scene_visual_masks[:, :, 1 : 1 + num_patch] = img_masks.unsqueeze(-1)
            scene_ids = torch.full(
                (bsize, num_images, scene_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
            scene_ids[:, :, 0] = cfg.vision_start_token_id
            scene_ids[:, :, -1] = cfg.vision_end_token_id
        else:
            scene_chunks = scene_emb
            scene_token_len = num_patch
            scene_visual_masks = img_masks.unsqueeze(-1).expand(-1, -1, num_patch)
            scene_ids = torch.full(
                (bsize, num_images, scene_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
        scene_pad_masks = img_masks.unsqueeze(-1).expand(-1, -1, scene_token_len)
        scene_tokens = scene_chunks.flatten(1, 2)
        scene_pad_masks = scene_pad_masks.flatten(1, 2)
        scene_visual_masks = scene_visual_masks.flatten(1, 2)
        scene_ids = scene_ids.flatten(1, 2)
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(
            dtype=embed_dtype
        )

        tactile_blocks: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        tactile_patch_visual_mask = None
        if self.tactile_settings.use_rgb:
            rgb_emb = all_img_emb[:, num_images:]
            rgb_patch_tokens, rgb_patch_mask = self.tactile_encoder.encode_rgb_embeddings(
                rgb_emb,
                tactile_sensor_mask=sensor_mask,
                tactile_rgb_mask=rgb_sensor_mask,
            )
            rgb_patch_tokens = rgb_patch_tokens.reshape(
                bsize,
                sensor_count,
                num_patch,
                -1,
            )
            rgb_patch_mask = rgb_patch_mask.reshape(
                bsize,
                sensor_count,
                num_patch,
            )
            if use_boundaries:
                rgb_start = self.qwenvl_with_expert.embed_special_token(
                    cfg.vision_start_token_id,
                    bsize,
                    sensor_count,
                    device,
                    embed_dtype,
                )
                rgb_end = self.qwenvl_with_expert.embed_special_token(
                    cfg.vision_end_token_id,
                    bsize,
                    sensor_count,
                    device,
                    embed_dtype,
                )
                rgb_chunks = torch.cat([rgb_start, rgb_patch_tokens, rgb_end], dim=2)
                rgb_token_len = num_patch + 2
                rgb_visual_masks = torch.zeros(
                    bsize,
                    sensor_count,
                    rgb_token_len,
                    dtype=torch.bool,
                    device=device,
                )
                rgb_visual_masks[:, :, 1 : 1 + num_patch] = rgb_patch_mask
                rgb_ids = torch.full(
                    (bsize, sensor_count, rgb_token_len),
                    visual_token_id,
                    dtype=torch.long,
                    device=device,
                )
                rgb_ids[:, :, 0] = cfg.vision_start_token_id
                rgb_ids[:, :, -1] = cfg.vision_end_token_id
            else:
                rgb_chunks = rgb_patch_tokens
                rgb_token_len = num_patch
                rgb_visual_masks = rgb_patch_mask
                rgb_ids = torch.full(
                    (bsize, sensor_count, rgb_token_len),
                    visual_token_id,
                    dtype=torch.long,
                    device=device,
                )
            rgb_pad_masks = rgb_sensor_mask.unsqueeze(-1).expand(
                -1, -1, rgb_token_len
            )
            # Keep the tactile sequence length fixed, but make every embedding
            # in a missing sensor block a true zero placeholder.  This also
            # clears the otherwise non-zero vision boundary embeddings.
            rgb_chunks = rgb_chunks * rgb_pad_masks.unsqueeze(-1).to(
                dtype=rgb_chunks.dtype
            )
            tactile_blocks.append(
                (
                    rgb_chunks.flatten(1, 2),
                    rgb_pad_masks.flatten(1, 2),
                    rgb_ids.flatten(1, 2),
                    rgb_visual_masks.flatten(1, 2),
                )
            )
            tactile_patch_visual_mask = rgb_patch_mask

        marker_token_mask = None
        if self.tactile_settings.use_markers and not _slow_only:
            if marker_displacement_history is None:
                marker_displacement_history = torch.zeros(
                    bsize,
                    sensor_count,
                    self.tactile_settings.marker_history_length,
                    self.tactile_settings.num_markers,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                marker_valid_mask = torch.zeros(
                    marker_displacement_history.shape[:-1],
                    dtype=torch.bool,
                    device=device,
                )
                marker_history_valid_mask = torch.zeros(
                    marker_displacement_history.shape[:3],
                    dtype=torch.bool,
                    device=device,
                )
            marker_dtype = self.tactile_encoder.marker_dtype
            marker_displacement_history = marker_displacement_history.to(
                device=device, dtype=marker_dtype
            )
            if marker_valid_mask is not None:
                marker_valid_mask = marker_valid_mask.to(device=device, dtype=torch.bool)
            if marker_history_valid_mask is not None:
                marker_history_valid_mask = marker_history_valid_mask.to(
                    device=device, dtype=torch.bool
                )
            marker_tokens, marker_token_mask = self.tactile_encoder.encode_markers(
                marker_displacement_history,
                marker_valid_mask,
                marker_history_valid_mask,
                sensor_mask,
                marker_contact_state,
            )
            # Sensor-major, then time-major and spatial-minor within each sensor.
            marker_tokens = marker_tokens.to(dtype=embed_dtype).flatten(1, 2)
            marker_token_mask = marker_token_mask.flatten(1, 2)
            marker_ids = torch.full(
                (bsize, marker_tokens.shape[1]),
                cfg.text_config.eos_token_id,
                dtype=torch.long,
                device=device,
            )
            tactile_blocks.append(
                (
                    marker_tokens,
                    marker_token_mask,
                    marker_ids,
                    torch.zeros_like(marker_token_mask),
                )
            )

        if self.use_depth_align and self.align_type == "query":
            def _get_align_tokens(tokens):
                weights = tokens.view(
                    self.num_task_tokens,
                    tokens.shape[0] // self.num_task_tokens,
                    tokens.shape[1],
                )
                return weights.mean(dim=1)

            align_pad_masks = torch.ones(
                bsize,
                self.num_task_tokens,
                device=device,
                dtype=lang_masks.dtype,
            )
            fake_align_ids = torch.full(
                (bsize, self.num_task_tokens),
                cfg.text_config.eos_token_id,
                dtype=torch.long,
                device=device,
            )
            current_task = _get_align_tokens(self.depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_current_video_patch", False)
                and getattr(self, "use_current_shared_task_proj", False)
            ):
                current_video_task = _get_align_tokens(self.current_video_align_embs)
                current_task = self.current_shared_task_proj(
                    torch.cat([current_task, current_video_task], dim=-1)
                )
            align_embs = current_task.repeat(bsize, 1, 1).to(
                device=device,
                dtype=embed_dtype,
            )
            parts = [scene_tokens]
            masks = [scene_pad_masks]
            input_ids = [scene_ids]
            visual_masks = [scene_visual_masks]

            def _append(tokens, token_masks, token_ids, token_visual_masks=None):
                parts.append(tokens)
                masks.append(token_masks)
                input_ids.append(token_ids)
                visual_masks.append(
                    torch.zeros_like(token_masks)
                    if token_visual_masks is None
                    else token_visual_masks
                )

            future_align_embs = None
            if self.use_future_depth:
                future_task = _get_align_tokens(self.future_depth_align_embs)
                if (
                    getattr(self, "use_future_video", False)
                    and getattr(self, "use_future_video_patch", True)
                    and getattr(self, "future_video_share_future_depth_query", False)
                    and getattr(self, "use_shared_future_task_proj", False)
                ):
                    future_video_task = _get_align_tokens(self.future_video_align_embs)
                    future_task = self.future_shared_task_proj(
                        torch.cat([future_task, future_video_task], dim=-1)
                    )
                future_align_embs = future_task.repeat(bsize, 1, 1).to(
                    device=device,
                    dtype=embed_dtype,
                )
            if (
                not self.use_future_depth
                and getattr(self, "use_future_video", False)
                and getattr(self, "future_video_share_future_depth_query", False)
            ):
                raise ValueError("share_future_depth_query=True requires use_future_depth")

            for segment_name in prefix_query_segments(
                use_depth_align=True,
                use_future_depth=self.use_future_depth,
                use_future_video=getattr(self, "use_future_video", False),
                use_future_video_cls=getattr(self, "use_future_video_cls", False),
                use_future_video_patch=getattr(self, "use_future_video_patch", True),
                future_video_share_future_depth_query=getattr(
                    self,
                    "future_video_share_future_depth_query",
                    False,
                ),
            ):
                if segment_name == "language":
                    _append(lang_emb, lang_masks, lang_tokens.to(device))
                    for block in tactile_blocks:
                        _append(*block)
                elif _slow_only and not _include_task_queries:
                    continue
                elif segment_name == "current_depth":
                    _append(align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_video_cls":
                    cls_emb = self.future_video_cls_align_emb.weight.repeat(
                        bsize, 1, 1
                    ).to(device=device, dtype=embed_dtype)
                    cls_mask = torch.ones(
                        bsize, 1, device=device, dtype=lang_masks.dtype
                    )
                    cls_ids = torch.full(
                        (bsize, 1),
                        cfg.text_config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _append(cls_emb, cls_mask, cls_ids)
                elif segment_name == "future_video":
                    video_embs = _get_align_tokens(self.future_video_align_embs).repeat(
                        bsize, 1, 1
                    ).to(device=device, dtype=embed_dtype)
                    _append(video_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_depth":
                    _append(future_align_embs, align_pad_masks, fake_align_ids)
                else:
                    raise ValueError(f"Unsupported prefix query segment: {segment_name}")
            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)
        else:
            parts = [scene_tokens, lang_emb]
            masks = [scene_pad_masks, lang_masks]
            input_ids = [scene_ids, lang_tokens.to(device)]
            visual_masks = [scene_visual_masks, torch.zeros_like(lang_masks)]
            for tokens, mask, ids, visual_mask in tactile_blocks:
                parts.append(tokens)
                masks.append(mask)
                input_ids.append(ids)
                visual_masks.append(visual_mask)
            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)

        pad_masks = pad_masks.to(dtype=torch.bool)
        if getattr(self.config, "vlm_causal", False):
            att_masks = torch.ones(
                (bsize, embs.shape[1]),
                device=device,
                dtype=torch.bool,
            )
        else:
            att_masks = torch.zeros(
                (bsize, embs.shape[1]),
                device=device,
                dtype=torch.bool,
            )

        rope_view_masks = img_masks
        if self.tactile_settings.use_rgb:
            rope_view_masks = torch.cat([img_masks, rgb_sensor_mask], dim=1)
        rope_grid_thw = all_grid[rope_view_masks]
        if rope_grid_thw.numel() == 0:
            rope_grid_thw = all_grid.reshape(-1, 3)[:1]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )

        scene_patch_mask = img_masks.unsqueeze(-1).expand(-1, -1, num_patch)
        filtered_deepstack = []
        for deepstack in all_deepstack:
            per_batch = []
            for batch_index in range(bsize):
                selected = [
                    deepstack[batch_index, :num_images][
                        scene_patch_mask[batch_index]
                    ]
                ]
                if self.tactile_settings.use_rgb:
                    selected.append(
                        deepstack[batch_index, num_images:][
                            tactile_patch_visual_mask[batch_index]
                        ]
                    )
                per_batch.append(torch.cat(selected, dim=0))
            filtered_deepstack.append(torch.cat(per_batch, dim=0))

        if self.tactile_settings.debug_shapes and not self._tactile_debug_logged:
            marker_shape = None
            if marker_token_mask is not None:
                marker_shape = (
                    bsize,
                    sensor_count,
                    marker_token_mask.shape[1] // sensor_count,
                    embs.shape[-1],
                )
            logger.info(
                "VTLA prefix shapes: scene=%s tactile_rgb=%s marker=%s "
                "context=%s mask=%s prefix_length=%d rgb_valid=%.4f marker_valid=%.4f",
                tuple(scene_tokens.shape),
                None
                if not self.tactile_settings.use_rgb
                else (bsize, sensor_count * (num_patch + (2 if use_boundaries else 0)), embs.shape[-1]),
                marker_shape,
                tuple(embs.shape),
                tuple(pad_masks.shape),
                embs.shape[1],
                0.0 if rgb_sensor_mask is None else rgb_sensor_mask.float().mean().item(),
                0.0 if marker_token_mask is None else marker_token_mask.float().mean().item(),
            )
            self._tactile_debug_logged = True

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            filtered_deepstack,
        )
        if _return_cache_metadata:
            return result + (prefix_input_ids, rope_grid_thw)
        return result

    def _require_vtla_slow_cache_contract(self) -> None:
        settings = self.tactile_settings
        if self.tactile_encoder is None or not settings.enabled:
            raise RuntimeError("VTLA slow-cache inference requires tactile encoding")
        if not settings.use_rgb or not settings.use_markers:
            raise RuntimeError("VTLA slow-cache inference requires tactile RGB and markers")
        if (
            settings.num_sensors != 1
            or settings.marker_history_length != 4
            or settings.num_markers != 48
            or settings.marker_tokenization.mode != "point_spatiotemporal"
        ):
            raise RuntimeError(
                "VTLA slow-cache inference is scoped to one sensor and 4x48 "
                "point-spatiotemporal marker tokens"
            )
        if settings.marker_contact_gate.target != "marker_only":
            raise RuntimeError("VTLA slow-cache inference requires gate target marker_only")
        if not getattr(self.config, "vlm_causal", False):
            raise RuntimeError("VTLA slow-cache inference requires vlm_causal=true")
        if not getattr(self.config, "use_cache", False):
            raise RuntimeError("VTLA slow-cache inference requires use_cache=true")

    def _build_task_query_tokens(self, batch_size, device, dtype):
        if not (self.use_depth_align and self.align_type == "query"):
            hidden_size = self.qwenvl_with_expert.qwenvl.config.text_config.hidden_size
            return (
                torch.empty(batch_size, 0, hidden_size, device=device, dtype=dtype),
                torch.empty(batch_size, 0, device=device, dtype=torch.bool),
                torch.empty(batch_size, 0, device=device, dtype=torch.long),
            )

        def _get_align_tokens(tokens):
            weights = tokens.view(
                self.num_task_tokens,
                tokens.shape[0] // self.num_task_tokens,
                tokens.shape[1],
            )
            return weights.mean(dim=1)

        cfg = self.qwenvl_with_expert.qwenvl.config
        standard_mask = torch.ones(
            batch_size, self.num_task_tokens, device=device, dtype=torch.bool
        )
        standard_ids = torch.full(
            (batch_size, self.num_task_tokens),
            cfg.text_config.eos_token_id,
            device=device,
            dtype=torch.long,
        )
        current_task = _get_align_tokens(self.depth_align_embs)
        if (
            getattr(self, "use_future_video", False)
            and getattr(self, "use_current_video_patch", False)
            and getattr(self, "use_current_shared_task_proj", False)
        ):
            current_task = self.current_shared_task_proj(
                torch.cat(
                    [current_task, _get_align_tokens(self.current_video_align_embs)], dim=-1
                )
            )
        future_task = None
        if self.use_future_depth:
            future_task = _get_align_tokens(self.future_depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_future_video_patch", True)
                and getattr(self, "future_video_share_future_depth_query", False)
                and getattr(self, "use_shared_future_task_proj", False)
            ):
                future_task = self.future_shared_task_proj(
                    torch.cat(
                        [future_task, _get_align_tokens(self.future_video_align_embs)], dim=-1
                    )
                )

        parts = []
        masks = []
        ids = []
        for name in prefix_query_segments(
            use_depth_align=True,
            use_future_depth=self.use_future_depth,
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self, "future_video_share_future_depth_query", False
            ),
        ):
            if name == "language":
                continue
            if name == "current_depth":
                value, mask, token_ids = current_task, standard_mask, standard_ids
            elif name == "future_video_cls":
                value = self.future_video_cls_align_emb.weight
                mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
                token_ids = torch.full(
                    (batch_size, 1),
                    cfg.text_config.eos_token_id,
                    device=device,
                    dtype=torch.long,
                )
            elif name == "future_video":
                value = _get_align_tokens(self.future_video_align_embs)
                mask, token_ids = standard_mask, standard_ids
            elif name == "future_depth":
                value, mask, token_ids = future_task, standard_mask, standard_ids
            else:
                raise ValueError(f"Unsupported prefix query segment: {name}")
            parts.append(value.repeat(batch_size, 1, 1).to(device=device, dtype=dtype))
            masks.append(mask)
            ids.append(token_ids)
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1), torch.cat(ids, dim=1)

    def _build_fast_vtla_tail(
        self,
        *,
        marker_displacement_history,
        marker_valid_mask,
        marker_history_valid_mask,
        marker_contact_state,
        tactile_sensor_mask,
        batch_size,
        device,
        dtype,
    ):
        self._require_vtla_slow_cache_contract()
        if marker_displacement_history is None:
            raise ValueError("Fast VTLA inference requires marker_displacement_history")
        if tactile_sensor_mask is None:
            tactile_sensor_mask = torch.ones(
                batch_size, 1, device=device, dtype=torch.bool
            )
        elif tactile_sensor_mask.ndim == 1 and batch_size == 1:
            tactile_sensor_mask = tactile_sensor_mask.unsqueeze(0)
        tactile_sensor_mask = tactile_sensor_mask.to(device=device, dtype=torch.bool)
        marker_displacement_history = marker_displacement_history.to(
            device=device, dtype=self.tactile_encoder.marker_dtype
        )
        marker_valid_mask = marker_valid_mask.to(device=device, dtype=torch.bool)
        marker_history_valid_mask = marker_history_valid_mask.to(
            device=device, dtype=torch.bool
        )
        if marker_contact_state is not None:
            marker_contact_state = marker_contact_state.to(device=device)

        if device.type == "cuda":
            marker_start = torch.cuda.Event(enable_timing=True)
            marker_end = torch.cuda.Event(enable_timing=True)
            marker_start.record()
        else:
            marker_started = time.perf_counter()
        marker_tokens, marker_mask = self.tactile_encoder.encode_markers(
            marker_displacement_history,
            marker_valid_mask,
            marker_history_valid_mask,
            tactile_sensor_mask,
            marker_contact_state,
        )
        if device.type == "cuda":
            marker_end.record()
            marker_end.synchronize()
            marker_ms = float(marker_start.elapsed_time(marker_end))
        else:
            marker_ms = (time.perf_counter() - marker_started) * 1000.0
        marker_tokens = marker_tokens.to(dtype=dtype).flatten(1, 2)
        marker_mask = marker_mask.flatten(1, 2)
        expected_markers = 4 * 48
        if marker_tokens.shape[1] != expected_markers:
            raise RuntimeError(
                f"Expected {expected_markers} marker tokens, got {marker_tokens.shape[1]}"
            )
        cfg = self.qwenvl_with_expert.qwenvl.config
        marker_ids = torch.full(
            marker_mask.shape,
            cfg.text_config.eos_token_id,
            device=device,
            dtype=torch.long,
        )

        query_started = time.perf_counter()
        query_tokens, query_mask, query_ids = self._build_task_query_tokens(
            batch_size, device, dtype
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        query_ms = (time.perf_counter() - query_started) * 1000.0
        embs = torch.cat([marker_tokens, query_tokens], dim=1)
        pad_masks = torch.cat([marker_mask, query_mask], dim=1).bool()
        input_ids = torch.cat([marker_ids, query_ids], dim=1)
        att_masks = torch.ones_like(pad_masks)
        return (
            embs,
            pad_masks,
            att_masks,
            input_ids,
            expected_markers,
            int(query_tokens.shape[1]),
            {"marker_mlp_ms": marker_ms, "task_query_ms": query_ms},
        )

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        query_spans = prefix_query_token_spans(
            prefix_len=hidden_states.shape[1],
            num_task_tokens=self.num_task_tokens,
            use_depth_align=True,
            use_future_depth=getattr(self, "use_future_depth", False),
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self,
                "future_video_share_future_depth_query",
                False,
            ),
        )
        start, end = query_spans["current_depth"]
        return hidden_states[:, start:end, :]

    def sample_cascaded_times(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        base_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample one time in each non-overlapping cascaded FM interval."""

        split = self.tactile_refinement_settings.tau_split
        if base_time is None:
            base_slow = torch.rand(batch_size, device=device, dtype=dtype)
            base_tactile = torch.rand(batch_size, device=device, dtype=dtype)
        elif base_time.ndim == 1:
            if tuple(base_time.shape) != (batch_size,):
                raise ValueError("base_time must be [B] or [B,2]")
            base_slow = base_time.to(device=device, dtype=dtype)
            # A legacy [B] training time controls the slow sample. The tactile
            # half still receives an independent sample as required by the
            # two-loss cascaded objective. Use [B,2] for fully explicit times.
            base_tactile = torch.rand(batch_size, device=device, dtype=dtype)
        elif base_time.ndim == 2 and tuple(base_time.shape) == (batch_size, 2):
            slow_time = base_time[:, 0].to(device=device, dtype=dtype)
            tactile_time = base_time[:, 1].to(device=device, dtype=dtype)
            if not ((slow_time >= split).all() and (slow_time <= 1).all()):
                raise ValueError("Explicit slow time must lie in [tau_split, 1]")
            if not ((tactile_time >= 0).all() and (tactile_time <= split).all()):
                raise ValueError("Explicit tactile time must lie in [0, tau_split]")
            return slow_time, tactile_time
        else:
            raise ValueError("base_time must be [B] or [B,2]")
        if not ((base_slow >= 0).all() and (base_slow <= 1).all()):
            raise ValueError("base_time values must lie in [0,1]")
        slow_time = split + (1.0 - split) * base_slow
        tactile_time = split * base_tactile
        return slow_time, tactile_time

    def _slow_context_mask_from_pad(self, pad_mask: Tensor) -> Tensor:
        cache = VTLASlowCache(
            past_key_values={},
            pad_masks=pad_mask,
            att_masks=torch.ones_like(pad_mask),
            input_ids=torch.zeros_like(pad_mask, dtype=torch.long),
            position_ids=torch.zeros(
                3, pad_mask.shape[0], pad_mask.shape[1],
                device=pad_mask.device, dtype=torch.long,
            ),
            rope_grid_thw=torch.empty(0, 3, device=pad_mask.device, dtype=torch.long),
            prefix_len=pad_mask.shape[1],
            includes_task_queries=True,
        )
        return self._tactile_slow_context_mask(cache)

    def _compute_alignment_losses(
        self,
        outputs_embeds: Tensor,
        *,
        depth_targets,
        img_masks,
        future_depth_targets,
        future_video_targets,
        future_video_cls_targets,
        future_video_current_patch,
    ):
        align_metrics: dict[str, Tensor] = {}
        if self.config.align_params == {}:
            return 0, 0, 0, None, None, None, None, align_metrics
        loss_depth, loss_future_depth, depth_preds, future_depth_preds = (
            self.depth_emb_forward(
                outputs_embeds, depth_targets, img_masks, future_depth_targets
            )
        )
        loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
        loss_future_depth = loss_future_depth * self.config.align_params.get(
            "future_depth_loss_weight", 1.0
        )
        loss_future_video = 0
        future_video_preds = None
        current_video_preds = None
        if getattr(self, "use_future_video", False):
            loss_video, future_video_preds, video_metrics = self.video_emb_forward(
                outputs_embeds,
                future_video_targets,
                future_video_cls_targets=future_video_cls_targets,
                future_video_current_patch=future_video_current_patch,
            )
            video_total_loss = loss_video
            if (
                getattr(self, "use_current_video_patch", False)
                and future_video_current_patch is not None
            ):
                (
                    current_video_loss,
                    current_video_preds,
                    current_video_metrics,
                ) = self.current_video_emb_forward(
                    outputs_embeds, future_video_current_patch
                )
                video_total_loss = video_total_loss + current_video_loss
                video_metrics.update(current_video_metrics)
                video_metrics["align/current_video_loss"] = current_video_loss.detach()
            video_cfg = self.config.align_params.get("video", {})
            video_weight = video_cfg.get(
                "future_video_loss_weight",
                self.config.align_params.get(
                    "future_video_loss_weight",
                    self.config.align_params["depth_loss_weight"],
                ),
            )
            loss_future_video = video_total_loss * video_weight
            align_metrics.update(video_metrics)
            if "align/current_video_loss" in align_metrics:
                align_metrics["align/current_video_loss_weighted"] = (
                    align_metrics["align/current_video_loss"] * video_weight
                )
            align_metrics["align/future_video_loss"] = loss_video.detach()
            align_metrics["align/future_video_loss_weighted"] = (
                loss_video * video_weight
            ).detach()
            align_metrics["align/video_loss"] = video_total_loss.detach()
            align_metrics["align/video_loss_weighted"] = loss_future_video.detach()
        self.steps += 1
        return (
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
            align_metrics,
        )

    def _forward_cascaded(
        self,
        *,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise,
        time_values,
        loss_type,
        depth_targets,
        image_grid_thw,
        future_depth_targets,
        future_video_targets,
        future_video_cls_targets,
        future_video_current_patch,
        tactile_rgb,
        tactile_rgb_grid_thw,
        marker_displacement_history,
        marker_valid_mask,
        marker_history_valid_mask,
        marker_contact_state,
        tactile_sensor_mask,
        tactile_rgb_mask,
    ):
        """Train both halves of cascaded Flow Matching on every batch."""

        dtype, device, batch = state.dtype, state.device, state.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        slow_time, tactile_time = self.sample_cascaded_times(
            batch, device, dtype, time_values
        )
        target_velocity = noise - actions
        x_slow = slow_time[:, None, None] * noise + (
            1.0 - slow_time[:, None, None]
        ) * actions
        x_tactile = tactile_time[:, None, None] * noise + (
            1.0 - tactile_time[:, None, None]
        ) * actions
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self._embed_prefix_vtla(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            image_grid_thw=image_grid_thw,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=None,
            marker_valid_mask=None,
            marker_history_valid_mask=None,
            marker_contact_state=None,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
            _slow_only=True,
            _include_task_queries=True,
        )
        slow_time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix(state, x_slow, slow_time)
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention = make_att_2d_masks(pad_masks, att_masks)
        prefix_len = prefix_pad_masks.shape[1]
        if self.block_future_depth_to_action:
            attention = block_suffix_to_fv_(
                attention,
                suffix_row_start=prefix_len,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )
        attention = self._block_suffix_to_future_video_if_enabled_(
            attention, suffix_row_start=prefix_len, prefix_len=prefix_len
        )
        position_ids = self._build_full_position_ids(
            prefix_position_ids, prefix_pad_masks, suffix_pad_masks
        )
        (slow_hidden, suffix_out), _, router_logits = (
            self.qwenvl_with_expert.forward(
                attention_mask=attention,
                position_ids=position_ids,
                vlm_position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
                ada_cond=(
                    slow_time_embs
                    if getattr(self.config, "adanorm_time", False)
                    else None
                ),
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        )
        action_context = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = action_context
        slow_velocity = self.action_out_proj(
            suffix_out.to(self.action_out_proj.weight.dtype)
        ).to(dtype=dtype)
        marker_tokens, marker_mask = self.tactile_encoder.encode_markers(
            marker_displacement_history.to(
                device=device, dtype=self.tactile_encoder.marker_dtype
            ),
            marker_valid_mask.to(device=device, dtype=torch.bool),
            marker_history_valid_mask.to(device=device, dtype=torch.bool),
            tactile_sensor_mask,
            marker_contact_state.to(device=device),
        )
        marker_tokens = marker_tokens.flatten(1, 2)
        marker_mask = marker_mask.flatten(1, 2)
        tactile_velocity = self._predict_tactile_velocity(
            state=state,
            x_t=x_tactile,
            timestep=tactile_time,
            marker_tokens=marker_tokens,
            marker_mask=marker_mask,
            action_context_tokens=action_context,
            action_context_mask=torch.ones(
                action_context.shape[:2], device=device, dtype=torch.bool
            ),
            slow_context_tokens=slow_hidden,
            slow_context_mask=self._slow_context_mask_from_pad(prefix_pad_masks),
        )
        if loss_type == "fm":
            slow_losses = F.mse_loss(
                target_velocity, slow_velocity, reduction="none"
            )
            tactile_losses = F.mse_loss(
                target_velocity, tactile_velocity, reduction="none"
            )
        elif loss_type == "L1_fm":
            slow_losses = F.l1_loss(
                target_velocity, slow_velocity, reduction="none"
            )
            tactile_losses = F.l1_loss(
                target_velocity, tactile_velocity, reduction="none"
            )
        else:
            raise ValueError(f"Unsupported cascaded loss_type={loss_type!r}")
        loss_cfg = self.tactile_refinement_settings.loss
        losses = (
            loss_cfg.slow_weight * slow_losses
            + loss_cfg.tactile_weight * tactile_losses
        )
        seq_wise_loss, router_z_loss, metrics = self._moe_losses_and_metrics(
            router_logits, slow_losses
        )
        metrics.update(
            {
                "loss/slow_fm": slow_losses.mean().detach(),
                "loss/tactile_fm": tactile_losses.mean().detach(),
                "tactile_refinement/marker_valid_ratio": marker_mask.float()
                .mean()
                .detach(),
                "tactile_refinement/gate_on_ratio": self.tactile_encoder
                .last_marker_diagnostics["contact_on_ratio"],
                "tactile_refinement/tau_split": losses.new_tensor(
                    self.tactile_refinement_settings.tau_split
                ),
            }
        )
        (
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
            align_metrics,
        ) = self._compute_alignment_losses(
            slow_hidden,
            depth_targets=depth_targets,
            img_masks=img_masks,
            future_depth_targets=future_depth_targets,
            future_video_targets=future_video_targets,
            future_video_cls_targets=future_video_cls_targets,
            future_video_current_patch=future_video_current_patch,
        )
        metrics.update(align_metrics)
        return (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            seq_wise_loss,
            router_z_loss,
            metrics,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        )

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        loss_type="fm",
        depth_targets=None,
        image_grid_thw=None,
        future_depth_targets=None,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
        tactile_rgb=None,
        tactile_rgb_grid_thw=None,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
    ) -> Tensor:
        if self.tactile_refinement_settings.cascaded_enabled:
            return self._forward_cascaded(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                actions=actions,
                noise=noise,
                time_values=time,
                loss_type=loss_type,
                depth_targets=depth_targets,
                image_grid_thw=image_grid_thw,
                future_depth_targets=future_depth_targets,
                future_video_targets=future_video_targets,
                future_video_cls_targets=future_video_cls_targets,
                future_video_current_patch=future_video_current_patch,
                tactile_rgb=tactile_rgb,
                tactile_rgb_grid_thw=tactile_rgb_grid_thw,
                marker_displacement_history=marker_displacement_history,
                marker_valid_mask=marker_valid_mask,
                marker_history_valid_mask=marker_history_valid_mask,
                marker_contact_state=marker_contact_state,
                tactile_sensor_mask=tactile_sensor_mask,
                tactile_rgb_mask=tactile_rgb_mask,
            )
        dtype = state.dtype
        device = state.device
        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)
        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
        )
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        prefix_len = prefix_pad_masks.shape[1]
        if self.block_future_depth_to_action:
            att_2d_masks = block_suffix_to_fv_(
                att_2d_masks,
                suffix_row_start=prefix_len,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )

        att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            att_2d_masks,
            suffix_row_start=prefix_len,
            prefix_len=prefix_len,
        )
        position_ids = self._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad_masks)

        (outputs_embeds, suffix_out), _, router_logits_list = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        align_metrics = {}
        if self.config.align_params != {}:
            loss_depth, loss_future_depth, depth_preds, future_depth_preds = self.depth_emb_forward(outputs_embeds, depth_targets, img_masks,future_depth_targets,)
            loss_depth = loss_depth * self.config.align_params["depth_loss_weight"]
            loss_future_depth = loss_future_depth * self.config.align_params.get("future_depth_loss_weight", 1.0)
            loss_future_video = 0
            future_video_preds = None
            current_video_preds = None
            if getattr(self, "use_future_video", False):
                loss_video, future_video_preds, video_metrics = self.video_emb_forward(
                    outputs_embeds,
                    future_video_targets,
                    future_video_cls_targets=future_video_cls_targets,
                    future_video_current_patch=future_video_current_patch,
                )
                video_total_loss = loss_video
                if (
                    getattr(self, "use_current_video_patch", False)
                    and future_video_current_patch is not None
                ):
                    current_video_loss, current_video_preds, current_video_metrics = self.current_video_emb_forward(
                        outputs_embeds,
                        future_video_current_patch,
                    )
                    video_total_loss = video_total_loss + current_video_loss
                    video_metrics.update(current_video_metrics)
                    video_metrics["align/current_video_loss"] = current_video_loss.detach()
                video_cfg = self.config.align_params.get("video", {})
                video_weight = video_cfg.get(
                    "future_video_loss_weight",
                    self.config.align_params.get(
                        "future_video_loss_weight",
                        self.config.align_params["depth_loss_weight"],
                    ),
                )
                loss_future_video = video_total_loss * video_weight
                align_metrics.update(video_metrics)
                if "align/current_video_loss" in align_metrics:
                    align_metrics["align/current_video_loss_weighted"] = (
                        align_metrics["align/current_video_loss"] * video_weight
                    )
                align_metrics["align/future_video_loss"] = loss_video.detach()
                align_metrics["align/future_video_loss_weighted"] = (loss_video * video_weight).detach()
                align_metrics["align/video_loss"] = video_total_loss.detach()
                align_metrics["align/video_loss_weighted"] = loss_future_video.detach()
            self.steps += 1
        else:
            loss_depth = 0
            loss_future_depth = 0
            loss_future_video = 0
            depth_preds = None
            future_depth_preds = None
            future_video_preds = None
            current_video_preds = None

        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)

        if self.tactile_settings.debug_shapes and not getattr(
            self, "_tactile_flow_debug_logged", False
        ):
            logger.info(
                "VTLA Flow Matching shapes: state=%s noisy_actions=%s flow_time=%s "
                "predicted_velocity=%s target_velocity=%s",
                tuple(state.shape),
                tuple(x_t.shape),
                tuple(time.shape),
                tuple(v_t.shape),
                tuple(u_t.shape),
            )
            self._tactile_flow_debug_logged = True

        if loss_type == "fm":
            losses = F.mse_loss(u_t, v_t, reduction="none")
        elif loss_type == "L1_fm":
            losses = F.l1_loss(u_t, v_t, reduction="none")

        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(
            router_logits_list, losses
        )
        if self.tactile_encoder is not None:
            for name, value in self.tactile_encoder.last_marker_diagnostics.items():
                if value.numel() == 1:
                    moe_metrics[f"tactile/{name}"] = value
                else:
                    for region_index, scalar in enumerate(value.flatten()):
                        moe_metrics[f"tactile/{name}/region_{region_index}"] = scalar
        if align_metrics:
            moe_metrics.update(align_metrics)
        return losses, loss_depth, loss_future_depth, loss_future_video, depth_preds, seq_wise_loss, router_z_loss, moe_metrics, future_depth_preds, future_video_preds, current_video_preds

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
        tactile_rgb=None,
        tactile_rgb_grid_thw=None,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
    ) -> Tensor:
        """Do a full Qwen3-VL inference forward and compute the action."""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        predict_velocity_fn = self.predict_velocity
        if getattr(self, "_use_compile_predict_velocity", False):
            predict_velocity_fn = getattr(self, "_compiled_predict_velocity", None)
            if predict_velocity_fn is None:
                predict_velocity_fn = torch.compile(
                    self.predict_velocity,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                self._compiled_predict_velocity = predict_velocity_fn

        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)
            v_t = predict_velocity_fn(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                prefix_position_ids=prefix_position_ids,
            )

            x_t += dt * v_t
            time += dt
        print(f"Denoise {count} steps")
        return x_t

    @torch.no_grad()
    def build_slow_cache(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        *,
        image_grid_thw,
        tactile_rgb,
        tactile_rgb_grid_thw,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
        scene_timestamp: float | None = None,
        tactile_rgb_timestamp: float | None = None,
        include_task_queries: bool = False,
    ) -> VTLASlowCache:
        """Build the reusable Scene -> Language -> tactile RGB VLM cache."""

        self._require_vtla_slow_cache_contract()
        profile: dict[str, float] = {}
        total_started = time.perf_counter()
        (
            slow_embs,
            slow_pad_masks,
            slow_att_masks,
            slow_position_ids,
            slow_visual_masks,
            slow_deepstack,
            slow_input_ids,
            rope_grid_thw,
        ) = self._embed_prefix_vtla(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            image_grid_thw=image_grid_thw,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=None,
            marker_valid_mask=None,
            marker_history_valid_mask=None,
            marker_contact_state=None,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
            _slow_only=True,
            _include_task_queries=include_task_queries,
            _return_cache_metadata=True,
            _profile_ms=profile,
        )
        slow_attention = make_att_2d_masks(slow_pad_masks, slow_att_masks)
        if slow_embs.is_cuda:
            vlm_start = torch.cuda.Event(enable_timing=True)
            vlm_end = torch.cuda.Event(enable_timing=True)
            vlm_start.record()
        else:
            vlm_started = time.perf_counter()
        slow_outputs, slow_kv, _ = self.qwenvl_with_expert.forward(
            attention_mask=slow_attention,
            position_ids=slow_position_ids,
            vlm_position_ids=slow_position_ids,
            past_key_values=None,
            inputs_embeds=[slow_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            visual_pos_masks=slow_visual_masks,
            deepstack_visual_embeds=slow_deepstack,
        )
        if slow_embs.is_cuda:
            vlm_end.record()
            vlm_end.synchronize()
            profile["slow_vlm_ms"] = float(vlm_start.elapsed_time(vlm_end))
        else:
            profile["slow_vlm_ms"] = (time.perf_counter() - vlm_started) * 1000.0
        profile["slow_cache_build_ms"] = (time.perf_counter() - total_started) * 1000.0
        version = int(getattr(self, "_vtla_slow_cache_version", 0)) + 1
        self._vtla_slow_cache_version = version
        return VTLASlowCache(
            past_key_values=slow_kv,
            pad_masks=slow_pad_masks,
            att_masks=slow_att_masks,
            input_ids=slow_input_ids,
            position_ids=slow_position_ids,
            rope_grid_thw=rope_grid_thw,
            prefix_len=int(slow_embs.shape[1]),
            version=version,
            scene_timestamp=scene_timestamp,
            tactile_rgb_timestamp=tactile_rgb_timestamp,
            profile_ms=profile,
            final_hidden_states=slow_outputs[0] if include_task_queries else None,
            includes_task_queries=include_task_queries,
        )

    @torch.no_grad()
    def build_cascaded_slow_context(self, *args, **kwargs) -> VTLASlowCache:
        """Build Scene/Language/TacRGB/Query context for cascaded inference."""

        if not self.tactile_refinement_settings.cascaded_enabled:
            raise RuntimeError("cascaded_flow is not enabled")
        kwargs["include_task_queries"] = True
        return self.build_slow_cache(*args, **kwargs)

    def _tactile_slow_context_mask(self, slow_cache: VTLASlowCache) -> Tensor:
        mask = slow_cache.pad_masks.clone().to(dtype=torch.bool)
        if not slow_cache.includes_task_queries or not self.use_depth_align:
            return mask
        spans = prefix_query_token_spans(
            prefix_len=mask.shape[1],
            num_task_tokens=self.num_task_tokens,
            use_depth_align=True,
            use_future_depth=getattr(self, "use_future_depth", False),
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self, "future_video_share_future_depth_query", False
            ),
        )
        for name, (start, end) in spans.items():
            if name.startswith("future_"):
                mask[:, start:end] = False
        return mask

    def _predict_tactile_velocity(
        self,
        *,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        marker_tokens: Tensor,
        marker_mask: Tensor,
        action_context_tokens: Tensor,
        action_context_mask: Tensor,
        slow_context_tokens: Tensor,
        slow_context_mask: Tensor,
    ) -> Tensor:
        modules = (
            self.tactile_action_expert,
            self.tactile_state_proj,
            self.tactile_action_in_proj,
            self.tactile_action_out_proj,
            self.tactile_context_proj,
            self.tactile_plan_proj,
            self.tactile_marker_proj,
            self.tactile_time_mlp,
        )
        if any(module is None for module in modules):
            raise RuntimeError("Cascaded tactile modules are not initialized")
        assert self.tactile_action_expert is not None
        assert self.tactile_state_proj is not None
        assert self.tactile_action_in_proj is not None
        assert self.tactile_action_out_proj is not None
        assert self.tactile_context_proj is not None
        assert self.tactile_plan_proj is not None
        assert self.tactile_marker_proj is not None
        assert self.tactile_time_mlp is not None
        if state.ndim != 2 or x_t.ndim != 3:
            raise ValueError("state and x_t must be [B,D] and [B,T,D]")
        if action_context_tokens.ndim != 3 or action_context_mask.ndim != 2:
            raise ValueError("action context and mask must be [B,T,D] and [B,T]")
        if action_context_tokens.shape[:2] != action_context_mask.shape:
            raise ValueError("action context and mask lengths differ")
        if action_context_tokens.shape[0] != state.shape[0]:
            raise ValueError("action context batch does not match state")
        if action_context_tokens.shape[1] != x_t.shape[1]:
            raise ValueError("action context must cover the full action horizon")
        hidden_dtype = self.tactile_action_in_proj.weight.dtype
        state_token = self.tactile_state_proj(state.to(hidden_dtype)).unsqueeze(1)
        action_tokens = self.tactile_action_in_proj(x_t.to(hidden_dtype))
        time_embedding = sinusoidal_time_embedding(
            timestep.to(device=x_t.device), action_tokens.shape[-1]
        ).to(dtype=hidden_dtype)
        time_embedding = self.tactile_time_mlp(time_embedding).unsqueeze(1)
        action_positions = sinusoidal_position_embedding(
            torch.arange(x_t.shape[1], device=x_t.device), action_tokens.shape[-1]
        ).to(dtype=hidden_dtype)
        hidden = torch.cat(
            [state_token, action_tokens + time_embedding + action_positions.unsqueeze(0)],
            dim=1,
        )
        marker_context = self.tactile_marker_proj(marker_tokens.to(hidden_dtype))
        action_context = self.tactile_plan_proj(
            action_context_tokens.to(hidden_dtype)
        )
        slow_context = self.tactile_context_proj(
            slow_context_tokens.to(hidden_dtype)
        )
        hidden = self.tactile_action_expert(
            hidden,
            marker_context,
            marker_mask,
            action_context,
            action_context_mask,
            slow_context,
            slow_context_mask,
        )
        return self.tactile_action_out_proj(hidden[:, 1:]).to(dtype=x_t.dtype)

    @torch.no_grad()
    def build_slow_action_plan(
        self,
        slow_context: VTLASlowCache,
        state: Tensor,
        *,
        noise: Tensor | None = None,
        action_offset: int = 0,
    ) -> SlowActionPlan:
        """Integrate the original 36-layer Action Expert from 1 to tau_split."""

        if not self.tactile_refinement_settings.cascaded_enabled:
            raise RuntimeError("cascaded_flow is not enabled")
        if not slow_context.includes_task_queries:
            raise ValueError("Cascaded slow context must include task queries")
        if slow_context.final_hidden_states is None:
            raise ValueError("Cascaded slow context is missing final hidden states")
        if not 0 <= action_offset <= self.config.n_action_steps:
            raise ValueError("action_offset is outside the action chunk")
        batch, device, dtype = state.shape[0], state.device, state.dtype
        if noise is None:
            noise = torch.randn(
                batch,
                self.config.n_action_steps,
                self.config.max_action_dim,
                device=device,
                dtype=dtype,
            )
        expected_shape = (
            batch,
            self.config.n_action_steps,
            self.config.max_action_dim,
        )
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"noise must be {expected_shape}, got {tuple(noise.shape)}")
        settings = self.tactile_refinement_settings
        step_size = (1.0 - settings.tau_split) / settings.inference.slow_steps
        x_t = noise.clone()
        started = time.perf_counter()
        for step in range(settings.inference.slow_steps):
            tau = 1.0 - step * step_size
            velocity = self.predict_velocity(
                state,
                slow_context.pad_masks,
                slow_context.past_key_values,
                x_t,
                torch.full((batch,), tau, device=device, dtype=dtype),
                prefix_position_ids=slow_context.position_ids,
            )
            x_t = x_t - step_size * velocity
        context_started = time.perf_counter()
        _, action_context = self.predict_velocity(
            state,
            slow_context.pad_masks,
            slow_context.past_key_values,
            x_t,
            torch.full(
                (batch,), settings.tau_split, device=device, dtype=dtype
            ),
            prefix_position_ids=slow_context.position_ids,
            return_action_hidden=True,
        )
        context_finished = time.perf_counter()
        profile = {
            "slow_action_stage_ms": (context_started - started) * 1000.0,
            "slow_action_plan_ms": (context_finished - started) * 1000.0,
        }
        profile["slow_action_context_ms"] = (
            context_finished - context_started
        ) * 1000.0
        return SlowActionPlan(
            x_split=x_t.detach().clone(),
            action_context=action_context.detach().clone(),
            tau_split=settings.tau_split,
            noise=noise.detach().clone(),
            state=state.detach().clone(),
            slow_context=slow_context,
            action_offset=int(action_offset),
            version=slow_context.version,
            profile_ms=profile,
        )

    @torch.no_grad()
    def refine_action_with_tactile(
        self,
        plan: SlowActionPlan,
        *,
        marker_displacement_history: Tensor,
        marker_valid_mask: Tensor,
        marker_history_valid_mask: Tensor,
        marker_contact_state: Tensor,
        state: Tensor,
        tactile_sensor_mask: Tensor | None = None,
        action_offset: int | None = None,
        fixed_action_prefix: Tensor | None = None,
        return_profile: bool = False,
    ):
        """Integrate only the tactile expert from tau_split to zero."""

        if not self.tactile_refinement_settings.cascaded_enabled:
            raise RuntimeError("cascaded_flow is not enabled")
        if plan.version != plan.slow_context.version:
            raise ValueError("SlowActionPlan context version is inconsistent")
        if plan.slow_context.final_hidden_states is None:
            raise ValueError("SlowActionPlan has no reusable Slow hidden states")
        offset = plan.action_offset if action_offset is None else int(action_offset)
        if not 0 <= offset <= plan.x_split.shape[1]:
            raise ValueError("action_offset is outside the action chunk")
        if state.shape[0] != plan.x_split.shape[0]:
            raise ValueError("latest state batch does not match SlowActionPlan")
        total_started = time.perf_counter()
        marker_started = time.perf_counter()
        marker_tokens, marker_mask = self.tactile_encoder.encode_markers(
            marker_displacement_history.to(
                device=state.device, dtype=self.tactile_encoder.marker_dtype
            ),
            marker_valid_mask.to(device=state.device, dtype=torch.bool),
            marker_history_valid_mask.to(device=state.device, dtype=torch.bool),
            tactile_sensor_mask,
            marker_contact_state.to(device=state.device),
        )
        marker_tokens = marker_tokens.flatten(1, 2)
        marker_mask = marker_mask.flatten(1, 2)
        marker_ms = (time.perf_counter() - marker_started) * 1000.0
        if marker_tokens.shape[1] != 192:
            raise RuntimeError(f"Expected 192 Marker tokens, got {marker_tokens.shape[1]}")
        x_t = plan.x_split.detach().clone().to(device=state.device, dtype=state.dtype)
        fixed_prefix = x_t[:, :offset].clone()
        if fixed_action_prefix is not None:
            expected_prefix_shape = (x_t.shape[0], offset, x_t.shape[2])
            if tuple(fixed_action_prefix.shape) != expected_prefix_shape:
                raise ValueError(
                    "fixed_action_prefix must have shape "
                    f"{expected_prefix_shape}, got {tuple(fixed_action_prefix.shape)}"
                )
            fixed_prefix = fixed_action_prefix.to(
                device=state.device, dtype=state.dtype
            ).detach().clone()
        settings = self.tactile_refinement_settings
        step_size = settings.tau_split / settings.inference.tactile_steps
        slow_tokens = plan.slow_context.final_hidden_states.to(device=state.device)
        slow_mask = self._tactile_slow_context_mask(plan.slow_context).to(
            device=state.device
        )
        action_context = plan.action_context.to(device=state.device)
        action_context_mask = torch.ones(
            action_context.shape[:2], device=state.device, dtype=torch.bool
        )
        expert_started = time.perf_counter()
        for step in range(settings.inference.tactile_steps):
            tau = settings.tau_split - step * step_size
            velocity = self._predict_tactile_velocity(
                state=state,
                x_t=x_t,
                timestep=torch.full(
                    (state.shape[0],), tau, device=state.device, dtype=state.dtype
                ),
                marker_tokens=marker_tokens,
                marker_mask=marker_mask,
                action_context_tokens=action_context,
                action_context_mask=action_context_mask,
                slow_context_tokens=slow_tokens,
                slow_context_mask=slow_mask,
            )
            x_t = x_t - step_size * velocity
            if offset:
                x_t[:, :offset] = fixed_prefix
        expert_ms = (time.perf_counter() - expert_started) * 1000.0
        output = x_t
        profile = {
            "tactile_marker_encode_ms": marker_ms,
            "tactile_expert_ms": expert_ms,
            "tactile_refinement_ms": (time.perf_counter() - total_started) * 1000.0,
        }
        return (output, profile) if return_profile else output

    @torch.no_grad()
    def extend_slow_cache(
        self,
        slow_cache: VTLASlowCache,
        *,
        marker_displacement_history,
        marker_valid_mask,
        marker_history_valid_mask,
        marker_contact_state,
        tactile_sensor_mask=None,
        dtype=None,
    ) -> VTLAFastPrefix:
        """Append Marker/Query into a disposable working cache."""

        self._require_vtla_slow_cache_contract()
        device = slow_cache.pad_masks.device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        (
            tail_embs,
            tail_pad_masks,
            tail_att_masks,
            tail_input_ids,
            marker_count,
            query_count,
            profile,
        ) = self._build_fast_vtla_tail(
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            batch_size=slow_cache.pad_masks.shape[0],
            device=device,
            dtype=dtype,
        )
        full_pad = torch.cat([slow_cache.pad_masks, tail_pad_masks], dim=1)
        full_att = torch.cat([slow_cache.att_masks, tail_att_masks], dim=1)
        full_ids = torch.cat([slow_cache.input_ids, tail_input_ids], dim=1)
        full_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            full_ids,
            full_pad.long(),
            image_grid_thw=slow_cache.rope_grid_thw,
            video_grid_thw=None,
        )
        full_attention = make_att_2d_masks(full_pad, full_att)
        tail_length = tail_embs.shape[1]
        continuation_attention = full_attention[:, -tail_length:, :]
        working_kv = {
            layer: {
                "key_states": values["key_states"],
                "value_states": values["value_states"],
            }
            for layer, values in slow_cache.past_key_values.items()
        }
        if tail_embs.is_cuda:
            continuation_start = torch.cuda.Event(enable_timing=True)
            continuation_end = torch.cuda.Event(enable_timing=True)
            continuation_start.record()
        else:
            continuation_started = time.perf_counter()
        _, full_kv, _ = self.qwenvl_with_expert.forward(
            attention_mask=continuation_attention,
            position_ids=full_position_ids[:, :, -tail_length:],
            vlm_position_ids=full_position_ids[:, :, -tail_length:],
            past_key_values=working_kv,
            inputs_embeds=[tail_embs, None],
            use_cache=True,
            fill_kv_cache=False,
            append_kv_cache=True,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )
        if tail_embs.is_cuda:
            continuation_end.record()
            continuation_end.synchronize()
            profile["marker_query_continuation_ms"] = float(
                continuation_start.elapsed_time(continuation_end)
            )
        else:
            profile["marker_query_continuation_ms"] = (
                time.perf_counter() - continuation_started
            ) * 1000.0
        if any(
            values["key_states"].shape[1] != slow_cache.prefix_len
            for values in slow_cache.past_key_values.values()
        ):
            raise RuntimeError("Permanent VTLA slow cache was mutated during continuation")
        return VTLAFastPrefix(
            past_key_values=full_kv,
            pad_masks=full_pad,
            position_ids=full_position_ids,
            marker_token_count=marker_count,
            query_token_count=query_count,
            profile_ms=profile,
        )

    @torch.no_grad()
    def sample_actions_fast(
        self,
        slow_cache: VTLASlowCache,
        state,
        *,
        marker_displacement_history,
        marker_valid_mask,
        marker_history_valid_mask,
        marker_contact_state,
        tactile_sensor_mask=None,
        noise=None,
        return_profile: bool = False,
    ):
        """Replan from the current marker/state snapshot without any RGB ViT."""

        total_started = time.perf_counter()
        bsize, device, dtype = state.shape[0], state.device, state.dtype
        if noise is None:
            noise = torch.randn(
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
                device=device,
                dtype=dtype,
            )
        fast_prefix = self.extend_slow_cache(
            slow_cache,
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            dtype=dtype,
        )
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        flow_time = torch.tensor(1.0, dtype=dtype, device=device)
        fm_started = time.perf_counter()
        predict_velocity_fn = self.predict_velocity
        if getattr(self, "_use_compile_predict_velocity", False):
            predict_velocity_fn = getattr(self, "_compiled_predict_velocity", None)
            if predict_velocity_fn is None:
                predict_velocity_fn = torch.compile(
                    self.predict_velocity,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                self._compiled_predict_velocity = predict_velocity_fn
        count = 0
        while flow_time >= -dt / 2:
            count += 1
            velocity = predict_velocity_fn(
                state,
                fast_prefix.pad_masks,
                fast_prefix.past_key_values,
                x_t,
                flow_time.expand(bsize),
                prefix_position_ids=fast_prefix.position_ids,
            )
            x_t += dt * velocity
            flow_time += dt
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile = dict(fast_prefix.profile_ms)
        profile["fm_sampling_ms"] = (time.perf_counter() - fm_started) * 1000.0
        profile["fast_replan_total_ms"] = (time.perf_counter() - total_started) * 1000.0
        self.last_vtla_fast_profile_ms = profile
        print(f"Denoise {count} steps")
        return (x_t, profile) if return_profile else x_t

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
        return_action_hidden: bool = False,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states."""
        if prefix_position_ids is None:
            raise ValueError("FlowMatchingV2.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        if self.block_future_depth_to_action:
            # Query rows here are all suffix (state/action), so row start is 0.
            full_att_2d_masks = block_suffix_to_fv_(
                full_att_2d_masks,
                suffix_row_start=0,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )
        full_att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            full_att_2d_masks,
            suffix_row_start=0,
            prefix_len=prefix_len,
        )

        full_position_ids = self._build_full_position_ids(
            prefix_position_ids,
            prefix_pad_masks,
            suffix_pad_masks,
        )
        position_ids = full_position_ids[:, :, -suffix_len:]

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        if return_action_hidden:
            return v_t, suffix_out
        return v_t

    def _moe_losses_and_metrics(self, router_logits_list, losses):
        router_z_loss_coeff = getattr(self.config, "router_z_loss_coeff", 0)
        router_z_loss = losses.new_zeros(())
        router_z_layer_losses = None  # per-layer raw z-loss (pre-coeff), for monitoring
        if router_z_loss_coeff > 0 and router_logits_list:
            router_z_layer_losses = [
                torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
                for logits in router_logits_list
            ]
            router_z_loss = router_z_loss_coeff * torch.stack(router_z_layer_losses).mean()

        seq_wise_loss_coeff = getattr(self.config, "sequence_wise_loss_coeff", 0)
        seq_wise_loss = 0
        seqwise_layer_losses = None  # per-layer raw seq-wise balance loss (pre-coeff), for monitoring
        if seq_wise_loss_coeff > 0 and router_logits_list:
            # router_logits are [B*T, E] (action-expert tokens, fixed length T per sample).
            # per_sequence -> balance experts within each sample's T tokens (DeepSeek-V3 intent);
            # global -> treat the whole B*T batch as one sequence.
            mode = getattr(self.config, "sequence_wise_mode", "per_sequence")
            score_func = getattr(self.config, "router_activation", "softmax")
            if mode == "global":
                seq_lengths = None
            else:
                B = losses.shape[0]
                N = router_logits_list[0].shape[0]
                seq_lengths = [N // B] * B
            seqwise_layer_losses = triton_sequence_wise_balance_loss(
                router_logits_list=tuple(router_logits_list),
                top_k=getattr(self.config, "token_top_k", 4),
                seq_lengths=seq_lengths,
                padding_len=0,
                score_func=score_func,
            )
            if seqwise_layer_losses:
                seq_wise_loss = seq_wise_loss_coeff * torch.stack(seqwise_layer_losses).mean()

        moe_metrics = {}
        if router_logits_list:
            token_moe_layers_list = sorted(getattr(self.config, "token_moe_layers", None) or [])
            all_moe_indices = token_moe_layers_list
            token_expert_counts = []
            # Per-layer token-MoE stats, collected for moe_summary/* cross-layer aggregates.
            tok_maxvio, tok_minvio, tok_minload, tok_entropy, tok_sigmoid = [], [], [], [], []
            tok_bias = []  # per-layer max(|e_score_correction_bias|) (loss-free); >1 -> bias dominates sigmoid score
            any_dead = None  # OR-accumulated bool: any token-MoE layer with a 0-count expert
            with torch.no_grad():
                for i, logits in enumerate(router_logits_list):
                    layer_id = all_moe_indices[i] if i < len(all_moe_indices) else i
                    num_experts = logits.shape[-1]
                    routing_probs = F.softmax(logits, dim=1, dtype=torch.float)
                    moe_block = self.qwenvl_with_expert.qwen_expert.model.layers[layer_id].mlp
                    if hasattr(moe_block, "last_tokens_per_expert"):
                        # Global (all-reduced), biased, true top-k load from the load-balance hook.
                        counts = moe_block.last_tokens_per_expert.clone()
                        if counts.sum() == 0:
                            # Buffer not yet populated by the load-balance hook (first step
                            # after run start / resume) -> skip this layer to avoid a spurious
                            # has_dead_expert / min_load_ratio spike on the very first viz.
                            continue
                    else:
                        _, selected = torch.topk(routing_probs, 1, dim=-1)
                        counts = F.one_hot(selected.squeeze(-1), num_classes=num_experts).float().sum(dim=0)
                    avg_load = counts.mean()
                    denom = avg_load.clamp(min=1e-9)
                    maxvio = (counts.max() - avg_load) / denom          # peak overload  (>=0, larger=worse)
                    minvio = (avg_load - counts.min()) / denom          # valley underload (=1 -> dead expert)
                    min_load_ratio = counts.min() / denom               # =0 -> dead expert
                    # entropy is rank-local (this rank's routing_probs, last micro-batch).
                    per_sample_entropy = -(routing_probs * routing_probs.clamp(min=1e-9).log()).sum(dim=-1)
                    entropy = per_sample_entropy.mean()
                    ll = f"{layer_id:02d}"
                    token_expert_counts.append((layer_id, counts))
                    moe_metrics[f"moe_maxvio/layer{ll}"] = maxvio
                    moe_metrics[f"moe_minvio/layer{ll}"] = minvio
                    moe_metrics[f"moe_minload/layer{ll}"] = min_load_ratio
                    moe_metrics[f"moe_entropy_rank0/layer{ll}"] = entropy
                    tok_maxvio.append(maxvio)
                    tok_minvio.append(minvio)
                    tok_minload.append(min_load_ratio)
                    tok_entropy.append(entropy)
                    dead = counts.min() == 0
                    any_dead = dead if any_dead is None else (any_dead | dead)
                    if hasattr(moe_block, "avg_topk_sigmoid_score"):
                        sig = moe_block.avg_topk_sigmoid_score.detach().reshape(()).to(denom)
                        moe_metrics[f"moe_topksigmoid_rank0/layer{ll}"] = sig
                        tok_sigmoid.append(sig)
                    if hasattr(moe_block, "e_score_correction_bias"):
                        bias_absmax = moe_block.e_score_correction_bias.detach().abs().max().to(denom)
                        moe_metrics[f"moe_bias/layer{ll}"] = bias_absmax
                        tok_bias.append(bias_absmax)
                # ---- moe_summary/* : cross-layer aggregates over token-MoE layers (written every step) ----
                if tok_maxvio:
                    moe_metrics["moe_summary/maxvio_avg"] = torch.stack(tok_maxvio).mean()
                    moe_metrics["moe_summary/maxvio_max"] = torch.stack(tok_maxvio).max()
                    moe_metrics["moe_summary/minvio_avg"] = torch.stack(tok_minvio).mean()
                    moe_metrics["moe_summary/minvio_max"] = torch.stack(tok_minvio).max()
                    moe_metrics["moe_summary/min_load_ratio"] = torch.stack(tok_minload).min()
                    moe_metrics["moe_summary/has_dead_expert"] = any_dead.float()
                    moe_metrics["moe_summary/entropy_avg_rank0"] = torch.stack(tok_entropy).mean()
                if tok_sigmoid:
                    moe_metrics["moe_summary/topk_sigmoid_avg_rank0"] = torch.stack(tok_sigmoid).mean()
                if tok_bias:
                    moe_metrics["moe_summary/bias_absmax"] = torch.stack(tok_bias).max()
                # ---- moe_seqwise/* : per-layer raw sequence-wise balance loss (pre-coeff) + average ----
                if seqwise_layer_losses and len(seqwise_layer_losses) == len(all_moe_indices):
                    sw_vals = []
                    for lid, sw in zip(all_moe_indices, seqwise_layer_losses):
                        v = sw.detach()
                        moe_metrics[f"moe_seqwise/layer{lid:02d}"] = v
                        sw_vals.append(v)
                    moe_metrics["moe_seqwise/avg"] = torch.stack(sw_vals).mean()
                # ---- moe_zloss/* : per-layer raw router z-loss (pre-coeff) + average/weighted loss ----
                if router_z_layer_losses and len(router_z_layer_losses) == len(all_moe_indices):
                    zl_vals = []
                    for lid, zl in zip(all_moe_indices, router_z_layer_losses):
                        v = zl.detach()
                        moe_metrics[f"moe_zloss/layer{lid:02d}"] = v
                        zl_vals.append(v)
                    moe_metrics["moe_zloss/avg_raw"] = torch.stack(zl_vals).mean()
                    moe_metrics["moe_zloss/weighted"] = router_z_loss.detach()
                if token_expert_counts:
                    moe_metrics["_token_moe_expert_counts"] = token_expert_counts
        return seq_wise_loss, router_z_loss, moe_metrics


class LingbotVlaV2Policy(PreTrainedModel):
    config_class = LingbotVLAV2Config
    name = "torch_lingbot_vla_v2"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm", "FixAdaRMSNorm"]

    def get_parallel_plan(self):
        from lingbotvla.distributed.parallel_plan import ParallelPlan
        from torch.distributed._tensor import Shard

        ep_plan = {
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.gate_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.up_proj": Shard(0),
            "model.qwenvl_with_expert.qwen_expert.model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(ep_plan=ep_plan)

    @classmethod
    def get_weight_loader(cls):
        return LingBotVLAWeightLoader()

    def __init__(self, config: LingbotVLAV2Config, eval: bool = False):
        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = FlowMatchingV2(config, eval)
        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head
        self.reset()
        torch.set_float32_matmul_precision("high")

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    def checkpoint_allowed_missing_parameters(
        self,
        names: set[str],
        *,
        loaded_parameter_names: Optional[set[str]] = None,
    ) -> set[str]:
        """Allow the tactile namespace only for a genuinely tactile-free checkpoint."""

        if not getattr(self.config, "tactile_enabled", False):
            return set()
        tactile_prefix = "model.tactile_encoder."
        refinement_prefixes = (
            "model.tactile_action_expert.",
            "model.tactile_state_proj.",
            "model.tactile_action_in_proj.",
            "model.tactile_action_out_proj.",
            "model.tactile_context_proj.",
            "model.tactile_plan_proj.",
            "model.tactile_marker_proj.",
            "model.tactile_time_mlp.",
        )
        refinement_enabled = getattr(
            self.config, "tactile_refinement_enabled", False
        )
        checkpoint_has_refinement = bool(
            loaded_parameter_names
            and any(
                any(name.startswith(prefix) for prefix in refinement_prefixes)
                for name in loaded_parameter_names
            )
        )
        checkpoint_has_tactile = bool(
            loaded_parameter_names
            and any(name.startswith(tactile_prefix) for name in loaded_parameter_names)
        )
        allowed = set()
        if not checkpoint_has_tactile:
            allowed.update(
                name for name in names if name.startswith(tactile_prefix)
            )
        if refinement_enabled and not checkpoint_has_refinement:
            allowed.update(
                name
                for name in names
                if any(name.startswith(prefix) for prefix in refinement_prefixes)
            )
        if refinement_enabled and checkpoint_has_refinement:
            allowed.update(
                name
                for name in names
                if (
                    name.startswith("model.tactile_plan_proj.")
                    or (
                        name.startswith("model.tactile_action_expert.")
                        and (".plan_norm." in name or ".plan_attention." in name)
                    )
                )
            )
        if allowed:
            logger.warning(
                "Initializing checkpoint-compatible VTLA parameters: %s",
                sorted(allowed),
            )
        return allowed

    def forward(
        self,
        images,
        img_masks,
        state,
        lang_tokens,
        lang_masks,
        actions,
        joint_mask=None,
        action_is_pad=None,
        noise=None,
        time=None,
        depth_targets=None,
        image_grid_thw=None,
        future_depth_targets=None,
        future_video_targets=None,
        future_video_cls_targets=None,
        future_video_current_patch=None,
        tactile_rgb=None,
        tactile_rgb_grid_thw=None,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
        **kwargs
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}
        if getattr(self.config, "action_fp32", False):
            state = state.float()
            actions = actions.float()
        (
            losses,
            loss_depth,
            loss_future_depth,
            loss_future_video,
            depth_preds,
            seq_wise_loss,
            router_z_loss,
            moe_metrics,
            future_depth_preds,
            future_video_preds,
            current_video_preds,
        ) = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            loss_type=self.config.loss_type,
            depth_targets=depth_targets,
            image_grid_thw=image_grid_thw,
            future_depth_targets=future_depth_targets,
            future_video_targets=future_video_targets,
            future_video_cls_targets=future_video_cls_targets,
            future_video_current_patch=future_video_current_patch,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
        )

        if joint_mask is not None:
            if "repeat" in self.config.loss_type:
                joint_mask = joint_mask.repeat(2, 1, 1)
            assert len(joint_mask.shape) == 3
            
            masked_losses = losses * joint_mask
            valid_counts = joint_mask.sum(dim=(1, 2)).clamp(min=1)
            batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
            loss_vla = masked_losses.sum() / joint_mask.sum().clamp(min=1)
        else:
            losses = losses[:, :, : self.config.action_dim]
            batch_mean_losses = losses.mean(dim=(1, 2))
            loss_vla = losses.mean()

        loss_dict["batch_mean_losses"] = batch_mean_losses.detach()
        total_loss = (
            loss_vla
            + loss_depth
            + loss_future_depth
            + loss_future_video
            + seq_wise_loss
            + router_z_loss
        )
        loss_dict["router_z_loss"] = router_z_loss.detach() if torch.is_tensor(router_z_loss) else router_z_loss
        if moe_metrics:
            loss_dict.update(moe_metrics)
        return total_loss, loss_vla, loss_depth, loss_future_depth, loss_future_video, seq_wise_loss, loss_dict, depth_preds, future_depth_preds, future_video_preds, current_video_preds

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
        tactile_rgb=None,
        tactile_rgb_grid_thw=None,
        marker_displacement_history=None,
        marker_valid_mask=None,
        marker_history_valid_mask=None,
        marker_contact_state=None,
        tactile_sensor_mask=None,
        tactile_rgb_mask=None,
    ) -> Tensor:
        return self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            image_grid_thw=image_grid_thw,
            tactile_rgb=tactile_rgb,
            tactile_rgb_grid_thw=tactile_rgb_grid_thw,
            marker_displacement_history=marker_displacement_history,
            marker_valid_mask=marker_valid_mask,
            marker_history_valid_mask=marker_history_valid_mask,
            marker_contact_state=marker_contact_state,
            tactile_sensor_mask=tactile_sensor_mask,
            tactile_rgb_mask=tactile_rgb_mask,
        )

    def build_slow_cache(self, *args, **kwargs) -> VTLASlowCache:
        return self.model.build_slow_cache(*args, **kwargs)

    def sample_actions_fast(self, *args, **kwargs):
        return self.model.sample_actions_fast(*args, **kwargs)

    def build_cascaded_slow_context(self, *args, **kwargs) -> VTLASlowCache:
        return self.model.build_cascaded_slow_context(*args, **kwargs)

    def build_slow_action_plan(self, *args, **kwargs) -> SlowActionPlan:
        return self.model.build_slow_action_plan(*args, **kwargs)

    def refine_action_with_tactile(self, *args, **kwargs):
        return self.model.refine_action_with_tactile(*args, **kwargs)


ModelClass = LingbotVlaV2Policy

__all__ = [
    "LingbotVlaV2Policy",
    "VTLASlowCache",
    "VTLAFastPrefix",
    "SlowActionPlan",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLTextModel",
    "Qwen3VLPreTrainedModel",
    "Qwen2ForCausalLM",
]
