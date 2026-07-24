"""Temporal tactile encoders and gated Qwen-prefix fusion for LingBot V2.

The primary RGB path uses the repository's self-contained DINOv2 ViT-S/14 and
requires a local checkpoint.  A compact native patch transformer remains an
explicit fallback for architecture/debug ablations.  No weight is downloaded
at construction or inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _reset_module(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
        return
    reset = getattr(module, "_reset_parameters", None)
    if callable(reset):
        reset()


def _as_bool_mask(value: Optional[Tensor], shape: Tuple[int, ...], device) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if tuple(value.shape) != shape:
        raise ValueError(f"Expected mask shape {shape}, got {tuple(value.shape)}.")
    return value.to(device=device, dtype=torch.bool)


def _safe_transformer_mask(valid: Tensor) -> Tuple[Tensor, Tensor]:
    """Make a key-padding mask safe for samples whose every key is invalid."""
    present = valid.any(dim=-1)
    first = torch.zeros_like(valid)
    first[:, 0] = True
    safe_valid = valid | ((~present).unsqueeze(-1) & first)
    return safe_valid, present


class FixedTokenResampler(nn.Module):
    """Resample a variable set of valid features to a fixed token count."""

    def __init__(self, dim: int, num_tokens: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_tokens = int(num_tokens)
        self.queries = nn.Parameter(torch.empty(1, self.num_tokens, dim))
        self.key_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, std=0.02)
        for module in (self.key_norm, self.query_norm, self.attention, self.output_norm):
            _reset_module(module)

    def forward(self, features: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B,N,D], got {tuple(features.shape)}.")
        if valid.shape != features.shape[:2]:
            raise ValueError(
                f"Feature/mask shape mismatch: {tuple(features.shape)} vs {tuple(valid.shape)}."
            )
        safe_valid, present = _safe_transformer_mask(valid.to(dtype=torch.bool))
        features = features * valid.unsqueeze(-1).to(features.dtype)
        queries = self.queries.expand(features.shape[0], -1, -1).to(features.dtype)
        output, _ = self.attention(
            self.query_norm(queries),
            self.key_norm(features),
            self.key_norm(features),
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        output = self.output_norm(output + queries)
        output = output * present[:, None, None].to(output.dtype)
        token_mask = present[:, None].expand(-1, self.num_tokens)
        return output, token_mask


class TactileRGBEncoder(nn.Module):
    """Encode K synchronized tactile RGB frames into fixed prefix tokens."""

    def __init__(self, params: Dict[str, Any], prefix_hidden_dim: int):
        super().__init__()
        self.backbone_name = str(params["rgb_backbone"])
        self.history_steps = int(params["history_steps"])
        self.history_stride = int(params.get("history_stride", 1))
        self.history_frequency_hz = float(params.get("history_frequency_hz", 30.0))
        self.input_size = int(params["rgb_input_size"])
        patch_size = int(params["rgb_patch_size"])
        heads = int(params["rgb_encoder_heads"])
        if self.backbone_name == "dinov2_vits14":
            if patch_size != 14:
                raise ValueError("dinov2_vits14 requires tactile_params.rgb_patch_size=14.")
            if not params.get("rgb_backbone_path"):
                raise ValueError(
                    "dinov2_vits14 requires tactile_params.rgb_backbone_path pointing "
                    "to local pretrained weights; online download is intentionally disabled."
                )
            self.backbone_pretrain_size = int(
                params.get("rgb_backbone_pretrain_size", 518)
            )
            if self.backbone_pretrain_size % patch_size != 0:
                raise ValueError(
                    "dinov2_vits14 requires rgb_backbone_pretrain_size to be "
                    "divisible by rgb_patch_size."
                )
            from lingbotvla.models.vla.vision_models.MoGe.moge.model.dinov2.hub.backbones import (
                dinov2_vits14,
            )

            self.backbone = dinov2_vits14(
                pretrained=False,
                img_size=self.backbone_pretrain_size,
                block_chunks=0,
            )
            dim = int(self.backbone.embed_dim)
            self.native_patch_embed = None
            self.native_spatial_embedding = None
            self.native_frame_encoder = None
        elif self.backbone_name == "native_patch_transformer":
            self.backbone_pretrain_size = self.input_size
            dim = int(params["rgb_encoder_dim"])
            layers = int(params["rgb_encoder_layers"])
            self.backbone = None
            self.native_patch_embed = nn.Conv2d(
                3, dim, kernel_size=patch_size, stride=patch_size
            )
            patch_count = (self.input_size // patch_size) ** 2
            self.native_spatial_embedding = nn.Parameter(
                torch.empty(1, 1, patch_count, dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.native_frame_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=layers, enable_nested_tensor=False
            )
        else:
            raise ValueError(
                "Unsupported tactile RGB backbone "
                f"{self.backbone_name!r}; expected 'dinov2_vits14' or "
                "'native_patch_transformer'."
            )
        self.temporal_embedding = nn.Parameter(torch.empty(1, self.history_steps, 1, dim))
        self.resampler = FixedTokenResampler(dim, int(params["rgb_tokens"]), heads)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, prefix_hidden_dim),
        )
        self.register_buffer(
            "rgb_mean",
            torch.tensor(params["rgb_mean"], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(params["rgb_std"], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.freeze_backbone = bool(params["freeze_rgb_backbone"])
        self.backbone_path = params.get("rgb_backbone_path")
        self.reset_parameters()
        if self.freeze_backbone:
            backbone_module = self.backbone
            if backbone_module is None:
                backbone_module = nn.ModuleList(
                    [self.native_patch_embed, self.native_frame_encoder]
                )
            backbone_module.requires_grad_(False)

    @property
    def num_tokens(self) -> int:
        return self.resampler.num_tokens

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            if self.backbone is not None:
                self.backbone.eval()
            else:
                self.native_patch_embed.eval()
                self.native_frame_encoder.eval()
        return self

    def reset_parameters(self) -> None:
        if self.backbone is not None:
            for module in self.backbone.modules():
                if module is self.backbone:
                    continue
                _reset_module(module)
            # Parameters without a reset_parameters hook in the vendored DINO.
            if hasattr(self.backbone, "init_weights"):
                self.backbone.init_weights()
            if hasattr(self.backbone, "mask_token"):
                nn.init.zeros_(self.backbone.mask_token)
            for module in self.backbone.modules():
                if module.__class__.__name__ == "LayerScale" and hasattr(module, "gamma"):
                    nn.init.ones_(module.gamma)
        else:
            self.native_patch_embed.reset_parameters()
            nn.init.normal_(self.native_spatial_embedding, std=0.02)
            for module in self.native_frame_encoder.modules():
                if module is self.native_frame_encoder:
                    continue
                _reset_module(module)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        self.resampler.reset_parameters()
        for module in self.output_projection:
            _reset_module(module)

    @torch.no_grad()
    def load_local_backbone(self) -> None:
        """Load RGB backbone weights from disk without any network access."""
        if not self.backbone_path:
            return
        path = Path(self.backbone_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tactile RGB backbone checkpoint not found: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            for key in ("state_dict", "model", "encoder"):
                if key in raw and isinstance(raw[key], dict):
                    raw = raw[key]
                    break
        if not isinstance(raw, dict):
            raise TypeError(f"Unsupported tactile RGB checkpoint payload in {path}.")
        prefixes = ("model.tactile_rgb_encoder.", "tactile_rgb_encoder.", "module.")
        state = {}
        for key, value in raw.items():
            mapped = key
            for prefix in prefixes:
                if mapped.startswith(prefix):
                    mapped = mapped[len(prefix) :]
                    break
            state[mapped] = value
        if self.backbone is not None:
            backbone_state = {
                (key[len("backbone.") :] if key.startswith("backbone.") else key): value
                for key, value in state.items()
                if not key.startswith(("resampler.", "output_projection.", "temporal_embedding"))
            }
            incompatible = self.backbone.load_state_dict(backbone_state, strict=True)
        else:
            incompatible = self.load_state_dict(state, strict=False)
            allowed_missing = {
                key
                for key in self.state_dict()
                if key.startswith("resampler.")
                or key.startswith("output_projection.")
                or key == "temporal_embedding"
            }
            disallowed_missing = set(incompatible.missing_keys) - allowed_missing
            if disallowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Incompatible native tactile RGB backbone checkpoint: "
                    f"missing={sorted(disallowed_missing)}, "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )

    def forward(self, history: Tensor, history_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if history.ndim != 5:
            raise ValueError(f"Expected tactile RGB [B,K,3,H,W], got {tuple(history.shape)}.")
        batch, steps, channels, height, width = history.shape
        if steps != self.history_steps or channels != 3:
            raise ValueError(
                f"Expected K={self.history_steps}, C=3, got K={steps}, C={channels}."
            )
        if height != self.input_size or width != self.input_size:
            raise ValueError(
                f"Expected tactile RGB {self.input_size}x{self.input_size}, got {height}x{width}."
            )
        mask = _as_bool_mask(history_mask, (batch, steps), history.device)
        pixels = history.float()
        if history.dtype == torch.uint8:
            pixels = pixels / 255.0
        pixels = (pixels - self.rgb_mean) / self.rgb_std
        if self.backbone is not None:
            backbone_dtype = next(self.backbone.parameters()).dtype
            features = self.backbone.forward_features(
                pixels.flatten(0, 1).to(dtype=backbone_dtype)
            )
            patches = features["x_norm_patchtokens"]
        else:
            pixels = pixels.to(dtype=self.native_patch_embed.weight.dtype)
            patches = self.native_patch_embed(pixels.flatten(0, 1))
            patches = patches.flatten(2).transpose(1, 2)
        patch_count = patches.shape[1]
        patches = patches.view(batch, steps, patch_count, -1)
        if self.backbone is None:
            patches = patches + self.native_spatial_embedding
            patches = self.native_frame_encoder(patches.flatten(0, 1)).view(
                batch, steps, patch_count, -1
            )
        patches = patches + self.temporal_embedding.to(patches.dtype)
        valid = mask[:, :, None].expand(-1, -1, patch_count).reshape(batch, -1)
        tokens, token_mask = self.resampler(patches.reshape(batch, -1, patches.shape[-1]), valid)
        tokens = self.output_projection(tokens)
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, token_mask


class TactileMarkerEncoder(nn.Module):
    """Encode temporal normalized marker flow into fixed prefix tokens."""

    def __init__(self, params: Dict[str, Any], prefix_hidden_dim: int):
        super().__init__()
        self.history_steps = int(params["history_steps"])
        self.history_stride = int(params.get("history_stride", 1))
        self.history_frequency_hz = float(params.get("history_frequency_hz", 30.0))
        self.marker_count = int(params["marker_count"])
        self.marker_dim = int(params["marker_dim"])
        self.include_velocity = bool(params["marker_include_velocity"])
        dim = int(params["marker_encoder_dim"])
        heads = int(params["marker_encoder_heads"])
        layers = int(params["marker_encoder_layers"])
        feature_dim = self.marker_dim * (2 if self.include_velocity else 1)
        self.point_mlp = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.marker_embedding = nn.Parameter(torch.empty(1, 1, self.marker_count, dim))
        self.temporal_embedding = nn.Parameter(torch.empty(1, self.history_steps, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.resampler = FixedTokenResampler(dim, int(params["marker_tokens"]), heads)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, prefix_hidden_dim),
        )
        self.reset_parameters()

    @property
    def num_tokens(self) -> int:
        return self.resampler.num_tokens

    def reset_parameters(self) -> None:
        nn.init.normal_(self.marker_embedding, std=0.02)
        nn.init.normal_(self.temporal_embedding, std=0.02)
        for tree in (self.point_mlp, self.temporal_encoder, self.output_projection):
            for module in tree.modules():
                if module is tree:
                    continue
                _reset_module(module)
        self.resampler.reset_parameters()

    def _build_flow_features(
        self,
        flow: Tensor,
        valid: Tensor,
        timestamps: Optional[Tensor],
    ) -> Tensor:
        if not self.include_velocity:
            return flow
        batch, steps = flow.shape[:2]
        velocity = torch.zeros_like(flow)
        velocity_valid = torch.zeros_like(valid)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if timestamps is not None:
            if tuple(timestamps.shape) != (batch, steps):
                raise ValueError(
                    f"Expected tactile timestamps [B,K]={batch, steps}, "
                    f"got {tuple(timestamps.shape)}."
                )
            delta_t = timestamps[:, 1:] - timestamps[:, :-1]
            time_valid = torch.isfinite(delta_t) & (delta_t > 1e-6)
            safe_delta_t = torch.where(time_valid, delta_t, torch.ones_like(delta_t))
            velocity[:, 1:] = (flow[:, 1:] - flow[:, :-1]) / safe_delta_t[
                :, :, None, None
            ].to(flow.dtype)
            pair_valid = pair_valid & time_valid[:, :, None]
        else:
            fallback_dt = self.history_stride / self.history_frequency_hz
            velocity[:, 1:] = (flow[:, 1:] - flow[:, :-1]) / fallback_dt
        velocity_valid[:, 1:] = pair_valid
        velocity = velocity * velocity_valid.unsqueeze(-1).to(velocity.dtype)
        return torch.cat([flow, velocity], dim=-1)

    def forward(
        self,
        marker_flow: Tensor,
        marker_valid_mask: Optional[Tensor] = None,
        history_mask: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if marker_flow.ndim != 4:
            raise ValueError(f"Expected marker flow [B,K,M,D], got {tuple(marker_flow.shape)}.")
        batch, steps, count, marker_dim = marker_flow.shape
        expected = (self.history_steps, self.marker_count, self.marker_dim)
        if (steps, count, marker_dim) != expected:
            raise ValueError(
                f"Expected marker shape [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(marker_flow.shape)}."
            )
        history_valid = _as_bool_mask(history_mask, (batch, steps), marker_flow.device)
        marker_valid = _as_bool_mask(
            marker_valid_mask, (batch, steps, count), marker_flow.device
        )
        valid = marker_valid & history_valid[:, :, None]
        flow = marker_flow.to(dtype=self.point_mlp[0].weight.dtype)
        features = self._build_flow_features(flow, valid, timestamps)
        encoded = self.point_mlp(features)
        encoded = encoded + self.marker_embedding + self.temporal_embedding
        encoded = encoded.reshape(batch, steps * count, -1)
        flat_valid = valid.reshape(batch, steps * count)
        safe_valid, _ = _safe_transformer_mask(flat_valid)
        encoded = self.temporal_encoder(encoded, src_key_padding_mask=~safe_valid)
        encoded = encoded * flat_valid.unsqueeze(-1).to(encoded.dtype)
        tokens, token_mask = self.resampler(encoded, flat_valid)
        tokens = self.output_projection(tokens)
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, token_mask


class GatedTactileFusion(nn.Module):
    """Apply sample-level ablation/dropout and concatenate fixed tactile tokens."""

    def __init__(
        self,
        prefix_hidden_dim: int,
        rgb_enabled: bool,
        marker_enabled: bool,
        params: Dict[str, Any],
    ):
        super().__init__()
        self.rgb_enabled = bool(rgb_enabled)
        self.marker_enabled = bool(marker_enabled)
        self.rgb_dropout_prob = float(params["rgb_dropout_prob"])
        self.marker_dropout_prob = float(params["marker_dropout_prob"])
        self.all_dropout_prob = float(params["all_tactile_dropout_prob"])
        self.gate_init = float(params["gate_init"])
        if self.rgb_enabled:
            self.rgb_gate = nn.Parameter(torch.tensor(self.gate_init))
            self.rgb_gate_proj = nn.Linear(prefix_hidden_dim, 1)
            self.rgb_modality_embedding = nn.Parameter(torch.empty(1, 1, prefix_hidden_dim))
        if self.marker_enabled:
            self.marker_gate = nn.Parameter(torch.tensor(self.gate_init))
            self.marker_gate_proj = nn.Linear(prefix_hidden_dim, 1)
            self.marker_modality_embedding = nn.Parameter(torch.empty(1, 1, prefix_hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.rgb_enabled:
            nn.init.constant_(self.rgb_gate, self.gate_init)
            nn.init.zeros_(self.rgb_gate_proj.weight)
            nn.init.zeros_(self.rgb_gate_proj.bias)
            nn.init.normal_(self.rgb_modality_embedding, std=0.02)
        if self.marker_enabled:
            nn.init.constant_(self.marker_gate, self.gate_init)
            nn.init.zeros_(self.marker_gate_proj.weight)
            nn.init.zeros_(self.marker_gate_proj.bias)
            nn.init.normal_(self.marker_modality_embedding, std=0.02)

    @staticmethod
    def _force_mask(force_mask, batch: int, device) -> Tensor:
        if force_mask is None:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        if isinstance(force_mask, bool):
            return torch.full((batch,), force_mask, dtype=torch.bool, device=device)
        if force_mask.numel() == 1:
            return force_mask.to(device=device, dtype=torch.bool).expand(batch)
        if tuple(force_mask.shape) != (batch,):
            raise ValueError(f"Force mask must be scalar or [B], got {tuple(force_mask.shape)}.")
        return force_mask.to(device=device, dtype=torch.bool)

    def _sample_keep(
        self,
        base_present: Tensor,
        branch_dropout_prob: float,
        force_mask,
        all_drop: Tensor,
    ) -> Tensor:
        batch = base_present.shape[0]
        branch_drop = torch.zeros(batch, dtype=torch.bool, device=base_present.device)
        if self.training and branch_dropout_prob > 0:
            branch_drop = torch.rand(batch, device=base_present.device) < branch_dropout_prob
        forced = self._force_mask(force_mask, batch, base_present.device)
        return base_present.to(dtype=torch.bool) & ~branch_drop & ~all_drop & ~forced

    def forward(
        self,
        rgb_tokens: Optional[Tensor] = None,
        rgb_token_mask: Optional[Tensor] = None,
        rgb_present: Optional[Tensor] = None,
        marker_tokens: Optional[Tensor] = None,
        marker_token_mask: Optional[Tensor] = None,
        marker_present: Optional[Tensor] = None,
        force_mask_rgb=None,
        force_mask_marker=None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        reference = rgb_tokens if rgb_tokens is not None else marker_tokens
        if reference is None:
            raise ValueError("At least one enabled tactile branch must provide fixed tokens.")
        batch = reference.shape[0]
        all_drop = torch.zeros(batch, dtype=torch.bool, device=reference.device)
        if self.training and self.all_dropout_prob > 0:
            all_drop = torch.rand(batch, device=reference.device) < self.all_dropout_prob

        outputs = []
        masks = []
        metrics: Dict[str, Tensor] = {}
        if self.rgb_enabled:
            if rgb_tokens is None or rgb_token_mask is None or rgb_present is None:
                raise ValueError("RGB tactile branch is enabled but fixed RGB tokens are missing.")
            keep = self._sample_keep(
                rgb_present, self.rgb_dropout_prob, force_mask_rgb, all_drop
            )
            mask = rgb_token_mask.to(torch.bool) & keep[:, None]
            gate = torch.sigmoid(
                self.rgb_gate + self.rgb_gate_proj(rgb_tokens.mean(dim=1))
            ).unsqueeze(1)
            tokens = gate * (rgb_tokens + self.rgb_modality_embedding.to(rgb_tokens.dtype))
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
            outputs.append(tokens)
            masks.append(mask)
            metrics["tactile/rgb_gate"] = gate.mean().detach()
            metrics["tactile/rgb_presence"] = keep.float().mean().detach()
        if self.marker_enabled:
            if marker_tokens is None or marker_token_mask is None or marker_present is None:
                raise ValueError("Marker tactile branch is enabled but fixed marker tokens are missing.")
            keep = self._sample_keep(
                marker_present, self.marker_dropout_prob, force_mask_marker, all_drop
            )
            mask = marker_token_mask.to(torch.bool) & keep[:, None]
            gate = torch.sigmoid(
                self.marker_gate + self.marker_gate_proj(marker_tokens.mean(dim=1))
            ).unsqueeze(1)
            tokens = gate * (
                marker_tokens + self.marker_modality_embedding.to(marker_tokens.dtype)
            )
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
            outputs.append(tokens)
            masks.append(mask)
            metrics["tactile/marker_gate"] = gate.mean().detach()
            metrics["tactile/marker_presence"] = keep.float().mean().detach()
        metrics["tactile/all_dropout_fraction"] = all_drop.float().mean().detach()
        return torch.cat(outputs, dim=1), torch.cat(masks, dim=1), metrics


__all__ = [
    "FixedTokenResampler",
    "GatedTactileFusion",
    "TactileMarkerEncoder",
    "TactileRGBEncoder",
]
