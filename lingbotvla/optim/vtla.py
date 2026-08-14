"""Optimizer grouping helpers for the minimal VTLA extension."""

from __future__ import annotations

from typing import Any

from torch import nn


ACTION_PARAMETER_FRAGMENTS = (
    "qwen_expert.",
    "state_proj.",
    "action_in_proj.",
    "action_out_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
)

TACTILE_PARAMETER_FRAGMENTS = (
    "tactile_encoder.",
    "tactile_expert.",
    "marker_to_tactile_proj.",
    "tactile_time_embedder.",
    "tactile_action_in_proj.",
    "tactile_action_out_proj.",
)


def build_vtla_param_groups(
    model: nn.Module,
    *,
    new_modules_lr: float,
    action_expert_lr: float,
    base_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Return disjoint tactile/action/base groups, excluding frozen params."""

    if min(new_modules_lr, action_expert_lr, base_lr) <= 0:
        raise ValueError("All VTLA optimizer learning rates must be positive")
    groups: dict[str, dict[str, Any]] = {
        "tactile": {
            "params": [],
            "lr": float(new_modules_lr),
            "weight_decay": float(weight_decay),
        },
        "action_expert": {
            "params": [],
            "lr": float(action_expert_lr),
            "weight_decay": float(weight_decay),
        },
        "base_trainable": {
            "params": [],
            "lr": float(base_lr),
            "weight_decay": float(weight_decay),
        },
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            raise ValueError(f"Trainable parameter {name} appears more than once")
        seen.add(identifier)
        if any(fragment in name for fragment in TACTILE_PARAMETER_FRAGMENTS):
            target = "tactile"
        elif any(fragment in name for fragment in ACTION_PARAMETER_FRAGMENTS):
            target = "action_expert"
        else:
            target = "base_trainable"
        groups[target]["params"].append(parameter)

    result = []
    for name, group in groups.items():
        if group["params"]:
            group["name"] = name
            result.append(group)
    if not any(group["name"] == "tactile" for group in result):
        raise RuntimeError("VTLA is enabled but no trainable tactile parameters were found")
    return result


def summarize_vtla_param_groups(
    model: nn.Module,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return counts used by the one-shot training startup log."""

    return {
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "groups": {
            group["name"]: {
                "lr": group["lr"],
                "parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in groups
        },
    }


__all__ = [
    "ACTION_PARAMETER_FRAGMENTS",
    "TACTILE_PARAMETER_FRAGMENTS",
    "build_vtla_param_groups",
    "summarize_vtla_param_groups",
]
