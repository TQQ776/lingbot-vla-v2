#!/usr/bin/env python3
"""Paired offline tactile ablations for a trained LingBot VTLA checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
from collections.abc import Mapping
from dataclasses import replace
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402
from lingbotvla.data.vla_data.base_dataset import VLADataset  # noqa: E402
from lingbotvla.tactile_contact import (  # noqa: E402
    contact_state_is_visible_numpy,
)


CONDITIONS = (
    "full",
    "full_repeat",
    "zero_marker_content",
    "shuffle_marker_history",
    "zero_tactile_rgb",
    "no_tactile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired tactile ablations on fixed LeRobot anchors. This is an "
            "offline model evaluation and never sends robot commands."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--norm-path", type=Path, required=True)
    parser.add_argument("--robot", default="tacthru_umi_v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument(
        "--contact-fraction",
        type=float,
        default=0.75,
        help="Fraction of anchors sampled while the precomputed global gate is visible.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--one-anchor-per-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample at most one anchor from each episode (default: enabled).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of the default BF16 inference path.",
    )
    return parser.parse_args()


def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, np.ndarray):
            result[key] = value.copy()
        else:
            result[key] = copy.deepcopy(value)
    return result


def image_to_hwc_uint8(value: Any, *, key: str) -> np.ndarray:
    image = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"{key} must be a three-dimensional image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.5:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8))


def relative_index(dataset: Any, physical_index: int) -> int:
    index_map = getattr(dataset, "_absolute_to_relative_idx", None)
    if index_map is None:
        return physical_index
    return int(index_map[physical_index])


def load_raw_observation(
    eval_dataset: VLADataset,
    physical_index: int,
    scene_image_keys: list[str],
) -> dict[str, Any]:
    raw = eval_dataset.check_lerobot_item(eval_dataset.dataset[physical_index])
    if eval_dataset.marker_contact_state_cache is not None:
        raw["marker_contact_state"] = eval_dataset.marker_contact_state_cache[
            relative_index(eval_dataset.dataset, physical_index)
        ].clone()
    for key in scene_image_keys:
        raw[key] = image_to_hwc_uint8(raw[key], key=key)
    return raw


def choose_physical_indices(
    eval_dataset: VLADataset,
    *,
    num_samples: int,
    contact_fraction: float,
    seed: int,
    one_anchor_per_episode: bool,
) -> list[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 <= contact_fraction <= 1.0:
        raise ValueError("contact_fraction must be between 0 and 1")
    candidates = list(eval_dataset.sample_indices or range(len(eval_dataset.dataset)))
    if len(candidates) < num_samples:
        raise ValueError(
            f"Requested {num_samples} anchors but only {len(candidates)} are available"
        )
    cache = eval_dataset.marker_contact_state_cache
    if cache is None:
        rng = random.Random(seed)
        return sorted(rng.sample(candidates, num_samples))

    visible: list[int] = []
    hidden: list[int] = []
    for physical_index in candidates:
        state = cache[relative_index(eval_dataset.dataset, physical_index)].cpu().numpy()
        if bool(np.any(contact_state_is_visible_numpy(state))):
            visible.append(physical_index)
        else:
            hidden.append(physical_index)

    desired_visible = round(num_samples * contact_fraction)
    if one_anchor_per_episode:
        selected = choose_episode_stratified_indices(
            eval_dataset,
            candidates=candidates,
            visible=set(visible),
            desired_visible=desired_visible,
            num_samples=num_samples,
            seed=seed,
        )
        selected_visible = sum(index in set(visible) for index in selected)
        print(
            f"Selected {len(selected)} anchors from {len(selected)} unique episodes: "
            f"contact-visible={selected_visible}, "
            f"contact-hidden={len(selected) - selected_visible}"
        )
        return selected

    visible_count = min(desired_visible, len(visible))
    hidden_count = min(num_samples - visible_count, len(hidden))
    remaining = num_samples - visible_count - hidden_count
    if remaining:
        extra_visible = min(remaining, len(visible) - visible_count)
        visible_count += extra_visible
        remaining -= extra_visible
    if remaining:
        hidden_count += min(remaining, len(hidden) - hidden_count)
        remaining = num_samples - visible_count - hidden_count
    if remaining:
        raise ValueError("Not enough gate-stratified anchors to satisfy num_samples")

    rng = random.Random(seed)
    selected = rng.sample(visible, visible_count) + rng.sample(hidden, hidden_count)
    rng.shuffle(selected)
    print(
        f"Selected {len(selected)} anchors: contact-visible={visible_count}, "
        f"contact-hidden={hidden_count}"
    )
    return selected


def episode_bounds(episodes: Any) -> list[tuple[int, int, int]]:
    records: list[tuple[int, Any]] = []
    if hasattr(episodes, "iterrows"):
        records = [(int(index), record) for index, record in episodes.iterrows()]
    elif isinstance(episodes, Mapping):
        if "dataset_from_index" in episodes:
            starts = episodes["dataset_from_index"]
            ends = episodes["dataset_to_index"]
            records = [
                (index, {"dataset_from_index": start, "dataset_to_index": end})
                for index, (start, end) in enumerate(zip(starts, ends))
            ]
        else:
            records = [(int(index), record) for index, record in episodes.items()]
    else:
        records = list(enumerate(episodes))

    result: list[tuple[int, int, int]] = []
    for fallback_index, record in records:
        episode_index = int(record.get("episode_index", fallback_index))
        result.append(
            (
                episode_index,
                int(record["dataset_from_index"]),
                int(record["dataset_to_index"]),
            )
        )
    return result


def choose_episode_stratified_indices(
    eval_dataset: VLADataset,
    *,
    candidates: list[int],
    visible: set[int],
    desired_visible: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    candidate_set = set(candidates)
    options: dict[int, dict[str, list[int]]] = {}
    for episode_index, start, end in episode_bounds(eval_dataset.dataset_meta.episodes):
        episode_candidates = [
            index for index in range(start, end) if index in candidate_set
        ]
        if not episode_candidates:
            continue
        options[episode_index] = {
            "visible": [index for index in episode_candidates if index in visible],
            "hidden": [index for index in episode_candidates if index not in visible],
        }
    if len(options) < num_samples:
        raise ValueError(
            f"Requested {num_samples} independent episodes but only {len(options)} "
            "contain complete action anchors; use --no-one-anchor-per-episode to override"
        )

    rng = random.Random(seed)
    selected: list[int] = []
    used_episodes: set[int] = set()

    def select_category(category: str, count: int) -> int:
        episode_ids = [
            episode_index
            for episode_index, category_options in options.items()
            if episode_index not in used_episodes and category_options[category]
        ]
        rng.shuffle(episode_ids)
        for episode_index in episode_ids[:count]:
            selected.append(rng.choice(options[episode_index][category]))
            used_episodes.add(episode_index)
        return min(count, len(episode_ids))

    desired_hidden = num_samples - desired_visible
    selected_hidden = select_category("hidden", desired_hidden)
    selected_visible = select_category("visible", desired_visible)
    remaining = num_samples - selected_hidden - selected_visible
    if remaining:
        episode_ids = [
            episode_index
            for episode_index in options
            if episode_index not in used_episodes
        ]
        rng.shuffle(episode_ids)
        for episode_index in episode_ids[:remaining]:
            category = "visible" if options[episode_index]["visible"] else "hidden"
            selected.append(rng.choice(options[episode_index][category]))
            used_episodes.add(episode_index)
    if len(selected) != num_samples:
        raise ValueError("Could not construct the requested episode-stratified sample")
    rng.shuffle(selected)
    return selected


def marker_history_keys(policy: LingbotVLAv2Server) -> tuple[list[str], list[str]]:
    settings = policy.vla.feature_transform.tactile_settings
    return list(settings.marker_displacement_keys), list(settings.marker_valid_mask_keys)


def make_condition_observation(
    observation: dict[str, Any],
    *,
    condition: str,
    policy: LingbotVLAv2Server,
    physical_index: int,
    seed: int,
) -> dict[str, Any]:
    result = clone_observation(observation)
    displacement_keys, valid_keys = marker_history_keys(policy)
    settings = policy.vla.feature_transform.tactile_settings

    if condition == "shuffle_marker_history":
        for sensor_index, (displacement_key, valid_key) in enumerate(
            zip(displacement_keys, valid_keys)
        ):
            history = result[displacement_key]
            history_length = int(history.shape[0])
            if history_length != int(settings.marker_history_length):
                raise ValueError(
                    f"{displacement_key} must contain the real "
                    f"{settings.marker_history_length}-frame history, got {tuple(history.shape)}"
                )
            generator = torch.Generator().manual_seed(
                seed + 1009 * physical_index + 17 * sensor_index
            )
            order = torch.randperm(history_length, generator=generator)
            if torch.equal(order, torch.arange(history_length)):
                order = torch.roll(order, shifts=1)
            result[displacement_key] = history[order]
            if valid_key in result:
                result[valid_key] = result[valid_key][order]

    if condition == "zero_tactile_rgb":
        for key in settings.rgb_keys:
            result[key] = torch.zeros_like(result[key])

    if condition == "no_tactile":
        for displacement_key, valid_key in zip(displacement_keys, valid_keys):
            result[displacement_key] = torch.zeros_like(result[displacement_key])
            if valid_key in result:
                result[valid_key] = torch.zeros_like(result[valid_key], dtype=torch.bool)
        for key in settings.rgb_keys:
            result[key] = torch.zeros_like(result[key])
        result["marker_contact_state"] = torch.zeros(
            int(settings.num_sensors), dtype=torch.int8
        )
        result["tactile_sensor_mask"] = torch.zeros(
            int(settings.num_sensors), dtype=torch.bool
        )

    return result


def set_zero_marker_content(policy: LingbotVLAv2Server, enabled: bool) -> Any:
    encoder = policy.vla.model.tactile_encoder
    if encoder is None:
        raise RuntimeError("Loaded checkpoint does not contain a tactile encoder")
    original = encoder.settings
    encoder.settings = replace(
        original,
        marker_ablation=replace(
            original.marker_ablation,
            zero_marker_content=enabled,
        ),
    )
    return original


def concatenate_action(result: dict[str, Any], action_keys: list[str]) -> np.ndarray:
    values = []
    for key in action_keys:
        value = result[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(values, axis=-1)


def run_condition(
    policy: LingbotVLAv2Server,
    observations: list[dict[str, Any]],
    physical_indices: list[int],
    *,
    condition: str,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    normalized_predictions: list[np.ndarray] = []
    action_keys = list(policy.vla.feature_transform.org_features["actions"])
    original_settings = set_zero_marker_content(
        policy, condition == "zero_marker_content"
    )
    try:
        for batch_start in range(0, len(observations), batch_size):
            batch_indices = physical_indices[batch_start : batch_start + batch_size]
            batch = [
                make_condition_observation(
                    observation,
                    condition=condition,
                    policy=policy,
                    physical_index=physical_index,
                    seed=seed,
                )
                for observation, physical_index in zip(
                    observations[batch_start : batch_start + batch_size],
                    batch_indices,
                )
            ]
            # Reusing the seed for every condition gives the same Flow Matching
            # initial noise for the same anchor and batch position.
            set_inference_seed(seed + batch_start)
            output, normalized = policy._infer_batch(batch, return_normalized=True)
            predictions.append(concatenate_action(output, action_keys))
            normalized_predictions.append(normalized.detach().cpu().float().numpy())
            print(
                f"condition={condition} batch={batch_start // batch_size + 1}/"
                f"{(len(observations) + batch_size - 1) // batch_size} complete"
            )
    finally:
        policy.vla.model.tactile_encoder.settings = original_settings
    return np.concatenate(predictions, axis=0), np.concatenate(normalized_predictions, axis=0)


def metric_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    per_sample_mse = np.mean(np.square(error), axis=(1, 2))
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "per_dimension_mae": np.mean(np.abs(error), axis=(0, 1)).tolist(),
        "per_sample_mse": per_sample_mse.tolist(),
    }


def paired_delta_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    delta = prediction - reference
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
        "per_dimension_mae": np.mean(np.abs(delta), axis=(0, 1)).tolist(),
    }


def write_anchor_csv(
    path: Path,
    metadata: list[dict[str, int]],
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "sample_index",
        "physical_index",
        "episode_index",
        "frame_index",
        "contact_state",
    ]
    for condition in predictions:
        fieldnames.extend([f"{condition}_mse", f"{condition}_mae"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, row_metadata in enumerate(metadata):
            row: dict[str, Any] = {"sample_index": sample_index, **row_metadata}
            for condition, prediction in predictions.items():
                error = prediction[sample_index] - target[sample_index]
                row[f"{condition}_mse"] = float(np.mean(np.square(error)))
                row[f"{condition}_mae"] = float(np.mean(np.abs(error)))
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    args.model_path = args.model_path.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.norm_path = args.norm_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.model_path, args.data_path, args.norm_path):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Offline evaluation only: no robot or gripper command can be sent.")
    print(f"checkpoint={args.model_path}")
    print(f"dataset={args.data_path}")
    set_inference_seed(args.seed)
    policy = LingbotVLAv2Server(
        path_to_pi_model=str(args.model_path),
        robot_norm_path=str(args.norm_path),
        use_length=int(50),
        chunk_ret=True,
        use_bf16=not args.fp32,
        use_fp32=args.fp32,
        use_compile=False,
    )
    policy.reset(args.robot)
    settings = policy.vla.feature_transform.tactile_settings
    if not settings.enabled or not settings.use_markers:
        raise RuntimeError("The loaded checkpoint is not marker-enabled VTLA")

    image_size = int(getattr(policy.data_config, "img_size", 256))
    robot_config_root = getattr(
        policy.data_config,
        "robot_config_root",
        str(PROJECT_ROOT / "configs" / "robot_configs"),
    )
    eval_dataset = VLADataset(
        repo_id=str(args.data_path),
        data_name=args.robot,
        dataset_config=policy.data_config,
        robot_config_root=robot_config_root,
        config=policy.config,
        processor=policy.processor,
        chunk_size=int(policy.config.chunk_size),
        image_size=(image_size, image_size),
        do_nomalize=True,
        feature_transform=policy.vla.feature_transform,
        image_augment=False,
        use_future_image=False,
    )
    physical_indices = choose_physical_indices(
        eval_dataset,
        num_samples=args.num_samples,
        contact_fraction=args.contact_fraction,
        seed=args.seed,
        one_anchor_per_episode=args.one_anchor_per_episode,
    )

    scene_image_keys = list(policy.vla.feature_transform.org_features["images"])
    observations = [
        load_raw_observation(eval_dataset, index, scene_image_keys)
        for index in physical_indices
    ]
    for observation in observations:
        for key in settings.marker_displacement_keys:
            expected = (
                int(settings.marker_history_length),
                int(settings.num_markers),
                2,
            )
            if tuple(observation[key].shape) != expected:
                raise ValueError(
                    f"Evaluation did not load the real marker history for {key}: "
                    f"expected {expected}, got {tuple(observation[key].shape)}"
                )

    action_keys = list(policy.vla.feature_transform.org_features["actions"])
    targets = np.stack(
        [concatenate_action(observation, action_keys) for observation in observations],
        axis=0,
    )
    metadata: list[dict[str, int]] = []
    for physical_index, observation in zip(physical_indices, observations):
        contact = np.asarray(observation["marker_contact_state"]).reshape(-1)
        metadata.append(
            {
                "physical_index": int(physical_index),
                "episode_index": int(np.asarray(observation["episode_index"])),
                "frame_index": int(np.asarray(observation["frame_index"])),
                "contact_state": int(contact[0]),
            }
        )

    predictions: dict[str, np.ndarray] = {}
    normalized_predictions: dict[str, np.ndarray] = {}
    for condition in args.conditions:
        predictions[condition], normalized_predictions[condition] = run_condition(
            policy,
            observations,
            physical_indices,
            condition=condition,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    metrics = {
        condition: metric_summary(prediction, targets)
        for condition, prediction in predictions.items()
    }
    paired_deltas: dict[str, Any] = {}
    if "full" in predictions:
        paired_deltas = {
            condition: paired_delta_summary(prediction, predictions["full"])
            for condition, prediction in predictions.items()
            if condition != "full"
        }
    report = {
        "checkpoint": str(args.model_path),
        "dataset": str(args.data_path),
        "norm_path": str(args.norm_path),
        "seed": args.seed,
        "dtype": "fp32" if args.fp32 else "bf16",
        "num_samples": len(observations),
        "action_shape": list(targets.shape),
        "conditions": list(predictions),
        "anchors": metadata,
        "metrics": metrics,
        "paired_output_deltas_vs_full": paired_deltas,
        "limitation": (
            "All evaluated episodes were included in training; these are in-sample "
            "paired ablations, not a held-out generalization result."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_anchor_csv(args.output_dir / "per_anchor.csv", metadata, targets, predictions)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        target=targets,
        physical_indices=np.asarray(physical_indices, dtype=np.int64),
        **{f"prediction_{key}": value for key, value in predictions.items()},
        **{
            f"normalized_prediction_{key}": value
            for key, value in normalized_predictions.items()
        },
    )

    print(json.dumps({"metrics": metrics, "paired_output_deltas_vs_full": paired_deltas}, indent=2))
    print(f"Wrote evaluation artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
