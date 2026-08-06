#!/usr/bin/env python3
"""Compute robust TacThru no-contact thresholds for marker contact gating."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingbotvla.tactile_contact import build_marker_region_mapping_numpy


@contextmanager
def open_zarr_group(path: Path) -> Iterator[Any]:
    if path.is_dir():
        yield zarr.open_group(str(path), mode="r")
        return
    zip_store_cls = getattr(zarr, "ZipStore", None) or zarr.storage.ZipStore
    store = zip_store_cls(str(path), mode="r")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        store.close()


def parse_episode_range(value: str, total: int) -> tuple[int, int]:
    pieces = value.split(":")
    if len(pieces) != 2 or not all(piece.strip().isdigit() for piece in pieces):
        raise ValueError("--episodes must be START:END with an exclusive END")
    start, end = map(int, pieces)
    if start < 0 or end <= start or end > total:
        raise ValueError(f"Episode range {start}:{end} is outside 0:{total}")
    return start, end


def _robust_threshold(values: np.ndarray, multiplier: float) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite calibration values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    on = median + multiplier * mad
    off = median + 0.5 * multiplier * mad
    eps = max(np.finfo(np.float32).eps, median * 1e-6)
    if on <= off + eps:
        on = max(float(np.quantile(values, 0.999)), median + eps)
        off = min(float(np.quantile(values, 0.99)), on * 0.8)
    if off >= on:
        off = on * 0.8
    return median, mad, on, off


def load_reference_regions(path: Path, *, sensor_name: str, num_markers: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        reference = np.load(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload.get("sensor_names", [sensor_name])
        if names != [sensor_name]:
            raise ValueError(f"Reference sensor order {names} does not match {[sensor_name]}")
        reference = np.asarray(
            payload.get("reference_xy", payload.get("marker_reference_xy")),
            dtype=np.float32,
        )
    if reference.ndim == 3:
        reference = reference[0]
    if reference.shape != (num_markers, 2) or not np.isfinite(reference).all():
        raise ValueError(f"Reference coordinates must be [{num_markers},2]")
    region_ids, _ = build_marker_region_mapping_numpy(
        reference,
        num_regions=4,
        region_layout="2x2",
    )
    return region_ids[0]


def compute_contact_stats(
    displacement: np.ndarray,
    valid: np.ndarray,
    region_ids: np.ndarray,
    *,
    topk_markers: int,
    mad_multiplier: float,
) -> dict[str, Any]:
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError("Marker displacement must be [T,N,2]")
    if valid.shape != displacement.shape[:-1]:
        raise ValueError("Marker valid mask must be [T,N]")
    amplitude = np.linalg.norm(displacement, axis=-1)
    amplitude = np.where(valid, amplitude, np.nan)
    point_median = np.nanmedian(amplitude, axis=0)
    point_mad = np.nanmedian(np.abs(amplitude - point_median[None]), axis=0)
    point_threshold = point_median + mad_multiplier * point_mad
    if not np.isfinite(point_threshold).all():
        raise ValueError("At least one marker has no valid no-contact calibration samples")

    num_regions = int(region_ids.max()) + 1
    regional_scores = np.zeros((len(displacement), num_regions), dtype=np.float64)
    for region_index in range(num_regions):
        region_values = amplitude[:, region_ids == region_index]
        for frame_index, row in enumerate(region_values):
            finite = row[np.isfinite(row)]
            if len(finite):
                k = min(topk_markers, len(finite))
                regional_scores[frame_index, region_index] = np.partition(
                    finite, len(finite) - k
                )[-k:].mean()
    global_score = regional_scores.max(axis=1)
    global_median, global_mad, on, off = _robust_threshold(
        global_score, mad_multiplier
    )
    return {
        "point_noise_median": point_median.tolist(),
        "point_noise_mad": point_mad.tolist(),
        "suggested_point_threshold": point_threshold.tolist(),
        "global_score_median": global_median,
        "global_score_mad": global_mad,
        "suggested_global_on_threshold": on,
        "suggested_global_off_threshold": off,
        "regional_score_quantiles": {
            "q50": np.quantile(regional_scores, 0.50, axis=0).tolist(),
            "q95": np.quantile(regional_scores, 0.95, axis=0).tolist(),
            "q99": np.quantile(regional_scores, 0.99, axis=0).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="No-contact TacThru .zarr or .zarr.zip")
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", required=True, help="START:END, END exclusive")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=0,
        help="Use only the first N frames from each selected episode; 0 uses all frames.",
    )
    parser.add_argument("--marker-key", default="tacthru_l_marker")
    parser.add_argument("--valid-key", default=None)
    parser.add_argument("--sensor-name", default="left")
    parser.add_argument("--reference-xy", type=Path, required=True)
    parser.add_argument("--topk-markers", type=int, default=3)
    parser.add_argument("--mad-multiplier", type=float, default=6.0)
    parser.add_argument("--sample-hz", type=float, default=30.0)
    parser.add_argument(
        "--selection-note",
        default="User-declared no-contact calibration frames",
    )
    args = parser.parse_args()
    if args.frames_per_episode < 0 or args.topk_markers <= 0:
        parser.error("frame count must be non-negative and top-k must be positive")
    if not np.isfinite(args.mad_multiplier) or args.mad_multiplier <= 0:
        parser.error("--mad-multiplier must be positive and finite")

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    with open_zarr_group(source) as root:
        marker = np.asarray(root[f"data/{args.marker_key}"][:], dtype=np.float32)
        valid_array = (
            np.asarray(root[f"data/{args.valid_key}"][:], dtype=np.bool_)
            if args.valid_key
            else None
        )
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        start_episode, end_episode = parse_episode_range(args.episodes, len(episode_ends))
        chunks = []
        valid_chunks = []
        for episode_index in range(start_episode, end_episode):
            start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
            end = int(episode_ends[episode_index])
            if args.frames_per_episode:
                end = min(end, start + args.frames_per_episode)
            current = marker[start:end]
            chunks.append(current)
            if valid_array is not None:
                valid_chunks.append(valid_array[start:end])
            else:
                valid_chunks.append(np.isfinite(current).all(axis=-1))
        displacement = np.concatenate(chunks, axis=0)
        valid = np.concatenate(valid_chunks, axis=0)

    region_ids = load_reference_regions(
        args.reference_xy,
        sensor_name=args.sensor_name,
        num_markers=displacement.shape[1],
    )
    stats = compute_contact_stats(
        displacement,
        valid,
        region_ids,
        topk_markers=args.topk_markers,
        mad_multiplier=args.mad_multiplier,
    )
    payload = {
        "version": 1,
        "units": "normalized_displacement; 2*(current-reference)/[width,height]",
        "source": str(source),
        "marker_key": args.marker_key,
        "num_markers": int(displacement.shape[1]),
        "sensor_names": [args.sensor_name],
        **stats,
        "metadata": {
            "sample_hz": args.sample_hz,
            "num_frames": int(len(displacement)),
            "episode_start": start_episode,
            "episode_end_exclusive": end_episode,
            "frames_per_episode": args.frames_per_episode,
            "selection_note": args.selection_note,
            "topk_markers": args.topk_markers,
            "mad_multiplier": args.mad_multiplier,
            "global_threshold_rule": (
                "median(global_topk_region_score) + {6,3}*MAD for on/off; "
                "quantile fallback when MAD is zero"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
