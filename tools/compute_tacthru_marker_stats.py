#!/usr/bin/env python3
"""Compute VTLA marker statistics from an explicit training-only Zarr split."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr


@contextmanager
def open_zarr_group(path: Path) -> Iterator[Any]:
    """Open directory or zip-backed Zarr without mutating it."""

    if path.is_dir():
        yield zarr.open_group(str(path), mode="r")
        return
    zip_store_cls = getattr(zarr, "ZipStore", None) or zarr.storage.ZipStore
    store = zip_store_cls(str(path), mode="r")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        store.close()


def parse_episode_range(value: str, total_episodes: int) -> tuple[int, int]:
    """Parse mandatory ``START:END`` with an exclusive END."""

    pieces = value.split(":")
    if len(pieces) != 2 or not all(piece.strip().isdigit() for piece in pieces):
        raise ValueError("--train-episodes must be START:END with an exclusive END")
    start, end = (int(piece) for piece in pieces)
    if start < 0 or end <= start or end > total_episodes:
        raise ValueError(
            f"Training episode range {start}:{end} is outside 0:{total_episodes}"
        )
    return start, end


def compute_stats(
    marker_array: Any,
    episode_ends: np.ndarray,
    episode_start: int,
    episode_end: int,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Accumulate float64 statistics over valid [dx,dy,vx,vy] rows."""

    total = np.zeros(4, dtype=np.float64)
    total_sq = np.zeros(4, dtype=np.float64)
    count = 0
    frame_count = 0
    for episode_index in range(episode_start, episode_end):
        start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
        end = int(episode_ends[episode_index])
        previous: np.ndarray | None = None
        for chunk_start in range(start, end, chunk_frames):
            chunk_end = min(end, chunk_start + chunk_frames)
            current = np.asarray(
                marker_array[chunk_start:chunk_end],
                dtype=np.float32,
            )
            if current.ndim != 3 or current.shape[-1] != 2:
                raise ValueError(
                    f"Marker array must be [T,N,2], got {tuple(current.shape)}"
                )
            if previous is None:
                previous_frames = np.concatenate([current[:1], current[:-1]], axis=0)
            else:
                previous_frames = np.concatenate(
                    [previous[None], current[:-1]], axis=0
                )
            velocity = current - previous_frames
            features = np.concatenate([current, velocity], axis=-1).reshape(-1, 4)
            valid = np.isfinite(features).all(axis=-1)
            valid_features = features[valid].astype(np.float64, copy=False)
            total += valid_features.sum(axis=0)
            total_sq += np.square(valid_features).sum(axis=0)
            count += len(valid_features)
            frame_count += len(current)
            previous = current[-1].copy()
    if count == 0:
        raise ValueError("No finite marker samples were found in the training split")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0):
        raise ValueError(f"Training marker statistics contain zero std: {std.tolist()}")
    return mean, std, count, frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="TacThru .zarr or .zarr.zip")
    parser.add_argument("output", type=Path, help="Output marker statistics JSON")
    parser.add_argument(
        "--train-episodes",
        required=True,
        help="Training-only episode range START:END (END exclusive)",
    )
    parser.add_argument("--marker-key", default="tacthru_l_marker")
    parser.add_argument("--chunk-frames", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    with open_zarr_group(source) as root:
        if "data" not in root or args.marker_key not in root["data"]:
            raise KeyError(f"Missing data/{args.marker_key}")
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        start, end = parse_episode_range(args.train_episodes, len(episode_ends))
        mean, std, count, frame_count = compute_stats(
            root["data"][args.marker_key],
            episode_ends,
            start,
            end,
            args.chunk_frames,
        )

    payload = {
        "schema_version": 1,
        "source": str(source),
        "marker_key": args.marker_key,
        "training_split_only": True,
        "train_episode_start": start,
        "train_episode_end_exclusive": end,
        "train_episode_count": end - start,
        "train_frame_count": frame_count,
        "valid_marker_count": count,
        "feature_order": ["dx", "dy", "vx", "vy"],
        "marker_mean": mean.tolist(),
        "marker_std": std.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
