#!/usr/bin/env python3
"""Compare global 8-token and regional 32-token TacThru marker encoders."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingbotvla.models.vla.lingbot_vla.tactile_vtla import (
    TactileTokenEncoder,
    TactileVTLAConfig,
)


def benchmark(mapping: dict, *, mode: str, args: argparse.Namespace) -> dict:
    mapping = deepcopy(mapping)
    mapping["use_rgb"] = False
    mapping["marker_contact_gate"] = {
        "enabled": False,
        "mode": "none",
        "target": "marker_only",
    }
    mapping["gate_tactile_rgb"] = False
    if mode == "global":
        mapping["marker_tokenization"] = {
            "mode": "global",
            "num_regions": 1,
            "region_layout": "1x1",
            "aggregation": "mean_max",
            "include_reference_xy": False,
            "point_hidden_dim": 128,
            "region_hidden_dim": 512,
        }
        mapping["marker_position_encoding"] = {
            "temporal_type": "learned",
            "spatial_type": "none",
            "use_real_time": False,
            "combination": "additive",
        }
    settings = TactileVTLAConfig.from_mapping(mapping)
    device = torch.device(args.device)
    encoder = TactileTokenEncoder(settings, context_dim=args.context_dim).to(device)
    history = torch.randn(
        args.batch_size,
        settings.num_sensors,
        settings.marker_history_length,
        settings.num_markers,
        2,
        device=device,
    )
    valid = torch.ones(history.shape[:-1], dtype=torch.bool, device=device)
    history_valid = torch.ones(history.shape[:3], dtype=torch.bool, device=device)
    sensors = torch.ones(history.shape[:2], dtype=torch.bool, device=device)
    for _ in range(args.warmup):
        encoder.encode_markers(history, valid, history_valid, sensors)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    tokens = None
    for _ in range(args.iterations):
        tokens, _ = encoder.encode_markers(history, valid, history_valid, sensors)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "mode": mode,
        "shape": list(tokens.shape),
        "tokens_per_sensor": settings.marker_tokens_per_sensor,
        "mean_forward_ms": elapsed * 1000.0 / args.iterations,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mapping = payload["train"]["tactile"]
    print(
        json.dumps(
            [benchmark(mapping, mode=mode, args=args) for mode in ("global", "regional")],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
