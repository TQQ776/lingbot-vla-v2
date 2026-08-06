#!/usr/bin/env python3
"""Resolve one compact VTLA ablation preset into a complete training YAML."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def deep_merge(target: dict, override: dict) -> dict:
    result = deepcopy(target)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(base: dict, presets: dict, name: str) -> dict:
    if name not in presets:
        raise KeyError(f"Unknown ablation {name!r}; choose from {sorted(presets)}")
    result = deepcopy(base)
    result["train"]["tactile"] = deep_merge(
        result["train"]["tactile"], presets[name]
    )
    result["train"]["output_dir"] = f"output/vtla_ablation_{name}"
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "configs/vla/tacthru_umi"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--base",
        type=Path,
        default=default_dir / "tacthru_umi_vtla_rgb_marker.yaml",
    )
    parser.add_argument(
        "--presets",
        type=Path,
        default=default_dir / "ablations/presets.yaml",
    )
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    presets = yaml.safe_load(args.presets.read_text(encoding="utf-8"))
    resolved = resolve_config(base, presets, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
