#!/usr/bin/env bash
# Launch reproducible LingBot V2 tactile ablations on one RGB+marker superset.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_LAUNCHER="$ROOT_DIR/scripts/train_tacthru_umi_v2.sh"
CONFIG="$ROOT_DIR/configs/vla/tacthru_umi/tacthru_umi_tactile.yaml"
ROBOT_CONFIG="$ROOT_DIR/configs/robot_configs/tacthru_umi_v2_tactile.yaml"
source "$ROOT_DIR/scripts/tactile_train_contract.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_tacthru_umi_v2_tactile.sh DATASET [options] [-- train overrides]

This launcher converts one wrist+tactile-RGB+marker superset and reuses it for
all four architecture ablations. It never changes the original wrist-only
launcher or its data/checkpoint paths.

Tactile options:
  --tactile-mode MODE       none | rgb | marker | rgb-marker (default: rgb-marker)
  --tactile-stage STAGE     adapters | expert | full (default: full)

All other wrapper options, including --task, --skip-convert, --skip-norm,
--check-only and arguments after --, are forwarded to train_tacthru_umi_v2.sh.
If omitted, dataset, norm and output paths are derived from the source name.

Examples:
  bash scripts/train_tacthru_umi_v2_tactile.sh data/task.zarr.zip \
    --tactile-mode rgb-marker --tactile-stage adapters --check-only

  bash scripts/train_tacthru_umi_v2_tactile.sh data/task.zarr.zip \
    --tactile-mode marker --tactile-stage full --skip-convert --skip-norm
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

INPUT="$1"
shift
TACTILE_MODE="rgb-marker"
TACTILE_STAGE="full"
OUTPUT_DIR=""
LEROBOT_DIR=""
FORWARD_ARGS=()
TRAIN_OVERRIDE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tactile-mode)
      [[ $# -ge 2 ]] || { echo "--tactile-mode requires a value" >&2; exit 2; }
      TACTILE_MODE="$2"
      shift 2
      ;;
    --tactile-stage)
      [[ $# -ge 2 ]] || { echo "--tactile-stage requires a value" >&2; exit 2; }
      TACTILE_STAGE="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --lerobot-dir)
      [[ $# -ge 2 ]] || { echo "--lerobot-dir requires a value" >&2; exit 2; }
      LEROBOT_DIR="$2"
      shift 2
      ;;
    --)
      shift
      TRAIN_OVERRIDE_ARGS=("$@")
      FORWARD_ARGS+=(-- "${TRAIN_OVERRIDE_ARGS[@]}")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$TACTILE_MODE" in
  none|rgb|marker|rgb-marker) ;;
  *) echo "--tactile-mode must be none, rgb, marker, or rgb-marker" >&2; exit 2 ;;
esac
case "$TACTILE_STAGE" in
  adapters|expert|full) ;;
  *) echo "--tactile-stage must be adapters, expert, or full" >&2; exit 2 ;;
esac
if [[ "$TACTILE_MODE" == none && "$TACTILE_STAGE" != full ]]; then
  echo "--tactile-stage must be full when --tactile-mode=none" >&2
  exit 2
fi
tactile_validate_train_overrides "${TRAIN_OVERRIDE_ARGS[@]}"
if [[ ! -e "$INPUT" ]]; then
  echo "Dataset does not exist: $INPUT" >&2
  exit 1
fi
for required in "$BASE_LAUNCHER" "$CONFIG" "$ROBOT_CONFIG"; do
  [[ -f "$required" ]] || { echo "Required file is missing: $required" >&2; exit 1; }
done

INPUT_ABS="$(readlink -f "$INPUT")"
STEM="$(basename "$INPUT_ABS")"
STEM="${STEM%.zarr.zip}"
STEM="${STEM%.zarr}"
STEM="$(printf '%s' "$STEM" | tr -cs '[:alnum:]_.-' '_')"

if [[ -f "$INPUT_ABS/meta/info.json" ]]; then
  LEROBOT_DIR="$INPUT_ABS"
elif [[ -z "$LEROBOT_DIR" ]]; then
  LEROBOT_DIR="$ROOT_DIR/data/lerobot/${STEM}_tacthru_umi_v2_tactile_v1"
fi
LEROBOT_DIR="$(readlink -m "$LEROBOT_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/output/${STEM}_tactile_ablations/${TACTILE_MODE}_${TACTILE_STAGE}"
fi
OUTPUT_DIR="$(readlink -m "$OUTPUT_DIR")"
FORWARD_ARGS=(
  --lerobot-dir "$LEROBOT_DIR"
  --output-dir "$OUTPUT_DIR"
  "${FORWARD_ARGS[@]}"
)
NORM_FILE="${NORM_FILE:-$ROOT_DIR/assets/norm_stats/${STEM}_tacthru_umi_v2_tactile_v1.json}"
NORM_FILE="$(readlink -m "$NORM_FILE")"
INIT_MODEL_DIR="$(readlink -m "${MODEL_DIR:-$ROOT_DIR/models/lingbot-vla-v2-6b}")"
BASE_MODEL_ASSET_DIR="$(readlink -m "${BASE_MODEL_ASSET_DIR:-$ROOT_DIR/models/lingbot-vla-v2-6b}")"

mkdir -p "$OUTPUT_DIR"
"${PYTHON:-$(command -v python)}" - \
  "$OUTPUT_DIR" "$INPUT_ABS" "$LEROBOT_DIR" "$NORM_FILE" \
  "$TACTILE_MODE" "$TACTILE_STAGE" "$CONFIG" \
  "$INIT_MODEL_DIR" "$BASE_MODEL_ASSET_DIR" \
  "${TRAIN_OVERRIDE_ARGS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, source, dataset, norm, mode, stage, config, init_model, base_assets = map(
    Path, sys.argv[1:10]
)
mode = mode.name
stage = stage.name
train_overrides = sys.argv[10:]
manifest_path = output / "tactile_experiment.json"

config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
contract = {
    "schema_version": 1,
    "dataset_tactile_mode": "rgb-marker",
    "train_tactile_mode": mode,
    "tactile_train_stage": stage,
    "source": str(source),
    "lerobot_dataset": str(dataset),
    "norm_stats": str(norm),
    "config": str(config),
    "config_sha256": config_sha,
    "initial_model": str(init_model),
    "base_model_assets": str(base_assets),
    "train_overrides": train_overrides,
    "wrist_only_baseline": "tacthru-umi-v2-wrist-only-20260724",
}

if manifest_path.is_file():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_contract = {key: previous.get(key) for key in contract}
    if previous_contract != contract:
        raise SystemExit(
            "Refusing cross-experiment resume in the same output directory.\n"
            f"existing={json.dumps(previous_contract, ensure_ascii=False, sort_keys=True)}\n"
            f"requested={json.dumps(contract, ensure_ascii=False, sort_keys=True)}"
        )
else:
    existing = [path.name for path in output.iterdir()]
    if existing:
        raise SystemExit(
            f"Refusing non-empty output without tactile_experiment.json: {output}; "
            f"entries={existing[:20]}"
        )
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(config.parents[3]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_head = None
    payload = {
        **contract,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
PY

echo "[tactile-train] dataset superset: $LEROBOT_DIR"
echo "[tactile-train] architecture:    $TACTILE_MODE"
echo "[tactile-train] stage:           $TACTILE_STAGE"
echo "[tactile-train] norm:            $NORM_FILE"
echo "[tactile-train] output:          $OUTPUT_DIR"

CONFIG="$CONFIG" \
DATA_NAME=tacthru_umi_v2_tactile \
ROBOT_CONFIG="$ROBOT_CONFIG" \
DATASET_TACTILE_MODE=rgb-marker \
TRAIN_TACTILE_MODE="$TACTILE_MODE" \
TACTILE_TRAIN_STAGE="$TACTILE_STAGE" \
NORM_FILE="$NORM_FILE" \
  exec bash "$BASE_LAUNCHER" "$INPUT_ABS" "${FORWARD_ARGS[@]}"
