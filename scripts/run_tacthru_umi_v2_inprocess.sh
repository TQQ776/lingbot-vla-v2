#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LINGBOT_V2_INPROCESS_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
CHECKPOINT="${LINGBOT_V2_CHECKPOINT:-$PROJECT_ROOT/output/insert_ethernet_vtla_rgb_marker_all201_v1/checkpoints/global_step_40000/hf_ckpt}"
NORM_STATS="${LINGBOT_V2_NORM_STATS:-$PROJECT_ROOT/assets/norm_stats/insert_ethernet_cable_ml_0721_201_vtla_all201.json}"
QWEN_PATH="${QWEN3VL_PATH:-$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct}"
WARMUP="${LINGBOT_V2_WARMUP:-1}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {health|synthetic|run} [client arguments...]" >&2
  exit 2
fi
COMMAND="$1"
shift
case "$COMMAND" in
  health|synthetic|run) ;;
  *)
    echo "First argument must be health, synthetic, or run; got: $COMMAND" >&2
    exit 2
    ;;
esac

case "$WARMUP" in
  1|true|TRUE|yes|YES) WARMUP_FLAG="--warmup" ;;
  0|false|FALSE|no|NO) WARMUP_FLAG="--no-warmup" ;;
  *)
    echo "LINGBOT_V2_WARMUP must be 0/1 or true/false, got: $WARMUP" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing in-process LingBot V2 Python: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Missing final HF checkpoint directory: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$NORM_STATS" ]]; then
  echo "Missing normalization statistics: $NORM_STATS" >&2
  exit 1
fi
if [[ ! -d "$QWEN_PATH" ]]; then
  echo "Missing Qwen3-VL model directory: $QWEN_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3VL_PATH="$QWEN_PATH"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" -m deploy.tacthru_umi_v2.realman_client "$COMMAND" \
  --transport inprocess \
  --project-root "$PROJECT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --norm-stats "$NORM_STATS" \
  --qwen-path "$QWEN_PATH" \
  "$WARMUP_FLAG" \
  "$@"
