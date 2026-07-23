#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LINGBOT_V2_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
CHECKPOINT="${LINGBOT_V2_CHECKPOINT:-$PROJECT_ROOT/output/pull_tissue_v2_formal/checkpoints/global_step_40000/hf_ckpt}"
NORM_STATS="${LINGBOT_V2_NORM_STATS:-$PROJECT_ROOT/assets/norm_stats/tacthru_umi_v2.json}"
QWEN_PATH="${QWEN3VL_PATH:-$PROJECT_ROOT/models/Qwen3-VL-4B-Instruct}"
HTTP_KEEP_ALIVE="${LINGBOT_V2_HTTP_KEEP_ALIVE:-1}"

case "${HTTP_KEEP_ALIVE}" in
  1|true|TRUE|yes|YES) HTTP_KEEP_ALIVE_FLAG="--http-keep-alive" ;;
  0|false|FALSE|no|NO) HTTP_KEEP_ALIVE_FLAG="--no-http-keep-alive" ;;
  *)
    echo "LINGBOT_V2_HTTP_KEEP_ALIVE must be 0/1 or true/false, got: ${HTTP_KEEP_ALIVE}" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing LingBot V2 Python: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Missing final HF checkpoint directory: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$NORM_STATS" ]]; then
  echo "Missing TacThru UMI normalization statistics: $NORM_STATS" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3VL_PATH="$QWEN_PATH"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" -m deploy.tacthru_umi_v2.http_server \
  --project-root "$PROJECT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --norm-stats "$NORM_STATS" \
  --qwen-path "$QWEN_PATH" \
  "${HTTP_KEEP_ALIVE_FLAG}" \
  "$@"
