#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LINGBOT_V2_CHECKPOINT="${LINGBOT_V2_TACTILE_CHECKPOINT:-$PROJECT_ROOT/output/tacthru_umi_v2_tactile/checkpoints/global_step_40000/hf_ckpt}"
export LINGBOT_V2_NORM_STATS="${LINGBOT_V2_TACTILE_NORM_STATS:-$PROJECT_ROOT/assets/norm_stats/tacthru_umi_v2_tactile_v1.json}"
PORT="${LINGBOT_V2_TACTILE_PORT:-18082}"
MAX_BODY_MB="${LINGBOT_V2_TACTILE_MAX_BODY_MB:-32}"

extra_flags=()
case "${LINGBOT_V2_ALLOW_MISSING_TACTILE:-0}" in
  1|true|TRUE|yes|YES) extra_flags+=(--allow-missing-tactile) ;;
  0|false|FALSE|no|NO) extra_flags+=(--no-allow-missing-tactile) ;;
  *) echo "LINGBOT_V2_ALLOW_MISSING_TACTILE must be 0/1 or true/false" >&2; exit 2 ;;
esac
case "${LINGBOT_V2_ALLOW_TACTILE_ABLATION:-0}" in
  1|true|TRUE|yes|YES) extra_flags+=(--allow-tactile-ablation) ;;
  0|false|FALSE|no|NO) extra_flags+=(--no-allow-tactile-ablation) ;;
  *) echo "LINGBOT_V2_ALLOW_TACTILE_ABLATION must be 0/1 or true/false" >&2; exit 2 ;;
esac

echo "[lingbot-v2-tactile-server] protocol=v1+v2 port=${PORT} checkpoint=${LINGBOT_V2_CHECKPOINT}"
exec bash "$PROJECT_ROOT/scripts/run_tacthru_umi_v2_server.sh" \
  --port "$PORT" \
  --max-body-mb "$MAX_BODY_MB" \
  "${extra_flags[@]}" \
  "$@"
