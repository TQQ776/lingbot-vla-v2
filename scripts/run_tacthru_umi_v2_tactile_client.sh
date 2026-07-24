#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 health|synthetic|run [arguments...]" >&2
  exit 2
fi

command_name="$1"
shift
if [[ "$command_name" == "health" ]]; then
  exec bash "$PROJECT_ROOT/scripts/run_tacthru_umi_v2_client.sh" health "$@"
fi

tactile_mode="${LINGBOT_V2_TACTILE_MODE:-rgb-marker}"
extra_flags=(--protocol-version 2 --tactile-mode "$tactile_mode")
case "${LINGBOT_V2_ALLOW_MISSING_TACTILE:-0}" in
  1|true|TRUE|yes|YES) extra_flags+=(--allow-missing-tactile) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "LINGBOT_V2_ALLOW_MISSING_TACTILE must be 0/1 or true/false" >&2; exit 2 ;;
esac
case "${LINGBOT_V2_FORCE_MASK_TACTILE_RGB:-0}" in
  1|true|TRUE|yes|YES) extra_flags+=(--force-mask-tactile-rgb) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "LINGBOT_V2_FORCE_MASK_TACTILE_RGB must be 0/1 or true/false" >&2; exit 2 ;;
esac
case "${LINGBOT_V2_FORCE_MASK_TACTILE_MARKER:-0}" in
  1|true|TRUE|yes|YES) extra_flags+=(--force-mask-marker) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "LINGBOT_V2_FORCE_MASK_TACTILE_MARKER must be 0/1 or true/false" >&2; exit 2 ;;
esac

exec bash "$PROJECT_ROOT/scripts/run_tacthru_umi_v2_client.sh" \
  "$command_name" \
  "${extra_flags[@]}" \
  "$@"
