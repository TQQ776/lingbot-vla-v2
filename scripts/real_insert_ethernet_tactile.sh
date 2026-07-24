#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-1}"
EXECUTE_REAL="${LINGBOT_V2_EXECUTE:-0}"
MOTION_PROFILE="${LINGBOT_V2_MOTION_PROFILE:-smooth-conservative}"
PORT="${LINGBOT_V2_TACTILE_PORT:-18082}"

case "$MOTION_PROFILE" in
  legacy)
    DEFAULT_EXEC_END_STEP=3
    DEFAULT_MAX_POS_SPEED=0.02
    DEFAULT_MAX_ROT_SPEED=0.05
    ;;
  smooth-conservative)
    DEFAULT_EXEC_END_STEP=12
    DEFAULT_MAX_POS_SPEED=0.03
    DEFAULT_MAX_ROT_SPEED=0.05
    ;;
  *) echo "LINGBOT_V2_MOTION_PROFILE must be legacy or smooth-conservative" >&2; exit 2 ;;
esac

EXEC_END_STEP="${EXEC_END_STEP:-$DEFAULT_EXEC_END_STEP}"
MAX_POS_SPEED="${LINGBOT_V2_MAX_POS_SPEED:-$DEFAULT_MAX_POS_SPEED}"
MAX_ROT_SPEED="${LINGBOT_V2_MAX_ROT_SPEED:-$DEFAULT_MAX_ROT_SPEED}"
GRIPPER_STARTUP_WIDTH_M="${GRIPPER_STARTUP_WIDTH_M:-0.004}"
REQUEST_TIMEOUT_S="${LINGBOT_V2_REQUEST_TIMEOUT_S:-5.0}"
MAX_REJECTS="${LINGBOT_V2_MAX_CONSECUTIVE_ROUNDTRIP_REJECTS:-3}"
TACTILE_CFG="${LINGBOT_V2_TACTILE_SENSOR_CFG:-/mnt/models/VTLA-RDT/tacthru/cfg/sensor/ml.yaml}"

if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer, got: $STEPS" >&2
  exit 2
fi
if ! [[ "$EXEC_END_STEP" =~ ^[0-9]+$ ]] || (( EXEC_END_STEP <= 2 || EXEC_END_STEP > 50 )); then
  echo "EXEC_END_STEP must be in [3,50], got: $EXEC_END_STEP" >&2
  exit 2
fi

execute_flags=()
workspace_flags=()
case "$EXECUTE_REAL" in
  1|true|TRUE|yes|YES)
    execute_flags=(--execute)
    if [[ -z "${WORKSPACE_MIN_XYZ:-}" || -z "${WORKSPACE_MAX_XYZ:-}" ]]; then
      echo "Real tactile execution requires WORKSPACE_MIN_XYZ and WORKSPACE_MAX_XYZ." >&2
      exit 2
    fi
    read -r -a workspace_min <<<"$WORKSPACE_MIN_XYZ"
    read -r -a workspace_max <<<"$WORKSPACE_MAX_XYZ"
    if (( ${#workspace_min[@]} != 3 || ${#workspace_max[@]} != 3 )); then
      echo "WORKSPACE_MIN_XYZ/MAX_XYZ must each contain exactly three numbers." >&2
      exit 2
    fi
    workspace_flags=(
      --workspace-min-xyz "${workspace_min[@]}"
      --workspace-max-xyz "${workspace_max[@]}"
    )
    ;;
  0|false|FALSE|no|NO) ;;
  *) echo "LINGBOT_V2_EXECUTE must be 0/1 or true/false" >&2; exit 2 ;;
esac

echo "[lingbot-v2-tactile-client] mode=${LINGBOT_V2_TACTILE_MODE:-rgb-marker} port=${PORT} execute=${EXECUTE_REAL}"
echo "[lingbot-v2-tactile-client] original scripts/real_insert_ethernet.sh remains the protocol-v1 rollback path"

exec bash "$PROJECT_ROOT/scripts/run_tacthru_umi_v2_tactile_client.sh" run \
  --server-url "http://127.0.0.1:${PORT}" \
  --timeout "$REQUEST_TIMEOUT_S" \
  --http-keep-alive \
  --instruction "Insert the Ethernet cable." \
  --tacthru-repo /mnt/models/VTLA-RDT/tacthru \
  --camera-cfg /mnt/models/VTLA-RDT/tacthru/cfg/camera/synria_c10.yaml \
  --tactile-sensor-cfg "$TACTILE_CFG" \
  --robot-cfg /mnt/models/VTLA-RDT/tacthru/cfg/robot/realman.yaml \
  --gripper-cfg /mnt/models/VTLA-RDT/tacthru/cfg/gripper/synria_gloria.yaml \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --steps "$STEPS" \
  --rate-hz 1 \
  --preview \
  "${execute_flags[@]}" \
  --local-tactile-guard \
  --max-tactile-age-s "${LINGBOT_V2_MAX_TACTILE_AGE_S:-0.15}" \
  --max-tactile-skew-s "${LINGBOT_V2_MAX_TACTILE_SKEW_S:-0.05}" \
  --min-valid-markers "${LINGBOT_V2_MIN_VALID_MARKERS:-40}" \
  --assumed-gripper-width-m "$GRIPPER_STARTUP_WIDTH_M" \
  --gripper-startup-width-m "$GRIPPER_STARTUP_WIDTH_M" \
  --gripper-startup-tolerance-m 0.005 \
  --gripper-startup-timeout-s 3.0 \
  --gripper-action-select threshold \
  --gripper-hold-closed-below-m 0.010 \
  --gripper-hold-closed-target-m "$GRIPPER_STARTUP_WIDTH_M" \
  --gripper-open-lookahead-start-step "${LINGBOT_V2_GRIPPER_LOOKAHEAD_START_STEP:-25}" \
  --gripper-open-lookahead-consecutive-steps "${LINGBOT_V2_GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS:-5}" \
  --gripper-exit-policy prompt \
  --exec-start-step 2 \
  --exec-end-step "$EXEC_END_STEP" \
  --max-roundtrip-s 2.0 \
  --max-consecutive-roundtrip-rejects "$MAX_REJECTS" \
  --max-sensor-skew-s 0.10 \
  --max-pos-speed "$MAX_POS_SPEED" \
  --max-rot-speed "$MAX_ROT_SPEED" \
  --max-target-delta-m 0.050 \
  --max-target-rotation-rad 0.10 \
  --max-step-delta-m 0.020 \
  --max-step-rotation-rad 0.10 \
  --max-observation-drift-m 0.01 \
  --max-observation-drift-rotation-rad 0.10 \
  --max-scheduled-duration-s 2.0 \
  --verify-position-tolerance-m 0.005 \
  --verify-rotation-tolerance-rad 0.05 \
  --verify-gripper-tolerance-m 0.005 \
  --verification-timeout-s 2.0 \
  "${workspace_flags[@]}"
