#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-1}"
MOTION_PROFILE="${LINGBOT_V2_MOTION_PROFILE:-legacy}"
EXECUTE_REAL="${LINGBOT_V2_EXECUTE:-1}"

# The conservative profile keeps blocking completion verification and every
# existing safety bound. It only lengthens each interpolated chunk and raises
# translational speed to TacThru's documented real-robot value.
case "${MOTION_PROFILE}" in
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
  *)
    echo "LINGBOT_V2_MOTION_PROFILE must be legacy or smooth-conservative, got: ${MOTION_PROFILE}" >&2
    exit 2
    ;;
esac

EXEC_END_STEP="${EXEC_END_STEP:-${DEFAULT_EXEC_END_STEP}}"
MAX_POS_SPEED="${LINGBOT_V2_MAX_POS_SPEED:-${DEFAULT_MAX_POS_SPEED}}"
MAX_ROT_SPEED="${LINGBOT_V2_MAX_ROT_SPEED:-${DEFAULT_MAX_ROT_SPEED}}"
GRIPPER_STARTUP_WIDTH_M="${GRIPPER_STARTUP_WIDTH_M:-0.004}"
GRIPPER_STARTUP_TOLERANCE_M="${GRIPPER_STARTUP_TOLERANCE_M:-0.005}"
GRIPPER_STARTUP_TIMEOUT_S="${GRIPPER_STARTUP_TIMEOUT_S:-3.0}"
REQUEST_TIMEOUT_S="${LINGBOT_V2_REQUEST_TIMEOUT_S:-5.0}"
TACTILE_SENSOR_CFG="${LINGBOT_V2_TACTILE_SENSOR_CFG:-/mnt/models/VTLA-RDT/tacthru/cfg/sensor/ml.yaml}"
MAX_TACTILE_AGE_S="${LINGBOT_V2_MAX_TACTILE_AGE_S:-0.25}"
MAX_TACTILE_SKEW_S="${LINGBOT_V2_MAX_TACTILE_SKEW_S:-0.20}"
MIN_VALID_MARKERS="${LINGBOT_V2_MIN_VALID_MARKERS:-40}"
# Set this to 0 to restore the previous fail-fast behavior without editing code.
MAX_CONSECUTIVE_ROUNDTRIP_REJECTS="${LINGBOT_V2_MAX_CONSECUTIVE_ROUNDTRIP_REJECTS:-3}"
GRIPPER_LOOKAHEAD_RELEASE="${LINGBOT_V2_GRIPPER_LOOKAHEAD_RELEASE:-1}"
GRIPPER_LOOKAHEAD_START_STEP="${LINGBOT_V2_GRIPPER_LOOKAHEAD_START_STEP:-25}"
GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS="${LINGBOT_V2_GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS:-5}"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer, got: ${STEPS}" >&2
  exit 2
fi
if ! [[ "${EXEC_END_STEP}" =~ ^[0-9]+$ ]] || (( EXEC_END_STEP <= 2 || EXEC_END_STEP > 50 )); then
  echo "EXEC_END_STEP must be an integer in [3, 50], got: ${EXEC_END_STEP}" >&2
  exit 2
fi
if ! [[ "${MAX_CONSECUTIVE_ROUNDTRIP_REJECTS}" =~ ^[0-9]+$ ]]; then
  echo "LINGBOT_V2_MAX_CONSECUTIVE_ROUNDTRIP_REJECTS must be a non-negative integer, got: ${MAX_CONSECUTIVE_ROUNDTRIP_REJECTS}" >&2
  exit 2
fi
if ! [[ "${MIN_VALID_MARKERS}" =~ ^[0-9]+$ ]] || (( MIN_VALID_MARKERS > 48 )); then
  echo "LINGBOT_V2_MIN_VALID_MARKERS must be an integer in [0, 48], got: ${MIN_VALID_MARKERS}" >&2
  exit 2
fi
EXECUTE_FLAGS=()
case "${EXECUTE_REAL}" in
  1|true|TRUE|yes|YES) EXECUTE_FLAGS=(--execute) ;;
  0|false|FALSE|no|NO) ;;
  *)
    echo "LINGBOT_V2_EXECUTE must be 0/1 or true/false, got: ${EXECUTE_REAL}" >&2
    exit 2
    ;;
esac
GRIPPER_LOOKAHEAD_FLAGS=()
case "${GRIPPER_LOOKAHEAD_RELEASE}" in
  1|true|TRUE|yes|YES)
    if ! [[ "${GRIPPER_LOOKAHEAD_START_STEP}" =~ ^[0-9]+$ ]]; then
      echo "LINGBOT_V2_GRIPPER_LOOKAHEAD_START_STEP must be a non-negative integer, got: ${GRIPPER_LOOKAHEAD_START_STEP}" >&2
      exit 2
    fi
    if ! [[ "${GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "LINGBOT_V2_GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS must be a positive integer, got: ${GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS}" >&2
      exit 2
    fi
    if (( GRIPPER_LOOKAHEAD_START_STEP + GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS > 50 )); then
      echo "Gripper lookahead start + consecutive steps must fit the 50-step action chunk." >&2
      exit 2
    fi
    GRIPPER_LOOKAHEAD_FLAGS=(
      --gripper-open-lookahead-start-step "${GRIPPER_LOOKAHEAD_START_STEP}"
      --gripper-open-lookahead-consecutive-steps "${GRIPPER_LOOKAHEAD_CONSECUTIVE_STEPS}"
    )
    ;;
  0|false|FALSE|no|NO)
    # Omitting both CLI options restores the original selected-window-only policy.
    ;;
  *)
    echo "LINGBOT_V2_GRIPPER_LOOKAHEAD_RELEASE must be 0/1 or true/false, got: ${GRIPPER_LOOKAHEAD_RELEASE}" >&2
    exit 2
    ;;
esac
if [[ -z "${WORKSPACE_MIN_XYZ:-}" || -z "${WORKSPACE_MAX_XYZ:-}" ]]; then
  echo "Set task-specific Realman Base bounds before real execution:" >&2
  echo "  WORKSPACE_MIN_XYZ='X Y Z' WORKSPACE_MAX_XYZ='X Y Z' bash scripts/real_insert_ethernet.sh" >&2
  exit 2
fi

read -r -a workspace_min <<<"${WORKSPACE_MIN_XYZ}"
read -r -a workspace_max <<<"${WORKSPACE_MAX_XYZ}"
if (( ${#workspace_min[@]} != 3 || ${#workspace_max[@]} != 3 )); then
  echo "WORKSPACE_MIN_XYZ and WORKSPACE_MAX_XYZ must each contain exactly three numbers." >&2
  exit 2
fi

HTTP_KEEP_ALIVE="${LINGBOT_V2_HTTP_KEEP_ALIVE:-1}"
case "${HTTP_KEEP_ALIVE}" in
  1|true|TRUE|yes|YES) HTTP_KEEP_ALIVE_FLAG="--http-keep-alive" ;;
  0|false|FALSE|no|NO) HTTP_KEEP_ALIVE_FLAG="--no-http-keep-alive" ;;
  *)
    echo "LINGBOT_V2_HTTP_KEEP_ALIVE must be 0/1 or true/false, got: ${HTTP_KEEP_ALIVE}" >&2
    exit 2
    ;;
esac

cd /mnt/models/VTLA-RDT/lingbot-vla-v2

echo "[lingbot-v2-client] motion_profile=${MOTION_PROFILE} exec_window=[2,${EXEC_END_STEP}) max_pos_speed=${MAX_POS_SPEED}m/s max_rot_speed=${MAX_ROT_SPEED}rad/s"
if [[ ${#EXECUTE_FLAGS[@]} -eq 0 ]]; then
  echo "[lingbot-v2-client] dry-run mode: no arm trajectory or gripper command will be sent"
fi

bash scripts/run_tacthru_umi_v2_client.sh run \
  --server-url http://127.0.0.1:18081 \
  --timeout "${REQUEST_TIMEOUT_S}" \
  "${HTTP_KEEP_ALIVE_FLAG}" \
  --instruction "Insert the Ethernet cable." \
  --tacthru-repo /mnt/models/VTLA-RDT/tacthru \
  --camera-cfg /mnt/models/VTLA-RDT/tacthru/cfg/camera/synria_c10.yaml \
  --tactile-sensor-cfg "${TACTILE_SENSOR_CFG}" \
  --robot-cfg /mnt/models/VTLA-RDT/tacthru/cfg/robot/realman.yaml \
  --gripper-cfg /mnt/models/VTLA-RDT/tacthru/cfg/gripper/synria_gloria.yaml \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --steps "${STEPS}" \
  --rate-hz 1 \
  --preview \
  "${EXECUTE_FLAGS[@]}" \
  --assumed-gripper-width-m "${GRIPPER_STARTUP_WIDTH_M}" \
  --gripper-startup-width-m "${GRIPPER_STARTUP_WIDTH_M}" \
  --gripper-startup-tolerance-m "${GRIPPER_STARTUP_TOLERANCE_M}" \
  --gripper-startup-timeout-s "${GRIPPER_STARTUP_TIMEOUT_S}" \
  --gripper-action-select threshold \
  --gripper-hold-closed-below-m 0.010 \
  --gripper-hold-closed-target-m "${GRIPPER_STARTUP_WIDTH_M}" \
  "${GRIPPER_LOOKAHEAD_FLAGS[@]}" \
  --gripper-exit-policy prompt \
  --exec-start-step 2 \
  --exec-end-step "${EXEC_END_STEP}" \
  --max-roundtrip-s 2.0 \
  --max-consecutive-roundtrip-rejects "${MAX_CONSECUTIVE_ROUNDTRIP_REJECTS}" \
  --max-sensor-skew-s 0.10 \
  --max-tactile-skew-s "${MAX_TACTILE_SKEW_S}" \
  --max-tactile-age-s "${MAX_TACTILE_AGE_S}" \
  --min-valid-markers "${MIN_VALID_MARKERS}" \
  --max-pos-speed "${MAX_POS_SPEED}" \
  --max-rot-speed "${MAX_ROT_SPEED}" \
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
  --workspace-min-xyz "${workspace_min[@]}" \
  --workspace-max-xyz "${workspace_max[@]}"
