#!/usr/bin/env bash
set -euo pipefail

# 首次真实夹爪验证保持 STEPS=1；确认硬件反馈、夹持宽度和机械臂轨迹后，
# 再逐步增加到当前批准的默认 STEPS=20：STEPS=20 bash scripts/real_realman.sh
STEPS="${STEPS:-20}"
if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer, got: ${STEPS}" >&2
  exit 2
fi

# 默认复用 SSH 隧道中的同一条 HTTP/1.1 连接。
# 随时回退：LINGBOT_V2_HTTP_KEEP_ALIVE=0 bash scripts/real_realman.sh
HTTP_KEEP_ALIVE="${LINGBOT_V2_HTTP_KEEP_ALIVE:-1}"
case "${HTTP_KEEP_ALIVE}" in
  1|true|TRUE|yes|YES) HTTP_KEEP_ALIVE_FLAG="--http-keep-alive" ;;
  0|false|FALSE|no|NO) HTTP_KEEP_ALIVE_FLAG="--no-http-keep-alive" ;;
  *)
    echo "LINGBOT_V2_HTTP_KEEP_ALIVE must be 0/1 or true/false, got: ${HTTP_KEEP_ALIVE}" >&2
    exit 2
    ;;
esac

if [[ -z "${LINGBOT_V2_API_KEY:-}" ]]; then
  echo "LINGBOT_V2_API_KEY is not set in this terminal." >&2
  exit 2
fi

cd /mnt/models/VTLA-RDT/lingbot-vla-v2

# 夹爪策略：当前执行窗口预测宽度 <=12mm 时闭合到4mm，否则打开到初始宽度45mm。
# 每轮重新判断，不保留闭合锁存；机械臂执行[2,8)共6个waypoint。
bash scripts/run_tacthru_umi_v2_client.sh run \
  --server-url http://127.0.0.1:18081 \
  "${HTTP_KEEP_ALIVE_FLAG}" \
  --instruction "Pull the tissue" \
  --tacthru-repo /mnt/models/VTLA-RDT/tacthru \
  --camera-cfg /mnt/models/VTLA-RDT/tacthru/cfg/camera/synria_c10.yaml \
  --robot-cfg /mnt/models/VTLA-RDT/tacthru/cfg/robot/realman.yaml \
  --gripper-cfg /mnt/models/VTLA-RDT/tacthru/cfg/gripper/synria_gloria.yaml \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --steps "${STEPS}" \
  --rate-hz 1 \
  --preview \
  --execute \
  --gripper-action-select threshold \
  --gripper-hold-closed-below-m 0.012 \
  --gripper-hold-closed-target-m 0.004 \
  --gripper-exit-policy prompt \
  --exec-start-step 2 \
  --exec-end-step 8 \
  --max-roundtrip-s 1.6 \
  --max-sensor-skew-s 0.10 \
  --max-pos-speed 0.02 \
  --max-rot-speed 0.05 \
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
  --workspace-min-xyz -0.049504 -0.453634 0.043029 \
  --workspace-max-xyz 0.496991 -0.228485 0.282270
