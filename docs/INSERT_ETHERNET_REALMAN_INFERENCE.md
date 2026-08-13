# 插网线：LingBot V2 服务器 + 本地 Realman 联合推理

任务指令以训练数据 `meta/tasks.parquet` 为准：

```text
Insert the Ethernet cable.
```

## 1. B200 服务器

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2

export LINGBOT_V2_CHECKPOINT='/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2/output/insert_ethernet_cable_ml_0721_201_v2_normfix/checkpoints/global_step_40000/hf_ckpt'
export LINGBOT_V2_NORM_STATS='/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2/assets/norm_stats/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_float64.json'
export QWEN3VL_PATH='/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2/models/Qwen3-VL-4B-Instruct'
export CUDA_VISIBLE_DEVICES=0

env -u LINGBOT_V2_API_KEY bash scripts/run_tacthru_umi_v2_server.sh \
  --host 127.0.0.1 \
  --port 18081 \
  --use-compile \
  --warmup \
  --warmup-instruction "Insert the Ethernet cable."
```

首次真机验证先不加 `--use-compile`。服务启动后保持终端运行。
同一台机器运行 server 和真机 client 时，不需要 SSH 隧道。client 退出只会关闭
本轮相机、触觉、夹爪、机械臂和 HTTP 连接，不会结束 server；下一轮重新运行 client
即可复用已加载且已完成 warmup 的模型。
若启动日志提示 `using checkpoint norm instead of the robot config default`，表示正在用该
checkpoint 的训练 norm 覆盖共享 robot YAML 中的旧默认值，属于预期行为。

## 2. 本地 SSH 隧道

新开本地终端：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2
bash scripts/run_tacthru_umi_v2_tunnel.sh
```

脚本从 `~/.ssh/config` 读取当前 SSH 端口，并保持前台运行；`Ctrl-C` 关闭隧道。

## 3. Health 与无硬件推理

再开一个本地终端：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

bash scripts/run_tacthru_umi_v2_client.sh health \
  --server-url http://127.0.0.1:18081 \
  --http-keep-alive

bash scripts/run_tacthru_umi_v2_client.sh synthetic \
  --server-url http://127.0.0.1:18081 \
  --http-keep-alive \
  --instruction "Insert the Ethernet cable." \
  --state '0,0,0,0,0,0,1,0.004'
```

## 4. 真机 dry-run

下面的命令会打开腕部相机并读取机械臂状态，但不会启动 Gloria，也不会下发轨迹：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

bash scripts/run_tacthru_umi_v2_client.sh run \
  --server-url http://127.0.0.1:18081 \
  --http-keep-alive \
  --instruction "Insert the Ethernet cable." \
  --tacthru-repo /mnt/models/VTLA-RDT/tacthru \
  --camera-cfg /mnt/models/VTLA-RDT/tacthru/cfg/camera/synria_c10.yaml \
  --robot-cfg /mnt/models/VTLA-RDT/tacthru/cfg/robot/realman.yaml \
  --gripper-cfg /mnt/models/VTLA-RDT/tacthru/cfg/gripper/synria_gloria.yaml \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --assumed-gripper-width-m 0.004 \
  --preview \
  --steps 5
```

从 dry-run 日志中的 `controller_target_pos_after_adapter_safety_m` 和实际工位范围确定
Realman Base 坐标系的 `WORKSPACE_MIN_XYZ`、`WORKSPACE_MAX_XYZ`。不要复用抽纸任务边界。

## 5. 首次真实执行

把网线预先放在夹爪中，并把下面两组 workspace 占位值换成实测边界：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

WORKSPACE_MIN_XYZ='X_MIN Y_MIN Z_MIN' \
WORKSPACE_MAX_XYZ='X_MAX Y_MAX Z_MAX' \
STEPS=1 \
EXEC_END_STEP=3 \
bash scripts/real_insert_ethernet.sh
```

执行时有两次独立的 Space 门控：

1. 第一次允许 Gloria 直接初始化到 `4 mm`，不会先打开到 `45 mm`。
2. 程序必须读到新鲜、有效的舵机位置反馈，并确认实际宽度在 `4 +/- 5 mm` 内；否则终止。
3. 第二次 Space 由操作者确认网线确实夹牢；在此之前不会发送第一条在线推理请求。

每个可执行 action 窗口使用二值夹爪策略：任一预测宽度 `<=10 mm`，整段保持
`4 mm`；只有窗口内所有预测宽度都 `>10 mm` 才打开到 Gloria 配置的 `45 mm`。

首次命令默认只执行 action 索引 `2` 的一个 waypoint。确认图像、坐标、反馈和轨迹都正确后，
才逐步增加到六个 waypoint 和更多推理轮次：

```bash
WORKSPACE_MIN_XYZ='X_MIN Y_MIN Z_MIN' \
WORKSPACE_MAX_XYZ='X_MAX Y_MAX Z_MAX' \
STEPS=20 \
EXEC_END_STEP=8 \
bash scripts/real_insert_ethernet.sh
```

`real_insert_ethernet.sh` 默认固定执行半开区间 `[2, EXEC_END_STEP)`，不会根据网络延迟
向后平移；因此上面的 `EXEC_END_STEP=8` 始终执行 `[2,8)`。如需恢复原来的在线延迟
补偿，可设置 `LINGBOT_V2_FIXED_EXEC_WINDOW=0`。
