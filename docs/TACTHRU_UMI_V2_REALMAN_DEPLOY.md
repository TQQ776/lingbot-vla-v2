# LingBot-VLA V2 + TacThru UMI 真机部署

这套部署把 GPU 推理和真机控制分开，结构与 `RDT-client` 相同，但动作坐标和输入格式严格按本次 V2 数据集实现。

```text
本地 Realman client                         B200 inference server
┌──────────────────────────┐              ┌──────────────────────────┐
│ 腕部相机 1280×960 BGR     │              │ 最终 hf_ckpt              │
│ → 中心裁剪 224×224 RGB    │  HTTP/JPEG   │ LingbotVLAv2Server        │
│                           ├─────────────►│ FeatureTransform + norm   │
│ T_start_current + 夹爪宽度│              │                          │
│                           │◄─────────────┤ 50×8 action chunk         │
│ 安全预检 → RealmanAdapter │              │ episode-frame absolute   │
└──────────────────────────┘              └──────────────────────────┘
```

当前模型只使用腕部 RGB；触觉 RGB 和 marker 不会发送，也不会进入模型。

## 1. 坐标与数据语义

这是部署中最重要的约束。TacThru UMI 示教在每个 episode 开始时把 Vive tracker 设为原点，训练数据中的位姿因此位于 episode 起点 `new_tcp` 坐标系，而不是机械臂 Base 坐标系。

client 在 episode 开始时记录：

```text
T_base_start
```

发送给 server 的 8D state 是：

```text
T_start_current = inverse(T_base_start) @ T_base_current
state = xyz + quaternion_xyzw + gripper_width_m
```

V2 server 的官方 `FeatureTransform.unapply()` 已把模型内部的 `quaternion_local` 相对动作还原成 episode 坐标系下的绝对 8D target。client 执行时使用：

```text
T_base_target = T_base_start @ T_start_target
```

不能使用 RDT 的 `T_base_current @ relative_action`，否则会重复应用相对变换。第 8 维始终是夹爪绝对宽度（米），不是 `[0,1]` 开合度。

## 2. 新增文件

```text
deploy/tacthru_umi_v2/protocol.py       HTTP/JPEG 协议和严格 schema
deploy/tacthru_umi_v2/http_server.py    B200 推理 server
deploy/tacthru_umi_v2/realman_client.py 本地相机、网络和运行循环
deploy/tacthru_umi_v2/realman_runtime.py Realman/Gloria 桥接与安全预检
deploy/tacthru_umi_v2/transforms.py     episode/Base 位姿转换
scripts/run_tacthru_umi_v2_server.sh    server 启动入口
scripts/run_tacthru_umi_v2_client.sh    client 启动入口
```

server 使用 V2 项目的 Python 环境；client 使用 TacThru Python 3.11 Realman 环境。两边不共享 CUDA 或机械臂依赖。

## 3. 在 B200 启动 server

最终模型路径必须指向 Hugging Face 权重目录，不能指向 DCP checkpoint 父目录：

```text
output/pull_tissue_v2_formal/checkpoints/global_step_40000/hf_ckpt
```

建议设置 API key，再启动 server：

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2

export LINGBOT_V2_API_KEY='请换成随机长字符串'
export CUDA_VISIBLE_DEVICES=0

bash scripts/run_tacthru_umi_v2_server.sh \
  --host 127.0.0.1 \
  --port 18081 \
  --warmup
```

server wrapper 默认启用 HTTP/1.1 Keep-Alive，但仍只监听 SSH 隧道后的 loopback 地址。服务端回退到原来的 HTTP/1.0 `Connection: close`：

```bash
LINGBOT_V2_HTTP_KEEP_ALIVE=0 \
bash scripts/run_tacthru_umi_v2_server.sh \
  --host 127.0.0.1 \
  --port 18081 \
  --use-compile \
  --warmup
```

服务端模式改变需要重启 server；模型、checkpoint 和 norm 不受影响。

默认关闭 `torch.compile`，优先保证真机部署稳定。完成普通推理 dry-run 后，如果需要降低延迟，可重启时增加 `--use-compile`，并再次完成 synthetic 和 dry-run 验证。

真机执行默认要求 SSH 隧道，因为明文 HTTP 无法防止返回动作被网络中间人篡改：

```bash
ssh -p <当前SSH端口> -N \
  -L 18081:127.0.0.1:18081 \
  tqq@172.16.41.254
```

使用隧道时，本地 `--server-url` 写 `http://127.0.0.1:18081`。

平台 TCP 端口映射只建议用于可信内网中的 health、synthetic 和 dry-run。`--execute` 会拒绝非 localhost 的明文 HTTP；若不使用 SSH 隧道，必须在 server 前配置 HTTPS/mTLS。

## 4. 本地检查 server

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2
export LINGBOT_V2_API_KEY='与server相同的字符串'

bash scripts/run_tacthru_umi_v2_client.sh health \
  --server-url http://127.0.0.1:18081 \
  --http-keep-alive
```

health 必须明确显示：

```text
ready: true
robot_config: tacthru_umi_v2
pose_frame: episode_start_new_tcp
chunk_size: 50
camera_key: observation.images.camera_wrist_left
tactile_enabled: false
action_spec.quaternion_order: xyzw
action_spec.gripper_unit: m
contract.robot_config_sha256: ...
contract.norm_stats_sha256: ...
transport.http_keep_alive_enabled: true
```

## 5. 无硬件 synthetic 推理

这一步不打开相机、机械臂或夹爪：

```bash
bash scripts/run_tacthru_umi_v2_client.sh synthetic \
  --server-url http://127.0.0.1:18081 \
  --http-keep-alive \
  --instruction "Pull the tissue"
```

成功标准是返回 `action_shape: [50, 8]`，且没有协议、norm、checkpoint 或 CUDA 错误。

synthetic 和真机 step 日志会记录 `latency`，包括 JPEG/JSON 编码、TCP连接、请求写入、等待响应头、响应读取、解析、连接是否复用，以及服务端 `read/decode/lock/inference/encode` 分阶段时间。即使 `roundtrip_s` 超过安全上限，客户端也会先写入 `roundtrip_rejected` 事件，保留该响应的服务端推理时间，然后拒绝执行动作。

本地真机 wrapper 默认启用 Keep-Alive。只回退客户端、不重启 server：

```bash
LINGBOT_V2_HTTP_KEEP_ALIVE=0 bash scripts/real_realman.sh
```

此时客户端立即恢复原来的 `urllib + Connection: close`；启用了 Keep-Alive 的新 server 仍兼容该旧路径。

## 6. 真机 dry-run：默认不发送动作

先把电脑接入机械臂和相机所在网络，然后执行：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

bash scripts/run_tacthru_umi_v2_client.sh run \
  --server-url http://127.0.0.1:18081 \
  --instruction "Pull the tissue" \
  --tacthru-repo /mnt/models/VTLA-RDT/tacthru \
  --camera-cfg /mnt/models/VTLA-RDT/tacthru/cfg/camera/synria_c10.yaml \
  --robot-cfg /mnt/models/VTLA-RDT/tacthru/cfg/robot/realman.yaml \
  --gripper-cfg /mnt/models/VTLA-RDT/tacthru/cfg/gripper/synria_gloria.yaml \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --assumed-gripper-width-m 0.045 \
  --preview \
  --steps 5
```

不写 `--execute` 时：

- 不调用 `execute_waypoints()`；
- 不启动 Gloria gripper，因此不会触发它的初始化移动；
- 不发送运动命令，但 Realman SDK 会连接机械臂、选择运行模式并切换/更新 `new_tcp` tool frame；
- dry-run 的第 8 维使用 `--assumed-gripper-width-m`，应填写当前实际夹爪宽度；
- 每一步会记录 224×224 RGB 摘要、8D state、选中的完整 8D 动作、转换后的 Base target 和安全检查结果。

日志默认位于：

```text
logs/realman/YYYY.MM.DD/HH.MM.SS_tacthru_umi_v2_dryrun/
```

先检查日志中的 `controller_target_pos_after_adapter_safety_m`、旋转变化、执行窗口和夹爪宽度是否符合任务。这个字段已经包含 UMI→controller TCP 变换和桌面安全 `z_lift`。

## 7. 首次短窗口真机执行

执行前必须根据机械臂实际摆放填写 Base 坐标系 workspace 边界。边界约束的是经过 adapter 桌面安全修正后的最终 controller TCP，而不是模型原始 UMI 点。不要直接复制示例数值；从当前位姿、工作台尺寸和 dry-run 的 `controller_target_pos_after_adapter_safety_m` 确定边界。

```bash
bash scripts/run_tacthru_umi_v2_client.sh run \
  --server-url http://127.0.0.1:18081 \
  --instruction "Pull the tissue" \
  --realman-ip 192.168.1.18 \
  --realman-port 8080 \
  --preview \
  --steps 1 \
  --execute \
  --gripper-action-select threshold \
  --gripper-hold-closed-below-m 0.012 \
  --gripper-hold-closed-target-m 0.004 \
  --exec-start-step 2 \
  --exec-end-step 3 \
  --max-roundtrip-s 5 \
  --max-pos-speed 0.03 \
  --max-rot-speed 0.08 \
  --max-target-delta-m 0.06 \
  --max-target-rotation-rad 0.60 \
  --max-step-delta-m 0.02 \
  --max-step-rotation-rad 0.25 \
  --workspace-min-xyz <X_MIN> <Y_MIN> <Z_MIN> \
  --workspace-max-xyz <X_MAX> <Y_MAX> <Z_MAX>
```

`--execute` 后还必须按一次 Space。只有按下 Space 后，client 才会启用轨迹下发并启动 Gloria gripper；夹爪初始化本身可能移动夹爪。首次运行应保持急停可触及，并只执行一个短 chunk。

执行窗口是半开区间：`--exec-start-step 2 --exec-end-step 3` 实际只选择索引 `2`。夹爪对当前窗口的预测宽度做二值判断：最小值 `<12 mm` 时命令 `4 mm`，否则命令 Gloria 配置中的初始宽度 `45 mm`；等于阈值时打开，且不保留 episode 闭合锁存。正常退出时 client 会在关闭 Gloria、解除舵机力矩前再次等待 Space；先固定或取走夹持物，避免掉落。

推理延迟会按 30 Hz 时间语义换算为已过期步数。client 会跳过过期前缀，并在 50 步范围内平移短执行窗口；它不会盲目执行已经过时的第 2 步。

## 8. 内置安全检查

任何一项失败都会在调用机械臂前拒绝整个 chunk：

- 协议版本、request/session ID 和 health contract；
- checkpoint 训练配置、robot mapping、norm 路径与 SHA256 绑定；
- 固定使用训练时的 30 Hz、50 步 action chunk；
- action 必须严格为 `50×8` 且全部 finite；
- 四元数必须是可归一化的 `xyzw`；
- 夹爪宽度必须位于配置范围且单位为米；
- 相机/机械臂时间偏差、robot/gripper state 新鲜度、请求往返时间和 observation 后机械臂漂移；
- 所有选中目标经 UMI→TCP 和桌面 `z_lift` 后的 workspace；
- 当前位姿到目标的最大平移/旋转；
- 相邻 waypoint 的最大平移/旋转；
- 线速度和角速度限制会拉伸 waypoint 时间戳；
- 最大计划时长、严格递增时间戳和首 waypoint 过期检查；
- 非 stream 模式在下发后轮询验证最终 TCP 位姿和夹爪反馈；
- 预览窗口中的 q/Esc 会请求软件停止轨迹，但不等同于物理急停；
- server 使用单飞推理锁，避免多个请求同时访问同一个 stateful policy。

这些检查不能替代 Realman 急停、工作台碰撞评估和人工看护。当前链路尚没有完整 IK 可达性、关节限位、自碰撞/场景碰撞预规划；Realman SDK 的内部错误码也没有通过 TacThru ring buffer 暴露给 client。当前 TacThru adapter 会做指尖桌面高度修正，但不能把它当作完整碰撞规划器，因此只建议有人看护的渐进式真机测试，不建议无人值守运行。

如果 TacThru 不位于项目的 sibling 目录，除了 CLI 的 `--tacthru-repo`，启动 wrapper 前还需设置 `TACTHRU_REPO` 或 `REALMAN_PYTHON`，以便脚本先找到 Python 3.11 Realman 环境。
