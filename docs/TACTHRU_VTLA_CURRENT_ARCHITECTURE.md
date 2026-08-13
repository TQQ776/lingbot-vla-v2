# 当前 LingBot-VLA 2.0 + TacThru VTLA 架构

![当前 LingBot-VLA 2.0 + TacThru VTLA 架构](./TACTHRU_VTLA_CURRENT_ARCHITECTURE.svg)

这张图对应当前 `tacthru_umi_vtla_rgb_marker.yaml` 和已训练的
`insert_ethernet_vtla_regional32_contact_merged160_v1` 架构，而不是原始
LingBot-VLA 2.0 论文图的泛化示意。

## 核心结论

1. 腕部 RGB 和 TacThru RGB 共用冻结的 Qwen3-VL ViT。
2. Marker 不经过 ViT。它走 Regional Marker Encoder：每帧 48 个点经过
   Point MLP 和 2x2 区域 mean/max pooling，得到 4 个区域 token；8 帧一共
   32 个 marker token。
3. 场景、语言、触觉 RGB、marker 和 learned query 共同组成 Qwen3-VL
   prefix，都会经过 36 层 Understanding stream 并形成每层 K/V context。
4. Action Expert 不是简单串联在完整 VLM 后面。每一层都会把 prefix 和
   action suffix 的 Q/K/V 放进联合注意力；注意力输出再进入各自的 FFN。
   Prefix 分支使用冻结的 Qwen3-VL Dense FFN，Action 分支使用可训练 MoE FFN。
5. 推理时 prefix 只计算一次并缓存 K/V。之后 10 个 Flow Matching Euler
   步重复计算 state、当前 noisy action 和 time 构成的 action suffix。

## 当前张量与配置

| 模块 | 当前值 |
|---|---|
| 场景相机 | `observation.images.camera_wrist_left`，当前帧 |
| 触觉 RGB | `observation.images.tactile_left`，1 个传感器，当前帧 |
| Marker 输入 | `[B,1,8,48,2]`，最后一维是 `(dx,dy)` |
| Marker token | `8 帧 x 4 区域 = 32` 个/传感器 |
| Marker 位置编码 | temporal sincos + spatial sincos |
| Contact gate | 全局 active-count hysteresis，作用目标为 `marker_only` |
| Understanding stream | Qwen3-VL，36 层，hidden size 2560，冻结 |
| Action stream | 36 层，hidden size 768，所有层使用 token MoE |
| MoE | 32 routed experts，Top-4，另有 shared expert，sigmoid router |
| Robot state | 物理 8D，pad 到统一 55D |
| 模型动作 | 50 steps x 55D |
| 物理动作 | EEF pose 7D + gripper 1D |
| Flow Matching | 10 个 Euler 去噪步 |
| 真机默认执行窗口 | `[2,8)`，即每个 chunk 执行第 2 到第 7 步 |

## Gate 的准确含义

当前 gate 的 `target` 是 `marker_only`，因此：

- gate 关闭时，32 个 marker token 全部被 attention mask 屏蔽；
- TacThru RGB token 不会被这个 gate 屏蔽；
- 一个全局 gate 状态控制整段 8 帧 marker token，而不是分别控制四个区域；
- 配置虽然保留 `regional_soft_gate: true`，但当前
  `global_active_count_hysteresis_soft_region` 前向路径不会乘区域 soft gate。

训练时，gate 状态按 episode 顺序预计算，避免跨 episode 污染 hysteresis
状态。真机推理时，`TacThruSource` 按在线帧序列维护同一状态机。

## 训练与推理的边界

- Future RGB/depth 是辅助训练目标，不是真机推理时可提供的未来输入。
- Current/Future query 是 learned query token；future query 对 action suffix
  使用 attention block，避免未来监督信息泄漏给动作预测。
- 图中粉色虚线表示只在训练期存在的 target 或 loss；实线表示推理数据流。

## 代码依据

- VTLA prefix 和 token 拼接：`lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
- Marker encoder 和 gate mask：`lingbotvla/models/vla/lingbot_vla/tactile_vtla.py`
- 双流联合注意力和 Action MoE：`lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
- 在线/离线 gate 状态机：`lingbotvla/tactile_contact.py`
- 当前训练配置：`configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml`

可编辑图源是 `TACTHRU_VTLA_CURRENT_ARCHITECTURE.dot`。重新生成图像：

```bash
dot -Tsvg docs/TACTHRU_VTLA_CURRENT_ARCHITECTURE.dot \
  -o docs/TACTHRU_VTLA_CURRENT_ARCHITECTURE.svg

dot -Tpng -Gdpi=180 docs/TACTHRU_VTLA_CURRENT_ARCHITECTURE.dot \
  -o docs/TACTHRU_VTLA_CURRENT_ARCHITECTURE.png
```
