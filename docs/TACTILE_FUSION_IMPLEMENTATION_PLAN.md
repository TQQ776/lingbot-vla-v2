# LingBot VLA V2 触觉融合详细修改计划

> 状态：设计计划，尚未实施代码修改
> 目标任务：Realman + Gloria 执行 Insert Ethernet Cable
> 修改范围：只修改 `lingbot-vla-v2`；TacThru 项目只作为只读数据和算法参考
> 核心约束：默认行为保持当前 wrist-only 基线，所有新功能可配置、可消融、可回退

> 基线警告：当前 `tacthru-umi-v2` 工作树包含大量尚未提交的可运行 V2 适配代码，`origin/main` 不能作为“触觉修改前版本”。实施任何触觉代码前，必须先按第 15 节建立当前 wrist-only 定向快照。

## 1. 目标与验收原则

本计划的最终目标不是简单增加一张触觉图片，而是让 LingBot V2 同时利用：

1. 腕部 RGB：全局定位、网口和插头姿态。
2. 触觉 RGB：接触轮廓、接触面积、局部形状。
3. `48 x 2` marker flow：切向形变、滑移、偏斜和回弹。
4. 最近若干帧触觉历史：接触变化速度和任务阶段。

动作输出保持不变：

```text
[50, 8] = xyz(3) + quaternion_xyzw(4) + gripper_width(1)
```

必须同时满足以下工程要求：

- 训练时可以独立选择是否使用触觉 RGB。
- 训练时可以独立选择是否使用 marker。
- 可以训练 wrist-only、RGB-only tactile、marker-only、RGB+marker 四类模型。
- 同一个已训练 checkpoint 可以在评估时强制屏蔽 RGB 或 marker，测量模型实际依赖程度。
- 可以配置训练期间的随机模态 dropout，但它必须与强制消融区分。
- 触觉传感器丢帧时不能导致模型或服务端崩溃。
- `train.tactile_rgb_enabled=false` 且 `train.tactile_marker_enabled=false` 时，当前训练和推理行为应保持不变。
- 不覆盖旧数据集、旧配置、旧 checkpoint 和旧真机脚本。
- 回退不依赖删除文件或 `git reset --hard`。

## 2. 当前基线与已知事实

当前 Insert Ethernet 源 Zarr 数据包含：

```text
camera0_rgb       : [N, 224, 224, 3]
tacthru_l_rgb     : [N, 224, 224, 3]
tacthru_l_marker  : [N, 48, 2]
```

其中 `tacthru_l_marker` 是归一化 marker flow：

```text
marker_flow = current_marker - reference_marker
```

它不是牛顿力、法向压力或深度图。

当前 LingBot V2 链路仍是：

```text
state8 + wrist_rgb -> server -> [50, 8] action chunk
```

现状限制：

- `tools/convert_tacthru_zarr_to_lerobot_v2.py` 当前没有导出触觉 RGB 和 marker。
- `configs/vla/tacthru_umi/tacthru_umi.yaml` 当前只配置腕部图像。
- `configs/robot_configs/tacthru_umi_v2.yaml` 当前没有触觉字段。
- `deploy/tacthru_umi_v2/protocol.py` 当前协议不启用触觉。
- `deploy/tacthru_umi_v2/http_server.py` 当前请求只接受现有图像槽位。
- 当前 wrist-only checkpoint 没有学习过触觉，不能在推理时临时追加触觉输入。
- V2 底层能够处理多路配置图像，但 Depth/DINO 教师只读取第一路图像，因此腕部 RGB 必须始终排在第一位。

## 3. 总体架构

最终推荐采用专用触觉编码器和门控 token 融合：

```text
wrist RGB
    -> current Qwen3-VL image path -------------------+
                                                       |
recent K tactile RGB frames                            |
    -> TactileRGBEncoder -> TemporalEncoder -> tokens -+-> GatedTactileFusion
                                                       |          |
recent K marker-flow frames                            |          v
    -> MarkerEncoder -> TemporalEncoder -> tokens -----+   V2 Action Expert
                                                                  |
robot state + instruction ----------------------------------------+
                                                                  |
                                                                  v
                                                          [50, 8] actions
```

同时保留本地高频触觉安全分支：

```text
marker flow
    +-> send to V2 for learned action prediction
    +-> local 20 Hz contact/slip guard for stop, slow, retract and release gating
```

远程 V2 负责较长时间尺度动作，本地触觉分支负责毫秒级安全响应。两者不能互相替代。

## 4. 配置设计

### 4.1 配置语义必须分层

当前 `train_lingbotvla.py` 会把 model 与 train 配置合并进 `LingbotVLAV2Config`，部署加载器也会从 checkpoint 的 `lingbotvla_cli.yaml` 恢复这部分配置。因此用户可见的唯一架构开关放在 `train` 下最容易保证训练、checkpoint 和部署使用同一真源：

```text
train.tactile_rgb_enabled
train.tactile_marker_enabled
```

仍然必须区分三种行为：

1. `*_enabled`：正式架构消融；决定是否读取数据、创建 encoder 和保存参数。
2. `*_dropout_prob`：训练期间按完整样本随机丢弃已启用模态。
3. `force_mask_*`：评估或 dry-run 时对同一 checkpoint 强制遮蔽已存在分支。

数据加载是否读取触觉由两个架构开关派生，不再增加另一套容易冲突的 `load_*` 用户开关。静态数据键名仍放在 `data` 配置中。

### 4.2 建议新增的配置结构

以下键名是计划中的统一接口，实施时应由配置 dataclass 或配置校验函数强制验证：

```yaml
data:
  tactile_rgb_key: observation.images.tactile_left
  tactile_marker_key: observation.tactile.marker_flow_left
  tactile_marker_valid_key: observation.tactile.marker_valid_left

train:
  tactile_rgb_enabled: false
  tactile_marker_enabled: false
  tactile_params:
    schema_version: 1
    history_steps: 4
    history_stride: 1
    max_timestamp_skew_s: 0.05
    rgb_backbone: dinov2_vits14
    rgb_backbone_path: null
    freeze_rgb_backbone: true
    rgb_tokens: 8
    marker_count: 48
    marker_dim: 2
    marker_tokens: 8
    marker_reference_mode: auto
    marker_include_velocity: true
    marker_normalization: image_size_xy
    fusion_type: gated_prefix
    gate_init: -4.0
    teacher_attention_to_tactile: false
    missing_policy: mask
    rgb_dropout_prob: 0.10
    marker_dropout_prob: 0.10
    all_tactile_dropout_prob: 0.05
    auxiliary_contact_loss_weight: 0.0
    auxiliary_slip_loss_weight: 0.0
    auxiliary_phase_loss_weight: 0.0

eval:
  force_mask_tactile_rgb: false
  force_mask_tactile_marker: false

deploy:
  allow_tactile_ablation: false
  allow_missing_tactile: false
  tactile_fallback: stop
  local_tactile_guard_enabled: true
```

其中 `data`、`train`、`eval` 应接入现有训练/评估 dataclass；上面的 `deploy` 是部署合同，不假设它会被 `lingbotvla_cli.yaml` 自动解析。实施时分别落到 server/client argparse 或环境变量，并由 `/health` 返回最终生效值。模型所需模态仍以 checkpoint 中保存的两个 train 开关为权威来源，客户端不得自行猜测。

训练 wrapper 同时提供简写：

```text
--tactile-mode none|rgb|marker|rgb-marker
```

它只负责映射到两个独立布尔值，底层仍允许直接覆盖：

```text
--train.tactile_rgb_enabled true|false
--train.tactile_marker_enabled true|false
```

### 4.3 配置校验规则

启动训练、评估或服务端时必须检查：

- 两个 `train.*_enabled=false` 时，数据 batch 和模型结构必须与当前 wrist-only 路径一致。
- `train.tactile_rgb_enabled=true` 时必须存在配置的 tactile RGB key。
- `train.tactile_marker_enabled=true` 时必须存在 marker key、正确 shape 和 normalization metadata。
- `force_mask_*` 只能屏蔽已经存在于 checkpoint 中的分支。
- `dropout_prob` 范围必须是 `[0, 1)`。
- temporal horizon 必须至少为 1。
- marker 数量、归一化方法和 checkpoint metadata 必须匹配。
- `marker_reference_mode=auto` 只有在数据真实包含 reference XY 时才启用；当前旧 InsertEthernet 数据不得伪造 reference。
- 训练配置、数据 contract 和 checkpoint contract 不一致时应明确报错，不能静默降级。

## 5. 消融实验设计

### 5.1 独立训练的架构消融

所有实验必须使用相同的数据划分、随机种子、训练步数、batch size 和评估流程。

| ID | 触觉 RGB | Marker | 时间窗口 | 融合方式 | 目的 |
|---|---:|---:|---:|---|---|
| A0 | 否 | 否 | 1 | none | 当前 wrist-only 基线 |
| A1 | 是 | 否 | 1 | multi-image 或 RGB adapter | 测量触觉 RGB 单独贡献 |
| A2 | 否 | 是 | 1 | marker token adapter | 测量 marker 单独贡献 |
| A3 | 是 | 是 | 1 | gated token adapter | 测量单帧双触觉贡献 |
| A4 | 是 | 是 | 4 | gated token adapter | 测量时序触觉贡献，推荐主模型 |
| A5 | 是 | 是 | 4 | gated token adapter + modality dropout | 测量缺失模态鲁棒性 |
| A6 | 是 | 是 | 4 | state concat marker | 低延迟 marker 对照 |
| A7 | 是 | 是 | 4 | marker-render multi-image | 图像化 marker 对照 |

其中 A7 仅作为快速对照，不作为最终首选。把 marker 画成 RGB 图会损失原始浮点精度，并把结构化物理量交给普通图像编码器重新学习。

### 5.2 同一 checkpoint 的强制遮蔽实验

对 A4 或 A5 的同一个 checkpoint，运行以下四组：

| ID | `force_mask_tactile_rgb` | `force_mask_tactile_marker` | 目的 |
|---|---:|---:|---|
| O0 | false | false | 完整触觉性能 |
| O1 | true | false | 只保留 marker，测量 RGB 依赖 |
| O2 | false | true | 只保留 RGB，测量 marker 依赖 |
| O3 | true | true | 同模型无触觉，测量总体触觉依赖 |

强制遮蔽不能简单输入一张黑图。正确实现应当：

- 保持固定 token shape，避免破坏 `torch.compile` 图。
- 将被遮蔽模态 token 置零或替换为 missing token。
- 将对应 attention/presence mask 设为 false。
- 将对应融合 gate 强制设为零。
- 在日志中记录实际生效的 mask。

### 5.3 随机模态 dropout

训练 dropout 必须按“整个样本、整个时间窗口”屏蔽，而不是随机屏蔽窗口中的某一帧：

```text
rgb_dropout_prob    推荐从 0.10 开始
marker_dropout_prob 推荐从 0.10 开始
```

待基线稳定后再测试更高概率。所有 dropout 值必须写入 checkpoint metadata 和实验日志。

## 6. 数据转换和数据合同

### 6.1 新数据集必须版本化

不得覆盖现有 wrist-only LeRobot 数据集。建议新目录命名：

```text
data/lerobot/insert_ethernet_v2_tactile_v1
```

只转换一份同时包含 wrist、tactile RGB 和 marker 的 superset 数据集。A0/A1/A2/A3 等消融通过配置选择字段，不为每种组合重新转换数据，避免转换差异破坏公平比较。只有明确关闭触觉时才允许继续使用旧 wrist-only 数据集；配置要求触觉而字段缺失时必须直接报错。

建议字段：

```text
observation.images.camera_wrist_left     uint8 [H, W, 3]
observation.images.tactile_left          uint8 [H, W, 3]
observation.tactile.marker_flow_left     float32 [48, 2]
observation.tactile.marker_valid_left    bool [48]
observation.tactile.timestamp            float64 []
observation.state                        float32 [8]
action                                   float32 [8]
```

对于未来新录数据，建议额外保存：

```text
observation.tactile.marker_reference_xy  float32 [48, 2]
observation.tactile.rgb_timestamp        float64 []
observation.robot_timestamp              float64 []
```

当前旧 Zarr 若没有 absolute marker/reference，只保留已有 flow，并在 metadata 中明确来源和归一化方法，不能伪造原始坐标。

### 6.2 Marker 归一化必须统一

当前 ML48 传感器实际分辨率为 `640 x 480`，正确归一化应为：

```text
flow_x = delta_x / 640 * 2
flow_y = delta_y / 480 * 2
```

不能复制真机旧路径中的统一 `/400`。LingBot V2 的转换器和在线客户端应共用同一个纯函数，例如：

```python
normalize_marker_flow(delta_xy, image_width, image_height)
```

该函数需要单元测试覆盖 X/Y 两个尺度，并将 `image_width`、`image_height` 和 normalization version 写入数据集 metadata。

### 6.3 时间窗口构建

不要在磁盘中重复保存 4 份相同数据。Dataset 在读取时基于 episode 边界构造：

```text
[t-3, t-2, t-1, t]
```

要求：

- 不能跨 episode 取帧。
- episode 开头可以 repeat-first，但必须同时产生 temporal mask。
- marker velocity 使用真实时间戳计算，不能假设所有帧间隔完全相同。
- RGB、marker、robot state 的时间偏差必须写入 QA 报告。

### 6.4 数据质量检查

转换完成后至少输出：

- episode 数和每条长度。
- 触觉 RGB 缺帧率。
- marker 有效点数量分布。
- marker flow 的 X/Y min、max、mean、std 和分位数。
- 无接触片段的 flow 漂移。
- wrist/tactile/robot timestamp skew 的 p50、p95、max。
- RGB 全黑、冻结帧和重复帧统计。
- train/validation/test episode 列表及其 hash。

## 7. 数据增强

现有对普通相机的统一增强不能直接套用全部触觉模态。

### 7.1 Wrist RGB

保留当前视觉增强策略，作为环境外观泛化来源。

### 7.2 Tactile RGB

只使用较弱增强：

- 小幅亮度和对比度变化。
- 轻微传感器噪声。
- 少量模糊或压缩噪声，用于模拟传输变化。

默认禁止：

- 水平或垂直翻转。
- 大角度旋转。
- 会移动接触位置语义的大幅随机裁剪。
- 强烈色相变化。

### 7.3 Marker

允许：

- 小幅坐标噪声。
- 少量 marker point dropout，并同步更新 valid mask。
- 轻微 reference drift 模拟，但必须有上限。

禁止直接改变 X/Y 符号或交换 X/Y 轴。

## 8. 模型模块设计

### 8.1 TactileRGBEncoder

建议新增独立模块，而不是长期复用普通相机增强和全部 Qwen image tokens：

```text
[B, K, 3, 224, 224]
  -> ViT-Small/DINOv2
  -> patch tokens
  -> temporal attention
  -> token resampler
  -> [B, 8, prefix_hidden_dim]
```

训练初期冻结视觉 encoder，只训练 projection、temporal layer 和 resampler；后续再决定是否解冻高层。

权重来源必须固定：优先复用项目已有 `timm`/DINOv2 依赖，`rgb_backbone_path` 指向本地权重，训练和部署阶段不得临时联网下载。readiness 检查需要验证文件存在、SHA256、模型名称和输出维度；若引入新依赖，应单独锁定版本并记录到环境说明。

### 8.2 MarkerEncoder

每个 marker 推荐输入：

```text
reference_x, reference_y,
flow_x, flow_y,
velocity_x, velocity_y,
valid_mask
```

处理路径：

```text
[B, K, 48, feature_dim]
  -> shared point MLP
  -> marker index/reference-position embedding
  -> temporal transformer
  -> point/token resampler
  -> [B, 8, prefix_hidden_dim]
```

如果旧数据没有 reference XY，第一版使用固定 marker index embedding + flow/velocity，不应伪造 reference。

### 8.3 GatedTactileFusion

本计划钉死采用 `gated_prefix`，不同时实现另一套 768D Action Expert cross-attention。RGB 和 marker token 先投影到 Qwen prefix hidden dimension（当前模型约 2560，实际值从 config 读取），再作为非 visual prefix token 插入。Action suffix 可以 attention 到这些 token；Depth/DINO teacher 的 visual/deepstack mask 必须排除它们。

融合模块应满足：

- tactile RGB 与 marker 各自有 presence mask。
- 两个分支各自有可观测 gate。
- gate 在模态缺失或强制屏蔽时严格为 0。
- 输出 hidden size 与 Qwen prefix hidden dimension 一致。
- 记录每层或最终 gate 均值，便于确认模型是否真正使用触觉。
- 固定最大 token 数，避免每种模态组合触发不可控的动态 shape。

### 8.4 辅助任务

动作损失仍是主损失。只有数据标签可靠时才启用辅助头：

```text
L_total = L_action
        + lambda_contact * L_contact
        + lambda_slip * L_slip
        + lambda_phase * L_phase
```

可选标签：

- contact / no-contact。
- stable / slipping。
- approach / first-contact / insertion / seated / release。

没有标定力传感器时，不得把 marker flow 监督命名为真实 force loss。

### 8.5 旧 checkpoint 兼容

兼容规则必须写入自动测试：

1. 旧 model config 没有 tactile 配置时，解释为 `tactile.enabled=false`。
2. 关闭触觉时不实例化新增 adapter，原模块名称和参数 shape 保持不变。
3. 旧 wrist-only checkpoint 使用严格加载。
4. 从旧 checkpoint 初始化触觉模型时，只允许明确白名单中的新触觉参数缺失：

   ```text
   tactile_rgb_encoder.*
   tactile_marker_encoder.*
   tactile_fusion.*
   ```

5. 禁止对整个模型无条件使用 `strict=False`；任何非 tactile 前缀的 missing/unexpected key 都必须中止。
6. 启动日志必须列出新初始化参数、参数量和 checkpoint contract。

触觉 checkpoint metadata 至少保存：

```text
contract_version
input_schema_version
tactile_adapter_version
enabled_modalities
temporal_horizon
dataset_manifest_sha256
episode_split_sha256
marker_normalization_sha256
training_config_sha256
norm_stats_sha256
git_commit
```

## 9. 训练策略

### 9.1 初始化

- 从当前 Insert Ethernet wrist-only checkpoint 初始化原 V2 参数。
- 新增触觉模块使用独立初始化。
- checkpoint loader 必须明确列出 expected missing keys 和 unexpected keys。
- 不允许用 `strict=False` 后完全静默忽略不匹配。

### 9.2 分阶段训练

建议采用：

1. 阶段 T1：冻结原 V2 主干，训练触觉编码器、projection、fusion gate。
2. 阶段 T2：解冻 Action Expert 和上层融合模块，联合训练。
3. 阶段 T3：使用较小学习率进行端到端微调。

每一阶段都生成独立 checkpoint 子目录和完整配置快照。

### 9.3 数据划分

至少保留约 20 个完整 episode 作为测试集，不能随机按帧划分。测试集应覆盖：

- 不同插头初始偏差。
- 不同网口位置和姿态。
- 接触后顺利插入。
- 偏斜、卡住和恢复。
- 插到底和夹爪释放。

## 10. 部署协议

### 10.1 协议版本

新增协议版本，不直接改变现有 v1 请求语义：

```text
v1: state8 + wrist_rgb
v2: state8 + wrist_rgb + optional tactile_rgb + optional marker payload
```

建议 v2 请求包含：

```text
schema_version
request_id
session_id
state8
wrist_rgb
wrist_timestamp
tactile_rgb_history [K, H, W, C] (optional)
tactile_rgb_timestamps [K] (optional)
tactile_rgb_history_mask [K] (optional)
marker_flow [K, 48, 2] (optional)
marker_valid_mask [K, 48] (optional)
marker_timestamps [K] (optional)
marker_history_mask [K] (optional)
modality_presence
```

服务端 `/health` 必须返回：

- checkpoint 支持的 modality。
- temporal horizon。
- marker normalization version。
- 是否允许缺失触觉。
- contract SHA256。

### 10.2 客户端开关

建议提供：

```text
--tactile-mode off|rgb|marker|rgb-marker
--tactile-temporal-horizon K
--allow-missing-tactile
--force-mask-tactile-rgb
--force-mask-marker
--local-tactile-guard / --no-local-tactile-guard
```

默认必须为：

```text
--tactile-mode off
```

真机 `--execute` 模式默认设置 `allow_tactile_ablation=false`。强制遮蔽只允许 synthetic、offline evaluation 和 dry-run；不能在正在执行的 episode 中热切换模态、checkpoint、协议或 temporal horizon。

### 10.3 本地触觉安全分支

本地 guard 只能增加限制，不能绕过现有：

- workspace bounds。
- max position/rotation speed。
- observation drift。
- sensor skew。
- command verification。
- 急停和人工确认。

本地 guard 需要记录触发原因，例如：

```text
contact_magnitude_high
slip_detected
marker_tracking_lost
tactile_stale
possible_jam
release_confirmed
```

## 11. 日志和可观测性

每个训练 batch 或合理采样间隔记录：

- RGB/marker 是否存在。
- 随机 dropout mask。
- 强制消融 mask。
- marker magnitude、velocity 和 valid count。
- fusion gate 均值。
- 每个触觉分支 token norm。
- 主动作 loss 和辅助 loss。

每次在线请求记录：

- wrist/tactile/state timestamp 和 skew。
- tactile encode、marker encode、fusion、model、HTTP 各阶段耗时。
- 实际使用的 modality。
- 缺失或过期触觉处理结果。
- contract hash 和 checkpoint 路径。

日志中不得只写 `tactile=true`，必须明确是 RGB、marker 还是两者。

## 12. 代码和文件修改清单

下面是计划修改或新增的主要位置。实施前需再次核对当前分支的真实类名和配置注册方式。

### 12.1 数据和配置

- 修改 `tools/convert_tacthru_zarr_to_lerobot_v2.py`
  - 新增版本化触觉输出。
  - 保留旧 wrist-only 默认行为。
  - 增加 marker normalization metadata 和 QA。
  - 新增显式参数 `--include-tactile-rgb`、`--include-tactile-marker`、`--tactile-side left`；未传参数时继续产生旧 schema。
- 新增公共触觉包，供 converter、dataset 和真机 client 共用，避免离线/在线重复实现：
  - `lingbotvla/tactile/schema.py`
  - `lingbotvla/tactile/transforms.py`
  - 统一 RGB/BGR、marker shape、history mask，以及 `dx/width*2`、`dy/height*2`。
- 新增 `configs/vla/tacthru_umi/tacthru_umi_tactile.yaml`，保留 `configs/vla/tacthru_umi/tacthru_umi.yaml` 不变。
- 新增 `configs/robot_configs/tacthru_umi_v2_tactile.yaml`，保留 `configs/robot_configs/tacthru_umi_v2.yaml` 不变。
- 如果新配置使用新的 `data_name`，同步扩展 dataset 的 `is_tacthru_umi_v2()` 判定，不能复制一套不一致的状态/动作映射。
- 修改 `lingbotvla/data/vla_data/utils.py`
  - 注册触觉字段、presence mask 和 temporal metadata。
- 修改 `lingbotvla/data/vla_data/base_dataset.py`
  - wrist 保持当前采样；tactile RGB 和 marker 使用独立历史时间偏移。
  - episode 开头 repeat-first + history mask，禁止跨 episode。
  - 对禁用模态不解码视频、不读取 marker，避免无效 I/O。
- 修改 `lingbotvla/data/vla_data/transform.py`
  - 分离 wrist/tactile/marker 增强。
  - 增加训练 dropout 和强制 mask。
- 修改 `lingbotvla/data/dataset.py` 和 `lingbotvla/data/multi_vla_dataset.py`
  - 透传两个触觉开关、history 和 mask；两个开关都关闭时保持旧构造路径。
- 修改 `scripts/compute_norm_stats.py`
  - 在 superset norm 中增加 marker stats；marker 禁用时额外统计项不参与模型。
- 修改 `scripts/check_tacthru_umi_v2_ready.py`
  - 根据配置动态检查必需 key、manifest、shape 和 normalization，而不是全局放宽。
  - 检查 `rgb_backbone_path`、权重 SHA256、依赖版本和 encoder 输出维度。

### 12.2 模型

- 修改 `tasks/vla/train_lingbotvla.py`
  - 在训练参数中注册两个独立开关和 `tactile_params`。
  - 将 tactile RGB、marker、history mask 和 presence mask 传给模型。
- 修改 `lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py`
  - 两个开关默认 false，确保旧 YAML 和旧 checkpoint 自动进入 wrist-only 路径。
- 新增独立 tactile 模块目录或文件，例如：
  - `tactile_rgb_encoder.py`
  - `marker_encoder.py`
  - `tactile_fusion.py`
- 修改 `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
  - 接收 tactile batch。
  - 创建固定形状 tactile tokens 和 masks。
  - 在 Action Expert 前接入 gated fusion。
  - 只有对应开关为 true 时才实例化 encoder；两个开关都为 false 时 module graph/state dict 与旧模型一致。
  - 触觉 token 不加入 Qwen visual deepstack mask；Depth/DINO teacher 默认不能通过辅助监督读取触觉 token。
- 核对 `lingbotvla/models/vla/vision_models/module_utils.py`
  - wrist RGB 保持第一视图。
  - Depth/DINO teacher 不误读 tactile RGB。
- 修改实际 checkpoint loader 所在的 `lingbotvla/models/module_utils.py` 或对应 loader
  - 从 wrist checkpoint 初始化触觉模型时只放行白名单触觉前缀。
  - 最终触觉 checkpoint 部署仍严格加载。
- 更新 FSDP/no-split/compile 策略，使新增 temporal transformer 和 resampler 能完成 eager/compile smoke。

### 12.3 训练脚本

- 新增触觉训练 launcher，保留现有训练脚本不变。
- 新增四个基础 preset：
  - `wrist_only`
  - `tactile_rgb_only`
  - `marker_only`
  - `tactile_rgb_marker`
- 训练输出目录中保存最终展开后的 YAML、git commit、dataset hash 和 contract hash。
- 四种正式消融使用独立 output 目录，禁止跨实验 resume。

### 12.4 部署

- 修改 `deploy/lingbot_vla_v2_policy.py`
  - 从 checkpoint 读取触觉开关并准备/堆叠可选触觉 tensor。
- 修改 `deploy/tacthru_umi_v2/protocol.py`
  - 增加 v2 schema，同时保留 v1。
- 修改 `deploy/tacthru_umi_v2/http_server.py`
  - 根据 checkpoint contract 解析可选触觉。
- 修改 `deploy/tacthru_umi_v2/realman_client.py`
  - 同步采集触觉、构造 marker history、发送 presence mask。
  - 增加本地 guard 和分阶段延迟日志。
- 新增触觉服务端启动脚本和真机脚本。
- 推荐新增 `scripts/run_tacthru_umi_v2_tactile_server.sh`、`scripts/run_tacthru_umi_v2_tactile_client.sh` 和 `scripts/real_insert_ethernet_tactile.sh`。
- 保留 `scripts/real_insert_ethernet.sh` 原样作为 wrist-only 回退入口。

## 13. 测试计划

建议新增：

```text
tests/test_tacthru_umi_v2_tactile_transforms.py
tests/test_tacthru_umi_v2_tactile_dataset.py
tests/test_tacthru_umi_v2_tactile_encoder.py
tests/test_tacthru_umi_v2_tactile_ablation.py
tests/test_tacthru_umi_v2_tactile_protocol.py
tests/test_tacthru_umi_v2_tactile_checkpoint_compat.py
```

同时扩展现有 converter 测试：旧参数仍断言 wrist-only converter 行为；只有显式 `--include-tactile-*` 才进入新的触觉 schema。

### 13.1 单元测试

- 四种 RGB/marker 配置组合能否通过配置校验。
- `force_mask` 是否真正清除对应分支影响。
- 模态 dropout 是否按样本和完整时间窗口执行。
- marker X/Y normalization 是否使用不同宽高。
- temporal window 是否不会跨 episode。
- 缺失 tactile RGB、部分 marker 无效、全部 marker 无效的处理。
- v1 请求是否仍能被 v1 服务端处理。
- v2 请求的 contract mismatch 是否明确报错。

### 13.2 回归测试

在两个触觉架构开关都为 false 时：

- 加载当前 wrist-only checkpoint 不应出现新 missing keys。
- synthetic inference 的 action shape 和 metadata 保持不变。
- 相同输入输出应在对应 dtype 的数值容差内与基线一致。
- `torch.compile` 单步测试保持通过。
- 原 `real_insert_ethernet.sh` 不需要增加任何触觉参数。

### 13.3 训练测试

依次执行：

1. 每个配置一个 batch forward。
2. 每个配置一个 batch forward + backward。
3. 每个配置 `max_steps=1` compile smoke。
4. 小数据集 overfit，确认 loss 能下降。
5. 完整 A0-A5 训练。

### 13.4 部署测试

顺序不得跳过：

1. synthetic payload。
2. 本地录制数据回放。
3. 真机传感器连接但不执行动作。
4. tactile dry-run。
5. 低速、单 chunk、人工确认执行。
6. 完整任务评估。

## 14. 评估指标

不能只比较训练 loss。至少记录：

- 完整插入并锁住成功率。
- 插入后正确释放夹爪成功率。
- 首次接触到锁住的时间。
- 卡住次数和自动恢复成功率。
- 误释放率。
- workspace/speed/sensor safety 拦截次数。
- 每轮 V2 请求的 p50、p95、max 延迟。
- 触觉编码额外耗时。
- RGB/marker gate 在任务各阶段的变化。

同一真实场景应进行足够重复试验，并记录失败类型，不能只报告最好的一次。

## 15. 完整回退设计

### 15.1 Git 回退

2026-07-23 的只读审计发现：

- 当前分支是 `tacthru-umi-v2`。
- 当前 `HEAD` 仍对应官方 `origin/main` 的 `69729b4`。
- 当前可运行的 TacThru UMI V2 训练、部署和真机控制包含大量未提交/未跟踪修改，约 28 项。
- `logs/` 约 24 MB，当前没有被 `.gitignore` 排除。

因此不能把 `origin/main` 或 `69729b4` 当作 wrist-only 回退点；这样会同时丢失目前已经可用的插网线 V2 链路。

正式修改前必须：

1. 确认当前工作树中哪些改动属于用户，不能覆盖。
2. 将 `/logs/` 和 `/.rollback/` 加入 `.gitignore`，但不删除已有日志。
3. 只显式暂存确认属于 wrist-only 基线的代码、配置、测试和文档；禁止直接 `git add .`。
4. 检查 staged diff，并运行当前测试、synthetic 推理和一次 dry-run。
5. 创建当前可运行 wrist-only 基线提交。
6. 创建标签，例如 `tacthru-umi-v2-wrist-only-20260723`，同时建立 backup branch。
7. 在独立分支或独立 worktree 开发触觉，例如 `feature/tactile-fusion-v1`。
8. 每个阶段单独提交，不把数据、模型、协议和真机控制混在一个提交中。

推荐使用独立 worktree，使现有目录一直保留为可运行的腕部版本：

```bash
git worktree add ../lingbot-vla-v2-tactile \
  -b feature/tactile-fusion-v1 \
  tacthru-umi-v2-wrist-only-20260723
```

在基线提交完成前，还应增加一个定向快照工具 `scripts/create_tactile_fusion_snapshot.sh`。它只备份本计划将修改的文件，并在 `.rollback/tactile_fusion_pre_<timestamp>/` 保存：

- 文件 SHA256。
- `git status --short`。
- `git diff --binary`。
- 修改前不存在的新文件清单。
- 生成的 `restore.sh`。

`restore.sh` 默认必须先比较当前 SHA，避免覆盖触觉开发期间之外的人工修改；默认支持 `--dry-run`，只有显式 `--force` 才允许覆盖冲突文件。这个定向快照是当前脏工作树下的第二层保障，不能替代基线提交和独立分支。

建议提交边界：

```text
commit 1: add dormant tactile config and contracts, defaults off
commit 2: add tactile dataset conversion and QA
commit 3: add tactile RGB baseline
commit 4: add marker encoder and masks
commit 5: add gated temporal fusion
commit 6: add protocol v2 and offline server/client
commit 7: add local tactile guard and real-hardware path
```

出现问题时优先使用 `git revert` 对应提交或切回基线分支，不使用破坏工作树的强制重置。

### 15.2 配置回退

两个触觉架构开关为 false，用于保证新代码仍能训练/加载旧 wrist-only checkpoint：

```yaml
train:
  tactile_rgb_enabled: false
  tactile_marker_enabled: false
deploy:
  allow_tactile_ablation: false
  local_tactile_guard_enabled: false
```

它不能把已经训练好的 tactile checkpoint 原地变回旧模型；关闭分支后该 checkpoint 会出现严格加载不匹配。`force_mask` 也只是鲁棒性/敏感性测试，不等价于真正回退。

真正的运行时回退必须成套切换到：

```text
旧 wrist-only checkpoint
+ 旧 norm stats
+ protocol v1 server/client
+ 原 scripts/real_insert_ethernet.sh
```

### 15.3 产物回退

- 旧 LeRobot 数据集保持原路径。
- 新触觉数据集使用新路径和 schema version。
- 旧 checkpoint 保持原路径，只读使用。
- 新 checkpoint 使用独立 output directory。
- 旧服务端和客户端协议保持 v1。
- 验证期间可使用不同端口运行 tactile v2 服务端。
- 旧 `scripts/real_insert_ethernet.sh` 不替换；新增 tactile 脚本。

### 15.4 运行时回退

运行时回退顺序：

1. 停止 tactile v2 客户端。
2. 启动旧 wrist-only checkpoint 和 v1 server。
3. 使用原 `scripts/real_insert_ethernet.sh`。
4. 确认 health contract、checkpoint 路径和 action shape。
5. 重新执行 dry-run 后才能恢复真机动作。

## 16. 分阶段实施顺序与门槛

### P0：冻结基线

工作：

- 保存当前配置、checkpoint、synthetic 输出和最近 dry-run 日志。
- 记录当前 p50/p95 推理延迟。
- 建立基线 git 标记和独立分支。

完成门槛：能够从保存的信息重现当前 wrist-only synthetic 和 dry-run。

### P1：加入关闭状态的配置骨架

工作：

- 加入配置 dataclass、validation 和 mask 数据结构。
- 所有默认值关闭触觉。

完成门槛：关闭触觉时训练、compile smoke、server 和 synthetic 输出通过回归。

### P2：数据转换与 QA

工作：

- 生成版本化触觉 LeRobot 数据集。
- 修复 marker normalization。
- 输出时间同步和 marker 质量报告。

完成门槛：所有字段、shape、episode 边界和统计检查通过。

### P3：触觉 RGB 基线

工作：

- 先利用现有多图能力训练 A1。
- wrist 始终为第一视图。

完成门槛：A1 能训练、离线推理，且与 A0 使用相同测试协议。

### P4：Marker 单分支

工作：

- 实现 marker encoder、valid mask 和 A2。
- 实现同 checkpoint `force_mask_tactile_marker`。

完成门槛：A2 可训练；marker 全缺失时不崩溃；mask 测试通过。

### P5：时序门控融合

工作：

- 实现 K 帧窗口、两个 resampler 和 gated fusion。
- 训练 A3-A5。

完成门槛：所有 O0-O3 强制消融可重复，日志能证明实际 mask 和 gate 状态。

### P6：协议 v2 与离线部署

工作：

- 增加版本化请求、health contract、传感器时间戳和分阶段延迟日志。
- 保留 v1 服务。

完成门槛：旧 v1 链路不受影响；v2 可用录制数据完成端到端回放。

### P7：本地 guard 与真机验证

工作：

- 本地高频触觉 guard。
- dry-run、低速执行和完整任务评估。

完成门槛：触觉断开、过期、marker 丢失时均进入安全状态；回退 wrist-only 流程经过验证。

## 17. 推荐的第一轮实验

第一轮不直接进行大规模结构搜索，执行：

1. A0：当前 wrist-only 基线。
2. A1：wrist + tactile RGB，单帧。
3. A2：wrist + marker token，单帧。
4. A3：wrist + tactile RGB + marker，单帧。
5. A4：A3 改为 4 帧时序。
6. 对 A4 执行 O0-O3 同 checkpoint 强制遮蔽。

如果 A4 在完整插入、卡住恢复和释放三个指标上稳定优于 A0，再启用辅助任务和更复杂的训练阶段。若触觉只提升训练 loss、没有提升 held-out episode 或真机成功率，应先排查同步、归一化、数据覆盖和模型是否真正打开 fusion gate，而不是继续扩大模型。

## 18. 实施过程中的禁止事项

- 不直接覆盖当前 wrist-only 配置和脚本。
- 不覆盖原 LeRobot 数据集。
- 不用黑图冒充严格的模态遮蔽。
- 不把 marker flow 称为真实力。
- 不把随机帧划分当作独立测试集。
- 不因触觉存在而放宽现有机械臂 workspace 和速度限制。
- 不在未完成 synthetic、回放和 dry-run 前启用真机执行。
- 不在没有配置快照和 contract hash 的情况下比较消融结果。
