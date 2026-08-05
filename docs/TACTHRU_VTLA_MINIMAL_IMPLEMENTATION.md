# LingBot-VLA 2.0 TacThru 最小 VTLA 实现

本文记录 `LingBot-VLA TacThru VTLA 改造任务.md` 的实际代码落地。实现只扩展
action expert 可见的 prefix 条件，不修改动作空间、动作头、Flow Matching 数学、
Euler 采样、MoE 路由或机器人控制器。

## 1. 仓库结构分析

- 图像数据入口：`lingbotvla/data/vla_data/transform.py::prepare_images`。Qwen3-VL
  image processor 输出 `pixel_values` 和 `image_grid_thw`。
- 视觉编码器入口：
  `QwenvlWithExpertV2Model.get_image_features()`，实际调用
  `self.qwenvl.visual(...)`，输出末层 image token 和 deepstack token。
- 语言 token：`prepare_language()` 生成 ID/mask，
  `embed_language_tokens()` 调用 Qwen3-VL text embedding。
- 视觉/语言 prefix：`FlowMatchingV2.embed_prefix()`。原始顺序是场景图像、语言、
  depth/future query。VTLA 顺序是场景图像、语言、触觉 RGB、marker、原任务 query。
- action expert：
  `Qwen2ForCausalLM`，位于
  `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py`，由
  `QwenvlWithExpertV2Model` 按层和 Qwen3-VL 联合执行。
- attention 不是单独的 cross-attention 模块。每层分别从 VLM prefix 和 action
  suffix 计算 Q/K/V，沿 token 轴拼接，再执行同一次 attention。action suffix 的
  Query 因而可读取 prefix 的 Key/Value。加入 prefix 的触觉 token 会自然成为条件。
- mask 语义：`make_att_2d_masks()` 中 `True` 表示有效且允许注意。
- position/RoPE：场景和触觉 RGB 均使用 Qwen image token ID、vision boundary 和
  multimodal RoPE；marker 使用合法 EOS text ID，获得连续的一维位置，不伪造二维坐标。
- 原始 VLM hidden/context 宽度来自 Qwen3-VL（当前配置为 2560）；action expert hidden
  width为 768。两条流在各自投影后产生兼容的 attention heads，不沿 hidden 维拼接。
- Flow Matching 原代码保持：

  ```python
  noise = torch.randn(actions.shape, ...)
  time = self.sample_time(...)
  x_t = time * noise + (1 - time) * actions
  u_t = noise - actions
  losses = mse_loss(u_t, v_t)
  ```

  `sample_time()` 仍使用 Beta(1.5, 1.0) 采样；推理仍从高斯噪声开始，用
  `x_t += dt * v_t` 做原有固定步数 Euler 积分。

## 2. 实现文件

- `lingbotvla/models/vla/lingbot_vla/tactile_vtla.py`
  - 严格配置校验；
  - `[dx,dy,vx,vy]` 构造和训练统计加载；
  - 共享 MLP `MarkerEncoder`；
  - RGB projection、modality embedding、sensor-side embedding；
  - 固定 token 数和显式 mask。
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
  - 训练和采样 API 增加触觉字段；
  - 场景 RGB 与触觉 RGB 在一次共享 Qwen ViT 调用中编码；
  - 触觉 token 插入 prefix；
  - 原 Flow Matching 和动作头不变。
- `lingbotvla/data/vla_data/tactile.py`
  - LeRobot/current inference 字段转成固定 shape 模型输入；
  - position/reference 优先，flow-only 数据等价兼容；
  - 缺失传感器和模态零占位并 mask。
- `lingbotvla/data/vla_data/base_dataset.py`
  - marker 查询前一帧和当前帧，episode 首帧速度确定为零；
  - tactile RGB 查询当前帧。
- `lingbotvla/data/vla_data/utils.py`
  - 将触觉样本接入现有 `FeatureTransform`，不混入 robot state。
- `lingbotvla/optim/vtla.py`、`tasks/vla/train_lingbotvla.py`
  - 冻结策略、互斥 optimizer 参数组、参数量和 LR 日志。
- `lingbotvla/models/module_utils.py`
  - 仅允许无触觉旧 checkpoint 缺失完整 `model.tactile_encoder.*` 命名空间。
- `deploy/lingbot_vla_v2_policy.py`
  - 单样本/批量采样将同一组触觉字段移动到设备并传入模型。
- `tools/convert_tacthru_zarr_to_lerobot_v2.py`
  - 新增可选 `--include-tactile`；默认 wrist-only 行为保留。
- `tools/compute_tacthru_marker_stats.py`
  - 从明确的训练 episode 范围计算 mean/std，episode 间不串速度。
- `tests/test_tactile_vtla.py`
  - shape、mask、缺失模态、梯度、冻结、数据和配置测试。

## 3. 数据合同与 shape

模型 API：

```text
tactile_rgb                 [B,S,L_patch_input,D_patch_input]
tactile_rgb_grid_thw        [B,S,3]
marker_positions            [B,S,N,2]
marker_reference            [B,S,N,2]
previous_marker_positions   [B,S,N,2]
marker_valid_mask           [B,S,N]
tactile_sensor_mask         [B,S]
tactile_rgb_mask            [B,S]
```

其中原始 RGB 在 `FeatureTransform` 前为 `[S,3,H,W]`；Qwen processor 后才变成上面的
flattened patch 输入。共享 ViT 输出 `[B,S,P,D_vlm]`。每个传感器的 marker MLP 输出
一个 `[B,S,D_vlm]` token，不在 sensor 维平均。

```text
marker_features = concat(position-reference, position-previous_position)
marker_features: [B,S,N,4]
marker_tokens:   [B,S,D_vlm]
rgb_tokens:      [B,S*P,D_vlm]（prefix 中每张图另有 start/end）
vtla_prefix:     [B,N_scene_vl + S*(P+2) + S + N_query,D_vlm]
vtla_mask:       [B,同一 token 长度]
```

现有 201 条插网线 Zarr 只有标准化 `marker_flow`。转换器写成
`position=flow, reference=0`；这与 `position-reference=flow` 完全等价。dataloader
查询同 episode 的前一帧 position，生成正确 velocity。没有使用随机填充值。

## 4. 数据转换和训练统计

全部 201 条均作为训练集，不再保留后 20 条验证集。远端训练直接复用已经转换完成的
`insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_v1` 数据目录。

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

python tools/compute_tacthru_marker_stats.py \
  /mnt/models/VTLA-RDT/tacthru/data/tasks/InsertEthernetCable/insert_ethernet_cable_ml_0721_201.zarr.zip \
  assets/norm_stats/insert_ethernet_cable_ml_0721_201_vtla_marker.json \
  --train-episodes 0:201

python tools/convert_tacthru_zarr_to_lerobot_v2.py \
  /mnt/models/VTLA-RDT/tacthru/data/tasks/InsertEthernetCable/insert_ethernet_cable_ml_0721_201.zarr.zip \
  data/lerobot/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_v1 \
  --include-tactile \
  --max-source-episodes 201 \
  --task 'Insert the Ethernet cable'
```

转换前可先执行相同命令并追加 `--check-only`。该模式不需要 LeRobot，只检查源 Zarr、
episode 切分、时间线和触觉字段；真正写出 LeRobot 数据集时仍需要训练环境的 LeRobot。

已生成的统计文件记录了 `training_split_only=true`、episode 范围、帧数、有效 marker
数和特征顺序。统计同时保存在模型 config 和 MarkerEncoder persistent buffer 中。

## 5. 训练命令和消融配置

```bash
# 原模型
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_base.yaml

# 仅触觉 RGB
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_rgb.yaml

# 仅 marker
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_marker.yaml

# RGB + marker
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml
```

三个 VTLA 配置冻结 Qwen3-VL/共享 ViT，训练新增模块和原 action expert，使用 AdamW：
新增模块 `1e-4`，action expert `1e-5`。任何冻结参数都不会进入 optimizer；其余原模型
仍需训练的模块保留 `train.lr`。

## 6. 推理入口

训练完成的 Hugging Face checkpoint 可启动原 WebSocket policy server：

```bash
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
python deploy/lingbot_vla_v2_policy.py \
  --model_path /path/to/vtla/huggingface/checkpoint \
  --port 8006 \
  --use_length 50 \
  --chunk_ret true \
  --use_compile false
```

传给 `LingbotVLAv2Server.infer()` 的 observation 除原 state/wrist/task 外，可直接带：

```text
tactile_rgb                 [S,H,W,3] uint8（原始 observation）
marker_positions            [S,N,2] float32
marker_reference            [S,N,2] float32
previous_marker_positions   [S,N,2] float32
marker_valid_mask           [S,N] bool
tactile_sensor_mask         [S] bool
```

推理复用同一 image processor、marker mean/std、token 顺序、mask、Flow Matching 采样和
动作反归一化。推理不会更新参数。实时客户端必须传入真实 previous marker；若缺失，首帧
确定性使用 current marker，速度为零。

RealMan HTTP 部署协议已升级为 v2。服务器从 checkpoint 的 `train.tactile` 自动恢复
base/RGB/marker/RGB+marker 合同；客户端根据 `/health` 自动打开所需 TacThru 模态。合同不匹配
会直接拒绝请求，不允许静默退化为无触觉推理。

```bash
# GPU 推理端；默认指向本次 all-201 RGB+marker 最终 checkpoint 和 norm
bash scripts/run_tacthru_umi_v2_server.sh --use-compile --warmup

# 机器人端先做不下发动作的真实传感器请求
LINGBOT_V2_EXECUTE=0 \
WORKSPACE_MIN_XYZ='-0.049504 -0.453634 0.043029' \
WORKSPACE_MAX_XYZ='0.496991 -0.228485 0.282270' \
bash scripts/real_insert_ethernet.sh
```

实时 marker 输入严格复现训练数据表示：

```text
flow_t     = 2 * (marker_t - marker_ref_t) / [frame_width, frame_height]
position   = flow_t
reference  = 0
previous   = flow_(t-1)（TacThru 相邻 30 Hz 帧）
```

episode 首帧令 `previous=position`，因此首帧 marker 速度为零。TacThru、腕部相机和机器人
state 的时间戳在发请求前进行新鲜度与 skew 检查。

## 7. checkpoint 兼容

- `tactile.enabled=false` 不创建触觉模块，旧无触觉训练/推理行为不增加 token。
- 旧 normfix checkpoint 可在 VTLA 训练初始化时缺失且只缺失
  `model.tactile_encoder.*`，日志列出初始化的 key。
- checkpoint 一旦含任意 tactile parameter，就必须包含完整 tactile parameter 集。
- tensor shape mismatch 和其他 missing key 仍报错，不静默跳过。
- 新 checkpoint 保存 marker mean/std buffer 和已解析的 config 数值；推理端即使没有原
  stats 文件，也可从 checkpoint config 恢复相同统计。

## 8. 当前验证与风险

- `57 passed`：CPU 测试覆盖编码 shape、双传感器、mask、缺失模态、backward、冻结
  视觉、optimizer 分组、数据兼容、训练统计 split、四个消融配置，以及实际
  `_embed_prefix_vtla()` 方法体的 token 顺序、mRoPE 输入和 deepstack 对齐。
- Flow Matching 插值/target/loss 与推理 Euler 更新有 AST 回归断言，防止触觉改造意外
  改写原数学过程。
- `--check-only --include-tactile` 已对真实 201 条 Zarr 执行成功。正式训练使用完整
  `[0,201)`：201 个 episode、88,363 帧、48 个 marker，30 Hz 连续时间线。
- all-201 RGB+marker 正式训练已在 GPU 服务器启动；最终 VTLA checkpoint 尚未同步回本地，
  因此本地还没有做最终 6B checkpoint 的 strict-load/forward。
- 真正写出 LeRobot 视频数据集仍依赖训练环境中的 LeRobot；服务器现有的 201 条
  `tactile_v1` 目标目录已通过 manifest/meta 检查，因此训练直接复用而不重复转换。
- 最终训练 checkpoint 生成并通过 strict-load/forward 前，不能把接口完成等同于触觉策略已经
  学会修正动作。
- HTTP v2 协议、假 policy 端到端转发、实时 marker 变换和原 RealMan 安全回归共 `39 passed`；
  `rdt-client` 环境也完成了触觉请求本地序列化往返。
- 尚未用训练完成的 6B VTLA checkpoint 做 GPU strict-load/forward，也未执行真实机械臂命令；
  机器人底层控制器没有修改。
- 触觉 RGB 增加 prefix 长度和共享 ViT 计算量，显存和延迟需在目标 GPU 上实测。

后续可选但不属于本次实现：action expert LoRA、接触 gate、快慢双流、时序 marker
Transformer、触觉辅助损失、力/阻抗输出和独立 tactile expert。
