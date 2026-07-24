# LingBot V2 触觉融合使用说明

本实现把 wrist RGB 保持为原 Qwen/Depth/DINO 的第一且唯一视觉视图。触觉 RGB
和 48×2 marker flow 使用独立编码器，生成非 visual 的固定 Qwen prefix token，
最终动作格式仍为 `[50, 8]`。

## 1. 不变项与新入口

- 原 wrist-only 训练入口不变：`scripts/train_tacthru_umi_v2.sh`
- 原 protocol v1 服务端/客户端和 `scripts/real_insert_ethernet.sh` 不变
- 新训练入口：`scripts/train_tacthru_umi_v2_tactile.sh`
- 新 protocol v2 服务端默认端口：`18082`
- 新真机脚本默认是 dry-run，不会自动执行机械臂

触觉架构有四种独立模式：

| 模式 | 触觉 RGB | Marker |
|---|---:|---:|
| `none` | 否 | 否 |
| `rgb` | 是 | 否 |
| `marker` | 否 | 是 |
| `rgb-marker` | 是 | 是 |

所有模式复用同一份 `wrist + tactile RGB + marker` superset 数据。训练 launcher
会为不同模式和阶段建立独立 output，并用 `tactile_experiment.json` 拒绝跨实验
resume。`none` 只允许配合 `full`；`rgb` 和 `rgb-marker` 需要触觉 DINOv2 权重；
`marker` 不需要该权重。

## 2. 权重、模型目录与数据前提

### 2.1 `BASE_MODEL_ASSET_DIR` 与 `MODEL_DIR` 不同

这两个目录不能混用：

- `BASE_MODEL_ASSET_DIR`：官方 LingBot V2 基础资产目录，主要提供训练时仍需使用的
  Depth/Future-Video teacher 等固定资产，例如 `depth/model.pt`、
  `dino_video/teacher_step_10000.pth` 和 `dino_video/config.yaml`。
- `MODEL_DIR`：本阶段用于初始化主模型权重的 HF checkpoint。T1 通常使用官方
  LingBot V2 模型，T2 应指向 T1 的 `hf_ckpt`，T3 应指向 T2 的 `hf_ckpt`。

默认情况下两者都指向 `models/lingbot-vla-v2-6b`。进入 T2/T3 后，只修改
`MODEL_DIR`，继续让 `BASE_MODEL_ASSET_DIR` 指向完整的官方基础资产：

```bash
BASE_MODEL_ASSET_DIR=/path/to/official/lingbot-vla-v2-6b \
MODEL_DIR=/path/to/previous_stage/hf_ckpt \
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb-marker --tactile-stage expert
```

如果把 `BASE_MODEL_ASSET_DIR` 错误地指向普通阶段 checkpoint，可能找不到 Depth 或
Future-Video teacher；如果忘记修改 `MODEL_DIR`，T2/T3 会重新从基础模型开始，而不是
继承上一阶段。

这里还有两种名称相近但用途不同的 DINO 权重：

- `BASE_MODEL_ASSET_DIR/dino_video/...`：原 V2 的 Future-Video teacher。
- `models/dinov2_vits14_pretrain.pth`：触觉 RGB 专用 DINOv2 ViT-S/14 backbone。

正式触觉 RGB 分支使用本项目内置的 DINOv2 ViT-S/14 实现，但不会联网下载权重。
需要准备本地权重：

```text
models/dinov2_vits14_pretrain.pth
```

该权重使用官方 518×518 预训练位置编码；编码器会按 DINOv2 标准方式插值到实际
224×224 触觉输入，不能用 224×224 重新构造骨干后再严格加载权重。

readiness 会检查文件存在、大小和 SHA256。`marker`、`none` 模式不要求该文件。
若只做代码 smoke，可显式把配置改为 `native_patch_transformer`；它不是推荐正式模型。

当前 Tac 数据中的 marker 已由数据处理流程按真实 640×480 tracker 坐标完成：

```text
flow_x = dx / 640 * 2
flow_y = dy / 480 * 2
```

converter 默认将其视为 `normalized`，不会再次归一化。只有旧 `/400` 产物才应显式
使用 `--marker-input-space legacy_400_normalized`。

## 3. 首次准备与检查

以下示例以插网线数据为例：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_tacthru_umi_v2_tactile.sh \
  /path/to/insert_ethernet_cable.zarr.zip \
  --task "Insert the Ethernet cable." \
  --tactile-mode rgb-marker \
  --tactile-stage adapters \
  --expected-source-episodes 201 \
  --check-only
```

该命令会准备统一 superset、norm stats 和 contract，但 `--check-only` 不启动训练。
生成路径按源数据名自动隔离，也可显式传 `--lerobot-dir`、`--output-dir`。

## 4. 四组基础消融

第一组完成转换和 norm 后，其余组复用同一 `--lerobot-dir`，并加
`--skip-convert --skip-norm`。`DATASET` 仍可保留原 Zarr 路径，因为显式
`--lerobot-dir` 指向已经转换好的目录：

```bash
# A0: wrist-only graph，仍使用同一 superset 的 wrist/state/action
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode none --tactile-stage full \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm

# A1: tactile RGB only
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb --tactile-stage full \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm

# A2: marker only
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode marker --tactile-stage full \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm

# A4: RGB + marker，K=4
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb-marker --tactile-stage full \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm
```

launcher 默认继承当前 40,000 step、50 action chunk 等 V2 参数。临时 smoke 参数仍放在
`--` 后，例如：

```bash
-- --train.max_steps 1 --train.save_steps 1 --train.enable_resume false
```

### 4.1 `--` 后的覆盖规则

`--` 后只允许覆盖普通训练超参数，例如 `max_steps`、`save_steps`、学习率或是否 resume。
传入的完整 override 列表会写进 `tactile_experiment.json`，参与 experiment contract
self-hash。因此即使只修改 `--train.lr`，也属于另一个实验，必须使用新的 output；
launcher 会拒绝在旧 output 中跨合同 resume。

以下保护字段禁止在 `--` 后覆盖，它们必须由 launcher、`--tactile-mode`、
`--tactile-stage` 或文档列出的环境变量统一生成：

```text
model.model_path / model.tokenizer_path
data.data_name / data.train_path / data.robot_config_root / data.norm_stats_file
data.tactile_* canonical keys
train.output_dir / train.align_params
train.tactile_rgb_enabled / train.tactile_marker_enabled
train.tactile_params / train.tactile_train_stage
train.tactile_*_sha256 contract fields
eval.force_mask_tactile_rgb / eval.force_mask_tactile_marker
```

不要使用 `--key=value` 等形式尝试绕过保护；这样会破坏可复现实验合同。训练时的
force-mask 也不是架构消融，应在允许消融的 synthetic/offline/dry-run 部署流程中使用。

## 5. T1/T2/T3 分阶段训练

- `adapters`：只训练触觉编码器、projection 和 fusion gate，并尊重冻结的 DINO backbone
- `expert`：在 adapters 基础上解冻 Action Expert 和 state/action/time projection
- `full`：恢复原 V2 的全量可训练路径，用较小学习率做最终联合微调

每个阶段必须使用不同 output。后一阶段通过 `MODEL_DIR` 指向前一阶段完整 HF
checkpoint，而 `BASE_MODEL_ASSET_DIR` 始终保留官方 teacher 资产；不能在同一 output
中跨阶段 resume：

```bash
# T1：两个目录均可使用默认官方基础模型
BASE_MODEL_ASSET_DIR=/path/to/official/lingbot-vla-v2-6b \
MODEL_DIR=/path/to/official/lingbot-vla-v2-6b \
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb-marker --tactile-stage adapters

# T2：主模型从 T1 初始化，teacher 资产仍来自官方目录
BASE_MODEL_ASSET_DIR=/path/to/official/lingbot-vla-v2-6b \
MODEL_DIR=/path/to/T1/hf_ckpt \
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb-marker --tactile-stage expert \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm

# T3：主模型从 T2 初始化
BASE_MODEL_ASSET_DIR=/path/to/official/lingbot-vla-v2-6b \
MODEL_DIR=/path/to/T2/hf_ckpt \
bash scripts/train_tacthru_umi_v2_tactile.sh DATASET \
  --tactile-mode rgb-marker --tactile-stage full \
  --lerobot-dir LEROBOT_SUPERSET --skip-convert --skip-norm \
  -- --train.lr 1.0e-5
```

## 6. Experiment contract 与哈希保护

触觉 launcher 会在每个 output 中维护 `tactile_experiment.json`。准备阶段完成后，合同
至少固定以下信息：

- dataset schema 模式、训练模态和训练阶段；
- 原始数据、LeRobot superset、norm、初始化模型和基础资产路径；
- 完整 `train_overrides`；
- 触觉 YAML SHA256、dataset conversion manifest SHA256、norm SHA256；
- RGB 模式下的触觉 DINOv2 backbone SHA256；
- Git commit 和最终 `contract_sha256` self-hash。

训练启动时，experiment self-hash、dataset manifest SHA 和 RGB backbone SHA 会写进
checkpoint 保存的 `lingbotvla_cli.yaml`。同一 output 只有在模式、阶段、初始化路径、
配置和 overrides 完全一致时才允许 resume；非空目录没有合同或合同不同都会被拒绝。
如果先对某个正式 output 执行 `--check-only`，应同时带上后续训练准备使用的全部
`--` overrides，否则正式训练改变 overrides 时会被视为另一个实验。

服务端加载触觉 checkpoint 时会进行 fail-closed 校验：

1. 重新计算 `tactile_experiment.json` 的 self-hash，并和合同及训练配置中的值比较。
2. 对 output 中冻结的 `tactile_dataset_manifest.json` 重新计算 SHA256，并同时与
   experiment、训练配置中的固定值比较。
3. 对当前实际 norm 文件重新计算 SHA256，并与 experiment 比较。
4. RGB 模式检查 experiment 与训练配置记录的 DINOv2 backbone SHA256 一致，并严格加载
   配置指定的本地 backbone。
5. 校验训练配置、robot config、norm 和 checkpoint index 的实际 SHA256，生成服务端
   `combined_sha256`；protocol v2 客户端请求必须携带相同 contract hash。
6. 交叉校验 experiment 的训练模态、阶段、LeRobot 路径、初始化模型路径与 checkpoint
   实际保存的 YAML。
7. 同时校验历史长度、marker `48×2`、`image_size_xy=[640,480]`、数据键及机器人
   state/action 映射，任何不一致都拒绝启动。

这能防止拿错 norm、数据合同、配置或 checkpoint 后仍静默推理。

## 7. 服务端与 dry-run

先在服务器设置实际 checkpoint 和 norm：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

export LINGBOT_V2_TACTILE_CHECKPOINT=/path/to/global_step_N/hf_ckpt
export LINGBOT_V2_TACTILE_NORM_STATS=/path/to/tactile_norm.json
export LINGBOT_V2_API_KEY='你自己的随机长字符串'

bash scripts/run_tacthru_umi_v2_tactile_server.sh \
  --host 127.0.0.1 \
  --use-compile \
  --warmup
```

本地通过 SSH 隧道访问 `127.0.0.1:18082` 后，先运行 synthetic，再运行真机 dry-run：

```bash
bash scripts/run_tacthru_umi_v2_tactile_client.sh synthetic \
  --server-url http://127.0.0.1:18082 \
  --instruction "Insert the Ethernet cable."

STEPS=20 bash scripts/real_insert_ethernet_tactile.sh
```

只有 synthetic、离线回放和 dry-run 均通过后，才可显式执行：

```bash
LINGBOT_V2_EXECUTE=1 \
WORKSPACE_MIN_XYZ='X_MIN Y_MIN Z_MIN' \
WORKSPACE_MAX_XYZ='X_MAX Y_MAX Z_MAX' \
STEPS=20 \
bash scripts/real_insert_ethernet_tactile.sh
```

不要因为增加触觉而放宽原 workspace、速度、位姿差或网络安全限制。

## 8. 同 checkpoint 消融

只允许 synthetic、offline evaluation 或 dry-run：

```bash
LINGBOT_V2_ALLOW_TACTILE_ABLATION=1 \
bash scripts/run_tacthru_umi_v2_tactile_server.sh --warmup

LINGBOT_V2_FORCE_MASK_TACTILE_RGB=1 \
bash scripts/run_tacthru_umi_v2_tactile_client.sh synthetic \
  --server-url http://127.0.0.1:18082 \
  --instruction "Insert the Ethernet cable."
```

执行模式会在打开硬件前拒绝 force-mask、缺失触觉或关闭本地 guard。

## 9. 已知限制与后续增强

- `initial_model` 和 `base_model_assets` 的规范化路径已经进入 experiment self-hash，路径
  被修改会改变实验合同；当前实现尚未对这两个目录的全部文件内容单独计算 SHA256。
- dataset conversion manifest 会冻结一份内容快照到训练 output；服务端重算该快照的
  SHA，因此部署服务器不需要重新挂载完整训练数据集。
- RGB backbone 的 SHA 固定在训练合同中，服务端会校验合同值并严格加载本地文件；后续
  可进一步增加“对部署机器上的 backbone 文件重新计算 SHA”的独立检查。
- 改变模式、阶段、`MODEL_DIR`、`BASE_MODEL_ASSET_DIR` 或任何 `--` 后 override 时，始终
  新建 output，不要复用旧目录。
