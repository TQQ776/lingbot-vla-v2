# TacThru UMI 数据微调 LingBot-VLA 2.0

本适配用于将 TacThru 的 UMI `.zarr/.zarr.zip` 数采数据转换为 LingBot-VLA 2.0 可直接使用的 LeRobot v3 数据，并按官方 native-depth post-training 方案微调 v2 6B 模型。

适配原则是“只改 embodiment 和数据接口，训练方案尽量保持官方 v2”：保留 UMI 的 30 Hz 原始时序、腕部普通 RGB、EEF 与夹爪数据；当前基线明确屏蔽触觉 RGB 和 marker flow。模型的 50-step action chunk、55D canonical 表示、MoE、Depth/DINO、Muon、学习率和正式训练步数沿用官方 real-robot 配置。TacThru 旧 Diffusion Policy 的 `action_horizon=16` 与 `down_sample_steps=3` 属于旧模型训练超参数，不继承到 v2。

保持官方 v2 不变的部分：

- `LingbotVLAV2Config`、Qwen3-VL-4B-Instruct；
- 50-step 连续 action chunk；
- 55D canonical state/action 与无效维 mask；
- 32 experts、Top-4 fused MoE、FSDP2；
- native-depth、future-depth、DINO-Video 双查询蒸馏；
- Muon、`lr=5e-5`、`max_steps=40000`、正式训练 `torch.compile=true`。

仅针对 TacThru UMI 调整的部分：

- 将 Zarr 转成 v2 支持的 LeRobot v3；
- 保持数据自身的 30 Hz 和 100 个 episode 边界；
- 将物理腕部相机 `camera0_rgb` 映射为唯一的 `camera_wrist_left` 视觉输入；
- 不复制、不训练 `tacthru_{l,r}_rgb` 和 `tacthru_{l,r}_marker`；
- 将绝对 EEF 位姿和夹爪宽度映射到 v2 的 `end/effector` canonical 槽位；
- 为当前数据重新计算 normalization statistics；
- Depth 和 DINO teacher 使用第一路也是唯一一路的腕部普通 RGB。

## 你当前要执行的完整流程

当前本地代码目录：

```text
/mnt/models/VTLA-RDT/lingbot-vla-v2
```

当前正确的 100-episode 数据：

```text
/mnt/models/VTLA-RDT/tacthru/data/tasks/PullTissue/pull_tissue_ml_0706_100.zarr.zip
```

回到集群局域网或 VPN 后，先在本地同步源码。默认是 dry-run，不会删除服务器上的任何文件：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

bash scripts/sync_code_to_server.sh --dry-run
bash scripts/sync_code_to_server.sh --apply
```

然后把原始数据单独传到服务器项目内部。源码同步脚本会排除根目录 `data/`，所以以后再次同步代码不会覆盖训练数据：

```bash
ssh -p 30175 tqq@172.16.41.254 \
  'mkdir -p /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2/data/source'

rsync -avP -e 'ssh -p 30175' \
  /mnt/models/VTLA-RDT/tacthru/data/tasks/PullTissue/pull_tissue_ml_0706_100.zarr.zip \
  tqq@172.16.41.254:/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2/data/source/
```

进入服务器，准备好官方环境和本页后面列出的全部权重后，先执行单 episode 检查：

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2

bash scripts/train_tacthru_umi_v2.sh \
  data/source/pull_tissue_ml_0706_100.zarr.zip \
  --max-source-episodes 1 \
  --overwrite \
  --check-only
```

检查通过后，重建完整 100-episode 转换数据并做 1-step 模型 smoke test：

```bash
bash scripts/train_tacthru_umi_v2.sh \
  data/source/pull_tissue_ml_0706_100.zarr.zip \
  --overwrite \
  --output-dir output/pull_tissue_v2_smoke \
  -- \
  --train.max_steps 1 \
  --train.save_steps 1 \
  --train.save_hf_weights false \
  --train.use_compile false \
  --train.enable_resume false
```

最后换一个干净的正式输出目录，再启动正式训练：

```bash
bash scripts/train_tacthru_umi_v2.sh \
  data/source/pull_tissue_ml_0706_100.zarr.zip \
  --skip-norm \
  --output-dir output/pull_tissue_v2_formal
```

转换器会核对已有转换清单并安全复用完整 LeRobot 数据。上面的最后一条命令只有在 smoke test 使用的确实是完整 100-episode 转换数据和对应 norm 时才能使用 `--skip-norm`。若更换数据、转换目录或统计文件，必须重新转换并重新计算 norm。

如果已经生成过旧的“拆成 300 个 10 Hz phase episode”的转换结果，它与当前 v2-native 方案不兼容，必须使用 `--overwrite` 重新生成，并重新计算 norm，不能复用旧统计文件。

## 最终训练命令

权重和 Python 环境准备完成后，在服务器项目根目录执行：

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2

bash scripts/train_tacthru_umi_v2.sh \
  /服务器可见的/pull_tissue_ml_0706_100.zarr.zip
```

脚本依次完成：

1. TacThru Zarr 转 LeRobot v3；
2. 计算 v2 normalization statistics；
3. 检查 LingBot-VLA v2、Qwen3-VL、MoGe、LingBot-Depth 和 DINO-Video 权重；
4. 使用单张 B200 启动 post-training。

仅准备和检查、不启动训练：

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET --check-only
```

## UMI 数据如何映射到 v2

转换后的原始状态和动作都是 8D：

```text
[x, y, z, qx, qy, qz, qw, gripper]
```

机器人配置将其映射为：

```text
end.position       = xyz + quaternion = 7D
effector.position  = gripper          = 1D
```

其中 EEF 动作使用 v2 原生局部四元数相对表示：

```yaml
subtract_state: true
relative_type: quaternion_local
```

夹爪动作保持绝对量：

```yaml
subtract_state: false
```

模型仍使用官方 55D canonical state/action 顺序：

```text
arm 14 + end 14 + effector 2 + waist 4 + head 2 + base 3 + hand 12
+ reserved 4 = 55D
```

未使用的槽位由 mask 屏蔽；TacThru 实际激活单臂 `end` 的前 7D 和 `effector` 的前 1D。

本页完成的是“数据转换 + v2 微调”链路。现有 TacThru 真机运行代码接收的是相对 10D `xyz + rot6d + gripper`，而 v2 通用 `unapply()` 最终恢复的是绝对 8D `xyz + quaternion + gripper`。因此训练完成后不能直接把通用 v2 输出送进现有真机控制器；部署阶段还需要一个明确的动作桥接层，把未归一化的局部相对四元数动作转换为 TacThru 所需的 rot6d 10D 格式，再复用现有 UMI 执行逻辑。

## 30 Hz UMI 时序

转换器保持原始时序和 episode 边界，不做降采样，也不做相位拆分：

```text
100 个 30 Hz 源 episode → 100 个 30 Hz LeRobot episode
16307 个源帧          → 16307 个转换帧
```

官方 v2 默认 `chunk_size=50`，因此每个训练样本连续读取：

```text
[t, t+1, t+2, ..., t+49]
```

这 50 个动作覆盖 `49 / 30 ≈ 1.633 s`。数据加载器只保留能取得完整 50-step chunk 的 anchor，丢弃每个 episode 最后 49 个会发生 padding 的起点；当前 100-episode 数据应得到 11407 个有效训练样本。

future image 使用第 `t+49` 帧，因此传给 DINO-Video 的真实有效帧率为 `30 / 49 ≈ 0.6122 Hz`。配置和 TacThru batch 都使用由数据 FPS 和 chunk 得到的这一真实值。

## 图像与 Depth

图像输入为：

```text
camera_wrist_left  camera0_rgb 物理腕部普通 RGB
```

原始数据不需要真实 Depth 图。v2 官方 native-depth 训练使用第一路 RGB 和官方教师权重产生几何/视频监督，因此服务器仍需要 MoGe、LingBot-Depth 和 DINO-Video 权重。当前 `data.cameras` 只声明 `camera_wrist_left`，所以腕部 RGB 是第一路也是唯一一路教师输入。

源 Zarr 可以继续包含 `tacthru_l_rgb`、`tacthru_r_rgb` 和 `tacthru_*_marker`，但转换器不会读取或复制它们。转换清单会记录源数据中发现了哪些被排除的触觉字段，并明确写入 `enabled=false`、`copied_to_lerobot=false`、`used_for_training=false`。源 Zarr 本身不会被修改。

转换器版本已提升为 4。若此前生成过包含触觉视觉的旧 LeRobot 目录，必须使用 `--overwrite` 重新转换，不能复用旧目录。

## 服务器权重目录

在服务器项目目录内使用 uv 创建并激活独立的 v2 环境：

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2
bash tools/create_train_uv_env.sh
source .venv/bin/activate
```

该脚本将 Python、依赖和 editable 项目安装在当前 v2 目录的 `.uv/`、`.venv/` 中，不修改旧版 `lingbot-vla/.venv`。安装阶段允许 GPU 暂时不可见，GPU 加回实例后再执行 readiness/smoke 检查。

脚本会优先使用项目内 `.venv/bin/python`；若没有 `.venv`，则使用当前已激活 Conda 环境中的 `python`。也可以显式设置 `PYTHON=/完整路径/bin/python`。

所有仅需下载、不参与本地代码开发的文件放在服务器项目内部：

```text
models/
├── lingbot-vla-v2-6b/
│   ├── config.json
│   ├── model-*.safetensors
│   ├── depth/model.pt
│   └── dino_video/
│       ├── config.yaml
│       └── teacher_step_10000.pth
├── Qwen3-VL-4B-Instruct/
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer/processor 文件
└── moge-2-vitb-normal/
    └── model.pt
```

在服务器已进入 v2 Python 环境后下载：

```bash
cd /root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2

python scripts/download_hf_model.py \
  --repo_id robbyant/lingbot-vla-v2-6b \
  --local_dir models

python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-VL-4B-Instruct \
  --local_dir models

python scripts/download_hf_model.py \
  --repo_id Ruicheng/moge-2-vitb-normal \
  --local_dir models
```

下载只应在训练服务器进行。本地开发目录不需要保存这些权重。

若服务器已有其他目录布局，可以用同名环境变量覆盖，不必修改代码：

```bash
MODEL_DIR=/path/to/lingbot-vla-v2-6b \
TOKENIZER_DIR=/path/to/Qwen3-VL-4B-Instruct \
MOGE_PATH=/path/to/moge/model.pt \
MORGBD_PATH=/path/to/depth/model.pt \
DINO_CKPT=/path/to/dino/teacher_step_10000.pth \
DINO_CONFIG=/path/to/dino/config.yaml \
bash scripts/train_tacthru_umi_v2.sh DATASET
```

## 默认训练设置

配置文件：`configs/vla/tacthru_umi/tacthru_umi.yaml`

```text
模型配置：LingbotVLAV2Config + Qwen3-VL-4B-Instruct
Action Expert：32 experts，Top-4 fused MoE
Depth/Video：官方 native-depth + DINO-Video
数据频率：30 Hz，保持 UMI 原始频率
chunk size：50（官方 v2 默认）
micro batch：4
gradient accumulation：1（不累计）
global batch：4
GPU：单张 B200
torch.compile：正式训练按官方配置开启；首次 smoke 通过命令行关闭
```

先做单步 smoke training：

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET -- \
  --train.max_steps 1 \
  --train.save_steps 1 \
  --train.save_hf_weights false \
  --train.use_compile false \
  --train.enable_resume false
```

单步测试仍会写一个用于检查恢复链路的 DCP checkpoint，但不会额外导出约几十 GB 的 Hugging Face 权重。

正式配置沿用官方 real-robot 策略，默认在第 40000 步保存一次。若需要中途恢复，可以显式设置例如
`--train.save_steps 5000 --train.save_hf_weights false`；请先确认服务器有足够空间，因为 DCP 会包含模型和优化器状态，当前训练代码不会自动轮转删除旧 checkpoint。

基础单步通过后，再用官方 `torch.compile=true` 做一次单步检查：

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET \
  --skip-norm \
  --output-dir output/pull_tissue_v2_compile_smoke \
  -- \
  --train.max_steps 1 \
  --train.save_steps 1 \
  --train.save_hf_weights false \
  --train.use_compile true \
  --train.enable_resume false
```

`--skip-norm` 只能在数据集没有变化且 `assets/norm_stats/tacthru_umi_v2.json` 确实对应当前数据时使用。

## 常用选项

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET \
  --task "Pull the tissue" \
  --lerobot-dir data/lerobot/pull_tissue_v2 \
  --output-dir output/pull_tissue_v2
```

转换一个源 episode 做数据 smoke test：

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET \
  --max-source-episodes 1 \
  --overwrite \
  --check-only
```

脚本默认要求转换清单中有 100 个源 episode。若正式数据量不是 100，显式指定：

```bash
bash scripts/train_tacthru_umi_v2.sh DATASET \
  --expected-source-episodes 实际数量
```

正式训练完整数据前，需要再次使用 `--overwrite` 生成完整转换结果，并重新计算 normalization statistics。

## 产物位置

```text
转换数据：data/lerobot/<数据集名>_tacthru_umi_v2/
统计文件：assets/norm_stats/tacthru_umi_v2.json
训练输出：output/<转换数据集目录名>/
```

本地源码目录建议保持为：

```text
/mnt/models/VTLA-RDT/lingbot-vla-v2
```

服务器运行目录：

```text
/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2
```

工作流是“本地只开发和提交代码，服务器同步代码并下载权重、转换数据、训练”。

训练完成后的 GPU server、本地 client、episode 坐标转换和 Realman 真机安全执行步骤见：

```text
docs/TACTHRU_UMI_V2_REALMAN_DEPLOY.md
```
