# Codex 任务：将 LingBot-VLA 2.0 最小改造成 TacThru VTLA

你正在一个现有的 **LingBot-VLA 2.0** 代码仓库中工作。

请先阅读仓库结构、模型定义、数据管线、训练脚本和推理脚本，然后完成一个最小可运行的 VTLA 改造：在原有视觉—语言—动作模型基础上加入 TacThru 的触觉 RGB 和 marker 数据。

不要凭空假设类名、文件名或张量接口。必须以仓库中的真实代码为准。

## 目标

将原模型：

[ v = v_ ( A_,  C_{},S ) ]

改为：

[ v = v_ ( A_,  C_{},S ) ]

其中：

[ C_{} = [ C_{}; Z_{}; Z_{}]. ]

要求：

- 保留原视觉语言主干；
- 保留原 Flow Matching 训练方法；
- 保留原动作空间；
- 保留原动作输出头；
- 保留原推理积分或去噪流程；
- 只扩展 action expert 的条件上下文；
- 第一版不实现触觉门控、快慢双流或额外损失。

# 一、先分析仓库

在修改任何代码前，先定位并总结以下内容：

1. 视觉语言主干的入口；
1. 图像经过视觉编码器后产生 token 的位置；
1. 语言和视觉 token 融合的位置；
1. 最终视觉语言上下文 context 的张量形状；
1. action expert 的类名和文件位置；
1. action expert 的 forward 接口；
1. action expert 如何使用视觉语言 context；
1. cross-attention 的 Query、Key、Value 来源；
1. Flow Matching 中：
    - 噪声动作如何生成；
    - flow time 如何采样；
    - 中间动作如何构造；
    - target velocity 如何构造；
    - loss 在哪里计算；
1. 推理时动作如何从噪声逐步生成；
1. 数据集返回的字段；
1. 多相机图像目前如何组织；
1. attention mask、position ids、RoPE 如何处理；
1. 模型冻结和 optimizer 参数分组在哪里配置。

输出一个简短的仓库分析说明后，再开始代码修改。

# 二、增加数据字段

在原数据样本中加入以下字段。

```
sample = {
    # 原字段
    "images": ...,
    "language": ...,
    "state": ...,
    "actions": ...,

    # 新字段
    "tactile_rgb": ...,
    "marker_positions": ...,
    "marker_reference": ...,
    "marker_valid_mask": ...,
}
```

建议张量形状：

```
tactile_rgb:
    [B, num_tactile_sensors, C, H, W]

marker_positions:
    [B, num_tactile_sensors, N, 2]

marker_reference:
    [B, num_tactile_sensors, N, 2]

marker_valid_mask:
    [B, num_tactile_sensors, N]
```

其中：

- num_tactile_sensors 可以为 1 或 2；
- N 为每个 TacThru 传感器的 marker 数量；
- marker 顺序在所有帧中必须固定；
- 缺失 marker 必须通过 mask 处理，不能直接填随机值。

如数据集中已有前一帧 marker，增加：

```
previous_marker_positions:
    [B, num_tactile_sensors, N, 2]
```

若没有，则在 dataset 中根据时间序列构造。

# 三、marker 特征预处理

根据原始 marker 坐标构造：

[ M_t=M_t-M_{} ]

以及：

[ V_t=M_t-M_{t-1}. ]

每个 marker 的输入特征为：

[ [ x, y, v_x, v_y]. ]

输出形状：

```
marker_features:
    [B, num_tactile_sensors, N, 4]
```

应用 marker_valid_mask，使无效 marker 对编码结果不产生影响。

增加训练集统计量：

```
marker_mean
marker_std
```

标准化：

```
marker_features = (
    marker_features - marker_mean
) / (marker_std + 1e-6)
```

要求：

- 统计量只使用训练集计算；
- 保存到 checkpoint 或配置文件；
- 验证和推理复用同一统计量；
- 不允许验证集或测试集参与统计。

# 四、实现轻量 MarkerEncoder

新增一个独立模块，例如：

```
class MarkerEncoder(nn.Module):
    ...
```

第一版采用简单 MLP，不使用 Transformer、GRU 或 PointNet。

参考实现：

```
from __future__ import annotations

import torch
from torch import nn


class MarkerEncoder(nn.Module):
    def __init__(
        self,
        num_markers: int,
        context_dim: int,
        input_features: int = 4,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        if num_markers <= 0:
            raise ValueError("num_markers must be positive")

        if context_dim <= 0:
            raise ValueError("context_dim must be positive")

        input_dim = num_markers * input_features

        self.num_markers = num_markers
        self.input_features = input_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(
        self,
        marker_features: torch.Tensor,
        marker_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            marker_features:
                [B, N, 4]

            marker_valid_mask:
                Optional bool or float tensor [B, N].

        Returns:
            marker_token:
                [B, 1, context_dim]
        """
        if marker_features.ndim != 3:
            raise ValueError(
                "marker_features must have shape [B, N, F]"
            )

        batch_size, num_markers, num_features = marker_features.shape

        if num_markers != self.num_markers:
            raise ValueError(
                f"Expected {self.num_markers} markers, "
                f"received {num_markers}"
            )

        if num_features != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} marker features, "
                f"received {num_features}"
            )

        x = marker_features

        if marker_valid_mask is not None:
            if marker_valid_mask.shape != (batch_size, num_markers):
                raise ValueError(
                    "marker_valid_mask must have shape [B, N]"
                )

            mask = marker_valid_mask.to(dtype=x.dtype).unsqueeze(-1)
            x = x * mask

        x = x.reshape(batch_size, -1)
        token = self.encoder(x)

        return token.unsqueeze(1)
```

如果存在左右两个 TacThru：

- 第一版优先共享同一个 MarkerEncoder；
- 为不同传感器增加独立 side embedding；
- 最终每个传感器输出一个 token。

结果形状：

```
marker_tokens:
    [B, num_tactile_sensors, context_dim]
```

不要在 sensor 维度上求平均。

# 五、增加触觉 RGB 编码

第一版优先复用原有视觉编码器，不从头创建大型 tactile ViT。

实现以下逻辑：

```
tactile_rgb_tokens = vision_encoder(tactile_rgb)
```

必须检查原视觉编码器是否接受：

```
[B, C, H, W]
```

如果有多个触觉传感器，将：

```
[B, S, C, H, W]
```

临时 reshape 为：

```
[B * S, C, H, W]
```

编码后再恢复 sensor 维度。

例如：

```
batch_size, num_sensors, channels, height, width = tactile_rgb.shape

flat_tactile_rgb = tactile_rgb.reshape(
    batch_size * num_sensors,
    channels,
    height,
    width,
)

flat_tokens = vision_encoder(flat_tactile_rgb)

tactile_rgb_tokens = flat_tokens.reshape(
    batch_size,
    num_sensors * flat_tokens.shape[1],
    flat_tokens.shape[2],
)
```

如果视觉编码器输出维度不等于 action context dimension，增加：

```
self.tactile_rgb_projection = nn.Linear(
    vision_output_dim,
    context_dim,
)
```

输出：

```
tactile_rgb_tokens:
    [B, num_tactile_tokens, context_dim]
```

复用视觉编码器时，默认冻结其预训练参数。

# 六、增加模态和传感器身份 embedding

新增可学习 embedding，用于区分：

- 原场景视觉 token；
- 触觉 RGB token；
- marker token；
- 左传感器；
- 右传感器。

至少实现：

```
self.tactile_rgb_modality_embedding
self.marker_modality_embedding
self.sensor_side_embeddings
```

例如：

```
self.tactile_rgb_modality_embedding = nn.Parameter(
    torch.zeros(1, 1, context_dim)
)

self.marker_modality_embedding = nn.Parameter(
    torch.zeros(1, 1, context_dim)
)

self.sensor_side_embeddings = nn.Embedding(
    num_tactile_sensors,
    context_dim,
)
```

加入方式：

```
token = (
    token
    + modality_embedding
    + side_embedding
)
```

如果仓库已有 camera embedding 或 modality embedding 机制，应复用原机制，不要重复造一套不兼容实现。

# 七、构造 VTLA context

原始上下文：

```
vision_language_context:
    [B, N_vl, context_dim]
```

新增：

```
tactile_rgb_tokens:
    [B, N_rgb, context_dim]

marker_tokens:
    [B, N_marker, context_dim]
```

沿 token 序列维拼接：

```
vtla_context = torch.cat(
    [
        vision_language_context,
        tactile_rgb_tokens,
        marker_tokens,
    ],
    dim=1,
)
```

结果：

```
vtla_context:
    [B, N_vl + N_rgb + N_marker, context_dim]
```

禁止沿 hidden dimension 拼接。

同时扩展 attention mask：

```
vtla_attention_mask = torch.cat(
    [
        vision_language_attention_mask,
        tactile_rgb_attention_mask,
        marker_attention_mask,
    ],
    dim=1,
)
```

要求：

- mask 的 dtype 与原实现一致；
- mask 的语义与原实现一致；
- 有效 token 为 1 还是 0，必须根据现有代码判断；
- 不得凭经验猜测；
- 新 token 必须参与 action expert 对 context 的 attention；
- padding token 不得参与 attention。

# 八、处理 position ids 和 RoPE

检查原 context 是否使用：

- 绝对位置 embedding；
- 1D RoPE；
- 多模态 RoPE；
- 图像二维 position ids；
- camera-specific position ids。

第一版原则：

1. 不修改原视觉语言 token 的 position ids；
1. 新增触觉 RGB token 使用与额外图像视角一致的位置编码逻辑；
1. marker token 使用独立、连续且合法的位置；
1. 不允许 position id 冲突导致 shape error；
1. 不允许直接给 marker 构造伪二维图像位置；
1. 若 action expert 只把 context 当作 Key/Value，优先使用最小兼容方案；
1. 若 RoPE 在 action expert 内部重新计算，确保新 token 的 position_ids 和 mask 一起传入。

在代码注释中解释最终采用的位置编码方案。

# 九、修改 action expert 条件

原接口可能类似：

```
velocity = action_expert(
    noisy_actions=noisy_actions,
    timestep=flow_time,
    context=vision_language_context,
    state=robot_state,
    context_attention_mask=context_attention_mask,
)
```

修改为：

```
velocity = action_expert(
    noisy_actions=noisy_actions,
    timestep=flow_time,
    context=vtla_context,
    state=robot_state,
    context_attention_mask=vtla_attention_mask,
)
```

不要改变：

- noisy_actions；
- flow_time；
- robot_state；
- 动作 token 编码；
- 输出 shape；
- Flow Matching target；
- action head；
- ODE solver；
- Euler 或其他积分器；
- action normalization。

如果 action expert 的 cross-attention 明确使用：

```
query = action_hidden
key = context
value = context
```

那么只需要让 key 和 value 使用 vtla_context。

第一版不要：

- 用 marker token 替换 action Query；
- 切换 Query；
- 加入接触 gate；
- 加入独立 tactile expert；
- 修改 MoE router；
- 修改已有 expert 数量；
- 修改 action expert 层数。

# 十、保持 Flow Matching 不变

训练逻辑继续使用仓库原实现。

理论形式为：

```
action_noise = torch.randn_like(actions)
flow_time = torch.rand(batch_size, device=actions.device)

action_tau = (
    (1.0 - flow_time_view) * action_noise
    + flow_time_view * actions
)

target_velocity = actions - action_noise

predicted_velocity = action_expert(
    noisy_actions=action_tau,
    timestep=flow_time,
    context=vtla_context,
    state=state,
    context_attention_mask=vtla_attention_mask,
)

loss = torch.mean(
    (predicted_velocity - target_velocity) ** 2
)
```

实际修改必须复用仓库原有 Flow Matching 实现，不要把已有插值路径重写成以上示例，除非仓库本身就是该公式。

总损失保持原损失：

```
total_loss = flow_matching_loss
```

不增加：

- marker 重建损失；
- tactile RGB 重建损失；
- 接触分类损失；
- 力预测损失；
- 对比学习损失。

# 十一、训练参数冻结

默认冻结：

- VLM；
- language model；
- 原视觉编码器；
- 原场景图像编码路径。

训练：

- MarkerEncoder；
- tactile RGB projection；
- modality embeddings；
- sensor side embeddings；
- action expert；
- 或 action expert 的 LoRA。

如果仓库支持 LoRA，优先提供两种配置：

### 配置 A：快速稳定版

训练：

- 所有新增模块；
- action expert LoRA。

冻结：

- action expert 原始权重；
- VLM；
- vision encoder。

### 配置 B：完整任务微调版

训练：

- 所有新增模块；
- action expert 全部或最后若干层。

冻结：

- VLM；
- vision encoder。

不要默认全量训练整个 VLM。

# 十二、optimizer 参数分组

为新增模块和预训练 action expert 使用不同学习率。

目标形式：

```
optimizer = torch.optim.AdamW(
    [
        {
            "params": new_module_params,
            "lr": 1e-4,
            "weight_decay": 1e-2,
        },
        {
            "params": action_expert_params,
            "lr": 1e-5,
            "weight_decay": 1e-2,
        },
    ]
)
```

new_module_params 至少包含：

- marker encoder；
- tactile RGB projection；
- modality embeddings；
- side embeddings。

确保：

- 一个参数不能出现在两个参数组；
- frozen 参数不能进入 optimizer；
- 输出日志中打印每个参数组的参数量；
- 配置文件中允许覆盖学习率。

# 十三、缺失触觉数据处理

模型应支持以下情况：

1. 单个触觉传感器；
1. 左右两个触觉传感器；
1. 某个传感器临时缺失；
1. marker 部分缺失；
1. tactile RGB 暂时缺失。

实现显式有效性 mask，例如：

```
tactile_sensor_mask:
    [B, num_tactile_sensors]
```

缺失模态不能通过删除 token 改变 batch 内序列长度。

优先采用：

- 固定 token 数量；
- 无效 token 置零；
- attention mask 屏蔽无效 token。

# 十四、推理修改

在推理入口增加：

```
tactile_rgb
marker_positions
marker_reference
previous_marker_positions
marker_valid_mask
tactile_sensor_mask
```

推理必须：

1. 使用训练阶段相同的 marker 预处理；
1. 使用训练集保存的 mean/std；
1. 使用相同的视觉图像归一化；
1. 使用相同的 modality embedding；
1. 构建相同顺序的 vtla_context；
1. 使用相同 attention mask；
1. 保持原 Flow Matching 采样步骤不变；
1. 保持原 action denormalization 不变；
1. 保持原机器人控制接口不变。

不得在推理阶段额外更新参数。

# 十五、配置项

新增清晰的配置字段，名称可根据仓库风格调整：

```
tactile:
  enabled: true

  num_sensors: 2
  num_markers: 64

  use_rgb: true
  use_markers: true

  share_vision_encoder: true
  freeze_vision_encoder: true

  marker_input_features: 4
  marker_hidden_dim: 512
  marker_tokens_per_sensor: 1

  add_modality_embedding: true
  add_sensor_side_embedding: true

  marker_stats_path: null

training:
  freeze_vlm: true
  train_action_expert: true
  use_action_expert_lora: false

  new_modules_lr: 1.0e-4
  action_expert_lr: 1.0e-5
```

配置关闭：

```
tactile:
  enabled: false
```

时，模型行为必须与原始 LingBot-VLA 2.0 一致。

# 十六、向后兼容

必须保证：

- 旧 checkpoint 可以在 tactile.enabled=false 时加载；
- 新 checkpoint 能保存新增模块；
- 加载旧 checkpoint 时新增参数可随机初始化，并明确打印 missing keys；
- 不能静默忽略 shape mismatch；
- 原始无触觉训练配置仍能运行；
- 原始无触觉推理仍能运行。

建议为新增模块使用独立命名空间，例如：

```
model.tactile_encoder
model.tactile_rgb_projection
model.tactile_embeddings
```

# 十七、单元测试

至少增加以下测试。

## 1. MarkerEncoder shape test

输入：

```
[B, N, 4]
```

输出：

```
[B, 1, context_dim]
```

## 2. 双传感器 shape test

输入：

```
[B, 2, N, 4]
```

输出：

```
[B, 2, context_dim]
```

## 3. tactile RGB shape test

输入：

```
[B, S, C, H, W]
```

输出：

```
[B, S * num_patches, context_dim]
```

## 4. context 拼接测试

检查：

```
vtla_context.shape[1] == (
    vl_context.shape[1]
    + tactile_rgb_tokens.shape[1]
    + marker_tokens.shape[1]
)
```

## 5. attention mask 测试

检查 mask 长度等于 VTLA context 长度。

## 6. backward 测试

执行一次：

```
loss.backward()
```

确认以下参数存在非空梯度：

- marker encoder；
- tactile RGB projection；
- modality embeddings；
- action expert 可训练参数。

## 7. frozen module 测试

确认 VLM 和冻结的 vision encoder 梯度为 None。

## 8. 无触觉回归测试

tactile.enabled=false 时：

- 原始输入可运行；
- 输出 shape 与原模型一致；
- 不创建额外 context token。

## 9. 缺失传感器测试

屏蔽一个传感器后：

- forward 不报错；
- token 数量保持固定；
- 被屏蔽 token 不参与 attention。

# 十八、调试日志

在 debug 模式下打印一次以下 shape：

```
vision_language_context
vision_language_attention_mask
tactile_rgb_tokens
marker_tokens
vtla_context
vtla_attention_mask
robot_state
noisy_actions
flow_time
predicted_velocity
target_velocity
```

只在首次 batch 或显式 debug 配置下打印，避免训练日志泛滥。

同时打印：

- 可训练参数总数；
- 冻结参数总数；
- 新增模块参数量；
- action expert 可训练参数量；
- optimizer 参数组学习率；
- 每个触觉传感器的有效比例。

# 十九、最低限度消融配置

创建以下配置文件或启动参数。

## 1. 原模型

```
tactile:
  enabled: false
```

## 2. 仅触觉 RGB

```
tactile:
  enabled: true
  use_rgb: true
  use_markers: false
```

## 3. 仅 marker

```
tactile:
  enabled: true
  use_rgb: false
  use_markers: true
```

## 4. RGB + marker

```
tactile:
  enabled: true
  use_rgb: true
  use_markers: true
```

这些配置必须复用相同训练和评估流程。

# 二十、实现限制

本次不要实现以下功能：

- 触觉中期训练；
- 接触分类器；
- 接触 gate；
- Adaptive Query；
- tactile fast stream；
- 独立 tactile action expert；
- tactile MoE router；
- marker 时序 Transformer；
- 触觉 RGB 重建；
- marker 重建；
- 力估计头；
- 阻抗控制输出；
- 新动作维度；
- 新 Flow Matching 路径；
- 修改 ODE solver；
- 修改机器人底层控制器。

# 二十一、代码质量要求

所有新增代码必须：

- 使用类型标注；
- 有清晰 docstring；
- 检查输入 shape；
- 对无效配置抛出明确异常；
- 避免硬编码 hidden dimension；
- 避免硬编码 marker 数量；
- 避免硬编码传感器数量；
- 遵循仓库原有代码风格；
- 使用仓库已有 registry、config 和 logging 系统；
- 不重复实现仓库已有工具；
- 不引入不必要的新依赖。

# 二十二、最终交付内容

完成后请输出：

1. 仓库结构分析；
1. 修改文件列表；
1. 每个文件的修改目的；
1. 完整代码补丁；
1. 新增配置文件；
1. 新增单元测试；
1. 训练启动命令；
1. 推理启动命令；
1. checkpoint 兼容说明；
1. 所有关键张量 shape；
1. 当前实现中仍存在的风险；
1. 后续可选改进，但不要在本次实现。

最终实现应满足：

```
vl_context = encode_vision_language(
    scene_images,
    language,
)

tactile_rgb_tokens = encode_tactile_rgb(
    tactile_rgb,
)

marker_tokens = encode_markers(
    marker_positions,
    marker_reference,
    previous_marker_positions,
    marker_valid_mask,
)

vtla_context = torch.cat(
    [
        vl_context,
        tactile_rgb_tokens,
        marker_tokens,
    ],
    dim=1,
)

predicted_velocity = action_expert(
    noisy_actions=noisy_actions,
    timestep=flow_time,
    context=vtla_context,
    state=robot_state,
    context_attention_mask=vtla_attention_mask,
)
```

核心原则：

> 不修改 LingBot-VLA 2.0 的 Flow Matching 数学过程和动作输出，只加入 TacThru RGB 编码、marker MLP 编码，并把新的触觉 token 添加到 action expert 的条件上下文中。
