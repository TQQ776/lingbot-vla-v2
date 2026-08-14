# LingBot-VLA-v2 原始 192-Token → T-Rex 风格三流 Cascaded-MoT 改造任务书

**基线仓库**：`TQQ776/lingbot-vla-v2`
**基线 commit**：`9122411c0d1624439c55eb1b298a0bb9f501e308`
**原则**：必须从这个原始 192-token 基线实现；不要 cherry-pick `192_speed` 或 `192_fast_online`。

---

## 1. 最终目标

把现有两流：

```text
Prefix / Qwen3-VL Understanding Stream
                +
Action Expert Stream
```

扩展为三流：

```text
Prefix / Qwen3-VL Understanding Stream
                +
Action Expert Stream
                +
Tactile Expert Stream
```

三条流逐层对齐：

```text
VLM layer 0    ↔ Action layer 0    ↔ Tactile layer 0
...
VLM layer 35   ↔ Action layer 35   ↔ Tactile layer 35
```

每条流参数独立。每层各自先产生 Q/K/V，拼接后做 Joint Attention，再按 stream 拆分，分别进入各自的 output projection / residual / FFN。

目标 Cascaded Flow：

\[
X_1 \xrightarrow[\tau\in[0.6,1]]{Action} X_{0.6}
\xrightarrow[\tau\in[0,0.6]]{Tactile} X_0
\]

默认：

```yaml
tau_split: 0.6
total_steps: 10
slow_steps: 4
tactile_steps: 6
```

---

## 2. 开工前必须阅读的基线文件

```text
lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py
lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py
lingbotvla/models/vla/lingbot_vla/tactile_vtla.py
lingbotvla/models/vla/lingbot_vla/flex_attention.py
lingbotvla/models/vla/lingbot_vla/utils.py
configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker_point192_history4.yaml
tests/test_tactile_vtla.py
tasks/vla/train_lingbotvla.py
```

先确认这些现状：

- `QwenvlWithExpertV2Model.forward()` 当前只有 VLM + Action 两条 stream。
- 两边逐层各自算 Q/K/V，concat 做 Joint Attention，再按 token span 拆回各自 decoder layer。
- Action Expert = 36 层，且现有代码已经要求 `action_num_layers == vlm_num_layers`。
- 原始 marker：`1 sensor × 4 history × 48 markers = 192 tokens`。
- action/state dim = 55；chunk = 50。
- TacRGB 使用共享 Qwen3-VL ViT。
- `vlm_causal=true`，attention backend=`flex_cached`。
- Marker 当前被放进 Prefix。
- Marker 已经有 temporal + spatial sin/cos position encoding，且包含 reference XY。

---

## 3. Legacy 与新模式必须同时保留

新增配置：

```yaml
tactile_refinement:
  enabled: true
  architecture: three_stream_mot
  mode: cascaded_flow
```

行为：

```text
enabled=false:
    完全保持原始192-token逻辑：
    TacRGB + Marker 都在 Prefix，VLM+Action 两流

enabled=true:
    TacRGB 仍在 Prefix
    Marker 不再进入 Prefix
    Marker 改进入第三条 Tactile stream
```

严禁新模式下 Marker 同时存在于 Prefix 和 Tactile stream。

旧 checkpoint 必须能 `strict=False` 加载，已有 VLM / Action / Marker Encoder 参数名尽量不改。

---

## 4. 三条 stream 的输入

### 4.1 Prefix

```text
Scene RGB
Language
TacRGB
Task Queries / Depth / Future-video Queries
```

即：

\[
P=[Scene,Language,TacRGB,Queries]
\]

新三流模式下 Prefix **不含 Marker192**。

### 4.2 Action stream

保留 LingBot 原 Action Expert 输入语义：

```text
State + Flow-time conditioning + X_tau action tokens
```

动作：

```text
X_tau: [B,50,55]
```

Action Expert 负责：

\[
1 \rightarrow 0.6
\]

不要重写已有 Action MoE。

### 4.3 Tactile stream

严格按 T-Rex fast-flow思想构造：

\[
T_\tau=
[M_{1:192}\mid e_\tau\mid E_X(X_{\tau,1:50})]
\]

即：

```text
192 Marker observation tokens
1   Flow-time token
50  X_tau action tokens
----------------------------
243 tactile-stream tokens
```

第一版先不加入 latest-state token；以后如有需要做成 config option。

---

## 5. Marker 处理

保留现有 `TactileTokenEncoder` / PointSpatiotemporalMarkerEncoder，不重新设计 Marker 特征。

输入：

```text
[B,1,4,48,2]
```

现有 encoder：

```text
→ [B,192,D_vlm]
```

原配置已经有：

```yaml
marker_position_encoding:
  temporal_type: sincos
  spatial_type: sincos
  use_real_time: true
```

所以**禁止再重复加第二套 marker temporal/spatial embedding**。

为保持旧 Marker checkpoint兼容，不要直接把现有 encoder输出维度从 VLM hidden 改成768。新增：

```python
marker_to_tactile_proj = nn.Linear(vlm_hidden_size, tactile_hidden_size)
```

得到：

```text
[B,192,768]
```

---

## 6. Time token 与 X_tau token

新增独立模块：

```python
tactile_time_embedder
tactile_action_in_proj = nn.Linear(55, 768)
tactile_action_out_proj = nn.Linear(768, 55)
```

其中：

```text
tau → tactile_time_embedder → [B,1,768]
X_tau [B,50,55] → tactile_action_in_proj → [B,50,768]
```

可以从现有 Action 对应权重 copy 初始化，但之后必须是独立 Parameter。

最终：

```python
tactile_seq = torch.cat(
    [marker_tokens_768, tactile_time_token, tactile_action_tokens],
    dim=1,
)
# [B,243,768]
```

---

## 7. 新建 36 层 Tactile Expert

最小侵入方案：复用现有 Qwen2 Action Expert decoder-layer接口，再实例化第三套独立 expert：

```python
self.tactile_expert
```

建议 config：

```yaml
tactile_expert:
  hidden_size: 768
  intermediate_size: 1536
  num_layers: 36
  num_attention_heads: 32
  num_key_value_heads: 8
  head_dim: 128
  init_from_action_expert: true
```

要求：

- 36层。
- 每层独立 input norm。
- 独立 Q/K/V/O。
- 独立 post-attention norm。
- 独立 tactile FFN。
- 允许 FFN 比 Action 小。
- 不得共享 Parameter object。

必须 assert：

```python
vlm_layers == action_layers == tactile_layers
```

`init_from_action_expert=true` 时，只做初始 copy：
- Q/K/V/O、Norm shape一致则 copy。
- FFN shape一致才 copy。
- 不一致部分正常初始化。
- copy后参数独立。

---

## 8. 三流 Joint Attention

把原先类似：

```python
inputs_embeds=[prefix_embs, suffix_embs]
```

扩展成三流：

```python
inputs_embeds=[prefix_embs, action_embs, tactile_embs]
```

或者改成显式 stream dict。

每一层：

```text
Prefix hidden  → Prefix Q/K/V
Action hidden  → Action Q/K/V
Tactile hidden → Tactile Q/K/V
```

然后：

\[
Q=[Q_P;Q_A;Q_T],\quad
K=[K_P;K_A;K_T],\quad
V=[V_P;V_A;V_T]
\]

做一次 Joint Attention。

之后严格按动态 span 拆分：

```text
Prefix output  → Prefix decoder layer后半段
Action output  → Action decoder layer后半段
Tactile output → Tactile decoder layer后半段
```

禁止靠写死 192/50 的绝对 index来切整个联合序列；应根据 stream lengths / spans 计算。

---

## 9. Attention Mask

跨 stream 必须满足：

```text
                    K / V
                 P      A      T
Query P          ✓      ✗      ✗
Query A          ✓      ✓      ✗
Query T          ✓      ✓      ✓
```

即：

\[
P\leftarrow P
\]

\[
A\leftarrow P+A
\]

\[
T\leftarrow P+A+T
\]

信息方向：

\[
P\rightarrow A\rightarrow T
\]

严禁：
- Prefix读取Action。
- Prefix读取Tactile。
- Action读取Tactile。

这样 Prefix / Action 的 KV 才可供 Fast tactile重复使用。

### 极重要

**不要简单套一个全局下三角 mask 就完事。**

先检查现有 LingBot：
- Prefix↔Prefix mask。
- Action↔Action mask。
- state/action token关系。
- depth/future-video block mask。

新三流 mask必须只增加跨 stream block约束，**不得无意改变原 stream 内 attention语义**。

建议新增：

```python
build_three_stream_attention_mask(...)
```

并写精确单元测试。

---

## 10. Position IDs / RoPE

保持：
- Prefix 使用现有 Qwen3-VL M-RoPE。
- Action 使用 Prefix 最大有效 position 后的连续位置。
- Tactile 再接在 Action 后面。

逻辑：

```text
Prefix positions
→ Action positions
→ Tactile positions
```

Tactile内部：

```text
Marker192 | Time1 | X_tau50
```

都要有合法 sequence position ids。

Marker已有的 temporal/spatial sincos encoding属于“Marker内容位置”；Transformer RoPE position属于“序列位置”。两者不是一回事，二者都保留。

建议扩展现有 `_build_full_position_ids()` 为三流版本，而不是另写不一致的位置逻辑。

---

## 11. Slow Action Flow：X1 → X0.6

新增 slow-plan/cache结构，例如：

```python
@dataclass(frozen=True)
class CascadedSlowPlan:
    x_split: Tensor
    tau_split: float
    noise: Tensor
    past_key_values: ...
    prefix_len: int
    action_len: int
    prefix_position_ids: Tensor
```

新增类似：

```python
build_cascaded_slow_plan(...)
```

默认：

```text
total_steps=10
slow_steps=4
dt=-0.1
```

Action velocity evaluation的 tau：

```text
1.0
0.9
0.8
0.7
```

4次 Euler update后得到：

```text
X0.6
```

Slow plan需保存：
- immutable `x_split = X0.6`
- Prefix per-layer KV
- Action per-layer KV
- position/meta信息

### 必须刷新 Action KV @ X0.6

最后一个 Euler update后，当前 cache可能还是对应更新前的 X0.7。

所以必须：

```text
得到 X0.6
↓
重新构造 Action sequence (state + tau=0.6 + X0.6)
↓
只重新跑 Action stream
↓
刷新每层 Action KV
```

最终 Fast读取的必须是：

\[
KV_{A@X_{0.6}}
\]

---

## 12. Fast Tactile Flow：X0.6 → X0

新增：

```python
refine_action_with_tactile(...)
```

每个 fast tick：

1. clone slow cache；
2. 从同一个 immutable `slow_plan.x_split` 开始；
3. 编码最新 Marker → 192 tokens；
4. 从 tau=0.6 开始；
5. 每个 Euler step 都重新构造：

\[
[Marker_{192}\mid e_\tau\mid Embed(X_\tau)_{50}]
\]

6. 当前243 tokens全部标记为 Tactile stream；
7. Prefix/Action不重新 forward，只通过 cached K/V参与attention；
8. 每步只取 Tactile输出最后50个 hidden：

```python
action_hidden = tactile_hidden[:, -50:, :]
```

9. 预测：

```python
v_tactile = tactile_action_out_proj(action_hidden)
# [B,50,55]
```

10. Euler：

\[
X \leftarrow X + dt\cdot v_T
\]

11. tau同步减小直到0。

最终 `X0` 就是动作 chunk，**不要做 `slow_action + delta_action`**。

---

## 13. Fast 多次修正与缓存语义

同一 slow plan 下：

```text
Marker(t1) → 从同一个 X0.6 开始 → X0(t1)
Marker(t2) → 从同一个 X0.6 开始 → X0(t2)
Marker(t3) → 从同一个 X0.6 开始 → X0(t3)
```

每次：
- clone cache。
- 不修改 slow plan里的KV。
- 不修改 `x_split`。
- 不把上一次 tactile输出 `X0` 当下一次的 `x_split`。

Fast tick 禁止重新跑：
- ViT
- Prefix Qwen
- Slow Action `1→0.6`

---

## 14. Contact Gate

保留原始：

```yaml
marker_contact_gate:
  target: marker_only
gate_tactile_rgb: false
```

因此：
- TacRGB始终保留在Slow Prefix。
- Marker gate只控制Fast中的 Marker条件。

第一版默认：

```yaml
tactile_refinement:
  gate_off_behavior: action_fallback
```

无接触时：

```text
Action: X1 → X0.6 → X0
```

即 Action Expert接管剩余 `[0,0.6]` flow。

这要求 Action Expert仍保留全区间 `[0,1]` 能力。

必须避免全 masked tactile token造成NaN。

---

## 15. 训练方式

Flow Matching目标：

\[
X_\tau=\tau\epsilon+(1-\tau)A
\]

\[
u=\epsilon-A
\]

### Action cascaded区间

随机：

\[
\tau_A\sim U(0.6,1)
\]

直接解析构造：

\[
X_{\tau_A}
\]

训练：

\[
L_{A,cascade}=\|v_A-u\|^2
\]

### Action full-range fallback能力

为了 gate-off，建议额外保留：

\[
\tau_{full}\sim U(0,1)
\]

可通过低概率/低权重 auxiliary action loss训练，不要求每个batch都跑第二次完整Action forward。

### Tactile区间

随机：

\[
\tau_T\sim U(0,0.6)
\]

解析构造：

\[
X_{\tau_T}=\tau_T\epsilon+(1-\tau_T)A
\]

Tactile输入：

\[
[Marker_{192}\mid e_{\tau_T}\mid X_{\tau_T,50}]
\]

读取 slow Prefix/Action KV，然后预测：

\[
v_T:[B,50,55]
\]

监督同一个：

\[
u=\epsilon-A
\]

损失：

\[
L_T=\|v_T-u\|^2
\]

现有 depth / future-video / MoE losses不得静默删除。

---

## 16. Tactile训练时的 Slow KV

不要把“训练的 X_tau”与“Action KV”混为一件事。

`X_tau` 可以解析构造，不需要先完整 rollout。

第一版建议为 tactile branch 构建 slow context时：

1. Prefix正常forward；
2. 在 boundary `tau=0.6` 解析构造：
   \[
   X_{0.6}^{analytic}=0.6\epsilon+0.4A
   \]
3. 用：
   ```text
   state + tau=0.6 + X0.6_analytic
   ```
   构造 Action KV；
4. tactile branch再随机采 `tau_T∈[0,0.6]`。

---

## 17. Boundary mismatch

训练常见：

\[
X_{0.6}^{train}=0.6\epsilon+0.4A
\]

推理实际：

\[
\hat X_{0.6}^{infer}=ActionRollout(X_1)
\]

两者存在 distribution gap。

新增：

```yaml
rollout_boundary_exposure_prob: 0.25
```

命中时：

```text
no_grad:
X1 → Slow Action rollout → X0.6_hat
→ refresh Action KV
→ 至少用 X0.6_hat 做一次 tau=0.6 tactile boundary loss
```

注意：这只能声称改善 boundary exposure，不要声称已经解决整个 `[0,0.6]` 的 rollout mismatch。

---

## 18. 推荐第一阶段冻结

第一阶段：

```yaml
freeze_vlm: true
freeze_action_expert: true
train_marker_encoder: true
train_marker_to_tactile_proj: true
train_tactile_expert: true
train_tactile_flow_heads: true
```

先验证第三流能利用已有 VLM/Action context学习后半段 flow。

第二阶段再考虑：
- Action最后若干层小学习率解冻；
- LoRA；
- 联合微调。

第一版不要同时大改所有已有能力。

---

## 19. 新配置文件

从：

```text
configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker_point192_history4.yaml
```

复制为例如：

```text
configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker_point192_history4_three_stream_mot.yaml
```

核心建议：

```yaml
train:
  tactile_refinement:
    enabled: true
    architecture: three_stream_mot
    mode: cascaded_flow
    tau_split: 0.6

    expert:
      hidden_size: 768
      intermediate_size: 1536
      num_layers: 36
      num_attention_heads: 32
      num_key_value_heads: 8
      head_dim: 128
      init_from_action_expert: true

    sequence:
      include_marker_tokens: true
      include_time_token: true
      include_action_tokens: true
      include_latest_state: false

    inference:
      total_steps: 10
      slow_steps: 4
      tactile_steps: 6
      gate_off_behavior: action_fallback
      clone_slow_cache_each_fast_tick: true

    training:
      freeze_vlm: true
      freeze_action_expert: true
      tactile_loss_weight: 1.0
      rollout_boundary_exposure_prob: 0.25
```

validate：
- `vlm_layers == action_layers == tactile_layers`
- `tau_split == 1 - slow_steps/total_steps`
- `tactile_steps == total_steps-slow_steps`
- 该配置 Marker tokens = 192
- chunk=50
- action_dim=55

模型内部不要把所有这些数字写死。

---

## 20. 文件修改建议

### `modeling_lingbot_vla_v2.py`

主要负责：
- Tactile Expert config/实例化。
- 三流 `QwenvlWithExpertV2Model.forward()`。
- 三流 spans / position ids。
- Slow plan/cache。
- Action upper flow。
- boundary Action KV refresh。
- tactile sequence builder。
- tactile velocity head。
- tactile lower flow。
- cascaded training branch。
- legacy两流兼容。

### `qwen2_action_expert.py`

优先复用现有 decoder layer：
- `compute_kqv=True`
- `output_atten=True`

如果能直接实例化另一套Qwen2 expert，就不要复制粘贴整个源文件。

### `tactile_vtla.py`

继续只负责 tactile representation：
- Marker192。
- reference XY。
- marker normalization。
- temporal/spatial encoding。
- contact gate。

不要把 Flow Matching逻辑塞进这个文件。

### `flex_attention.py`

扩展三流 block mask，同时保持原 stream内部mask语义。

### `tests/test_tactile_three_stream_mot.py`

新增独立测试文件。

---

## 21. 必须通过的测试

1. **Tactile sequence shape**
   ```text
   Marker=192, Time=1, X_tau=50, total=243
   ```

2. **三流层数**
   ```text
   VLM=36, Action=36, Tactile=36
   ```

3. **参数独立**
   Action/Tactile QKV不能是同一个Parameter object。

4. **Mask**
   ```text
   P sees P only
   A sees P+A
   T sees P+A+T
   ```
   并确认原 Prefix↔Prefix、Action↔Action语义没变。

5. **Fast不重跑Slow**
   已有slow cache后，一次 tactile refine：
   ```text
   VLM call count = 0
   Action call count = 0
   Tactile call count > 0
   ```

6. **x_split immutable**

7. **同输入确定性**（eval模式）

8. **Marker sensitivity**
   同 slow cache / x_split，只改变Marker，fast action应变化。

9. **Tactile velocity shape**
   `[B,50,55]`，且head明确读取最后50个hidden。

10. **Split schedule**
    `10 total / 4 slow / 6 tactile → tau_split=0.6`

11. **Action KV refresh**
    cache必须对应 `tau=0.6, X0.6`。

12. **Gate OFF**
    无NaN，默认Action fallback，输出 `[B,50,55]`。

13. **Legacy baseline**
    `tactile_refinement.enabled=false` 时旧192路径仍工作。

14. **Checkpoint compatibility**
    old checkpoint `strict=False`：
    - old keys正常恢复；
    - new tactile keys为expected missing；
    - 不出现大规模旧参数意外missing。

15. **Gradient**
    第一阶段：
    ```text
    VLM grad=None
    Action grad=None
    Tactile grad!=None
    marker_to_tactile_proj grad!=None
    ```

16. **Boundary exposure**
    probability=1时，真实Slow rollout必须被调用。

---

## 22. 明确禁止

不要：

- cherry-pick后续slow/fast分支。
- 把Tactile Expert做成独立6层cross-attention小网络。
- Marker同时进入Prefix与Tactile。
- 第三流只有Marker而没有 `tau + X_tau`。
- 直接把192个marker hidden硬映射成50步action。
- 改掉现有Action MoE核心逻辑。
- 让Prefix或Action读取Tactile。
- 每个fast tick重新跑ViT/VLM/Action upper flow。
- 把fast输出定义成 `slow_action + delta_action`。
- 训练时强制完整 rollout `1→0.6→0` 才能构造每个样本。
- 重复增加Marker temporal/spatial position encoding。
- 硬编码所有联合序列绝对index。
- 不经测试就改变原flex-attention同流mask语义。
- 声称analytic training已经完全解决boundary mismatch。

---

## 23. 推荐提交顺序

```text
Commit 1: 配置 + 36L Tactile Expert实例化
Commit 2: 三流 Joint Attention + mask tests
Commit 3: Marker从Prefix切到Tactile；实现243-token序列和velocity head
Commit 4: Slow Action 1→0.6 + cache + boundary KV refresh
Commit 5: Fast Tactile 0.6→0 + cache reuse + immutable xsplit
Commit 6: Cascaded training + analytic tau sampling + boundary exposure
Commit 7: Gate fallback + legacy/checkpoint兼容
Commit 8: 全测试 + profiling + 文档
```

---

## 24. 最终实际数据流

```text
                         SLOW
====================================================

Scene RGB ─ ViT ─┐
Language ─────────┤
TacRGB ─── ViT ───┤
Queries ──────────┘
        ↓
Prefix/Qwen stream, 36L
        ↓
cache KV_P

State + tau + X_tau
        ↓
Action Expert, 36L
        ↓
Flow: X1 → X0.6
        ↓
refresh Action KV exactly at X0.6
        ↓
cache KV_A + immutable X0.6


                         FAST
====================================================

Marker history [B,1,4,48,2]
        ↓
existing Marker Encoder
        ↓
192 Marker tokens
        ↓
marker_to_tactile_proj
        ↓

tau → tactile_time_embedder → 1 token

X_tau [B,50,55]
        ↓
tactile_action_in_proj
        ↓
50 action tokens

[Marker192 | tau1 | X_tau50]
        ↓
243 tactile-stream tokens
        ↓

cached KV_P ─┐
cached KV_A ─┼→ Tactile Expert layer l × 36
new Tactile ─┘
        ↓
only last 50 hidden
        ↓
Linear 768→55
        ↓
v_tactile [B,50,55]
        ↓
Euler
        ↓
X0.6 → X0
```

每层跨流可见性：

```text
                   K/V
               P    A    T
P Query        ✓    ✗    ✗
A Query        ✓    ✓    ✗
T Query        ✓    ✓    ✓
```

---

## 25. 最终验收标准

必须同时满足：

\[
oxed{
192\ Marker
+
36L\ TactileExpert
+
P/A/T\ JointAttention
+
单向可缓存Mask
+
[Marker_{192}|\tau|X_{\tau,50}]
+
X_1\xrightarrow{Action}X_{0.6}
\xrightarrow{Tactile}X_0
}
\]

并且：

```text
Fast tactile tick 不重新跑 ViT / Prefix-Qwen / Action upper-flow。
```

---

## 26. 完成后必须给出报告

请逐项输出：

1. 修改文件列表。
2. 每个文件修改内容。
3. 每层三流 tensor shapes。
4. attention mask实际矩阵。
5. 243-token tactile layout。
6. Slow tau序列。
7. Fast tau序列。
8. Action KV refresh具体位置。
9. 训练 Action/Tactile tau采样方式。
10. Gate OFF路径。
11. 旧192 checkpoint加载结果。
12. 冻结/训练参数统计。
13. 所有新增测试及结果。
14. 单次Slow latency。
15. 单次Fast tactile refinement latency。
16. 已知未解决问题。
17. 一张与最终代码一致的数据流图。

---

## 27. 实现状态（`three_vtla`）

实现基线保持为 `marker_token_192` / `9122411`，没有 cherry-pick
`192_speed` 或 `192_fast_online`。

当前已完成：

- 36 层独立 Tactile Expert；
- P/A/T Joint Attention 与单向可缓存 mask；
- `[Marker192 | tau1 | X_tau50]`；
- Action `1 -> 0.6`、边界 Action KV 刷新；
- Tactile `0.6 -> 0`、不可变 `X0.6` 与缓存复用；
- gate-off Action fallback；
- Marker/Gate在Fast Euler循环前只计算一次，Gate OFF跳过Tactile Expert；
- cascaded tau 训练和 rollout boundary exposure；
- 冻结Action阶段的Action loss仅监控，不混入优化loss；
- 带plan age、位姿/夹爪漂移、执行offset和scene version的三级Slow Plan有效性判断；
- 部署侧`reuse / refresh_action / rebuild`在线Slow/Fast调度和分段耗时回传；
- 第一阶段冻结与新模块优化器分组；
- legacy/checkpoint 兼容；
- 专用配置、测试和架构报告。

实现报告和最终数据流图：

```text
docs/VTLA_THREE_STREAM_MOT_ARCHITECTURE.md
```

最新增量验证覆盖模型、Gate、调度、HTTP、in-process、Realman安全路径及
旧VTLA/区域门控，共 `138 passed`、2条既有PyTorch DCP弃用warning。另有
一个基线测试在收集阶段要求当前基线并不存在的 `RetryableInferenceError`，
与本改造文件无关。

仍未完成：必须先训练约415M新增触觉侧参数，并在目标GPU用真实6B checkpoint
测量Slow/Fast/Gate-OFF latency和显存，才能判断是否达到真机实时频率。
