# VTLA Three-Stream Cascaded-MoT Architecture

This document reports the implementation derived from
`lingbot_192token_to_trex_three_stream_mot_prompt.md` on the
`marker_token_192` baseline.

## Data Flow

```mermaid
flowchart TD
    RGB[Scene RGB] --> VIT[Shared Qwen3-VL ViT]
    TRGB[TacRGB] --> VIT
    LANG[Language and task queries] --> P[Prefix / Qwen3-VL stream, 36 layers]
    VIT --> P
    P --> KVP[Cache Prefix K/V]

    STATE[State] --> A[Action Expert stream, 36 layers]
    NOISE[X1] --> A
    KVP --> A
    A -->|Euler at 1.0, 0.9, 0.8, 0.7| XS[X0.6]
    XS --> REFRESH[Re-run Action at tau=0.6 and X0.6]
    STATE --> REFRESH
    KVP --> REFRESH
    REFRESH --> KVA[Cache Action K/V at boundary]

    MARKER[1 sensor x 4 frames x 48 markers] --> ME[Existing Marker encoder]
    ME --> MP[marker_to_tactile_proj]
    MP --> TS[Marker192 | tau1 | X_tau50]
    XS --> TS
    KVP --> T[Tactile Expert stream, 36 layers]
    KVA --> T
    TS --> T
    T --> LAST[Last 50 tactile hidden states]
    LAST --> HEAD[Linear 768 to 55]
    HEAD -->|Euler at 0.6, 0.5, ..., 0.1| X0[X0 action chunk]
```

The per-layer cross-stream visibility is:

| Query / K-V | Prefix | Action | Tactile |
|---|---:|---:|---:|
| Prefix | yes | no | no |
| Action | yes | yes | no |
| Tactile | yes | yes | yes |

The Prefix-Prefix and Action-Action submatrices are copied from the legacy
two-stream mask. Only the new cross-stream blocks are added.

## Tensor Contract

| Stream | Input shape | Hidden size |
|---|---|---:|
| Prefix | `[B, P, D_vlm]` | Qwen3-VL hidden size |
| Action | `[B, 51, 768]` (`State1 + Action50`) | 768 |
| Tactile | `[B, 243, 768]` | 768 |

The Tactile layout is fixed by configuration rather than absolute joint-sequence
indices:

```text
[0, 192)   existing Marker observation tokens
[192, 193) independent tactile flow-time token
[193, 243) projected X_tau action tokens
```

Existing Marker temporal/spatial sin/cos encoding and reference XY content are
retained. Transformer M-RoPE sequence positions continue in the order Prefix,
Action, Tactile and do not replace Marker content positions.

## Slow and Fast Schedules

Slow Action Euler evaluations:

```text
tau = 1.0, 0.9, 0.8, 0.7
X1 -> X0.9 -> X0.8 -> X0.7 -> X0.6
```

After the final update, Action is evaluated again using exactly
`state + tau=0.6 + X0.6`. Its per-layer K/V is appended to the immutable Prefix
cache. Therefore Fast never reads stale Action K/V from `X0.7`.

Fast Tactile Euler evaluations:

```text
tau = 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
X0.6 -> X0.5 -> ... -> X0
```

Every Fast call clones the cache containers and starts from a clone of the same
`CascadedSlowPlan.x_split`. It does not feed the previous Fast result back as a
new boundary.

## Training

Action cascade training samples `tau_A ~ U(0.6, 1)` and constructs
`X_tau = tau * noise + (1 - tau) * action` analytically. Tactile training samples
`tau_T ~ U(0, 0.6)` and uses the same Flow Matching target `noise - action`.

The Tactile branch builds Action K/V at the analytic `X0.6`. With probability
`0.25`, a no-gradient Slow Action rollout supplies the boundary and a tactile
loss is evaluated at `tau=0.6`. This improves boundary exposure but does not
claim to eliminate the complete rollout distribution mismatch.

The first-stage YAML freezes Qwen3-VL and Action Expert. Trainable modules are:

- existing Marker encoder parameters;
- `marker_to_tactile_proj`;
- independent 36-layer Tactile Expert;
- `tactile_time_embedder`;
- tactile Action input/output projections.

Resolved parameter counts for the configured Qwen3-VL hidden size 2560 were
computed on the `meta` device (shape allocation only):

| Trainable component | Parameters |
|---|---:|
| Existing tactile/Marker encoder | 1,396,928 |
| 36-layer Tactile Expert | 410,796,288 |
| Marker to Tactile projection | 1,966,848 |
| Tactile time embedder | 1,181,184 |
| Tactile Action input/output heads | 85,303 |
| Total first-stage tactile trainable | 415,426,551 |

Depth, future-depth, future-video, Action MoE metrics, and router losses remain
in the original return path. The optional full-range Action auxiliary loss is
implemented but disabled in the frozen-Action first-stage configuration; the
pretrained Action Expert retains its original full-range ability.

## Contact Gate and Legacy Mode

TacRGB stays in Prefix because `gate_tactile_rgb=false` and the gate target is
`marker_only`. When contact is off, inference runs Action Expert from `X0.6` to
`X0`; it does not execute an all-masked Tactile sequence. Mixed batches select
Tactile or Action fallback per sample.

When `tactile_refinement.enabled=false`, Marker remains in Prefix and the exact
legacy VLM+Action two-stream path is used.

Old 192-token checkpoints can be loaded with `strict=False`; original VLM,
Action, and Marker names are unchanged and new Tactile parameters are expected
missing keys. The project weight loader recognizes those missing namespaces and
copies shape-compatible loaded Action decoder/projection values into independent
Tactile parameters. A newly migrated checkpoint still requires Tactile training
before deployment.

## Files

- `tactile_action_expert.py`: validated configuration, independent expert
  construction, Slow plan, cache cloning, copy initialization.
- `modeling_lingbot_vla_v2.py`: three-stream Joint Attention, 243-token builder,
  training, Slow/refresh/Fast inference, gate fallback, public online APIs.
- `utils.py`: exact P/A/T visibility matrix while preserving legacy submasks.
- `configuration_lingbot_vla.py`: serialized model configuration and first-stage
  shape validation.
- `optim/vtla.py` and `train_lingbotvla.py`: optimizer groups, freezing and
  runtime metadata.
- `test_tactile_three_stream_mot.py`: architecture and behavioral tests.

## Verification and Profiling

Unit tests verify the 243-token shape, 36/36/36 layer contract, independent
parameters, mask matrix, Marker sensitivity, deterministic Fast behavior,
immutable boundary/cache semantics, gate fallback, schedule, checkpoint
compatibility, and first-stage gradients.

Slow and Fast latency must be profiled on the target GPU with the actual 6B
checkpoint. This development environment does not contain the Qwen3-VL assets
or CUDA runtime needed to report meaningful hardware latency.

Known remaining system-level work: the model exposes separate
`build_cascaded_slow_plan()` and `refine_action_with_tactile()` APIs, but a
robot controller must schedule those APIs at its desired Slow/Fast rates and
replace a Slow plan when observations or the executed horizon make it stale.
