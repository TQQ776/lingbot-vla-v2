# LingBot-VLA 2.0 TacThru 8-frame VTLA implementation

This document records the implemented TacThru VTLA path. The change is limited
to tactile marker conditioning. Tactile RGB still shares the Qwen3-VL ViT with
scene images. The action suffix, action expert, MoE, Flow Matching target and
Euler integration are unchanged.

## Marker representation

For a calibrated marker position `M_t` and fixed no-contact reference
`M_ref`, both in pixels, the client and dataset use:

```text
dx = 2 * (x_t - x_ref) / image_width
dy = 2 * (y_t - y_ref) / image_height
```

One physical dataset row stores one normalized displacement frame:

```text
observation.tactile.marker_displacement_left  [48,2] float32
observation.tactile.marker_valid_left         [48]   bool
```

No explicit `vx,vy` feature is computed. Dynamic information comes from the
eight displacement frames `[D_(t-7), ..., D_t]`.

## Training history

`LeRobotDataset.get_delta_timestamps()` requests offsets `[-7,...,0] / 30`
for every enabled marker displacement and validity feature. LeRobot clamps
queries to the start of the same episode, so history cannot cross an episode
boundary. A prefix shorter than eight frames replicates the episode's earliest
frame. For `t=2` the result is:

```text
[D0,D0,D0,D0,D0,D0,D1,D2]
```

Scene RGB, state and action labels remain aligned to `t`. Replicated padding is
not included when marker normalization statistics are computed.

## Tensor contract

```text
stored dataset row                 [48,2]
LeRobot history query              [8,48,2]
single-sample transform            [S,8,48,2]
collated model input               [B,S,8,48,2]
marker_valid_mask                  [B,S,8,48]
marker_history_valid_mask          [B,S,8]
tactile_sensor_mask                [B,S]
shared per-frame MarkerEncoder     [B,S,8,D_vlm]
sensor-major/time-minor flatten    [B,S*8,D_vlm]
```

The shared MLP receives `48 * 2 = 96` values per frame. Each token is:

```text
MLP(D_sensor,time)
+ marker modality embedding
+ sensor-side embedding
+ temporal embedding[time]
```

The final token mask is the conjunction of sensor validity, history-frame
validity and at least one valid marker. The mask is applied again after all
biased layers and embeddings, so an invalid token is exactly zero.

Token order is sensor-major, time-minor:

```text
left[t-7], ..., left[t], right[t-7], ..., right[t]
```

Marker tokens are ordinary nonvisual prefix tokens. They receive normal
one-dimensional Transformer positions, do not enter the Qwen image mRoPE grid,
and additionally receive the learned marker temporal embedding.

## Prefix path

The implemented prefix order is:

```text
scene visual tokens
language tokens
tactile RGB tokens
marker history tokens
depth/future query tokens, when enabled
```

The VLM and action branches continue to participate in the existing joint
layer-wise attention. No separate tactile cross-attention block was added.

## Live history and protocol

`TacThruSource` maintains an independent `deque(maxlen=8)` for displacement and
validity. It consumes new TacThru ring-buffer frames by capture timestamp at
30 Hz, independently of the slower synchronous policy request loop. The first
valid frame fills all eight slots. New timestamps roll the deque; duplicate
timestamps do not append. Start, close and episode reset clear the history and
reinitialize it from the first current frame.

HTTP protocol v3 transmits:

```text
marker_displacement_history [S,8,48,2]
marker_valid_mask           [S,8,48]
marker_history_valid_mask   [S,8]
marker_sample_hz            30
```

The server rejects the wrong sensor count, history length, marker count,
channel count, mask shape, sampling rate or non-finite values. It never reshapes
or truncates marker history silently.

## Normalization statistics

`tools/compute_tacthru_marker_stats.py` computes only `[dx,dy]` over finite
markers in the selected real frames. Standard deviation is clamped to at least
`eps`. The current all-201 statistics cover 88,363 frames and 4,241,424 valid
markers. A legacy four-channel statistics file fails with an explicit request
to recompute the marker statistics.

## Checkpoint compatibility

A tactile-free base checkpoint may initialize the complete new tactile
namespace during training. A legacy one-token tactile checkpoint is not loaded
silently. Server-side migration requires:

```bash
export LINGBOT_V2_ALLOW_LEGACY_MARKER_REINIT=1
```

The loader then preserves non-marker and compatible tactile weights, while
reinitializing only the marker MLP first layer, its two-channel statistics
buffers and the temporal embedding. Missing, unexpected and mismatched keys are
reported; any unrelated mismatch remains fatal. This migration only makes the
checkpoint structurally loadable and does not replace history8 training.

## Main implementation files

- `lingbotvla/models/vla/lingbot_vla/tactile_vtla.py`: config, statistics,
  per-frame marker MLP, embeddings, masks and legacy checkpoint policy.
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`: marker block
  insertion into the existing prefix.
- `lingbotvla/data/vla_data/base_dataset.py`: same-episode history queries.
- `lingbotvla/data/vla_data/tactile.py`: padding and fixed-shape transform.
- `tools/convert_tacthru_zarr_to_lerobot_v2.py`: explicit per-frame displacement
  dataset and manifest.
- `deploy/tacthru_umi_v2/tactile_source.py`: timestamped live deque.
- `deploy/tacthru_umi_v2/protocol.py`: strict protocol v3 validation.
- `deploy/tacthru_umi_v2/http_server.py`: checkpoint contract and model mapping.

## Data flow

```text
TacThru absolute marker positions + fixed calibrated reference
        -> normalized [dx,dy]
        -> 8-frame episode-safe/timestamp-safe history
        -> [B,S,8,48,2]
        -> shared per-frame Marker MLP
        -> [B,S,8,D_vlm]
        -> modality + sensor + temporal embeddings
        -> [B,S*8,D_vlm]
        -> LingBot prefix and joint layer-wise attention
        -> action expert Flow Matching vector field
        -> unchanged Euler integration and action chunk
```
