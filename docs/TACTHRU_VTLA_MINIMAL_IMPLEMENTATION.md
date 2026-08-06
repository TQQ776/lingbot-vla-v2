# LingBot-VLA 2.0 TacThru marker VTLA

This implementation changes only TacThru marker tokenization, marker position
encoding, marker contact gating, and their configuration/diagnostics. The
following LingBot-VLA paths are unchanged:

- scene RGB and TacThru RGB use the existing shared Qwen3-VL vision encoder;
- the VLM prefix and Action Expert branches keep layer-wise joint Q/K/V attention;
- Action Expert MoE, state/action embeddings, output head, Flow Matching target,
  Euler integration, action chunk, and robot action representation are unchanged;
- depth and future-video distillation remain in their original positions.

## Marker representation and history

For marker position `M_t` and the fixed no-contact reference `M_ref`, in image
pixels, collection and deployment use:

```text
dx = 2 * (x_t - x_ref) / image_width
dy = 2 * (y_t - y_ref) / image_height
```

One dataset row stores `[48,2]` normalized displacement and `[48]` validity.
LeRobot requests offsets `[-7,...,0] / 30` to build `[S,8,48,2]`, oldest to
current. Episode prefixes replicate their earliest frame and never read from a
previous episode. No explicit velocity is stored; temporal change is represented
by the eight displacement frames.

The marker encoder applies training-set z-score internally. Contact score is
always computed from the input displacement before that z-score, so contact
threshold units remain:

```text
2 * (current_xy - reference_xy) / [image_width, image_height]
```

## Global and regional tokenization

`marker_tokenization.mode: global` preserves the legacy architecture:

```text
[B,S,8,48,2] -> one shared per-frame MarkerEncoder -> [B,S,8,D]
```

`marker_tokenization.mode: regional` uses a shared point encoder and a shared
region projection:

```text
[dx,dy,(x_ref,y_ref)] -> shared point MLP
                      -> masked mean/max by region
                      -> shared region projection
                      -> [B,S,8,4,D]
                      -> [B,S,32,D]
```

Invalid markers do not enter mean or max pooling. An all-invalid region emits an
exact zero token with mask `false`. The default `mean_max` aggregation concatenates
masked mean and max before projection. There are no independent per-region MLPs.

The fixed region order is:

```text
0 left-top, 1 right-top, 2 left-bottom, 3 right-bottom
```

Reference coordinates are normalized to `[-1,1]`. The x/y median is the split;
points on a split use right/bottom. Model, dataset gate, calibration, and live
deployment all call `build_marker_region_mapping_numpy()`. The repository copy of
the current single-sensor reference is:

```text
assets/tactile/tacthru_ml_marker_reference_48.json
```

## Position and identity encoding

Content and identity are separate. A regional token is:

```text
soft_gate * content[t,r]
+ temporal_position[t]
+ spatial_position[sensor,r]
+ marker_modality_embedding
+ sensor_embedding[sensor]
```

Temporal and spatial types independently support `none`, `learned`, and
`sincos`. Fixed sin-cos supports odd hidden dimensions by zero-padding unused
dimensions. With `use_real_time: true`, temporal values are
`[-7/30,...,-1/30,0]` seconds rather than integer indices. Spatial sin-cos uses
the mean reference coordinate of the markers assigned to each region.

Final token order is sensor-major, time-major, region-minor. The main model uses
the returned tensor length and never hard-codes 8 or 32.

## Marker-only contact gate

The default gate affects only marker attention masks and marker regional content.
It never disables TacThru RGB:

```text
rgb_mask    = sensor_valid & rgb_valid
marker_mask = sensor_valid & history_valid & region_valid & contact_visible
```

Global OFF makes every marker mask false after all biased layers and embeddings,
so hidden marker tokens are exact zeros. RGB remains visible during marker OFF.
`target: marker_and_rgb` exists only for the explicit negative-control preset.

For each valid marker, amplitude is `sqrt(dx^2 + dy^2)`. Per-region score is the
mean of the largest `topk_markers` amplitudes. A region contact candidate requires
both score over threshold and at least `min_active_markers` over their per-marker
thresholds. Global contact is any active region.

Supported modes are `none`, `hard`, `hard_hysteresis`,
`hard_hysteresis_hold`, and `hard_hysteresis_soft_region`. The default requires
two on frames, three off frames, then exposes three HOLD frames. Regional soft
weights are sigmoid values in `[0,1]` and scale content only; identity/position
embeddings are not scaled.

If fewer than `min_valid_markers_global` markers are tracked, the frame is
UNKNOWN rather than OFF. `hold_previous` preserves the previous visibility for
at most `max_unknown_hold_frames`; the next frame enters HOLD when supported,
otherwise OFF. Episode start/reset always sets the state machine to OFF.

Training does not keep gate state inside model forward. Dataset initialization
runs `MarkerContactGate.run_episode()` in episode order and caches the current
state per row. Live deployment calls the same NumPy-only
`MarkerContactGate.step()` for every new 30 Hz sensor timestamp. Duplicate ring
buffer timestamps are ignored. This avoids random-batch state leakage and keeps
offline/online transitions identical.

## Default main configuration

`configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml` resolves to:

```text
regional, 4 regions, mean_max, include_reference_xy=true
8 frames * 4 regions = 32 marker tokens per sensor
temporal=sincos real-time, spatial=sincos
gate=hard_hysteresis_soft_region, target=marker_only
top-k=3, min active=2, on/off/hold=2/3/3
gate_tactile_rgb=false
```

`TactileVTLAConfig.to_dict()` embeds resolved references and thresholds, so a
saved Hugging Face config remains usable when the original calibration path is
not mounted.

## Contact calibration

Generate no-contact thresholds from declared calibration frames:

```bash
.uv/convert-venv/bin/python tools/compute_marker_contact_stats.py \
  /path/to/insert_ethernet_cable_ml_0721_201.zarr.zip \
  assets/tactile/contact_calibration.json \
  --episodes 0:201 \
  --frames-per-episode 15 \
  --marker-key tacthru_l_marker \
  --sensor-name left \
  --reference-xy assets/tactile/tacthru_ml_marker_reference_48.json
```

Per-marker threshold is `median + mad_multiplier * MAD`. Global on/off are
`median(global regional top-k score) + {6,3} * MAD`, with documented quantile
fallback for zero MAD. `off_threshold < on_threshold` is validated.

The checked-in initial calibration uses the first 15 frames of all 201 episodes
and assumes those frames are pre-contact. This is an explicit experimental
assumption, not a hardware-certified no-contact recording. Recalibrate from a
dedicated unloaded sensor recording before treating thresholds as final.

## YAML-only ablations

Compact overrides are in `configs/vla/tacthru_umi/ablations/presets.yaml`:

```text
rgb_only
marker_global_8_no_gate
rgb_marker_global_8_no_gate
rgb_marker_regional_32_no_pos
rgb_marker_regional_32_time_only
rgb_marker_regional_32_space_only
rgb_marker_regional_32_spatiotemporal
rgb_marker_regional_32_hard_gate
rgb_marker_regional_32_hysteresis
rgb_marker_regional_32_full_gate
rgb_marker_regional_32_gate_rgb_control
marker_history_shuffle
marker_region_shuffle
```

Generate one complete config without duplicating the base YAML:

```bash
python tools/generate_vtla_ablation_config.py \
  rgb_marker_regional_32_full_gate /tmp/full_gate.yaml
```

Run an experiment using the same training path:

```bash
PYTHON=/path/to/training/bin/python \
  bash experiment/tacthru/run_vtla_ablation.sh \
  rgb_marker_regional_32_full_gate
```

## Runtime metadata and diagnostics

Rank zero writes these files before training data loading:

```text
OUTPUT_DIR/resolved_tactile_config.yaml
OUTPUT_DIR/marker_region_mapping.json
```

They contain tokenization, token count, position types, thresholds, gate state
durations, gate target, RGB gate status, calibration/reference provenance,
per-sensor region members and centers, and trainable tactile parameter count.
The same summary is printed and added to W&B when enabled.

`debug_shapes: true` logs raw/regional/flattened shapes and final prefix length
once. Training metrics expose ON/HOLD/OFF/UNKNOWN ratios, regional soft weights,
regional valid counts, marker valid-token ratio, and RGB valid-token ratio.

## Checkpoint compatibility

`load_vtla_checkpoint_state_dict()` is strict outside marker migration:

- global to the same global configuration loads all trainable legacy weights;
- legacy four-channel `[dx,dy,vx,vy]` to two-channel global requires
  `LINGBOT_V2_ALLOW_LEGACY_MARKER_REINIT=1`;
- global to regional reinitializes only the new point/region marker modules;
- learned temporal to fixed/none explicitly ignores the learned table;
- missing learned temporal/spatial tables keep normal PyTorch initialization;
- reference, mapping, fixed position, ablation-order, and threshold buffers are
  always rebuilt from the current resolved config instead of checkpoint values;
- unrelated missing, unexpected, or shape-mismatched keys are fatal.

The report lists loaded, intentionally ignored, intentionally reinitialized,
unexpected, forbidden-missing, shape-mismatched, and configuration-incompatible
keys. Structural migration does not replace training the regional encoder.

## Commands

Main all-201 training:

```bash
CONFIG="$PWD/configs/vla/tacthru_umi/tacthru_umi_vtla_rgb_marker.yaml" \
PYTHON=/path/to/training/bin/python \
bash scripts/train_tacthru_umi_v2.sh \
  data/lerobot/insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_history8_v2 \
  --skip-convert --skip-norm --expected-source-episodes 201 \
  --output-dir output/insert_ethernet_vtla_regional32_contact_all201_v1
```

Focused tests and benchmark:

```bash
python -m pytest -q \
  tests/test_tactile_vtla.py \
  tests/test_tactile_vtla_regional.py \
  tests/test_tacthru_umi_v2_deploy_protocol.py \
  tests/test_tacthru_umi_v2_tactile_source.py \
  tests/test_tacthru_umi_v2_http_e2e.py

python tools/benchmark_marker_encoder.py --device cuda
```
