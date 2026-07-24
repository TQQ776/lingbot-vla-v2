#!/usr/bin/env bash
# Convert TacThru UMI Zarr data, compute v2 normalization statistics, check
# the native-depth checkpoints, and launch LingBot-VLA 2.0 post-training.

set -euo pipefail
# train.sh pipes torchrun through tee; export pipefail so failures propagate
# back through that child Bash process instead of being hidden by tee.
export SHELLOPTS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/tactile_train_contract.sh"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="$(command -v python || true)"
fi
CONFIG="${CONFIG:-$ROOT_DIR/configs/vla/tacthru_umi/tacthru_umi.yaml}"
NORM_CONFIG="${NORM_CONFIG:-$ROOT_DIR/configs/vla/norm_compute/post_data.yaml}"
CONVERTER="$ROOT_DIR/tools/convert_tacthru_zarr_to_lerobot_v2.py"
READY_CHECK="$ROOT_DIR/scripts/check_tacthru_umi_v2_ready.py"
DATA_NAME="${DATA_NAME:-tacthru_umi_v2}"
ROBOT_CONFIG="${ROBOT_CONFIG:-$ROOT_DIR/configs/robot_configs/${DATA_NAME}.yaml}"
DATASET_TACTILE_MODE="${DATASET_TACTILE_MODE:-none}"
TRAIN_TACTILE_MODE="${TRAIN_TACTILE_MODE:-none}"
TACTILE_TRAIN_STAGE="${TACTILE_TRAIN_STAGE:-full}"
TACTILE_SIDE="${TACTILE_SIDE:-left}"
MARKER_INPUT_SPACE="${MARKER_INPUT_SPACE:-normalized}"
MARKER_IMAGE_WIDTH="${MARKER_IMAGE_WIDTH:-640}"
MARKER_IMAGE_HEIGHT="${MARKER_IMAGE_HEIGHT:-480}"

BASE_MODEL_ASSET_DIR="${BASE_MODEL_ASSET_DIR:-$ROOT_DIR/models/lingbot-vla-v2-6b}"
MODEL_DIR="${MODEL_DIR:-$BASE_MODEL_ASSET_DIR}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT_DIR/models/Qwen3-VL-4B-Instruct}"
NORM_FILE="${NORM_FILE:-$ROOT_DIR/assets/norm_stats/tacthru_umi_v2.json}"
MOGE_PATH="${MOGE_PATH:-$ROOT_DIR/models/moge-2-vitb-normal/model.pt}"
MORGBD_PATH="${MORGBD_PATH:-$BASE_MODEL_ASSET_DIR/depth/model.pt}"
DINO_CKPT="${DINO_CKPT:-$BASE_MODEL_ASSET_DIR/dino_video/teacher_step_10000.pth}"
DINO_CONFIG="${DINO_CONFIG:-$BASE_MODEL_ASSET_DIR/dino_video/config.yaml}"
NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_tacthru_umi_v2.sh DATASET [options] [-- train overrides]

DATASET is a TacThru .zarr/.zarr.zip path. An already converted local
LeRobot v3 directory is also accepted.

Options:
  --task TEXT                  Dataset language instruction (default: Pull the tissue).
  --lerobot-dir PATH           Converted dataset directory.
  --output-dir PATH            Training output directory.
  --repo-id ID                 LeRobot repo id recorded by the converter.
  --max-source-episodes N      Convert only the first N source episodes.
  --expected-source-episodes N Require this many source episodes (default: 100).
  --overwrite                  Recreate the converted LeRobot directory.
  --skip-convert               Treat DATASET as an existing LeRobot directory.
  --skip-norm                  Reuse assets/norm_stats/tacthru_umi_v2.json.
  --check-only                 Prepare and validate, but do not start training.
  -h, --help                   Show this help.

Any arguments after -- are passed to train_lingbotvla.py. For example:
  -- --train.max_steps 1 --train.enable_resume false

Environment overrides:
  PYTHON, CONFIG, DATA_NAME, ROBOT_CONFIG, DATASET_TACTILE_MODE,
  TRAIN_TACTILE_MODE, TACTILE_TRAIN_STAGE, TACTILE_SIDE,
  MARKER_INPUT_SPACE, MARKER_IMAGE_WIDTH, MARKER_IMAGE_HEIGHT,
  BASE_MODEL_ASSET_DIR, MODEL_DIR, TOKENIZER_DIR, NORM_FILE, MOGE_PATH,
  MORGBD_PATH, DINO_CKPT, DINO_CONFIG, NORM_NUM_WORKERS,
  CUDA_VISIBLE_DEVICES, MASTER_PORT, TORCHRUN
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

INPUT="$1"
shift

TASK="Pull the tissue"
LEROBOT_DIR=""
OUTPUT_DIR=""
REPO_ID=""
MAX_SOURCE_EPISODES=""
EXPECTED_SOURCE_EPISODES=100
EXPECTED_SOURCE_EPISODES_SET=0
OVERWRITE=0
SKIP_CONVERT=0
SKIP_NORM=0
CHECK_ONLY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires a value" >&2; exit 2; }
      TASK="$2"; shift 2 ;;
    --lerobot-dir)
      [[ $# -ge 2 ]] || { echo "--lerobot-dir requires a value" >&2; exit 2; }
      LEROBOT_DIR="$2"; shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"; shift 2 ;;
    --repo-id)
      [[ $# -ge 2 ]] || { echo "--repo-id requires a value" >&2; exit 2; }
      REPO_ID="$2"; shift 2 ;;
    --max-source-episodes)
      [[ $# -ge 2 ]] || { echo "--max-source-episodes requires a value" >&2; exit 2; }
      MAX_SOURCE_EPISODES="$2"; shift 2 ;;
    --expected-source-episodes)
      [[ $# -ge 2 ]] || { echo "--expected-source-episodes requires a value" >&2; exit 2; }
      EXPECTED_SOURCE_EPISODES="$2"; EXPECTED_SOURCE_EPISODES_SET=1; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    --skip-norm) SKIP_NORM=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown wrapper option: $1 (put training overrides after --)" >&2; exit 2 ;;
  esac
done

if [[ -n "$MAX_SOURCE_EPISODES" && "$EXPECTED_SOURCE_EPISODES_SET" -eq 0 ]]; then
  EXPECTED_SOURCE_EPISODES="$MAX_SOURCE_EPISODES"
fi

case "$DATASET_TACTILE_MODE" in
  none) DATASET_HAS_TACTILE_RGB=false; DATASET_HAS_TACTILE_MARKER=false ;;
  rgb) DATASET_HAS_TACTILE_RGB=true; DATASET_HAS_TACTILE_MARKER=false ;;
  marker) DATASET_HAS_TACTILE_RGB=false; DATASET_HAS_TACTILE_MARKER=true ;;
  rgb-marker) DATASET_HAS_TACTILE_RGB=true; DATASET_HAS_TACTILE_MARKER=true ;;
  *) echo "DATASET_TACTILE_MODE must be none, rgb, marker, or rgb-marker" >&2; exit 2 ;;
esac
case "$TRAIN_TACTILE_MODE" in
  none) TRAIN_TACTILE_RGB=false; TRAIN_TACTILE_MARKER=false ;;
  rgb) TRAIN_TACTILE_RGB=true; TRAIN_TACTILE_MARKER=false ;;
  marker) TRAIN_TACTILE_RGB=false; TRAIN_TACTILE_MARKER=true ;;
  rgb-marker) TRAIN_TACTILE_RGB=true; TRAIN_TACTILE_MARKER=true ;;
  *) echo "TRAIN_TACTILE_MODE must be none, rgb, marker, or rgb-marker" >&2; exit 2 ;;
esac
case "$TACTILE_TRAIN_STAGE" in
  adapters|expert|full) ;;
  *) echo "TACTILE_TRAIN_STAGE must be adapters, expert, or full" >&2; exit 2 ;;
esac
if [[ "$TRAIN_TACTILE_RGB" == true && "$DATASET_HAS_TACTILE_RGB" != true ]]; then
  echo "Training requests tactile RGB, but DATASET_TACTILE_MODE=$DATASET_TACTILE_MODE does not contain it." >&2
  exit 2
fi
if [[ "$TRAIN_TACTILE_MARKER" == true && "$DATASET_HAS_TACTILE_MARKER" != true ]]; then
  echo "Training requests marker input, but DATASET_TACTILE_MODE=$DATASET_TACTILE_MODE does not contain it." >&2
  exit 2
fi
if [[ "$TRAIN_TACTILE_MODE" == none && "$TACTILE_TRAIN_STAGE" != full ]]; then
  echo "TACTILE_TRAIN_STAGE must be full when TRAIN_TACTILE_MODE=none." >&2
  exit 2
fi
if ! [[ "$MARKER_IMAGE_WIDTH" =~ ^[1-9][0-9]*$ && "$MARKER_IMAGE_HEIGHT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MARKER_IMAGE_WIDTH/HEIGHT must be positive integers." >&2
  exit 2
fi
if [[ "$DATASET_TACTILE_MODE" != none || "$TRAIN_TACTILE_MODE" != none ]]; then
  tactile_validate_train_overrides "${EXTRA_ARGS[@]}"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/the/v2/environment/bin/python if needed." >&2
  exit 1
fi
if [[ ! "$NORM_NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NORM_NUM_WORKERS must be a positive integer, got: $NORM_NUM_WORKERS" >&2
  exit 2
fi
export PATH="$(dirname "$PYTHON"):$PATH"
if [[ -n "${TORCHRUN:-}" ]]; then
  TORCHRUN_BIN="$TORCHRUN"
else
  TORCHRUN_BIN="$(dirname "$PYTHON")/torchrun"
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  echo "torchrun was not found next to the selected Python: $TORCHRUN_BIN" >&2
  echo "Install PyTorch in that environment or set TORCHRUN=/full/path/to/torchrun." >&2
  exit 1
fi
export TORCHRUN="$TORCHRUN_BIN"

for required in "$CONFIG" "$NORM_CONFIG" "$ROBOT_CONFIG" "$CONVERTER" "$READY_CHECK" "$ROOT_DIR/train.sh"; do
  if [[ ! -f "$required" ]]; then
    echo "Required project file is missing: $required" >&2
    exit 1
  fi
done

if [[ ! -e "$INPUT" ]]; then
  echo "Dataset does not exist: $INPUT" >&2
  exit 1
fi
INPUT_ABS="$(readlink -f "$INPUT")"

if [[ -z "$LEROBOT_DIR" ]]; then
  STEM="$(basename "$INPUT_ABS")"
  STEM="${STEM%.zarr.zip}"
  STEM="${STEM%.zarr}"
  STEM="$(printf '%s' "$STEM" | tr -cs '[:alnum:]_.-' '_')"
  LEROBOT_DIR="$ROOT_DIR/data/lerobot/${STEM}_${DATA_NAME}"
fi
LEROBOT_DIR="$(readlink -m "$LEROBOT_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/output/$(basename "$LEROBOT_DIR")"
fi
OUTPUT_DIR="$(readlink -m "$OUTPUT_DIR")"
MODEL_DIR="$(readlink -m "$MODEL_DIR")"
TOKENIZER_DIR="$(readlink -m "$TOKENIZER_DIR")"
NORM_FILE="$(readlink -m "$NORM_FILE")"
MOGE_PATH="$(readlink -m "$MOGE_PATH")"
MORGBD_PATH="$(readlink -m "$MORGBD_PATH")"
DINO_CKPT="$(readlink -m "$DINO_CKPT")"
DINO_CONFIG="$(readlink -m "$DINO_CONFIG")"

mkdir -p "$(dirname "$LEROBOT_DIR")" "$(dirname "$NORM_FILE")" "$OUTPUT_DIR"

if [[ -f "$INPUT_ABS/meta/info.json" ]]; then
  LEROBOT_DIR="$INPUT_ABS"
elif [[ "$SKIP_CONVERT" -eq 1 ]]; then
  if [[ ! -f "$LEROBOT_DIR/meta/info.json" ]]; then
    echo "--skip-convert requires DATASET or --lerobot-dir to be an existing LeRobot directory." >&2
    echo "Missing: $LEROBOT_DIR/meta/info.json" >&2
    exit 1
  fi
else
  CONVERT_ARGS=(
    "$PYTHON" "$CONVERTER" "$INPUT_ABS" "$LEROBOT_DIR"
    --task "$TASK"
  )
  if [[ -n "$REPO_ID" ]]; then CONVERT_ARGS+=(--repo-id "$REPO_ID"); fi
  if [[ -n "$MAX_SOURCE_EPISODES" ]]; then
    CONVERT_ARGS+=(--max-source-episodes "$MAX_SOURCE_EPISODES")
  fi
  if [[ "$DATASET_HAS_TACTILE_RGB" == true ]]; then
    CONVERT_ARGS+=(--include-tactile-rgb)
  fi
  if [[ "$DATASET_HAS_TACTILE_MARKER" == true ]]; then
    CONVERT_ARGS+=(
      --include-tactile-marker
      --marker-input-space "$MARKER_INPUT_SPACE"
      --marker-image-width "$MARKER_IMAGE_WIDTH"
      --marker-image-height "$MARKER_IMAGE_HEIGHT"
    )
  fi
  if [[ "$DATASET_TACTILE_MODE" != none ]]; then
    CONVERT_ARGS+=(--tactile-side "$TACTILE_SIDE")
  fi
  if [[ "$OVERWRITE" -eq 1 ]]; then CONVERT_ARGS+=(--overwrite); fi
  "${CONVERT_ARGS[@]}"
fi

if [[ ! -f "$LEROBOT_DIR/meta/info.json" ]]; then
  echo "Converted LeRobot metadata is missing: $LEROBOT_DIR/meta/info.json" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CHECK_ARGS=(
  "$PYTHON" "$READY_CHECK"
  --dataset "$LEROBOT_DIR"
  --model "$MODEL_DIR"
  --tokenizer "$TOKENIZER_DIR"
  --moge "$MOGE_PATH"
  --morgbd "$MORGBD_PATH"
  --dino-checkpoint "$DINO_CKPT"
  --dino-config "$DINO_CONFIG"
  --expected-source-episodes "$EXPECTED_SOURCE_EPISODES"
  --dataset-tactile-mode "$DATASET_TACTILE_MODE"
  --tactile-mode "$TRAIN_TACTILE_MODE"
  --tactile-config "$CONFIG"
)

# The official v2 README asks for the complete Qwen3-VL checkpoint, not only
# its tokenizer/processor assets. Keep this stricter than the generic checker.
if ! compgen -G "$TOKENIZER_DIR/*.safetensors" >/dev/null; then
  echo "Qwen3-VL safetensors are missing: $TOKENIZER_DIR/*.safetensors" >&2
  echo "Download the complete Qwen/Qwen3-VL-4B-Instruct repository on the server." >&2
  exit 1
fi

# Fail early on an incompatible conversion or missing official checkpoint,
# before spending time computing normalization statistics.
"${CHECK_ARGS[@]}"

TACTILE_CONFIG_JSON="$("$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream) or {}
data = config.get("data") or {}
train = config.get("train") or {}
print(json.dumps({
    "params": train.get("tactile_params") or {},
    "rgb_key": data.get("tactile_rgb_key", "observation.images.tactile_left"),
    "marker_key": data.get("tactile_marker_key", "observation.tactile.marker_flow_left"),
    "marker_valid_key": data.get("tactile_marker_valid_key", "observation.tactile.marker_valid_left"),
    "timestamp_key": data.get("tactile_timestamp_key", "observation.tactile.timestamp"),
}, separators=(",", ":")))
PY
)"
TACTILE_PARAMS_JSON="$("$PYTHON" - "$TACTILE_CONFIG_JSON" <<'PY'
import json, sys
print(json.dumps(json.loads(sys.argv[1])["params"], separators=(",", ":")))
PY
)"
TACTILE_RGB_KEY="$("$PYTHON" - "$TACTILE_CONFIG_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["rgb_key"])
PY
)"
TACTILE_MARKER_KEY="$("$PYTHON" - "$TACTILE_CONFIG_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["marker_key"])
PY
)"
TACTILE_MARKER_VALID_KEY="$("$PYTHON" - "$TACTILE_CONFIG_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["marker_valid_key"])
PY
)"
TACTILE_TIMESTAMP_KEY="$("$PYTHON" - "$TACTILE_CONFIG_JSON" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["timestamp_key"])
PY
)"

if [[ "$SKIP_NORM" -eq 0 ]]; then
  rm -f "$NORM_FILE"
  CUDA_VISIBLE_DEVICES="${NORM_CUDA_VISIBLE_DEVICES:-0}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${NORM_MASTER_PORT:-62501}" \
    bash "$ROOT_DIR/train.sh" "$ROOT_DIR/scripts/compute_norm_stats.py" "$NORM_CONFIG" \
      --data.data_name "$DATA_NAME" \
      --data.train_path "$LEROBOT_DIR" \
      --data.robot_config_root "$ROOT_DIR/configs/robot_configs" \
      --data.norm_path "$NORM_FILE" \
      --data.tactile_rgb_key "$TACTILE_RGB_KEY" \
      --data.tactile_marker_key "$TACTILE_MARKER_KEY" \
      --data.tactile_marker_valid_key "$TACTILE_MARKER_VALID_KEY" \
      --data.tactile_timestamp_key "$TACTILE_TIMESTAMP_KEY" \
      --data.data_ratio_for_norm_compute 1 \
      --data.num_workers "$NORM_NUM_WORKERS" \
      --train.chunk_size 50 \
      --train.micro_batch_size 32 \
      --train.tactile_rgb_enabled false \
      --train.tactile_marker_enabled "$DATASET_HAS_TACTILE_MARKER" \
      --train.tactile_params "$TACTILE_PARAMS_JSON" \
      --train.output_dir "$OUTPUT_DIR/norm_run"
fi

if [[ ! -f "$NORM_FILE" ]]; then
  echo "Normalization statistics are missing: $NORM_FILE" >&2
  echo "Remove --skip-norm or provide the stats at the path above." >&2
  exit 1
fi

"${CHECK_ARGS[@]}" --norm "$NORM_FILE"

# The tactile launcher creates this manifest before preparation. Complete it
# only after both the converted-dataset contract and normalization file exist.
if [[ -f "$OUTPUT_DIR/tactile_experiment.json" ]]; then
  "$PYTHON" - \
    "$OUTPUT_DIR/tactile_experiment.json" \
    "$LEROBOT_DIR/tacthru_umi_v2_conversion.json" \
    "$NORM_FILE" "$CONFIG" "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

experiment_path, dataset_manifest, norm_path, config_path, root = map(Path, sys.argv[1:])
for required in (experiment_path, dataset_manifest, norm_path, config_path):
    if not required.is_file():
        raise SystemExit(f"Cannot finalize tactile experiment contract; missing {required}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = json.loads(experiment_path.read_text(encoding="utf-8"))
existing_contract_sha256 = payload.get("contract_sha256")
dataset_manifest_sha256 = sha256(dataset_manifest)
payload["dataset_manifest_sha256"] = dataset_manifest_sha256
payload["norm_stats_sha256"] = sha256(norm_path)
payload["config_sha256"] = sha256(config_path)
config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
tactile_params = (config_payload.get("train") or {}).get("tactile_params") or {}
if payload.get("train_tactile_mode") in {"rgb", "rgb-marker"}:
    backbone_value = tactile_params.get("rgb_backbone_path")
    if not backbone_value:
        raise SystemExit("RGB tactile experiment is missing train.tactile_params.rgb_backbone_path")
    backbone_path = Path(str(backbone_value)).expanduser()
    if not backbone_path.is_absolute():
        backbone_path = root / backbone_path
    if not backbone_path.is_file():
        raise SystemExit(f"RGB tactile backbone is missing: {backbone_path}")
    payload["rgb_backbone_path"] = str(backbone_path.resolve())
    payload["rgb_backbone_sha256"] = sha256(backbone_path)
else:
    payload["rgb_backbone_path"] = None
    payload["rgb_backbone_sha256"] = None
try:
    payload["git_head"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    payload["git_head"] = None

contract_fields = {
    key: payload.get(key)
    for key in (
        "schema_version",
        "dataset_tactile_mode",
        "train_tactile_mode",
        "tactile_train_stage",
        "source",
        "lerobot_dataset",
        "norm_stats",
        "initial_model",
        "base_model_assets",
        "train_overrides",
        "config_sha256",
        "dataset_manifest_sha256",
        "norm_stats_sha256",
        "rgb_backbone_sha256",
        "git_head",
    )
}
candidate_contract_sha256 = hashlib.sha256(
    json.dumps(contract_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
if existing_contract_sha256 and existing_contract_sha256 != candidate_contract_sha256:
    raise SystemExit(
        "Refusing to mutate or repair a finalized tactile experiment contract; "
        "use a new output directory when dataset, norm, code, model, "
        "configuration, or overrides change."
    )
payload["contract_sha256"] = candidate_contract_sha256

# Keep a content snapshot beside the experiment contract. The deployment
# process can then verify the actual manifest it is using instead of trusting
# only a path and two copied hash strings.
manifest_snapshot = experiment_path.parent / "tactile_dataset_manifest.json"
if manifest_snapshot.is_file():
    if sha256(manifest_snapshot) != dataset_manifest_sha256:
        raise SystemExit(
            "Refusing to replace a finalized tactile dataset manifest snapshot: "
            f"{manifest_snapshot}"
        )
else:
    shutil.copyfile(dataset_manifest, manifest_snapshot)
payload["dataset_manifest_snapshot"] = str(manifest_snapshot)
experiment_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
fi

TACTILE_CONTRACT_ARGS=()
if [[ "$DATASET_TACTILE_MODE" != none ]]; then
  EXPERIMENT_MANIFEST="$OUTPUT_DIR/tactile_experiment.json"
  if [[ ! -f "$EXPERIMENT_MANIFEST" ]]; then
    echo "Tactile training requires the versioned launcher contract: $EXPERIMENT_MANIFEST" >&2
    echo "Use scripts/train_tacthru_umi_v2_tactile.sh instead of bypassing the tactile launcher." >&2
    exit 1
  fi
  mapfile -t TACTILE_CONTRACT_VALUES < <("$PYTHON" - "$EXPERIMENT_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "contract_sha256",
    "dataset_manifest_sha256",
    "rgb_backbone_sha256",
):
    print(payload.get(key) or "")
PY
)
  if [[ ${#TACTILE_CONTRACT_VALUES[@]} -ne 3 || -z "${TACTILE_CONTRACT_VALUES[0]}" || -z "${TACTILE_CONTRACT_VALUES[1]}" ]]; then
    echo "Incomplete tactile experiment contract: $EXPERIMENT_MANIFEST" >&2
    exit 1
  fi
  TACTILE_CONTRACT_ARGS=(
    --train.tactile_experiment_contract_sha256 "${TACTILE_CONTRACT_VALUES[0]}"
    --train.tactile_dataset_manifest_sha256 "${TACTILE_CONTRACT_VALUES[1]}"
  )
  if [[ -n "${TACTILE_CONTRACT_VALUES[2]}" ]]; then
    TACTILE_CONTRACT_ARGS+=(
      --train.tactile_rgb_backbone_sha256 "${TACTILE_CONTRACT_VALUES[2]}"
    )
  fi
fi

echo "TacThru UMI v2 preparation is complete."
echo "  dataset: $LEROBOT_DIR"
echo "  norm:    $NORM_FILE"
echo "  model:   $MODEL_DIR"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "check-only complete; training was not started."
  exit 0
fi

"${CHECK_ARGS[@]}" --norm "$NORM_FILE" --require-gpu --require-flash

# align_params is a dictionary-valued CLI option, so update the complete JSON
# object to make teacher-path environment overrides effective during training.
ALIGN_PARAMS_JSON="$("$PYTHON" - "$CONFIG" "$MOGE_PATH" "$MORGBD_PATH" "$DINO_CKPT" "$DINO_CONFIG" <<'PY'
import json
import sys

import yaml

config_path, moge_path, morgbd_path, dino_ckpt, dino_config = sys.argv[1:]
with open(config_path, encoding="utf-8") as stream:
    align = yaml.safe_load(stream)["train"]["align_params"]
align["depth"]["moge_path"] = moge_path
align["depth"]["morgbd_path"] = morgbd_path
align["video"]["ckpt_path"] = dino_ckpt
align["video"]["config_path"] = dino_config
print(json.dumps(align, separators=(",", ":")))
PY
)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
MASTER_PORT="${MASTER_PORT:-62500}" \
  bash "$ROOT_DIR/train.sh" "$ROOT_DIR/tasks/vla/train_lingbotvla.py" "$CONFIG" \
    --model.model_path "$MODEL_DIR" \
    --model.tokenizer_path "$TOKENIZER_DIR" \
    --data.data_name "$DATA_NAME" \
    --data.train_path "$LEROBOT_DIR" \
    --data.robot_config_root "$ROOT_DIR/configs/robot_configs" \
    --data.norm_stats_file "$NORM_FILE" \
    --data.tactile_rgb_key "$TACTILE_RGB_KEY" \
    --data.tactile_marker_key "$TACTILE_MARKER_KEY" \
    --data.tactile_marker_valid_key "$TACTILE_MARKER_VALID_KEY" \
    --data.tactile_timestamp_key "$TACTILE_TIMESTAMP_KEY" \
    --train.output_dir "$OUTPUT_DIR" \
    --train.align_params "$ALIGN_PARAMS_JSON" \
    --train.tactile_rgb_enabled "$TRAIN_TACTILE_RGB" \
    --train.tactile_marker_enabled "$TRAIN_TACTILE_MARKER" \
    --train.tactile_params "$TACTILE_PARAMS_JSON" \
    --train.tactile_train_stage "$TACTILE_TRAIN_STAGE" \
    "${TACTILE_CONTRACT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
