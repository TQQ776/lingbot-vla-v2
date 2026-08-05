#!/usr/bin/env bash
# Convert TacThru UMI Zarr data, compute v2 normalization statistics, check
# the native-depth checkpoints, and launch LingBot-VLA 2.0 post-training.

set -euo pipefail
# train.sh pipes torchrun through tee; export pipefail so failures propagate
# back through that child Bash process instead of being hidden by tee.
export SHELLOPTS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="$(command -v python || true)"
fi
CONFIG="${CONFIG:-$ROOT_DIR/configs/vla/tacthru_umi/tacthru_umi.yaml}"
NORM_CONFIG="$ROOT_DIR/configs/vla/norm_compute/post_data.yaml"
CONVERTER="$ROOT_DIR/tools/convert_tacthru_zarr_to_lerobot_v2.py"
READY_CHECK="$ROOT_DIR/scripts/check_tacthru_umi_v2_ready.py"
ROBOT_CONFIG="$ROOT_DIR/configs/robot_configs/tacthru_umi_v2.yaml"

MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/lingbot-vla-v2-6b}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT_DIR/models/Qwen3-VL-4B-Instruct}"
NORM_FILE="${NORM_FILE:-$ROOT_DIR/assets/norm_stats/tacthru_umi_v2.json}"
MOGE_PATH="${MOGE_PATH:-$ROOT_DIR/models/moge-2-vitb-normal/model.pt}"
MORGBD_PATH="${MORGBD_PATH:-$MODEL_DIR/depth/model.pt}"
DINO_CKPT="${DINO_CKPT:-$MODEL_DIR/dino_video/teacher_step_10000.pth}"
DINO_CONFIG="${DINO_CONFIG:-$MODEL_DIR/dino_video/config.yaml}"
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
  PYTHON, CONFIG, MODEL_DIR, TOKENIZER_DIR, NORM_FILE, MOGE_PATH,
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
  LEROBOT_DIR="$ROOT_DIR/data/lerobot/${STEM}_tacthru_umi_v2"
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

if [[ "$SKIP_CONVERT" -eq 1 || -f "$INPUT_ABS/meta/info.json" ]]; then
  LEROBOT_DIR="$INPUT_ABS"
else
  CONVERT_ARGS=(
    "$PYTHON" "$CONVERTER" "$INPUT_ABS" "$LEROBOT_DIR"
    --task "$TASK"
  )
  if [[ -n "$REPO_ID" ]]; then CONVERT_ARGS+=(--repo-id "$REPO_ID"); fi
  if [[ -n "$MAX_SOURCE_EPISODES" ]]; then
    CONVERT_ARGS+=(--max-source-episodes "$MAX_SOURCE_EPISODES")
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
)

EXPECT_TACTILE="$($PYTHON - "$CONFIG" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
print("1" if config.get("train", {}).get("tactile", {}).get("enabled", False) else "0")
PY
)"
if [[ "$EXPECT_TACTILE" -eq 1 ]]; then
  CHECK_ARGS+=(--expect-tactile)
fi

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

if [[ "$SKIP_NORM" -eq 0 ]]; then
  rm -f "$NORM_FILE"
  CUDA_VISIBLE_DEVICES="${NORM_CUDA_VISIBLE_DEVICES:-0}" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="${NORM_MASTER_PORT:-62501}" \
    bash "$ROOT_DIR/train.sh" "$ROOT_DIR/scripts/compute_norm_stats.py" "$NORM_CONFIG" \
      --data.data_name tacthru_umi_v2 \
      --data.train_path "$LEROBOT_DIR" \
      --data.robot_config_root "$ROOT_DIR/configs/robot_configs" \
      --data.norm_path "$NORM_FILE" \
      --data.data_ratio_for_norm_compute 1 \
      --data.num_workers "$NORM_NUM_WORKERS" \
      --train.chunk_size 50 \
      --train.micro_batch_size 32 \
      --train.output_dir "$OUTPUT_DIR/norm_run"
fi

if [[ ! -f "$NORM_FILE" ]]; then
  echo "Normalization statistics are missing: $NORM_FILE" >&2
  echo "Remove --skip-norm or provide the stats at the path above." >&2
  exit 1
fi

"${CHECK_ARGS[@]}" --norm "$NORM_FILE"

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
    --data.data_name tacthru_umi_v2 \
    --data.train_path "$LEROBOT_DIR" \
    --data.robot_config_root "$ROOT_DIR/configs/robot_configs" \
    --data.norm_stats_file "$NORM_FILE" \
    --train.output_dir "$OUTPUT_DIR" \
    --train.align_params "$ALIGN_PARAMS_JSON" \
    "${EXTRA_ARGS[@]}"
