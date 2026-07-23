#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$ROOT_DIR/.uv/bin/uv}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-$ROOT_DIR/.uv/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl}"
RECREATE=0
RESUME=0

usage() {
  cat <<'EOF'
Usage: bash tools/create_train_uv_env.sh [--recreate|--resume] [--flash-attn-wheel PATH]

Creates a project-local uv environment at .venv for LingBot-VLA v2 training.
The environment uses Python 3.12, PyTorch 2.8.0, Flash-Attention 2.8.3,
LeRobot 0.4.2, and the pinned V2/depth dependencies. GPU availability is
reported but is not required during installation.

Environment overrides:
  UV_BIN, UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, VENV_DIR, PYTHON_VERSION,
  FLASH_ATTN_WHEEL, UV_LINK_MODE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    --resume) RESUME=1; shift ;;
    --flash-attn-wheel)
      [[ $# -ge 2 ]] || { echo "--flash-attn-wheel requires a path" >&2; exit 2; }
      FLASH_ATTN_WHEEL="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv was not found: $UV_BIN" >&2
  echo "Install/copy uv there or set UV_BIN=/full/path/to/uv." >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT_DIR/.uv/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$(dirname "$FLASH_ATTN_WHEEL")"

if [[ -e "$VENV_DIR" ]]; then
  if [[ "$RECREATE" -eq 1 ]]; then
    case "$VENV_DIR" in
      "$ROOT_DIR"/.venv|"$ROOT_DIR"/.venv/*) rm -rf "$VENV_DIR" ;;
      *) echo "Refusing to remove unexpected venv path: $VENV_DIR" >&2; exit 1 ;;
    esac
  elif [[ "$RESUME" -ne 1 ]]; then
    echo "Environment already exists: $VENV_DIR" >&2
    echo "Use --resume to finish it or --recreate to rebuild it." >&2
    exit 1
  fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" python install "$PYTHON_VERSION"
  PYTHON_BIN="$($UV_BIN python find "$PYTHON_VERSION")"
  "$UV_BIN" venv --python "$PYTHON_BIN" --seed "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
uv_pip() {
  "$UV_BIN" pip install --python "$PYTHON" "$@"
}

uv_pip --upgrade pip setuptools wheel
# Pin NumPy before resolving torchvision so the installer does not fetch a
# newer transient NumPy only to downgrade it when requirements.txt is applied.
uv_pip numpy==1.26.4
# The PyTorch 2.8 wheels resolve to the CUDA 12.8 stack in this environment.
# GPU visibility is not required during installation; the final validation
# still requires torch.version.cuda == 12.8.
uv_pip torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
uv_pip torchdata==0.11.0 torchcodec==0.6.0
uv_pip -r "$ROOT_DIR/requirements.txt"
uv_pip --no-deps numpydantic==1.9.0

if [[ ! -f "$FLASH_ATTN_WHEEL" ]]; then
  echo "Flash-Attention wheel is missing: $FLASH_ATTN_WHEEL" >&2
  exit 1
fi
uv_pip --no-deps "$FLASH_ATTN_WHEEL"

uv_pip --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
uv_pip --no-deps -e "$ROOT_DIR"

uv_pip -r "$ROOT_DIR/requirements-depth.txt"
# Depth packages have broad/conflicting dependency metadata. Restore the V2
# pins, then install only their local code without resolving those dependencies.
uv_pip -r "$ROOT_DIR/requirements.txt"
uv_pip --no-deps numpydantic==1.9.0

uv_pip --no-deps \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
uv_pip --no-deps -e "$ROOT_DIR/lingbotvla/models/vla/vision_models/lingbot-depth"
uv_pip --no-deps -e "$ROOT_DIR/lingbotvla/models/vla/vision_models/MoGe"
uv_pip huggingface_hub==0.34.3

"$PYTHON" - <<'PY'
import importlib.metadata as md

import accelerate
import cv2
import flash_attn
import lerobot
import mdm
import mlflow
import moge
import omegaconf
import peft
import qwen_vl_utils
import torch
import transformers
import utils3d
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from mdm.model.v2 import MDMModel
from moge.model.v2 import MoGeModel

assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.compiled_with_cxx11_abi() is True
assert md.version("transformers") == "4.57.3"
assert md.version("lerobot") == "0.4.2"
assert md.version("flash-attn") == "2.8.3"
assert md.version("zarr") == "2.18.3"

if torch.cuda.is_available():
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy

    print("LingbotVlaV2Policy GPU-aware import: OK")
else:
    print("LingbotVlaV2Policy GPU-aware import: skipped (CUDA device is not visible)")

print("python environment imports: OK")
for package in (
    "torch",
    "torchvision",
    "transformers",
    "flash-attn",
    "lerobot",
    "zarr",
    "huggingface-hub",
):
    print(package, md.version(package))
print("torch CUDA", torch.version.cuda, "available", torch.cuda.is_available())
PY

if ! "$UV_BIN" pip check --python "$PYTHON"; then
  echo "[WARN] uv pip check reported dependency metadata issues." >&2
  echo "[WARN] Local V2/depth packages are installed with --no-deps intentionally." >&2
fi

echo "V2 uv environment ready: $VENV_DIR"
