#!/usr/bin/env bash

set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/torchrun" ]]; then
  TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
else
  TORCHRUN="${TORCHRUN:-$(command -v torchrun || true)}"
fi
if [[ -z "$TORCHRUN" ]]; then
  echo "torchrun was not found; activate the v2 environment or create $ROOT_DIR/.venv" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
export HF_HUB_DISABLE_TELEMETRY=1 
export DISABLE_TELEMETRY=1 

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
  else
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
  fi
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"
NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=$NPROC_PER_NODE}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=127.0.0.1}
MASTER_PORT=${MASTER_PORT:=62500}


"$TORCHRUN" --nnodes="$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" "$@" 2>&1 | tee log.txt
