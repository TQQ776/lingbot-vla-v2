#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${LINGBOT_V2_SSH_HOST:-tqq@172.16.41.254}"
SSH_PORT="${LINGBOT_V2_SSH_PORT:-30175}"
REMOTE_DIR="${LINGBOT_V2_REMOTE_DIR:-/root/kube-user/ns100002-chenrui/cr/tqq/lingbot-vla-v2}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_code_to_server.sh [--dry-run|--apply]

The default is --dry-run. The script synchronizes source code only and never
uses --delete. Server-side weights, environments, datasets, norm statistics,
and training outputs are preserved.

Optional environment variables:
  LINGBOT_V2_SSH_HOST
  LINGBOT_V2_SSH_PORT
  LINGBOT_V2_REMOTE_DIR
EOF
}

mode="${1:---dry-run}"
case "$mode" in
  --dry-run) dry_run=(--dry-run) ;;
  --apply) dry_run=() ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

command -v rsync >/dev/null 2>&1 || {
  echo "rsync is required on the local machine." >&2
  exit 1
}

ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$SSH_PORT" "$SSH_HOST" \
  "test -d '$REMOTE_DIR'" || {
    echo "Cannot reach the v2 server repository. Connect to the cluster LAN/VPN first." >&2
    exit 1
  }

rsync -az --checksum --itemize-changes "${dry_run[@]}" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.uv/' \
  --exclude='.hf/' \
  --exclude='/models/' \
  --exclude='/data/' \
  --exclude='/output/' \
  --exclude='/wandb/' \
  --exclude='/runs/' \
  --exclude='/tmp/' \
  --exclude='**/__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.zarr' \
  --exclude='*.zarr.zip' \
  --exclude='/assets/norm_stats/tacthru*.json' \
  -e "ssh -p $SSH_PORT" \
  "$ROOT_DIR/" "$SSH_HOST:$REMOTE_DIR/"

if [[ "$mode" == "--dry-run" ]]; then
  echo "Dry run only. Review the list, then rerun with --apply."
else
  echo "v2 source synchronization completed; server runtime artifacts were preserved."
fi
