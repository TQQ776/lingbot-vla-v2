#!/usr/bin/env bash

set -euo pipefail

SNAPSHOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SNAPSHOT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$SNAPSHOT_DIR/../.." && pwd)"
fi

FORCE=0
FORCE_CONFLICTS=0
case "${1:---dry-run}" in
  --dry-run) ;;
  --force) FORCE=1 ;;
  --force-conflicts) FORCE=1; FORCE_CONFLICTS=1 ;;
  *)
    echo "Usage: $0 [--dry-run|--force|--force-conflicts]" >&2
    exit 2
    ;;
esac
[[ $# -le 1 ]] || { echo "Usage: $0 [--dry-run|--force|--force-conflicts]" >&2; exit 2; }

echo "snapshot=$SNAPSHOT_DIR"
echo "repository=$ROOT_DIR"

for required in paths.txt files.sha256 absent-list.txt working-tree.sha256 working-tree-absent.txt; do
  [[ -f "$SNAPSHOT_DIR/$required" ]] || {
    echo "Snapshot metadata is missing: $SNAPSHOT_DIR/$required" >&2
    exit 1
  }
done
(cd "$SNAPSHOT_DIR/files" && sha256sum --check --quiet "../files.sha256")

checksum_from_file() {
  local checksum_file="$1"
  local relative_path="$2"
  awk -v path="$relative_path" '$2 == path {print $1; exit}' "$checksum_file"
}

current_checksum() {
  local target_path="$1"
  if [[ -f "$target_path" ]]; then
    sha256sum "$target_path" | awk '{print $1}'
  elif [[ ! -e "$target_path" && ! -L "$target_path" ]]; then
    printf 'missing\n'
  else
    printf 'unsupported\n'
  fi
}

snapshot_checksum() {
  local relative_path="$1"
  local checksum
  checksum="$(checksum_from_file "$SNAPSHOT_DIR/working-tree.sha256" "$relative_path")"
  if [[ -n "$checksum" ]]; then
    printf '%s\n' "$checksum"
  elif grep -Fxq -- "$relative_path" "$SNAPSHOT_DIR/working-tree-absent.txt"; then
    printf 'missing\n'
  else
    printf 'unknown\n'
  fi
}

baseline_checksum() {
  local relative_path="$1"
  local checksum
  checksum="$(checksum_from_file "$SNAPSHOT_DIR/files.sha256" "$relative_path")"
  if [[ -n "$checksum" ]]; then
    printf '%s\n' "$checksum"
  elif grep -Fxq -- "$relative_path" "$SNAPSHOT_DIR/absent-list.txt"; then
    printf 'missing\n'
  else
    printf 'unknown\n'
  fi
}

CONFLICTS=0
while IFS= read -r relative_path; do
  [[ -n "$relative_path" ]] || continue
  target_path="$ROOT_DIR/$relative_path"
  current="$(current_checksum "$target_path")"
  captured="$(snapshot_checksum "$relative_path")"
  baseline="$(baseline_checksum "$relative_path")"
  if [[ "$current" == unsupported || "$captured" == unknown || "$baseline" == unknown ]]; then
    printf 'conflict %s (unsupported or incomplete snapshot metadata)\n' "$relative_path"
    CONFLICTS=$((CONFLICTS + 1))
  elif [[ "$current" != "$captured" && "$current" != "$baseline" ]]; then
    printf 'conflict %s (current=%s captured=%s baseline=%s)\n' \
      "$relative_path" "$current" "$captured" "$baseline"
    CONFLICTS=$((CONFLICTS + 1))
  fi
done <"$SNAPSHOT_DIR/paths.txt"

if [[ "$FORCE" -eq 1 && "$CONFLICTS" -gt 0 && "$FORCE_CONFLICTS" -eq 0 ]]; then
  echo "Refusing restore because $CONFLICTS path(s) changed after the snapshot." >&2
  echo "Review them first; --force-conflicts is the explicit destructive override." >&2
  exit 3
fi

while IFS= read -r checksum relative_path; do
  [[ -n "${relative_path:-}" ]] || continue
  source_path="$SNAPSHOT_DIR/files/$relative_path"
  target_path="$ROOT_DIR/$relative_path"
  target_checksum="$(current_checksum "$target_path")"
  if [[ "$target_checksum" == "$checksum" ]]; then
    printf 'keep    %s (already matches baseline)\n' "$relative_path"
    continue
  fi
  printf 'restore %s (current=%s baseline=%s)\n' \
    "$relative_path" "$target_checksum" "$checksum"
  if [[ "$FORCE" -eq 1 ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp -a "$source_path" "$target_path"
  fi
done <"$SNAPSHOT_DIR/files.sha256"

while IFS= read -r relative_path; do
  [[ -n "$relative_path" ]] || continue
  target_path="$(readlink -m "$ROOT_DIR/$relative_path")"
  if [[ ! -e "$target_path" && ! -L "$target_path" ]]; then
    printf 'keep    %s (already absent from baseline)\n' "$relative_path"
    continue
  fi
  printf 'remove  %s (absent from baseline)\n' "$relative_path"
  if [[ "$FORCE" -eq 1 ]]; then
    case "$target_path" in
      "$ROOT_DIR"/*) rm -f -- "$target_path" ;;
      *) echo "Refusing path outside repository: $relative_path" >&2; exit 1 ;;
    esac
  fi
done <"$SNAPSHOT_DIR/absent-list.txt"

if [[ "$FORCE" -eq 0 ]]; then
  echo "Dry-run only. Review every changed path, then re-run with --force."
  if [[ "$CONFLICTS" -gt 0 ]]; then
    echo "$CONFLICTS path(s) changed after snapshot creation and will not be overwritten by --force."
  fi
fi
