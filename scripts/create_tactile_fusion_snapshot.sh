#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_REF="${BASELINE_REF:-tacthru-umi-v2-wrist-only-20260724}"
PATHS_FILE="${PATHS_FILE:-$ROOT_DIR/scripts/tactile_fusion_paths.txt}"
OUTPUT_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Create a file-level baseline snapshot for tactile-fusion development.

Usage:
  bash scripts/create_tactile_fusion_snapshot.sh [options]

Options:
  --baseline REF   Git baseline used to discover changed paths.
  --paths-file FILE
                   Allowlist of tactile-owned repository paths.
  --output DIR     Snapshot directory. Must remain under .rollback/.
  --dry-run        Print the files that would be captured without writing.
  -h, --help       Show this help.

The snapshot stores each changed path as it existed in --baseline, not its
possibly modified working-tree contents. The generated restore.sh only
performs a dry-run by default. Restoring baseline files or removing paths
that do not exist in the baseline requires --force.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      [[ $# -ge 2 ]] || { echo "--baseline requires a value" >&2; exit 2; }
      BASELINE_REF="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --paths-file)
      [[ $# -ge 2 ]] || { echo "--paths-file requires a value" >&2; exit 2; }
      PATHS_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"
git rev-parse --verify "${BASELINE_REF}^{commit}" >/dev/null
PATHS_FILE="$(readlink -m "$PATHS_FILE")"
[[ -f "$PATHS_FILE" ]] || { echo "Tactile path allowlist is missing: $PATHS_FILE" >&2; exit 1; }

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/.rollback/tactile_fusion_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR="$(readlink -m "$OUTPUT_DIR")"
ROLLBACK_ROOT="$(readlink -m "$ROOT_DIR/.rollback")"
case "$OUTPUT_DIR/" in
  "$ROLLBACK_ROOT"/*) ;;
  *)
    echo "Snapshot output must stay under $ROLLBACK_ROOT: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

mapfile -t CANDIDATE_PATHS < <(
  sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$PATHS_FILE" |
    LC_ALL=C sort -u
)
if [[ ${#CANDIDATE_PATHS[@]} -eq 0 ]]; then
  echo "Tactile path allowlist is empty: $PATHS_FILE" >&2
  exit 1
fi
for relative_path in "${CANDIDATE_PATHS[@]}"; do
  if [[ "$relative_path" == /* || "$relative_path" == ".." || "$relative_path" == ../* || "$relative_path" == */../* ]]; then
    echo "Invalid repository-relative path in allowlist: $relative_path" >&2
    exit 1
  fi
done

mapfile -t PATHS < <(
  {
    git diff --name-only --diff-filter=ACDMRTUXB "$BASELINE_REF" -- "${CANDIDATE_PATHS[@]}"
    for relative_path in "${CANDIDATE_PATHS[@]}"; do
      if ! git cat-file -e "$BASELINE_REF:$relative_path" 2>/dev/null &&
         [[ -e "$relative_path" || -L "$relative_path" ]]; then
        printf '%s\n' "$relative_path"
      fi
    done
  } | sed '/^$/d' | LC_ALL=C sort -u
)

if [[ ${#PATHS[@]} -eq 0 ]]; then
  echo "No changed or untracked files relative to $BASELINE_REF."
  exit 0
fi

printf 'baseline=%s\noutput=%s\nfiles=%d\n' "$BASELINE_REF" "$OUTPUT_DIR" "${#PATHS[@]}"
printf '  %s\n' "${PATHS[@]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Snapshot output already exists: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR/files"

git status --short --branch >"$OUTPUT_DIR/git-status.txt"
git rev-parse HEAD >"$OUTPUT_DIR/head.txt"
printf '%s\n' "$BASELINE_REF" >"$OUTPUT_DIR/baseline-ref.txt"
cp "$PATHS_FILE" "$OUTPUT_DIR/tactile-fusion-paths.txt"
printf '%s\n' "${PATHS[@]}" >"$OUTPUT_DIR/paths.txt"
git diff --binary "$BASELINE_REF" -- "${PATHS[@]}" >"$OUTPUT_DIR/baseline-to-working-tree.patch"
: >"$OUTPUT_DIR/files.sha256"
: >"$OUTPUT_DIR/absent-list.txt"
: >"$OUTPUT_DIR/working-tree.sha256"
: >"$OUTPUT_DIR/working-tree-absent.txt"

for relative_path in "${PATHS[@]}"; do
  target_path="$(readlink -m "$ROOT_DIR/$relative_path")"
  case "$target_path" in
    "$ROOT_DIR"/*) ;;
    *)
      echo "Refusing path outside repository: $relative_path -> $target_path" >&2
      exit 1
      ;;
  esac

  if git cat-file -e "$BASELINE_REF:$relative_path" 2>/dev/null; then
    destination="$OUTPUT_DIR/files/$relative_path"
    mkdir -p "$(dirname "$destination")"
    git show "$BASELINE_REF:$relative_path" >"$destination"
    baseline_mode="$(git ls-tree "$BASELINE_REF" -- "$relative_path" | awk 'NR == 1 {print $1}')"
    if [[ "$baseline_mode" == "100755" ]]; then
      chmod 0755 "$destination"
    else
      chmod 0644 "$destination"
    fi
    sha256sum "$destination" | sed "s#  $OUTPUT_DIR/files/#  #" >>"$OUTPUT_DIR/files.sha256"
  else
    printf '%s\n' "$relative_path" >>"$OUTPUT_DIR/absent-list.txt"
  fi

  if [[ -f "$target_path" ]]; then
    printf '%s  %s\n' "$(sha256sum "$target_path" | awk '{print $1}')" "$relative_path" \
      >>"$OUTPUT_DIR/working-tree.sha256"
  elif [[ ! -e "$target_path" && ! -L "$target_path" ]]; then
    printf '%s\n' "$relative_path" >>"$OUTPUT_DIR/working-tree-absent.txt"
  else
    echo "Snapshot only supports regular files or absent paths: $relative_path" >&2
    exit 1
  fi
done

cp "$ROOT_DIR/scripts/restore_tactile_fusion_snapshot.sh" "$OUTPUT_DIR/restore.sh"
chmod 0755 "$OUTPUT_DIR/restore.sh"

echo "Snapshot created: $OUTPUT_DIR"
