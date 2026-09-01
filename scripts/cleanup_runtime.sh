#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT=${OMNI_PORTAL_RUNTIME_ROOT:-$REPO_ROOT/runtime-data}
APPLY=0

case ${1:-} in
  "") ;;
  --apply) APPLY=1 ;;
  --help|-h)
    printf 'Usage: ./scripts/cleanup_runtime.sh [--apply]\n'
    exit 0
    ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac

if [[ -s "$RUNTIME_ROOT/state/supervisor.pid" ]]; then
  pid=$(sed -n '1p' "$RUNTIME_ROOT/state/supervisor.pid")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf 'Refusing cleanup: supervisor pid %s is running. Stop it first.\n' "$pid" >&2
    exit 1
  fi
fi

files=(
  "$RUNTIME_ROOT/components/comprehension-model.gguf"
  "$RUNTIME_ROOT/components/comprehension-projector.gguf"
  "$RUNTIME_ROOT/components/tts-model.gguf"
  "$RUNTIME_ROOT/components/tts-projector.gguf"
  "$RUNTIME_ROOT/components/.robit-omni-portal-cache"
  "$RUNTIME_ROOT/state/access-url.txt"
  "$RUNTIME_ROOT/state/access-token.txt"
  "$RUNTIME_ROOT/state/comprehension-gpu.txt"
  "$RUNTIME_ROOT/state/gpu-lease-token.txt"
  "$RUNTIME_ROOT/state/supervisor.pid"
  "$RUNTIME_ROOT/state/tts-active.pid"
)

printf 'Runtime root: %s\n' "$RUNTIME_ROOT"
du -sh "$RUNTIME_ROOT" 2>/dev/null || true
for file in "${files[@]}"; do
  [[ -f "$file" ]] && printf '%s\n' "$file"
done
if [[ -d "$RUNTIME_ROOT/session-logs" ]]; then
  find "$RUNTIME_ROOT/session-logs" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.tmp' \) -print
fi

if ((!APPLY)); then
  printf 'Dry run only. Re-run with --apply to unlink the files listed above.\n'
  exit 0
fi

for file in "${files[@]}"; do
  [[ -f "$file" ]] && unlink "$file"
done
if [[ -d "$RUNTIME_ROOT/session-logs" ]]; then
  while IFS= read -r -d '' file; do unlink "$file"; done < <(
    find "$RUNTIME_ROOT/session-logs" -maxdepth 1 -type f \
      \( -name '*.json' -o -name '*.tmp' \) -print0
  )
fi
for directory in \
  "$RUNTIME_ROOT/session-logs" "$RUNTIME_ROOT/components" "$RUNTIME_ROOT/state"; do
  rmdir "$directory" 2>/dev/null || true
done
du -sh "$RUNTIME_ROOT" 2>/dev/null || true
printf 'Removed only known repository-owned runtime artifacts. Ollama data was untouched.\n'
