#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE=${OMNI_JETSON_REMOTE:-egg@192.168.1.31}
REMOTE_ROOT=${OMNI_JETSON_REPO:-/home/egg/Documents/egg_events/vendor/qwen-omni-adapters}
FULL_VALIDATE=0
TEST_TARGETS=()

usage() {
  printf '%s\n' \
    'Usage: scripts/jetson_dev_refresh.sh [--full] [pytest-path ...]' \
    '' \
    'Sync tracked source to the Jetson, validate it there, and reload only the' \
    'adapter, portal, and desktop harness. Resident comprehension, pointing,' \
    'and TTS workers must keep the same PIDs.' \
    '' \
    'Environment:' \
    '  OMNI_JETSON_REMOTE    SSH destination (default egg@192.168.1.31)' \
    '  OMNI_JETSON_REPO      Remote checkout path' \
    '  OMNI_JETSON_PASSWORD  Optional sshpass password; SSH keys are preferred'
}

while (($#)); do
  case "$1" in
    --full) FULL_VALIDATE=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *) TEST_TARGETS+=("$1") ;;
  esac
  shift
done

command -v rsync >/dev/null || {
  printf 'rsync is required for Jetson development refreshes.\n' >&2
  exit 1
}

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
RSYNC_SSH='ssh -o BatchMode=yes -o ConnectTimeout=8'
if [[ -n ${OMNI_JETSON_PASSWORD:-} ]]; then
  command -v sshpass >/dev/null || {
    printf 'OMNI_JETSON_PASSWORD is set but sshpass is unavailable.\n' >&2
    exit 1
  }
  export SSHPASS=$OMNI_JETSON_PASSWORD
  SSH=(sshpass -e ssh -o ConnectTimeout=8)
  RSYNC_SSH='sshpass -e ssh -o ConnectTimeout=8'
fi

printf -v REMOTE_ROOT_Q '%q' "$REMOTE_ROOT"

read_remote_pids() {
  "${SSH[@]}" "$REMOTE" "cd $REMOTE_ROOT_Q && .venv/bin/qwen-omni-daemon status" \
    | "$REPO_ROOT/.venv/bin/python" -c '
import json, sys
value = json.load(sys.stdin)
children = {item.get("name"): item.get("pid") for item in value.get("children", [])}
for name in ("comprehension", "pointing", "tts"):
    value = children.get(name)
    print("{}={}".format(name, "" if value is None else value))
'
}

BEFORE_PIDS=$(read_remote_pids)
printf 'Resident workers before refresh:\n%s\n' "$BEFORE_PIDS"

cd "$REPO_ROOT"
git ls-files -z \
  | rsync -az --from0 --files-from=- --relative -e "$RSYNC_SSH" ./ "$REMOTE:$REMOTE_ROOT/"

if ((FULL_VALIDATE)); then
  "${SSH[@]}" "$REMOTE" "cd $REMOTE_ROOT_Q && ./scripts/validate.sh"
else
  if ((${#TEST_TARGETS[@]} == 0)); then
    TEST_TARGETS=(
      tests/test_context_catalog.py
      tests/test_omni_adapter.py
      tests/test_omni_portal.py
      tests/test_background_agent.py
      tests/test_background_workflows.py
      tests/test_daemon.py
    )
  fi
  TEST_ARGS=()
  for target in "${TEST_TARGETS[@]}"; do
    printf -v target_q '%q' "$target"
    TEST_ARGS+=("$target_q")
  done
  "${SSH[@]}" "$REMOTE" \
    "cd $REMOTE_ROOT_Q && .venv/bin/python -m compileall -q src runtime portal harness clients && .venv/bin/python -m ruff check src runtime portal harness clients && PYTHONPATH=.:src .venv/bin/python -m pytest -q ${TEST_ARGS[*]}"
fi

if ! "${SSH[@]}" "$REMOTE" \
  "cd $REMOTE_ROOT_Q && .venv/bin/qwen-omni-daemon reload-python"; then
  printf '%s\n' \
    'The running supervisor predates code-only reload support.' \
    'Perform one normal service restart to bootstrap it; later refreshes preserve weights.' >&2
  exit 1
fi

"${SSH[@]}" "$REMOTE" "
  cd $REMOTE_ROOT_Q
  systemctl --user restart omni-call-harness.service
  systemctl --user is-active --quiet omni-call-harness.service
"

AFTER_PIDS=$(read_remote_pids)
printf 'Resident workers after refresh:\n%s\n' "$AFTER_PIDS"
if [[ "$BEFORE_PIDS" != "$AFTER_PIDS" ]]; then
  printf 'A resident model PID changed; the code-only refresh invariant failed.\n' >&2
  exit 1
fi

printf 'Jetson code refreshed without reloading resident model weights.\n'
