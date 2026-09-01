#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -x "$REPO_ROOT/.venv/bin/qwen-omni" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server" ]] \
  || [[ ! -x "$REPO_ROOT/vendor/llama.cpp/build/bin/llama-tts" ]]; then
  "$REPO_ROOT/scripts/bootstrap.sh"
fi

exec "$REPO_ROOT/portal/start.sh" --daemon
