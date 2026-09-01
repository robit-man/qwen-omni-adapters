#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Darwin ]] || {
  printf 'deploy-macos.sh is for macOS; use ./deploy.sh on the broker-managed Linux host.\n' >&2
  exit 1
}
"$REPO_ROOT/scripts/bootstrap.sh"
"$REPO_ROOT/services/macos/install.sh"
"$REPO_ROOT/.venv/bin/qwen-omni-daemon" status
