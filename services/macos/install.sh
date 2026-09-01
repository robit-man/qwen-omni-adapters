#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
LABEL=ai.robit.qwen-omni-adapters
DESTINATION="$HOME/Library/LaunchAgents/$LABEL.plist"

[[ $(uname -s) == Darwin ]] || { printf 'This installer is for macOS.\n' >&2; exit 1; }
[[ -x "$REPO_ROOT/.venv/bin/qwen-omni-daemon" ]] || {
  printf 'Run ./scripts/bootstrap.sh before installing the service.\n' >&2
  exit 1
}
mkdir -p "$HOME/Library/LaunchAgents" "$REPO_ROOT/runtime-data/logs"
sed \
  -e "s|@REPO_ROOT@|$REPO_ROOT|g" \
  -e "s|@PATH@|$PATH|g" \
  "$SCRIPT_DIR/$LABEL.plist.in" >"$DESTINATION"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$DESTINATION"
launchctl enable "gui/$UID/$LABEL"
launchctl kickstart -k "gui/$UID/$LABEL"
printf 'Installed %s as a user LaunchAgent.\n' "$LABEL"
printf 'Status: .venv/bin/qwen-omni-daemon status\n'
