#!/usr/bin/env bash
set -Eeuo pipefail

LABEL=ai.robit.qwen-omni-adapters
DESTINATION="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
[[ -f "$DESTINATION" ]] && unlink "$DESTINATION"
printf 'Removed %s. Model tags and runtime data were retained.\n' "$LABEL"
