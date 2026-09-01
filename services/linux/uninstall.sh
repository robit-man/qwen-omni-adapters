#!/usr/bin/env bash
set -Eeuo pipefail

sudo systemctl disable --now qwen-omni-adapters.service 2>/dev/null || true
sudo unlink /etc/systemd/system/qwen-omni-adapters.service 2>/dev/null || true
sudo systemctl daemon-reload
printf 'Removed qwen-omni-adapters.service. Model tags and runtime data were retained.\n'
