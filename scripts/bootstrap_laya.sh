#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
LAYA_VENV=${OMNI_LAYA_VENV:-$REPO_ROOT/.laya-venv}
LAYA_VERSION=${OMNI_LAYA_VERSION:-0.3.4}

command -v "$PYTHON" >/dev/null 2>&1 || {
  printf 'Missing Python: %s\n' "$PYTHON" >&2
  exit 1
}

# System site packages let Jetson deployments reuse their platform-specific
# PyTorch build. On a host without those packages, pip installs the portable
# dependencies into this isolated environment.
"$PYTHON" -m venv --system-site-packages "$LAYA_VENV"
"$LAYA_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$LAYA_VENV/bin/python" -m pip install \
  "laya==$LAYA_VERSION" "httpx>=0.27,<1" "PyYAML>=6,<7"
"$LAYA_VENV/bin/python" - <<'PY'
from importlib.metadata import version
from laya import Router

assert Router
print(f"Laya {version('laya')} installed; checkpoints load in the resident service.")
PY
