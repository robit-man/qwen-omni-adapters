#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${OMNI_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}

[[ -x "$PYTHON" ]] || {
  printf 'Missing repository Python: %s (run ./scripts/bootstrap.sh --skip-llama --skip-models)\n' "$PYTHON" >&2
  exit 1
}

cd "$REPO_ROOT"
"$PYTHON" -m compileall -q src runtime portal clients tests
"$PYTHON" -m ruff check src runtime portal clients tests
"$PYTHON" -m pytest
node portal/vad_harness.mjs >/dev/null
bash -n deploy.sh deploy-macos.sh portal/start.sh scripts/bootstrap.sh scripts/build_llama_cpp.sh scripts/cleanup_runtime.sh scripts/validate.sh services/linux/install.sh services/linux/uninstall.sh services/macos/install.sh services/macos/uninstall.sh
"$PYTHON" -m qwen_omni_adapters contract >/dev/null
printf 'All source, contract, VAD, and unit validation gates passed.\n'
