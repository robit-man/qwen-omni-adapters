#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
POINTING_VENV=${OMNI_POINTING_VENV:-$REPO_ROOT/.pointing-venv}
MODEL=${OMNI_POINTING_MODEL:-vikhyatk/moondream2}
REVISION=${OMNI_POINTING_REVISION:-9a7d4024050840e001defacec2b00727e89149e6}

if [[ $(uname -s) != Linux || $(uname -m) != aarch64 ]] \
  || [[ ! -r /proc/device-tree/compatible ]] \
  || ! tr '\0' '\n' </proc/device-tree/compatible | grep -Eiq 'tegra|nvidia'; then
  printf 'Structured point-head bootstrap is only required on NVIDIA Tegra; skipping.\n'
  exit 0
fi

command -v "$PYTHON" >/dev/null 2>&1 || {
  printf 'Missing Python: %s\n' "$PYTHON" >&2
  exit 1
}

python_minor=$(
  "$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
[[ $python_minor == 3.10 ]] || {
  printf 'JetPack 6 point-head runtime requires Python 3.10 (found %s).\n' "$python_minor" >&2
  exit 1
}

l4t_release=$(sed -n 's/^# R\([0-9][0-9]*\) (release), REVISION: \([0-9][0-9.]*\).*/R\1.\2/p' /etc/nv_tegra_release 2>/dev/null | head -n 1)
case $l4t_release in
  R36.2*|R36.3*)
    torch_wheel='https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl'
    torch_version_prefix='2.4.0a0+07cecf4168.nv24.'
    ;;
  R36.4*|R36.5*|R36.6*)
    torch_wheel='https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl'
    torch_version_prefix='2.5.0a0+872d972e41.nv24.'
    ;;
  *)
    printf 'Unsupported or unknown JetPack 6 L4T release: %s\n' "${l4t_release:-unknown}" >&2
    exit 1
    ;;
esac

"$PYTHON" -m venv "$POINTING_VENV"
"$POINTING_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$POINTING_VENV/bin/python" -m pip install 'numpy<2'
if ! "$POINTING_VENV/bin/python" - "$torch_version_prefix" <<'PY'
import importlib.metadata
import sys

try:
    installed = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    installed = ""
raise SystemExit(0 if installed.startswith(sys.argv[1]) else 1)
PY
then
  "$POINTING_VENV/bin/python" -m pip install "$torch_wheel"
fi
"$POINTING_VENV/bin/python" -m pip install \
  'transformers==4.51.3' 'accelerate==1.10.1' 'Pillow>=11.0.0'

# Download model code/config/weights without creating a CUDA context. The
# daemon performs the real CUDA/residency proof after it owns the lifecycle.
"$POINTING_VENV/bin/python" - "$MODEL" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
PY

"$POINTING_VENV/bin/python" - <<'PY'
import torch
import transformers

print(f"Point-head runtime ready: torch={torch.__version__} transformers={transformers.__version__}")
PY
