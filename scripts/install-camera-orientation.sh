#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${OMNI_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
MODEL_DIR=$REPO_ROOT/runtime-data/components/camera-orientation
MODEL_NAME=pico-s70-int8.onnx
MODEL_SHA256=cf456234feabc4f5f5023198e13c2785779ff8bcd3d223f44f61667b963e9c22
MODEL_REVISION=69076d99f16fafbc3d7e12958390972b5f291e81

[[ -x "$PYTHON" ]] || { printf 'Run the runtime bootstrap first.\n' >&2; exit 1; }
mkdir -p -- "$MODEL_DIR"
if ! printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_DIR/$MODEL_NAME" | sha256sum --status -c; then
  download=$(mktemp "$MODEL_DIR/.download.XXXXXXXX")
  trap 'rm -f -- "$download"' EXIT
  curl --fail --silent --show-error --location --retry 1 --max-time 90 --max-filesize 24082994 \
    "https://huggingface.co/ortusai/rightwayup/resolve/$MODEL_REVISION/$MODEL_NAME" -o "$download"
  printf '%s  %s\n' "$MODEL_SHA256" "$download" | sha256sum -c
  mv -f -- "$download" "$MODEL_DIR/$MODEL_NAME"
fi
"$PYTHON" -m pip install -e "$REPO_ROOT[camera-orientation]"
printf 'Camera roll model installed at %s\n' "$MODEL_DIR/$MODEL_NAME"
