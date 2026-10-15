#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
KEV_VENV=${OMNI_KEV_VENV:-$REPO_ROOT/runtime-data/kev/venv}
KEV_SOURCE=${OMNI_KEV_SOURCE:-$REPO_ROOT/vendor/kev}
MANIFEST=${OMNI_KEV_MANIFEST:-$REPO_ROOT/runtime-data/kev-models.json}
CODE_REVISION=2ecafc49fe1e8945cd600443e6c04c1cf0b3a78f

"$PYTHON" -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "Kev requires Python 3.12 or 3.13"'
if [[ ! -d "$KEV_SOURCE/.git" ]]; then
  git clone https://github.com/jaredpalmer/kev.git "$KEV_SOURCE"
fi
git -C "$KEV_SOURCE" diff --quiet HEAD -- || {
  printf 'Refusing to overwrite local Kev source changes: %s\n' "$KEV_SOURCE" >&2
  exit 1
}
if ! git -C "$KEV_SOURCE" cat-file -e "$CODE_REVISION^{commit}" 2>/dev/null; then
  git -C "$KEV_SOURCE" fetch origin "$CODE_REVISION"
fi
git -C "$KEV_SOURCE" checkout --detach "$CODE_REVISION"
"$PYTHON" -m venv "$KEV_VENV"
"$KEV_VENV/bin/python" -m pip install --upgrade pip
# Match the CUDA/Transformers/LoRA path already exercised on A100. PyPI's
# unrelated ORM also uses the name kev; install only our pinned source tree.
"$KEV_VENV/bin/python" -m pip install \
  'torch==2.8.0' 'transformers==5.19.0' 'peft==0.21.2' "$KEV_SOURCE"

"$KEV_VENV/bin/python" - "$MANIFEST" "$CODE_REVISION" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

checkpoints = []
for name, repository, revision in (
    ("adapter", "jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"),
    ("base", "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
):
    path = Path(snapshot_download(repo_id=repository, revision=revision,
                                 cache_dir=os.environ.get("OMNI_KEV_CACHE")))
    files = {str(item.relative_to(path)): {"size_bytes": item.stat().st_size, "sha256": digest(item)}
             for item in path.rglob("*") if item.is_file()}
    checkpoints.append({"name": name, "repository": repository, "revision": revision,
                        "path": str(path), "files": files})

package = Path(importlib.metadata.distribution("kev").locate_file("kev"))
manifest = {"schema": "robit.omni.kev-models.v1", "code_revision": sys.argv[2],
            "weights_revision": checkpoints[0]["revision"], "base_revision": checkpoints[1]["revision"],
            "checkpoints": checkpoints, "kev_package_root": str(package),
            "kev_code_files": {str(path.relative_to(package)): digest(path) for path in package.rglob("*.py")},
            "versions": {name: importlib.metadata.version(name) for name in ("kev", "torch", "transformers", "peft")},
            "model_loaded": False, "cuda_initialized": False}
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
output.chmod(0o600)
print(f"Pinned Kev runtime and local weights ready; manifest: {output}; no CUDA model loaded")
PY

printf 'Start through docker gpu run using %s and runtime/kev_server.py.\n' "$KEV_VENV/bin/python"
