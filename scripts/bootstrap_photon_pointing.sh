#!/usr/bin/env bash
# Dependency pins from the adjacent 2026-10-12 A100 Photon runtime evidence.
# Preparation installs files and verifies hashes; it never imports a GPU runtime.
set -Eeuo pipefail
umask 077

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
POINTING_ROOT=${OMNI_PHOTON_ROOT:-$REPO_ROOT/runtime-data/pointing}
PHOTON_VENV=${OMNI_PHOTON_VENV:-$POINTING_ROOT/venv}
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  printf 'This pinned Photon bootstrap is for Linux x86_64 CPython3.12; retain bootstrap_pointing.sh for Tegra.\n' >&2
  exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Pinned Photon wheels require CPython3.12"'
mkdir -p -- "$POINTING_ROOT"
# Reuse compatible host dependencies read-only. In a clean host, pip installs
# the same complete lock into this private venv without changing global Python.
"$PYTHON" -m venv --system-site-packages "$PHOTON_VENV"
requirements=$(mktemp "$POINTING_ROOT/photon-requirements.XXXXXXXX")
trap 'rm -f -- "$requirements"' EXIT
cat >"$requirements" <<'PHOTON_REQUIREMENTS'
# Linux x86_64 CPython3.12 portable wheel plan, resolved without GPU imports.
# Install only into an isolated environment; no runtime compatibility claim.
annotated-doc==0.0.4 --hash=sha256:571ac1dc6991c450b25a9c2d84a3705e2ae7a53467b5d111c24fa8baabbed320
anyio==4.12.1 --hash=sha256:d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c
certifi==2023.11.17 --hash=sha256:e036ab49d5b79556f99cfc2d9320b34cfbe5be05c5871b51de9329f0603b0474
click==8.3.1 --hash=sha256:981153a64e25f12d547d3426c367a4857371575ee7ad18df2a6183ab0545b2a6
cuda-bindings==13.2.0 --hash=sha256:46d8776a55d6d5da9dd6e9858fba2efcda2abe6743871dee47dd06eb8cb6d955
cuda-pathfinder==1.5.4 --hash=sha256:9563d3175ce1828531acf4b94e1c1c7d67208c347ca002493e2654878b26f4b7
cuda-toolkit==13.0.2 --hash=sha256:b198824cf2f54003f50d64ada3a0f184b42ca0846c1c94192fa269ecd97a66eb
filelock==3.25.2 --hash=sha256:ca8afb0da15f229774c9ad1b455ed96e85a81373065fb10446672f64444ddf70
fsspec==2026.2.0 --hash=sha256:98de475b5cb3bd66bedd5c4679e87b4fdfe1a3bf4d707b151b3c07e58c9a2437
h11==0.16.0 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86
hf-xet==1.4.3 --hash=sha256:fc360b70c815bf340ed56c7b8c63aacf11762a4b099b2fe2c9bd6d6068668c08
httpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55
httpx==0.28.1 --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad
huggingface-hub==1.10.1 --hash=sha256:6b981107a62fbe68c74374418983399c632e35786dcd14642a9f2972633c8b5a
idna==3.6 --hash=sha256:c05567e9c24a6b9faaa835c4821bad0590fbb9d5779e7caa6e1cc4978e7eb24f
jinja2==3.1.2 --hash=sha256:6088930bfe239f0e6710546ab9c19c9ef35e29792895fed6e6e31a023a182a61
kestrel==0.9.1 --hash=sha256:80640639279b18b2da61b08c06c550f8eaab6596d6d15978fd56f76707462cb7
kestrel-kernels==0.7.4 --hash=sha256:50a8b4bea28675db48920e58df0fa0911bb5667e42f83b153e99f3cac367e6a3
kestrel-kernels-bundle-a==0.7.4 --hash=sha256:a9054198c034a9f8ebfb3fb1cd15a68ae8b18a766632520ac6bde536efa78b95
kestrel-kernels-bundle-b==0.7.4 --hash=sha256:8e954e9cabb6a57264b22ef15f7b1e178242d339211572f2b462b4139187a7f6
kestrel-kernels-bundle-c==0.7.4 --hash=sha256:557dc5b73ef6ad150c6e8678f285f8dbf3e1b7edeb07980d2ae8880c223d88fe
kestrel-kernels-bundle-d==0.7.4 --hash=sha256:57829cdfa7303852f94677750769c9f042fb31e13579abc421c03d5c24bb327e
kestrel-kernels-bundle-e==0.7.4 --hash=sha256:500cf46fdb07b46516d76cc4bff7a67b9e4e78cad831443d7b908f22c3658843
kestrel-kernels-bundle-f==0.7.4 --hash=sha256:2bc734bc0c1aaa2b617c2340c461e5ebcdd163752afeb4519106694c6bbe5779
kestrel-kernels-bundle-g==0.7.4 --hash=sha256:7ff7e38df37f177a8273089c298d628f6f94cb6bd3c050c9c0a5f0d4177f2a54
kestrel-kernels-bundle-h==0.7.4 --hash=sha256:7ddd51d2807bd4eaddf714b89dad190b3cf2d16f61e7116c9ed23ed77d7eb33c
kestrel-native==0.1.8 --hash=sha256:aa065e659bce0d3cb5989c987793eebdc494c025c20309398b7fec9f8ad3f4ea
markdown-it-py==3.0.0 --hash=sha256:355216845c60bd96232cd8d8c40e8f9765cc86f46880e43a8fd22dc1a1a8cab1
mdurl==0.1.2 --hash=sha256:84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8
moondream==2.6.1 --hash=sha256:73adcfa8bd701a78f589b8e4b56696fbd305486c3d9bf8856f9bbb759f7d0ff4
mpmath==1.3.0 --hash=sha256:a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c
networkx==3.6.1 --hash=sha256:d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762
numpy==1.26.4 --hash=sha256:675d61ffbfa78604709862923189bad94014bef562cc35cf61d3a07bba02a7ed
nvidia-cublas==13.1.1.3 --hash=sha256:37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436
nvidia-cuda-cupti==13.0.85 --hash=sha256:4eb01c08e859bf924d222250d2e8f8b8ff6d3db4721288cf35d14252a4d933c8
nvidia-cuda-nvrtc==13.0.88 --hash=sha256:ad9b6d2ead2435f11cbb6868809d2adeeee302e9bb94bcf0539c7a40d80e8575
nvidia-cuda-runtime==13.0.96 --hash=sha256:7f82250d7782aa23b6cfe765ecc7db554bd3c2870c43f3d1821f1d18aebf0548
nvidia-cudnn-cu13==9.20.0.48 --hash=sha256:0c45dd8eeb50b603f07995b1b300c62ffe6a1980482b82b3bcf94a4ca9d49304
nvidia-cufft==12.0.0.61 --hash=sha256:6c44f692dce8fd5ffd3e3df134b6cdb9c2f72d99cf40b62c32dde45eea9ddad3
nvidia-cufile==1.15.1.6 --hash=sha256:08a3ecefae5a01c7f5117351c64f17c7c62efa5fffdbe24fc7d298da19cd0b44
nvidia-curand==10.4.0.35 --hash=sha256:1aee33a5da6e1db083fe2b90082def8915f30f3248d5896bcec36a579d941bfc
nvidia-cusolver==12.0.4.66 --hash=sha256:0a759da5dea5c0ea10fd307de75cdeb59e7ea4fcb8add0924859b944babf1112
nvidia-cusparse==12.6.3.3 --hash=sha256:2b3c89c88d01ee0e477cb7f82ef60a11a4bcd57b6b87c33f789350b59759360b
nvidia-cusparselt-cu13==0.8.1 --hash=sha256:786ce87568c303fadb5afcc7102d454cd3040d75f6f8626f5db460d1871f4dd0
nvidia-nccl-cu13==2.29.7 --hash=sha256:edd81538446786ec3b73972543e53bb43bcaf0bfc8ef76cb679fcc390ffe136d
nvidia-nvjitlink==13.0.88 --hash=sha256:13a74f429e23b921c1109976abefacc69835f2f433ebd323d3946e11d804e47b
nvidia-nvshmem-cu13==3.4.5 --hash=sha256:290f0a2ee94c9f3687a02502f3b9299a9f9fe826e6d0287ee18482e78d495b80
nvidia-nvtx==13.0.85 --hash=sha256:4936d1d6780fbe68db454f5e72a42ff64d1fd6397df9f363ae786930fd5c1cd4
packaging==24.0 --hash=sha256:2ddfb553fdf02fb784c234c7ba6ccc288296ceabec964ad2eae3777778130bc5
pillow==10.4.0 --hash=sha256:86dcb5a1eb778d8b25659d5e4341269e8590ad6b4e8b44d9f4b07f8d136c414a
pygments==2.20.0 --hash=sha256:81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176
pyyaml==6.0.1 --hash=sha256:6c22bec3fbe2524cde73d7ada88f6566758a8f7227bfbf93a408a9d86bcc12a0
rich==15.0.0 --hash=sha256:33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb
safetensors==0.8.0rc0 --hash=sha256:bf0d366f75f63867f1ede90f87090450c7cec320da1fc2a597f9bb8cb73460db
setuptools==68.1.2 --hash=sha256:3d8083eed2d13afc9426f227b24fd1659489ec107c0e86cec2ffdde5c92e790b
shellingham==1.5.4 --hash=sha256:7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686
sympy==1.14.0 --hash=sha256:e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5
tokenizers==0.22.2 --hash=sha256:369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67
torch==2.12.0 --hash=sha256:4b4f64c2c2b11f7510d93dd6412b87025ff6eddd6bb61c3b5a3d892ea20c4756
torch-c-dlpack-ext==0.1.5 --hash=sha256:e6f9da4bb9af70e27facc777458be62e10dbbbddda7672d16138db0553c5a524
tqdm==4.67.3 --hash=sha256:ee1e4c0e59148062281c49d80b25b67771a127c85fc9676d3be5f243206826bf
triton==3.7.0 --hash=sha256:8f111161d49bf903c0eaedde3962353a3d841c08a836839b7cc1025b8426efcf
typer==0.24.1 --hash=sha256:112c1f0ce578bfb4cab9ffdabc68f031416ebcc216536611ba21f04e9aa84c9e
typing-extensions==4.15.0 --hash=sha256:f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548
# Compatibility runtime for pinned kestrel-kernels0.7.4 AOT loader; isolated install only.
nvidia-cuda-runtime-cu12==12.9.79 --hash=sha256:25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3
PHOTON_REQUIREMENTS
"$PHOTON_VENV/bin/python" -m pip install --only-binary=:all: --require-hashes -r "$requirements"

"$PHOTON_VENV/bin/python" - "$POINTING_ROOT" <<'PY'
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

root = Path(sys.argv[1]).resolve()
files = {}

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

for name, repo, revision, filename, sha, size, source_env in (
    ('weights', 'moondream/moondream3.1-9B-A2B', '1da27e6c2d2f9118de14560df456881dc62d91f3',
     'model.safetensors', '26dde16af961d3a486c7b86b9a34fc8487be7917f7e62324b751400ef4d391b4',
     10505269576, 'OMNI_PHOTON_WEIGHTS_SOURCE'),
    ('tokenizer', 'moondream/starmie-v1', '35192e10a54e36eabe0a7cc57a2c1aab371cafc5',
     'tokenizer.json', '0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c',
     3694206, 'OMNI_PHOTON_TOKENIZER_SOURCE'),
):
    target = root / filename
    if target.exists():
        source = target
    elif os.environ.get(source_env):
        source = Path(os.environ[source_env]).resolve(strict=True)
    else:
        source = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))
    if source.stat().st_size != size or digest(source) != sha:
        raise ValueError(f'Exact {name} length/SHA-256 verification failed; existing files preserved')
    if not target.exists():
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    # The materialized inode belongs to this checkout and survives removal of
    # the acquisition checkout/cache. No runtime path points into an adjacent repo.
    files[name] = {'path': str(target), 'repository': repo, 'revision': revision,
                   'bytes': size, 'sha256': sha}

versions = {name: importlib.metadata.version(name) for name in (
    'moondream', 'kestrel', 'kestrel-native', 'kestrel-kernels', 'torch',
    'nvidia-cuda-runtime-cu12', *('kestrel-kernels-bundle-' + letter for letter in 'abcdefgh'))}
manifest = {'schema': 'robit.omni.photon-preparation.v1', 'model': 'moondream/moondream3.1-9B-A2B',
            'files': files, 'versions': versions, 'model_loaded': False, 'cuda_initialized': False,
            'model_license': 'https://moondream.ai/licenses/model/1.0',
            'sdk_source': 'https://pypi.org/project/moondream/2.6.1/'}
if any(name in sys.modules for name in ('torch', 'moondream', 'kestrel', 'transformers')):
    raise RuntimeError('Preparation unexpectedly imported a GPU runtime')
(root / 'preparation.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
printf '\nPrepared local Photon runtime; no CUDA context or model created.\n'
printf 'Set OMNI_POINTING_PYTHON=%s/bin/python\n' "$PHOTON_VENV"
printf 'Set OMNI_POINTING_MODEL=moondream/moondream3.1-9B-A2B\n'
printf 'Set OMNI_POINTING_REVISION=1da27e6c2d2f9118de14560df456881dc62d91f3\n'
printf 'A scoped lease and prepare transition are required before model loading.\n'
