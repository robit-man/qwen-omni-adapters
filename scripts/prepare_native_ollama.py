#!/usr/bin/env python3
"""Prepare exact-tag native Ollama runner bindings without duplicating weights.

This enables an experimental pinned Ollama build, not stock model compatibility
or release qualification. All existing sidecar/conditional adapter checks apply.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qwen_omni_adapters.audio_prefill import (
    TRAINED_AUDIO_SYSTEM_PROMPT, TRAINED_AUDIO_TEMPLATE_SUFFIX,
    TRAINED_AUDIO_PROMPT_SHA256, verify_audio_prefill_runtime,
)
from qwen_omni_adapters.ollama_sidecar import resolve_ollama_sidecar

def identity(path: Path) -> dict:
    path = path.absolute()
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('native artifact changed during hashing')
    return {'path': str(path), 'sha256': digest}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--models-dir', type=Path, required=True)
    p.add_argument('--config-dir', type=Path, required=True)
    p.add_argument('--engine-dir', type=Path, default=ROOT / 'vendor/llama.cpp/build/bin')
    a = p.parse_args()
    resolved = resolve_ollama_sidecar(model=a.model, models_dir=a.models_dir)
    if (resolved['profile'] != 'trained-audio-bridge' or
            resolved['inspection']['manifest']['schema'] != 'robit.ollama-audio-bridge.v2'):
        raise ValueError('native integration requires the indexed trained bridge profile')
    model = Path(resolved['standard_layers']['language_model']['path'])
    projector = Path(resolved['standard_layers']['projector']['path'])
    verify_audio_prefill_runtime(projector, a.engine_dir / 'llama-server')
    manifest = Path(resolved['manifest']).read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    libraries = sorted(set(a.engine_dir.glob('*.so*')) | set(a.engine_dir.glob('*.dylib')))
    cfg = {
        'schema': 'robit.omni.native-ollama.v1', 'manifest_digest': digest,
        'model': identity(model), 'projector': identity(projector),
        'sidecar': identity(Path(resolved['bundle'])),
        'engine': identity(a.engine_dir / 'llama-server'),
        'tts': identity(a.engine_dir / 'llama-tts'),
        'reference_encoder': identity(a.engine_dir / 'llama-tts-reference-encode'),
        'proof_script': identity(ROOT / 'native/ollama/gpu_proof.py'),
        'python': str(Path(sys.executable).absolute()),
        'libraries': [identity(lib) for lib in libraries],
        'voices': {'female': identity(ROOT / 'portal/voices/female_voice.wav'),
                   'male': identity(ROOT / 'portal/voices/default_voice.wav')},
        'system_prompt': TRAINED_AUDIO_SYSTEM_PROMPT,
        'audio_suffix': TRAINED_AUDIO_TEMPLATE_SUFFIX,
        'prompt_sha256': TRAINED_AUDIO_PROMPT_SHA256,
    }
    if resolved['audio_adapter'] is not None:
        cfg['audio_adapter'] = identity(Path(resolved['audio_adapter']['path']))
    a.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = a.config_dir / (digest + '.json')
    # Atomic exclusive creation prevents one installer from replacing another
    # runner's live interpretation of a manifest.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(cfg, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'configuration': str(target), 'manifest_sha256': digest,
                      'weight_bytes_copied': 0, 'CUDA_started': False,
                      'native_endpoint_qualified': False}))

if __name__ == '__main__':
    main()
