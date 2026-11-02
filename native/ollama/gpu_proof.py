"""Control-plane proof for Ollama-owned native CUDA workers; no inference."""
from __future__ import annotations
import os
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_omni_adapters.accelerator import is_tegra, process_is_gpu_resident
pid = int(sys.argv[1])
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not is_tegra() and (not gpu.startswith('GPU-') or ',' in gpu):
    raise SystemExit('native Omni worker requires its admitted exact GPU UUID')
for _ in range(100):
    if process_is_gpu_resident(pid, gpu):
        journal = os.environ.get('OLLAMA_OMNI_RESIDENCY_JOURNAL')
        if journal:
            receipt = {'pid': pid, 'gpu_uuid': gpu, 'resident': True,
                       'worker': Path('/proc', str(pid), 'comm').read_text().strip()}
            fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a') as stream:
                stream.write(json.dumps(receipt) + '\n')
        raise SystemExit(0)
    if not Path('/proc', str(pid)).exists():
        break
    time.sleep(.1)
raise SystemExit('native Omni accelerator residency unproven')
