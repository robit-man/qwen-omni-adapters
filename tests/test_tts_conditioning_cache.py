from __future__ import annotations

import hashlib
import os
from dataclasses import replace

from runtime.tts_server import Config, PersistentTTSWorker, _synthesis_spec


def test_reading_a_checkpoint_may_update_its_access_time(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"unchanged weights")
    os.utime(model, ns=(0, model.stat().st_mtime_ns))
    worker = PersistentTTSWorker(Config(binary=model, model=model, projector=model))
    assert worker._weight_identity(model) == hashlib.sha256(b"unchanged weights").hexdigest()


def test_changed_reference_and_checkpoint_cannot_reuse_loaded_conditioning(tmp_path):
    binary = tmp_path / "fake-tts"
    binary.write_text("""#!/usr/bin/env python3
import base64,sys
def frame(kind,payload=b''):
 sys.stdout.buffer.write(kind.encode()+len(payload).to_bytes(8,'little')+payload)
 sys.stdout.buffer.flush()
frame('R')
for line in sys.stdin.buffer:
 frame('A',base64.b64decode(line.strip()))
 frame('D')
""")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    projector = tmp_path / "projector.gguf"
    reference = tmp_path / "reference.wav"
    model.write_bytes(b"model-A")
    projector.write_bytes(b"codec-A")
    reference.write_bytes(b"voice-A")
    config = Config(binary=binary, model=model, projector=projector,
                    require_gpu=False, gpu_layers=0, timeout_s=5)
    worker = PersistentTTSWorker(config)
    spec = _synthesis_spec(config, {"text": "first", "speaker_file": str(reference)})
    try:
        assert b"".join(worker.stream(spec)) == b"first"
        pid_a = worker.pid
        assert b"".join(worker.stream(replace(spec, text="second"))) == b"second"
        assert worker.pid == pid_a
        reference.write_bytes(b"voice-B")  # same filename and size
        assert b"".join(worker.stream(spec)) == b"first"
        pid_b = worker.pid
        assert pid_b != pid_a
        model.write_bytes(b"model-B")
        assert b"".join(worker.stream(spec)) == b"first"
        pid_c = worker.pid
        assert pid_c != pid_b
        projector.write_bytes(b"codec-B")
        assert b"".join(worker.stream(spec)) == b"first"
        assert worker.pid != pid_c
        assert len(worker._weight_fingerprints) == 2
    finally:
        worker.close()
    assert worker.profile is None
    assert not worker.speaker_reference_active
