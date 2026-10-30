from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from qwen_omni_adapters.contract import OmniAdapterError, parse_adapter_request
from runtime.tts_server import (
    Config,
    PersistentTTSWorker,
    TTSError,
    _context_size,
    _encode_reference,
    _synthesis_spec,
)


def _request(speech):
    return {"model": "example", "messages": [{"role": "user", "content": "Read this."}],
            "speech": speech, "stream": False}


@pytest.mark.parametrize("text", [None, 2, "x" * 2049, "bad\x00text", "\ud800"])
def test_invalid_reference_transcript_is_rejected_before_any_synthesis(text):
    with pytest.raises(OmniAdapterError, match="reference_text"):
        parse_adapter_request(_request({"reference_text": text, "speaker_file": "/voice.wav"}))


def test_reference_transcript_requires_audio_and_matches_wire_schema():
    schema = json.loads(Path("docs/schema/request-v1.schema.json").read_text())
    invalid = _request({"reference_text": "The spoken reference."})
    assert not Draft202012Validator(schema).is_valid(invalid)
    with pytest.raises(OmniAdapterError, match="requires"):
        parse_adapter_request(invalid)
    valid = _request({"reference_text": "  Spoken reference.  ", "speaker_file": "/voice.wav"})
    Draft202012Validator(schema).validate(valid)
    assert parse_adapter_request(valid).speech["reference_text"] == "Spoken reference."


def test_full_reference_requires_bundle_component_and_budgets_whole_prompt(tmp_path):
    config = Config(binary=tmp_path / "tts", model=tmp_path / "model", projector=tmp_path / "codec",
                    adaptive_context=True, max_frames=96)
    body = {"text": "a" * 3600, "speaker_file": "/voice.wav"}
    ordinary = _synthesis_spec(config, body)
    assert _context_size(config, ordinary) == 4096
    with pytest.raises(TTSError, match="bundled reference encoder"):
        _synthesis_spec(config, {**body, "reference_text": "reference"})
    encoder = tmp_path / "encoder"
    encoder.write_bytes(b"encoder")
    enabled = replace(config, reference_encoder=encoder)
    full = _synthesis_spec(enabled, {**body, "reference_text": "b" * 100})
    assert _context_size(enabled, full) == 8192
    assert full.text == ordinary.text


def test_reference_files_are_ephemeral_and_cache_identity_covers_transcript_audio_and_encoder(tmp_path, monkeypatch):
    binary = tmp_path / "fake-tts"
    binary.write_text("""#!/usr/bin/env python3
import base64,json,sys
from pathlib import Path
profile=Path(sys.argv[sys.argv.index('--tts-reference-prompt-file')+1])
value=json.loads(profile.read_bytes())
speaker=Path(sys.argv[sys.argv.index('--tts-speaker-file')+1])
marker=value['reference_text'].encode()+speaker.read_bytes()
def frame(kind,data=b''):
 sys.stdout.buffer.write(kind.encode()+len(data).to_bytes(8,'little')+data)
 sys.stdout.buffer.flush()
frame('R')
for line in sys.stdin.buffer:
 frame('A',marker+base64.b64decode(line))
 frame('D')
""")
    binary.chmod(0o755)
    model, codec, encoder, voice = [tmp_path / n for n in ("model", "codec", "encoder", "voice.wav")]
    for p in (model, codec, encoder, voice):
        p.write_bytes(p.name.encode())
    config = Config(binary=binary, model=model, projector=codec, reference_encoder=encoder, timeout_s=5)
    encoded, snapshot_paths = [], []

    def encode(_config, source):
        snapshot_paths.append(source)
        encoded.append(source.read_bytes())
        return [[1] * 16] * 19

    monkeypatch.setattr("runtime.tts_server._encode_reference", encode)
    spec = _synthesis_spec(config, {"text": "first", "speaker_file": str(voice), "reference_text": "voice A"})
    worker = PersistentTTSWorker(config)
    try:
        assert b"".join(worker.stream(spec)) == b"voice Avoice.wavfirst"
        pid = worker.pid
        assert all(not path.exists() for path in snapshot_paths)
        assert b"".join(worker.stream(replace(spec, text="second"))) == b"voice Avoice.wavsecond"
        assert worker.pid == pid and len(encoded) == 1
        worker.close()
        assert b"".join(worker.stream(spec)) == b"voice Avoice.wavfirst"
        assert worker.pid != pid and len(encoded) == 1 and worker.reference_cache_hits == 1
        changed_text = replace(spec, reference_text="voice B")
        assert b"".join(worker.stream(changed_text)) == b"voice Bvoice.wavfirst"
        assert len(encoded) == 2
        voice.write_bytes(b"changed")
        assert b"".join(worker.stream(changed_text)) == b"voice Bchangedfirst"
        encoder.write_bytes(b"changed checkpoint")
        assert b"".join(worker.stream(changed_text)) == b"voice Bchangedfirst"
        assert len(encoded) == 4
        assert all(not path.exists() for path in snapshot_paths)
    finally:
        worker.close()


def test_reference_preparation_failure_removes_private_snapshot(tmp_path, monkeypatch):
    paths = []
    voice, encoder = tmp_path / "voice.wav", tmp_path / "encoder"
    voice.write_bytes(b"original")
    encoder.write_bytes(b"encoder")
    config = Config(binary=tmp_path / "tts", model=encoder, projector=encoder, reference_encoder=encoder)

    def fail(_config, source):
        paths.append(source)
        raise TTSError("encoding failed")

    monkeypatch.setattr("runtime.tts_server._encode_reference", fail)
    worker = PersistentTTSWorker(config)
    spec = _synthesis_spec(config, {"text": "target", "speaker_file": str(voice), "reference_text": "reference"})
    with pytest.raises(TTSError, match="encoding failed"):
        worker.ensure(spec)
    assert paths and all(not p.exists() for p in paths)
    assert worker.process is None


def test_encoder_admission_failure_restores_existing_reservation_before_loading(tmp_path, monkeypatch):
    encoder, binary = tmp_path / "encoder", tmp_path / "llama-tts-reference-encode"
    encoder.write_bytes(b"encoder")
    binary.write_bytes(b"not executed")
    config = Config(binary=tmp_path / "llama-tts", model=encoder, projector=encoder,
                    reference_encoder=encoder, lease_token="test-lease", gpu_uuid="GPU-test")
    calls = []

    def transition(action, *_):
        calls.append(action)
        if action == "prepare":
            raise TTSError("admission timed out")

    monkeypatch.setattr("runtime.tts_server._normalize_reference", lambda *_: (tmp_path / "normalized", 24000))
    monkeypatch.setattr("runtime.tts_server._broker_transition", transition)
    with pytest.raises(TTSError, match="admission timed out"):
        _encode_reference(config, tmp_path / "voice")
    assert calls == ["prepare", "ready"]
