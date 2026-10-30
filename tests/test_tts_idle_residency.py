from __future__ import annotations

import time
from dataclasses import replace

import pytest

from qwen_omni_adapters import tts_residency
from runtime.tts_server import (
    Config,
    TTSError,
    _batch_specs,
    _command,
    _context_size,
    _synthesis_spec,
    create_app,
)


@pytest.fixture
def native_protocol(tmp_path):
    binary = tmp_path / "protocol-worker"
    binary.write_text("""#!/usr/bin/env python3
import base64,sys,time
def frame(kind,payload=b''):
 sys.stdout.buffer.write(kind.encode()+len(payload).to_bytes(8,'little')+payload)
 sys.stdout.buffer.flush()
frame('R')
for line in sys.stdin.buffer:
 text=base64.b64decode(line.strip())
 pcm=b''.join(int(value).to_bytes(2,'little') for value in text)
 frame('A',pcm)
 if text==b'pause': time.sleep(.6)
 frame('A',pcm)
 frame('D')
""")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"protocol fixture, not model weights")
    return binary, model


def application(native_protocol, seconds, **options):
    binary, model = native_protocol
    app = create_app(Config(binary=binary, model=model, projector=model, gpu_layers=0,
                            require_gpu=False, idle_keepalive_s=seconds, timeout_s=3, **options))
    return app, app.test_client(), app.extensions["tts_idle_residency"]


def wait_for_release(controller):
    deadline = time.monotonic() + 3
    while controller.worker.pid is not None and time.monotonic() < deadline:
        time.sleep(.02)
    assert controller.worker.pid is None


@pytest.mark.parametrize("seconds", [0, 2])
def test_http_requests_retain_only_matching_profiles_when_opted_in(native_protocol, tmp_path, seconds):
    app, client, controller = application(native_protocol, seconds)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-A")
    try:
        body = {"text": "first", "speaker_file": str(reference)}
        assert client.post("/synthesize", json=body).status_code == 200
        first = controller.worker.pid
        assert bool(first) == bool(seconds)
        assert client.post("/synthesize", json={**body, "text": "second"}).status_code == 200
        assert controller.worker.pid == first
        if seconds:
            reference.write_bytes(b"reference-B")
            assert client.post("/synthesize", json=body).status_code == 200
            assert controller.worker.pid != first
        assert client.post("/residency", json={"action": "shed"}).status_code == 200
        assert controller.worker.pid is None
    finally:
        controller.stop()
        controller.worker.close()


def test_idle_timeout_does_not_interrupt_an_active_stream_and_cancel_discards_it(native_protocol):
    app, client, controller = application(native_protocol, .1)
    try:
        assert client.post("/synthesize", json={"text": "first"}).status_code == 200
        before = controller.worker.pid
        response = client.post("/synthesize/stream", json={"text": "pause"}, buffered=False)
        next(iter(response.response))
        process = controller.worker.process
        time.sleep(.35)  # exceed idle TTL while generation owns the request lock
        assert controller.worker.pid == before and process.poll() is None
        response.close()
        assert controller.worker.pid is None and process.poll() is not None
        assert client.get("/healthz").get_json()["idle_residency_state"] == "released:incomplete"
        assert client.post("/synthesize", json={"text": "fresh"}).status_code == 200
        assert controller.worker.pid != before
        wait_for_release(controller)
        assert controller.last_transition == "released:expired"
    finally:
        controller.stop()
        controller.worker.close()


@pytest.mark.parametrize("telemetry", ["low", "missing", "error"])
def test_idle_weights_are_released_when_unified_headroom_becomes_unsafe(native_protocol, monkeypatch, telemetry):
    monkeypatch.setattr(tts_residency, "is_tegra", lambda: True)
    monkeypatch.setattr(tts_residency, "gpu_facts", lambda: [{"vram_total_mib": 32000, "vram_used_mib": 10000}])
    app, client, controller = application(native_protocol, 30)
    try:
        assert client.post("/synthesize", json={"text": "first"}).status_code == 200
        assert controller.worker.pid is not None
        def changed():
            if telemetry == "error":
                raise OSError("telemetry unavailable")
            return None if telemetry == "missing" else [{"vram_total_mib": 32000, "vram_used_mib": 31000}]
        monkeypatch.setattr(tts_residency, "gpu_facts", changed)
        wait_for_release(controller)
        assert controller.last_transition == "released:headroom"
    finally:
        controller.stop()
        controller.worker.close()


@pytest.mark.parametrize("text,tier", [("Hello", 4096), ("a" * 4096, 8192),
                                    ("你" * 4096, 16384), ("𐍈" * 4096, 32768)])
def test_context_covers_complete_utf8_input_and_frame_budget(native_protocol, text, tier, tmp_path):
    binary, model = native_protocol
    config = Config(binary=binary, model=model, projector=model, adaptive_context=True)
    spec = _synthesis_spec(config, {"text": text})
    assert _context_size(config, spec) == tier
    command = _command(config, spec, tmp_path / "speech.wav")
    assert command[command.index("--ctx-size") + 1] == str(tier)
    assert command[command.index("--prompt") + 1] == text
    assert command[command.index("--n-predict") + 1] == "512"
    assert _context_size(config, spec) >= len(text.encode()) + spec.frames + 256
    with pytest.raises(TTSError, match="frame budget"):
        _context_size(config, replace(spec, frames=32768))
    if tier > 4096:
        with pytest.raises(TTSError, match="does not cover"):
            _context_size(config, replace(spec, context_size=4096))
    native = replace(config, adaptive_context=False)
    assert _context_size(native, spec) == 0
    assert "--ctx-size" not in _command(native, spec, tmp_path / "native.wav")


def test_context_tiers_reload_only_when_needed_and_batch_reserves_all_members(native_protocol, monkeypatch):
    app, client, controller = application(native_protocol, 30, adaptive_context=True)
    try:
        assert client.post("/synthesize", json={"text": "first"}).status_code == 200
        small = controller.worker.pid
        assert client.post("/synthesize", json={"text": "second"}).status_code == 200
        assert controller.worker.pid == small
        long_text = "𐍈" * 4096
        before = controller.worker.ensure
        observed = []
        def ensure(spec):
            before(spec)
            observed.append(controller.worker.pid)
        monkeypatch.setattr(controller.worker, "ensure", ensure)
        blocks = ["first", long_text, "second"]
        specs = _batch_specs(controller.worker.config, {"blocks": blocks})
        assert [spec.text for spec in specs] == blocks
        assert {spec.context_size for spec in specs} == {32768}
        response = client.post("/synthesize/stream/batch", json={"blocks": blocks})
        assert response.status_code == 200 and response.data
        assert len(observed) == 3 and len(set(observed)) == 1 and observed[0] != small
        command = controller.worker.process.args
        assert command[command.index("--ctx-size") + 1] == "32768"
    finally:
        controller.stop()
        controller.worker.close()
