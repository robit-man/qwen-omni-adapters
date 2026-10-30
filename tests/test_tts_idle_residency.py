from __future__ import annotations

import time

import pytest

from qwen_omni_adapters import tts_residency
from runtime.tts_server import Config, create_app


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


def application(native_protocol, seconds):
    binary, model = native_protocol
    app = create_app(Config(binary=binary, model=model, projector=model, gpu_layers=0,
                            require_gpu=False, idle_keepalive_s=seconds, timeout_s=3))
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
