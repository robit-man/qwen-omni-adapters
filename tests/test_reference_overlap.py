import json
import threading
from dataclasses import replace

import httpx
import pytest

from qwen_omni_adapters.contract import parse_adapter_request
from runtime import adapter_server as adapter
from runtime import tts_server as tts


def config(tmp_path):
    files = [tmp_path / name for name in ("tts", "model", "codec", "encoder", "voice.wav")]
    for path in files:
        path.write_bytes(path.name.encode())
    return tts.Config(
        binary=files[0],
        model=files[1],
        projector=files[2],
        reference_encoder=files[3],
        prepare_reference=True,
    ), files[-1]


def test_saved_reference_preparation_is_ephemeral_and_reused_without_loading_tts(
    tmp_path, monkeypatch
):
    cfg, voice = config(tmp_path)
    encoded = []

    def encode(_config, source, *, cancel_event=None):
        encoded.append(source)
        return [[1] * 16] * 19

    monkeypatch.setattr(tts, "_encode_reference", encode)
    worker = tts.PersistentTTSWorker(cfg)
    spec = tts._synthesis_spec(
        cfg, {"text": "future response", "speaker_file": str(voice), "reference_text": "reference"}
    )
    worker.prepare_reference(spec, threading.Event())
    assert worker.pid is None and not worker.ready
    assert worker.reference_encodes == 1 and not encoded[0].exists()
    cached = worker._reference_material(spec)
    assert json.loads(cached)["codes"] == [[1] * 16] * 19
    assert worker.reference_encodes == 1 and worker.reference_cache_hits == 1
    voice.write_bytes(b"new reference")
    worker.prepare_reference(spec, threading.Event())
    assert worker.reference_encodes == 2 and all(not p.exists() for p in encoded)


def test_actual_speech_cancels_unrelated_reference_work_before_waiting_for_the_request_lock(
    tmp_path, monkeypatch
):
    cfg, voice = config(tmp_path)
    started = threading.Event()
    cancelled = threading.Event()

    def encode(_config, _source, *, cancel_event=None):
        started.set()
        assert cancel_event.wait(2)
        cancelled.set()
        raise tts.TTSError("cancelled")

    monkeypatch.setattr(tts, "_encode_reference", encode)
    monkeypatch.setattr(tts.PersistentTTSWorker, "stream", lambda *_: iter([b"\0\0" * 240]))
    app = tts.create_app(cfg)
    result = []

    def prepare():
        with app.test_client() as client:
            result.append(
                client.post(
                    "/prepare-reference",
                    json={"speaker_file": str(voice), "reference_text": "reference"},
                ).json
            )

    thread = threading.Thread(target=prepare)
    try:
        thread.start()
        assert started.wait(2)
        with app.test_client() as client:
            response = client.post("/synthesize/stream", json={"text": "committed"})
            assert response.get_data() == b"\0\0" * 240
        thread.join(timeout=2)
        assert not thread.is_alive() and cancelled.is_set()
        assert result == [{"prepared": False}]
    finally:
        app.extensions["tts_idle_residency"].stop()


def test_preparation_defaults_off_and_skips_embedding_only_profiles(tmp_path):
    cfg, voice = config(tmp_path)
    app = tts.create_app(replace(cfg, prepare_reference=False))
    assert app.test_client().post("/prepare-reference", json={}).status_code == 404
    app.extensions["tts_idle_residency"].stop()
    app = tts.create_app(cfg)
    assert app.test_client().post("/prepare-reference", json={"speaker_file": str(voice)}).json == {
        "prepared": False
    }
    app.extensions["tts_idle_residency"].stop()


def test_adapter_prepares_only_reference_fields_and_bounds_background_work(monkeypatch):
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(adapter, "_REFERENCE_PREPARATION_SLOT", slot)
    started, release = threading.Event(), threading.Event()
    seen = []

    def handle(request):
        seen.append((request.url.path, json.loads(request.content)))
        started.set()
        assert release.wait(2)
        return httpx.Response(200, json={"prepared": True})

    cfg = adapter.Config(
        comprehension_url="",
        comprehension_model="",
        language_url="http://language",
        tts_url="http://speech/synthesize",
        timeout_s=3,
        prepare_reference=True,
    )
    parsed = parse_adapter_request(
        {
            "model": "example",
            "stream": False,
            "messages": [{"role": "user", "content": "uncommitted question"}],
            "response_modalities": ["text", "audio"],
            "speech": {"speaker_file": "/voice.wav", "reference_text": "reference"},
        }
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter._start_reference_preparation(parsed, cfg, client)
        assert started.wait(2)
        adapter._start_reference_preparation(parsed, cfg, client)
        release.set()
        assert slot.acquire(timeout=2)
        slot.release()
    assert seen == [
        ("/prepare-reference", {"speaker_file": "/voice.wav", "reference_text": "reference"})
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_preparation_overlaps_language_without_speaking_unresolved_tools(monkeypatch, stream):
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(adapter, "_REFERENCE_PREPARATION_SLOT", slot)
    prepared = threading.Event()
    paths = []
    tool = {"function": {"name": "lookup", "arguments": {"query": "current evidence"}}}

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/prepare-reference":
            assert json.loads(request.content) == {
                "speaker_file": "/voice.wav",
                "reference_text": "reference",
            }
            prepared.set()
            return httpx.Response(200, json={"prepared": True})
        assert request.url.path == "/api/chat"
        assert prepared.wait(2), "reference preparation did not run during language inference"
        result = {
            "message": {"role": "assistant", "content": "Uncommitted text", "tool_calls": [tool]},
            "done": True,
        }
        return httpx.Response(200, content=json.dumps(result) + "\n")

    parsed = parse_adapter_request(
        {
            "model": "example",
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": "Look up the latest evidence"}],
            "response_modalities": ["text", "audio"],
            "speech": {"speaker_file": "/voice.wav", "reference_text": "reference"},
        }
    )
    cfg = adapter.Config(
        comprehension_url="",
        comprehension_model="",
        language_url="http://language",
        tts_url="http://speech/synthesize",
        timeout_s=3,
        prepare_reference=True,
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if stream:
            events = [json.loads(raw) for raw in adapter.execute_stream(parsed, cfg, client)]
            assert not any(event["type"].startswith("audio_") for event in events)
            result = events[-1]["response"]
        else:
            result = adapter.execute(parsed, cfg, client)
        assert slot.acquire(timeout=2)
        slot.release()
    assert result["message"]["tool_calls"] == [tool]
    assert "audio" not in result["message"]
    assert set(paths) == {"/api/chat", "/prepare-reference"}
