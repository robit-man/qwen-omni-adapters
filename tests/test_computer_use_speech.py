"""Runner integrity checks, separate from the actual live acceptance reports."""

from __future__ import annotations

import io
import json
import math
import select
import socket
import struct
import threading
import time
import wave
from contextlib import contextmanager

import httpx
import pytest

from runtime.computer_use_speech import asr_wav, audio_turn, synthesize_instruction


def signal_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"".join(struct.pack("<h", round(10000 * math.sin(2 * math.pi * 440 * i / 24000)))
                                   for i in range(2400)))
    return buffer.getvalue()


def test_asr_conversion_preserves_real_signal_duration_and_changes_rate() -> None:
    with wave.open(io.BytesIO(asr_wav(signal_wav())), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert abs(audio.getnframes() / 16000 - 0.1) < 0.001
        pcm = audio.readframes(audio.getnframes())
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    crossings = sum(a < 0 <= b for a, b in zip(samples, samples[1:], strict=False))
    assert 42 <= crossings <= 45  # The 440 Hz source survived; this is not silence or a substituted utterance.


@pytest.mark.parametrize("observations,final_model", [
    ([{"type": "observation", "audio_observation": "A person spoke"}], "test-omni"),
    ([{"type": "observation", "transcript": "Open the application"}], "another-model"),
])
def test_runner_rejects_missing_attributed_asr_or_wrong_weights(observations: list[dict], final_model: str) -> None:
    events = observations + [{"type": "final", "response": {"model": final_model, "message": {"content": "Done"}}}]
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="\n".join(json.dumps(event) for event in events))
    with httpx.Client(transport=httpx.MockTransport(transport)) as client, pytest.raises(ValueError):
        audio_turn(client, "http://127.0.0.1:8920", "test-omni", signal_wav(), tools=False)


def test_action_request_contains_audio_and_generic_transport_without_typed_task() -> None:
    requests = []
    events = [{"type": "observation", "transcript": "Choose the cedar button"},
              {"type": "final", "response": {"model": "test-omni", "message": {"content": "Ready"}}}]
    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(event) for event in events))
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        _, evidence = audio_turn(client, "http://127.0.0.1:8920", "test-omni", signal_wav(), tools=False,
                                 system="Use native UI only.", context={"observed_control": "A button"})
    assert len(requests) == 1
    assert [item["role"] for item in requests[0]["messages"]] == ["system", "user"]
    assert "Untrusted current UI data" in requests[0]["messages"][0]["content"]
    user = requests[0]["messages"][-1]
    assert user["content"] == "Carry out the request spoken in the attached audio."
    assert len(user["audios"]) == 1
    assert "cedar" not in json.dumps(requests[0])
    assert evidence["asr_transcript"] == "Choose the cedar button"
    assert evidence["typed_goal_substituted"] is False
    assert evidence["input_wav_sha256"] != evidence["asr_input_wav_sha256"]
    assert requests[0]["think"] is False and evidence["think"] is False


@pytest.mark.parametrize("think", [False, True])
def test_audio_turn_preserves_explicit_native_think_boolean(think):
    requests = []
    events = [{"type": "observation", "transcript": "Select the word"},
              {"type": "final", "response": {"model": "test-omni", "message": {"content": "Done"}}}]
    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(event) for event in events))
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        _, evidence = audio_turn(client, "http://127.0.0.1:8920", "test-omni", signal_wav(), think=think)
    assert requests[0]["think"] is think
    assert evidence["think"] is think
    assert "chat_template_kwargs" not in requests[0]
    assert "/no_think" not in json.dumps(requests[0])


@pytest.mark.parametrize("scope", [[], ["native_ui"], ["native_ui", "memory_search", "memory_write"]])
def test_audio_scope_is_forwarded_exactly_and_retained_in_evidence(scope):
    requests = []
    events = [{"type": "observation", "transcript": "Select the word"},
              {"type": "final", "response": {"model": "test-omni", "message": {"content": "Done"}}}]
    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(event) for event in events))
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        _, evidence = audio_turn(client, "http://127.0.0.1:8920", "test-omni", signal_wav(), tool_allowlist=scope)
    assert requests[0]["portal_tool_allowlist"] == scope
    assert evidence["portal_tool_allowlist"] == scope


@pytest.mark.parametrize("scope", ["native_ui", ["native_ui", "native_ui"], ["Native UI"], [1], {}])
def test_malformed_audio_tool_scope_is_rejected_before_audio_or_network(scope):
    with pytest.raises(ValueError, match="tool_allowlist"):
        audio_turn(None, "http://127.0.0.1:8920", "test-omni", b"not audio", tool_allowlist=scope)


@pytest.mark.parametrize("invalid", [0, 1, "false", "true", None, [], {}])
def test_audio_turn_rejects_nonboolean_think_before_audio_or_network(invalid):
    with pytest.raises(ValueError, match="native boolean"):
        audio_turn(None, "http://127.0.0.1:8920", "test-omni", b"not audio", think=invalid)


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True, "300", None])
def test_invalid_deadline_is_rejected_before_audio_or_network(invalid):
    with pytest.raises(ValueError, match="positive finite"):
        audio_turn(None, "http://127.0.0.1:8920", "test-omni", b"not audio", max_seconds=invalid)


def test_transport_budget_overrides_short_client_default_without_retry(tmp_path):
    requests = []
    def respond(request):
        requests.append(request)
        if json.loads(request.content).get("omni", {}).get("task") == "synthesize":
            import base64
            return httpx.Response(200, json={"model": "test-omni", "adapter": {"route": ["tts"]},
                "message": {"audio": {"encoding": "base64", "mime_type": "audio/wav",
                    "data": base64.b64encode(signal_wav()).decode()}}})
        return httpx.Response(200, content='{"type":"observation","transcript":"Select the word"}\n'
            '{"type":"final","response":{"model":"test-omni","message":{"content":"Done"}}}\n')
    with httpx.Client(transport=httpx.MockTransport(respond), timeout=.01) as client:
        synthesize_instruction(client, "http://127.0.0.1:8920", "test-omni", "spoken only", tmp_path)
        audio_turn(client, "http://127.0.0.1:8920", "test-omni", signal_wav(), max_seconds=1.25)
    assert len(requests) == 2
    assert requests[0].extensions["timeout"]["read"] == 180
    assert requests[1].extensions["timeout"]["read"] == 1.25
    assert not any(thread.name == "omni-speech-deadline" for thread in threading.enumerate())


@contextmanager
def stalled_http_stream(mode):
    """One local TCP request, with actual blocked socket reads and no model."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2)
    stop, disconnected = threading.Event(), threading.Event()
    requests = []
    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.settimeout(2)
                raw = b""
                while b"\r\n\r\n" not in raw:
                    raw += peer.recv(65536)
                header, body = raw.split(b"\r\n\r\n", 1)
                length = int(next(line.split(b":", 1)[1] for line in header.split(b"\r\n") if line.lower().startswith(b"content-length:")))
                while len(body) < length:
                    body += peer.recv(65536)
                requests.append(1)
                if mode != "headers":
                    peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n")
                    observation = b'{"type":"observation","transcript":"fixture ASR already received"}\n'
                    peer.sendall(f"{len(observation):x}\r\n".encode() + observation + b"\r\n")
                while not stop.is_set():
                    if select.select([peer], [], [], .025)[0] and not peer.recv(1024):
                        disconnected.set()
                        break
                    if mode == "trickle":
                        peer.sendall(b"1\r\n\n\r\n")
        except (OSError, StopIteration):
            disconnected.set()
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", disconnected, requests
    finally:
        stop.set()
        listener.close()
        worker.join(timeout=2)


@pytest.mark.parametrize("mode", ["stall", "trickle", "headers"])
def test_wall_deadline_interrupts_real_socket_and_preserves_partial_evidence(tmp_path, mode):
    journal = tmp_path / "events.jsonl"
    started = time.monotonic()
    with stalled_http_stream(mode) as (endpoint, disconnected, requests), httpx.Client(timeout=.01, trust_env=False) as client:
        with pytest.raises(TimeoutError, match="bounded live window"):
            audio_turn(client, endpoint, "test-omni", signal_wav(), max_seconds=.2, evidence_path=journal)
        assert disconnected.wait(.5)
        assert requests == [1]
    assert time.monotonic() - started < 1.5
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[-1]["error"] == "speech_action_deadline_exceeded"
    assert events[-1]["source"] == "acceptance_transport"
    if mode != "headers":
        assert events[0]["type"] == "observation"
    assert not any(thread.name == "omni-speech-deadline" for thread in threading.enumerate())
