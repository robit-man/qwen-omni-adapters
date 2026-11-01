from __future__ import annotations

import base64
import io
import json
import struct
import wave

import httpx
import pytest

from qwen_omni_adapters.audio import is_digital_silence
from qwen_omni_adapters.contract import ADAPTER_SCHEMA, parse_adapter_request
from runtime.adapter_server import Config, execute, execute_stream


def pcm_wav(pcm=b"\0\0" * 32000, *, width=2, channels=1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def request_body(raw, *, require_speech=True, task="chat"):
    return {
        "model": "robit/ornith-1.5-omni:edge-v2",
        "messages": [{"role": "user", "content": "Describe this audio.",
                      "audios": [{"data": base64.b64encode(raw).decode()}]}],
        "omni": {"schema": ADAPTER_SCHEMA, "task": task, "require_speech": require_speech},
        "response_modalities": ["text", "audio"],
        "speech_mode": "always",
        "think": False,
        "stream": False,
    }


CONFIG = Config("http://comprehension/v1/chat/completions", "ornith",
                "http://language", "http://tts/synthesize", 1)


@pytest.mark.parametrize("raw", [pcm_wav(), pcm_wav(b"\0\0"), pcm_wav(b"\0\0" * 16385)])
def test_complete_zero_pcm_is_silence(raw):
    assert is_digital_silence(raw)


@pytest.mark.parametrize("raw", [
    pcm_wav(b"\1\0" + b"\0\0" * 31999),
    pcm_wav(b"\0\0" * 31999 + b"\0\x80"),
    pcm_wav(b"\0\0" * 16384 + b"\1\0"),
    pcm_wav(b""), pcm_wav()[:-1], b"", b"RIFF\0\0\0\0WAVE",
    pcm_wav(b"\x80" * 32000, width=1),
    pcm_wav(b"\0" * 96000, width=3),
    pcm_wav(b"\0" * 128000, width=4),
    pcm_wav(b"\0" * 128000, channels=2),
    pcm_wav(rate=24000),
])
def test_nonzero_truncated_empty_and_unsupported_are_never_proven_silent(raw):
    assert not is_digital_silence(raw)


def test_partial_sample_and_oversized_declared_data_are_not_silence():
    raw = bytearray(pcm_wav(b"\0\0" * 4))
    data = raw.index(b"data") + 4
    struct.pack_into("<I", raw, data, 9)
    raw.append(0)
    struct.pack_into("<I", raw, 4, len(raw) - 8)
    assert not is_digital_silence(bytes(raw))
    struct.pack_into("<I", raw, data, 0xFFFFFFFE)
    assert not is_digital_silence(bytes(raw))


@pytest.mark.parametrize("streaming", [False, True])
def test_silent_live_turn_uses_no_backend_and_preserves_provenance(streaming):
    def unexpected(_):
        pytest.fail("Exact silence must not invoke comprehension, language or cloning")

    body = request_body(pcm_wav())
    # Reference audio is not perceptual input and must neither be inspected nor
    # saved by the silence check.
    body["speech"] = {"speaker_audio": {"data": base64.b64encode(pcm_wav(b"\1\0")).decode()}}
    parsed = parse_adapter_request(body)
    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        if streaming:
            events = [json.loads(event) for event in execute_stream(parsed, CONFIG, client)]
            assert [event["type"] for event in events] == ["stage", "observation", "final"]
            result = events[-1]["response"]
        else:
            result = execute(parsed, CONFIG, client)
    assert result["message"]["content"] == ""
    assert "audio" not in result["message"]
    assert result["adapter"]["tts_skipped_reason"] == "required_speech_not_found"
    assert "all recorded samples are zero" in result["adapter"]["audio_observation"]
    assert result["adapter"]["evidence_provenance"]["current_visual_input"] is False


@pytest.mark.parametrize("variant", ["one_lsb", "mixed_audio", "image", "truncated"])
def test_nonzero_mixed_media_and_incomplete_audio_still_reach_neural_backend(variant):
    body = request_body(pcm_wav())
    message = body["messages"][0]
    if variant == "one_lsb":
        message["audios"][0]["data"] = base64.b64encode(pcm_wav(b"\1\0")).decode()
    elif variant == "mixed_audio":
        message["audios"].append({"data": base64.b64encode(pcm_wav(b"\1\0")).decode()})
    elif variant == "image":
        message["images"] = [base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 16).decode()]
    else:
        message["audios"][0]["data"] = base64.b64encode(pcm_wav()[:-1]).decode()
    calls = []

    def backend(request):
        calls.append(request.url.host)
        return httpx.Response(200, json={"choices": [{"message": {"content":
            "<speech_transcript></speech_transcript><audio_observation>Observed input.</audio_observation>"}}]})

    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        execute(parse_adapter_request(body), CONFIG, client)
    assert calls == ["comprehension"]


def test_silence_does_not_discard_explicit_text_question():
    calls = []

    def backend(request):
        calls.append(request.url.host)
        assert "all recorded samples are zero" in request.content.decode()
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "The recording is silent."}})

    body = request_body(pcm_wav(), require_speech=False)
    body["response_modalities"] = ["text"]
    body["speech_mode"] = "never"
    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        result = execute(parse_adapter_request(body), CONFIG, client)
    assert calls == ["language"]
    assert result["message"]["content"] == "The recording is silent."
