"""The semantic ASR verdict must reach history before any conversational reply."""

import base64
import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from qwen_omni_adapters.contract import ADAPTER_SCHEMA, parse_adapter_request
from runtime.adapter_server import Config, _parse_live_turn_route, execute, execute_stream

CASES = json.loads((Path(__file__).parent / "fixtures/live_speech_evidence.json").read_text())


def _request():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 160)
    return parse_adapter_request({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Current live audio.", "audios": [{
            "data": base64.b64encode(out.getvalue()).decode(),
        }]}],
        "omni": {"schema": ADAPTER_SCHEMA, "task": "chat", "require_speech": True},
        "response_modalities": ["text", "audio"],
        "speech_mode": "always",
        "think": False,
        "stream": False,
    })


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("candidate", [case["text"] for case in CASES if not case["speech"]])
def test_typed_invalid_speech_never_reaches_reply_tts_or_transcript(streaming, candidate):
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "comprehension":
            text = (f"<speech_transcript>{candidate}</speech_transcript>"
                    "<audio_observation>A steady fan.</audio_observation>")
        elif request.url.host == "language":
            payload = json.loads(request.content)
            assert "route live ASR turns" in payload["messages"][0]["content"]
            assert payload["messages"][-1]["content"] == candidate
            text = "INVALID_SPEECH"
        else:
            raise AssertionError("Invalid speech must not start TTS")
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    config = Config("http://comprehension", "test-model", "http://language",
                    "http://tts", 10, live_addressee_gate=True)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if streaming:
            events = [json.loads(chunk) for chunk in execute_stream(_request(), config, client)]
            observation = next(event for event in events if event["type"] == "observation")
            assert "transcript" not in observation
            assert candidate not in observation["content"]
            result = events[-1]["response"]
        else:
            result = execute(_request(), config, client)
    # One semantic validation call is permitted; conversational generation is not.
    assert hosts == ["comprehension", "language"]
    assert result["adapter"]["route"] == ["comprehension"]
    assert result["adapter"]["tts_skipped_reason"] == "required_speech_not_found"
    assert "input_transcript" not in result["adapter"]
    assert candidate not in result["adapter"]["observation"]
    assert result["adapter"]["audio_observation"] == "A steady fan."
    assert result["message"]["content"] == ""
    assert "audio" not in result["message"]


@pytest.mark.parametrize("verdict", ["", "uncertain", "INVALID_SPEECH|NONE", "<think>INVALID_SPEECH</think>"])
def test_unknown_verdict_cannot_discard_a_real_spoken_turn(verdict):
    assert _parse_live_turn_route(verdict).speech_valid is True
