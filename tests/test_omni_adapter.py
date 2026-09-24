from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import subprocess
import wave
from pathlib import Path

import httpx
import pytest

from clients.python_client import printable_response
from qwen_omni_adapters.audio import decode_wav_payload
from qwen_omni_adapters.context import configured_tools
from qwen_omni_adapters.contract import (
    ADAPTER_SCHEMA,
    MediaItem,
    OmniAdapterError,
    adapter_contract,
    parse_adapter_request,
)
from runtime.adapter_server import (
    TRAINED_AUDIO_PROMPT_SHA256,
    TRAINED_AUDIO_SUFFIX_PROMPT,
    TRAINED_AUDIO_SYSTEM_PROMPT,
    AdapterStageError,
    Config,
    _natural_live_reply,
    _tts_blocks_for_request,
    _tts_text_blocks,
    _video_audio,
    build_comprehension_payload,
    build_language_payload,
    execute,
    execute_stream,
)
from runtime.tts_server import Config as TTSConfig
from runtime.tts_server import PersistentTTSWorker, TTSError, _command, _synthesis_spec
from runtime.tts_server import create_app as create_tts_app


def _wav(sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * 160)
    return out.getvalue()


def _encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


def _gif() -> bytes:
    return b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"


def _tts_config(tmp_path: Path, **overrides) -> TTSConfig:
    values = {
        "binary": tmp_path / "llama-tts",
        "model": tmp_path / "model.gguf",
        "projector": tmp_path / "projector.gguf",
        "gpu_layers": 0,
    }
    values.update(overrides)
    return TTSConfig(**values)


def _base_request(**overrides):
    request = {
        "model": "robit/qwen3.8-omni:latest",
        "messages": [{"role": "user", "content": "What happened?"}],
        "omni": {"schema": ADAPTER_SCHEMA, "task": "chat"},
        "response_modalities": ["text"],
        "speech_mode": "auto",
        "think": True,
        "stream": False,
    }
    request.update(overrides)
    return request


def test_adapter_contract_separates_wire_schema_from_bundle_schema() -> None:
    contract = adapter_contract()

    assert contract["schema"] == ADAPTER_SCHEMA
    assert contract["transport"]["streaming_v1"] is False
    assert contract["compatibility"]["message_extensions"] == ["audios", "videos"]
    assert contract["media"]["video"]["max_items"] == 4
    assert "environmental" in contract["response"]["adapter"]["audio_observation"]


def test_tts_stream_window_validation_and_cli_arguments(tmp_path: Path) -> None:
    config = _tts_config(tmp_path)
    assert config.stream_frames == 8
    spec = _synthesis_spec(config, {"text": "Hello", "stream_frames": 12})
    command = _command(config, spec, tmp_path / "speech.wav", stream=True)

    assert command[-3:] == ["--tts-stream", "--tts-stream-frames", "12"]
    with pytest.raises(TTSError, match="between 1 and 72"):
        _synthesis_spec(config, {"text": "Hello", "stream_frames": 0})
    with pytest.raises(TTSError, match="between 1 and 72"):
        _synthesis_spec(config, {"text": "Hello", "stream_frames": 73})


def test_tts_text_blocks_have_no_aggregate_reply_ceiling() -> None:
    sentences = [f"{index:02d} {'x' * 390}." for index in range(40)]
    text = " ".join(sentences)

    blocks = _tts_text_blocks(text, {})

    assert len(blocks) == 40
    assert " ".join(blocks) == text


@pytest.mark.parametrize(
    ("reply", "user_text", "expected"),
    [
        (
            "The result is seven. I'm here to help with anything else.",
            "What is three plus four?",
            "The result is seven.",
        ),
        (
            "Your message came through. The result is seven.",
            "What is three plus four?",
            "The result is seven.",
        ),
        (
            "The result is seven, and I'm here to help.",
            "What is three plus four?",
            "The result is seven.",
        ),
        (
            "Your microphone was silent.",
            "Was my microphone silent?",
            "Your microphone was silent.",
        ),
        (
            "The phrase 'I'm here to help' sounds canned.",
            "Why does 'I'm here to help' sound canned?",
            "The phrase 'I'm here to help' sounds canned.",
        ),
        ("That tracks.", "Does that make sense?", "That tracks."),
    ],
)
def test_natural_live_reply_filters_only_unsolicited_assistant_filler(
    reply: str, user_text: str, expected: str
) -> None:
    assert _natural_live_reply(reply, user_text) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "I'm here to help.",
        "How can I assist you today?",
        "I didn't hear anything. Please say that again.",
    ],
)
def test_natural_live_reply_fails_closed_when_only_boilerplate_remains(
    reply: str,
) -> None:
    with pytest.raises(AdapterStageError, match="only disallowed assistant boilerplate"):
        _natural_live_reply(reply, "Tell me the result.")


def test_natural_live_reply_bounds_ordinary_speech_but_preserves_requested_detail() -> None:
    reply = "First point. Second point. Third point. Fourth point."

    assert _natural_live_reply(reply, "What happened?") == (
        "First point. Second point."
    )
    assert _natural_live_reply(reply, "Explain in detail what happened.") == reply


def test_natural_live_reply_drops_unrequested_heading_and_dangling_list_teaser() -> None:
    reply = "# Analysis\nThe direct answer is seven. A few more points:\n- First\n- Second"

    assert _natural_live_reply(reply, "What is the direct answer?") == (
        "The direct answer is seven."
    )


def test_live_spoken_reply_has_a_block_circuit_breaker(monkeypatch) -> None:
    monkeypatch.setenv("OMNI_TTS_BLOCK_CHARS", "80")
    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Explain it."}],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
        )
    )
    text = " ".join(f"Sentence {index} {'x' * 70}." for index in range(5))

    with pytest.raises(AdapterStageError, match="TTS block limit"):
        _tts_blocks_for_request(
            text,
            parsed,
            _adapter_config(spoken_max_tts_blocks=2),
        )


def test_persistent_tts_worker_reuses_one_process_and_streams_framed_pcm(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import base64
import sys

def frame(kind, data=b''):
    sys.stdout.buffer.write(kind.encode() + len(data).to_bytes(8, 'little') + data)
    sys.stdout.buffer.flush()

frame('R')
for line in sys.stdin.buffer:
    prompt = base64.b64decode(line.strip())
    frame('A', b'\\x01\\x00' * max(1, len(prompt)))
    frame('D')
"""
    )
    binary.chmod(0o755)
    config = _tts_config(tmp_path, binary=binary, persistent=True, timeout_s=5)
    spec = _synthesis_spec(config, {"text": "first", "stream_frames": 1})
    worker = PersistentTTSWorker(config)
    try:
        assert b"".join(worker.stream(spec)) == b"\x01\x00" * 5
        first_pid = worker.process.pid
        second = _synthesis_spec(config, {"text": "second", "stream_frames": 1})
        assert b"".join(worker.stream(second)) == b"\x01\x00" * 6
        assert worker.process.pid == first_pid
    finally:
        worker.close()


def test_persistent_tts_worker_reports_an_active_clone_reference(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import base64
import sys

def frame(kind, data=b''):
    sys.stdout.buffer.write(kind.encode() + len(data).to_bytes(8, 'little') + data)
    sys.stdout.buffer.flush()

frame('R')
for line in sys.stdin.buffer:
    prompt = base64.b64decode(line.strip())
    frame('A', b'\\x01\\x00' * max(1, len(prompt)))
    frame('D')
"""
    )
    binary.chmod(0o755)
    reference = tmp_path / "shipped-reference.wav"
    reference.write_bytes(_wav(16000))
    config = _tts_config(tmp_path, binary=binary, persistent=True, timeout_s=5)
    spec = _synthesis_spec(
        config,
        {"text": "Clone this shipped voice.", "speaker_file": str(reference)},
    )
    worker = PersistentTTSWorker(config)
    try:
        assert b"".join(worker.stream(spec))
        assert isinstance(worker.pid, int)
        assert worker.speaker_reference_active is True
    finally:
        worker.close()
    assert worker.pid is None
    assert worker.speaker_reference_active is False


def test_nonpersistent_tts_batch_reuses_one_process_for_the_whole_utterance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import base64
import sys

def frame(kind, data=b''):
    sys.stdout.buffer.write(kind.encode() + len(data).to_bytes(8, 'little') + data)
    sys.stdout.buffer.flush()

frame('R')
for line in sys.stdin.buffer:
    prompt = base64.b64decode(line.strip())
    frame('A', b'\\x01\\x00' * len(prompt))
    frame('D')
"""
    )
    binary.chmod(0o755)
    config = _tts_config(tmp_path, binary=binary, persistent=False, timeout_s=5)
    real_popen = subprocess.Popen
    starts = 0

    def counting_popen(*args, **kwargs):
        nonlocal starts
        starts += 1
        return real_popen(*args, **kwargs)

    monkeypatch.setattr("runtime.tts_server.subprocess.Popen", counting_popen)
    response = create_tts_app(config).test_client().post(
        "/synthesize/stream/batch",
        json={"blocks": ["first", "second"]},
    )

    assert response.status_code == 200
    assert response.data == b"\x01\x00" * 11
    assert response.headers["X-Audio-Blocks"] == "2"
    assert starts == 1


def test_persistent_tts_worker_discards_protocol_after_cancelled_stream(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import base64
import sys

def frame(kind, data=b''):
    sys.stdout.buffer.write(kind.encode() + len(data).to_bytes(8, 'little') + data)
    sys.stdout.buffer.flush()

frame('R')
for line in sys.stdin.buffer:
    prompt = base64.b64decode(line.strip())
    frame('A', prompt + b':head')
    frame('A', prompt + b':tail')
    frame('D')
"""
    )
    binary.chmod(0o755)
    config = _tts_config(tmp_path, binary=binary, persistent=True, timeout_s=5)
    worker = PersistentTTSWorker(config)
    try:
        first = worker.stream(
            _synthesis_spec(config, {"text": "first", "stream_frames": 1})
        )
        assert next(first) == b"first:head"
        first.close()
        assert not worker.ready

        second = worker.stream(
            _synthesis_spec(config, {"text": "second", "stream_frames": 1})
        )
        assert b"".join(second) == b"second:headsecond:tail"
    finally:
        worker.close()


def test_persistent_tts_patch_recreates_audio_helper_per_prompt() -> None:
    patch = Path("patches/llama.cpp-qwen3tts-persistent.patch").read_text()

    assert "+        mtmd_helper::gen_audio gen(lctx, mctx.get());" in patch
    assert "+        gen.reset();" not in patch
    assert "+        common_sampler_ptr request_sampler(" in patch
    assert "common_sampler_init(model, params.sampling)" in patch
    assert "+        common_sampler_reset(smpl);" not in patch
    assert "+        mtmd_gen_audio_reset_rng(mctx.get());" in patch
    assert "+void clip_reset_rng(struct clip_ctx * ctx)" in patch


def test_tts_accepts_bounded_wav_speaker_envelope(tmp_path: Path) -> None:
    reference = io.BytesIO()
    with wave.open(reference, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    envelope = {
        "mime_type": "audio/wav",
        "encoding": "base64",
        "data": _encoded(reference.getvalue()),
    }

    spec = _synthesis_spec(
        _tts_config(tmp_path), {"text": "Clone this voice.", "speaker_audio": envelope}
    )

    assert spec.speaker == ""
    assert spec.speaker_audio == reference.getvalue()
    with pytest.raises(TTSError, match="mutually exclusive"):
        _synthesis_spec(
            _tts_config(tmp_path),
            {
                "text": "No ambiguity.",
                "speaker_file": "/trusted/reference.wav",
                "speaker_audio": envelope,
            },
        )


def test_tts_http_stream_is_header_tagged_incremental_pcm(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import sys
import wave

args = sys.argv[1:]
output = args[args.index("--output") + 1]
with wave.open(output, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    wav.writeframes(bytes([1, 2, 3, 4]))
sys.stdout.buffer.write(bytes([1, 2]))
sys.stdout.buffer.flush()
sys.stdout.buffer.write(bytes([3, 4]))
sys.stdout.buffer.flush()
"""
    )
    os.chmod(binary, 0o755)
    config = _tts_config(
        tmp_path, binary=binary, stream_frames=12, persistent=False
    )
    response = (
        create_tts_app(config)
        .test_client()
        .post(
            "/synthesize/stream",
            json={"text": "Hello"},
        )
    )

    assert response.status_code == 200
    assert response.data == bytes([1, 2, 3, 4])
    assert response.headers["X-Audio-Codec"] == "pcm_s16le"
    assert response.headers["X-Audio-Sample-Rate"] == "24000"
    assert response.headers["X-Audio-Channels"] == "1"
    assert response.headers["X-Audio-Stream-Frames"] == "12"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Accel-Buffering"] == "no"


def test_example_client_redacts_audio_base64_by_default() -> None:
    response = {"message": {"audio": {"data": "YWJj", "decoded_bytes": 3}}}

    printable = printable_response(response, include_audio_base64=False)

    assert printable["message"]["audio"]["data"] == "<base64 omitted; 4 characters>"
    assert response["message"]["audio"]["data"] == "YWJj"
    assert printable_response(response, include_audio_base64=True) is response


def test_adapter_json_schemas_are_valid_json_and_use_v1_identifier() -> None:
    schema_dir = Path("docs/schema")
    request_schema = json.loads((schema_dir / "request-v1.schema.json").read_text())
    response_schema = json.loads((schema_dir / "response-v1.schema.json").read_text())
    voice_schema = json.loads((schema_dir / "voice-profile-v1.schema.json").read_text())

    assert request_schema["properties"]["omni"]["properties"]["schema"]["const"] == ADAPTER_SCHEMA
    assert (
        response_schema["properties"]["adapter"]["properties"]["schema"]["const"] == ADAPTER_SCHEMA
    )
    assert voice_schema["properties"]["schema"]["const"] == "robit.omni.voice-profile.v1"


def test_chat_request_routes_video_and_audio_through_all_three_stages() -> None:
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "What was said and shown?",
                "audios": [
                    {
                        "mime_type": "audio/wav",
                        "encoding": "base64",
                        "data": _encoded(_wav(16000)),
                    }
                ],
                "videos": [
                    {
                        "mime_type": "video/mp4",
                        "encoding": "base64",
                        "data": _encoded(_mp4()),
                        "sampling": {"fps": 2, "max_frames": 64, "include_audio": True},
                    }
                ],
            }
        ],
        response_modalities=["text", "audio"],
    )

    parsed = parse_adapter_request(request)

    assert parsed.route == ("comprehension", "language", "tts")
    assert parsed.input_modalities == ("text", "audio", "video")
    assert parsed.media[1].options["max_frames"] == 64
    assert parsed.passthrough["think"] is True


def test_stock_ollama_bare_base64_image_is_detected_by_signature() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"example"
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Describe this image.",
                "images": [_encoded(png)],
            }
        ]
    )

    parsed = parse_adapter_request(request)

    assert parsed.media[0].mime_type == "image/png"
    assert parsed.route == ("comprehension", "language")


@pytest.mark.parametrize(
    ("task", "message", "route"),
    [
        (
            "transcribe",
            {
                "role": "user",
                "content": "Transcribe.",
                "audios": [{"data": _encoded(_wav(16000))}],
            },
            ("comprehension",),
        ),
        (
            "describe",
            {
                "role": "user",
                "content": "Describe.",
                "videos": [{"data": _encoded(_mp4()), "mime_type": "video/mp4"}],
            },
            ("comprehension",),
        ),
        (
            "synthesize",
            {"role": "user", "content": "Read this exactly."},
            ("tts",),
        ),
    ],
)
def test_direct_tasks_select_one_component(task, message, route) -> None:
    request = _base_request(
        messages=[message],
        omni={"schema": ADAPTER_SCHEMA, "task": task},
    )

    assert parse_adapter_request(request).route == route


def test_transcribe_with_spoken_output_routes_directly_to_tts() -> None:
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Transcribe.",
                "audios": [{"data": _encoded(_wav(16000))}],
            }
        ],
        omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
        response_modalities=["text", "audio"],
        speech_mode="always",
    )

    assert parse_adapter_request(request).route == ("comprehension", "tts")


def test_transcribe_with_spoken_output_executes_tts() -> None:
    output_wav = _wav(24000)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello from audio"}}]},
            )
        if request.url.host == "tts":
            assert json.loads(request.content)["text"] == "hello from audio"
            return httpx.Response(200, content=output_wav, headers={"content-type": "audio/wav"})
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Transcribe.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
            response_modalities=["text", "audio"],
            speech_mode="always",
        )
    )
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen == ["comprehension", "tts"]
    assert result["message"]["content"] == "hello from audio"
    assert result["adapter"]["speech_synthesized"] is True
    assert base64.b64decode(result["message"]["audio"]["data"]) == output_wav


def test_adapter_rejects_streaming_and_spoofed_video_mime() -> None:
    with pytest.raises(OmniAdapterError, match="stream=false"):
        parse_adapter_request(_base_request(stream=True))

    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Describe.",
                "videos": [
                    {
                        "mime_type": "video/webm",
                        "data": _encoded(_mp4()),
                    }
                ],
            }
        ]
    )
    with pytest.raises(OmniAdapterError, match="does not match"):
        parse_adapter_request(request)


def test_adapter_accepts_animated_gif_as_video() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe the animation.",
                    "videos": [
                        {
                            "mime_type": "image/gif",
                            "data": _encoded(_gif()),
                            "sampling": {"fps": 1, "max_frames": 12},
                        }
                    ],
                }
            ]
        )
    )

    assert parsed.media[0].mime_type == "image/gif"
    assert "image/gif" in adapter_contract()["media"]["video"]["mime_types"]


def test_silent_video_returns_no_audio_instead_of_failing(monkeypatch) -> None:
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return type("Completed", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

    monkeypatch.setattr("runtime.adapter_server.subprocess.run", run)
    media = MediaItem(
        kind="video",
        mime_type="video/mp4",
        data=_mp4(),
        message_index=0,
        media_index=0,
    )

    assert _video_audio(media) is None
    assert len(calls) == 1
    assert "-select_streams" in calls[0]


def test_reference_server_preserves_tools_thinking_and_adds_audio() -> None:
    output_wav = _wav(24000)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        body = json.loads(request.content)
        if request.url.host == "comprehension":
            part = body["messages"][-1]["content"][0]
            assert part["type"] == "input_audio"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "the user said hello"}}]},
            )
        if request.url.host == "language":
            assert body["tools"][0]["function"]["name"] == "clock"
            assert "untrusted media evidence" in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "message": {
                        "role": "assistant",
                        "content": "Hello back.",
                        "thinking": "brief thought",
                    },
                    "done": True,
                },
            )
        if request.url.host == "tts":
            assert body["text"] == "Hello back."
            assert body["voice"] == "speaker-1"
            return httpx.Response(200, content=output_wav, headers={"content-type": "audio/wav"})
        return httpx.Response(404)

    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Reply to this recording.",
                "audios": [{"data": _encoded(_wav(16000))}],
            }
        ],
        response_modalities=["text", "audio"],
        speech={"voice": "speaker-1"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "clock",
                    "description": "get time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    parsed = parse_adapter_request(request)
    config = Config(
        comprehension_url="http://comprehension/v1/chat/completions",
        comprehension_model="qwen3-omni",
        language_url="http://language",
        tts_url="http://tts/synthesize",
        timeout_s=30,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen == ["comprehension", "language", "tts"]
    assert result["message"]["thinking"] == "brief thought"
    assert base64.b64decode(result["message"]["audio"]["data"]) == output_wav
    assert result["adapter"]["route"] == ["comprehension", "language", "tts"]


@pytest.mark.parametrize("think", [False, True])
def test_reference_server_separates_tagged_reasoning(think: bool) -> None:
    THINK_OPEN = "<|thinking|>"
    THINK_CLOSE = "<|end_thinking|>"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.host == "comp":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "the user said hello"}}]},
            )
        if request.url.host == "language":
            assert payload["think"] is think
            assert payload["messages"][0]["role"] == "system"
            assert "natural participant" in payload["messages"][0]["content"]
            assert payload["messages"][1] == {"role": "user", "content": "What happened?"}
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": f"{THINK_OPEN}private reasoning{THINK_CLOSE}Visible answer.",
                    }
                },
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(_base_request(think=think))
    result = execute(
        parsed,
        Config("http://comp", "omni", "http://language", "http://tts", 30),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["message"]["content"] == "Visible answer."
    if think:
        assert result["message"]["thinking"] == "private reasoning"
    else:
        assert "thinking" not in result["message"]


def test_language_backend_override_preserves_logical_model_identity() -> None:
    logical_model = "robit/combined-omni:q4km"
    core_model = "robit/core-language:27b"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == core_model
        return httpx.Response(
            200,
            json={
                "model": core_model,
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
            },
        )

    parsed = parse_adapter_request(_base_request(model=logical_model))
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
        language_model=core_model,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["model"] == logical_model
    assert result["adapter"]["language_backend_model"] == core_model


def test_comprehension_payload_tags_video_for_qwen_style_server(monkeypatch) -> None:
    # These fixtures stand in for a streamable clip; the pipe-probe is exercised
    # by its own tests rather than by shelling out to ffmpeg here.
    from runtime import adapter_server

    monkeypatch.setattr(adapter_server, "_video_is_pipe_readable", lambda _data: True)

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe.",
                    "videos": [
                        {
                            "mime_type": "video/mp4",
                            "data": _encoded(_mp4()),
                            "sampling": {"fps": 8, "max_frames": 96},
                        }
                    ],
                }
            ],
            omni={"task": "describe", "include_audio_from_video": False},
        )
    )
    payload = build_comprehension_payload(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
    )

    assert payload["messages"][-1]["content"][0]["type"] == "input_video"
    assert payload["messages"][-1]["content"][0]["sampling"] == {
        "fps": 2.0,
        "max_frames": 32,
    }
    assert payload["mm_processor_kwargs"]["use_audio_in_video"] is False
    assert payload["cache_prompt"] is False
    assert payload["max_tokens"] == 2048


def test_chat_comprehension_payload_forbids_conversational_media_reply() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Reply to this recording.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ]
        )
    )

    payload = build_comprehension_payload(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
    )

    system = payload["messages"][0]
    assert system["role"] == "system"
    assert "not a conversational assistant" in system["content"]
    assert "<speech_transcript>" in system["content"]
    assert "<audio_observation>" in system["content"]
    assert "never your response" in system["content"]
    media_parts = payload["messages"][-1]["content"]
    assert [part["type"] for part in media_parts] == ["input_audio", "text"]
    assert "Reply to this recording." not in media_parts[-1]["text"]
    assert "Analyze this audio" in media_parts[-1]["text"]
    assert "non-speech evidence" in media_parts[-1]["text"]
    assert "nothing else" in media_parts[-1]["text"]


def test_trained_audio_bridge_uses_the_release_gated_prompt_contract() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "This caller text must not perturb ASR.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
        )
    )
    payload = build_comprehension_payload(
        parsed,
        Config(
            "http://comp",
            "local-audio-bridge",
            "http://language",
            "http://tts",
            30,
            comprehension_disable_thinking=True,
            comprehension_repeat_penalty=1.1,
        ),
    )

    assert TRAINED_AUDIO_PROMPT_SHA256 == (
        "9f73862652e0226ec3f9690f0a783d1c21dc1113285b4dc18d0edd51f2766758"
    )
    assert payload["messages"][0] == {
        "role": "system",
        "content": TRAINED_AUDIO_SYSTEM_PROMPT,
    }
    assert [part["type"] for part in payload["messages"][1]["content"]] == [
        "input_audio",
        "text",
    ]
    assert payload["messages"][1]["content"][-1]["text"] == TRAINED_AUDIO_SUFFIX_PROMPT
    assert "This caller text" not in json.dumps(payload["messages"])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["repeat_penalty"] == 1.1
    assert payload["cache_prompt"] is False


def test_trained_audio_chat_uses_the_same_asr_prompt_as_training() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Answer what I said.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ]
        )
    )

    payload = build_comprehension_payload(
        parsed,
        Config(
            "http://comp",
            "local-audio-bridge",
            "http://language",
            "http://tts",
            30,
            comprehension_disable_thinking=True,
        ),
    )

    assert payload["messages"][0]["content"] == TRAINED_AUDIO_SYSTEM_PROMPT
    assert payload["messages"][-1]["content"][-1]["text"] == TRAINED_AUDIO_SUFFIX_PROMPT


def test_stream_exposes_only_tagged_input_transcript_to_clients() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<speech_transcript>Haha, same, just vibing.</speech_transcript>"
                                    "<audio_observation>Soft room tone and a fan.</audio_observation>"
                                    "<visual_observation>A person is visible.</visual_observation>"
                                )
                            }
                        }
                    ]
                },
            )
        if request.url.host == "language":
            body = json.loads(request.content)
            assert body["think"] is False
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert "natural participant" in body["messages"][0]["content"]
            assert body["messages"][1]["role"] == "user"
            current = body["messages"][-1]["content"]
            assert current.startswith("Haha, same, just vibing.\n\n")
            assert "Reply naturally." not in current
            assert "<speech_transcript>" not in current
            assert "Soft room tone and a fan." in current
            assert '<adapter_observation source="current_attached_media"' in current
            assert 'modalities="audio"' in current
            assert 'current_visual_input="false"' in current
            assert "never recast audio or tool data as something seen" in current
            assert "/no_think" not in current
            return httpx.Response(
                200,
                content=(
                    b'{"message":{"role":"assistant","content":"What is the vibe?"},"done":true}\n'
                ),
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Reply naturally.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            think=False,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config(
                "http://comprehension/v1/chat/completions",
                "qwen3-omni",
                "http://language",
                "http://tts/synthesize",
                30,
            ),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    observation = next(event for event in events if event["type"] == "observation")
    assert observation["transcript"] == "Haha, same, just vibing."
    assert observation["audio_observation"] == "Soft room tone and a fan."
    assert "A person is visible" in observation["content"]
    final = events[-1]["response"]
    assert final["adapter"]["input_transcript"] == "Haha, same, just vibing."
    assert final["adapter"]["audio_observation"] == "Soft room tone and a fan."
    assert final["adapter"]["evidence_provenance"] == {
        "current_media_modalities": ["audio"],
        "current_visual_input": False,
        "tool_data_is_visual_input": False,
        "prior_dialogue_is_current_observation": False,
    }
    assert final["message"]["content"] == "What is the vibe?"


def test_environmental_audio_does_not_become_user_transcript() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<speech_transcript></speech_transcript>"
                                    "<audio_observation>Rain, a car horn, and distant "
                                    "traffic.</audio_observation>"
                                )
                            }
                        }
                    ]
                },
            )
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=(
                    b'{"message":{"role":"assistant","content":"It sounds like a rainy street."},'
                    b'"done":true}\n'
                ),
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "What is happening around me?",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            think=False,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config(
                "http://comprehension/v1/chat/completions",
                "qwen3-omni",
                "http://language",
                "http://tts/synthesize",
                30,
            ),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    observation = next(event for event in events if event["type"] == "observation")
    assert "transcript" not in observation
    assert observation["audio_observation"] == "Rain, a car horn, and distant traffic."
    final = events[-1]["response"]
    assert "input_transcript" not in final["adapter"]
    assert final["adapter"]["audio_observation"] == ("Rain, a car horn, and distant traffic.")


def test_require_speech_stops_after_sound_only_comprehension() -> None:
    requested_hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        assert request.url.host == "comprehension"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<speech_transcript></speech_transcript>"
                                "<audio_observation>A fan and room tone.</audio_observation>"
                            )
                        }
                    }
                ]
            },
        )

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Live call audio.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config(
                "http://comprehension/v1/chat/completions",
                "qwen3-omni",
                "http://language",
                "http://tts/synthesize",
                30,
            ),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert requested_hosts == ["comprehension"]
    assert [event["type"] for event in events] == ["stage", "observation", "final"]
    assert events[1]["audio_observation"] == "A fan and room tone."
    final = events[-1]["response"]
    assert final["adapter"]["route"] == ["comprehension"]
    assert final["adapter"]["tts_skipped_reason"] == "required_speech_not_found"
    assert final["message"]["content"] == ""
    assert "audio" not in final["message"]


def test_require_speech_rejects_encoder_meta_commentary_as_silence() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        assert request.url.host == "comprehension"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<speech_transcript>The user is asking me to analyze an "
                                "audio file, but I don't actually have access to any audio "
                                "content in this conversation. Without the actual audio input, "
                                "I cannot produce a speech transcript because there is nothing "
                                "for me to perceive.</speech_transcript>"
                                "<audio_observation>No audio was received. No sound content is "
                                "available for analysis.</audio_observation>"
                            )
                        }
                    }
                ]
            },
        )

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Live call audio.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            _adapter_config(),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert requested_hosts == ["comprehension"]
    observation = events[1]
    assert observation["type"] == "observation"
    assert "transcript" not in observation
    assert "audio_observation" not in observation
    final = events[-1]["response"]
    assert final["adapter"]["tts_skipped_reason"] == "required_speech_not_found"
    assert "input_transcript" not in final["adapter"]
    assert "audio_observation" not in final["adapter"]


def test_live_model_may_observe_addressed_elsewhere_speech_without_tts() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<speech_transcript>Maya, I'll call you tomorrow."
                                    "</speech_transcript>"
                                )
                            }
                        }
                    ]
                },
            )
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=b'{"message":{"role":"assistant","content":""},"done":true}\n',
            )
        raise AssertionError(f"unexpected backend {request.url.host}")

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "The attached audio contains the current room speech.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            _adapter_config(),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert requested_hosts == ["comprehension", "language"]
    assert not any(event.get("type") == "audio_delta" for event in events)
    final = events[-1]["response"]
    assert final["message"]["content"] == ""
    assert final["adapter"]["tts_skipped_reason"] == "empty_assistant_response"


def test_ambiguous_live_room_speech_keeps_camera_optional_without_forcing_it() -> None:
    from runtime import adapter_server

    tools = [entry["schema"] for entry in configured_tools()]
    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Live room audio."}],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
                "tool_routing": "relevant",
            },
            tools=tools,
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )

    payload = adapter_server.build_language_payload(
        parsed,
        "<speech_transcript>Maya, I'll call you tomorrow.</speech_transcript>",
        "language",
        config=_adapter_config(),
    )

    assert "request_camera_view" in {
        item["function"]["name"] for item in payload["tools"]
    }
    assert payload.get("tool_choice") != "required"


def test_video_context_overflow_retries_with_lower_frame_cap(monkeypatch) -> None:
    # These fixtures stand in for a streamable clip; the pipe-probe is exercised
    # by its own tests rather than by shelling out to ffmpeg here.
    from runtime import adapter_server

    monkeypatch.setattr(adapter_server, "_video_is_pipe_readable", lambda _data: True)

    seen_caps = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        cap = body["messages"][-1]["content"][0]["sampling"]["max_frames"]
        seen_caps.append(cap)
        if len(seen_caps) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "exceed_context_size_error",
                        "message": "request exceeds the available context size",
                    }
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "video understood"}}]})

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe.",
                    "videos": [
                        {
                            "mime_type": "video/mp4",
                            "data": _encoded(_mp4()),
                            "sampling": {"fps": 1, "max_frames": 24},
                        }
                    ],
                }
            ],
            omni={"task": "describe", "include_audio_from_video": False},
        )
    )
    result = execute(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen_caps == [24, 16]
    assert result["message"]["content"] == "video understood"


def test_reference_server_streams_thinking_and_text_then_final_response() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                b'{"message":{"role":"assistant","thinking":"brief "},"done":false}\n'
                b'{"message":{"role":"assistant","content":"Hello "},"done":false}\n'
                b'{"message":{"role":"assistant","content":"back."},"done":true}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    parsed = parse_adapter_request(_base_request())
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            config,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert seen[0]["stream"] is True
    assert seen[0]["think"] is True
    assert [event["type"] for event in events] == [
        "stage",
        "delta",
        "delta",
        "delta",
        "final",
    ]
    final = events[-1]["response"]
    assert final["message"]["content"] == "Hello back."
    assert final["message"]["thinking"] == "brief"
    assert final["adapter"]["text_streamed"] is True
    assert final["adapter"]["audio_streamed"] is False


def test_reference_server_stream_parser_handles_split_think_tags() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'{"message":{"role":"assistant","content":"<thi"}}\n'
                b'{"message":{"role":"assistant","content":"nk>step one</th"}}\n'
                b'{"message":{"role":"assistant","content":"ink>Answer."},'
                b'"done":true}\n'
            ),
        )

    parsed = parse_adapter_request(_base_request(think=True))
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config("http://comp", "omni", "http://language", "http://tts", 30),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    deltas = [event["message"] for event in events if event["type"] == "delta"]
    assert "".join(str(delta.get("thinking") or "") for delta in deltas) == "step one"
    assert "".join(str(delta.get("content") or "") for delta in deltas) == "Answer."
    final = events[-1]["response"]["message"]
    assert final["thinking"] == "step one"
    assert final["content"] == "Answer."


def test_disabled_stream_suppresses_orphaned_closing_think_tag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert payload["messages"][0]["role"] == "system"
        assert "natural participant" in payload["messages"][0]["content"]
        assert payload["messages"][1] == {"role": "user", "content": "What happened?"}
        return httpx.Response(
            200,
            content=(
                b'{"message":{"role":"assistant","content":"hidden chain"}}\n'
                b'{"message":{"role":"assistant","content":"</thi"}}\n'
                b'{"message":{"role":"assistant","content":"nk>Visible answer."},'
                b'"done":true}\n'
            ),
        )

    parsed = parse_adapter_request(_base_request(think=False))
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config("http://comp", "omni", "http://language", "http://tts", 30),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    deltas = [event["message"] for event in events if event["type"] == "delta"]
    assert "".join(str(delta.get("content") or "") for delta in deltas) == ("Visible answer.")
    final = events[-1]["response"]["message"]
    assert final["content"] == "Visible answer."
    assert "thinking" not in final


def test_sanitized_stream_emits_content_free_liveness_pulses() -> None:
    from runtime.adapter_server import STREAM_LIVENESS_CHUNKS

    lines = b"".join(
        b'data: {"choices":[{"delta":{"content":"word "}}]}\n'
        for _ in range(STREAM_LIVENESS_CHUNKS)
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=lines)

    parsed = parse_adapter_request(_base_request(think=False))
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config(
                "http://comp",
                "omni",
                "http://language/v1/chat/completions",
                "http://tts",
                30,
                language_api="openai",
            ),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    progress = [event for event in events if event["type"] == "progress"]
    assert progress == [{"type": "progress", "stage": "language"}]
    assert events[-1]["response"]["message"]["content"] == (
        "word " * STREAM_LIVENESS_CHUNKS
    ).strip()


def test_reference_server_streams_pcm_and_keeps_final_wav_envelope() -> None:
    pcm = b"\x01\x00\x02\x00\x03\x00"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=(b'{"message":{"role":"assistant","content":"Speak."},"done":true}\n'),
            )
        if request.url.host == "tts":
            return httpx.Response(
                200,
                content=pcm,
                headers={"x-audio-codec": "pcm_s16le"},
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            response_modalities=["text", "audio"],
            speech_mode="always",
        )
    )
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            config,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert [path for path, _body in seen] == [
        "/api/chat",
        "/synthesize/stream/batch",
    ]
    assert seen[1][1]["stream_frames"] == 8
    assert seen[1][1]["blocks"] == ["Speak."]
    assert [event["type"] for event in events] == [
        "stage",
        "delta",
        "stage",
        "audio_start",
        "audio_delta",
        "audio_end",
        "final",
    ]
    streamed = base64.b64decode(events[4]["audio"]["data"])
    assert streamed == pcm
    final = events[-1]["response"]
    decoded = decode_wav_payload(final["message"]["audio"])
    assert decoded.sample_rate_hz == 24000
    assert decoded.frames == 3
    assert final["adapter"]["audio_streamed"] is True
    assert final["adapter"]["route"] == ["language", "tts"]


def test_live_stream_filters_boilerplate_before_display_and_tts() -> None:
    pcm = b"\x01\x00\x02\x00"
    tts_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=(
                    b'{"message":{"role":"assistant","thinking":"brief private thought",'
                    b'"content":"The answer is four. "}}\n'
                    b'{"message":{"role":"assistant","content":"I am here to help."},'
                    b'"done":true}\n'
                ),
            )
        if request.url.host == "tts":
            tts_batches.append(json.loads(request.content)["blocks"])
            return httpx.Response(
                200,
                content=pcm,
                headers={"x-audio-codec": "pcm_s16le"},
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "What is two plus two?"}],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=True,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            _adapter_config(),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    visible = "".join(
        str(event.get("message", {}).get("content") or "")
        for event in events
        if event["type"] == "delta"
    )
    assert visible == "The answer is four."
    assert tts_batches == [["The answer is four."]]
    final = events[-1]["response"]
    assert final["message"]["content"] == "The answer is four."
    assert final["message"]["thinking"] == "brief private thought"


def test_live_stream_stops_runaway_decoded_audio_without_changing_codec_window() -> None:
    oversized_pcm = b"\x01\x00" * 48_001
    seen_tts_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=b'{"message":{"role":"assistant","content":"A short answer."}}\n',
            )
        if request.url.host == "tts":
            seen_tts_payload.update(json.loads(request.content))
            return httpx.Response(
                200,
                content=oversized_pcm,
                headers={"x-audio-codec": "pcm_s16le"},
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Answer briefly."}],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )

    with pytest.raises(AdapterStageError, match="audio duration limit"):
        list(
            execute_stream(
                parsed,
                _adapter_config(spoken_max_audio_seconds=2.0),
                httpx.Client(transport=httpx.MockTransport(handler)),
            )
        )

    assert seen_tts_payload["stream_frames"] == 8


def test_long_tts_stream_uses_multiple_blocks_and_one_complete_wav(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OMNI_TTS_BLOCK_CHARS", "80")
    text = (
        "First sentence explains the long response in a calm and measured way. "
        "Second sentence contains enough additional detail to require another block. "
        "Third sentence proves that the final audio continues through the ending."
    )
    tts_batches = []
    pcm = b"\x01\x00\x02\x00"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=(
                    json.dumps(
                        {
                            "message": {"role": "assistant", "content": text},
                            "done": True,
                        }
                    )
                    + "\n"
                ).encode(),
            )
        if request.url.host == "tts":
            tts_batches.append(body["blocks"])
            return httpx.Response(
                200,
                content=pcm * len(body["blocks"]),
                headers={"x-audio-codec": "pcm_s16le"},
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )
    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            Config("http://comp", "omni", "http://language", "http://tts", 30),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert len(tts_batches) == 1
    assert len(tts_batches[0]) == len(_tts_text_blocks(text, {})) == 3
    assert " ".join(tts_batches[0]) == text
    assert sum(event["type"] == "audio_start" for event in events) == 1
    assert sum(event["type"] == "audio_delta" for event in events) == 1
    final = events[-1]["response"]
    decoded = decode_wav_payload(final["message"]["audio"])
    assert decoded.frames == 6
    assert final["adapter"]["tts_blocks"] == 3


def test_video_that_a_pipe_cannot_read_is_normalized(monkeypatch) -> None:
    """The backend pipes video to ffprobe, which cannot seek to a trailing moov."""

    from runtime import adapter_server

    item = MediaItem(
        kind="video", mime_type="video/mp4", data=b"trailing-moov", message_index=0, media_index=0
    )
    monkeypatch.setattr(adapter_server, "_video_is_pipe_readable", lambda _data: False)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"normalized")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(adapter_server.subprocess, "run", fake_run)

    assert adapter_server._normalized_video_data(item, fps=2.0, max_frames=8) == b"normalized"
    assert "+faststart" in calls[0]
    # The GIF-only duration cap must not truncate ordinary video.
    assert "-t" not in calls[0]


def test_pipe_readable_video_is_passed_through_untouched(monkeypatch) -> None:
    from runtime import adapter_server

    payload = b"already-streamable"
    item = MediaItem(
        kind="video", mime_type="video/mp4", data=payload, message_index=0, media_index=0
    )
    monkeypatch.setattr(adapter_server, "_video_is_pipe_readable", lambda _data: True)

    def refuse(*args, **kwargs):
        raise AssertionError("a pipe-readable clip must not be re-encoded")

    monkeypatch.setattr(adapter_server.subprocess, "run", refuse)

    assert adapter_server._normalized_video_data(item, fps=2.0, max_frames=8) is payload


def test_a_gif_is_still_normalized_and_duration_capped(monkeypatch) -> None:
    from runtime import adapter_server

    item = MediaItem(
        kind="video", mime_type="image/gif", data=b"gif-bytes", message_index=0, media_index=0
    )

    def pipe_readable(_data):
        raise AssertionError("a GIF is normalized without probing")

    monkeypatch.setattr(adapter_server, "_video_is_pipe_readable", pipe_readable)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(adapter_server.subprocess, "run", fake_run)

    assert adapter_server._normalized_video_data(item, fps=2.0, max_frames=8) == b"mp4"
    assert "-t" in calls[0]


def test_a_truncated_probe_counts_as_not_pipe_readable(monkeypatch) -> None:
    """ffprobe can print a codec and an error at once for a partial read."""

    from runtime import adapter_server

    monkeypatch.setattr(
        adapter_server.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=b"mpeg4\n", stderr=b"partial file\n"
        ),
    )

    assert adapter_server._video_is_pipe_readable(b"anything") is False



def _adapter_config(**overrides):
    values = {
        "comprehension_url": "http://comprehension/v1/chat/completions",
        "comprehension_model": "qwen3-omni",
        "language_url": "http://language",
        "tts_url": "http://tts/synthesize",
        "timeout_s": 30,
    }
    values.update(overrides)
    return Config(**values)


# -- optional comprehension worker ---------------------------------------


def test_media_routes_fail_clearly_when_comprehension_is_not_configured() -> None:
    """A deployment can run language and speech without the largest component."""

    from runtime import adapter_server

    config = dataclasses.replace(_adapter_config(), comprehension_url="")

    with pytest.raises(adapter_server.AdapterStageError, match="not configured"):
        adapter_server._require_comprehension(config)


def test_a_configured_comprehension_url_is_accepted() -> None:
    from runtime import adapter_server

    adapter_server._require_comprehension(_adapter_config())


# -- OpenAI-compatible language backend ----------------------------------


def test_the_openai_backend_posts_to_the_url_as_given() -> None:
    from runtime import adapter_server

    ollama = _adapter_config()
    assert adapter_server.language_request_url(ollama).endswith("/api/chat")

    openai = dataclasses.replace(
        ollama,
        language_api="openai",
        language_url="http://127.0.0.1:8901/v1/chat/completions",
    )
    assert (
        adapter_server.language_request_url(openai)
        == "http://127.0.0.1:8901/v1/chat/completions"
    )


def test_ollama_only_fields_are_dropped_for_an_openai_backend() -> None:
    """An OpenAI-compatible server rejects unknown top-level keys."""

    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Hello."}],
            keep_alive="30m",
            think=False,
            options={"temperature": 0.2, "num_predict": 64},
        )
    )

    payload = adapter_server.build_language_payload(
        parsed, None, "local-qwen3-omni", "openai"
    )

    assert "keep_alive" not in payload
    assert "think" not in payload
    assert "options" not in payload
    # The sampling controls the OpenAI schema does define are carried across.
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64
    assert payload["model"] == "local-qwen3-omni"


def test_the_ollama_backend_keeps_its_native_fields() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Hello."}],
            keep_alive="30m",
            options={"temperature": 0.2},
        )
    )

    payload = adapter_server.build_language_payload(parsed, None, "ornith", "ollama")

    assert payload["keep_alive"] == "30m"
    assert payload["options"] == {"temperature": 0.2, "num_predict": 4096}
    assert "max_tokens" not in payload


def test_language_output_is_bounded_for_both_backends() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Keep this bounded."}],
            options={"num_predict": 999_999},
        )
    )
    config = _adapter_config(language_max_output_tokens=2048)

    openai = adapter_server.build_language_payload(
        parsed, None, "local", "openai", config
    )
    ollama = adapter_server.build_language_payload(
        parsed, None, "local", "ollama", config
    )

    assert openai["max_tokens"] == 2048
    assert ollama["options"]["num_predict"] == 2048


def test_live_spoken_language_uses_the_smaller_output_bound_for_both_backends() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Give me the short answer."}],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
            },
            options={"num_predict": 999_999},
        )
    )
    config = _adapter_config(
        language_max_output_tokens=2048,
        spoken_language_max_output_tokens=192,
    )

    openai = adapter_server.build_language_payload(
        parsed, None, "local", "openai", config
    )
    ollama = adapter_server.build_language_payload(
        parsed, None, "local", "ollama", config
    )

    assert openai["max_tokens"] == 192
    assert ollama["options"]["num_predict"] == 192


def test_text_chat_is_not_constrained_by_the_live_spoken_output_bound() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            messages=[{"role": "user", "content": "Write a detailed analysis."}],
            options={"num_predict": 999_999},
        )
    )
    config = _adapter_config(
        language_max_output_tokens=2048,
        spoken_language_max_output_tokens=192,
    )

    payload = adapter_server.build_language_payload(
        parsed, None, "local", "openai", config
    )

    assert payload["max_tokens"] == 2048


def test_language_output_gets_a_server_default_when_the_client_omits_one() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(messages=[{"role": "user", "content": "Hello."}])
    )
    config = _adapter_config(language_max_output_tokens=1536)

    openai = adapter_server.build_language_payload(
        parsed, None, "local", "openai", config
    )
    ollama = adapter_server.build_language_payload(
        parsed, None, "local", "ollama", config
    )

    assert openai["max_tokens"] == 1536
    assert ollama["options"]["num_predict"] == 1536


def test_relevant_tool_routing_uses_recovered_speech_and_narrows_to_leaf_tools() -> None:
    from runtime import adapter_server

    tools = [entry["schema"] for entry in configured_tools()]
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "The attached audio contains the current request.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "tool_routing": "relevant",
            },
            tools=tools,
        )
    )

    payload = adapter_server.build_language_payload(
        parsed,
        "<speech_transcript>Can you open the web browser?</speech_transcript>",
        "ornith",
        "ollama",
    )
    names = {item["function"]["name"] for item in payload["tools"]}

    assert names == {"browser_interact"}
    assert payload["tool_choice"] == "required"

    laya_selected = adapter_server.build_language_payload(
        parsed,
        "<speech_transcript>Can you open the web browser?</speech_transcript>",
        "ornith",
        "ollama",
        decision_tool_names=("browser_interact", "gui_interact"),
    )
    laya_names = {
        item["function"]["name"] for item in laya_selected["tools"]
    }
    assert laya_names == {"browser_interact", "gui_interact"}
    assert "web_search" not in laya_names


def test_live_camera_request_exposes_a_required_relevant_tool_contract() -> None:
    tools = [entry["schema"] for entry in configured_tools()]
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "The attached audio contains the current request.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
                "tool_routing": "relevant",
            },
            tools=tools,
        )
    )
    observation = (
        "<speech_transcript>I'm talking to my buddy; if you look at the camera "
        "you can see.</speech_transcript>"
        "<audio_observation>No non-speech sounds detected.</audio_observation>"
    )

    payload = build_language_payload(parsed, observation, "ornith", "ollama")
    assert payload["tool_choice"] == "required"
    assert "request_camera_view" in {
        item["function"]["name"] for item in payload["tools"]
    }
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "request_camera_view"
    ]
    assert "<required_tool_action>" in payload["messages"][0]["content"]
    assert "<audio_observation>" not in payload["messages"][-1]["content"]


def test_live_camera_refusal_is_hidden_and_retried_as_a_native_tool_call() -> None:
    language_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<speech_transcript>What are you seeing on your "
                                    "cameras?</speech_transcript>"
                                    "<audio_observation>No non-speech sounds "
                                    "detected.</audio_observation>"
                                )
                            }
                        }
                    ]
                },
            )
        if request.url.host == "language":
            body = json.loads(request.content)
            language_requests.append(body)
            assert [item["function"]["name"] for item in body["tools"]] == [
                "request_camera_view"
            ]
            if len(language_requests) == 1:
                assert body["stream"] is True
                return httpx.Response(
                    200,
                    content=(
                        b'{"message":{"role":"assistant","content":"I do not have '
                        b'camera access."},"done":true}\n'
                    ),
                )
            assert body["stream"] is False
            assert "adapter_execution_context" in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "request_camera_view",
                                    "arguments": {"mode": "still"},
                                }
                            }
                        ],
                    }
                },
            )
        raise AssertionError(f"unexpected backend {request.url.host}")

    camera_schema = next(
        entry["schema"]
        for entry in configured_tools()
        if entry["schema"]["function"]["name"] == "request_camera_view"
    )
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "The attached audio contains the current request.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "require_speech": True,
                "tool_routing": "relevant",
            },
            tools=[camera_schema],
            response_modalities=["text", "audio"],
            speech_mode="always",
            think=False,
        )
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            _adapter_config(),
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    visible = "".join(
        str(event.get("message", {}).get("content") or "")
        for event in events
        if event.get("type") == "delta"
    )
    assert "camera access" not in visible
    final = events[-1]["response"]
    assert final["message"]["tool_calls"][0]["function"] == {
        "name": "request_camera_view",
        "arguments": {"mode": "still"},
    }
    assert "audio" not in final["message"]
    assert len(language_requests) == 2


def test_fresh_visual_evidence_cannot_request_the_same_camera_bridge_again() -> None:
    tools = [entry["schema"] for entry in configured_tools()]
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "What can you see?",
                    "images": [_encoded(b"\x89PNG\r\n\x1a\nexample")],
                }
            ],
            omni={
                "schema": ADAPTER_SCHEMA,
                "task": "chat",
                "tool_routing": "relevant",
            },
            tools=tools,
        )
    )

    payload = build_language_payload(
        parsed,
        "<visual_observation>A red mug is on the desk.</visual_observation>",
        "ornith",
        "ollama",
    )

    assert "request_camera_view" not in {
        item["function"]["name"] for item in payload["tools"]
    }


def test_client_tool_routing_preserves_supplied_contract() -> None:
    from runtime import adapter_server

    tools = [
        {
            "type": "function",
            "function": {
                "name": "custom_clock",
                "description": "Return the time.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    parsed = parse_adapter_request(_base_request(tools=tools))

    payload = adapter_server.build_language_payload(
        parsed, None, "ornith", "ollama"
    )

    assert payload["tools"] == tools


def test_invalid_tool_routing_mode_is_rejected() -> None:
    with pytest.raises(OmniAdapterError, match="tool_routing"):
        parse_adapter_request(
            _base_request(
                omni={
                    "schema": ADAPTER_SCHEMA,
                    "task": "chat",
                    "tool_routing": "guess",
                }
            )
        )


def test_an_openai_response_is_normalized_into_the_ollama_shape() -> None:
    from runtime import adapter_server

    normalized = adapter_server._language_result(
        {
            "model": "local-qwen3-omni",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello there.",
                        "reasoning_content": "thinking out loud",
                    }
                }
            ],
        },
        "openai",
    )

    assert normalized["message"]["content"] == "Hello there."
    assert normalized["message"]["thinking"] == "thinking out loud"
    assert normalized["done"] is True


def test_an_openai_response_without_a_message_is_rejected() -> None:
    from runtime import adapter_server

    with pytest.raises(adapter_server.AdapterStageError, match="no assistant message"):
        adapter_server._language_result({"choices": []}, "openai")


def test_openai_stream_tool_call_fragments_are_reassembled() -> None:
    from runtime import adapter_server

    calls: dict[int, dict] = {}
    adapter_server._merge_openai_tool_call_deltas(
        calls,
        [
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "shell", "arguments": ""},
            }
        ],
    )
    adapter_server._merge_openai_tool_call_deltas(
        calls,
        [{"index": 0, "function": {"arguments": '{"command":"printf '} }],
    )
    complete = adapter_server._merge_openai_tool_call_deltas(
        calls,
        [{"index": 0, "function": {"arguments": 'ok"}'}}],
    )

    assert complete == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command":"printf ok"}',
            },
        }
    ]


def test_think_false_avoids_the_broken_false_template_branch() -> None:
    """Chain of thought before the first spoken word is pure added latency.

    `think` is an Ollama field and is dropped for an OpenAI-shaped backend, so
    this Qwen template's explicit false branch returns only newlines on a
    multi-turn prompt. Omission answers normally without a reasoning channel.
    """

    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(messages=[{"role": "user", "content": "Hello."}], think=False)
    )

    payload = adapter_server.build_language_payload(parsed, None, "m", "openai")

    assert "chat_template_kwargs" not in payload
    assert payload["messages"][-1]["content"] == "Hello."
    assert "/no_think" not in json.dumps(payload["messages"])
    assert "reasoning_format" not in payload


def test_native_think_true_still_enables_reasoning_on_the_openai_path() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(messages=[{"role": "user", "content": "Hello."}], think=True)
    )

    payload = adapter_server.build_language_payload(parsed, None, "m", "openai")

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_format" not in payload


def test_ornith_profile_explicitly_disables_hidden_reasoning_for_think_false() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(messages=[{"role": "user", "content": "Hello."}], think=False)
    )
    config = _adapter_config(
        language_api="openai",
        language_disable_thinking=True,
    )

    payload = adapter_server.build_language_payload(
        parsed, None, "ornith", "openai", config
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "/no_think" not in json.dumps(payload["messages"])


def test_explicit_think_true_overrides_the_ornith_no_thinking_profile() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(messages=[{"role": "user", "content": "Hello."}], think=True)
    )
    config = _adapter_config(
        language_api="openai",
        language_disable_thinking=True,
    )

    payload = adapter_server.build_language_payload(
        parsed, None, "ornith", "openai", config
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_comprehension_does_not_set_the_thinking_flag() -> None:
    """Qwen3-Omni's template degenerates on a multimodal prompt when thinking
    is explicitly disabled: measured on an AGX Orin, the identical request
    returns only newlines with the flag and a correct tagged transcript
    without it. Perception emits no reasoning for extraction prompts anyway.
    """

    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
            messages=[
                {
                    "role": "user",
                    "content": "",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
        )
    )

    payload = adapter_server.build_comprehension_payload(parsed, _adapter_config())

    assert "chat_template_kwargs" not in payload
    assert "reasoning_format" not in payload
    assert payload["temperature"] == 0
    assert "repeat_penalty" not in payload


def test_trained_bridge_comprehension_uses_its_no_thinking_prefill() -> None:
    from runtime import adapter_server

    parsed = parse_adapter_request(
        _base_request(
            omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
            messages=[
                {
                    "role": "user",
                    "content": "",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
        )
    )

    payload = adapter_server.build_comprehension_payload(
        parsed,
        _adapter_config(
            comprehension_disable_thinking=True,
            comprehension_repeat_penalty=1.1,
        ),
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["repeat_penalty"] == 1.1
    assert payload["temperature"] == 0
    assert "reasoning_format" not in payload
