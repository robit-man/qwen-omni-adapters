"""Opt-in recorder guards; these are not physical microphone acceptance."""

import base64
import hashlib
import json

import httpx
import numpy as np
import pytest

from harness.audio import to_wav
from harness.call import CallConfig, CallSession, TurnResult


def test_submitted_audio_trace_matches_real_payload_and_cannot_replace_audio():
    traced = []
    payloads = []
    samples = np.full(640, 0.1, dtype=np.float32)
    expected = to_wav(samples)

    def observe(kind, evidence):
        traced.append((kind, dict(evidence)))
        # A collector changing its own event cannot alter the submitted WAV.
        evidence["wav"] = b"collector-owned replacement"

    call = CallSession(CallConfig(token="unit-token", model="unit-model", evidence_trace=observe))

    def record_payload(payload):
        payloads.append(payload)
        return TurnResult(echo_suppressed=True)

    call._run = record_payload
    try:
        call.take_turn(samples, segments=2)
    finally:
        call.close()
    assert len(traced) == 1 and traced[0][0] == "turn_audio"
    assert traced[0][1]["wav"] == expected
    assert traced[0][1]["segments"] == 2
    assert traced[0][1]["wav_sha256"] == hashlib.sha256(expected).hexdigest()
    attachment = payloads[0]["messages"][-1]["audios"][0]
    assert base64.b64decode(attachment["data"]) == expected


def test_stream_collector_cannot_change_the_events_used_by_the_call():
    traced = []
    original = {"type": "tool", "tools": [{"name": "native_ui", "result": {"status": "verified"}}]}

    def observe(kind, evidence):
        traced.append(kind)
        evidence["event"]["tools"][0]["name"] = "collector-only-change"

    call = CallSession(CallConfig(token="unit-token", model="unit-model", evidence_trace=observe))
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=json.dumps(original) + "\n"))
    try:
        with httpx.Client(transport=transport) as client:
            events = list(call._events({"model": "unit-model"}, client=client))
    finally:
        call.close()
    assert traced == ["stream_event"]
    assert events == [original]


def test_default_config_does_not_opt_in_to_capture_or_tool_recording():
    config = CallConfig()
    assert config.evidence_trace is None
    assert config.content_trace is None


def test_explicit_voice_session_is_isolated_and_default_scope_is_preserved():
    default = CallSession(CallConfig(token="unit-token", model="unit-model"))
    isolated = CallSession(CallConfig(token="unit-token", model="unit-model", portal_session_id="c" * 64))
    try:
        assert default.portal_session_id == hashlib.sha256(b"omni-voice-session\0unit-token").hexdigest()
        assert isolated.portal_session_id == "c" * 64
        assert isolated.portal_session_id != default.portal_session_id
    finally:
        default.close()
        isolated.close()


@pytest.mark.parametrize("scope", ["", "short", "private words with spaces" * 2, "x" * 129, 1])
def test_explicit_voice_session_rejects_unbounded_or_nonopaque_identities(scope):
    with pytest.raises(ValueError, match="opaque"):
        CallSession(CallConfig(token="unit-token", model="unit-model", portal_session_id=scope))


def test_voice_scope_is_copied_forwarded_and_omitted_when_tools_are_off():
    selected = ["native_ui"]
    call = CallSession(CallConfig(token="unit-token", model="unit-model", portal_tool_allowlist=selected))
    selected.append("shell")
    try:
        assert call._build_payload(b"audio", 1, None, with_tools=True)["portal_tool_allowlist"] == ["native_ui"]
        assert "portal_tool_allowlist" not in call._build_payload(b"audio", 1, None, with_tools=False)
    finally:
        call.close()


def test_empty_explicit_voice_scope_is_preserved_instead_of_becoming_unscoped():
    call = CallSession(CallConfig(token="unit-token", model="unit-model", portal_tool_allowlist=[]))
    try:
        assert call._build_payload(b"audio", 1, None, with_tools=True)["portal_tool_allowlist"] == []
    finally:
        call.close()


@pytest.mark.parametrize("scope", ["native_ui", ["native_ui", "native_ui"], ["Native UI"], [1], {}])
def test_voice_tool_scope_rejects_malformed_input(scope):
    with pytest.raises(ValueError, match="tool scope"):
        CallSession(CallConfig(token="unit-token", model="unit-model", portal_tool_allowlist=scope))


def test_pcm_playback_chunks_are_not_duplicated_into_the_evidence_stream():
    observed = []
    call = CallSession(CallConfig(token="unit-token", model="unit-model",
                                 evidence_trace=lambda kind, value: observed.append((kind, value))))
    event = {"type": "audio", "data": "unit-pcm-placeholder"}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=json.dumps(event) + "\n"))
    try:
        with httpx.Client(transport=transport) as client:
            assert list(call._events({"model": "unit-model"}, client=client)) == [event]
    finally:
        call.close()
    assert observed == []
