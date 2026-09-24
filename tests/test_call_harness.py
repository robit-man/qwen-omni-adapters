"""The local harness must behave like the browser's live-call mode.

Its VAD is a port of ``portal/static/call_vad.js``, and the port is only worth
having if it agrees with the original: the browser's thresholds were tuned in
real rooms, and a harness that disagreed in the same room would be blamed on
the model rather than on the listener.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import __main__ as harness_main  # noqa: E402
from harness.audio import (  # noqa: E402
    MicrophoneStream,
    SpeakerStream,
    probe_audio_server,
)
from harness.call import (  # noqa: E402
    LIVE_CALL_SYSTEM_PROMPT,
    CallConfig,
    CallSession,
    TurnResult,
    _accepted_utterance_preempts,
)
from harness.location import BrowserLocationProvider  # noqa: E402
from harness.vad import Vad, VadConfig  # noqa: E402

FRAME_MS = 20.0
RATE = 16_000
FRAME = int(RATE * FRAME_MS / 1000)


def test_local_voice_location_uses_browser_and_discards_raw_ip() -> None:
    provider = BrowserLocationProvider(
        runner=lambda _url, _timeout: (
            "<html><body><pre>"
            '{"success":true,"ip":"203.0.113.42","city":"Seattle",'
            '"region":"Washington","country":"United States",'
            '"latitude":47.6062,"longitude":-122.3321,'
            '"connection":{"isp":"private"},'
            '"timezone":{"id":"America/Los_Angeles","utc":"-07:00"}}'
            "</pre></body></html>"
        )
    )

    location = provider.refresh_now()

    assert location == {
        "city": "Seattle",
        "region": "Washington",
        "region_code": "",
        "country": "United States",
        "country_code": "",
        "continent": "",
        "continent_code": "",
        "latitude": 47.61,
        "longitude": -122.33,
        "timezone": {
            "id": "America/Los_Angeles",
            "abbreviation": "",
            "utc_offset": "-07:00",
        },
    }
    assert "203.0.113.42" not in json.dumps(location)
    assert "connection" not in location


def test_local_voice_payload_supplies_sanitized_location_to_the_portal() -> None:
    location = {"city": "Seattle", "country": "United States"}
    call = CallSession(
        CallConfig(
            token="t",
            model="m",
            client_location_reader=lambda: location,
        )
    )

    payload = call._build_payload(b"audio", 1, None, with_tools=True)

    assert payload["portal_client_location"] == location
    assert call._build_payload(b"audio", 1, None, with_tools=False).get(
        "portal_client_location"
    ) is None


def test_harness_status_is_private_and_contains_only_bounded_liveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness_main.os, "getpid", lambda: 4242)
    monkeypatch.setattr(harness_main.time, "time", lambda: 1234.5)

    harness_main._write_harness_status(
        tmp_path,
        state="listening",
        indicator_backend="ayatana-appindicator3",
        detail="x" * 500,
    )

    path = tmp_path / harness_main.HARNESS_STATUS_FILE
    value = json.loads(path.read_text(encoding="utf-8"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert value == {
        "detail": "x" * 160,
        "indicator_backend": "ayatana-appindicator3",
        "pid": 4242,
        "schema": "robit.omni-call-harness.status.v1",
        "state": "listening",
        "updated_at": 1234.5,
    }


def test_missing_startup_token_is_waitable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMNI_PORTAL_TOKEN", raising=False)
    monkeypatch.setattr(harness_main, "_repo_root", lambda: tmp_path)

    assert harness_main._available_token(None) == ""

    with pytest.raises(SystemExit, match="no portal token"):
        harness_main._read_token(None)


def test_portal_wait_does_not_send_a_request_until_token_exists(monkeypatch) -> None:
    tokens = iter(["", "new-token"])
    clock = iter([0.0, 0.0, 0.0])
    requests: list[tuple[str, dict[str, str]]] = []

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"model": "test-model"}

    def get(url: str, *, headers: dict[str, str], timeout: float) -> Response:
        assert timeout == 5.0
        requests.append((url, headers))
        return Response()

    monkeypatch.setattr(harness_main.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(harness_main.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(harness_main.httpx, "get", get)

    status, token = harness_main._wait_for_portal(
        "http://127.0.0.1:8920", "", 30.0, lambda: next(tokens)
    )

    assert status == {"model": "test-model"}
    assert token == "new-token"
    assert requests == [
        (
            "http://127.0.0.1:8920/api/status",
            {"Authorization": "Bearer new-token"},
        )
    ]


def test_audio_probe_requires_a_real_capture_source_and_sink(monkeypatch) -> None:
    outputs = {
        "sources": "1\talsa_input.usb-mic\tmodule-alsa-card.c\ts16le 1ch 16000Hz\tIDLE\n",
        "sinks": "2\talsa_output.hdmi\tmodule-alsa-card.c\ts16le 2ch 48000Hz\tIDLE\n",
    }

    def run(command, **_kwargs):
        kind = command[-1]
        return type("Completed", (), {"returncode": 0, "stdout": outputs[kind], "stderr": ""})()

    monkeypatch.setattr(harness_main.subprocess, "run", run)

    assert probe_audio_server() == {"sources": 1, "sinks": 1}


def test_audio_probe_rejects_monitor_only_capture(monkeypatch) -> None:
    def run(command, **_kwargs):
        output = (
            "1\talsa_output.hdmi.monitor\tmodule-alsa-card.c\ts16le 2ch 48000Hz\tIDLE\n"
            if command[-1] == "sources"
            else "2\talsa_output.hdmi\tmodule-alsa-card.c\ts16le 2ch 48000Hz\tIDLE\n"
        )
        return type("Completed", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    monkeypatch.setattr(harness_main.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="no usable sources"):
        probe_audio_server()


def tone(level: float) -> np.ndarray:
    """A frame whose RMS is ``level``."""

    return np.full(FRAME, level, dtype=np.float32)


def feed(vad: Vad, level: float, ms: float, start: float = 0.0):
    """Push ``ms`` of audio at one level; return the results and the clock."""

    results = []
    now = start
    for _ in range(int(ms / FRAME_MS)):
        now += FRAME_MS
        results.append(vad.process(tone(level), now, FRAME_MS))
    return results, now


def calibrated() -> tuple[Vad, float]:
    vad = Vad(VadConfig())
    _, now = feed(vad, 0.001, VadConfig().calibration_ms + FRAME_MS)
    return vad, now


# -- the shape of a spoken turn -------------------------------------------


def test_quiet_room_is_never_mistaken_for_speech() -> None:
    vad, now = calibrated()
    results, _ = feed(vad, 0.001, 2_000, now)
    assert {result.event for result in results} == {"idle"}
    assert not vad.speaking


def test_speech_is_confirmed_before_it_is_believed() -> None:
    """A click is not a turn: the level has to hold for startConfirmMs."""

    vad, now = calibrated()
    results, now = feed(vad, 0.25, VadConfig().start_confirm_ms - FRAME_MS, now)
    assert [result.event for result in results][-1] == "candidate"
    assert not vad.speaking

    more, _ = feed(vad, 0.25, FRAME_MS * 3, now)
    assert any(result.event == "start" for result in more)
    assert vad.speaking


def test_respeaker_native_gate_rejects_far_end_audio_without_losing_preroll() -> None:
    vad, now = calibrated()
    blocked, now = feed(vad, 0.25, 500, now)
    # Re-run those frames through the explicit native gate: loud playback is
    # not local speech even though an amplitude-only detector would accept it.
    vad.reset(now)
    results = []
    for _ in range(25):
        now += FRAME_MS
        results.append(
            vad.process(tone(0.25), now, FRAME_MS, native_speech=False)
        )
    assert {result.event for result in results} == {"idle"}
    assert not vad.speaking

    # Once the DSP says near-end speech is present, normal confirmation starts.
    admitted = []
    for _ in range(12):
        now += FRAME_MS
        admitted.append(
            vad.process(tone(0.25), now, FRAME_MS, native_speech=True)
        )
    assert any(result.event == "start" for result in admitted)
    assert blocked


def test_a_full_utterance_survives_a_pause_in_the_middle() -> None:
    """People pause mid-sentence; that is not the end of the turn."""

    vad, now = calibrated()
    _, now = feed(vad, 0.25, 600, now)
    assert vad.speaking

    # Shorter than silence_ms: still the same turn.
    mid, now = feed(vad, 0.001, VadConfig().silence_ms - 200, now)
    assert all(result.event == "active" for result in mid)
    assert vad.speaking

    _, now = feed(vad, 0.25, 400, now)
    tail, _ = feed(vad, 0.001, VadConfig().silence_ms + 100, now)
    finished = [result for result in tail if result.event == "utterance"]
    assert len(finished) == 1
    assert finished[0].utterance is not None
    assert finished[0].utterance.samples().size > 0


def test_a_brief_noise_is_rejected_rather_than_answered() -> None:
    """Under minActiveMs of speech is a cough, and must not start a turn."""

    vad, now = calibrated()
    _, now = feed(vad, 0.25, VadConfig().start_confirm_ms + FRAME_MS, now)
    results, _ = feed(vad, 0.001, VadConfig().silence_ms + 100, now)
    events = [result.event for result in results]
    assert "rejected" in events
    assert "utterance" not in events


def test_the_first_syllable_is_kept() -> None:
    """Pre-roll exists so the turn does not begin mid-word."""

    vad, now = calibrated()
    # Quiet frames fill the pre-roll, then speech begins.
    _, now = feed(vad, 0.001, FRAME_MS * 5, now)
    _, now = feed(vad, 0.25, 600, now)
    tail, _ = feed(vad, 0.001, VadConfig().silence_ms + 100, now)
    utterance = next(r.utterance for r in tail if r.event == "utterance")
    assert utterance is not None
    # More audio than the speech alone: the quiet run-up came with it.
    assert utterance.samples().size > int(RATE * 0.6)


def test_loud_speech_escapes_calibration_instead_of_being_swallowed() -> None:
    """Someone who speaks immediately should not lose their first sentence."""

    vad = Vad(VadConfig())
    results, _ = feed(vad, 0.4, 300)

    # Loud enough to clear the escape threshold on the first frame, so the
    # calibration window is abandoned rather than waited out.
    assert results[0].event != "calibrating"
    assert any(result.event == "start" for result in results)
    assert vad.speaking

    # A quiet room still calibrates properly.
    patient = Vad(VadConfig())
    quiet, _ = feed(patient, 0.001, 200)
    assert {result.event for result in quiet} == {"calibrating"}


def test_the_noise_floor_rises_with_a_noisy_room() -> None:
    """Otherwise a fan becomes a permanent speaker."""

    vad, now = calibrated()
    quiet_floor = vad.noise_floor
    feed(vad, 0.010, 4_000, now)
    assert vad.noise_floor > quiet_floor


def test_conversation_content_trace_is_injectable_and_end_to_end(monkeypatch) -> None:
    traced: list[tuple[str, str, dict]] = []
    call = CallSession(
        CallConfig(
            token="t",
            model="m",
            content_trace=lambda event, text, details: traced.append(
                (event, text, details)
            ),
        )
    )
    monkeypatch.setattr(
        call,
        "_events",
        lambda _payload: iter(
            [
                {
                    "type": "observation",
                    "transcript": "Please do the thing.",
                    "audio_observation": "Quiet room tone.",
                },
                {
                    "type": "final",
                    "response": {
                        "message": {"content": "I did the thing."}
                    },
                },
            ]
        ),
    )

    result = call._run({"omni": {"task": "chat"}})

    assert result.transcript == "Please do the thing."
    assert ("heard", "Please do the thing.", {}) in traced
    assert ("audio_observation", "Quiet room tone.", {}) in traced
    generated = next(item for item in traced if item[0] == "generated")
    assert generated[1] == "I did the thing."

    monkeypatch.setattr(
        call,
        "_run",
        lambda *_args, **_kwargs: TurnResult(spoke_seconds=1.25),
    )
    call._speak_finished("I did the thing.")
    assert ("tts_input", "I did the thing.", {}) in traced
    playback = next(item for item in traced if item[0] == "playback")
    assert playback[1] == "I did the thing."
    assert playback[2]["spoke_seconds"] == 1.25
    call.close()


# -- the request the browser makes ----------------------------------------


def session() -> CallSession:
    return CallSession(
        CallConfig(portal_url="http://127.0.0.1:8920", token="t", model="m")
    )


def test_voice_scope_is_stable_across_harness_restarts() -> None:
    first = session()
    second = session()
    other = CallSession(CallConfig(token="different", model="m"))

    assert first.portal_session_id == second.portal_session_id
    assert first.portal_session_id != other.portal_session_id
    assert first._client.cookies.get("omni_portal_session") == first.portal_session_id

    first.close()
    second.close()
    other.close()


def test_a_turn_asks_for_speech_and_gets_tools_without_reasoning() -> None:
    """Reasoning is silence the other person has to sit through."""

    payload = session()._build_payload(b"wav", 1, None)

    assert payload["speech_mode"] == "always"
    assert payload["response_modalities"] == ["text", "audio"]
    assert payload["omni"]["task"] == "chat"
    # Nothing was said -> stop after comprehension rather than inventing a turn.
    assert payload["omni"]["require_speech"] is True
    assert payload["think"] is False
    assert payload["portal_auto_tools"] is True
    assert payload["portal_camera_bridge"] is False
    # Host execution is structurally delegated to the checkpointed worker;
    # synchronous foreground shell loops can no longer strand a spoken turn.
    assert payload["portal_shell_bridge"] is False
    assert payload["portal_background_bridge"] is False
    assert payload["stream"] is True
    # The live-call instructions, plus the clock appended per turn.
    assert payload["messages"][0]["content"].startswith(LIVE_CALL_SYSTEM_PROMPT)
    # The normal chat request hears and answers in one pass.
    assert "latest spoken turn" in payload["messages"][-1]["content"]
    assert payload["messages"][-1]["audios"][0]["data"] == "d2F2"


def test_a_live_background_worker_is_exposed_and_its_progress_is_context() -> None:
    class Worker:
        def context_summary(self) -> str:
            return "Persistent background work:\n- abc: running — build the app"

    call = session()
    call.background_agent = Worker()  # type: ignore[assignment]
    payload = call._build_payload(b"wav", 1, None)

    assert payload["portal_background_bridge"] is True
    assert "portal_require_tool_decision" not in payload
    assert "abc: running" in payload["messages"][0]["content"]
    assert "never claim that work advanced" in payload["messages"][0]["content"]


def test_every_spoken_turn_is_one_grounded_auto_tool_pass() -> None:
    """No tool-less classification pass: the answer pass carries the tools.

    A request that needs the web, a camera, a shell, or a durable task is
    handled in the same grounded pass; there is no separate dispatcher text
    that could be mistaken for an action or slip into a refusal.
    """

    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=True,
            camera_enabled=False,
        )
    )
    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return TurnResult(
            transcript="create a tone on my desktop",
            reply="I’ve created the tone on your desktop.",
            tools_used=["background_task"],
        )

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert len(payloads) == 1
    assert payloads[0]["portal_auto_tools"] is True
    assert "response_format" not in payloads[0]
    assert "LIVE_ROUTE" not in payloads[0]["messages"][0]["content"]
    assert result.reply == "I’ve created the tone on your desktop."
    assert result.tools_used == ["background_task"]


def test_failure_is_logged_and_carried_into_the_next_prompt() -> None:
    call = session()
    call._note_failure(
        ValueError("live turn dispatcher returned invalid JSON"),
        raw='{"mode": "reply", "reply": "cut off',
        transcript="what do you remember",
    )
    assert "Previous-turn operation note" in call._pending_failure_note
    assert "invalid JSON" in call._pending_failure_note
    assert "what do you remember" in call._pending_failure_note
    payload = call._build_payload(b"wav", 1, None)
    system = payload["messages"][0]["content"]
    assert "Previous-turn operation note" in system
    assert "invalid JSON" in system
    assert "what do you remember" in system


def test_a_host_action_request_runs_one_grounded_pass_with_background_tools() -> None:
    """A durable-task request is answered in the same tool pass it starts in.

    The portal executes the background_task tool server-side and returns the
    grounded reply; the harness never creates a second request or rearranges
    the turn into a tool-less reply.
    """

    call = CallSession(
        CallConfig(
            token="t",
            model="m",
            tools_enabled=True,
            camera_enabled=False,
            prepare_speech=lambda: None,
            restore_after_speech=lambda: None,
        )
    )
    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        task = payload["omni"]["task"]  # type: ignore[index]
        if task == "synthesize":
            return TurnResult(spoke_seconds=1.0)
        return TurnResult(
            transcript="create a tone on my desktop",
            reply="I’ve started that as a background task.",
            tools_used=["background_task"],
        )

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert len(payloads) == 2  # one grounded answer pass, then the TTS pass
    assert payloads[0]["omni"]["task"] == "chat"  # type: ignore[index]
    assert payloads[0]["portal_auto_tools"] is True
    assert "response_format" not in payloads[0]
    assert result.tools_used == ["background_task"]
    assert result.reply == "I’ve started that as a background task."
    assert not result.reply.startswith("I’ve created")


def test_background_task_tool_event_wakes_the_persistent_agent_in_the_pass() -> None:
    """The durable worker is resumed from the tool event, not a parser branch."""

    wakes: list[bool] = []

    class Worker:
        store = object()

        def context_summary(self) -> str:
            return ""

        def wake(self) -> None:
            wakes.append(True)

    call = CallSession(CallConfig(token="t", model="m"))
    call.background_agent = Worker()  # type: ignore[assignment]
    call._events = lambda _payload: iter(  # type: ignore[method-assign]
        [
            {"type": "observation", "transcript": "create a tone on my desktop"},
            {
                "type": "tool",
                "phase": "complete",
                "name": "background_task",
                "tools": [{"name": "background_task", "arguments": {"action": "start"}}],
            },
            {"type": "final", "response": {"message": {"content": "Started."}}},
        ]
    )

    result = call._run({"messages": []})

    assert wakes == [True]
    assert "background_task" in result.tools_used
    assert result.reply == "Started."


def test_an_embodied_turn_keeps_camera_bridge_for_capture_or_motion() -> None:
    call = CallSession(
        CallConfig(token="t", model="m", camera_enabled=True),
        frame_grabber=lambda **_kwargs: None,  # type: ignore[arg-type]
    )

    payload = call._build_payload(b"wav", 1, None)

    assert payload["portal_camera_bridge"] is True

    payload_with_evidence = call._build_payload(
        b"wav",
        1,
        {"mime_type": "image/jpeg", "encoding": "base64", "data": "eA=="},
    )
    assert payload_with_evidence["portal_camera_bridge"] is True

    payload_with_motion = call._build_payload(
        b"wav",
        1,
        {"mime_type": "video/mp4", "encoding": "base64", "data": "eA=="},
    )
    assert payload_with_motion["portal_camera_bridge"] is False


def test_a_still_is_attached_as_an_image_and_a_clip_as_a_video() -> None:
    still = {"mime_type": "image/jpeg", "encoding": "base64", "data": "x"}
    payload = session()._build_payload(b"wav", 1, still)
    assert payload["messages"][-1]["images"] == [still]
    assert "images" in payload["messages"][-1]

    clip = {"mime_type": "video/mp4", "encoding": "base64", "data": "y"}
    payload = session()._build_payload(b"wav", 1, clip)
    assert payload["messages"][-1]["videos"] == [clip]
    assert "images" not in payload["messages"][-1]


def test_media_is_framed_as_evidence_rather_than_a_scene_to_narrate() -> None:
    """Asked "can you hear me?", a model handed a picture describes the room."""

    frame = {"mime_type": "image/jpeg", "encoding": "base64", "data": "x"}
    content = session()._build_payload(b"wav", 1, frame)["messages"][-1]["content"]

    assert "fresh ambient view" in content
    assert "otherwise ignore it" in content
    assert "Never inventory or mention the scene" in content


def test_nothing_visual_is_sent_when_nothing_visual_was_asked() -> None:
    payload = session()._build_payload(b"wav", 1, None)
    message = payload["messages"][-1]

    assert "images" not in message and "videos" not in message
    assert "camera" not in message["content"].lower()


def test_only_the_dialogue_carries_to_the_next_turn() -> None:
    """History is text. Replaying audio would re-hear an answered question."""

    call = session()
    call._remember(TurnResult(transcript="what is that", reply="a kettle"))
    payload = call._build_payload(b"wav", 1, None)

    history = payload["messages"][1:-1]
    assert history == [
        {"role": "user", "content": "what is that"},
        {"role": "assistant", "content": "a kettle"},
    ]
    assert all("audios" not in message for message in history)
    assert "audios" in payload["messages"][-1]


def test_sound_with_no_speech_is_remembered_as_context() -> None:
    call = session()
    call._remember(TurnResult(audio_observation="a door closed", reply=""))
    payload = call._build_payload(b"wav", 1, None)

    assert payload["messages"][1] == {"role": "user", "content": "a door closed"}


def test_history_is_bounded() -> None:
    """A long call must not grow its own prompt without limit."""

    call = session()
    for index in range(50):
        call._remember(TurnResult(transcript=f"q{index}", reply=f"a{index}"))

    assert len(call._history) <= call.config.history_turns * 2
    assert call._history[-1]["content"] == "a49"


def test_conversation_context_falls_off_with_elapsed_time() -> None:
    call = session()
    for index in range(12):
        role = "user" if index % 2 == 0 else "assistant"
        call._append_history(role, f"m{index}")

    now = call._history_times[-1]
    assert len(call._history_for_prompt(now + 60)) == 12

    after_twenty_minutes = call._history_for_prompt(now + 20 * 60)
    assert len(after_twenty_minutes) == 6
    assert all("20 minutes ago" in item["content"] for item in after_twenty_minutes)

    assert call._history_for_prompt(now + 2 * 86400) == []


def test_interrupted_speech_is_not_recorded_as_fully_heard() -> None:
    call = session()
    call._remember(TurnResult(transcript="tell me more", reply="A long answer."))

    call._mark_interrupted("A long answer.", spoke_seconds=0.8)

    assert "interrupted before it finished" in call._history[-1]["content"]
    assert "may not have heard" in call._history[-1]["content"]


def test_unheard_reply_is_removed_from_context() -> None:
    call = session()
    call._remember(TurnResult(transcript="hello", reply="Hello there."))

    call._mark_interrupted("Hello there.", spoke_seconds=0.0)

    assert call._history == [{"role": "user", "content": "hello"}]
    assert len(call._history_times) == 1


def test_a_turn_has_no_transcription_gate_before_the_answer() -> None:
    """Memory cannot insert a blocking ASR request ahead of conversation."""

    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=False,
            camera_enabled=False,
        )
    )

    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return TurnResult(transcript="hello", reply="hello back")

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert result.reply == "hello back"
    assert len(payloads) == 1
    assert payloads[0]["omni"]["task"] == "chat"  # type: ignore[index]
    assert "audios" in payloads[0]["messages"][-1]  # type: ignore[index]


def test_tools_ride_the_only_answering_pass_like_the_portal() -> None:
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=True,
            camera_enabled=False,
        )
    )
    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return TurnResult(
            transcript="what is the news",
            reply="Here is what I found.",
            tools_used=["web_search"],
        )

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert len(payloads) == 1
    assert payloads[0]["portal_auto_tools"] is True
    assert result.tools_used == ["web_search"]


def test_an_empty_observation_does_not_start_a_second_pass() -> None:
    """Noise must not reach tools, cameras, memory, or another model request."""

    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=True,
            camera_enabled=False,
        )
    )

    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return TurnResult()

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert result == TurnResult()
    assert len(payloads) == 1


def test_live_turn_never_injects_a_previous_turns_prefetched_memory() -> None:
    call = CallSession(CallConfig(token="t", model="m"))
    payload = call._build_payload(b"wav", 1, None, with_tools=False)
    system = payload["messages"][0]["content"]

    assert "prefetched" not in system
    assert "preceding conversation" not in system


def test_recent_near_verbatim_playback_echo_never_becomes_a_turn() -> None:
    call = CallSession(CallConfig(token="t", model="m"))
    call._note_spoken("I'm here, ready to talk. How can I assist you?", 2.0)
    call._events = lambda _payload: iter(  # type: ignore[method-assign]
        [
            {
                "type": "observation",
                "transcript": "I'm here ready to talk how can I assist you",
            },
            {"type": "audio_delta", "audio": {"data": "AAAA"}},
        ]
    )

    result = call._run({"messages": []})

    assert result.echo_suppressed
    assert result.spoke_seconds == 0.0
    assert call._history == []


def test_a_real_followup_is_not_mistaken_for_playback_echo() -> None:
    call = CallSession(CallConfig(token="t", model="m"))
    call._note_spoken("I'm here, ready to talk. How can I assist you?", 2.0)

    assert not call._is_recent_playback_echo("Can you check tomorrow's weather?")


def test_constrained_host_finishes_text_before_swapping_to_speech() -> None:
    """Comprehension and TTS must never be resident at the same time."""

    order: list[str] = []
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=False,
            camera_enabled=False,
            prepare_speech=lambda: order.append("evict"),
            restore_after_speech=lambda: order.append("restore"),
        )
    )

    class RecordingMemory:
        def remember(self, _text: str, *, kind: str) -> None:
            assert kind == "exchange"
            order.append("memory")

    call.memory = RecordingMemory()  # type: ignore[assignment]
    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        task = payload["omni"]["task"]  # type: ignore[index]
        if task == "chat":
            order.append("chat")
            assert payload["response_modalities"] == ["text"]
            assert payload["speech_mode"] == "never"
            return TurnResult(transcript="hello", reply="Yes, I hear you.")
        order.append("tts")
        assert payload["messages"] == [
            {"role": "user", "content": "Yes, I hear you."}
        ]
        return TurnResult(spoke_seconds=1.0, first_audio_ms=25.0)

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert order == ["chat", "evict", "tts", "restore", "memory"]
    assert [payload["omni"]["task"] for payload in payloads] == [  # type: ignore[index]
        "chat",
        "synthesize",
    ]
    assert result.error == ""
    assert result.spoke_seconds == 1.0


def test_comprehension_is_restored_when_synthesis_fails() -> None:
    order: list[str] = []
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=False,
            camera_enabled=False,
            prepare_speech=lambda: order.append("evict"),
            restore_after_speech=lambda: order.append("restore"),
        )
    )

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        task = payload["omni"]["task"]  # type: ignore[index]
        if task == "chat":
            return TurnResult(transcript="hello", reply="Hello.")
        order.append("tts")
        return TurnResult(error="TTS failed")

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert order == ["evict", "tts", "restore"]
    assert result.error == "TTS failed"


def test_failed_foreground_tool_loop_does_not_invent_canned_speech() -> None:
    order: list[str] = []
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=True,
            camera_enabled=False,
            prepare_speech=lambda: order.append("evict"),
            restore_after_speech=lambda: order.append("restore"),
        )
    )

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        task = payload["omni"]["task"]  # type: ignore[index]
        assert task == "chat"
        order.append("chat")
        return TurnResult(
            transcript="create the audio file",
            error="safe tool loop stopped without actionable progress",
        )

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert order == ["chat"]
    assert result.followup == ""
    assert "without actionable progress" in result.error
    assert result.spoke_seconds == 0


def test_live_prompt_routes_mutating_verified_work_to_the_persistent_agent() -> None:
    assert "Use supplied tools to complete requested outcomes" in LIVE_CALL_SYSTEM_PROMPT
    assert "background_task" in LIVE_CALL_SYSTEM_PROMPT
    assert "change approach after failure" in LIVE_CALL_SYSTEM_PROMPT


def test_live_context_requires_tools_and_grounded_alternatives() -> None:
    class Worker:
        def context_summary(self) -> str:
            return ""

    call = CallSession(CallConfig(token="t", model="m"))
    call.background_agent = Worker()  # type: ignore[assignment]

    payload = call._build_payload(b"wav", 1, None, with_tools=True)
    system = payload["messages"][0]["content"]

    assert "<execution_policy>" in system
    assert "Act through the supplied tools" in system
    assert "until the requested outcome is verified" in system
    assert "brief handoff acknowledgment" in system



# -- current vision is normalized; motion upgrades stay model-driven -------


def test_camera_intent_comes_from_the_structured_tool_event() -> None:
    call = session()
    call._events = lambda _payload: iter(  # type: ignore[method-assign]
        [
            {"type": "observation", "transcript": "what happened over there"},
            {
                "type": "tool",
                "phase": "start",
                "tools": [
                    {
                        "name": "request_camera_view",
                        "arguments": {"mode": "motion"},
                    }
                ],
            },
            {
                "type": "final",
                "response": {"message": {"content": "I need a fresh view."}},
            },
        ]
    )

    result = call._run({"messages": []})

    assert result.tools_used == ["request_camera_view"]
    assert result.camera_requested is True
    assert result.camera_motion is True


def test_unrelated_spoken_turn_does_not_capture_or_attach_an_ambient_still() -> None:
    still = {"mime_type": "image/jpeg", "encoding": "base64", "data": "eA=="}
    captured: list[bool] = []
    payloads: list[dict[str, object]] = []

    def grabber(*, motion: bool) -> dict[str, object]:
        captured.append(motion)
        return still

    call = CallSession(
        CallConfig(token="t", model="m", tools_enabled=True, camera_enabled=True),
        frame_grabber=grabber,  # type: ignore[arg-type]
    )

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return TurnResult(
            transcript="look up the latest news",
            reply="Here is the news.",
            tools_used=["web_search"],
        )

    call._run = run  # type: ignore[method-assign]

    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert captured == []
    assert len(payloads) == 1
    assert "images" not in payloads[0]["messages"][-1]  # type: ignore[operator]
    assert payloads[0]["portal_camera_bridge"] is True
    assert result.reply == "Here is the news."
    assert result.camera_requested is False


def test_explicit_camera_tool_requests_the_right_capture_mode() -> None:
    captured: list[bool] = []
    still = {"mime_type": "image/jpeg", "encoding": "base64", "data": "aQ=="}
    clip = {"mime_type": "video/mp4", "encoding": "base64", "data": "dg=="}

    def grabber(*, motion: bool) -> dict[str, object]:
        captured.append(motion)
        return clip if motion else still

    call = CallSession(
        CallConfig(token="t", model="m", tools_enabled=True, camera_enabled=True),
        frame_grabber=grabber,  # type: ignore[arg-type]
    )
    responses = iter(
        [
            TurnResult(
                transcript="what just happened",
                reply="Let me check.",
                tools_used=["request_camera_view"],
                camera_requested=True,
                camera_motion=True,
            ),
            TurnResult(transcript="what just happened", reply="The box fell over."),
        ]
    )
    payloads: list[dict[str, object]] = []

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        payloads.append(payload)
        return next(responses)

    call._run = run  # type: ignore[method-assign]
    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert captured == [True]
    assert result.followup == "The box fell over."
    assert "images" not in payloads[0]["messages"][-1]  # type: ignore[operator]
    assert payloads[1]["messages"][-1]["videos"] == [clip]  # type: ignore[index]


# -- compact ordinary context ---------------------------------------------


def test_runtime_facts_are_fetched_explicitly_instead_of_eagerly_injected() -> None:
    payload = session()._build_payload(b"RIFF", segments=1, frame=None)
    system = payload["messages"][0]

    assert system["role"] == "system"
    assert "natural participant in a live conversation" in system["content"]
    assert "The current date and time is" not in system["content"]
    assert "This machine is in" not in system["content"]


# -- talking over a reply --------------------------------------------------


def test_interrupting_needs_no_model_weights() -> None:
    """Hearing that someone started is the VAD; it loads nothing.

    This is why a reply can be interrupted on a host that has evicted the
    comprehension weights to make room for speech: detecting the start of
    speech is signal processing, and working out what was said happens after
    playback is already cut off.
    """

    vad = Vad(VadConfig())
    _, now = feed(vad, 0.001, VadConfig().calibration_ms + FRAME_MS)
    results, _ = feed(vad, 0.25, VadConfig().start_confirm_ms + FRAME_MS * 2, now)

    assert any(result.event == "start" for result in results)


def test_a_turn_does_not_run_on_the_thread_that_holds_the_microphone() -> None:
    """Otherwise nothing hears the room for the whole reply.

    take_turn used to be called inline in the capture loop, so for the entire
    turn -- including playback -- no frame was read and no interruption could
    be noticed. request_barge existed and nothing could ever call it.
    """

    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    # The turn is handed to a worker, and the loop keeps reading frames.
    assert "threading.Thread(" in source
    assert "session.take_turn(" not in source.split("def worker(")[0]
    assert "for frame in microphone.frames():" in source
    assert "session.request_barge()" in source


def test_a_dead_microphone_pipe_is_an_error_the_supervisor_can_restart() -> None:
    class ClosedPipe:
        def read(self, _size: int) -> bytes:
            return b""

    class DeadRecorder:
        stdout = ClosedPipe()

        def poll(self) -> int:
            return 7

    microphone = MicrophoneStream()
    microphone._process = DeadRecorder()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="microphone capture.*exit code 7"):
        next(microphone.frames())


def test_capture_cleanup_does_not_poison_the_supervisors_stop_event() -> None:
    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    assert "worker_stop.set()" in source
    assert "\n        stop.set()\n" not in source


def test_public_link_comes_only_from_the_current_live_daemon(
    tmp_path: Path, monkeypatch
) -> None:
    import harness.__main__ as harness_main

    state = tmp_path / "runtime-data" / "state" / "daemon-status.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "state": "ready",
                "pid": 101,
                "access_url": (
                    "https://current-link.trycloudflare.com/#access=current_token"
                ),
                "children": [{"name": "cloudflared", "pid": 202}],
            }
        ),
        encoding="utf-8",
    )
    live = {101, 202}
    monkeypatch.setattr(harness_main, "_pid_alive", lambda pid: pid in live)

    assert harness_main._published_access_url(tmp_path).startswith(
        "https://current-link.trycloudflare.com/"
    )
    live.remove(202)
    assert harness_main._published_access_url(tmp_path) == ""


def test_talking_over_a_reply_is_refused_without_echo_cancellation() -> None:
    """A bare microphone hears the speakers and would interrupt every answer."""

    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    assert "array.present" in source
    assert "can_barge" in source


def test_native_respeaker_gate_applies_only_during_playback() -> None:
    """The DSP detector is too conservative to gate ordinary room speech."""

    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    assert "array.present and speaking_since is not None" in source


def test_a_barge_ducks_pauses_resumes_or_commits_without_a_hard_cut() -> None:
    call = CallSession(
        CallConfig(portal_url="http://127.0.0.1:8920", token="t", model="m")
    )
    actions: list[object] = []

    class Speaker:
        def duck(self) -> None:
            actions.append("duck")

        def pause(self) -> None:
            actions.append("pause")

        def resume(self) -> None:
            actions.append("resume")

        def stop(self, *, fade_s: float = 0.0) -> None:
            actions.append(("stop", fade_s))

    call._active_speaker = Speaker()  # type: ignore[assignment]
    call.request_duck()
    call.request_pause()
    call.resume_reply()
    call.request_barge()

    assert call._barge.is_set()
    assert actions == ["duck", "pause", "resume", ("stop", 0.12)]


def test_an_accepted_utterance_during_speech_preparation_cancels_the_announcement() -> None:
    restored: list[bool] = []
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            prepare_speech=lambda: call.request_barge(),
            restore_after_speech=lambda: restored.append(True),
        )
    )
    call._run = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("cancelled announcement must not start TTS")
    )

    result = call.announce("A background task has a long result to report.")

    assert result.interrupted is True
    assert restored == [True]


def test_speaker_lets_pulse_keep_one_continuous_adaptive_stream(monkeypatch) -> None:
    """Two decoder packets pre-roll one continuous Pulse stream."""

    import io

    command: list[str] = []
    processes: list[Process] = []

    class Sink(io.BytesIO):
        def close(self) -> None:
            pass

    class Process:
        stdin = Sink()
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 0

        def terminate(self):
            self.done = True

        def kill(self):
            self.done = True

        def send_signal(self, _signal):
            pass

    def popen(args, **_kwargs):
        command.extend(args)
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr("harness.audio.subprocess.Popen", popen)
    speaker = SpeakerStream(rate_hz=1_000)
    speaker.start()
    first = np.full(10, 1_000, dtype="<i2").tobytes()
    second = np.full(10, 2_000, dtype="<i2").tobytes()

    assert speaker.write(first)
    assert command == []
    assert speaker.write(second)
    speaker.finish()

    assert not any(argument.startswith("--latency-msec=") for argument in command)
    assert not any(argument.startswith("--process-time-msec=") for argument in command)
    assert "--stream-name=Omni conversational voice" in command
    assert len(processes) == 1
    assert processes[0].stdin.getvalue() == first + second
    assert speaker.timing()["startup_buffer_ms"] == 20.0


def test_streamed_pcm_blocks_are_written_byte_exactly_to_one_timeline(monkeypatch) -> None:
    import io

    class Sink(io.BytesIO):
        def close(self) -> None:
            pass

    class Process:
        stdin = Sink()
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 0

        def terminate(self):
            self.done = True

        def kill(self):
            self.done = True

        def send_signal(self, _signal):
            pass

    speaker = SpeakerStream(rate_hz=1_000)
    process = Process()
    monkeypatch.setattr(
        "harness.audio.subprocess.Popen", lambda *_args, **_kwargs: process
    )
    first = np.full(10, 1_000, dtype="<i2").tobytes()
    second = np.full(10, 2_000, dtype="<i2").tobytes()

    speaker.start()
    assert speaker.write(first)
    assert speaker.write(second)
    speaker.finish()
    assert process.stdin.getvalue() == first + second
    assert speaker.played_seconds == 0.02


def test_speech_during_a_turn_is_kept_rather_than_dropped() -> None:
    """Carrying on while it answers must compound, not replace or vanish."""

    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    # Held until the turn finishes, then answered together.
    assert "waiting.add(" in source
    assert "if busy.is_set():" in source
    # The interrupting speech is kept, but submitted audio is never duplicated.
    worker = source.split("def worker(", 1)[1].split("turns =", 1)[0]
    assert "waiting.prepend(" not in worker


@pytest.mark.parametrize(
    ("busy", "reply_started", "can_barge", "announcement", "expected"),
    [
        (False, False, True, False, False),
        (True, False, True, False, False),
        (True, True, True, False, True),
        (True, True, False, False, False),
        (True, False, False, True, True),
    ],
)
def test_only_audible_foreground_replies_or_background_announcements_preempt(
    busy: bool,
    reply_started: bool,
    can_barge: bool,
    announcement: bool,
    expected: bool,
) -> None:
    assert (
        _accepted_utterance_preempts(
            busy=busy,
            reply_started=reply_started,
            can_barge=can_barge,
            background_announcement=announcement,
        )
        is expected
    )


def test_the_capture_loop_uses_a_two_stage_interruption() -> None:
    import inspect

    from harness.call import run_call_loop

    source = inspect.getsource(run_call_loop)

    assert "session.request_duck()" in source
    assert "session.request_pause()" in source
    assert "session.resume_reply()" in source
    assert source.index("session.request_duck()") < source.index(
        "session.request_barge()"
    )


# -- never waiting when it does not have to --------------------------------


def test_the_reload_is_not_waited_for_before_the_turn_can_finish() -> None:
    """Waiting there stopped anything being answered for half a minute.

    The evicted worker takes about thirty seconds to read its weights back.
    Blocking on that before the turn completed spent it while the speaker was
    still listening to the reply, or thinking about what to say next.
    """

    import inspect

    from harness.residency import SpeechResidency

    restore = inspect.getsource(SpeechResidency.restore)

    assert "await_ready" not in restore
    assert "while time.monotonic() < deadline" not in restore
    assert "in the background" in restore
    assert "threading.Thread(" in restore


def test_comprehension_readiness_repairs_a_failed_background_reload() -> None:
    import inspect

    from harness.residency import SpeechResidency

    source = inspect.getsource(SpeechResidency.await_ready)

    assert "not self._is_active()" in source
    assert "self._ensure_started()" in source


def test_background_comprehension_recovery_never_gives_up() -> None:
    import inspect

    from harness.residency import SpeechResidency

    source = inspect.getsource(SpeechResidency._restore_until_started)

    assert "while True" in source
    assert "gave up" not in source
    assert SpeechResidency.ready_timeout_s == 120.0


def test_a_turn_waits_for_comprehension_only_when_it_needs_it() -> None:
    order: list[str] = []
    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            tools_enabled=False,
            camera_enabled=False,
            await_comprehension=lambda: order.append("await"),
            prepare_speech=lambda: order.append("evict"),
            restore_after_speech=lambda: order.append("restore"),
        )
    )

    def run(payload: dict[str, object], **_kwargs: object) -> TurnResult:
        task = payload["omni"]["task"]  # type: ignore[index]
        order.append("chat" if task == "chat" else "tts")
        if task == "chat":
            return TurnResult(transcript="hello", reply="Hello.")
        return TurnResult(spoke_seconds=1.0)

    call._run = run  # type: ignore[method-assign]
    call.take_turn(np.zeros(RATE, dtype=np.float32))

    # Readiness is checked first, then the turn runs and hands speech the room.
    assert order == ["await", "chat", "evict", "tts", "restore"]


def test_a_worker_that_never_comes_back_fails_one_turn_not_the_call() -> None:
    def refuse() -> None:
        raise TimeoutError("did not become ready")

    call = CallSession(
        CallConfig(
            portal_url="http://127.0.0.1:8920",
            token="t",
            model="m",
            await_comprehension=refuse,
        )
    )

    result = call.take_turn(np.zeros(RATE, dtype=np.float32))

    assert "comprehension is not ready" in result.error
