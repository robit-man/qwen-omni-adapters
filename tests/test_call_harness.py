"""The local harness must behave like the browser's live-call mode.

Its VAD is a port of ``portal/static/call_vad.js``, and the port is only worth
having if it agrees with the original: the browser's thresholds were tuned in
real rooms, and a harness that disagreed in the same room would be blamed on
the model rather than on the listener.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.call import (  # noqa: E402
    LIVE_CALL_SYSTEM_PROMPT,
    CallConfig,
    CallSession,
    TurnResult,
)
from harness.vad import Vad, VadConfig  # noqa: E402

FRAME_MS = 20.0
RATE = 16_000
FRAME = int(RATE * FRAME_MS / 1000)


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


# -- the request the browser makes ----------------------------------------


def session() -> CallSession:
    return CallSession(
        CallConfig(portal_url="http://127.0.0.1:8920", token="t", model="m")
    )


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
    assert payload["stream"] is True
    # The live-call instructions, plus the clock appended per turn.
    assert payload["messages"][0]["content"].startswith(LIVE_CALL_SYSTEM_PROMPT)
    # The normal chat request hears and answers in one pass.
    assert "latest spoken turn" in payload["messages"][-1]["content"]
    assert payload["messages"][-1]["audios"][0]["data"] == "d2F2"


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

    assert "the question is about something visible" in content
    assert "do not inventory the scene" in content


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



# -- the cameras are offered only when the words reach for them ------------


def test_a_conversational_turn_never_reaches_for_the_cameras() -> None:
    """The bug: "can you hear me okay?" answered, then narrated the room."""

    from harness.vision_intent import wants_vision

    for spoken in (
        "can you hear me okay hello",
        "what is the capital of france",
        "how are you doing today",
        "tell me a joke",
        "what time is it",
    ):
        assert wants_vision(spoken) is False, spoken


def test_a_question_about_something_visible_does() -> None:
    from harness.vision_intent import wants_vision

    for spoken in (
        "what am I holding",
        "can you see this",
        "look at the screen",
        "read that label for me",
        "what is this thing",
        "how many people are in the room",
    ):
        assert wants_vision(spoken) is True, spoken


def test_a_question_about_time_asks_for_a_clip_not_a_still() -> None:
    from harness.vision_intent import wants_motion

    assert wants_motion("what just happened") is True
    assert wants_motion("did you see that") is True
    assert wants_motion("what am I holding") is False


# -- knowing when it is ----------------------------------------------------


def test_the_model_is_told_the_current_date_and_time() -> None:
    """Without a clock a model answers "what day is it" from its training."""

    from datetime import datetime

    from harness.call import grounding_preamble

    fixed = datetime(2026, 9, 17, 14, 5)
    preamble = grounding_preamble(fixed)

    assert "Thursday 17 September 2026" in preamble
    assert "14:05" in preamble


def test_the_grounding_rides_on_every_turn() -> None:
    payload = session()._build_payload(b"RIFF", segments=1, frame=None)
    system = payload["messages"][0]

    assert system["role"] == "system"
    assert "The current date and time is" in system["content"]
    # The live-call instructions are still there, not replaced by it.
    assert "live two-way spoken conversation" in system["content"]


def test_the_clock_is_read_per_turn_not_once_at_import() -> None:
    """A process listening for a week must not still think it is Monday."""

    import harness.call as call_module

    seen: list[str] = []
    for _ in range(2):
        payload = session()._build_payload(b"RIFF", segments=1, frame=None)
        seen.append(payload["messages"][0]["content"])

    # Same call, freshly rendered each time rather than a module constant.
    assert "The current date and time is" not in call_module.LIVE_CALL_SYSTEM_PROMPT
    assert all("The current date and time is" in content for content in seen)
