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

from harness.audio import MicrophoneStream, SpeakerStream  # noqa: E402
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


def test_completed_background_recall_is_bounded_and_added_to_next_turn() -> None:
    call = CallSession(CallConfig(token="t", model="m", memory_recall=1))

    class Recalled:
        def stamped(self) -> str:
            return "[yesterday] The user's dog is called Biscuit."

    call._recalled = [Recalled()]
    payload = call._build_payload(b"wav", 1, None, with_tools=False)
    system = payload["messages"][0]["content"]

    assert "dog is called Biscuit" in system
    assert "Use it only if it also bears on the current words" in system


def test_observation_prefetches_memory_without_waiting_for_it() -> None:
    call = CallSession(CallConfig(token="t", model="m", memory_recall=3))
    queued: list[tuple[str, int]] = []

    class RecordingMemory:
        def recall_later(self, query: str, *, limit: int) -> None:
            queued.append((query, limit))

    call.memory = RecordingMemory()  # type: ignore[assignment]
    call._events = lambda _payload: iter(  # type: ignore[method-assign]
        [
            {"type": "observation", "transcript": "what is my puppy's name"},
            {"type": "final", "response": {"message": {"content": "Biscuit."}}},
        ]
    )

    result = call._run({"messages": []})

    assert result.reply == "Biscuit."
    assert queued == [("what is my puppy's name", 3)]


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


def test_speaker_lets_pulse_keep_one_continuous_adaptive_stream(monkeypatch) -> None:
    """Tiny forced Pulse buffers underflow between incremental decoder yields."""

    command: list[str] = []

    class Process:
        stdin = None

        def poll(self):
            return None

    def popen(args, **_kwargs):
        command.extend(args)
        return Process()

    monkeypatch.setattr("harness.audio.subprocess.Popen", popen)
    SpeakerStream().start()

    assert not any(argument.startswith("--latency-msec=") for argument in command)
    assert not any(argument.startswith("--process-time-msec=") for argument in command)
    assert "--stream-name=Omni conversational voice" in command


def test_streamed_pcm_blocks_are_written_byte_exactly_to_one_timeline() -> None:
    import io

    class Process:
        stdin = io.BytesIO()

        def poll(self):
            return None

    speaker = SpeakerStream(rate_hz=1_000)
    process = Process()
    speaker._process = process  # type: ignore[assignment]
    first = np.full(10, 1_000, dtype="<i2").tobytes()
    second = np.full(10, 2_000, dtype="<i2").tobytes()

    assert speaker.write(first)
    assert speaker.write(second)
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
