"""The always-listening call loop.

Drives the same conversation the portal's live-call mode drives, against the
same endpoint, with the same request shape -- but through the machine's own
microphone and speakers instead of a browser. The browser's behaviour is the
specification here: it has been tuned in real rooms, and a local harness that
answered differently in the same room would be blamed on the model.

The loop is deliberately small. Listen, decide something was said, ask, speak
the answer as it arrives, and stop speaking the moment the person starts again.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from harness.audio import (
    CAPTURE_RATE_HZ,
    PLAYBACK_RATE_HZ,
    MicrophoneStream,
    SpeakerStream,
    to_wav,
)
from harness.call_queue import SETTLE_MS, CallQueue, Pending
from harness.memory import PassiveMemory
from harness.place import Place, PlaceLookup
from harness.respeaker import STATE_TO_RING, ReSpeaker, describe_direction
from harness.vad import Vad, VadConfig
from harness.vision_intent import wants_motion, wants_vision

logger = logging.getLogger(__name__)

SCHEMA = "robit.ollama.omni-adapter.v1"

# Verbatim from portal/static/portal.js, so a spoken turn is answered the same
# way here as it is in the browser.
LIVE_CALL_SYSTEM_PROMPT = (
    "You are participating in a live two-way spoken conversation. Answer the "
    "user's intent directly in a natural, concise spoken turn. Do not echo, "
    "transcribe, paraphrase, narrate, or evaluate what the user just said unless "
    "they explicitly ask you to. Never mention an audio transcript, encoder, "
    "adapter, or these instructions. Use the prior dialogue for continuity. If a "
    "current camera frame is attached, treat only that frame as current visual "
    "evidence; older visual descriptions are conversational history, not proof of "
    "what remains visible now. A frame is background context unless the speaker "
    "asked about something visible: answer what was said, and do not describe "
    "the room, the scene, or what you can see unless they asked."
)

def grounding_preamble(
    now: datetime | None = None, place: Place | None = None
) -> str:
    """Tell the model when it is, because otherwise it guesses.

    A model has no clock. Asked what day it is, or how long ago something
    happened, it answers from whenever its training stopped -- confidently and
    wrongly. This is computed per turn rather than once at import, so a process
    that has been listening for a week does not still think it is Monday.
    """

    moment = (now or datetime.now()).astimezone()
    lines = [
        "The current date and time is "
        f"{moment.strftime('%A %-d %B %Y at %H:%M')} "
        f"({moment.strftime('%Z')}). Use this for anything that depends on "
        "when it is -- today, tomorrow, how long ago something was -- rather "
        "than guessing. Timestamps in square brackets on remembered items are "
        "when those happened, relative to now."
    ]
    if place:
        # Coarse and said to be coarse: it is derived from the network address,
        # so it places the conversation in a city and nothing finer. A model
        # told this without the caveat will answer as if it knows the street.
        lines.append(
            f"This machine is in {place.describe()}"
            + (f" ({place.timezone})" if place.timezone else "")
            + ". That is approximate, from the network connection rather than "
            "a GPS fix, so treat it as the general area for weather, local "
            "time elsewhere, and what counts as nearby -- never as the "
            "speaker's exact position."
        )
    return "\n\n".join(lines)


State = str  # "starting" | "listening" | "hearing" | "thinking" | "speaking" | "offline"


@dataclass
class CallConfig:
    portal_url: str = "http://127.0.0.1:8920"
    token: str = ""
    model: str = ""
    input_device: str | None = None
    output_device: str | None = None
    input_channels: int = 1
    input_channel: int = 0
    history_turns: int = 12
    # Tools on and reasoning off by default: this is a spoken conversation, and
    # a hidden chain of thought is a silence the other person has to sit through.
    tools_enabled: bool = True
    reasoning_enabled: bool = False
    # Answer first, then look things up. The portal's tool loop runs before it
    # replies, so a turn that needs a web search stays silent for as long as
    # the search takes. Splitting it means the first answer arrives at
    # conversational speed and anything the tools turn up follows.
    chained_tools: bool = True
    camera_enabled: bool = True
    camera_device: str = "/dev/video0"
    request_timeout_s: float = 300.0
    vad: VadConfig = field(default_factory=VadConfig)
    # How to fetch the portal token again when the one in hand is refused.
    token_reader: Callable[[], str] | None = None
    # Where completed exchanges are journaled outside the conversation path.
    memory_path: str = ""
    # Unified-memory hosts may need to evict comprehension before loading TTS.
    # When configured, chat stays text-only until every reasoning/tool/vision
    # pass is complete, then these callbacks bracket one direct synthesis pass.
    prepare_speech: Callable[[], None] | None = None
    restore_after_speech: Callable[[], None] | None = None
    # Whether the speaker may talk over a reply in progress.
    #
    # Detecting that someone has started speaking is the VAD, which runs on
    # raw microphone frames and loads nothing -- so this works even while the
    # comprehension weights are evicted for speech. What cannot be done
    # without them is working out *what* was said, and that happens after
    # playback has already been cut off and comprehension restored.
    #
    # It needs echo cancellation to be safe: without it the microphone hears
    # the reply coming out of the speakers and the harness interrupts itself
    # on every turn. The ReSpeaker's processed channel is cancelled; a bare
    # microphone is not, so this stays off unless the array is present.
    barge_in_enabled: bool = True
    # Ignore the first moment of playback, so the tail of the speaker's own
    # question cannot count as an interruption of the answer to it.
    barge_in_grace_s: float = 0.6
    # How long a finished utterance waits for the speaker to carry on.
    #
    # People stop to think mid-sentence, and every one of those pauses looks
    # like the end of a turn to a voice detector. Waiting a moment lets the
    # rest of the sentence join the first half instead of arriving as a second
    # question answered separately.
    utterance_settle_s: float = SETTLE_MS / 1000.0


@dataclass
class TurnResult:
    transcript: str = ""
    audio_observation: str = ""
    reply: str = ""
    tools_used: list[str] = field(default_factory=list)
    followup: str = ""
    spoke_seconds: float = 0.0
    interrupted: bool = False
    error: str = ""
    first_audio_ms: float | None = None
    total_ms: float = 0.0


class _PortalError(RuntimeError):
    """An HTTP failure from the portal, with its status kept for retry logic."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"portal returned HTTP {status}: {detail}")
        self.status = status


class CallSession:
    """One continuous conversation: history, transport, and the turn itself."""

    def __init__(
        self,
        config: CallConfig,
        *,
        on_state: Callable[[State, str], None] | None = None,
        frame_grabber: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.config = config
        self._history: list[dict[str, Any]] = []
        self._on_state = on_state or (lambda state, detail: None)
        self._frame_grabber = frame_grabber
        self._client = httpx.Client(timeout=httpx.Timeout(config.request_timeout_s))
        self._barge = threading.Event()
        # Where the voice came from, when a ReSpeaker array can say.
        self.direction: str = ""
        # Roughly where this machine is, when the network will say.
        self.place: Place | None = None
        self.memory: PassiveMemory | None = (
            PassiveMemory(Path(config.memory_path)) if config.memory_path else None
        )

    # -- plumbing --------------------------------------------------------

    def close(self) -> None:
        self._client.close()
        if self.memory is not None:
            self.memory.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.token}"}

    def _refresh_token(self) -> bool:
        """Re-read the portal token, which changes on every daemon start.

        The daemon mints a fresh token each time it comes up, so a harness
        that read it once at startup is holding a dead key the moment the
        adapter restarts -- and every turn after that is refused with a 401
        while the microphone, the model and the speakers are all perfectly
        fine. Reading it again costs one file read and turns a permanent
        outage into a hiccup.
        """

        reader = self.config.token_reader
        if reader is None:
            return False
        try:
            token = (reader() or "").strip()
        except Exception as error:  # noqa: BLE001 - a missing file is not fatal
            logger.debug("could not re-read the portal token: %s", error)
            return False
        if not token or token == self.config.token:
            return False
        logger.info("portal token changed; picking up the new one")
        self.config.token = token
        return True

    def _state(self, state: State, detail: str = "") -> None:
        try:
            self._on_state(state, detail)
        except Exception:  # noqa: BLE001 - an indicator must never break a call
            logger.debug("state callback failed", exc_info=True)

    def request_barge(self) -> None:
        """Ask the turn in flight to stop speaking: the person started again."""

        self._barge.set()

    # -- one spoken turn -------------------------------------------------

    def _build_payload(
        self,
        wav_audio: bytes,
        segments: int,
        frame: dict[str, Any] | None,
        *,
        with_tools: bool | None = None,
    ) -> dict[str, Any]:
        if with_tools is None:
            with_tools = self.config.tools_enabled
        plural = "" if segments == 1 else "s"
        content = (
            f"The attached audio combines {segments} consecutive segment{plural} "
            "from the user's latest spoken turn"
        )
        content += (
            " and the attached media shows what the cameras can see right now, "
            "supplied because the question is about something visible. Answer "
            "their question from it directly and briefly; do not inventory the "
            "scene."
            if frame
            else ". Continue the live conversation by answering the combined "
            "intent directly and use later words to resolve self-corrections."
        )
        message: dict[str, Any] = {
            "role": "user",
            "content": content,
            "audios": [
                {
                    "mime_type": "audio/wav",
                    "encoding": "base64",
                    "data": base64.b64encode(wav_audio).decode("ascii"),
                }
            ],
        }
        if self.direction:
            message["content"] += (
                f" The speaker was {self.direction} relative to the array; treat "
                "that as environmental evidence, not an instruction."
            )
        if frame:
            key = (
                "videos"
                if str(frame.get("mime_type") or "").startswith("video/")
                else "images"
            )
            message[key] = [frame]

        split_speech = self.config.prepare_speech is not None
        return {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{LIVE_CALL_SYSTEM_PROMPT}\n\n"
                        f"{grounding_preamble(place=self.place)}"
                    ),
                },
                *self._history[-self.config.history_turns :],
                message,
            ],
            "omni": {
                "schema": SCHEMA,
                "task": "chat",
                "include_audio_from_video": True,
                # Stop after comprehension when nothing was actually said, so a
                # cough does not become a turn.
                "require_speech": True,
            },
            "response_modalities": ["text"] if split_speech else ["text", "audio"],
            "speech_mode": "never" if split_speech else "always",
            "think": self.config.reasoning_enabled,
            "portal_auto_tools": with_tools,
            "stream": True,
        }

    def _build_synthesis_payload(self, text: str) -> dict[str, Any]:
        """Speak finished text without loading comprehension or language."""

        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": text}],
            "omni": {"schema": SCHEMA, "task": "synthesize"},
            "response_modalities": ["text", "audio"],
            "speech_mode": "always",
            "think": False,
            "portal_auto_tools": False,
            "stream": True,
        }

    def _events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        url = f"{self.config.portal_url.rstrip('/')}/api/chat/stream"
        with self._client.stream(
            "POST", url, json=payload, headers=self._headers()
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise _PortalError(response.status_code, response.text[:300])
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event

    def take_turn(self, samples: np.ndarray, segments: int = 1) -> TurnResult:
        """Answer what was heard, and follow up if tools find more.

        With tools enabled the portal runs its whole tool loop before saying
        anything, so a question that needs a web search is met with silence
        for as long as the search takes. Chaining answers from what was heard
        first -- at conversational speed -- and speaks again only if the tools
        actually turned something up. A follow-up that used no tools has
        nothing to add, so it is not spoken.
        """

        audio = to_wav(samples)
        chained = self.config.tools_enabled and self.config.chained_tools

        # No imagery on the first pass. A camera frame attached to every turn
        # makes the picture the subject: asked "can you hear me okay?", the
        # model answers and then starts describing the room. The cameras are
        # offered only once the words have reached for them.
        result = self._run(
            self._build_payload(audio, segments, None, with_tools=not chained)
        )
        self._remember(result)
        if result.error or result.interrupted:
            return result
        if not result.transcript and not result.audio_observation:
            return result

        looking = (
            self.config.camera_enabled
            and self._frame_grabber is not None
            and wants_vision(result.transcript)
        )
        if chained or looking:
            frame = None
            if looking:
                self._state("thinking", "looking")
                frame = self._frame_grabber(motion=wants_motion(result.transcript))

            follow = self._run(
                self._build_payload(
                    audio,
                    segments,
                    frame,
                    with_tools=True,
                ),
                speak_only_if_useful=not looking,
            )
            useful = bool(follow.tools_used or looking)
            if useful and follow.reply.strip() and not follow.error:
                result.followup = follow.reply.strip()
                result.tools_used = follow.tools_used
                result.spoke_seconds += follow.spoke_seconds
                self._history.append(
                    {"role": "assistant", "content": follow.reply.strip()}
                )
            if follow.error or follow.interrupted:
                return result

        if self.config.prepare_speech is not None:
            speech = self._speak_finished(result.followup or result.reply)
            result.spoke_seconds += speech.spoke_seconds
            result.first_audio_ms = speech.first_audio_ms
            result.total_ms += speech.total_ms
            result.interrupted = speech.interrupted
            result.error = speech.error
            if speech.error or speech.interrupted:
                return result

        # Scheduling the daemon worker is the final operation. It can never
        # overlap comprehension, tool use, TTS, or comprehension restoration.
        self._persist(result)
        return result

    def _speak_finished(self, text: str) -> TurnResult:
        """Evict heavyweight listeners, synthesize once, then restore them."""

        if not text.strip():
            return TurnResult()
        prepare = self.config.prepare_speech
        restore = self.config.restore_after_speech
        if prepare is None:
            return TurnResult(error="speech residency callback is not configured")

        self._state("thinking", "making room for speech")
        try:
            prepare()
        except Exception as error:  # noqa: BLE001 - report one failed turn
            return TurnResult(error=f"could not make room for speech: {error}")

        speech = TurnResult()
        try:
            speech = self._run(self._build_synthesis_payload(text))
        finally:
            if restore is not None:
                self._state("thinking", "restoring comprehension")
                try:
                    restore()
                except Exception as error:  # noqa: BLE001 - keep the listener alive
                    detail = f"could not restore comprehension: {error}"
                    speech.error = f"{speech.error}; {detail}" if speech.error else detail
        return speech

    def _run(
        self,
        payload: dict[str, Any],
        *,
        speak_only_if_useful: bool = False,
    ) -> TurnResult:
        """One request: stream it, and speak the audio as it arrives."""

        self._barge.clear()
        result = TurnResult()
        started = time.monotonic()

        speaker = SpeakerStream(PLAYBACK_RATE_HZ, self.config.output_device)
        speaking = False
        self._state("thinking", "")

        def events() -> Iterator[dict[str, Any]]:
            """Stream the turn, taking a fresh token if this one is refused."""

            try:
                yield from self._events(payload)
            except _PortalError as error:
                if error.status != 401 or not self._refresh_token():
                    raise
                yield from self._events(payload)

        try:
            for event in events():
                if self._barge.is_set():
                    result.interrupted = True
                    break
                kind = event.get("type")
                if kind == "observation":
                    result.transcript = str(event.get("transcript") or "").strip()
                    result.audio_observation = str(
                        event.get("audio_observation") or ""
                    ).strip()
                    self._state("thinking", result.transcript)
                elif kind == "tool":
                    name = str(event.get("name") or event.get("tool") or "").strip()
                    if name:
                        result.tools_used.append(name)
                        self._state("thinking", f"using {name}")
                elif kind == "delta":
                    message = event.get("message")
                    if isinstance(message, dict):
                        result.reply += str(message.get("content") or "")
                elif kind == "audio_delta":
                    audio = event.get("audio")
                    if not isinstance(audio, dict):
                        continue
                    chunk = base64.b64decode(str(audio.get("data") or ""))
                    if not chunk:
                        continue
                    if speak_only_if_useful and not result.tools_used:
                        # Nothing was looked up, so this pass has nothing the
                        # first answer did not already say. Collect the text
                        # for the record and stay quiet.
                        continue
                    if not speaking:
                        speaker.start()
                        speaking = True
                        result.first_audio_ms = (time.monotonic() - started) * 1000
                        self._state("speaking", result.reply[:60])
                    if not speaker.write(chunk):
                        break
                elif kind == "final":
                    response = event.get("response")
                    if isinstance(response, dict):
                        message = response.get("message")
                        if isinstance(message, dict) and message.get("content"):
                            result.reply = str(message["content"])
                elif kind == "error":
                    result.error = str(event.get("error") or "stream error")
                    break
        except Exception as error:  # noqa: BLE001 - one bad turn is not fatal
            result.error = f"{type(error).__name__}: {error}"
            logger.warning("call turn failed: %s", result.error)
        finally:
            if speaking:
                if self._barge.is_set() or result.error:
                    speaker.stop()
                    result.interrupted = self._barge.is_set()
                else:
                    speaker.finish()
                result.spoke_seconds = speaker.played_seconds
            else:
                speaker.stop()

        result.total_ms = (time.monotonic() - started) * 1000
        return result

    def _remember(self, result: TurnResult) -> None:
        """Keep the dialogue, not the audio: only text carries to the next turn.

        Persistent journaling happens only after all answer/tool/TTS work has
        finished, separately from this prompt history update.
        """

        spoken = result.transcript or result.audio_observation
        reply = result.reply.strip()
        if spoken:
            self._history.append({"role": "user", "content": spoken})
        if reply:
            self._history.append({"role": "assistant", "content": reply})
        limit = self.config.history_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def _persist(self, result: TurnResult) -> None:
        """Queue semantic storage after the whole turn, with no synchronous I/O."""

        if self.memory is None:
            return
        transcript = result.transcript
        reply = " ".join(
            part for part in (result.reply.strip(), result.followup.strip()) if part
        )
        if transcript and reply:
            self.memory.remember(f"{transcript} — {reply}", kind="exchange")
        elif transcript:
            self.memory.remember(transcript, kind="heard")
        elif result.audio_observation:
            self.memory.remember(result.audio_observation, kind="sound")


def run_call_loop(
    config: CallConfig,
    *,
    on_state: Callable[[State, str], None] | None = None,
    on_turn: Callable[[TurnResult], None] | None = None,
    frame_grabber: Callable[[], dict[str, Any] | None] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Listen until told to stop, answering when the speaker is actually done.

    Two things this gets right that answering per detected utterance does not.

    A finished utterance is not a turn yet: it waits briefly, and anything said
    in that moment joins it. People stop to think mid-sentence, and every one
    of those pauses looks like the end of a turn to a voice detector, so
    without this the two halves arrive as two questions and get two answers.

    And the microphone is never ignored. A turn runs on its own thread so the
    capture loop keeps reading frames throughout -- including while the reply
    is being spoken, which is the only moment an interruption can happen.
    Interrupting costs no weights: hearing that someone has started is signal
    processing. When it happens, the question that was being answered goes
    back in the queue, because it never actually got answered, and joins
    whatever the speaker went on to say.
    """

    stop = stop or threading.Event()
    session = CallSession(config, on_state=on_state, frame_grabber=frame_grabber)
    vad = Vad(config.vad)
    outer_notify = on_state or (lambda state, detail: None)

    array = ReSpeaker()
    array.start()

    # Looked up in the background so the first spoken turn does not pay for it,
    # and left unknown when there is no network rather than delaying anything.
    places = PlaceLookup()
    places.refresh_async()

    # Echo cancellation is what makes talking over a reply safe. Without it the
    # microphone hears the speakers and the harness interrupts itself.
    can_barge = config.barge_in_enabled and array.present
    if config.barge_in_enabled and not array.present:
        logger.info(
            "talking over replies is disabled: no echo-cancelling array, so the "
            "microphone would hear the speakers and interrupt every answer"
        )

    speaking_since: float | None = None
    busy = threading.Event()
    waiting = CallQueue(CAPTURE_RATE_HZ)
    lock = threading.Lock()
    work: queue.Queue[Pending] = queue.Queue(maxsize=1)

    def notify(state: State, detail: str = "") -> None:
        nonlocal speaking_since
        speaking_since = time.monotonic() if state == "speaking" else None
        array.set_state(STATE_TO_RING.get(state, "trace"))
        outer_notify(state, detail)

    session._on_state = notify

    def worker() -> None:
        """Take turns one at a time, off the thread that holds the microphone."""

        while not stop.is_set():
            try:
                pending = work.get(timeout=0.2)
            except queue.Empty:
                continue
            busy.set()
            try:
                session.direction = describe_direction(array.direction)
                session.place = places.place or None
                result = session.take_turn(
                    pending.audio(), segments=max(1, pending.segments)
                )
                if result.interrupted:
                    # It was cut off, so the question stands. Put it back in
                    # front of whatever the speaker said over the top of it.
                    with lock:
                        waiting.prepend(pending.audio(), pending.active_ms)
                    logger.info("re-queued the interrupted question")
                elif on_turn:
                    on_turn(result)
            except Exception as error:  # noqa: BLE001 - one turn is not the call
                logger.warning("turn failed: %s", error)
            finally:
                busy.clear()
                notify("listening", "")

    turns = threading.Thread(target=worker, name="omni-call-turn", daemon=True)
    turns.start()

    microphone = MicrophoneStream(
        device=config.input_device,
        rate_hz=CAPTURE_RATE_HZ,
        channels=config.input_channels,
        channel=config.input_channel,
    )
    frame_ms = microphone.frame_ms
    now_ms = 0.0
    settle_until: float | None = None
    try:
        with microphone:
            notify("listening", "")
            for frame in microphone.frames():
                if stop.is_set():
                    return
                now_ms += frame_ms
                verdict = vad.process(frame, now_ms, frame_ms)
                now = time.monotonic()

                if verdict.event in {"candidate", "start", "active"}:
                    # Still talking, so nothing is finished being said.
                    settle_until = None
                    if verdict.event == "start":
                        if not busy.is_set():
                            notify("hearing", "")
                        elif (
                            can_barge
                            and speaking_since is not None
                            and now - speaking_since >= config.barge_in_grace_s
                        ):
                            logger.info("interrupted while speaking")
                            session.request_barge()
                elif verdict.event == "utterance" and verdict.utterance is not None:
                    with lock:
                        waiting.add(
                            verdict.utterance.samples(),
                            verdict.utterance.active_duration_ms,
                        )
                        segments = waiting.segments
                    settle_until = now + config.utterance_settle_s
                    if segments > 1:
                        logger.info("carried on speaking; %d segments so far", segments)
                    # The speaker's own voice has been in the microphone and
                    # the room has changed, so the floor is learned again.
                    vad.reset(now_ms, calibrate=True)

                if settle_until is None or now < settle_until:
                    continue
                if busy.is_set():
                    # Hold it: answering the first half while the second is
                    # still being spoken is what produced two replies.
                    continue
                with lock:
                    if not waiting:
                        settle_until = None
                        continue
                    pending = waiting.take()
                settle_until = None
                try:
                    work.put_nowait(pending)
                except queue.Full:
                    with lock:
                        waiting.prepend(pending.audio(), pending.active_ms)
    finally:
        stop.set()
        turns.join(timeout=5)
        array.stop()
        session.close()
