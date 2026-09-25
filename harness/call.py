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
import difflib
import hashlib
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
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
from harness.background_agent import BackgroundAgent
from harness.call_queue import SETTLE_MS, CallQueue, Pending
from harness.memory import PassiveMemory, memory_capacity_available
from harness.respeaker import STATE_TO_RING, ReSpeaker, describe_direction
from harness.vad import Vad, VadConfig
from portal.background_tasks import BackgroundTaskStore
from qwen_omni_adapters.context import context_text
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPressure

logger = logging.getLogger(__name__)

SCHEMA = "robit.ollama.omni-adapter.v1"

LIVE_CALL_SYSTEM_PROMPT = context_text("prompts", "live_call_system")


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
    # Maximum prompt-history messages. The usable window shrinks when the room
    # has been quiet for a while, so yesterday's topic cannot hijack today.
    history_turns: int = 12
    # Tools on and reasoning off by default: this is a spoken conversation, and
    # a hidden chain of thought is a silence the other person has to sit through.
    tools_enabled: bool = True
    reasoning_enabled: bool = False
    # The portal owns its tool loop: schemas ride on the spoken request and it
    # keeps executing safe calls until the model returns a final answer.
    camera_enabled: bool = True
    camera_device: str = "/dev/video0"
    request_timeout_s: float = 300.0
    vad: VadConfig = field(default_factory=VadConfig)
    # How to fetch the portal token again when the one in hand is refused.
    token_reader: Callable[[], str] | None = None
    # Approximate location comes from an isolated client-side Chromium lookup.
    # The callback returns only sanitized geography; raw IP/provider fields
    # never enter the request, logs, history, or model context.
    client_location_reader: Callable[[], Mapping[str, Any] | None] | None = None
    # Where completed exchanges are journaled outside the conversation path.
    memory_path: str = ""
    memory_calibration_path: str = ""
    # Shared crash-safe handoff between the portal and the stepwise background
    # agent. Empty disables background work without affecting normal tools.
    background_task_path: str = ""
    # Intermediate task state is already visible in the indicator. Constrained
    # action-residency profiles keep these updates silent so an optional status
    # sentence does not reload TTS into the task's reclaimed working set.
    # Terminal completion remains announced independently.
    background_progress_speech: bool = True
    # Optional constrained-host transition run before a durable action slice.
    # It may shed independently resident TTS, but must not remove the shared
    # comprehension/ASR trunk or pointing worker.
    prepare_background_action: Callable[[], None] | None = None
    # Unified-memory hosts may need to evict comprehension before loading TTS.
    # When configured, chat stays text-only until every reasoning/tool/vision
    # pass is complete, then these callbacks bracket one direct synthesis pass.
    prepare_speech: Callable[[], None] | None = None
    restore_after_speech: Callable[[], None] | None = None
    # Block until the evicted worker can hear again. Called at the point a turn
    # actually needs comprehension rather than when the last reply finished, so
    # its reload overlaps the reply still playing and the pause before anyone
    # speaks again.
    await_comprehension: Callable[[], None] | None = None
    comprehension_ready: Callable[[], bool] | None = None
    # Exact conversational text is sensitive and therefore opt-in. Tests and
    # explicitly configured deployments can inject a structured sink to trace
    # recognition -> generation -> TTS -> playback end to end.
    content_trace: Callable[[str, str, dict[str, Any]], None] | None = None
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
    # First duck under a possible interruption. Only sustained speech earns a
    # full fade and pause; a rejected false start rises back into the sentence.
    barge_in_pause_s: float = 0.45
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
    camera_requested: bool = False
    camera_motion: bool = False
    followup: str = ""
    spoke_seconds: float = 0.0
    interrupted: bool = False
    echo_suppressed: bool = False
    error: str = ""
    first_audio_ms: float | None = None
    total_ms: float = 0.0


@dataclass
class BargeInState:
    """Reconsider near-end speech after playback's echo-safe grace window."""

    started_at: float | None = None
    ducked_at: float | None = None
    paused: bool = False

    @property
    def active(self) -> bool:
        return self.started_at is not None

    def reset(self) -> None:
        self.started_at = None
        self.ducked_at = None
        self.paused = False

    def observe(
        self,
        event: str,
        *,
        now: float,
        reply_started_at: float,
        grace_s: float,
        pause_s: float,
    ) -> tuple[str, ...]:
        """Return the duck/pause/resume transitions for one VAD event."""

        if event == "start" or (event == "active" and self.started_at is None):
            self.started_at = now
            self.ducked_at = None
            self.paused = False

        if event in {"start", "active"} and self.started_at is not None:
            actions: list[str] = []
            if self.ducked_at is None and now - reply_started_at >= grace_s:
                self.ducked_at = now
                actions.append("duck")
            if (
                self.ducked_at is not None
                and not self.paused
                and now - self.started_at >= pause_s
            ):
                self.paused = True
                actions.append("pause")
            return tuple(actions)

        if event == "rejected":
            actions = ("resume",) if self.ducked_at is not None else ()
            self.reset()
            return actions
        return ()


class _PortalError(RuntimeError):
    """An HTTP failure from the portal, with its status kept for retry logic."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"portal returned HTTP {status}: {detail}")
        self.status = status


def _accepted_utterance_preempts(
    *,
    busy: bool,
    reply_started: bool,
    can_barge: bool,
    background_announcement: bool,
) -> bool:
    """Return whether accepted near-end audio should cancel current work.

    Before a foreground reply has produced audio, another accepted sound is
    ambiguous: it may be the person continuing, or a non-speech event accepted
    for later comprehension. Preserve it for the next turn instead of silently
    discarding a reply that never became audible. An actual streaming reply may
    be barged when echo cancellation makes that safe. Background announcements
    remain subordinate to a person and may be preempted before playback.
    """

    return busy and (
        background_announcement or (reply_started and can_barge)
    )


def _foreground_lane_still_owned(
    *,
    accepted_audio_waiting: bool,
    queued_turn: bool,
) -> bool:
    """Keep the model lane only for accepted speech, never a raw VAD candidate."""

    return accepted_audio_waiting or queued_turn


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
        self._history_times: list[float] = []
        self._on_state = on_state or (lambda state, detail: None)
        self._frame_grabber = frame_grabber
        # Foreground and durable work are one embodied conversation and must
        # own the same single visible browser. Deriving the opaque cookie from
        # the daemon token keeps it stable across harness-only restarts without
        # persisting or exposing that token. A daemon restart clears the portal
        # browser store and naturally yields a new scope.
        self.portal_session_id = hashlib.sha256(
            b"omni-voice-session\0" + config.token.encode("utf-8")
        ).hexdigest()
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.request_timeout_s),
            cookies={"omni_portal_session": self.portal_session_id},
        )
        self._barge = threading.Event()
        self._speaker_lock = threading.Lock()
        self._active_speaker: SpeakerStream | None = None
        self._last_spoken_text = ""
        self._last_spoken_at = 0.0
        # Where the voice came from, when a ReSpeaker array can say.
        self.direction: str = ""
        self.memory: PassiveMemory | None = (
            PassiveMemory(Path(config.memory_path)) if config.memory_path else None
        )
        self.background_agent: BackgroundAgent | None = None
        self._pending_failure_note = ""

    # -- plumbing --------------------------------------------------------

    def close(self) -> None:
        if self.background_agent is not None:
            self.background_agent.close()
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

    def _trace_content(self, event: str, text: str, **details: Any) -> None:
        sink = self.config.content_trace
        if sink is None:
            return
        try:
            sink(event, text, dict(details))
        except Exception:  # noqa: BLE001 - diagnostics must never break a call
            logger.debug("conversation content trace failed", exc_info=True)

    def _speaker_action(self, action: str) -> None:
        with self._speaker_lock:
            speaker = self._active_speaker
        if speaker is None:
            return
        getattr(speaker, action)()

    def request_duck(self) -> None:
        """Lower a possible interruption while deciding whether it is speech."""

        self._speaker_action("duck")

    def request_pause(self) -> None:
        """Pause at silence after the possible interruption persists."""

        self._speaker_action("pause")

    def resume_reply(self) -> None:
        """A noise candidate was rejected; continue the unfinished sentence."""

        self._speaker_action("resume")

    def request_barge(self) -> None:
        """Commit an interruption after actual speech has been accepted."""

        self._barge.set()
        with self._speaker_lock:
            speaker = self._active_speaker
        if speaker is not None:
            speaker.stop(fade_s=0.12)

    @staticmethod
    def _echo_words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.casefold())

    def _is_recent_playback_echo(self, transcript: str) -> bool:
        """Recognize a near-verbatim copy of the reply that just left speakers.

        Hardware AEC and native DSP VAD are the first line of defense. This is
        deliberately narrow: it catches only a substantial near-copy within a
        short acoustic tail, not merely a semantically similar user response.
        """

        if time.monotonic() - self._last_spoken_at > 90.0:
            return False
        heard = self._echo_words(transcript)
        spoken = self._echo_words(self._last_spoken_text)
        if len(heard) < 3 or not spoken:
            return False
        if heard == spoken:
            return True
        shorter, longer = (heard, spoken) if len(heard) <= len(spoken) else (spoken, heard)
        if len(shorter) >= 6 and len(shorter) / len(longer) >= 0.55:
            width = len(shorter)
            if any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1)):
                return True
        return (
            min(len(heard), len(spoken)) >= 6
            and difflib.SequenceMatcher(a=heard, b=spoken, autojunk=False).ratio()
            >= 0.88
        )

    def _note_spoken(self, text: str, seconds: float) -> None:
        if text.strip() and seconds > 0.0:
            self._last_spoken_text = text.strip()
            self._last_spoken_at = time.monotonic()

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
            " and the attached image is a fresh ambient view from the embodied "
            "client's cameras. Use it only when it materially helps answer the "
            "speaker's request; otherwise ignore it. Never inventory or mention "
            "the scene merely because the view is present. This is real current "
            "visual evidence from the camera capture you requested: answer with "
            "the same scope as the question. Give a casual one-sentence overview "
            "for a broad casual question; when asked about one item or feature, "
            "focus on it at the requested detail and do not inventory unrelated "
            "objects or make a list unless asked. Never narrate the frame or "
            "disclaim camera access, sight, or permission to describe it."
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
        system_content = LIVE_CALL_SYSTEM_PROMPT
        if self.background_agent is not None:
            background = self.background_agent.context_summary()
            if background:
                system_content += f"\n\n{background}"
            if with_tools:
                system_content += "\n\n" + context_text(
                    "directives", "foreground_execution"
                )
        if self._pending_failure_note:
            system_content += f"\n\n{self._pending_failure_note}"
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                *self._history_for_prompt(),
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
            # Camera access is a model-selected capability, not eager context.
            # Keeping only the tiny bridge on the audio pass prevents an
            # unrelated ambient frame from steering an ordinary spoken answer.
            "portal_camera_bridge": bool(
                with_tools
                and self.config.camera_enabled
                and self._frame_grabber is not None
                and frame is None
            ),
            # Host execution belongs to the checkpointed worker. Giving the
            # foreground both shell and background_task let a small model pick
            # shell, enter a synchronous retry loop, and strand the spoken
            # turn. The worker still receives unrestricted raw Bash and every
            # result; only the latency-critical selection is made structural.
            "portal_shell_bridge": False,
            # One compact handoff contract lets a spoken turn return promptly
            # while a checkpointed worker performs sustained tool chains.
            "portal_background_bridge": bool(
                with_tools and self.background_agent is not None
            ),
            "stream": True,
        }
        if with_tools and self.config.client_location_reader is not None:
            try:
                client_location = self.config.client_location_reader()
            except Exception:  # noqa: BLE001 - location never gates conversation
                client_location = None
            if client_location:
                payload["portal_client_location"] = dict(client_location)
        return payload

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
        """Answer what was heard, with tools in the answer-producing pass.

        This is the same chain as the cloudflared browser: the portal exposes
        its safe schemas, executes every requested call, and returns the final
        grounded answer. A camera-enabled client exposes a structured capture
        bridge, but the first pass is audio-only. The model must determine that
        the request materially needs current visual evidence before it can ask
        for a still or motion clip; transcript keywords never decide that
        locally.
        """

        self._barge.clear()
        # The comprehension weights are needed from the first request. If the
        # last reply evicted them, their reload has been running since that
        # reply began playing, so this usually returns at once.
        if self.config.await_comprehension is not None:
            self._state("thinking", "waiting to hear")
            try:
                self.config.await_comprehension()
            except Exception as error:  # noqa: BLE001 - one turn is not the call
                return TurnResult(error=f"comprehension is not ready: {error}")

        audio = to_wav(samples)

        # Every spoken turn is one grounded answer pass with the portal's real
        # tools auto-executed: the model either answers conversationally or
        # calls the smallest tool that accomplishes the request, and the answer
        # it speaks comes from that completed work. There is no tool-less
        # classification pass and no separate text for it to confuse with an
        # action.
        payload = self._build_payload(
            audio,
            segments,
            None,
            with_tools=self.config.tools_enabled,
        )
        result = self._run(payload)
        if result.echo_suppressed:
            return result
        if result.interrupted:
            return result

        if result.error:
            self._remember(result)
            self._note_failure(
                result.error,
                transcript=result.transcript or result.audio_observation,
            )
            self._mark_interrupted(result.reply, 0.0)
            return result
        if not result.transcript and not result.audio_observation:
            return result

        looking = (
            self.config.camera_enabled
            and self._frame_grabber is not None
            and result.camera_requested
        )
        capture_failed = False
        if looking:
            # A second pass, this time with the evidence the model requested.
            # In split-speech mode neither provisional text nor a tool request
            # is spoken; only this grounded answer reaches TTS.
            self._state("thinking", "looking")
            frame = self._frame_grabber(motion=result.camera_motion)
            if frame is None:
                # This is an evidence-backed capture failure, not a broad
                # capability denial. It is safe to speak only after the
                # structured request actually tried the camera path.
                capture_failed = True
                result.reply = "The camera capture failed just now."
                result.followup = ""
                result.spoke_seconds = 0.0
                result.first_audio_ms = None
            else:
                follow = self._run(
                    self._build_payload(
                        audio,
                        segments,
                        frame,
                        with_tools=self.config.tools_enabled,
                    ),
                )
                result.tools_used = list(
                    dict.fromkeys([*result.tools_used, *follow.tools_used])
                )
                result.reply = follow.reply.strip()
                result.followup = ""
                result.spoke_seconds = follow.spoke_seconds
                result.first_audio_ms = follow.first_audio_ms
                result.total_ms += follow.total_ms
                result.error = follow.error
                if follow.interrupted:
                    result.interrupted = True
                    self._remember(result)
                    self._mark_interrupted(result.reply, result.spoke_seconds)
                    return result
                if follow.error:
                    self._remember(result)
                    return result

        if self.config.prepare_speech is not None or capture_failed:
            speech = self._speak_finished(result.reply)
            result.spoke_seconds += speech.spoke_seconds
            result.first_audio_ms = speech.first_audio_ms
            result.total_ms += speech.total_ms
            result.interrupted = speech.interrupted
            result.error = speech.error
            if speech.interrupted:
                self._remember(result)
                self._mark_interrupted(result.reply, result.spoke_seconds)
                return result
            if speech.error:
                self._remember(result)
                return result

        self._note_spoken(result.reply, result.spoke_seconds)
        self._remember(result)

        # Scheduling the daemon worker is the final operation. It can never
        # overlap comprehension, tool use, TTS, or comprehension restoration.
        self._persist(result)
        self._pending_failure_note = ""
        return result

    def _note_failure(
        self,
        error: BaseException | str,
        *,
        raw: str = "",
        transcript: str = "",
        final_error: str = "",
    ) -> None:
        """Log the root cause and carry it into the next prompt generation.

        The system cannot retract a spoken claim, but it can tell the model
        exactly what failed so the next generation can retry knowingly. The
        note is injected into the next turn's system context and cleared once
        that turn has read it.
        """

        detail = str(error)
        if final_error and final_error != detail:
            detail = f"{detail}; fallback also failed: {final_error}"
        if raw:
            logger.warning(
                "live turn error recorded: %s; raw dispatch: %r", detail, raw[:400]
            )
        else:
            logger.warning("live turn error recorded: %s", detail)
        transcript = (transcript or "").strip()
        note = (
            "Previous-turn operation note (the user did not hear or see this): an "
            "earlier attempt to handle the last spoken request failed before a "
            "verified reply was produced. Error:\n"
            f"{detail}\n"
        )
        if transcript:
            note += (
                f"The request that failed was: {transcript!r}. If the user asks you to "
                "try that again, retry it as a fresh request; do not recite this note "
                "or blame the user."
            )
        else:
            note += (
                "If the user retries the request they just made, treat it as fresh. "
                "Do not recite this note."
            )
        self._pending_failure_note = note

    def _speak_finished(self, text: str) -> TurnResult:
        """Evict heavyweight listeners, synthesize once, then restore them."""

        if not text.strip():
            return TurnResult()
        if self._barge.is_set():
            return TurnResult(interrupted=True)
        prepare = self.config.prepare_speech
        restore = self.config.restore_after_speech
        if prepare is not None:
            self._state("thinking", "making room for speech")
            try:
                prepare()
            except Exception as error:  # noqa: BLE001 - report one failed turn
                return TurnResult(error=f"could not make room for speech: {error}")
        if self._barge.is_set():
            if restore is not None:
                try:
                    restore()
                except Exception as error:  # noqa: BLE001 - keep the listener alive
                    return TurnResult(
                        interrupted=True,
                        error=f"could not restore comprehension: {error}",
                    )
            return TurnResult(interrupted=True)

        speech = TurnResult()
        self._trace_content("tts_input", text)
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
        self._trace_content(
            "playback",
            text,
            spoke_seconds=round(speech.spoke_seconds, 3),
            interrupted=speech.interrupted,
            error=speech.error,
        )
        return speech

    def _run(
        self,
        payload: dict[str, Any],
        *,
        speak_only_if_useful: bool = False,
    ) -> TurnResult:
        """One request: stream it, and speak the audio as it arrives."""

        result = TurnResult()
        started = time.monotonic()

        speaker = SpeakerStream(PLAYBACK_RATE_HZ, self.config.output_device)
        with self._speaker_lock:
            self._active_speaker = speaker
        speaking = False
        first_delta_ms: float | None = None
        audio_blocks: set[str] = set()
        declared_audio_blocks = 0
        last_audio_at: float | None = None
        last_audio_block: str | None = None
        max_block_gap_ms = 0.0
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
                    if result.transcript:
                        self._trace_content("heard", result.transcript)
                    if result.audio_observation:
                        self._trace_content(
                            "audio_observation", result.audio_observation
                        )
                    if result.transcript and self._is_recent_playback_echo(
                        result.transcript
                    ):
                        result.echo_suppressed = True
                        logger.info(
                            "suppressed near-verbatim playback echo: %r",
                            result.transcript[:160],
                        )
                        break
                    self._state("thinking", result.transcript)
                elif kind == "tool":
                    items = event.get("tools")
                    if not isinstance(items, list):
                        items = [event]
                    phase = str(event.get("phase") or "").strip()
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        name = str(
                            item.get("name")
                            or item.get("tool")
                            or event.get("name")
                            or event.get("tool")
                            or ""
                        ).strip()
                        if not name:
                            continue
                        if name not in result.tools_used:
                            result.tools_used.append(name)
                        arguments = item.get("arguments")
                        if name == "request_camera_view":
                            result.camera_requested = True
                            if isinstance(arguments, dict):
                                result.camera_motion = (
                                    str(arguments.get("mode") or "still") == "motion"
                                )
                        if name == "background_task" and self.background_agent is not None:
                            self.background_agent.wake()
                        if phase != "complete":
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
                    arrived = time.monotonic()
                    block = str(audio.get("block") or "0")
                    audio_blocks.add(block)
                    try:
                        declared_audio_blocks = max(
                            declared_audio_blocks, int(audio.get("blocks") or 0)
                        )
                    except (TypeError, ValueError):
                        pass
                    if (
                        last_audio_at is not None
                        and last_audio_block is not None
                        and block != last_audio_block
                    ):
                        max_block_gap_ms = max(
                            max_block_gap_ms, (arrived - last_audio_at) * 1000.0
                        )
                    last_audio_at = arrived
                    last_audio_block = block
                    if result.camera_requested:
                        # request_camera_view is a client-side bridge. Any
                        # language/TTS produced after its placeholder result is
                        # provisional until the harness attaches the requested
                        # frame in a second pass.
                        continue
                    if speak_only_if_useful and not result.tools_used:
                        # Nothing was looked up, so this pass has nothing the
                        # first answer did not already say. Collect the text
                        # for the record and stay quiet.
                        continue
                    if not speaking:
                        speaker.start()
                        speaking = True
                        first_delta_ms = (arrived - started) * 1000.0
                        result.first_audio_ms = first_delta_ms
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
                    speaker.stop(fade_s=0.12 if self._barge.is_set() else 0.0)
                    result.interrupted = self._barge.is_set()
                else:
                    speaker.finish()
                    result.interrupted = self._barge.is_set()
                result.spoke_seconds = speaker.played_seconds
                audible_started_at = speaker.audible_started_at
                if audible_started_at is not None:
                    result.first_audio_ms = (audible_started_at - started) * 1000.0
                timing = speaker.timing()
                logger.info(
                    "playback timing: chunks=%d blocks=%d pcm=%.3fs "
                    "first_delta=%.1fms audible=%.1fms startup_wait=%.1fms "
                    "startup_buffer=%.1fms max_packet_gap=%.1fms "
                    "max_block_gap=%.1fms predicted_starvation=%.1fms "
                    "source_wait=%.1fms write_block=%.1fms peak_queue=%.1fms "
                    "interrupted=%s",
                    timing["chunks"],
                    declared_audio_blocks or len(audio_blocks),
                    timing["pcm_seconds"],
                    first_delta_ms or 0.0,
                    result.first_audio_ms or 0.0,
                    timing["startup_wait_ms"],
                    timing["startup_buffer_ms"],
                    timing["max_arrival_gap_ms"],
                    max_block_gap_ms,
                    timing["max_predicted_starvation_ms"],
                    timing["max_source_wait_ms"],
                    timing["max_write_block_ms"],
                    timing["peak_buffer_ms"],
                    result.interrupted,
                )
            else:
                speaker.stop()
            with self._speaker_lock:
                if self._active_speaker is speaker:
                    self._active_speaker = None

        result.total_ms = (time.monotonic() - started) * 1000
        omni = payload.get("omni")
        task = str(omni.get("task") or "") if isinstance(omni, dict) else ""
        if task != "synthesize" and result.reply.strip() and not result.camera_requested:
            self._trace_content(
                "generated",
                result.reply.strip(),
                interrupted=result.interrupted,
                error=result.error,
                tools=list(result.tools_used),
            )
        return result

    def _append_history(self, role: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        self._history.append({"role": role, "content": content})
        self._history_times.append(time.time())
        limit = self.config.history_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]
            self._history_times = self._history_times[-limit:]

    def _history_for_prompt(self, now: float | None = None) -> list[dict[str, Any]]:
        """Recent role-preserving continuity with no model-copyable age labels."""

        if not self._history:
            return []
        now = now or time.time()
        newest_age = max(0.0, now - self._history_times[-1])
        maximum = self.config.history_turns
        if newest_age > 86400:
            return []
        if newest_age > 21600:
            maximum = min(maximum, 2)
        elif newest_age > 3600:
            maximum = min(maximum, 4)
        elif newest_age > 900:
            maximum = min(maximum, 6)
        elif newest_age > 300:
            maximum = min(maximum, 8)

        return [
            {"role": message["role"], "content": str(message["content"])}
            for message in self._history[-maximum:]
        ]

    def _mark_interrupted(self, reply: str, spoke_seconds: float) -> None:
        """Record that generated text was not necessarily heard in full."""

        reply = reply.strip()
        for index in range(len(self._history) - 1, -1, -1):
            message = self._history[index]
            if message.get("role") != "assistant" or message.get("content") != reply:
                continue
            if spoke_seconds < 0.15:
                self._history.pop(index)
                self._history_times.pop(index)
            else:
                message["content"] = (
                    f"{reply}\n[This spoken reply was interrupted before it "
                    "finished. The user may not have heard its later words.]"
                )
            return

    def _remember(self, result: TurnResult) -> None:
        """Keep the dialogue, not the audio: only text carries to the next turn.

        Persistent journaling happens only after all answer/tool/TTS work has
        finished, separately from this prompt history update.
        """

        spoken = result.transcript or result.audio_observation
        reply = result.reply.strip()
        if spoken:
            self._append_history("user", spoken)
        if reply:
            self._append_history("assistant", reply)

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

    def announce(self, text: str) -> TurnResult:
        """Speak a background completion when the live conversation is idle."""

        self._barge.clear()
        text = text.strip()
        if not text:
            return TurnResult()
        speech = self._speak_finished(text)
        self._note_spoken(text, speech.spoke_seconds)
        if not speech.error:
            self._append_history("assistant", f"[Background task update] {text}")
        return speech


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
    processing. Playback first ducks, then pauses only for sustained speech. A
    rejected false start resumes; an accepted utterance becomes the next turn.
    """

    stop = stop or threading.Event()
    # A dead parecord pipe ends this invocation, not the whole harness. Keep a
    # private event for its turn worker so cleanup can join that thread without
    # poisoning the caller-owned stop event that permits the outer supervisor
    # to reopen the microphone.
    worker_stop = threading.Event()
    agent_stop = threading.Event()
    memory_governor = MemoryGovernor()
    foreground_active = threading.Event()
    near_end_active = threading.Event()
    session = CallSession(config, on_state=on_state, frame_grabber=frame_grabber)
    vad = Vad(config.vad)
    outer_notify = on_state or (lambda state, detail: None)

    array = ReSpeaker()
    array.start()

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
    background_announcement_active = threading.Event()
    waiting = CallQueue(CAPTURE_RATE_HZ)
    lock = threading.Lock()
    work: queue.Queue[Pending] = queue.Queue(maxsize=1)
    background_announcements: queue.Queue[dict[str, Any]] = queue.Queue()
    background_store: BackgroundTaskStore | None = None

    if config.background_task_path and config.tools_enabled:
        background_store = BackgroundTaskStore(config.background_task_path)

        def queue_progress(update: dict[str, Any]) -> None:
            """Pause agent work while its optional spoken checkpoint has the floor."""

            acknowledged = threading.Event()
            background_announcements.put(
                {**update, "_kind": "progress", "_ack": acknowledged}
            )
            while not acknowledged.wait(0.2):
                if stop.is_set() or agent_stop.is_set():
                    return

        for pending_announcement in background_store.pending_announcements():
            background_announcements.put(
                {**pending_announcement, "_kind": "terminal"}
            )
        session.background_agent = BackgroundAgent(
            store=background_store,
            portal_url=config.portal_url,
            token=config.token,
            model=config.model,
            portal_session_id=session.portal_session_id,
            foreground_active=foreground_active,
            stop=agent_stop,
            token_reader=config.token_reader,
            await_language=config.await_comprehension,
            on_complete=lambda item: background_announcements.put(
                {**item, "_kind": "terminal"}
            ),
            on_progress=(queue_progress if config.background_progress_speech else None),
            request_timeout_s=config.request_timeout_s,
            memory_governor=memory_governor,
            prepare_action_residency=config.prepare_background_action,
        )
        session.background_agent.start()

    if session.memory is not None:
        calibration_path = (
            Path(config.memory_calibration_path)
            if config.memory_calibration_path
            else None
        )

        def admit_memory(encoder_payload_gib: float) -> bool:
            # Encoder work is lower priority than both a person and an agent
            # task. Wait until the shared Qwen worker has finished restoring,
            # then use its live calibration and current MemAvailable rather
            # than a fixed board-size assumption.
            if foreground_active.is_set():
                return False
            try:
                memory_governor.require(
                    "deferred memory encoding", reserve_gib=encoder_payload_gib
                )
            except MemoryPressure:
                return False
            if (
                session.background_agent is not None
                and (
                    session.background_agent.active.is_set()
                    or session.background_agent.has_work()
                )
            ):
                return False
            if (
                config.comprehension_ready is not None
                and not config.comprehension_ready()
            ):
                return False
            if calibration_path is None:
                return True
            return memory_capacity_available(
                calibration_path, encoder_payload_gib
            )

        session.memory.set_admission(admit_memory)

    def notify(state: State, detail: str = "") -> None:
        nonlocal speaking_since
        speaking_since = time.monotonic() if state == "speaking" else None
        array.set_state(STATE_TO_RING.get(state, "trace"))
        outer_notify(state, detail)

    session._on_state = notify

    def worker() -> None:
        """Take turns one at a time, off the thread that holds the microphone."""

        while not stop.is_set() and not worker_stop.is_set():
            pending: Pending | None = None
            announcement: dict[str, Any] | None = None
            try:
                pending = work.get_nowait()
            except queue.Empty:
                with lock:
                    person_waiting = bool(waiting)
                if not foreground_active.is_set() and not person_waiting:
                    try:
                        announcement = background_announcements.get_nowait()
                    except queue.Empty:
                        pass
                if announcement is None:
                    worker_stop.wait(0.1)
                    continue
            busy.set()
            foreground_active.set()
            if announcement is not None:
                background_announcement_active.set()
            if pending is not None:
                # The utterance that produced this work item is now owned by
                # the foreground worker. A later VAD start will set this again.
                near_end_active.clear()
            try:
                if pending is not None:
                    session.direction = describe_direction(array.direction)
                    result = session.take_turn(
                        pending.audio(), segments=max(1, pending.segments)
                    )
                    if result.interrupted:
                        logger.info("reply yielded to the speaker")
                    if on_turn:
                        on_turn(result)
                elif announcement is not None:
                    task_id = str(announcement.get("task_id") or "")
                    report = str(announcement.get("result") or "").strip()
                    kind = str(announcement.get("_kind") or "terminal")
                    if report:
                        logger.info("announcing %s background task %s", kind, task_id)
                        result = session.announce(report)
                        if result.error:
                            logger.warning(
                                "background %s %s could not be spoken: %s",
                                kind,
                                task_id,
                                result.error,
                            )
                        elif (
                            kind == "terminal"
                            and background_store is not None
                        ):
                            background_store.mark_announced(task_id)
            except Exception as error:  # noqa: BLE001 - one turn is not the call
                logger.warning("turn failed: %s", error)
            finally:
                background_announcement_active.clear()
                if announcement is not None:
                    acknowledged = announcement.get("_ack")
                    if isinstance(acknowledged, threading.Event):
                        acknowledged.set()
                busy.clear()
                with lock:
                    accepted_audio_waiting = bool(waiting)
                if not _foreground_lane_still_owned(
                    accepted_audio_waiting=accepted_audio_waiting,
                    queued_turn=not work.empty(),
                ):
                    foreground_active.clear()
                if session.background_agent is not None:
                    session.background_agent.wake()
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
    barge = BargeInState()
    try:
        with microphone:
            listening_announced = False
            for frame in microphone.frames():
                if stop.is_set():
                    return
                if not listening_announced:
                    # Popen alone does not prove the selected source works.
                    # Publish listening only after a full capture frame arrives.
                    notify("listening", "")
                    listening_announced = True
                now_ms += frame_ms
                verdict = vad.process(
                    frame,
                    now_ms,
                    frame_ms,
                    # The array's native detector is conservative at room
                    # distance. Use it only to distinguish near-end speech
                    # from the far-end audio while our own speaker has the
                    # floor; ordinary listening keeps the proven adaptive VAD.
                    native_speech=(
                        array.speech_detected
                        if array.present and speaking_since is not None
                        else None
                    ),
                )
                now = time.monotonic()

                if verdict.event in {"candidate", "start", "active"}:
                    near_end_active.set()
                    # Still talking, so nothing is finished being said.
                    settle_until = None
                    if verdict.event == "start":
                        # A confirmed near-end speech start immediately owns the
                        # shared model lane. Waiting until end-of-utterance leaves a
                        # long background generation running while the person talks,
                        # delaying ASR and making the system appear uninterruptible.
                        foreground_active.set()
                        if session.background_agent is not None:
                            session.background_agent.wake()
                        if not busy.is_set():
                            notify("hearing", "")
                    if (
                        verdict.event in {"start", "active"}
                        and can_barge
                        and speaking_since is not None
                    ):
                        for action in barge.observe(
                            verdict.event,
                            now=now,
                            reply_started_at=speaking_since,
                            grace_s=config.barge_in_grace_s,
                            pause_s=config.barge_in_pause_s,
                        ):
                            if action == "duck":
                                logger.info("possible interruption; ducking reply")
                                session.request_duck()
                            elif action == "pause":
                                logger.info("sustained interruption; pausing reply")
                                session.request_pause()
                elif verdict.event == "rejected":
                    near_end_active.clear()
                    for action in barge.observe(
                        "rejected",
                        now=now,
                        reply_started_at=speaking_since or now,
                        grace_s=config.barge_in_grace_s,
                        pause_s=config.barge_in_pause_s,
                    ):
                        if action == "resume":
                            logger.info("interruption rejected; resuming reply")
                            session.resume_reply()
                    if not busy.is_set():
                        notify("listening", "")
                        foreground_active.clear()
                        if session.background_agent is not None:
                            session.background_agent.wake()
                elif verdict.event == "utterance" and verdict.utterance is not None:
                    # Confirmed speech has owned the foreground/model lane since
                    # its VAD start. Acceptance keeps that ownership through ASR,
                    # reasoning, and reply; rejected speech releases it above.
                    foreground_active.set()
                    if session.background_agent is not None:
                        session.background_agent.wake()
                    should_preempt = _accepted_utterance_preempts(
                        busy=busy.is_set(),
                        reply_started=speaking_since is not None,
                        can_barge=can_barge,
                        background_announcement=background_announcement_active.is_set(),
                    )
                    if should_preempt:
                        # A person always wins over a background announcement.
                        # Foreground work is interrupted only after its reply
                        # has actually started streaming to the speaker.
                        if barge.active:
                            logger.info("interruption confirmed; yielding to speaker")
                        session.request_barge()
                    elif busy.is_set() and speaking_since is None:
                        logger.info(
                            "accepted audio queued while the current foreground "
                            "reply is not yet audible"
                        )
                    barge.reset()
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
        worker_stop.set()
        agent_stop.set()
        turns.join(timeout=5)
        array.stop()
        session.close()
