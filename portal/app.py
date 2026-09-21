"""Authenticated, phone-first web portal for the Robit Omni adapter.

The portal is intentionally a narrow same-origin proxy. It never exposes the
component workers or Ollama directly, pins requests to one published model,
and executes only an allowlisted set of explicit web, document, and session-memory
demonstration tools.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    render_template,
    request,
    stream_with_context,
)
from waitress import serve

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_omni_adapters.audio import AudioContractError, decode_wav_payload
from qwen_omni_adapters.context import context_text
from qwen_omni_adapters.decision_plane import DecisionPlane, DecisionState, DecisionWaveResult
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPolicy

try:
    from portal.background_tasks import BackgroundTaskStore
    from portal.documents import DocumentError, SessionDocumentStore
    from portal.environment import portal_behavior_system_message
    from portal.tools import (
        DISCOVERY_TOOLS,
        SAFE_TOOLS,
        PortalToolHarness,
        ToolInputError,
        discover_tool_names,
        tool_result_json,
        tool_schemas,
        tool_use_instructions,
    )
except ModuleNotFoundError:  # Direct script execution from portal/.
    from background_tasks import BackgroundTaskStore
    from documents import DocumentError, SessionDocumentStore
    from environment import portal_behavior_system_message
    from tools import (
        DISCOVERY_TOOLS,
        SAFE_TOOLS,
        PortalToolHarness,
        ToolInputError,
        discover_tool_names,
        tool_result_json,
        tool_schemas,
        tool_use_instructions,
    )

ADAPTER_SCHEMA = "robit.ollama.omni-adapter.v1"
DEFAULT_MODEL = "robit/qwen3.8-27b-e03-obliterated-omni:q4km"
TOOL_RESULT_POLICY = context_text("directives", "tool_result_policy")
VOICE_PROFILE_SCHEMA = "robit.omni.voice-profile.v1"
QWEN3_TTS_LANGUAGES = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
VOICE_SPEECH_FIELDS = {
    "language",
    "speaker_file",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "max_frames",
}
VOICE_CLIENT_FIELDS = {
    "clone_enabled",
    "preset",
    "speaker_audio",
    "language",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "max_frames",
}
MAX_SPEAKER_REFERENCE_BYTES = 10 * 1024 * 1024
SESSION_COOKIE_NAME = "omni_portal_session"
DIAGNOSTIC_TTL_SECONDS = 5 * 60
# Productive chains have no numeric round ceiling. This guard only stops a
# model that keeps changing searches/calls without obtaining actionable data.
MAX_STALLED_TOOL_ROUNDS = 8
DIAGNOSTIC_NUMERIC_FIELDS = {
    "queue_wait_ms",
    "upstream_headers_ms",
    "first_upstream_byte_ms",
    "response_headers_ms",
    "first_event_ms",
    "first_text_ms",
    "tts_stage_ms",
    "audio_start_ms",
    "first_audio_delta_ms",
    "complete_ms",
    "total_ms",
    "status",
    "tool_round",
    "decision_latency_ms",
    "decision_confidence",
}
DIAGNOSTIC_BOOLEAN_FIELDS = {
    "audio_requested",
    "thinking_requested",
    "has_audio_input",
    "has_image_input",
    "has_video_input",
    "has_document_input",
    "tools_requested",
    "tool_ok",
    "decision_shadow",
    "decision_escalated",
    "decision_fast_path",
}
DIAGNOSTIC_STRING_FIELDS = {
    "request_id",
    "task",
    "outcome",
    "tool_name",
    "media_id",
    "decision_wave",
    "decision_id",
    "decision_value",
    "decision_source",
    "decision_band",
    "decision_checkpoint",
}

def load_voice_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read voice profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise TypeError("voice profile must be a JSON object")
    if profile.get("schema") != VOICE_PROFILE_SCHEMA:
        raise RuntimeError(f"voice profile schema must be {VOICE_PROFILE_SCHEMA}")
    unknown = set(profile) - VOICE_SPEECH_FIELDS - {"schema", "name", "presets"}
    if unknown:
        raise RuntimeError(f"unknown voice profile fields: {sorted(unknown)}")
    language = str(profile.get("language") or "en").strip()
    if language not in QWEN3_TTS_LANGUAGES:
        supported = ", ".join(sorted(QWEN3_TTS_LANGUAGES))
        raise RuntimeError(f"voice profile language must be one of: {supported}")
    profile["language"] = language

    def resolve_speaker(value: Any, field: str) -> str:
        speaker_value = str(value or "").strip()
        if not speaker_value:
            raise RuntimeError(f"voice profile {field} must not be empty")
        speaker_path = Path(speaker_value).expanduser()
        if not speaker_path.is_absolute():
            speaker_path = path.parent / speaker_path
        speaker_path = speaker_path.resolve()
        if not speaker_path.is_file():
            raise RuntimeError(f"voice profile {field} does not exist: {speaker_path}")
        if speaker_path.suffix.lower() not in {".wav", ".mp3"}:
            raise RuntimeError(f"voice profile {field} must be WAV or MP3")
        return str(speaker_path)

    speaker = str(profile.get("speaker_file") or "").strip()
    if speaker:
        profile["speaker_file"] = resolve_speaker(speaker, "speaker_file")
    else:
        profile.pop("speaker_file", None)

    raw_presets = profile.get("presets", [])
    if not isinstance(raw_presets, list):
        raise TypeError("voice profile presets must be an array")
    presets: list[dict[str, Any]] = []
    preset_ids: set[str] = set()
    for index, raw_preset in enumerate(raw_presets):
        if not isinstance(raw_preset, Mapping):
            raise TypeError(f"voice profile preset {index} must be an object")
        unknown_preset = set(raw_preset) - {"id", "label", "speaker_file", "default"}
        if unknown_preset:
            raise RuntimeError(f"unknown voice preset fields: {sorted(unknown_preset)}")
        preset_id = str(raw_preset.get("id") or "").strip()
        if (
            not preset_id
            or len(preset_id) > 64
            or not all(character.isalnum() or character in {"-", "_"} for character in preset_id)
        ):
            raise RuntimeError("voice preset id must be 1–64 safe characters")
        if preset_id in preset_ids:
            raise RuntimeError(f"duplicate voice preset id: {preset_id}")
        preset_ids.add(preset_id)
        label = str(raw_preset.get("label") or "").strip()
        if not label or len(label) > 80:
            raise RuntimeError("voice preset label must be 1–80 characters")
        is_default = raw_preset.get("default", False)
        if not isinstance(is_default, bool):
            raise TypeError("voice preset default must be a boolean")
        presets.append(
            {
                "id": preset_id,
                "label": label,
                "speaker_file": resolve_speaker(
                    raw_preset.get("speaker_file"),
                    f"preset {preset_id} speaker_file",
                ),
                "default": is_default,
            }
        )
    if presets:
        defaults = [preset for preset in presets if preset["default"]]
        if len(defaults) != 1:
            raise RuntimeError("voice profile presets require exactly one default")
        profile["presets"] = presets
        profile["speaker_file"] = defaults[0]["speaker_file"]
    else:
        profile.pop("presets", None)
    numeric_ranges = {
        "temperature": (0.0, 2.0),
        "top_k": (0, 1000),
        "top_p": (0.0, 1.0),
        "max_frames": (1, 2048),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        if key not in profile:
            continue
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"voice profile {key} must be numeric")
        if not minimum <= value <= maximum:
            raise RuntimeError(f"voice profile {key} must be between {minimum} and {maximum}")
    if "seed" in profile and (
        isinstance(profile["seed"], bool) or not isinstance(profile["seed"], int)
    ):
        raise RuntimeError("voice profile seed must be an integer")
    if not -1 <= int(profile.get("seed", 42)) <= 2_147_483_647:
        raise RuntimeError("voice profile seed must be -1 or a 32-bit non-negative integer")
    return profile


@dataclass(frozen=True)
class PortalConfig:
    adapter_url: str
    adapter_health_url: str
    comprehension_health_url: str
    tts_health_url: str
    ollama_health_url: str
    model: str
    access_token: str
    voice_profile: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 1200
    max_body_bytes: int = 96 * 1024 * 1024
    inference_slots: int = 1
    max_inflight_requests: int = 4
    session_log_dir: Path | None = None
    session_log_ttl_s: float = DIAGNOSTIC_TTL_SECONDS
    background_task_path: Path | None = None
    memory_policy: MemoryPolicy | None = None
    decision_plane_enabled: bool = False

    @classmethod
    def from_environment(cls) -> PortalConfig:
        adapter_url = os.environ.get("OMNI_ADAPTER_URL", "http://127.0.0.1:8910/api/chat").strip()
        access_token = os.environ.get("OMNI_PORTAL_TOKEN", "").strip()
        if len(access_token) < 24:
            raise RuntimeError("OMNI_PORTAL_TOKEN must contain at least 24 characters")
        profile_path = Path(
            os.environ.get(
                "OMNI_VOICE_PROFILE",
                str(Path(__file__).resolve().parent / "voice-profile.json"),
            )
        ).expanduser()
        return cls(
            adapter_url=adapter_url,
            adapter_health_url=os.environ.get(
                "OMNI_ADAPTER_HEALTH_URL", "http://127.0.0.1:8910/healthz"
            ).strip(),
            comprehension_health_url=os.environ.get(
                # Empty means this deployment runs no comprehension worker.
                "OMNI_COMPREHENSION_HEALTH_URL", "http://127.0.0.1:8901/health"
            ).strip(),
            tts_health_url=os.environ.get(
                "OMNI_TTS_HEALTH_URL", "http://127.0.0.1:8892/healthz"
            ).strip(),
            ollama_health_url=os.environ.get(
                "OMNI_OLLAMA_HEALTH_URL", "http://127.0.0.1:11434/api/tags"
            ).strip(),
            model=os.environ.get("OMNI_MODEL", DEFAULT_MODEL).strip(),
            access_token=access_token,
            voice_profile=load_voice_profile(profile_path),
            timeout_s=float(os.environ.get("OMNI_PORTAL_TIMEOUT_S", "1200")),
            max_body_bytes=int(os.environ.get("OMNI_PORTAL_MAX_BODY_BYTES", str(96 * 1024 * 1024))),
            inference_slots=max(1, int(os.environ.get("OMNI_PORTAL_INFERENCE_SLOTS", "1"))),
            max_inflight_requests=max(
                1, int(os.environ.get("OMNI_PORTAL_MAX_INFLIGHT_REQUESTS", "4"))
            ),
            session_log_dir=Path(
                os.environ.get(
                    "OMNI_PORTAL_SESSION_LOG_DIR",
                    "runtime-data/session-logs",
                )
            ).expanduser(),
            session_log_ttl_s=max(
                1.0,
                float(
                    os.environ.get(
                        "OMNI_PORTAL_SESSION_LOG_TTL_S",
                        str(DIAGNOSTIC_TTL_SECONDS),
                    )
                ),
            ),
            background_task_path=Path(
                os.environ.get(
                    "OMNI_BACKGROUND_TASKS",
                    str(
                        Path(os.environ.get("OMNI_REPO_ROOT") or ".")
                        / "runtime-data/state/background-tasks.json"
                    ),
                )
            ).expanduser(),
            memory_policy=MemoryPolicy.from_environment(),
            decision_plane_enabled=os.environ.get(
                "OMNI_DECISION_PLANE_ENABLED", "0"
            ).strip().lower()
            not in {"0", "false", "no", "off"},
        )


class PortalError(RuntimeError):
    """A safe, user-visible portal failure."""


class PortalRequestError(ValueError):
    """A safe client request validation failure."""


@dataclass
class _InferenceTicket:
    session_id: str
    released: bool = False


class _InferenceQueue:
    """Bounded, session-counted queue around isolated upstream requests."""

    def __init__(self, slots: int, max_inflight: int) -> None:
        self.slots = max(1, slots)
        self.max_inflight = max(self.slots, max_inflight)
        self._gate = threading.BoundedSemaphore(self.slots)
        self._lock = threading.Lock()
        self._inflight = 0
        self._active = 0
        self._sessions: dict[str, int] = {}

    def acquire(self, session_id: str, timeout_s: float) -> _InferenceTicket | None:
        with self._lock:
            if self._inflight >= self.max_inflight:
                return None
            self._inflight += 1
            self._sessions[session_id] = self._sessions.get(session_id, 0) + 1
        if not self._gate.acquire(timeout=timeout_s):
            self._remove_inflight(session_id)
            return None
        with self._lock:
            self._active += 1
        return _InferenceTicket(session_id=session_id)

    def _remove_inflight(self, session_id: str) -> None:
        with self._lock:
            self._inflight -= 1
            remaining = self._sessions.get(session_id, 1) - 1
            if remaining > 0:
                self._sessions[session_id] = remaining
            else:
                self._sessions.pop(session_id, None)

    def release(self, ticket: _InferenceTicket) -> None:
        with self._lock:
            if ticket.released:
                return
            ticket.released = True
            self._active -= 1
        self._gate.release()
        self._remove_inflight(ticket.session_id)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "users": len(self._sessions),
                "inflight": self._inflight,
                "active": self._active,
                "queued": self._inflight - self._active,
                "slots": self.slots,
                "limit": self.max_inflight,
            }


@dataclass
class _DiagnosticSession:
    last_seen: float
    last_seen_at: str
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=240))
    request_ids: set[str] = field(default_factory=set)
    timer: threading.Timer | None = None


class _SessionDiagnostics:
    """Short-lived, content-redacted diagnostic journals keyed by session."""

    def __init__(self, directory: Path | None, ttl_s: float) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self.ttl_s = max(0.05, ttl_s)
        self._lock = threading.Lock()
        self._sessions: dict[str, _DiagnosticSession] = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for stale in self.directory.glob("*.json"):
                stale.unlink(missing_ok=True)
            for stale in self.directory.glob("*.tmp"):
                stale.unlink(missing_ok=True)

    @staticmethod
    def _key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_text() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _schedule_locked(self, key: str, record: _DiagnosticSession) -> None:
        if record.timer is not None:
            return
        timer = threading.Timer(self.ttl_s, self._expire, args=(key,))
        timer.daemon = True
        record.timer = timer
        timer.start()

    def _expire(self, key: str) -> None:
        path: Path | None = None
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return
            record.timer = None
            remaining = self.ttl_s - (time.monotonic() - record.last_seen)
            if remaining > 0:
                timer = threading.Timer(remaining, self._expire, args=(key,))
                timer.daemon = True
                record.timer = timer
                timer.start()
                return
            self._sessions.pop(key, None)
            if self.directory is not None:
                path = self.directory / f"{key}.json"
        if path is not None:
            path.unlink(missing_ok=True)

    def _persist_locked(self, key: str, record: _DiagnosticSession) -> None:
        if self.directory is None:
            return
        path = self.directory / f"{key}.json"
        temporary = self.directory / f"{key}.tmp"
        payload = {
            "schema": "robit.omni.session-diagnostics.v1",
            "session": key[:16],
            "last_seen_at": record.last_seen_at,
            "ttl_seconds": self.ttl_s,
            "events": list(record.events),
        }
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        temporary.replace(path)

    def open_session(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            now = time.monotonic()
            record = self._sessions.get(key)
            if record is None:
                record = _DiagnosticSession(now, self._now_text())
                self._sessions[key] = record
                self._schedule_locked(key, record)
            else:
                record.last_seen = now
                record.last_seen_at = self._now_text()

    def touch(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()

    def begin_request(self, session_id: str, request_id: str, fields: Mapping[str, Any]) -> None:
        self.open_session(session_id)
        key = self._key(session_id)
        with self._lock:
            record = self._sessions[key]
            record.request_ids.add(request_id)
            self._append_locked(
                key,
                record,
                "request_received",
                {"request_id": request_id, **dict(fields)},
            )

    def record(
        self,
        session_id: str,
        event: str,
        fields: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> bool:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return False
            if request_id is not None and request_id not in record.request_ids:
                return False
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()
            self._append_locked(key, record, event, fields or {})
            return True

    def _append_locked(
        self,
        key: str,
        record: _DiagnosticSession,
        event: str,
        fields: Mapping[str, Any],
    ) -> None:
        item = {"at": self._now_text(), "event": event}
        item.update(_diagnostic_fields(fields))
        record.events.append(item)
        self._persist_locked(key, record)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return {"ttl_seconds": self.ttl_s, "events": []}
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()
            return {
                "ttl_seconds": self.ttl_s,
                "events": copy.deepcopy(list(record.events)),
            }

    def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.pop(key, None)
            if record is not None and record.timer is not None:
                record.timer.cancel()
        if self.directory is not None:
            (self.directory / f"{key}.json").unlink(missing_ok=True)
            (self.directory / f"{key}.tmp").unlink(missing_ok=True)


def _diagnostic_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in DIAGNOSTIC_NUMERIC_FIELDS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or numeric > 86_400_000:
            continue
        result[key] = (
            int(numeric) if key in {"status", "tool_round"} else round(numeric, 2)
        )
    for key in DIAGNOSTIC_BOOLEAN_FIELDS:
        value = raw.get(key)
        if isinstance(value, bool):
            result[key] = value
    for key in DIAGNOSTIC_STRING_FIELDS:
        value = raw.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if (
            value
            and len(value) <= 64
            and all(character.isalnum() or character in {"-", "_", "."} for character in value)
        ):
            result[key] = value
    return result


def _request_diagnostic_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    omni = payload.get("omni")
    task = str(omni.get("task") or "chat") if isinstance(omni, Mapping) else "chat"
    messages = payload.get("messages")
    message_items = messages if isinstance(messages, list) else []
    modalities = payload.get("response_modalities")
    output_items = modalities if isinstance(modalities, list) else []
    return {
        "task": task,
        "audio_requested": "audio" in output_items,
        "thinking_requested": payload.get("think") is True,
        "has_audio_input": any(
            isinstance(message, Mapping) and bool(message.get("audios"))
            for message in message_items
        ),
        "has_image_input": any(
            isinstance(message, Mapping) and bool(message.get("images"))
            for message in message_items
        ),
        "has_video_input": any(
            isinstance(message, Mapping) and bool(message.get("videos"))
            for message in message_items
        ),
        "has_document_input": any(
            isinstance(message, Mapping) and bool(message.get("documents"))
            for message in message_items
        ),
    }


def _request_media_digests(payload: Mapping[str, Any]) -> list[str]:
    """Return bounded content identities without retaining or decoding uploaded media."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    result: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        for media_field in ("audios", "images", "videos", "documents"):
            values = message.get(media_field)
            if not isinstance(values, list):
                continue
            for envelope in values:
                if not isinstance(envelope, Mapping):
                    continue
                encoded = envelope.get("data")
                if not isinstance(encoded, str) or not encoded:
                    continue
                digest = hashlib.sha256(
                    f"{media_field}\0{encoded}".encode()
                ).hexdigest()[:16]
                result.append(f"{media_field[:-1]}-{digest}")
                if len(result) >= 8:
                    return result
    return result


def _voice_override(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PortalRequestError("portal_voice must be an object")
    unknown = set(raw) - VOICE_CLIENT_FIELDS
    if unknown:
        raise PortalRequestError(f"unknown portal_voice fields: {sorted(unknown)}")

    clone_enabled = raw.get("clone_enabled", False)
    if not isinstance(clone_enabled, bool):
        raise PortalRequestError("portal_voice.clone_enabled must be a boolean")
    result: dict[str, Any] = {"clone_enabled": clone_enabled}

    if "language" in raw:
        language = str(raw["language"] or "").strip()
        if language not in QWEN3_TTS_LANGUAGES:
            supported = ", ".join(sorted(QWEN3_TTS_LANGUAGES))
            raise PortalRequestError(f"portal_voice.language must be one of: {supported}")
        result["language"] = language

    numeric_ranges = {
        "temperature": (0.0, 2.0, float),
        "top_k": (0, 1000, int),
        "top_p": (0.0, 1.0, float),
        "max_frames": (1, 2048, int),
    }
    for key, (minimum, maximum, expected_type) in numeric_ranges.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PortalRequestError(f"portal_voice.{key} must be numeric")
        if expected_type is int and not isinstance(value, int):
            raise PortalRequestError(f"portal_voice.{key} must be an integer")
        if not minimum <= value <= maximum:
            raise PortalRequestError(f"portal_voice.{key} must be between {minimum} and {maximum}")
        result[key] = expected_type(value)

    if "seed" in raw:
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PortalRequestError("portal_voice.seed must be an integer")
        if not -1 <= seed <= 2_147_483_647:
            raise PortalRequestError(
                "portal_voice.seed must be -1 or a 32-bit non-negative integer"
            )
        result["seed"] = seed

    reference = raw.get("speaker_audio")
    if reference is not None:
        if not clone_enabled:
            raise PortalRequestError("portal_voice.speaker_audio requires clone_enabled=true")
        try:
            decoded = decode_wav_payload(reference, max_bytes=MAX_SPEAKER_REFERENCE_BYTES)
        except AudioContractError as exc:
            raise PortalRequestError(f"invalid voice reference: {exc}") from exc
        if decoded.duration_ms < 500:
            raise PortalRequestError("voice reference must be at least 0.5 seconds")
        if decoded.duration_ms > 30_000:
            raise PortalRequestError("voice reference must be no longer than 30 seconds")
        result["speaker_audio"] = {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(decoded.data).decode("ascii"),
        }
    preset = raw.get("preset")
    if preset is not None:
        if not clone_enabled:
            raise PortalRequestError("portal_voice.preset requires clone_enabled=true")
        if reference is not None:
            raise PortalRequestError("portal_voice.preset and speaker_audio are mutually exclusive")
        preset_id = str(preset or "").strip()
        if (
            not preset_id
            or len(preset_id) > 64
            or not all(character.isalnum() or character in {"-", "_"} for character in preset_id)
        ):
            raise PortalRequestError("portal_voice.preset is invalid")
        result["preset"] = preset_id
    return result


def _json_object(response: httpx.Response, stage: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise PortalError(f"{stage} did not return JSON") from exc
    if not isinstance(data, dict):
        raise PortalError(f"{stage} returned a non-object JSON response")
    return data


def _tool_arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, Mapping):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


_TEXT_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
    re.IGNORECASE,
)


def _text_tool_call(payload: str, index: int) -> dict[str, Any] | None:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    function = decoded.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = decoded.get("name") or decoded.get("tool") or function
        arguments = (
            decoded.get("arguments")
            or decoded.get("args")
            or decoded.get("parameters")
            or {}
        )
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(arguments, Mapping):
        arguments = {"value": arguments}
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"text-tool-{index}-{digest}",
        "type": "function",
        "function": {"name": name.strip(), "arguments": dict(arguments)},
    }


def _response_tool_calls(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return []
    structured = message.get("tool_calls")
    if isinstance(structured, list) and structured:
        return [call for call in structured if isinstance(call, Mapping)]
    content = message.get("content")
    if not isinstance(content, str) or "<tool_call" not in content.lower():
        return []
    matches = list(_TEXT_TOOL_CALL_PATTERN.finditer(content))
    parsed = [
        call
        for index, match in enumerate(matches)
        if (call := _text_tool_call(match.group(1), index)) is not None
    ]
    if not matches or len(parsed) != len(matches):
        raise PortalError("model emitted a malformed textual tool call")
    if isinstance(message, dict):
        message["content"] = _TEXT_TOOL_CALL_PATTERN.sub("", content).strip()
        message["tool_calls"] = parsed
    return parsed


def _without_media(messages: list[Any]) -> list[Any]:
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        if isinstance(message, dict):
            message.pop("audios", None)
            message.pop("images", None)
            message.pop("videos", None)
            message.pop("documents", None)
            if message.get("tool_calls"):
                # Private reasoning from completed rounds is neither evidence
                # nor conversation continuity. Carrying it forward only burns
                # context that recalled memories and actual results need.
                message.pop("thinking", None)
    return cleaned


def _latest_user_context(messages: list[Any]) -> str:
    """Return current user text for an explicit server-side helper handoff."""

    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text") or "").strip()
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part)
    return ""


def _latest_concrete_tool_context(messages: list[Any]) -> str:
    """Return the newest concrete result, skipping discovery metadata."""

    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        if message.get("tool_name") == "tool_search":
            continue
        return str(message.get("content") or "").strip()
    return ""


def _tool_followup(
    payload: Mapping[str, Any],
    response: Mapping[str, Any],
    harness: PortalToolHarness,
    session_id: str,
    seen: set[str],
    blocked_tools: set[str] | None = None,
    decision_observer: Any | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return None, [], False
    calls = _response_tool_calls(response)
    if not calls:
        return None, [], False

    followup = copy.deepcopy(dict(payload))
    messages = _without_media(list(followup.get("messages") or []))
    assistant = {
        key: copy.deepcopy(value)
        for key, value in message.items()
        if key in {"role", "content", "tool_calls"}
    }
    assistant["role"] = "assistant"
    messages.append(assistant)
    executed: list[dict[str, Any]] = []
    discovered: list[str] = []
    active: list[str] = []
    known_names = {item["function"]["name"] for item in SAFE_TOOLS}
    made_progress = False
    blocked_tools = blocked_tools or set()
    if decision_observer is not None:
        decision_observer(
            "pre_action",
            DecisionState(
                user_request=_latest_user_context(messages),
                current_goal=_latest_user_context(messages),
                current_phase="pre_action",
                last_action={
                    "calls": [
                        {
                            "name": str(call.get("function", {}).get("name") or "")
                            if isinstance(call.get("function"), Mapping)
                            else "",
                            "arguments": dict(_tool_arguments(call)),
                        }
                        for call in calls
                        if isinstance(call, Mapping)
                    ]
                },
                authorization_state={
                    "blocked_tools": sorted(blocked_tools),
                    "server_allowlist": True,
                },
            ),
        )
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        name = (
            str(function.get("name") or "")
            if isinstance(function, Mapping)
            else ""
        )
        arguments = dict(_tool_arguments(call))
        if name == "subagent_delegate" and not str(
            arguments.get("context") or ""
        ).strip():
            source = arguments.get("context_source")
            if source == "current_user_message":
                arguments["context"] = _latest_user_context(messages)
            elif source == "latest_non_discovery_tool_result":
                arguments["context"] = _latest_concrete_tool_context(messages)
        fingerprint = hashlib.sha256(
            f"{name}\0{json.dumps(arguments, sort_keys=True, default=str)}".encode()
        ).hexdigest()
        duplicate = fingerprint in seen
        blocked = name in blocked_tools
        if duplicate:
            result = {
                "error": "duplicate_tool_call",
                "message": "This exact tool call already ran in the current turn; use its prior result.",
            }
        elif blocked:
            seen.add(fingerprint)
            result = {
                "error": "tool_not_available_in_this_execution_profile",
                "message": (
                    "Host execution is delegated in this live turn. Call "
                    "background_task with action=start and a self-contained objective."
                ),
            }
        else:
            seen.add(fingerprint)
            made_progress = True
            result = harness.execute(session_id, name, arguments)
            if name == "tool_search" and isinstance(result, Mapping):
                # Discovery must respect the same execution profile as direct
                # schemas. Otherwise hiding foreground shell only delays it by
                # one model round, which is how a synchronous loop returned.
                result = dict(result)
                for field in ("available_tools", "suggested_tools"):
                    values = result.get(field)
                    if isinstance(values, list):
                        result[field] = [
                            str(item) for item in values if str(item) not in blocked_tools
                        ]
                results = result.get("results")
                if isinstance(results, list):
                    result["results"] = [
                        item
                        for item in results
                        if not isinstance(item, Mapping)
                        or str(item.get("name") or "") not in blocked_tools
                    ]
                available = result.get("available_tools")
                if isinstance(available, list):
                    discovered.extend(str(item) for item in available)
            elif name in known_names:
                active.append(name)
        content = tool_result_json(result)
        tool_message: dict[str, Any] = {
            "role": "tool",
            "tool_name": name or "unknown",
            "content": content,
        }
        if call.get("id"):
            tool_message["tool_call_id"] = str(call["id"])
        messages.append(tool_message)
        display_arguments = {
            key: copy.deepcopy(value)
            for key, value in arguments.items()
            if key
            in {
                "query",
                "url",
                "topic",
                "key",
                "mode",
                "num_results",
                "max_results",
                "max_length",
                "objective",
                "completion_criteria",
                "action",
                "guidance",
                "role",
                "context_source",
                "task_id",
                "command",
                "cwd",
                "timeout_seconds",
            }
        }
        ok = "error" not in result
        if name == "shell":
            ok = (
                ok
                and result.get("exit_code") == 0
                and result.get("timed_out") is not True
            )
        executed.append(
            {
                "id": str(call.get("id") or fingerprint[:12]),
                "name": name or "unknown",
                "arguments": display_arguments,
                "ok": ok,
                "result": content,
                "status": "complete",
                "duplicate": duplicate,
            }
        )
    if decision_observer is not None:
        decision_observer(
            "post_action",
            DecisionState(
                user_request=_latest_user_context(messages),
                current_goal=_latest_user_context(messages),
                current_phase="post_action",
                last_action={
                    "calls": [
                        {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                        }
                        for item in executed
                    ]
                },
                last_observation=json.dumps(
                    _tool_trace(executed), ensure_ascii=False, default=str
                ),
                pending_requirements=("verify goal completion",),
            ),
        )
    followup["messages"] = messages
    # A required choice applies only to the first live decision. Once the
    # model selected a real tool, normal iterative tool use resumes.
    followup.pop("tool_choice", None)
    # Keep only the tool actively doing the work, or freshly discovered
    # candidates. This stays tiny while allowing iterative shell work to fix
    # or verify a command without paying for another discovery inference.
    # An empty discovery preserves the current concrete schema: ffmpeg, for
    # example, is a program inside shell rather than a separate portal tool.
    current = [
        str(item.get("function", {}).get("name") or "")
        for item in followup.get("tools", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("function"), Mapping)
        and str(item.get("function", {}).get("name") or "") != "tool_search"
    ]
    # A concrete call narrows the candidates to the tool actually chosen. On a
    # later discovery attempt, preserve that active tool and add alternatives
    # after it, still capped at three.
    # A discovery result has already made the capability decision. Retaining
    # unrelated bridge schemas from the first live request beside that answer
    # wastes the constrained model window and can prevent the chosen tool from
    # running at all. Concrete calls stay active for iterative use; an empty
    # discovery preserves the prior set because programs inside shell are not
    # separate portal tools.
    concrete = list(
        dict.fromkeys(active if active else discovered if discovered else current)
    )[:3]
    followup["tools"] = copy.deepcopy(
        [*DISCOVERY_TOOLS, *tool_schemas(concrete)]
    )
    return followup, executed, made_progress


def _tool_round_productive(executed: list[dict[str, Any]]) -> bool:
    """Whether a round returned usable evidence or completed an action."""

    for item in executed:
        if item.get("duplicate") or item.get("ok") is not True:
            continue
        if str(item.get("name") or "") == "tool_search":
            continue
        try:
            result = json.loads(str(item.get("result") or "{}"))
        except ValueError:
            return True
        if not isinstance(result, Mapping):
            return True
        if result.get("found") is False:
            continue
        if "results" in result and not result.get("results"):
            continue
        return True
    return False


def _tool_trace(executed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return call/result evidence suitable for the session UI."""

    return [
        {
            "id": str(item.get("id") or "")[:80],
            "name": str(item.get("name") or "unknown")[:80],
            "arguments": dict(item.get("arguments") or {}),
            "ok": item.get("ok") is True,
            "status": str(item.get("status") or "complete"),
            "result": str(item.get("result") or ""),
        }
        for item in executed
    ]


def _tool_start_trace(calls: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for call in calls:
        function = call.get("function")
        name = str(function.get("name") or "unknown") if isinstance(function, Mapping) else "unknown"
        arguments = _tool_arguments(call)
        fingerprint = hashlib.sha256(
            f"{name}\0{json.dumps(arguments, sort_keys=True, default=str)}".encode()
        ).hexdigest()
        trace.append(
            {
                "id": str(call.get("id") or fingerprint[:12])[:80],
                "name": name,
                "arguments": {
                    key: copy.deepcopy(value)
                    for key, value in arguments.items()
                    if key
                    in {
                        "query",
                        "url",
                        "topic",
                        "key",
                        "mode",
                        "num_results",
                        "max_results",
                        "max_length",
                        "objective",
                        "completion_criteria",
                        "action",
                        "guidance",
                        "role",
                        "task_id",
                    }
                },
                "ok": False,
                "status": "running",
                "result": "",
            }
        )
    return trace


def _record_media_diagnostics(
    diagnostics: _SessionDiagnostics,
    session_id: str,
    request_id: str,
    media_ids: list[str],
) -> None:
    """Record request-local media identities without retaining media or descriptions."""

    for media_id in media_ids[:8]:
        diagnostics.record(
            session_id,
            "media_observed",
            {"request_id": request_id, "media_id": media_id},
            request_id=request_id,
        )


def _record_tool_diagnostics(
    diagnostics: _SessionDiagnostics,
    session_id: str,
    request_id: str,
    round_index: int,
    phase: str,
    items: list[Mapping[str, Any]],
) -> None:
    """Record tool name/status only; arguments and results remain outside diagnostics."""

    for item in items:
        if phase == "started":
            function = item.get("function")
            name = (
                str(function.get("name") or "unknown")
                if isinstance(function, Mapping)
                else "unknown"
            )
            fields: dict[str, Any] = {
                "request_id": request_id,
                "tool_name": name,
                "tool_round": round_index,
            }
        else:
            fields = {
                "request_id": request_id,
                "tool_name": str(item.get("name") or "unknown"),
                "tool_round": round_index,
                "tool_ok": item.get("ok") is True,
            }
        diagnostics.record(
            session_id,
            f"tool_call_{phase}",
            fields,
            request_id=request_id,
        )


def _probe(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url, timeout=5)
        return {"ok": response.status_code < 400, "status": response.status_code}
    except httpx.HTTPError:
        return {"ok": False, "status": None}


def create_app(
    config: PortalConfig | None = None,
    client: httpx.Client | None = None,
    web_client: httpx.Client | None = None,
    web_browser_runner: Any | None = None,
    decision_plane: DecisionPlane | None = None,
) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        static_folder=str(root / "static"),
        static_url_path="/assets",
        template_folder=str(root / "templates"),
    )
    runtime = config or PortalConfig.from_environment()
    session = client or httpx.Client(timeout=runtime.timeout_s)
    inference_queue = _InferenceQueue(
        slots=runtime.inference_slots,
        max_inflight=runtime.max_inflight_requests,
    )
    diagnostics = _SessionDiagnostics(
        directory=runtime.session_log_dir,
        ttl_s=runtime.session_log_ttl_s,
    )
    documents = SessionDocumentStore(ttl_s=runtime.session_log_ttl_s)
    plane = decision_plane
    if plane is None and runtime.decision_plane_enabled:
        plane = DecisionPlane.from_environment()
    decision_executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="omni-decision-shadow")
        if plane is not None
        else None
    )

    def observe_decision_wave(
        wave: str,
        state: DecisionState,
        *,
        session_id: str = "",
        request_id: str = "",
        wait: bool = False,
    ) -> DecisionWaveResult | None:
        if plane is None:
            return None

        def run() -> DecisionWaveResult:
            result = plane.evaluate(state=state, wave=wave)
            if session_id:
                for name, item in result.results.items():
                    diagnostics.record(
                        session_id,
                        "decision",
                        {
                            "request_id": request_id,
                            "decision_wave": wave,
                            "decision_id": name,
                            "decision_value": str(item.value),
                            "decision_confidence": item.confidence,
                            "decision_latency_ms": item.latency_ms,
                            "decision_band": item.band,
                            "decision_source": item.source,
                            "decision_checkpoint": item.checkpoint,
                            "decision_shadow": item.shadow,
                            "decision_escalated": item.escalated,
                            "decision_fast_path": item.fast_path_taken,
                        },
                        request_id=request_id,
                    )
            return result

        if wait or not plane.shadow_mode or decision_executor is None:
            return run()
        decision_executor.submit(run)
        return None

    def route_input_tools(
        payload: Mapping[str, Any], *, session_id: str, request_id: str
    ) -> list[str]:
        diagnostics_fields = _request_diagnostic_fields(payload)
        if any(
            diagnostics_fields.get(name) is True
            for name in ("has_audio_input", "has_image_input", "has_video_input")
        ):
            # Before comprehension, embodied requests contain only a transport
            # description (for example, "the attached audio...") rather than
            # the speaker's words. Routing that placeholder wastes a shadow
            # wave and can occupy the single resident Laya worker exactly when
            # the adapter submits the useful post-comprehension transcript.
            return []
        messages = payload.get("messages")
        text = _latest_user_context(messages if isinstance(messages, list) else [])
        if not text:
            return []
        result = observe_decision_wave(
            "input_routing",
            DecisionState(
                user_request=text,
                current_goal=text,
                current_phase="input_routing",
                available_tool_families=tuple(
                    str(name) for name in (plane.config.get("tool_families", {}) if plane else {})
                ),
                authorization_state={"portal_tools_opted_in": True},
            ),
            session_id=session_id,
            request_id=request_id,
            wait=bool(plane is not None and not plane.shadow_mode),
        )
        if result is None:
            return []
        family = result.results.get("tool_family")
        if family is None or not family.fast_path_taken:
            return []
        family_map = plane.config.get("tool_families", {}) if plane is not None else {}
        members = family_map.get(str(family.value), []) if isinstance(family_map, Mapping) else []
        candidates = {str(item) for item in members}
        # Agreement with the existing independent ranker makes this a
        # reversible schema optimization, not unilateral model tool choice.
        return [name for name in discover_tool_names(text) if name in candidates][:3]

    def run_subagent(objective: str, role: str, context: str) -> Mapping[str, Any]:
        system_content = context_text("prompts", "subagent_system").format(
            role=role
        )
        user_content = f"<objective>\n{objective}\n</objective>"
        if context:
            user_content += f"\n\n<delegated_context>\n{context}\n</delegated_context>"
        delegated_payload = {
            "model": runtime.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "omni": {"schema": ADAPTER_SCHEMA, "task": "chat"},
            "response_modalities": ["text"],
            "speech_mode": "never",
            "think": False,
            "stream": False,
        }
        delegated_response = session.post(runtime.adapter_url, json=delegated_payload)
        delegated_data = _json_object(delegated_response, "sub-agent adapter")
        if delegated_response.status_code >= 400:
            detail = str(delegated_data.get("error") or "sub-agent request failed")
            raise ToolInputError(detail[:500])
        delegated_message = delegated_data.get("message")
        if not isinstance(delegated_message, Mapping):
            raise ToolInputError("sub-agent adapter returned no assistant message")
        content = str(delegated_message.get("content") or "").strip()
        if not content:
            raise ToolInputError("sub-agent returned an empty result")
        return {
            "content": content,
            "model": str(delegated_data.get("model") or runtime.model),
            "provenance": {
                "source_type": "isolated_model_completion",
                "role": role,
                "tools_available": False,
                "media_available": False,
                "reasoning_enabled": False,
            },
            "claim_limits": (
                "This is model-generated analysis, not independent external evidence. "
                "The parent must verify claims with source-bearing tools when needed."
            ),
        }

    tool_harness = PortalToolHarness(
        documents,
        ttl_s=runtime.session_log_ttl_s,
        web_client=web_client,
        browser_runner=web_browser_runner,
        subagent_runner=run_subagent,
        background_tasks=(
            BackgroundTaskStore(runtime.background_task_path)
            if runtime.background_task_path is not None
            else None
        ),
        memory_governor=(
            MemoryGovernor(runtime.memory_policy)
            if runtime.memory_policy is not None
            else None
        ),
    )
    app.config["MAX_CONTENT_LENGTH"] = runtime.max_body_bytes

    def authorized() -> bool:
        value = request.headers.get("Authorization", "")
        supplied = value[7:].strip() if value.lower().startswith("bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied, runtime.access_token)

    def request_session_id() -> str:
        supplied = str(request.cookies.get(SESSION_COOKIE_NAME) or "").strip()
        if 16 <= len(supplied) <= 128 and all(
            character.isalnum() or character in {"-", "_"} for character in supplied
        ):
            return supplied
        return secrets.token_urlsafe(24)

    def apply_voice_profile(payload: dict[str, Any]) -> None:
        client_voice = _voice_override(payload.pop("portal_voice", None))
        speech = {
            key: copy.deepcopy(value)
            for key, value in runtime.voice_profile.items()
            if key in VOICE_SPEECH_FIELDS
        }
        clone_enabled = client_voice.pop("clone_enabled", None)
        speaker_audio = client_voice.pop("speaker_audio", None)
        preset_id = client_voice.pop("preset", None)
        if clone_enabled is False:
            speech.pop("speaker_file", None)
        elif clone_enabled is True:
            if speaker_audio is None and not speech.get("speaker_file"):
                raise PortalRequestError(
                    "voice clone is enabled but no reference audio is configured"
                )
            if speaker_audio is not None:
                speech.pop("speaker_file", None)
                speech["speaker_audio"] = speaker_audio
            elif preset_id is not None:
                presets = {
                    str(preset["id"]): preset for preset in runtime.voice_profile.get("presets", [])
                }
                selected = presets.get(preset_id)
                if selected is None:
                    raise PortalRequestError(f"unknown voice preset: {preset_id}")
                speech["speaker_file"] = str(selected["speaker_file"])
        speech.update(client_voice)
        payload.pop("speech", None)
        if speech:
            payload["speech"] = speech

    def apply_reasoning_mode(payload: dict[str, Any]) -> None:
        think = payload.get("think", False)
        if not isinstance(think, bool):
            raise PortalRequestError("portal think must be a boolean")
        payload["think"] = think

    def apply_client_location(
        payload: dict[str, Any], session_id: str, *, tools_enabled: bool
    ) -> None:
        location = payload.pop("portal_client_location", None)
        if location is None or not tools_enabled:
            return
        try:
            tool_harness.set_client_location(session_id, location)
        except ToolInputError as exc:
            raise PortalRequestError(str(exc)) from exc

    def apply_system_policy(payload: dict[str, Any], *, tools_enabled: bool) -> None:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise PortalRequestError("messages must be a non-empty array")
        environment = portal_behavior_system_message()
        environment["content"] += f"\n\n{TOOL_RESULT_POLICY}"
        if tools_enabled:
            environment["content"] += f"\n\n{tool_use_instructions()}"
        if isinstance(messages[0], dict) and messages[0].get("role") == "system":
            existing = str(messages[0].get("content") or "").strip()
            messages[0]["content"] = (
                f"{existing}\n\n{environment['content']}" if existing else environment["content"]
            )
        else:
            messages.insert(0, environment)

    def apply_document_context(payload: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise PortalRequestError("messages must be a non-empty array")
        last_user: dict[str, Any] | None = None
        uploads: list[Any] = []
        for raw_message in messages:
            if not isinstance(raw_message, dict):
                continue
            raw_documents = raw_message.pop("documents", [])
            if raw_documents:
                if raw_message.get("role") != "user":
                    raise PortalRequestError("documents are accepted only on user messages")
                if not isinstance(raw_documents, list):
                    raise PortalRequestError("message.documents must be an array")
                uploads.extend(raw_documents)
            if raw_message.get("role") == "user":
                last_user = raw_message
        if last_user is None:
            raise PortalRequestError("messages must contain a user message")
        query = str(last_user.get("content") or "").strip()
        try:
            context, accepted = documents.prepare(session_id, uploads, query)
        except DocumentError as exc:
            raise PortalRequestError(str(exc)) from exc
        if context:
            last_user["content"] = f"{context}\n\n<user_request>\n{query}\n</user_request>"
        return accepted

    @app.after_request
    def secure_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' data: blob:; connect-src 'self' https://ipwho.is; "
            "script-src 'self'; style-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if (
            request.path.startswith("/api/")
            or request.path.startswith("/assets/")
            or request.path == "/"
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "request exceeds the portal upload limit"}), 413

    @app.get("/")
    def index():
        browser_session = request_session_id()
        response = make_response(
            render_template(
                "index.html",
                model=runtime.model,
                max_upload_mib=runtime.max_body_bytes // (1024 * 1024),
                session_scope=hashlib.sha256(
                    f"robit-omni-browser-cache:{browser_session}".encode()
                ).hexdigest(),
                browser_context={
                    "live_call_system": context_text(
                        "prompts", "live_call_system"
                    ),
                    "media_conversation_system": context_text(
                        "prompts", "media_conversation_system"
                    ),
                },
            )
        )
        response.set_cookie(
            SESSION_COOKIE_NAME,
            browser_session,
            max_age=24 * 60 * 60,
            secure=True,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify(
            {
                "ok": True,
                "service": "robit-omni-portal",
                "decision_plane": plane.health() if plane is not None else {"enabled": False},
            }
        )

    @app.get("/api/status")
    def status():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        session_id = request_session_id()
        diagnostics.touch(session_id)
        stages = {
            "adapter": _probe(session, runtime.adapter_health_url),
            "tts": _probe(session, runtime.tts_health_url),
            "ollama": _probe(session, runtime.ollama_health_url),
        }
        decision_health = plane.health() if plane is not None else {"enabled": False}
        if runtime.comprehension_health_url:
            stages["comprehension"] = _probe(session, runtime.comprehension_health_url)
        else:
            # A deployment can intentionally run without the comprehension
            # worker -- it is by far the largest component, and a host that
            # only needs language and speech should not be reported unhealthy
            # for declining to load it. Media routes still fail loudly when
            # asked for; see _require_comprehension in the adapter.
            stages["comprehension"] = {"ok": True, "status": None, "enabled": False}
        return jsonify(
            {
                "ok": all(item["ok"] for item in stages.values()),
                "model": runtime.model,
                "schema": ADAPTER_SCHEMA,
                "stages": stages,
                "safe_tools": SAFE_TOOLS,
                "tool_execution": {
                    "automatic": True,
                    "streaming": True,
                    "client_opt_in": True,
                    "default_enabled": False,
                    "round_limit": None,
                    "call_limit": None,
                    "termination": [
                        "model_final",
                        "repeated_nonproductive_rounds",
                        "request_timeout",
                        "client_disconnect",
                    ],
                },
                "location": {
                    "scope": "browser_session",
                    "delivery": "get_user_location tool",
                    "source": "browser HTTPS IP geolocation",
                    "precision": "approximate",
                    "raw_ip_retained": False,
                    **tool_harness.location_stats(session_id),
                },
                "documents": {
                    "supported": ["pdf", "docx", "utf-8 text/code"],
                    "retrieval": "session-isolated hashed lexical embeddings",
                    **documents.stats(session_id),
                },
                "memory": {
                    "scope": "browser_session",
                    "ttl_seconds": runtime.session_log_ttl_s,
                    **tool_harness.memory_stats(session_id),
                },
                "web": {
                    "discovery": "local_chromium",
                    "search_api": False,
                    "index_scope": "browser_session",
                    **tool_harness.web_stats(session_id),
                },
                "workspace": {
                    "scope": "browser_session",
                    **tool_harness.workspace_stats(session_id),
                },
                "subagents": {
                    "scope": "browser_session",
                    "execution": "synchronous_isolated_text_only",
                    "tools_available_to_helper": False,
                    **tool_harness.subagent_stats(session_id),
                },
                "voice_profile": {
                    "name": str(runtime.voice_profile.get("name") or "default"),
                    "language": str(runtime.voice_profile.get("language") or "en"),
                    "speaker_reference": bool(runtime.voice_profile.get("speaker_file")),
                    "temperature": float(runtime.voice_profile.get("temperature", 0.7)),
                    "top_k": int(runtime.voice_profile.get("top_k", 40)),
                    "top_p": float(runtime.voice_profile.get("top_p", 0.9)),
                    "seed": int(runtime.voice_profile.get("seed", 42)),
                    "max_frames": int(runtime.voice_profile.get("max_frames", 512)),
                    "clone_mode": "speaker_embedding",
                    "client_reference_wav": True,
                    "presets": [
                        {
                            "id": str(preset["id"]),
                            "label": str(preset["label"]),
                            "default": bool(preset["default"]),
                        }
                        for preset in runtime.voice_profile.get("presets", [])
                    ],
                },
                "streaming": {
                    "text": True,
                    "audio": True,
                    "audio_transport": "pcm_s16le_deltas_with_final_wav",
                    "barge_in": True,
                },
                "audio_understanding": {
                    "speech_transcription": True,
                    "environmental_sound_analysis": True,
                    "evidence_field": "adapter.audio_observation",
                },
                "runtime_environment": {
                    "delivery": "tool_only",
                    "tool": "get_system_snapshot",
                    "refreshed_each_call": True,
                    "includes": [
                        "date/time",
                        "CPU/load",
                        "RAM",
                        "NVIDIA GPUs",
                        "network counters",
                    ],
                    "excludes": [
                        "hostnames",
                        "addresses",
                        "processes",
                        "credentials",
                        "session content",
                    ],
                },
                "requests": inference_queue.snapshot(),
                "decision_plane": {
                    **decision_health,
                    "metrics": plane.metrics() if plane is not None else {},
                },
            }
        )

    @app.get("/api/activity")
    def activity():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        diagnostics.touch(request_session_id())
        return jsonify(inference_queue.snapshot())

    @app.route("/api/diagnostics", methods=["GET", "POST", "DELETE"])
    def session_diagnostics():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        session_id = request_session_id()
        if request.method == "DELETE":
            diagnostics.clear(session_id)
            documents.clear(session_id)
            tool_harness.clear(session_id)
            return Response(status=204)
        if request.method == "GET":
            return jsonify(diagnostics.snapshot(session_id))
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "diagnostic event must be an object"}), 400
        event = str(payload.get("event") or "").strip()
        if event not in {"client_stream_timing", "page_leave"}:
            return jsonify({"error": "unsupported diagnostic event"}), 400
        request_id = str(payload.get("request_id") or "").strip() or None
        accepted = diagnostics.record(
            session_id,
            event,
            payload,
            request_id=request_id,
        )
        return jsonify({"accepted": accepted})

    @app.get("/api/tools")
    def list_tools():
        """Advertise the tools this adapter can execute itself."""

        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        return jsonify({"schema": ADAPTER_SCHEMA, "tools": copy.deepcopy(SAFE_TOOLS)})

    @app.post("/api/tools/<name>/call")
    def call_tool(name: str):
        """Run one tool and return its result.

        The portal already executes these inside its own agentic loop, but a
        caller that drives its own loop -- an embodied runtime mixing these
        with tools of its own, say -- needs to run a single one and keep
        control of the conversation. Without this, such a caller has to keep
        a second service alive purely to execute tools, which is the whole
        cost the single weights package is meant to remove.
        """

        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        known = {item["function"]["name"] for item in SAFE_TOOLS}
        if name not in known:
            return jsonify({"error": f"unknown tool: {name}"}), 404
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            return jsonify({"error": "request body must be a JSON object"}), 400
        arguments = payload.get("arguments", payload)
        if not isinstance(arguments, Mapping):
            return jsonify({"error": "arguments must be an object"}), 400
        session_id = request_session_id()
        started = time.monotonic()
        result = tool_harness.execute(session_id, name, arguments)
        return jsonify(
            {
                "schema": ADAPTER_SCHEMA,
                "tool": name,
                "result": result,
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        )

    @app.post("/api/chat")
    def chat():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        ticket: _InferenceTicket | None = None
        session_id = request_session_id()
        request_id = secrets.token_urlsafe(9)
        started = time.monotonic()
        request_logged = False
        outcome_status = 500
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            auto_tools = payload.pop("portal_auto_tools", False) is True
            camera_bridge = payload.pop("portal_camera_bridge", False) is True
            shell_bridge = payload.pop("portal_shell_bridge", False) is True
            background_bridge = payload.pop("portal_background_bridge", False) is True
            payload.pop("portal_require_tool_decision", None)
            if payload.get("model") != runtime.model:
                return jsonify({"error": "portal model tag is fixed"}), 400
            if payload.get("stream") is not False:
                return jsonify({"error": "portal requires stream=false"}), 400
            diagnostic_fields = _request_diagnostic_fields(payload)
            diagnostic_fields["tools_requested"] = auto_tools
            diagnostic_media_ids = _request_media_digests(payload)
            apply_client_location(payload, session_id, tools_enabled=auto_tools)
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)
            apply_system_policy(payload, tools_enabled=auto_tools)
            observed_media = tool_harness.observe_request(session_id, payload) if auto_tools else []
            accepted_documents = apply_document_context(payload, session_id)
            diagnostics.begin_request(
                session_id,
                request_id,
                diagnostic_fields,
            )
            _record_media_diagnostics(
                diagnostics, session_id, request_id, diagnostic_media_ids
            )
            request_logged = True
            if auto_tools:
                routed_tools = route_input_tools(
                    payload, session_id=session_id, request_id=request_id
                )
                initial_tools = [*DISCOVERY_TOOLS, *tool_schemas(routed_tools)]
                if camera_bridge:
                    initial_tools.extend(tool_schemas(["request_camera_view"]))
                if shell_bridge:
                    initial_tools.extend(tool_schemas(["shell"]))
                if background_bridge:
                    initial_tools.extend(tool_schemas(["background_task"]))
                payload["tools"] = copy.deepcopy(
                    list(
                        {
                            str(item.get("function", {}).get("name") or ""): item
                            for item in initial_tools
                            if isinstance(item, Mapping)
                        }.values()
                    )
                )
            queue_started = time.monotonic()
            ticket = inference_queue.acquire(session_id, runtime.timeout_s)
            if ticket is None:
                outcome_status = 503
                return jsonify({"error": "inference queue is full or timed out"}), 503
            diagnostics.record(
                session_id,
                "queue_acquired",
                {
                    "request_id": request_id,
                    "queue_wait_ms": (time.monotonic() - queue_started) * 1000,
                },
                request_id=request_id,
            )

            executed: list[dict[str, Any]] = []
            seen_tool_calls: set[str] = set()
            current_payload: dict[str, Any] = payload
            round_index = 0
            stalled_rounds = 0
            while True:
                upstream = session.post(runtime.adapter_url, json=current_payload)
                data = _json_object(upstream, "adapter")
                if upstream.status_code >= 400:
                    outcome_status = upstream.status_code
                    response = jsonify(data)
                    response.headers["X-Omni-Request-ID"] = request_id
                    return response, upstream.status_code
                if not auto_tools:
                    break
                calls = _response_tool_calls(data)
                if calls:
                    round_index += 1
                    _record_tool_diagnostics(
                        diagnostics,
                        session_id,
                        request_id,
                        round_index,
                        "started",
                        calls,
                    )
                followup, round_tools, _made_progress = _tool_followup(
                    current_payload,
                    data,
                    tool_harness,
                    session_id,
                    seen_tool_calls,
                    {"shell"} if background_bridge and not shell_bridge else set(),
                    decision_observer=lambda wave, state: observe_decision_wave(
                        wave,
                        state,
                        session_id=session_id,
                        request_id=request_id,
                    ),
                )
                if followup is None:
                    break
                if _tool_round_productive(round_tools):
                    stalled_rounds = 0
                else:
                    stalled_rounds += 1
                if stalled_rounds >= MAX_STALLED_TOOL_ROUNDS:
                    raise PortalError(
                        "safe tool loop stopped after repeated rounds without actionable progress"
                    )
                executed.extend(round_tools)
                _record_tool_diagnostics(
                    diagnostics,
                    session_id,
                    request_id,
                    round_index,
                    "completed",
                    round_tools,
                )
                current_payload = followup

            data["portal"] = {
                "schema": "robit.omni-phone-portal.v1",
                "safe_tools_executed": _tool_trace(executed),
                "documents_indexed": accepted_documents,
                "media_observed": observed_media,
            }
            outcome_status = 200
            response = jsonify(data)
            response.headers["X-Omni-Request-ID"] = request_id
            return response
        except PortalRequestError as exc:
            outcome_status = 400
            return jsonify({"error": str(exc)}), 400
        except (httpx.HTTPError, PortalError) as exc:
            outcome_status = 502
            return jsonify({"error": str(exc)}), 502
        finally:
            if ticket is not None:
                inference_queue.release(ticket)
            if request_logged:
                diagnostics.record(
                    session_id,
                    "request_complete",
                    {
                        "request_id": request_id,
                        "status": outcome_status,
                        "total_ms": (time.monotonic() - started) * 1000,
                    },
                    request_id=request_id,
                )

    @app.post("/api/chat/stream")
    def chat_stream():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        auto_tools = payload.pop("portal_auto_tools", False) is True
        camera_bridge = payload.pop("portal_camera_bridge", False) is True
        shell_bridge = payload.pop("portal_shell_bridge", False) is True
        background_bridge = payload.pop("portal_background_bridge", False) is True
        payload.pop("portal_require_tool_decision", None)
        if payload.get("model") != runtime.model:
            return jsonify({"error": "portal model tag is fixed"}), 400
        if payload.get("stream") is not True:
            return jsonify({"error": "stream endpoint requires stream=true"}), 400
        session_id = request_session_id()
        request_id = secrets.token_urlsafe(9)
        started = time.monotonic()
        diagnostic_fields = _request_diagnostic_fields(payload)
        diagnostic_fields["tools_requested"] = auto_tools
        diagnostic_media_ids = _request_media_digests(payload)
        try:
            apply_client_location(payload, session_id, tools_enabled=auto_tools)
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)
            apply_system_policy(payload, tools_enabled=auto_tools)
            observed_media = tool_harness.observe_request(session_id, payload) if auto_tools else []
            accepted_documents = apply_document_context(payload, session_id)
            diagnostics.begin_request(
                session_id,
                request_id,
                diagnostic_fields,
            )
            _record_media_diagnostics(
                diagnostics, session_id, request_id, diagnostic_media_ids
            )
            if auto_tools:
                routed_tools = route_input_tools(
                    payload, session_id=session_id, request_id=request_id
                )
                initial_tools = [*DISCOVERY_TOOLS, *tool_schemas(routed_tools)]
                if camera_bridge:
                    initial_tools.extend(tool_schemas(["request_camera_view"]))
                if shell_bridge:
                    initial_tools.extend(tool_schemas(["shell"]))
                if background_bridge:
                    initial_tools.extend(tool_schemas(["background_task"]))
                payload["tools"] = copy.deepcopy(
                    list(
                        {
                            str(item.get("function", {}).get("name") or ""): item
                            for item in initial_tools
                            if isinstance(item, Mapping)
                        }.values()
                    )
                )
        except PortalRequestError as exc:
            return jsonify({"error": str(exc)}), 400
        queue_started = time.monotonic()
        ticket = inference_queue.acquire(session_id, runtime.timeout_s)
        if ticket is None:
            diagnostics.record(
                session_id,
                "request_complete",
                {
                    "request_id": request_id,
                    "status": 503,
                    "total_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
            return jsonify({"error": "inference queue is full or timed out"}), 503
        diagnostics.record(
            session_id,
            "queue_acquired",
            {
                "request_id": request_id,
                "queue_wait_ms": (time.monotonic() - queue_started) * 1000,
            },
            request_id=request_id,
        )

        try:
            upstream_request = session.build_request(
                "POST", runtime.adapter_url.rstrip("/") + "/stream", json=payload
            )
            upstream = session.send(upstream_request, stream=True)
            diagnostics.record(
                session_id,
                "upstream_headers",
                {
                    "request_id": request_id,
                    "status": upstream.status_code,
                    "upstream_headers_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
        except httpx.HTTPError as exc:
            inference_queue.release(ticket)
            diagnostics.record(
                session_id,
                "request_complete",
                {
                    "request_id": request_id,
                    "status": 502,
                    "outcome": "upstream_error",
                    "total_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
            return jsonify({"error": str(exc)}), 502

        def relay():
            first_byte = True
            current_upstream = upstream
            current_payload = payload
            executed: list[dict[str, Any]] = []
            seen_tool_calls: set[str] = set()
            final_status = upstream.status_code
            stalled_rounds = 0

            def event_bytes(event: Mapping[str, Any]) -> bytes:
                return (json.dumps(event, separators=(",", ":")) + "\n").encode()

            try:
                if current_upstream.status_code >= 400:
                    yield from current_upstream.iter_bytes()
                    return
                round_index = 0
                while True:
                    final_response: dict[str, Any] | None = None
                    stream_failed = False
                    for line in current_upstream.iter_lines():
                        if first_byte:
                            first_byte = False
                            diagnostics.record(
                                session_id,
                                "first_upstream_byte",
                                {
                                    "request_id": request_id,
                                    "first_upstream_byte_ms": (
                                        time.monotonic() - started
                                    )
                                    * 1000,
                                },
                                request_id=request_id,
                            )
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            yield event_bytes(
                                {"type": "error", "error": "adapter returned an invalid stream event"}
                            )
                            stream_failed = True
                            break
                        if not isinstance(event, Mapping):
                            yield event_bytes(
                                {"type": "error", "error": "adapter returned a non-object stream event"}
                            )
                            stream_failed = True
                            break
                        if event.get("type") == "final":
                            candidate = event.get("response")
                            if isinstance(candidate, dict):
                                final_response = candidate
                            continue
                        yield event_bytes(event)
                        if event.get("type") == "error":
                            stream_failed = True
                            break
                    current_upstream.close()
                    if stream_failed:
                        return
                    if final_response is None:
                        yield event_bytes(
                            {"type": "error", "error": "adapter stream ended without a final response"}
                        )
                        return

                    if not auto_tools:
                        followup = None
                        round_tools: list[dict[str, Any]] = []
                        _made_progress = False
                    else:
                        calls = _response_tool_calls(final_response)
                        if calls:
                            round_index += 1
                            _record_tool_diagnostics(
                                diagnostics,
                                session_id,
                                request_id,
                                round_index + 1,
                                "started",
                                calls,
                            )
                            yield event_bytes(
                                {
                                    "type": "tool",
                                    "phase": "start",
                                    "round": round_index + 1,
                                    "tools": _tool_start_trace(calls),
                                }
                            )
                        followup, round_tools, _made_progress = _tool_followup(
                            current_payload,
                            final_response,
                            tool_harness,
                            session_id,
                            seen_tool_calls,
                            {"shell"} if background_bridge and not shell_bridge else set(),
                            decision_observer=lambda wave, state: observe_decision_wave(
                                wave,
                                state,
                                session_id=session_id,
                                request_id=request_id,
                            ),
                        )
                    if followup is None:
                        final_response["portal"] = {
                            "schema": "robit.omni-phone-portal.v1",
                            "safe_tools_executed": _tool_trace(executed),
                            "documents_indexed": accepted_documents,
                            "media_observed": observed_media,
                        }
                        yield event_bytes({"type": "final", "response": final_response})
                        return
                    if _tool_round_productive(round_tools):
                        stalled_rounds = 0
                    else:
                        stalled_rounds += 1
                    if stalled_rounds >= MAX_STALLED_TOOL_ROUNDS:
                        final_status = 502
                        yield event_bytes(
                            {
                                "type": "error",
                                "error": (
                                    "safe tool loop stopped after repeated rounds "
                                    "without actionable progress"
                                ),
                            }
                        )
                        return
                    executed.extend(round_tools)
                    _record_tool_diagnostics(
                        diagnostics,
                        session_id,
                        request_id,
                        round_index + 1,
                        "completed",
                        round_tools,
                    )
                    yield event_bytes(
                        {
                            "type": "tool",
                            "phase": "complete",
                            "round": round_index + 1,
                            "tools": _tool_trace(round_tools),
                        }
                    )
                    current_payload = followup
                    try:
                        next_request = session.build_request(
                            "POST",
                            runtime.adapter_url.rstrip("/") + "/stream",
                            json=current_payload,
                        )
                        current_upstream = session.send(next_request, stream=True)
                    except httpx.HTTPError as exc:
                        final_status = 502
                        yield event_bytes({"type": "error", "error": str(exc)[:500]})
                        return
                    final_status = current_upstream.status_code
                    if current_upstream.status_code >= 400:
                        current_upstream.read()
                        yield event_bytes(
                            {
                                "type": "error",
                                "error": f"tool follow-up returned HTTP {current_upstream.status_code}: {current_upstream.text[:500]}",
                            }
                        )
                        return
            finally:
                current_upstream.close()
                inference_queue.release(ticket)
                diagnostics.record(
                    session_id,
                    "request_complete",
                    {
                        "request_id": request_id,
                        "status": final_status,
                        "total_ms": (time.monotonic() - started) * 1000,
                    },
                    request_id=request_id,
                )

        return Response(
            stream_with_context(relay()),
            status=upstream.status_code,
            content_type=upstream.headers.get(
                "content-type", "application/x-ndjson; charset=utf-8"
            ),
            headers={
                "X-Accel-Buffering": "no",
                "X-Omni-Request-ID": request_id,
            },
        )

    return app


if __name__ == "__main__":
    serve(
        create_app(),
        host=os.environ.get("OMNI_PORTAL_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_PORTAL_PORT", "8920")),
        threads=int(os.environ.get("OMNI_PORTAL_THREADS", "12")),
        channel_timeout=int(os.environ.get("OMNI_PORTAL_TIMEOUT_S", "1200")) + 60,
    )
