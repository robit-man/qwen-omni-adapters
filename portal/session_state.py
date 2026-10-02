"""Durable, session-isolated state for browser reconnects.

The browser keeps the rich local presentation in IndexedDB.  This store keeps
only the bounded textual/action state that can advance while the page is away,
so a returning client can merge newer work by stable turn ID and sequence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_TURNS = 48
MAX_TEXT_CHARS = 64_000
MAX_TOOL_RESULT_CHARS = 24_000


def _bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    return str(value or "")[:limit]


def _task_ids(value: Any) -> set[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            task_id = item.get("task_id")
            if isinstance(task_id, str) and task_id:
                found.add(task_id[:80])
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def _normalized_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in value[:64]:
        if not isinstance(raw, Mapping):
            continue
        arguments = raw.get("arguments")
        result.append(
            {
                "id": _bounded_text(raw.get("id"), 80),
                "name": _bounded_text(raw.get("name") or "unknown", 80),
                "arguments": copy.deepcopy(dict(arguments))
                if isinstance(arguments, Mapping)
                else {},
                "ok": raw.get("ok") is True,
                "status": (
                    "running" if raw.get("status") == "running" else "complete"
                ),
                "result": _bounded_text(raw.get("result"), MAX_TOOL_RESULT_CHARS),
            }
        )
    return result


def _merge_tools(current: list[dict[str, Any]], incoming: Any) -> list[dict[str, Any]]:
    trace = copy.deepcopy(current)
    for item in _normalized_tools(incoming):
        pending = next(
            (
                index
                for index in range(len(trace) - 1, -1, -1)
                if trace[index].get("status") == "running"
                and (
                    (item.get("id") and trace[index].get("id") == item.get("id"))
                    or (
                        not item.get("id")
                        and trace[index].get("name") == item.get("name")
                    )
                )
            ),
            -1,
        )
        if pending >= 0 and item.get("status") != "running":
            trace[pending] = item
        else:
            trace.append(item)
    return trace[-64:]


class SessionContinuationStore:
    """Crash-safe bounded turn journal keyed by the opaque portal cookie."""

    def __init__(self, directory: Path, ttl_s: float | None = None) -> None:
        self.directory = directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_s = None if ttl_s is None else max(60.0, float(ttl_s))
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._last_persist: dict[str, float] = {}

    @staticmethod
    def _key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _empty(self) -> dict[str, Any]:
        return {
            "schema": "robit.omni.browser-continuation.v1",
            "sequence": 0,
            "updated_at": time.time(),
            "turns": [],
        }

    def _load_locked(self, key: str) -> dict[str, Any]:
        existing = self._records.get(key)
        if existing is not None:
            return existing
        path = self._path(key)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            value = self._empty()
        if not isinstance(value, dict) or value.get("schema") != self._empty()["schema"]:
            value = self._empty()
        if (
            self.ttl_s is not None
            and time.time() - float(value.get("updated_at") or 0) > self.ttl_s
        ):
            path.unlink(missing_ok=True)
            value = self._empty()
        value["turns"] = [
            item for item in value.get("turns", []) if isinstance(item, dict)
        ][-MAX_TURNS:]
        self._records[key] = value
        return value

    def _persist_locked(self, key: str, record: Mapping[str, Any]) -> None:
        path = self._path(key)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._last_persist[key] = time.monotonic()

    @staticmethod
    def _turn(record: dict[str, Any], turn_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in record["turns"] if item.get("turn_id") == turn_id),
            None,
        )

    def begin(
        self,
        session_id: str,
        turn_id: str,
        request_id: str,
        user_content: str,
    ) -> int:
        key = self._key(session_id)
        with self._lock:
            record = self._load_locked(key)
            turn = self._turn(record, turn_id)
            if turn is None:
                turn = {
                    "turn_id": turn_id,
                    "request_id": request_id,
                    "sequence": 0,
                    "status": "running",
                    "stage": "queued",
                    "user": {"content": _bounded_text(user_content)},
                    "assistant": {
                        "content": "",
                        "thinking": "",
                        "tool_trace": [],
                    },
                    "task_ids": [],
                    "started_at": time.time(),
                    "updated_at": time.time(),
                }
                record["turns"].append(turn)
                record["turns"] = record["turns"][-MAX_TURNS:]
            else:
                turn["request_id"] = request_id
                turn["status"] = "running"
            record["sequence"] = int(record.get("sequence") or 0) + 1
            record["updated_at"] = time.time()
            turn["sequence"] = record["sequence"]
            turn["updated_at"] = record["updated_at"]
            self._persist_locked(key, record)
            return int(record["sequence"])

    def event(
        self,
        session_id: str,
        turn_id: str,
        event: Mapping[str, Any],
    ) -> int:
        key = self._key(session_id)
        with self._lock:
            record = self._load_locked(key)
            turn = self._turn(record, turn_id)
            if turn is None:
                return 0
            event_type = str(event.get("type") or "")
            assistant = turn.setdefault("assistant", {})
            force_persist = event_type in {
                "observation",
                "reset",
                "tool",
                "final",
                "error",
            }
            changed = True
            if event_type == "delta":
                message = event.get("message")
                message = message if isinstance(message, Mapping) else {}
                assistant["content"] = _bounded_text(
                    str(assistant.get("content") or "")
                    + str(message.get("content") or "")
                )
                assistant["thinking"] = _bounded_text(
                    str(assistant.get("thinking") or "")
                    + str(message.get("thinking") or "")
                )
            elif event_type == "observation":
                transcript = _bounded_text(event.get("transcript"))
                observation = _bounded_text(event.get("audio_observation"))
                if transcript:
                    turn["user"]["content"] = transcript
                    turn["input_transcript"] = transcript
                if observation:
                    turn["audio_observation"] = observation
            elif event_type == "reset":
                assistant["content"] = ""
                assistant["thinking"] = ""
                if event.get("reason") == "browser_network_retry":
                    assistant["tool_trace"] = []
            elif event_type == "stage":
                turn["stage"] = _bounded_text(event.get("stage"), 40)
            elif event_type == "tool":
                assistant["tool_trace"] = _merge_tools(
                    list(assistant.get("tool_trace") or []), event.get("tools")
                )
                task_ids = set(turn.get("task_ids") or [])
                for tool in assistant["tool_trace"]:
                    if tool.get("name") != "background_task":
                        continue
                    try:
                        task_ids.update(_task_ids(json.loads(tool.get("result") or "{}")))
                    except (TypeError, ValueError):
                        continue
                turn["task_ids"] = sorted(task_ids)[:32]
            elif event_type == "final":
                response = event.get("response")
                response = response if isinstance(response, Mapping) else {}
                message = response.get("message")
                message = message if isinstance(message, Mapping) else {}
                assistant["content"] = _bounded_text(message.get("content"))
                assistant["thinking"] = _bounded_text(message.get("thinking"))
                portal = response.get("portal")
                portal = portal if isinstance(portal, Mapping) else {}
                if isinstance(portal.get("safe_tools_executed"), list):
                    assistant["tool_trace"] = _normalized_tools(
                        portal.get("safe_tools_executed")
                    )
                adapter = response.get("adapter")
                adapter = adapter if isinstance(adapter, Mapping) else {}
                transcript = _bounded_text(adapter.get("input_transcript"))
                observation = _bounded_text(adapter.get("audio_observation"))
                if transcript:
                    turn["user"]["content"] = transcript
                    turn["input_transcript"] = transcript
                if observation:
                    turn["audio_observation"] = observation
                turn["status"] = "complete"
                turn["stage"] = "complete"
            elif event_type == "error":
                turn["status"] = "error"
                turn["stage"] = "error"
                turn["error"] = _bounded_text(event.get("error"), 1000)
            elif event_type in {"audio_start", "audio_delta", "audio_end"}:
                changed = False
            else:
                changed = False
            if not changed:
                return int(record.get("sequence") or 0)
            record["sequence"] = int(record.get("sequence") or 0) + 1
            record["updated_at"] = time.time()
            turn["sequence"] = record["sequence"]
            turn["updated_at"] = record["updated_at"]
            if force_persist or time.monotonic() - self._last_persist.get(key, 0) >= 0.5:
                self._persist_locked(key, record)
            return int(record["sequence"])

    def snapshot(self, session_id: str, after: int = 0) -> dict[str, Any]:
        key = self._key(session_id)
        with self._lock:
            record = self._load_locked(key)
            turns = [
                copy.deepcopy(item)
                for item in record["turns"]
                if int(item.get("sequence") or 0) > max(0, int(after))
            ]
            task_ids = sorted(
                {
                    str(task_id)
                    for item in record["turns"]
                    for task_id in item.get("task_ids", [])
                    if task_id
                }
            )
            return {
                "schema": record["schema"],
                "sequence": int(record.get("sequence") or 0),
                "turns": turns,
                "task_ids": task_ids,
            }

    def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            self._records.pop(key, None)
            self._last_persist.pop(key, None)
            self._path(key).unlink(missing_ok=True)
            for temporary in self.directory.glob(f"{key}.*.tmp"):
                temporary.unlink(missing_ok=True)
