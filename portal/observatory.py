"""Read-only, bounded observability projection for the Omni portal.

The observatory deliberately reads existing runtime stores instead of creating a
second event pipeline.  Every returned item names its retention scope and source
so live telemetry, session journals, and durable memory are not conflated.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

OBSERVATORY_SCHEMA = "robit.omni.observatory.v1"
MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_MEMORY_TEXT = 220


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _iso_from_epoch(value: Any) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    try:
        return datetime.fromtimestamp(numeric).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


def _bounded_text(value: Any, limit: int = MAX_MEMORY_TEXT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)].rstrip()}…"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            payload = source.read(MAX_STATE_BYTES + 1)
    except OSError:
        return {}
    if len(payload) > MAX_STATE_BYTES:
        return {}
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _file_age(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _daemon_projection(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not raw:
        return {
            "available": False,
            "state": "unavailable",
            "source_age_seconds": _file_age(path),
        }
    children = raw.get("children")
    service_names = [
        _bounded_text(item.get("name"), 48)
        for item in (children if isinstance(children, list) else [])
        if isinstance(item, Mapping) and item.get("name")
    ]
    updated_at = _iso_from_epoch(raw.get("updated_at"))
    accelerator = raw.get("accelerator")
    accelerator = accelerator if isinstance(accelerator, Mapping) else {}
    return {
        "available": True,
        "state": _bounded_text(raw.get("state") or "unknown", 32),
        "updated_at": updated_at,
        "source_age_seconds": _file_age(path),
        "model": _bounded_text(raw.get("model"), 160),
        "services": service_names[:16],
        "comprehension_context_tokens": raw.get("comprehension_context_tokens"),
        "comprehension_context_ceiling": raw.get("comprehension_context_ceiling"),
        "platform": _bounded_text(raw.get("platform"), 48),
        "accelerator": {
            key: accelerator[key]
            for key in ("machine", "tegra", "tegra_soc", "gpu_memory_model")
            if key in accelerator
        },
    }


def _harness_projection(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not raw:
        return {
            "available": False,
            "state": "unavailable",
            "source_age_seconds": _file_age(path),
        }
    updated_epoch = raw.get("updated_at")
    try:
        age = max(0.0, time.time() - float(updated_epoch))
    except (TypeError, ValueError):
        age = _file_age(path)
    return {
        "available": True,
        "state": _bounded_text(raw.get("state") or "unknown", 32),
        "updated_at": _iso_from_epoch(updated_epoch),
        "age_seconds": round(age, 1) if age is not None else None,
        "fresh": bool(age is not None and age <= 15),
        "indicator_backend": _bounded_text(raw.get("indicator_backend"), 80),
    }


def _passive_memory_projection(path: Path, *, limit: int = 24) -> dict[str, Any]:
    base: dict[str, Any] = {
        "available": False,
        "scope": "durable_host",
        "entries": 0,
        "recent": [],
        "kinds": {},
    }
    if not path.is_file():
        return {**base, "reason": "memory store not present"}
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=0.1,
        )
        connection.row_factory = sqlite3.Row
        aggregate = connection.execute(
            "SELECT COUNT(*) AS total, AVG(strength) AS mean_strength, "
            "MIN(created_at) AS oldest, MAX(created_at) AS newest, "
            "SUM(uses) AS recalls FROM memories"
        ).fetchone()
        kind_rows = connection.execute(
            "SELECT kind, COUNT(*) AS total FROM memories GROUP BY kind "
            "ORDER BY total DESC LIMIT 16"
        ).fetchall()
        recent_rows = connection.execute(
            "SELECT id, text, kind, created_at, last_used_at, uses, strength "
            "FROM memories ORDER BY created_at DESC LIMIT ?",
            (max(1, min(64, int(limit))),),
        ).fetchall()
    except (OSError, sqlite3.DatabaseError) as exc:
        return {**base, "reason": f"memory store unreadable: {type(exc).__name__}"}
    finally:
        if connection is not None:
            connection.close()
    total = int(aggregate["total"] or 0) if aggregate is not None else 0
    oldest = float(aggregate["oldest"] or 0) if aggregate is not None else 0.0
    return {
        **base,
        "available": True,
        "entries": total,
        "mean_strength": round(float(aggregate["mean_strength"] or 0.0), 3),
        "recalls": int(aggregate["recalls"] or 0),
        "oldest_at": _iso_from_epoch(oldest),
        "oldest_days": round((time.time() - oldest) / 86400.0, 1) if oldest else 0.0,
        "newest_at": _iso_from_epoch(aggregate["newest"]),
        "kinds": {str(row["kind"]): int(row["total"]) for row in kind_rows},
        "recent": [
            {
                "id": int(row["id"]),
                "text": _bounded_text(row["text"]),
                "kind": _bounded_text(row["kind"], 48),
                "created_at": _iso_from_epoch(row["created_at"]),
                "last_used_at": _iso_from_epoch(row["last_used_at"]),
                "uses": int(row["uses"] or 0),
                "strength": round(float(row["strength"] or 0.0), 3),
            }
            for row in recent_rows
        ],
    }


def _archive_projection(path: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "available": False,
        "scope": "durable_host",
        "records_observed": 0,
        "recent": [],
        "status_counts": {},
    }
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            partial = size > MAX_ARCHIVE_BYTES
            if partial:
                source.seek(max(0, size - MAX_ARCHIVE_BYTES))
                source.readline()
            text = source.read(MAX_ARCHIVE_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return {**base, "reason": "task archive not present"}

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("Task "):
            if current:
                records.append(current)
            current = {"task_id": _bounded_text(line[5:], 64)}
            continue
        if current is None:
            continue
        for prefix, key in (
            ("Status: ", "status"),
            ("Created: ", "created_at"),
            ("Updated: ", "updated_at"),
            ("Objective: ", "objective"),
            ("Result: ", "result"),
        ):
            if line.startswith(prefix):
                current[key] = _bounded_text(line[len(prefix) :], 260)
                break
    if current:
        records.append(current)
    statuses = Counter(str(item.get("status") or "unknown") for item in records)
    return {
        **base,
        "available": True,
        "records_observed": len(records),
        "status_counts": dict(statuses),
        "recent": records[-24:][::-1],
        "partial": partial,
        "coverage_note": (
            "Newest archive window only; counts are not lifetime totals."
            if partial
            else "Entire current archive file observed."
        ),
    }


def _virtual_storage_projection(root: Path) -> dict[str, Any]:
    try:
        entries = list(root.iterdir())
    except OSError:
        return {
            "available": False,
            "scope": "durable_session_corpora",
            "corpora": 0,
            "bytes": 0,
        }
    stores = [item for item in entries if item.name.endswith(".sqlite3")]
    total = 0
    for item in entries:
        if not item.name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm")):
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return {
        "available": True,
        "scope": "durable_session_corpora",
        "corpora": len(stores),
        "bytes": total,
    }


def _timeline(
    diagnostics: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    raw_events = diagnostics.get("events")
    for index, event in enumerate(raw_events if isinstance(raw_events, list) else []):
        if not isinstance(event, Mapping):
            continue
        event_name = _bounded_text(event.get("event") or "signal", 64)
        detail = ""
        if event.get("tool_name"):
            detail = _bounded_text(event.get("tool_name"), 80)
        elif event.get("task"):
            detail = _bounded_text(event.get("task"), 80)
        elif event.get("outcome"):
            detail = _bounded_text(event.get("outcome"), 80)
        items.append(
            {
                "id": f"diagnostic-{index}",
                "at": event.get("at"),
                "category": "tools" if event_name.startswith("tool_") else "conversation",
                "label": event_name.replace("_", " "),
                "detail": detail,
                "source": "session diagnostic journal",
                "retention": "session · short-lived",
                "status": "error" if "error" in event_name else "ok",
            }
        )
    for task in tasks:
        task_id = _bounded_text(task.get("task_id"), 64)
        items.append(
            {
                "id": f"task-{task_id}",
                "at": _iso_from_epoch(task.get("updated_at")),
                "category": "work",
                "label": _bounded_text(task.get("status") or "task", 48),
                "detail": _bounded_text(task.get("objective"), 180),
                "source": "background task store",
                "retention": "durable while live + bounded recent",
                "status": (
                    "error"
                    if task.get("status") in {"blocked", "cancelled"}
                    else "ok"
                ),
            }
        )
    for item in memory.get("recent", []) if isinstance(memory.get("recent"), list) else []:
        if not isinstance(item, Mapping):
            continue
        items.append(
            {
                "id": f"memory-{item.get('id')}",
                "at": item.get("created_at"),
                "category": "memory",
                "label": _bounded_text(item.get("kind") or "memory", 48),
                "detail": _bounded_text(item.get("text"), 180),
                "source": "passive semantic memory",
                "retention": "durable · relevance-decayed",
                "status": "ok",
            }
        )
    for item in archive.get("recent", []) if isinstance(archive.get("recent"), list) else []:
        if not isinstance(item, Mapping):
            continue
        items.append(
            {
                "id": f"archive-{item.get('task_id')}",
                "at": item.get("updated_at") or item.get("created_at"),
                "category": "work",
                "label": f"archived · {_bounded_text(item.get('status') or 'unknown', 32)}",
                "detail": _bounded_text(item.get("objective"), 180),
                "source": "task archive",
                "retention": "durable archive",
                "status": "error" if item.get("status") == "blocked" else "ok",
            }
        )
    items.sort(key=lambda item: str(item.get("at") or ""), reverse=True)
    return items[:120]


class ObservatoryReader:
    """Project the existing stores into one bounded, authenticated snapshot."""

    def __init__(
        self,
        *,
        state_root: Path,
        memory_path: Path,
        virtual_context_root: Path,
        environment_sampler: Callable[[], Mapping[str, Any]],
        environment_ttl_s: float = 5.0,
    ) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        self.memory_path = Path(memory_path).expanduser().resolve()
        self.virtual_context_root = Path(virtual_context_root).expanduser().resolve()
        self.environment_sampler = environment_sampler
        self.environment_ttl_s = max(1.0, environment_ttl_s)
        self._lock = threading.Lock()
        self._environment_sampled_at = 0.0
        self._environment: dict[str, Any] = {}

    def _environment_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._environment and now - self._environment_sampled_at < self.environment_ttl_s:
                return dict(self._environment)
            try:
                sampled = self.environment_sampler()
                self._environment = dict(sampled) if isinstance(sampled, Mapping) else {}
            except Exception as exc:  # noqa: BLE001 - observability is never load-bearing
                self._environment = {
                    "captured_at": _now_iso(),
                    "error": f"environment sample failed: {type(exc).__name__}",
                }
            self._environment_sampled_at = now
            return dict(self._environment)

    def snapshot(
        self,
        *,
        diagnostics: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]],
        virtual_context: Mapping[str, Any],
        session_memory: Mapping[str, Any],
        location: Mapping[str, Any],
        services: Mapping[str, Any],
        requests: Mapping[str, Any],
    ) -> dict[str, Any]:
        daemon_path = self.state_root / "daemon-status.json"
        harness_path = self.state_root / "harness-status.json"
        archive_path = self.state_root / "background-task-archive.log"
        memory = _passive_memory_projection(self.memory_path)
        archive = _archive_projection(archive_path)
        task_list = [dict(item) for item in tasks if isinstance(item, Mapping)]
        live_tasks = [
            item
            for item in task_list
            if item.get("status") not in {"completed", "blocked", "cancelled"}
        ]
        return {
            "schema": OBSERVATORY_SCHEMA,
            "captured_at": _now_iso(),
            "scope": {
                "label": "retained evidence",
                "caveat": (
                    "This view covers evidence retained by the current runtime stores; "
                    "it is not a recording of unretained audio, video, or private reasoning."
                ),
            },
            "live": {
                "daemon": _daemon_projection(daemon_path),
                "harness": _harness_projection(harness_path),
                "services": dict(services),
                "requests": dict(requests),
                "environment": self._environment_snapshot(),
            },
            "memory": {
                "passive": memory,
                "session": dict(session_memory),
                "virtual_context": dict(virtual_context),
                "virtual_storage": _virtual_storage_projection(self.virtual_context_root),
            },
            "work": {
                "live": live_tasks,
                "recent": task_list[-24:][::-1],
                "archive": archive,
            },
            "location": dict(location),
            "timeline": _timeline(diagnostics, task_list, memory, archive),
            "retention": [
                {
                    "id": "live",
                    "label": "Live state",
                    "detail": "Service, hardware, battery, network, and voice-loop state",
                    "scope": "sampled now",
                },
                {
                    "id": "session",
                    "label": "Session journal",
                    "detail": "Content-redacted request, media, decision, and tool events",
                    "scope": f"expires after {diagnostics.get('ttl_seconds', 0):g}s idle",
                },
                {
                    "id": "working",
                    "label": "Working context",
                    "detail": "Session evidence corpus and the bounded model working set",
                    "scope": "durable until explicit session clear",
                },
                {
                    "id": "memory",
                    "label": "Passive memory",
                    "detail": "Semantic memories that strengthen on recall and decay when unused",
                    "scope": "durable · relevance-decayed",
                },
                {
                    "id": "tasks",
                    "label": "Work ledger",
                    "detail": "Current background objectives, progress, outcomes, and archive",
                    "scope": "durable + bounded current set",
                },
            ],
        }
