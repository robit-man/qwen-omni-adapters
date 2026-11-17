"""Read-only, bounded observability projection for the Omni portal.

The observatory deliberately reads existing runtime stores instead of creating a
second event pipeline.  Every returned item names its retention scope and source
so live telemetry, session journals, and durable memory are not conflated.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

OBSERVATORY_SCHEMA = "robit.omni.observatory.v1"
MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_MEMORY_TEXT = 220


class RuntimeModelReader:
    """Read existing worker inventories without starting or warming any model."""

    def __init__(self, client: Any, *, comprehension_url: str, tts_url: str, ollama_url: str):
        from urllib.parse import urlsplit, urlunsplit

        def endpoint(url: str, path: str) -> str:
            parts = urlsplit(url)
            return urlunsplit((parts.scheme, parts.netloc, path, "", "")) if url else ""

        self.client = client
        self.urls = {"comprehension": endpoint(comprehension_url, "/props"),
                     "tts": tts_url, "ollama": endpoint(ollama_url, "/api/ps")}
        self._lock = threading.Lock()
        self._sampled_at = 0.0
        self._cached: dict[str, Any] = {}

    def _get(self, url: str) -> dict[str, Any]:
        if not url:
            return {}
        try:
            response = self.client.get(url, timeout=2)
            if response.status_code != 200 or len(response.content) > 2 * 1024 * 1024:
                return {}
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - an inventory probe must not affect inference
            return {}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._cached and time.monotonic() - self._sampled_at < 5:
                return dict(self._cached)
            components: list[dict[str, Any]] = []
            props = self._get(self.urls["comprehension"])
            if props.get("model_path"):
                path = Path(str(props["model_path"]))
                entry: dict[str, Any] = {
                    "role": "Language + perception", "name": path.name[:180],
                    "state": "sleeping" if props.get("is_sleeping") is True else "loaded",
                    "source": "llama.cpp worker /props", "bytes": None,
                    "precision": _bounded_text(props.get("model_ftype"), 48),
                }
                try:
                    with path.open("rb") as artifact:
                        if artifact.read(4) == b"GGUF":
                            entry["bytes"] = path.stat().st_size
                except OSError:
                    pass
                components.append(entry)
            tts = self._get(self.urls["tts"])
            if tts:
                components.append({
                    "role": "Speech synthesis", "name": "Qwen3-TTS",
                    "state": "loaded" if tts.get("persistent_ready") is True else "released" if tts.get("ok") is True else "unavailable",
                    "policy": tts.get("residency_policy") if tts.get("residency_policy") in {"request", "persistent"} else "request",
                    "gpu_resident": tts.get("gpu_resident") if isinstance(tts.get("gpu_resident"), bool) else None,
                    "source": "TTS worker health", "bytes": None,
                })
            models = self._get(self.urls["ollama"]).get("models")
            for model in (models if isinstance(models, list) else [])[:16]:
                if not isinstance(model, Mapping):
                    continue
                components.append({
                    "role": "Ollama runner", "name": _bounded_text(model.get("name") or model.get("model"), 180),
                    "state": "loaded", "bytes": model.get("size") if isinstance(model.get("size"), int) else None,
                    "vram_bytes": model.get("size_vram") if isinstance(model.get("size_vram"), int) else None,
                    "source": "Ollama /api/ps",
                })
            self._cached = {"captured_at": _now_iso(), "components": components,
                            "loaded": sum(item["state"] == "loaded" for item in components),
                            "reported_weight_bytes": sum(item.get("bytes") or 0 for item in components if item["state"] == "loaded")}
            self._sampled_at = time.monotonic()
            return dict(self._cached)


def _diagnostic_signals(event: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Typed instrumentation only; never infer speech/vision from message words."""

    kind = event.get("event")
    if kind == "inference_stage":
        return [{"comprehension": ("perception", "Perception started"),
                 "language": ("llm", "LLM inference started"),
                 "tts": ("speaking", "Speech synthesis started")}.get(
                     event.get("stage"), ("llm", "Inference stage"))]
    if kind == "perception_observed":
        return [(signal, label) for flag, signal, label in (
            ("speech_present", "hearing", "Speech recognized"),
            ("audio_observation_present", "audio", "Sound observed"),
            ("visual_observation_present", "vision", "Visual observation received"),
        ) if event.get(flag) is True] or [("perception", "Perception completed")]
    if kind in {"audio_stream_started", "audio_stream_ended"}:
        return [("speaking", "Audio output started" if kind == "audio_stream_started" else "Audio output completed")]
    if kind == "request_received":
        return [(signal, label) for flag, signal, label in (
            ("has_audio_input", "audio", "Audio input received"),
            ("has_image_input", "vision", "Image input received"),
            ("has_video_input", "vision", "Video input received"),
            ("has_document_input", "documents", "Document input received"),
        ) if event.get(flag) is True] or [("llm", "Text request received")]
    if kind in {"tool_call_started", "tool_call_completed"}:
        return [("tools", str(kind).replace("_", " "))]
    return [("llm", str(kind or "signal").replace("_", " "))]


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
    capture_epoch = raw.get("capture_at")
    try:
        capture_age = max(0.0, time.time() - float(capture_epoch))
    except (TypeError, ValueError):
        capture_age = None
    try:
        age = max(0.0, time.time() - float(updated_epoch))
    except (TypeError, ValueError):
        age = _file_age(path)
    return {
        "available": True,
        "state": _bounded_text(raw.get("state") or "unknown", 32),
        "updated_at": _iso_from_epoch(raw.get("state_changed_at", updated_epoch)),
        "age_seconds": round(age, 1) if age is not None else None,
        "fresh": bool(age is not None and age <= 15),
        "capture_at": _iso_from_epoch(capture_epoch),
        "capture_fresh": bool(capture_age is not None and capture_age <= 15),
        "indicator_backend": _bounded_text(raw.get("indicator_backend"), 80),
    }


def _passive_memory_projection(path: Path, *, limit: int = 24, include_content: bool = False) -> dict[str, Any]:
    base: dict[str, Any] = {
        "available": False,
        "scope": "durable_host",
        "entries": 0,
        "recent": [],
        "kinds": {},
        "content_available": include_content,
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
            for row in recent_rows if include_content
        ],
    }


def _durable_memory_projection(path: Path) -> dict[str, Any]:
    """Aggregate source/index structure without exposing the owner voice archive."""

    base: dict[str, Any] = {"available": False, "scope": "device_aggregate"}
    if not path.is_file():
        return {**base, "reason": "Source archive not present"}
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.1)
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        counts = connection.execute(
            "SELECT COUNT(*) AS events, SUM(vector IS NOT NULL) AS indexed, "
            "MIN(occurred_at) AS oldest, MAX(occurred_at) AS newest FROM history_events"
        ).fetchone()
        kinds = connection.execute(
            "SELECT kind, COUNT(*) AS total FROM history_events GROUP BY kind ORDER BY total DESC LIMIT 16"
        ).fetchall()
        windows = connection.execute(
            "SELECT COUNT(*) AS total, SUM(dirty != 0) AS pending FROM history_buckets"
        ).fetchone()
        recent = connection.execute(
            "SELECT bucket, start, end, dirty, substr(source_ids,1,65537) AS source_ids "
            "FROM history_buckets ORDER BY start DESC LIMIT 16"
        ).fetchall()
        links = connection.execute(
            "SELECT l.memory_id, COUNT(*) AS sources FROM history_links l "
            "JOIN history_events e ON e.id=l.event_id JOIN memories m ON m.id=l.memory_id "
            "GROUP BY l.memory_id ORDER BY sources DESC LIMIT 64"
        ).fetchall()
    except (OSError, sqlite3.DatabaseError):
        return {**base, "reason": "Source archive unavailable"}
    finally:
        if connection is not None:
            connection.close()
    temporal = []
    for row in recent:
        try:
            members = json.loads(row["source_ids"] or "null")
        except (TypeError, ValueError):
            members = None
        temporal.append({
            "unit": _bounded_text(str(row["bucket"]).split(":", 1)[0], 16),
            "start_at": _iso_from_epoch(row["start"]), "end_at": _iso_from_epoch(row["end"]),
            "pending": bool(row["dirty"]),
            "sources": len(members) if isinstance(members, list) else None,
        })
    events, indexed = int(counts["events"] or 0), int(counts["indexed"] or 0)
    return {
        **base, "available": True, "events": events, "indexed_events": indexed,
        "unindexed_events": events - indexed,
        "oldest_at": _iso_from_epoch(counts["oldest"]), "newest_at": _iso_from_epoch(counts["newest"]),
        "kinds": {_bounded_text(row["kind"], 48): int(row["total"]) for row in kinds},
        "windows": int(windows["total"] or 0), "pending_windows": int(windows["pending"] or 0),
        "recent_windows": temporal,
        "cluster_sources": [{"memory_id": int(row["memory_id"]), "sources": int(row["sources"])} for row in links],
    }


def _task_plan_projection(task: Mapping[str, Any]) -> dict[str, Any]:
    """Expose committed controller contracts and receipts, never infer progress."""
    state = task.get("task_state") or {}
    controller = state.get("controller") or {}
    frontier = controller.get("frontier") or {}
    contracts = [dict(item) for item in frontier.get("steps", [])[:16] if isinstance(item, Mapping)]
    for field in ("last_contract", "pending_contract"):
        item = controller.get(field)
        if not isinstance(item, Mapping) or not item.get("contract_id"):
            continue
        found = next((index for index, old in enumerate(contracts) if old.get("contract_id") == item["contract_id"]), None)
        if found is None:
            contracts.append(dict(item))
        else:
            contracts[found].update(item)
    pending = controller.get("pending_contract") or {}
    nodes = []
    for item in contracts[:18]:
        evidence = [str(item[key])[:128] for key in ("evidence_id", "action_evidence_id", "verification_evidence_id") if item.get(key)]
        nodes.append({"id": str(item.get("contract_id") or "")[:128],
            "subtask": _bounded_text(item.get("subtask"), 300), "decision": _bounded_text(item.get("decision"), 24),
            "family": _bounded_text(item.get("capability_family"), 40), "operation": _bounded_text(item.get("executor_operation"), 40),
            "status": _bounded_text(item.get("status") or "planned", 40),
            "active": item.get("contract_id") == pending.get("contract_id"),
            "target": _bounded_text(item.get("effect_target"), 300),
            "expected_effect": _bounded_text(item.get("expected_effect"), 40),
            "acceptance": _bounded_text(item.get("acceptance_test"), 500),
            "verification_family": _bounded_text(item.get("verification_family"), 40),
            "selection_reason": _bounded_text(item.get("reason"), 300), "evidence_ids": evidence,
            "planned_environment_version": item.get("environment_version_at_plan"),
            "action_environment_version": item.get("environment_version_after_action")})
    ids = {node["id"] for node in nodes}
    steps = [str(item.get("contract_id") or "") for item in frontier.get("steps", [])[:16] if isinstance(item, Mapping)]
    edges = [{"from": first, "to": second, "kind": "committed successor"}
        for first, second in zip(steps, steps[1:], strict=False) if first in ids and second in ids]
    reports = [item for item in state.get("audit_reports", [])[-8:] if isinstance(item, Mapping)]
    return {"schema": "robit.omni.task-plan.v1", "task_id": _bounded_text(task.get("task_id"), 64),
        "objective": _bounded_text(task.get("objective"), 500), "status": _bounded_text(task.get("status"), 40),
        "phase": _bounded_text(controller.get("phase"), 40), "next_transition": _bounded_text(controller.get("next_transition"), 40),
        "revision": controller.get("frontier_generation", 0), "environment_version": (state.get("environment") or {}).get("version", 0),
        "nodes": nodes, "edges": edges,
        "requirements": [{"text": _bounded_text(item.get("text"), 300), "status": _bounded_text(item.get("status"), 40),
            "evidence_ids": [str(value)[:128] for value in item.get("evidence_ids", [])[:8]]}
            for item in state.get("requirements", [])[:12] if isinstance(item, Mapping)],
        "audits": [{key: item.get(key) for key in ("evidence_id", "contract_id", "executor_succeeded", "contract_satisfied", "authority", "environment_version_after", "milestone_progress") if key in item} for item in reports],
        "coverage": "Current committed frontier and bounded audit tail; no private reasoning or inferred completion"}


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
    for event in raw_events if isinstance(raw_events, list) else []:
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
        identity = hashlib.sha256(json.dumps(dict(event), sort_keys=True).encode()).hexdigest()[:20]
        for signal, label in _diagnostic_signals(event):
            items.append({
                "id": f"diagnostic-{identity}-{signal}",
                "at": event.get("at"),
                "signal": signal,
                "event": event_name,
                "category": "tools" if event_name.startswith("tool_") else "conversation",
                "label": label,
                "detail": detail,
                "source": "session diagnostic journal",
                "retention": "session · short-lived",
                "status": "error" if event.get("tool_ok") is False or "error" in event_name else "ok",
                "request_id": _bounded_text(event.get("request_id"), 128),
                "tool_round": event.get("tool_round") if isinstance(event.get("tool_round"), int) else None,
            })
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
        self._harness_state: str | None = None
        self._harness_events: deque[dict[str, Any]] = deque(maxlen=120)

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
        conversation: Mapping[str, Any] | None = None,
        owner_authorized: bool = False,
    ) -> dict[str, Any]:
        daemon_path = self.state_root / "daemon-status.json"
        harness_path = self.state_root / "harness-status.json"
        archive_path = self.state_root / "background-task-archive.log"
        memory = _passive_memory_projection(self.memory_path, include_content=owner_authorized)
        archive = _archive_projection(archive_path)
        task_list = [dict(item) for item in tasks if isinstance(item, Mapping)]
        live_tasks = [
            item
            for item in task_list
            if item.get("status") not in {"completed", "blocked", "cancelled"}
        ]
        harness = _harness_projection(harness_path)
        with self._lock:
            if harness.get("available") and harness.get("state") != self._harness_state:
                self._harness_state = harness["state"]
                self._harness_events.append({
                    "id": f"voice-{harness.get('updated_at')}-{self._harness_state}",
                    "at": harness.get("updated_at"), "category": "conversation",
                    "signal": {"hearing": "hearing", "speaking": "speaking", "thinking": "llm"}.get(self._harness_state, "presence"),
                    "label": f"Voice loop · {self._harness_state}",
                    "detail": "Observed state transition", "source": "voice-loop status",
                    "retention": "bounded observed transitions", "status": "ok",
                })
            voice_events = list(self._harness_events)
        timeline = _timeline(diagnostics, task_list, memory, archive) + voice_events
        timeline.sort(key=lambda item: str(item.get("at") or ""), reverse=True)
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
                "harness": harness,
                "activity": dict(diagnostics.get("activity") or {}),
                "services": dict(services),
                "requests": dict(requests),
                "environment": self._environment_snapshot(),
            },
            "memory": {
                "passive": memory,
                "durable": _durable_memory_projection(self.memory_path),
                "session": dict(session_memory),
                "conversation": dict(conversation or {}),
                "virtual_context": dict(virtual_context),
                "virtual_storage": _virtual_storage_projection(self.virtual_context_root),
            },
            "work": {
                "plans": [_task_plan_projection(item) for item in task_list[-24:][::-1]],
                "live": live_tasks,
                "recent": task_list[-24:][::-1],
                "archive": archive,
            },
            "location": dict(location),
            "timeline": timeline[:240],
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
