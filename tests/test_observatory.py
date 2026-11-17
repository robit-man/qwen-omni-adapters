from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal.app import _SessionDiagnostics
from portal.observatory import (
    OBSERVATORY_SCHEMA,
    ObservatoryReader,
    RuntimeModelReader,
    _durable_memory_projection,
    _harness_projection,
    _timeline,
)
from portal.session_state import SessionContinuationStore
from portal.tools import SessionMemoryStore


def _memory_store(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE memories ("
        "id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL, "
        "vector BLOB NOT NULL, created_at REAL NOT NULL, last_used_at REAL NOT NULL, "
        "uses INTEGER NOT NULL, strength REAL NOT NULL)"
    )
    now = time.time()
    connection.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "We are building the observability dashboard.", "turn", b"[]", now, now, 2, 0.85),
    )
    connection.commit()
    connection.close()


def test_observatory_projects_existing_stores_without_sensitive_daemon_fields(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    memory = tmp_path / "memory.sqlite3"
    virtual = tmp_path / "virtual-context"
    virtual.mkdir()
    (virtual / "session.sqlite3").write_bytes(b"retained")
    _memory_store(memory)
    now = time.time()
    (state / "daemon-status.json").write_text(
        json.dumps(
            {
                "state": "ready",
                "updated_at": now,
                "model": "model:test",
                "access_url": "https://secret.invalid/#access=secret",
                "pid": 999,
                "children": [{"name": "portal", "pid": 123}],
                "accelerator": {"machine": "aarch64", "tegra": True},
            }
        )
    )
    (state / "harness-status.json").write_text(
        json.dumps(
            {
                "schema": "robit.omni-call-harness.status.v1",
                "state": "listening",
                "updated_at": now,
            }
        )
    )
    (state / "background-task-archive.log").write_text(
        "Task abc123\nStatus: completed\nCreated: 2026-10-07T09:00:00-07:00\n"
        "Updated: 2026-10-07T09:30:00-07:00\nObjective: Verify the dashboard\n"
        "Result: Complete\n\n"
    )
    reader = ObservatoryReader(
        state_root=state,
        memory_path=memory,
        virtual_context_root=virtual,
        environment_sampler=lambda: {
            "captured_at": "2026-10-07T10:00:00-07:00",
            "memory": {"used_percent": 42.0},
            "battery": {"available": True, "percentage": 88, "voltage_v": 12.4},
        },
    )

    snapshot = reader.snapshot(
        diagnostics={
            "ttl_seconds": 300,
            "events": [
                {
                    "at": "2026-10-07T10:00:00-07:00",
                    "event": "tool_call_completed",
                    "tool_name": "get_system_snapshot",
                }
            ],
        },
        tasks=[
            {
                "task_id": "live123",
                "status": "running",
                "objective": "Implement the graph",
                "updated_at": now,
            }
        ],
        virtual_context={"mode": "active", "chunks": 12},
        session_memory={"entries": 2, "chars": 48},
        location={"available": False},
        services={"adapter": {"ok": True, "status": 200}},
        requests={"active": 1, "queued": 0},
        owner_authorized=True,
    )

    assert snapshot["schema"] == OBSERVATORY_SCHEMA
    assert snapshot["live"]["daemon"]["services"] == ["portal"]
    assert "access_url" not in snapshot["live"]["daemon"]
    assert "pid" not in snapshot["live"]["daemon"]
    assert snapshot["memory"]["passive"]["entries"] == 1
    assert snapshot["memory"]["passive"]["recent"][0]["text"] == (
        "We are building the observability dashboard."
    )
    assert snapshot["memory"]["virtual_storage"]["corpora"] == 1
    assert snapshot["work"]["live"][0]["task_id"] == "live123"
    assert snapshot["work"]["archive"]["records_observed"] == 1
    assert {item["category"] for item in snapshot["timeline"]} >= {
        "tools",
        "memory",
        "work",
    }


def test_observatory_diagnostic_view_joins_active_sessions_but_remains_redacted() -> None:
    diagnostics = _SessionDiagnostics(directory=None, ttl_s=300)
    diagnostics.begin_request(
        "session-one",
        "request-one",
        {"task": "chat", "content": "must not survive"},
    )
    diagnostics.begin_request(
        "session-two",
        "request-two",
        {"task": "transcribe", "prompt": "also private"},
    )

    snapshot = diagnostics.observatory_snapshot()

    assert snapshot["active_sessions"] == 2
    assert len(snapshot["events"]) == 2
    assert {item["task"] for item in snapshot["events"]} == {"chat", "transcribe"}
    assert "must not survive" not in json.dumps(snapshot)
    assert "also private" not in json.dumps(snapshot)


def test_active_chat_stages_follow_concurrent_request_lifetimes_and_session_clear():
    diagnostics = _SessionDiagnostics(directory=None, ttl_s=300)
    diagnostics.begin_request("one", "private-id-one", {"task": "chat"})
    diagnostics.begin_request("two", "private-id-two", {"task": "chat"})
    diagnostics.record("one", "inference_stage", {
        "request_id": "private-id-one", "stage": "language", "content": "private words",
    }, request_id="private-id-one")
    activity = diagnostics.observatory_snapshot()["activity"]
    assert activity == {"source": "portal requests", "stages": {"language": 1, "queued": 1}}
    assert "private" not in json.dumps(activity)
    assert not diagnostics.record("two", "request_complete", {
        "request_id": "private-id-one",
    }, request_id="private-id-one")
    diagnostics.record("one", "request_complete", {
        "request_id": "private-id-one",
    }, request_id="private-id-one")
    assert diagnostics.observatory_snapshot()["activity"]["stages"] == {"queued": 1}
    diagnostics.clear("two")
    assert diagnostics.observatory_snapshot()["activity"]["stages"] == {}
    diagnostics.clear("one")


def test_microphone_freshness_uses_capture_heartbeat_without_moving_transition_time(tmp_path, monkeypatch):
    from harness.__main__ import HARNESS_STATUS_FILE, _write_harness_status
    monkeypatch.setattr(time, "time", lambda: 2000)
    _write_harness_status(tmp_path, state="listening", indicator_backend="test",
        state_changed_at=1000, capture_at=1999)
    status = _harness_projection(tmp_path / HARNESS_STATUS_FILE)
    assert status["fresh"] and status["capture_fresh"]
    assert status["updated_at"] != status["capture_at"]
    transition_at = status["updated_at"]
    monkeypatch.setattr(time, "time", lambda: 2020)
    stale = _harness_projection(tmp_path / HARNESS_STATUS_FILE)
    assert not stale["fresh"] and not stale["capture_fresh"]
    assert stale["updated_at"] == transition_at


def test_durable_structure_retains_sources_when_semantic_cache_is_empty(tmp_path):
    from harness.history import ConversationHistory

    path = tmp_path / "memory.sqlite3"
    _memory_store(path)
    connection = sqlite3.connect(path)
    connection.execute("DELETE FROM memories")
    connection.commit()
    connection.close()
    archive = ConversationHistory(path)
    # A real source row, with private source content deliberately absent from the projection.
    with archive._db:
        archive._db.execute(
            "INSERT INTO history_events(event_key,text,kind,occurred_at,recorded_at,source_json,content_sha256) VALUES(?,?,?,?,?,?,?)",
            ("private-key", "private voice marker", "heard", time.time(), time.time(), '{"speaker":"private-person"}', "source-hash"),
        )
    before = path.read_bytes()
    projection = _durable_memory_projection(path)
    assert projection["events"] == 1
    assert projection["unindexed_events"] == 1
    assert projection["indexed_events"] == 0
    assert projection["kinds"] == {"heard": 1}
    assert projection["cluster_sources"] == []
    assert "private" not in json.dumps(projection)
    assert "source-hash" not in json.dumps(projection)
    assert path.read_bytes() == before
    archive.close()


def test_session_inspection_is_cookie_scoped_bounded_and_cleared(tmp_path):
    notes = SessionMemoryStore(ttl_s=300)
    notes.write("one", "project", "task", "one-marker")
    notes.write("two", "project", "task", "two-marker")
    journal = SessionContinuationStore(tmp_path / "sessions")
    journal.begin("one", "private-turn-id", "request-one", "one conversation")
    journal.event("one", "private-turn-id", {"type": "delta", "message": {"content": "one reply", "thinking": "hidden reasoning"}})
    journal.begin("two", "other-turn", "request-two", "two conversation")
    first = {"notes": notes.inspection("one"), "conversation": journal.observatory_snapshot("one")}
    text = json.dumps(first)
    assert "one-marker" in text and "one conversation" in text
    assert first["conversation"]["messages"] == 2
    assert "two-marker" not in text and "two conversation" not in text
    assert "hidden reasoning" not in text and "private-turn-id" not in text
    notes.clear("one")
    journal.clear("one")
    assert notes.inspection("one")["recent"] == []
    assert journal.observatory_snapshot("one")["recent"] == []
    assert notes.inspection("two")["entries"] == 1


def test_worker_inventory_reads_actual_loaded_artifacts_without_starting_models(tmp_path):
    model = tmp_path / "worker.gguf"
    model.write_bytes(b"GGUF" + bytes(4096))
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(model), "is_sleeping": False})
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True, "persistent_ready": False, "residency_policy": "request", "gpu_resident": None})
        return httpx.Response(200, json={"models": []})

    reader = RuntimeModelReader(httpx.Client(transport=httpx.MockTransport(handle)),
        comprehension_url="http://worker/health", tts_url="http://speech/healthz", ollama_url="http://ollama/api/tags")
    snapshot = reader.snapshot()
    assert snapshot["loaded"] == 1
    assert snapshot["reported_weight_bytes"] == model.stat().st_size
    assert snapshot["components"][0]["name"] == "worker.gguf"
    assert str(tmp_path) not in json.dumps(snapshot)
    assert snapshot["components"][1]["state"] == "released"
    assert all(method == "GET" for method, _path in requests)
    assert {path for _method, path in requests} == {"/props", "/healthz", "/api/ps"}
    reader.snapshot()
    assert len(requests) == 3


def test_timeline_keeps_typed_sensory_and_inference_events_without_content():
    at = "2026-11-16T10:00:00-08:00"
    raw = [
        {"at": at, "event": "request_received", "has_audio_input": True, "has_image_input": True},
        {"at": at, "event": "perception_observed", "speech_present": True, "audio_observation_present": True, "transcript": "private words"},
        {"at": at, "event": "inference_stage", "stage": "language"},
        {"at": at, "event": "audio_stream_started"},
    ]
    result = _timeline({"events": raw}, [], {}, {})
    assert {item["signal"] for item in result} == {"audio", "vision", "hearing", "llm", "speaking"}
    assert "private words" not in json.dumps(result)
    assert {item["id"] for item in result} == {item["id"] for item in _timeline({"events": list(reversed(raw))}, [], {}, {})}


def test_noise_only_observation_never_becomes_heard_speech():
    result = _timeline({"events": [{"event": "perception_observed", "speech_present": False, "audio_observation_present": True}]}, [], {}, {})
    assert [item["signal"] for item in result] == ["audio"]


def test_plan_projection_preserves_receipt_states_and_does_not_infer_completion():
    from portal.observatory import _task_plan_projection
    task = {"task_id": "plan", "objective": "Acquire, change and verify", "status": "running", "task_state": {
        "requirements": [{"text": "Observed result must match", "status": "pending", "evidence_ids": []}],
        "environment": {"version": 7}, "controller": {"phase": "audit", "next_transition": "verify", "frontier_generation": 2,
            "frontier": {"steps": [{"contract_id": "read", "subtask": "Inspect current state", "decision": "retrieve", "status": "evidence_acquired", "evidence_id": "receipt-read"},
                {"contract_id": "write", "subtask": "Change the declared target", "decision": "act", "status": "planned"}]},
            "pending_contract": {"contract_id": "write", "status": "awaiting_audit", "action_evidence_id": "receipt-action", "capability_family": "shell", "reason": "The declared target is a file", "acceptance_test": "Exact bytes", "environment_version_after_action": 7}},
        "audit_reports": [{"contract_id": "write", "executor_succeeded": True, "contract_satisfied": False, "authority": "mutation", "private_thought": "never expose"}]}}
    plan = _task_plan_projection(task)
    assert plan["edges"] == [{"from": "read", "to": "write", "kind": "committed successor"}]
    assert plan["nodes"][1]["status"] == "awaiting_audit"
    assert plan["nodes"][1]["active"] is True
    assert plan["requirements"][0]["status"] == "pending"
    assert plan["nodes"][1]["selection_reason"] == "The declared target is a file"
    assert "never expose" not in json.dumps(plan)
