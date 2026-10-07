from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portal.app import _SessionDiagnostics
from portal.observatory import OBSERVATORY_SCHEMA, ObservatoryReader


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
