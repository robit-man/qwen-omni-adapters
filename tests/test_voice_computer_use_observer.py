from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from runtime.verify_voice_computer_use import (
    contains_marker,
    journal_summary,
    memory_summary,
    session_keys,
    task_summary,
)


def _record(message: str, *, stamp: int = 20, pid: int = 42) -> str:
    return json.dumps({"__REALTIME_TIMESTAMP": stamp * 1_000_000, "_PID": str(pid), "MESSAGE": message})


def _trace(event: str, text: str, **extra) -> str:
    return "conversation_trace " + json.dumps({"event": event, "text": text, **extra})


def test_observer_counts_only_full_live_asr_for_the_declared_marker_and_process() -> None:
    lines = [
        _record("turn: heard='Copper Finch truncated text' reply='done'"),
        _record(_trace("heard", "Copper Finch do private work"), stamp=5),
        _record(_trace("heard", "Copper Finch wrong process"), pid=99),
        _record(_trace("heard", "Copper Finch open the owned note")),
        _record(_trace("generated", "The owned note was saved", tools=["computer_use", "not-a-tool"], interrupted=False)),
        _record("sustained interruption; pausing reply"),
        _record(_trace("heard", "Unrelated private conversation")),
        _record(_trace("generated", "Unrelated private answer", tools=["shell"])),
    ]
    result = journal_summary(lines, "Copper Finch", 10, 42)
    assert len(result["full_live_transcripts"]) == 1
    assert len(result["generated_replies"]) == 1
    assert len(result["interruption_signals"]) == 1
    assert result["observed_tools"] == ["computer_use"]
    assert "private" not in json.dumps(result)
    assert "owned note" not in json.dumps(result)


def test_marker_correlation_has_word_boundaries_and_never_infers_intent_from_generated_text() -> None:
    assert contains_marker("Copper, Finch: begin", "copper finch")
    assert not contains_marker("Copper Finches unrelated", "copper finch")
    result = journal_summary([_record(_trace("generated", "Copper Finch completed", tools=["computer_use"]))], "Copper Finch", 10, 42)
    assert result["full_live_transcripts"] == []
    assert result["generated_replies"] == []


def test_completed_task_claim_needs_current_environment_verification_and_scoped_receipts() -> None:
    base = {"task_id": "owned", "objective": "Copper Finch edit the note", "created_at": 20,
            "status": "completed", "task_state": {"environment": {"version": 2},
                "controller": {"last_checkpoint_evidence_ids": ["fresh"]},
                "audit_reports": [{"evidence_id": "fresh", "authority": "verification", "environment_version_after": 1}]},
            "evidence_records": [{"evidence_id": "fresh", "tool": "computer_use", "result": "private note contents", "result_sha256": "a"*64}]}
    assert task_summary([base], "Copper Finch", 10)[0]["fresh_checkpoint_verification"] is False
    base["task_state"]["audit_reports"][0]["environment_version_after"] = 2
    result = task_summary([base, {**base, "objective": "Unrelated secret task"}], "Copper Finch", 10)
    assert len(result) == 1 and result[0]["fresh_checkpoint_verification"] is True
    assert "private note contents" not in json.dumps(result)
    assert "Copper Finch edit" not in json.dumps(result)


def test_persistent_memory_is_read_only_marker_scoped_and_redacted(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE memories(id INTEGER,text TEXT,kind TEXT,created_at REAL,last_used_at REAL,uses INTEGER)")
        connection.executemany("INSERT INTO memories VALUES(?,?,?,?,?,?)", [
            (1, "Copper Finch saved private file bytes", "exchange", 20, 21, 1),
            (2, "Unrelated private secret", "exchange", 20, 22, 5),
            (3, "Copper Finch obsolete record", "exchange", 1, 2, 1),
        ])
    result = memory_summary(path, "Copper Finch", 10)
    assert result["available"] and len(result["matching_rows"]) == 1
    assert "private" not in json.dumps(result)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3


def test_session_diagnostic_scopes_do_not_expose_token_or_cookie() -> None:
    keys = session_keys("SECRET-TOKEN", ["owned-task"])
    assert len(keys) == 2 and len(set(keys)) == 2
    assert all(len(key) == 64 and all(char in "0123456789abcdef" for char in key) for key in keys)
    assert "SECRET-TOKEN" not in str(keys)
