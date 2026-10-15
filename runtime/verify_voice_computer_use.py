#!/usr/bin/env python3
"""Observe an assisted live microphone test without recording or executing work.

The human speaks the marker. Full heard/generated events come from the live
call harness, never a typed substitute. Output contains evidence hashes and
counts; transcript, cookie, token and unrelated session content remain private.
This collector reports evidence, not a claim that a whole task or suite passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TOOLS = {"native_ui", "computer_use", "browser_interact", "gui_interact", "memory_write", "memory_read",
         "memory_search", "background_task", "web_search", "web_fetch", "web_crawl",
         "workspace_file", "shell", "file_deliver", "task_checkpoint", "task_manage",
         "task_audit", "tool_search"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def contains_marker(text: Any, marker: str) -> bool:
    # This is test-evidence correlation with a human-declared nonce, not action routing.
    def normalize(value: Any) -> str:
        return " ".join(re.findall(r"\w+", str(value).casefold()))
    return (" " + normalize(marker) + " ") in (" " + normalize(text) + " ")


def read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def journal_summary(lines: list[str], marker: str, since: float, pid: int) -> dict[str, Any]:
    heard, replies, interruptions, tools = [], [], [], set()
    active = False
    for line in lines[-5000:]:
        try:
            record = json.loads(line)
            stamp = float(record.get("__REALTIME_TIMESTAMP", 0)) / 1_000_000
            if stamp < since or str(record.get("_PID")) != str(pid):
                continue
            message = str(record.get("MESSAGE", ""))
            prefix, separator, raw = message.partition("conversation_trace ")
            if separator:
                event = json.loads(raw)
                if event.get("event") == "heard":
                    active = contains_marker(event.get("text", ""), marker)
                    if active:
                        heard.append({"at": stamp, "transcript_sha256": digest(event["text"]),
                                      "source": "live_harness_observation"})
                elif active and event.get("event") == "generated":
                    names = {name for name in event.get("tools", []) if name in TOOLS}
                    tools.update(names)
                    replies.append({"at": stamp, "reply_sha256": digest(event.get("text", "")),
                                    "tools": sorted(names), "interrupted": event.get("interrupted") is True,
                                    "error_present": bool(event.get("error"))})
            elif active and any(fragment in message for fragment in (
                "sustained interruption; pausing", "interruption understood; yielding",
                "confirmed interruption; cancelling", "interruption rejected; resuming",
            )):
                interruptions.append({"at": stamp, "event_sha256": digest(message)})
        except (ValueError, TypeError, AttributeError):
            continue
    return {"full_live_transcripts": heard, "generated_replies": replies,
            "interruption_signals": interruptions, "observed_tools": sorted(tools)}


def task_summary(tasks: list[dict[str, Any]], marker: str, since: float) -> list[dict[str, Any]]:
    result = []
    for task in tasks:
        if not isinstance(task, dict) or float(task.get("created_at", 0)) < since or not contains_marker(task.get("objective", ""), marker):
            continue
        state = task.get("task_state", {})
        version = state.get("environment", {}).get("version", 0)
        selected = set(state.get("controller", {}).get("last_checkpoint_evidence_ids", []))
        fresh = any(report.get("evidence_id") in selected and report.get("authority") == "verification"
                    and report.get("environment_version_after") == version
                    for report in state.get("audit_reports", []) if isinstance(report, dict))
        receipts = []
        for evidence in task.get("evidence_records", []):
            if not isinstance(evidence, dict) or evidence.get("tool") not in TOOLS:
                continue
            receipts.append({"evidence_id_sha256": digest(evidence.get("evidence_id")),
                             "tool": evidence["tool"], "receipt_sha256": digest(evidence),
                             "stored_result_sha256": evidence.get("result_sha256", "")[:64]})
        result.append({"task_id_sha256": digest(task.get("task_id")), "status": task.get("status"),
                       "objective_sha256": digest(task.get("objective")), "environment_version": version,
                       "fresh_checkpoint_verification": fresh, "receipts": receipts,
                       "action_count": len(task.get("actions", []))})
    return result


def memory_summary(path: Path, marker: str, since: float) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "matching_rows": []}
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
            rows = connection.execute("SELECT id,text,kind,created_at,last_used_at,uses FROM memories "
                "WHERE created_at >= ? OR last_used_at >= ? ORDER BY last_used_at DESC LIMIT 256", (since, since)).fetchall()
        matching = [{"row_id_sha256": digest(row[0]), "content_sha256": digest(row[1]),
                     "kind": row[2] if row[2] in {"turn", "exchange", "heard", "sound"} else "other",
                     "created_at": row[3], "last_used_at": row[4], "uses": row[5]}
                    for row in rows if contains_marker(row[1], marker)]
        return {"available": True, "matching_rows": matching}
    except (OSError, sqlite3.DatabaseError):
        return {"available": False, "matching_rows": [], "unreadable": True}


def session_keys(token: str, task_ids: list[str]) -> list[str]:
    foreground = hashlib.sha256(b"omni-voice-session\0" + token.encode()).hexdigest()
    sessions = [foreground]
    for task_id in task_ids:
        hashed = hashlib.sha256(f"{foreground}\0{task_id}\0background-controller-v6".encode()).hexdigest()
        sessions.append("background-" + hashed[:40])
    return [hashlib.sha256(session.encode()).hexdigest() for session in sessions]


def _journal(since: float) -> list[str]:
    result = subprocess.run(  # noqa: S603 - read-only fixed service query.
        ["journalctl", "--user", "-u", "omni-call-harness.service", "--since", f"@{since:.6f}",
         "--no-pager", "--output=json", "--lines=5000"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return result.stdout[-8 * 1024 * 1024:].splitlines() if result.returncode == 0 else []


def collect(root: Path, marker: str, since: float, token: str, lines: list[str], states: list[dict[str, Any]]) -> dict[str, Any]:
    status = read_json(root / "state/harness-status.json")
    pid = int(status.get("pid", 0))
    tasks = read_json(root / "state/background-tasks.json").get("tasks", [])
    owned_tasks = [task for task in tasks if isinstance(task, dict) and float(task.get("created_at", 0)) >= since
                   and contains_marker(task.get("objective", ""), marker)]
    diagnostics = []
    for key in session_keys(token, [str(task["task_id"]) for task in owned_tasks]):
        events = read_json(root / "session-logs" / (key + ".json")).get("events", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            diagnostics.append({"event_sha256": digest(event),
                                "tool": event.get("tool_name") if event.get("tool_name") in TOOLS else None})
    journal = journal_summary(lines, marker, since, pid)
    try:
        os.kill(pid, 0)
        alive = pid > 0
    except (OSError, ValueError):
        alive = False
    return {"schema": "robit.omni.voice-computer-use-observation.v1", "status": "evidence_collected",
            "marker_sha256": digest(marker), "since": since, "collected_at": time.time(),
            "harness_alive": alive, "harness_states": states,
            "full_live_asr_trace_present": bool(journal["full_live_transcripts"]),
            "journal": journal, "tasks": task_summary(owned_tasks, marker, since),
            "passive_memory": memory_summary(root / "memory.sqlite3", marker, since),
            "scoped_diagnostic_events": diagnostics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-file", type=Path, default=REPO / "runtime-data/state/access-token.txt")
    parser.add_argument("--since", type=float, default=None)
    parser.add_argument("--duration", type=float, default=120)
    args = parser.parse_args()
    if not 0 <= args.duration <= 600 or not 4 <= len(args.marker.strip()) <= 80:
        parser.error("Use a 4–80 character spoken marker and a 0–600 second observation window")
    since = args.since if args.since is not None else time.time()
    root, states, deadline = REPO / "runtime-data", [], time.monotonic() + args.duration
    while True:
        status = read_json(root / "state/harness-status.json")
        state = {key: status.get(key) for key in ("state", "updated_at")}
        if not states or state != states[-1]:
            states.append(state)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    report = collect(root, args.marker, since, args.token_file.read_text().strip(), _journal(since), states)
    report["requested_model"] = args.model
    destination = root / ("voice-computer-use-" + secrets.token_hex(6) + ".json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    destination.chmod(0o600)
    print(json.dumps({"status": report["status"], "artifact": str(destination),
                      "full_live_asr_trace_present": report["full_live_asr_trace_present"],
                      "matching_task_count": len(report["tasks"]),
                      "matching_memory_rows": len(report["passive_memory"]["matching_rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
