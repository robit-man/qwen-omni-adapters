"""Immutable, scoped tool and procedural evidence. History never grants permission."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class BehaviorMemory:
    def __init__(self, path: Path, *, owner_session: str = "", episodic_root: Path | None = None):
        self.path, self.owner_session, self.episodic_root = Path(path), owner_session, episodic_root

    @contextmanager
    def _write_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _db(self, *, write=False):
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            self.path.chmod(0o600)
        db = sqlite3.connect(str(self.path) if write else f"file:{self.path}?mode=ro", uri=not write, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("""CREATE TABLE IF NOT EXISTS behavior_sources(
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, kind TEXT NOT NULL,
                    outcome TEXT NOT NULL, occurred_at REAL NOT NULL, body TEXT NOT NULL, sha256 TEXT NOT NULL)""")
                db.execute("CREATE INDEX IF NOT EXISTS behavior_scope ON behavior_sources(scope,occurred_at)")
                db.execute("CREATE TABLE IF NOT EXISTS cleared_scopes(scope TEXT PRIMARY KEY,cleared_at REAL NOT NULL)")
            yield db
            if write:
                db.commit()
        finally:
            db.close()

    def _append(self, session, identity, kind, outcome, stamp, body, *, scope_hash=None, origin_created_at=None):
        with self._write_lock():
            return self._append_locked(session, identity, kind, outcome, stamp, body,
                scope_hash=scope_hash, origin_created_at=origin_created_at)

    def _append_locked(self, session, identity, kind, outcome, stamp, body, *, scope_hash=None, origin_created_at=None):
        scope = scope_hash or _hash(session)
        source_id = _hash(scope + "\0" + identity)
        body = {**body, "source_id": source_id, "kind": kind, "outcome": outcome,
            "occurred_at": stamp, "applicability": "unverified", "permission_granted": False}
        encoded = _json(body)
        with self._db(write=True) as db:
            cutoff = db.execute("SELECT cleared_at FROM cleared_scopes WHERE scope=?", (scope,)).fetchone()
            if cutoff and (origin_created_at if origin_created_at is not None else stamp) <= cutoff[0]:
                return
            existing = db.execute("SELECT sha256 FROM behavior_sources WHERE id=?", (source_id,)).fetchone()
            if existing and existing[0] != _hash(encoded):
                raise ValueError("Behavior source identity conflicts with retained evidence")
            db.execute("INSERT OR IGNORE INTO behavior_sources VALUES(?,?,?,?,?,?,?)",
                (source_id, scope, kind, outcome, stamp, encoded, _hash(encoded)))
        # Source commits before optional semantic/graph indexing. Replays repair
        # an interrupted index write without counting another independent case.
        if kind == "behavior" and self.episodic_root is not None:
            from qwen_omni_adapters.episodic_memory import EpisodicMemory
            key = "device-owner" if scope == _hash(self.owner_session) else scope
            store = EpisodicMemory(self.episodic_root / f"{key}.sqlite3")
            try:
                store.append(event_key=f"behavior:{source_id}", episode=f"behavior:{body['task_id']}",
                    modality="tool", authority="tool_result", actor="audited_controller", text=encoded,
                    occurred_at=stamp, provenance={"origin": "audited_behavior_source", "behavior_source_id": source_id,
                        "conversation_eligible": False, "current_environment_evidence": False})
            finally:
                store.close()

    def invocation(self, session, *, episode, sequence, tool, ok, occurred_at):
        self._append(session, f"invocation:{episode}:{sequence}", "invocation", "returned" if ok else "failed", occurred_at,
            {"tool": tool, "request_id": episode, "sequence": sequence,
                "claim_limits": "Tool return status is not verification of task completion"})

    def observe_task(self, session: str, task: Mapping):
        origin = task.get("origin_scope")
        if not session or not isinstance(origin, str) or len(origin) != 64 or any(char not in "0123456789abcdef" for char in origin) or not task.get("created_at"):
            return
        state = task.get("task_state") or {}
        controller = state.get("controller") or {}
        contracts = list((controller.get("frontier") or {}).get("steps") or [])
        contracts += [controller.get("last_contract"), controller.get("pending_contract")]
        by_id = {item.get("contract_id"): item for item in contracts if isinstance(item, Mapping)}
        actions = {item.get("call_id"): item for item in task.get("actions", []) if isinstance(item, Mapping)}
        for audit in state.get("audit_reports", []):
            if not isinstance(audit, Mapping) or not audit.get("audit_id") or not audit.get("evidence_id"):
                continue
            action = actions.get(audit["evidence_id"])
            contract = by_id.get(audit.get("contract_id"))
            if not action or not contract or not action.get("at"):
                continue
            # A successful mutation awaiting a verifier stays a candidate.
            # An informational read does not become a verified procedure.
            outcome = "failed" if audit.get("executor_succeeded") is False else "candidate"
            if (audit.get("executor_succeeded") is True and audit.get("contract_satisfied") is True
                and audit.get("authority") == "verification" and audit.get("contract_verification_matched") is True):
                outcome = "verified_effect"
            body = {"task_id": str(task.get("task_id")), "contract_id": str(audit["contract_id"]),
                "trigger": str(task.get("objective") or "")[:1500], "step": str(contract.get("subtask") or "")[:800],
                "family": str(contract.get("capability_family") or ""), "operation": str(contract.get("executor_operation") or ""),
                "expected_effect": str(contract.get("expected_effect") or ""),
                "acceptance_test": str(contract.get("acceptance_test") or "")[:1000],
                "required_tools": [str(action.get("tool") or "")],
                "source_audit": {key: audit.get(key) for key in ("audit_id", "evidence_id", "authority", "resource_key",
                    "resource_fingerprint", "executor_succeeded", "contract_satisfied", "contract_verification_matched",
                    "acceptance_semantics_checked", "environment_version_before", "environment_version_after")},
                "claim_limits": "Historical observed approach. Verified effect does not establish complete-task success or present applicability."}
            self._append(session, f"audit:{task.get('task_id')}:{audit['audit_id']}", "behavior", outcome, float(action["at"]), body,
                scope_hash=origin, origin_created_at=float(task["created_at"]))

    def view(self, session: str, *, include_content=True):
        try:
            return self._view(session, include_content=include_content)
        except (OSError, sqlite3.Error, ValueError):
            return {"available": False, "content_available": False, "recent": [], "links": [], "tools": []}

    def _view(self, session: str, *, include_content=True):
        result = {"available": True, "content_available": include_content, "invocations": 0,
            "behaviors": 0, "verified_effect": 0, "failed": 0, "candidate": 0, "recent": [], "links": [], "tools": []}
        if not self.path.exists():
            return result
        with self._db() as db:
            scope = _hash(session)
            for row in db.execute("SELECT kind,outcome,COUNT(*) FROM behavior_sources WHERE scope=? GROUP BY kind,outcome", (scope,)):
                result["invocations" if row[0] == "invocation" else "behaviors"] += row[2]
                if row[0] == "behavior":
                    result[row[1]] += row[2]
            result["tools"] = [{"name": row[0], "returns": row[1], "failed": row[2]} for row in db.execute(
                "SELECT json_extract(body,'$.tool'),COUNT(*),SUM(outcome='failed') FROM behavior_sources WHERE scope=? AND kind='invocation' GROUP BY json_extract(body,'$.tool') ORDER BY COUNT(*) DESC LIMIT 12", (scope,))]
            if include_content:
                rows = db.execute("SELECT body,sha256 FROM behavior_sources WHERE scope=? AND kind='behavior' ORDER BY occurred_at DESC,id DESC LIMIT 24", (scope,)).fetchall()
                result["recent"] = [{**json.loads(row[0]), "source_sha256": row[1]} for row in rows if _hash(row[0]) == row[1]]
                groups = {}
                for item in reversed(result["recent"]):
                    key = (item["task_id"], item["contract_id"])
                    previous = groups.get(key)
                    if previous:
                        result["links"].append({"from": previous["source_id"], "to": item["source_id"], "kind": "later evidence for same contract"})
                    groups[key] = item
        return result

    def clear(self, session):
        if session == self.owner_session:
            return
        with self._write_lock(), self._db(write=True) as db:
            db.execute("DELETE FROM behavior_sources WHERE scope=?", (_hash(session),))
            db.execute("INSERT OR REPLACE INTO cleared_scopes VALUES(?,?)", (_hash(session), time.time()))
