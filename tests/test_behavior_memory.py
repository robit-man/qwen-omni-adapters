from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time

import pytest

from qwen_omni_adapters.behavior_memory import BehaviorMemory
from qwen_omni_adapters.episodic_memory import EpisodicMemory


def trajectory(session="owner"):
    contract = {"contract_id": "step-a", "subtask": "Write and verify the report", "capability_family": "filesystem",
        "executor_operation": "write", "expected_effect": "change_environment", "acceptance_test": "Exact saved report"}
    return {"task_id": "task-a", "origin_scope": hashlib.sha256(session.encode()).hexdigest(), "created_at": 99,
        "objective": "Save the measured report", "actions": [],
        "task_state": {"controller": {"frontier": {"steps": [contract]}}, "audit_reports": []}}


def append_audit(task, identity, *, ok, authority, satisfied=False, matched=False):
    task["actions"].append({"call_id": identity, "tool": "workspace_file", "at": 100 + len(task["actions"]),
        "arguments": '{"api_key":"secret-never-copy"}', "outcome": "private output"})
    task["task_state"]["audit_reports"].append({"audit_id": f"audit-{identity}", "evidence_id": identity,
        "contract_id": "step-a", "authority": authority, "executor_succeeded": ok,
        "contract_satisfied": satisfied, "contract_verification_matched": matched,
        "resource_fingerprint": "observed-content-hash", "environment_version_after": 1})


def test_failed_action_mutation_and_verification_remain_distinct_sources(tmp_path):
    root = tmp_path / "episodes"
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3", owner_session="owner", episodic_root=root)
    task = trajectory("owner")
    append_audit(task, "failed", ok=False, authority="inspection")
    memory.observe_task("owner", task)
    append_audit(task, "write", ok=True, authority="mutation")
    memory.observe_task("owner", task)
    append_audit(task, "verify", ok=True, authority="verification", satisfied=True, matched=True)
    memory.observe_task("owner", task)
    memory.observe_task("owner", task)
    result = memory.view("owner")
    assert (result["behaviors"], result["failed"], result["candidate"], result["verified_effect"]) == (3, 1, 1, 1)
    assert len(result["links"]) == 2
    assert all(item["applicability"] == "unverified" and not item["permission_granted"] for item in result["recent"])
    assert "secret-never-copy" not in json.dumps(result) and "private output" not in json.dumps(result)
    assert memory.path.stat().st_mode & 0o777 == 0o600
    store = EpisodicMemory(root / "device-owner.sqlite3")
    try:
        found = store.search("measured report")
        assert found["results"] and store.stats()["events"] == 3
        assert found["source_is_current_visual_evidence"] is False
    finally:
        store.close()


def test_inspection_success_and_incomplete_verification_never_admit_success(tmp_path):
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3")
    task = trajectory("alice")
    append_audit(task, "inspect", ok=True, authority="inspection", satisfied=True)
    append_audit(task, "not-verified", ok=True, authority="verification", satisfied=False, matched=True)
    memory.observe_task("alice", task)
    assert memory.view("alice")["verified_effect"] == 0
    assert memory.view("alice")["candidate"] == 2


def test_read_only_projection_scopes_and_trash_preserve_owner_and_other_browser(tmp_path):
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3", owner_session="owner")
    assert memory.view("alice")["behaviors"] == 0 and not memory.path.exists()
    for session in ("alice", "bob", "owner"):
        memory.invocation(session, episode="request-a", sequence=1, tool="mcp_call", ok=False, occurred_at=123)
        task = trajectory(session)
        append_audit(task, "verify", ok=True, authority="verification", satisfied=True, matched=True)
        memory.observe_task(session, task)
    memory.clear("alice")
    memory.clear("owner")
    assert memory.view("alice")["invocations"] == 0 and not memory.view("alice")["recent"]
    assert memory.view("bob")["invocations"] == 1 and memory.view("owner")["behaviors"] == 1
    private = memory.view("owner", include_content=False)
    assert private["verified_effect"] == 1 and not private["recent"] and not private["links"]


def test_conflicting_occurrence_and_tampered_source_are_not_accepted(tmp_path):
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3")
    task = trajectory("alice")
    append_audit(task, "source", ok=False, authority="inspection")
    memory.observe_task("alice", task)
    changed = copy.deepcopy(task)
    changed["task_state"]["audit_reports"][0]["executor_succeeded"] = True
    with pytest.raises(ValueError, match="conflicts"):
        memory.observe_task("alice", changed)
    with sqlite3.connect(memory.path) as db:
        db.execute("UPDATE behavior_sources SET body=?", ('{"forged":true}',))
    assert memory.view("alice")["recent"] == []


def test_unknown_legacy_attribution_is_not_backfilled(tmp_path):
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3")
    task = trajectory()
    append_audit(task, "legacy", ok=True, authority="verification", satisfied=True, matched=True)
    del task["actions"][0]["at"]
    memory.observe_task("alice", task)
    memory.observe_task("", trajectory())
    assert memory.view("alice")["behaviors"] == 0


def test_browser_task_is_not_learned_under_native_executor_scope_and_trash_blocks_replay(tmp_path):
    memory = BehaviorMemory(tmp_path / "behavior.sqlite3", owner_session="owner")
    task = trajectory("browser")
    append_audit(task, "verify", ok=True, authority="verification", satisfied=True, matched=True)
    memory.observe_task("owner", task)
    assert memory.view("browser")["verified_effect"] == 1
    assert memory.view("owner")["behaviors"] == 0
    memory.clear("browser")
    memory.observe_task("owner", task)
    assert memory.view("browser")["behaviors"] == 0
    task["created_at"] = time.time() + 1
    memory.observe_task("owner", task)
    assert memory.view("browser")["behaviors"] == 1
