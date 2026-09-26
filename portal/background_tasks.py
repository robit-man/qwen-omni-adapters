"""Small crash-safe handoff store for long-running voice tasks.

The portal accepts the task while the foreground turn is still being answered;
the always-listening harness claims it afterwards and advances it one model or
tool step at a time.  The JSON file is deliberately boring: it is inspectable,
survives either process restarting, and is protected across processes with a
file lock.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import secrets
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_omni_adapters.context import context_text

TERMINAL_STATUSES = {"completed", "blocked", "cancelled"}
MAX_EXPIRED_RESUMES = 3
MAX_TASK_STATE_RECORDS = 128
STAGNANT_ACTION_RETIRE_THRESHOLD = 2


def _initial_task_state(
    objective: str, completion_criteria: str
) -> dict[str, Any]:
    """Create the externally owned state used across executor contexts.

    This state is deliberately separate from the renewable model transcript.
    Model prose cannot update it.  Only store operations backed by an executor
    receipt or an accepted checkpoint are allowed to advance it.
    """

    requirement_text = completion_criteria or objective
    return {
        "schema": "robit.omni.background-task-state.v1",
        "version": 0,
        "requirements": [
            {
                "requirement_id": "root",
                "text": requirement_text,
                "status": "pending",
                "evidence_ids": [],
            }
        ],
        "knowledge": {"version": 0, "records": []},
        "environment": {"version": 0, "artifacts": []},
        "controller": {
            "phase": "prethink",
            "active_requirement_id": "root",
            "current_subtask": objective,
            "next_transition": "prethink",
            "unresolved_evidence": [],
            "closed_evidence_slots": [],
            "last_audit_id": "",
            "stagnation": {"fingerprint": "", "count": 0},
            "retired_action_families": [],
            "executor_generation": 0,
        },
        "audit_reports": [],
    }


def _ensure_task_state(task: dict[str, Any]) -> dict[str, Any]:
    state = task.get("task_state")
    if not isinstance(state, dict) or state.get("schema") != (
        "robit.omni.background-task-state.v1"
    ):
        state = _initial_task_state(
            str(task.get("objective") or ""),
            str(task.get("completion_criteria") or ""),
        )
        task["task_state"] = state
    return state


def _apply_audit_report(task: dict[str, Any], report: Mapping[str, Any]) -> None:
    """Advance durable state from one deterministic executor audit."""

    state = _ensure_task_state(task)
    audit_id = str(report.get("audit_id") or "")[:128]
    if not audit_id:
        return
    reports = state.setdefault("audit_reports", [])
    if any(
        isinstance(item, Mapping) and str(item.get("audit_id") or "") == audit_id
        for item in reports
    ):
        return

    normalized = copy.deepcopy(dict(report))
    controller = state.setdefault("controller", {})
    knowledge = state.setdefault("knowledge", {"version": 0, "records": []})
    environment = state.setdefault("environment", {"version": 0, "artifacts": []})
    environment_before = int(environment.get("version") or 0)
    authority = str(normalized.get("authority") or "missing")
    evidence_id = str(normalized.get("evidence_id") or "")[:128]
    evidence_slot = str(normalized.get("evidence_slot") or "")[:128]

    closed_slots = controller.setdefault("closed_evidence_slots", [])
    slot_is_new = bool(evidence_slot) and not any(
        isinstance(item, Mapping)
        and str(item.get("slot_id") or "") == evidence_slot
        and int(item.get("environment_version") or 0) == environment_before
        for item in closed_slots
    )
    epistemic_progress = bool(
        slot_is_new
        and authority in {"discovery", "inspection", "concrete", "verification"}
    )
    if epistemic_progress:
        closed_slots.append(
            {
                "slot_id": evidence_slot,
                "environment_version": environment_before,
                "evidence_id": evidence_id,
            }
        )
        controller["closed_evidence_slots"] = closed_slots[-MAX_TASK_STATE_RECORDS:]
        records = knowledge.setdefault("records", [])
        records.append(
            {
                "evidence_id": evidence_id,
                "slot_id": evidence_slot,
                "authority": authority,
                "target": str(normalized.get("target") or "")[:500],
                "environment_version": environment_before,
            }
        )
        knowledge["records"] = records[-MAX_TASK_STATE_RECORDS:]
        knowledge["version"] = int(knowledge.get("version") or 0) + 1

    changed_paths = normalized.get("changed_paths")
    changed_paths = (
        [str(path)[:4096] for path in changed_paths if str(path)][:32]
        if isinstance(changed_paths, list)
        else []
    )
    environmental_progress = authority == "mutation" and bool(changed_paths)
    if environmental_progress:
        environment["version"] = environment_before + 1
        artifacts = environment.setdefault("artifacts", [])
        for path in changed_paths:
            artifacts.append(
                {
                    "path": path,
                    "evidence_id": evidence_id,
                    "environment_version": environment["version"],
                }
            )
        # The current artifact view is bounded; immutable versions remain in
        # evidence_records and audit_reports.
        current_by_path: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and str(artifact.get("path") or ""):
                current_by_path[str(artifact["path"])] = dict(artifact)
        environment["artifacts"] = list(current_by_path.values())[-64:]

    state_progress = epistemic_progress or environmental_progress
    fingerprint = str(normalized.get("state_fingerprint") or "")[:128]
    stagnation = controller.setdefault("stagnation", {})
    if state_progress:
        stagnation.update({"fingerprint": "", "count": 0})
    elif fingerprint:
        previous = str(stagnation.get("fingerprint") or "")
        stagnation.update(
            {
                "fingerprint": fingerprint,
                "count": int(stagnation.get("count") or 0) + 1
                if previous == fingerprint
                else 1,
            }
        )

    normalized["epistemic_progress"] = epistemic_progress
    normalized["environmental_progress"] = environmental_progress
    normalized["environment_version_before"] = environment_before
    normalized["environment_version_after"] = int(environment.get("version") or 0)
    normalized["stagnation_count"] = int(stagnation.get("count") or 0)
    retired = [
        str(value)[:160]
        for value in controller.get("retired_action_families", [])
        if str(value)
    ]
    # New information can be useful without repairing the failed execution
    # path that triggered recovery. Keep retirement per typed action family
    # until an audited milestone or environment mutation advances the task;
    # an unrelated clock/system lookup must not reopen the stagnant route.
    if environmental_progress or normalized.get("milestone_progress") is True:
        retired = []
    elif normalized["stagnation_count"] >= STAGNANT_ACTION_RETIRE_THRESHOLD:
        action_family = str(normalized.get("action_family") or "")[:160]
        if action_family:
            retired = list(dict.fromkeys([*retired, action_family]))[-32:]
    controller["retired_action_families"] = retired
    reports.append(normalized)
    state["audit_reports"] = reports[-MAX_TASK_STATE_RECORDS:]
    state["version"] = int(state.get("version") or 0) + 1
    controller["phase"] = "prethink"
    controller["next_transition"] = "prethink"
    controller["last_audit_id"] = audit_id
    controller["last_transition"] = str(normalized.get("transition") or "")[:40]


def _apply_checkpoint_state(
    task: dict[str, Any], transition: Mapping[str, Any]
) -> None:
    """Record a checkpoint as controller state, never as environmental fact."""

    state = _ensure_task_state(task)
    action = str(transition.get("action") or "")
    evidence_ids = [
        str(value)[:128]
        for value in transition.get("evidence_ids", [])
        if str(value)
    ][:16]
    remaining = [
        " ".join(str(value).split())[:300]
        for value in transition.get("remaining_requirements", [])
        if str(value).strip()
    ][:8]
    requirements = state.setdefault("requirements", [])
    root = next(
        (
            item
            for item in requirements
            if isinstance(item, dict) and item.get("requirement_id") == "root"
        ),
        None,
    )
    if root is not None:
        root["status"] = (
            "completed" if action == "complete" else "blocked" if action == "blocked" else "pending"
        )
        # A checkpoint advances the controller frontier; it must not erase
        # evidence accepted at an earlier frontier.  Keep a bounded,
        # insertion-ordered page table while the immutable receipts remain in
        # evidence_records.  This lets a later filesystem milestone retain the
        # source acquisition that justified it.
        previous_ids = [
            str(value)[:128]
            for value in root.get("evidence_ids", [])
            if str(value)
        ]
        root["evidence_ids"] = list(
            dict.fromkeys([*previous_ids, *evidence_ids])
        )[-32:]
    controller = state.setdefault("controller", {})
    controller["phase"] = "terminal" if action in {"complete", "blocked"} else "prethink"
    controller["next_transition"] = "stop" if action in {"complete", "blocked"} else "prethink"
    if remaining:
        controller["current_subtask"] = remaining[0]
    controller["remaining_requirements"] = remaining
    controller["last_checkpoint_evidence_ids"] = evidence_ids
    if action == "progress":
        controller["executor_generation"] = int(
            controller.get("executor_generation") or 0
        ) + 1
        controller["stagnation"] = {"fingerprint": "", "count": 0}
        controller["retired_action_families"] = []
    state["version"] = int(state.get("version") or 0) + 1


class BackgroundTaskStore:
    """Inter-process task records with leases and atomic checkpoints."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"schema": "robit.omni.background-tasks.v1", "tasks": []}

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty()
        if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
            return self._empty()
        return value

    def _write_unlocked(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(value, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _mutate(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            value = self._read_unlocked()
            result = callback(value)
            self._write_unlocked(value)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return result

    def _inspect(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            result = callback(self._read_unlocked())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return result

    @staticmethod
    def _public(task: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(task, dict):
            _ensure_task_state(task)
        public = {
            key: copy.deepcopy(task.get(key))
            for key in (
                "task_id",
                "objective",
                "completion_criteria",
                "status",
                "created_at",
                "updated_at",
                "round",
                "resume_count",
                "progress",
                "current_stage",
                "tools_used",
                "actions",
                "guidance",
                "compaction",
                "task_state",
                "result",
                "error",
            )
            if task.get(key) not in (None, "", [])
        }
        evidence_records = task.get("evidence_records")
        if isinstance(evidence_records, list) and evidence_records:
            # The status/list surface stays small. Detailed immutable receipts
            # are paged explicitly through expand_evidence instead of being
            # copied into every task poll.
            public["evidence_record_count"] = len(evidence_records)
        return public

    def create(self, objective: str, completion_criteria: str = "") -> dict[str, Any]:
        objective = str(objective).strip()
        if not objective:
            raise ValueError("objective is required")
        if len(objective) > 6000:
            raise ValueError("objective is too long")
        completion_criteria = str(completion_criteria).strip()
        if len(completion_criteria) > 2000:
            raise ValueError("completion_criteria is too long")

        def create(value: dict[str, Any]) -> dict[str, Any]:
            now = time.time()
            task = {
                "task_id": secrets.token_hex(6),
                "objective": objective,
                "completion_criteria": completion_criteria,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "round": 0,
                "progress": ["Accepted from the live conversation."],
                "messages": [],
                # Every concrete contract, including unrestricted shell, is
                # discovered from the tiny tool_search surface. Preloading
                # shell made browser/desktop jobs grab the first visible tool
                # and retry commands instead of discovering computer use.
                "active_tools": [],
                "tools_used": [],
                "actions": [],
                "current_stage": context_text("task_stages", "queued"),
                "guidance": [],
                "applied_guidance_ids": [],
                # These survive transcript compaction, so duplicate/stalled
                # work cannot become novel merely because older chat messages
                # were summarized away.
                "tool_fingerprints": [],
                "result_digests": [],
                # Compaction may page detailed tool turns out of the working
                # transcript, but never destroys their only task-local copy.
                # Records are append-only and addressed by the original call ID.
                "evidence_records": [],
                "task_state": _initial_task_state(objective, completion_criteria),
            }
            tasks = value.setdefault("tasks", [])
            tasks.append(task)
            # Terminal records are useful conversational context, but this is
            # not an archive. Keep every live task and only the newest finished
            # records once the file has become crowded.
            live = [item for item in tasks if item.get("status") not in TERMINAL_STATUSES]
            done = [item for item in tasks if item.get("status") in TERMINAL_STATUSES]
            value["tasks"] = [*live, *done[-24:]]
            return self._public(task)

        return self._mutate(create)

    def list(self) -> list[dict[str, Any]]:
        return self._inspect(
            lambda value: [self._public(item) for item in value.get("tasks", [])]
        )

    def archive_terminal(self, archive_path: Path | str) -> int:
        """Move finished tasks to a human-readable append-only local log."""

        destination = Path(archive_path).expanduser().resolve()

        def archive(value: dict[str, Any]) -> int:
            tasks = value.get("tasks", [])
            finished = [
                item for item in tasks if item.get("status") in TERMINAL_STATUSES
            ]
            if not finished:
                return 0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as output:
                for item in finished:
                    created = datetime.fromtimestamp(
                        float(item.get("created_at") or 0)
                    ).astimezone()
                    updated = datetime.fromtimestamp(
                        float(item.get("updated_at") or 0)
                    ).astimezone()
                    output.write(f"Task {item.get('task_id', 'unknown')}\n")
                    output.write(f"Status: {item.get('status', 'unknown')}\n")
                    output.write(f"Created: {created.isoformat(timespec='seconds')}\n")
                    output.write(f"Updated: {updated.isoformat(timespec='seconds')}\n")
                    output.write(f"Objective: {item.get('objective', '')}\n")
                    tools = item.get("tools_used")
                    if isinstance(tools, list) and tools:
                        output.write(
                            f"Tools used: {', '.join(str(tool) for tool in tools)}\n"
                        )
                    actions = item.get("actions")
                    if isinstance(actions, list) and actions:
                        output.write("Actions:\n")
                        for action in actions:
                            if not isinstance(action, Mapping):
                                continue
                            marker = "ok" if action.get("ok") is True else "failed"
                            output.write(
                                f"  - {action.get('tool', 'unknown')} "
                                f"{action.get('arguments', '{}')} -> {marker}: "
                                f"{action.get('outcome', '')}\n"
                            )
                    progress = item.get("progress")
                    if isinstance(progress, list) and progress:
                        output.write("Steps:\n")
                        for step in progress:
                            output.write(f"  - {' '.join(str(step).split())}\n")
                    if item.get("result"):
                        output.write(f"Result: {item['result']}\n")
                    if item.get("error"):
                        output.write(f"Error: {item['error']}\n")
                    output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            value["tasks"] = [
                item for item in tasks if item.get("status") not in TERMINAL_STATUSES
            ]
            return len(finished)

        return int(self._mutate(archive))

    def get(self, task_id: str) -> dict[str, Any] | None:
        def find(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") == task_id:
                    return self._public(item)
            return None

        return self._inspect(find)

    def claim_next(
        self,
        owner: str,
        lease_s: float = 60.0,
        *,
        exclude_task_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Claim pending work, or work whose prior process lease expired."""

        excluded = {str(value) for value in (exclude_task_ids or set())}

        def claim(value: dict[str, Any]) -> dict[str, Any] | None:
            now = time.time()
            candidates = []
            for item in value.get("tasks", []):
                if str(item.get("task_id") or "") in excluded:
                    continue
                status = item.get("status")
                expired = (
                    status == "running"
                    and float(item.get("lease_until") or 0) < now
                )
                if expired:
                    resume_count = int(item.get("resume_count") or 0) + 1
                    item["resume_count"] = resume_count
                    if resume_count >= MAX_EXPIRED_RESUMES:
                        item["status"] = "blocked"
                        item["updated_at"] = now
                        item["error"] = (
                            "The background worker stopped repeatedly before the "
                            "task could reach another verified checkpoint."
                        )
                        item.setdefault("progress", []).append(
                            "Stopped after three expired worker leases; manual review is required."
                        )
                        item["progress"] = item["progress"][-32:]
                        item["announcement_pending"] = True
                        item.pop("lease_until", None)
                        item.pop("owner", None)
                        continue
                if status == "pending" or expired:
                    candidates.append((item, expired))
            if not candidates:
                return None
            item, expired = min(
                candidates,
                key=lambda candidate: (
                    float(candidate[0].get("last_claimed_at") or 0),
                    float(candidate[0].get("created_at") or 0),
                ),
            )
            _ensure_task_state(item)
            item["status"] = "running"
            item["owner"] = owner
            item["lease_until"] = now + max(5.0, lease_s)
            item["last_claimed_at"] = now
            item["updated_at"] = now
            item["current_stage"] = context_text("task_stages", "preparing")
            if expired:
                item.setdefault("progress", []).append(
                    "Resumed after the previous worker stopped."
                )
            return copy.deepcopy(item)

        return self._mutate(claim)

    def checkpoint(
        self,
        task_id: str,
        owner: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        active_tools: list[str] | None = None,
        tools_used: list[str] | None = None,
        applied_guidance_ids: list[str] | None = None,
        tool_fingerprints: list[str] | None = None,
        result_digests: list[str] | None = None,
        progress: str = "",
        current_stage: str = "",
        result: str = "",
        error: str = "",
        status: str = "running",
        lease_s: float = 60.0,
        controller_transition: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        def update(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id or item.get("owner") != owner:
                    continue
                if item.get("status") == "cancelled":
                    return self._public(item)
                now = time.time()
                item["updated_at"] = now
                item["round"] = int(item.get("round") or 0) + 1
                # A successfully persisted worker checkpoint ends any prior
                # crash streak. Only consecutive expired leases without an
                # intervening checkpoint should trigger manual review.
                item["resume_count"] = 0
                previous_status = str(item.get("status") or "")
                item["status"] = status
                if status == "running":
                    item["lease_until"] = now + max(5.0, lease_s)
                    if current_stage:
                        item["current_stage"] = current_stage[:300]
                else:
                    item.pop("lease_until", None)
                    item.pop("owner", None)
                    if status == "pending" and current_stage:
                        item["current_stage"] = current_stage[:300]
                    else:
                        item.pop("current_stage", None)
                if (
                    status in {"completed", "blocked"}
                    and previous_status not in TERMINAL_STATUSES
                ):
                    # Completion speech is performed by a different thread and
                    # can be separated from this checkpoint by a process
                    # restart. Keep that tiny piece of delivery state durable
                    # so finished work is never silently lost.
                    item["announcement_pending"] = True
                if messages is not None:
                    item["messages"] = copy.deepcopy(messages)
                if active_tools is not None:
                    item["active_tools"] = list(dict.fromkeys(active_tools))[:3]
                if tools_used is not None:
                    item["tools_used"] = list(dict.fromkeys(tools_used))[-16:]
                if applied_guidance_ids is not None:
                    item["applied_guidance_ids"] = list(
                        dict.fromkeys(applied_guidance_ids)
                    )[-64:]
                if tool_fingerprints is not None:
                    item["tool_fingerprints"] = list(
                        dict.fromkeys(str(value) for value in tool_fingerprints if value)
                    )[-2048:]
                if result_digests is not None:
                    item["result_digests"] = list(
                        dict.fromkeys(str(value) for value in result_digests if value)
                    )[-2048:]
                if progress:
                    entries = item.setdefault("progress", [])
                    entries.append(progress[:1000])
                    item["progress"] = entries[-32:]
                if result:
                    item["result"] = result[:12000]
                if error:
                    item["error"] = error[:2000]
                else:
                    item.pop("error", None)
                if controller_transition is not None:
                    _apply_checkpoint_state(item, controller_transition)
                return self._public(item)
            return None

        return self._mutate(update)

    def release_owner(
        self,
        owner: str,
        *,
        current_stage: str = "Paused for an orderly worker restart",
    ) -> int:
        """Relinquish live leases during a controlled service shutdown."""

        owner = str(owner)

        def release(value: dict[str, Any]) -> int:
            now = time.time()
            count = 0
            for item in value.get("tasks", []):
                if item.get("status") != "running" or item.get("owner") != owner:
                    continue
                item["status"] = "pending"
                item["updated_at"] = now
                item["resume_count"] = 0
                item["current_stage"] = current_stage[:300]
                item.pop("lease_until", None)
                item.pop("owner", None)
                count += 1
            return count

        return int(self._mutate(release))

    def resume_after_review(self, task_id: str, reason: str) -> dict[str, Any] | None:
        """Resume only a lease-exhausted task after explicit human review."""

        reason = " ".join(str(reason).split())
        if not reason:
            raise ValueError("review reason is required")

        def resume(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id:
                    continue
                error = str(item.get("error") or "")
                if item.get("status") != "blocked" or not error.startswith(
                    "The background worker stopped repeatedly"
                ):
                    return self._public(item)
                now = time.time()
                item["status"] = "pending"
                item["updated_at"] = now
                item["resume_count"] = 0
                item["current_stage"] = context_text("task_stages", "queued")
                item["announcement_pending"] = False
                item.pop("error", None)
                item.pop("lease_until", None)
                item.pop("owner", None)
                item.setdefault("progress", []).append(
                    f"Human review resumed the preserved task: {reason[:500]}"
                )
                item["progress"] = item["progress"][-32:]
                return self._public(item)
            return None

        return self._mutate(resume)

    def compact_context(
        self,
        task_id: str,
        owner: str,
        *,
        messages: list[dict[str, Any]],
        receipt: Mapping[str, Any],
        evidence_records: list[Mapping[str, Any]] | None = None,
        lease_s: float = 60.0,
    ) -> dict[str, Any] | None:
        """Atomically replace a claimed task's renewable transcript.

        Compaction is control-plane maintenance, not task progress. It therefore
        refreshes the lease without incrementing the task round or manufacturing
        a progress entry.
        """

        def compact(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id or item.get("owner") != owner:
                    continue
                if item.get("status") != "running":
                    return self._public(item)
                now = time.time()
                item["messages"] = copy.deepcopy(messages)
                item["compaction"] = copy.deepcopy(dict(receipt))
                archived = item.setdefault("evidence_records", [])
                archived_ids = {
                    str(record.get("evidence_id") or "")
                    for record in archived
                    if isinstance(record, Mapping)
                }
                for record in evidence_records or []:
                    evidence_id = str(record.get("evidence_id") or "")
                    if not evidence_id or evidence_id in archived_ids:
                        continue
                    archived.append(copy.deepcopy(dict(record)))
                    archived_ids.add(evidence_id)
                item["updated_at"] = now
                item["lease_until"] = now + max(5.0, lease_s)
                return self._public(item)
            return None

        return self._mutate(compact)

    def expand_evidence(
        self, task_id: str, evidence_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Page immutable pre-compaction tool receipts by original call ID."""

        requested = {str(value) for value in evidence_ids if str(value)}

        def expand(value: dict[str, Any]) -> list[dict[str, Any]]:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id:
                    continue
                return [
                    copy.deepcopy(dict(record))
                    for record in item.get("evidence_records", [])
                    if isinstance(record, Mapping)
                    and str(record.get("evidence_id") or "") in requested
                ]
            return []

        return self._inspect(expand)

    def record_action(
        self,
        task_id: str,
        owner: str,
        *,
        call_id: str,
        tool: str,
        arguments: str,
        outcome: str,
        ok: bool,
        receipt: Mapping[str, Any] | None = None,
        evidence_record: Mapping[str, Any] | None = None,
        audit_report: Mapping[str, Any] | None = None,
        recorded_at: float | None = None,
    ) -> dict[str, Any] | None:
        """Append an audit entry and its immutable expandable evidence atomically."""

        def record(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id or item.get("owner") != owner:
                    continue
                if item.get("status") != "running":
                    return self._public(item)
                now = time.time()
                actions = item.setdefault("actions", [])
                bounded_call_id = str(call_id)[:128]
                action_already_recorded = any(
                    isinstance(action, Mapping)
                    and action.get("call_id") == bounded_call_id
                    for action in actions
                )
                if not action_already_recorded:
                    action = {
                        "call_id": bounded_call_id,
                        "tool": str(tool or "unknown")[:120],
                        "arguments": str(arguments)[:2000],
                        "outcome": str(outcome)[:1200],
                        "ok": bool(ok),
                    }
                    if receipt:
                        action["receipt"] = copy.deepcopy(dict(receipt))
                    if audit_report:
                        action["audit_id"] = str(
                            audit_report.get("audit_id") or ""
                        )[:128]
                    action_time = now if recorded_at is None else float(recorded_at)
                    if action_time > 0:
                        action["at"] = action_time
                    actions.append(action)
                    item["actions"] = actions[-32:]
                if evidence_record:
                    evidence_id = str(evidence_record.get("evidence_id") or "")
                    archived = item.setdefault("evidence_records", [])
                    already_archived = any(
                        isinstance(record, Mapping)
                        and str(record.get("evidence_id") or "") == evidence_id
                        for record in archived
                    )
                    if evidence_id and not already_archived:
                        archived.append(copy.deepcopy(dict(evidence_record)))
                if audit_report:
                    _apply_audit_report(item, audit_report)
                item["updated_at"] = now
                return self._public(item)
            return None

        return self._mutate(record)

    def renew_executor_context(
        self,
        task_id: str,
        owner: str,
        *,
        messages: list[dict[str, Any]],
        current_stage: str = "",
        lease_s: float = 60.0,
    ) -> dict[str, Any] | None:
        """Persist a fresh executor working set without inventing a task round."""

        def renew(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id or item.get("owner") != owner:
                    continue
                if item.get("status") != "running":
                    return self._public(item)
                now = time.time()
                item["messages"] = copy.deepcopy(messages)
                item["active_tools"] = []
                item["updated_at"] = now
                item["lease_until"] = now + max(5.0, lease_s)
                if current_stage:
                    item["current_stage"] = current_stage[:300]
                return self._public(item)
            return None

        return self._mutate(renew)

    def update_stage(
        self,
        task_id: str,
        owner: str,
        stage: str,
        *,
        lease_s: float = 60.0,
    ) -> dict[str, Any] | None:
        """Publish a live stage without fabricating a completed checkpoint."""

        stage = str(stage).strip()
        if not stage:
            raise ValueError("stage is required")

        def update(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id or item.get("owner") != owner:
                    continue
                if item.get("status") != "running":
                    return self._public(item)
                now = time.time()
                item["current_stage"] = stage[:300]
                item["updated_at"] = now
                item["lease_until"] = now + max(5.0, lease_s)
                return self._public(item)
            return None

        return self._mutate(update)

    def pending_announcements(self) -> list[dict[str, Any]]:
        """Return terminal reports that have not yet reached the speaker."""

        return self._inspect(
            lambda value: [
                self._public(item)
                for item in value.get("tasks", [])
                if item.get("status") in {"completed", "blocked"}
                and item.get("announcement_pending") is True
            ]
        )

    def mark_announced(self, task_id: str) -> bool:
        """Persist that a terminal report was spoken or intentionally interrupted."""

        def mark(value: dict[str, Any]) -> bool:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id:
                    continue
                if item.get("announcement_pending") is not True:
                    return False
                item["announcement_pending"] = False
                item["announced_at"] = time.time()
                return True
            return False

        return bool(self._mutate(mark))

    def add_guidance(
        self,
        task_id: str,
        content: str,
        *,
        provenance: str = "external_control_request",
    ) -> dict[str, Any] | None:
        content = str(content).strip()
        if not content:
            raise ValueError("guidance is required")
        if len(content) > 4000:
            raise ValueError("guidance is too long")

        def add(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id:
                    continue
                if item.get("status") in TERMINAL_STATUSES:
                    return self._public(item)
                guidance = item.setdefault("guidance", [])
                guidance.append(
                    {
                        "guidance_id": secrets.token_hex(5),
                        "content": content,
                        "received_at": time.time(),
                        "provenance": str(provenance)[:80],
                    }
                )
                if item.get("status") == "waiting_input":
                    item["status"] = "pending"
                    item["current_stage"] = context_text("task_stages", "queued")
                item["guidance"] = guidance[-64:]
                item["updated_at"] = time.time()
                item.setdefault("progress", []).append(
                    f"New conversational guidance received: {content[:400]}"
                )
                return self._public(item)
            return None

        return self._mutate(add)

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        def cancel(value: dict[str, Any]) -> dict[str, Any] | None:
            for item in value.get("tasks", []):
                if item.get("task_id") != task_id:
                    continue
                if item.get("status") not in TERMINAL_STATUSES:
                    item["status"] = "cancelled"
                    item["updated_at"] = time.time()
                    item.pop("lease_until", None)
                    item.pop("owner", None)
                    item.setdefault("progress", []).append("Cancelled by the user.")
                return self._public(item)
            return None

        return self._mutate(cancel)

    def remove(self, task_id: str) -> bool:
        """Delete one task record, including a live record the user overrides."""

        normalized = str(task_id).strip()
        if not normalized:
            return False

        def remove(value: dict[str, Any]) -> bool:
            tasks = value.get("tasks", [])
            for index, item in enumerate(tasks):
                if item.get("task_id") == normalized:
                    del tasks[index]
                    return True
            return False

        return bool(self._mutate(remove))
