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

TERMINAL_STATUSES = {"completed", "blocked", "cancelled"}


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
        return {
            key: copy.deepcopy(task.get(key))
            for key in (
                "task_id",
                "objective",
                "completion_criteria",
                "status",
                "created_at",
                "updated_at",
                "round",
                "progress",
                "current_stage",
                "tools_used",
                "guidance",
                "result",
                "error",
            )
            if task.get(key) not in (None, "", [])
        }

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
                "active_tools": ["shell"],
                "tools_used": [],
                "current_stage": "Queued",
                "guidance": [],
                "applied_guidance_ids": [],
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

    def claim_next(self, owner: str, lease_s: float = 60.0) -> dict[str, Any] | None:
        """Claim pending work, or work whose prior process lease expired."""

        def claim(value: dict[str, Any]) -> dict[str, Any] | None:
            now = time.time()
            for item in value.get("tasks", []):
                status = item.get("status")
                expired = status == "running" and float(item.get("lease_until") or 0) < now
                if status != "pending" and not expired:
                    continue
                item["status"] = "running"
                item["owner"] = owner
                item["lease_until"] = now + max(5.0, lease_s)
                item["updated_at"] = now
                item["current_stage"] = "Preparing task"
                if expired:
                    item.setdefault("progress", []).append(
                        "Resumed after the previous worker stopped."
                    )
                return copy.deepcopy(item)
            return None

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
        progress: str = "",
        current_stage: str = "",
        result: str = "",
        error: str = "",
        status: str = "running",
        lease_s: float = 60.0,
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
                return self._public(item)
            return None

        return self._mutate(update)

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

    def add_guidance(self, task_id: str, content: str) -> dict[str, Any] | None:
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
                    }
                )
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
