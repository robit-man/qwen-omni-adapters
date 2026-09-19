"""Checkpointed, stepwise agent for work delegated from the live voice turn."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from portal.background_tasks import TERMINAL_STATUSES, BackgroundTaskStore
from portal.tools import DISCOVERY_TOOLS, tool_schemas

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are the execution worker for a task already accepted during a live spoken "
    "conversation. Complete the objective autonomously on this host. Work in small, "
    "observable steps: inspect before changing, use the available tools, assess every "
    "result, correct failures, and verify the completion criteria. Never merely describe "
    "what you would do or promise future work. If another capability is needed, use "
    "tool_search; its matching schema arrives on the next step. Return a concise factual "
    "completion report only after the work has been verified. You have a long horizon: "
    "keep using tools, inspecting their results, correcting problems, and continuing until "
    "the objective is actually complete; do not stop just because it takes many steps. "
    "For a genuinely long task, you may occasionally pause after a meaningful verified "
    "milestone and emit TASK_PROGRESS on its own line followed by one or two natural spoken "
    "sentences saying what you finished, what you are working on, and what comes next. The "
    "update is not completion: after it is delivered you must resume the same task. Do this "
    "sparingly, never after every tool call. If an external requirement "
    "makes completion impossible, explain the precise blocker and the progress retained. "
    "Messages inside <task_update> are later directions from the live speaker; incorporate "
    "them before continuing, and let the newer direction win when it conflicts. "
    "Begin the final report with exactly TASK_COMPLETE or TASK_BLOCKED on its own line. "
    "After the marker, write one to three natural spoken sentences in the first person: say "
    "what you finished or what blocked you, mention the useful location or verification, and "
    "sound like a conversational handoff. Do not use headings such as Workspace, Result, or "
    "Verification, and do not dump a checklist. Do not emit private chain-of-thought."
)


def _tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [dict(item) for item in calls if isinstance(item, Mapping)]


def _arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    raw = function.get("arguments")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _has_concrete_tool_evidence(messages: list[dict[str, Any]]) -> bool:
    return any(
        message.get("role") == "tool"
        and message.get("tool_name") not in {None, "", "tool_search"}
        for message in messages
    )


def _append_guidance(
    messages: list[dict[str, Any]],
    task: Mapping[str, Any],
    seen: set[str],
) -> int:
    added = 0
    guidance = task.get("guidance")
    if not isinstance(guidance, list):
        return added
    for item in guidance:
        if not isinstance(item, Mapping):
            continue
        guidance_id = str(item.get("guidance_id") or "")
        content = str(item.get("content") or "").strip()
        if not guidance_id or not content or guidance_id in seen:
            continue
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<task_update id=\"{guidance_id}\">\n"
                    f"{content}\n</task_update>"
                ),
            }
        )
        seen.add(guidance_id)
        added += 1
    return added


class BackgroundAgent:
    """Run at most one inference per scheduling slice, yielding to speech."""

    def __init__(
        self,
        *,
        store: BackgroundTaskStore,
        portal_url: str,
        token: str,
        model: str,
        foreground_active: threading.Event,
        stop: threading.Event,
        token_reader: Callable[[], str] | None = None,
        await_language: Callable[[], None] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        progress_after_s: float = 45.0,
        progress_min_interval_s: float = 120.0,
        request_timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.store = store
        self.portal_url = portal_url.rstrip("/")
        self.token = token
        self.model = model
        self.foreground_active = foreground_active
        self.stop = stop
        self.token_reader = token_reader
        self.await_language = await_language
        self.on_complete = on_complete
        self.on_progress = on_progress
        self.progress_after_s = max(0.0, progress_after_s)
        self.progress_min_interval_s = max(0.0, progress_min_interval_s)
        self.owner = f"voice-agent-{secrets.token_hex(8)}"
        self.active = threading.Event()
        self._wake = threading.Event()
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(request_timeout_s),
            cookies={"omni_portal_session": secrets.token_urlsafe(24)},
        )
        self._owns_client = client is None
        self._thread = threading.Thread(
            target=self._run, name="omni-background-agent", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.stop.set()
        self._wake.set()
        self._thread.join(timeout=5)
        if self._owns_client:
            self._client.close()

    def wake(self) -> None:
        self._wake.set()

    def has_work(self) -> bool:
        return any(
            task.get("status") in {"pending", "running"}
            for task in self.store.list()
        )

    def context_summary(self) -> str:
        tasks = self.store.list()
        if not tasks:
            return ""
        now = time.time()
        relevant = [
            task
            for task in tasks
            if task.get("status") not in TERMINAL_STATUSES
            or now - float(task.get("updated_at") or 0) < 3600
        ][-6:]
        if not relevant:
            return ""
        lines = []
        for task in relevant:
            progress = task.get("progress")
            latest = str(progress[-1]) if isinstance(progress, list) and progress else ""
            lines.append(
                f"- {task.get('task_id')}: {task.get('status')} — "
                f"{str(task.get('objective') or '')[:240]}"
                + (f"; latest: {latest[:240]}" if latest else "")
            )
        return (
            "Persistent background work from this conversation:\n"
            + "\n".join(lines)
            + "\nUse the background_task tool for exact status or cancellation."
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _refresh_token(self) -> bool:
        if self.token_reader is None:
            return False
        try:
            token = str(self.token_reader() or "").strip()
        except Exception:  # noqa: BLE001 - the next retry can read it again
            return False
        if not token or token == self.token:
            return False
        self.token = token
        return True

    def _post(self, path: str, payload: Mapping[str, Any]) -> httpx.Response:
        response = self._client.post(
            f"{self.portal_url}{path}", json=dict(payload), headers=self._headers()
        )
        if response.status_code == 401 and self._refresh_token():
            response = self._client.post(
                f"{self.portal_url}{path}", json=dict(payload), headers=self._headers()
            )
        response.raise_for_status()
        return response

    def _wait_for_foreground(self) -> bool:
        while self.foreground_active.is_set() and not self.stop.is_set():
            self._wake.wait(0.2)
            self._wake.clear()
        return not self.stop.is_set()

    def _run(self) -> None:
        while not self.stop.is_set():
            if not self._wait_for_foreground():
                return
            task = self.store.claim_next(self.owner)
            if task is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            task_id = str(task["task_id"])
            logger.info("background task %s started: %s", task_id, task["objective"])
            self.active.set()
            try:
                self._execute(task)
            except Exception as error:  # noqa: BLE001 - checkpoint and retry later
                detail = f"{type(error).__name__}: {error}"
                logger.warning("background task %s yielded after: %s", task_id, detail)
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    progress=f"Worker yielded after a transient failure: {detail}",
                    error=detail,
                    status="pending",
                )
                self._wake.wait(1.0)
                self._wake.clear()
            finally:
                self.active.clear()

    def _execute(self, task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        task_started_at = time.monotonic()
        last_progress_at: float | None = None
        messages = copy.deepcopy(task.get("messages") or [])
        if not messages:
            criteria = str(task.get("completion_criteria") or "").strip()
            request = f"<objective>\n{task['objective']}\n</objective>"
            if criteria:
                request += f"\n\n<completion_criteria>\n{criteria}\n</completion_criteria>"
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ]
        active_tools = [
            name
            for name in task.get("active_tools", ["shell"])
            if isinstance(name, str) and name != "background_task"
        ]
        if "shell" not in active_tools:
            active_tools.insert(0, "shell")
        seen: set[str] = set()
        seen_guidance = {
            str(item) for item in task.get("applied_guidance_ids", []) if str(item)
        }

        while not self.stop.is_set():
            current = self.store.get(task_id)
            if current is None or current.get("status") == "cancelled":
                logger.info("background task %s cancelled", task_id)
                return
            added_guidance = _append_guidance(messages, current, seen_guidance)
            if added_guidance:
                logger.info(
                    "background task %s accepted %d conversational update(s)",
                    task_id,
                    added_guidance,
                )
            if not self._wait_for_foreground():
                return
            # This deployment serves text cognition from the same Qwen3-Omni
            # worker as comprehension. TTS evicts it, so a background step
            # must wait for the asynchronous restore rather than racing a
            # dead port. This waits for existing weights; it does not load a
            # separate Ornith/Ollama model.
            if self.await_language is not None:
                self.await_language()
            if self.foreground_active.is_set():
                continue
            schemas = [
                *copy.deepcopy(DISCOVERY_TOOLS),
                *tool_schemas(list(dict.fromkeys(active_tools))[:3]),
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
                "response_modalities": ["text"],
                "speech_mode": "never",
                # Spoken foreground turns optimize for latency, but this worker
                # is doing multi-step planning where an impulsive call can
                # waste far more time than deliberation costs. Native thinking
                # remains a separate backend channel and is never spoken or
                # copied into the durable task transcript.
                "think": True,
                "tools": schemas,
                "portal_auto_tools": False,
                "stream": False,
            }
            data = self._post("/api/chat", payload).json()
            message = data.get("message")
            if not isinstance(message, Mapping):
                raise RuntimeError("background inference returned no assistant message")
            assistant = {
                key: copy.deepcopy(value)
                for key, value in message.items()
                if key in {"role", "content", "tool_calls"}
            }
            assistant["role"] = "assistant"
            messages.append(assistant)
            calls = _tool_calls(data)
            if not calls:
                report = str(message.get("content") or "").strip()
                if not report:
                    raise RuntimeError("background agent returned an empty final report")
                if not _has_concrete_tool_evidence(messages):
                    # A fluent promise or fabricated completion is not work.
                    # Keep it in the checkpoint for audit, explicitly reject
                    # it, then give the model another isolated step in which
                    # the concrete shell schema is still visible.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That completion report is unsupported: no concrete tool "
                                "has run. Continue the accepted task now by calling shell "
                                "or another discovered action tool. Do not report completion "
                                "until the returned evidence verifies the objective."
                            ),
                        }
                    )
                    checkpoint = self.store.checkpoint(
                        task_id,
                        self.owner,
                        messages=messages,
                        active_tools=active_tools,
                        applied_guidance_ids=list(seen_guidance),
                        progress=(
                            "Rejected an unsupported completion report; no action tool "
                            "had executed."
                        ),
                        status="running",
                    )
                    if checkpoint is None or checkpoint.get("status") == "cancelled":
                        return
                    continue
                if report.startswith("TASK_PROGRESS"):
                    spoken = report.removeprefix("TASK_PROGRESS").lstrip(" :\n")
                    if not spoken:
                        spoken = "I reached a useful checkpoint and I’m continuing the task."
                    now = time.monotonic()
                    eligible = (
                        now - task_started_at >= self.progress_after_s
                        and (
                            last_progress_at is None
                            or now - last_progress_at >= self.progress_min_interval_s
                        )
                    )
                    checkpoint = self.store.checkpoint(
                        task_id,
                        self.owner,
                        messages=messages,
                        active_tools=active_tools,
                        applied_guidance_ids=list(seen_guidance),
                        progress=f"Spoken milestone: {spoken}",
                        status="running",
                    )
                    if checkpoint is None or checkpoint.get("status") == "cancelled":
                        return
                    if eligible and self.on_progress is not None:
                        last_progress_at = now
                        self.on_progress(
                            {
                                "task_id": task_id,
                                "status": "running",
                                "result": spoken,
                            }
                        )
                        continuation = "The milestone update was delivered."
                    else:
                        continuation = (
                            "The milestone was checkpointed without interrupting the speaker."
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{continuation} Resume the same accepted task now. "
                                "Continue using tools and do not report completion until "
                                "the objective has been verified."
                            ),
                        }
                    )
                    continue
                latest = self.store.get(task_id)
                if latest is not None and _append_guidance(
                    messages, latest, seen_guidance
                ):
                    checkpoint = self.store.checkpoint(
                        task_id,
                        self.owner,
                        messages=messages,
                        active_tools=active_tools,
                        applied_guidance_ids=list(seen_guidance),
                        progress=(
                            "A newer spoken update arrived before completion; "
                            "reassessing the task."
                        ),
                        status="running",
                    )
                    if checkpoint is None or checkpoint.get("status") == "cancelled":
                        return
                    continue
                status = "completed"
                if report.startswith("TASK_BLOCKED"):
                    status = "blocked"
                    report = report.removeprefix("TASK_BLOCKED").lstrip(" :\n")
                elif report.startswith("TASK_COMPLETE"):
                    report = report.removeprefix("TASK_COMPLETE").lstrip(" :\n")
                if not report:
                    report = (
                        "The background task completed."
                        if status == "completed"
                        else "The background task is blocked."
                    )
                completed = self.store.checkpoint(
                    task_id,
                    self.owner,
                    messages=messages,
                    active_tools=active_tools,
                    applied_guidance_ids=list(seen_guidance),
                    progress=(
                        "Work completed and assessed."
                        if status == "completed"
                        else "Work stopped at a reported blocker."
                    ),
                    result=report,
                    status=status,
                )
                logger.info("background task %s %s: %s", task_id, status, report[:300])
                if completed is not None and self.on_complete is not None:
                    self.on_complete(completed)
                return

            progress_parts: list[str] = []
            for call in calls:
                function = call.get("function")
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, Mapping)
                    else ""
                )
                arguments = _arguments(call)
                fingerprint = hashlib.sha256(
                    f"{name}\0{json.dumps(arguments, sort_keys=True, default=str)}".encode()
                ).hexdigest()
                if fingerprint in seen:
                    result: Any = {
                        "error": "duplicate_tool_call",
                        "message": "This exact call already ran; assess its result and choose a different next step.",
                    }
                else:
                    seen.add(fingerprint)
                    response = self._post(
                        f"/api/tools/{name}/call", {"arguments": arguments}
                    ).json()
                    result = response.get("result", response)
                if name == "tool_search" and isinstance(result, Mapping):
                    available = result.get("available_tools")
                    if isinstance(available, list):
                        active_tools = list(
                            dict.fromkeys(
                                [
                                    "shell",
                                    *(
                                        str(item)
                                        for item in available
                                        if str(item) != "background_task"
                                    ),
                                ]
                            )
                        )[:3]
                elif name and name != "background_task":
                    active_tools = list(dict.fromkeys(["shell", name]))[:3]
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": name or "unknown",
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
                if call.get("id"):
                    tool_message["tool_call_id"] = str(call["id"])
                messages.append(tool_message)
                if name == "shell":
                    command = str(arguments.get("command") or "").replace("\n", " ")
                    exit_code = result.get("exit_code") if isinstance(result, Mapping) else None
                    progress_parts.append(
                        f"Ran shell step ({command[:180]}); exit={exit_code}."
                    )
                else:
                    progress_parts.append(f"Ran {name or 'unknown'} and retained its result.")

            checkpoint = self.store.checkpoint(
                task_id,
                self.owner,
                messages=messages,
                active_tools=active_tools,
                applied_guidance_ids=list(seen_guidance),
                progress=" ".join(progress_parts),
                status="running",
            )
            if checkpoint is None or checkpoint.get("status") == "cancelled":
                return
            # The next reasoning pass is a separate request. That boundary is
            # the scheduling point where a live utterance gets the model first.
            time.sleep(0.05)
