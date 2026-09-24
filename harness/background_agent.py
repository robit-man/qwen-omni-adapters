"""Checkpointed, stepwise agent for work delegated from the live voice turn."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from portal.background_tasks import TERMINAL_STATUSES, BackgroundTaskStore
from portal.tools import DISCOVERY_TOOLS, tool_schemas
from qwen_omni_adapters.context import context_text, context_value
from qwen_omni_adapters.decision_plane import DecisionPlane, DecisionState
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPressure

logger = logging.getLogger(__name__)

MAX_TASK_CONTEXT_MESSAGES = 28
MAX_TASK_CONTEXT_BYTES = 96 * 1024
MAX_RETAINED_TASK_MESSAGES = 12
MAX_TOOL_RESULT_CHARS = 24_000
MAX_CHECKPOINT_REPORT_CHARS = 1_000
MAX_ACTION_ARGUMENT_CHARS = 2_000
MAX_ACTION_OUTCOME_CHARS = 1_200
COMPUTER_ACTION_TOOLS = {"browser_interact", "gui_interact"}

_SENSITIVE_AUDIT_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|secret|token|api[_-]?key)",
    re.IGNORECASE,
)

AGENT_SYSTEM_PROMPT = context_text("prompts", "background_agent_system")


def _task_system_prompt(task: Mapping[str, Any]) -> str:
    """Pin the durable task contract ahead of all renewable worker context."""

    objective = " ".join(str(task.get("objective") or "").split())
    criteria = " ".join(str(task.get("completion_criteria") or "").split())
    contract = [
        "<current_task>",
        f"Objective: {objective}",
    ]
    if criteria:
        contract.append(f"Completion criteria: {criteria}")
    guidance = task.get("guidance")
    if isinstance(guidance, list):
        directions = [
            " ".join(str(item.get("content") or "").split())
            for item in guidance[-8:]
            if isinstance(item, Mapping) and str(item.get("content") or "").strip()
        ]
        if directions:
            contract.append("Later user directions, oldest to newest:")
            contract.extend(f"- {direction}" for direction in directions)
    contract.extend(
        [
            "Keep every action causally relevant to this task. Ignore unrelated topics "
            "from model state or prior work.",
            "</current_task>",
        ]
    )
    task_contract = "\n".join(contract)
    return f"{task_contract}\n\n{AGENT_SYSTEM_PROMPT}"


TASK_CHECKPOINT_TOOL = context_value("control_tools", "task_checkpoint")
TASK_COMPACT_TOOL = context_value("control_tools", "task_compact")
TASK_RECOVERY_TOOL = context_value("control_tools", "task_recovery")


class _ForegroundPreempted(RuntimeError):
    """Background inference was cancelled for an accepted spoken turn."""


class _NonRetryableBackgroundError(RuntimeError):
    """A request is invalid and cannot improve by resending the same payload."""


class _MalformedToolCall(RuntimeError):
    """The model emitted invalid structured arguments and must replan smaller."""


_HTTP_STATUS_PATTERN = re.compile(r"\bHTTP\s+([45]\d\d)\b", re.IGNORECASE)
_TRANSIENT_CLIENT_STATUSES = {408, 409, 425, 429}


def _stream_error(message: str) -> RuntimeError:
    """Classify an upstream error without coupling to backend-specific prose."""

    lowered = message.casefold()
    if "parse tool call arguments as json" in lowered or (
        "tool call arguments" in lowered and "parse error" in lowered
    ):
        return _MalformedToolCall(message)
    match = _HTTP_STATUS_PATTERN.search(message)
    if match is not None:
        status = int(match.group(1))
        if 400 <= status < 500 and status not in _TRANSIENT_CLIENT_STATUSES:
            return _NonRetryableBackgroundError(message)
    return RuntimeError(message)


def _tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [dict(item) for item in calls if isinstance(item, Mapping)]


def _inference_diagnostics(
    response: Mapping[str, Any], step_token_limit: int
) -> dict[str, Any]:
    """Return useful planner telemetry without retaining private reasoning text."""

    message = response.get("message")
    if not isinstance(message, Mapping):
        message = {}
    thinking_chars = len(str(message.get("thinking") or ""))
    content_chars = len(str(message.get("content") or ""))
    tool_call_count = len(_tool_calls(response))
    try:
        eval_count = max(0, int(response.get("eval_count") or 0))
    except (TypeError, ValueError):
        eval_count = 0
    if tool_call_count:
        classification = "structured_action"
    elif eval_count >= step_token_limit:
        classification = "output_budget_exhausted_without_action"
    elif thinking_chars and not content_chars:
        classification = "thinking_only_without_action"
    else:
        classification = "required_action_missing"
    return {
        "classification": classification,
        "done_reason": str(response.get("done_reason") or "")[:80],
        "prompt_eval_count": max(0, int(response.get("prompt_eval_count") or 0)),
        "eval_count": eval_count,
        "step_token_limit": step_token_limit,
        "thinking_chars": thinking_chars,
        "content_chars": content_chars,
        "tool_call_count": tool_call_count,
    }


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


def _repair_malformed_tool_history(messages: list[dict[str, Any]]) -> int:
    """Make retained rejected calls parseable without turning them into evidence."""

    repaired = 0
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            raw = function.get("arguments")
            if not isinstance(raw, str):
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                function["arguments"] = {}
                repaired += 1
                continue
            if not isinstance(parsed, Mapping):
                function["arguments"] = {}
                repaired += 1
    return repaired


def _append_malformed_tool_recovery(messages: list[dict[str, Any]]) -> None:
    """Keep one current repair instruction instead of an unbounded retry pile."""

    directive = context_text("directives", "background_malformed_tool_call")
    messages[:] = [
        message
        for message in messages
        if not (
            message.get("role") == "user"
            and str(message.get("content") or "") == directive
        )
    ]
    messages.append({"role": "user", "content": directive})


def _call_fingerprint(name: str, arguments: Mapping[str, Any]) -> str:
    def normalized(value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.casefold().split())
        if isinstance(value, Mapping):
            return {str(key): normalized(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    return hashlib.sha256(
        f"{name}\0{json.dumps(normalized(arguments), sort_keys=True, default=str)}".encode()
    ).hexdigest()


def _seen_tool_fingerprints(messages: list[dict[str, Any]]) -> set[str]:
    """Rebuild duplicate protection from a task restored after a restart."""

    seen: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = (
                str(function.get("name") or "")
                if isinstance(function, Mapping)
                else ""
            )
            if name:
                seen.add(_call_fingerprint(name, _arguments(call)))
    return seen


def _latest_tool_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Return the most recent external attempt for immediate-loop protection.

    A fingerprint is not a permanent prohibition. The same verification command
    is often exactly what should run after a repair changes the target state.
    Only an unchanged, immediately repeated external call is rejected; any
    intervening concrete attempt gives the worker a new causal state to assess.
    """

    for message in reversed(messages):
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = (
                str(function.get("name") or "")
                if isinstance(function, Mapping)
                else ""
            )
            if name and name not in {
                "task_checkpoint",
                "task_compact",
                "task_recovery",
            }:
                return _call_fingerprint(name, _arguments(call))
    return ""


def _compact_task_messages(
    messages: list[dict[str, Any]],
    task: Mapping[str, Any],
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Turn a long transcript into a fresh chain with its durable state intact."""

    serialized_bytes = len(
        json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8")
    )
    if len(messages) <= 3 or (
        force
        and len(messages) <= MAX_RETAINED_TASK_MESSAGES + 3
        and serialized_bytes <= MAX_TASK_CONTEXT_BYTES // 2
    ) or (
        not force
        and len(messages) <= MAX_TASK_CONTEXT_MESSAGES
        and serialized_bytes <= MAX_TASK_CONTEXT_BYTES
    ):
        return messages
    head = copy.deepcopy(messages[:2])
    tail_start = max(2, len(messages) - MAX_RETAINED_TASK_MESSAGES)
    if (
        tail_start > 2
        and messages[tail_start].get("role") == "tool"
        and messages[tail_start - 1].get("role") == "assistant"
    ):
        tail_start -= 1
    progress = task.get("progress")
    progress_lines = (
        [f"- {str(item)[:500]}" for item in progress[-10:]]
        if isinstance(progress, list)
        else []
    )
    guidance = task.get("guidance")
    guidance_lines = []
    if isinstance(guidance, list):
        guidance_lines = [
            f"- {str(item.get('content') or '')[:500]}"
            for item in guidance[-8:]
            if isinstance(item, Mapping) and str(item.get("content") or "").strip()
        ]
    tools = task.get("tools_used")
    tool_names = ", ".join(str(item) for item in tools) if isinstance(tools, list) else ""
    actions = task.get("actions")
    action_lines: list[str] = []
    if isinstance(actions, list):
        for action in actions[-10:]:
            if not isinstance(action, Mapping):
                continue
            outcome = " ".join(str(action.get("outcome") or "").split())[:280]
            action_lines.append(
                "- "
                + str(action.get("call_id") or "unknown")[:80]
                + " | "
                + str(action.get("tool") or "unknown")[:80]
                + (" | succeeded" if action.get("ok") is True else " | failed")
                + (f" | {outcome}" if outcome else "")
            )
    sections = [
        "<retained_checkpoint>",
        "Older detailed reasoning/tool rounds were compacted. Continue from the objective "
        "and the retained concrete state below; do not repeat completed or failed calls.",
    ]
    if progress_lines:
        sections.extend(["Recent durable checkpoints:", *progress_lines])
    if guidance_lines:
        sections.extend(["Spoken guidance that remains authoritative:", *guidance_lines])
    if tool_names:
        sections.append(f"Tools already used: {tool_names}")
    if action_lines:
        sections.extend(
            [
                "Recent durable action receipts (orientation only; reverify any "
                "terminal criterion with fresh tool evidence):",
                *action_lines,
            ]
        )
    sections.append("</retained_checkpoint>")
    checkpoint = {"role": "user", "content": "\n".join(sections)}
    tail = copy.deepcopy(messages[tail_start:])
    for message in tail:
        if message.get("role") != "tool":
            continue
        try:
            value = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            value = str(message.get("content") or "")
        message["content"] = json.dumps(
            _bounded_tool_result(value), ensure_ascii=False, default=str
        )
    return [*head, checkpoint, *tail]


def _computer_action_state(task: Mapping[str, Any]) -> str:
    """Render durable orientation without replaying the whole motor transcript."""

    sections = [
        "<computer_action_state>",
        "This is a compact orientation summary, not current visual evidence. "
        "Ground the next pointer action only in the newest returned screenshot and "
        "its coordinate_space metadata; old coordinates are not reusable.",
    ]
    progress = task.get("progress")
    if isinstance(progress, list) and progress:
        sections.append("Recent durable progress:")
        sections.extend(
            f"- {' '.join(str(item).split())[:500]}" for item in progress[-4:]
        )
    actions = task.get("actions")
    if isinstance(actions, list) and actions:
        sections.append("Recent action receipts:")
        for action in actions[-6:]:
            if not isinstance(action, Mapping):
                continue
            tool_name = str(action.get("tool") or "unknown")[:80]
            arguments = " ".join(str(action.get("arguments") or "").split())[:500]
            outcome = " ".join(str(action.get("outcome") or "").split())[:500]
            if tool_name in COMPUTER_ACTION_TOOLS:
                # Exact coordinates and element IDs expire with their image.
                # Durable orientation needs the kind of action and page/window
                # identity, not stale motor targets.
                try:
                    parsed_arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    parsed_arguments = {}
                arguments = f"action={str(parsed_arguments.get('action') or 'unknown')[:80]}"
                try:
                    parsed_outcome = json.loads(outcome)
                except (TypeError, ValueError):
                    parsed_outcome = {}
                if isinstance(parsed_outcome, Mapping):
                    orientation = {
                        key: parsed_outcome[key]
                        for key in {
                            "url",
                            "title",
                            "rendered",
                            "challenge",
                            "error",
                            "message",
                        }
                        if key in parsed_outcome
                    }
                    outcome = json.dumps(orientation, ensure_ascii=False)[:500]
            sections.append(
                "- "
                + str(action.get("call_id") or "unknown")[:80]
                + " | "
                + tool_name
                + (" | succeeded" if action.get("ok") is True else " | failed")
                + (f" | args={arguments}" if arguments else "")
                + (f" | result={outcome}" if outcome else "")
            )
    sections.append("</computer_action_state>")
    return "\n".join(sections)


def _computer_action_messages(
    messages: list[dict[str, Any]],
    task: Mapping[str, Any],
    active_tools: list[str],
    *,
    recovery_required: bool,
) -> list[dict[str, Any]]:
    """Build a short visual-motor request while preserving durable task state.

    The task transcript remains authoritative and is still checkpointed in full.
    Only the inference view is narrowed: objective/policy, a bounded receipt
    summary, and at most the two newest computer-use cycles. This prevents a
    miss several screenshots ago from competing with the current coordinate
    frame without discarding long-horizon progress.
    """

    if recovery_required or not COMPUTER_ACTION_TOOLS.intersection(active_tools):
        return messages
    motor_rounds: list[int] = []
    discovery_rounds: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        names: set[str] = set()
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if isinstance(function, Mapping):
                names.add(str(function.get("name") or ""))
        if names.intersection(COMPUTER_ACTION_TOOLS):
            motor_rounds.append(index)
        elif "tool_search" in names:
            discovery_rounds.append(index)
    if motor_rounds:
        tail_start = motor_rounds[-2] if len(motor_rounds) > 1 else motor_rounds[-1]
    elif discovery_rounds:
        tail_start = discovery_rounds[-1]
    else:
        return messages
    head = copy.deepcopy(messages[:2])
    state = {"role": "user", "content": _computer_action_state(task)}
    scoped = [*head, state, *copy.deepcopy(messages[tail_start:])]
    newest_image_retained = False
    for message in reversed(scoped):
        images = message.get("images")
        if not isinstance(images, list) or not images:
            continue
        if not newest_image_retained:
            newest_image_retained = True
            continue
        message.pop("images", None)
        message["content"] = (
            str(message.get("content") or "")
            + "\n[An older screenshot was superseded by the newest visual evidence.]"
        ).strip()
    return scoped


def _context_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "messages": len(messages),
        "bytes": len(
            json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8")
        ),
    }


def _compaction_available(messages: list[dict[str, Any]]) -> bool:
    metrics = _context_metrics(messages)
    large_enough = (
        metrics["messages"] > MAX_RETAINED_TASK_MESSAGES + 3
        or metrics["bytes"] > MAX_TASK_CONTEXT_BYTES // 2
    )
    if not large_enough:
        return False

    # Hysteresis: a compaction receipt and the control messages around it can
    # leave the fresh chain just over the low-water message threshold. Do not
    # offer compaction again until a real external action has added new
    # evidence. Otherwise a small deterministic model can select the visible
    # maintenance tool forever instead of returning to the task.
    latest_compaction = -1
    latest_external_result = -1
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if name == "task_compact":
            latest_compaction = index
        elif name not in {
            "",
            "task_checkpoint",
            "task_recovery",
            "tool_search",
        } and not _is_duplicate_tool_result(message):
            latest_external_result = index
    return latest_compaction <= latest_external_result


def _compaction_receipt(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    task: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    evidence_ids = list(_tool_evidence(after))[-16:]
    guidance = task.get("guidance")
    return {
        "schema": "robit.omni.task-compaction.v1",
        "reason": reason,
        "before": _context_metrics(before),
        "after": _context_metrics(after),
        "retained": {
            "objective": bool(str(task.get("objective") or "").strip()),
            "completion_criteria": bool(
                str(task.get("completion_criteria") or "").strip()
            ),
            "latest_guidance_count": min(
                8, len(guidance) if isinstance(guidance, list) else 0
            ),
            "fresh_evidence_ids": evidence_ids,
        },
    }


def _result_digest(name: str, result: Any) -> str:
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{name}\0{rendered}".encode()).hexdigest()


def _bounded_tool_result(result: Any) -> Any:
    """Keep durable evidence useful without letting one tool inflate context."""

    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) <= MAX_TOOL_RESULT_CHARS:
        return result
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    if not isinstance(result, Mapping):
        return {
            "truncated": True,
            "original_sha256": digest,
            "preview": rendered[: MAX_TOOL_RESULT_CHARS - 200],
        }
    bounded: dict[str, Any] = {
        key: value
        for key, value in result.items()
        if key
        in {
            "error",
            "message",
            "exit_code",
            "timed_out",
            "blocked",
            "challenge",
            "url",
            "title",
            "rendered",
        }
    }
    remaining = MAX_TOOL_RESULT_CHARS - len(
        json.dumps(bounded, ensure_ascii=False, default=str)
    ) - 300
    for key, value in result.items():
        if key in bounded or remaining <= 0:
            continue
        text = json.dumps(value, ensure_ascii=False, default=str)
        bounded[str(key)] = text[:remaining]
        remaining -= min(len(text), remaining)
    bounded["truncated"] = True
    bounded["original_sha256"] = digest
    return bounded


def _audit_json(value: Any, limit: int) -> str:
    """Render useful local action detail without retaining secrets or bulk payloads."""

    def clean(item: Any, *, key: str = "", depth: int = 0) -> Any:
        if _SENSITIVE_AUDIT_KEY.search(key):
            return "[redacted]"
        if depth >= 6:
            return "[nested value omitted]"
        if isinstance(item, Mapping):
            return {
                str(child_key): clean(
                    child_value, key=str(child_key), depth=depth + 1
                )
                for child_key, child_value in list(item.items())[:48]
                if str(child_key) != "screenshot"
            }
        if isinstance(item, list):
            return [clean(child, depth=depth + 1) for child in item[:32]]
        if isinstance(item, str):
            if key.casefold() == "stdin":
                return f"[omitted {len(item)} characters]"
            if item.startswith("data:") or len(item) > 1_000:
                return f"{item[:320]}… [{len(item)} characters]"
        return item

    rendered = json.dumps(
        clean(value), ensure_ascii=False, sort_keys=True, default=str
    )
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: limit - 24].rstrip()}… [{len(rendered)} chars]"


def _tool_evidence(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        evidence_id = str(message.get("tool_call_id") or "").strip()
        name = str(message.get("tool_name") or "").strip()
        if not evidence_id or not name or name in {
            "tool_search",
            "task_checkpoint",
            "task_compact",
            "task_recovery",
        }:
            continue
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            result = {}
        if isinstance(result, Mapping) and result.get("error") == "duplicate_tool_call":
            continue
        evidence[evidence_id] = {"name": name, "result": result}
    return evidence


def _freshest_evidence_id(messages: list[dict[str, Any]]) -> str:
    """Return the newest concrete, non-control result available to the worker."""

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        evidence_id = str(message.get("tool_call_id") or "").strip()
        name = str(message.get("tool_name") or "").strip()
        if not evidence_id or name in {
            "",
            "tool_search",
            "task_checkpoint",
            "task_recovery",
        }:
            continue
        if _is_duplicate_tool_result(message):
            continue
        return evidence_id
    return ""


def _checkpoint_available(messages: list[dict[str, Any]]) -> bool:
    """Allow one checkpoint attempt only after a newer concrete tool result."""

    latest_action = -1
    latest_control = -1
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user" and "<task_update " in str(message.get("content") or ""):
            latest_control = index
            continue
        if role != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if name in {"task_checkpoint", "task_recovery"}:
            latest_control = index
        elif _is_duplicate_tool_result(message):
            # A locally rejected replay is control feedback, not new evidence.
            # Require a materially different real action before checkpointing.
            latest_control = index
        elif name and name != "tool_search":
            latest_action = index
    return latest_action > latest_control


def _recovery_required(messages: list[dict[str, Any]]) -> bool:
    """Whether a capability failure still needs a typed recovery transition."""

    pending = False
    for message in messages:
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            continue
        if name == "task_recovery":
            if isinstance(result, Mapping) and result.get("accepted") is True:
                pending = False
            continue
        if (
            isinstance(result, Mapping)
            and result.get("disposition") == "change_capability"
            and result.get("task_blocked") is False
        ):
            pending = True
    return pending


def _is_duplicate_tool_result(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "tool":
        return False
    try:
        result = json.loads(str(message.get("content") or "{}"))
    except ValueError:
        return False
    return isinstance(result, Mapping) and result.get("error") == "duplicate_tool_call"


def _result_failed_or_blocked(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("error") or result.get("timed_out") is True:
        return True
    if result.get("blocked") is True or result.get("challenge") is True:
        return True
    exit_code = result.get("exit_code")
    return exit_code is not None and exit_code != 0


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
        portal_session_id: str | None = None,
        foreground_active: threading.Event,
        stop: threading.Event,
        token_reader: Callable[[], str] | None = None,
        await_language: Callable[[], None] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        progress_after_s: float = 45.0,
        progress_min_interval_s: float = 120.0,
        retry_initial_s: float = 1.0,
        retry_max_s: float = 30.0,
        request_timeout_s: float = 300.0,
        step_token_limit: int | None = None,
        memory_governor: MemoryGovernor | None = None,
        max_slice_rounds: int = 12,
        max_slice_tool_calls: int = 16,
        max_slice_stalls: int = 3,
        slice_backoff_s: float = 10.0,
        client: httpx.Client | None = None,
        decision_plane: DecisionPlane | None = None,
    ) -> None:
        self.store = store
        self.portal_url = portal_url.rstrip("/")
        self.token = token
        self.model = model
        self.foreground_active = foreground_active
        self.stop = stop
        self.token_reader = token_reader
        self.await_language = await_language
        self.decision_plane = decision_plane
        if self.decision_plane is None and os.environ.get(
            "OMNI_DECISION_PLANE_ENABLED", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}:
            self.decision_plane = DecisionPlane.from_environment()
        self._decision_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-background-system1")
            if self.decision_plane is not None
            else None
        )
        self.on_complete = on_complete
        self.on_progress = on_progress
        self.progress_after_s = max(0.0, progress_after_s)
        self.progress_min_interval_s = max(0.0, progress_min_interval_s)
        self.retry_initial_s = max(0.01, retry_initial_s)
        self.retry_max_s = max(self.retry_initial_s, retry_max_s)
        configured_step_limit = step_token_limit
        if configured_step_limit is None:
            try:
                configured_step_limit = int(
                    os.environ.get("OMNI_BACKGROUND_STEP_TOKENS", "768")
                )
            except ValueError:
                configured_step_limit = 768
        self.step_token_limit = max(128, min(4096, configured_step_limit))
        self.memory_governor = memory_governor
        self.max_slice_rounds = max(2, int(max_slice_rounds))
        self.max_slice_tool_calls = max(1, int(max_slice_tool_calls))
        self.max_slice_stalls = max(1, int(max_slice_stalls))
        self.slice_backoff_s = max(0.1, float(slice_backoff_s))
        self._failures: dict[str, int] = {}
        self._resource_deferrals: dict[str, int] = {}
        self._slice_deferrals: dict[str, int] = {}
        self._quiesced_tasks: set[str] = set()
        self.owner = f"voice-agent-{secrets.token_hex(8)}"
        self.active = threading.Event()
        self._wake = threading.Event()
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(request_timeout_s),
            cookies={
                "omni_portal_session": portal_session_id or secrets.token_urlsafe(24)
            },
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
        if self._decision_executor is not None:
            self._decision_executor.shutdown(wait=False, cancel_futures=True)
        if self.decision_plane is not None:
            self.decision_plane.close()

    def _observe_decision(self, wave: str, state: DecisionState) -> None:
        plane = self.decision_plane
        if plane is None:
            return

        def run() -> None:
            try:
                result = plane.evaluate(state=state, wave=wave)
                logger.info(
                    "background System-1 wave=%s latency=%.1fms decisions=%s shadow=%s",
                    wave,
                    result.latency_ms,
                    ",".join(
                        f"{name}:{item.value}@{item.confidence:.3f}"
                        for name, item in result.results.items()
                    ),
                    plane.shadow_mode,
                )
            except Exception as exc:  # noqa: BLE001 - optimizer failure is non-fatal
                logger.warning("background System-1 wave %s unavailable: %s", wave, exc)

        if plane.shadow_mode and self._decision_executor is not None:
            self._decision_executor.submit(run)
        else:
            run()

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

    def _record_action(
        self,
        task_id: str,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        historical: bool = False,
    ) -> None:
        try:
            self.store.record_action(
                task_id,
                self.owner,
                call_id=call_id,
                tool=name or "unknown",
                arguments=_audit_json(arguments, MAX_ACTION_ARGUMENT_CHARS),
                outcome=_audit_json(result, MAX_ACTION_OUTCOME_CHARS),
                ok=not _result_failed_or_blocked(result),
                recorded_at=0 if historical else None,
            )
        except Exception as error:  # noqa: BLE001 - auditing must not stop the task
            logger.warning("could not audit background tool call %s: %s", name, error)

    def _restore_action_audit(
        self, task_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Recover retained calls from tasks created before action auditing existed."""

        outcomes: dict[str, Any] = {}
        for message in messages:
            if message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if not call_id:
                continue
            try:
                outcomes[call_id] = json.loads(str(message.get("content") or "{}"))
            except ValueError:
                outcomes[call_id] = str(message.get("content") or "")
        for message in messages:
            calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, Mapping)
                    else ""
                )
                call_id = str(call.get("id") or "")
                if name and call_id in outcomes:
                    self._record_action(
                        task_id,
                        call_id,
                        name,
                        _arguments(call),
                        outcomes[call_id],
                        historical=True,
                    )

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
        if self.memory_governor is not None:
            # The portal owns per-tool admission and emergency cancellation.
            # Reaching that control point is itself bounded and must remain
            # possible inside the soft-to-hard safety band.
            self.memory_governor.require_hard_floor(
                f"background tool control {path.rsplit('/', 2)[-2]}"
            )
        response = self._client.post(
            f"{self.portal_url}{path}", json=dict(payload), headers=self._headers()
        )
        if response.status_code == 401 and self._refresh_token():
            response = self._client.post(
                f"{self.portal_url}{path}", json=dict(payload), headers=self._headers()
            )
        response.raise_for_status()
        if self.memory_governor is not None and self.memory_governor.under_hard_pressure():
            raise MemoryPressure(
                "background tool result",
                self.memory_governor.available_gib(),
                self.memory_governor.policy.hard_floor_gib,
            )
        return response

    def _chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run cancellable background inference through the streaming route."""

        body = dict(payload)
        body["stream"] = True
        for attempt in range(2):
            if self.memory_governor is not None:
                image_turn = any(
                    isinstance(message, Mapping) and bool(message.get("images"))
                    for message in body.get("messages", [])
                )
                reserve = (
                    self.memory_governor.policy.operation_reserve_gib * 1.5
                    if image_turn
                    else None
                )
                if image_turn:
                    self.memory_governor.require(
                        "background inference", reserve_gib=reserve
                    )
                else:
                    # A compacted text continuation reuses the resident trunk.
                    # Its watchdog still cancels at the hard floor.
                    self.memory_governor.require_hard_floor(
                        "background inference continuation"
                    )
            with self._client.stream(
                "POST",
                f"{self.portal_url}/api/chat/stream",
                json=body,
                headers=self._headers(),
            ) as response:
                if response.status_code == 401 and attempt == 0 and self._refresh_token():
                    continue
                response.raise_for_status()
                done = threading.Event()

                def cancel() -> None:
                    response.close()

                watcher = None
                tripped = threading.Event()
                if self.memory_governor is not None:
                    watcher, tripped = self.memory_governor.watch(
                        "background-inference", cancel, done
                    )
                stream_items: queue.Queue[tuple[str, Any]] = queue.Queue()
                content_type = response.headers.get("content-type", "")

                def read_stream(
                    stream: httpx.Response = response,
                    response_content_type: str = content_type,
                    items: queue.Queue[tuple[str, Any]] = stream_items,
                ) -> None:
                    try:
                        if "application/json" in response_content_type:
                            stream.read()
                            items.put(("json", stream.json()))
                        else:
                            for line in stream.iter_lines():
                                items.put(("line", line))
                    except Exception as error:  # noqa: BLE001 - forwarded below
                        items.put(("error", error))
                    finally:
                        items.put(("done", None))

                stream_reader = threading.Thread(
                    target=read_stream,
                    name="omni-background-stream-reader",
                    daemon=True,
                )
                stream_reader.start()
                try:
                    final: dict[str, Any] | None = None
                    while True:
                        if self.foreground_active.is_set():
                            response.close()
                            raise _ForegroundPreempted(
                                "background inference yielded to foreground speech"
                            )
                        if tripped.is_set() and self.memory_governor is not None:
                            response.close()
                            raise MemoryPressure(
                                "background inference",
                                self.memory_governor.available_gib(),
                                self.memory_governor.policy.hard_floor_gib,
                            )
                        try:
                            kind, value = stream_items.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if kind == "done":
                            break
                        if kind == "error":
                            if isinstance(value, Exception):
                                raise value
                            raise RuntimeError("background inference stream failed")
                        if kind == "json":
                            if not isinstance(value, dict):
                                raise RuntimeError(
                                    "background inference returned invalid JSON"
                                )
                            return value
                        line = str(value)
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, Mapping):
                            continue
                        if event.get("type") == "error":
                            raise _stream_error(
                                str(event.get("error") or "stream error")
                            )
                        if event.get("type") == "final" and isinstance(
                            event.get("response"), Mapping
                        ):
                            final = dict(event["response"])
                    if final is None:
                        raise RuntimeError("background inference stream ended without a final")
                    return final
                except httpx.HTTPError as error:
                    if self.foreground_active.is_set():
                        raise _ForegroundPreempted(
                            "background inference yielded to foreground speech"
                        ) from error
                    if tripped.is_set() and self.memory_governor is not None:
                        raise MemoryPressure(
                            "background inference",
                            self.memory_governor.available_gib(),
                            self.memory_governor.policy.hard_floor_gib,
                        ) from error
                    raise
                finally:
                    done.set()
                    response.close()
                    stream_reader.join(timeout=1.0)
                    if watcher is not None:
                        watcher.join(timeout=1.0)
        raise RuntimeError("background inference authorization failed")

    def _wait_for_foreground(self) -> bool:
        yielded = False
        while self.foreground_active.is_set() and not self.stop.is_set():
            if self.active.is_set() and not yielded:
                logger.info(
                    "background work yielded to a live human interjection; task context "
                    "remains checkpointed"
                )
                yielded = True
            self._wake.wait(0.2)
            self._wake.clear()
        if yielded and not self.stop.is_set():
            logger.info("live interjection finished; resuming checkpointed background work")
        return not self.stop.is_set()

    def _run(self) -> None:
        capacity_deferrals = 0
        while not self.stop.is_set():
            if not self._wait_for_foreground():
                return
            if not self.has_work():
                capacity_deferrals = 0
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if self.memory_governor is not None:
                try:
                    # Claiming and compacting are bounded control work. Using
                    # the soft floor here made compaction unreachable precisely
                    # when it was needed.
                    self.memory_governor.require_hard_floor(
                        "background scheduler control"
                    )
                except MemoryPressure as error:
                    capacity_deferrals += 1
                    delay = min(
                        self.memory_governor.policy.wait_max_s,
                        self.memory_governor.policy.wait_initial_s
                        * (2 ** min(capacity_deferrals - 1, 8)),
                    )
                    if capacity_deferrals == 1 or capacity_deferrals & (
                        capacity_deferrals - 1
                    ) == 0:
                        logger.warning(
                            "background scheduler waiting %.1fs for runtime capacity: %s",
                            delay,
                            error,
                        )
                    self._wake.wait(delay)
                    self._wake.clear()
                    continue
                capacity_deferrals = 0
            task = self.store.claim_next(
                self.owner, exclude_task_ids=self._quiesced_tasks
            )
            if task is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            task_id = str(task["task_id"])
            logger.info("background task %s started: %s", task_id, task["objective"])
            self.active.set()
            try:
                available = (
                    self.memory_governor.available_gib()
                    if self.memory_governor is not None
                    else float("inf")
                )
                constrained = bool(
                    self.memory_governor is not None
                    and available < self.memory_governor.required_gib()
                )
                retained = copy.deepcopy(task.get("messages") or [])
                compacted = _compact_task_messages(
                    retained,
                    task,
                    force=constrained,
                )
                if compacted != retained:
                    receipt = _compaction_receipt(
                        retained,
                        compacted,
                        task,
                        reason=(
                            "memory_pressure_pre_admission"
                            if constrained
                            else "proactive_context_limit"
                        ),
                    )
                    self.store.compact_context(
                        task_id,
                        self.owner,
                        messages=compacted,
                        receipt=receipt,
                    )
                    task["messages"] = compacted
                    task["compaction"] = receipt
                    logger.info(
                        "background task %s compacted before admission: %s",
                        task_id,
                        json.dumps(receipt, sort_keys=True),
                    )
                if self.await_language is not None:
                    self.store.update_stage(
                        task_id,
                        self.owner,
                        context_text("task_stages", "waiting_language"),
                    )
                    self.await_language()
                if self.memory_governor is not None:
                    self.memory_governor.require_hard_floor(
                        "background task continuation"
                    )
                self._execute(task)
                self._failures.pop(task_id, None)
                self._resource_deferrals.pop(task_id, None)
            except _ForegroundPreempted:
                logger.info("background task %s preempted by a spoken turn", task_id)
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    current_stage=context_text(
                        "task_stages", "waiting_spoken_turn"
                    ),
                    status="pending",
                )
            except MemoryPressure as error:
                deferrals = self._resource_deferrals.get(task_id, 0) + 1
                self._resource_deferrals[task_id] = deferrals
                delay = min(
                    self.memory_governor.policy.wait_max_s
                    if self.memory_governor is not None
                    else self.retry_max_s,
                    (self.memory_governor.policy.wait_initial_s
                    if self.memory_governor is not None
                    else self.retry_initial_s)
                    * (2 ** min(deferrals - 1, 8)),
                )
                if deferrals == 1 or deferrals & (deferrals - 1) == 0:
                    logger.warning(
                        "background task %s deferred by the runtime memory governor; "
                        "retrying in %.1fs: %s",
                        task_id,
                        delay,
                        error,
                    )
                # Resource pressure is scheduler state. It is deliberately not
                # added to messages, progress, evidence, or the task's error.
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    current_stage=context_text("task_stages", "waiting_capacity"),
                    status="pending",
                )
                self._wake.wait(delay)
                self._wake.clear()
            except _NonRetryableBackgroundError as error:
                detail = f"{type(error).__name__}: {error}"
                self._quiesced_tasks.add(task_id)
                logger.error(
                    "background task %s stopped retrying an invalid backend request: %s",
                    task_id,
                    error,
                )
                # Preserve the accepted task without spinning on an identical
                # invalid request. A corrected worker starts with an empty
                # quiescence set and resumes this pending task after restart.
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    current_stage=context_text(
                        "task_stages", "waiting_corrected_worker"
                    ),
                    error=detail,
                    status="pending",
                )
            except Exception as error:  # noqa: BLE001 - checkpoint and retry later
                detail = f"{type(error).__name__}: {error}"
                failures = self._failures.get(task_id, 0) + 1
                self._failures[task_id] = failures
                delay = min(
                    self.retry_max_s,
                    self.retry_initial_s * (2 ** min(failures - 1, 10)),
                )
                logger.warning(
                    "background task %s yielded after: %s; retrying in %.1fs",
                    task_id,
                    detail,
                    delay,
                )
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    current_stage=context_text(
                        "task_stages", "retry_backend"
                    ).format(delay=delay),
                    error=detail,
                    status="pending",
                )
                self._wake.wait(delay)
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
                {"role": "system", "content": _task_system_prompt(task)},
                {"role": "user", "content": request},
            ]
        elif messages[0].get("role") == "system":
            # Durable tasks keep evidence and calls across service restarts, but
            # worker policy is executable code, not frozen task data. Always
            # apply the current generic policy when resuming retained work.
            messages[0] = {
                "role": "system",
                "content": _task_system_prompt(task),
            }
        else:
            messages.insert(
                0,
                {"role": "system", "content": _task_system_prompt(task)},
            )
        self._restore_action_audit(task_id, messages)
        seen = {
            *_seen_tool_fingerprints(messages),
            *(str(value) for value in task.get("tool_fingerprints", []) if value),
        }
        last_tool_fingerprint = _latest_tool_fingerprint(messages)
        result_digests = {
            str(value) for value in task.get("result_digests", []) if value
        }
        compacted = _compact_task_messages(messages, task)
        if len(compacted) < len(messages):
            logger.info(
                "background task %s compacted %d retained messages into %d for a fresh "
                "checkpoint chain",
                task_id,
                len(messages),
                len(compacted),
            )
            messages = compacted
        active_tools = [
            name
            for name in task.get("active_tools", [])
            if isinstance(name, str) and name != "background_task"
        ]
        tools_used = [
            name for name in task.get("tools_used", []) if isinstance(name, str) and name
        ]
        suppress_discovery = False
        seen_guidance = {
            str(item) for item in task.get("applied_guidance_ids", []) if str(item)
        }
        slice_rounds = 0
        slice_tool_calls = 0
        stalls = 0
        recovery_required = _recovery_required(messages)
        self._observe_decision(
            "input_routing",
            DecisionState(
                user_request=str(task.get("objective") or ""),
                current_goal=str(task.get("objective") or ""),
                current_phase="background_admission",
                pending_requirements=(
                    str(task.get("completion_criteria") or "verified completion"),
                ),
                authorization_state={"durable_task_accepted": True},
            ),
        )
        while not self.stop.is_set():
            if (
                slice_rounds >= self.max_slice_rounds
                or slice_tool_calls >= self.max_slice_tool_calls
                or stalls >= self.max_slice_stalls
            ):
                deferrals = self._slice_deferrals.get(task_id, 0) + 1
                self._slice_deferrals[task_id] = deferrals
                delay = min(
                    self.retry_max_s,
                    self.slice_backoff_s * (2 ** min(deferrals - 1, 5)),
                )
                self.store.checkpoint(
                    task_id,
                    self.owner,
                    messages=messages,
                    active_tools=active_tools,
                    tools_used=tools_used,
                    applied_guidance_ids=list(seen_guidance),
                    tool_fingerprints=list(seen),
                    result_digests=list(result_digests),
                    current_stage=context_text(
                        "task_stages", "yielded_slice"
                    ).format(delay=delay),
                    status="pending",
                )
                self._wake.wait(delay)
                self._wake.clear()
                return
            current = self.store.get(task_id)
            if current is None or current.get("status") == "cancelled":
                logger.info("background task %s cancelled", task_id)
                return
            added_guidance = _append_guidance(messages, current, seen_guidance)
            if added_guidance:
                active_tools = []
                suppress_discovery = False
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
                self.store.update_stage(
                    task_id,
                    self.owner,
                    context_text("task_stages", "waiting_language"),
                )
                self.await_language()
            if self.foreground_active.is_set():
                continue
            self.store.update_stage(
                task_id,
                self.owner,
                context_text("task_stages", "planning"),
            )
            repaired_history = _repair_malformed_tool_history(messages)
            if repaired_history:
                _append_malformed_tool_recovery(messages)
                logger.warning(
                    "background task %s repaired %d malformed retained tool call(s) "
                    "before replanning",
                    task_id,
                    repaired_history,
                )
            can_checkpoint = _checkpoint_available(messages) and not recovery_required
            schemas = (
                [copy.deepcopy(TASK_RECOVERY_TOOL)]
                if recovery_required
                else [
                    *(
                        [copy.deepcopy(TASK_COMPACT_TOOL)]
                        if _compaction_available(messages)
                        else []
                    ),
                    *(
                        [copy.deepcopy(TASK_CHECKPOINT_TOOL)]
                        if can_checkpoint
                        else []
                    ),
                    *([] if suppress_discovery else copy.deepcopy(DISCOVERY_TOOLS)),
                    *tool_schemas(list(dict.fromkeys(active_tools))[:3]),
                ]
            )
            # A discovery result already chose the capability. Once a concrete
            # contract is active, every round is an action/checkpoint round,
            # not an open-ended deliberation round. Keep native thinking for
            # initial planning only; otherwise a small model can consume the
            # entire output ceiling in the private channel without ever
            # emitting the required structured call.
            action_after_discovery = bool(
                messages
                and messages[-1].get("role") == "tool"
                and messages[-1].get("tool_name") == "tool_search"
            )
            structured_action_phase = action_after_discovery or bool(active_tools)
            inference_messages = _computer_action_messages(
                messages,
                current,
                active_tools,
                recovery_required=recovery_required,
            )
            if inference_messages is not messages:
                durable_metrics = _context_metrics(messages)
                scoped_metrics = _context_metrics(inference_messages)
                logger.info(
                    "background task %s computer-action scope: "
                    "messages=%d->%d bytes=%d->%d",
                    task_id,
                    durable_metrics["messages"],
                    scoped_metrics["messages"],
                    durable_metrics["bytes"],
                    scoped_metrics["bytes"],
                )
            payload = {
                "model": self.model,
                "messages": inference_messages,
                "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
                "response_modalities": ["text"],
                "speech_mode": "never",
                # Spoken foreground turns optimize for latency, but this worker
                # is doing multi-step planning where an impulsive call can
                # waste far more time than deliberation costs. Native thinking
                # remains a separate backend channel and is never spoken or
                # copied into the durable task transcript.
                "think": not structured_action_phase,
                "options": {"num_predict": self.step_token_limit},
                # Background rounds are independently checkpointed. Reusing a
                # llama.cpp prompt slot keeps discarded history resident and
                # defeats transcript compaction on unified-memory Jetsons.
                "cache_prompt": False,
                "tools": schemas,
                # A background worker exists to act. Before a concrete tool
                # result exists, checkpointing is unavailable; afterwards a
                # required choice is either the next action or an
                # evidence-backed checkpoint. Free prose cannot strand a
                # durable task between those states.
                "tool_choice": "required",
                "portal_auto_tools": False,
                "stream": False,
            }
            try:
                data = self._chat(payload)
            except _MalformedToolCall as error:
                # Replaying identical state makes a deterministic small model
                # reproduce the same oversized/truncated JSON forever. Persist
                # a task-neutral repair instruction and replan inside this
                # slice; the rejected generation never becomes task evidence.
                slice_rounds += 1
                stalls += 1
                _repair_malformed_tool_history(messages)
                _append_malformed_tool_recovery(messages)
                checkpoint = self.store.checkpoint(
                    task_id,
                    self.owner,
                    messages=messages,
                    active_tools=active_tools,
                    tools_used=tools_used,
                    applied_guidance_ids=list(seen_guidance),
                    tool_fingerprints=list(seen),
                    result_digests=list(result_digests),
                    current_stage=context_text("task_stages", "replanning"),
                    status="running",
                )
                if checkpoint is None or checkpoint.get("status") == "cancelled":
                    return
                logger.warning(
                    "background task %s rejected malformed structured arguments; "
                    "replanning with a smaller-call constraint: %s",
                    task_id,
                    error,
                )
                continue
            slice_rounds += 1
            # Browser screenshots are one-pass perception evidence. Once the
            # model has inspected one, retain the DOM/tool summary but never
            # checkpoint megabytes of base64 into the long-horizon transcript.
            for retained in messages:
                if retained.pop("images", None):
                    retained["content"] = (
                        str(retained.get("content") or "")
                        + "\n[The rendered screenshot was inspected in this reasoning pass.]"
                    ).strip()
            message = data.get("message")
            if not isinstance(message, Mapping):
                raise RuntimeError("background inference returned no assistant message")
            logger.info(
                "background task %s inference diagnostics: %s",
                task_id,
                json.dumps(
                    _inference_diagnostics(data, self.step_token_limit),
                    sort_keys=True,
                ),
            )
            assistant = {
                key: copy.deepcopy(value)
                for key, value in message.items()
                if key in {"role", "content", "tool_calls"}
            }
            assistant["role"] = "assistant"
            messages.append(assistant)
            calls = _tool_calls(data)
            if not calls:
                stalls += 1
                messages.append(
                    {
                        "role": "user",
                        "content": context_text(
                            "directives", "background_no_call"
                        ),
                    }
                )
                continue

            self._observe_decision(
                "pre_action",
                DecisionState(
                    user_request=str(task.get("objective") or ""),
                    current_goal=str(task.get("objective") or ""),
                    current_phase="pre_action",
                    last_action={
                        "calls": [
                            {
                                "name": str(call.get("function", {}).get("name") or "")
                                if isinstance(call.get("function"), Mapping)
                                else "",
                                "arguments": _arguments(call),
                            }
                            for call in calls
                        ]
                    },
                    pending_requirements=(
                        str(task.get("completion_criteria") or "verified completion"),
                    ),
                    authorization_state={
                        "durable_task_accepted": True,
                        "server_allowlist": True,
                    },
                    retry_count=stalls,
                ),
            )

            # A person may have spoken while this inference was running. Yield
            # before acting, then re-read task control state. A targeted update
            # invalidates the model's now-stale proposed calls and gets a fresh
            # reasoning pass; an unrelated interjection simply lets them run
            # after the foreground turn finishes.
            if not self._wait_for_foreground():
                return
            latest = self.store.get(task_id)
            if latest is None or latest.get("status") == "cancelled":
                return
            redirected = _append_guidance(messages, latest, seen_guidance)
            if redirected:
                # The proposed calls have not executed, so do not leave an
                # assistant tool-call message with missing results in the next
                # prompt. Keep the newly appended human updates and replan.
                del messages[-redirected - 1]
                active_tools = []
                suppress_discovery = False
                checkpoint = self.store.checkpoint(
                    task_id,
                    self.owner,
                    messages=messages,
                    active_tools=active_tools,
                    tools_used=tools_used,
                    applied_guidance_ids=list(seen_guidance),
                    progress=(
                        "A live interjection redirected the task before its pending action; "
                        "replanning from the retained checkpoint."
                    ),
                    status="running",
                    current_stage=context_text("task_stages", "replanning"),
                )
                if checkpoint is None or checkpoint.get("status") == "cancelled":
                    return
                continue

            progress_parts: list[str] = []
            observed_actions: list[dict[str, Any]] = []
            for call in calls:
                function = call.get("function")
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, Mapping)
                    else ""
                )
                arguments = _arguments(call)
                call_id = str(call.get("id") or secrets.token_hex(6))
                if name == "task_compact":
                    latest = self.store.get(task_id) or task
                    before = copy.deepcopy(messages)
                    compacted = _compact_task_messages(
                        messages,
                        latest,
                        force=True,
                    )
                    receipt = _compaction_receipt(
                        before,
                        compacted,
                        latest,
                        reason="agent_requested",
                    )
                    messages = compacted
                    compact_result = {
                        **receipt,
                        "compacted": compacted != before,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "tool_call_id": call_id,
                            "content": json.dumps(compact_result),
                        }
                    )
                    self._record_action(
                        task_id, call_id, name, arguments, compact_result
                    )
                    self.store.compact_context(
                        task_id,
                        self.owner,
                        messages=messages,
                        receipt=receipt,
                    )
                    stalls = 0
                    continue
                if name == "task_recovery":
                    evidence_id = str(arguments.get("evidence_id") or "").strip()
                    failure_scope = str(arguments.get("failure_scope") or "").strip()
                    unmet = " ".join(
                        str(arguments.get("unmet_requirement") or "").split()
                    )
                    capability_query = " ".join(
                        str(arguments.get("capability_query") or "").split()
                    )
                    evidence = _tool_evidence(messages).get(evidence_id)
                    evidence_result = (
                        evidence.get("result")
                        if isinstance(evidence, Mapping)
                        else None
                    )
                    valid = (
                        failure_scope
                        in {"arguments", "capability", "assumption", "task"}
                        and bool(unmet)
                        and len(unmet) <= 300
                        and bool(capability_query)
                        and len(capability_query) <= 200
                        and isinstance(evidence_result, Mapping)
                        and evidence_result.get("disposition")
                        == "change_capability"
                        and evidence_result.get("task_blocked") is False
                    )
                    recovery_result = (
                        {
                            "accepted": True,
                            "failure_scope": failure_scope,
                            "unmet_requirement": unmet,
                            "capability_query": capability_query,
                        }
                        if valid
                        else {
                            "error": "invalid_recovery_assessment",
                            "message": (
                                "Reference a capability-scoped failed evidence ID and "
                                "provide a bounded capability-only query."
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "tool_call_id": call_id,
                            "content": json.dumps(recovery_result),
                        }
                    )
                    self._record_action(
                        task_id, call_id, name, arguments, recovery_result
                    )
                    if valid:
                        recovery_required = False
                        active_tools = []
                        suppress_discovery = False
                        stalls = 0
                        messages.append(
                            {
                                "role": "user",
                                "content": context_text(
                                    "directives", "background_recovery"
                                ).format(capability_query=capability_query),
                            }
                        )
                    else:
                        stalls += 1
                    continue
                if name == "task_checkpoint":
                    latest = self.store.get(task_id)
                    if latest is not None and _append_guidance(
                        messages, latest, seen_guidance
                    ):
                        # The terminal assertion was based on stale directions.
                        stalls += 1
                        checkpoint_result = {
                            "error": "new_guidance",
                            "retryable": True,
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "tool_call_id": call_id,
                                "content": json.dumps(checkpoint_result),
                            }
                        )
                        self._record_action(
                            task_id, call_id, name, arguments, checkpoint_result
                        )
                        continue
                    action = str(arguments.get("action") or "")
                    report = " ".join(str(arguments.get("report") or "").split())
                    criteria_assessment = " ".join(
                        str(arguments.get("criteria_assessment") or "").split()
                    )
                    raw_ids = arguments.get("evidence_ids")
                    evidence_ids = (
                        [str(value) for value in raw_ids if str(value)]
                        if isinstance(raw_ids, list)
                        else []
                    )
                    evidence = _tool_evidence(messages)
                    selected = [evidence.get(value) for value in evidence_ids]
                    valid_refs = bool(selected) and all(item is not None for item in selected)
                    freshest_evidence_id = _freshest_evidence_id(messages)
                    cites_freshest = bool(freshest_evidence_id) and (
                        freshest_evidence_id in evidence_ids
                    )
                    failed = [
                        item
                        for item in selected
                        if item is not None and _result_failed_or_blocked(item["result"])
                    ]
                    recovery_remains = any(
                        isinstance(item, Mapping)
                        and isinstance(item.get("result"), Mapping)
                        and (
                            item["result"].get("task_blocked") is False
                            or item["result"].get("disposition")
                            == "change_capability"
                        )
                        for item in selected
                        if item is not None
                    )
                    valid = (
                        action in {"progress", "complete", "blocked"}
                        and bool(report)
                        and len(report) <= MAX_CHECKPOINT_REPORT_CHARS
                        and bool(criteria_assessment)
                        and len(criteria_assessment) <= MAX_CHECKPOINT_REPORT_CHARS
                        and valid_refs
                        and cites_freshest
                        and (
                            (
                                action == "blocked"
                                and bool(failed)
                                and not recovery_remains
                            )
                            or (action != "blocked" and not failed)
                        )
                    )
                    if not valid:
                        stalls += 1
                        valid_ids = [eid for eid, item in evidence.items() if item is not None and not _result_failed_or_blocked(item["result"])]
                        failed_ids = [eid for eid, item in evidence.items() if item is not None and _result_failed_or_blocked(item["result"])]
                        checkpoint_result = {
                            "error": "unsupported_checkpoint",
                            "message": (
                                "Reference existing successful tool calls for "
                                "progress/complete, or a concrete failed tool call "
                                "with no remaining alternative for blocked. Include "
                                "the freshest concrete result and explicitly assess "
                                "the completion criteria."
                            ),
                            "valid_evidence_ids": valid_ids[:16],
                            "failed_evidence_ids": failed_ids[:8],
                            "freshest_evidence_id": freshest_evidence_id,
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "tool_call_id": call_id,
                                "content": json.dumps(checkpoint_result),
                            }
                        )
                        self._record_action(
                            task_id, call_id, name, arguments, checkpoint_result
                        )
                        continue
                    if action == "progress":
                        now = time.monotonic()
                        eligible = (
                            now - task_started_at >= self.progress_after_s
                            and (
                                last_progress_at is None
                                or now - last_progress_at >= self.progress_min_interval_s
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "tool_call_id": call_id,
                                "content": json.dumps({"accepted": True}),
                            }
                        )
                        self._record_action(
                            task_id,
                            call_id,
                            name,
                            arguments,
                            {"accepted": True, "action": action},
                        )
                        checkpoint = self.store.checkpoint(
                            task_id,
                            self.owner,
                            messages=messages,
                            active_tools=active_tools,
                            tools_used=tools_used,
                            applied_guidance_ids=list(seen_guidance),
                            tool_fingerprints=list(seen),
                            result_digests=list(result_digests),
                            progress=report,
                            status="running",
                            current_stage=context_text(
                                "task_stages", "continuing_checkpoint"
                            ),
                        )
                        if checkpoint is None or checkpoint.get("status") == "cancelled":
                            return
                        if eligible and self.on_progress is not None:
                            last_progress_at = now
                            self.on_progress(
                                {"task_id": task_id, "status": "running", "result": report}
                            )
                        stalls = 0
                        continue
                    status = "completed" if action == "complete" else "blocked"
                    self._record_action(
                        task_id,
                        call_id,
                        name,
                        arguments,
                        {"accepted": True, "status": status},
                    )
                    completed = self.store.checkpoint(
                        task_id,
                        self.owner,
                        messages=messages,
                        active_tools=active_tools,
                        tools_used=tools_used,
                        applied_guidance_ids=list(seen_guidance),
                        tool_fingerprints=list(seen),
                        result_digests=list(result_digests),
                        progress=(
                            "Work completed with referenced evidence."
                            if status == "completed"
                            else "Work stopped at a verified external blocker."
                        ),
                        result=report,
                        status=status,
                    )
                    logger.info(
                        "background task %s %s: %s", task_id, status, report[:300]
                    )
                    if completed is not None and self.on_complete is not None:
                        self.on_complete(completed)
                    return
                fingerprint = _call_fingerprint(name, arguments)
                if fingerprint == last_tool_fingerprint:
                    result: Any = {
                        "error": "duplicate_tool_call",
                        "message": (
                            "This exact call was the immediately preceding external "
                            "attempt; assess its result and change the next step. It may "
                            "be run again after a materially different intervening action "
                            "changes or newly inspects relevant state."
                        ),
                    }
                    if name == "tool_search":
                        suppress_discovery = True
                    elif name in active_tools:
                        active_tools = [item for item in active_tools if item != name]
                    stalls += 1
                else:
                    seen.add(fingerprint)
                    last_tool_fingerprint = fingerprint
                    slice_tool_calls += 1
                    self.store.update_stage(
                        task_id,
                        self.owner,
                        context_text("task_stages", "running_tool").format(
                            tool=name or "unknown"
                        ),
                    )
                    try:
                        response = self._post(
                            f"/api/tools/{name}/call", {"arguments": arguments}
                        ).json()
                    except Exception as error:
                        self._record_action(
                            task_id,
                            call_id,
                            name,
                            arguments,
                            {
                                "error": type(error).__name__,
                                "message": str(error),
                            },
                        )
                        raise
                    result = response.get("result", response)
                    if (
                        isinstance(result, Mapping)
                        and result.get("error") == "resource_pressure"
                    ):
                        self._record_action(
                            task_id, call_id, name, arguments, result
                        )
                        raise MemoryPressure(
                            name or "tool",
                            self.memory_governor.available_gib()
                            if self.memory_governor is not None
                            else 0.0,
                            self.memory_governor.policy.hard_floor_gib
                            if self.memory_governor is not None
                            else 0.0,
                        )
                    if name:
                        tools_used = list(dict.fromkeys([*tools_used, name]))[-16:]
                    if name != "tool_search":
                        suppress_discovery = False
                change_capability = (
                    isinstance(result, Mapping)
                    and result.get("disposition") == "change_capability"
                )
                if change_capability:
                    active_tools = []
                    suppress_discovery = False
                    recovery_required = True
                elif name == "tool_search" and isinstance(result, Mapping):
                    available = result.get("available_tools")
                    if isinstance(available, list):
                        active_tools = list(
                            dict.fromkeys(
                                [
                                    *(
                                        str(item)
                                        for item in available
                                        if str(item) != "background_task"
                                    ),
                                ]
                            )
                        )[:3]
                elif name and name != "background_task":
                    active_tools = [name]
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": name or "unknown",
                    "content": "",
                }
                digest = _result_digest(name, result)
                repeated_result = digest in result_digests
                failed_result = _result_failed_or_blocked(result)
                if failed_result:
                    stalls += 1
                else:
                    stalls = 0
                    self._slice_deferrals.pop(task_id, None)
                if not repeated_result:
                    result_digests.add(digest)
                screenshot = None
                if name in {"browser_interact", "gui_interact"} and isinstance(result, Mapping):
                    screenshot = result.get("screenshot")
                    result = {
                        key: value
                        for key, value in result.items()
                        if key != "screenshot"
                    }
                result = _bounded_tool_result(result)
                self._record_action(task_id, call_id, name, arguments, result)
                observed_actions.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                tool_message["content"] = json.dumps(
                    result, ensure_ascii=False, default=str
                )
                tool_message["tool_call_id"] = call_id
                messages.append(tool_message)
                if change_capability:
                    messages.append(
                        {
                            "role": "user",
                            "content": context_text(
                                "directives", "background_recovery_required"
                            ),
                        }
                    )
                if _is_duplicate_tool_result(tool_message):
                    messages.append(
                        {
                            "role": "user",
                            "content": context_text(
                                "directives", "background_duplicate_call"
                            ),
                        }
                    )
                if isinstance(screenshot, Mapping) and screenshot.get("data"):
                    messages.append(
                        {
                            "role": "user",
                            "content": context_text(
                                "directives", "background_visual_evidence"
                            ),
                            "images": [dict(screenshot)],
                        }
                    )
                if name == "shell":
                    command = str(arguments.get("command") or "").replace("\n", " ")
                    exit_code = result.get("exit_code") if isinstance(result, Mapping) else None
                    progress_parts.append(
                        f"Ran shell step ({command[:180]}); exit={exit_code}."
                    )
                else:
                    progress_parts.append(f"Ran {name or 'unknown'} and retained its result.")

            if observed_actions:
                self._observe_decision(
                    "post_action",
                    DecisionState(
                        user_request=str(task.get("objective") or ""),
                        current_goal=str(task.get("objective") or ""),
                        current_phase="post_action",
                        last_action={
                            "calls": [
                                {
                                    "call_id": item["call_id"],
                                    "name": item["name"],
                                    "arguments": item["arguments"],
                                }
                                for item in observed_actions
                            ]
                        },
                        last_observation=json.dumps(
                            [
                                {
                                    "call_id": item["call_id"],
                                    "name": item["name"],
                                    "result": item["result"],
                                }
                                for item in observed_actions
                            ],
                            ensure_ascii=False,
                            default=str,
                        ),
                        pending_requirements=(
                            str(task.get("completion_criteria") or "verified completion"),
                        ),
                        retry_count=stalls,
                    ),
                )

            concrete_evidence = _tool_evidence(messages)
            newest_evidence_ids = [
                str(item["call_id"])
                for item in observed_actions
                if str(item["call_id"]) in concrete_evidence
            ]
            if newest_evidence_ids:
                messages.append(
                    {
                        "role": "user",
                        "content": context_text(
                            "directives", "background_self_check"
                        ).format(evidence_ids=",".join(newest_evidence_ids)),
                    }
                )

            checkpoint = self.store.checkpoint(
                task_id,
                self.owner,
                messages=messages,
                active_tools=active_tools,
                tools_used=tools_used,
                applied_guidance_ids=list(seen_guidance),
                tool_fingerprints=list(seen),
                result_digests=list(result_digests),
                progress=" ".join(progress_parts),
                status="running",
                current_stage=context_text(
                    "task_stages",
                    "self_check" if newest_evidence_ids else "assessing_result",
                ),
            )
            if checkpoint is None or checkpoint.get("status") == "cancelled":
                return
            # The next reasoning pass is a separate request. That boundary is
            # the scheduling point where a live utterance gets the model first.
            time.sleep(0.05)
