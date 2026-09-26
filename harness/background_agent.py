"""Checkpointed, stepwise agent for work delegated from the live voice turn."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import os
import queue
import re
import secrets
import shlex
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from portal.background_tasks import TERMINAL_STATUSES, BackgroundTaskStore
from portal.tools import DISCOVERY_TOOLS, tool_schemas
from qwen_omni_adapters.context import (
    configured_tool_families,
    context_text,
    context_value,
    without_parent_frame_coordinates,
)
from qwen_omni_adapters.decision_plane import DecisionPlane, DecisionState
from qwen_omni_adapters.memory import MemoryGovernor, MemoryPressure

logger = logging.getLogger(__name__)

MAX_TASK_CONTEXT_MESSAGES = 128
MAX_TASK_CONTEXT_BYTES = 96 * 1024
MAX_RETAINED_TASK_MESSAGES = 12
MIN_TASK_CONTEXT_BYTES = 32 * 1024
MIN_RETAINED_TASK_MESSAGES = 4
MAX_FOCUS_MEMORY_CHARS = 8_000
MIN_FOCUS_MEMORY_CHARS = 900
DEFAULT_RESIDENT_CONTEXT_TOKENS = 16_384
MAX_TOOL_RESULT_CHARS = 24_000
MAX_CHECKPOINT_REPORT_CHARS = 1_000
MAX_ACTION_ARGUMENT_CHARS = 2_000
MAX_ACTION_OUTCOME_CHARS = 1_200
VISUAL_CONTEXT_ACCOUNTING_BYTES = 8 * 1024
COMPACTION_HIGH_WATER_FRACTION = 0.72
COMPUTER_ACTION_TOOLS = {"browser_interact", "gui_interact"}
WEB_EVIDENCE_TOOLS = {"browser_interact", "web_crawl", "web_fetch", "web_search"}
MILESTONE_SOURCE_TOOLS = {
    "document_search",
    "ocr_pdf",
    "structured_read",
    "web_crawl",
    "web_fetch",
}

_HTTP_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

_SENSITIVE_AUDIT_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|secret|token|api[_-]?key)",
    re.IGNORECASE,
)

AGENT_SYSTEM_PROMPT = context_text("prompts", "background_agent_system")
COMPACT_AGENT_SYSTEM_PROMPT = context_text(
    "prompts", "background_agent_system_compact"
)
TASK_START_REQUEST = (
    "<task_start>Begin the task pinned in <current_task>. Choose the smallest "
    "evidence-producing action and continue until its criteria are verified.</task_start>"
)
MAX_VIRTUAL_QUERY_CHARS = 1_200
MAX_PHASE_ACTIONS = 8

_TYPED_TOOL_FAMILIES = frozenset(configured_tool_families())
_CAMERA_DEVICE_RE = re.compile(r"\b(?:camera|webcam|video\s+feed)\b", re.IGNORECASE)
_CAMERA_DEVICE_ACTION_RE = re.compile(
    r"\b(?:capture|check|describe|identify|look|observe|see|show|use|view|watch)\b",
    re.IGNORECASE,
)
_PHYSICAL_SCENE_RE = re.compile(
    r"\b(?:holding|wearing|physical\s+scene|surroundings)\b", re.IGNORECASE
)
_PHYSICAL_SCENE_ACTION_RE = re.compile(
    r"\b(?:describe|identify|look|monitor|observe|see|show|view|watch)\b",
    re.IGNORECASE,
)
_ROOM_OBSERVATION_RE = re.compile(
    r"\b(?:look|monitor|observe|see|watch)\b[^.?!\n]{0,80}\broom\b|"
    r"\broom\b[^.?!\n]{0,80}\b(?:motion|moving|look|monitor|observe|see|watch)\b",
    re.IGNORECASE,
)
_NEGATED_CAMERA_RE = re.compile(
    r"\b(?:do\s+not|don't|never|no|without)\s+(?:use\s+)?(?:the\s+)?"
    r"(?:physical\s+)?(?:camera|webcam|video\s+feed)\b",
    re.IGNORECASE,
)


def _resident_task_context_tokens() -> int:
    """Read the live language window used by the background controller.

    Jetson can resize its resident KV window after the harness has started.
    The task controller therefore cannot size L1/L2 state from a model-card
    maximum or a process-start snapshot. Use the same live state file as the
    portal and fail closed at the supported 4K floor when it is unavailable.
    """

    try:
        configured = int(
            os.environ.get(
                "OMNI_COMPREHENSION_CONTEXT_TOKENS",
                str(DEFAULT_RESIDENT_CONTEXT_TOKENS),
            )
        )
    except ValueError:
        configured = DEFAULT_RESIDENT_CONTEXT_TOKENS
    configured = max(4_096, configured)
    candidates: list[Path] = []
    explicit = os.environ.get("OMNI_COMPREHENSION_CONTEXT_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    repo_root = os.environ.get("OMNI_REPO_ROOT", "").strip()
    if repo_root:
        candidates.append(
            Path(repo_root).expanduser()
            / "runtime-data/state/comprehension-context-tokens"
        )
    for path in candidates:
        try:
            selected = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        return max(4_096, min(configured, selected))
    if candidates:
        return min(4_096, configured)
    return min(configured, DEFAULT_RESIDENT_CONTEXT_TOKENS)


def _task_context_limits(
    resident_context_tokens: int | None = None,
) -> dict[str, int]:
    """Derive task-chain and pinned-focus limits from the live KV tier."""

    resident = max(
        4_096,
        int(
            resident_context_tokens
            if resident_context_tokens is not None
            else _resident_task_context_tokens()
        ),
    )
    return {
        "resident_context_tokens": resident,
        # The raw chain is virtual-memory input, not the final prompt. Keep it
        # broad enough for high-recall ingestion while bounding recurrent JSON
        # and checkpoint churn more tightly when Tegra downshifts to 4K.
        "context_bytes": min(
            MAX_TASK_CONTEXT_BYTES,
            max(MIN_TASK_CONTEXT_BYTES, resident * 8),
        ),
        "retained_messages": max(
            MIN_RETAINED_TASK_MESSAGES,
            min(MAX_RETAINED_TASK_MESSAGES, resident // 1_365),
        ),
        # At the 4K floor this leaves room for the immutable objective,
        # completion contract, current query, and output headroom. Larger KV
        # tiers can page in a broader frontier without changing raw storage.
        "focus_chars": min(
            MAX_FOCUS_MEMORY_CHARS,
            max(MIN_FOCUS_MEMORY_CHARS, (resident * 3) // 10),
        ),
        # Leave the remaining window for the selected tool grammar, chat
        # template, output headroom, and a fresh action result.  This is a
        # token-pressure trigger, not a message-count trigger.
        "compaction_high_water_tokens": max(
            2_048,
            int(resident * COMPACTION_HIGH_WATER_FRACTION),
        ),
    }


def _resident_virtual_query_chars(
    resident_context_tokens: int | None = None,
) -> int:
    """Bound the retrieval anchor without repeating the whole pinned task."""

    resident = _task_context_limits(resident_context_tokens)[
        "resident_context_tokens"
    ]
    return min(MAX_VIRTUAL_QUERY_CHARS, max(420, resident // 10))


def _uncheckpointed_action_count(task: Mapping[str, Any]) -> int:
    """Count concrete attempts since the latest accepted phase boundary."""

    boundary = 0.0
    guidance = task.get("guidance")
    if isinstance(guidance, list):
        boundary = max(
            (
                float(item.get("received_at") or 0)
                for item in guidance
                if isinstance(item, Mapping)
            ),
            default=0.0,
        )
    actions = task.get("actions")
    if not isinstance(actions, list):
        return 0
    for action in reversed(actions):
        if not isinstance(action, Mapping):
            continue
        recorded_at = float(action.get("at") or 0)
        if recorded_at <= boundary:
            break
        if str(action.get("tool") or "") != "task_checkpoint":
            continue
        try:
            outcome = json.loads(str(action.get("outcome") or "{}"))
        except ValueError:
            continue
        if (
            isinstance(outcome, Mapping)
            and outcome.get("accepted") is True
            and outcome.get("action") == "progress"
        ):
            boundary = recorded_at
            break
    return sum(
        1
        for action in actions
        if isinstance(action, Mapping)
        and float(action.get("at") or 0) > boundary
        and str(action.get("tool") or "")
        not in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}
    )


def _task_virtual_query(
    task: Mapping[str, Any], *, resident_context_tokens: int | None = None
) -> str:
    """Keep paging anchored to the durable task, not a compaction artifact."""

    query_chars = _resident_virtual_query_chars(resident_context_tokens)
    objective = " ".join(str(task.get("objective") or "").split())
    guidance = task.get("guidance")
    latest_direction = ""
    if isinstance(guidance, list):
        latest_direction = next(
            (
                " ".join(str(item.get("content") or "").split())
                for item in reversed(guidance)
                if isinstance(item, Mapping)
                and str(item.get("content") or "").strip()
            ),
            "",
        )
    prefix = "Advance and verify the pinned task. Objective: "
    if not latest_direction:
        return f"{prefix}{objective}"[:query_chars]

    # A long objective must not crowd the newest correction out of the paging
    # query.  The objective remains pinned in full in the system contract; this
    # compact retrieval anchor reserves a stable share for current direction.
    direction_prefix = " Latest user direction: "
    direction_budget = min(
        len(latest_direction),
        max(1, (query_chars * 2) // 5),
    )
    objective_budget = max(
        0,
        query_chars
        - len(prefix)
        - len(direction_prefix)
        - direction_budget,
    )
    return (
        f"{prefix}{objective[:objective_budget]}"
        f"{direction_prefix}{latest_direction[:direction_budget]}"
    )


def _audited_task_state(task: Mapping[str, Any]) -> str:
    """Render the external manager/auditor state without executor narration."""

    state = task.get("task_state")
    if not isinstance(state, Mapping):
        return ""
    controller = state.get("controller")
    controller = controller if isinstance(controller, Mapping) else {}
    environment = state.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    knowledge = state.get("knowledge")
    knowledge = knowledge if isinstance(knowledge, Mapping) else {}
    requirements = state.get("requirements")
    requirements = requirements if isinstance(requirements, list) else []
    reports = state.get("audit_reports")
    reports = reports if isinstance(reports, list) else []
    last_audit = next(
        (dict(item) for item in reversed(reports) if isinstance(item, Mapping)),
        {},
    )
    artifacts = environment.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    records = knowledge.get("records")
    records = records if isinstance(records, list) else []
    stagnation = controller.get("stagnation")
    stagnation = dict(stagnation) if isinstance(stagnation, Mapping) else {}
    compact = {
        "state_version": int(state.get("version") or 0),
        "requirements": [
            {
                "requirement_id": str(item.get("requirement_id") or "")[:80],
                "status": str(item.get("status") or "")[:40],
                "text": str(item.get("text") or "")[:500],
                "evidence_ids": list(item.get("evidence_ids") or [])[:8],
            }
            for item in requirements[-8:]
            if isinstance(item, Mapping)
        ],
        "controller": {
            "phase": str(controller.get("phase") or "")[:40],
            "current_subtask": str(controller.get("current_subtask") or "")[:500],
            "next_transition": str(controller.get("next_transition") or "")[:40],
            "remaining_requirements": list(
                controller.get("remaining_requirements") or []
            )[:8],
            "executor_generation": int(controller.get("executor_generation") or 0),
            "stagnation": stagnation,
        },
        "environment": {
            "version": int(environment.get("version") or 0),
            "artifacts": [dict(item) for item in artifacts[-12:] if isinstance(item, Mapping)],
        },
        "knowledge": {
            "version": int(knowledge.get("version") or 0),
            "recent_records": [
                dict(item) for item in records[-8:] if isinstance(item, Mapping)
            ],
        },
        "last_audit": last_audit,
    }
    return (
        '<audited_task_state schema="robit.omni.background-task-state.v1">\n'
        + json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)
        + "\n</audited_task_state>"
    )


def _task_system_prompt(
    task: Mapping[str, Any],
    *,
    resident_context_tokens: int | None = None,
    expand_available: bool | None = None,
) -> str:
    """Pin the durable task contract ahead of all renewable worker context."""

    objective = " ".join(str(task.get("objective") or "").split())
    criteria = " ".join(str(task.get("completion_criteria") or "").split())
    contract = [
        "<current_task>",
        f"Objective: {objective}",
    ]
    if criteria:
        contract.append(f"Completion criteria: {criteria}")
    audited_state = _audited_task_state(task)
    if audited_state:
        contract.append(audited_state)
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
    limits = _task_context_limits(resident_context_tokens)
    focus_memory = _focus_memory(
        task,
        expand_available=expand_available,
        max_chars=limits["focus_chars"],
    )
    if focus_memory:
        # This is L1 pinned working state, not merely a post-compaction note.
        # A successful source or artifact must remain conspicuous on the very
        # next action round so ordinary transcript growth cannot make the
        # controller rediscover or recreate it.
        contract.append(focus_memory)
    constrained = limits["resident_context_tokens"] <= 8_192
    contract.extend(
        [
            (
                "<execution_frontier>Advance the earliest unmet prerequisite. Never "
                "verify a downstream target before evidence shows its required artifact "
                "or input exists; if absent, return to the missing prerequisite."
                "</execution_frontier>"
                if constrained
                else "<execution_frontier>Advance the earliest unmet prerequisite in the "
                "task's stated sequence. A downstream verification, launch, or presentation "
                "step cannot precede concrete evidence that its required artifact or input "
                "exists. If a probe proves a downstream target absent, return to the "
                "earliest missing prerequisite instead of probing variants of that absent "
                "target.</execution_frontier>"
            ),
            "Keep every action causally relevant to this task. Ignore unrelated topics "
            "from model state or prior work.",
            "</current_task>",
        ]
    )
    task_contract = "\n".join(contract)
    controller_policy = (
        COMPACT_AGENT_SYSTEM_PROMPT if constrained else AGENT_SYSTEM_PROMPT
    )
    return f"{task_contract}\n\n{controller_policy}"


def _fresh_executor_messages(
    task: Mapping[str, Any], *, reason: str
) -> list[dict[str, Any]]:
    """Start a clean bounded executor from external audited state."""

    audit = _latest_audit(task)
    handoff = {
        "reason": reason,
        "last_audit_id": str(audit.get("audit_id") or ""),
        "last_evidence_id": str(audit.get("evidence_id") or ""),
        "next_transition": _task_controller_value(task, "next_transition")
        or "prethink",
    }
    return [
        {"role": "system", "content": _task_system_prompt(task)},
        {
            "role": "user",
            "content": (
                '<executor_handoff schema="robit.omni.executor-handoff.v1">'
                + json.dumps(handoff, ensure_ascii=False, sort_keys=True)
                + "</executor_handoff>\n"
                "Choose the next bounded evidence-producing action from the audited "
                "state. Do not reconstruct or continue discarded private reasoning."
            ),
        },
    ]


def _background_discovery_preflight(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the typed router input without interpreting request words."""

    family = str(arguments.get("family") or "").strip()
    if family in _TYPED_TOOL_FAMILIES or family == "uncertain":
        return None
    return {
        "error": "invalid_tool_family",
        "message": (
            "Select one family from the tool_search schema, or choose uncertain. "
            "Natural-language query routing is not supported."
        ),
        "retryable": True,
        "task_progress": False,
        "failure_scope": "arguments",
    }


def _task_allows_physical_camera(task: Mapping[str, Any]) -> bool:
    """Return whether durable task scope explicitly depends on a physical scene.

    Browser screenshots and desktop frames are separate first-class tools and
    never authorize an ambient camera. Authorization can come from the original
    objective, its completion contract, or a later human direction; generated
    summaries and tool results cannot broaden it.
    """

    scoped_text: list[str] = [
        str(task.get("objective") or ""),
        str(task.get("completion_criteria") or ""),
    ]
    guidance = task.get("guidance")
    if isinstance(guidance, list):
        scoped_text.extend(
            str(item.get("content") or "")
            for item in guidance[-8:]
            if isinstance(item, Mapping)
        )
    for source in scoped_text:
        for sentence in re.split(r"[.?!\n]+", source):
            if not sentence.strip() or _NEGATED_CAMERA_RE.search(sentence):
                continue
            if (
                _CAMERA_DEVICE_RE.search(sentence)
                and _CAMERA_DEVICE_ACTION_RE.search(sentence)
            ):
                return True
            if (
                _PHYSICAL_SCENE_RE.search(sentence)
                and _PHYSICAL_SCENE_ACTION_RE.search(sentence)
            ):
                return True
            if _ROOM_OBSERVATION_RE.search(sentence):
                return True
    return False


def _camera_scope_rejection() -> dict[str, Any]:
    return {
        "error": "physical_camera_outside_task_scope",
        "message": (
            "The durable task does not explicitly depend on the current physical scene. "
            "Use browser_interact for a rendered webpage, gui_interact for the desktop, "
            "or discover another task-relevant mechanism."
        ),
        "retryable": True,
        "task_progress": False,
        "failure_scope": "capability",
    }


def _filter_background_discovery(result: Any, task: Mapping[str, Any]) -> Any:
    """Remove physical-camera routing when the pinned task did not authorize it."""

    if not isinstance(result, Mapping) or _task_allows_physical_camera(task):
        return result
    filtered = dict(result)
    removed = False
    for field in ("available_tools", "suggested_tools"):
        values = filtered.get(field)
        if not isinstance(values, list):
            continue
        kept = [str(item) for item in values if str(item) != "request_camera_view"]
        removed = removed or len(kept) != len(values)
        filtered[field] = kept
    results = filtered.get("results")
    if isinstance(results, list):
        kept_results = [
            item
            for item in results
            if not (
                isinstance(item, Mapping)
                and str(item.get("name") or "") == "request_camera_view"
            )
        ]
        removed = removed or len(kept_results) != len(results)
        filtered["results"] = kept_results
    if not removed:
        return filtered
    filtered["scope_filtered_tools"] = ["request_camera_view"]
    if not filtered.get("available_tools"):
        return _camera_scope_rejection()
    return filtered


_TOOL_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "description",
    "examples",
    "title",
}


def _compact_tool_schema(value: Any) -> Any:
    """Remove prose annotations while preserving executable JSON constraints.

    The full catalog remains the source of truth and tool_search result. A
    constrained worker has already selected one capability, so repeating every
    property description in the next 4K action round wastes scarce L0 tokens.
    Names, types, enums, required fields, bounds, and additionalProperties stay
    exact; only non-executable documentation is projected out.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _compact_tool_schema(item)
            for key, item in value.items()
            if str(key) not in _TOOL_SCHEMA_ANNOTATION_KEYS
        }
    if isinstance(value, list):
        return [_compact_tool_schema(item) for item in value]
    return copy.deepcopy(value)


def _background_tool_contract(
    active_tools: list[str],
    *,
    recovery_required: bool,
    phase_boundary: bool,
    expand_available: bool,
    can_checkpoint: bool,
    resident_context_tokens: int | None = None,
    recovery_exploration: bool = False,
) -> list[dict[str, Any]]:
    """Expose the smallest complete action space for one controller round."""

    resident = _task_context_limits(resident_context_tokens)[
        "resident_context_tokens"
    ]
    if recovery_required:
        schemas = [copy.deepcopy(TASK_RECOVERY_TOOL)]
    elif phase_boundary:
        schemas = [copy.deepcopy(TASK_CHECKPOINT_TOOL)]
    else:
        tool_limit = 1 if resident <= 8_192 else 3
        concrete = [
            name
            for name in dict.fromkeys(active_tools)
            if tool_schemas([name])
        ][:tool_limit]
        schemas = tool_schemas(concrete)
        # A routed family is the current phase's action space, not a hint that
        # competes with the router on every subsequent step. Keep using the
        # concrete capability until a checkpoint resets the phase or its
        # executor returns a typed capability failure. This prevents repeated
        # classify -> inspect -> classify loops while preserving a general,
        # model-selected transition at the phase boundary.
        if not concrete or recovery_exploration:
            schemas.extend(copy.deepcopy(DISCOVERY_TOOLS))
        # On a constrained tier, paging remains available beside the one active
        # leaf; broad discovery does not. Larger tiers use the same state
        # machine so behavior is independent of the current KV allocation.
        if expand_available:
            schemas.append(copy.deepcopy(TASK_EXPAND_TOOL))
        if can_checkpoint:
            schemas.append(copy.deepcopy(TASK_CHECKPOINT_TOOL))
    if resident <= 8_192:
        return [_compact_tool_schema(schema) for schema in schemas]
    return schemas


def _task_expand_available(
    messages: list[dict[str, Any]], *, compacted: bool
) -> bool:
    """Offer one evidence page-in, then force a return to task actions.

    ``task_expand`` reads immutable receipts; it does not discover capabilities or
    change external state. Leaving it resident immediately after a page-in lets a
    small controller mistake repeated lexical queries for forward progress. A new
    concrete result (or phase boundary) can make older evidence relevant again, but
    routing and failed actions must recover through ``tool_search`` instead.
    """

    if not compacted:
        return False
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if name == "task_expand":
            return False
        if name == "tool_search":
            return False
        if name in {"task_checkpoint", "task_compact", "task_recovery"}:
            continue
        if not name:
            continue
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            result = {}
        return (
            not _result_failed_or_blocked(result)
            and _supports_durable_progress({"name": name, "result": result})
        )
    return True


TASK_CHECKPOINT_TOOL = context_value("control_tools", "task_checkpoint")
TASK_COMPACT_TOOL = context_value("control_tools", "task_compact")
TASK_EXPAND_TOOL = context_value("control_tools", "task_expand")
TASK_RECOVERY_TOOL = context_value("control_tools", "task_recovery")
LOCAL_CONTROL_TOOL_NAMES = {
    "task_checkpoint",
    "task_compact",
    "task_expand",
    "task_recovery",
}
NON_STICKY_RESULT_TOOLS = {
    "document_search",
    "get_current_time",
    "get_portal_capabilities",
    "get_system_snapshot",
    "get_user_location",
    "memory_search",
    "memory_write",
    "request_camera_view",
    "web_crawl",
    "web_fetch",
    "web_search",
}
CAPABILITY_RECOVERY_ALTERNATIVES = {
    "shell": ["workspace_file"],
    "workspace_file": ["shell"],
}
MAX_FAILED_CAPABILITY_ATTEMPTS = 3


class _ForegroundPreempted(RuntimeError):
    """Background inference was cancelled for an accepted spoken turn."""


class _NonRetryableBackgroundError(RuntimeError):
    """A request is invalid and cannot improve by resending the same payload."""


class _MalformedToolCall(RuntimeError):
    """The model emitted invalid structured arguments and must replan smaller."""


def _background_portal_session(seed: str, task_id: str) -> str:
    """Return a stable task-local portal namespace without exposing either input."""

    digest = hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest()
    return f"background-{digest[:40]}"


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
    diagnostic = {
        "classification": classification,
        "done_reason": str(response.get("done_reason") or "")[:80],
        "prompt_eval_count": max(0, int(response.get("prompt_eval_count") or 0)),
        "eval_count": eval_count,
        "step_token_limit": step_token_limit,
        "thinking_chars": thinking_chars,
        "content_chars": content_chars,
        "tool_call_count": tool_call_count,
    }
    portal = response.get("portal")
    virtual = portal.get("virtual_context") if isinstance(portal, Mapping) else None
    if isinstance(virtual, Mapping):
        diagnostic["virtual_context"] = {
            key: virtual.get(key)
            for key in (
                "mode",
                "physical_context_tokens",
                "working_tokens",
                "working_set_fallback",
            )
            if virtual.get(key) is not None
        }
    return diagnostic


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


_STRICT_VISUAL_TARGET = re.compile(
    r"target=(?P<label>[^<\n]{1,160}?)\s+"
    r"point=\(\s*(?P<x>\d{1,4})\s*,\s*(?P<y>\d{1,4})\s*\)\s+"
    r"bbox=\(\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*\)",
    re.IGNORECASE,
)

_VISUAL_REFERRING_EXPRESSIONS = (
    re.compile(
        r"\bclick\s+(?:on\s+)?(?:the\s+)?(?P<label>[\w][^\"“”<>\n]{0,120}?)"
        r"(?=\s*(?:[\"”.;,]|\bat\b|\bin\b|\bnear\b|\blocated\b|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btarget\s+(?:is\s+|[-—:]\s*)(?:the\s+)?"
        r"(?P<label>[\w][^\"“”<>\n]{0,120}?)"
        r"(?=\s*(?:[\"”.;,]|\bat\b|\bin\b|\bnear\b|\blocated\b|$))",
        re.IGNORECASE,
    ),
)


def _visual_referring_expression(observation: str) -> str:
    """Return one unambiguous natural-language target from current evidence."""

    labels: dict[str, str] = {}
    for pattern in _VISUAL_REFERRING_EXPRESSIONS:
        for match in pattern.finditer(observation):
            label = " ".join(match.group("label").split()).strip(" -—:,.\"")
            if label:
                labels.setdefault(label.casefold(), label[:160])
    return next(iter(labels.values())) if len(labels) == 1 else ""


def _ground_visual_click(
    arguments: Mapping[str, Any], observation: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Admit one unambiguous current-frame perception point directly."""

    grounded = dict(arguments)
    if (
        str(grounded.get("action") or "") != "visual_click"
        or str(grounded.get("coordinate_unit") or "") != "normalized_1000"
    ):
        return grounded, None
    matches = list(_STRICT_VISUAL_TARGET.finditer(observation))
    proposed = {"x": grounded.get("x"), "y": grounded.get("y")}
    if len(matches) == 1:
        match = matches[0]
        x, y = int(match.group("x")), int(match.group("y"))
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            return grounded, None
        label = " ".join(match.group("label").split())[:160]
        grounded.update({"x": x, "y": y, "target": label})
        return grounded, {
            "source": "strict_current_visual_observation",
            "target": label,
            "proposed": proposed,
            "executed": {"x": x, "y": y},
        }
    if matches:
        return grounded, None
    label = _visual_referring_expression(observation)
    if not label:
        return grounded, None
    grounded["target"] = label
    return grounded, {
        "source": "current_visual_referring_expression",
        "target": label,
        "proposed": proposed,
    }


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
    fingerprint_arguments = dict(arguments)
    if name == "shell":
        # An absolute leading ``cd`` determines the command's execution root;
        # changing the redundant tool-level cwd does not make the call a new
        # action.  Without this normalization a controller can alternate the
        # same inspection between ``cwd`` omitted/present forever while the
        # duplicate guard sees distinct JSON.
        command = str(fingerprint_arguments.get("command") or "")
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            tokens = []
        target_index = 2 if tokens[:2] == ["cd", "--"] else 1
        if (
            tokens[:1] == ["cd"]
            and len(tokens) > target_index
            and Path(tokens[target_index]).is_absolute()
        ):
            fingerprint_arguments.pop("cwd", None)

    def normalized(value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.casefold().split())
        if isinstance(value, Mapping):
            return {str(key): normalized(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    return hashlib.sha256(
        f"{name}\0{json.dumps(normalized(fingerprint_arguments), sort_keys=True, default=str)}".encode()
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
    """Return the most recent concrete attempt for immediate-loop protection.

    A fingerprint is not a permanent prohibition. The same verification command
    is often exactly what should run after a repair changes the target state.
    Only an unchanged, immediately repeated concrete call is rejected; any
    intervening concrete attempt gives the worker a new causal state to assess.
    Routing and local-control calls do not change the external target, so they
    must not erase this boundary across checkpoints or worker restarts.
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
            if name and name not in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}:
                return _call_fingerprint(name, _arguments(call))
    return ""


def _causal_tail_start(
    messages: list[dict[str, Any]], retained_messages: int
) -> int:
    """Keep two complete recent action/observation cycles across compaction."""

    default_start = max(2, len(messages) - retained_messages)
    paired_assistant: dict[str, int] = {}
    external_pairs: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                call_id = str(call.get("id") or "")
                if call_id:
                    paired_assistant[call_id] = index
            continue
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if not name or name in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}:
            continue
        call_id = str(message.get("tool_call_id") or "")
        external_pairs.append(paired_assistant.get(call_id, index))
    if external_pairs:
        default_start = min(default_start, external_pairs[-2])
    if (
        default_start > 2
        and messages[default_start].get("role") == "tool"
        and messages[default_start - 1].get("role") == "assistant"
    ):
        default_start -= 1
    return max(2, default_start)


def _compact_task_messages(
    messages: list[dict[str, Any]],
    task: Mapping[str, Any],
    *,
    force: bool = False,
    resident_context_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Turn a long transcript into a fresh chain with its durable state intact."""

    limits = _task_context_limits(resident_context_tokens)
    retained_messages = limits["retained_messages"]
    metrics = _context_metrics(messages)
    if len(messages) <= 3 or (
        force
        and len(messages) <= retained_messages + 3
        and metrics["estimated_tokens"]
        <= limits["compaction_high_water_tokens"] // 2
    ) or (
        not force
        and len(messages) <= MAX_TASK_CONTEXT_MESSAGES
        and metrics["estimated_tokens"]
        < limits["compaction_high_water_tokens"]
    ):
        return messages
    head = copy.deepcopy(messages[:2])
    if head and head[0].get("role") == "system":
        # A compacted chain gets exactly one current L1 frontier. Previously
        # the old system prompt and a second full focus ledger were both kept,
        # so a 4K resident model received >5K tokens of pinned state alone.
        head[0] = {
            "role": "system",
            "content": _task_system_prompt(
                task,
                resident_context_tokens=limits["resident_context_tokens"],
                expand_available=True,
            ),
        }
    tail_start = _causal_tail_start(messages, retained_messages)
    sections = [
        '<retained_checkpoint schema="robit.omni.task-page.v2">',
        "Older rounds were paged out losslessly. The current system contract contains "
        "the sole resident typed frontier; exact receipts remain recoverable by its "
        "task_expand evidence pointers or exact-text query. Continue from the newest "
        "native tool cycle.",
    ]
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
    actions = task.get("actions")
    if isinstance(actions, list) and actions:
        verified_receipts = [
            action.get("receipt")
            for action in actions[-32:]
            if isinstance(action, Mapping)
            and isinstance(action.get("receipt"), Mapping)
            and action.get("ok") is True
        ]
        if verified_receipts:
            sections.append(
                "Verified historical action target ledger for progress reports only; "
                "copy target text exactly and never reuse it as current coordinates:"
            )
            sections.extend(
                f"- {json.dumps(receipt, ensure_ascii=False, sort_keys=True)}"
                for receipt in verified_receipts[-24:]
            )
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
                            "visual_change",
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
    summary, and normally only the newest computer-use cycle. This prevents a
    miss several screenshots ago from competing with the current coordinate
    frame without discarding long-horizon progress. Visual refinement carries
    its immediately preceding full-frame semantic orientation inside the newest
    cycle, so retaining another complete DOM/screenshot result is unnecessary.
    A locally rejected action that returned no replacement frame retains one
    preceding cycle because that earlier screenshot is still current evidence.
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
        tail_start = motor_rounds[-1]
        if len(motor_rounds) > 1 and not any(
            isinstance(message.get("images"), list) and message.get("images")
            for message in messages[tail_start:]
        ):
            tail_start = motor_rounds[-2]
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


def _discard_visual_frames(
    messages: list[dict[str, Any]],
    *,
    replacement_note: str,
) -> int:
    """Discard superseded screenshots while retaining their transcript slot.

    The newest screenshot remains useful across a local control rejection, such as an
    immediately duplicated snapshot: no external action happened, so it is still the
    current frame. Callers invoke this helper only when a real computer-use result
    supplies a replacement frame or invalidates the old one.
    """

    discarded = 0
    for message in messages:
        if not message.pop("images", None):
            continue
        discarded += 1
        message["content"] = (
            str(message.get("content") or "") + f"\n[{replacement_note}]"
        ).strip()
    return discarded


def _sanitize_checkpoint_history(
    messages: list[dict[str, Any]],
    *,
    call_id: str = "",
) -> int:
    """Keep model-authored checkpoint claims out of recurrent task context.

    ``task_checkpoint`` is a control request, not evidence.  Its free-form
    report, criteria assessment, remaining-work list, and adjacent assistant
    prose are useful to the validator and terminal task record once, but
    replaying them on the next round lets an unsupported claim become apparent
    history.  Retain only the action and provenance pointers needed to pair the
    call with its authoritative tool result.
    """

    sanitized = 0
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        changed = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            if str(function.get("name") or "") != "task_checkpoint":
                continue
            if call_id and str(call.get("id") or "") != call_id:
                continue
            arguments = _arguments(call)
            evidence = arguments.get("evidence_ids")
            safe_arguments = {
                "action": str(arguments.get("action") or ""),
                "evidence_ids": (
                    [str(value) for value in evidence if str(value)][:16]
                    if isinstance(evidence, list)
                    else []
                ),
            }
            call["function"] = {
                **dict(function),
                "arguments": safe_arguments,
            }
            sanitized += 1
            changed = True
        if changed:
            # Narrative beside a checkpoint is another unvalidated claim about
            # task state.  The following tool result is the sole authority.
            message["content"] = ""
    return sanitized


def _durable_task_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a crash-safe transcript without screenshots or checkpoint claims."""

    durable = copy.deepcopy(messages)
    _sanitize_checkpoint_history(durable)
    _discard_visual_frames(
        durable,
        replacement_note=(
            "The rendered screenshot was inspected in this worker slice; take a fresh "
            "snapshot after a restart or scheduling yield."
        ),
    )
    return durable


def _context_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    measured: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        item = {key: value for key, value in message.items() if key != "images"}
        images = message.get("images")
        if isinstance(images, list) and images:
            image_count += len(images)
            item["images"] = [
                {
                    key: value
                    for key, value in image.items()
                    if key != "data"
                }
                if isinstance(image, Mapping)
                else {"type": "unknown"}
                for image in images
            ]
        measured.append(item)
    accounted_bytes = len(
        json.dumps(measured, ensure_ascii=False, default=str).encode("utf-8")
    ) + image_count * VISUAL_CONTEXT_ACCOUNTING_BYTES
    return {
        "messages": len(messages),
        # Raw PNG/JPEG base64 bytes are transport size, not language context.
        # Charge one bounded multimodal-token estimate per live frame so a
        # single fresh screenshot does not falsely trigger transcript
        # compaction and discard the only actionable visual state.
        "bytes": accounted_bytes,
        # The resident tokenizer remains authoritative at request packing
        # time.  This conservative local estimate is only a high-water alarm;
        # it prevents a tiny-message counter from compacting a half-empty KV
        # window while still bounding a task before the packer rejects it.
        "estimated_tokens": (accounted_bytes + 3) // 4,
    }


def _compaction_available(
    messages: list[dict[str, Any]],
    *,
    resident_context_tokens: int | None = None,
) -> bool:
    limits = _task_context_limits(resident_context_tokens)
    metrics = _context_metrics(messages)
    large_enough = (
        metrics["estimated_tokens"] >= limits["compaction_high_water_tokens"]
        or metrics["messages"] > MAX_TASK_CONTEXT_MESSAGES
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


def _compaction_tool_available(
    messages: list[dict[str, Any]], active_tools: list[str]
) -> bool:
    """Expose manual compaction only when no already-scoped computer loop is active."""

    return _compaction_available(messages) and not COMPUTER_ACTION_TOOLS.intersection(
        active_tools
    )


def _compaction_receipt(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    task: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    evidence_ids = list(_tool_evidence(after))[-16:]
    guidance = task.get("guidance")
    limits = _task_context_limits()
    return {
        "schema": "robit.omni.task-compaction.v1",
        "reason": reason,
        "before": _context_metrics(before),
        "after": _context_metrics(after),
        "resident_context_tokens": limits["resident_context_tokens"],
        "working_set_limits": {
            "context_bytes": limits["context_bytes"],
            "retained_messages": limits["retained_messages"],
            "focus_chars": limits["focus_chars"],
            "compaction_high_water_tokens": limits[
                "compaction_high_water_tokens"
            ],
        },
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


def _unchanged_result_digest(
    name: str, arguments: Mapping[str, Any], result: Any
) -> str:
    """Fingerprint causal outcomes without rewarding cosmetic retry changes."""

    if (
        name == "workspace_file"
        and str(arguments.get("action") or "") in {"list", "read"}
    ):
        # Read-only probes do not change filesystem state. Supplying an
        # explicit default depth, page size, or another cosmetic argument
        # cannot make the same returned observation new evidence.
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(f"{name}\0observed\0{rendered}".encode()).hexdigest()
    if name == "shell" and _result_failed_or_blocked(result) and isinstance(result, Mapping):
        # Shell commonly reports the same failure on either stream depending on
        # redirection. The command text, cwd echo, timeout, and supplied stdin are
        # attempted remedies, not changes to the observed external state.
        diagnostic = " ".join(
            " ".join(str(result.get(field) or "").casefold().split())
            for field in ("stdout", "stderr", "message", "error")
            if str(result.get(field) or "").strip()
        )
        causal = {
            "exit_code": result.get("exit_code"),
            "timed_out": result.get("timed_out") is True,
            "diagnostic": diagnostic,
        }
        rendered = json.dumps(causal, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(f"{name}\0failed\0{rendered}".encode()).hexdigest()
    return hashlib.sha256(
        f"{_call_fingerprint(name, arguments)}\0{_result_digest(name, result)}".encode()
    ).hexdigest()


def _latest_external_result_digest(messages: list[dict[str, Any]]) -> str:
    """Restore the last causal result boundary from durable task messages."""

    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    last_digest = ""
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping):
                    continue
                call_id = str(call.get("id") or "")
                name = str(function.get("name") or "")
                if call_id and name:
                    calls[call_id] = (name, _arguments(call))
            continue
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if (
            not name
            or name == "tool_search"
            or name in LOCAL_CONTROL_TOOL_NAMES
            or name in COMPUTER_ACTION_TOOLS
        ):
            continue
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            continue
        if isinstance(result, Mapping) and result.get("error") in {
            "duplicate_tool_call",
            "repeated_unchanged_result",
        }:
            continue
        call_id = str(message.get("tool_call_id") or "")
        call_name, arguments = calls.get(call_id, (name, {}))
        last_digest = _unchanged_result_digest(call_name or name, arguments, result)
    return last_digest


def _audit_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _canonical_http_url(value: Any) -> str:
    """Normalize a public-web candidate for provenance-bound comparison."""

    raw = str(value or "").strip().rstrip(".,;:!?)]}")
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    if scheme not in {"http", "https"} or not hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _urls_in_value(value: Any) -> list[str]:
    """Extract bounded HTTP(S) URLs from a trusted task or tool-result value."""

    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            return
        for match in _HTTP_URL.findall(item):
            normalized = _canonical_http_url(match)
            if normalized and normalized not in found:
                found.append(normalized)

    visit(value)
    return found[:64]


def _authorized_web_fetch_urls(
    messages: list[dict[str, Any]], task: Mapping[str, Any]
) -> list[str]:
    """Return URLs grounded in user intent or prior external evidence.

    Assistant-authored calls are deliberately excluded. Otherwise an invented
    URL would authorize itself merely by appearing in the proposed call.
    """

    found: list[str] = []

    def retain(values: Any) -> None:
        for value in _urls_in_value(values):
            if value not in found:
                found.append(value)

    # The newest external result is the most useful retry menu.
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if str(message.get("tool_name") or "") not in WEB_EVIDENCE_TOOLS:
            continue
        content = message.get("content")
        try:
            parsed = json.loads(str(content or ""))
        except ValueError:
            parsed = str(content or "")
        retain(parsed)

    retain(task.get("objective"))
    retain(task.get("completion_criteria"))
    guidance = task.get("guidance")
    if isinstance(guidance, list):
        for item in reversed(guidance):
            if isinstance(item, Mapping):
                retain(item.get("content"))
    return found[:64]


def _web_fetch_preflight(
    messages: list[dict[str, Any]],
    task: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Fail closed when a model invents a fetch target after discovery."""

    requested = _canonical_http_url(arguments.get("url"))
    if not requested:
        # The portal remains authoritative for malformed and non-HTTP input.
        return None
    allowed = _authorized_web_fetch_urls(messages, task)
    if requested in allowed:
        return None
    return {
        "error": "undiscovered_url",
        "message": (
            "The requested URL was absent from the task/user input and prior "
            "tool evidence. Choose an exact URL from allowed_urls; do not invent "
            "or reconstruct a source address."
        ),
        "failure_scope": "arguments",
        "retryable": True,
        "task_blocked": False,
        "allowed_urls": allowed[:12],
    }


def _focus_memory(
    task: Mapping[str, Any],
    *,
    expand_available: bool | None = None,
    max_chars: int = MAX_FOCUS_MEMORY_CHARS,
) -> str:
    """Render durable, typed focus records instead of another prose summary."""

    if expand_available is None:
        expand_available = bool(task.get("compaction"))
    bounded_max = max(MIN_FOCUS_MEMORY_CHARS, int(max_chars))
    constrained = bounded_max <= 2_000

    def expansion_pointer(call_id: str) -> dict[str, Any]:
        if not expand_available:
            return {}
        return {"page_in_evidence_id": call_id}

    actions = task.get("actions")
    if not isinstance(actions, list):
        return ""
    sources_by_url: dict[str, dict[str, Any]] = {}
    artifacts_by_path: dict[str, dict[str, Any]] = {}
    inspections_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoints: list[dict[str, Any]] = []
    failures_by_target: dict[tuple[str, str, str], dict[str, Any]] = {}
    other_successes: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        call_id = str(action.get("call_id") or "")[:128]
        tool = str(action.get("tool") or "")[:120]
        ok = action.get("ok") is True
        arguments = _audit_mapping(action.get("arguments"))
        outcome = _audit_mapping(action.get("outcome"))
        authority = _evidence_authority({"name": tool, "result": outcome})
        if tool == "web_fetch" and ok:
            url = str(arguments.get("url") or "").strip()[:500]
            if url:
                # Re-fetching a mutable URL supersedes its resident pointer;
                # every version remains immutable in evidence_records.
                sources_by_url.pop(url, None)
                sources_by_url[url] = {
                    "evidence_id": call_id,
                    "status": "acquired",
                    "source_url": url,
                    **expansion_pointer(call_id),
                }
        elif tool == "browser_interact" and ok:
            browser_action = str(arguments.get("action") or "snapshot")[:80]
            url = _canonical_http_url(
                outcome.get("url") or arguments.get("url")
            )
            if url and outcome.get("rendered") is True:
                # A real rendered HTTP page is source-bearing browser evidence.
                # A successful transport receipt for about:blank or another
                # empty viewport is only an inspection and cannot satisfy a
                # research/acquisition prerequisite.
                sources_by_url.pop(url, None)
                sources_by_url[url] = {
                    "evidence_id": call_id,
                    "status": "browser_rendered",
                    "source_url": url,
                    "title": str(outcome.get("title") or "")[:200],
                    **expansion_pointer(call_id),
                }
            else:
                target = str(
                    outcome.get("url")
                    or arguments.get("url")
                    or browser_action
                )[:500]
                key = (f"browser_{browser_action}", target)
                inspections_by_target.pop(key, None)
                inspections_by_target[key] = {
                    "evidence_id": call_id,
                    "status": f"browser_{browser_action}",
                    "target": target,
                    "task_progress": False,
                    **expansion_pointer(call_id),
                }
        elif tool == "workspace_file" and ok:
            path = str(
                outcome.get("path") or arguments.get("path") or ""
            ).strip()[:500]
            action = str(
                outcome.get("action") or arguments.get("action") or "changed"
            )
            if path and action in {"mkdir", "write", "replace"}:
                artifacts_by_path.pop(path, None)
                artifacts_by_path[path] = {
                    "evidence_id": call_id,
                    "status": action,
                    "path": path,
                    "sha256": str(outcome.get("sha256") or "")[:128],
                    "validation": str(outcome.get("validation") or "")[:120],
                    **expansion_pointer(call_id),
                }
            elif path and action in {"list", "read"}:
                key = (action, path)
                inspections_by_target.pop(key, None)
                inspections_by_target[key] = {
                    "evidence_id": call_id,
                    "status": action,
                    "path": path,
                    "sha256": str(outcome.get("sha256") or "")[:128],
                    "task_progress": False,
                    **expansion_pointer(call_id),
                }
        elif tool == "task_checkpoint" and ok:
            accepted = outcome.get("accepted") is True
            if accepted:
                checkpoints.append(
                    {
                        "checkpoint_id": call_id,
                        "action": str(arguments.get("action") or "progress"),
                        "evidence_ids": list(arguments.get("evidence_ids") or [])[:16],
                        "authority": "model_checkpoint_control_not_task_evidence",
                    }
                )
        elif ok and authority == "inspection":
            target = str(
                outcome.get("path")
                or arguments.get("path")
                or outcome.get("url")
                or arguments.get("url")
                or outcome.get("cwd")
                or arguments.get("cwd")
                or tool
            ).strip()[:500]
            observation = {
                field: outcome[field]
                for field in (
                    "action",
                    "path",
                    "cwd",
                    "exit_code",
                    "stdout",
                    "stderr",
                    "timed_out",
                )
                if field in outcome
            }
            key = (tool, target)
            inspections_by_target.pop(key, None)
            inspections_by_target[key] = {
                "evidence_id": call_id,
                "status": "inspected",
                "tool": tool,
                "target": target,
                "observation": observation,
                "task_progress": False,
                **expansion_pointer(call_id),
            }
        elif (
            ok
            and authority != "discovery"
            and tool not in LOCAL_CONTROL_TOOL_NAMES
            and tool != "tool_search"
        ):
            target = str(
                arguments.get("path")
                or arguments.get("url")
                or arguments.get("command")
                or ""
            ).strip()[:300]
            other_successes.append(
                {
                    "evidence_id": call_id,
                    "tool": tool,
                    "target": target,
                    "result": " ".join(
                        str(action.get("outcome") or "").split()
                    )[:180],
                    **expansion_pointer(call_id),
                }
            )
        if not ok and tool not in LOCAL_CONTROL_TOOL_NAMES and tool != "tool_search":
            target = str(
                arguments.get("path")
                or arguments.get("url")
                or arguments.get("command")
                or ""
            ).strip()[:300]
            error = str(
                outcome.get("error") or outcome.get("message") or "failed"
            )[:120]
            key = (tool, target, error)
            failures_by_target.pop(key, None)
            failures_by_target[key] = {
                "evidence_id": call_id,
                "tool": tool,
                "target": target,
                "diagnostic": " ".join(
                    str(action.get("outcome") or "").split()
                )[:220],
                **expansion_pointer(call_id),
            }
    sources = list(sources_by_url.values())
    artifacts = list(artifacts_by_path.values())
    inspections = list(inspections_by_target.values())
    failures = list(failures_by_target.values())
    if not any(
        (sources, artifacts, inspections, checkpoints, failures, other_successes)
    ):
        return ""

    def tagged(name: str, records: list[dict[str, Any]]) -> list[str]:
        if not records:
            return []
        return [
            f"<{name}>",
            *(
                f'<focus_item id="{html.escape(str(record.get("evidence_id") or record.get("checkpoint_id") or ""))}">'
                + html.escape(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    )
                )
                + "</focus_item>"
                for record in records
            ),
            f"</{name}>",
        ]

    paging_contract = (
        (
            "Recover omitted detail with task_expand using a page_in_evidence_id, or "
            "query by exact path, URL, symbol, or error; paging is not task progress. "
            "task_expand is never an action or argument of another tool. "
            if constrained
            else "Some detailed receipts have left active context. To recover a missing "
            "detail, invoke the separate task_expand control tool with "
            '{"evidence_ids":["the page_in_evidence_id"]}. task_expand is never an action '
            "or argument of workspace_file or another external tool. Expansion is "
            "paging, not new progress. If the needed nonresident record is not listed, "
            'invoke task_expand with {"query":"an exact path, URL, symbol, or error"}. '
        )
        if expand_available
        else (
            "Exact recent receipts remain in the active tool transcript. "
            if constrained
            else "Detailed recent receipts remain in the active tool transcript; no "
            "paging control is available or needed. "
        )
    )
    selected = {
        "phase_checkpoints": checkpoints[-2:],
        "acquired_sources": sources[-8:],
        "artifacts": artifacts[-12:],
        "other_successes": other_successes[-4:],
        "failed_attempts": failures[-4:],
        "inspections": inspections[-2:],
    }

    def render() -> str:
        omitted = {
            "checkpoints": max(
                0, len(checkpoints) - len(selected["phase_checkpoints"])
            ),
            "sources": max(0, len(sources) - len(selected["acquired_sources"])),
            "artifacts": max(0, len(artifacts) - len(selected["artifacts"])),
            "successes": max(
                0, len(other_successes) - len(selected["other_successes"])
            ),
            "failures": max(0, len(failures) - len(selected["failed_attempts"])),
            "inspections": max(0, len(inspections) - len(selected["inspections"])),
        }
        focus_contract = (
            "Typed records are authoritative. Do not redo a retained source or "
            "artifact. "
            + paging_contract
            + "Read/list is observation, not progress. Checkpoints are boundaries, "
            "not proof. Omitted records remain lossless."
            if constrained
            else "These typed records remain authoritative across compaction. "
            "Do not redo an acquired source or artifact merely because its original "
            "turn is not resident. "
            + paging_contract
            + "Read/list inspections are observations, never completed work. Do not "
            "repeat an equivalent inspection unless causal state changed. Phase "
            "checkpoints are control boundaries, not proof. Recompute task state from "
            "the pinned completion contract and typed records. Omitted residents remain "
            "lossless in external evidence storage."
        )
        sections = [
            '<focus_memory schema="robit.omni.background-focus.v2" '
            + " ".join(
                f'omitted_{key}="{value}"' for key, value in omitted.items()
            )
            + ">",
            f"<focus_contract>{focus_contract}</focus_contract>",
            *tagged("phase_checkpoints", selected["phase_checkpoints"]),
            *tagged("acquired_sources", selected["acquired_sources"]),
            *tagged("artifacts", selected["artifacts"]),
            *tagged("other_successes", selected["other_successes"]),
            *tagged("failed_attempts", selected["failed_attempts"]),
            *tagged("inspections", selected["inspections"]),
            "</focus_memory>",
        ]
        return "\n".join(sections)

    # Evict low-authority/recoverable residents until L1 fits. Raw evidence
    # and action receipts are append-only in the store, so this changes only
    # the working set and reports exactly how many records are nonresident.
    eviction_order = (
        "inspections",
        "phase_checkpoints",
        "failed_attempts",
        "other_successes",
        "artifacts",
        "acquired_sources",
    )
    minimum = {
        "inspections": 0,
        "phase_checkpoints": 0,
        "other_successes": 0,
        "failed_attempts": 0,
        "artifacts": 0,
        "acquired_sources": 0,
    }
    rendered = render()
    while len(rendered) > bounded_max:
        removed = False
        for name in eviction_order:
            records = selected[name]
            if len(records) > minimum[name]:
                records.pop(0)
                removed = True
                break
        if not removed:
            break
        rendered = render()
    return rendered


def _compaction_evidence_records(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Archive model-visible tool receipts before their turns leave L0 context."""

    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping):
                    continue
                call_id = str(call.get("id") or "")
                name = str(function.get("name") or "")
                if call_id and name:
                    calls[call_id] = (name, _arguments(call))
            continue
        if message.get("role") != "tool":
            continue
        evidence_id = str(message.get("tool_call_id") or "")
        name = str(message.get("tool_name") or "")
        if (
            not evidence_id
            or not name
            or name == "tool_search"
            or name in LOCAL_CONTROL_TOOL_NAMES
        ):
            continue
        call_name, arguments = calls.get(evidence_id, (name, {}))
        raw_result = str(message.get("content") or "")
        records.append(
            {
                "evidence_id": evidence_id,
                "tool": call_name or name,
                "arguments": _audit_json(arguments, 4_000),
                # This is the already bounded, model-visible receipt. Preserve
                # its full text so EXPAND can recover exact fetched passages,
                # symbols, values, and diagnostics instead of a 320-character
                # audit preview.
                "result": raw_result[:MAX_TOOL_RESULT_CHARS],
                "result_sha256": hashlib.sha256(raw_result.encode("utf-8")).hexdigest(),
            }
        )
    return records


def _search_task_evidence(
    task: Mapping[str, Any], query: str, *, limit: int = 8
) -> list[dict[str, Any]]:
    """Locate immutable task receipts that are not resident in the focus ledger.

    This is a bounded exact/lexical page-table lookup, not model-authored
    summarization. The global virtual-context store still performs hybrid and
    recursive retrieval; this local path guarantees that a known file, URL,
    symbol, or diagnostic can recover its exact task receipt without first
    knowing an evicted evidence ID.
    """

    normalized = " ".join(str(query).casefold().split())[:500]
    terms = tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9_./:@+-]{2,}", normalized)
            if token
        )
    )
    records = task.get("evidence_records")
    if not normalized or not isinstance(records, list):
        return []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        candidate = copy.deepcopy(dict(record))
        rendered = json.dumps(
            candidate, ensure_ascii=False, sort_keys=True, default=str
        ).casefold()
        exact = normalized in rendered
        matched = sum(1 for term in terms if term in rendered)
        if not exact and matched == 0:
            continue
        # Exact distinctive spans dominate; token coverage breaks partial
        # matches, and recency resolves otherwise equivalent versions.
        coverage = matched / max(1, len(terms))
        score = (10.0 if exact else 0.0) + coverage * 4.0
        scored.append((score, index, candidate))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _score, _index, record in scored[: max(1, min(8, limit))]]


def _guard_repeated_unchanged_result(
    name: str,
    arguments: Mapping[str, Any],
    result: Any,
    last_digest: str,
    prior_digests: set[str] | None = None,
) -> tuple[Any, str, str, bool]:
    """Require a causal boundary between identical non-visual tool results."""

    digest = _unchanged_result_digest(name, arguments, result)
    if (
        not name
        or name == "tool_search"
        or name in LOCAL_CONTROL_TOOL_NAMES
        or name in COMPUTER_ACTION_TOOLS
    ):
        return result, last_digest, digest, False
    repeated_failed_outcome = bool(
        prior_digests
        and digest in prior_digests
        and _result_failed_or_blocked(result)
    )
    if (not last_digest or digest != last_digest) and not repeated_failed_outcome:
        return result, digest, digest, False
    recovery: dict[str, Any] = {
        "error": "repeated_unchanged_result",
        "message": (
            "This call repeated a previously observed failed or unchanged bounded "
            "result without producing new task evidence. Reassess retained evidence "
            "and choose an action that changes or inspects different state."
        ),
        "task_blocked": False,
    }
    if name == "web_fetch":
        recovery.update(
            {
                "failure_scope": "arguments",
                "disposition": "change_capability",
                "alternative_tools": ["web_search", "browser_interact"],
            }
        )
    elif name in CAPABILITY_RECOVERY_ALTERNATIVES:
        recovery.update(
            {
                "failure_scope": "capability",
                "disposition": "change_capability",
                "alternative_tools": CAPABILITY_RECOVERY_ALTERNATIVES[name],
            }
        )
    return recovery, last_digest, digest, True


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
        if (
            not evidence_id
            or not name
            or name == "tool_search"
            or name in LOCAL_CONTROL_TOOL_NAMES
        ):
            continue
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            result = {}
        if isinstance(result, Mapping) and result.get("error") == "duplicate_tool_call":
            continue
        evidence[evidence_id] = {"name": name, "result": result}
    return evidence


def _action_audit_evidence(task: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Recover compacted evidence from the bounded, immutable action audit."""

    evidence: dict[str, dict[str, Any]] = {}
    actions = task.get("actions")
    if not isinstance(actions, list):
        return evidence
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        evidence_id = str(action.get("call_id") or "").strip()
        name = str(action.get("tool") or "").strip()
        if not evidence_id or not name or name in {
            "tool_search",
            "task_checkpoint",
            "task_compact",
            "task_recovery",
        }:
            continue
        try:
            result = json.loads(str(action.get("outcome") or "{}"))
        except ValueError:
            continue
        if isinstance(result, Mapping):
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
            "task_compact",
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
        if name == "task_checkpoint":
            try:
                result = json.loads(str(message.get("content") or "{}"))
            except ValueError:
                result = {}
            if (
                isinstance(result, Mapping)
                and result.get("error") == "unsupported_checkpoint"
                and result.get("retryable") is True
            ):
                # The result carries the exact admissible IDs. Permit one
                # control-plane correction without manufacturing another
                # external action merely to make checkpointing available.
                continue
            latest_control = index
        elif name == "task_recovery":
            latest_control = index
        elif name == "task_compact":
            # Compaction changes only the representation of already observed
            # state. It is neither task evidence nor a reason to invalidate a
            # checkpoint against the newest real external result.
            continue
        elif name == "task_expand":
            # Paging a retained receipt changes only the working representation.
            continue
        elif _is_duplicate_tool_result(message):
            # A locally rejected replay performed no external action and cannot
            # invalidate the preceding successful evidence. Keep checkpointing
            # available against that prior result; the validator still requires
            # the freshest real evidence ID and rejects the duplicate itself.
            continue
        elif name and name != "tool_search":
            latest_action = index
    return latest_action > latest_control


def _checkpoint_retry_pending(messages: list[dict[str, Any]]) -> bool:
    """Whether the immediately preceding result offered one bounded ID repair."""

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if str(message.get("tool_name") or "") != "task_checkpoint":
            return False
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            return False
        return (
            isinstance(result, Mapping)
            and result.get("error") == "unsupported_checkpoint"
            and result.get("retryable") is True
        )
    return False


def _normalize_progress_evidence(
    action: str,
    evidence_ids: list[str],
    evidence: Mapping[str, Mapping[str, Any]],
    freshest_evidence_id: str,
) -> tuple[list[str], bool]:
    """Repair progress provenance only; terminal assertions remain exact."""

    selected = [evidence.get(value) for value in evidence_ids]
    already_valid = (
        bool(selected)
        and all(item is not None for item in selected)
        and freshest_evidence_id in evidence_ids
    )
    if action != "progress" or already_valid or not freshest_evidence_id:
        return evidence_ids, False
    freshest = evidence.get(freshest_evidence_id)
    if freshest is None or not _supports_durable_progress(freshest):
        return evidence_ids, False
    return [freshest_evidence_id], True


def _evidence_authority(item: Mapping[str, Any] | None) -> str:
    """Classify a receipt without interpreting free-form model claims.

    Discovery identifies a possible next source/capability. Inspection observes
    state but does not create it. Concrete receipts can establish phase
    progress. Failed receipts can support only recovery or a proven blocker.
    """

    if not isinstance(item, Mapping):
        return "missing"
    name = str(item.get("name") or "")
    result = item.get("result")
    if _result_failed_or_blocked(result):
        return "failed"
    if name in {"tool_search", "web_search"}:
        return "discovery"
    if isinstance(result, Mapping):
        provenance = result.get("provenance")
        if isinstance(provenance, Mapping) and (
            provenance.get("authority") == "discovery_only"
            or provenance.get("citation_ready") is False
        ):
            return "discovery"
        if result.get("mode") == "discover":
            return "discovery"
        declared_authority = str(result.get("evidence_authority") or "")
        if declared_authority in {"mutation", "verification"}:
            return declared_authority
        if declared_authority in {
            "inspection",
            "unchanged_effect",
            "unverified_effect",
        }:
            return "inspection"
        if result.get("task_progress") is False:
            return "inspection"
    return "concrete"


def _task_environment_version(task: Mapping[str, Any]) -> int:
    state = task.get("task_state")
    if not isinstance(state, Mapping):
        return 0
    environment = state.get("environment")
    if not isinstance(environment, Mapping):
        return 0
    return int(environment.get("version") or 0)


def _task_controller_value(task: Mapping[str, Any], key: str) -> Any:
    state = task.get("task_state")
    if not isinstance(state, Mapping):
        return None
    controller = state.get("controller")
    if not isinstance(controller, Mapping):
        return None
    return controller.get(key)


def _action_audit_report(
    task: Mapping[str, Any],
    *,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Create an executor-independent, typed state-transition audit."""

    authority = _evidence_authority({"name": name, "result": result})
    operation = str(
        arguments.get("action") or arguments.get("intent") or "execute"
    ).strip()[:80]
    action_family = f"{name}:{operation}"
    environment_version = _task_environment_version(task)

    if name == "shell":
        # The typed shell intent defines the state domain.  A caller can name a
        # narrower evidence target explicitly; otherwise repeated inspections
        # of the same working tree close one slot instead of becoming endless
        # variations of ls/find/pwd.
        target = str(
            arguments.get("evidence_target")
            or arguments.get("cwd")
            or (result.get("cwd") if isinstance(result, Mapping) else "")
            or Path.cwd()
        )
    elif name == "workspace_file":
        target = f"{operation}:{arguments.get('path') or ''}"
    elif name in {"web_fetch", "web_crawl"}:
        target = str(arguments.get("url") or "")
    elif name == "web_search":
        target = str(arguments.get("query") or "")
    elif name in COMPUTER_ACTION_TOOLS:
        target = str(
            (result.get("url") if isinstance(result, Mapping) else "")
            or arguments.get("url")
            or arguments.get("target")
            or operation
        )
    else:
        target = str(
            arguments.get("path")
            or arguments.get("url")
            or arguments.get("query")
            or arguments.get("document_id")
            or name
        )
    target = " ".join(target.split())[:500]
    slot_seed = f"{action_family}\0{target}\0{environment_version}"
    evidence_slot = hashlib.sha256(slot_seed.encode()).hexdigest()[:24]

    changed_paths: list[str] = []
    if isinstance(result, Mapping):
        effect = result.get("effect_receipt")
        if isinstance(effect, Mapping) and isinstance(effect.get("changed_paths"), list):
            changed_paths = [
                str(path) for path in effect["changed_paths"] if str(path)
            ][:32]
        elif authority == "mutation" and str(result.get("path") or ""):
            changed_paths = [str(result["path"])]

    milestone_progress = bool(
        authority in {"mutation", "verification"}
        or (
            authority == "concrete"
            and (
                name in MILESTONE_SOURCE_TOOLS
                or (
                    name in COMPUTER_ACTION_TOOLS
                    and (
                        operation != "snapshot"
                        or (
                            isinstance(result, Mapping)
                            and isinstance(
                                result.get("verified_visual_observation"), Mapping
                            )
                        )
                    )
                )
            )
        )
    )

    transition = (
        "replan"
        if authority == "failed"
        else "retrieve"
        if authority in {"discovery", "inspection", "concrete"}
        and name in {
            *WEB_EVIDENCE_TOOLS,
            "document_search",
            "memory_read",
            "memory_search",
            "structured_read",
        }
        else "verify"
        if authority == "verification"
        else "act"
    )
    current_subtask = str(
        _task_controller_value(task, "current_subtask")
        or task.get("completion_criteria")
        or task.get("objective")
        or ""
    )
    unresolved = _task_controller_value(task, "unresolved_evidence")
    unresolved = unresolved if isinstance(unresolved, list) else []
    state_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "subtask": current_subtask,
                "environment_version": environment_version,
                "unresolved_evidence": unresolved,
                "action_family": action_family,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:24]
    return {
        "audit_id": f"audit-{str(call_id)[:118]}",
        "evidence_id": str(call_id)[:128],
        "transition": transition,
        "action_family": action_family,
        "authority": authority,
        "target": target,
        "evidence_slot": evidence_slot,
        "state_fingerprint": state_fingerprint,
        "changed_paths": changed_paths,
        "executor_succeeded": not _result_failed_or_blocked(result),
        # A tool can succeed and even close a new knowledge slot without
        # satisfying a durable phase. This typed bit is deliberately narrower
        # than epistemic/environmental progress and is checked independently
        # at every checkpoint.
        "milestone_progress": milestone_progress,
    }


def _latest_audit(task: Mapping[str, Any]) -> dict[str, Any]:
    state = task.get("task_state")
    if not isinstance(state, Mapping):
        return {}
    reports = state.get("audit_reports")
    if not isinstance(reports, list):
        return {}
    return next(
        (dict(item) for item in reversed(reports) if isinstance(item, Mapping)),
        {},
    )


def _completion_is_audited(
    task: Mapping[str, Any], evidence_ids: list[str]
) -> bool:
    """Require post-mutation verification before terminal completion."""

    state = task.get("task_state")
    if not isinstance(state, Mapping):
        # Compatibility for tasks created before the audited-state schema.
        return True
    environment = state.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    environment_version = int(environment.get("version") or 0)
    reports = state.get("audit_reports")
    reports = reports if isinstance(reports, list) else []
    selected = [
        item
        for item in reports
        if isinstance(item, Mapping)
        and str(item.get("evidence_id") or "") in evidence_ids
    ]
    if not selected:
        return False
    if not any(item.get("milestone_progress") is True for item in selected):
        return False
    if environment_version <= 0:
        return any(
            str(item.get("authority") or "")
            in {"concrete", "verification", "inspection"}
            for item in selected
        )
    return any(
        str(item.get("authority") or "") == "verification"
        and int(item.get("environment_version_after") or -1) == environment_version
        for item in selected
    )


def _checkpoint_has_milestone(
    task: Mapping[str, Any], evidence_ids: list[str]
) -> bool:
    state = task.get("task_state")
    if not isinstance(state, Mapping):
        return False
    reports = state.get("audit_reports")
    if not isinstance(reports, list):
        return False
    selected_ids = {str(value) for value in evidence_ids}
    return any(
        isinstance(item, Mapping)
        and str(item.get("evidence_id") or "") in selected_ids
        and item.get("milestone_progress") is True
        for item in reports
    )


def _supports_durable_progress(item: Mapping[str, Any] | None) -> bool:
    return _evidence_authority(item) in {"concrete", "mutation", "verification"}


def _direct_alternative_tools(result: Mapping[str, Any]) -> list[str]:
    alternatives = result.get("alternative_tools")
    if not isinstance(alternatives, list):
        return []
    return list(
        dict.fromkeys(
            str(item)
            for item in alternatives
            if str(item) != "background_task" and tool_schemas([str(item)])
        )
    )[:3]


def _successor_tools(name: str, result: Any) -> list[str]:
    """Keep only capabilities that remain causally useful after this result."""

    if isinstance(result, Mapping) and result.get("error") in {
        "browser_navigation_error",
        "duplicate_tool_call",
        "repeated_unchanged_result",
    }:
        # A locally rejected/no-change call, or a rendered network-error page,
        # cannot advance through another sticky call in the same action space.
        # Return to discovery so the controller can select a materially
        # different capability.
        return []
    alternatives = (
        _direct_alternative_tools(result) if isinstance(result, Mapping) else []
    )
    if alternatives:
        return alternatives
    if not name or name == "background_task" or name in NON_STICKY_RESULT_TOOLS:
        return []
    return [name]


def _structured_action_phase(
    messages: list[dict[str, Any]], active_tools: list[str]
) -> bool:
    """Native thinking is only for the task's first unresolved planning turn."""

    if active_tools:
        return True
    for message in reversed(messages):
        if message.get("role") == "tool":
            return True
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            return True
    return False


def _latest_result_requires_replan(messages: list[dict[str, Any]]) -> bool:
    """Give the model a bounded planning pass after typed non-progress evidence."""

    for message in reversed(messages):
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                # One native-thinking pass already had the chance to replan.
                # If it emitted prose instead of an action, the local retry
                # directive keeps that plan resident and the next pass must
                # execute without reopening another identical thought cycle.
                return False
            continue
        if message.get("role") != "tool":
            continue
        name = str(message.get("tool_name") or "")
        if name in {"", "tool_search", "task_checkpoint", "task_compact"}:
            continue
        try:
            result = json.loads(str(message.get("content") or "{}"))
        except ValueError:
            return False
        return (
            isinstance(result, Mapping)
            and result.get("task_progress") is False
            and not _result_failed_or_blocked(result)
        )
    return False


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
            pending = not bool(_direct_alternative_tools(result))
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


def _trailing_capability_failures(task: Mapping[str, Any]) -> dict[str, int]:
    """Recover actual executor failures, never successful inspections."""

    counts: dict[str, int] = {}
    actions = task.get("actions")
    if not isinstance(actions, list):
        return counts
    for action in reversed(actions):
        if not isinstance(action, Mapping):
            continue
        name = str(action.get("tool") or "")
        if not name or name in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search", "web_search"}:
            continue
        outcome = _audit_mapping(action.get("outcome"))
        if action.get("ok") is True and not _result_failed_or_blocked(outcome):
            break
        if _result_failed_or_blocked(outcome) or action.get("ok") is False:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _apply_capability_retry_budget(
    name: str,
    result: Any,
    failures: dict[str, int],
) -> Any:
    """Turn repeated capability failure into an observable route change.

    This never upgrades failure into progress. It prevents a controller from
    spending every renewed slice on superficial variants of the same dead
    action space. A concrete success clears the budget and makes every
    capability available again.
    """

    if not name or name in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search", "web_search"}:
        return result
    if not _result_failed_or_blocked(result):
        # A successful inspection proves that the capability executed. Whether
        # it advanced the task is tracked separately by the audited evidence
        # slot and stagnation state; it is not a capability failure.
        if _evidence_authority({"name": name, "result": result}) == "inspection":
            failures.pop(name, None)
        else:
            failures.clear()
        return result
    failures[name] = failures.get(name, 0) + 1
    if failures[name] < MAX_FAILED_CAPABILITY_ATTEMPTS:
        return result
    alternatives = [
        candidate
        for candidate in CAPABILITY_RECOVERY_ALTERNATIVES.get(name, [])
        if tool_schemas([candidate])
    ]
    return {
        "error": "capability_retry_exhausted",
        "message": (
            f"{name} failed {failures[name]} times without a successful "
            "executor result. "
            "Change capability before trying this action space again."
        ),
        "failure_scope": "capability",
        "disposition": "change_capability",
        "alternative_tools": alternatives,
        "task_blocked": False,
        "last_result": _bounded_tool_result(result),
    }


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
        prepare_action_residency: Callable[[], None] | None = None,
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
        self.prepare_action_residency = prepare_action_residency
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
        self._portal_session_seed = portal_session_id or secrets.token_urlsafe(24)
        self._active_portal_session = ""
        self.active = threading.Event()
        self._wake = threading.Event()
        self._client = client or httpx.Client(timeout=httpx.Timeout(request_timeout_s))
        self._owns_client = client is None
        self._thread = threading.Thread(
            target=self._run, name="omni-background-agent", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.stop.set()
        self._wake.set()
        # Relinquish the durable lease before waiting for an in-flight HTTP
        # request.  Service managers may enforce a stop deadline shorter than
        # the request timeout; releasing after join can therefore turn an
        # orderly deployment into a false expired-lease crash.
        released = self.store.release_owner(self.owner)
        if released:
            logger.info(
                "released %d background task lease(s) for orderly shutdown", released
            )
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
        headers = {"Authorization": f"Bearer {self.token}"}
        if self._active_portal_session:
            headers["Cookie"] = "omni_portal_session=" + self._active_portal_session
        return headers

    def _record_action(
        self,
        task_id: str,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        result: Any,
        *,
        historical: bool = False,
    ) -> dict[str, Any] | None:
        try:
            receipt = None
            if name in COMPUTER_ACTION_TOOLS and isinstance(result, Mapping):
                raw_receipt = result.get("action_receipt")
                if isinstance(raw_receipt, Mapping):
                    action = str(raw_receipt.get("action") or "")[:80]
                    target = " ".join(
                        str(raw_receipt.get("target") or "").split()
                    )[:240]
                    if action and target:
                        # Coordinates expire with their screenshot. Persist only
                        # the verified semantic action identity needed for task
                        # continuity and accurate reporting after scope reduction.
                        receipt = {"action": action, "target": target}
            evidence_record = None
            if name not in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}:
                evidence_result = json.dumps(
                    _bounded_tool_result(result),
                    ensure_ascii=False,
                    default=str,
                )
                evidence_record = {
                    "evidence_id": str(call_id)[:128],
                    "tool": name or "unknown",
                    "arguments": _audit_json(arguments, 4_000),
                    "result": evidence_result[:MAX_TOOL_RESULT_CHARS],
                    "result_sha256": hashlib.sha256(
                        evidence_result.encode("utf-8")
                    ).hexdigest(),
                }
            current = self.store.get(task_id) or {}
            audit_report = (
                None
                if name in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}
                else _action_audit_report(
                    current,
                    call_id=call_id,
                    name=name or "unknown",
                    arguments=arguments,
                    result=result,
                )
            )
            return self.store.record_action(
                task_id,
                self.owner,
                call_id=call_id,
                tool=name or "unknown",
                arguments=_audit_json(arguments, MAX_ACTION_ARGUMENT_CHARS),
                outcome=_audit_json(result, MAX_ACTION_OUTCOME_CHARS),
                ok=not _result_failed_or_blocked(result),
                receipt=receipt,
                evidence_record=evidence_record,
                audit_report=audit_report,
                recorded_at=0 if historical else None,
            )
        except Exception as error:  # noqa: BLE001 - auditing must not stop the task
            logger.warning("could not audit background tool call %s: %s", name, error)
            return None

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
            if self.foreground_active.is_set():
                continue
            if self.prepare_action_residency is not None:
                try:
                    self.prepare_action_residency()
                except Exception as error:  # noqa: BLE001 - governor remains authoritative
                    logger.warning(
                        "could not prepare background action residency: %s", error
                    )
                if self.foreground_active.is_set():
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
                        evidence_records=_compaction_evidence_records(retained),
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
        # Browser/web indexes and virtual memory are task-local. Sharing the
        # foreground cookie, or one cookie across durable tasks, lets an old
        # project's assistant narration outrank the empty current workspace.
        self._active_portal_session = _background_portal_session(
            self._portal_session_seed, task_id
        )
        task_started_at = time.monotonic()
        last_progress_at: float | None = None
        messages = copy.deepcopy(task.get("messages") or [])
        # Older tasks can contain free-form checkpoint reports from a previous
        # runtime.  Strip those claims before either restoring the audit or
        # presenting the retained transcript to the model.
        _sanitize_checkpoint_history(messages)
        if not messages:
            messages = [
                {"role": "system", "content": _task_system_prompt(task)},
                {"role": "user", "content": TASK_START_REQUEST},
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
        # Tasks accepted by an older worker may retain a second full copy of
        # objective/criteria in the initial user turn. The authoritative copy
        # is deterministically pinned in the system contract above; collapse
        # only that known legacy envelope before the next bounded request.
        if (
            len(messages) > 1
            and messages[1].get("role") == "user"
            and str(messages[1].get("content") or "").lstrip().startswith("<objective>")
        ):
            messages[1] = {"role": "user", "content": TASK_START_REQUEST}
        self._restore_action_audit(task_id, messages)
        seen = {
            *_seen_tool_fingerprints(messages),
            *(str(value) for value in task.get("tool_fingerprints", []) if value),
        }
        last_tool_fingerprint = _latest_tool_fingerprint(messages)
        result_digests = {
            str(value) for value in task.get("result_digests", []) if value
        }
        capability_failures = _trailing_capability_failures(task)
        last_external_result_digest = _latest_external_result_digest(messages)
        phase_action_count = _uncheckpointed_action_count(task)
        active_tools = [
            name
            for name in task.get("active_tools", [])
            if isinstance(name, str) and name != "background_task"
        ]
        tools_used = [
            name for name in task.get("tools_used", []) if isinstance(name, str) and name
        ]
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
                    messages=_durable_task_messages(messages),
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
            # The objective is stable, but accepted milestones and user
            # directions evolve across a long task. Keep that current plan in
            # the pinned system contract instead of hoping semantic retrieval
            # will recover it from an old compacted turn.
            messages[0] = {
                "role": "system",
                "content": _task_system_prompt(current),
            }
            added_guidance = _append_guidance(messages, current, seen_guidance)
            if added_guidance:
                active_tools = []
                phase_action_count = 0
                logger.info(
                    "background task %s accepted %d conversational update(s)",
                    task_id,
                    added_guidance,
                )
            if _compaction_tool_available(messages, active_tools):
                before = copy.deepcopy(messages)
                compacted = _compact_task_messages(messages, current, force=True)
                if compacted != before:
                    receipt = _compaction_receipt(
                        before,
                        compacted,
                        current,
                        reason="automatic_context_limit",
                    )
                    messages = compacted
                    self.store.compact_context(
                        task_id,
                        self.owner,
                        messages=_durable_task_messages(messages),
                        receipt=receipt,
                        evidence_records=_compaction_evidence_records(before),
                    )
                    messages[0] = {
                        "role": "system",
                        "content": _task_system_prompt(
                            current, expand_available=True
                        ),
                    }
                    logger.info(
                        "background task %s compacted automatically before inference: %s",
                        task_id,
                        json.dumps(receipt, sort_keys=True),
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
            phase_boundary = (
                phase_action_count >= MAX_PHASE_ACTIONS and can_checkpoint
            )
            # Before compaction the exact result is still resident in the
            # ordinary transcript. Offer paging only after older turns may
            # have left L0; otherwise the maintenance action needlessly
            # competes with the next concrete task action.
            expand_available = _task_expand_available(
                messages,
                compacted=bool(current.get("compaction")),
            )
            resident_context_tokens = _resident_task_context_tokens()
            replan_after_inspection = _latest_result_requires_replan(messages)
            schemas = _background_tool_contract(
                active_tools,
                recovery_required=recovery_required,
                phase_boundary=phase_boundary,
                expand_available=expand_available,
                can_checkpoint=can_checkpoint,
                resident_context_tokens=resident_context_tokens,
                recovery_exploration=replan_after_inspection,
            )
            offered_tool_names = {
                str(function.get("name") or "")
                for schema in schemas
                if isinstance(schema, Mapping)
                and isinstance((function := schema.get("function")), Mapping)
            }
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
            structured_action_phase = action_after_discovery or _structured_action_phase(
                messages, active_tools
            )
            inference_messages = _computer_action_messages(
                messages,
                current,
                active_tools,
                recovery_required=recovery_required,
            )
            if phase_boundary:
                inference_messages = [*inference_messages]
                inference_messages.append(
                    {
                        "role": "user",
                        "content": context_text(
                            "directives", "background_phase_boundary"
                        ).format(
                            action_count=phase_action_count,
                            evidence_id=_freshest_evidence_id(messages),
                        ),
                    }
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
                "think": not structured_action_phase or replan_after_inspection,
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
                "portal_background_worker": True,
                "portal_virtual_query": _task_virtual_query(
                    current,
                    resident_context_tokens=resident_context_tokens,
                ),
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
                    messages=_durable_task_messages(messages),
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
            message = data.get("message")
            if not isinstance(message, Mapping):
                raise RuntimeError("background inference returned no assistant message")
            adapter_metadata = data.get("adapter")
            round_visual_observation = (
                str(adapter_metadata.get("observation") or "")[:3000]
                if isinstance(adapter_metadata, Mapping)
                and "<visual_observation>"
                in str(adapter_metadata.get("observation") or "")
                else ""
            )
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
            calls = _tool_calls(data)
            if len(calls) > 1:
                logger.warning(
                    "background task %s proposed %d tool calls; executing only the "
                    "first so every external result receives a fresh self-check",
                    task_id,
                    len(calls),
                )
                calls = calls[:1]
            if calls:
                # The durable protocol must describe only calls that the worker
                # will actually execute. Later proposed calls were planned
                # without seeing the first result and are therefore stale.
                assistant["tool_calls"] = copy.deepcopy(calls)
            else:
                assistant.pop("tool_calls", None)
            messages.append(assistant)
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
                phase_action_count = 0
                checkpoint = self.store.checkpoint(
                    task_id,
                    self.owner,
                    messages=_durable_task_messages(messages),
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
            reset_executor_context = False
            for call in calls:
                function = call.get("function")
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, Mapping)
                    else ""
                )
                arguments = _arguments(call)
                grounding_receipt = None
                if name == "browser_interact":
                    arguments, grounding_receipt = _ground_visual_click(
                        arguments, round_visual_observation
                    )
                call_id = str(call.get("id") or secrets.token_hex(6))
                if (
                    name in {*LOCAL_CONTROL_TOOL_NAMES, "request_camera_view"}
                    and name not in offered_tool_names
                ):
                    rejected_result = {
                        "error": "tool_not_offered",
                        "message": (
                            "This tool was not in the action contract for the current "
                            "inference round. Use one of the currently offered tools."
                        ),
                        "offered_tools": sorted(offered_tool_names),
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name or "unknown",
                            "tool_call_id": call_id,
                            "content": json.dumps(rejected_result),
                        }
                    )
                    self._record_action(
                        task_id,
                        call_id,
                        name or "unknown",
                        arguments,
                        rejected_result,
                    )
                    stalls += 1
                    continue
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
                        messages=_durable_task_messages(messages),
                        receipt=receipt,
                        evidence_records=_compaction_evidence_records(before),
                    )
                    stalls = 0
                    continue
                if name == "task_expand":
                    raw_ids = arguments.get("evidence_ids")
                    evidence_ids = (
                        list(dict.fromkeys(str(value) for value in raw_ids if str(value)))[:8]
                        if isinstance(raw_ids, list)
                        else []
                    )
                    query = " ".join(str(arguments.get("query") or "").split())[:500]
                    records = self.store.expand_evidence(task_id, evidence_ids)
                    if query:
                        latest_task = self.store.get(task_id) or current
                        records.extend(_search_task_evidence(latest_task, query, limit=8))
                    records = list(
                        {
                            str(record.get("evidence_id") or ""): record
                            for record in records
                            if isinstance(record, Mapping)
                            and str(record.get("evidence_id") or "")
                        }.values()
                    )[:8]
                    found = {str(record.get("evidence_id") or "") for record in records}
                    expand_result = (
                        {
                            "expanded": records,
                            "missing_evidence_ids": [
                                value for value in evidence_ids if value not in found
                            ],
                            "matched_query": query,
                            "authority": "retained_pre_compaction_tool_receipts",
                            "task_progress": False,
                        }
                        if (evidence_ids or query) and records
                        else {
                            "error": "evidence_not_found",
                            "message": (
                                "Use evidence IDs present in tagged focus items or query "
                                "with an exact path, URL, symbol, or diagnostic from the task."
                            ),
                            "requested_evidence_ids": evidence_ids,
                            "query": query,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "tool_call_id": call_id,
                            "content": json.dumps(expand_result),
                        }
                    )
                    self._record_action(
                        task_id, call_id, name, arguments, expand_result
                    )
                    stalls = 0 if records else stalls + 1
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
                    # The validator below already holds a private parsed copy
                    # of these arguments.  Remove the model-authored prose from
                    # the recurrent transcript before any rejection, progress
                    # continuation, compaction, or persistence can replay it as
                    # if it were established task history.
                    _sanitize_checkpoint_history(messages, call_id=call_id)
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
                    raw_remaining = arguments.get("remaining_requirements")
                    remaining_requirements = (
                        [
                            " ".join(str(value).split())[:300]
                            for value in raw_remaining
                            if str(value).strip()
                        ][:8]
                        if isinstance(raw_remaining, list)
                        else []
                    )
                    raw_ids = arguments.get("evidence_ids")
                    evidence_ids = (
                        [str(value) for value in raw_ids if str(value)]
                        if isinstance(raw_ids, list)
                        else []
                    )
                    evidence = {
                        **_action_audit_evidence(latest or current),
                        **_tool_evidence(messages),
                    }
                    freshest_evidence_id = _freshest_evidence_id(messages)
                    evidence_ids, evidence_normalized = _normalize_progress_evidence(
                        action,
                        evidence_ids,
                        evidence,
                        freshest_evidence_id,
                    )
                    selected = [evidence.get(value) for value in evidence_ids]
                    valid_refs = bool(selected) and all(item is not None for item in selected)
                    evidence_authorities = [
                        _evidence_authority(item) for item in selected
                    ]
                    supports_progress = any(
                        authority in {"concrete", "mutation", "verification"}
                        for authority in evidence_authorities
                    )
                    supports_milestone = _checkpoint_has_milestone(
                        latest or current, evidence_ids
                    )
                    supports_completion = (
                        supports_progress
                        and all(
                            authority
                            in {"concrete", "mutation", "verification", "inspection"}
                            for authority in evidence_authorities
                        )
                        and _completion_is_audited(
                            latest or current, evidence_ids
                        )
                    )
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
                        and (
                            (action == "complete" and not remaining_requirements)
                            or (
                                action in {"progress", "blocked"}
                                and bool(remaining_requirements)
                            )
                        )
                        and valid_refs
                        and cites_freshest
                        and (
                            (
                                action == "progress"
                                and supports_progress
                                and supports_milestone
                            )
                            or (action == "complete" and supports_completion)
                            or action == "blocked"
                        )
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
                        valid_ids = [
                            eid
                            for eid, item in evidence.items()
                            if _evidence_authority(item)
                            in {"concrete", "mutation", "verification", "inspection"}
                        ]
                        failed_ids = [eid for eid, item in evidence.items() if item is not None and _result_failed_or_blocked(item["result"])]
                        retryable = not _checkpoint_retry_pending(messages)
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
                            "remaining_requirements_required": (
                                "non-empty for progress/blocked; empty for complete"
                            ),
                            "evidence_authority_required": (
                                "progress requires a typed milestone receipt (source "
                                "acquisition, verified mutation, external interaction, or "
                                "verification); completion requires the same and, after any "
                                "environment mutation, verification from the current version; "
                                "a successful information probe alone never proves a milestone"
                            ),
                            "retryable": retryable,
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
                        active_tools = []
                        phase_action_count = 0
                        accepted_result = {
                            "accepted": True,
                            "action": action,
                            "evidence_ids": evidence_ids,
                            "evidence_normalized": evidence_normalized,
                        }
                        neutral_progress = (
                            "Phase checkpoint retained against concrete evidence "
                            f"{', '.join(evidence_ids[:4])}; work remains."
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": name,
                                "tool_call_id": call_id,
                                "content": json.dumps(accepted_result),
                            }
                        )
                        self._record_action(
                            task_id,
                            call_id,
                            name,
                            arguments,
                            accepted_result,
                        )
                        checkpoint = self.store.checkpoint(
                            task_id,
                            self.owner,
                            messages=_durable_task_messages(messages),
                            active_tools=active_tools,
                            tools_used=tools_used,
                            applied_guidance_ids=list(seen_guidance),
                            tool_fingerprints=list(seen),
                            result_digests=list(result_digests),
                            progress=neutral_progress,
                            status="running",
                            current_stage=context_text(
                                "task_stages", "continuing_checkpoint"
                            ),
                            controller_transition={
                                "action": action,
                                "evidence_ids": evidence_ids,
                                "remaining_requirements": remaining_requirements,
                            },
                        )
                        if checkpoint is None or checkpoint.get("status") == "cancelled":
                            return
                        if eligible and self.on_progress is not None:
                            last_progress_at = now
                            self.on_progress(
                                {
                                    "task_id": task_id,
                                    "status": "running",
                                    "result": neutral_progress,
                                }
                            )
                        latest_task = self.store.get(task_id) or current
                        messages = _fresh_executor_messages(
                            latest_task,
                            reason="verified_phase_checkpoint",
                        )
                        renewed = self.store.renew_executor_context(
                            task_id,
                            self.owner,
                            messages=_durable_task_messages(messages),
                            current_stage=context_text(
                                "task_stages", "continuing_checkpoint"
                            ),
                        )
                        if renewed is None or renewed.get("status") == "cancelled":
                            return
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
                        messages=_durable_task_messages(messages),
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
                        controller_transition={
                            "action": action,
                            "evidence_ids": evidence_ids,
                            "remaining_requirements": remaining_requirements,
                        },
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
                    if name in active_tools:
                        active_tools = [item for item in active_tools if item != name]
                    stalls += 1
                else:
                    seen.add(fingerprint)
                    if name not in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}:
                        last_tool_fingerprint = fingerprint
                    slice_tool_calls += 1
                    self.store.update_stage(
                        task_id,
                        self.owner,
                        context_text("task_stages", "running_tool").format(
                            tool=name or "unknown"
                        ),
                    )
                    if name == "tool_search":
                        preflight_result = _background_discovery_preflight(arguments)
                    elif name == "request_camera_view" and not _task_allows_physical_camera(
                        latest or current
                    ):
                        preflight_result = _camera_scope_rejection()
                    elif name == "web_fetch":
                        preflight_result = _web_fetch_preflight(
                            messages,
                            latest or current,
                            arguments,
                        )
                    else:
                        preflight_result = None
                    if preflight_result is not None:
                        # Discovery is evidence about where a fetch may go. A
                        # model-authored URL is not. Keep the leaf tool active
                        # and return the exact grounded choices without issuing
                        # any network request.
                        result = preflight_result
                    else:
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
                    if name == "tool_search":
                        result = _filter_background_discovery(
                            result, latest or current
                        )
                    if (
                        name == "workspace_file"
                        and str(arguments.get("action") or "") in {"list", "read"}
                        and isinstance(result, Mapping)
                        and not _result_failed_or_blocked(result)
                    ):
                        result = dict(result)
                        result["task_progress"] = False
                        result["evidence_authority"] = "inspection"
                    if (
                        name in COMPUTER_ACTION_TOOLS
                        and str(arguments.get("action") or "") == "snapshot"
                        and round_visual_observation
                        and isinstance(result, Mapping)
                        and isinstance(result.get("visual_change"), Mapping)
                        and result["visual_change"].get("materially_changed") is False
                        and not isinstance(
                            result.get("verified_visual_observation"), Mapping
                        )
                    ):
                        # The model perceived the frame immediately before asking for a
                        # no-op verification snapshot, and the executor proved that the
                        # returned frame is unchanged. Preserve that tagged semantic
                        # reading as fresh evidence so exact text or terminal state does
                        # not vanish when image tensors are later compacted.
                        result = dict(result)
                        result["verified_visual_observation"] = {
                            "provenance": "current_unchanged_browser_frame",
                            "observation": without_parent_frame_coordinates(
                                round_visual_observation,
                                max_chars=3000,
                            ),
                        }
                    if grounding_receipt is not None and isinstance(result, Mapping):
                        existing_grounding = result.get("visual_grounding")
                        result = dict(result)
                        if isinstance(existing_grounding, Mapping):
                            result["visual_grounding"] = {
                                **existing_grounding,
                                "semantic_source": grounding_receipt,
                            }
                        else:
                            result["visual_grounding"] = grounding_receipt
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
                (
                    result,
                    last_external_result_digest,
                    digest,
                    repeated_result,
                ) = _guard_repeated_unchanged_result(
                    name,
                    arguments,
                    result,
                    last_external_result_digest,
                    result_digests,
                )
                result = _apply_capability_retry_budget(
                    name,
                    result,
                    capability_failures,
                )
                change_capability = (
                    isinstance(result, Mapping)
                    and result.get("disposition") == "change_capability"
                )
                if change_capability:
                    direct_alternatives = _direct_alternative_tools(result)
                    if direct_alternatives:
                        # The local executor has already supplied a concrete,
                        # allowlisted equivalent capability. Expose it immediately;
                        # another classification and discovery pair only adds two
                        # generative rounds and lets small models retry a dead path.
                        active_tools = direct_alternatives
                        recovery_required = False
                    else:
                        active_tools = []
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
                    else:
                        active_tools = []
                elif name and name != "background_task":
                    active_tools = _successor_tools(name, result)
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": name or "unknown",
                    "content": "",
                }
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
                if isinstance(screenshot, Mapping) and screenshot.get("data"):
                    _discard_visual_frames(
                        messages,
                        replacement_note=(
                            "This rendered screenshot was superseded by a newer "
                            "computer-use frame."
                        ),
                    )
                elif (
                    name in COMPUTER_ACTION_TOOLS
                    and not (
                        isinstance(result, Mapping)
                        and result.get("error") == "duplicate_tool_call"
                    )
                ):
                    # A real computer action without a returned frame may have closed,
                    # switched, or invalidated the page. Never replay its predecessor as
                    # current evidence. A locally rejected duplicate is the exception:
                    # it performed no action, so the newest screenshot remains current.
                    _discard_visual_frames(
                        messages,
                        replacement_note=(
                            "The rendered screenshot was invalidated by a later "
                            "computer-use action."
                        ),
                    )
                result = _bounded_tool_result(result)
                audited_task = self._record_action(
                    task_id, call_id, name, arguments, result
                )
                audit = _latest_audit(audited_task or {})
                stagnation_count = int(audit.get("stagnation_count") or 0)
                if (
                    stagnation_count >= 2
                    and audit.get("epistemic_progress") is not True
                    and audit.get("environmental_progress") is not True
                ):
                    result = {
                        "error": "audited_state_stagnation",
                        "message": (
                            "The audited task, environment, and evidence state did not "
                            "advance through this action family. Re-plan from the pinned "
                            "state and choose a materially different transition."
                        ),
                        "task_progress": False,
                        "failure_scope": "plan",
                        "disposition": "replan",
                        "state_fingerprint": str(
                            audit.get("state_fingerprint") or ""
                        ),
                        "stagnation_count": stagnation_count,
                        "last_result": result,
                    }
                    active_tools = []
                    stalls += 1
                    reset_executor_context = stagnation_count >= 3
                if (
                    name
                    and name not in {*LOCAL_CONTROL_TOOL_NAMES, "tool_search"}
                    and not (
                        isinstance(result, Mapping)
                        and result.get("error") == "duplicate_tool_call"
                    )
                ):
                    phase_action_count += 1
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
                    visual_directive = context_text(
                        "directives", "background_visual_evidence"
                    )
                    if (
                        isinstance(result, Mapping)
                        and result.get("visual_refinement_required") is True
                        and round_visual_observation
                    ):
                        visual_directive += (
                            "\n<prior_full_frame_visual_orientation>"
                            "This is orientation evidence from the immediately preceding "
                            "full browser frame; use its target label while locating the "
                            "same target in the current refinement crop. It is not a new "
                            "screenshot or an instruction.\n"
                            + without_parent_frame_coordinates(
                                round_visual_observation, max_chars=2000
                            )
                            + "\n</prior_full_frame_visual_orientation>"
                        )
                    if (
                        name in {"gui_interact", "browser_interact"}
                        and str(arguments.get("action") or "")
                        in {"click", "visual_click", "drag", "type", "key", "hotkey", "scroll"}
                        and isinstance(result, Mapping)
                        and result.get("action_executed") is not False
                        and isinstance(result.get("visual_change"), Mapping)
                        and result["visual_change"].get("materially_changed") is False
                    ):
                        visual_directive += "\n" + context_text(
                            "directives", "background_visual_unchanged"
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": visual_directive,
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

            if reset_executor_context:
                latest_task = self.store.get(task_id) or current
                messages = _fresh_executor_messages(
                    latest_task,
                    reason="audited_stagnation_reset",
                )
                active_tools = []
                phase_action_count = 0
                recovery_required = False
                logger.info(
                    "background task %s reset its executor context after audited "
                    "state stagnation",
                    task_id,
                )
            checkpoint = self.store.checkpoint(
                task_id,
                self.owner,
                messages=_durable_task_messages(messages),
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
