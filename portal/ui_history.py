"""Request-local model-history projection without input or executor authority.

Only portal route code may construct this object and register a just-executed
server result. Nothing in a request JSON object can construct that provenance.
The payload carries no origin marker or new adapter field.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _digest(value: Any) -> str:
    # Exactly portal.computer_use._digest: default ASCII escaping and no NaN.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _plain(value: Any) -> Any:
    """Ignore only media removed by the existing index-preserving portal pass."""
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        result = {key: item for key, item in value.items()
                  if key not in {"images", "audios", "videos", "documents"}}
        if result.get("tool_calls"):
            result.pop("thinking", None)
        return result
    return value


def _snapshot(value: Any) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("session_id"), str)
            and bool(value["session_id"]) and isinstance(value.get("revision"), str)
            and bool(value["revision"]) and isinstance(value.get("nodes"), list)
            and isinstance(value.get("scope"), dict)
            and isinstance(value["scope"].get("id"), str)
            and value["scope"].get("kind") in ("desktop", "browser"))


def _identity(result: dict[str, Any], operation: str) -> tuple[tuple[str, str], bool] | None:
    receipt = result.get("ui_state_receipt")
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "robit.omni.ui-state-receipt.v1"):
        return None
    # Only an actual observe/verify dispatch observes fresh top-level state.
    # Execute/workflow can return cached before/after/terminal bodies; these
    # remain eligible historical pairs but never retire another observation.
    fresh = operation in {"observe", "verify"} and not result.get("history_projection")
    effect = result.get("effect_receipt")
    if isinstance(effect, dict) and effect.get("effect_state") == "replayed":
        fresh = False
    choices = [(result.get("state"), fresh)]
    choices.extend((result.get(name), False) for name in ("after", "terminal", "before"))
    outcomes = result.get("outcomes")
    if isinstance(outcomes, list) and outcomes and isinstance(outcomes[-1], dict):
        choices.extend((outcomes[-1].get(name), False) for name in ("after", "before"))
    for state, positive in choices:
        if not _snapshot(state):
            continue
        if (receipt.get("session_id") == _digest(state["session_id"])
                and receipt.get("scope_id") == state["scope"]["id"]
                and receipt.get("revision") == state["revision"]):
            return ((receipt["session_id"], state["scope"]["kind"]),
                    positive and not state.get("history_projection"))
    return None


def _references(contract: Any) -> set[str]:
    """Read only explicit typed control references; never walk arbitrary data."""
    if not isinstance(contract, dict):
        return set()
    refs = [contract.get("node_id")]
    for key, field in (("target", "node_id"), ("args", "to_node_id")):
        if isinstance(contract.get(key), dict):
            refs.append(contract[key].get(field))
    for key in ("expect", "predicates", "final_predicates"):
        predicates = contract.get(key)
        if isinstance(predicates, list) and len(predicates) <= 20:
            refs.extend(item.get("node_id") for item in predicates if isinstance(item, dict))
    return {ref for ref in refs if isinstance(ref, str) and ref and ref != "$target"}


def _retired(result: dict[str, Any], controller: tuple[str, str],
             arguments: dict[str, Any]) -> dict[str, Any]:
    """Retire known snapshot positions; leave all immutable receipt trees exact."""
    def snapshot(value: Any, refs: set[str]) -> Any:
        if not _snapshot(value) or (_digest(value["session_id"]), value["scope"]["kind"]) != controller:
            return copy.deepcopy(value)
        metadata = {key: copy.deepcopy(item) for key, item in value.items()
                    if key not in {"nodes", "text", "window_visibility"}}
        # Historical readback is restricted to the selected/explicit effect
        # controls. It carries no operation, geometry, or executable offer.
        if len(refs) <= 24:
            evidence = [{key: copy.deepcopy(node[key]) for key in
                         ("id", "generation", "scope_id", "role", "name", "value",
                          "checked", "selected", "focused", "text_selection", "text_observation", "text_insertion")
                          if key in node}
                        for node in value["nodes"] if isinstance(node, dict)
                        and node.get("id") in refs and not node.get("sensitive")
                        and node.get("visible") is not False]
            if evidence:
                metadata["historical_control_evidence"] = evidence
        metadata["history_projection"] = {
            "schema": "robit.omni.ui-history-projection.v1",
            "historical": True, "current_observation": False,
            "executable_node_map": False, "absence_claims_supported": False,
            "retired_fields": [key for key in ("nodes", "text", "window_visibility") if key in value],
            "retired_node_count": len(value["nodes"]), "model_snapshot_sha256": _digest(value),
        }
        return metadata

    compact = copy.deepcopy(result)
    refs = _references(arguments) | _references(arguments.get("step")) | _references(result.get("action"))
    selected = result.get("selected_action_id")
    candidates = result.get("candidates")
    if isinstance(selected, str) and selected and isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("id") == selected:
                refs |= _references(candidate)
    for key in ("state", "before", "after", "terminal"):
        if key in result:
            compact[key] = snapshot(result[key], refs)
    outcomes = result.get("outcomes")
    if isinstance(outcomes, list):
        for outcome, original in zip(compact["outcomes"], outcomes, strict=True):
            if not isinstance(original, dict):
                continue
            outcome_refs = _references(original.get("action"))
            for key in ("before", "after"):
                if key in original:
                    outcome[key] = snapshot(original[key], outcome_refs)
    for key in ("candidates", "selected_action_id"):
        compact.pop(key, None)
    compact["history_projection"] = {
        "schema": "robit.omni.ui-history-projection.v1",
        "historical": True, "candidate_offers_retired": True,
        "executable_candidate_offers": False,
    }
    return compact


@dataclass
class _Origin:
    assistant_index: int
    tool_index: int
    assistant_digest: str
    tool_digest: str
    controller: tuple[str, str]
    arguments: dict[str, Any]
    retired: bool = False


class ServerUiHistory:
    """One request's private provenance ledger; never built from client fields.

    `record_server_pair` must be called only at the existing trusted dispatch
    site, immediately after the portal appends the executed tool result.
    Initial client messages are never registered. Prefix rewrites invalidate
    every old registration, even when a lookalike occupies the old index.
    """

    def __init__(self) -> None:
        self._origins: list[_Origin] = []
        self._prefix_size = 0
        self._prefix_digest: str | None = None
        self._disabled = False

    def record_server_pair(self, messages: list[dict[str, Any]], *, assistant_index: int,
                           tool_index: int, operation: str) -> int:
        if self._disabled:
            return 0
        try:
            prefix_changed = self._prefix_digest is not None and (
                len(messages) < self._prefix_size
                or _digest(_plain(messages[:self._prefix_size])) != self._prefix_digest)
            # Validate the full prefix before any model-history mutation.
            _digest(_plain(messages))
        except (ValueError, TypeError, RecursionError):
            prefix_changed = True
        if prefix_changed:
            # Unknown packing/shedding cannot transfer or re-establish origin
            # during this request. A new route-local object is the boundary.
            self._origins.clear()
            self._disabled = True
            return 0
        count = self._record(messages, assistant_index, tool_index, operation)
        # Prefix integrity still covers retired messages. Their index records
        # are no longer needed, so long valid chains cannot fill the ledger.
        self._origins = [origin for origin in self._origins if not origin.retired]
        self._prefix_size = len(messages)
        self._prefix_digest = _digest(_plain(messages))
        return count

    def _record(self, messages: list[dict[str, Any]], ai: int, ti: int, operation: str) -> int:
        if (type(ai) is not int or type(ti) is not int
                or not (0 <= ai < ti < len(messages)) or len(self._origins) >= 256
                or any(origin.tool_index == ti for origin in self._origins)):
            return 0
        assistant, tool = messages[ai], messages[ti]
        if not isinstance(assistant, dict) or not isinstance(tool, dict):
            return 0
        name, call_id, content = tool.get("tool_name"), tool.get("tool_call_id"), tool.get("content")
        if (assistant.get("role") != "assistant" or tool.get("role") != "tool"
                or name not in ("native_ui", "computer_use") or not isinstance(call_id, str)
                or not call_id or not isinstance(content, str)
                or any(tool.get(key) for key in ("images", "audios", "videos"))):
            return 0
        calls = assistant.get("tool_calls")
        matching = [call for call in calls if
            isinstance(call, dict) and call.get("id") == call_id
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == name] if isinstance(calls, list) else []
        if (len(matching) != 1 or not isinstance(operation, str)
                or operation not in {"observe", "verify", "execute", "workflow"}):
            return 0
        try:
            arguments = matching[0]["function"].get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict) or arguments.get("operation", "observe") != operation:
                return 0
            _digest(arguments)
            result = json.loads(content)
            _digest(result)
        except (ValueError, TypeError, RecursionError):
            return 0
        if not isinstance(result, dict) or (identity := _identity(result, operation)) is None:
            return 0
        controller, positive = identity
        retired = 0
        if positive:
            for origin in self._origins:
                if origin.retired or origin.controller != controller:
                    continue
                if origin.assistant_index >= len(messages) or origin.tool_index >= len(messages):
                    continue
                old_assistant, old_tool = messages[origin.assistant_index], messages[origin.tool_index]
                if (_digest(_plain(old_assistant)) != origin.assistant_digest
                        or _digest(_plain(old_tool)) != origin.tool_digest):
                    continue
                old_result = json.loads(old_tool["content"])
                old_tool["content"] = json.dumps(_retired(old_result, controller, origin.arguments), ensure_ascii=False,
                                                 separators=(",", ":"))
                origin.tool_digest = _digest(_plain(old_tool))
                origin.retired = True
                retired += 1
        self._origins.append(_Origin(ai, ti, _digest(_plain(assistant)), _digest(_plain(tool)),
                                     controller, copy.deepcopy(arguments)))
        return retired
