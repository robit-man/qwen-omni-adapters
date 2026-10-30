"""Private host-compiled immediate effects, separate from planner predictions.

The compiler never interprets explanatory prose, shortcut names or UI labels.
It creates no executable offer and grants no input authority. Callers must use
the existing fresh native resolution, authorization and verification path.
Application-specific input ends at one observation boundary; an optional typed
comparison describes only a UI transition, never completion of the user task.
Canonical public workflows remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .computer_use import VERBS, ComputerUseError, parse_intent, parse_step
from .text_entry import complete_text

_SCHEMA = "robit.omni.ui-immediate-effect.v1"
_TRANSITIONS = ("focus_changed_within_scope", "popup_opened", "document_changed")
_BOUNDARY = frozenset({"click", "double_click", "right_click", "invoke", "key_chord", "scroll", "drag_drop"})


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ComputerUseError("Invalid immediate-effect JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _operation(step: Mapping[str, Any]) -> dict:
    if not isinstance(step, Mapping):
        raise ComputerUseError("Expected a typed immediate operation")
    # Reuse canonical verb/argument validation for both kinds of proposal.
    # This placeholder is never an output predicate or execution instruction.
    proposed = copy.deepcopy(dict(step))
    if "expect" not in proposed:
        proposed["expect"] = [{"kind": "focused", "node_id": "$target"}]
    parsed = parse_step(proposed)
    parsed.pop("expect")
    return parsed


def _source(state: Any) -> dict:
    if not isinstance(state, Mapping) or not isinstance(state.get("scope"), Mapping):
        raise ComputerUseError("Expected an observed transition source")
    scope_kind = state["scope"].get("kind")
    if not isinstance(scope_kind, str) or scope_kind not in {"native", "desktop", "browser"}:
        raise ComputerUseError("Invalid observed transition scope")
    source = {"session_id": state.get("session_id"), "scope_id": state["scope"].get("id"),
              "revision": state.get("revision")}
    if any(not isinstance(value, str) or not 1 <= len(value) <= 512 for value in source.values()):
        raise ComputerUseError("Invalid observed transition source")
    return source


def transition_descriptor_schema(before: Mapping[str, Any]) -> dict:
    """Typed nonliteral choices pinned to the exact current observation ref."""
    return {"type": "object", "additionalProperties": False, "required": ["kind", "source_ref"],
        "properties": {"kind": {"enum": list(_TRANSITIONS)},
                       "source_ref": {"const": _source(before)}}}


def immediate_effect_schema() -> dict:
    """Private handoff schema; accepting this shape grants no authority."""
    source = {"type": "object", "additionalProperties": False,
        "required": ["session_id", "scope_id", "revision"], "properties": {
            key: {"type": "string", "minLength": 1, "maxLength": 512}
            for key in ("session_id", "scope_id", "revision")}}
    nullable_source = {"anyOf": [source, {"type": "null"}]}
    transition = {"type": "object", "additionalProperties": False, "required": ["kind", "source_ref"],
        "properties": {"kind": {"enum": list(_TRANSITIONS)}, "source_ref": source}}
    predicates = []
    for kind, fields in {
        "focused": {}, "active_window": {}, "value": {"value": {"type": "string", "maxLength": 4096}},
        "checked": {"checked": {"type": "boolean"}},
        "text_selection": {key: {"type": "integer", "minimum": 0, "maximum": 1_000_000}
                           for key in ("start", "end")},
    }.items():
        predicates.append({"type": "object", "additionalProperties": False,
            "required": ["kind", "node_id", *fields], "properties": {
                "kind": {"const": kind}, "node_id": {"const": "$target"}, **fields}})
    return {"type": "object", "additionalProperties": False,
        "required": ["schema", "operation", "mode", "predicates", "reason", "source_ref", "transition",
                     "non_authorizing", "task_progress"],
        "properties": {"schema": {"const": _SCHEMA}, "operation": {"enum": sorted(VERBS)},
            "mode": {"enum": ["verified_step", "observation_boundary", "requires_replan"]},
            "predicates": {"type": "array", "maxItems": 2, "items": {"oneOf": predicates}},
            "reason": {"type": "string", "minLength": 1, "maxLength": 100},
            "source_ref": nullable_source, "transition": {"anyOf": [transition, {"type": "null"}]},
            "non_authorizing": {"const": True}, "task_progress": {"const": False}},
        "allOf": [{"if": {"properties": {"mode": {"const": "verified_step"}}},
                   "then": {"properties": {"predicates": {"minItems": 1}}},
                   "else": {"properties": {"predicates": {"maxItems": 0}}}}]}


def _typed_text_result(operation: dict, before: Any) -> tuple[str | None, str | None]:
    """Derive the same exact value, preserving why insertion remains unknown."""
    if not isinstance(before, Mapping) or not isinstance(before.get("nodes"), list):
        return None, None
    matches = [node for node in before["nodes"] if isinstance(node, dict) and all(
        node.get("id" if key == "node_id" else key) == value for key, value in operation["target"].items())]
    if len(matches) != 1:
        return None, None
    node = matches[0]
    coverage = node.get("text_observation")
    if (node.get("visible") is not True or node.get("enabled") is not True
            or node.get("sensitive") is not False or node.get("focused") is not True
            or node.get("scope_id") != before.get("scope", {}).get("id")
            or not isinstance(node.get("operations"), list) or "type_text" not in node["operations"]
            or any(not isinstance(node.get(key), str) or not node[key] for key in ("id", "generation"))
            or not isinstance(coverage, dict)
            or any(type(coverage.get(key)) is not int for key in ("character_count", "selection_count"))
            or not isinstance(coverage.get("captured_range"), dict)
            or any(type(coverage["captured_range"].get(key)) is not int for key in ("start", "end"))
            or not complete_text(node)):
        return None, None
    value = node["value"]
    count = node["text_observation"]["selection_count"]
    if count == 0 and not value and "text_selection" not in node:
        return operation["args"]["text"], None
    insertion = node.get("text_insertion")
    if count == 0 and "text_selection" not in node and insertion is not None:
        # Executor-observed Unicode character offset of an unselected caret.
        if (not isinstance(insertion, dict) or set(insertion) != {"caret"} or type(insertion["caret"]) is not int
                or not 0 <= insertion["caret"] <= len(value)):
            return None, "typing_insertion_range_not_observed"
        caret = insertion["caret"]
        return value[:caret] + operation["args"]["text"] + value[caret:], None
    selection = node.get("text_selection")
    if (count != 1 or not isinstance(selection, dict) or set(selection) != {"start", "end"}
            or any(type(selection.get(key)) is not int for key in ("start", "end"))
            or not 0 <= selection["start"] < selection["end"] <= len(value)):
        return None, "typing_insertion_range_not_observed"
    return value[:selection["start"]] + operation["args"]["text"] + value[selection["end"]:], None


@dataclass(frozen=True)
class EffectContract:
    """Immutable compilation product; JSON exports never expose mutable internals."""

    operation: str
    mode: str
    reason: str
    _operation_json: str
    _predicates_json: str
    _source_json: str
    _transition_json: str
    _before_sha256: str | None

    def as_dict(self) -> dict:
        return {"schema": _SCHEMA, "operation": self.operation, "mode": self.mode,
            "predicates": json.loads(self._predicates_json), "reason": self.reason,
            "source_ref": json.loads(self._source_json), "transition": json.loads(self._transition_json),
            "non_authorizing": True, "task_progress": False}

    def bind(self, step: Mapping[str, Any]) -> dict:
        """Bind only the unchanged proposal, for subsequent native admission."""
        operation = _operation(step)
        if _json(operation) != self._operation_json:
            raise ComputerUseError("Immediate operation changed after effect compilation")
        if self.mode == "verified_step":
            return parse_step({**operation, "expect": json.loads(self._predicates_json)})
        if self.mode == "observation_boundary":
            return parse_intent(operation)
        raise ComputerUseError("Immediate effect requires replanning before input")

    def compare(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
        """Compare a selected typed UI transition; this is not an effect receipt."""
        transition = json.loads(self._transition_json)
        result = {"schema": "robit.omni.ui-transition-comparison.v1", "status": "unknown",
            "transition": transition, "reason": "transition_descriptor_unavailable",
            "input_authority": False, "task_progress": False, "resource_acquired": False}
        if transition is None:
            return result
        try:
            if _digest(before) != self._before_sha256 or _source(before) != transition["source_ref"]:
                return {**result, "reason": "transition_source_changed"}
            _source(after)
            if (before["session_id"] != after["session_id"] or any(
                    before["scope"].get(key) != after["scope"].get(key)
                    for key in ("kind", "id", "window_id", "process_id"))):
                return {**result, "reason": "transition_scope_changed"}
            result.update(before_sha256=self._before_sha256, after_sha256=_digest(after))
            observed = _compare_transition(transition["kind"], before, after)
        except (ComputerUseError, TypeError, ValueError, KeyError):
            return {**result, "reason": "transition_observation_invalid"}
        return {**result, "status": "unknown" if observed is None else "observed" if observed else "absent",
                "reason": "transition_evidence_incomplete" if observed is None else "typed_observation_comparison"}


def _selection_value(operation: dict, before: Mapping[str, Any] | None) -> str | None:
    """Selection may verify unchanged text only from complete native pre-text."""
    if not isinstance(before, Mapping) or not isinstance(before.get("nodes"), list):
        return None
    matches = [node for node in before["nodes"] if isinstance(node, dict) and all(
        node.get("id" if key == "node_id" else key) == value
        for key, value in operation["target"].items())]
    if len(matches) != 1:
        return None
    node = matches[0]
    coverage = node.get("text_observation")
    if (node.get("visible") is not True or node.get("enabled") is not True
            or node.get("sensitive") is not False
            or node.get("scope_id") != before.get("scope", {}).get("id")
            or any(not isinstance(node.get(key), str) or not node[key] for key in ("id", "generation"))
            or not isinstance(node.get("operations"), list) or "select_text" not in node["operations"]
            or not isinstance(coverage, dict)
            or any(type(coverage.get(key)) is not int for key in ("character_count", "selection_count"))
            or not isinstance(coverage.get("captured_range"), dict)
            or any(type(coverage["captured_range"].get(key)) is not int for key in ("start", "end"))
            or not complete_text(node) or len(node["value"]) > 4096
            or operation["args"]["end"] > len(node["value"])):
        return None
    return node["value"]


def compile_immediate_transition(step: Mapping[str, Any], before: Mapping[str, Any] | None = None, *,
                                 transition: Mapping[str, Any] | None = None) -> EffectContract:
    """Compile only typed operation semantics; caller prose is never an oracle.

    ``before`` must be the caller's fresh executor observation, not model output.
    Derived typing predicates must be revalidated at input by the existing native
    dispatcher. Missing caret/selection knowledge yields no inferred insertion.
    """
    operation = _operation(step)
    verb, args = operation["verb"], operation["args"]
    source = _source(before) if before is not None else None
    if transition is not None and (not isinstance(transition, Mapping)
            or set(transition) != {"kind", "source_ref"} or transition.get("kind") not in _TRANSITIONS
            or source is None or transition.get("source_ref") != source):
        raise ComputerUseError("Expected a source-bound typed transition descriptor")
    predicates, mode, reason = [], "verified_step", "host_operation_effect"
    if verb in {"focus", "switch_window"}:
        predicates = [{"kind": "focused" if verb == "focus" else "active_window", "node_id": "$target"}]
    elif verb in {"set_value", "set_checked"}:
        kind = {"set_value": "value", "set_checked": "checked"}[verb]
        predicates = [{"kind": kind, "node_id": "$target", **args}]
    elif verb == "select_text":
        value = _selection_value(operation, before)
        predicates = [{"kind": "text_selection", "node_id": "$target", **args}]
        if value is None:
            # The existing native range contract remains usable on large or
            # incompletely captured documents. It cannot establish that all
            # document text stayed unchanged; that needs separate evidence.
            reason = "selection_range_only_text_preservation_unknown"
        else:
            predicates.append({"kind": "value", "node_id": "$target", "value": value})
    elif verb == "type_text":
        value, typing_reason = _typed_text_result(operation, before)
        if value is None or len(value) > 4096:
            mode, reason = "requires_replan", typing_reason or "typing_complete_value_not_observed"
        else:
            predicates = [{"kind": "value", "node_id": "$target", "value": value}]
    elif verb in _BOUNDARY:
        mode, reason = "observation_boundary", "operation_effect_requires_observation"
    else:
        mode, reason = "requires_replan", "operation_effect_not_supported"
    return EffectContract(verb, mode, reason, _json(operation), _json(predicates), _json(source),
                          _json(dict(transition) if transition is not None else None),
                          _digest(before) if before is not None else None)


def _nodes(state: Mapping[str, Any]) -> list[dict] | None:
    nodes = state.get("nodes")
    if (not isinstance(nodes, list) or len(nodes) > 600 or any(not isinstance(node, dict) for node in nodes)
            or any(any(not isinstance(node.get(key), str) or not node[key]
                       for key in ("id", "generation", "scope_id", "role"))
                   or any(type(node.get(key)) is not bool for key in ("visible", "sensitive"))
                   or "focused" in node and type(node["focused"]) is not bool for node in nodes)
            or len({node["id"] for node in nodes}) != len(nodes)):
        return None
    # include_windows can legitimately carry foreign window roots. They remain
    # outside this comparison's exact active scope and grant no focus evidence.
    return [node for node in nodes if node["scope_id"] == state["scope"]["id"]]


def _document(state: Mapping[str, Any]) -> tuple | None:
    document, nodes = state.get("document_evidence"), _nodes(state)
    if not isinstance(document, dict) or nodes is None:
        return None
    match = [node for node in nodes if node["id"] == document.get("node_id")]
    digest = document.get("provider_identity_sha256")
    if (document.get("status") != "observed" or document.get("provider") != "atspi2"
            or document.get("source") != "atspi.Document.DocURL" or len(match) != 1
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(document.get("url"), str) or not document["url"]
            or document["url"] != state["scope"].get("url")
            or document.get("scope_id") != state["scope"]["id"]
            or not isinstance(document.get("window_id"), str) or not document["window_id"]
            or document.get("window_id") != state["scope"].get("window_id")):
        return None
    node = match[0]
    if (node.get("role") != "document_web" or node.get("visible") is not True
            or node.get("sensitive") is not False or document.get("generation") != node["generation"]):
        return None
    return tuple(document[key] for key in ("url", "node_id", "generation", "provider_identity_sha256"))


def _compare_transition(kind: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> bool | None:
    if kind == "document_changed":
        old_document, new_document = _document(before), _document(after)
        return None if old_document is None or new_document is None else old_document != new_document
    old, new = _nodes(before), _nodes(after)
    if old is None or new is None or before.get("truncated") is not False or after.get("truncated") is not False:
        return None
    if kind == "focus_changed_within_scope":
        focused = [[(node["id"], node["generation"]) for node in nodes
                    if node.get("focused") is True and node.get("visible") is True and node.get("sensitive") is False]
                   for nodes in (old, new)]
        return None if any(len(items) != 1 for items in focused) else focused[0] != focused[1]
    if kind == "popup_opened":
        old_visible = {(node["id"], node["generation"]) for node in old
                       if node.get("role") in {"menu", "popup_menu"} and node.get("visible") is True}
        new_visible = {(node["id"], node["generation"]) for node in new
                       if node.get("role") in {"menu", "popup_menu"} and node.get("visible") is True
                       and node.get("sensitive") is False}
        added = new_visible - old_visible
        if any(identity in {item[0] for item in old_visible} for identity, _generation in added):
            return None  # A recycled provider identity does not prove a popup opened.
        return bool(added)
    return None
