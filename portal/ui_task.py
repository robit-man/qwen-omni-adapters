"""Server-owned native goal → structured phase → guarded controller loop.

Callbacks own authenticated session scope and transport deadlines. ``observe``
returns a bounded production observation receipt, including mapped windows;
that whole receipt is current_state. No model tool-discovery loop runs here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from portal.tools import tool_schemas
from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    parse_intent,
    parse_predicates,
    parse_step,
)


def phase_response_schema(allowed_tools: Iterable[str]) -> dict[str, Any]:
    """Closed planner output; canonical nested action contracts remain intact."""
    allowed = frozenset(allowed_tools)
    schemas = {item["function"]["name"]: item["function"]["parameters"]
               for item in tool_schemas(["native_ui", "memory_write", "memory_search"])}
    ui = schemas["native_ui"]
    remaining = {"type": "array", "maxItems": 25,
                 "items": {"type": "string", "minLength": 1, "maxLength": 240}}

    def branch(kind, properties, required):
        return {"type": "object", "additionalProperties": False,
                "properties": {"kind": {"const": kind}, "remaining_criteria": remaining, **properties},
                "required": ["kind", *required]}

    alternatives = [branch("ui", {name: ui["properties"][name] for name in ("steps", "final_predicates")},
                           ["steps", "final_predicates"])]
    for name in ("memory_write", "memory_search"):
        if name in allowed:
            properties = copy.deepcopy(schemas[name]["properties"])
            for key, limit in {"topic": 64, "key": 128, "value": 4096, "query": 500}.items():
                if key in properties:
                    properties[key].update(minLength=1, maxLength=limit)
            alternatives.append(branch(name, properties, schemas[name]["required"]))
    alternatives.extend([
        branch("visual_inspection", {}, []),
        branch("complete", {"message": {"type": "string", "minLength": 1, "maxLength": 8000}}, ["message"]),
        # The pinned llama.cpp grammar parser rejects an exactly 2,000-fold
        # repetition. Keep this private planner field below that boundary.
        branch("blocked", {"reason": {"type": "string", "minLength": 1, "maxLength": 1999}}, ["reason"])])
    return copy.deepcopy({"$defs": ui["$defs"], "oneOf": alternatives})


# Keep the private schema bounded without shortening the canonical 25-step phase.
_MAX_GROUNDED_SCHEMA_BYTES = 56 * 1024


def _grounded_phase_schema(allowed_tools: Iterable[str], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Ground only the first position in positive, model-visible capabilities.

    Semantic selectors may remain ambiguous. Named visual discovery is still
    guarded by the native adapter; this projection creates no input authority.
    Unknown/oversized observations retain the unchanged non-UI phase choices.
    """
    schema = phase_response_schema(allowed_tools)
    fallback = {"$defs": copy.deepcopy(schema["$defs"]),
                "oneOf": copy.deepcopy([branch for branch in schema["oneOf"]
                           if branch["properties"]["kind"]["const"] != "ui"])}

    def non_ui():
        return fallback

    state = observation.get("state") if isinstance(observation, Mapping) else None
    nodes = state.get("nodes") if isinstance(state, dict) else None
    if not isinstance(nodes, list) or len(nodes) > 80:
        return non_ui()
    canonical = schema["$defs"]["step"]["oneOf"]
    # A switch has one causal effect. Resolve $target only after the controller
    # has selected the current window; independently authored IDs can name the
    # previous window even when the requested switch itself succeeds.
    switch_effect = {"type": "array", "minItems": 1, "maxItems": 1, "items": {
        "type": "object", "additionalProperties": False, "required": ["kind", "node_id"],
        "properties": {"kind": {"const": "active_window"}, "node_id": {"const": "$target"}}}}
    switch = copy.deepcopy(canonical[0])
    switch["properties"]["verb"] = {"const": "switch_window"}
    switch["properties"]["expect"] = switch_effect
    other_zero = copy.deepcopy(canonical[0])
    other_zero["properties"]["verb"]["enum"].remove("switch_window")
    schema["$defs"]["causal_step"] = {"oneOf": [other_zero,
        {"$ref": "#/$defs/step/oneOf/1"}, switch]}
    arguments = {verb: copy.deepcopy(canonical[0])
                 for verb in canonical[0]["properties"]["verb"]["enum"]}
    for condition in canonical[1]["allOf"]:
        verb = condition["if"]["properties"]["verb"]["const"]
        required = condition["then"]["properties"]["args"]["required"]
        index = next(index for index, item in enumerate(schema["$defs"]["step_args"]["oneOf"])
                     if item["required"] == required)
        branch = copy.deepcopy(canonical[1])
        branch.pop("allOf")
        branch["properties"]["args"] = {"$ref": f"#/$defs/step_args/oneOf/{index}"}
        arguments[verb] = branch
    eligible: dict[str, list[dict]] = {verb: [] for verb in arguments}
    identities = set()
    for node in nodes:
        if (not isinstance(node, dict) or any(not isinstance(node.get(key), str)
                or len(node[key]) > 512 or (not node[key] and key != "name")
                for key in ("id", "generation", "scope_id", "role", "name"))
                or node["id"] in identities
                or any(type(node.get(key)) is not bool for key in ("visible", "enabled", "sensitive"))
                or not isinstance(node.get("operations"), list)
                or len(node["operations"]) > len(arguments)
                or any(not isinstance(verb, str) or verb not in arguments for verb in node["operations"])):
            return non_ui()
        identities.add(node["id"])
        if node["visible"] is True and node["enabled"] is True and node["sensitive"] is False:
            for verb in set(node["operations"]):
                if verb == "type_text" and node.get("focused") is not True:
                    continue
                eligible[verb].append(node)

    def target(properties, required):
        return {"type": "object", "additionalProperties": False,
                "properties": properties, "required": required}

    first, boundaries = [], []
    for verb, controls in eligible.items():
        selectors = []
        if controls:
            selectors.append(target({"node_id": {"enum": sorted({node["id"] for node in controls})}}, ["node_id"]))
            groups: dict[tuple[str, str], set[str]] = {}
            scopes: dict[str, set[str]] = {}
            for node in controls:
                groups.setdefault((node["role"], node["scope_id"]), set()).add(node["name"])
                scopes.setdefault(node["scope_id"], set()).add(node["name"])
            for (role, scope), names in sorted(groups.items()):
                selectors.append(target({"role": {"const": role}, "scope_id": {"const": scope},
                                         "name": {"enum": sorted(names)}}, ["role"]))
            for scope, names in sorted(scopes.items()):
                selectors.append(target({"scope_id": {"const": scope}, "name": {"enum": sorted(names)}}, ["name"]))
        if verb in {"invoke", "click", "double_click", "right_click"}:
            # No guessed ID: existing exact-match/protection/frame checks decide
            # whether a named semantic target can use visual discovery.
            discovery = copy.deepcopy(schema["$defs"]["target"]["oneOf"][1]["properties"])
            discovery["name"]["minLength"] = 1
            selectors.append(target(discovery, ["name"]))
        if selectors:
            branch = copy.deepcopy(arguments[verb])
            branch["properties"]["verb"] = {"const": verb}
            branch["properties"]["target"] = {"anyOf": selectors}
            if verb == "switch_window":
                branch["properties"]["expect"] = copy.deepcopy(switch_effect)
            if verb == "focus":
                branch["properties"]["expect"] = copy.deepcopy(switch_effect)
                branch["properties"]["expect"]["items"]["properties"]["kind"] = {"const": "focused"}
            if verb in {"invoke", "click", "double_click", "right_click"}:
                branch["properties"].pop("expect")
                branch["required"].remove("expect")
                boundaries.append(branch)
            else:
                first.append(branch)
    if not first and not boundaries:
        return non_ui()
    if first:
        schema["$defs"]["grounded_first_step"] = {"oneOf": first}
        schema["oneOf"][0]["properties"]["steps"] = {"anyOf": [
            {"type": "array", "prefixItems": [{"$ref": "#/$defs/grounded_first_step"}]
                 + [{"$ref": "#/$defs/causal_step"}] * (length - 1), "minItems": length, "maxItems": length}
            for length in range(1, 26)]}
    else:
        schema["oneOf"].pop(0)
    if boundaries:
        schema["oneOf"].append({"type": "object", "additionalProperties": False,
            "required": ["kind", "step"], "properties": {
                "kind": {"const": "ui_boundary"}, "step": {"oneOf": boundaries},
                "remaining_criteria": copy.deepcopy(fallback["oneOf"][0]["properties"]["remaining_criteria"])},
            "description": "One guarded input followed by semantic observation and replanning. "
                           "No effect predicate, verified progress, or following step."})
    if len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > _MAX_GROUNDED_SCHEMA_BYTES:
        return non_ui()
    return schema


def _accepted_visual_reading(result: Mapping[str, Any]) -> bool:
    reading, receipt = result.get("reading"), result.get("visual_inspection_receipt")
    if (result.get("status") not in ("observed", "captured_snapshot")
            or result.get("dispatch_status") != "not_dispatched" or result.get("evidence_authority") != "inspection"
            or result.get("task_progress") is not False
            or not isinstance(reading, str) or not reading.strip() or len(reading) > 6000
            or not isinstance(receipt, dict) or receipt.get("schema") != "robit.omni.native-visual-inspection.v1"
            or receipt.get("authority") != "untrusted_tool_observation" or receipt.get("snapshot_only") is not True
            or receipt.get("temporal_scope") != "captured_snapshot"
            or receipt.get("current_ui_verified") is not False or receipt.get("input_authority") is not False):
        return False
    before, after = receipt.get("frame_sha256"), receipt.get("checked_frame_sha256")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in (before, after)):
        return False
    same = before == after
    if receipt.get("post_capture_pixels_match") is not same or same != (result["status"] == "observed"):
        return False
    if not same:
        document = receipt.get("document_evidence")
        if (not isinstance(receipt.get("document_region"), dict) or not isinstance(document, dict)
                or document.get("status") != "observed" or document.get("source") != "atspi.Document.DocURL"
                or document.get("url") != receipt.get("source_url")):
            return False
    return True


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep receipts and readback; never multiply historical UI maps in planning."""
    fields = ("status", "error", "reason", "message", "phase_handoff", "terminal_receipt",
              "verification_receipt", "ui_state_receipt", "effect_receipt", "action_receipt",
              "policy_receipts", "point_receipts", "selection_receipts", "evidence_authority", "resource", "stored",
              "updated", "unchanged", "topic", "key", "characters", "scope", "query", "results",
              "reading", "visual_inspection_receipt", "ui_boundary_receipt", "observed_changes")
    compact = {key: copy.deepcopy(result[key]) for key in fields if key in result}
    if not _accepted_visual_reading(result):
        compact.pop("reading", None)
    outcomes = []
    for item in result.get("outcomes", [])[:25]:
        action = item.get("action", {})
        outcome = {key: copy.deepcopy(item[key]) for key in
                   ("status", "reason", "error", "receipt", "effect_receipt", "verification", "policy_receipt", "point_receipt")
                   if key in item}
        outcome["action"] = {key: copy.deepcopy(action[key]) for key in
                             ("id", "node_id", "verb", "args", "expect", "idempotency_key") if key in action}
        after = item.get("after", {})
        wanted = {action.get("node_id")} | {predicate.get("node_id") for predicate in action.get("expect", [])}
        outcome["observed_effects"] = [{key: copy.deepcopy(node[key]) for key in
            ("id", "generation", "scope_id", "role", "name", "value", "checked", "selected", "focused", "text_selection", "text_observation")
            if key in node} for node in after.get("nodes", []) if node.get("id") in wanted and not node.get("sensitive")][:21]
        outcomes.append(outcome)
    if outcomes:
        compact["outcomes"] = outcomes
    return compact


def _text_observation_free_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy actual snapshot nodes; similarly named action/authority fields stay."""
    copied = copy.deepcopy(state)
    nodes = copied.get("nodes")
    for node in nodes if isinstance(nodes, list) else []:
        if isinstance(node, dict):
            node.pop("text_observation", None)
    return copied


def _phase_observation_fingerprint(observed: Mapping[str, Any]) -> dict[str, Any] | None:
    copied = copy.deepcopy(dict(observed))
    state = observed["state"]
    projection = state.get("model_projection", {})
    anchors = ("raw_source_sha256", "text_observation_semantic_sha256")
    anchored = isinstance(projection, dict) and any(key in projection for key in anchors)
    nodes = state.get("nodes")
    metadata = isinstance(nodes, list) and any(isinstance(node, dict) and "text_observation" in node for node in nodes)
    if not anchored and not metadata:
        return copied
    normalized = _text_observation_free_state(state)
    if anchored:
        if (any(not isinstance(projection.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", projection[key]) is None
                for key in anchors) or observed.get("content_sha256") != projection[anchors[0]]):
            return None
        copied["content_sha256"] = projection[anchors[1]]
        for key in anchors:
            normalized["model_projection"].pop(key)
    else:
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if observed.get("content_sha256") != digest(state):
            return None
        copied["content_sha256"] = digest(normalized)
    copied["state"] = normalized
    return copied


def _zero_effect_ui_phase_key(result: Mapping[str, Any], observed: Mapping[str, Any]) -> str | None:
    """Identify an already verified UI phase, never a planner progress claim.

    Fresh terminal and per-input observations remain part of the exact binding.
    Opaque action/step IDs do not make unchanged effective inputs new work.
    Incomplete or uncertain effect evidence cannot establish this equivalence.
    """
    outcomes, terminal = result.get("outcomes"), result.get("terminal")
    if (result.get("status") != "complete" or result.get("error")
            or result.get("terminal_receipt", {}).get("status") != "verified"
            or not isinstance(outcomes, list) or not outcomes or not isinstance(terminal, dict)
            or not isinstance(observed.get("state"), dict)):
        return None
    steps = []
    fields = ("session_id", "scope_id", "node_id", "generation", "verb", "args", "expect")
    for outcome in outcomes:
        action, effect = outcome.get("action", {}), outcome.get("effect_receipt", {})
        if (outcome.get("status") != "verified" or action.get("kind") != "action"
                or outcome.get("receipt", {}).get("status") not in {"dispatched", "already_satisfied"}
                or effect.get("effect_state") not in {"absent", "unchanged", "replayed"}
                or any(key not in action for key in fields)):
            return None
        before, after = outcome.get("before"), outcome.get("after")
        if any(not isinstance(state, dict) or not all(key in state for key in
                   ("session_id", "scope", "revision", "nodes")) for state in (before, after, terminal)):
            return None
        binding = {key: action[key] for key in fields}
        # Effect predicates are a conjunction; ordering conveys no new intent.
        binding["expect"] = sorted(binding["expect"], key=lambda item: json.dumps(item, sort_keys=True))
        steps.append({"action": binding, "before": _text_observation_free_state(before),
                      "after": _text_observation_free_state(after)})
    fingerprint_observed = _phase_observation_fingerprint(observed)
    if fingerprint_observed is None:
        return None
    encoded = json.dumps({"observed": fingerprint_observed, "steps": steps,
                          "terminal": _text_observation_free_state(terminal)}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class NativePhaseCompilationError(Exception):
    """Trusted compiler JSON-decode failure; never carries the model's reply."""

    code = "invalid_native_phase_json"


def _new_verified_ui_prefix(result: Mapping[str, Any], seen: set[str]) -> bool:
    """Allow recovery after a new bound input/effect transition, not goal credit."""
    novel = False
    for outcome in result.get("outcomes", []):
        if not isinstance(outcome, dict) or outcome.get("status") != "verified":
            break
        action, receipt, effect = (outcome.get(key, {}) for key in ("action", "receipt", "effect_receipt"))
        before, after = outcome.get("before", {}), outcome.get("after", {})
        if any(not isinstance(value, dict) for value in (action, receipt, effect, before, after)):
            break
        if receipt.get("status") != "dispatched" or effect.get("effect_state") != "applied":
            continue
        before_scope, after_scope = before.get("scope", {}), after.get("scope", {})
        before_semantic = before.get("base_semantic_revision", before.get("revision"))
        after_semantic = after.get("base_semantic_revision", after.get("revision"))
        fields = ("session_id", "scope_id", "node_id", "generation", "verb", "revision", "id", "idempotency_key")
        if (action.get("kind") != "action" or any(not isinstance(action.get(key), str) or not action[key] for key in fields)
                or not isinstance(action.get("args"), dict) or not isinstance(action.get("expect"), list)
                or not isinstance(before_scope, dict) or not isinstance(after_scope, dict)
                or not after_scope.get("id") or not isinstance(before.get("nodes"), list)
                or before.get("session_id") != action["session_id"] or after.get("session_id") != action["session_id"]
                or before_scope.get("id") != action["scope_id"]
                or action["revision"] != before.get("revision")
                or effect.get("schema") != "robit.omni.ui-effect-receipt.v1"
                or effect.get("action_id") != action["id"] or effect.get("idempotency_key") != action["idempotency_key"]
                or effect.get("before_revision") != before.get("revision")
                or effect.get("after_revision") != after.get("revision")
                or effect.get("before_semantic_revision") != before_semantic
                or effect.get("after_semantic_revision") != after_semantic
                or not before_semantic or not after_semantic or before_semantic == after_semantic):
            break
        targets = [node for node in before["nodes"] if isinstance(node, dict) and node.get("id") == action["node_id"]]
        if (len(targets) != 1 or targets[0].get("generation") != action["generation"]
                or targets[0].get("visible") is not True or targets[0].get("enabled") is not True
                or targets[0].get("sensitive") is not False or action["verb"] not in targets[0].get("operations", [])):
            break
        try:
            predicates = json.dumps(action["expect"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if effect.get("predicates_sha256") != hashlib.sha256(predicates).hexdigest():
                break
            # Opaque input IDs and changed verification wording do not make an
            # already observed semantic transition new. Authority bindings stay.
            transition = {"action": {key: action[key] for key in
                ("session_id", "scope_id", "node_id", "generation", "verb", "args")},
                "before": {"scope": before_scope, "semantic_revision": before_semantic},
                "after": {"scope": after_scope, "semantic_revision": after_semantic}}
            signature = hashlib.sha256(json.dumps(transition, sort_keys=True, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError):
            break
        novel |= signature not in seen
        seen.add(signature)
    return novel


def run_native_goal(goal: str, *, observe: Callable[[], dict], compile_phase: Callable[[dict], dict],
                    execute: Callable[[str, dict], dict], cancel: Callable[[], None],
                    allowed_tools: Iterable[str], max_phases: int = 40, max_seconds: float = 1200,
                    model: str = "", clock: Callable[[], float] = time.monotonic,
                    cancelled: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Compile each phase once, execute internally, then return to the planner.

    The caller enforces callback I/O deadlines using remaining_seconds. Checks
    around every callback prevent a late plan from dispatching input. Complete
    is explicitly a planner claim; only independent task oracles score success.
    """
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 20000:
        raise ValueError("Expected a bounded nonempty native goal")
    if type(max_phases) is not int or not 1 <= max_phases <= 40:
        raise ValueError("Expected 1–40 native phases")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Expected a positive finite native task deadline")
    allowed = frozenset(allowed_tools)
    if "native_ui" not in allowed:
        raise ValueError("Native task requires explicit native_ui permission")
    started, traces, ledger, last_result = clock(), [], [], None
    failed, normal, cleaned, phases, claims = 0, False, False, 0, []
    zero_effect_ui_phases: set[str] = set()
    applied_ui_transitions: set[str] = set()
    boundary_observations: set[str] = set()

    def boundary():
        if cancelled is not None and cancelled():
            return "cancelled"
        return "deadline_exceeded" if clock() - started >= max_seconds else None

    def finish(status, message):
        return {"type": "final", "response": {"model": model,
            "message": {"role": "assistant", "content": message},
            "portal": {"safe_tools_executed": copy.deepcopy(traces), "native_task": {
                "status": status, "task_verified": False, "phases": phases,
                "elapsed_s": max(0, clock() - started), "remaining_criteria_authority": "planner_claim",
                "remaining_criteria": copy.deepcopy(claims),
                "ledger": copy.deepcopy(ledger)}}}}

    try:
        for phases in range(1, max_phases + 1):
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            yield {"type": "progress", "stage": "native_task_observe", "phase": phases}
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            current = observe()
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            schema = _grounded_phase_schema(allowed, current)
            validator = Draft202012Validator(schema)
            packet = {"goal": goal, "current_state": copy.deepcopy(current),
                      "ledger": copy.deepcopy(ledger[-12:]), "last_result": copy.deepcopy(last_result),
                      "response_schema": copy.deepcopy(schema),
                      "remaining_seconds": max_seconds - (clock() - started)}
            yield {"type": "progress", "stage": "native_task_plan", "phase": phases}
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            packet["remaining_seconds"] = max_seconds - (clock() - started)
            try:
                plan = compile_phase(packet)
            except NativePhaseCompilationError:
                if reason := boundary():
                    yield finish(reason, "Native task stopped: " + reason)
                    return
                last_result = {"error": NativePhaseCompilationError.code,
                    "dispatch_status": "not_dispatched", "input_attempted": False,
                    "new_effect_count": 0,
                    "reason": "The compiler reply was not complete valid JSON. Return a smaller complete phase "
                              "matching the unchanged schema, without repairing or replaying the invalid reply."}
                ledger.append({"phase": phases, "kind": "invalid", **last_result})
                ledger = ledger[-12:]
                failed += 1
                if failed >= 3:
                    yield finish("blocked", "Three consecutive native phases were invalid or unverified.")
                    return
                continue
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            try:
                if len(json.dumps(plan, allow_nan=False).encode()) > 131072:
                    raise ComputerUseError("Compiled phase exceeds bounded size")
                validator.validate(plan)
                kind = plan["kind"]
                if kind == "ui":
                    for step in plan["steps"]:
                        parse_step(step)
                    parse_predicates(plan["final_predicates"], terminal=True)
                if kind == "ui_boundary":
                    parse_intent(plan["step"])
            except (ValidationError, ComputerUseError, ValueError, TypeError) as exc:
                last_result = {"error": "invalid_native_phase", "dispatch_status": "not_dispatched",
                               "reason": str(exc.message if isinstance(exc, ValidationError) else exc)[:600]}
                ledger.append({"phase": phases, "kind": "invalid", **last_result})
                failed += 1
                if failed >= 3:
                    yield finish("blocked", "Three consecutive native phases were invalid or unverified.")
                    return
                continue
            claims = copy.deepcopy(plan.get("remaining_criteria", []))
            if kind in {"complete", "blocked"}:
                normal = kind == "complete"
                yield finish("planner_claim" if normal else "blocked", plan.get("message", plan.get("reason", "")))
                return
            name = "native_ui" if kind in {"ui", "ui_boundary", "visual_inspection"} else kind
            arguments = ({"operation": "workflow", "steps": plan["steps"], "final_predicates": plan["final_predicates"]}
                         if kind == "ui" else {"operation": "observation_boundary", "intent": plan["step"]}
                         if kind == "ui_boundary" else {"operation": "inspect_visual"} if kind == "visual_inspection" else {key: value for key, value in plan.items()
                                                if key not in {"kind", "remaining_criteria"}})
            if name not in allowed:
                raise ValueError("Compiled phase exceeds immutable tool scope")
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            tool = {"id": uuid.uuid4().hex, "name": name, "arguments": copy.deepcopy(arguments),
                    "ok": False, "status": "running", "result": ""}
            yield {"type": "tool", "phase": "start", "round": phases, "tools": [copy.deepcopy(tool)]}
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            result = execute(name, copy.deepcopy(arguments))
            if not isinstance(result, dict):
                raise ValueError("Native executor returned no typed receipt")
            if kind == "ui":
                if any(item.get("effect_receipt", {}).get("effect_state") == "applied"
                       for item in result.get("outcomes", [])):
                    zero_effect_ui_phases.clear()
                if signature := _zero_effect_ui_phase_key(result, current):
                    if signature in zero_effect_ui_phases:
                        result = {**result, "error": "ui_phase_repeated_without_new_effect",
                            "reason": "This verified UI phase repeats the same effective actions and observed state "
                                      "without a new applied effect. Replan from the retained evidence."}
                    else:
                        zero_effect_ui_phases.add(signature)
            content = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            replayed_only = (kind == "ui" and bool(result.get("outcomes"))
                             and all(item.get("effect_receipt", {}).get("effect_state") == "replayed"
                                     for item in result["outcomes"]))
            ok = not result.get("error") and not replayed_only and (kind != "ui" or result.get("status") == "complete")
            if kind == "visual_inspection":
                ok = ok and _accepted_visual_reading(result)
            novel_boundary = False
            if kind == "ui_boundary":
                receipt = result.get("ui_boundary_receipt", {})
                ok = (result.get("status") == "observed_unverified" and receipt.get("observed") is True
                      and receipt.get("verified") is False and receipt.get("task_progress") is False
                      and type(receipt.get("new_effect_count")) is int and receipt["new_effect_count"] == 0
                      and result.get("effect_receipt", {}).get("effect_state") == "unknown")
                key = receipt.get("novelty_key")
                if (ok and receipt.get("input_attempted") is True and receipt.get("cached") is False
                        and isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key)):
                    novel_boundary = key not in boundary_observations
                    boundary_observations.add(key)
            recoverable_prefix = kind == "ui" and _new_verified_ui_prefix(result, applied_ui_transitions)
            tool.update(ok=ok, status="complete", result=content)
            traces.append(copy.deepcopy(tool))
            last_result = _compact_result(result)
            ledger.append({"phase": phases, "kind": kind, "ok": ok,
                           "remaining_criteria": copy.deepcopy(plan.get("remaining_criteria", [])),
                           "criteria_authority": "planner_claim", "result": {
                               key: copy.deepcopy(last_result[key]) for key in
                               ("status", "error", "reason", "phase_handoff", "ui_state_receipt",
                                "verification_receipt", "effect_receipt", "stored", "updated", "unchanged", "visual_inspection_receipt",
                                "ui_boundary_receipt", "observed_changes")
                               if key in last_result}})
            ledger = ledger[-12:]
            if kind == "ui_boundary":
                # Observation protocol success is not verified task progress.
                failed += not novel_boundary
            else:
                failed = 0 if ok or recoverable_prefix else failed + 1
            yield {"type": "tool", "phase": "complete", "round": phases, "tools": [copy.deepcopy(tool)]}
            if failed >= 3:
                yield finish("blocked", "Three consecutive native phases were invalid or unverified.")
                return
        yield finish("phase_budget_exceeded", "Native task reached its bounded phase budget.")
    except Exception as exc:
        reason = boundary()
        diagnostic = type(exc).__name__
        if reason is None:
            code = getattr(exc, "code", None)
            if (isinstance(code, str) and len(code) <= 128
                    and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code)):
                diagnostic += f" ({code})"
        yield finish(reason or "blocked", "Native task stopped: " + reason if reason else
                     "Native task callback failed: " + diagnostic)
    finally:
        if not normal and not cleaned:
            cleaned = True
            cancel()
