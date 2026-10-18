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

from portal import ui_decision
from portal.computer_use import parse_subgoal
from portal.tools import tool_schemas
from portal.ui_observations import NativeObservationArchive
from portal.ui_plans import NativePlanArchive
from portal.ui_recovery import NativeStrategyLedger
from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    parse_intent,
    parse_predicates,
    parse_step,
)
from qwen_omni_adapters.ui_effects import compile_immediate_transition, transition_descriptor_schema


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


def _compact_grounded_targets(schema: dict) -> dict:
    """Share repeated target schemas without discarding any admitted choice.

    Working-state wrapping can push a valid grounded schema past the wire
    budget. Repeated selectors describe the same controls across verbs; local
    references preserve both their constraints and their generation order.
    """
    compact = copy.deepcopy(schema)
    groups: dict[str, list[dict]] = {}

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get("target"), dict):
                target = properties["target"]
                if "$ref" not in target:
                    key = json.dumps(target, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    groups.setdefault(key, []).append(properties)
            for item in value.values():
                visit(item)

    visit(compact)
    definitions = compact.setdefault("$defs", {})
    for properties in groups.values():
        if len(properties) < 2:
            continue
        index = len(definitions)
        while (name := f"shared_grounded_target_{index}") in definitions:
            index += 1
        definitions[name] = properties[0]["target"]
        for owner in properties:
            owner["target"] = {"$ref": f"#/$defs/{name}"}
    return compact


def _grounded_phase_schema(allowed_tools: Iterable[str], observation: Mapping[str, Any], *,
                           intent_choice: bool = False, effect_contracts: bool = False) -> dict[str, Any]:
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

    first, boundaries, intent_selectors = [], [], []
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
            if intent_choice:
                for selector in selectors:
                    if selector not in intent_selectors:
                        intent_selectors.append(copy.deepcopy(selector))
                # Positive current controls go to primitive Choice. The four
                # missing-target discovery paths remain, but the host rejects
                # a discovery-shaped selector that actually matches a control.
                selectors = []
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
            if (verb in {"invoke", "click", "double_click", "right_click"}
                    or (intent_choice or effect_contracts) and verb in {"key_chord", "scroll"}
                    or effect_contracts and verb == "drag_drop"):
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
        if intent_choice or effect_contracts:
            boundary = schema["oneOf"][-1]
            boundary["required"].append("objective")
            boundary["properties"]["objective"] = {"type": "string", "minLength": 1, "maxLength": 500}
            boundary["description"] += " Supply an explicit planner objective within the exact user goal."
            if intent_choice:
                boundary["description"] += (" For a positive native control the verb is only an audited "
                    "preference for primitive Choice; a genuinely missing visual target retains the requested verb.")
    if intent_selectors:
        schema["oneOf"].append({"type": "object", "additionalProperties": False,
            "required": ["kind", "subgoal"], "properties": {
                "kind": {"const": "ui_subgoal"}, "subgoal": {"type": "object",
                    "additionalProperties": False, "required": ["id", "family", "objective", "target"],
                    "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 200},
                        "family": {"const": "pointer_or_activation"},
                        "objective": {"type": "string", "minLength": 1, "maxLength": 500},
                        "target": {"anyOf": intent_selectors}}},
                "remaining_criteria": copy.deepcopy(fallback["oneOf"][0]["properties"]["remaining_criteria"])},
            "description": "State a planner objective within the exact user goal and a current semantic selector. "
                "The active policy chooses a finite native primitive; the host resolves it freshly. "
                "No input authority or expected effect is supplied. Prefer this family for positive targets."})
    # This is a non-authorizing proposal. Optional text coverage can be omitted
    # by the bounded model projection; its absence must not hide an advertised
    # capability. The executor obtains full fresh coverage before any write.
    text_controls = [node for node in nodes if node["visible"] and node["enabled"] and not node["sensitive"]
                     and {"set_value", "type_text"}.intersection(node["operations"])]
    if intent_choice and text_controls:
        text_family = {"type": "object", "additionalProperties": False,
            "required": ["id", "family", "objective", "target", "mode", "text"], "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 200},
                "family": {"const": "text_entry"}, "mode": {"const": "replace"},
                "objective": {"type": "string", "minLength": 1, "maxLength": 500},
                "text": {"type": "string", "minLength": 1, "maxLength": 2048},
                "target": {"anyOf": [
                    target({"node_id": {"enum": [node["id"] for node in text_controls]}}, ["node_id"]),
                    target({"role": {"enum": sorted({node["role"] for node in text_controls})},
                            "name": {"enum": sorted({node["name"] for node in text_controls})}}, ["role"])]}}}
        existing = next((branch for branch in schema["oneOf"] if branch["properties"]["kind"]["const"] == "ui_subgoal"), None)
        if existing is not None:
            existing["properties"]["subgoal"] = {"oneOf": [existing["properties"]["subgoal"], text_family]}
            existing["description"] += " For exact text replacement use text_entry; the controller selects and verifies the input strategy."
        else:
            schema["oneOf"].append({"type": "object", "additionalProperties": False,
                "required": ["kind", "subgoal"], "properties": {"kind": {"const": "ui_subgoal"}, "subgoal": text_family,
                    "remaining_criteria": copy.deepcopy(fallback["oneOf"][0]["properties"]["remaining_criteria"])},
                "description": "For text replacement supply content and intent once; the controller chooses a strategy and verifies exact readback."})
        submit_controls = [node for node in text_controls if "key_chord" in node["operations"]]
        if submit_controls:
            submit_family = copy.deepcopy(text_family)
            submit_family["properties"]["family"] = {"const": "text_submit"}
            submit_family["properties"]["target"] = target(
                {"node_id": {"enum": [node["id"] for node in submit_controls]}}, ["node_id"])
            branch = next(b for b in schema["oneOf"] if b["properties"]["kind"]["const"] == "ui_subgoal")
            subgoal = branch["properties"]["subgoal"]
            branch["properties"]["subgoal"] = {"oneOf": [*subgoal.get("oneOf", [subgoal]), submit_family]}
            branch["description"] += " text_submit verifies replacement, presses Return once, then returns observed unknown consequences."
    if len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > _MAX_GROUNDED_SCHEMA_BYTES:
        schema = _compact_grounded_targets(schema)
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
              "policy_receipts", "point_receipts", "selection_receipts", "intent_selection_receipt", "evidence_authority", "resource", "stored",
              "updated", "unchanged", "topic", "key", "characters", "scope", "query", "results",
              "reading", "visual_inspection_receipt", "ui_boundary_receipt", "observed_changes", "text_entry_receipt", "text_submit_receipt",
              "recalled_observation", "observation_recall_receipt", "pointer_recovery", "pointer_recovery_receipt",
              "action_handoff", "action_handoff_revalidation", "immediate_effect_contract",
              "immediate_transition_receipt", "planner_workflow_claim", "strategy_filter_receipts", "transition",
              "verified_prefix_receipt", "verification_only_receipt", "recalled_plan", "plan_recall_receipt")
    fields = (*fields, "control_catalogue", "control_expansion_receipt")
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


_MAX_PLANNER_LEDGER_BYTES = 8192


def _planner_ledger(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound historical prompt evidence without changing the execution ledger."""
    if not ledger:
        return []

    def encoded(value, *, canonical=False):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=canonical).encode()

    def digest(value):
        return hashlib.sha256(encoded(value, canonical=True)).hexdigest()

    def fields(row):
        result = row.get("result")
        return len(row) - 1 + len(result) if isinstance(result, dict) else len(row)

    summaries, omitted_fields = [], []
    for index, row in enumerate(ledger):
        view = copy.deepcopy(row)
        omitted = {}
        if index < len(ledger) - 1 and "remaining_criteria" in view:
            omitted["remaining_criteria"] = digest(view.pop("remaining_criteria"))
        if isinstance(view.get("result"), dict) and "observed_changes" in view["result"]:
            omitted["result.observed_changes"] = digest(view["result"].pop("observed_changes"))
        view["history_reference"] = {"row_sha256": digest(row), "omitted_fields_sha256": omitted}
        summaries.append(view)
        omitted_fields.append(len(omitted))

    for start in range(len(ledger) + 1):
        omitted = ledger[:start]
        references = [{"row_sha256": digest(row), **({"phase": row["phase"]}
                      if type(row.get("phase")) is int and 1 <= row["phase"] <= 40 else {})}
                      for row in omitted[-12:]]
        metadata = {"schema": "robit.omni.planner-history.v1", "temporal_scope": "historical",
            "evidence_authority": "untrusted_receipt_summary", "input_authority": False,
            "current_ui_verified": False, "omitted_facts_are_not_absent": True,
            "hash_serialization": "sorted-key JSON, ensure_ascii=false, allow_nan=false, default separators, UTF-8",
            "byte_limit": _MAX_PLANNER_LEDGER_BYTES, "source_rows": len(ledger),
            "retained_rows": len(ledger) - start, "omitted_phase_count": start,
            "omitted_field_count": sum(fields(row) for row in omitted) + sum(omitted_fields[start:]),
            "field_count_scope": "row fields other than result, plus direct result fields",
            "source_ledger_sha256": digest(ledger), "omitted_phase_references": references,
            "omitted_reference_count": max(0, start - len(references)),
            "latest_criteria_status": ("retained" if start < len(ledger) else "omitted")
                if "remaining_criteria" in ledger[-1] else "not_provided",
            "latest_criteria_count": len(ledger[-1]["remaining_criteria"])
                if isinstance(ledger[-1].get("remaining_criteria"), list) else None,
            "latest_criteria_sha256": digest(ledger[-1]["remaining_criteria"])
                if "remaining_criteria" in ledger[-1] else None}
        projected = copy.deepcopy(summaries[start:])
        if projected:
            projected[-1]["history_projection"] = metadata
        else:
            projected = [{"history_projection": metadata}]
        if len(encoded(projected)) <= _MAX_PLANNER_LEDGER_BYTES:
            return projected
    raise ValueError("Historical planner metadata exceeds its byte bound")


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
    prefix = result.get("verified_prefix_receipt", {})
    contexts = prefix.get("step_contexts", [])
    if (result.get("status") == "complete" and not result.get("error")
            and prefix.get("all_immediate_readbacks_verified") is True and contexts
            and not result.get("outcomes")):
        bindings = []
        for context in contexts:
            receipt = context.get("verification_only_receipt", {})
            target = receipt.get("target", {})
            if (receipt.get("schema") != "robit.omni.native-verification-only.v1"
                    or receipt.get("status") != "verified" or receipt.get("input_attempted") is not False
                    or any(key not in target for key in ("node_id", "generation", "scope_id"))
                    or not isinstance(receipt.get("predicates"), list) or not receipt["predicates"]):
                return None
            # A repeated independently observed idempotent predicate is not a
            # new effect, even if coverage metadata or unrelated UI changes.
            bindings.append({"target": {key: target[key] for key in ("node_id", "generation", "scope_id")},
                **{key: receipt.get(key) for key in ("operation", "predicates", "args_sha256")},
                "session_id": receipt.get("fresh_source_ref", {}).get("session_id")})
        return hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
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


def _fresh_prefix_continuation(result, *, plan, observed, working, phase):
    """A current host no-input boundary can request planning, never authorize input.

    Internal readbacks legitimately renew revisions after the outer observation.
    Freshness is joined through the accepted current-phase handoff and the
    executor's own source references, not whole-window revision equality.
    """
    try:
        prefix = result["verified_prefix_receipt"]
        continuation = prefix["continuation"]
        contract = result["immediate_effect_contract"]
        source = continuation["source_ref"]
        state = observed["state"]
        handoff = working.action_handoff
        count = prefix["attempted_step_count"]
        planned = len(plan["steps"])
        if (result.get("status") != "needs_replan" or result.get("error")
                or result.get("dispatch_status") != "not_dispatched"
                or result.get("input_attempted") is not False or result.get("task_progress") is not False
                or type(result.get("new_effect_count")) is not int or result["new_effect_count"] != 0
                or any(result.get(key) != [] for key in
                    ("outcomes", "policy_receipts", "selection_receipts", "point_receipts"))
                or any(result.get(key) for key in ("dispatch_record", "receipt", "pointer_recovery"))
                or result.get("effect_receipt", {}).get("schema") != "robit.omni.ui-effect-receipt.v1"
                or result["effect_receipt"].get("effect_state") != "absent"
                or result["effect_receipt"].get("environment_may_have_changed") is not False
                or prefix.get("schema") != "robit.omni.native-verified-prefix.v1"
                or prefix.get("stop") != "unannotated_observation_boundary"
                or any(prefix.get(key) is not False for key in
                    ("input_authority", "task_progress", "goal_verified", "model_terminal_predicates_verified",
                     "all_immediate_readbacks_verified"))
                or any(type(prefix.get(key)) is not int for key in
                    ("attempted_step_count", "verified_prefix_count", "planned_step_count", "unattempted_step_count"))
                or not 0 < count == prefix["verified_prefix_count"] < planned <= 25
                or prefix["planned_step_count"] != planned
                or prefix["unattempted_step_count"] != planned - count
                or type(continuation.get("step_index")) is not int or continuation["step_index"] != count
                or continuation.get("schema") != "robit.omni.native-prefix-continuation.v1"
                or continuation.get("input_authority") is not False
                or continuation.get("step") != plan["steps"][count]
                or continuation.get("goal_sha256") != working.goal_sha256
                or continuation.get("objective") != working.commitment["objective"]
                or not isinstance(handoff, dict)
                or source != contract.get("source_ref")
                or set(source) != {"session_id", "scope_id", "revision"}
                or any(not isinstance(value, str) or not 1 <= len(value) <= 512 for value in source.values())
                or source["session_id"] != state["session_id"] or source["scope_id"] != state["scope"]["id"]):
            return False
        # Revalidate the already accepted claim; this does not refresh it.
        ui_decision.validate_action_handoff(handoff, goal=working.goal,
            source_epoch=working.epoch, source_phase=phase, state=state,
            objective=working.commitment["objective"])
        contexts = prefix["step_contexts"]
        if (not isinstance(contexts, list) or len(contexts) != count
                or contexts[0].get("action_handoff") != handoff):
            return False
        expected = compile_immediate_transition(continuation["step"],
            {"session_id": source["session_id"], "scope": {"id": source["scope_id"],
             "kind": state["scope"]["kind"]}, "revision": source["revision"]}).as_dict()
        if contract != expected or expected["mode"] != "observation_boundary":
            return False
        for index, context in enumerate(contexts):
            receipt = context["verification_only_receipt"]
            effect = context["immediate_effect_contract"]
            target = receipt["target"]
            references = [effect["source_ref"], receipt["original_source_ref"], receipt["fresh_source_ref"]]
            issued = plan["steps"][index]
            nodes = [node for node in state["nodes"] if node.get("id") == target.get("node_id")]
            if (len(nodes) != 1 or ui_decision._target_binding(nodes[0]) != target
                    or any(nodes[0].get("id" if key == "node_id" else key) != value
                           for key, value in issued["target"].items())
                    or index == 0 and target != handoff["target"]):
                return False
            before = {**state, "revision": effect["source_ref"]["revision"]}
            if effect != compile_immediate_transition(issued, before).as_dict():
                return False
            if (type(context.get("step_index")) is not int or context["step_index"] != index
                    or context.get("status") != "complete"
                    or type(context.get("outcome_count")) is not int or context["outcome_count"] != 0
                    or receipt.get("schema") != "robit.omni.native-verification-only.v1"
                    or receipt.get("status") != "verified"
                    or any(receipt.get(key) is not False for key in ("input_authority", "input_attempted", "task_progress"))
                    or effect.get("schema") != "robit.omni.ui-immediate-effect.v1"
                    or effect.get("mode") != "verified_step" or effect.get("non_authorizing") is not True
                    or effect.get("task_progress") is not False
                    or effect.get("operation") != plan["steps"][index]["verb"]
                    or receipt.get("operation") != effect["operation"]
                    or effect["source_ref"] != receipt["original_source_ref"]
                    or any(not isinstance(ref, dict) or set(ref) != set(source)
                        or any(not isinstance(value, str) or not 1 <= len(value) <= 512 for value in ref.values())
                        or any(ref[key] != source[key] for key in ("session_id", "scope_id")) for ref in references)
                    or any(not isinstance(target.get(key), str) or not target[key]
                           for key in ("node_id", "generation", "scope_id"))
                    or target["scope_id"] != source["scope_id"]
                    or not effect.get("predicates")
                    or receipt.get("predicates") != [{**predicate, "node_id": target["node_id"]}
                                                     for predicate in effect["predicates"]]
                    or receipt.get("predicates_sha256") != working.digest(receipt["predicates"])
                    or receipt.get("args_sha256") != working.digest(plan["steps"][index].get("args", {}))):
                return False
        phase_handoff = result["phase_handoff"]
        return (phase_handoff.get("schema") == "robit.omni.ui-phase-handoff.v1"
            and phase_handoff.get("executor") == "host_verified_prefix"
            and all(phase_handoff.get(key) is False for key in
                    ("terminal_verified", "goal_verified", "immediate_prefix_verified"))
            and type(phase_handoff.get("new_effect_count")) is int and phase_handoff["new_effect_count"] == 0
            and phase_handoff.get("next") == "replan")
    except (KeyError, TypeError, ValueError, AttributeError, ComputerUseError):
        return False


class NativePhaseCompilationError(Exception):
    """Trusted compiler JSON-decode failure; never carries the model's reply."""

    code = "invalid_native_phase_json"


class NativePhaseReplan(Exception):
    """A validated private planner decision requests another phase without input."""

    code = "native_planner_requested_replan"

    def __init__(self, reason: str):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError("Expected a bounded nonblank planner replan reason")
        self.reason = reason
        super().__init__(self.code)


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


def _working_phase_schema(allowed, phase_schema, state, *, pointer_recovery=None, observed_state=None,
                          action_handoffs=False):
    """Wrap, without changing, canonical phase contracts and root-relative refs."""
    def wrap(inner):
        inner = copy.deepcopy(inner)
        catalogue = (observed_state or {}).get("model_projection", {}).get("control_catalogue", {})
        discovery_refs = [row["reference"] for row in catalogue.get("entries", [])
                          if isinstance(row, dict) and isinstance(row.get("reference"), str)]
        if ("native_ui" in allowed and catalogue.get("schema") == "robit.omni.native-control-catalogue.v1"
                and catalogue.get("input_authority") is False and discovery_refs):
            inner["oneOf"].append({"type": "object", "additionalProperties": False,
                "properties": {"kind": {"const": "inspect_controls"},
                    "reference": {"enum": discovery_refs}, "remaining_criteria": {
                        "type": "array", "maxItems": 25,
                        "items": {"type": "string", "minLength": 1, "maxLength": 240}}},
                "required": ["kind", "reference"]})
        if pointer_recovery is not None and "native_ui" in allowed:
            offer = pointer_recovery
            if (offer.get("schema") != "robit.omni.pointer-recovery-proposal.v1"
                    or offer.get("input_authority") is not False or not re.fullmatch(r"[0-9a-f]{32}", offer.get("reference", ""))
                    or not isinstance(offer.get("choices"), list) or not 1 <= len(offer["choices"]) <= 12):
                raise ComputerUseError("Invalid host pointer recovery proposal")
            steps = []
            for choice in offer["choices"]:
                if (set(choice) != {"node_id", "verb"} or choice["verb"] not in {"click", "double_click", "right_click", "invoke"}
                        or not isinstance(choice["node_id"], str) or not 1 <= len(choice["node_id"]) <= 512):
                    raise ComputerUseError("Invalid host pointer recovery choice")
                steps.append({"type": "object", "additionalProperties": False,
                    "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "verb": {"const": choice["verb"]}, "target": {"type": "object", "additionalProperties": False,
                            "properties": {"node_id": {"const": choice["node_id"]}}, "required": ["node_id"]},
                        "args": {"type": "object", "additionalProperties": False, "properties": {}}},
                    "required": ["id", "verb", "target", "args"]})
            inner["oneOf"].append({"type": "object", "additionalProperties": False,
                "properties": {"kind": {"const": "ui_pointer_recovery"},
                    "reference": {"const": offer["reference"]}, "objective": {"type": "string", "minLength": 1, "maxLength": 500},
                    "step": {"oneOf": steps}, "remaining_criteria": {
                        "type": "array", "maxItems": 25, "items": {"type": "string", "minLength": 1, "maxLength": 240}}},
                "required": ["kind", "reference", "objective", "step"]})
        references = [item["reference"] for item in state.get("observations", {}).get("entries", [])]
        if references:
            inner["oneOf"].append({"type": "object", "additionalProperties": False,
                "properties": {"kind": {"const": "recall_observation"},
                    "reference": {"enum": references}, "remaining_criteria": {
                        "type": "array", "maxItems": 25, "items": {"type": "string", "minLength": 1, "maxLength": 240}}},
                "required": ["kind", "reference"]})
        plans = state.get("plan_claims", {}).get("entries", [])
        if plans:
            width = state["plan_claims"]["fragment_characters"]
            maximum = max(row["serialized_characters"] for row in plans)
            inner["oneOf"].append({"type": "object", "additionalProperties": False,
                "properties": {"kind": {"const": "recall_plan"},
                    "reference": {"enum": [row["reference"] for row in plans]},
                    "offset": {"enum": list(range(0, maximum, width))},
                    "remaining_criteria": {"type": "array", "maxItems": 25,
                        "items": {"type": "string", "minLength": 1, "maxLength": 240}}},
                "required": ["kind", "reference", "offset"]})
        definitions = copy.deepcopy(inner["$defs"])
        if action_handoffs:
            definitions["native_action_handoff"] = ui_decision.action_handoff_schema()
        if action_handoffs and observed_state is not None:
            definitions["native_immediate_transition"] = transition_descriptor_schema(observed_state)
        branches = inner["oneOf"]
        definitions["native_working_phase"] = {"oneOf": [copy.deepcopy(branch) for branch in branches
            if branch["properties"]["kind"]["const"] not in {"complete", "blocked"}]}
        definitions["native_commitment"] = {"type": "object", "additionalProperties": False,
            "required": ["objective", "intended_resource", "unmet_constraints"], "properties": {
                "objective": {"type": "string", "minLength": 1, "maxLength": 500},
                "intended_resource": {"type": "string", "maxLength": 512},
                "unmet_constraints": {"type": "array", "maxItems": 10,
                    "items": {"type": "string", "minLength": 1, "maxLength": 240}}}}
        phase_commitment = copy.deepcopy(definitions["native_commitment"])
        phase_commitment["required"].remove("objective")
        del phase_commitment["properties"]["objective"]
        definitions["native_phase_commitment"] = phase_commitment
        options = []
        decisions = ["start"] if state["source_phase"] == 1 else ["recover"] if state["recovery_mode"] else ["continue", "advance"]
        working = definitions["native_working_phase"]["oneOf"]
        groups = {False: [], True: []}
        for index, branch in enumerate(working):
            kind = branch["properties"]["kind"]["const"]
            owns_objective = (kind == "ui_subgoal" or
                              kind in {"ui_boundary", "ui_pointer_recovery"} and "objective" in branch.get("required", []))
            groups[owns_objective].append({"$ref": f"#/$defs/native_working_phase/oneOf/{index}"})
        choices = [(decisions, {"oneOf": refs}, owns_objective)
                   for owns_objective, refs in groups.items() if refs]
        for decision, phase, owns_objective in [*choices, *[
            (["verify" if branch["properties"]["kind"]["const"] == "complete" else "blocked"], copy.deepcopy(branch), False)
            for branch in branches if branch["properties"]["kind"]["const"] == "blocked" or (
                branch["properties"]["kind"]["const"] == "complete" and not state["recovery_mode"])]]:
            options.append({"type": "object", "additionalProperties": False,
                # llama.cpp emits properties in schema order. Select the phase
                # before its branch-specific commitment; otherwise generating
                # commitment.objective prematurely excludes all subgoals.
                "required": ["assessment", "phase", "commitment"], "properties": {
                    "assessment": {"type": "object", "additionalProperties": False,
                        "required": ["source_epoch", "source_phase", "decision", "summary"], "properties": {
                            "source_epoch": {"const": state["source_epoch"]},
                            "source_phase": {"const": state["source_phase"]},
                            "decision": {"enum": decision},
                            "summary": {"type": "string", "minLength": 1, "maxLength": 500}}},
                    "phase": phase, "commitment": {"$ref": "#/$defs/native_phase_commitment" if owns_objective
                                                   else "#/$defs/native_commitment"},
                    **({"action_handoff": {"$ref": "#/$defs/native_action_handoff"}} if action_handoffs else {}),
                    **({"transition": {"$ref": "#/$defs/native_immediate_transition"}}
                       if action_handoffs and observed_state is not None else {})}})
        return {"$defs": definitions, "oneOf": options}
    schema = wrap(phase_schema)
    if len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > _MAX_GROUNDED_SCHEMA_BYTES:
        schema = _compact_grounded_targets(schema)
    if len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > _MAX_GROUNDED_SCHEMA_BYTES:
        fallback = phase_response_schema(allowed)
        fallback["oneOf"] = [branch for branch in fallback["oneOf"] if branch["properties"]["kind"]["const"] != "ui"]
        schema = wrap(fallback)
    if len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > _MAX_GROUNDED_SCHEMA_BYTES:
        raise ComputerUseError("Native working schema exceeds the unchanged grammar budget")
    return schema


class _NativeWorkingState:
    """Bounded historical host facts, separate from renewable planner claims."""
    def __init__(self, goal, *, strategy_ledger=None):
        self.goal = goal
        self.goal_sha256 = hashlib.sha256(goal.encode()).hexdigest()
        self.epoch, self.commitment, self.assessment = uuid.uuid4().hex, None, None
        self.phase_refs, self.facts, self.latest, self.last_unverified = [], [], None, None
        self.receipts: set[str] = set()
        self.recovery = False
        self.observations = NativeObservationArchive(self.epoch)
        self.action_handoff = None
        self.strategy_ledger = strategy_ledger
        self.plans = NativePlanArchive(self.epoch, self.goal_sha256)

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()

    def view(self, phase, failed):
        facts = copy.deepcopy(self.facts[-8:])
        value = {"schema": "robit.omni.native-working-state.v1", "input_authority": False,
            "goal_sha256": self.goal_sha256, "source_epoch": self.epoch, "source_phase": phase,
            "commitment": copy.deepcopy(self.commitment), "previous_assessment": copy.deepcopy(self.assessment),
            "phase_ref_format": ["source_phase", "host_result_sha256", "actual_outcome_count"],
            "phase_refs": copy.deepcopy(self.phase_refs), "executed_facts": facts,
            "omitted_executed_facts": len(self.facts) - len(facts),
            "all_executed_facts_sha256": self.digest(self.facts),
            "latest_host_outcome": copy.deepcopy(self.latest),
            "last_unverified_host_outcome": copy.deepcopy(self.last_unverified), "recovery_mode": self.recovery,
            "consecutive_unverified_count": failed, "fact_authority": "historical_executor_receipts_not_goal_progress",
            "observations": self.observations.view(),
            "references": "Phase/outcome hashes are audit pointers. Observation references can be expanded with recall_observation; historical only, no input authority."}
        if self.strategy_ledger is not None:
            value["strategy_recovery"] = self.strategy_ledger.view()
        if self.plans.records:
            value["plan_claims"] = self.plans.view()
        while len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 12288 and facts:
            facts.pop(0)
            value["omitted_executed_facts"] += 1
        recovery = value.get("strategy_recovery", {})
        while len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 12288 and recovery.get("facts"):
            recovery["facts"].pop(0)
            recovery["omitted_fact_count"] += 1
        # Keep full claims internally. A legal Unicode claim may consume more
        # bytes than its character bound; omissions are explicit, never absence.
        for group, key in (("commitment", "unmet_constraints"), ("previous_assessment", "summary"),
                           ("commitment", "intended_resource"), ("commitment", "objective")):
            if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) <= 12288:
                break
            claim = value[group]
            if claim is not None and key in claim:
                removed = claim.pop(key)
                claim.setdefault("omitted_fields", {})[key] = {"sha256": self.digest(removed),
                    "status": "unavailable_in_bounded_view_not_absent", "items_or_characters": len(removed)}
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 12288:
            raise ComputerUseError("Native working state exceeds its bounded evidence representation")
        return value

    def planning_view(self, phase, failed):
        value = self.view(phase, failed)
        value["observations"] = self.observations.planning_view()
        return value

    def accept(self, reply, *, state=None, previous_result=None):
        commitment, assessment, phase = reply["commitment"], reply["assessment"], reply["phase"]
        objective_path = ("/objective" if phase["kind"] in {"ui_boundary", "ui_pointer_recovery"} and "objective" in phase else
                          "/subgoal/objective" if phase["kind"] == "ui_subgoal" else None)
        if objective_path is not None:
            if "objective" in commitment:
                raise ComputerUseError("Phase-owned objective must be supplied once, in the phase")
            objective = phase["objective"] if objective_path == "/objective" else phase["subgoal"]["objective"]
            commitment = {**commitment, "objective": objective,
                "objective_source": {"path": objective_path, "source_phase": assessment["source_phase"],
                                     "phase_sha256": self.digest(phase)}}
        if not commitment["objective"].strip() or not assessment["summary"].strip():
            raise ComputerUseError("Working commitment and assessment must be nonblank")
        handoff = None
        if "action_handoff" in reply:
            if state is None:
                raise ComputerUseError("Action handoff requires its captured observation")
            selector = (phase.get("subgoal", {}).get("target") if phase["kind"] == "ui_subgoal" else
                        phase.get("step", {}).get("target") if phase["kind"] in {"ui_boundary", "ui_pointer_recovery"}
                        else phase.get("steps", [{}])[0].get("target") if phase["kind"] == "ui" else None)
            claimed = reply["action_handoff"].get("target", {})
            target = next((node for node in state.get("nodes", []) if node.get("id") == claimed.get("node_id")), None)
            if (not isinstance(selector, dict) or target is None or any(
                    target.get("id" if key == "node_id" else key) != value for key, value in selector.items())):
                raise ComputerUseError("Action handoff does not describe the proposed target")
            previous = previous_result or {}
            recovery = phase["kind"] == "ui_pointer_recovery"
            if recovery and not isinstance(previous.get("action_handoff"), dict):
                raise ComputerUseError("Recovery handoff has no retained source")
            handoff = ui_decision.validate_action_handoff(reply["action_handoff"], goal=self.goal,
                source_epoch=self.epoch, source_phase=assessment["source_phase"], state=state,
                objective=commitment["objective"], target=target,
                **({"predecessor": previous["action_handoff"], "recovery_reference": phase["reference"]}
                   if recovery else {}))
        self.commitment = {**copy.deepcopy(commitment), "authority": "planner_claim", "input_authority": False}
        self.assessment = {**copy.deepcopy(assessment), "authority": "planner_claim", "input_authority": False}
        self.action_handoff = handoff

    def record(self, phase, kind, result, *, recover):
        if self.strategy_ledger is not None:
            self.strategy_ledger.record(result, source_phase=phase)
        digest = self.digest(result)
        actual = result.get("outcomes", [])
        if not actual and isinstance(result.get("dispatch_record"), dict):
            actual = [result["dispatch_record"]]
        self.phase_refs.append([phase, digest, len(actual)])
        self.recovery = recover
        fields = {key: copy.deepcopy(result[key]) for key in
                  ("status", "error", "reason", "dispatch_status", "input_attempted") if key in result}
        effect = result.get("effect_receipt", {})
        if "effect_state" in effect:
            fields["effect_state"] = copy.deepcopy(effect["effect_state"])
        if isinstance(result.get("ui_boundary_receipt"), dict):
            fields["ui_boundary_receipt"] = {key: copy.deepcopy(result["ui_boundary_receipt"][key]) for key in
                ("input_attempted", "adapter_called", "cached", "observed", "verified", "environment_may_have_changed",
                 "task_progress", "new_effect_count") if key in result["ui_boundary_receipt"]}
        omitted = {}
        for key in ("reason", "error", "status", "effect_state", "ui_boundary_receipt", "dispatch_status", "input_attempted"):
            if len(json.dumps(fields, ensure_ascii=False, allow_nan=False).encode()) <= 1536:
                break
            if key in fields:
                omitted[key] = self.digest(fields.pop(key))
        self.latest = {"source_phase": phase, "result_sha256": digest, "facts": fields,
                       "omitted_fields_sha256": omitted, "input_authority": False}
        if recover:
            self.last_unverified = copy.deepcopy(self.latest)
        for index, item in enumerate(actual):
            action, receipt, effect = item.get("action", {}), item.get("receipt", {}), item.get("effect_receipt", {})
            outcome_hash = self.digest(item)
            fact = {"source_phase": phase, "source_outcome_index": index, "outcome_sha256": outcome_hash,
                "status": item.get("status"), "receipt_status": receipt.get("status"),
                "input_attempted": receipt.get("input_attempted"), "effect_state": effect.get("effect_state"),
                "receipt_seen_before": outcome_hash in self.receipts,
                "action": {key: copy.deepcopy(action[key]) for key in ("node_id", "generation", "verb") if key in action},
                "args_sha256": self.digest(action.get("args")), "input_authority": False}
            fact["input_attempted_available"] = "input_attempted" in receipt
            fact["effect_state_available"] = "effect_state" in effect
            before = item.get("before", {})
            targets = [node for node in before.get("nodes", []) if node.get("id") == action.get("node_id")]
            if len(targets) == 1 and targets[0].get("sensitive") is False:
                fact["observed_referent"] = {key: copy.deepcopy(targets[0][key]) for key in
                    ("id", "generation", "scope_id", "role", "name") if key in targets[0]}
            self.facts.append(fact)
            self.receipts.add(outcome_hash)


def run_native_goal(goal: str, *, observe: Callable[[], dict], compile_phase: Callable[[dict], dict],
                    execute: Callable[[str, dict], dict], cancel: Callable[[], None],
                    allowed_tools: Iterable[str], max_phases: int = 40, max_seconds: float = 1200,
                    model: str = "", clock: Callable[[], float] = time.monotonic,
                    cancelled: Callable[[], bool] | None = None,
                    intent_choice: bool = False, working_state: bool = False,
                    effect_contracts: bool = False,
                    strategy_ledger: NativeStrategyLedger | None = None) -> Iterator[dict[str, Any]]:
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
    if type(effect_contracts) is not bool or effect_contracts and not working_state:
        raise ValueError("Immediate effect contracts require native working state")
    if strategy_ledger is not None and (not isinstance(strategy_ledger, NativeStrategyLedger) or not effect_contracts):
        raise ValueError("Strategy history requires the private immediate-effect protocol")
    allowed = frozenset(allowed_tools)
    if "native_ui" not in allowed:
        raise ValueError("Native task requires explicit native_ui permission")
    started, traces, ledger, last_result = clock(), [], [], None
    failed, normal, cleaned, phases, claims = 0, False, False, 0, []
    zero_effect_ui_phases: set[str] = set()
    applied_ui_transitions: set[str] = set()
    boundary_observations: set[str] = set()
    pointer_resolution_pending = False
    prefix_continuation_pending = False
    working = _NativeWorkingState(goal, strategy_ledger=strategy_ledger) if working_state else None

    def boundary():
        if cancelled is not None and cancelled():
            return "cancelled"
        return "deadline_exceeded" if clock() - started >= max_seconds else None

    def finish(status, message):
        event = {"type": "final", "response": {"model": model,
            "message": {"role": "assistant", "content": message},
            "portal": {"safe_tools_executed": copy.deepcopy(traces), "native_task": {
                "status": status, "task_verified": False, "phases": phases,
                "elapsed_s": max(0, clock() - started), "remaining_criteria_authority": "planner_claim",
                "remaining_criteria": copy.deepcopy(claims),
                "ledger": copy.deepcopy(ledger)}}}}
        if working is not None:
            event["response"]["portal"]["native_task"]["working_state"] = working.view(phases, failed)
        return event

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
            if working is not None:
                working.observations.remember(phases, current)
            schema = _grounded_phase_schema(allowed, current, intent_choice=intent_choice,
                                            effect_contracts=effect_contracts)
            working_view = working.planning_view(phases, failed) if working is not None else None
            if working is not None:
                schema = _working_phase_schema(allowed, schema, working_view,
                    pointer_recovery=(last_result or {}).get("pointer_recovery"),
                    observed_state=current.get("state"), action_handoffs=effect_contracts)
            validator = Draft202012Validator(schema)
            packet = {"goal": goal, "current_state": copy.deepcopy(current),
                      "ledger": _planner_ledger(ledger[-12:]), "last_result": copy.deepcopy(last_result),
                      "response_schema": copy.deepcopy(schema),
                      "remaining_seconds": max_seconds - (clock() - started)}
            if working is not None:
                packet["working_state"] = working_view
            yield {"type": "progress", "stage": "native_task_plan", "phase": phases}
            if reason := boundary():
                yield finish(reason, "Native task stopped: " + reason)
                return
            packet["remaining_seconds"] = max_seconds - (clock() - started)
            try:
                plan = compile_phase(packet)
            except (NativePhaseCompilationError, NativePhaseReplan) as error:
                if reason := boundary():
                    yield finish(reason, "Native task stopped: " + reason)
                    return
                replanning = isinstance(error, NativePhaseReplan)
                last_result = {"error": error.code,
                    "dispatch_status": "not_dispatched", "input_attempted": False,
                    "new_effect_count": 0,
                    "reason": ("Planner requested replanning; its reason is an unverified claim: " + error.reason
                               if replanning else
                               "The compiler reply was not complete valid JSON. Return a smaller complete phase "
                               "matching the unchanged schema, without repairing or replaying the invalid reply.")}
                ledger.append({"phase": phases, "kind": "replan" if replanning else "invalid", **last_result})
                ledger = ledger[-12:]
                if working is not None:
                    working.record(phases, "replan" if replanning else "invalid", last_result, recover=True)
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
                if working is not None:
                    reply = plan
                    plan = plan["phase"]
                kind = plan["kind"]
                if kind == "ui":
                    for step in plan["steps"]:
                        parse_step(step)
                    parse_predicates(plan["final_predicates"], terminal=True)
                if kind == "ui_boundary":
                    parse_intent(plan["step"])
                    if intent_choice and plan["step"]["verb"] in {"invoke", "click", "double_click", "right_click"}:
                        parse_subgoal({"id": plan["step"]["id"], "family": "pointer_or_activation",
                                       "objective": plan["objective"], "target": plan["step"]["target"]})
                if kind == "ui_subgoal":
                    parse_subgoal(plan["subgoal"])
                if working is not None:
                    working.accept(reply, state=current.get("state"), previous_result=last_result)
            except (ValidationError, ComputerUseError, ValueError, TypeError) as exc:
                last_result = {"error": "invalid_native_phase", "dispatch_status": "not_dispatched",
                               "reason": str(exc.message if isinstance(exc, ValidationError) else exc)[:600]}
                if working is not None:
                    last_result["input_attempted"] = False
                ledger.append({"phase": phases, "kind": "invalid", **last_result})
                if working is not None:
                    working.record(phases, "invalid", last_result, recover=True)
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
            retained_plan = working.plans.remember(phases, plan) if kind == "ui" and effect_contracts else None
            name = "native_ui" if kind in {"ui", "ui_boundary", "ui_subgoal", "ui_pointer_recovery", "visual_inspection", "recall_observation", "recall_plan", "inspect_controls"} else kind
            arguments = ({"operation": "verified_prefix", "steps": plan["steps"],
                          "objective": working.commitment["objective"], "source_phase": phases}
                         if kind == "ui" and effect_contracts else
                         {"operation": "workflow", "steps": plan["steps"], "final_predicates": plan["final_predicates"]}
                         if kind == "ui" else {"operation": "resolve_pointer", "reference": plan["reference"],
                             "intent": plan["step"], "objective": plan["objective"]}
                         if kind == "ui_pointer_recovery" else {"operation": "observation_boundary", "intent": plan["step"]}
                         if kind == "ui_boundary" else {"operation": "intent_boundary", "subgoal": plan["subgoal"],
                             "feedback": {key: copy.deepcopy(last_result[key]) for key in
                                 ("status", "error", "reason", "effect_receipt", "ui_boundary_receipt")
                                 if key in last_result} if last_result is not None else None}
                         if kind == "ui_subgoal" else {"operation": "inspect_visual"} if kind == "visual_inspection"
                         else {"operation": "recall_observation", "reference": plan["reference"]} if kind == "recall_observation"
                         else {"operation": "recall_plan", "reference": plan["reference"], "offset": plan["offset"]} if kind == "recall_plan"
                         else {"operation": "inspect_controls", "reference": plan["reference"]} if kind == "inspect_controls"
                         else {key: value for key, value in plan.items()
                                                if key not in {"kind", "remaining_criteria"}})
            if kind == "ui_boundary" and (intent_choice or effect_contracts):
                arguments.update(objective=plan["objective"], feedback={key: copy.deepcopy(last_result[key]) for key in
                    ("status", "error", "reason", "effect_receipt", "ui_boundary_receipt")
                    if key in last_result} if last_result is not None else None)
            if name == "native_ui" and working is not None and working.action_handoff is not None:
                arguments["action_handoff"] = copy.deepcopy(working.action_handoff)
            if name == "native_ui" and effect_contracts and kind in {"ui", "ui_boundary", "ui_subgoal", "ui_pointer_recovery"}:
                if "transition" in reply:
                    arguments["transition"] = copy.deepcopy(reply["transition"])
                if kind == "ui_boundary":
                    arguments.update(operation="immediate_transition", step=arguments.pop("intent"))
                    arguments.pop("feedback", None)
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
            # Private task-owned expansion never calls a desktop input adapter.
            result = (working.observations.recall(plan["reference"]) if kind == "recall_observation" and working is not None
                      else working.plans.safe_recall(plan["reference"], plan["offset"]) if kind == "recall_plan" and working is not None
                      else execute(name, copy.deepcopy(arguments)))
            if not isinstance(result, dict):
                raise ValueError("Native executor returned no typed receipt")
            if kind == "ui" and effect_contracts:
                # The complete original instructions remain retrievable claims.
                attempted = result.get("verified_prefix_receipt", {}).get("attempted_step_count", 0)
                result["planner_workflow_claim"] = {"authority": "unverified_planner_plan",
                    "input_authority": False, "terminal_verified": False,
                    "retained_plan": retained_plan, "suffix_start_index": attempted,
                    "original_step_count": len(plan["steps"]),
                    "steps_sha256": _NativeWorkingState.digest(plan["steps"]),
                    "remaining_steps": copy.deepcopy(plan["steps"][attempted:attempted + 5]),
                    "omitted_remaining_steps": max(0, len(plan["steps"]) - attempted - 5),
                    "terminal_predicates_sha256": _NativeWorkingState.digest(plan["final_predicates"])}
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
            recoverable_prefix = kind == "ui" and _new_verified_ui_prefix(result, applied_ui_transitions)
            if (kind == "ui" and result.get("status") == "complete" and not result.get("error")
                    and not recoverable_prefix and _new_verified_ui_prefix(result, set())):
                result = {**result, "error": "ui_phase_repeated_transition",
                    "reason": "This input repeats a previously verified native state transition. "
                              "Its effect is real, but it adds no new observed state. Recover with a "
                              "different applicable action or stop with unmet criteria; prior receipts remain historical."}
            content = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            replayed_only = (kind == "ui" and bool(result.get("outcomes"))
                             and all(item.get("effect_receipt", {}).get("effect_state") == "replayed"
                                     for item in result["outcomes"]))
            ok = not result.get("error") and not replayed_only and (kind != "ui" or result.get("status") == "complete")
            if kind == "ui_subgoal" and plan["subgoal"]["family"] == "text_entry":
                ok = ok and result.get("status") == "complete" and result.get("text_entry_receipt", {}).get("exact_value_verified") is True
            if kind == "visual_inspection":
                ok = ok and _accepted_visual_reading(result)
            if kind == "recall_observation":
                ok = ok and result.get("observation_recall_receipt", {}).get("first_expansion_in_task") is True
            if kind == "recall_plan":
                ok = ok and result.get("plan_recall_receipt", {}).get("first_expansion_in_task") is True
            if kind == "inspect_controls":
                ok = ok and result.get("control_expansion_receipt", {}).get("first_expansion_in_task") is True
            # An unresolved host continuation gets one planner opportunity,
            # independent of fresh tokens, observations or rewritten claims.
            # Only a new producer-bound verified input effect resolves it.
            if recoverable_prefix:
                prefix_continuation_pending = False
            deferred_prefix = (kind == "ui" and effect_contracts and working is not None
                and not prefix_continuation_pending
                and _fresh_prefix_continuation(result, plan=plan, observed=current,
                    working=working, phase=phases))
            if deferred_prefix:
                prefix_continuation_pending = True
            novel_boundary = False
            is_boundary = (kind in {"ui_boundary", "ui_pointer_recovery"}
                or kind == "ui_subgoal" and plan["subgoal"]["family"] != "text_entry"
                or kind == "ui" and result.get("immediate_effect_contract", {}).get("mode") == "observation_boundary")
            offer = result.get("pointer_recovery", {})
            deferred_pointer = (working is not None and is_boundary and result.get("status") == "abstained"
                and isinstance(offer, dict) and offer.get("schema") == "robit.omni.pointer-recovery-proposal.v1"
                and offer.get("input_authority") is False and bool(offer.get("choices"))
                and result.get("ui_boundary_receipt", {}).get("input_attempted") is False
                and result.get("effect_receipt", {}).get("effect_state") == "absent")
            if is_boundary:
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
            tool.update(ok=ok, status="complete", result=content)
            traces.append(copy.deepcopy(tool))
            last_result = _compact_result(result)
            ledger.append({"phase": phases, "kind": kind, "ok": ok,
                           "deferred_to_planner": deferred_prefix or deferred_pointer and not pointer_resolution_pending,
                           "remaining_criteria": copy.deepcopy(plan.get("remaining_criteria", [])),
                           "criteria_authority": "planner_claim", "result": {
                               key: copy.deepcopy(last_result[key]) for key in
                               ("status", "error", "reason", "phase_handoff", "ui_state_receipt",
                                "verification_receipt", "effect_receipt", "stored", "updated", "unchanged", "visual_inspection_receipt",
                                "ui_boundary_receipt", "observed_changes", "intent_selection_receipt", "text_entry_receipt",
                                "observation_recall_receipt", "pointer_recovery_receipt", "control_expansion_receipt")
                               if key in last_result}})
            if working is not None:
                ledger[-1].update(planner_assessment=copy.deepcopy(working.assessment),
                                  planner_commitment=copy.deepcopy(working.commitment))
            ledger = ledger[-12:]
            if is_boundary:
                # Observation protocol success is not verified task progress.
                # One fast-policy deferral hands the attempt to the slow planner.
                # Repeating fast decisions without resolving it still spends the
                # failure budget, even if each mints a different proposal token.
                if recoverable_prefix:
                    failed = 0
                failed += (not novel_boundary and not deferred_prefix
                           and not (deferred_pointer and not pointer_resolution_pending))
            elif kind == "inspect_controls":
                # New knowledge permits reasoning, never erases action failures.
                failed += not ok
            else:
                no_input_prefix = (result.get("verified_prefix_receipt", {}).get("all_immediate_readbacks_verified") is True
                    and not result.get("outcomes"))
                failed = (failed if ok else failed + 1) if no_input_prefix else 0 if ok or recoverable_prefix else failed + 1
            pointer_resolution_pending = deferred_pointer
            if working is not None:
                working.record(phases, kind, result, recover=not ok or is_boundary
                    or result.get("status") in {"needs_replan", "unresolved", "rejected", "abstained", "cancelled"})
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
