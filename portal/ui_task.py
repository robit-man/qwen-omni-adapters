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
from qwen_omni_adapters.computer_use import ComputerUseError, parse_predicates, parse_step


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
              "reading", "visual_inspection_receipt")
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
            ("id", "generation", "scope_id", "role", "name", "value", "checked", "selected", "focused", "text_selection")
            if key in node} for node in after.get("nodes", []) if node.get("id") in wanted and not node.get("sensitive")][:21]
        outcomes.append(outcome)
    if outcomes:
        compact["outcomes"] = outcomes
    return compact


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
        steps.append({"action": binding, "before": before, "after": after})
    encoded = json.dumps({"observed": observed, "steps": steps, "terminal": terminal}, sort_keys=True,
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
    schema = phase_response_schema(allowed)
    validator = Draft202012Validator(schema)
    started, traces, ledger, last_result = clock(), [], [], None
    failed, normal, cleaned, phases, claims = 0, False, False, 0, []
    zero_effect_ui_phases: set[str] = set()
    applied_ui_transitions: set[str] = set()

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
            name = "native_ui" if kind in {"ui", "visual_inspection"} else kind
            arguments = ({"operation": "workflow", "steps": plan["steps"], "final_predicates": plan["final_predicates"]}
                         if kind == "ui" else {"operation": "inspect_visual"} if kind == "visual_inspection" else {key: value for key, value in plan.items()
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
            recoverable_prefix = kind == "ui" and _new_verified_ui_prefix(result, applied_ui_transitions)
            tool.update(ok=ok, status="complete", result=content)
            traces.append(copy.deepcopy(tool))
            last_result = _compact_result(result)
            ledger.append({"phase": phases, "kind": kind, "ok": ok,
                           "remaining_criteria": copy.deepcopy(plan.get("remaining_criteria", [])),
                           "criteria_authority": "planner_claim", "result": {
                               key: copy.deepcopy(last_result[key]) for key in
                               ("status", "error", "reason", "phase_handoff", "ui_state_receipt",
                                "verification_receipt", "effect_receipt", "stored", "updated", "unchanged", "visual_inspection_receipt")
                               if key in last_result}})
            ledger = ledger[-12:]
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
