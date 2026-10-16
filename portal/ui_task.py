"""Server-owned native goal → structured phase → guarded controller loop.

Callbacks own authenticated session scope and transport deadlines. ``observe``
returns a bounded production observation receipt, including mapped windows;
that whole receipt is current_state. No model tool-discovery loop runs here.
"""
from __future__ import annotations

import copy
import json
import math
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
        branch("complete", {"message": {"type": "string", "minLength": 1, "maxLength": 8000}}, ["message"]),
        # The pinned llama.cpp grammar parser rejects an exactly 2,000-fold
        # repetition. Keep this private planner field below that boundary.
        branch("blocked", {"reason": {"type": "string", "minLength": 1, "maxLength": 1999}}, ["reason"])])
    return copy.deepcopy({"$defs": ui["$defs"], "oneOf": alternatives})


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep receipts and readback; never multiply historical UI maps in planning."""
    fields = ("status", "error", "reason", "message", "phase_handoff", "terminal_receipt",
              "verification_receipt", "ui_state_receipt", "effect_receipt", "action_receipt",
              "policy_receipts", "point_receipts", "evidence_authority", "resource", "stored",
              "updated", "unchanged", "topic", "key", "characters", "scope", "query", "results")
    compact = {key: copy.deepcopy(result[key]) for key in fields if key in result}
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
            plan = compile_phase(packet)
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
            name = "native_ui" if kind == "ui" else kind
            arguments = ({"operation": "workflow", "steps": plan["steps"], "final_predicates": plan["final_predicates"]}
                         if kind == "ui" else {key: value for key, value in plan.items()
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
            content = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            replayed_only = (kind == "ui" and bool(result.get("outcomes"))
                             and all(item.get("effect_receipt", {}).get("effect_state") == "replayed"
                                     for item in result["outcomes"]))
            ok = not result.get("error") and not replayed_only and (kind != "ui" or result.get("status") == "complete")
            tool.update(ok=ok, status="complete", result=content)
            traces.append(copy.deepcopy(tool))
            last_result = _compact_result(result)
            ledger.append({"phase": phases, "kind": kind, "ok": ok,
                           "remaining_criteria": copy.deepcopy(plan.get("remaining_criteria", [])),
                           "criteria_authority": "planner_claim", "result": {
                               key: copy.deepcopy(last_result[key]) for key in
                               ("status", "error", "reason", "phase_handoff", "ui_state_receipt",
                                "verification_receipt", "effect_receipt", "stored", "updated", "unchanged")
                               if key in last_result}})
            ledger = ledger[-12:]
            failed = 0 if ok else failed + 1
            yield {"type": "tool", "phase": "complete", "round": phases, "tools": [copy.deepcopy(tool)]}
            if failed >= 3:
                yield finish("blocked", "Three consecutive native phases were invalid or unverified.")
                return
        yield finish("phase_budget_exceeded", "Native task reached its bounded phase budget.")
    except Exception as exc:
        reason = boundary()
        yield finish(reason or "blocked", "Native task stopped: " + reason if reason else
                     "Native task callback failed: " + type(exc).__name__)
    finally:
        if not normal and not cleaned:
            cleaned = True
            cancel()
