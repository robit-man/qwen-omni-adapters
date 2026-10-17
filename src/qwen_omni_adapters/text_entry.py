"""Effect-oriented native text replacement with explicit, finite strategies.

This internal task family never turns set_value into a hidden typing fallback.
Each call selects one strategy. Failed writes remain historical evidence and
are excluded from subsequent choices for this control generation and value.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from .computer_use import ComputerUseError, parse_step, predicate_satisfied


def parse_text_entry(value: Any) -> dict:
    if (not isinstance(value, dict) or set(value) != {"id", "family", "objective", "target", "mode", "text"}
            or value.get("family") != "text_entry" or value.get("mode") != "replace"
            or not isinstance(value.get("objective"), str) or not value["objective"].strip()
            or len(value["objective"]) > 500 or not isinstance(value.get("text"), str)
            or not 1 <= len(value["text"]) <= 2048 or "\x00" in value["text"]):
        raise ComputerUseError("Expected a bounded text-entry replacement subgoal")
    parse_step({"id": value["id"], "verb": "set_value", "target": value["target"],
                "args": {"value": value["text"]},
                "expect": [{"kind": "value", "node_id": "$target", "value": value["text"]}]})
    return copy.deepcopy(value)


def complete_text(node: dict) -> bool:
    coverage, value = node.get("text_observation", {}), node.get("value")
    return (isinstance(value, str) and coverage.get("truncated") is False
            and coverage.get("character_count") == len(value)
            and coverage.get("captured_range") == {"start": 0, "end": len(value)}
            and coverage.get("selection_count") in {0, 1}
            and (coverage["selection_count"] == 0 or isinstance(node.get("text_selection"), dict)))


def candidates(controller, state: dict, intent: dict) -> tuple[list[dict], list[dict]]:
    bank, history = [], []
    for node in state["nodes"]:
        if (not all(node.get("id" if key == "node_id" else key) == value for key, value in intent["target"].items())
                or not node["visible"] or not node["enabled"] or node["sensitive"] or not complete_text(node)):
            continue
        previous = controller.text_write_history(state, node, intent["text"])
        history.extend(previous)
        failed = {row["strategy"] for row in previous if not row["effect_verified"]}
        operations = set(node["operations"])
        strategies = []
        if node["value"] == intent["text"]:
            strategies.append("verify_existing")
        else:
            if ("type_text" in operations and (node.get("focused") or "focus" in operations)
                    and (not node["value"] or "select_text" in operations)):
                strategies.append("physical_replace")
            if "set_value" in operations:
                strategies.append("native_set")
        # Transport is an executor decision: equivalent implementations must
        # not compete for semantic Choice probability. Prefer a completely
        # observed physical replacement; use the explicit setter when that
        # transport is unavailable/previously failed. Never retry either one.
        applicable = [strategy for strategy in strategies if strategy not in failed]
        for strategy in applicable[:1]:
            bank.append({"id": f"text-{len(bank)}", "kind": "text_strategy", "strategy": strategy,
                "node_id": node["id"], "generation": node["generation"], "role": node["role"], "name": node["name"],
                "effect": "exact_full_value", "non_authorizing": True,
                "strategy_source": "host_capabilities_and_execution_history"})
    if len(bank) > 48:
        bank = []
    return [*bank, {"id": "text-abstain", "kind": "abstain"}], history


def run_text_entry(controller, intent: dict, choose, cancel=None) -> dict:
    """Select once, freshly bind, execute a verified prefix, then independently read.

    Deterministic focus/select/type transitions consume the selected strategy;
    none invents document content or a semantic outcome. Stop on the first
    unverified prefix, retaining all dispatched/uncertain receipts.
    """
    intent = parse_text_entry(intent)
    outcomes, policy, history = [], None, []
    selected = None

    def result(status, reason, terminal=None):
        return {"status": status, "reason": reason, "outcomes": outcomes,
                "terminal": terminal, "text_entry_receipt": {
                    "schema": "robit.omni.native-text-entry.v1", "mode": "replace",
                    "strategy": selected.get("strategy") if selected else None,
                    "selection": policy, "history": history, "goal_verified": False,
                    "exact_value_verified": status == "complete"}}

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise ComputerUseError("Text entry cancelled")

    try:
        check_cancel()
        initial = controller.observe()
        bank, history = candidates(controller, initial, intent)
        policy = choose(copy.deepcopy(initial), copy.deepcopy(bank), copy.deepcopy(history))
        check_cancel()
        selected = next((item for item in bank if item["id"] == policy.get("choice")), None)
        if selected is None or selected["kind"] != "text_strategy":
            return result("abstained", "No applicable text strategy selected")
        original = next(node for node in initial["nodes"] if node["id"] == selected["node_id"])
        current = controller.observe()

        def target(state, value, *, focused=False, selection=None, input_guard=True):
            if (state["session_id"] != initial["session_id"] or state["scope"] != initial["scope"]
                    or any(state.get(key) != initial.get(key) for key in ("window_scope_sha256", "window_scope_count"))):
                return None
            node = next((node for node in state["nodes"] if node["id"] == original["id"]), None)
            if (node is None or not complete_text(node) or node.get("value") != value
                    or not node["visible"] or not node["enabled"] or node["sensitive"]
                    or any(node.get(key) != original.get(key) for key in ("generation", "scope_id", "role", "name"))
                    or input_guard and node.get("bounds") != original.get("bounds")
                    or focused and not node.get("focused")
                    or selection is not None and node.get("text_selection") != selection):
                return None
            return node

        if target(current, original["value"]) != original:
            return result("needs_replan", "Text target changed while choosing a strategy", current)
        expected = {"kind": "value", "node_id": original["id"], "value": intent["text"]}
        old_value = original["value"]

        def step(verb, args, expect, *, focused=False, selection=None):
            check_cancel()
            instruction = {"id": hashlib.sha256(intent["id"].encode()).hexdigest() + ":" + verb,
                           "verb": verb, "target": {"node_id": original["id"]},
                           "args": args, "expect": expect}

            def guarded(state, offers):
                node = target(state, old_value, focused=focused, selection=selection)
                legal = [item for item in offers if item["kind"] == "action" and item["node_id"] == original["id"]]
                return {"choice": legal[0]["id"] if node is not None and len(legal) == 1
                        else next(item["id"] for item in offers if item["kind"] == "abstain")}

            outcome = controller.run_step(instruction, guarded, cancel)
            outcomes.append(outcome)
            return outcome.get("status") == "verified"

        strategy = selected["strategy"]
        if strategy == "native_set":
            if not step("set_value", {"value": intent["text"]}, [expected]):
                return result("needs_replan", "Native write did not verify; reconcile before choosing a different strategy")
        elif strategy == "physical_replace":
            if not original.get("focused") and not step("focus", {}, [{"kind": "focused", "node_id": original["id"]}]):
                return result("needs_replan", "Text-entry focus prefix did not verify")
            selected_range = {"start": 0, "end": len(old_value)} if old_value else None
            if old_value and not step("select_text", selected_range,
                    [{"kind": "text_selection", "node_id": original["id"], **selected_range}], focused=True):
                return result("needs_replan", "Text-entry selection prefix did not verify")
            if not step("type_text", {"text": intent["text"]}, [expected], focused=True, selection=selected_range):
                return result("needs_replan", "Physical write did not verify; no input retry was attempted")
        check_cancel()
        terminal = controller.observe()
        check_cancel()
        if target(terminal, intent["text"], input_guard=False) is None or not predicate_satisfied(expected, terminal):
            return result("needs_replan", "Independent text-entry readback did not match", terminal)
        return result("complete", "Exact full value independently verified", terminal)
    except Exception as exc:
        return result("cancelled" if cancel is not None and cancel.is_set() else "needs_replan", str(exc)[:500])
