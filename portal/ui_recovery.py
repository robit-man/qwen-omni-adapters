"""Bounded native strategy history derived only from typed executor evidence.

This module performs no input, model call, I/O or authorization. Operation
identity omits planner wording, step IDs and whole-window revision churn.
Relevant state can justify new reasoning after a proved unchanged action; it
cannot clear uncertain input. Capacity exhaustion retains all existing locks.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any

from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    parse_predicates,
    parse_step,
    predicate_satisfied,
)
from qwen_omni_adapters.ui_effects import compile_immediate_transition

_TARGET_FIELDS = ("id", "generation", "scope_id", "role", "name", "visible", "enabled", "sensitive",
                  "operations", "bounds", "value", "focused", "checked", "selected", "text_selection",
                  "text_observation", "text_insertion", "structure")
_SCOPE_FIELDS = ("kind", "id", "window_id", "process_id")


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ComputerUseError("Strategy evidence is not bounded JSON") from exc
    if len(encoded) > 1024 * 1024:
        raise ComputerUseError("Strategy evidence exceeds its byte bound")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 512


def _bound_operation(state: Mapping[str, Any], action: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    if (not isinstance(state, Mapping) or not _text(state.get("session_id"))
            or not isinstance(state.get("scope"), dict) or not _text(state["scope"].get("id"))
            or not isinstance(state.get("nodes"), list) or len(state["nodes"]) > 600
            or not isinstance(action, Mapping)):
        raise ComputerUseError("Strategy requires a current native observation and typed operation")
    target = action.get("target")
    if "node_id" in action:
        if target is not None and target != {"node_id": action["node_id"]}:
            raise ComputerUseError("Strategy target identities disagree")
        target = {"node_id": action["node_id"]}
    parsed = parse_step({"id": "strategy-binding", "verb": action.get("verb"), "target": target,
        "args": action.get("args", {}), "expect": [{"kind": "focused", "node_id": "$target"}]})
    nodes = state["nodes"]
    if any(not isinstance(node, dict) or not _text(node.get("id")) for node in nodes):
        raise ComputerUseError("Strategy observation contains invalid identities")
    if len({node["id"] for node in nodes}) != len(nodes):
        raise ComputerUseError("Strategy observation contains ambiguous identities")
    matching = [node for node in nodes if all(node.get("id" if key == "node_id" else key) == value
                for key, value in parsed["target"].items())]
    if len(matching) != 1:
        raise ComputerUseError("Strategy requires one actual observed target")
    node = matching[0]
    if (any(not _text(node.get(key)) for key in ("generation", "scope_id", "role"))
            or any(type(node.get(key)) is not bool for key in ("visible", "enabled", "sensitive"))
            or node["sensitive"] or not isinstance(node.get("operations"), list)
            or any(not isinstance(verb, str) for verb in node["operations"])):
        raise ComputerUseError("Strategy target is invalid or protected")
    for key, expected in (("generation", node["generation"]), ("session_id", state["session_id"]),
                          ("scope_id", state["scope"]["id"])):
        if key in action and action[key] != expected:
            raise ComputerUseError("Strategy action source binding changed")
    scope = {key: copy.deepcopy(state["scope"][key]) for key in _SCOPE_FIELDS if key in state["scope"]}
    operation = {"session_id": state["session_id"], "scope": scope, "node_id": node["id"],
        "generation": node["generation"], "target_scope_id": node["scope_id"],
        "verb": parsed["verb"], "args": parsed["args"]}
    relevant = {"target": {key: copy.deepcopy(node[key]) for key in _TARGET_FIELDS if key in node},
                "resource": state["scope"].get("url")}
    relevant["target"]["operations"] = sorted(set(node["operations"]))
    if parsed["verb"] == "drag_drop":
        drop = [item for item in nodes if item["id"] == parsed["args"]["to_node_id"]]
        if len(drop) != 1 or drop[0].get("sensitive") is not False:
            raise ComputerUseError("Strategy drop target is unavailable")
        relevant["destination"] = {key: copy.deepcopy(drop[0][key]) for key in _TARGET_FIELDS if key in drop[0]}
    return operation, relevant, node


def strategy_identity(state: Mapping[str, Any], action: Mapping[str, Any]) -> dict:
    """Return advisory identities; none are executable candidate identifiers."""
    operation, relevant, node = _bound_operation(state, action)
    operation_id, relevant_id = _digest(operation), _digest(relevant)
    return {"operation_id": operation_id, "strategy_id": _digest([operation_id, relevant_id]),
        "relevant_state_sha256": relevant_id, "target": {"node_id": node["id"],
            "generation": node["generation"], "scope_id": node["scope_id"]},
        "verb": operation["verb"], "args_sha256": _digest(operation["args"]),
        "input_authority": False}


def _bound_effect(record: dict, effect: dict) -> bool:
    action, before, after = (record.get(key, {}) for key in ("action", "before", "after"))
    if any(not isinstance(value, dict) for value in (action, before, after, effect)):
        return False
    if (effect.get("schema") != "robit.omni.ui-effect-receipt.v1"
            or not isinstance(before.get("scope"), dict) or not isinstance(after.get("scope"), dict)
            or not _text(action.get("id")) or not _text(action.get("idempotency_key"))
            or effect.get("action_id") != action["id"] or effect.get("idempotency_key") != action["idempotency_key"]
            or not _text(before.get("revision")) or not _text(after.get("revision"))
            or action.get("revision") != before["revision"]
            or any(effect.get(key) != value for key, value in (
                ("before_revision", before["revision"]), ("after_revision", after["revision"]),
                ("before_semantic_revision", before.get("base_semantic_revision", before["revision"])),
                ("after_semantic_revision", after.get("base_semantic_revision", after["revision"]))))):
        return False
    try:
        predicates = parse_predicates(action.get("expect"))
        scope_matches = (before.get("scope", {}).get("id") == after.get("scope", {}).get("id")
                         or action.get("verb") == "switch_window"
                         and after.get("scope", {}).get("id") == action.get("node_id"))
        return (effect.get("predicates_sha256") == _digest(predicates)
                and before.get("session_id") == after.get("session_id")
                and scope_matches
                and all(predicate_satisfied(predicate, after) for predicate in predicates))
    except (ComputerUseError, TypeError, ValueError):
        return False


def _classification(record: dict, outer: dict) -> str:
    dispatch = record.get("receipt", {})
    dispatch = dispatch if isinstance(dispatch, dict) else {}
    boundary = outer.get("ui_boundary_receipt", {})
    boundary = boundary if isinstance(boundary, dict) else {}
    effect = record.get("effect_receipt", outer.get("effect_receipt", {}))
    effect = effect if isinstance(effect, dict) else {}
    status = record.get("status", outer.get("status"))
    raw_action = record.get("action")
    selection_abstained = (status == "abstained" and isinstance(raw_action, dict) and raw_action.get("kind") == "abstain"
                           and not record.get("dispatch_attempted") and not dispatch)
    definitely_no_input = (dispatch.get("input_attempted") is False
        or boundary.get("input_attempted") is False and boundary.get("adapter_called") is False)
    explicit_rejection_without_input = (dispatch.get("input_attempted") is False
                                       and dispatch.get("status") in {"rejected", "already_satisfied"})
    attempted = (dispatch.get("status") == "dispatched" and dispatch.get("input_attempted") is not False
                 or record.get("dispatch_attempted") is True and not explicit_rejection_without_input
                 or dispatch.get("input_attempted") is True or boundary.get("input_attempted") is True
                 or effect.get("environment_may_have_changed") is True)
    if selection_abstained:
        return "policy_deferred_no_input"
    if definitely_no_input and not attempted:
        if dispatch.get("status") == "already_satisfied" and status == "verified":
            return "already_satisfied_no_input"
        if status == "abstained" and isinstance(outer.get("intent_selection_receipt"), dict):
            return "policy_deferred_no_input"
        return "rejected_no_input"
    if attempted:
        if _observed_transition(record, outer):
            return "observed_transition"
        if status == "verified" and _bound_effect(record, effect):
            if effect.get("effect_state") == "unchanged" and (
                    record["before"].get("base_semantic_revision", record["before"]["revision"])
                    == record["after"].get("base_semantic_revision", record["after"]["revision"])):
                return "proven_no_effect"
            if effect.get("effect_state") == "applied" and (
                    record["before"].get("base_semantic_revision", record["before"]["revision"])
                    != record["after"].get("base_semantic_revision", record["after"]["revision"])):
                return "applied"
        return "input_effect_uncertain"
    return "informational_no_bound_input"


def _observed_transition(record: dict, outer: dict) -> bool:
    """Recompute a bound UI transition; reported comparison status is no proof."""
    compiled, receipt = outer.get("immediate_effect_contract"), outer.get("immediate_transition_receipt")
    before, after, action = (record.get(key) for key in ("before", "after", "action"))
    if (any(not isinstance(value, dict) for value in (compiled, receipt, before, after, action))
            or record.get("status") != "observed_unverified" or record.get("dispatch_attempted") is not True
            or record.get("receipt", {}).get("status") != "dispatched"
            or receipt.get("schema") != "robit.omni.native-immediate-transition.v1"
            or receipt.get("compiler") != "host_operation_semantics"
            or receipt.get("input_authority") is not False or receipt.get("task_progress") is not False
            or receipt.get("goal_verified") is not False or compiled.get("mode") != "observation_boundary"
            or not isinstance(compiled.get("transition"), dict)
            or any(not isinstance(value.get("scope"), dict) or value["scope"].get("provider") != "atspi2"
                   for value in (before, after))):
        return False
    original = receipt.get("original_source_ref")
    if original is not None:
        handoff = outer.get("action_handoff", {})
        source = handoff.get("source_observation") if isinstance(handoff, dict) else None
        if (not isinstance(original, dict) or not isinstance(source, dict)
                or set(original) != {"session_id", "scope_id", "revision"}
                or any(not _text(original.get(key)) for key in ("session_id", "scope_id", "revision"))
                or original.get("revision") != source.get("revision")
                or original.get("scope_id") != source.get("scope_id")
                or _digest(original.get("session_id")) != source.get("session_sha256")
                or source.get("scope_sha256") != _digest(before["scope"])
                or original.get("session_id") != before.get("session_id")
                or original.get("scope_id") != before["scope"].get("id")):
            return False
    try:
        _bound_operation(before, action)
        step = {"id": action["id"], "verb": action["verb"],
                "target": {"node_id": action["node_id"]}, "args": action["args"]}
        contract = compile_immediate_transition(step, before, transition=compiled["transition"])
        comparison = contract.compare(before, after)
        return (contract.as_dict() == compiled and comparison.get("status") == "observed"
                and receipt.get("comparison") == comparison
                and transition_related_to_target(compiled["transition"]["kind"], action, before, after))
    except (ComputerUseError, TypeError, ValueError, KeyError):
        return False


def transition_related_to_target(kind: str, action: dict, before: dict, after: dict) -> bool:
    """A scope change alone cannot settle this particular control's input.

    Only positive current provider relationships qualify. The generic descriptor
    has no expected destination ref, so unrelated focus/resource transitions and
    missing, clipped or inconsistent ancestry remain uncertain.
    """
    if (any(not isinstance(value, dict) for value in (action, before, after))
            or any(not isinstance(value.get("nodes"), list)
                   or any(not isinstance(node, dict) for node in value["nodes"]) for value in (before, after))):
        return False
    old = [node for node in before["nodes"] if node.get("id") == action.get("node_id")]
    new = [node for node in after["nodes"] if node.get("id") == action.get("node_id")]
    if (len(old) != 1 or len(new) != 1
            or old[0].get("generation") != action.get("generation")
            or new[0].get("generation") != action.get("generation")
            or old[0].get("scope_id") != new[0].get("scope_id")):
        return False
    target = new[0]
    if target.get("sensitive") is not False:
        return False
    if kind == "focus_changed_within_scope":
        return (target.get("role") != "window" and old[0].get("focused") is False
                and target.get("focused") is True and target.get("visible") is True)
    if kind == "document_changed":
        documents = [state.get("document_evidence", {}) for state in (before, after)]
        return (target.get("role") == "document_web" and all(
            document.get("node_id") == action.get("node_id")
            and document.get("generation") == action.get("generation") for document in documents))
    if kind != "popup_opened":
        return False
    previously_visible = {(node.get("id"), node.get("generation")) for node in before.get("nodes", [])
                          if node.get("role") in {"menu", "popup_menu"} and node.get("visible") is True}
    for popup in after.get("nodes", []):
        if (popup.get("role") not in {"menu", "popup_menu"} or popup.get("visible") is not True
                or popup.get("sensitive") is not False or popup.get("scope_id") != target.get("scope_id")
                or (popup.get("id"), popup.get("generation")) in previously_visible):
            continue
        structure = popup.get("structure", {})
        ancestors = structure.get("ancestors") if isinstance(structure, dict) else None
        ancestor_fields = {"id", "generation", "role", "name", "visible", "enabled"}
        if (not isinstance(ancestors, list) or not 1 <= len(ancestors) <= 3
                or any(not isinstance(ancestor, dict) or set(ancestor) != ancestor_fields
                       or any(not _text(ancestor.get(key)) for key in ("id", "generation", "role"))
                       or not isinstance(ancestor.get("name"), str) or len(ancestor["name"]) > 512
                       or any(type(ancestor.get(key)) is not bool for key in ("visible", "enabled"))
                       for ancestor in ancestors)
                or len({ancestor["id"] for ancestor in ancestors}) != len(ancestors)
                or popup.get("id") in {ancestor["id"] for ancestor in ancestors}
                or structure.get("parent_id") != ancestors[-1].get("id")):
            continue
        if any(all(ancestor.get(key) == target.get(key) for key in ancestor_fields)
               for ancestor in ancestors):
            return True
    return False


class NativeStrategyLedger:
    """Task-owned bounded facts, with sticky uncertain-operation locks."""

    def __init__(self, max_entries: int = 128) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 256:
            raise ComputerUseError("Expected a bounded native strategy ledger capacity")
        self.max_entries = max_entries
        self._rows: dict[str, dict] = {}
        self._uncertain: dict[str, str] = {}
        self._operations: dict[str, dict] = {}
        self._informational: list[dict] = []
        self.capacity_exhausted = False

    def record(self, result: Mapping[str, Any], *, source_phase: int | None = None) -> list[dict]:
        if not isinstance(result, Mapping) or (source_phase is not None
                and (type(source_phase) is not int or not 1 <= source_phase <= 40)):
            raise ComputerUseError("Invalid strategy result or phase")
        outer = dict(result)
        records = outer.get("outcomes")
        if not isinstance(records, list):
            records = [outer["dispatch_record"]] if isinstance(outer.get("dispatch_record"), dict) else [outer]
        if len(records) > 25:
            raise ComputerUseError("Strategy result exceeds the native phase bound")
        observed = []
        for record in records:
            if not isinstance(record, dict):
                raise ComputerUseError("Invalid strategy outcome")
            classification = _classification(record, outer)
            fact = {"classification": classification, "source_phase": source_phase,
                "result_sha256": _digest(record), "input_authority": False, "task_progress": False,
                "repeated_receipt_count": 0, "same_strategy_result_count": 1,
                "claim_limit": "observed_ui_history_not_external_effect_proof"}
            try:
                identity = strategy_identity(record.get("before"), record.get("action"))
                operation, _relevant, _node = _bound_operation(record["before"], record["action"])
            except (ComputerUseError, TypeError, ValueError):
                # Unbound policy/inspection results are useful facts, not locks.
                fact["bound_operation"] = False
                self._informational.append(fact)
                self._informational = self._informational[-8:]
                if classification == "input_effect_uncertain":
                    self.capacity_exhausted = True  # Unknown input cannot be safely forgotten.
                observed.append(copy.deepcopy(fact))
                continue
            fact.update(identity, bound_operation=True)
            row_key = _digest([identity["strategy_id"], classification])
            if row_key in self._rows:
                previous = self._rows[row_key]
                previous["same_strategy_result_count"] += 1
                if fact["result_sha256"] in {previous["result_sha256"], previous.get("latest_result_sha256")}:
                    previous["repeated_receipt_count"] += 1
                previous["latest_result_sha256"] = fact["result_sha256"]
                previous["last_source_phase"] = source_phase
                observed.append(copy.deepcopy(previous))
                continue
            if len(self._rows) >= self.max_entries:
                self.capacity_exhausted = True
                observed.append({**fact, "retained": False, "reason": "strategy_capacity_exhausted"})
                continue
            self._rows[row_key] = fact
            self._operations[identity["operation_id"]] = operation
            if classification == "input_effect_uncertain":
                self._uncertain[identity["operation_id"]] = row_key
            observed.append(copy.deepcopy(fact))
        return observed

    def assess(self, state: Mapping[str, Any], action: Mapping[str, Any]) -> dict:
        identity = strategy_identity(state, action)
        rows = [row for row in self._rows.values() if row["operation_id"] == identity["operation_id"]]
        uncertain = self._uncertain.get(identity["operation_id"])
        same_failed = [row for row in rows if row["classification"] == "proven_no_effect"
                       and row["relevant_state_sha256"] == identity["relevant_state_sha256"]]
        relevant_changed = bool(rows and all(row["relevant_state_sha256"] != identity["relevant_state_sha256"] for row in rows))
        classification = (self._rows[uncertain]["classification"] if uncertain else "proven_no_effect"
                          if same_failed else rows[-1]["classification"] if rows else "not_attempted")
        blocked = self.capacity_exhausted or bool(uncertain or same_failed)
        reason = ("uncertain_input_requires_reconciliation" if uncertain else "strategy_capacity_exhausted"
                  if self.capacity_exhausted else "same_strategy_without_new_effect" if same_failed else
                  "relevant_state_changed_replan" if relevant_changed else "not_blocked_by_history")
        evidence = ([{"kind": "independent_effect_reconciliation", "operation_id": identity["operation_id"]}]
                    if uncertain else [{"kind": "different_strategy_or_evidence", "operation_id": identity["operation_id"]}]
                    if same_failed else [])
        return {**identity, "classification": classification, "blocked": blocked,
            "repeat_status": reason, "relevant_state_changed": relevant_changed,
            "evidence_needed": evidence, "task_progress": False}

    def view(self, state: Mapping[str, Any] | None = None, candidates: Iterable[Mapping[str, Any]] = ()) -> dict:
        facts = [*self._rows.values(), *self._informational]
        value = {"schema": "robit.omni.native-strategy-recovery.v1", "input_authority": False,
            "task_progress": False, "fact_authority": "historical_executor_evidence",
            "capacity_exhausted": self.capacity_exhausted, "retained_entry_count": len(self._rows),
            "omitted_fact_count": max(0, len(facts) - 12), "facts": copy.deepcopy(facts[-12:]),
            "allowed_alternatives": [], "blocked_strategies": [], "evidence_needed": [],
            "omitted_alternative_count": 0, "omitted_blocked_count": 0, "omitted_evidence_count": 0}
        supplied = list(islice(candidates, 51))
        if len(supplied) > 50:
            raise ComputerUseError("Strategy alternatives exceed the current candidate bound")
        if state is not None:
            for candidate in supplied:
                try:
                    assessment = self.assess(state, candidate)
                    operation, _relevant, node = _bound_operation(state, candidate)
                    if (node.get("visible") is not True or node.get("enabled") is not True
                            or operation["verb"] not in node["operations"]):
                        continue
                except (ComputerUseError, TypeError, ValueError):
                    continue
                if assessment["blocked"]:
                    value["blocked_strategies"].append({key: assessment[key] for key in
                        ("operation_id", "strategy_id", "repeat_status", "classification")})
                    value["evidence_needed"].extend(assessment["evidence_needed"])
                else:
                    value["allowed_alternatives"].append({"alternative_ref": f"alternative_{len(value['allowed_alternatives'])}",
                        "operation_id": assessment["operation_id"], "verb": operation["verb"],
                        "target": assessment["target"], "args_sha256": assessment["args_sha256"],
                        "input_authority": False})
        for group, counter in (("facts", "omitted_fact_count"), ("allowed_alternatives", "omitted_alternative_count"),
                               ("blocked_strategies", "omitted_blocked_count"), ("evidence_needed", "omitted_evidence_count")):
            while len(json.dumps(value, ensure_ascii=False).encode()) > 12288 and value[group]:
                value[group].pop(0)
                value[counter] += 1
        if len(json.dumps(value, ensure_ascii=False).encode()) > 12288:
            raise ComputerUseError("Strategy view exceeds its planner-context bound")
        return value

    def reconcile(self, operation_id: str, evidence: Mapping[str, Any], *, predicates: list[dict]) -> dict:
        """Resolve only independently observed, host-derived idempotent effects."""
        row_key = self._uncertain.get(operation_id)
        if row_key is None or not isinstance(evidence, Mapping):
            raise ComputerUseError("No uncertain operation has bound reconciliation evidence")
        operation = self._operations[operation_id]
        verb, args, node_id = operation["verb"], operation["args"], operation["node_id"]
        if verb in {"focus", "switch_window"}:
            expected = [{"kind": "focused" if verb == "focus" else "active_window", "node_id": node_id}]
        elif verb in {"set_value", "set_checked", "select_text"}:
            expected = [{"kind": {"set_value": "value", "set_checked": "checked", "select_text": "text_selection"}[verb],
                         "node_id": node_id, **args}]
        else:
            raise ComputerUseError("Uncertain non-idempotent input requires task-specific reconciliation")
        state, receipt = evidence.get("state"), evidence.get("verification_receipt")
        if (not isinstance(state, dict) or not isinstance(receipt, dict) or predicates != expected
                or not isinstance(state.get("scope"), dict) or not _text(state.get("revision"))
                or evidence.get("status") != "verified" or receipt.get("schema") != "robit.omni.ui-verification.v1"
                or receipt.get("status") != "verified" or not _text(receipt.get("id"))
                or evidence.get("content_sha256") != _digest(state)
                or receipt.get("predicates_sha256") != _digest(expected)
                or receipt.get("revision") != state.get("revision")
                or receipt.get("session_id") != _digest(operation["session_id"])
                or state.get("session_id") != operation["session_id"]
                or receipt.get("scope_id") != operation["scope"]["id"]
                or any(state["scope"].get(key) != operation["scope"].get(key) for key in _SCOPE_FIELDS)
                or not all(predicate_satisfied(predicate, state) for predicate in expected)):
            raise ComputerUseError("Independent effect reconciliation did not verify")
        _operation, _relevant, node = _bound_operation(state, {"node_id": node_id, "generation": operation["generation"],
            "verb": verb, "args": args})
        if node["generation"] != operation["generation"]:
            raise ComputerUseError("Independent effect reconciliation target changed")
        self._rows[row_key]["classification"] = "effect_independently_observed"
        self._rows[row_key]["reconciliation_sha256"] = _digest(dict(evidence))
        del self._uncertain[operation_id]
        return {"status": "effect_independently_observed", "operation_id": operation_id,
                "input_authority": False, "task_progress": False}
