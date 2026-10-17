"""Host outcomes persist; planner assessment/commitment never confer authority."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator
from test_ui_task import ALLOWED, GOAL, Fixture, disappear_before_execution, final, phase

from portal.ui_task import _NativeWorkingState, _working_phase_schema, phase_response_schema


def envelope(packet, plan, *, decision=None, objective="Write and commit the owned draft", resource="owned draft"):
    state = packet["working_state"]
    if decision is None:
        decision = ("blocked" if plan["kind"] == "blocked" else "verify" if plan["kind"] == "complete" else
                    "start" if state["source_phase"] == 1 else "recover" if state["recovery_mode"] else "continue")
    return {"assessment": {"source_epoch": state["source_epoch"], "source_phase": state["source_phase"],
                            "decision": decision, "summary": "Assess the latest host receipt without claiming task completion."},
            "commitment": {**({} if "objective" in plan or plan["kind"] == "ui_subgoal" else {"objective": objective}),
                           "intended_resource": resource, "unmet_constraints": ["Verify actual delivery"]},
            "phase": copy.deepcopy(plan)}


def test_working_state_wraps_same_phase_and_preserves_executed_prefix():
    fixture = Fixture([])
    def compile_phase(packet):
        fixture.packets.append(copy.deepcopy(packet))
        return envelope(packet, phase() if len(fixture.packets) == 1 else {"kind": "complete", "message": "Claim only"})
    fixture.compile = compile_phase
    events = list(fixture.run(working_state=True))
    assert len(fixture.executions) == 1 and len(fixture.adapter.calls) == 2
    state = fixture.packets[1]["working_state"]
    assert state["commitment"]["authority"] == "planner_claim" and state["input_authority"] is False
    assert state["phase_refs"][0][0] == 1 and state["phase_refs"][0][2] == 2
    assert len(state["executed_facts"]) == 2
    assert final(events)["portal"]["native_task"]["task_verified"] is False


def test_legacy_direct_call_still_uses_unchanged_plain_phase():
    fixture = Fixture([phase(), {"kind": "complete", "message": "Claim only"}])
    events = list(fixture.run())
    assert all("working_state" not in packet for packet in fixture.packets)
    assert len(fixture.adapter.calls) == 2
    assert "working_state" not in final(events)["portal"]["native_task"]


def working_fixture(planner, **kwargs):
    fixture = Fixture([], **kwargs)
    def compile_phase(packet):
        fixture.packets.append(copy.deepcopy(packet))
        return planner(packet, len(fixture.packets))
    fixture.compile = compile_phase
    return fixture


def test_first_complete_can_verify_current_observation_without_fabricating_an_input():
    fixture = working_fixture(lambda packet, _: envelope(packet, {"kind": "complete", "message": "Claim only"}))
    events = list(fixture.run(working_state=True))
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"
    assert not fixture.executions and not fixture.adapter.calls
    assert final(events)["portal"]["native_task"]["task_verified"] is False


@pytest.mark.parametrize("fault", ["epoch", "phase", "decision", "host_fields", "empty_summary", "empty_objective", "extra"])
def test_invalid_or_stale_assessment_cannot_dispatch_or_reset_failure(fault):
    def planner(packet, number):
        reply = envelope(packet, phase())
        if fault == "epoch":
            reply["assessment"]["source_epoch"] = "other-task"
        elif fault == "phase":
            reply["assessment"]["source_phase"] = 0
        elif fault == "decision":
            reply["assessment"]["decision"] = "continue"
        elif fault == "host_fields":
            reply["commitment"]["recovery_mode"] = False
        elif fault == "empty_summary":
            reply["assessment"]["summary"] = " \t"
        elif fault == "empty_objective":
            reply["commitment"]["objective"] = " "
        else:
            reply["authorization"] = True
        return reply
    fixture = working_fixture(planner)
    events = list(fixture.run(working_state=True))
    assert not fixture.executions and not fixture.adapter.calls
    assert final(events)["portal"]["native_task"]["phases"] == 3
    assert fixture.packets[2]["working_state"]["consecutive_unverified_count"] == 2
    assert fixture.packets[1]["working_state"]["latest_host_outcome"]["facts"]["input_attempted"] is False
    assert fixture.packets[1]["working_state"]["commitment"] is None


def test_unexecuted_suffix_is_not_evidence_and_renaming_commitment_does_not_reset_host_failure():
    fixture = disappear_before_execution(working_fixture(lambda packet, number:
        envelope(packet, phase(), objective="Changed claim " + str(number), resource="Changed resource " + str(number))), times=3)
    events = list(fixture.run(working_state=True))
    state = fixture.packets[2]["working_state"]
    assert state["recovery_mode"] is True and state["consecutive_unverified_count"] == 2
    assert state["phase_refs"][0][2] == 1
    assert all(fact["action"].get("node_id") != "commit" for fact in state["executed_facts"])
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert len(fixture.executions) == 3 and not fixture.adapter.calls


@pytest.mark.parametrize("decision", ["continue", "advance", "verify", "start"])
def test_previous_unverified_outcome_requires_recovery_not_claimed_advancement(decision):
    fixture = disappear_before_execution(working_fixture(lambda packet, number:
        envelope(packet, phase(), decision=None if number == 1 else decision)))
    events = list(fixture.run(working_state=True))
    assert len(fixture.executions) == 1 and not fixture.adapter.calls
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert fixture.packets[1]["working_state"]["last_unverified_host_outcome"]["source_phase"] == 1


def test_verified_prefix_and_failure_survive_twelve_row_eviction_and_useful_knowledge():
    first = phase()
    first["steps"][1]["expect"][0]["value"] = "Not observed by this controlled adapter"
    def planner(packet, number):
        if number == 1:
            return envelope(packet, first)
        if number == 15:
            return envelope(packet, {"kind": "blocked", "reason": "Do not claim task success"})
        return envelope(packet, {"kind": "memory_search", "query": "A general acquired constraint"})
    fixture = working_fixture(planner)
    events = list(fixture.run(working_state=True))
    state = fixture.packets[-1]["working_state"]
    assert len(fixture.packets[-1]["ledger"]) <= 12
    assert state["phase_refs"][0][0] == 1 and len(state["phase_refs"]) == 14
    assert state["executed_facts"][0]["source_phase"] == 1
    assert state["last_unverified_host_outcome"]["source_phase"] == 1
    assert state["latest_host_outcome"]["source_phase"] == 14
    assert all(fact["input_authority"] is False for fact in state["executed_facts"])
    assert final(events)["portal"]["native_task"]["task_verified"] is False


def test_cached_rejected_receipt_is_historical_not_new_obstruction_or_input():
    state = _NativeWorkingState(GOAL)
    item = {"status": "rejected", "action": {"node_id": "source", "generation": "1", "verb": "drag_drop", "args": {"to_node_id": "dest"}},
        "receipt": {"status": "rejected", "reason": "native_pointer_target_occluded"},
        "effect_receipt": {"effect_state": "absent", "environment_may_have_changed": False}}
    result = {"status": "needs_replan", "reason": "native_pointer_target_occluded", "outcomes": [item]}
    state.record(1, "ui", result, recover=True)
    state.record(2, "ui", copy.deepcopy(result), recover=True)
    facts = state.view(3, 2)["executed_facts"]
    assert facts[0]["receipt_seen_before"] is False and facts[1]["receipt_seen_before"] is True
    assert facts[1]["effect_state"] == "absent" and facts[1]["input_attempted_available"] is False
    assert facts[0]["outcome_sha256"] == facts[1]["outcome_sha256"]


def test_abstention_without_dispatch_record_retains_explicit_tri_state_host_facts():
    state = _NativeWorkingState(GOAL)
    result = {"status": "abstained", "reason": "intent_policy_abstained", "effect_receipt": {"effect_state": "absent"},
        "ui_boundary_receipt": {"input_attempted": False, "adapter_called": False, "cached": False, "observed": False}}
    state.record(1, "ui_subgoal", result, recover=True)
    facts = state.view(2, 1)["latest_host_outcome"]["facts"]
    assert facts["ui_boundary_receipt"]["input_attempted"] is False
    assert facts["ui_boundary_receipt"]["adapter_called"] is False
    assert facts["effect_state"] == "absent" and facts["status"] == "abstained"
    assert state.view(2, 1)["phase_refs"][0][2] == 0


@pytest.mark.parametrize("character", ["é", "😀", "\x00", "\\", "\n"])
def test_max_legal_unicode_claims_and_forty_refs_are_bounded_without_destroying_source(character):
    state = _NativeWorkingState(GOAL)
    reply = {"assessment": {"source_epoch": state.epoch, "source_phase": 1, "decision": "start", "summary": character * 500 + "x"},
        "commitment": {"objective": character * 499 + "x", "intended_resource": character * 512,
                       "unmet_constraints": [character * 239 + "x"] * 10}, "phase": phase()}
    reply["assessment"]["summary"] = character * 499 + "x"
    state.accept(reply)
    saved = copy.deepcopy(state.commitment)
    for number in range(1, 41):
        state.record(number, "invalid", {"status": "rejected", "error": "invalid_native_phase", "reason": "r" * 1000,
            "dispatch_status": "not_dispatched", "input_attempted": False}, recover=True)
    view = state.view(40, 2)
    assert len(json.dumps(view, ensure_ascii=False).encode()) <= 12288
    assert len(view["phase_refs"]) == 40 and state.commitment == saved
    if "unmet_constraints" not in view["commitment"]:
        omission = view["commitment"]["omitted_fields"]["unmet_constraints"]
        assert omission["sha256"] == state.digest(saved["unmet_constraints"])
        assert omission["status"] == "unavailable_in_bounded_view_not_absent"


def test_envelope_root_refs_canonical_defs_and_whole_schema_cap():
    state = _NativeWorkingState(GOAL).view(1, 0)
    canonical = phase_response_schema(ALLOWED)
    schema = _working_phase_schema(ALLOWED, canonical, state)
    for name, value in canonical["$defs"].items():
        assert schema["$defs"][name] == value
    Draft202012Validator(schema).validate(envelope({"working_state": state}, phase()))
    assert len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()) <= 57344
    assert hashlib.sha256(GOAL.encode()).hexdigest() == state["goal_sha256"]


@pytest.mark.parametrize("boundary", ["compile_cancel", "observe_cancel", "late_plan", "tool_start_cancel"])
def test_working_envelope_preserves_cancellation_and_deadline_before_input(boundary):
    now = [0]
    fixture = working_fixture(lambda packet, _: envelope(packet, phase()), clock=lambda: now[0])
    if boundary in {"compile_cancel", "late_plan"}:
        original = fixture.compile
        def compile_phase(packet):
            reply = original(packet)
            if boundary == "compile_cancel":
                fixture.cancelled.set()
            else:
                now[0] = 5
            return reply
        fixture.compile = compile_phase
    elif boundary == "observe_cancel":
        original = fixture.observe
        def observe():
            result = original()
            fixture.cancelled.set()
            return result
        fixture.observe = observe
    events = []
    for event in fixture.run(working_state=True, max_seconds=5):
        events.append(event)
        if boundary == "tool_start_cancel" and event.get("type") == "tool" and event.get("phase") == "start":
            fixture.cancelled.set()
    task = final(events)["portal"]["native_task"]
    assert task["status"] == ("deadline_exceeded" if boundary == "late_plan" else "cancelled")
    assert not fixture.executions and not fixture.adapter.calls
    assert task["working_state"]["phase_refs"] == []
    assert task["task_verified"] is False


def test_working_state_keeps_forty_phase_ceiling_and_does_not_credit_knowledge_as_mutation():
    fixture = working_fixture(lambda packet, _: envelope(packet, {"kind": "memory_search", "query": "A general constraint"}))
    events = list(fixture.run(working_state=True))
    task = final(events)["portal"]["native_task"]
    assert task["status"] == "phase_budget_exceeded" and task["phases"] == 40
    assert len(task["working_state"]["phase_refs"]) == 40
    assert all(ref[2] == 0 for ref in task["working_state"]["phase_refs"])
    assert task["working_state"]["executed_facts"] == []
    assert not fixture.adapter.calls and task["task_verified"] is False


@pytest.mark.parametrize("kind", ["ui_boundary", "ui_subgoal"])
def test_phase_owned_objective_rejects_duplicate_generation(kind):
    state = _NativeWorkingState(GOAL)
    plan = ({"kind": kind, "objective": "Actually requested objective"} if kind == "ui_boundary" else
            {"kind": kind, "subgoal": {"objective": "Actually requested objective"}})
    reply = envelope({"working_state": state.view(1, 0)}, plan, objective="A conflicting claim")
    reply["commitment"]["objective"] = "A conflicting claim"
    with pytest.raises(ValueError, match="objective"):
        state.accept(reply)
    assert state.commitment is None and state.assessment is None


@pytest.mark.parametrize("kind", ["ui_boundary", "ui_subgoal"])
def test_phase_objective_has_one_source_and_host_preserves_its_exact_provenance(kind):
    from test_grounded_first_step import node, observation

    from portal.ui_task import _grounded_phase_schema

    state = _NativeWorkingState(GOAL)
    current = observation(node(role="button", name="Input", operations=("click", "double_click")))
    schema = _working_phase_schema(ALLOWED, _grounded_phase_schema(ALLOWED, current, intent_choice=True), state.view(1, 0))
    objective = "Open and read the input document to learn its source and delivery requirements."
    plan = ({"kind": kind, "objective": objective, "step": {
        "id": "open-input", "verb": "click", "target": {"name": "Input"}, "args": {}}}
        if kind == "ui_boundary" else {"kind": kind, "subgoal": {
        "id": "open-input", "family": "pointer_or_activation", "objective": objective,
        "target": {"node_id": "current"}}})
    reply = envelope({"working_state": state.view(1, 0)}, plan)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(reply), list(validator.iter_errors(reply))
    original = copy.deepcopy(reply)
    state.accept(reply)
    assert reply == original
    assert state.commitment["objective"] == objective
    assert state.commitment["objective_source"] == {
        "path": "/objective" if kind == "ui_boundary" else "/subgoal/objective",
        "source_phase": 1, "phase_sha256": state.digest(plan)}
    assert state.commitment["input_authority"] is False
    for duplicate in (objective, "A differently worded objective"):
        invalid = copy.deepcopy(reply)
        invalid["commitment"]["objective"] = duplicate
        assert not validator.is_valid(invalid)


def test_shadow_boundary_retains_its_one_explicit_commitment_objective():
    from test_grounded_first_step import node, observation

    from portal.ui_task import _grounded_phase_schema

    state = _NativeWorkingState(GOAL)
    current = observation(node(role="button", name="Input", operations=("click",)))
    schema = _working_phase_schema(ALLOWED, _grounded_phase_schema(ALLOWED, current), state.view(1, 0))
    plan = {"kind": "ui_boundary", "step": {
        "id": "open-input", "verb": "click", "target": {"name": "Input"}, "args": {}}}
    reply = envelope({"working_state": state.view(1, 0)}, plan)
    assert Draft202012Validator(schema).is_valid(reply)
    state.accept(reply)
    assert state.commitment["objective"] == reply["commitment"]["objective"]
    del reply["commitment"]["objective"]
    assert not Draft202012Validator(schema).is_valid(reply)


@pytest.mark.parametrize("objective,resource", [
    ("Compare observed policy versions", "authoritative browser document"),
    ("Organize the permitted originals", "selected Files item"),
    ("Revise the draft and verify delivery", "editable document"),
    ("Recover after the last unresolved action", "resource not yet established"),
])
def test_generic_claims_do_not_fabricate_an_observed_resource_or_effect(objective, resource):
    state = _NativeWorkingState(GOAL)
    state.accept(envelope({"working_state": state.view(1, 0)}, phase(), objective=objective, resource=resource))
    view = state.view(1, 0)
    assert view["commitment"]["intended_resource"] == resource
    assert view["executed_facts"] == [] and view["latest_host_outcome"] is None
    assert view["input_authority"] is False and view["commitment"]["input_authority"] is False


def test_whole_envelope_overcap_withholds_ui_without_shortening_canonical_steps():
    canonical = phase_response_schema(ALLOWED)
    crowded = copy.deepcopy(canonical)
    crowded["oneOf"][0]["description"] = "x" * 57344
    state = _NativeWorkingState(GOAL).view(1, 0)
    schema = _working_phase_schema(ALLOWED, crowded, state)
    kinds = {branch["properties"]["kind"]["const"] for branch in schema["$defs"]["native_working_phase"]["oneOf"]}
    assert "ui" not in kinds and {"memory_search", "memory_write", "visual_inspection"} <= kinds
    assert schema["$defs"]["step"] == canonical["$defs"]["step"]
    assert canonical["oneOf"][0]["properties"]["steps"]["maxItems"] == 25
    Draft202012Validator(schema).validate(envelope({"working_state": state}, {"kind": "memory_search", "query": "evidence"}))
