"""CPU acceptance of mandatory phase compilation and host execution boundaries."""
from __future__ import annotations

import copy
import json
import threading

import pytest
from jsonschema import Draft202012Validator

from portal.computer_use import ComputerUseStore, _digest
from portal.tools import tool_schemas
from portal.ui_task import (
    NativePhaseCompilationError,
    _new_verified_ui_prefix,
    _zero_effect_ui_phase_key,
    phase_response_schema,
    run_native_goal,
)

GOAL = "Write the exact owned draft, then commit it."
ALLOWED = frozenset({"native_ui", "memory_write", "memory_search"})


def phase():
    return {"kind": "ui", "steps": [
        {"id": "write", "verb": "set_value", "target": {"role": "textbox", "name": "Draft"},
         "args": {"value": "Exact draft"}, "expect": [{"kind": "value", "node_id": "$target", "value": "Exact draft"}]},
        {"id": "commit", "verb": "invoke", "target": {"role": "button", "name": "Commit"},
         "expect": [{"kind": "value", "node_id": "receipt", "value": "Exact draft"}]}],
        "final_predicates": [{"kind": "value", "node_id": "receipt", "value": "Exact draft"}],
        "remaining_criteria": ["Independent whole-task verification"]}


class Adapter:
    def __init__(self, ambiguous=False):
        self.value = self.saved = ""
        self.calls, self.timeline = [], []
        self.ambiguous = ambiguous

    def observe(self):
        self.timeline.append("observe")
        nodes = [{"id": "draft", "role": "textbox", "name": "Draft", "value": self.value, "operations": ["set_value"]},
                 {"id": "commit", "role": "button", "name": "Commit", "operations": ["invoke"]},
                 {"id": "receipt", "role": "label", "name": "Receipt", "value": self.saved, "operations": []}]
        if self.ambiguous:
            nodes.append({"id": "other", "role": "textbox", "name": "Draft", "value": "", "operations": ["set_value"]})
        return {"session_id": "session", "scope": {"id": "form", "kind": "desktop"},
                "revision": self.value + ":" + self.saved, "nodes": [
                    {"generation": "1", "scope_id": "form", "visible": True, "enabled": True,
                     "sensitive": False, **node} for node in nodes]}

    def perform(self, action):
        self.timeline.append(action["verb"])
        self.calls.append(copy.deepcopy(action))
        if action["verb"] == "set_value":
            assert action["node_id"] == "draft"
            self.value = action["args"]["value"]
        else:
            self.saved = self.value
        return {"status": "dispatched"}


class Fixture:
    def __init__(self, plans, *, ambiguous=False, allowed=ALLOWED, clock=lambda: 0):
        self.plans = iter(plans)
        self.adapter = Adapter(ambiguous)
        self.store = ComputerUseStore(None, adapter_factory=lambda *_: self.adapter)
        self.packets, self.executions, self.policy_calls = [], [], []
        self.cancel_count = 0
        self.allowed, self.clock = allowed, clock
        self.cancelled = threading.Event()

    def observe(self):
        return self.store.act("session", {"operation": "observe", "surface": "desktop"})

    def compile(self, packet):
        self.adapter.timeline.append("compile")
        self.packets.append(copy.deepcopy(packet))
        return next(self.plans)

    def execute(self, name, arguments):
        self.executions.append((name, copy.deepcopy(arguments)))
        if name == "native_ui":
            return self.store.act("session", {**arguments, "surface": "desktop"}, goal=GOAL)
        return {"stored": True, "scope": "browser_session"} if name == "memory_write" else {"query": arguments["query"], "results": []}

    def cancel(self):
        self.cancel_count += 1
        self.cancelled.set()
        self.store.act("session", {"operation": "cancel", "surface": "desktop"})

    def run(self, **kwargs):
        return run_native_goal(GOAL, observe=self.observe, compile_phase=self.compile, execute=self.execute,
                               cancel=self.cancel, cancelled=self.cancelled.is_set, allowed_tools=self.allowed,
                               clock=self.clock, model="owned-model", **kwargs)


def disappear_before_execution(fixture, *, times=1):
    """A positive planner target genuinely disappears before fresh execution."""
    original_execute = fixture.execute
    remaining = times

    def execute(name, arguments):
        nonlocal remaining
        if name != "native_ui" or remaining == 0:
            return original_execute(name, arguments)
        remaining -= 1
        original_observe = fixture.adapter.observe

        def missing():
            state = original_observe()
            state["nodes"] = [node for node in state["nodes"] if node["id"] != "draft"]
            return state

        fixture.adapter.observe = missing
        try:
            return original_execute(name, arguments)
        finally:
            fixture.adapter.observe = original_observe

    fixture.execute = execute
    return fixture


def final(events):
    return events[-1]["response"]


def test_strict_phase_schema_keeps_canonical_defs_and_excludes_forbidden_capabilities():
    canonical = copy.deepcopy(tool_schemas(["native_ui", "memory_write", "memory_search"]))
    schema = phase_response_schema({"native_ui"})
    assert schema["$defs"] == canonical[0]["function"]["parameters"]["$defs"]
    Draft202012Validator(schema).validate(phase())
    for value in ({"kind": "execute", "action_id": "opaque"}, {"kind": "memory_write", "topic": "t", "key": "k", "value": "v"},
                  {"kind": "ui", **{k: v for k, v in phase().items() if k != "kind"}, "authorization": True}):
        assert not Draft202012Validator(schema).is_valid(value)
    schema["$defs"].clear()
    assert tool_schemas(["native_ui", "memory_write", "memory_search"]) == canonical


def test_actual_controller_runs_two_inputs_between_compiler_calls_and_delegates_ambiguity():
    fixture = Fixture([phase(), {"kind": "complete", "message": "Planner reports the phase complete."}], ambiguous=True)

    class Policy:
        def decide(self, state, actions, *, context):
            fixture.policy_calls.append(copy.deepcopy((state, actions, context)))
            return {"choice": next(action["id"] for action in actions if action.get("node_id") == "draft"),
                    "receipt": {"status": "accepted", "model": "fast-policy"}}

    fixture.store.provider = Policy()
    events = list(fixture.run())
    assert [action["verb"] for action in fixture.adapter.calls] == ["set_value", "invoke"]
    compiles = [i for i, item in enumerate(fixture.adapter.timeline) if item == "compile"]
    assert len(compiles) == 2
    assert fixture.adapter.timeline[compiles[0] + 1:compiles[1]].count("set_value") == 1
    assert fixture.adapter.timeline[compiles[0] + 1:compiles[1]].count("invoke") == 1
    assert len(fixture.policy_calls) == 1
    assert fixture.policy_calls[0][2]["goal"] == GOAL
    assert all(packet["goal"] == GOAL for packet in fixture.packets)
    assert "state" in fixture.packets[0]["current_state"]
    second = fixture.packets[1]
    assert second["last_result"]["phase_handoff"]["new_effect_count"] == 2
    assert "before" not in second["last_result"]["outcomes"][0]
    assert "after" not in second["last_result"]["outcomes"][0]
    effects = second["last_result"]["outcomes"][1]["observed_effects"]
    assert next(node for node in effects if node["id"] == "receipt")["value"] == "Exact draft"
    assert "outcomes" not in second["ledger"][0]["result"]
    result = final(events)
    assert result["portal"]["native_task"]["status"] == "planner_claim"
    assert result["portal"]["native_task"]["task_verified"] is False
    raw = result["portal"]["safe_tools_executed"][0]["result"]
    assert '"before"' in raw and '"after"' in raw  # Full execution evidence survives projection.
    assert fixture.cancel_count == 0


def test_first_unverified_phase_returns_to_planner_with_actual_failed_step_and_fresh_state():
    bad = phase()
    fixture = disappear_before_execution(Fixture([bad, phase(), {"kind": "complete", "message": "Claim."}]))
    events = list(fixture.run())
    assert fixture.packets[1]["last_result"]["phase_handoff"]["remaining_steps"][0] == {"index": 0, "id": "write"}
    assert fixture.packets[1]["last_result"]["status"] == "needs_replan"
    assert len(fixture.adapter.calls) == 2
    assert len(fixture.executions) == 2
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"


@pytest.mark.parametrize("bad", [
    {"kind": "ui", "steps": [], "final_predicates": []},
    {"kind": "ui", "steps": [{**phase()["steps"][0], "args": {}}], "final_predicates": phase()["final_predicates"]},
    {"kind": "ui", "steps": phase()["steps"], "final_predicates": [{"kind": "focused", "node_id": "$target"}]},
    {"kind": "memory_write", "topic": "x", "key": "y", "value": "z"},
    {"kind": "memory_write", "topic": "x", "key": "y", "value": "z" * 4097},
    {"kind": "ui", "steps": phase()["steps"], "final_predicates": phase()["final_predicates"], "code": "input()"},
])
def test_invalid_or_out_of_scope_plan_cannot_reach_any_executor(bad):
    fixture = Fixture([bad] * 3, allowed={"native_ui"})
    events = list(fixture.run())
    assert fixture.executions == [] and fixture.adapter.calls == []
    assert len(fixture.packets) == 3
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert fixture.cancel_count == 1


def test_memory_uses_exact_authorized_tools_and_claim_labels_have_no_verification_authority():
    fixture = Fixture([{"kind": "memory_write", "topic": "t", "key": "k", "value": "Observed note"},
                       {"kind": "memory_search", "query": "Observed note", "max_results": 2},
                       {"kind": "complete", "message": "Claim", "remaining_criteria": ["Unverified criterion"]}])
    events = list(fixture.run())
    assert fixture.executions == [("memory_write", {"topic": "t", "key": "k", "value": "Observed note"}),
                                  ("memory_search", {"query": "Observed note", "max_results": 2})]
    result = final(events)["portal"]["native_task"]
    assert result["status"] == "planner_claim" and result["task_verified"] is False
    assert result["remaining_criteria_authority"] == "planner_claim"
    assert result["remaining_criteria"] == ["Unverified criterion"]


class NoEffectAdapter(Adapter):
    """Real controller receipts for already-active + unchanged/cached inputs."""

    version = 0

    def observe(self):
        state = super().observe()
        state["revision"] = str(self.version)
        state["nodes"][0].update(focused=True, text_selection={"start": 0, "end": 18},
                                  operations=["focus", "key_chord"])
        state["nodes"].append({"id": "form", "role": "window", "name": "Owned",
            "generation": "1", "scope_id": "form", "visible": True, "enabled": True,
            "sensitive": False, "operations": ["switch_window"]})
        return state

    def perform(self, action):
        if action["verb"] == "switch_window":
            return {"status": "already_satisfied", "input_attempted": False}
        self.calls.append(copy.deepcopy(action))
        return {"status": "dispatched", "backend": "x11_guarded_xtest"}


def no_effect_phase(suffix=""):
    return {"kind": "ui", "steps": [
        {"id": "active" + suffix, "verb": "switch_window", "target": {"node_id": "form"},
         "args": {}, "expect": [{"kind": "active_window", "node_id": "$target"}]},
        {"id": "focus" + suffix, "verb": "focus", "target": {"node_id": "draft"},
         "args": {}, "expect": [{"kind": "focused", "node_id": "$target"}]},
        {"id": "select" + suffix, "verb": "key_chord", "target": {"node_id": "draft"},
         "args": {"keys": ["Control_L", "a"]},
         "expect": [{"kind": "text_selection", "node_id": "$target", "start": 0, "end": 18}]}],
        "final_predicates": [{"kind": "active_window", "node_id": "form"},
            {"kind": "focused", "node_id": "draft"},
            {"kind": "text_selection", "node_id": "draft", "start": 0, "end": 18}],
        "remaining_criteria": ["Unfinished delivery"]}


def no_effect_fixture(plans):
    fixture = Fixture(plans)
    fixture.adapter = NoEffectAdapter()
    return fixture


def completed_tools(events):
    return [event["tools"][0] for event in events
            if event.get("type") == "tool" and event.get("phase") == "complete"]


def test_repeated_mixed_zero_effect_ui_phase_does_not_reset_failures():
    fixture = no_effect_fixture([no_effect_phase()] * 4 + [{"kind": "complete", "message": "Claim"}])
    events = list(fixture.run())
    tools = completed_tools(events)
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert [tool["ok"] for tool in tools] == [True, False, False, False]
    result = fixture.packets[2]["last_result"]
    assert [item["effect_receipt"]["effect_state"] for item in result["outcomes"]] == ["absent", "replayed", "replayed"]
    assert result["phase_handoff"]["new_effect_count"] == 0
    assert result["error"] == "ui_phase_repeated_without_new_effect"
    assert "same effective actions and observed state" in result["reason"]
    assert result["status"] == "complete" and result["phase_handoff"]["terminal_verified"] is True
    assert all(item["status"] == "verified" for item in result["outcomes"])
    assert len(fixture.adapter.calls) == 2  # Cached inputs are never repeated.
    assert fixture.cancel_count == 1


def test_planner_step_id_or_claim_renaming_does_not_make_zero_effect_phase_new():
    plans = [no_effect_phase(str(index)) for index in range(4)]
    for index, plan in enumerate(plans):
        plan["remaining_criteria"] = ["Planner claim " + str(index)]
    fixture = no_effect_fixture(plans + [{"kind": "complete", "message": "Claim"}])
    events = list(fixture.run())
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert [tool["ok"] for tool in completed_tools(events)] == [True, False, False, False]


def test_first_verified_no_effect_phase_remains_valid_and_has_no_mutation_credit():
    fixture = no_effect_fixture([no_effect_phase(), {"kind": "complete", "message": "Claim"}])
    events = list(fixture.run())
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"
    assert completed_tools(events)[0]["ok"] is True
    result = fixture.packets[1]["last_result"]
    assert result["phase_handoff"]["new_effect_count"] == 0
    assert result["evidence_authority"] == "inspection"


@pytest.mark.parametrize("change", ["input_state", "planning_state", "arguments", "effects"])
def test_changed_observed_state_or_effective_action_is_not_the_same_zero_effect_phase(change):
    second = no_effect_phase()
    if change == "arguments":
        second["steps"][2]["args"]["keys"] = ["Control_L", "c"]
    if change == "effects":
        second["steps"][2]["expect"] = [{"kind": "focused", "node_id": "$target"}]
    fixture = no_effect_fixture([no_effect_phase(), second, {"kind": "complete", "message": "Claim"}])
    compile_phase = fixture.compile
    observe = fixture.observe

    def changed_compile(packet):
        if len(fixture.packets) == 1 and change == "input_state":
            fixture.adapter.version += 1
        return compile_phase(packet)

    def changed_observe():
        result = observe()
        if len(fixture.packets) == 1 and change == "planning_state":
            result["new_mapped_window_evidence"] = "fresh host observation"
        return result

    fixture.compile, fixture.observe = changed_compile, changed_observe
    events = list(fixture.run())
    assert [tool["ok"] for tool in completed_tools(events)] == [True, True]
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"


def test_actual_mutation_clears_zero_effect_repetition_history():
    fixture = no_effect_fixture([no_effect_phase(), {"kind": "ui", "steps": [
        {"id": "change-state", "verb": "focus", "target": {"node_id": "draft"}, "args": {},
         "expect": [{"kind": "focused", "node_id": "$target"}]}],
        "final_predicates": [{"kind": "focused", "node_id": "draft"}]},
        no_effect_phase(), {"kind": "complete", "message": "Claim"}])
    perform = fixture.adapter.perform

    def mutation(action):
        fixture.adapter.version += 1
        return perform(action)

    execute = fixture.execute

    def changed_execute(name, arguments):
        if len(fixture.executions) == 1:
            fixture.adapter.perform = mutation
        result = execute(name, arguments)
        fixture.adapter.perform = perform
        if len(fixture.executions) == 2:
            assert result["outcomes"][0]["effect_receipt"]["effect_state"] == "applied"
            fixture.adapter.version = 0  # Independently return to the initial state.
        return result

    fixture.execute = changed_execute
    events = list(fixture.run())
    assert [tool["ok"] for tool in completed_tools(events)] == [True, True, True]
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"


def test_memory_phases_do_not_use_the_ui_repetition_guard():
    memory = {"kind": "memory_write", "topic": "t", "key": "k", "value": "Source-bound note"}
    search = {"kind": "memory_search", "query": "Source-bound note", "max_results": 2}
    fixture = Fixture([memory, memory, search, search, {"kind": "complete", "message": "Claim"}])
    events = list(fixture.run())
    assert all(tool["ok"] for tool in completed_tools(events))
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"


def test_accepted_visual_knowledge_phases_remain_outside_ui_repetition_guard():
    from test_native_visual_inspection import _inspection_receipt

    fixture = Fixture([{"kind": "visual_inspection"}] * 2 + [{"kind": "complete", "message": "Claim"}])
    fixture.adapter.inspect_visual = lambda: {"status": "observed", "reading": "Captured source fact.",
        "visual_inspection_receipt": _inspection_receipt()}
    events = list(fixture.run())
    assert [tool["ok"] for tool in completed_tools(events)] == [True, True]
    assert fixture.adapter.calls == []


@pytest.mark.parametrize("fault", ["unverified", "unknown_effect", "applied", "missing_before",
                                  "missing_terminal", "unverified_terminal", "missing_current_state"])
def test_incomplete_or_uncertain_proof_cannot_establish_zero_effect_equivalence(fault):
    fixture = no_effect_fixture([no_effect_phase(), {"kind": "complete", "message": "Claim"}])
    events = list(fixture.run())
    result = json.loads(completed_tools(events)[0]["result"])
    observed = fixture.packets[0]["current_state"]
    assert _zero_effect_ui_phase_key(result, observed)
    if fault == "unverified":
        result["outcomes"][0]["status"] = "unresolved"
    elif fault in {"unknown_effect", "applied"}:
        result["outcomes"][0]["effect_receipt"]["effect_state"] = "unknown" if fault == "unknown_effect" else "applied"
    elif fault == "missing_before":
        result["outcomes"][0].pop("before")
    elif fault == "missing_terminal":
        result.pop("terminal")
    elif fault == "unverified_terminal":
        result["terminal_receipt"]["status"] = "unresolved"
    else:
        observed = {}
    assert _zero_effect_ui_phase_key(result, observed) is None


def test_semantically_invalid_later_step_cannot_dispatch_valid_earlier_step():
    invalid = phase()
    invalid["steps"][1] = {"id": "bad-range", "verb": "select_text", "target": {"node_id": "draft"},
                           "args": {"start": 5, "end": 1},
                           "expect": [{"kind": "focused", "node_id": "$target"}]}
    # JSON Schema alone cannot express this cross-field ordering constraint.
    Draft202012Validator(phase_response_schema(ALLOWED)).validate(invalid)
    fixture = Fixture([invalid] * 3)
    result = final(list(fixture.run()))
    assert fixture.executions == [] and fixture.adapter.calls == []
    assert result["portal"]["native_task"]["status"] == "blocked"


def test_authorized_memory_still_rejects_oversized_value_before_execution():
    fixture = Fixture([{"kind": "memory_write", "topic": "t", "key": "k", "value": "x" * 4097}] * 3)
    assert final(list(fixture.run()))["portal"]["native_task"]["status"] == "blocked"
    assert fixture.executions == []


def test_three_unverified_phases_stop_without_replaying_any_input():
    bad = phase()
    fixture = disappear_before_execution(Fixture([bad] * 4), times=3)
    result = final(list(fixture.run()))
    assert len(fixture.packets) == len(fixture.executions) == 3
    assert fixture.adapter.calls == []
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert fixture.cancel_count == 1


def partial_write_phase(value):
    plan = phase()
    plan["steps"][0]["args"]["value"] = value
    plan["steps"][0]["expect"][0]["value"] = value
    plan["steps"][1]["target"] = {"node_id": "missing"}
    return plan


def test_three_novel_verified_prefixes_reach_next_planner_without_success_relabel():
    fixture = Fixture([partial_write_phase(value) for value in ("Alpha", "Beta", "Gamma")]
                      + [{"kind": "complete", "message": "CPU planner claim."}])
    events = list(fixture.run())
    assert len(fixture.packets) == 4
    assert len(fixture.adapter.calls) == 3
    tools = completed_tools(events)
    assert [tool["ok"] for tool in tools] == [False, False, False]
    for packet in fixture.packets[1:]:
        result = packet["last_result"]
        assert result["status"] == "needs_replan" and result["error"] == "ui_workflow_unresolved"
        assert result["phase_handoff"]["new_effect_count"] == 1
        assert [item["effect_receipt"]["effect_state"] for item in result["outcomes"]] == ["applied", "absent"]
        assert result["phase_handoff"]["terminal_verified"] is False
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"
    assert final(events)["portal"]["native_task"]["task_verified"] is False


def produced_partial_result():
    fixture = Fixture([])
    plan = partial_write_phase("Alpha")
    return fixture.execute("native_ui", {"operation": "workflow", "steps": plan["steps"],
                                         "final_predicates": plan["final_predicates"]})


@pytest.mark.parametrize("corruption", [
    "unverified", "already_active", "unchanged", "unknown", "replayed", "schema", "action_id", "idempotency",
    "before_revision", "after_revision", "before_semantic", "after_semantic", "session", "scope", "generation",
    "action_revision", "missing_before", "missing_after", "protected", "disabled", "invisible", "operation", "predicates",
    "before_session", "after_session", "before_scope", "before_generation", "duplicate_target", "target_id",
    "receipt_status", "same_semantic_state",
])
def test_partial_recovery_requires_producer_bound_verified_applied_prefix(corruption):
    result = produced_partial_result()
    outcome = result["outcomes"][0]
    action, receipt, effect = outcome["action"], outcome["receipt"], outcome["effect_receipt"]
    if corruption == "unverified":
        outcome["status"] = "unresolved"
    elif corruption == "already_active":
        receipt.update(status="already_satisfied", input_attempted=False)
    elif corruption in {"unchanged", "unknown", "replayed"}:
        effect["effect_state"] = corruption
    elif corruption == "schema":
        effect["schema"] = "untrusted"
    elif corruption in {"action_id", "idempotency", "before_revision", "after_revision"}:
        key = "idempotency_key" if corruption == "idempotency" else corruption
        effect[key] = "mismatch"
    elif corruption in {"before_semantic", "after_semantic"}:
        effect[corruption + "_revision"] = "mismatch"
    elif corruption in {"session", "scope", "generation", "action_revision"}:
        key = {"session": "session_id", "scope": "scope_id", "generation": "generation", "action_revision": "revision"}[corruption]
        action[key] = "mismatch"
    elif corruption in {"missing_before", "missing_after"}:
        outcome.pop(corruption.removeprefix("missing_"))
    elif corruption in {"before_session", "after_session"}:
        outcome[corruption.removesuffix("_session")]["session_id"] = "mismatch"
    elif corruption == "before_scope":
        outcome["before"]["scope"]["id"] = "mismatch"
    elif corruption in {"before_generation", "duplicate_target", "target_id"}:
        target = next(node for node in outcome["before"]["nodes"] if node["id"] == action["node_id"])
        if corruption == "duplicate_target":
            outcome["before"]["nodes"].append(copy.deepcopy(target))
        else:
            target["generation" if corruption == "before_generation" else "id"] = "mismatch"
    elif corruption == "receipt_status":
        receipt["status"] = "rejected"
    elif corruption == "same_semantic_state":
        outcome["before"]["base_semantic_revision"] = outcome["after"]["revision"]
        effect["before_semantic_revision"] = outcome["after"]["revision"]
    elif corruption in {"protected", "disabled", "invisible", "operation"}:
        target = next(node for node in outcome["before"]["nodes"] if node["id"] == action["node_id"])
        if corruption == "operation":
            target["operations"] = []
        else:
            target[{"protected": "sensitive", "disabled": "enabled", "invisible": "visible"}[corruption]] = corruption == "protected"
    else:
        effect["predicates_sha256"] = "mismatch"
    result["phase_handoff"]["new_effect_count"] = 999  # A scalar claim cannot repair the receipt.
    seen = set()
    assert _new_verified_ui_prefix(result, seen) is False
    assert seen == set()


def test_applied_outcome_after_first_unverified_is_not_a_recovery_prefix():
    result = produced_partial_result()
    result["outcomes"].reverse()
    assert _new_verified_ui_prefix(result, set()) is False


def test_input_id_step_id_or_expectation_relabel_does_not_make_transition_new():
    result = produced_partial_result()
    seen = set()
    assert _new_verified_ui_prefix(result, seen) is True
    renamed = copy.deepcopy(result)
    action = renamed["outcomes"][0]["action"]
    action.update(id="new-opaque-action", idempotency_key="new-opaque-key", description="New planner label")
    action["expect"].append({"kind": "visible", "role": "textbox", "name": "Draft", "visible": True})
    effect = renamed["outcomes"][0]["effect_receipt"]
    effect.update(action_id=action["id"], idempotency_key=action["idempotency_key"],
                  predicates_sha256=_digest(action["expect"]))
    renamed["phase_handoff"]["verified_steps"] = [{"id": "renamed", "index": 0}]
    assert _new_verified_ui_prefix(renamed, seen) is False
    assert len(seen) == 1


def test_capture_revision_changes_use_producer_base_semantic_revision_for_novelty():
    result = produced_partial_result()
    seen = set()
    assert _new_verified_ui_prefix(result, seen) is True
    framed = copy.deepcopy(result)
    outcome = framed["outcomes"][0]
    for direction in ("before", "after"):
        snapshot = outcome[direction]
        snapshot["base_semantic_revision"] = snapshot["revision"]
        snapshot["revision"] = "capture-" + direction
        outcome["effect_receipt"][direction + "_revision"] = snapshot["revision"]
    outcome["action"]["revision"] = outcome["before"]["revision"]
    assert _new_verified_ui_prefix(framed, seen) is False
    assert len(seen) == 1


class WindowCycleAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.active = "left"

    def observe(self):
        self.timeline.append("observe")
        return {"session_id": "session", "scope": {"id": self.active, "kind": "desktop"},
                "revision": self.active, "nodes": [{"id": name, "generation": "1", "scope_id": name,
                    "role": "window", "name": name, "visible": True, "enabled": True, "sensitive": False,
                    "operations": ["switch_window"]} for name in ("left", "right")]}

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        self.active = action["node_id"]
        return {"status": "dispatched"}


def test_actual_window_ABABA_cycle_cannot_keep_buying_recovery_with_new_expectations():
    plans = []
    for index, target in enumerate(("right", "left", "right", "left", "right")):
        expect = [{"kind": "active_window", "node_id": "$target"}]
        expect += [{"kind": "visible", "role": "window", "name": target, "visible": True}] * index
        plans.append({"kind": "ui", "steps": [
            {"id": "switch" + str(index), "verb": "switch_window", "target": {"node_id": target},
             "args": {}, "expect": expect},
            {"id": "missing", "verb": "focus", "target": {"node_id": "missing"},
             "args": {}, "expect": [{"kind": "focused", "node_id": "$target"}]}],
            "final_predicates": [{"kind": "active_window", "node_id": target}]})
    fixture = Fixture(plans + [{"kind": "complete", "message": "Cannot reach."}])
    fixture.adapter = WindowCycleAdapter()
    events = list(fixture.run())
    assert len(fixture.adapter.calls) == len(fixture.packets) == 5
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert all(tool["ok"] is False for tool in completed_tools(events))
    for tool in completed_tools(events):
        result = json.loads(tool["result"])
        assert result["phase_handoff"]["new_effect_count"] == 1
        assert result["phase_handoff"]["goal_verified"] is False


@pytest.mark.parametrize("late", ["cancel", "deadline"])
def test_partial_recovery_never_dispatches_a_late_next_phase(late):
    now = [0]
    fixture = Fixture([partial_write_phase("Alpha"), partial_write_phase("Beta")], clock=lambda: now[0])
    compile_phase = fixture.compile

    def late_compile(packet):
        if fixture.packets:
            if late == "cancel":
                fixture.cancelled.set()
            else:
                now[0] = 6
        return compile_phase(packet)

    fixture.compile = late_compile
    events = list(fixture.run(max_seconds=5))
    assert len(fixture.adapter.calls) == len(fixture.executions) == 1
    assert final(events)["portal"]["native_task"]["status"] == ("cancelled" if late == "cancel" else "deadline_exceeded")


def test_novel_partial_recovery_keeps_the_forty_phase_ceiling_and_no_task_verification():
    fixture = Fixture([partial_write_phase("Draft " + str(index)) for index in range(40)])
    events = list(fixture.run(max_phases=40))
    assert len(fixture.adapter.calls) == len(fixture.packets) == 40
    assert final(events)["portal"]["native_task"]["status"] == "phase_budget_exceeded"
    assert final(events)["portal"]["native_task"]["task_verified"] is False


def test_late_compiler_result_never_dispatches_and_phase_budget_cancels():
    now = [0]
    fixture = Fixture([phase()], clock=lambda: now[0])
    original = fixture.compile

    def late(packet):
        result = original(packet)
        now[0] = 6
        return result

    fixture.compile = late
    result = final(list(fixture.run(max_seconds=5)))
    assert result["portal"]["native_task"]["status"] == "deadline_exceeded"
    assert fixture.executions == [] and fixture.cancel_count == 1
    limited = Fixture([phase()])
    assert final(list(limited.run(max_phases=1)))["portal"]["native_task"]["status"] == "phase_budget_exceeded"
    assert limited.cancel_count == 1


def test_cancel_after_compilation_and_generator_close_before_dispatch_stop_owned_work():
    fixture = Fixture([phase()])
    original = fixture.compile

    def cancelled_plan(packet):
        result = original(packet)
        fixture.cancelled.set()
        return result

    fixture.compile = cancelled_plan
    assert final(list(fixture.run()))["portal"]["native_task"]["status"] == "cancelled"
    assert fixture.executions == [] and fixture.cancel_count == 1
    closed = Fixture([phase()])
    generator = closed.run()
    for event in generator:
        if event.get("type") == "tool" and event.get("phase") == "start":
            break
    generator.close()
    assert closed.executions == [] and closed.cancel_count == 1


def test_cancel_while_plan_progress_is_yielded_cannot_start_compiler_call():
    fixture = Fixture([phase()])
    generator = fixture.run()
    assert next(generator)["stage"] == "native_task_observe"
    assert next(generator)["stage"] == "native_task_plan"
    fixture.cancelled.set()
    result = final(list(generator))
    assert fixture.packets == [] and fixture.executions == []
    assert result["portal"]["native_task"]["status"] == "cancelled"
    assert fixture.cancel_count == 1


def test_callback_failure_preserves_completed_raw_receipts_and_cancels_owned_scope():
    fixture = Fixture([phase()])
    original = fixture.compile

    def fail_after_phase(packet):
        if fixture.packets:
            raise TimeoutError("Owned planner transport deadline")
        return original(packet)

    fixture.compile = fail_after_phase
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert result["message"]["content"] == "Native task callback failed: TimeoutError"
    assert len(result["portal"]["safe_tools_executed"]) == 1
    assert len(fixture.adapter.calls) == 2 and fixture.cancel_count == 1


@pytest.mark.parametrize("options", [{"max_phases": 0}, {"max_phases": True}, {"max_seconds": 0},
                                     {"max_seconds": float("nan")}, {"max_seconds": float("inf")}])
def test_invalid_host_budgets_fail_before_observation_or_planning(options):
    fixture = Fixture([])
    with pytest.raises(ValueError):
        list(fixture.run(**options))
    assert fixture.packets == [] and fixture.adapter.timeline == []


def test_callback_cancellation_exception_finishes_cancelled_without_replanning():
    fixture = Fixture([phase()])

    def cancelled_plan(_packet):
        fixture.cancelled.set()
        raise TimeoutError("Cancelled planner settled with an exception")

    fixture.compile = cancelled_plan
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "cancelled"
    assert fixture.executions == [] and fixture.cancel_count == 1


def compilation_fixture(plans):
    fixture = Fixture(plans)
    compile_phase = fixture.compile

    def compile_or_raise(packet):
        result = compile_phase(packet)
        if isinstance(result, Exception):
            raise result
        return result

    fixture.compile = compile_or_raise
    return fixture


def test_compilation_error_reobserves_and_recovers_before_any_input():
    fixture = compilation_fixture([NativePhaseCompilationError(), phase(),
                                   {"kind": "complete", "message": "Claim."}])
    events = list(fixture.run())
    assert len(fixture.packets) == 3
    assert fixture.adapter.timeline[:4] == ["observe", "compile", "observe", "compile"]
    feedback = fixture.packets[1]["last_result"]
    assert feedback["error"] == "invalid_native_phase_json"
    assert feedback["dispatch_status"] == "not_dispatched" and feedback["input_attempted"] is False
    assert feedback["new_effect_count"] == 0
    assert "smaller complete phase" in feedback["reason"]
    assert len(fixture.executions) == 1
    assert [action["verb"] for action in fixture.adapter.calls] == ["set_value", "invoke"]
    assert final(events)["portal"]["native_task"]["status"] == "planner_claim"


def test_three_compilation_errors_use_the_existing_failure_budget_without_dispatch():
    fixture = compilation_fixture([NativePhaseCompilationError()] * 4)
    result = final(list(fixture.run()))
    task = result["portal"]["native_task"]
    assert task["status"] == "blocked" and task["phases"] == 3
    assert len(fixture.packets) == 3 and len(task["ledger"]) == 3
    assert fixture.adapter.timeline == ["observe", "compile"] * 3
    assert fixture.executions == [] and fixture.adapter.calls == [] and fixture.cancel_count == 1
    assert all(item["kind"] == "invalid" and item["input_attempted"] is False for item in task["ledger"])


@pytest.mark.parametrize("failure", ["schema", "unverified"])
def test_compilation_and_existing_invalid_unverified_phases_share_one_counter(failure):
    bad = {"kind": "ui", "steps": [], "final_predicates": []}
    if failure == "unverified":
        bad = phase()
    fixture = compilation_fixture([NativePhaseCompilationError(), bad, NativePhaseCompilationError(),
                                   {"kind": "complete", "message": "Cannot reach."}])
    if failure == "unverified":
        disappear_before_execution(fixture)
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert len(fixture.packets) == 3 and fixture.adapter.calls == []
    assert len(fixture.executions) == (1 if failure == "unverified" else 0)


def test_verified_executor_success_retains_its_existing_failure_counter_reset():
    fixture = compilation_fixture([NativePhaseCompilationError(), phase(), NativePhaseCompilationError(),
        NativePhaseCompilationError(), {"kind": "complete", "message": "Claim."}])
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "planner_claim"
    assert len(fixture.packets) == 5 and len(fixture.executions) == 1
    assert fixture.packets[2]["last_result"]["phase_handoff"]["new_effect_count"] == 2
    assert len(fixture.adapter.calls) == 2


@pytest.mark.parametrize("late", ["cancel", "deadline"])
def test_late_compilation_error_obeys_boundary_before_recording_or_replanning(late):
    now = [0]
    fixture = Fixture([], clock=lambda: now[0])

    def expired(_packet):
        if late == "cancel":
            fixture.cancelled.set()
        else:
            now[0] = 6
        raise NativePhaseCompilationError()

    fixture.compile = expired
    result = final(list(fixture.run(max_seconds=5)))
    task = result["portal"]["native_task"]
    assert task["status"] == ("cancelled" if late == "cancel" else "deadline_exceeded")
    assert task["ledger"] == [] and task["phases"] == 1
    assert fixture.executions == [] and fixture.cancel_count == 1


@pytest.mark.parametrize("error", [TimeoutError("private transport canary"), ValueError("private callback canary"),
                                  json.JSONDecodeError("private decode canary", "private raw reply", 0)])
def test_untyped_compiler_faults_remain_fatal_and_redacted(error):
    fixture = compilation_fixture([error, {"kind": "complete", "message": "Cannot reach."}])
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert len(fixture.packets) == 1 and fixture.executions == []
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("origin", ["observe", "execute"])
def test_typed_compilation_error_outside_compiler_callback_remains_fatal(origin):
    fixture = Fixture([phase(), {"kind": "complete", "message": "Cannot reach."}])

    def wrong_boundary(*_):
        raise NativePhaseCompilationError()

    setattr(fixture, origin, wrong_boundary)
    result = final(list(fixture.run()))
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert result["portal"]["native_task"]["phases"] == 1
    assert len(fixture.packets) == (0 if origin == "observe" else 1)
    assert fixture.adapter.calls == [] and fixture.cancel_count == 1


def test_compilation_error_does_not_extend_the_phase_budget():
    fixture = compilation_fixture([NativePhaseCompilationError()] * 2)
    result = final(list(fixture.run(max_phases=1)))
    assert result["portal"]["native_task"]["status"] == "phase_budget_exceeded"
    assert len(fixture.packets) == 1 and fixture.executions == [] and fixture.cancel_count == 1
