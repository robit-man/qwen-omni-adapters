"""CPU acceptance of mandatory phase compilation and host execution boundaries."""
from __future__ import annotations

import copy
import json
import threading

import pytest
from jsonschema import Draft202012Validator

from portal.computer_use import ComputerUseStore
from portal.tools import tool_schemas
from portal.ui_task import _zero_effect_ui_phase_key, phase_response_schema, run_native_goal

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
    bad["steps"][0]["target"] = {"node_id": "missing"}
    fixture = Fixture([bad, phase(), {"kind": "complete", "message": "Claim."}])
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
    bad["steps"][0]["target"] = {"node_id": "missing"}
    fixture = Fixture([bad] * 4)
    result = final(list(fixture.run()))
    assert len(fixture.packets) == len(fixture.executions) == 3
    assert fixture.adapter.calls == []
    assert result["portal"]["native_task"]["status"] == "blocked"
    assert fixture.cancel_count == 1


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
