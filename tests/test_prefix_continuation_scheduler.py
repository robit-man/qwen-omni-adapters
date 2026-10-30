"""Controller handoff opportunity only; these CPU adapters prove no live task."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_native_effect_pipeline import NativeAdapter
from test_native_working_state import envelope
from test_ui_task import ALLOWED, GOAL, final

from portal import ui_decision
from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_task import run_native_goal

OBJECTIVE = "Inspect the owned brief before completing the requested draft"


def yield_plan(*, label="pending", target="commit"):
    return {"kind": "ui", "steps": [
        {"id": "verify-current", "verb": "switch_window", "target": {"node_id": "form"},
         "expect": [{"kind": "active_window", "node_id": "$target"}]},
        {"id": label, "verb": "invoke", "target": {"node_id": target}, "args": {},
         "expect": [{"kind": "text", "text": "An unverified planner prediction"}]},
        {"id": "untouched-suffix", "verb": "set_value", "target": {"node_id": "draft"},
         "args": {"value": "This suffix must not execute"},
         "expect": [{"kind": "value", "node_id": "$target", "value": "This suffix must not execute"}]}],
        "final_predicates": [{"kind": "text", "text": "Planner claim only"}]}


def write_plan(value="Fresh verified value"):
    return {"kind": "ui", "steps": [{"id": "actual-write", "verb": "set_value",
        "target": {"node_id": "draft"}, "args": {"value": value},
        "expect": [{"kind": "value", "node_id": "$target", "value": value}]}],
        "final_predicates": [{"kind": "value", "node_id": "draft", "value": value}]}


def bound_handoff(packet, plan, objective):
    state = packet["current_state"]["state"]
    node = next(node for node in state["nodes"] if node["id"] == plan["steps"][0]["target"]["node_id"])
    family = "switch_window" if plan["steps"][0]["verb"] == "switch_window" else "text_entry"
    return {"schema": "robit.omni.native-action-handoff.v1",
        "authority": "planner_claim_not_effect_or_input_authority", "input_authority": False,
        "goal_sha256": hashlib.sha256(packet["goal"].encode()).hexdigest(),
        "source_epoch": packet["working_state"]["source_epoch"],
        "source_phase": packet["working_state"]["source_phase"],
        "source_observation": ui_decision._observation_binding(state),
        "target": ui_decision._target_binding(node), "operation_family": family,
        "objective": objective, "next_operation": "Inspect the selected resource",
        "needed_evidence": ["The actual selected resource"],
        "expected_transition": {"schema": "robit.omni.native-expected-transition.v1",
            "mode": "observe_after_input", "operation_family": family,
            "authority": "host_operation_descriptor_not_observed_effect", "input_authority": False,
            "task_progress": False}}


class Scheduler:
    def __init__(self, plans, *, mutate=None):
        self.plans = iter(plans)
        self.adapter = NativeAdapter()
        original_observe = self.adapter.observe
        self.observation_number = 0
        def observe_with_window():
            self.observation_number += 1
            state = original_observe()
            state["nodes"].append({"id": "form", "generation": "1", "scope_id": "form",
                "role": "window", "name": "Owned workspace", "operations": ["switch_window"],
                "visible": True, "enabled": True, "sensitive": False})
            return state
        self.adapter.observe = observe_with_window
        self.store = ComputerUseStore(None, adapter_factory=lambda *_: self.adapter)
        self.stopped = self.store.begin_task("session", private_boundaries=True)
        self.packets, self.results, self.executions, self.observed = [], [], [], []
        self.mutate = mutate

    def observe(self):
        with self.store.task_scope("session", "desktop", self.stopped):
            observed = self.store.planning_observation("session")
            self.observed.append(copy.deepcopy(observed))
            return observed

    def compile(self, packet):
        self.packets.append(copy.deepcopy(packet))
        plan = next(self.plans)
        if callable(plan):
            plan = plan(self, packet)
        if plan == "invalid":
            return {"not": "a structured phase"}
        objective = plan.pop("test_objective", OBJECTIVE)
        reply = envelope(packet, plan, objective=objective)
        if plan["kind"] == "ui":
            reply["action_handoff"] = bound_handoff(packet, plan, objective)
        return reply

    def execute(self, name, arguments):
        self.executions.append((name, copy.deepcopy(arguments)))
        if name != "native_ui":
            return {"results": [], "query": arguments.get("query", "")}
        assert arguments["operation"] == "verified_prefix"
        with self.store.task_scope("session", "desktop", self.stopped):
            result = run_verified_prefix(arguments["steps"], execute_step=lambda step, index:
                self.store.immediate_transition("session", step, goal=GOAL, objective=arguments["objective"],
                    action_handoff=arguments.get("action_handoff") if index == 0 else None,
                    defer_unannotated_boundary=index > 0),
                cancelled=self.stopped.is_set, source_phase=arguments["source_phase"])
        if self.mutate is not None:
            # Distinct JSON receipt fields stay distinct under corruption,
            # even when the in-process producer originally shared one dict.
            result = json.loads(json.dumps(result))
            self.mutate(result, self, arguments)
        self.results.append(copy.deepcopy(result))
        return result

    def run(self, **kwargs):
        return list(run_native_goal(GOAL, observe=self.observe, compile_phase=self.compile,
            execute=self.execute, cancel=self.stopped.set, cancelled=self.stopped.is_set,
            allowed_tools=ALLOWED, working_state=True, effect_contracts=True, **kwargs))


def blocked():
    return {"kind": "blocked", "reason": "Independent whole-task criteria remain unmet"}


def test_two_failures_then_host_noinput_continuation_gets_one_fresh_planner_handoff():
    scheduler = Scheduler(["invalid", "invalid", yield_plan(), blocked()])
    events = scheduler.run()
    assert len(scheduler.packets) == 4 and len(scheduler.executions) == 1
    packet = scheduler.packets[-1]
    receipt = packet["last_result"]["verified_prefix_receipt"]
    assert receipt["stop"] == "unannotated_observation_boundary"
    assert receipt["continuation"]["step"] == yield_plan()["steps"][1]
    assert packet["last_result"]["planner_workflow_claim"]["remaining_steps"] == yield_plan()["steps"][1:]
    assert packet["working_state"]["consecutive_unverified_count"] == 2
    assert packet["ledger"][-1]["deferred_to_planner"] is True
    assert packet["current_state"] == scheduler.observed[-1] and len(scheduler.observed) == 4
    assert scheduler.observation_number > 4 and not scheduler.adapter.calls
    assert final(events)["portal"]["native_task"]["task_verified"] is False


CORRUPTIONS = [
    ("input_attempted", None), ("input_attempted", True), ("dispatch_status", "dispatched"),
    ("new_effect_count", False), ("new_effect_count", 1), ("task_progress", True),
    ("error", "ordinary_unresolved"), ("effect_receipt.effect_state", "unknown"),
    ("effect_receipt.environment_may_have_changed", True),
    ("outcomes", [{"status": "verified"}]), ("policy_receipts", [{"status": "accepted"}]),
    ("point_receipts", [{"status": "observed"}]), ("selection_receipts", [{"selected": True}]),
    ("dispatch_record", {"input_attempted": True}), ("receipt", {"status": "dispatched"}),
    ("pointer_recovery", {"reference": "unrelated"}),
    ("verified_prefix_receipt.stop", "unverified_operation"),
    ("verified_prefix_receipt.input_authority", True),
    ("verified_prefix_receipt.all_immediate_readbacks_verified", True),
    ("verified_prefix_receipt.attempted_step_count", True),
    ("verified_prefix_receipt.verified_prefix_count", 0),
    ("verified_prefix_receipt.unattempted_step_count", 1),
    ("verified_prefix_receipt.planned_step_count", 2),
    ("verified_prefix_receipt.continuation.input_authority", True),
    ("verified_prefix_receipt.continuation.goal_sha256", "foreign-goal"),
    ("verified_prefix_receipt.continuation.objective", "Unrelated objective"),
    ("verified_prefix_receipt.continuation.step_index", 2),
    ("verified_prefix_receipt.continuation.step.target.node_id", "draft"),
    ("verified_prefix_receipt.continuation.source_ref.revision", "mismatched-native-revision"),
    ("verified_prefix_receipt.step_contexts.0.action_handoff.source_epoch", "old-task"),
    ("verified_prefix_receipt.step_contexts.0.action_handoff.source_phase", 1),
    ("verified_prefix_receipt.step_contexts.0.verification_only_receipt.input_attempted", None),
    ("verified_prefix_receipt.step_contexts.0.verification_only_receipt.status", "unresolved"),
    ("verified_prefix_receipt.step_contexts.0.verification_only_receipt.target.node_id", "draft"),
    ("verified_prefix_receipt.step_contexts.0.verification_only_receipt.predicates_sha256", "unbound"),
    ("verified_prefix_receipt.step_contexts.0.verification_only_receipt.args_sha256", "unbound"),
    ("verified_prefix_receipt.step_contexts.0.immediate_effect_contract.operation", "focus"),
    ("phase_handoff.new_effect_count", True), ("phase_handoff.terminal_verified", True),
]


def set_path(result, path, value):
    keys = path.split(".")
    target = result
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[int(keys[-1]) if isinstance(target, list) else keys[-1]] = copy.deepcopy(value)


@pytest.mark.parametrize("path,value", CORRUPTIONS)
def test_unbound_contradictory_and_ordinary_receipts_spend_the_existing_failure_budget(path, value):
    scheduler = Scheduler(["invalid", "invalid", yield_plan(), blocked()],
        mutate=lambda result, *_: set_path(result, path, value))
    events = scheduler.run()
    task = final(events)["portal"]["native_task"]
    assert task["status"] == "blocked" and task["phases"] == 3
    assert len(scheduler.packets) == 3 and not scheduler.adapter.calls
    assert task["task_verified"] is False


@pytest.mark.parametrize("change", ["none", "step_ids", "objective", "observation", "target"])
def test_repackaged_current_continuations_cannot_rearm_the_unresolved_handoff(change):
    def changed(scheduler, packet):
        plan = yield_plan(label="new-step-id" if change == "step_ids" else "pending")
        if change == "objective":
            plan["test_objective"] = "A differently worded version of the same unfinished objective"
        if change == "observation":
            scheduler.adapter.saved = "Unrelated observation change, not an executor verified input"
        if change == "target":
            plan["steps"][1]["target"] = {"node_id": "draft"}
            plan["steps"][1]["verb"] = "double_click"
        return plan
    scheduler = Scheduler(["invalid", "invalid", yield_plan(), changed, blocked()])
    events = scheduler.run()
    assert len(scheduler.packets) == 4 and not scheduler.adapter.calls
    task = final(events)["portal"]["native_task"]
    assert task["status"] == "blocked" and task["working_state"]["consecutive_unverified_count"] == 3
    assert scheduler.packets[-1]["working_state"]["consecutive_unverified_count"] == 2


@pytest.mark.parametrize("interleave", ["memory", "idempotent", "invalid"])
def test_knowledge_noop_and_invalid_phases_do_not_rearm_the_consumed_handoff(interleave):
    middle = ({"kind": "memory_search", "query": "Existing constraint"} if interleave == "memory" else
              {**yield_plan(), "steps": yield_plan()["steps"][:1]} if interleave == "idempotent" else "invalid")
    scheduler = Scheduler([yield_plan(), middle, yield_plan(), yield_plan(), yield_plan(), blocked()])
    events = scheduler.run()
    rows = final(events)["portal"]["native_task"]["ledger"]
    deferred = [row for row in rows if row.get("deferred_to_planner")]
    assert len(deferred) == 1 and not scheduler.adapter.calls
    assert final(events)["portal"]["native_task"]["status"] == "blocked"


def test_genuine_new_verified_input_effect_resolves_and_rearms_one_handoff():
    scheduler = Scheduler(["invalid", "invalid", yield_plan(), write_plan(), yield_plan(), blocked()])
    events = scheduler.run()
    assert len(scheduler.packets) == 6
    assert [action["verb"] for action in scheduler.adapter.calls] == ["set_value"]
    assert scheduler.adapter.value == "Fresh verified value"
    assert scheduler.packets[-1]["ledger"][-1]["deferred_to_planner"] is True
    assert scheduler.packets[-1]["working_state"]["consecutive_unverified_count"] == 0
    assert final(events)["portal"]["native_task"]["task_verified"] is False


@pytest.mark.parametrize("corruption", ["replayed", "already_satisfied", "unknown", "scalar_only"])
def test_scalar_or_nonfresh_effect_cannot_rearm_a_consumed_handoff(corruption):
    def mutate(result, scheduler, arguments):
        if arguments["steps"][0]["verb"] != "set_value":
            return
        outcome = result["outcomes"][0]
        if corruption == "already_satisfied":
            outcome["receipt"].update(status="already_satisfied", input_attempted=False)
        else:
            outcome["effect_receipt"]["effect_state"] = "replayed" if corruption == "replayed" else "unknown"
        result["phase_handoff"]["new_effect_count"] = 999
    scheduler = Scheduler([yield_plan(), write_plan(), yield_plan(), yield_plan(), yield_plan(), blocked()], mutate=mutate)
    events = scheduler.run()
    rows = final(events)["portal"]["native_task"]["ledger"]
    assert sum(bool(row.get("deferred_to_planner")) for row in rows) == 1
    assert final(events)["portal"]["native_task"]["status"] == "blocked"


@pytest.mark.parametrize("fault", ["self_consistent_other_target", "self_consistent_other_predicate", "foreign_native_source"])
def test_self_consistent_receipts_must_still_match_the_issued_plan_and_task(fault):
    def mutate(result, *_):
        prefix = result["verified_prefix_receipt"]
        context = prefix["step_contexts"][0]
        receipt = context["verification_only_receipt"]
        if fault == "self_consistent_other_target":
            receipt["target"]["node_id"] = "draft"
            receipt["predicates"][0]["node_id"] = "draft"
            receipt["predicates_sha256"] = ui_decision._digest(receipt["predicates"])
        elif fault == "self_consistent_other_predicate":
            context["immediate_effect_contract"]["predicates"][0]["kind"] = "focused"
            receipt["predicates"][0]["kind"] = "focused"
            receipt["predicates_sha256"] = ui_decision._digest(receipt["predicates"])
        else:
            # Changing every internally agreeing source reference cannot move
            # historical receipts into another current native task/session.
            refs = [prefix["continuation"]["source_ref"], result["immediate_effect_contract"]["source_ref"],
                    context["immediate_effect_contract"]["source_ref"], receipt["original_source_ref"],
                    receipt["fresh_source_ref"]]
            for ref in refs:
                ref["session_id"] = "foreign-session"
    scheduler = Scheduler(["invalid", "invalid", yield_plan(), blocked()], mutate=mutate)
    events = scheduler.run()
    assert len(scheduler.packets) == 3 and not scheduler.adapter.calls
    assert final(events)["portal"]["native_task"]["working_state"]["consecutive_unverified_count"] == 3


def test_replaying_a_previously_verified_semantic_transition_does_not_rearm_again():
    def reset_after_second_handoff(result, scheduler, arguments):
        if arguments["source_phase"] == 3:
            # This controlled external reset is not an executor effect. The
            # next planner still sees the actual current empty value first.
            scheduler.adapter.value = ""
    scheduler = Scheduler([yield_plan(), write_plan(), yield_plan(), write_plan(),
                           yield_plan(), yield_plan(), blocked()], mutate=reset_after_second_handoff)
    events = scheduler.run()
    rows = final(events)["portal"]["native_task"]["ledger"]
    assert sum(bool(row.get("deferred_to_planner")) for row in rows) == 2
    assert len(scheduler.adapter.calls) == 1  # The old input is cached rather than dispatched again.
    assert scheduler.results[3]["outcomes"][0]["effect_receipt"]["effect_state"] == "replayed"
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
