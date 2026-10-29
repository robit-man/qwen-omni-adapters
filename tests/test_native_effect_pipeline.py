"""Generated phases use private effect compilation through a real controller.

Controlled provider observations verify the integration contract, not live
desktop capability or loaded-model whole-task success.
"""
import copy

from test_native_working_state import envelope
from test_ui_task import ALLOWED, GOAL, Adapter, final, phase

from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_recovery import NativeStrategyLedger
from portal.ui_task import run_native_goal


class NativeAdapter(Adapter):
    def prepare_intent(self, intent):
        pass

    def finish_intent(self):
        pass

    def boundary_observe(self):
        return self.observe()


def pipeline(plan, *, ineffective=False, repeat=False, invalid_before=0):
    adapter = NativeAdapter()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    stopped = store.begin_task("session", private_boundaries=True)
    controller = store._sessions[("session", "desktop")].controller
    controller.max_observations = 1
    controller.verification_timeout_s = .02
    controller.observation_interval_s = .001
    packets, results = [], []
    strategies = NativeStrategyLedger()
    if ineffective:
        def no_effect(action):
            adapter.calls.append(copy.deepcopy(action))
            return {"status": "dispatched", "input_attempted": True}
        adapter.perform = no_effect

    def observe():
        with store.task_scope("session", "desktop", stopped):
            return store.planning_observation("session")

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        number = len(packets) - invalid_before
        if number <= 0:
            return {"not": "a valid structured phase"}
        selected = copy.deepcopy(plan)
        if repeat and number == 2:
            selected["steps"][0]["id"] = "a-new-wording-is-not-a-new-strategy"
        return envelope(packet, selected if number == 1 or repeat and number == 2 else
                        {"kind": "blocked", "reason": "Independent task criteria remain unverified."})

    def execute(name, arguments):
        assert name == "native_ui" and arguments["operation"] in {"immediate_transition", "verified_prefix"}
        with store.task_scope("session", "desktop", stopped, strategy_ledger=strategies) as receipts:
            def execute_step(step, index):
                return store.immediate_transition("session", step, goal=GOAL, objective=arguments["objective"])
            result = (run_verified_prefix(arguments["steps"], execute_step=execute_step,
                strategy_ledger=strategies, source_phase=arguments["source_phase"], cancelled=stopped.is_set)
                if arguments["operation"] == "verified_prefix" else execute_step(arguments["step"], 0))
            if receipts:
                result["strategy_filter_receipts"] = copy.deepcopy(receipts)
        results.append(copy.deepcopy(result))
        return result

    events = list(run_native_goal(GOAL, observe=observe, compile_phase=compile_phase,
        execute=execute, cancel=stopped.set, cancelled=stopped.is_set,
        allowed_tools=ALLOWED, working_state=True, effect_contracts=True, strategy_ledger=strategies))
    return adapter, packets, results, final(events)


def test_first_generated_write_gets_exact_readback_without_claiming_remaining_workflow():
    plan = phase()
    plan["steps"][0]["expect"] = [{"kind": "focused", "node_id": "$target"}]
    following = copy.deepcopy(plan["steps"][0])
    following.update(id="must-replan-first", args={"value": "Unexecuted suffix"})
    plan["steps"].append(following)
    adapter, packets, results, terminal = pipeline(plan)
    assert adapter.value == adapter.saved == "Exact draft"
    assert [item["verb"] for item in adapter.calls] == ["set_value", "invoke"]
    assert adapter.calls[0]["expect"] == [{"kind": "value", "node_id": "draft", "value": "Exact draft"}]
    assert results[0]["status"] == "observed_unverified" and results[0]["task_progress"] is False
    retained = packets[1]["last_result"]["planner_workflow_claim"]
    assert retained["remaining_steps"] == [following]
    assert retained["original_step_count"] == 3 and retained["terminal_verified"] is False
    assert retained["retained_plan"]["reference"] in {
        row["reference"] for row in packets[1]["working_state"]["plan_claims"]["entries"]}
    assert packets[1]["working_state"]["phase_refs"][0][2] == 2
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_ineffective_write_cannot_pass_using_model_predicate_or_execute_following_commit():
    plan = phase()
    plan["steps"][0]["expect"] = [{"kind": "visible", "name": "Draft", "visible": True}]
    adapter, packets, results, terminal = pipeline(plan, ineffective=True)
    assert adapter.value == adapter.saved == "" and len(adapter.calls) == 1
    assert results[0]["status"] == "needs_replan"
    assert results[0]["error"] == "ui_immediate_effect_unresolved"
    assert packets[1]["working_state"]["recovery_mode"] is True
    assert packets[1]["working_state"]["consecutive_unverified_count"] == 1
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_opaque_first_operation_retains_actual_boundary_fact_without_fabricated_effect():
    plan = {"kind": "ui_boundary", "objective": "Activate the observed commit control", "step": {
        "id": "commit", "verb": "invoke", "target": {"node_id": "commit"}, "args": {}}}
    adapter, packets, results, terminal = pipeline(plan)
    assert len(adapter.calls) == 1 and adapter.calls[0]["verb"] == "invoke"
    assert results[0]["immediate_effect_contract"]["mode"] == "observation_boundary"
    assert results[0]["ui_boundary_receipt"]["verified"] is False
    assert results[0]["effect_receipt"]["effect_state"] == "unknown"
    state = packets[1]["working_state"]
    assert state["phase_refs"][0][2] == 1
    assert state["executed_facts"][0]["action"]["verb"] == "invoke"
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_failed_strategy_is_shared_with_actual_offer_filter_across_planner_phases():
    adapter, packets, results, terminal = pipeline(phase(), ineffective=True, repeat=True)
    assert len(adapter.calls) == 1 and len(results) == 2
    assert results[1]["status"] == "needs_replan"
    assert any(receipt["blocked"] for receipt in results[1]["strategy_filter_receipts"])
    facts = packets[1]["working_state"]["strategy_recovery"]["facts"]
    assert facts[0]["classification"] == "input_effect_uncertain"
    assert facts[0]["input_authority"] is False
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_new_verified_prefix_resets_prior_failure_streak_before_opaque_yield():
    adapter, packets, results, terminal = pipeline(phase(), invalid_before=2)
    assert len(adapter.calls) == 2 and len(results) == 1
    assert packets[2]["working_state"]["consecutive_unverified_count"] == 2
    assert packets[3]["working_state"]["consecutive_unverified_count"] == 0
    assert len(packets[3]["working_state"]["phase_refs"]) == 3
    assert terminal["portal"]["native_task"]["task_verified"] is False


def test_already_satisfied_prefix_does_not_erase_prior_failure_streak():
    plan = phase()
    plan["steps"] = [plan["steps"][0]]
    plan["steps"][0]["args"]["value"] = ""
    adapter, packets, results, terminal = pipeline(plan, invalid_before=2)
    assert not adapter.calls and len(results) == 1
    assert packets[3]["working_state"]["consecutive_unverified_count"] == 2
    assert terminal["portal"]["native_task"]["task_verified"] is False
