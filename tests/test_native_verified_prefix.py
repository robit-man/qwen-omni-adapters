"""Verified prefixes preserve chaining without executing uncertain suffixes."""
import copy

import pytest
from test_native_effect_pipeline import NativeAdapter
from test_ui_task import GOAL, phase

from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_recovery import NativeStrategyLedger
from qwen_omni_adapters.computer_use import ComputerUseError


def run(steps, *, hook=None):
    adapter = NativeAdapter()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task("session", private_boundaries=True)
    ledger = NativeStrategyLedger()
    callbacks = []
    with store.task_scope("session", "desktop", cancel, strategy_ledger=ledger):
        def execute_step(step, index):
            callbacks.append(copy.deepcopy(step))
            if index:
                assert ledger.view()["retained_entry_count"] or not adapter.calls
            if hook is not None:
                hook(adapter, cancel, index)
            return store.immediate_transition("session", step, goal=GOAL, objective="Write the requested exact values")
        result = run_verified_prefix(steps, execute_step=execute_step, cancelled=cancel.is_set,
            strategy_ledger=ledger, source_phase=1)
    return adapter, callbacks, ledger, result


def writes(*values):
    return [{"id": f"write-{index}", "verb": "set_value", "target": {"node_id": "draft"},
             "args": {"value": value}, "expect": [{"kind": "text", "text": "An incorrect model prediction"}]}
            for index, value in enumerate(values)]


def test_three_actual_verified_writes_chain_using_fresh_effects_and_no_model_oracle():
    adapter, callbacks, ledger, result = run(writes("first", "second", "third"))
    assert len(callbacks) == len(adapter.calls) == 3 and adapter.value == "third"
    assert [a["expect"][0]["value"] for a in adapter.calls] == ["first", "second", "third"]
    assert result["verified_prefix_receipt"]["verified_prefix_count"] == 3
    assert result["verified_prefix_receipt"]["all_immediate_readbacks_verified"] is True
    assert result["phase_handoff"]["terminal_verified"] is False
    assert result["phase_handoff"]["goal_verified"] is False
    assert result["task_progress"] is False
    before = [r["classification"] for r in ledger.view()["facts"]]
    ledger.record(result, source_phase=1)
    assert [r["classification"] for r in ledger.view()["facts"]] == before


def test_opaque_operation_executes_once_then_suffix_is_untouched():
    steps = [writes("first")[0], phase()["steps"][1], writes("forbidden suffix")[0]]
    adapter, callbacks, ledger, result = run(steps)
    assert len(callbacks) == len(adapter.calls) == 2
    assert adapter.value == adapter.saved == "first"
    assert result["status"] == "observed_unverified"
    receipt = result["verified_prefix_receipt"]
    assert (receipt["verified_prefix_count"], receipt["unattempted_step_count"]) == (1, 1)
    assert receipt["stop"] == "opaque_input_boundary"
    ledger.record(result, source_phase=1)
    assert ledger.view()["facts"][-1]["classification"] == "input_effect_uncertain"


def test_cancellation_between_operations_stops_remaining_input():
    def cancel_at_second(adapter, cancel, index):
        if index == 1:
            cancel.set()
    adapter, callbacks, _, result = run(writes("first", "second", "third"), hook=cancel_at_second)
    assert len(callbacks) == 2 and len(adapter.calls) == 1 and adapter.value == "first"
    assert result["status"] == "cancelled"


@pytest.mark.parametrize("steps", [[], writes(*[str(n) for n in range(26)]),
    [writes("valid")[0], {"id": "bad", "verb": "invented", "target": {"node_id": "draft"}, "expect": []}]])
def test_entire_plan_is_validated_before_any_input(steps):
    called = []
    with pytest.raises(ComputerUseError):
        run_verified_prefix(steps, execute_step=lambda step, index: called.append(step))
    assert not called


def test_callback_failure_keeps_verified_prefix_and_blocks_unknown_input_replay():
    ledger = NativeStrategyLedger()
    def execute_step(step, index):
        raise RuntimeError("Controlled callback failure; input delivery cannot be established")
    result = run_verified_prefix(writes("first", "second"), execute_step=execute_step, strategy_ledger=ledger)
    assert result["status"] == "needs_replan" and result["input_attempted"] is None
    assert result["verified_prefix_receipt"]["attempted_step_count"] == 1
    assert ledger.capacity_exhausted is True


def test_all_idempotent_readbacks_prove_no_new_inputs_or_task_progress():
    adapter, _, _, result = run(writes("", "", ""))
    assert not adapter.calls
    assert result["verified_prefix_receipt"]["all_immediate_readbacks_verified"] is True
    assert result["phase_handoff"]["new_effect_count"] == 0 and result["task_progress"] is False


def test_callback_cannot_claim_complete_without_a_verified_immediate_contract():
    called = []
    def execute_step(step, index):
        called.append(index)
        return {"status": "complete", "task_progress": False}
    result = run_verified_prefix(writes("first", "second"), execute_step=execute_step)
    assert called == [0] and result["status"] == "needs_replan"
    assert result["verified_prefix_receipt"]["all_immediate_readbacks_verified"] is False
