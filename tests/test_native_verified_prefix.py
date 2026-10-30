"""Verified prefixes preserve chaining without executing uncertain suffixes."""
import copy
import hashlib

import pytest
from test_native_effect_pipeline import NativeAdapter
from test_ui_task import GOAL, phase

from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_recovery import NativeStrategyLedger
from qwen_omni_adapters.computer_use import ComputerUseError


def run(steps, *, hook=None, defer=False):
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
            return store.immediate_transition("session", step, goal=GOAL, objective="Write the requested exact values",
                defer_unannotated_boundary=defer and index > 0)
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


@pytest.mark.parametrize("verb,args", [
    ("click", {}), ("double_click", {}), ("right_click", {}), ("invoke", {}),
    ("key_chord", {"keys": ["Control_L", "a"]}), ("scroll", {"direction": "down"}),
    ("drag_drop", {"to_node_id": "draft"}),
])
def test_later_unannotated_opaque_step_yields_unattempted_without_a_failed_strategy(verb, args):
    current = {"id": "fresh-claim-needed", "verb": verb, "target": {"node_id": "commit"},
               "args": args, "expect": [{"kind": "text", "text": "An unverified model prediction"}]}
    steps = [writes("first")[0], current, writes("forbidden suffix")[0]]
    original = copy.deepcopy(steps)
    adapter, callbacks, ledger, result = run(steps, defer=True)
    assert steps == original and len(callbacks) == 2
    assert len(adapter.calls) == 1 and adapter.value == "first" and adapter.saved == ""
    assert not result.get("pointer_recovery") and not result["policy_receipts"] and not result["point_receipts"]
    receipt = result["verified_prefix_receipt"]
    assert receipt["stop"] == "unannotated_observation_boundary"
    assert (receipt["attempted_step_count"], receipt["verified_prefix_count"], receipt["unattempted_step_count"]) == (1, 1, 2)
    continuation = receipt["continuation"]
    assert continuation["step"] == current and continuation["step_index"] == 1
    assert continuation["objective"] == "Write the requested exact values"
    assert continuation["goal_sha256"] == hashlib.sha256(GOAL.encode()).hexdigest()
    assert continuation["input_authority"] is False and result["input_attempted"] is False
    assert result["effect_receipt"]["effect_state"] == "absent" and result["task_progress"] is False
    assert result["phase_handoff"]["new_effect_count"] == 1  # Only the preceding verified write.
    ledger.record(result, source_phase=1)
    assert [fact["classification"] for fact in ledger.view()["facts"]] == ["applied"]
    assert ledger.capacity_exhausted is False


def test_verified_suffixes_and_first_opaque_operation_keep_existing_behavior():
    adapter, _, _, result = run(writes("first", "second", "third"), defer=True)
    assert len(adapter.calls) == 3 and adapter.value == "third"
    assert result["verified_prefix_receipt"]["all_immediate_readbacks_verified"] is True
    assert "continuation" not in result["verified_prefix_receipt"]
    adapter, _, _, result = run([phase()["steps"][1], writes("suffix")[0]], defer=True)
    assert len(adapter.calls) == 1 and result["verified_prefix_receipt"]["attempted_step_count"] == 1
    assert result["verified_prefix_receipt"]["stop"] == "opaque_input_boundary"


def test_current_window_noop_then_opaque_yield_reaches_no_policy_grounding_or_input(monkeypatch):
    adapter = NativeAdapter()
    observe = adapter.observe
    def with_window():
        state = observe()
        state["nodes"].append({"id": "form", "generation": "1", "scope_id": "form", "role": "window",
            "name": "Owned workspace", "operations": ["switch_window"],
            "visible": True, "enabled": True, "sensitive": False})
        return state
    monkeypatch.setattr(adapter, "observe", with_window)
    def forbidden(*args, **kwargs):
        pytest.fail("An unattempted suffix cannot reach policy, grounding or input")
    monkeypatch.setattr(adapter, "prepare_intent", forbidden)
    monkeypatch.setattr(adapter, "perform", forbidden)
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    monkeypatch.setattr(store, "observation_boundary", forbidden)
    cancel = store.begin_task("session", private_boundaries=True)
    ledger = NativeStrategyLedger()
    steps = [{"id": "already-active", "verb": "switch_window", "target": {"node_id": "form"},
              "expect": [{"kind": "active_window", "node_id": "$target"}]}, phase()["steps"][1]]
    with store.task_scope("session", "desktop", cancel, strategy_ledger=ledger):
        result = run_verified_prefix(steps, execute_step=lambda step, index: store.immediate_transition(
            "session", step, goal=GOAL, objective="Read the supplied source", defer_unannotated_boundary=index > 0),
            strategy_ledger=ledger, source_phase=1)
    assert not adapter.calls and not result["outcomes"]
    receipt = result["verified_prefix_receipt"]
    assert receipt["attempted_step_count"] == 1 and receipt["continuation"]["step"] == steps[1]
    assert receipt["continuation"]["step_index"] == 1 and result["phase_handoff"]["new_effect_count"] == 0
    assert not ledger.capacity_exhausted and ledger.view()["retained_entry_count"] == 0


@pytest.mark.parametrize("contradiction", [
    {"input_attempted": True}, {"new_effect_count": False}, {"new_effect_count": 1},
    {"dispatch_record": {"dispatch_attempted": True}}, {"outcomes": [{"status": "verified"}]},
    {"policy_receipts": [{"status": "accepted"}]}, {"point_receipts": [{"status": "observed"}]},
])
def test_contradictory_callback_cannot_hide_an_attempt_as_unattempted(contradiction):
    _, _, _, deferred = run([writes("first")[0], phase()["steps"][1]], defer=True)
    deferred.pop("outcomes")
    continuation = deferred["verified_prefix_receipt"].pop("continuation")
    deferred["prefix_continuation"] = continuation
    deferred.update(copy.deepcopy(contradiction))
    def execute_step(step, index):
        if index == 0:
            return {"status": "complete", "immediate_effect_contract": {"mode": "verified_step"},
                "verification_only_receipt": {"status": "verified", "input_attempted": False},
                "phase_handoff": {"terminal_verified": True}}
        return deferred
    result = run_verified_prefix(writes("first", "second"), execute_step=execute_step)
    receipt = result["verified_prefix_receipt"]
    assert receipt["attempted_step_count"] == 2 and receipt["stop"] == "opaque_input_boundary"
    assert "continuation" not in receipt


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
