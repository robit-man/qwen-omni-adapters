from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture, _switch_fixture, scoped_namespace_fixture

from portal.computer_use import ComputerUseStore, _effect_result
from portal.native_ui import NativeUiUnavailable
from qwen_omni_adapters.computer_use import RapidUiController


def activation(state):
    target = state["scope"]["id"]
    return {"id": "activate-current", "verb": "switch_window", "target": {"node_id": target},
            "args": {}, "expect": [{"kind": "active_window", "node_id": target}]}


def forbid_inputs(monkeypatch, fixture):
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *_a, **_kw: pytest.fail("No XTEST input"))
    monkeypatch.setattr("httpx.post", lambda *_a, **_kw: pytest.fail("No model inference"))
    monkeypatch.setattr(fixture.adapter, "_visual_capture", lambda *_a: pytest.fail("No visual capture"))
    fixture.bridge._mapped_window = lambda *_a: pytest.fail("No inactive window mapping")


def test_already_active_workflow_continues_real_typed_step_and_independent_terminal(monkeypatch):
    fixture = Fixture("already-active")
    forbid_inputs(monkeypatch, fixture)
    first = fixture.adapter.observe()
    title = next(n for n in first["nodes"] if n["name"] == "Title")
    policy = SimpleNamespace(decide=lambda *_a, **_kw: pytest.fail("No Kev call for unique controls"))
    store = ComputerUseStore(None, provider=policy, adapter_factory=lambda *_a: fixture.adapter)
    result = store.act("task", {"operation": "workflow", "surface": "desktop", "steps": [activation(first),
        {"id": "next-typed-step", "verb": "set_value", "target": {"node_id": title["id"]},
         "args": {"value": "Reviewed"}, "expect": [{"kind": "value", "node_id": "$target", "value": "Reviewed"}]}],
        "final_predicates": [{"kind": "value", "node_id": title["id"], "value": "Reviewed"}]})
    assert result["status"] == "complete" and result["terminal_receipt"]["status"] == "verified"
    noop, edited = result["outcomes"]
    assert noop["status"] == "verified" and noop["receipt"] == {
        "status": "already_satisfied", "input_attempted": False, "reason": "native_window_already_active"}
    assert noop["verification"]["observations"] == 1 and noop["after"] == noop["before"]
    assert noop["dispatch_status"] == "not_dispatched" and noop["effect_receipt"]["effect_state"] == "absent"
    assert noop["evidence_authority"] != "mutation" and not noop["task_progress"]
    assert edited["receipt"]["status"] == "dispatched" and fixture.title.events == [("set_value", "Reviewed")]
    assert len(result["phase_handoff"]["verified_steps"]) == 2 and result["phase_handoff"]["new_effect_count"] == 1
    assert result["policy_receipts"] == result["point_receipts"] == []


def test_noop_only_workflow_gets_no_input_or_mutation_credit(monkeypatch):
    fixture = Fixture()
    forbid_inputs(monkeypatch, fixture)
    store = ComputerUseStore(None, adapter_factory=lambda *_a: fixture.adapter)
    state = fixture.adapter.observe()
    step = activation(state)
    result = store.act("task", {"operation": "workflow", "surface": "desktop", "steps": [step],
                               "final_predicates": step["expect"]})
    assert result["status"] == "complete" and result["phase_handoff"]["new_effect_count"] == 0
    assert result["evidence_authority"] == "inspection" and not result["task_progress"]
    assert sum(o["receipt"]["status"] == "dispatched" for o in result["outcomes"]) == 0
    assert result["outcomes"][0]["effect_receipt"]["environment_may_have_changed"] is False


@pytest.mark.parametrize("change", ["session", "scope", "revision", "generation", "hidden", "disabled", "sensitive"])
def test_fresh_independent_noop_verification_rejects_binding_changes(change):
    class Adapter:
        calls = 0
        def observe(self):
            state = {"session_id": "session", "revision": "revision", "scope": {"kind": "desktop", "id": "window"},
                     "nodes": [{"id": "window", "generation": "1", "scope_id": "window", "role": "window",
                                "name": "Owned", "visible": True, "enabled": True, "sensitive": False,
                                "operations": ["switch_window"]}]}
            if self.calls:
                if change in {"session", "revision"}:
                    state["session_id" if change == "session" else change] = "changed"
                elif change == "scope":
                    state["scope"]["id"] = "other"
                else:
                    key = {"generation": "generation", "hidden": "visible", "disabled": "enabled", "sensitive": "sensitive"}[change]
                    state["nodes"][0][key] = "2" if change == "generation" else change == "sensitive"
            return state
        def perform(self, _action):
            self.calls += 1
            return {"status": "already_satisfied", "input_attempted": False}
    adapter = Adapter()
    control = RapidUiController(adapter, authorize=lambda *_a: True)
    result = control.run_step(activation(adapter.observe()))
    assert result["status"] == "rejected" and adapter.calls == 1
    assert result["receipt"]["status"] == "already_satisfied"
    assert _effect_result(result)["effect_receipt"]["effect_state"] == "absent"


@pytest.mark.parametrize("fault", ["missing_false", "zero_false", "true_input", "other_verb", "additional_predicate", "wrong_target_predicate"])
def test_new_receipt_is_not_a_general_skip_or_input_fallback(fault):
    class Adapter:
        calls = 0
        def observe(self):
            return {"session_id": "s", "revision": "r", "scope": {"kind": "desktop", "id": "w"},
                "nodes": [{"id": "w", "generation": "1", "scope_id": "w", "role": "window", "name": "Owned",
                           "visible": True, "enabled": True, "sensitive": False, "operations": ["switch_window", "focus"]}]}
        def perform(self, _action):
            self.calls += 1
            return {"status": "already_satisfied", **({} if fault == "missing_false" else {
                "input_attempted": 0 if fault == "zero_false" else fault == "true_input"})}
    adapter = Adapter()
    step = activation(adapter.observe())
    if fault == "other_verb":
        step["verb"] = "focus"
    elif fault == "additional_predicate":
        step["expect"].append({"kind": "visible", "name": "Owned", "visible": True})
    elif fault == "wrong_target_predicate":
        step["expect"][0]["node_id"] = "other"
    result = RapidUiController(adapter, authorize=lambda *_a: True).run_step(step)
    assert result["status"] == "rejected" and adapter.calls == 1
    assert result["reason"] == "Invalid already-satisfied window receipt"


def test_native_noop_rejects_extra_predicate_before_any_input(monkeypatch):
    fixture = Fixture()
    forbid_inputs(monkeypatch, fixture)
    step = activation(fixture.adapter.observe())
    step["expect"].append({"kind": "focused", "node_id": step["target"]["node_id"]})
    result = RapidUiController(fixture.adapter, authorize=lambda *_a: True).run_step(step)
    assert result["status"] == "rejected" and result["receipt"]["reason"] == "native_already_active_expectation_invalid"


@pytest.mark.parametrize("fault", ["host_denied", "cancelled", "provider_guard", "foreground", "active_state", "mapping"])
def test_native_noop_keeps_authority_cancellation_and_final_native_guards(monkeypatch, fault):
    fixture = Fixture()
    forbid_inputs(monkeypatch, fixture)
    control = RapidUiController(fixture.adapter, authorize=lambda *_a: fault != "host_denied")
    state = control.observe()
    action = next(a for a in control.enumerate(activation(state), state) if a["kind"] == "action")
    cancel = threading.Event()
    if fault == "cancelled":
        cancel.set()
    elif fault == "provider_guard":
        fixture.bridge._provider_guard = lambda: False
    elif fault in {"foreground", "active_state", "mapping"}:
        def guard():
            if fault == "foreground":
                fixture.foreground["window_id"] = "18"
            elif fault == "active_state":
                fixture.window.state.values.remove("ACTIVE")
            else:
                fixture.bridge.window_ids[state["scope"]["id"]] = "18"
            return True
        fixture.bridge._provider_guard = guard
    result = control.execute(action["id"], cancel)
    assert result["status"] in {"rejected", "cancelled"}
    assert result.get("receipt", {}).get("status") != "dispatched"


def test_noop_respects_full_namespace_and_scoped_identity_proof(monkeypatch):
    fixture, _proof, calls, _geometry, owners = scoped_namespace_fixture(monkeypatch)
    fixture.bridge._mapped_window = lambda *_a: pytest.fail("No inactive namespace mapping")
    state = fixture.adapter.observe()
    step = activation(state)
    result = RapidUiController(fixture.adapter, authorize=lambda *_a: True).run_step(step)
    assert result["status"] == "verified" and result["receipt"]["status"] == "already_satisfied"
    assert calls and owners and set(owners) == {"17"}
    assert result["before"] == result["after"]


@pytest.mark.parametrize("fault", ["out_of_scope", "reused_pid", "namespace_proof_changed"])
def test_noop_cannot_cross_scoped_or_namespace_identity(monkeypatch, fault):
    fixture, proof, _calls, _geometry, _owners = scoped_namespace_fixture(monkeypatch)
    fixture.bridge._mapped_window = lambda *_a: pytest.fail("No inactive namespace mapping")
    control = RapidUiController(fixture.adapter, authorize=lambda *_a: True)
    state = control.observe()
    action = next(a for a in control.enumerate(activation(state), state) if a["kind"] == "action")
    if fault == "out_of_scope":
        fixture.foreground["window_id"] = "18"
    elif fault == "reused_pid":
        monkeypatch.setattr("portal.native_ui._process_start", lambda _pid: "421")
    else:
        proof["digest"] = "b" * 64
    result = control.execute(action["id"])
    assert result["status"] in {"rejected", "unresolved"}
    assert result.get("receipt", {}).get("status") != "already_satisfied"


def test_uncertain_real_activation_is_never_replayed_as_already_satisfied(monkeypatch):
    fixture, _target, _other, _mapping, activations, control, action = _switch_fixture(monkeypatch)
    original = fixture.adapter.observe
    def observe():
        if activations:
            raise NativeUiUnavailable("native_observation_unavailable")
        return original()
    monkeypatch.setattr(fixture.adapter, "observe", observe)
    first = control.execute(action["id"])
    assert first["status"] == "unresolved" and first["receipt"]["status"] == "dispatched"
    assert control.execute(action["id"]) == first and len(activations) == 1


def test_noop_replay_retains_original_receipt_without_another_native_call(monkeypatch):
    fixture = Fixture()
    forbid_inputs(monkeypatch, fixture)
    control = RapidUiController(fixture.adapter, authorize=lambda *_a: True)
    state = control.observe()
    action = next(a for a in control.enumerate(activation(state), state) if a["kind"] == "action")
    first = control.execute(action["id"])
    assert first["status"] == "verified" and first["receipt"]["status"] == "already_satisfied"
    monkeypatch.setattr(fixture.adapter, "perform", lambda *_a: pytest.fail("No repeated native call"))
    monkeypatch.setattr(fixture.adapter, "observe", lambda: pytest.fail("Cached receipt is not a new input"))
    assert control.execute(action["id"]) == first
