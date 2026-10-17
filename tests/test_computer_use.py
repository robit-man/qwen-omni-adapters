from __future__ import annotations

import copy
import threading
import time

import pytest

from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    RapidUiController,
    parse_step,
    predicate_satisfied,
    run_workflow,
)


def snapshot() -> dict:
    return {
        "session_id": "fixture-session", "revision": "r1",
        "scope": {"id": "owned-page", "kind": "browser", "url": "https://example.test/"},
        "nodes": [{"id": "field", "generation": "g1", "scope_id": "main-frame",
                   "role": "textbox", "name": "Title", "visible": True, "enabled": True,
                   "sensitive": False, "operations": ["set_value", "focus"], "value": ""}],
        "text": "Ready",
    }


def step(value: str = "Report", identity: str = "set-title") -> dict:
    return {"id": identity, "verb": "set_value", "target": {"role": "textbox", "name": "Title"},
            "args": {"value": value}, "expect": [{"kind": "value", "node_id": "$target", "value": value}]}


@pytest.mark.parametrize("verb,fields", [("select_text", "end, start"), ("key_chord", "keys"), ("set_value", "value"), ("type_text", "text")])
def test_missing_args_reports_exact_required_fields_without_guessing_values(verb, fields):
    request = {**step(), "verb": verb}
    request.pop("args")
    with pytest.raises(ComputerUseError) as error:
        parse_step(request)
    assert str(error.value) == f"{verb} args must contain exactly these fields: {fields}"


class Adapter:
    def __init__(self) -> None:
        self.state = snapshot()
        self.dispatched: list[dict] = []
        self.reads = 0
        self.mutate = True
        self.fail_after_input = False
        self.observation_hook = None
        self.perform_hook = None

    def observe(self) -> dict:
        self.reads += 1
        if self.observation_hook:
            self.observation_hook(self)
        return copy.deepcopy(self.state)

    def perform(self, action: dict) -> dict:
        self.dispatched.append(copy.deepcopy(action))
        if self.perform_hook:
            self.perform_hook(self)
        if self.mutate:
            self.state["nodes"][0]["value"] = action["args"]["value"]
            self.state["revision"] = f"r{len(self.dispatched) + 1}"
            self.state["text"] = f"Applied {action['args']['value']}"
        if self.fail_after_input:
            raise RuntimeError("transport disconnected after delivery")
        return {"status": "dispatched"}


def controller(adapter: Adapter, **kwargs) -> RapidUiController:
    return RapidUiController(adapter, authorize=lambda _action, _state: True,
                             observation_interval_s=0.001, **kwargs)


def candidate(control: RapidUiController, request: dict | None = None) -> dict:
    current = control.observe()
    return next(action for action in control.enumerate(request or step(), current) if action["kind"] == "action")


def test_finite_candidates_bind_exact_arguments_and_current_target() -> None:
    adapter = Adapter()
    control = controller(adapter)
    action = candidate(control)
    action["args"]["value"] = "Injected"
    action["expect"][0]["value"] = "Injected"
    selected = control.candidate(action["id"])
    assert selected["args"] == {"value": "Report"}
    assert selected["expect"] == [{"kind": "value", "node_id": "field", "value": "Report"}]
    outcome = control.execute(action["id"])
    assert outcome["status"] == "verified"
    assert adapter.dispatched[0]["args"] == {"value": "Report"}
    assert outcome["verification"]["observations"] == 1


def test_no_host_authority_denies_even_confident_policy() -> None:
    adapter = Adapter()
    control = RapidUiController(adapter)
    outcome = control.run_step(step(), lambda _state, actions: {"choice": actions[-1]["id"], "confidence": 1})
    assert outcome["status"] == "rejected"
    assert "authority" in outcome["reason"]
    assert adapter.dispatched == []


@pytest.mark.parametrize("truthy", [1, "yes", [True]])
def test_authorizer_must_return_exact_boolean_true(truthy) -> None:
    adapter = Adapter()
    control = RapidUiController(adapter, authorize=lambda _action, _state: truthy)
    assert control.run_step(step())["status"] == "rejected"
    assert adapter.dispatched == []


@pytest.mark.parametrize("change", ["revision", "generation", "session", "scope", "disabled", "hidden", "sensitive", "operation", "missing"])
def test_stale_or_ineligible_target_is_rejected_before_input(change: str) -> None:
    adapter = Adapter()
    control = controller(adapter)
    action = candidate(control)
    if change == "revision":
        adapter.state["revision"] = "r-other"
    elif change == "generation":
        adapter.state["nodes"][0]["generation"] = "g-other"
    elif change == "session":
        adapter.state["session_id"] = "another-session"
    elif change == "scope":
        adapter.state["scope"]["id"] = "another-page"
    elif change == "disabled":
        adapter.state["nodes"][0]["enabled"] = False
    elif change == "hidden":
        adapter.state["nodes"][0]["visible"] = False
    elif change == "sensitive":
        adapter.state["nodes"][0]["sensitive"] = True
    elif change == "operation":
        adapter.state["nodes"][0]["operations"] = []
    else:
        adapter.state["nodes"] = []
    assert control.execute(action["id"])["status"] == "rejected"
    assert adapter.dispatched == []


def test_native_step_redecides_once_on_unrelated_pre_input_revision_change():
    adapter = Adapter()
    control = controller(adapter)
    control.enable_native_task_protocol()
    decisions = []

    def choose(state, actions):
        selected = next(a for a in actions if a["kind"] == "action")
        decisions.append((state["revision"], selected["id"]))
        if len(decisions) == 1:
            adapter.state.update(revision="background-update", text="Unrelated content updated")
        return selected["id"]

    outcome = control.run_step(step(), choose)
    assert outcome["status"] == "verified"
    assert [r for r, _ in decisions] == ["r1", "background-update"]
    assert len(adapter.dispatched) == 1
    assert adapter.dispatched[0]["id"] == decisions[1][1]
    assert outcome["pre_input_refresh"]["rejected_action_id"] == decisions[0][1]
    rejected = control.execute(decisions[0][1])
    assert rejected["status"] == "rejected"
    assert rejected["dispatch_attempted"] is False
    assert len(adapter.dispatched) == 1


@pytest.mark.parametrize("change", ["generation", "value", "focused", "bounds", "scope", "missing"])
def test_native_refresh_does_not_renew_changed_operands(change):
    adapter = Adapter()
    control = controller(adapter)
    control.enable_native_task_protocol()
    calls = []

    def choose(state, actions):
        calls.append(state)
        adapter.state["revision"] = "changed"
        node = adapter.state["nodes"][0]
        if change == "scope":
            adapter.state["scope"]["url"] = "https://another.test/"
        elif change == "missing":
            adapter.state["nodes"] = []
        elif change == "bounds":
            node["bounds"] = {"x": 10, "y": 10, "width": 20, "height": 20}
        else:
            node[change] = True if change == "focused" else "changed"
        return next(a["id"] for a in actions if a["kind"] == "action")

    result = control.run_step(step(), choose)
    assert result["status"] == "rejected"
    assert "pre_input_refresh" not in result
    assert len(calls) == 1
    assert adapter.dispatched == []


def test_native_refresh_stops_after_second_pre_input_rejection():
    adapter = Adapter()
    control = controller(adapter)
    control.enable_native_task_protocol()
    calls = []

    def choose(state, actions):
        calls.append(state)
        adapter.state["revision"] = "churn-" + str(len(calls))
        return next(a["id"] for a in actions if a["kind"] == "action")

    result = control.run_step(step(), choose)
    assert result["status"] == "rejected"
    assert len(calls) == 2
    assert result["dispatch_attempted"] is False
    assert adapter.dispatched == []


@pytest.mark.parametrize("mode", ["public", "boundary"])
def test_public_steps_and_policy_bound_boundaries_do_not_automatically_refresh(mode):
    adapter = Adapter()
    adapter.prepare_intent = lambda _step: None
    control = controller(adapter)
    if mode == "boundary":
        control.enable_native_task_protocol()
    calls = []

    def choose(state, actions):
        calls.append(state)
        adapter.state["revision"] = "changed"
        return next(a["id"] for a in actions if a["kind"] == "action")

    request = step()
    if mode == "boundary":
        request.pop("expect")
        request.update(verb="click", args={})
        adapter.state["nodes"][0]["operations"].append("click")
        outcome = control.run_boundary(request, choose)
    else:
        outcome = control.run_step(request, choose)
    assert outcome["status"] == "rejected"
    assert len(calls) == 1
    assert adapter.dispatched == []


@pytest.mark.parametrize("failure", ["uncertain", "no_effect", "adapter_rejection"])
def test_native_refresh_never_retries_after_entering_dispatch(failure):
    adapter = Adapter()
    control = controller(adapter, max_observations=1)
    control.enable_native_task_protocol()
    if failure == "uncertain":
        adapter.fail_after_input = True
    elif failure == "no_effect":
        adapter.mutate = False
    else:
        def reject(action):
            adapter.dispatched.append(action)
            return {"status": "rejected", "reason": "ui_revision_changed"}
        adapter.perform = reject
    calls = []

    def choose(state, actions):
        calls.append(state)
        return next(a["id"] for a in actions if a["kind"] == "action")

    result = control.run_step(step(), choose)
    assert result["status"] in {"unresolved", "rejected"}
    assert "pre_input_refresh" not in result
    assert len(calls) == len(adapter.dispatched) == 1


def test_workflow_refresh_keeps_policy_bound_to_current_step():
    adapter = Adapter()
    control = controller(adapter)
    control.enable_native_task_protocol()
    factories, decisions = [], []

    def bind(index, request):
        factories.append(index)
        def choose(state, actions):
            decisions.append((index, request["id"]))
            if len(decisions) == 1:
                adapter.state["revision"] = "background"
            return next(a["id"] for a in actions if a["kind"] == "action")
        return choose

    result = control.run_workflow([step("First", "first"), step("Second", "second")],
        final_predicates=[{"kind": "value", "node_id": "field", "value": "Second"}], provider_for_step=bind)
    assert result["status"] == "complete"
    assert factories == [0, 1]
    assert decisions == [(0, "first"), (0, "first"), (1, "second")]
    assert [a["args"]["value"] for a in adapter.dispatched] == ["First", "Second"]


@pytest.mark.parametrize("change", ["cancel", "authority", "focus_during_refresh", "old_offer"])
def test_fresh_native_decision_preserves_cancellation_authority_and_offer_guards(change):
    adapter = Adapter()
    control = controller(adapter)
    control.enable_native_task_protocol()
    cancel = threading.Event()
    calls = []

    def observe(current):
        if change == "focus_during_refresh" and current.reads == 3:
            current.state["nodes"][0]["focused"] = True
    adapter.observation_hook = observe

    def choose(state, actions):
        choice = next(a["id"] for a in actions if a["kind"] == "action")
        calls.append(choice)
        if len(calls) == 1:
            adapter.state["revision"] = "background"
        elif change == "cancel":
            cancel.set()
        elif change == "authority":
            control.authorize = lambda *_: False
        elif change == "old_offer":
            return calls[0]
        return choice

    outcome = control.run_step(step(), choose, cancel)
    assert outcome["status"] in {"cancelled", "rejected", "unresolved"}
    assert len(calls) == (1 if change == "focus_during_refresh" else 2)
    assert adapter.dispatched == []


def test_successful_dispatch_without_expected_effect_is_unresolved_and_not_replayed() -> None:
    adapter = Adapter()
    adapter.mutate = False
    control = controller(adapter, max_observations=3)
    action = candidate(control)
    first = control.execute(action["id"])
    assert first["status"] == "unresolved"
    assert first["verification"]["observations"] == 3
    assert control.execute(action["id"]) == first
    assert len(adapter.dispatched) == 1


@pytest.mark.parametrize("mutate", [False, True])
def test_set_value_requires_its_value_even_when_planner_only_checks_existing_focus(mutate):
    adapter = Adapter()
    adapter.state["nodes"][0]["focused"] = True
    adapter.mutate = mutate
    control = controller(adapter, max_observations=2)
    control.enable_native_task_protocol()
    request = {**step(), "expect": [{"kind": "focused", "node_id": "$target"}]}
    outcome = control.run_step(request)
    assert outcome["status"] == ("verified" if mutate else "unresolved")
    assert {"kind": "value", "node_id": "field", "value": "Report"} in outcome["action"]["expect"]
    assert control.execute(outcome["action"]["id"]) == outcome
    if not mutate:
        assert control.run_step(request) == outcome
    assert len(adapter.dispatched) == 1


@pytest.mark.parametrize("observed", ["", "Repo", "Report extra"])
def test_set_value_cannot_verify_a_different_value_with_a_satisfied_extra_predicate(observed):
    adapter = Adapter()
    adapter.mutate = False
    adapter.state["nodes"][0].update(value=observed, focused=True)
    result = controller(adapter, max_observations=1).run_step(
        {**step(), "expect": [{"kind": "focused", "node_id": "$target"}]})
    assert result["status"] == "unresolved"


def test_value_predicate_does_not_treat_a_truncated_matching_prefix_as_full_value():
    state = snapshot()
    state["nodes"][0].update(value="Report", text_observation={
        "character_count": 12, "captured_range": {"start": 0, "end": 6}, "truncated": True})
    assert not predicate_satisfied({"kind": "value", "node_id": "field", "value": "Report"}, state)


def test_model_additional_value_constraint_cannot_override_requested_value():
    adapter = Adapter()
    result = controller(adapter, max_observations=1).run_step(
        {**step(), "expect": [{"kind": "value", "node_id": "$target", "value": "Different"}]})
    assert result["status"] == "unresolved"
    assert adapter.state["nodes"][0]["value"] == "Report"


def test_mandatory_value_effect_preserves_candidate_predicate_budget():
    adapter = Adapter()
    result = controller(adapter).run_step({**step(), "expect": [
        {"kind": "focused", "node_id": "$target"} for _ in range(20)]})
    assert result["status"] == "unresolved" and "20 predicates" in result["reason"]
    assert not adapter.dispatched


def test_exception_after_input_is_not_replayed_even_after_new_enumeration() -> None:
    adapter = Adapter()
    adapter.mutate = False
    adapter.fail_after_input = True
    control = controller(adapter)
    first = control.run_step(step())
    second = control.run_step(step())
    assert first["status"] == second["status"] == "unresolved"
    assert first["action"]["id"] == second["action"]["id"]
    assert len(adapter.dispatched) == 1


def test_cancellation_after_dispatch_never_repeats_input() -> None:
    adapter = Adapter()
    cancel = threading.Event()
    adapter.perform_hook = lambda _adapter: cancel.set()
    control = controller(adapter)
    action = candidate(control)
    first = control.execute(action["id"], cancel)
    assert first["status"] == "cancelled"
    cancel.clear()
    assert control.execute(action["id"]) == first
    assert len(adapter.dispatched) == 1


def test_pre_cancelled_input_dispatches_nothing() -> None:
    adapter = Adapter()
    cancel = threading.Event()
    cancel.set()
    control = controller(adapter)
    action = candidate(control)
    assert control.execute(action["id"], cancel)["status"] == "cancelled"
    cancel.clear()
    assert control.execute(action["id"])["status"] == "cancelled"
    assert adapter.dispatched == []


def test_simultaneous_duplicate_stays_unknown_and_dispatches_once() -> None:
    adapter = Adapter()
    entered = threading.Event()
    release = threading.Event()

    def held(_adapter):
        entered.set()
        assert release.wait(2)

    adapter.perform_hook = held
    control = controller(adapter)
    action = candidate(control)
    results = []
    worker = threading.Thread(target=lambda: results.append(control.execute(action["id"])))
    worker.start()
    assert entered.wait(2)
    try:
        duplicate = control.execute(action["id"])
        assert duplicate["status"] == "unresolved"
        assert "unverified" in duplicate["reason"]
        assert len(adapter.dispatched) == 1
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert results[0]["status"] == "verified"


def test_sensitive_values_are_removed_and_no_action_can_target_them() -> None:
    adapter = Adapter()
    adapter.state["nodes"][0].update(sensitive=True, value="secret", selected=["secret"])
    control = controller(adapter)
    state = control.observe()
    assert "value" not in state["nodes"][0]
    assert "selected" not in state["nodes"][0]
    assert state["nodes"][0]["operations"] == []
    actions = control.enumerate(step(), state)
    assert [action["kind"] for action in actions] == ["observe", "abstain"]
    assert control.run_step(step())["status"] == "abstained"


def test_ambiguous_targets_abstain_without_a_policy() -> None:
    adapter = Adapter()
    duplicate = copy.deepcopy(adapter.state["nodes"][0])
    duplicate["id"] = "second-field"
    adapter.state["nodes"].append(duplicate)
    control = controller(adapter)
    assert control.run_step(step())["status"] == "abstained"
    assert adapter.dispatched == []


def test_scope_narrowing_resolves_ambiguous_controls() -> None:
    adapter = Adapter()
    duplicate = copy.deepcopy(adapter.state["nodes"][0])
    duplicate.update(id="second-field", scope_id="iframe")
    adapter.state["nodes"].append(duplicate)
    request = step()
    request["target"]["scope_id"] = "main-frame"
    assert controller(adapter).run_step(request)["status"] == "verified"


def test_provider_cannot_substitute_unknown_action_or_mutate_candidates() -> None:
    adapter = Adapter()
    control = controller(adapter)
    assert control.run_step(step(), lambda _state, _actions: "invented-input")["status"] == "unresolved"

    def policy(_state, actions):
        selected = actions[-1]
        selected["args"]["value"] = "Injected"
        return {"choice": selected["id"]}

    assert control.run_step(step(), policy)["status"] == "verified"
    assert adapter.dispatched[0]["args"]["value"] == "Report"


def test_supplied_snapshot_cannot_forge_current_nodes() -> None:
    adapter = Adapter()
    control = controller(adapter)
    state = control.observe()
    state["nodes"][0]["name"] = "Invented"
    with pytest.raises(ComputerUseError, match="current UI"):
        control.enumerate(step(), state)


def test_old_candidate_ids_are_invalidated_by_new_enumeration() -> None:
    adapter = Adapter()
    control = controller(adapter)
    old = candidate(control)
    new = candidate(control)
    assert old["id"] != new["id"]
    assert control.execute(old["id"])["status"] == "rejected"
    assert adapter.dispatched == []


def test_delayed_effect_uses_bounded_observation_instead_of_repeating_input() -> None:
    adapter = Adapter()
    adapter.mutate = False

    def delayed(current):
        if current.dispatched and current.reads >= 4:
            current.state["nodes"][0]["value"] = "Report"
            current.state["revision"] = "r2"

    adapter.observation_hook = delayed
    result = controller(adapter, max_observations=4).run_step(step())
    assert result["status"] == "verified"
    assert result["verification"]["observations"] == 2
    assert len(adapter.dispatched) == 1


def test_verification_deadline_prevents_late_success() -> None:
    adapter = Adapter()

    def slow_after_input(current):
        if current.dispatched:
            time.sleep(0.02)

    adapter.observation_hook = slow_after_input
    result = controller(adapter, verification_timeout_s=0.005).run_step(step())
    assert result["status"] == "unresolved"
    assert result["verification"]["observations"] == 1


def test_workflow_requires_fresh_independent_terminal_verification() -> None:
    adapter = Adapter()
    control = controller(adapter)
    result = run_workflow(control, [step()], final_predicates=[{"kind": "text", "text": "Applied Report"}])
    assert result["status"] == "complete"
    assert adapter.reads == 4
    assert result["terminal_receipt"]["revision"] == "r2"
    assert result["terminal_receipt"]["action_ids"] == [result["outcomes"][0]["action"]["id"]]


def test_verified_clicks_do_not_satisfy_unmet_terminal_criteria() -> None:
    adapter = Adapter()
    result = controller(adapter).run_workflow([step()], final_predicates=[{"kind": "text", "text": "Server persisted Report"}])
    assert result["outcomes"][0]["status"] == "verified"
    assert result["status"] == "needs_replan"
    assert "terminal_receipt" not in result


def test_fresh_terminal_observation_rejects_prior_success_and_scope_changes() -> None:
    adapter = Adapter()

    def changed_terminal(current):
        if current.reads >= 4:
            current.state["scope"]["id"] = "unrelated-page"

    adapter.observation_hook = changed_terminal
    result = controller(adapter).run_workflow([step()], final_predicates=[{"kind": "text", "text": "Applied Report"}])
    assert result["status"] == "needs_replan"
    assert "another UI session" in result["reason"]


def test_workflow_stops_at_first_unverified_effect() -> None:
    adapter = Adapter()
    adapter.mutate = False
    result = controller(adapter, max_observations=1).run_workflow(
        [step(), step("Next", "next-step")], final_predicates=[{"kind": "text", "text": "Next"}])
    assert result["status"] == "needs_replan"
    assert len(result["outcomes"]) == len(adapter.dispatched) == 1


def test_workflow_validates_all_steps_and_terminal_criteria_before_input() -> None:
    adapter = Adapter()
    control = controller(adapter)
    bad = step()
    bad["args"]["code"] = "arbitrary code"
    with pytest.raises(ComputerUseError):
        control.run_workflow([step(), bad], final_predicates=[{"kind": "text", "text": "Applied Report"}])
    with pytest.raises(ComputerUseError):
        control.run_workflow([step()], final_predicates=[])
    with pytest.raises(ComputerUseError):
        control.run_workflow([step()], final_predicates=step()["expect"])
    with pytest.raises(ComputerUseError):
        control.run_workflow([step()] * 26, final_predicates=[{"kind": "text", "text": "Report"}])
    assert adapter.dispatched == []


@pytest.mark.parametrize("modify", [
    lambda value: value.update(verb="execute_code"),
    lambda value: value.update(expect=[]),
    lambda value: value.update(expect=[{"kind": "download_saved", "filename": "report.txt"}]),
    lambda value: value.update(expect=[{"kind": ["text"], "text": "Report"}]),
    lambda value: value.update(expect=[{"kind": "checked", "node_id": "$target", "checked": 1}]),
    lambda value: value["target"].update(role=""),
    lambda value: value.update(target={"scope_id": "main-frame"}),
    lambda value: value["args"].update(keys=["Enter"]),
    lambda value: value.update(args={}),
    lambda value: value.update(permissions=["allow"]),
])
def test_step_validation_rejects_unbounded_or_forged_contracts(modify) -> None:
    value = step()
    modify(value)
    with pytest.raises(ComputerUseError):
        parse_step(value)


def test_typed_predicates_require_actual_state_and_never_a_dispatch_receipt() -> None:
    state = snapshot()
    state["nodes"][0].update(checked=False, selected=["A"], focused=True)
    assert predicate_satisfied({"kind": "checked", "node_id": "field", "checked": False}, state)
    assert predicate_satisfied({"kind": "selected", "node_id": "field", "values": ["A"]}, state)
    assert predicate_satisfied({"kind": "focused", "node_id": "field"}, state)
    assert predicate_satisfied({"kind": "url", "url": "https://example.test/"}, state)
    assert predicate_satisfied({"kind": "visible", "name": "Missing", "visible": False}, state)
    assert not predicate_satisfied({"kind": "text", "text": "Saved"}, state, {"status": "dispatched"})
    assert not predicate_satisfied({"kind": "download_saved"}, state, {"status": "dispatched", "verified": True})


def test_truncated_observation_cannot_prove_a_control_absent() -> None:
    state = snapshot()
    state["truncated"] = True
    assert not predicate_satisfied({"kind": "visible", "name": "Missing", "visible": False}, state)
    assert predicate_satisfied({"kind": "visible", "name": "Title", "visible": True}, state)
    state["truncated"] = False
    assert predicate_satisfied({"kind": "visible", "name": "Missing", "visible": False}, state)


@pytest.mark.parametrize("bad", [
    {"max_observations": 0}, {"max_observations": True},
    {"verification_timeout_s": 0}, {"verification_timeout_s": float("nan")},
    {"observation_interval_s": 2}, {"authorize": "allow"},
])
def test_controller_rejects_invalid_budgets(bad) -> None:
    with pytest.raises(ComputerUseError):
        RapidUiController(Adapter(), **bad)
