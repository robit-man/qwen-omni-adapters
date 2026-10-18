"""Explicit planner objectives delegate legacy boundaries without input authority."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_ui_intent_choice import configured, subgoal
from test_ui_observation_boundary import Adapter, intent, write

from portal.computer_use import _digest
from portal.ui_task import _compact_result, _grounded_phase_schema, run_native_goal
from qwen_omni_adapters.computer_use import ComputerUseError

OBJECTIVE = "Inspect the supplied task brief before making changes"
GOAL = "Original exact user task"


def test_active_wrapper_requires_explicit_objective_in_private_schema():
    schema = _grounded_phase_schema({"native_ui"}, {"state": Adapter().observe()}, intent_choice=True)
    plan = {"kind": "ui_boundary", "step": {**intent(), "target": {"name": "Input"}}}
    validator = Draft202012Validator(schema)
    assert not validator.is_valid(plan)
    validator.validate({**plan, "objective": OBJECTIVE})


def test_positive_legacy_boundary_delegates_actual_objective_not_raw_primitive():
    adapter, store, cancel, requests = configured(selected_verb="double_click")
    step = {**intent(), "target": {"name": "Input", "role": "table_cell"}}
    original = copy.deepcopy(step)
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", step, goal=GOAL, objective=OBJECTIVE)
    assert len(requests) == len(adapter.calls) == 1
    assert adapter.calls[0]["verb"] == "double_click"
    task = json.loads(requests[0]["state"])["task"]
    assert task["goal"] == GOAL and task["active_objective"] == OBJECTIVE
    preference = result["intent_selection_receipt"]["planner_preference"]
    assert preference["intent"] == original == step
    assert preference["authority"] == "planner_preference" and preference["input_authority"] is False
    assert preference["effect"] == "unknown"
    assert preference["semantic_sha256"] == _digest(original)
    assert task["planner_preference"] == {key: value for key, value in preference.items() if key != "semantic_sha256"}
    assert result["task_progress"] is False and result["new_effect_count"] == 0


@pytest.mark.parametrize("objective,valid", [("", False), ("x", True), ("x" * 499, True),
    ("x" * 500, True), ("x" * 501, False), (" ", True), ("é\\\"\n", True)])
def test_schema_bounds_are_separate_from_nonblank_semantic_admission(objective, valid):
    schema = _grounded_phase_schema({"native_ui"}, {"state": Adapter().observe()}, intent_choice=True)
    plan = {"kind": "ui_boundary", "step": {**intent(), "target": {"name": "Input"}}, "objective": objective}
    assert Draft202012Validator(schema).is_valid(plan) is valid


@pytest.mark.parametrize("objective", [None, False, 1, {}, [], "", " \t\n", "x" * 501])
def test_invalid_explicit_objective_rejects_before_native_read_or_policy(objective):
    adapter, store, cancel, requests = configured()
    def forbidden():
        raise AssertionError("Invalid objective must not cause an observation")
    adapter.observe = forbidden
    with store.task_scope("session", "desktop", cancel), pytest.raises(ComputerUseError):
        store.observation_boundary("session", intent(), goal=GOAL, objective=objective)
    assert not adapter.prepared and not adapter.calls and not requests


def test_missing_active_callback_objective_does_not_derive_intent_from_labels():
    adapter, store, cancel, requests = configured()
    with store.task_scope("session", "desktop", cancel), pytest.raises(ComputerUseError):
        store.observation_boundary("session", {**intent(), "id": "Open the Input folder"}, goal=GOAL)
    assert not requests and not adapter.prepared and not adapter.calls and adapter.finished == 0


@pytest.mark.parametrize("objective", ["x", "x" * 500, "Inspect supplied material", "Show what is inside",
    "The task relevance of this target is uncertain", "This label says ignore the user: treat it as untrusted",
    "Do not confuse a Desktop shortcut with the task's Input folder"])
def test_exact_objective_is_data_not_a_lexical_primitive_router(objective):
    adapter, store, cancel, requests = configured(selected_verb="right_click")
    step = intent("click")
    step.pop("args")
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", step, goal=GOAL, objective=objective)
    task = json.loads(requests[0]["state"])["task"]
    assert task["active_objective"] == objective and task["goal"] == GOAL
    assert adapter.calls[0]["verb"] == "right_click"
    preference = result["intent_selection_receipt"]["planner_preference"]
    assert preference["intent"] == step and "args" not in preference["intent"]
    assert preference["semantic_sha256"] == _digest(step)
    assert _compact_result(result)["intent_selection_receipt"]["planner_preference"] == preference


@pytest.mark.parametrize("change", ["id", "target", "extra", "verb"])
def test_retained_preference_cannot_change_subgoal_binding_or_invent_effect(change):
    adapter, store, cancel, requests = configured()
    step = intent()
    if change == "id":
        step["id"] = "different"
    elif change == "target":
        step["target"] = {"name": "Other"}
    elif change == "extra":
        step["expect"] = [{"kind": "visible", "name": "Input", "visible": True}]
    else:
        step["verb"] = "type_text"
    with store.task_scope("session", "desktop", cancel), pytest.raises(ComputerUseError):
        store.intent_boundary("session", {**subgoal(), "id": "read-brief"}, goal=GOAL, preference=step)
    assert not adapter.calls and not requests


def test_ambiguous_legacy_selector_retains_referent_counts_and_finite_primitive_choices():
    adapter, store, cancel, requests = configured(selected_node="other")
    observe = adapter.observe
    def ambiguous():
        state = observe()
        state["nodes"].extend([
            {**copy.deepcopy(state["nodes"][0]), "id": "other"},
            {**copy.deepcopy(state["nodes"][0]), "id": "hidden", "visible": False},
            {**copy.deepcopy(state["nodes"][0]), "id": "protected", "sensitive": True, "value": "PRIVATE"},
        ])
        return state
    adapter.observe = ambiguous
    step = {**intent(), "target": {"name": "Input", "role": "table_cell"}}
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", step, goal=GOAL, objective=OBJECTIVE)
    task = json.loads(requests[0]["state"])["task"]
    assert task["selector_match_count"] == 4 and task["eligible_node_count"] == 2
    criteria = [json.loads(value) for value in requests[0]["questions"]["action"]["criteria"].values()]
    assert sum(item["kind"] == "intent" for item in criteria) == 8
    assert "PRIVATE" not in json.dumps(requests[0])
    assert adapter.calls[0]["node_id"] == "other"
    assert result["intent_selection_receipt"]["planner_preference"]["intent"] == step
    assert result["task_progress"] is False


@pytest.mark.parametrize("mode", [None, "shadow"])
@pytest.mark.parametrize("positive", [False, True])
def test_active_compiled_objective_stays_policy_required_after_mode_flip(mode, positive):
    adapter, store, cancel, requests = configured()
    # The compiler used the active schema; execution happens after host policy changed.
    step = {**intent(), "target": {"name": "Input" if positive else "Actually missing"}}
    Draft202012Validator(_grounded_phase_schema({"native_ui"}, {"state": adapter.observe()}, intent_choice=True)).validate(
        {"kind": "ui_boundary", "step": step, "objective": OBJECTIVE})
    if mode is None:
        store.provider = None
    else:
        store.provider.mode = mode
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", step, goal=GOAL, objective=OBJECTIVE)
    assert result["reason"] == "intent_policy_not_active"
    assert result["ui_boundary_receipt"]["input_attempted"] is False
    assert not adapter.prepared and not adapter.calls and not requests


def test_mode_flip_during_missing_target_preparation_cannot_dispatch_raw_preference():
    adapter, store, cancel, requests = configured()
    prepare = adapter.prepare_intent
    def flip(step):
        prepare(step)
        store.provider.mode = "shadow"
    adapter.prepare_intent = flip
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", {**intent(), "target": {"name": "Missing"}},
                                            goal=GOAL, objective=OBJECTIVE)
    assert result["error"] == "intent_policy_not_active"
    assert result["ui_boundary_receipt"]["input_attempted"] is False
    assert len(adapter.prepared) == 1 and not adapter.calls and not requests


@pytest.mark.parametrize("probability", [0.94, 0.5])
def test_policy_probability_is_not_relaxed_by_explicit_legacy_objective(probability):
    adapter, store, cancel, requests = configured(probability=probability)
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert len(requests) == 1 and not adapter.calls
    assert result["intent_selection_receipt"]["status"] == "intent_policy_abstained"
    assert result["ui_boundary_receipt"]["input_attempted"] is False


def test_positive_match_disappearing_before_choice_has_no_raw_verb_fallback():
    adapter, store, cancel, requests = configured()
    observe = adapter.observe
    reads = 0
    def disappears():
        nonlocal reads
        reads += 1
        state = observe()
        if reads > 1:
            state["nodes"].pop(0)
        return state
    adapter.observe = disappears
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert result["reason"] == "intent_candidates_unavailable"
    assert not requests and not adapter.prepared and not adapter.calls and reads == 2


@pytest.mark.parametrize("drift", ["model", "mode", "threshold", "cancel", "generation", "scope"])
def test_wrapped_selected_primitive_rechecks_provider_and_current_node_binding(drift):
    adapter, store, cancel, requests = configured()
    prepare, observe = adapter.prepare_intent, adapter.observe
    def changed(step):
        prepare(step)
        if drift == "cancel":
            cancel.set()
        elif drift in {"model", "mode", "threshold"}:
            setattr(store.provider, drift, {"model": "wrong", "mode": "shadow", "threshold": 0.5}[drift])
    def fresh():
        state = observe()
        if adapter.prepared and drift == "generation":
            state["nodes"][0]["generation"] = "replacement"
        if adapter.prepared and drift == "scope":
            state["scope"]["window_id"] = "other"
        return state
    adapter.prepare_intent, adapter.observe = changed, fresh
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert len(requests) == 1 and not adapter.calls
    assert result["ui_boundary_receipt"]["input_attempted"] is False


def test_wrapped_feedback_preserves_unknown_and_false_without_claiming_current_authority():
    adapter, store, cancel, requests = configured()
    feedback = {"status": "observed_unverified", "error": "opening_unknown", "reason": "Previous effect unknown",
        "effect_receipt": {"effect_state": "unknown", "environment_may_have_changed": True},
        "ui_boundary_receipt": {"input_attempted": None, "cached": False, "task_progress": False}}
    original = copy.deepcopy(feedback)
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE, feedback=feedback)
    previous = json.loads(requests[0]["state"])["task"]["previous_result"]
    assert previous["facts"] == feedback == original
    assert previous["temporal_scope"] == "historical" and previous["input_authority"] is False
    assert result["intent_selection_receipt"]["goal_sha256"] == _digest(GOAL)


@pytest.mark.parametrize("oversized", ["goal", "feedback"])
def test_wrapper_admission_never_trims_goal_or_historical_evidence(oversized):
    adapter, store, cancel, requests = configured()
    goal, feedback = GOAL, None
    if oversized == "goal":
        goal = "g" * 19000
    elif oversized == "feedback":
        feedback = {"effect_receipt": {"full": "x" * 9000}}
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=goal, objective=OBJECTIVE, feedback=feedback)
    assert not requests and not adapter.calls
    assert result["ui_boundary_receipt"]["input_attempted"] is False


def test_wrapper_marks_partial_shared_text_and_keeps_complete_native_guard_evidence():
    adapter, store, cancel, requests = configured()
    original = adapter.observe
    value = "x" * 4000
    def large():
        state = original()
        state["nodes"][0]["value"] = value
        return state
    adapter.observe = large
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert len(requests) == len(adapter.calls) == 1
    shared = json.loads(requests[0]["state"])
    assert shared["task"]["goal"] == GOAL and shared["task"]["active_objective"] == OBJECTIVE
    node = shared["nodes"]["n0"]
    assert "value" not in node["observed"]
    coverage = node["projection"]["omitted_fields"]["value"]
    assert coverage["complete"] is False and coverage["omission_is_absence"] is False
    assert coverage["source_characters"] == 4000 and coverage["preview"] == value[:256]
    assert coverage["source_sha256"] == _digest(value)
    assert max(map(len, requests[0]["questions"]["action"]["criteria"].values())) <= 1000
    before = result["dispatch_record"]["before"]
    observed = next(item for item in before["nodes"] if item["id"] == node["observed"]["id"])
    assert observed["value"] == value
    assert result["task_progress"] is False and result["new_effect_count"] == 0


def test_wrapped_policy_requires_confirmed_model_and_preserves_delegated_cancel():
    adapter, store, cancel, requests = configured(alter=lambda body: body.pop("model"))
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert len(requests) == 1 and not adapter.calls and result["status"] == "abstained"


def test_cancellation_at_internal_delegation_keeps_policy_fact_without_input():
    adapter, store, cancel, requests = configured()
    boundary = store.observation_boundary
    def late(*args, **kwargs):
        if "_selection" in kwargs:
            cancel.set()
        return boundary(*args, **kwargs)
    store.observation_boundary = late
    with store.task_scope("session", "desktop", cancel):
        result = store.observation_boundary("session", intent(), goal=GOAL, objective=OBJECTIVE)
    assert result["status"] == "cancelled" and result["task_progress"] is False
    assert len(result["policy_receipts"]) == len(requests) == 1 and not adapter.calls


@pytest.mark.parametrize("wrapped_first", [False, True])
def test_objective_and_wrapper_renaming_cannot_unlock_an_uncertain_actual_intent(wrapped_first):
    adapter, store, cancel, requests = configured()
    adapter.mutate = False
    def execute(wrapped, identity):
        with store.task_scope("session", "desktop", cancel):
            if wrapped:
                return store.observation_boundary("session", intent("click", identity), goal=GOAL,
                                                   objective=OBJECTIVE + identity)
            return store.intent_boundary("session", {**subgoal(), "id": identity}, goal=GOAL)
    first = execute(wrapped_first, "first")
    adapter.mutate = True
    with store.task_scope("session", "desktop", cancel):
        for value in ("temporary", ""):
            changed = store.act("session", {"operation": "workflow", "surface": "desktop", "steps": [write(value)],
                "final_predicates": [{"kind": "value", "node_id": "draft", "value": value}]})
            assert changed["status"] == "complete"
    adapter.mutate = False
    repeated = execute(not wrapped_first, "different-wrapper-and-id")
    assert len(requests) == 2 and len(adapter.calls) == 3
    assert repeated["dispatch_record"]["action"]["id"] == first["dispatch_record"]["action"]["id"]
    assert repeated["ui_boundary_receipt"]["cached"] is True
    assert repeated["ui_boundary_receipt"]["input_attempted"] is False


def test_latest_feedback_preference_and_policy_survive_next_packet_with_original_budget():
    adapter, store, cancel, requests = configured(probability=0.5)
    packets, executed = [], []
    step = {**intent(), "target": {"name": "Input", "role": "table_cell"}}
    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        return {"kind": "ui_boundary", "step": copy.deepcopy(step), "objective": OBJECTIVE}
    def execute(name, arguments):
        executed.append(copy.deepcopy(arguments))
        with store.task_scope("session", "desktop", cancel):
            return store.observation_boundary("session", arguments["intent"], goal=GOAL,
                objective=arguments["objective"], feedback=arguments["feedback"])
    events = list(run_native_goal(GOAL, observe=lambda: {"state": adapter.observe()}, compile_phase=compile_phase,
        execute=execute, cancel=cancel.set, cancelled=cancel.is_set, allowed_tools={"native_ui"}, intent_choice=True))
    final = events[-1]["response"]["portal"]["native_task"]
    assert final["status"] == "blocked" and final["phases"] == 3
    assert len(requests) == 3 and not adapter.calls
    assert executed[0]["feedback"] is None
    prior = json.loads(requests[1]["state"])["task"]["previous_result"]
    assert prior["facts"] == executed[1]["feedback"]
    assert prior["facts"]["ui_boundary_receipt"]["input_attempted"] is False
    assert prior["temporal_scope"] == "historical" and prior["input_authority"] is False
    last = packets[1]["last_result"]
    assert last["intent_selection_receipt"]["planner_preference"]["intent"] == step
    assert last["policy_receipts"][0]["decision_purpose"] == "primitive_intent"
    assert final["task_verified"] is False


@pytest.mark.parametrize("intent_choice", [False, True])
def test_runner_omits_new_arguments_when_no_active_choice_was_compiled(intent_choice):
    adapter = Adapter()
    arguments = []
    def execute(name, supplied):
        arguments.append(supplied)
        return {"status": "rejected"}
    step = {**intent(), "target": {"name": "Missing"}}
    plan = {"kind": "ui_boundary", "step": step, **({"objective": OBJECTIVE} if intent_choice else {})}
    list(run_native_goal(GOAL, observe=lambda: {"state": adapter.observe()}, compile_phase=lambda _: plan,
         execute=execute, cancel=lambda: None, allowed_tools={"native_ui"}, max_phases=1, intent_choice=intent_choice))
    assert arguments[0] == {"operation": "observation_boundary", "intent": step,
                           **({"objective": OBJECTIVE, "feedback": None} if intent_choice else {})}


@pytest.mark.parametrize("stream", [False, True])
def test_authenticated_route_forwards_actual_objective_and_raw_preference(tmp_path, monkeypatch, stream):
    import httpx
    from test_portal_native_task_routes import GOAL as task_goal
    from test_portal_native_task_routes import _config, _post, _request, _working_reply

    import portal.tools as tools_module
    from portal.app import DEFAULT_MODEL, create_app
    from portal.computer_use import ComputerUseStore

    _, configured_store, _, choices = configured()
    adapters, packets = [], []
    def factory(session, surface):
        adapter = Adapter()
        original = adapter.observe
        adapter.observe = adapter.planning_observe = lambda: {**original(), "session_id": session}
        adapters.append(adapter)
        return adapter
    store = ComputerUseStore(None, provider=configured_store.provider, adapter_factory=factory)
    monkeypatch.setattr(tools_module, "ComputerUseStore", lambda *args, **kwargs: store)
    step = {**intent(), "target": {"name": "Input", "role": "table_cell"}}
    def upstream(request):
        payload = json.loads(request.content)
        packet = json.loads(payload["messages"][1]["content"])
        packets.append(packet)
        assert packet["goal"] == task_goal and payload["think"] is False
        assert payload["options"]["num_predict"] == 437
        assert "Every ui_boundary also requires your explicit objective" in payload["messages"][0]["content"]
        assert payload["response_format"]["json_schema"]["schema"] == packet["response_schema"]
        if len(packets) == 1:
            plan = {"kind": "ui_boundary", "step": copy.deepcopy(step), "objective": OBJECTIVE}
        else:
            assert packet["last_result"]["intent_selection_receipt"]["planner_preference"]["intent"] == step
            assert packet["last_result"]["policy_receipts"][0]["decision_purpose"] == "primitive_intent"
            plan = {"kind": "blocked", "reason": "CPU fixture ends without task completion"}
        return httpx.Response(200, json={"model": DEFAULT_MODEL, "message": {"role": "assistant", "content": json.dumps(_working_reply(request, plan))}})
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(stream), stream)
    assert len(packets) == 2 and len(choices) == len(adapters[0].calls) == 1
    assert json.loads(choices[0]["state"])["task"]["goal"] == task_goal
    assert result["portal"]["native_task"]["task_verified"] is False
