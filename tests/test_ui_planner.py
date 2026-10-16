"""CPU contracts for the planner/controller boundary, not live GUI acceptance."""
from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from portal.computer_use import ComputerUseStore
from portal.tools import tool_schemas
from portal.ui_planner import planner_call_error, planner_tool_schemas


def step(step_id="write"):
    return {"id": step_id, "verb": "set_value", "target": {"role": "textbox", "name": "Title"},
            "args": {"value": "Report"},
            "expect": [{"kind": "value", "node_id": "$target", "value": "Report"}]}


def workflow(*, duplicate_ids=False):
    return {"operation": "workflow", "surface": "desktop", "steps": [step("same" if duplicate_ids else "write"),
            {"id": "same" if duplicate_ids else "commit", "verb": "invoke",
             "target": {"role": "button", "name": "Commit"},
             "expect": [{"kind": "value", "node_id": "receipt", "value": "Saved Report"}]}],
            "final_predicates": [{"kind": "value", "node_id": "receipt", "value": "Saved Report"}]}


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
def test_projected_schema_removes_action_handoff_but_keeps_valid_phase_contract(name):
    canonical = tool_schemas([name])[0]
    original = copy.deepcopy(canonical)
    projected = planner_tool_schemas([canonical])[0]
    parameters = projected["function"]["parameters"]
    base = canonical["function"]["parameters"]
    assert canonical == original
    assert parameters["$defs"] == base["$defs"]
    assert not {"step", "action_id"} & parameters["properties"].keys()
    assert set(parameters["properties"]["operation"]["enum"]) == {"observe", "workflow", "verify", "cancel", "inspect_visual"}
    valid = [ {"operation": "observe"}, {"operation": "cancel"},
              {"operation": "verify", "predicates": [{"kind": "value", "node_id": "receipt", "value": "Saved Report"}]},
              workflow()]
    for arguments in valid:
        if name == "native_ui":
            arguments.pop("surface", None)
        Draft202012Validator(parameters).validate(arguments)
        assert planner_call_error(name, arguments) is None
    for arguments in ({"operation": "execute", "action_id": "a-real-offer"},
                      {"operation": "observe", "step": step()}):
        Draft202012Validator(base).validate(arguments)
        assert not Draft202012Validator(parameters).is_valid(arguments)
        assert planner_call_error(name, arguments)["dispatch_status"] == "not_dispatched"
    # Nested definitions are independently owned; the cached public catalog is untouched.
    parameters["$defs"].clear()
    assert canonical == original


def test_non_ui_contracts_and_direct_default_observe_are_preserved():
    schemas = tool_schemas(["memory_search", "native_ui"])
    before = copy.deepcopy(schemas)
    projected = planner_tool_schemas(schemas)
    assert projected[0] == before[0]
    assert projected[0] is not schemas[0]
    assert schemas == before
    assert planner_call_error("native_ui", {}) is None
    assert planner_call_error("memory_search", {"operation": ["execute"]}) is None


@pytest.mark.parametrize("operation", [[], {}, None, True, 4, "execute", "invented"])
def test_malformed_or_per_action_operations_fail_closed_without_shape_exceptions(operation):
    result = planner_call_error("native_ui", {"operation": operation, "action_id": "known"})
    assert result["error"] == "ui_planner_phase_required"
    assert result["dispatch_status"] == "not_dispatched"


class FormAdapter:
    def __init__(self, session, *, ambiguous=False, reject_commit=False):
        self.session = session
        self.value = self.receipt = ""
        self.revision = 0
        self.ambiguous, self.reject_commit = ambiguous, reject_commit
        self.events = []

    def observe(self):
        self.events.append(("observe", self.revision))
        nodes = [{"id": "title", "role": "textbox", "name": "Title", "value": self.value,
                  "description": "Intended local draft", "operations": ["set_value"]},
                 {"id": "commit", "role": "button", "name": "Commit", "operations": ["invoke"]},
                 {"id": "receipt", "role": "label", "name": "Receipt", "value": self.receipt, "operations": []}]
        if self.ambiguous:
            nodes.append({"id": "other", "role": "textbox", "name": "Title", "value": "",
                          "description": "Unrelated draft", "operations": ["set_value"]})
        return {"session_id": self.session, "revision": str(self.revision),
                "scope": {"id": "form", "kind": "desktop"}, "nodes": [
                    {"generation": "1", "scope_id": "form", "visible": True,
                     "enabled": True, "sensitive": False, **node} for node in nodes]}

    def perform(self, action):
        self.events.append(("perform", copy.deepcopy(action)))
        if action["verb"] == "set_value":
            assert action["node_id"] == "title"
            self.value = action["args"]["value"]
        elif self.reject_commit:
            return {"status": "rejected", "reason": "Commit unavailable"}
        else:
            self.receipt = "Saved " + self.value
        self.revision += 1
        return {"status": "dispatched"}


class AmbiguityPolicy:
    def __init__(self):
        self.calls = []

    def decide(self, state, actions, *, context):
        self.calls.append(copy.deepcopy((state, actions, context)))
        selected = next(action for action in actions if action.get("node_id") == "title")
        return {"choice": selected["id"], "receipt": {"status": "accepted", "model": "fast-policy"}}


def store_fixture(**kwargs):
    adapter = FormAdapter("session", **kwargs)
    store = ComputerUseStore(None, adapter_factory=lambda session, surface: adapter)
    return store, adapter


def test_one_phase_resolves_fresh_per_step_and_uses_policy_only_for_real_ambiguity():
    store, adapter = store_fixture(ambiguous=True)
    policy = AmbiguityPolicy()
    store.provider = policy
    result = store.act("session", workflow(), goal="Save the intended local draft.")
    assert result["status"] == "complete"
    performed = [event[1] for event in adapter.events if event[0] == "perform"]
    assert [action["verb"] for action in performed] == ["set_value", "invoke"]
    assert [action["revision"] for action in performed] == ["0", "1"]
    first, second = [index for index, event in enumerate(adapter.events) if event[0] == "perform"]
    assert any(event == ("observe", 1) for event in adapter.events[first + 1:second])
    assert adapter.events[-1] == ("observe", 2)  # Independent terminal observation.
    assert len(policy.calls) == 1
    state, actions, context = policy.calls[0]
    assert state["revision"] == "0" and context["goal"] == "Save the intended local draft."
    assert context["step"] == step()
    assert {action.get("node_id") for action in actions if action["kind"] == "action"} == {"title", "other"}
    assert result["outcomes"][1]["action"]["expect"][0]["node_id"] == "receipt"  # Effect is not the button.
    assert result["phase_handoff"] == {
        "schema": "robit.omni.ui-phase-handoff.v1", "executor": "rapid_ui_controller",
        "status": "phase_complete", "verified_steps": [{"index": 0, "id": "write"}, {"index": 1, "id": "commit"}],
        "remaining_steps": [], "new_effect_count": 2, "terminal_verified": True,
        "goal_verified": False, "next": "plan_next_phase"}


def test_phase_stops_at_first_failed_step_and_retains_unattempted_positions():
    store, adapter = store_fixture(reject_commit=True)
    arguments = workflow()
    arguments["steps"].append(step("later"))
    result = store.act("session", arguments)
    handoff = result["phase_handoff"]
    assert result["status"] == "needs_replan"
    assert handoff["verified_steps"] == [{"index": 0, "id": "write"}]
    assert handoff["remaining_steps"] == [{"index": 1, "id": "commit"}, {"index": 2, "id": "later"}]
    assert handoff["new_effect_count"] == 1 and handoff["terminal_verified"] is False
    assert len([event for event in adapter.events if event[0] == "perform"]) == 2


def test_verified_duplicate_step_ids_do_not_hide_independent_terminal_failure():
    store, _ = store_fixture()
    arguments = workflow(duplicate_ids=True)
    arguments["final_predicates"][0]["value"] = "Unobserved final effect"
    result = store.act("session", arguments)
    handoff = result["phase_handoff"]
    assert all(outcome["status"] == "verified" for outcome in result["outcomes"])
    assert result["status"] == "needs_replan"
    assert handoff["verified_steps"] == [{"index": 0, "id": "same"}, {"index": 1, "id": "same"}]
    assert handoff["remaining_steps"] == []
    assert handoff["status"] == "needs_replan" and handoff["next"] == "replan"
    assert handoff["terminal_verified"] is False and handoff["goal_verified"] is False


def test_cached_action_receipt_is_not_new_input_or_fresh_terminal_success():
    store, adapter = store_fixture()
    arguments = {"operation": "workflow", "surface": "desktop", "steps": [step()],
                 "final_predicates": [{"kind": "value", "node_id": "title", "value": "Report"}]}
    first = store.act("session", arguments)
    assert first["phase_handoff"]["new_effect_count"] == 1
    adapter.value, adapter.revision = "", 0  # Same bound preimage; historical intent remains idempotent.
    replay = store.act("session", arguments)
    assert replay["outcomes"][0]["action"]["idempotency_key"] == first["outcomes"][0]["action"]["idempotency_key"]
    assert replay["outcomes"][0]["effect_receipt"]["effect_state"] == "replayed"
    assert len([event for event in adapter.events if event[0] == "perform"]) == 1
    assert replay["phase_handoff"]["new_effect_count"] == 0
    assert replay["phase_handoff"]["terminal_verified"] is False
    assert replay["status"] == "needs_replan"
