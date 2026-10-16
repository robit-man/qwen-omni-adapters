"""Typed focus/activation controls; CPU fixtures do not measure planner adherence."""
from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from portal.tools import tool_schemas
from portal.ui_task import phase_response_schema
from qwen_omni_adapters.computer_use import RapidUiController, predicate_satisfied


def node(identity, role, *, focused=None, operations=(), **fields):
    value = {"id": identity, "generation": "1", "scope_id": "window", "role": role,
             "name": "Repeated title", "visible": True, "enabled": True,
             "sensitive": False, "operations": list(operations), **fields}
    if focused is not None:
        value["focused"] = focused
    return value


def state(*nodes):
    return {"session_id": "session", "revision": "stable", "scope": {"id": "window", "kind": "desktop"},
            "nodes": list(nodes)}


@pytest.mark.parametrize("target,expected", [
    (node("window", "window", focused=False), (False, True)),
    (node("editor", "textbox", focused=True), (True, False)),
    (node("window", "window", focused=True), (True, True)),
    (node("inactive", "window", focused=True), (True, False)),
    (node("window", "window"), (False, True)),
    (node("window", "window", focused="true"), (False, True)),
    (node("window", "textbox", focused=False, name="Active window"), (False, False)),
    (node("inactive", "window", focused=False, name="Keyboard focus"), (False, False)),
    (node("window", "window", focused=True, sensitive=True), (False, False)),
])
def test_focus_and_activation_use_exact_typed_facts(target, expected):
    observed = state(target)
    original = copy.deepcopy(observed)
    assert tuple(predicate_satisfied({"kind": kind, "node_id": target["id"]}, observed)
                 for kind in ("focused", "active_window")) == expected
    assert not predicate_satisfied({"kind": "focused", "node_id": "unknown"}, observed)
    assert not predicate_satisfied({"kind": "active_window", "node_id": "unknown"}, observed)
    assert observed == original


class Adapter:
    def __init__(self, observed):
        self.observed, self.calls = observed, []

    def observe(self):
        return copy.deepcopy(self.observed)

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        return {"status": "dispatched", "backend": "x11_guarded_xtest"}


def test_same_title_windows_stay_ambiguous_and_do_not_acquire_focus_capability():
    observed = state(node("window", "window", focused=False, operations=["switch_window"]),
                     node("other", "window", focused=False, operations=["switch_window"]),
                     node("editor", "textbox", focused=True, operations=["focus"]))
    adapter = Adapter(observed)
    controller = RapidUiController(adapter)
    current = controller.observe()
    switch = {"id": "activate", "verb": "switch_window", "target": {"role": "window", "name": "Repeated title"},
              "args": {}, "expect": [{"kind": "active_window", "node_id": "$target"}]}
    actions = controller.enumerate(switch, current)
    assert {action["node_id"] for action in actions if action["kind"] == "action"} == {"window", "other"}
    focus = {"id": "focus", "verb": "focus", "target": {"node_id": "window"}, "args": {},
             "expect": [{"kind": "focused", "node_id": "$target"}]}
    assert not any(action["kind"] == "action" for action in controller.enumerate(focus, current))
    focus["target"] = {"node_id": "editor"}
    assert [action["node_id"] for action in controller.enumerate(focus, current)
            if action["kind"] == "action"] == ["editor"]
    assert adapter.calls == [] and adapter.observed == observed


def test_dispatched_click_does_not_prove_keyboard_focus_or_change_state():
    adapter = Adapter(state(node("window", "window", focused=False, operations=["click"])))
    controller = RapidUiController(adapter, authorize=lambda *_: True, max_observations=1)
    step = {"id": "click", "verb": "click", "target": {"node_id": "window"}, "args": {},
            "expect": [{"kind": "focused", "node_id": "$target"}]}
    outcome = controller.run_step(step)
    assert outcome["receipt"]["status"] == "dispatched" and len(adapter.calls) == 1
    assert outcome["status"] == "unresolved"
    assert outcome["before"] == outcome["after"] == adapter.observed
    assert predicate_satisfied({"kind": "active_window", "node_id": "window"}, outcome["after"])
    assert not predicate_satisfied(step["expect"][0] | {"node_id": "window"}, outcome["after"])


def test_shared_predicate_annotations_preserve_legal_provider_focused_windows():
    catalogs = {item["function"]["name"]: item["function"]["parameters"]
                for item in tool_schemas(["computer_use", "native_ui"])}
    branches = []
    for schema in [*catalogs.values(), phase_response_schema(["native_ui"])]:
        predicates = {item["properties"]["kind"]["const"]: item
                      for item in schema["$defs"]["predicate"]["oneOf"]}
        branches.append({kind: predicates[kind] for kind in ("focused", "active_window")})
        validator = Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/predicate"})
        for kind in ("focused", "active_window"):
            assert isinstance(predicates[kind].get("description"), str) and predicates[kind]["description"].strip()
            validator.validate({"kind": kind, "node_id": "provider-focused-window"})
            assert not validator.is_valid({"kind": kind, "node_id": ""})
            assert not validator.is_valid({"kind": kind, "node_id": "window", "focused": True})
    assert branches[0] == branches[1] == branches[2]
