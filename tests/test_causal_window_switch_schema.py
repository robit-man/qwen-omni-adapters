"""Private planner window effects bind to the window actually selected."""
from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from portal import ui_task
from qwen_omni_adapters.computer_use import RapidUiController


def state():
    return {"session_id": "owned", "revision": "before", "scope": {"id": "editor", "kind": "desktop"},
            "nodes": [{"id": identity, "generation": "1", "scope_id": identity, "role": "window",
                       "name": name, "visible": True, "enabled": True, "sensitive": False,
                       "focused": False, "operations": ["switch_window", "key_chord"]}
                      for identity, name in (("editor", "Draft"), ("files", 'Files café "owned"'))]}


def switch(target=None, effects=None):
    return {"id": "switch", "verb": "switch_window", "target": target or {"node_id": "files"},
            "args": {}, "expect": effects if effects is not None else [{"kind": "active_window", "node_id": "$target"}]}


def phase(steps):
    return {"kind": "ui", "steps": steps, "final_predicates": [{"kind": "active_window", "node_id": "files"}]}


def validator():
    return Draft202012Validator(ui_task._grounded_phase_schema({"native_ui"}, {"state": state()}))


@pytest.mark.parametrize("position", [1, 2, 25])
@pytest.mark.parametrize("effects", [
    [{"kind": "active_window", "node_id": "editor"}],  # Actual 091 cross-window failure.
    [{"kind": "active_window", "node_id": "files"}],  # Model must use the runtime binding.
    [{"kind": "focused", "node_id": "$target"}],
    [{"kind": "text", "text": "unrelated visible label"}],
    [{"kind": "active_window", "node_id": "$target"}, {"kind": "focused", "node_id": "$target"}],
    [],
])
def test_wrong_or_unrelated_window_effects_are_rejected_at_every_phase_position(position, effects):
    steps = [switch() for _ in range(position - 1)] + [switch(effects=effects)]
    assert not validator().is_valid(phase(steps))


@pytest.mark.parametrize("target", [{"node_id": "files"}, {"role": "window", "name": 'Files café "owned"'},
                                    {"name": 'Files café "owned"', "scope_id": "files"}])
def test_observed_exact_and_semantic_window_targets_keep_runtime_effect_binding(target):
    assert validator().is_valid(phase([switch(target)]))


def test_future_semantic_window_is_allowed_later_but_not_as_invented_first_id():
    future = switch({"role": "window", "name": 'Not observed yet λ "dialog"'})
    future["id"] = "future-switch"
    assert validator().is_valid(phase([switch(), future]))
    assert not validator().is_valid(phase([switch({"node_id": "invented"})]))
    assert not validator().is_valid(phase([switch()] * 26))


def test_private_restriction_preserves_public_canonical_definitions_and_observation():
    observation = {"state": state()}
    before = copy.deepcopy(observation)
    public = ui_task.phase_response_schema({"native_ui", "memory_search"})
    derived = ui_task._grounded_phase_schema({"native_ui", "memory_search"}, observation)
    assert observation == before
    assert {key: derived["$defs"][key] for key in public["$defs"]} == public["$defs"]
    assert [item for item in derived["oneOf"][1:]
            if item["properties"]["kind"]["const"] != "ui_boundary"] == public["oneOf"][1:]
    boundary = {"kind": "ui_boundary", "step": {"id": "one", "verb": "click", "target": {"name": "Unmapped control"}}}
    assert Draft202012Validator(derived).is_valid(boundary)
    assert not Draft202012Validator(public).is_valid(boundary)
    # Existing direct-client contracts may name independent effects. Only the
    # private native goal planner uses the causal window-switch restriction.
    bad_for_private = phase([switch(effects=[{"kind": "active_window", "node_id": "editor"}])])
    assert Draft202012Validator(public).is_valid(bad_for_private)
    assert not Draft202012Validator(derived).is_valid(bad_for_private)


def test_decoder_bypass_cannot_dispatch_the_actual_cross_window_failure():
    packets, calls = [], []
    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        if len(packets) == 1:
            return phase([switch(effects=[{"kind": "active_window", "node_id": "editor"}])])
        assert packet["last_result"]["error"] == "invalid_native_phase"
        return {"kind": "blocked", "reason": "No input authorized by this invalid phase"}

    list(ui_task.run_native_goal("Unchanged original task", observe=lambda: {"state": state()},
         compile_phase=compile_phase, execute=lambda *args: calls.append(args), cancel=lambda: None,
         allowed_tools={"native_ui"}, max_phases=2))
    assert len(packets) == 2 and calls == []


def test_controller_resolves_effect_to_the_selected_window_and_verifies_fresh_activation():
    class Adapter:
        def __init__(self):
            self.current = state()
            self.calls = []

        def observe(self):
            return copy.deepcopy(self.current)

        def perform(self, action):
            self.calls.append(copy.deepcopy(action))
            self.current["scope"]["id"] = action["node_id"]
            self.current["revision"] = "after"
            return {"status": "dispatched"}

    adapter = Adapter()
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    result = controller.run_workflow([switch()], final_predicates=[{"kind": "active_window", "node_id": "files"}])
    assert result["status"] == "complete"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["node_id"] == "files"
    assert adapter.calls[0]["expect"] == [{"kind": "active_window", "node_id": "files"}]
    assert adapter.current["nodes"][1]["focused"] is False
