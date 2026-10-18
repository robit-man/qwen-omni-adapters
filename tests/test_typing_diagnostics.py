"""Typing uncertainty diagnostics never supply an insertion range or input right."""
from __future__ import annotations

import copy

import pytest
from test_ui_effect_contract import observed, step

from portal.computer_use import ComputerUseStore
from qwen_omni_adapters.computer_use import ComputerUseError
from qwen_omni_adapters.ui_effects import compile_immediate_transition

LEGACY = "typing_complete_value_not_observed"
RANGE = "typing_insertion_range_not_observed"


@pytest.mark.parametrize("value,role,kind", [
    ("https://docs.python.org/3.12/library/pathlib.html#pathlib.Path.mkdir", "textbox", "browser"),
    ("An existing paragraph in an editor", "paragraph", "native"),
    ("café 中 Ελληνικά", "entry", "desktop"),
    ("مرحبا بالعالم", "combobox", "native"),
    ("👩🏽‍💻 e\u0301\nsecond line", "terminal", "desktop"),
])
@pytest.mark.parametrize("inserted", ["New text", "\n", "\r"])
def test_complete_nonempty_text_without_insertion_range_reports_the_missing_evidence(value, role, kind, inserted):
    before = observed(value)
    before["scope"]["kind"] = kind
    before["nodes"][0]["role"] = role
    original = copy.deepcopy(before)
    proposal = step("type_text", {"text": inserted})
    effect = compile_immediate_transition(proposal, before)
    exported = effect.as_dict()
    assert before == original
    assert effect.mode == "requires_replan" and exported["reason"] == RANGE
    assert exported["predicates"] == []
    assert exported["source_ref"] == {"session_id": before["session_id"],
        "scope_id": before["scope"]["id"], "revision": before["revision"]}
    assert exported["non_authorizing"] is True and exported["task_progress"] is False
    with pytest.raises(ComputerUseError):
        effect.bind(proposal)


@pytest.mark.parametrize("selection", [
    {"start": 1, "end": 1}, {"start": -1, "end": 2}, {"start": 0, "end": 99},
    {"start": True, "end": 2}, {"start": 0, "end": 2, "extra": "not an insertion range"},
])
def test_complete_text_with_an_invalid_selection_range_still_grants_no_insertion(selection):
    proposal = step("type_text", {"text": "New"})
    effect = compile_immediate_transition(proposal, observed("old", selection=selection))
    assert effect.reason == RANGE and effect.mode == "requires_replan"
    assert effect.as_dict()["predicates"] == []
    with pytest.raises(ComputerUseError):
        effect.bind(proposal)


@pytest.mark.parametrize("fault", [
    "no_state", "missing_nodes", "missing_target", "ambiguous", "unfocused", "foreign_scope",
    "hidden", "disabled", "sensitive", "no_typing", "bad_operations", "missing_generation",
    "missing_value", "nonstring_value", "missing_coverage", "nonmapping_coverage", "truncated",
    "wrong_character_count", "boolean_character_count", "boolean_selection_count", "multiple_selections",
    "boolean_capture_start", "bad_capture_range", "missing_selection_for_count1", "nonmapping_selection",
])
def test_missing_or_ineligible_value_evidence_keeps_legacy_reason_and_rejection(fault):
    before = observed("old")
    node = before["nodes"][0]
    coverage = node["text_observation"]
    if fault == "no_state":
        before = None
    elif fault == "missing_nodes":
        before.pop("nodes")
    elif fault == "missing_target":
        node["id"] = "different"
    elif fault == "ambiguous":
        before["nodes"].append(copy.deepcopy(node))
    elif fault in {"unfocused", "hidden", "disabled", "sensitive"}:
        node[{"unfocused": "focused", "hidden": "visible", "disabled": "enabled", "sensitive": "sensitive"}[fault]] = fault == "sensitive"
    elif fault == "foreign_scope":
        node["scope_id"] = "different-window"
    elif fault == "no_typing":
        node["operations"] = ["focus"]
    elif fault == "bad_operations":
        node["operations"] = None
    elif fault == "missing_generation":
        node.pop("generation")
    elif fault == "missing_value":
        node.pop("value")
    elif fault == "nonstring_value":
        node["value"] = 3
    elif fault == "missing_coverage":
        node.pop("text_observation")
    elif fault == "nonmapping_coverage":
        node["text_observation"] = None
    elif fault == "truncated":
        coverage["truncated"] = True
    elif fault in {"wrong_character_count", "boolean_character_count"}:
        coverage["character_count"] = 4 if fault == "wrong_character_count" else True
    elif fault in {"boolean_selection_count", "multiple_selections"}:
        coverage["selection_count"] = True if fault == "boolean_selection_count" else 2
    elif fault == "boolean_capture_start":
        coverage["captured_range"]["start"] = False
    elif fault == "bad_capture_range":
        coverage["captured_range"]["end"] = 2
    else:
        coverage["selection_count"] = 1
        if fault == "nonmapping_selection":
            node["text_selection"] = "not a range"
    proposal = step("type_text", {"text": "New"})
    effect = compile_immediate_transition(proposal, before)
    assert effect.reason == LEGACY and effect.mode == "requires_replan"
    assert effect.as_dict()["predicates"] == []
    with pytest.raises(ComputerUseError):
        effect.bind(proposal)


@pytest.mark.parametrize("before,selection,inserted,expected", [
    ("", None, "New", "New"),
    ("old", {"start": 0, "end": 3}, "New", "New"),
    ("café 中", {"start": 5, "end": 6}, "Ελληνικά", "café Ελληνικά"),
    ("first\nlast", {"start": 5, "end": 6}, " — ", "first — last"),
])
def test_existing_supported_insertions_keep_exact_predicates_and_binding(before, selection, inserted, expected):
    proposal = step("type_text", {"text": inserted})
    effect = compile_immediate_transition(proposal, observed(before, selection=selection))
    assert effect.reason == "host_operation_effect" and effect.mode == "verified_step"
    assert effect.bind(proposal)["expect"] == [{"kind": "value", "node_id": "$target", "value": expected}]


def test_predicted_value_over_existing_4096_bound_keeps_legacy_rejection():
    proposal = step("type_text", {"text": "zz"})
    effect = compile_immediate_transition(proposal, observed("x" * 4096, selection={"start": 0, "end": 1}))
    assert effect.mode == "requires_replan" and effect.reason == LEGACY
    assert effect.as_dict()["predicates"] == []
    with pytest.raises(ComputerUseError):
        effect.bind(proposal)


def test_actual_controller_propagates_range_diagnostic_before_any_native_input():
    class NoInput:
        def __init__(self):
            self.calls = []

        def observe(self):
            return observed("Existing document text")

        def perform(self, action):
            self.calls.append(action)
            pytest.fail("An unknown insertion range cannot authorize native input")

    adapter = NoInput()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancelled = store.begin_task("session", private_boundaries=True)
    with store.task_scope("session", "desktop", cancelled):
        result = store.immediate_transition("session", step("type_text", {"text": "\n"}),
            goal="Edit the owned document", objective="Perform the requested next document operation")
    assert not adapter.calls and result["input_attempted"] is False
    assert result["dispatch_status"] == "not_dispatched" and result["new_effect_count"] == 0
    assert result["error"] == "ui_immediate_effect_requires_replan" and result["reason"] == RANGE
    assert result["immediate_effect_contract"]["reason"] == RANGE
    assert result["effect_receipt"]["effect_state"] == "absent"
    assert result["ui_boundary_receipt"]["adapter_called"] is False
