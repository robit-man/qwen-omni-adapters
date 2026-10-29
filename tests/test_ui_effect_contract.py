"""Private effect compilation does not turn planner explanations into oracles.

These are contract/controller regressions, not loaded-model task acceptance.
"""
from __future__ import annotations

import copy

import pytest

from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    RapidUiController,
    parse_intent,
    parse_step,
    predicate_satisfied,
)
from qwen_omni_adapters.ui_effects import (
    compile_immediate_transition,
    immediate_effect_schema,
    transition_descriptor_schema,
)


def step(verb="key_chord", args=None, *, expect=None):
    request = {"id": "prerequisite", "verb": verb, "target": {"node_id": "field"},
               "args": {"keys": ["Control_L", "l"]} if args is None else args}
    if expect is not None:
        request["expect"] = expect
    return request


def observed(value="", *, selection=None):
    node = {"id": "field", "generation": "g1", "scope_id": "window", "role": "textbox",
            "name": "Location", "visible": True, "enabled": True, "sensitive": False,
            "focused": True, "operations": ["type_text", "set_value", "focus", "key_chord"],
            "value": value, "text_observation": {"character_count": len(value),
                "captured_range": {"start": 0, "end": len(value)}, "truncated": False,
                "selection_count": int(selection is not None)}}
    if selection is not None:
        node["text_selection"] = selection
    return {"session_id": "session", "revision": "r1", "scope": {"kind": "native", "id": "window"},
            "nodes": [node], "text": "Actual native text"}


@pytest.mark.parametrize("narrative", [
    "Workbench file manager surface text", "The location field becomes available.",
    "Open the chooser as a prerequisite to reading corrections.json", "Résumé 中 Ελληνικά",
])
def test_keyboard_prerequisite_is_observed_without_literal_narrative_or_source_credit(narrative):
    proposal = step(expect=[{"kind": "text", "text": narrative}])
    original = copy.deepcopy(proposal)
    effect = compile_immediate_transition(proposal)
    assert proposal == original
    assert effect.mode == "observation_boundary"
    assert effect.as_dict() == {
        "schema": "robit.omni.ui-immediate-effect.v1", "operation": "key_chord",
        "mode": "observation_boundary", "predicates": [],
        "reason": "operation_effect_requires_observation", "non_authorizing": True,
        "task_progress": False, "source_ref": None, "transition": None,
    }
    # The compiler does not silently rewrite the canonical public contract.
    assert parse_step(proposal) == proposal
    assert parse_intent(effect.bind(proposal)) == {key: value for key, value in proposal.items() if key != "expect"}


@pytest.mark.parametrize(("verb", "args", "predicate"), [
    ("focus", {}, {"kind": "focused", "node_id": "$target"}),
    ("switch_window", {}, {"kind": "active_window", "node_id": "$target"}),
    ("set_value", {"value": "https://docs.python.org/"},
     {"kind": "value", "node_id": "$target", "value": "https://docs.python.org/"}),
    ("set_checked", {"checked": False}, {"kind": "checked", "node_id": "$target", "checked": False}),
    ("select_text", {"start": 1, "end": 4},
     {"kind": "text_selection", "node_id": "$target", "start": 1, "end": 4}),
])
def test_supported_operations_compile_argument_bound_immediate_effects(verb, args, predicate):
    proposal = step(verb, args, expect=[{"kind": "text", "text": "An unrelated narrative"}])
    effect = compile_immediate_transition(proposal)
    assert effect.mode == "verified_step"
    bound = effect.bind(proposal)
    assert bound["expect"] == [predicate]
    assert parse_step(bound) == bound
    assert effect.as_dict()["task_progress"] is False
    exported = effect.as_dict()
    exported["predicates"][0]["kind"] = "text"
    assert effect.bind(proposal)["expect"] == [predicate]


@pytest.mark.parametrize(("verb", "args"), [
    ("click", {}), ("double_click", {}), ("right_click", {}), ("invoke", {}),
    ("scroll", {"direction": "down"}), ("key_chord", {"keys": ["Return"]}),
])
def test_unknown_application_specific_effects_stop_at_a_single_input_boundary(verb, args):
    proposal = step(verb, args)
    effect = compile_immediate_transition(proposal)
    assert effect.mode == "observation_boundary"
    assert effect.as_dict()["predicates"] == []
    assert parse_intent(effect.bind(proposal)) == proposal


@pytest.mark.parametrize(("before", "selection", "inserted", "expected"), [
    ("", None, "New", "New"),
    ("old", {"start": 0, "end": 3}, "New", "New"),
    ("first old last", {"start": 6, "end": 9}, "new", "first new last"),
    ("café 中", {"start": 5, "end": 6}, "Ελληνικά", "café Ελληνικά"),
])
def test_typing_derives_exact_complete_value_from_observed_replacement_range(before, selection, inserted, expected):
    proposal = step("type_text", {"text": inserted}, expect=[{"kind": "focused", "node_id": "$target"}])
    effect = compile_immediate_transition(proposal, observed(before, selection=selection))
    assert effect.mode == "verified_step"
    assert effect.bind(proposal)["expect"] == [{"kind": "value", "node_id": "$target", "value": expected}]


@pytest.mark.parametrize("fault", ["no_state", "nonempty_caret_unknown", "truncated", "incomplete",
                                  "wrong_count", "ambiguous", "unfocused", "sensitive", "wrong_scope",
                                  "multiple_selections", "bad_selection", "hidden", "disabled",
                                  "bad_operations", "bad_coverage", "boolean_count", "bad_identity"])
def test_typing_uncertainty_requires_replanning_before_input(fault):
    state = observed("old", selection={"start": 0, "end": 3})
    node = state["nodes"][0]
    if fault == "no_state":
        state = None
    elif fault == "nonempty_caret_unknown":
        node.pop("text_selection")
        node["text_observation"]["selection_count"] = 0
    elif fault == "truncated":
        node["text_observation"]["truncated"] = True
    elif fault == "incomplete":
        node.pop("text_observation")
    elif fault == "wrong_count":
        node["text_observation"]["character_count"] += 1
    elif fault == "ambiguous":
        state["nodes"].append(copy.deepcopy(node))
    elif fault == "unfocused":
        node["focused"] = False
    elif fault == "sensitive":
        node["sensitive"] = True
    elif fault == "wrong_scope":
        node["scope_id"] = "foreign"
    elif fault == "multiple_selections":
        node["text_observation"]["selection_count"] = 2
    elif fault == "bad_selection":
        node["text_selection"]["end"] = 99
    elif fault == "hidden":
        node["visible"] = False
    elif fault == "disabled":
        node["enabled"] = False
    elif fault == "bad_operations":
        node["operations"] = None
    elif fault == "bad_coverage":
        node["text_observation"] = None
    elif fault == "boolean_count":
        node["text_observation"]["selection_count"] = True
    else:
        node["generation"] = None
    proposal = step("type_text", {"text": "new"})
    effect = compile_immediate_transition(proposal, state)
    assert effect.mode == "requires_replan"
    assert effect.as_dict()["predicates"] == []
    with pytest.raises(ComputerUseError):
        effect.bind(proposal)


def test_rebinding_cannot_change_text_or_target_behind_the_compiled_contract():
    proposal = step("set_value", {"value": "requested"})
    effect = compile_immediate_transition(proposal)
    for mutation in ({"args": {"value": "different"}}, {"target": {"node_id": "other"}},
                     {"id": "different-attempt"}, {"verb": "focus", "args": {}}):
        with pytest.raises(ComputerUseError):
            effect.bind({**proposal, **mutation})


def test_compiled_exact_write_still_requires_observed_value_not_focus_or_dispatch():
    class NoWrite:
        def __init__(self):
            self.calls = []

        def observe(self):
            return observed("unchanged")

        def perform(self, action):
            self.calls.append(action)
            return {"status": "dispatched"}

    adapter = NoWrite()
    controller = RapidUiController(adapter, authorize=lambda *_: True, max_observations=1)
    proposal = step("set_value", {"value": "requested"}, expect=[{"kind": "focused", "node_id": "$target"}])
    bound = compile_immediate_transition(proposal).bind(proposal)
    result = controller.run_step(bound)
    assert len(adapter.calls) == 1 and result["status"] == "unresolved"
    assert predicate_satisfied({"kind": "focused", "node_id": "field"}, result["after"])
    assert not predicate_satisfied({"kind": "value", "node_id": "field", "value": "requested"}, result["after"])


def test_valid_application_specific_drag_and_selection_effects_are_not_guessed():
    for proposal in (step("drag_drop", {"to_node_id": "folder"}), step("select", {"option": "draft"})):
        effect = compile_immediate_transition(proposal)
        assert effect.mode == "requires_replan"
        with pytest.raises(ComputerUseError):
            effect.bind(proposal)


@pytest.mark.parametrize("proposal", [
    {"id": "x", "verb": "launch", "target": {"node_id": "field"}},
    step("set_value", {"value": 5}), step("key_chord", {"keys": ["Control_L", "$(bad)"]}),
    {**step(), "authorized": True}, step("type_text", {"text": "\x00"}),
    step(expect=[{"kind": "unbounded_claim", "text": "already done"}]),
])
def test_malformed_contracts_are_rejected_instead_of_reinterpreted(proposal):
    with pytest.raises(ComputerUseError):
        compile_immediate_transition(proposal)


def transition(kind, state):
    return {"kind": kind, "source_ref": {"session_id": state["session_id"],
        "scope_id": state["scope"]["id"], "revision": state["revision"]}}


def complete_observed():
    return {**observed(), "truncated": False}


def test_typed_focus_transition_compares_scoped_provider_facts_without_resource_credit():
    before = complete_observed()
    after = copy.deepcopy(before)
    after["revision"] = "r2"
    after["nodes"][0]["focused"] = False
    other = {**copy.deepcopy(after["nodes"][0]), "id": "next-field", "focused": True}
    after["nodes"].append(other)
    effect = compile_immediate_transition(step(), before,
        transition=transition("focus_changed_within_scope", before))
    result = effect.compare(before, after)
    assert result["status"] == "observed"
    assert result["task_progress"] is result["resource_acquired"] is result["input_authority"] is False
    assert effect.mode == "observation_boundary" and effect.as_dict()["predicates"] == []
    assert effect.compare(before, before)["status"] == "absent"


def test_typed_popup_transition_depends_on_role_and_visibility_not_label_words():
    before = complete_observed()
    after = copy.deepcopy(before)
    after["revision"] = "r2"
    menu = {**copy.deepcopy(after["nodes"][0]), "id": "popup", "role": "popup_menu",
            "name": "Completely arbitrary text", "focused": False}
    after["nodes"].append(menu)
    effect = compile_immediate_transition(step("click", {}), before,
        transition=transition("popup_opened", before))
    assert effect.compare(before, after)["status"] == "observed"
    menu["role"], menu["name"] = "label", "File chooser popup menu"
    assert effect.compare(before, after)["status"] == "absent"
    menu["role"], menu["visible"] = "popup_menu", False
    assert effect.compare(before, after)["status"] == "absent"


def document(url):
    state = complete_observed()
    state["scope"].update(window_id="17", url=url)
    state["nodes"].append({**copy.deepcopy(state["nodes"][0]), "id": "document", "role": "document_web"})
    state["document_evidence"] = {"status": "observed", "source": "atspi.Document.DocURL", "url": url,
        "node_id": "document", "generation": "g1", "scope_id": "window", "window_id": "17",
        "provider": "atspi2", "provider_identity_sha256": "b" * 64}
    return state


def test_document_transition_requires_provider_binding_and_does_not_prove_requested_resource_or_readability():
    before, after = document("https://example.test/old"), document("https://example.test/new")
    after["revision"] = "r2"
    effect = compile_immediate_transition(step("key_chord", {"keys": ["Return"]}), before,
        transition=transition("document_changed", before))
    result = effect.compare(before, after)
    assert result["status"] == "observed" and not result["resource_acquired"]
    assert effect.compare(before, before)["status"] == "absent"
    after["document_evidence"]["url"] = "https://example.test/mismatched"
    assert effect.compare(before, after)["status"] == "unknown"
    after = copy.deepcopy(before)
    after["nodes"][0]["value"] = "https://example.test/new"
    after["text"] = "Requested page is now open"
    assert effect.compare(before, after)["status"] == "absent"


@pytest.mark.parametrize("fault", ["changed_before", "wrong_session", "wrong_scope", "incomplete",
                                  "no_focused_node", "two_focused_nodes", "duplicate_ids"])
def test_transition_uncertainty_never_becomes_observed_success(fault):
    before = complete_observed()
    effect = compile_immediate_transition(step(), before,
        transition=transition("focus_changed_within_scope", before))
    after = copy.deepcopy(before)
    if fault == "changed_before":
        before["revision"] = "different"
    elif fault == "wrong_session":
        after["session_id"] = "foreign"
    elif fault == "wrong_scope":
        after["scope"]["id"] = "foreign"
    elif fault == "incomplete":
        after["truncated"] = True
    elif fault == "no_focused_node":
        after["nodes"][0]["focused"] = False
    elif fault == "two_focused_nodes":
        after["nodes"].append({**copy.deepcopy(after["nodes"][0]), "id": "other"})
    else:
        after["nodes"].append(copy.deepcopy(after["nodes"][0]))
    assert effect.compare(before, after)["status"] == "unknown"


def test_descriptor_and_compilation_schema_are_closed_source_bound_and_json_compatible():
    from jsonschema import Draft202012Validator

    before = complete_observed()
    selected = transition("popup_opened", before)
    schema = Draft202012Validator(transition_descriptor_schema(before))
    assert schema.is_valid(selected)
    assert not schema.is_valid({**selected, "expected_text": "Narrative"})
    assert not schema.is_valid({**selected, "source_ref": {**selected["source_ref"], "revision": "stale"}})
    effect = compile_immediate_transition(step(), before, transition=selected)
    compiled_schema = Draft202012Validator(immediate_effect_schema())
    assert compiled_schema.is_valid(effect.as_dict())
    assert not compiled_schema.is_valid({**effect.as_dict(), "input_authority": True})
    assert not compiled_schema.is_valid({**effect.as_dict(), "predicates": [{"kind": "text", "text": "Narrative"}]})
    for malformed in ({**selected, "expected_text": "Narrative"}, {**selected, "kind": "requested_file_opened"},
                      {**selected, "source_ref": {**selected["source_ref"], "revision": "stale"}}):
        with pytest.raises(ComputerUseError):
            compile_immediate_transition(step(), before, transition=malformed)


def test_foreign_window_focus_is_excluded_and_switching_scope_remains_unknown():
    before = complete_observed()
    before["nodes"].append({**copy.deepcopy(before["nodes"][0]), "id": "foreign", "scope_id": "foreign"})
    after = copy.deepcopy(before)
    effect = compile_immediate_transition(step(), before,
        transition=transition("focus_changed_within_scope", before))
    assert effect.compare(before, after)["status"] == "absent"
    after["scope"]["id"] = "foreign"
    assert effect.compare(before, after)["status"] == "unknown"


def test_a_popup_provider_generation_change_does_not_prove_a_new_popup_opened():
    before = complete_observed()
    before["nodes"].append({**copy.deepcopy(before["nodes"][0]), "id": "popup", "role": "menu"})
    after = copy.deepcopy(before)
    after["nodes"][1]["generation"] = "g2"
    effect = compile_immediate_transition(step(), before, transition=transition("popup_opened", before))
    assert effect.compare(before, after)["status"] == "unknown"


def test_document_evidence_without_an_actual_window_binding_is_unknown():
    before, after = document("https://example.test/old"), document("https://example.test/new")
    for state in (before, after):
        state["scope"].pop("window_id")
        state["document_evidence"].pop("window_id")
    effect = compile_immediate_transition(step(), before, transition=transition("document_changed", before))
    assert effect.compare(before, after)["status"] == "unknown"
