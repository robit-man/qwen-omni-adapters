from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from qwen_omni_adapters.context import configured_tools


def validator(name):
    function = next(entry["schema"]["function"] for entry in configured_tools()
                    if entry["schema"]["function"]["name"] == name)
    Draft202012Validator.check_schema(function["parameters"])
    return Draft202012Validator(function["parameters"])


def step():
    return {"id": "select-control", "verb": "key_chord", "target": {"node_id": "observed-node"},
            "args": {"keys": ["Control_L", "a"]},
            "expect": [{"kind": "focused", "node_id": "$target"}]}


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("arguments", [
    {"operation": "observe"}, {"operation": "observe", "step": step()},
    {"operation": "execute", "action_id": "offered-opaque-action"},
    {"operation": "workflow", "steps": [step()],
     "final_predicates": [{"kind": "focused", "node_id": "independently-observed-node"}]},
    {"operation": "verify", "predicates": [{"kind": "text", "text": "fresh visible evidence"}]},
    {"operation": "cancel"},
])
def test_legal_operation_forms_validate(name, arguments):
    assert validator(name).is_valid(arguments)


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("arguments", [
    {"operation": "execute", "step": step()},
    {"operation": "execute", "action_id": "offered-id", "step": step()},
    {"operation": "execute", "action_id": ""},
    {"operation": "observe", "action_id": "offered-id"},
    {"operation": "workflow", "steps": [step()]},
    {"operation": "verify"},
    {"operation": "cancel", "step": step()},
    {"operation": "execute", "action_id": "offered-id", "args": {}},
])
def test_mixed_or_incomplete_operation_forms_are_rejected(name, arguments):
    assert not validator(name).is_valid(arguments)


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
def test_exact_captured_bad_qwen_call_is_rejected(name):
    captured = {"operation": "execute", "step": {
        "id": "select-all", "verb": "key_chord", "target": {"node_id": "captured-atspi-node"},
        "args": {"keys": ["ctrl+a"]}, "expect": [{"role": "textbox", "name": "", "visible": True}]}}
    assert not validator(name).is_valid(captured)
    # Independently test each argument error, so a bad operation cannot hide it.
    captured["operation"] = "observe"
    captured["step"]["expect"] = [{"kind": "focused", "node_id": "$target"}]
    assert not validator(name).is_valid(captured)
    captured["step"]["args"]["keys"] = ["Control_L", "a"]
    captured["step"]["expect"] = [{"role": "textbox", "name": "", "visible": True}]
    assert not validator(name).is_valid(captured)


@pytest.mark.parametrize("predicate", [
    {"kind": "value", "node_id": "$target", "value": ""},
    {"kind": "checked", "node_id": "$target", "checked": True},
    {"kind": "selected", "node_id": "$target", "values": ["Local"]},
    {"kind": "focused", "node_id": "$target"},
    {"kind": "text_selection", "node_id": "$target", "start": 0, "end": 3},
    {"kind": "active_window", "node_id": "$target"},
    {"kind": "visible", "role": "button", "name": "", "visible": True},
    {"kind": "text", "text": "Fresh result"}, {"kind": "url", "url": "https://example.com/"},
])
def test_complete_typed_predicate_variants_validate(predicate):
    request = {"operation": "observe", "step": step()}
    request["step"]["expect"] = [predicate]
    assert validator("native_ui").is_valid(request)
    incomplete = copy.deepcopy(request)
    incomplete["step"]["expect"][0].pop("kind")
    assert not validator("native_ui").is_valid(incomplete)


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("operation,field", [("verify", "predicates"), ("workflow", "final_predicates")])
def test_independent_terminal_checks_reject_unresolved_target(name, operation, field):
    request = {"operation": operation, field: [{"kind": "focused", "node_id": "$target"}]}
    if operation == "workflow":
        request["steps"] = [step()]
    assert not validator(name).is_valid(request)
    request[field][0]["node_id"] = "independently-observed-node"
    assert validator(name).is_valid(request)


def test_tool_definitions_reuse_local_refs_and_preserve_surface_contract():
    schemas = {entry["schema"]["function"]["name"]: entry["schema"]["function"]
               for entry in configured_tools()}
    for name in ("native_ui", "computer_use"):
        function = schemas[name]
        parameters = function["parameters"]
        assert parameters["properties"]["step"] == {"$ref": "#/$defs/step"}
        assert parameters["properties"]["steps"]["items"] == {"$ref": "#/$defs/step"}
        assert "execute requires action_id only" in function["description"]
        assert 'Begin discovery with {"operation":"observe"}' in function["description"]
    assert validator("computer_use").is_valid({"operation": "observe", "surface": "desktop"})
    assert not validator("native_ui").is_valid({"operation": "observe", "surface": "desktop"})


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
def test_guidance_prefers_bounded_semantic_workflows_and_actual_effects(name):
    function = next(entry["schema"]["function"] for entry in configured_tools()
                    if entry["schema"]["function"]["name"] == name)
    guidance = function["description"]
    assert guidance.index('Begin discovery with {"operation":"observe"}') < guidance.index("prefer one bounded workflow")
    assert "exact observed role/name targets" in guidance
    assert "copy the full fresh ID verbatim" in guidance
    assert "first unverified effect" in guidance
    assert "focused predicate proves only focus" in guidance
    assert "actual observed effect control" in guidance and "do not default to $target" in guidance
    assert "value or readable display text, checked, selected, text_selection" in guidance
    assert "click requests guarded physical mouse input" in guidance
    assert "invoke requests an accessibility action" in guidance
    assert "Never claim completion from input dispatch" in guidance
    assert "OBSERVED_NODE_ID" not in guidance and "RETURNED_SELECTED_ACTION_ID" not in guidance
    assert "focus-control" not in guidance


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("verb,args", [
    ("set_value", {"value": "observed content"}), ("set_checked", {"checked": True}),
    ("type_text", {"text": "literal keyboard input"}),
    ("select", {"option": "Local"}), ("scroll", {"direction": "down"}),
    ("key_chord", {"keys": ["Control_L", "a"]}), ("drag_drop", {"to_node_id": "observed-destination"}),
    ("select_text", {"start": 0, "end": 5}),
])
def test_argument_bearing_verbs_require_their_exact_fields(name, verb, args):
    request = {"operation": "observe", "step": {**step(), "verb": verb, "args": args}}
    assert validator(name).is_valid(request)
    request["step"].pop("args")
    assert not validator(name).is_valid(request)
    request["step"]["args"] = {}
    assert not validator(name).is_valid(request)
    request["step"]["args"] = {"value": "wrong field"} if verb != "set_value" else {"keys": ["a"]}
    assert not validator(name).is_valid(request)


@pytest.mark.parametrize("verb", ["invoke", "focus", "click", "double_click", "right_click", "switch_window"])
def test_zero_argument_verbs_allow_omission_but_target_is_never_empty(verb):
    request = {"operation": "observe", "step": {**step(), "verb": verb}}
    request["step"].pop("args")
    assert validator("native_ui").is_valid(request)
    request["step"]["args"] = {}
    assert validator("native_ui").is_valid(request)
    for target in ({}, {"scope_id": "observed-scope"}):
        request["step"]["target"] = target
        assert not validator("native_ui").is_valid(request)


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
@pytest.mark.parametrize("text", ["", "nul\x00inside", "x"*2049, None, 42])
def test_native_keyboard_text_schema_rejects_invalid_text_in_every_operation_form(name, text):
    typed = {**step(), "verb": "type_text", "args": {"text": text}}
    for request in ({"operation": "observe", "step": typed},
                    {"operation": "workflow", "steps": [typed], "final_predicates": [{"kind": "text", "text": "Verified"}]}):
        assert not validator(name).is_valid(request)


@pytest.mark.parametrize("name", ["native_ui", "computer_use"])
def test_keyboard_text_catalog_keeps_native_method_distinct_from_setter(name):
    function = next(entry["schema"]["function"] for entry in configured_tools()
                    if entry["schema"]["function"]["name"] == name)
    assert "Native type_text {text}" in function["description"]
    assert "failed setter never automatically retries through typing" in function["description"]
    assert validator(name).is_valid({"operation": "observe", "step": {
        **step(), "verb": "type_text", "args": {"text": "é\n" + "x"*2046}}})
