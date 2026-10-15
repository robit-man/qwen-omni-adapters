from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture

from harness.background_agent import _ui_receipt_authority
from portal.browser_ui import semantic_state
from portal.computer_use import ComputerUseStore
from qwen_omni_adapters.computer_use import ComputerUseError, RapidUiController, parse_step


def request(text="literal typed text"):
    return {"id": "type-observed-control", "verb": "type_text", "target": {"name": "Title"},
            "args": {"text": text}, "expect": [{"kind": "value", "node_id": "$target", "value": text}]}


def typing_fixture(monkeypatch, role="entry"):
    fixture = Fixture("typing-session")
    fixture.title.role = role
    fixture.title.state.values.add("FOCUSED")
    geometry = {"x": 40, "y": 30, "width": 800, "height": 600}
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: dict(geometry))
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        assert kwargs["timeout"] == 5 and "shell" not in kwargs
        fixture.title.value = argv[-1]
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    return fixture, commands, geometry


@pytest.mark.parametrize("role", ["entry", "text", "combo box"])
def test_type_text_uses_one_literal_xtest_argv_on_observed_editable_text_roles(monkeypatch, role):
    fixture, commands, _geometry = typing_fixture(monkeypatch, role)
    text = '--option $(literal) `literal` "quoted" é\nnext line'
    action = fixture.action("Title", "type_text", {"text": text})
    assert fixture.adapter.perform(action) == {"status": "dispatched", "backend": "x11_guarded_xtest",
                                             "reason": "native_text_input_invoked"}
    assert commands == [["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text]]
    assert fixture.title.events == []  # No implicit focus, setter, selection or clipboard operation.


@pytest.mark.parametrize("ineligible", ["hidden", "disabled", "readonly", "not-editable", "sensitive", "no-text", "other-role"])
def test_keyboard_typing_is_never_advertised_without_current_eligible_text_metadata(monkeypatch, ineligible):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    if ineligible == "hidden":
        fixture.title.state.values.remove("SHOWING")
    elif ineligible == "disabled":
        fixture.title.state.values.remove("SENSITIVE")
    elif ineligible == "readonly":
        fixture.title.state.values.add("READ_ONLY")
    elif ineligible == "not-editable":
        fixture.title.state.values.remove("EDITABLE")
    elif ineligible == "sensitive":
        fixture.title.role = "password text"
    elif ineligible == "no-text":
        fixture.title.interfaces.remove("Text")
    else:
        fixture.title.role = "push button"
    state = fixture.adapter.observe()
    assert all("type_text" not in node["operations"] for node in state["nodes"])
    assert commands == []


@pytest.mark.parametrize("text", ["", "nul\x00inside", "x"*2049, None, True])
def test_core_and_native_typing_reject_invalid_text_before_any_input(monkeypatch, text):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    with pytest.raises(ComputerUseError):
        parse_step(request(text))
    assert fixture.adapter.perform(fixture.action("Title", "type_text", {"text": text}))["status"] == "rejected"
    assert commands == []


def test_typing_contract_requires_exact_text_field_and_accepts_bounded_unicode():
    assert parse_step(request("é"*2048))["args"] == {"text": "é"*2048}
    for arguments in ({}, {"value": "wrong field"}, {"text": "valid", "code": "not authorized"}):
        with pytest.raises(ComputerUseError):
            parse_step({**request(), "args": arguments})


@pytest.mark.parametrize("changed", ["revision", "scope", "generation", "session", "focus", "readonly", "editable", "sensitive"])
def test_stale_or_ineligible_typing_never_posts_native_keyboard_input(monkeypatch, changed):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    action = fixture.action("Title", "type_text", {"text": "Never dispatched"})
    if changed in {"revision", "scope", "generation", "session"}:
        key = {"revision": "revision", "scope": "scope_id", "generation": "generation", "session": "session_id"}[changed]
        action[key] = "different"
    elif changed == "focus":
        fixture.title.state.values.remove("FOCUSED")
    elif changed == "readonly":
        fixture.title.state.values.add("READ_ONLY")
    elif changed == "editable":
        fixture.title.state.values.remove("EDITABLE")
    else:
        fixture.title.role = "password text"
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert commands == []


@pytest.mark.parametrize("lost_at", [3, 4])
@pytest.mark.parametrize("lost", ["focus", "editable", "readonly", "sensitive"])
def test_typing_rechecks_native_focus_and_editability_at_final_dispatch_boundaries(monkeypatch, lost_at, lost):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    action = fixture.action("Title", "type_text", {"text": "Never dispatched"})
    reads = 0
    def race():
        nonlocal reads
        reads += 1
        if reads == lost_at:
            if lost == "readonly":
                fixture.title.state.values.add("READ_ONLY")
            elif lost == "sensitive":
                fixture.title.role = "password text"
            else:
                fixture.title.state.values.remove("FOCUSED" if lost == "focus" else "EDITABLE")
    fixture.title.state_hook = race
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert commands == []


@pytest.mark.parametrize("at_geometry_read", [2, 3])
@pytest.mark.parametrize("changed", ["geometry", "window", "pid"])
def test_typing_has_fresh_geometry_and_foreground_binding_until_actual_command(monkeypatch, at_geometry_read, changed):
    fixture, commands, geometry = typing_fixture(monkeypatch)
    action = fixture.action("Title", "type_text", {"text": "Never dispatched"})
    reads = 0
    def dimensions(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        if reads == at_geometry_read:
            if changed == "geometry":
                geometry["width"] += 1
            else:
                fixture.foreground["window_id" if changed == "window" else "process_id"] = "18" if changed == "window" else 99
        return dict(geometry)
    monkeypatch.setattr("portal.native_ui._x11_geometry", dimensions)
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert commands == []


def test_partial_typing_timeout_is_retained_and_same_action_is_never_replayed(monkeypatch):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    def uncertain(argv, **_kwargs):
        commands.append(argv)
        fixture.title.value = "partial"
        raise subprocess.TimeoutExpired(argv, 5)
    monkeypatch.setattr("portal.native_ui.subprocess.run", uncertain)
    controller = RapidUiController(fixture.adapter, authorize=lambda *_args: True, max_observations=1)
    state = controller.observe()
    action = next(item for item in controller.enumerate(request(), state) if item["kind"] == "action")
    outcome = controller.execute(action["id"])
    assert outcome["status"] == "unresolved"
    assert outcome["receipt"] == {"status": "dispatched", "backend": "x11_guarded_xtest", "reason": "native_text_input_effect_unresolved"}
    assert controller.execute(action["id"]) == outcome
    assert len(commands) == 1 and fixture.title.events == [] and fixture.title.value == "partial"


def test_verified_keyboard_typing_retains_existing_executor_mutation_authority(monkeypatch):
    fixture, commands, _geometry = typing_fixture(monkeypatch)
    store = ComputerUseStore(None, adapter_factory=lambda *_args: fixture.adapter)
    observed = store.act("typing-session", {"surface": "desktop", "operation": "observe", "step": request()})
    result = store.act("typing-session", {"surface": "desktop", "operation": "execute", "action_id": observed["selected_action_id"]})
    assert result["status"] == "verified" and _ui_receipt_authority(result) == "mutation"
    assert result["effect_receipt"]["effect_state"] == "applied"
    assert len(commands) == 1


def test_explicit_browser_adapter_does_not_advertise_native_typing():
    state = semantic_state({"document_generation": "g1", "elements": [
        {"id": "input", "tag": "input", "type": "text", "role": "textbox", "value": ""}]}, "browser-session", "tab")
    assert "type_text" not in state["nodes"][0]["operations"]
