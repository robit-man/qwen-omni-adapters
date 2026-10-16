"""Literal multiline typing must not turn a single-line control into Return actions."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture

from qwen_omni_adapters.computer_use import RapidUiController


@pytest.fixture
def text_target(monkeypatch):
    fixture = Fixture("multiline-text")
    fixture.title.state.values.update({"FOCUSED", "MULTI_LINE"})
    monkeypatch.setattr(fixture.api.StateType, "MULTI_LINE", "MULTI_LINE", raising=False)
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_: {
        "x": 10, "y": 20, "width": 800, "height": 600})
    fixture.inputs = []

    def run(argv, **kwargs):
        fixture.inputs.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    return fixture


@pytest.mark.parametrize("fault", ["single-line", "unknown-state", "unfocused", "readonly", "hidden", "disabled"])
def test_multiline_input_requires_current_positive_native_capability_before_input(text_target, monkeypatch, fault):
    fixture = text_target
    action = fixture.action("Title", "type_text", {"text": "alpha\nbeta"})
    if fault == "unknown-state":
        monkeypatch.delattr(fixture.api.StateType, "MULTI_LINE")
    elif fault == "single-line":
        fixture.title.state.values.remove("MULTI_LINE")
    elif fault == "unfocused":
        fixture.title.state.values.remove("FOCUSED")
    elif fault == "readonly":
        fixture.title.state.values.add("READ_ONLY")
    elif fault == "hidden":
        fixture.title.state.values.remove("SHOWING")
    else:
        fixture.title.state.values.remove("SENSITIVE")
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    assert fixture.inputs == [] and fixture.title.events == []


def test_single_line_typing_does_not_require_multiline_capability(text_target, monkeypatch):
    fixture = text_target
    monkeypatch.delattr(fixture.api.StateType, "MULTI_LINE")
    result = fixture.adapter.perform(fixture.action("Title", "type_text", {"text": "plain -- literal"}))
    assert result["status"] == "dispatched" and len(fixture.inputs) == 1
    assert fixture.inputs[0][0] == ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", "plain -- literal"]
    assert fixture.title.events == []


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "no-observed-effect"])
def test_partial_or_unverified_multiline_dispatch_has_no_setter_clipboard_or_retry(text_target, monkeypatch, failure):
    fixture = text_target

    def run(argv, **kwargs):
        fixture.inputs.append((argv, kwargs))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=1 if failure == "nonzero" else 0)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    controller = RapidUiController(fixture.adapter, authorize=lambda *_: True,
        max_observations=1, verification_timeout_s=.01)
    step = {"id": "literal-lines", "verb": "type_text", "target": {"name": "Title"},
        "args": {"text": "alpha\nbeta"},
        "expect": [{"kind": "value", "node_id": "$target", "value": "alpha\nbeta"}]}
    first = controller.run_step(step)
    again = controller.run_step(step)
    assert first["status"] == "unresolved" and again == first
    assert first["receipt"]["backend"] == "x11_guarded_xtest"
    assert len(fixture.inputs) == 1 and fixture.inputs[0][1]["timeout"] == 5
    assert not fixture.inputs[0][1].get("shell", False)
    assert fixture.title.events == []
