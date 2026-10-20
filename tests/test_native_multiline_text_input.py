"""Literal multiline typing must not turn a single-line control into Return actions."""
from __future__ import annotations

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


def literal(fixture, monkeypatch, *, caret=0, failure=None, value=""):
    fixture.title.value, fixture.title.caret, fixture.title.selection = value, caret, None
    text = fixture.api.Text
    text.get_caret_offset = lambda node: node.caret
    text.get_n_selections = lambda node: 0 if node.selection is None else 1
    text.get_selection = lambda node, _index: SimpleNamespace(
        start_offset=node.selection[0], end_offset=node.selection[1])
    text.add_selection = text.set_selection = lambda *_args: True
    calls = []

    def insert(node, position, value, length):
        calls.append(("insert", position, value, length))
        if failure == "raises":
            raise RuntimeError("provider failure")
        if failure in {"false", "no-observed-effect"}:
            return failure == "no-observed-effect"
        node.value = node.value[:position] + value + node.value[position:]
        node.caret = position + len(value)
        return True

    def delete(node, start, end):
        calls.append(("delete", start, end))
        node.value = node.value[:start] + node.value[end:]
        node.selection, node.caret = None, start
        return True

    monkeypatch.setattr(fixture.api, "EditableText", SimpleNamespace(insert_text=insert, delete_text=delete),
                        raising=False)
    return calls


AUTOINDENTED = "def f(x):\n    if x:\n        return 1\n    return 2\n"


@pytest.mark.parametrize("value,caret,inserted,expected", [
    ("", 0, AUTOINDENTED, AUTOINDENTED),
    ("head\ntail", 5, "café\n  中\n", "head\ncafé\n  中\ntail"),
    ("abc", 3, "\n\tx", "abc\n\tx"),
])
def test_multiline_text_is_inserted_literally_at_the_issued_caret_without_keyboard_return(
        text_target, monkeypatch, value, caret, inserted, expected):
    fixture = text_target
    calls = literal(fixture, monkeypatch, caret=caret, value=value)
    result = fixture.adapter.perform(fixture.action("Title", "type_text", {"text": inserted}))
    assert result == {"status": "dispatched", "backend": "atspi_editable_text_literal",
                      "reason": "native_text_input_invoked"}
    assert calls == [("insert", caret, inserted, len(inserted.encode("utf-8")))]
    assert fixture.title.value == expected and fixture.inputs == []  # No XTEST Return, no setter.


def test_multiline_text_replaces_exactly_the_issued_selection(text_target, monkeypatch):
    fixture = text_target
    calls = literal(fixture, monkeypatch, value="keep OLD keep")
    fixture.title.selection = (5, 8)
    result = fixture.adapter.perform(fixture.action("Title", "type_text", {"text": "new\n  lines"}))
    assert result["status"] == "dispatched" and result["backend"] == "atspi_editable_text_literal"
    assert calls == [("delete", 5, 8), ("insert", 5, "new\n  lines", 11)]
    assert fixture.title.value == "keep new\n  lines keep" and fixture.inputs == []


@pytest.mark.parametrize("fault,reason", [
    ("no_caret", "native_multiline_insertion_point_not_observed"),
    ("truncated", "native_multiline_insertion_point_not_observed"),
    ("no_editable", "native_multiline_literal_transport_unavailable"),
])
def test_multiline_without_issued_insertion_point_or_literal_transport_sends_nothing(
        text_target, monkeypatch, fault, reason):
    fixture = text_target
    calls = literal(fixture, monkeypatch, value="abc", caret=1)
    if fault == "no_caret":
        del fixture.api.Text.get_caret_offset
    elif fault == "truncated":
        fixture.title.value = "z" * 2100
    else:
        fixture.title.interfaces.remove("EditableText")
    result = fixture.adapter.perform(fixture.action("Title", "type_text", {"text": "a\nb"}))
    assert result == {"status": "rejected", "reason": reason}
    assert calls == [] and fixture.inputs == [] and fixture.title.events == []


@pytest.mark.parametrize("drift", ["caret", "value", "selection"])
def test_live_insertion_point_drift_at_dispatch_rejects_before_literal_input(text_target, monkeypatch, drift):
    fixture = text_target
    calls = literal(fixture, monkeypatch, value="abc", caret=1)
    action = fixture.action("Title", "type_text", {"text": "a\nb"})
    real_provider_guard = fixture.bridge._provider_guard

    def drift_then_guard():  # After the perform-entry observation, before the final reads.
        if drift == "caret":
            fixture.title.caret = 2
        elif drift == "value":
            fixture.title.value = "abX"
        else:
            fixture.title.selection = (0, 1)
        return real_provider_guard()

    fixture.bridge._provider_guard = drift_then_guard
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected" and calls == [] and fixture.inputs == []


@pytest.mark.parametrize("failure", ["raises", "false", "no-observed-effect"])
def test_partial_or_unverified_multiline_dispatch_has_no_setter_clipboard_or_retry(text_target, monkeypatch, failure):
    fixture = text_target
    calls = literal(fixture, monkeypatch, failure=failure)
    controller = RapidUiController(fixture.adapter, authorize=lambda *_: True,
        max_observations=1, verification_timeout_s=.01)
    step = {"id": "literal-lines", "verb": "type_text", "target": {"name": "Title"},
        "args": {"text": "alpha\nbeta"},
        "expect": [{"kind": "value", "node_id": "$target", "value": "alpha\nbeta"}]}
    first = controller.run_step(step)
    again = controller.run_step(step)
    assert first["status"] == "unresolved" and again == first
    assert first["receipt"]["backend"] == "atspi_editable_text_literal"
    assert len(calls) == 1 and fixture.inputs == []
    assert fixture.title.events == []
