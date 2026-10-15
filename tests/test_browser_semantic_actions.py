"""Canonical browser gestures stay bound to fresh semantic nodes and effects."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from portal import browser
from portal.browser_ui import semantic_state
from qwen_omni_adapters.computer_use import predicate_satisfied


def page():
    return {"document_generation": "doc", "url": "http://localhost/owned", "viewport": {"width": 800, "height": 600},
            "elements": [
                {"id": "source", "tag": "button", "role": "button", "label": "Drag source",
                 "draggable": True, "x": 10, "y": 20, "width": 20, "height": 20},
                {"id": "drop", "tag": "div", "role": "region", "label": "Drop area",
                 "x": 110, "y": 70, "width": 20, "height": 20},
                {"id": "text", "tag": "textarea", "role": "textbox", "label": "Text",
                 "value": "alpha beta", "text_selectable": True, "text_selection": None,
                 "x": 40, "y": 140, "width": 200, "height": 50},
            ]}


@pytest.fixture
def semantic_browser(monkeypatch):
    store = browser.BrowserAutomationStore(chromium_bin="/usr/bin/google-chrome")
    state = semantic_state(page(), "session", "tab")
    calls = []
    session = SimpleNamespace(port=4242, target_id="tab", page_socket="")
    monkeypatch.setattr(store, "_session_locked", lambda *_: session)
    monkeypatch.setattr(store, "_page_socket", lambda *_: "ws://127.0.0.1/tab")
    monkeypatch.setattr(store, "_semantic_snapshot", lambda *_: copy.deepcopy(state))
    monkeypatch.setattr(store, "_refresh_element", lambda _session, _cdp, node_id: {
        "id": node_id, **next(item["bounds"] for item in state["nodes"] if item["id"] == node_id),
    })

    class Cdp:
        def __init__(self, *_args):
            pass

        def call(self, method, arguments):
            assert method != "Page.captureScreenshot"
            calls.append((method, arguments))
            return {}

        def close(self):
            pass

    monkeypatch.setattr(browser, "_Cdp", Cdp)
    monkeypatch.setattr(store, "_evaluate", lambda *_: True)

    def action(verb, args=None, node_id="source"):
        node = next(item for item in state["nodes"] if item["id"] == node_id)
        return {"node_id": node_id, "generation": node["generation"], "scope_id": state["scope"]["id"],
                "revision": state["revision"], "verb": verb, "args": args or {}}

    return store, state, calls, action


@pytest.mark.parametrize("verb,button,counts", [
    ("click", "left", [1, 1]), ("double_click", "left", [1, 1, 2, 2]), ("right_click", "right", [1, 1]),
])
def test_canonical_pointer_gestures_emit_actual_pressed_released_events(semantic_browser, verb, button, counts):
    store, _state, calls, action = semantic_browser
    assert store.semantic_perform("session", action(verb))["status"] == "dispatched"

    events = [event for method, event in calls if method == "Input.dispatchMouseEvent"]
    assert [event["clickCount"] for event in events] == counts
    assert [event["type"] for event in events] == ["mousePressed", "mouseReleased"] * (len(counts) // 2)
    assert all(event["button"] == button and event["x"] == 20 and event["y"] == 30 for event in events)
    assert events[0]["buttons"] == (1 if button == "left" else 2)
    assert events[-1]["buttons"] == 0


@pytest.mark.parametrize("direction,dx,dy", [("left", -400, 0), ("right", 400, 0), ("up", 0, -400), ("down", 0, 400)])
def test_canonical_scroll_uses_current_target_geometry(semantic_browser, direction, dx, dy):
    store, _state, calls, action = semantic_browser
    result = store.semantic_perform("session", action("scroll", {"direction": direction}))

    assert result["status"] == "dispatched"
    assert calls == [("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": 20, "y": 30, "deltaX": dx, "deltaY": dy})]


def test_drag_drop_uses_jointly_revalidated_code_owned_geometry(semantic_browser, monkeypatch):
    store, _state, calls, action = semantic_browser
    scripts = []

    def evaluate(_cdp, script):
        scripts.append(script)
        return {"ok": True, "source": {"x": 12, "y": 22, "width": 20, "height": 20},
                "destination": {"x": 112, "y": 72, "width": 20, "height": 20}}

    monkeypatch.setattr(store, "_evaluate", evaluate)
    result = store.semantic_perform("session", action("drag_drop", {"to_node_id": "drop"}))

    assert result["status"] == "dispatched"
    assert len(scripts) == 1 and 'resolve("source")' in scripts[0] and 'resolve("drop")' in scripts[0]
    assert "document.elementFromPoint" in scripts[0]
    assert calls[0][1]["x"] == 22 and calls[0][1]["y"] == 32
    assert calls[-1][1]["x"] == 122 and calls[-1][1]["y"] == 82
    assert calls[-1][1]["type"] == "mouseReleased"


@pytest.mark.parametrize("change", ["missing", "disabled", "sensitive", "foreign_scope"])
def test_drag_drop_rejects_unavailable_destination_before_input(semantic_browser, change):
    store, state, calls, action = semantic_browser
    destination = state["nodes"][1]
    if change == "missing":
        state["nodes"].remove(destination)
    elif change == "foreign_scope":
        destination["scope_id"] = "other-document"
    else:
        destination["enabled" if change == "disabled" else "sensitive"] = change != "disabled"
    assert store.semantic_perform("session", action("drag_drop", {"to_node_id": "drop"}))["status"] == "rejected"
    assert calls == []


def test_drag_drop_rejects_occlusion_without_dispatching(semantic_browser, monkeypatch):
    store, _state, calls, action = semantic_browser
    monkeypatch.setattr(store, "_evaluate", lambda *_: {"ok": False})
    with pytest.raises(browser.BrowserAutomationError, match="visible and unoccluded"):
        store.semantic_perform("session", action("drag_drop", {"to_node_id": "drop"}))
    assert calls == []


def test_text_selection_is_observed_and_invalidates_old_revision(semantic_browser, monkeypatch):
    store, _state, _calls, action = semantic_browser
    scripts = []
    monkeypatch.setattr(store, "_evaluate", lambda _cdp, script: scripts.append(script) or {"ok": True})
    result = store.semantic_perform("session", action("select_text", {"start": 0, "end": 5}, "text"))
    assert result["status"] == "dispatched"
    assert "el.setSelectionRange(0, 5)" in scripts[0]

    raw = page()
    before = semantic_state(raw, "session", "tab")
    assert not predicate_satisfied({"kind": "text_selection", "node_id": "text", "start": 0, "end": 5}, before, {})
    raw["elements"][2]["text_selection"] = {"start": 0, "end": 5}
    after = semantic_state(raw, "session", "tab")
    assert predicate_satisfied({"kind": "text_selection", "node_id": "text", "start": 0, "end": 5}, after, {})
    assert after["revision"] != before["revision"]


def test_text_selection_cannot_exceed_observed_value(semantic_browser):
    store, _state, calls, action = semantic_browser
    assert store.semantic_perform("session", action("select_text", {"start": 0, "end": 50}, "text"))["status"] == "rejected"
    assert calls == []


def test_key_chord_focuses_current_target_and_releases_modifiers(semantic_browser, monkeypatch):
    store, _state, calls, action = semantic_browser
    focused = []
    monkeypatch.setattr(store, "_evaluate", lambda _cdp, script: focused.append(script) or True)
    assert store.semantic_perform("session", action("key_chord", {"keys": ["ctrl", "a"]}, "text"))["status"] == "dispatched"

    assert "document.activeElement === el" in focused[0]
    events = [event for method, event in calls if method == "Input.dispatchKeyEvent"]
    assert [(event["type"], event["key"]) for event in events] == [
        ("keyDown", "Control"), ("keyDown", "a"), ("keyUp", "a"), ("keyUp", "Control"),
    ]
    assert events[1]["modifiers"] == 2 and "text" not in events[1]
    assert events[-1]["modifiers"] == 0


@pytest.mark.parametrize("keys", [["ctrl", "ctrl", "a"], ["ctrl", "a", "b"], ["execute_code"], ["ctrl+a"]])
def test_key_chord_rejects_unsupported_names_without_input(keys):
    calls = []
    with pytest.raises(browser.BrowserAutomationError):
        browser.BrowserAutomationStore._key_chord(SimpleNamespace(call=lambda *args: calls.append(args)), keys)
    assert calls == []


def test_key_failure_releases_already_pressed_modifiers():
    events = []

    def dispatch(_method, event):
        events.append(event)
        if event["key"] == "a":
            raise browser.BrowserAutomationError("transport failed")

    with pytest.raises(browser.BrowserAutomationError):
        browser.BrowserAutomationStore._key_chord(SimpleNamespace(call=dispatch), ["ctrl", "a"])
    assert events[-1]["key"] == "Control" and events[-1]["type"] == "keyUp"


def test_only_applicable_drag_selection_and_no_browser_window_switch_are_advertised():
    state = semantic_state(page(), "session", "tab")
    assert "drag_drop" in state["nodes"][0]["operations"]
    assert "drag_drop" not in state["nodes"][1]["operations"]
    assert "select_text" in state["nodes"][2]["operations"]
    assert "select_text" not in state["nodes"][0]["operations"]
    assert all("switch_window" not in node["operations"] for node in state["nodes"])
    raw = page()
    raw["elements"][2].update(sensitive=True, text_selection={"start": 0, "end": 5})
    protected = semantic_state(raw, "session", "tab")["nodes"][2]
    assert protected["operations"] == [] and "text_selection" not in protected


def test_stale_browser_state_never_dispatches_pointer_or_keyboard(semantic_browser):
    store, _state, calls, action = semantic_browser
    stale = action("double_click")
    stale["revision"] = "older-state"
    assert store.semantic_perform("session", stale)["status"] == "rejected"
    assert calls == []
