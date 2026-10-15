from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from portal.browser import BrowserAutomationStore
from portal.browser_ui import BrowserUiAdapter, semantic_state


def _page():
    return {"document_generation": "document-one", "url": "http://localhost/form",
            "title": "Settings", "visible_text": "Title Ready Preview", "viewport": {"width": 800},
            "elements": [
                {"id": "e1", "tag": "input", "type": "text", "role": "textbox",
                 "label": "Title", "value": "", "width": 150, "height": 30},
                {"id": "e2", "tag": "input", "type": "checkbox", "role": "checkbox",
                 "label": "Ready", "checked": False, "width": 20, "height": 20},
                {"id": "e3", "tag": "button", "role": "button", "text": "Preview",
                 "width": 100, "height": 30},
            ]}


def test_semantic_revision_is_stable_but_navigation_value_and_scope_invalidate():
    page = _page()
    original = semantic_state(page, "session", "tab")
    assert semantic_state(copy.deepcopy(page), "session", "tab") == original
    for field, value in (("document_generation", "document-two"), ("url", "http://localhost/other")):
        changed = copy.deepcopy(page)
        changed[field] = value
        assert semantic_state(changed, "session", "tab")["revision"] != original["revision"]
    page["elements"][0]["value"] = "new title"
    assert semantic_state(page, "session", "tab")["revision"] != original["revision"]
    assert semantic_state(_page(), "another-session", "tab")["revision"] != original["revision"]


def test_semantic_operations_are_native_and_sensitive_values_are_removed():
    page = _page()
    page["elements"].append({"id": "e4", "tag": "input", "type": "password", "value": "secret"})
    state = semantic_state(page, "session", "tab")
    pointer = ["focus", "click", "double_click", "right_click", "scroll", "key_chord"]
    assert state["nodes"][0]["operations"] == [*pointer, "set_value"]
    assert state["nodes"][1]["operations"] == [*pointer, "set_checked"]
    assert state["nodes"][2]["operations"] == [*pointer, "invoke"]
    assert state["nodes"][3]["sensitive"] is True
    assert state["nodes"][3]["operations"] == []
    assert state["nodes"][3]["value"] == ""


@pytest.mark.parametrize("changed", [False, True])
def test_semantic_dispatch_revalidates_under_store_lock_and_skips_screenshots(monkeypatch, changed):
    store = BrowserAutomationStore()
    session = SimpleNamespace(port=9999, target_id="tab", page_socket="", elements={})
    monkeypatch.setattr(store, "_session_locked", lambda session_id: session)
    monkeypatch.setattr(store, "_page_socket", lambda port, target: "ws://localhost/devtools/page/tab")
    page = _page()
    observed = semantic_state(page, "session", "tab")
    if changed:
        page["document_generation"] = "document-two"
    operations = []

    class Cdp:
        def __init__(self, *args):
            pass

        def call(self, *args):
            raise AssertionError("Semantic execution must not capture screenshots")

        def close(self):
            pass

    monkeypatch.setattr("portal.browser._Cdp", Cdp)
    monkeypatch.setattr(store, "_evaluate", lambda cdp, expression: copy.deepcopy(page))
    monkeypatch.setattr(store, "_refresh_element", lambda session, cdp, node_id: {"id": node_id})

    def set_value(cdp, element, value, *, semantic=False):
        assert store._lock._is_owned()
        assert semantic
        operations.append((element["id"], value))

    monkeypatch.setattr(store, "_set_value", set_value)
    receipt = store.semantic_perform("session", {
        "node_id": "e1", "generation": "document-one", "scope_id": observed["scope"]["id"],
        "revision": observed["revision"], "verb": "set_value", "args": {"value": "Updated"},
    })
    assert receipt["status"] == ("rejected" if changed else "dispatched")
    assert operations == ([] if changed else [("e1", "Updated")])


def test_visual_fallback_is_semantic_first_and_bound_to_current_source_frame():
    state = semantic_state(_page(), "session", "tab")
    calls = []
    current = True
    visual = {"node": {"id": "visual-apply", "name": "Canvas Apply"}, "revision": "visual-rev"}
    browser = SimpleNamespace(
        pointing_url="http://127.0.0.1:8940",
        semantic_observe=lambda session: copy.deepcopy(state),
        semantic_ground=lambda *args: calls.append(args) or visual,
        semantic_ground_is_current=lambda *args: current,
    )
    adapter = BrowserUiAdapter(browser, "session")
    adapter.prepare_step({"verb": "invoke", "target": {"role": "button", "name": "Preview"}})
    assert calls == []
    adapter.prepare_step({"verb": "invoke", "target": {"role": "button", "name": "Canvas Apply"}})
    assert len(calls) == 1
    assert adapter.observe()["revision"] == "visual-rev"
    current = False
    assert all(node["id"] != "visual-apply" for node in adapter.observe()["nodes"])


@pytest.mark.parametrize("stale", [False, True])
def test_visual_dispatch_rechecks_pixels_and_converts_dpr(monkeypatch, stale):
    from PIL import Image

    store = BrowserAutomationStore()
    session = SimpleNamespace(port=9999, target_id="tab")
    state = {"revision": "base", "scope": {"viewport": {"width": 800, "height": 600}}}
    node = {"id": "visual", "scope_id": "scope", "generation": "generation"}
    visual = {"node": node, "revision": "visual-rev", "base_revision": "base",
              "frame_sha256": "original", "x": 500, "y": 500, "receipt": {"target": "Apply"}}
    monkeypatch.setattr(store, "_session_locked", lambda session_id: session)
    monkeypatch.setattr(store, "_page_socket", lambda *args: "ws://localhost/tab")
    monkeypatch.setattr("portal.browser._Cdp", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(store, "_semantic_snapshot", lambda *args: state)
    monkeypatch.setattr(store, "_semantic_capture", lambda *args: (
        "changed" if stale else "original", Image.new("RGB", (1600, 1200))))
    clicks = []
    monkeypatch.setattr(store, "_click", lambda cdp, element: clicks.append(element))
    receipt = store.semantic_visual_perform("session", {
        "node_id": "visual", "scope_id": "scope", "generation": "generation",
        "revision": "visual-rev", "verb": "invoke",
    }, visual)
    assert receipt["status"] == ("rejected" if stale else "dispatched")
    if stale:
        assert clicks == []
    else:
        assert clicks[0]["x"] == pytest.approx(399.75)
        assert clicks[0]["y"] == pytest.approx(299.75)
