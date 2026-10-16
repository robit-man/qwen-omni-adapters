"""A missing child is unknown coverage, never disappearance or input authority."""
from __future__ import annotations

import pytest
from test_native_ui import Fixture, Node, _named_node, document_fixture

from portal.native_ui import NativeUiUnavailable
from qwen_omni_adapters.computer_use import predicate_satisfied


def test_null_child_preserves_current_siblings_and_explicit_incomplete_coverage():
    fixture = Fixture()
    fixture.window.children.insert(0, None)

    snapshot = fixture.adapter.observe()

    assert snapshot["truncated"] is True
    assert _named_node(snapshot, "Title")["enabled"] is True
    assert _named_node(snapshot, "Preview")["operations"] == ["invoke"]
    assert not predicate_satisfied({"kind": "visible", "name": "Unknown child", "visible": False}, snapshot)
    action = fixture.action()
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert fixture.button.events == ["invoke"]


def test_null_child_retains_identity_but_never_old_handle_or_stale_action():
    fixture = Fixture()
    original = fixture.action()
    original_item = _named_node(fixture.adapter.observe(), "Preview")
    index = fixture.window.children.index(fixture.button)
    fixture.window.children[index] = None

    partial = fixture.adapter.observe()

    assert partial["truncated"] is True
    assert all(node["id"] != original["node_id"] for node in partial["nodes"])
    assert original["node_id"] not in fixture.bridge.handles
    assert ":1.42:/preview" in fixture.bridge.previous
    assert fixture.bridge.generations[":1.42:/preview"] == 1
    assert fixture.adapter.perform(original) == {
        "status": "rejected", "reason": "stale_native_revision_or_scope"}
    current_revision_old_node = {**original, "revision": partial["revision"]}
    assert fixture.adapter.perform(current_revision_old_node) == {
        "status": "rejected", "reason": "native_target_missing"}
    assert fixture.button.events == []

    fixture.window.children[index] = fixture.button
    recovered = _named_node(fixture.adapter.observe(), "Preview")
    assert (recovered["id"], recovered["generation"]) == (
        original_item["id"], original_item["generation"])


def test_null_branch_does_not_prevent_complete_sibling_identity_retirement():
    fixture = Fixture()
    fixture.window.children.remove(fixture.button)
    unknown = Node("/unknown-child", "label", "Unknown branch child")
    complete_branch = Node("/complete", "panel", children=[fixture.button])
    incomplete_branch = Node("/incomplete", "panel", children=[unknown])
    fixture.window.children.extend([complete_branch, incomplete_branch])
    initial = fixture.adapter.observe()
    original = _named_node(initial, "Preview")
    original_unknown = _named_node(initial, "Unknown branch child")
    complete_branch.children.clear()
    incomplete_branch.children[:] = [None]

    partial = fixture.adapter.observe()

    assert partial["truncated"] is True
    assert ":1.42:/preview" not in fixture.bridge.previous
    assert ":1.42:/unknown-child" in fixture.bridge.previous
    assert original["id"] not in fixture.bridge.handles
    assert original_unknown["id"] not in fixture.bridge.handles
    complete_branch.children.append(fixture.button)
    incomplete_branch.children[:] = [unknown]
    recovered = fixture.adapter.observe()
    retired = _named_node(recovered, "Preview")
    retained = _named_node(recovered, "Unknown branch child")
    assert retired["generation"] == "2" and retired["id"] != original["id"]
    assert (retained["id"], retained["generation"]) == (
        original_unknown["id"], original_unknown["generation"])


@pytest.mark.parametrize("fault,code", [
    ("desktop", "native_desktop_unavailable"),
    ("provider", "native_observation_unavailable"),
    ("defunct", "native_observation_unavailable"),
    ("state_none", "native_observation_unavailable"),
    ("read_failure", "native_observation_unavailable"),
])
def test_null_child_compatibility_never_swallows_critical_provider_failures(fault, code):
    fixture = Fixture()
    fixture.adapter.observe()
    previous = set(fixture.bridge.previous)
    generations = dict(fixture.bridge.generations)
    handles = dict(fixture.bridge.handles)
    if fault == "desktop":
        fixture.api.get_desktop = lambda _index: None
    elif fault == "provider":
        fixture.desktop.children[:] = [None]
    elif fault == "defunct":
        fixture.button.state.values.add("DEFUNCT")
    elif fault == "state_none":
        fixture.button.get_state_set = lambda: None
    else:
        def denied(_index):
            raise PermissionError("private provider details")
        fixture.window.get_child_at_index = denied

    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()

    assert error.value.code == code
    assert "private provider details" not in str(error.value)
    assert fixture.bridge.previous == previous
    assert fixture.bridge.generations == generations
    assert fixture.bridge.handles == handles
    assert fixture.button.events == []


def test_null_child_in_document_root_scan_still_withholds_url_evidence():
    fixture, document = document_fixture()
    fixture.window.children.insert(0, None)

    state = fixture.adapter.observe()

    assert state["truncated"] is True
    assert _named_node(state, document.name)["role"] == "document_web"
    assert "url" not in state["scope"]
    assert state["document_evidence"]["reason"] == "native_document_root_scan_incomplete"


def test_null_descendant_within_proven_document_keeps_existing_root_url_contract():
    fixture, document = document_fixture()
    document.children[:] = [None, Node("/known-paragraph", "paragraph", "Known content")]

    state = fixture.adapter.observe()

    assert state["truncated"] is True
    assert _named_node(state, "Known content")["visible"] is True
    assert state["scope"]["url"] == "about:blank"
    assert state["document_evidence"]["status"] == "observed"
