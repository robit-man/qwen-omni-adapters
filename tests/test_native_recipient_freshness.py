"""Unrelated window drift admits focused text input only for an unchanged issued recipient."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_native_typing import typing_fixture
from test_native_ui import Node

STALE = {"status": "rejected", "reason": "stale_native_revision_or_scope"}


def recipient_fixture(monkeypatch, value="alpha", caret=5):
    fixture, commands, geometry = typing_fixture(monkeypatch, role="text")
    fixture.title.value, fixture.title.caret, fixture.title.selection = value, caret, None
    text = fixture.api.Text
    text.get_caret_offset = lambda node: node.caret
    text.get_n_selections = lambda node: 0 if node.selection is None else 1
    text.get_selection = lambda node, _index: SimpleNamespace(
        start_offset=node.selection[0], end_offset=node.selection[1])
    text.add_selection = text.set_selection = lambda *_args: True
    monkeypatch.setattr("portal.native_ui.subprocess.run", recorder(fixture, commands))
    return fixture, commands, geometry


def recorder(fixture, commands):
    def run(argv, **kwargs):
        commands.append(argv)
        assert "shell" not in kwargs
        if argv[1] == "type":
            fixture.title.value += argv[-1]
        return SimpleNamespace(returncode=0)
    return run


def drift(fixture):
    """Unrelated structural change after the action was issued (e.g. a progress bar)."""
    fixture.window.children.append(Node("/progress", "progress bar", ""))


@pytest.mark.parametrize("verb,args", [("type_text", {"text": "[x"}), ("key_chord", {"keys": ["shift", "Return"]})])
def test_unrelated_drift_with_unchanged_recipient_dispatches_once_with_joined_receipt(monkeypatch, verb, args):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    action = fixture.action("Title", verb, args)
    drift(fixture)
    result = fixture.adapter.perform(action)
    assert result["status"] == "dispatched" and result["backend"] == "x11_guarded_xtest"
    exception = result["freshness_exception"]
    assert exception["kind"] == "focused_recipient_context_unchanged"
    assert exception["issued_revision"] == action["revision"] != exception["dispatch_revision"]
    assert len(exception["recipient_sha256"]) == 64 and "alpha" not in str(exception)
    assert len(commands) == 1
    assert fixture.adapter.take_freshness_rejection() is None


def test_same_revision_dispatch_does_not_claim_the_exception(monkeypatch):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    result = fixture.adapter.perform(fixture.action("Title", "type_text", {"text": "b"}))
    assert result["status"] == "dispatched" and "freshness_exception" not in result and len(commands) == 1


@pytest.mark.parametrize("change", ["caret", "value", "selection", "focus", "generation", "foreground",
                                    "readonly", "name", "bounds_parent_index"])
def test_any_recipient_or_context_change_with_drift_rejects_before_input(monkeypatch, change):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    action = fixture.action("Title", "type_text", {"text": "x"})
    drift(fixture)
    if change == "caret":
        fixture.title.caret = 2
    elif change == "value":
        fixture.title.value = "alphA"
    elif change == "selection":
        fixture.title.selection = (0, 2)
    elif change == "focus":
        fixture.title.state.values.discard("FOCUSED")
    elif change == "generation":
        action["generation"] = "different"
    elif change == "foreground":
        fixture.foreground = {"window_id": "18", "process_id": 42}
    elif change == "readonly":
        fixture.title.state.values.add("READ_ONLY")
    elif change == "name":
        fixture.title.name = "Renamed"
    else:
        fixture.window.children.insert(0, Node("/inserted", "label", "Before"))
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    assert "freshness_exception" not in result
    assert commands == []


def test_issued_witness_is_one_use(monkeypatch):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    action = fixture.action("Title", "key_chord", {"keys": ["shift", "Return"]})
    drift(fixture)
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    drift(fixture)  # Further drift; the consumed witness cannot be replayed.
    assert fixture.adapter.perform(action) == STALE
    assert len(commands) == 1


def test_a_newer_observation_supersedes_the_issued_witness(monkeypatch):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    action = fixture.action("Title", "type_text", {"text": "x"})
    drift(fixture)
    fixture.adapter.observe()  # Newer issuance; the old action names a superseded revision.
    fixture.window.children.append(Node("/progress-2", "progress bar", ""))
    assert fixture.adapter.perform(action) == STALE
    assert commands == []


def test_non_recipient_verbs_and_window_scope_keys_keep_strict_revision_equality(monkeypatch):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    action = fixture.action("Preview", "invoke")
    drift(fixture)
    assert fixture.adapter.perform(action) == STALE
    window = next(node for node in fixture.adapter.observe()["nodes"] if node["id"] == node["scope_id"])
    state = fixture.adapter.observe()
    scoped = {**fixture.action("Title", "key_chord", {"keys": ["Escape"]}), "node_id": window["id"],
              "generation": window["generation"], "revision": state["revision"]}
    drift(fixture)
    assert fixture.adapter.perform(scoped) == STALE
    assert commands == [] and fixture.button.events == []


@pytest.mark.parametrize("unknown", ["no_caret_api", "invalid_caret", "truncated"])
def test_unknown_caret_or_incomplete_value_never_authorizes_drifted_input(monkeypatch, unknown):
    fixture, commands, _geometry = recipient_fixture(monkeypatch)
    if unknown == "no_caret_api":
        del fixture.api.Text.get_caret_offset
    elif unknown == "invalid_caret":
        fixture.title.caret = 99
    else:
        fixture.title.value = "z" * 2100
        fixture.title.caret = 0
    action = fixture.action("Title", "type_text", {"text": "x"})
    drift(fixture)
    assert fixture.adapter.perform(action) == STALE
    assert commands == []
