"""Unrelated application state must not invalidate a scoped native read."""
from __future__ import annotations

import pytest
from test_native_ui import Fixture, Node, _named_node

from portal.native_ui import NativeUiUnavailable


def foreign_application(fixture):
    window = Node("/other-window", "frame", "Other application")
    application = Node("/other-app", "application", children=[window])
    application.process_id = 99
    fixture.desktop.children.insert(0, application)
    fixture.bridge._mapped_window = lambda _node, pid: str(1000 + pid)
    return application


@pytest.mark.parametrize("include_windows,process_scope", [(False, None), (False, 42), (True, 42)])
@pytest.mark.parametrize("fault", ["defunct", "read_error"])
def test_foreign_state_failure_does_not_break_scoped_read_or_input(include_windows, process_scope, fault):
    fixture = Fixture()
    foreign = foreign_application(fixture)
    fixture.bridge.process_id = process_scope
    state_reads = []

    def failed_state():
        state_reads.append(True)
        raise RuntimeError("foreign application state disappeared")

    if fault == "defunct":
        foreign.state.values.add("DEFUNCT")
        foreign.state_hook = lambda: state_reads.append(True)
    else:
        foreign.get_state_set = failed_state

    snapshot = fixture.bridge.observe(include_windows=include_windows)

    assert _named_node(snapshot, "Title")["value"] == ""
    assert snapshot["truncated"] is False
    assert ":1.42:/other-app" in fixture.bridge.previous
    assert not any(node["name"] == "Other application" for node in snapshot["nodes"])
    action = fixture.action()
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert fixture.button.events == ["invoke"]
    assert state_reads == []


@pytest.mark.parametrize("pid", [None, 0, -1, 2**31, True, "99", 42])
def test_unknown_or_controlled_application_state_failure_remains_fatal(pid):
    fixture = Fixture()
    foreign = foreign_application(fixture)
    fixture.bridge.observe()
    previous, handles = set(fixture.bridge.previous), dict(fixture.bridge.handles)
    foreign.process_id = pid
    foreign.state.values.add("DEFUNCT")

    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge.observe()

    assert error.value.code == "native_observation_unavailable"
    assert fixture.bridge.previous == previous
    assert fixture.bridge.handles == handles
    assert fixture.button.events == []


def test_foreign_process_metadata_failure_is_not_treated_as_exclusion():
    fixture = Fixture()
    foreign = foreign_application(fixture)

    def unavailable_pid():
        raise RuntimeError("application owner unknown")

    foreign.get_process_id = unavailable_pid
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge.observe()
    assert error.value.code == "native_observation_unavailable"


@pytest.mark.parametrize("discovery", ["all_windows", "namespace_candidate"])
def test_foreign_state_remains_required_when_discovering_other_providers(discovery):
    fixture = Fixture()
    foreign = foreign_application(fixture)
    foreign.state.values.add("DEFUNCT")
    if discovery == "namespace_candidate":
        fixture.application.process_id = 100
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge.observe(include_windows=discovery == "all_windows")
    assert error.value.code == "native_observation_unavailable"


def test_excluded_foreign_subtree_retains_identity_but_never_old_input_handles():
    fixture = Fixture()
    foreign = foreign_application(fixture)
    initial = _named_node(fixture.bridge.observe(include_windows=True), "Other application")
    assert initial["id"] in fixture.bridge.handles

    foreign.state.values.add("DEFUNCT")
    scoped = fixture.bridge.observe()

    assert initial["id"] not in fixture.bridge.handles
    assert all(node["id"] != initial["id"] for node in scoped["nodes"])
    assert ":1.42:/other-window" in fixture.bridge.previous
    assert fixture.bridge.generations[":1.42:/other-window"] == 1

    foreign.state.values.remove("DEFUNCT")
    restored = _named_node(fixture.bridge.observe(include_windows=True), "Other application")
    assert (restored["id"], restored["generation"]) == (initial["id"], initial["generation"])

    fixture.desktop.children.remove(foreign)
    fixture.bridge.observe()
    assert ":1.42:/other-window" not in fixture.bridge.previous
    fixture.desktop.children.insert(0, foreign)
    replaced = _named_node(fixture.bridge.observe(include_windows=True), "Other application")
    assert replaced["generation"] == "2" and replaced["id"] != initial["id"]
