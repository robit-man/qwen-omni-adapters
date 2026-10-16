from __future__ import annotations

import io
import json
import os
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from portal.native_ui import (
    NativeUiAdapter,
    NativeUiUnavailable,
    _AtspiBridge,
    _JsonBridge,
    _x11_foreground,
)

STATES = SimpleNamespace(
    **{
        name: name
        for name in (
            "ACTIVE",
            "CHECKABLE",
            "CHECKED",
            "DEFUNCT",
            "EDITABLE",
            "FOCUSABLE",
            "FOCUSED",
            "PRESSED",
            "READ_ONLY",
            "SENSITIVE",
            "SELECTED",
            "SHOWING",
            "VISIBLE",
        )
    }
)


class State:
    def __init__(self, *values: str) -> None:
        self.values = set(values)

    def contains(self, value: str) -> bool:
        return value in self.values


class Node:
    def __init__(
        self,
        path: str,
        role: str,
        name: str = "",
        *,
        children=None,
        interfaces=(),
        value="",
        states=None,
    ) -> None:
        self.app = SimpleNamespace(bus_name=":1.42")
        self.path, self.role, self.name = path, role, name
        self.children, self.interfaces = list(children or []), list(interfaces)
        self.value = value
        self.state = State(*(states if states is not None else ["SENSITIVE", "VISIBLE", "SHOWING"]))
        self.events: list[Any] = []
        self.selected: list[int] = []
        self.fail_action = False
        self.fail_selection = False
        self.state_hook = None
        self.process_id = 42

    def get_process_id(self):
        return self.process_id

    def clear_cache_single(self):
        pass

    def get_state_set(self):
        if self.state_hook:
            self.state_hook()
        return self.state

    def get_role_name(self):
        return self.role

    def get_name(self):
        return self.name

    def get_interfaces(self):
        return self.interfaces

    def get_child_count(self):
        return len(self.children)

    def get_child_at_index(self, index):
        return self.children[index]

    def get_action_iface(self):
        return self

    def get_n_actions(self):
        return 1

    def get_action_name(self, _index):
        return "click"

    def do_action(self, _index):
        self.events.append("invoke")
        if self.fail_action:
            raise RuntimeError("backend failed after submitting input")
        if self.role in {"check box", "toggle button"}:
            self.state.values.symmetric_difference_update({"CHECKED"})
        return True

    def get_text_iface(self):
        return self

    def get_editable_text_iface(self):
        return self

    def set_text_contents(self, value):
        self.events.append(("set_value", value))
        self.value = value
        return True

    def get_component_iface(self):
        return self

    def grab_focus(self):
        self.events.append("focus")
        self.state.values.add("FOCUSED")
        return True

    def get_selection_iface(self):
        return self

    def get_n_selected_children(self):
        return len(self.selected)

    def get_selected_child(self, index):
        return self.children[self.selected[index]]

    def clear_selection(self):
        self.events.append("clear_selection")
        self.selected.clear()
        return True

    def select_child(self, index):
        self.events.append(("select", index))
        if self.fail_selection:
            raise RuntimeError("selection failed after clearing")
        self.selected = [index]
        return True


class Fixture:
    def __init__(self, session="one") -> None:
        self.title = Node(
            "/title",
            "entry",
            "Title",
            interfaces=["Text", "EditableText", "Component"],
            states=["SENSITIVE", "VISIBLE", "SHOWING", "EDITABLE", "FOCUSABLE"],
        )
        self.check = Node("/ready", "check box", "Ready", interfaces=["Action"])
        self.button = Node("/preview", "push button", "Preview", interfaces=["Action"])
        self.secret = Node(
            "/password",
            "password text",
            "secret-value-as-name",
            interfaces=["Text", "EditableText", "Action", "Component"],
            value="never-export-this",
        )
        self.selection = Node(
            "/select",
            "list box",
            "Items",
            interfaces=["Selection"],
            children=[Node("/a", "list item", "A"), Node("/b", "list item", "B")],
        )
        self.window = Node(
            "/window",
            "frame",
            "Fixture",
            children=[self.title, self.check, self.button, self.secret, self.selection],
            states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"],
        )
        self.application = Node("/app", "application", children=[self.window])
        self.desktop = Node("/desktop", "desktop frame", children=[self.application])
        self.api = SimpleNamespace(
            StateType=STATES,
            init=lambda: None,
            set_timeout=lambda *_args: None,
            get_desktop=lambda _index: self.desktop,
            Text=SimpleNamespace(
                get_character_count=lambda node: len(node.value),
                get_text=lambda node, start, end: node.value[start:end],
            ),
        )
        context = SimpleNamespace(pending=lambda: False)
        self.glib = SimpleNamespace(MainContext=SimpleNamespace(default=lambda: context))
        self.foreground = {"window_id": "17", "process_id": 42}
        self.bridge = _AtspiBridge(self.api, self.glib, session,
                                   foreground=lambda: dict(self.foreground))
        self.adapter = NativeUiAdapter(session, bridge_runner=self.request)

    def request(self, request):
        return (
            self.bridge.observe(include_windows=request.get("include_windows", False),
                                window_target=request.get("window_target"))
            if request["op"] == "observe"
            else self.bridge.perform(request["action"])
        )

    def action(self, name="Preview", verb="invoke", args=None):
        snapshot = self.adapter.observe()
        node = next(node for node in snapshot["nodes"] if node["name"] == name)
        return {
            "session_id": self.bridge.session_id,
            "kind": "action",
            "id": "opaque-choice",
            "node_id": node["id"],
            "generation": node["generation"],
            "scope_id": snapshot["scope"]["id"],
            "revision": snapshot["revision"],
            "verb": verb,
            "args": args or {},
            "expect": [{"kind": "text", "text": "Verified"}],
            "idempotency_key": "owned-by-controller",
        }


def legacy_cache_fixture() -> Fixture:
    fixture = Fixture()
    pending = [fixture.desktop]
    while pending:
        node = pending.pop()
        pending.extend(node.children)
        node.clear_cache_single = None
        node.cache_refreshes = 0
        node.cache_hook = None

        def clear_cache(target=node):
            target.cache_refreshes += 1
            if target.cache_hook:
                target.cache_hook()

        node.clear_cache = clear_cache
    return fixture


def test_cache_invalidation_prefers_the_single_node_api() -> None:
    fixture = Fixture()
    calls = []
    node = SimpleNamespace(clear_cache_single=lambda: calls.append("single"),
                           clear_cache=lambda: calls.append("descendants"))
    fixture.adapter.observe()
    before = fixture.bridge.reads
    fixture.bridge._clear_cache(node)
    assert calls == ["single"]
    assert fixture.bridge.reads == before + 1


@pytest.mark.parametrize("single", [None, False])
def test_cache_invalidation_supports_the_documented_older_api(single) -> None:
    fixture = Fixture()
    calls = []
    fixture.adapter.observe()
    node = SimpleNamespace(clear_cache_single=single, clear_cache=lambda: calls.append("cleared"))
    fixture.bridge._clear_cache(node)
    assert calls == ["cleared"]
    fixture.bridge._clear_cache(SimpleNamespace(clear_cache=lambda: calls.append("absent-single")))
    assert calls == ["cleared", "absent-single"]


def test_cache_invalidation_missing_both_apis_fails_closed() -> None:
    fixture = Fixture()
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge._clear_cache(SimpleNamespace())
    assert error.value.code == "native_cache_invalidation_unavailable"


@pytest.mark.parametrize("old_only", [False, True])
def test_cache_invalidation_errors_never_retry_or_bypass_the_budget(old_only) -> None:
    fixture = Fixture()
    calls = []

    def broken():
        calls.append("failed")
        raise RuntimeError("provider cache failure")

    fixture.adapter.observe()
    node = SimpleNamespace(clear_cache=broken) if old_only else SimpleNamespace(
        clear_cache_single=broken, clear_cache=lambda: calls.append("forbidden-retry"))
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge._clear_cache(node)
    assert error.value.code == "native_observation_unavailable"
    assert calls == ["failed"]
    fixture.bridge.reads = fixture.bridge.MAX_READS
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge._clear_cache(node)
    assert error.value.code == "native_observation_budget_exceeded"
    assert calls == ["failed"]
    fixture.bridge.reads = 0
    fixture.bridge.deadline = 0
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge._clear_cache(node)
    assert error.value.code == "native_observation_budget_exceeded"
    assert calls == ["failed"]


def test_older_cache_api_observes_and_dispatches_with_fresh_invalidations() -> None:
    fixture = legacy_cache_fixture()
    action = fixture.action()
    result = fixture.adapter.perform(action)
    assert result == {"status": "dispatched", "reason": "native_pattern_invoked"}
    assert fixture.button.events == ["invoke"]
    assert fixture.desktop.cache_refreshes >= 2
    assert fixture.window.cache_refreshes >= 4
    assert fixture.button.cache_refreshes >= 4


def test_older_cache_api_rejects_a_stale_candidate() -> None:
    fixture = legacy_cache_fixture()
    action = fixture.action()
    fixture.button.name = "Changed target"
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    assert result["reason"] == "stale_native_revision_or_scope"
    assert fixture.button.events == []


@pytest.mark.parametrize("change", ["disabled", "hidden", "protected", "inactive-window"])
def test_older_cache_refresh_preserves_native_input_guards(change) -> None:
    fixture = legacy_cache_fixture()
    action = fixture.action()
    target = fixture.window if change == "inactive-window" else fixture.button

    def changed_on_final_refresh():
        if target.cache_refreshes != 4:
            return
        if change == "disabled":
            target.state.values.discard("SENSITIVE")
        elif change == "hidden":
            target.state.values.discard("SHOWING")
        elif change == "protected":
            target.role = "password text"
        else:
            target.state.values.discard("ACTIVE")

    target.cache_hook = changed_on_final_refresh
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    assert result["reason"] in {"native_target_changed_before_dispatch", "native_focus_changed_before_dispatch",
                                "stale_native_revision_or_scope"}
    assert fixture.button.events == []


def test_older_cache_refresh_revalidates_focused_keyboard_target(monkeypatch) -> None:
    fixture = legacy_cache_fixture()
    fixture.title.state.values.add("FOCUSED")
    action = fixture.action("Title", "key_chord", {"keys": ["Escape"]})
    inputs = []
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda _identity: {"x": 0})
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda command, **_kw: inputs.append(command))

    def lose_focus():
        if fixture.title.cache_refreshes == 5:
            fixture.title.state.values.discard("FOCUSED")

    fixture.title.cache_hook = lose_focus
    result = fixture.adapter.perform(action)
    assert result["status"] == "rejected"
    assert result["reason"] == "native_key_target_not_focused"
    assert inputs == []
    assert fixture.title.events == []


def test_snapshot_is_stable_scoped_and_redacts_protected_controls() -> None:
    fixture = Fixture()
    first, second = fixture.adapter.observe(), fixture.adapter.observe()
    assert first == second
    assert first["scope"]["kind"] == "desktop"
    password = next(node for node in first["nodes"] if node["sensitive"])
    assert password["name"] == "[protected]"
    assert password["operations"] == []
    assert "value" not in password
    assert "secret-value-as-name" not in str(first)
    assert "never-export-this" not in str(first)
    assert all(
        node["generation"] == "1" and node["scope_id"] == first["scope"]["id"]
        for node in first["nodes"]
    )
    # GTK4 uses SENSITIVE without ENABLED; these real native buttons remain legal.
    assert next(node for node in first["nodes"] if node["name"] == "Preview")["enabled"]


def test_native_identity_is_isolated_per_session_and_reappearance_changes_generation() -> None:
    fixture, other = Fixture(), Fixture("two")
    first = fixture.action()
    assert first["node_id"] != other.action()["node_id"]
    assert other.adapter.perform(first)["status"] == "rejected"
    fixture.window.children.remove(fixture.button)
    fixture.adapter.observe()
    fixture.window.children.append(fixture.button)
    renewed = fixture.action()
    assert renewed["generation"] == "2"
    assert renewed["node_id"] != first["node_id"]


def test_failed_active_window_observation_does_not_publish_text_disappearance() -> None:
    fixture = Fixture()
    initial = fixture.adapter.observe()
    original = next(node for node in initial["nodes"] if node["name"] == "Title")
    generations = dict(fixture.bridge.generations)
    previous = set(fixture.bridge.previous)
    handles = dict(fixture.bridge.handles)
    fixture.window.state.values.remove("ACTIVE")
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_active_window_unavailable"
    assert fixture.bridge.generations == generations
    assert fixture.bridge.previous == previous
    assert fixture.bridge.handles == handles
    fixture.window.state.values.add("ACTIVE")
    current = next(node for node in fixture.adapter.observe()["nodes"] if node["name"] == "Title")
    assert (current["id"], current["generation"]) == (original["id"], "1")


def test_new_node_seen_only_during_failed_read_does_not_consume_a_generation() -> None:
    fixture = Fixture()
    fixture.adapter.observe()
    generations = dict(fixture.bridge.generations)
    added = Node("/new", "label", "New")
    fixture.window.children.append(added)
    def unreadable():
        raise RuntimeError("Native name read failed")
    added.get_name = unreadable
    with pytest.raises(NativeUiUnavailable):
        fixture.adapter.observe()
    assert fixture.bridge.generations == generations
    added.get_name = lambda: "New"
    current = next(node for node in fixture.adapter.observe()["nodes"] if node["name"] == "New")
    assert current["generation"] == "1"


def test_late_failed_visibility_read_does_not_commit_new_window_history(monkeypatch) -> None:
    fixture = Fixture()
    initial = fixture.adapter.observe()
    generations, previous = dict(fixture.bridge.generations), set(fixture.bridge.previous)
    history = fixture.bridge._window_history.copy()
    fixture.window.state.values.remove("ACTIVE")
    popup = Node("/new-popup", "dialog", "New popup", states=["ACTIVE", "VISIBLE", "SHOWING", "SENSITIVE"])
    fixture.application.children.append(popup)
    original = fixture.bridge._window_visibility
    def unavailable(*_args, **_kwargs):
        raise NativeUiUnavailable("native_observation_budget_exceeded")
    monkeypatch.setattr(fixture.bridge, "_window_visibility", unavailable)
    with pytest.raises(NativeUiUnavailable):
        fixture.adapter.observe()
    assert fixture.bridge.generations == generations
    assert fixture.bridge.previous == previous
    assert fixture.bridge._window_history == history
    monkeypatch.setattr(fixture.bridge, "_window_visibility", original)
    fixture.application.children.remove(popup)
    fixture.window.state.values.add("ACTIVE")
    current = fixture.adapter.observe()
    assert current["scope"] == initial["scope"]
    assert next(node for node in current["nodes"] if node["name"] == "Title")["generation"] == "1"


@pytest.mark.parametrize(
    ("name", "verb", "args", "event"),
    [
        ("Preview", "invoke", {}, "invoke"),
        ("Title", "set_value", {"value": "Native fixture"}, ("set_value", "Native fixture")),
        ("Title", "focus", {}, "focus"),
        ("Ready", "set_checked", {"checked": True}, "invoke"),
        ("Items", "select", {"option": "B"}, ("select", 1)),
    ],
)
def test_native_patterns_dispatch_and_observation_reports_actual_effects(
    name, verb, args, event
) -> None:
    fixture = Fixture()
    action = fixture.action(name, verb, args)
    result = fixture.adapter.perform(action)
    assert result == {"status": "dispatched", "reason": "native_pattern_invoked"}
    node = {
        "Preview": fixture.button,
        "Title": fixture.title,
        "Ready": fixture.check,
        "Items": fixture.selection,
    }[name]
    assert event in node.events
    after = fixture.adapter.observe()
    observed = next(item for item in after["nodes"] if item["name"] == name)
    if verb == "set_value":
        assert observed["value"] == "Native fixture"
    elif verb == "set_checked":
        assert observed["checked"] is True
    elif verb == "focus":
        assert observed["focused"] is True
    elif verb == "select":
        assert observed["selected"] == ["B"]


def test_set_checked_is_a_state_operation_without_repeated_toggle() -> None:
    fixture = Fixture()
    fixture.adapter.perform(fixture.action("Ready", "set_checked", {"checked": True}))
    result = fixture.adapter.perform(fixture.action("Ready", "set_checked", {"checked": True}))
    assert result["reason"] == "native_checked_already_satisfied"
    assert fixture.check.events == ["invoke"]


@pytest.mark.parametrize("change", ["text", "disabled", "hidden", "missing", "scope", "generation"])
def test_stale_or_ineligible_action_never_dispatches(change) -> None:
    fixture = Fixture()
    action = fixture.action()
    if change == "text":
        fixture.title.value = "changed after policy choice"
    elif change == "disabled":
        fixture.button.state.values.remove("SENSITIVE")
    elif change == "hidden":
        fixture.button.state.values.remove("SHOWING")
    elif change == "missing":
        fixture.window.children.remove(fixture.button)
    elif change == "scope":
        action["scope_id"] = "another-window"
    else:
        action["generation"] = "999"
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert fixture.button.events == []


def test_native_permission_or_read_failure_is_not_reported_as_a_missing_target() -> None:
    fixture = Fixture()
    action = fixture.action()

    def fail():
        raise PermissionError("private dbus details")

    fixture.button.get_state_set = fail
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.perform(action)
    assert error.value.code == "native_observation_unavailable"
    assert "private dbus details" not in str(error.value)
    assert fixture.button.events == []


def test_inconsistent_optional_action_names_withhold_invocation() -> None:
    fixture = Fixture()
    fixture.button.get_n_actions = lambda: 8

    def unavailable(_index):
        raise RuntimeError("provider advertises an unavailable action")

    fixture.button.get_action_name = unavailable
    state = fixture.adapter.observe()
    button = next(node for node in state["nodes"] if node["name"] == "Preview")
    assert "invoke" not in button["operations"]
    assert fixture.adapter.perform(fixture.action())["status"] == "rejected"
    assert fixture.button.events == []
    assert next(node for node in state["nodes"] if node["name"] == "Title")["value"] == ""


def test_optional_action_metadata_never_swallows_observation_budget() -> None:
    fixture = Fixture()

    def exhaust_budget():
        fixture.bridge.reads = fixture.bridge.MAX_READS
        return 8

    fixture.button.get_n_actions = exhaust_budget
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_observation_budget_exceeded"
    assert fixture.button.events == []


def test_unknown_target_is_rejected_after_a_successful_fresh_observation() -> None:
    fixture = Fixture()
    action = {**fixture.action(), "node_id": "forged-id"}
    assert fixture.adapter.perform(action)["reason"] == "native_target_missing"
    assert fixture.button.events == []


@pytest.mark.parametrize("race", ["focus", "password", "checked"])
def test_native_input_revalidates_after_resolving_the_native_interface(race) -> None:
    fixture = Fixture()
    target = fixture.check if race == "checked" else fixture.button
    action = fixture.action("Ready", "set_checked", {"checked": True}) if race == "checked" else fixture.action()
    calls = 0

    def get_interface():
        nonlocal calls
        calls += 1
        if calls == 2:  # Fresh observation succeeded; final interface resolution changes state.
            if race == "focus":
                fixture.window.state.values.remove("ACTIVE")
            elif race == "password":
                target.role = "password text"
            else:
                target.state.values.add("CHECKED")
        return target

    target.get_action_iface = get_interface
    result = fixture.adapter.perform(action)
    assert result["status"] == ("dispatched" if race == "checked" else "rejected")
    assert target.events == []


def test_action_session_must_match_even_when_all_native_handles_match() -> None:
    fixture = Fixture()
    result = fixture.adapter.perform({**fixture.action(), "session_id": "another-session"})
    assert result == {"status": "rejected", "reason": "native_session_mismatch"}
    assert fixture.button.events == []


@pytest.mark.parametrize(
    ("verb", "args"),
    [
        ("key_chord", {"keys": ["--window"]}),
        ("set_value", {"value": "x", "code": "arbitrary"}),
        ("invoke", {"command": "touch forbidden"}),
        ("set_checked", {"checked": 1}),
        ("set_value", {"value": "x" * 8193}),
        ("select", {"option": None}),
    ],
)
def test_unimplemented_operations_and_free_form_arguments_are_rejected(verb, args) -> None:
    fixture = Fixture()
    action = {**fixture.action(), "verb": verb, "args": args}
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert fixture.button.events == []


def test_partial_native_selection_and_throwing_input_are_dispatched_unresolved() -> None:
    fixture = Fixture()
    fixture.selection.selected = [0]
    fixture.selection.fail_selection = True
    result = fixture.adapter.perform(fixture.action("Items", "select", {"option": "B"}))
    assert result == {"status": "dispatched", "reason": "native_pattern_effect_unresolved"}
    assert fixture.selection.selected == []
    fixture.button.fail_action = True
    result = fixture.adapter.perform(fixture.action())
    assert result["status"] == "dispatched"
    assert fixture.button.events == ["invoke"]


def test_ambiguous_native_selection_rejects_before_clearing() -> None:
    fixture = Fixture()
    fixture.selection.children[0].name = "B"
    fixture.selection.selected = [0]
    result = fixture.adapter.perform(fixture.action("Items", "select", {"option": "B"}))
    assert result["reason"] == "native_option_not_unique"
    assert fixture.selection.events == []


def test_traversal_and_failed_reads_are_bounded_without_claiming_target_absence() -> None:
    fixture = Fixture()
    fixture.bridge.MAX_NODES = 5
    observed = fixture.adapter.observe()
    assert observed["truncated"] is True
    assert len(observed["nodes"]) <= 5
    fixture.bridge.MAX_READS = 2
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_observation_budget_exceeded"


def test_inactive_window_subtrees_cannot_consume_the_active_tree_budget() -> None:
    fixture = Fixture()
    inactive = Node(
        "/inactive", "frame", children=[Node(f"/hidden{i}", "label", str(i)) for i in range(1000)]
    )
    fixture.application.children.insert(0, inactive)
    observed = fixture.adapter.observe()
    assert observed["truncated"] is False
    assert any(node["name"] == "Preview" for node in observed["nodes"])
    assert all(node["name"] != "999" for node in observed["nodes"])


def closed_popup_fixture():
    fixture = Fixture()
    fixture.adapter.observe()
    popup = Node("/popup", "window", children=[Node("/popup-menu", "menu")],
                 states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"])
    fixture.window.state.values.remove("ACTIVE")
    fixture.application.children.append(popup)
    opened = fixture.adapter.observe()
    popup_item = next(node for node in opened["nodes"] if node["role"] == "window")
    fixture.application.children.remove(popup)
    popup.state.values.difference_update({"ACTIVE", "SHOWING", "VISIBLE"})
    fixture.window.state.values.add("ACTIVE")
    return fixture, popup, popup_item


def test_detached_known_popup_has_fresh_hidden_visibility_without_input_authority() -> None:
    fixture, _popup, popup_item = closed_popup_fixture()
    state = fixture.adapter.observe()
    receipt = next(item for item in state["window_visibility"] if item["id"] == popup_item["id"])
    assert receipt == {"id": popup_item["id"], "generation": popup_item["generation"],
        "owner_window_id": "17", "source": "retained_atspi_window", "active": False, "visible": False, "showing": False}
    assert all(node["id"] != popup_item["id"] for node in state["nodes"])
    assert popup_item["id"] not in fixture.bridge.handles
    action = {"session_id": "one", "scope_id": state["scope"]["id"], "revision": state["revision"],
              "node_id": popup_item["id"], "generation": popup_item["generation"], "verb": "focus", "args": {}}
    assert fixture.adapter.perform(action)["reason"] == "native_target_missing"


@pytest.mark.parametrize("invalid", ["defunct", "read-error", "generation", "identity", "role"])
def test_unknown_defunct_or_reused_retained_window_never_proves_hidden(invalid) -> None:
    fixture, popup, popup_item = closed_popup_fixture()
    if invalid == "defunct":
        popup.state.values.add("DEFUNCT")
    elif invalid == "read-error":
        def failed_state():
            raise RuntimeError("optional detached window state unavailable")
        popup.get_state_set = failed_state
    elif invalid == "generation":
        raw = popup.app.bus_name + ":" + popup.path
        fixture.bridge.generations[raw] += 1
    elif invalid == "identity":
        popup.path = "/reused-popup"
    else:
        popup.role = "password text"
    assert all(item["id"] != popup_item["id"] for item in fixture.adapter.observe()["window_visibility"])


def test_retained_window_budget_errors_remain_fail_closed() -> None:
    fixture, popup, _popup_item = closed_popup_fixture()
    original = fixture.bridge._read
    def read(function):
        if getattr(function, "__self__", None) is popup and getattr(function, "__name__", "") == "get_state_set":
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        return original(function)
    fixture.bridge._read = read
    with pytest.raises(NativeUiUnavailable, match="Native accessibility") as exc:
        fixture.adapter.observe()
    assert exc.value.code == "native_observation_budget_exceeded"


def test_retained_window_history_is_bounded_lru_and_owner_isolated() -> None:
    fixture = Fixture()
    first = fixture.adapter.observe()["scope"]["id"]
    for index in range(17):
        fixture.application.children = [Node(f"/window-{index}", "frame", states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"])]
        state = fixture.adapter.observe()
    assert len(fixture.bridge._window_history) == 16
    assert len(state["window_visibility"]) == 16
    assert all(item["id"] != first for item in state["window_visibility"])
    fixture.application.process_id = 99
    fixture.foreground.update(process_id=99, window_id="21")
    state = fixture.adapter.observe()
    assert all(item["owner_window_id"] == "21" for item in state["window_visibility"])


def test_persistent_bridge_refreshes_desktop_registration_cache() -> None:
    class CachedDesktop(Node):
        def clear_cache_single(self):
            self.cached = list(self.children)

        def get_child_count(self):
            return len(self.cached)

        def get_child_at_index(self, index):
            return self.cached[index]

    fixture = Fixture()
    desktop = CachedDesktop("/desktop", "desktop frame", children=[fixture.application])
    desktop.clear_cache_single()
    fixture.desktop = desktop
    fixture.adapter.observe()
    new_window = Node("/new-window", "frame", "Newly registered fixture",
                      states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"])
    desktop.children = [Node("/new-app", "application", children=[new_window])]
    state = fixture.adapter.observe()
    assert any(node["name"] == "Newly registered fixture" for node in state["nodes"])
    assert all(node["name"] != "Fixture" for node in state["nodes"])


def test_host_process_scope_disambiguates_compositor_projection_without_guessing() -> None:
    fixture = Fixture()
    proxy = Node("/compositor-window", "frame", "Fixture",
                 states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"])
    compositor = Node("/compositor", "application", children=[proxy])
    compositor.process_id = 99
    fixture.desktop.children.insert(0, compositor)
    state = fixture.adapter.observe()  # Actual X11 foreground PID excludes the compositor projection.
    assert any(node["name"] == "Preview" for node in state["nodes"])
    bridge = _AtspiBridge(fixture.api, fixture.glib, "scoped-session", process_id=42,
                         foreground=lambda: dict(fixture.foreground))
    state = bridge.observe()
    assert any(node["name"] == "Preview" for node in state["nodes"])
    assert len([node for node in state["nodes"] if node["role"] == "window"]) == 1
    with pytest.raises(ValueError):
        NativeUiAdapter("invalid-scope", process_id=True)
    fixture.foreground["process_id"] = 99
    with pytest.raises(NativeUiUnavailable) as error:
        bridge.observe()
    assert error.value.code == "native_foreground_process_scope_changed"


def test_foreground_manager_change_at_input_boundary_rejects_native_input() -> None:
    fixture = Fixture()
    action = fixture.action()
    calls = 0

    def get_interface():
        nonlocal calls
        calls += 1
        if calls == 2:
            fixture.foreground["window_id"] = "18"
        return fixture.button

    fixture.button.get_action_iface = get_interface
    result = fixture.adapter.perform(action)
    assert result["reason"] == "native_foreground_changed_before_dispatch"
    assert fixture.button.events == []


def test_foreground_mapper_uses_only_fixed_queries_and_rechecks_window(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":fixture")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls = []

    def run(command, **options):
        calls.append(command)
        assert options["timeout"] == 1
        return SimpleNamespace(returncode=0, stdout="42" if command[1] == "getwindowpid" else "17")

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert _x11_foreground() == {"window_id": "17", "process_id": 42}
    assert calls == [["xdotool", "getactivewindow"], ["xdotool", "getwindowpid", "17"],
                     ["xdotool", "getactivewindow"]]
    monkeypatch.setenv("WAYLAND_DISPLAY", "fixture-wayland")
    with pytest.raises(NativeUiUnavailable) as error:
        _x11_foreground()
    assert error.value.code == "native_foreground_wayland_unsupported"


def test_foreground_mapper_rejects_unmapped_or_racing_windows(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":fixture")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    responses = iter(["17", "42", "18"])
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=next(responses)))
    with pytest.raises(NativeUiUnavailable) as error:
        _x11_foreground()
    assert error.value.code == "native_foreground_changed_during_observation"
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=1, stdout="unmapped"))
    with pytest.raises(NativeUiUnavailable) as error:
        _x11_foreground()
    assert error.value.code == "native_foreground_unmapped"


def test_closed_session_fails_explicitly_and_sensitive_bridge_values_are_scrubbed_again() -> None:
    fixture = Fixture()
    snapshot = fixture.adapter.observe()
    protected = next(node for node in snapshot["nodes"] if node["sensitive"])
    protected.update(
        name="leaked", value="leaked", selected=["leaked"], checked=True, operations=["invoke"]
    )
    adapter = NativeUiAdapter("one", bridge_runner=lambda _request: snapshot)
    assert "leaked" not in str(adapter.observe())
    adapter.close()
    with pytest.raises(NativeUiUnavailable) as error:
        adapter.observe()
    assert error.value.code == "native_session_closed"


def test_key_chord_is_window_bound_fixed_argv_with_fresh_foreground_check(monkeypatch) -> None:
    fixture = Fixture()
    action = fixture.action("Fixture", "key_chord", {"keys": ["ctrl", "s"]})
    commands = []
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: {"x": 0, "y": 0, "width": 100, "height": 100})
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda command, **_kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0))
    assert fixture.adapter.perform(action) == {"status": "dispatched", "backend": "x11_guarded_xtest",
                                               "reason": "native_key_input_invoked"}
    assert commands == [["xdotool", "key", "--clearmodifiers", "ctrl+s"]]
    fixture.foreground["window_id"] = "18"
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert len(commands) == 1


@pytest.mark.parametrize("target", ["Fixture", "Title"])
@pytest.mark.parametrize("outcome", ["ok", "nonzero", "oserror", "timeout"])
def test_key_chord_attempt_attests_backend_and_never_retries_uncertain_input(monkeypatch, target, outcome) -> None:
    fixture = Fixture()
    fixture.title.state.values.add("FOCUSED")
    action = fixture.action(target, "key_chord", {"keys": ["ctrl", "a"]})
    attempts = []
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda _identity: {"x": 0, "y": 0, "width": 100, "height": 100})

    def run(command, **options):
        attempts.append(command)
        assert options == {"capture_output": True, "timeout": 2, "check": False}
        if outcome == "oserror":
            raise OSError("input command may have partially run")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 2)
        return SimpleNamespace(returncode=0 if outcome == "ok" else 1)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    assert fixture.adapter.perform(action) == {
        "status": "dispatched", "backend": "x11_guarded_xtest",
        "reason": "native_key_input_invoked" if outcome == "ok" else "native_key_input_effect_unresolved",
    }
    assert attempts == [["xdotool", "key", "--clearmodifiers", "ctrl+a"]]
    assert fixture.title.events == []  # No second focus, setter, or fallback.


def test_key_chord_changed_geometry_rejects_before_attempt_without_backend_claim(monkeypatch) -> None:
    fixture = Fixture()
    action = fixture.action("Fixture", "key_chord", {"keys": ["ctrl", "s"]})
    reads, attempts = [], []

    def geometry(_identity):
        reads.append(True)
        return {"x": 0 if len(reads) == 1 else 10, "y": 0, "width": 100, "height": 100}

    monkeypatch.setattr("portal.native_ui._x11_geometry", geometry)
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda command, **_kw: attempts.append(command))
    assert fixture.adapter.perform(action) == {"status": "rejected", "reason": "native_key_binding_changed_before_dispatch"}
    assert len(reads) == 2 and attempts == []


def test_native_key_chord_is_offered_only_on_focused_eligible_control_besides_window(monkeypatch) -> None:
    fixture = Fixture()
    initial = fixture.adapter.observe()
    assert "key_chord" not in next(node for node in initial["nodes"] if node["name"] == "Title")["operations"]
    assert fixture.adapter.perform(fixture.action("Title", "focus"))["status"] == "dispatched"
    fixture.secret.state.values.add("FOCUSED")
    fixture.check.state.values.add("FOCUSED")
    fixture.check.state.values.remove("SENSITIVE")
    state = fixture.adapter.observe()
    assert "key_chord" in next(node for node in state["nodes"] if node["name"] == "Title")["operations"]
    assert all("key_chord" not in node["operations"] for node in state["nodes"] if node["sensitive"] or not node["enabled"])
    commands = []
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: {"x": 0, "y": 0, "width": 100, "height": 100})
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    action = fixture.action("Title", "key_chord", {"keys": ["ctrl", "a"]})
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert commands == [["xdotool", "key", "--clearmodifiers", "ctrl+a"]]
    assert fixture.title.events == ["focus"]  # No implicit second focus/input fallback.


@pytest.mark.parametrize("lost_at", [3, 4])
def test_native_focused_control_key_chord_rechecks_focus_at_both_final_boundaries(monkeypatch, lost_at) -> None:
    fixture = Fixture()
    fixture.title.state.values.add("FOCUSED")
    action = fixture.action("Title", "key_chord", {"keys": ["ctrl", "a"]})
    reads, commands = 0, []
    def change_focus():
        nonlocal reads
        reads += 1
        if reads == lost_at:
            fixture.title.state.values.remove("FOCUSED")
    fixture.title.state_hook = change_focus
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda *_args, **_kwargs: {"x": 0, "y": 0, "width": 100, "height": 100})
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    assert fixture.adapter.perform(action) == {"status": "rejected", "reason": "native_key_target_not_focused"}
    assert commands == [] and fixture.title.events == []


@pytest.mark.parametrize("keys", [["ctrl+s"], ["s;rm"], ["$(pwd)"], ["--clearmodifiers"], [], ["a"]*5])
def test_key_chord_rejects_unbounded_or_non_key_tokens(keys) -> None:
    fixture = Fixture()
    action = fixture.action("Fixture", "key_chord", {"keys": keys})
    assert fixture.adapter.perform(action)["reason"] == "invalid_native_arguments"


def test_native_sessions_serialize_observation_and_input_across_instances() -> None:
    entered, release, second_called = threading.Event(), threading.Event(), threading.Event()

    def first_runner(_request):
        entered.set()
        assert release.wait(2)
        return {"session_id": "first", "nodes": []}

    def second_runner(_request):
        second_called.set()
        return {"session_id": "second", "nodes": []}

    first = NativeUiAdapter("first", bridge_runner=first_runner)
    second = NativeUiAdapter("second", bridge_runner=second_runner)
    left, right = threading.Thread(target=first.observe), threading.Thread(target=second.observe)
    left.start()
    assert entered.wait(1)
    right.start()
    try:
        assert not second_called.wait(0.05)
    finally:
        release.set()
        left.join(2)
        right.join(2)
    assert second_called.is_set()


def test_complete_native_error_reply_allows_a_later_fresh_observation(monkeypatch) -> None:
    monkeypatch.setattr("portal.native_ui.uuid.uuid4", lambda: SimpleNamespace(hex="request"))
    read_fd, write_fd = os.pipe()
    bridge = _JsonBridge.__new__(_JsonBridge)
    bridge.session_id, bridge.timeout_s, bridge.closed = "fixture-session", 1, False
    bridge.process_id = None
    bridge.window_scope = None
    with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb") as writer:
        bridge.process = SimpleNamespace(stdin=io.BytesIO(), stdout=reader)
        envelope = {"schema": "robit.omni.native-ui.v1", "id": "request"}
        writer.write(json.dumps({**envelope, "ok": False, "error": "native_active_window_unavailable"}).encode() + b"\n")
        writer.flush()
        with pytest.raises(NativeUiUnavailable) as error:
            bridge.request({"op": "observe"})
        assert error.value.code == "native_active_window_unavailable"
        assert bridge.closed is False
        writer.write(json.dumps({**envelope, "ok": True, "result": {"recovered": True}}).encode() + b"\n")
        writer.flush()
        assert bridge.request({"op": "observe"}) == {"recovered": True}


def document_fixture(uri="about:blank"):
    fixture = Fixture()
    document = Node("/document", "document web", "Committed page", interfaces=["Document"])
    document.document_url = uri
    document.get_document_iface = lambda: document
    fixture.window.children.append(document)
    fixture.api.Document = SimpleNamespace(
        get_document_attribute_value=lambda node, attribute: node.document_url if attribute == "DocURL" else ""
    )
    return fixture, document


def test_native_document_url_proves_committed_navigation_not_addressbar_text():
    from qwen_omni_adapters.computer_use import predicate_satisfied

    fixture, document = document_fixture("https://docs.python.org/3/library/pathlib.html")
    fixture.title.value = "https://www.gnu.org/software/bash/manual/"
    first = fixture.adapter.observe()
    evidence = first["document_evidence"]
    node = next(node for node in first["nodes"] if node["role"] == "document_web")
    assert first["scope"]["url"] == document.document_url
    assert evidence == {"status": "observed", "source": "atspi.Document.DocURL", "url": document.document_url,
                        "node_id": node["id"], "generation": node["generation"], "scope_id": first["scope"]["id"],
                        "window_id": "17", "provider": "atspi2",
                        "provider_identity_sha256": evidence["provider_identity_sha256"]}
    assert len(evidence["provider_identity_sha256"]) == 64
    assert not predicate_satisfied({"kind": "url", "url": fixture.title.value}, first)
    assert fixture.adapter.observe()["revision"] == first["revision"]
    document.document_url = fixture.title.value
    second = fixture.adapter.observe()
    assert predicate_satisfied({"kind": "url", "url": fixture.title.value}, second)
    assert second["revision"] != first["revision"]
    assert second["document_evidence"]["node_id"] == node["id"]
    assert not document.events and not fixture.title.events


def test_native_document_url_preserves_provider_uri_and_model_projection():
    from portal.app import _model_ui_observation

    uri = "https://example.org/A%2fb?q=%2f#ExactCase"
    fixture, _ = document_fixture(uri)
    snapshot = fixture.adapter.observe()
    projected = _model_ui_observation(snapshot)
    assert snapshot["scope"]["url"] == uri
    assert projected["scope"] == snapshot["scope"]
    assert projected["document_evidence"] == snapshot["document_evidence"]


def test_native_document_url_ignores_chrome_inactive_and_nested_documents():
    fixture, document = document_fixture()
    fixture.window.interfaces.append("Document")
    fixture.window.get_document_iface = lambda: fixture.window
    fixture.window.document_url = "chrome://browser/content/browser.xhtml"
    nested = Node("/iframe", "document web", interfaces=["Document"])
    nested.get_document_iface = lambda: nested
    nested.document_url = "https://iframe.example/"
    document.children.append(nested)
    inactive_doc = Node("/inactive-doc", "document web", interfaces=["Document"])
    inactive_doc.get_document_iface = lambda: inactive_doc
    inactive_doc.document_url = "https://inactive.example/"
    fixture.application.children.append(Node("/inactive-window", "frame", children=[inactive_doc]))
    state = fixture.adapter.observe()
    assert state["scope"]["url"] == "about:blank"
    document.state.values.remove("SHOWING")
    hidden = fixture.adapter.observe()
    assert "url" not in hidden["scope"]  # Never promote a nested document through its hidden root.
    assert hidden["document_evidence"]["status"] == "unavailable"


@pytest.mark.parametrize("same_uri", [False, True])
def test_native_document_url_requires_unique_visible_top_level_document(same_uri):
    fixture, _ = document_fixture()
    second = Node("/second-doc", "document web", interfaces=["Document"])
    second.get_document_iface = lambda: second
    second.document_url = "about:blank" if same_uri else "https://second.example/"
    fixture.window.children.append(second)
    state = fixture.adapter.observe()
    assert "url" not in state["scope"]
    assert state["document_evidence"]["reason"] == "document_root_not_unique"


@pytest.mark.parametrize("uri", [None, 42, "", "http://", "https://[bad", "https://example.org:bad/",
                                   "https://example.org/has space", "about:blank\x00", "relative/path"])
def test_native_document_url_withholds_malformed_provider_values(uri):
    fixture, _ = document_fixture(uri)
    state = fixture.adapter.observe()
    assert "url" not in state["scope"]
    assert state["document_evidence"]["status"] == "unavailable"


@pytest.mark.parametrize("unavailable", ["api", "method", "iface", "error", "defunct", "hidden", "role"])
def test_optional_document_capability_failure_preserves_native_observation(unavailable):
    fixture, document = document_fixture()
    if unavailable == "api":
        del fixture.api.Document
    elif unavailable == "method":
        fixture.api.Document = SimpleNamespace()
    elif unavailable == "iface":
        document.interfaces.clear()
    elif unavailable == "error":
        def fail(*_args):
            raise RuntimeError("provider unavailable")
        fixture.api.Document.get_document_attribute_value = fail
    elif unavailable == "defunct":
        # The already valid ordinary snapshot must still survive an optional read failure.
        document.get_document_iface = lambda: document.state.values.add("DEFUNCT") or document
    elif unavailable == "hidden":
        document.state.values.remove("VISIBLE")
    else:
        document.role = "panel"
    state = fixture.adapter.observe()
    assert state["nodes"] and "url" not in state["scope"]
    assert state["document_evidence"]["status"] == "unavailable"


def test_document_body_truncation_does_not_hide_structural_root_uniqueness():
    fixture, document = document_fixture()
    document.children = [Node(f"/body-{index}", "paragraph") for index in range(fixture.bridge.MAX_CHILDREN + 1)]
    state = fixture.adapter.observe()
    assert state["truncated"] is True
    assert state["scope"]["url"] == "about:blank"
    document.children.clear()
    fixture.window.children.extend(Node(f"/chrome-{index}", "panel") for index in range(fixture.bridge.MAX_CHILDREN))
    incomplete = fixture.adapter.observe()
    assert incomplete["truncated"] is True
    assert "url" not in incomplete["scope"]
    assert incomplete["document_evidence"]["reason"] == "native_document_root_scan_incomplete"


@pytest.mark.parametrize("change", ["uri", "foreground", "identity", "role", "visibility"])
def test_native_document_url_rechecks_fresh_document_and_foreground_binding(change):
    fixture, document = document_fixture()
    reads = []

    def getter(node, _attribute):
        reads.append(True)
        value = node.document_url
        if len(reads) == 1:
            if change == "uri":
                node.document_url = "https://changed.example/"
            elif change == "foreground":
                fixture.foreground["window_id"] = "18"
            elif change == "identity":
                node.path = "/changed-document"
            elif change == "role":
                node.role = "panel"
            else:
                node.state.values.remove("SHOWING")
        return value

    fixture.api.Document.get_document_attribute_value = getter
    state = fixture.adapter.observe()
    assert "url" not in state["scope"]
    assert state["document_evidence"]["status"] == "unavailable"


def test_optional_document_reads_preserve_fatal_budget_and_transactional_history():
    fixture, document = document_fixture()
    fixture.adapter.observe()
    previous, generations = set(fixture.bridge.previous), dict(fixture.bridge.generations)

    def exhaust(node, _attribute):
        fixture.bridge.reads = fixture.bridge.MAX_READS
        return node.document_url

    fixture.api.Document.get_document_attribute_value = exhaust
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_observation_budget_exceeded"
    assert fixture.bridge.previous == previous
    assert fixture.bridge.generations == generations


def test_document_url_provenance_tracks_positive_disappearance_and_generation():
    fixture, document = document_fixture()
    first = fixture.adapter.observe()
    fixture.window.children.remove(document)
    missing = fixture.adapter.observe()
    assert "url" not in missing["scope"]
    fixture.window.children.append(document)
    renewed = fixture.adapter.observe()
    assert renewed["document_evidence"]["generation"] == "2"
    assert renewed["document_evidence"]["node_id"] != first["document_evidence"]["node_id"]


def test_hidden_generic_ancestor_cannot_hide_a_second_visible_document_root():
    fixture, _ = document_fixture()
    other = Node("/second-visible-doc", "document web", interfaces=["Document"])
    other.get_document_iface = lambda: other
    other.document_url = "https://second.example/"
    fixture.window.children.append(Node("/hidden-container", "panel", children=[other], states=[]))
    state = fixture.adapter.observe()
    assert "url" not in state["scope"]
    assert state["document_evidence"]["reason"] == "document_root_not_unique"


def test_protected_nonleaf_withholds_document_root_completeness():
    fixture, _ = document_fixture()
    fixture.secret.children.append(Node("/protected-descendant", "panel"))
    state = fixture.adapter.observe()
    assert "url" not in state["scope"]
    assert state["document_evidence"]["reason"] == "native_document_root_scan_incomplete"
    assert "never-export-this" not in json.dumps(state)


def _named_node(state, name):
    return next(node for node in state['nodes'] if node['name'] == name)


def test_unchanged_tail_keeps_generation_through_planning_budget_omission():
    fixture = Fixture()
    tail = Node('/unchanged-tail', 'label', 'Unchanged tail')
    fixture.window.children.append(tail)
    for index in range(2):
        window = Node(f'/other-window-{index}', 'frame', f'Other {index}')
        app = Node(f'/other-app-{index}', 'application', children=[window])
        app.process_id = 100 + index
        fixture.desktop.children.insert(index, app)
    fixture.bridge._mapped_window = lambda _node, pid: str(1000 + pid)
    fixture.bridge.MAX_NODES = 13
    initial = fixture.bridge.observe()
    first = _named_node(initial, 'Unchanged tail')
    assert initial['truncated'] is False
    planning = fixture.bridge.observe(include_windows=True)
    assert planning['truncated'] is True
    assert not any(node['id'] == first['id'] for node in planning['nodes'])
    assert sum(node['operations'] == ['switch_window'] for node in planning['nodes']) == 2
    assert fixture.window.children[-1] is tail  # Same exact provider object and native path throughout.
    renewed = _named_node(fixture.bridge.observe(), 'Unchanged tail')
    assert (renewed['id'], renewed['generation']) == (first['id'], first['generation'])


def test_untruncated_inactive_window_exclusion_is_not_document_disappearance():
    fixture = Fixture()
    other_entry = Node('/other-entry', 'entry', 'Other document')
    other_window = Node('/other-window', 'frame', 'Other editor', children=[other_entry])
    fixture.application.children.append(other_window)
    initial = fixture.bridge.observe()
    first = _named_node(initial, 'Title')
    fixture.window.state.values.remove('ACTIVE')
    other_window.state.values.add('ACTIVE')
    fixture.foreground['window_id'] = '18'
    other = fixture.bridge.observe()
    assert other['truncated'] is False
    assert not any(node['id'] == first['id'] for node in other['nodes'])
    other_window.state.values.remove('ACTIVE')
    fixture.window.state.values.add('ACTIVE')
    fixture.foreground['window_id'] = '17'
    current = _named_node(fixture.bridge.observe(), 'Title')
    assert (current['id'], current['generation']) == (first['id'], first['generation'])


def test_complete_active_scope_disappearance_still_renews_generation():
    fixture = Fixture()
    first = _named_node(fixture.bridge.observe(), 'Preview')
    fixture.window.children.remove(fixture.button)
    missing = fixture.bridge.observe()
    assert missing['truncated'] is False
    assert not any(node['id'] == first['id'] for node in missing['nodes'])
    fixture.window.children.append(fixture.button)
    current = _named_node(fixture.bridge.observe(), 'Preview')
    assert current['generation'] == '2'
    assert current['id'] != first['id']


@pytest.mark.parametrize("limit", ["MAX_NODES", "MAX_CHILDREN", "MAX_DEPTH"])
def test_incomplete_child_coverage_preserves_unknown_identities_without_input_handles(limit):
    fixture = Fixture()
    first = _named_node(fixture.bridge.observe(), "Preview")
    original_limit = getattr(fixture.bridge, limit)
    setattr(fixture.bridge, limit, {"MAX_NODES": 4, "MAX_CHILDREN": 2, "MAX_DEPTH": 1}[limit])
    partial = fixture.bridge.observe()
    assert partial["truncated"] is True
    assert first["id"] not in fixture.bridge.handles
    assert ":1.42:/preview" in fixture.bridge.previous
    assert all(node["id"] != first["id"] for node in partial["nodes"])
    action = {"session_id": fixture.bridge.session_id, "revision": partial["revision"],
              "scope_id": partial["scope"]["id"], "node_id": first["id"],
              "generation": first["generation"], "verb": "invoke", "args": {}}
    assert fixture.bridge.perform(action) == {"status": "rejected", "reason": "native_target_missing"}
    assert not fixture.button.events
    setattr(fixture.bridge, limit, original_limit)
    current = _named_node(fixture.bridge.observe(), "Preview")
    assert (current["id"], current["generation"]) == (first["id"], first["generation"])


def test_complete_removal_retires_identity_even_when_other_branch_is_truncated():
    fixture = Fixture()
    fixture.window.children.remove(fixture.button)
    completed_branch = Node("/complete-branch", "panel", children=[fixture.button])
    other_branch = Node("/incomplete-branch", "panel")
    fixture.window.children.extend([completed_branch, other_branch])
    first = _named_node(fixture.bridge.observe(), "Preview")
    completed_branch.children.clear()
    other_branch.children = [Node(f"/unseen-{index}", "label") for index in range(129)]
    partial = fixture.bridge.observe()
    assert partial["truncated"] is True
    assert ":1.42:/preview" not in fixture.bridge.previous
    completed_branch.children.append(fixture.button)
    current = _named_node(fixture.bridge.observe(), "Preview")
    assert current["generation"] == "2" and current["id"] != first["id"]


def test_complete_parent_removal_retires_a_previously_excluded_window_subtree():
    fixture = Fixture()
    other = Node("/other-window", "frame", children=[Node("/other-text", "entry", "Other")],
                 states=["ACTIVE", "SENSITIVE", "SHOWING", "VISIBLE"])
    fixture.window.state.values.remove("ACTIVE")
    fixture.application.children.append(other)
    first = _named_node(fixture.bridge.observe(), "Other")
    other.state.values.remove("ACTIVE")
    fixture.window.state.values.add("ACTIVE")
    fixture.bridge.observe()
    assert ":1.42:/other-text" in fixture.bridge.previous
    fixture.application.children.remove(other)
    fixture.bridge.observe()
    assert ":1.42:/other-text" not in fixture.bridge.previous
    assert ":1.42:/other-window" not in fixture.bridge._identity_children
    fixture.window.state.values.remove("ACTIVE")
    other.state.values.add("ACTIVE")
    fixture.application.children.append(other)
    current = _named_node(fixture.bridge.observe(), "Other")
    assert current["generation"] == "2" and current["id"] != first["id"]


@pytest.mark.parametrize("uncertain", ["duplicate", "cycle"])
def test_duplicate_or_cyclic_native_coverage_cannot_prove_identity_absence(uncertain):
    fixture = Fixture()
    first = _named_node(fixture.bridge.observe(), "Preview")
    fixture.window.children.remove(fixture.button)
    if uncertain == "duplicate":
        fixture.window.children.append(fixture.title)
    else:
        fixture.selection.children.append(fixture.window)
    partial = fixture.bridge.observe()
    assert partial["truncated"] is True
    assert first["id"] not in fixture.bridge.handles
    assert ":1.42:/preview" in fixture.bridge.previous
    fixture.window.children.append(fixture.button)
    current = _named_node(fixture.bridge.observe(), "Preview")
    assert (current["id"], current["generation"]) == (first["id"], first["generation"])


def test_failed_read_rolls_back_identity_child_coverage_and_presence():
    fixture = Fixture()
    fixture.bridge.observe()
    old_coverage = {parent: set(children) for parent, children in fixture.bridge._identity_children.items()}
    old_presence = set(fixture.bridge.previous)
    old_generations = dict(fixture.bridge.generations)
    new = Node("/new-before-failure", "label", "New")
    fixture.window.children.insert(0, new)
    fixture.button.get_name = lambda: (_ for _ in ()).throw(RuntimeError("late provider failure"))
    with pytest.raises(NativeUiUnavailable):
        fixture.bridge.observe()
    assert fixture.bridge._identity_children == old_coverage
    assert fixture.bridge.previous == old_presence
    assert fixture.bridge.generations == old_generations
    assert ":1.42:/new-before-failure" not in fixture.bridge.generations


def test_identity_edge_storage_has_a_fatal_transactional_budget():
    fixture = Fixture()
    fixture.bridge.observe()
    old_coverage = {parent: set(children) for parent, children in fixture.bridge._identity_children.items()}
    old_presence = set(fixture.bridge.previous)
    old_generations = dict(fixture.bridge.generations)
    fixture.bridge.MAX_IDENTITIES = len(old_generations)
    fixture.check.children.append(fixture.title)  # Another edge, no additional provider identity.
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge.observe()
    assert error.value.code == "native_identity_budget_exceeded"
    assert fixture.bridge._identity_children == old_coverage
    assert fixture.bridge.previous == old_presence
    assert fixture.bridge.generations == old_generations


def _switch_fixture(monkeypatch):
    fixture = Fixture()
    target = Node('/notes', 'frame', 'Notes')
    unrelated = Node('/unrelated', 'frame', 'Unrelated')
    fixture.application.children.extend([target, unrelated])
    visible_mapping = {'target': '101', 'unrelated': '102'}
    activations = []

    def mapped(node, _pid):
        if node is target:
            return visible_mapping['target']
        if node is unrelated:
            return visible_mapping['unrelated']
        return '201' if node.name == 'Notes' else '17'

    def run(argv, **_kwargs):
        assert argv == ['xdotool', 'windowactivate', '--sync', visible_mapping['target']]
        activations.append(argv)
        fixture.window.state.values.remove('ACTIVE')
        target.state.values.add('ACTIVE')
        fixture.foreground['window_id'] = visible_mapping['target']
        return SimpleNamespace(returncode=0)

    fixture.bridge._mapped_window = mapped
    monkeypatch.setattr('portal.native_ui._process_start', lambda pid: f'{pid}01')
    monkeypatch.setattr('portal.native_ui.subprocess.run', run)
    step = {'id': 'switch-to-notes', 'verb': 'switch_window',
            'target': {'role': 'window', 'name': 'Notes'}, 'args': {},
            'expect': [{'kind': 'active_window', 'node_id': '$target'}]}
    fixture.adapter.prepare_step(step)
    from qwen_omni_adapters.computer_use import RapidUiController

    control = RapidUiController(fixture.adapter, authorize=lambda _action, _state: True,
                               max_observations=1, observation_interval_s=.001)
    state = control.observe()
    action = next(a for a in control.enumerate(step, state) if a['kind'] == 'action')
    return fixture, target, unrelated, visible_mapping, activations, control, action


def test_unrelated_inactive_membership_must_not_block_an_unchanged_switch_target(monkeypatch):
    fixture, target, unrelated, mapping, activations, control, action = _switch_fixture(monkeypatch)
    first = control._state
    mapping['unrelated'] = None
    fresh = fixture.adapter.observe()
    old_target = next(n for n in first['nodes'] if n['id'] == action['node_id'])
    new_target = next(n for n in fresh['nodes'] if n['id'] == action['node_id'])
    assert old_target == new_target
    assert fresh['scope'] == first['scope']
    assert first['revision'] == fresh['revision']
    assert all(n == next(m for m in fresh['nodes'] if m['id'] == n['id'])
               for n in first['nodes'] if n['scope_id'] == first['scope']['id'])
    assert target.path == '/notes' and unrelated.path == '/unrelated'
    outcome = control.execute(action['id'])
    assert outcome['status'] == 'verified', outcome.get('reason')
    assert len(activations) == 1


def test_native_executor_has_the_same_global_revision_conflict(monkeypatch):
    fixture, _target, _unrelated, mapping, activations, _control, action = _switch_fixture(monkeypatch)
    mapping['unrelated'] = None
    receipt = fixture.bridge.perform(action)
    assert receipt['status'] == 'dispatched', receipt.get('reason')
    assert len(activations) == 1


@pytest.mark.parametrize('change', ['active_text', 'target_missing', 'target_hidden', 'target_disabled', 'foreground', 'authority'])
def test_fail_closed_controls_for_a_future_scoped_switch_revision(monkeypatch, change):
    fixture, target, _unrelated, _mapping, activations, control, action = _switch_fixture(monkeypatch)
    if change == 'active_text':
        fixture.title.value = 'Changed current document'
    elif change == 'target_missing':
        fixture.application.children.remove(target)
    elif change == 'target_hidden':
        target.state.values.remove('SHOWING')
    elif change == 'target_disabled':
        target.state.values.remove('SENSITIVE')
    elif change == 'foreground':
        fixture.foreground['window_id'] = '999'
    else:
        control.authorize = lambda _action, _state: False
    outcome = control.execute(action['id'])
    assert outcome['status'] == 'rejected'
    assert not activations


@pytest.mark.parametrize("change", ["xid", "pid", "process_start", "generation", "query"])
def test_prepared_switch_binds_exact_target_identity_mapping_and_query(monkeypatch, change):
    fixture, target, _unrelated, mapping, activations, control, action = _switch_fixture(monkeypatch)
    if change == "xid":
        mapping["target"] = "201"
    elif change == "pid":
        target.process_id = 43
    elif change == "process_start":
        monkeypatch.setattr("portal.native_ui._process_start", lambda _pid: "new-start")
    elif change == "generation":
        fixture.application.children.remove(target)
        fixture.adapter.observe()
        fixture.application.children.append(target)
    else:
        fixture.adapter.prepare_step({"id": "different-query", "verb": "switch_window",
            "target": {"node_id": action["node_id"]}, "args": {}, "expect": action["expect"]})
    result = control.execute(action["id"])
    assert result["status"] == ("unresolved" if change == "pid" else "rejected")
    assert not activations


def test_switch_query_preserves_all_observed_matches_and_new_ambiguity(monkeypatch):
    fixture, _target, _unrelated, _mapping, activations, control, action = _switch_fixture(monkeypatch)
    fixture.application.children.append(Node("/notes-second", "frame", "Notes"))
    result = control.execute(action["id"])
    assert result["status"] == "rejected"
    step = {"id": "ambiguous-notes", "verb": "switch_window", "target": {"role": "window", "name": "Notes"},
            "args": {}, "expect": [{"kind": "active_window", "node_id": "$target"}]}
    fresh = control.run_step(step)
    assert fresh["status"] == "abstained"
    assert sum(node["name"] == "Notes" for node in control._state["nodes"]) == 2
    assert not activations


def test_prepared_switch_query_is_copied_and_planning_and_non_switch_clear_it(monkeypatch):
    fixture, _target, _unrelated, _mapping, _activations, _control, action = _switch_fixture(monkeypatch)
    step = {"id": "copy-query", "verb": "switch_window", "target": {"name": "Notes"},
            "args": {}, "expect": action["expect"]}
    fixture.adapter.prepare_step(step)
    step["target"]["name"] = "Unrelated"
    scoped = fixture.adapter.observe()
    assert fixture.bridge._window_target == {"name": "Notes"}
    assert {n["name"] for n in scoped["nodes"] if "switch_window" in n["operations"]} == {"Notes"}
    assert scoped["window_query_sha256"]
    broad = fixture.adapter.planning_observe()
    assert broad["window_query_sha256"] is None
    assert {n["name"] for n in broad["nodes"] if "switch_window" in n["operations"]} == {"Notes", "Unrelated"}
    fixture.adapter.prepare_step({"id": "ordinary", "verb": "focus", "target": {"name": "Title"},
                                  "expect": [{"kind": "focused", "node_id": "$target"}]})
    ordinary = fixture.adapter.observe()
    assert ordinary["window_query_sha256"] is None
    assert fixture.bridge._window_target is None


def test_unprepared_direct_native_switch_keeps_broad_compatibility(monkeypatch):
    fixture, _target, _unrelated, _mapping, activations, _control, action = _switch_fixture(monkeypatch)
    broad = fixture.bridge.observe(include_windows=True)
    assert broad["window_query_sha256"] is None
    action = {**action, "revision": broad["revision"]}
    assert fixture.bridge.perform(action)["status"] == "dispatched"
    assert len(activations) == 1


def test_switch_rechecks_positive_mapping_at_final_activation_boundary(monkeypatch):
    fixture, _target, _unrelated, mapping, activations, control, action = _switch_fixture(monkeypatch)
    original = fixture.bridge._mapped_window
    calls = []

    def moving_mapping(node, owner):
        if node.name == "Notes":
            calls.append(True)
            if len(calls) >= 4:  # core read, bridge read, builder guard, final activation guard
                return "201"
        return original(node, owner)

    fixture.bridge._mapped_window = moving_mapping
    result = control.execute(action["id"])
    assert result["status"] == "rejected"
    assert result["receipt"]["reason"] == "native_switch_window_mapping_changed"
    assert mapping["target"] == "101"
    assert not activations


def test_scoped_switch_still_rejects_active_evidence_displaced_by_traversal_cap(monkeypatch):
    fixture, target, _unrelated, _mapping, activations, control, _action = _switch_fixture(monkeypatch)
    fixture.application.children.remove(target)
    fixture.application.children.insert(0, target)
    fixture.window.children.extend(Node(f"/bounded-tail-{i}", "label", f"Tail {i}") for i in range(20))
    fixture.bridge.MAX_NODES = 20
    step = {"id": "capped-switch", "verb": "switch_window", "target": {"name": "Notes"},
            "expect": [{"kind": "active_window", "node_id": "$target"}]}
    fixture.adapter.prepare_step(step)
    before = control.observe()
    action = next(a for a in control.enumerate(step, before) if a["kind"] == "action")
    fixture.application.children.insert(0, Node("/unrelated-new", "frame", "Another"))
    result = control.execute(action["id"])
    assert result["status"] == "rejected"
    assert result["before"]["truncated"] is True
    assert not activations


def test_switch_process_start_unavailable_aborts_target_query(monkeypatch):
    fixture, _target, _unrelated, _mapping, activations, control, action = _switch_fixture(monkeypatch)

    def unavailable(_owner):
        raise OSError("process disappeared")

    monkeypatch.setattr("portal.native_ui._process_start", unavailable)
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_switch_window_binding_unavailable"
    assert control.execute(action["id"])["status"] == "unresolved"
    assert not activations


def test_failed_native_read_rolls_back_prepared_query_and_current_handles(monkeypatch):
    fixture, _target, _unrelated, _mapping, activations, _control, _action = _switch_fixture(monkeypatch)
    target, handles = dict(fixture.bridge._window_target), dict(fixture.bridge.handles)
    previous, generations = set(fixture.bridge.previous), dict(fixture.bridge.generations)
    fixture.button.get_name = lambda: (_ for _ in ()).throw(RuntimeError("provider failure"))
    with pytest.raises(NativeUiUnavailable):
        fixture.bridge.observe(include_windows=True, window_target={"name": "Unrelated"})
    assert fixture.bridge._window_target == target
    assert fixture.bridge.handles == handles
    assert fixture.bridge.previous == previous
    assert fixture.bridge.generations == generations
    assert not activations


@pytest.mark.parametrize("query", [{}, {"scope_id": "x"}, {"name": 4}, {"node_id": ""}, {"unknown": "x"}])
def test_invalid_window_query_is_rejected_without_publishing_state(monkeypatch, query):
    fixture, _target, _unrelated, _mapping, activations, _control, _action = _switch_fixture(monkeypatch)
    target, handles = dict(fixture.bridge._window_target), dict(fixture.bridge.handles)
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.bridge.observe(include_windows=True, window_target=query)
    assert error.value.code == "native_window_query_invalid"
    assert fixture.bridge._window_target == target and fixture.bridge.handles == handles
    assert not activations


@pytest.mark.parametrize("failure", ["unreadable_start", "mismatched_owner"])
def test_uncertain_binding_of_one_matching_window_cannot_create_false_uniqueness(monkeypatch, failure):
    fixture, _target, _unrelated, _mapping, activations, control, action = _switch_fixture(monkeypatch)
    prior_handles = dict(fixture.bridge.handles)
    prior_presence = set(fixture.bridge.previous)
    prior_query = dict(fixture.bridge._window_target)
    second = Node("/notes-second", "frame", "Notes")
    fixture.application.children.append(second)
    if failure == "mismatched_owner":
        second.process_id = 43
    else:
        reads = []

        def start(owner):
            reads.append(owner)
            if len(reads) % 2 == 0:
                raise OSError("one matching process identity is unknown")
            return f"{owner}01"

        monkeypatch.setattr("portal.native_ui._process_start", start)
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_switch_window_binding_unavailable"
    assert fixture.bridge.handles == prior_handles
    assert fixture.bridge.previous == prior_presence
    assert fixture.bridge._window_target == prior_query
    assert control.execute(action["id"])["status"] == "unresolved"
    assert not activations


def test_window_binding_read_budget_is_fatal_and_rolls_back_query(monkeypatch):
    fixture, _target, _unrelated, _mapping, activations, _control, _action = _switch_fixture(monkeypatch)
    prior_handles = dict(fixture.bridge.handles)
    prior_query = dict(fixture.bridge._window_target)

    def exhaust(owner):
        fixture.bridge.reads = fixture.bridge.MAX_READS
        return f"{owner}01"

    monkeypatch.setattr("portal.native_ui._process_start", exhaust)
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.observe()
    assert error.value.code == "native_observation_budget_exceeded"
    assert fixture.bridge.handles == prior_handles
    assert fixture.bridge._window_target == prior_query
    assert not activations


def namespace_fixture(monkeypatch):
    """A proxy provider with independently mocked private proof; no real GI/X11."""
    fixture = Fixture()
    fixture.application.process_id = 99
    fixture.window.interfaces = ["Component"]
    fixture.api.CoordType = SimpleNamespace(SCREEN="screen", WINDOW="window")
    geometry = {"x": 70, "y": 64, "width": 925, "height": 1016}
    fixture.window.get_extents = lambda _space: SimpleNamespace(**geometry)
    monkeypatch.setattr("portal.native_ui._x11_geometry", lambda _identity: dict(geometry))
    proof = {"source": "flatpak_namespace_metadata", "identity": dict(fixture.foreground),
             "reported_pid": 99, "bus_name": ":1.42", "values": {"private": "never-model-facing"}, "digest": "a" * 64}
    calls = []

    def provider(application, reported):
        calls.append((application, reported))
        return json.loads(json.dumps(proof)) if reported == 99 else None

    fixture.bridge._provider_proof = provider
    return fixture, proof, calls, geometry


def test_namespace_provider_observation_keeps_host_identity_and_private_proof(monkeypatch):
    fixture, proof, calls, _geometry = namespace_fixture(monkeypatch)
    state = fixture.adapter.observe()
    assert state["scope"]["window_id"] == "17"
    assert "Title" in [item["name"] for item in state["nodes"]]
    assert fixture.bridge._provider_binding["proof"] == proof
    assert fixture.bridge._provider_binding["application"] == ":1.42:/app"
    assert fixture.adapter._provider_digest == proof["digest"]
    assert "never-model-facing" not in json.dumps(state)
    assert "_namespace_provider_digest" not in state and len(calls) == 2


def test_direct_host_provider_does_not_load_namespace_helper(monkeypatch):
    fixture = Fixture()
    other = Node("/other-app", "application")
    other.process_id = 99
    fixture.desktop.children.insert(0, other)
    monkeypatch.setattr("portal.native_ui._identity_helper", lambda: pytest.fail("direct apps need no namespace proof"))
    fixture.adapter.observe()
    assert fixture.bridge._provider_binding is None and fixture.adapter._provider_digest is None


def test_namespace_proof_digest_changes_revision_without_changing_ui_ids(monkeypatch):
    fixture, proof, _calls, _geometry = namespace_fixture(monkeypatch)
    first = fixture.adapter.observe()
    proof["digest"] = "b" * 64
    second = fixture.adapter.observe()
    assert first["revision"] != second["revision"]
    assert [(node["id"], node["generation"]) for node in first["nodes"]] == [
        (node["id"], node["generation"]) for node in second["nodes"]]


@pytest.mark.parametrize("fault", ["ambiguous_provider", "ambiguous_frame", "missing_frame", "ineligible_frame", "geometry", "incomplete", "proof_changed"])
def test_namespace_observation_rejects_unknown_ambiguous_or_changed_binding_and_rolls_back(monkeypatch, fault):
    fixture, proof, calls, geometry = namespace_fixture(monkeypatch)
    first = fixture.adapter.observe()
    old_binding = fixture.bridge._provider_binding
    old_handles = fixture.bridge.handles
    old_generations = dict(fixture.bridge.generations)
    if fault == "ambiguous_provider":
        other = Node("/other-app", "application")
        other.process_id = 99
        fixture.desktop.children.append(other)
    elif fault == "ambiguous_frame":
        fixture.application.children.append(Node("/second-frame", "frame", states=["ACTIVE", "VISIBLE", "SHOWING", "SENSITIVE"]))
    elif fault == "missing_frame":
        fixture.window.state.values.discard("ACTIVE")
    elif fault == "ineligible_frame":
        fixture.window.state.values.discard("SENSITIVE")
    elif fault == "geometry":
        fixture.window.get_extents = lambda _space: SimpleNamespace(**{**geometry, "x": 71})
    elif fault == "incomplete":
        fixture.bridge.MAX_CHILDREN = 0
    elif fault == "proof_changed":
        original = fixture.bridge._provider_proof

        def changing(application, reported):
            result = original(application, reported)
            if len(calls) % 2 == 0:
                result["digest"] = "b" * 64
            return result

        fixture.bridge._provider_proof = changing
    with pytest.raises(NativeUiUnavailable):
        fixture.adapter.observe()
    assert fixture.bridge._provider_binding is old_binding
    assert fixture.bridge.handles is old_handles
    assert fixture.bridge.generations == old_generations
    assert first["revision"]
    assert not fixture.button.events


@pytest.mark.parametrize("verb", ["invoke", "set_value", "focus", "key_chord", "type_text"])
def test_namespace_final_proof_change_rejects_before_any_native_input(monkeypatch, verb):
    fixture, _proof, _calls, _geometry = namespace_fixture(monkeypatch)
    fixture.title.state.values.add("FOCUSED")
    name = "Preview" if verb == "invoke" else "Fixture" if verb == "key_chord" else "Title"
    args = {"value": "new"} if verb == "set_value" else {"text": "new"} if verb == "type_text" else {"keys": ["Escape"]} if verb == "key_chord" else {}
    action = fixture.action(name, verb, args)
    original = fixture.bridge._refresh_provider_binding
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1
        if refreshes > 1:
            raise NativeUiUnavailable("native_provider_identity_changed")
        return original()

    fixture.bridge._refresh_provider_binding = refresh
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *args, **kwargs: pytest.fail("no input on rejected proof"))
    result = fixture.adapter.perform(action)
    assert result == {"status": "rejected", "reason": "native_provider_changed_before_dispatch"}
    assert not any(node.events for node in (fixture.title, fixture.button, fixture.window))


def test_namespace_input_budget_error_propagates_without_dispatch(monkeypatch):
    fixture, _proof, _calls, _geometry = namespace_fixture(monkeypatch)
    action = fixture.action()
    original = fixture.bridge._refresh_provider_binding
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1
        if refreshes > 1:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        return original()

    fixture.bridge._refresh_provider_binding = refresh
    with pytest.raises(NativeUiUnavailable, match="budget_exceeded"):
        fixture.adapter.perform(action)
    assert not fixture.button.events


@pytest.mark.parametrize("property_text", ["_NET_WM_PID(CARDINAL) = 99", "_NET_WM_PID: malformed", ""])
def test_xres_fallback_requires_exact_missing_property_not_malformed_or_present(monkeypatch, property_text):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=property_text) if command[0] == "xprop" else SimpleNamespace(
            returncode=0 if command[1] == "getactivewindow" else 1, stdout="17")

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    monkeypatch.setattr("portal.native_ui._identity_helper", lambda: pytest.fail("property is not absent"))
    with pytest.raises(NativeUiUnavailable, match="foreground_unmapped"):
        _x11_foreground({"DISPLAY": ":9"})
    assert len(calls) == 3


def test_xres_missing_property_fallback_keeps_foreground_two_key_shape(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "xprop":
            assert kwargs["env"]["LC_ALL"] == "C"
            return SimpleNamespace(returncode=0, stdout="_NET_WM_PID:  not found.\n")
        return SimpleNamespace(returncode=0 if command[1] == "getactivewindow" else 1, stdout="17")

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    monkeypatch.setattr("portal.native_ui._identity_helper", lambda: SimpleNamespace(
        xres_owner=lambda xid, environment: {"pid": 42, "start": "1"}))
    assert _x11_foreground({"DISPLAY": ":9"}) == {"window_id": "17", "process_id": 42}
    assert calls[-1] == ["xdotool", "getactivewindow"]


@pytest.mark.parametrize(("namespace", "changed", "expected_reads", "expected_status"), [
    (False, False, 2, "dispatched"), (True, False, 3, "dispatched"), (True, True, 3, "rejected")])
def test_extra_visual_provider_refresh_is_namespace_only(monkeypatch, namespace, changed, expected_reads, expected_status):
    from test_native_manipulation import _visual_fixture

    from qwen_omni_adapters.computer_use import RapidUiController

    fixture, adapter, _frame, _requests = _visual_fixture(monkeypatch)
    observations = 0

    def request(payload):
        nonlocal observations
        state = fixture.request(payload)
        if payload["op"] == "observe":
            observations += 1
            if namespace:
                state["_namespace_provider_digest"] = "a" * 64
                if changed and observations == 3:
                    state["revision"] = "b" * 64
        return state

    adapter._runner = request
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    action = next(item for item in controller.enumerate(adapter._step, state) if item["kind"] == "action")
    attempts = []
    monkeypatch.setattr("portal.native_ui._pointer_dispatch", lambda *args, **kwargs: attempts.append(args) or {"status": "dispatched"})
    result = adapter.perform(action)
    assert result["status"] == expected_status
    assert observations == expected_reads
    assert len(attempts) == (expected_status == "dispatched")
    assert "_namespace_provider_digest" not in state


@pytest.mark.parametrize("failure", [None, "peer_error", "budget"])
def test_namespace_credentials_use_authenticated_bus_and_close_only_private_connection(monkeypatch, failure):
    import sys

    fixture = Fixture()
    fixture.adapter.observe()
    closed, methods = [], []

    class Session:
        def call_sync(self, *args):
            assert args[:4] == ("org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus", "GetAddress")
            return SimpleNamespace(unpack=lambda: ("unix:path=private-a11y",))

        def close_sync(self, *_args):
            pytest.fail("shared session bus must remain open")

    class Bus:
        def call_sync(self, *args):
            methods.append(args[3])
            assert args[:3] == ("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus")
            assert args[4] == ("(s)", (":1.42",)) and args[7] == 500
            if args[3] == "GetConnectionUnixUser" and failure == "peer_error":
                raise RuntimeError("provider vanished")
            return SimpleNamespace(unpack=lambda: (99 if args[3] == "GetConnectionUnixProcessID" else os.getuid(),))

        def close_sync(self, _cancellable):
            closed.append(True)

    def connect(address, *_args):
        assert address == "unix:path=private-a11y"
        if failure == "budget":
            fixture.bridge.MAX_READS = fixture.bridge.reads
        return Bus()

    gio = SimpleNamespace(Cancellable=lambda: SimpleNamespace(cancel=lambda: None),
        BusType=SimpleNamespace(SESSION="session"), bus_get_sync=lambda *_args: Session(),
        DBusCallFlags=SimpleNamespace(NONE=0),
        DBusConnectionFlags=SimpleNamespace(AUTHENTICATION_CLIENT=1, MESSAGE_BUS_CONNECTION=2),
        DBusConnection=SimpleNamespace(new_for_address_sync=connect))
    monkeypatch.setitem(sys.modules, "gi.repository", SimpleNamespace(Gio=gio))
    fixture.glib.VariantType = SimpleNamespace(new=lambda value: value)
    fixture.glib.Variant = lambda *args: args
    if failure:
        with pytest.raises(NativeUiUnavailable, match="budget_exceeded" if failure == "budget" else "observation_unavailable"):
            fixture.bridge._credentials(":1.42")
    else:
        assert fixture.bridge._credentials(":1.42") == {"pid": 99, "uid": os.getuid()}
        assert methods == ["GetConnectionUnixProcessID", "GetConnectionUnixUser"]
    assert closed == [True]


def test_namespace_late_preinput_budget_is_not_mislabeled_as_dispatch(monkeypatch):
    fixture, _proof, _calls, _geometry = namespace_fixture(monkeypatch)
    action = fixture.action("Fixture", "key_chord", {"keys": ["Escape"]})
    original = fixture.bridge._refresh_provider_binding
    count = 0

    def refresh():
        nonlocal count
        count += 1
        if count == 3:
            raise NativeUiUnavailable("native_provider_identity_budget_exceeded")
        return original()

    fixture.bridge._refresh_provider_binding = refresh
    monkeypatch.setattr("portal.native_ui.subprocess.run", lambda *args, **kwargs: pytest.fail("must not input"))
    with pytest.raises(NativeUiUnavailable, match="identity_budget_exceeded"):
        fixture.adapter.perform(action)
    assert not fixture.window.events


def test_namespace_uncertain_native_attempt_is_not_replayed(monkeypatch):
    from qwen_omni_adapters.computer_use import RapidUiController

    fixture, _proof, _calls, _geometry = namespace_fixture(monkeypatch)
    fixture.button.fail_action = True
    control = RapidUiController(fixture.adapter, authorize=lambda *_args: True, max_observations=1, verification_timeout_s=.1)
    step = {"id": "preview", "verb": "invoke", "target": {"role": "button", "name": "Preview"},
            "args": {}, "expect": [{"kind": "text", "text": "Not actually verified"}]}
    offered = control.enumerate(step, control.observe())
    action = next(item for item in offered if item["kind"] == "action")
    first = control.execute(action["id"])
    second = control.execute(action["id"])
    assert first["status"] == "unresolved" and second["status"] == "unresolved"
    assert first["receipt"] == second["receipt"]
    assert fixture.button.events == ["invoke"]


def test_namespace_observation_never_offers_inactive_window_even_with_positive_legacy_mapping(monkeypatch):
    fixture, _proof, _calls, _geometry = namespace_fixture(monkeypatch)
    inactive = Node("/namespace-inactive", "frame", "Other Flatpak window")
    inactive.process_id = 99
    fixture.application.children.append(inactive)
    fixture.bridge._mapped_window = lambda *_args: pytest.fail("foreground proof must not reach inactive mapping")
    state = fixture.adapter.planning_observe()
    assert all(node["name"] != "Other Flatpak window" for node in state["nodes"])
    assert all("switch_window" not in node["operations"] for node in state["nodes"])
    assert not any(record[0] is inactive for record in fixture.bridge.handles.values())


def scoped_namespace_fixture(monkeypatch):
    from portal.native_ui import normalize_native_task_scope

    fixture, proof, calls, geometry = namespace_fixture(monkeypatch)
    scope = {"schema": "robit.omni.native-task-scope.v1",
             "windows": [{"window_id": "17", "process_id": 42, "process_start": "420"}]}
    fixture.bridge.window_scope = normalize_native_task_scope(scope)
    owners = []

    def run(command, **kwargs):
        if command[0] == "xdotool":
            assert command[1] == "getwindowpid"
            return SimpleNamespace(returncode=1, stdout="")
        assert command[0] == "xprop"
        if command[-1] == "_NET_WM_PID":
            return SimpleNamespace(returncode=0, stdout="_NET_WM_PID:  not found.\n")
        assert command[-1] == "WM_TRANSIENT_FOR"
        return SimpleNamespace(returncode=0, stdout="WM_TRANSIENT_FOR:  not found.\n")

    def owner(xid, environment):
        owners.append(xid)
        return {"pid": 42, "start": "420"}

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    monkeypatch.setattr("portal.native_ui._identity_helper", lambda: SimpleNamespace(xres_owner=owner))
    monkeypatch.setattr("portal.native_ui._process_start", lambda _pid: "420")
    return fixture, proof, calls, geometry, owners


def test_scoped_namespace_identity_requires_both_full_proof_and_exact_window_authority(monkeypatch):
    fixture, _proof, calls, _geometry, owners = scoped_namespace_fixture(monkeypatch)
    action = fixture.action("Preview", "invoke")
    result = fixture.adapter.perform(action)
    assert result == {"status": "dispatched", "reason": "native_pattern_invoked"}
    assert fixture.button.events == ["invoke"]
    assert calls and owners and set(owners) == {"17"}


@pytest.mark.parametrize("fault", ["unlisted_xid", "wrong_pid", "reused_pid", "missing_namespace_proof", "namespace_changed_before_input"])
def test_flatpak_identity_proof_cannot_expand_or_bypass_session_window_scope(monkeypatch, fault):
    fixture, _proof, _calls, _geometry, _owners = scoped_namespace_fixture(monkeypatch)
    action = fixture.action("Preview", "invoke")
    if fault == "unlisted_xid":
        fixture.foreground["window_id"] = "18"  # Same PID, different unowned XID, no positive transient.
    elif fault == "wrong_pid":
        fixture.bridge.window_scope["windows"][0]["process_id"] = 43
    elif fault == "reused_pid":
        monkeypatch.setattr("portal.native_ui._process_start", lambda _pid: "421")
    elif fault == "missing_namespace_proof":
        fixture.bridge._provider_proof = lambda *_args: None
    else:
        original, refreshes = fixture.bridge._refresh_provider_binding, 0

        def refresh():
            nonlocal refreshes
            refreshes += 1
            if refreshes > 1:
                fixture.foreground["window_id"] = "18"
            return original()

        fixture.bridge._refresh_provider_binding = refresh
    if fault == "namespace_changed_before_input":
        assert fixture.adapter.perform(action)["status"] == "rejected"
    else:
        with pytest.raises(NativeUiUnavailable):
            fixture.adapter.perform(action)
    assert fixture.button.events == []
