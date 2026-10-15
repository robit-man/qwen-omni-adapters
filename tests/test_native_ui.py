from __future__ import annotations

import io
import json
import os
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
            self.bridge.observe(include_windows=request.get("include_windows", False))
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
    assert fixture.adapter.perform(action)["status"] == "dispatched"
    assert commands == [["xdotool", "key", "--clearmodifiers", "ctrl+s"]]
    fixture.foreground["window_id"] = "18"
    assert fixture.adapter.perform(action)["status"] == "rejected"
    assert len(commands) == 1


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
