"""CPU controls for exact task-owned native window authority."""
from __future__ import annotations

import copy
import io
import json
import sys
from types import SimpleNamespace

import pytest
from test_native_ui import _switch_fixture

import portal.native_ui as native
from portal.native_ui import (
    NativeUiUnavailable,
    native_task_scope_sha256,
    normalize_native_task_scope,
)

SCOPE = {"schema": "robit.omni.native-task-scope.v1", "windows": [
    {"window_id": "17", "process_id": 42, "process_start": "4201"}]}


def _identity_provider(monkeypatch, owners=None, transients=None, starts=None, activate=None):
    owners = owners if owners is not None else {"17": 42, "101": 42, "102": 42, "18": 42}
    transients = transients if transients is not None else {}
    starts = starts if starts is not None else {42: "4201"}
    monkeypatch.setattr(native, "_process_start", lambda pid: starts.get(pid, "unknown"))

    def run(argv, **_kwargs):
        if argv[:2] == ["xdotool", "windowactivate"] and activate:
            return activate(argv)
        if argv[:2] == ["xdotool", "getwindowpid"]:
            value = owners.get(argv[2])
            return SimpleNamespace(returncode=0 if value else 1, stdout=str(value or ""))
        assert argv[:2] == ["xprop", "-id"] and argv[3:] == ["WM_TRANSIENT_FOR"]
        value = transients.get(argv[2])
        return SimpleNamespace(returncode=0, stdout=(
            f"WM_TRANSIENT_FOR(WINDOW): window id # {hex(int(value))}\n" if value else "WM_TRANSIENT_FOR: not found."))

    monkeypatch.setattr(native.subprocess, "run", run)
    return owners, transients, starts


def _restrict_current_fixture(fixture, monkeypatch):
    # Fixed host configuration represents the runner's independently owned base.
    # The pre-patch implementation silently ignores this missing authority.
    fixture.bridge.window_scope = copy.deepcopy(SCOPE)
    fixture.adapter._window_scope = copy.deepcopy(SCOPE)
    activate = native.subprocess.run
    return _identity_provider(monkeypatch, activate=activate)


def test_same_pid_unowned_window_is_not_offered_under_task_scope(monkeypatch):
    fixture, _target, _other, _mapping, activations, controller, action = _switch_fixture(monkeypatch)
    _restrict_current_fixture(fixture, monkeypatch)
    state = controller.observe()
    assert action["node_id"] not in {node["id"] for node in state["nodes"]}
    assert activations == []


def test_previously_offered_outside_window_cannot_activate_under_task_scope(monkeypatch):
    fixture, _target, _other, _mapping, activations, controller, action = _switch_fixture(monkeypatch)
    _restrict_current_fixture(fixture, monkeypatch)
    outcome = controller.execute(action["id"])
    assert activations == []
    assert outcome["status"] in {"rejected", "unresolved"}


def test_scope_is_canonical_fresh_and_order_independent():
    raw = copy.deepcopy(SCOPE)
    raw["windows"].insert(0, {"window_id": "101", "process_id": 42, "process_start": "4201"})
    normalized = normalize_native_task_scope(raw)
    assert [row["window_id"] for row in normalized["windows"]] == ["17", "101"]
    assert native_task_scope_sha256(raw) == native_task_scope_sha256(normalized)
    raw["windows"][0]["process_start"] = "changed"
    assert normalized["windows"][1]["process_start"] == "4201"


@pytest.mark.parametrize("change", ["extra", "empty", "too_many", "duplicate", "pid_start", "bool_pid",
                                    "zero_xid", "large_xid", "leading_zero", "negative_pid", "bad_start"])
def test_scope_rejects_malformed_or_contradictory_identities(change):
    value = copy.deepcopy(SCOPE)
    row = value["windows"][0]
    if change == "extra":
        value["title"] = "not permission"
    elif change == "empty":
        value["windows"] = []
    elif change == "too_many":
        value["windows"] = [{**row, "window_id": str(i + 1)} for i in range(17)]
    elif change == "duplicate":
        value["windows"].append(dict(row))
    elif change == "pid_start":
        value["windows"].append({**row, "window_id": "18", "process_start": "4202"})
    else:
        key, bad = {"bool_pid": ("process_id", True), "zero_xid": ("window_id", "0"),
                    "large_xid": ("window_id", str(2**32)), "leading_zero": ("window_id", "017"),
                    "negative_pid": ("process_id", -1), "bad_start": ("process_start", "42 private")}[change]
        row[key] = bad
    with pytest.raises(ValueError, match="Invalid native task"):
        normalize_native_task_scope(value)


@pytest.mark.parametrize("change", ["base", "direct_transient", "same_pid_old", "nested", "foreign_pid",
                                    "recycled", "base_closed", "wrong_parent", "unknown"])
def test_exact_live_base_or_direct_same_process_transient_only(monkeypatch, change):
    owners, transients, starts = _identity_provider(monkeypatch)
    identity = {"window_id": "18", "process_id": 42}
    transients["18"] = "17"
    if change == "base":
        identity["window_id"] = "17"
    elif change == "same_pid_old":
        identity["window_id"] = "101"
    elif change == "nested":
        transients.update({"18": "101", "101": "17"})
    elif change == "foreign_pid":
        owners["18"] = 99
        identity["process_id"] = 99
    elif change == "recycled":
        starts[42] = "4202"
    elif change == "base_closed":
        owners.pop("17")
    elif change == "wrong_parent":
        transients["18"] = "102"
    elif change == "unknown":
        owners.pop("18")
    assert native._window_scope_allows(SCOPE, identity) is (change in {"base", "direct_transient"})


def test_unowned_foreground_fails_before_any_native_state_or_input(monkeypatch):
    fixture, _target, _other, _mapping, activations, _controller, _action = _switch_fixture(monkeypatch)
    _restrict_current_fixture(fixture, monkeypatch)
    fixture.foreground["window_id"] = "101"
    previous = set(fixture.bridge.previous)
    with pytest.raises(NativeUiUnavailable) as error:
        fixture.adapter.planning_observe()
    assert error.value.code == "native_task_window_out_of_scope"
    assert fixture.bridge.previous == previous and activations == []


def test_mapped_authorized_window_is_offered_and_digest_changes_revision(monkeypatch):
    fixture, _target, _other, _mapping, activations, controller, action = _switch_fixture(monkeypatch)
    unscoped = controller.observe()
    _restrict_current_fixture(fixture, monkeypatch)
    fixture.bridge.window_scope["windows"].append({"window_id": "101", "process_id": 42, "process_start": "4201"})
    fixture.adapter._window_scope = copy.deepcopy(fixture.bridge.window_scope)
    scoped = controller.observe()
    assert action["node_id"] in {node["id"] for node in scoped["nodes"]}
    assert scoped["window_scope_sha256"] == native_task_scope_sha256(fixture.bridge.window_scope)
    assert scoped["window_scope_count"] == 2 and scoped["revision"] != unscoped["revision"]
    assert "process_start" not in json.dumps({k: v for k, v in scoped.items() if k != "nodes"})
    assert activations == []


def test_unmapped_inactive_provider_never_creates_offer_or_handle(monkeypatch):
    fixture, target, _other, _mapping, _activations, controller, action = _switch_fixture(monkeypatch)
    _restrict_current_fixture(fixture, monkeypatch)
    fixture.bridge.window_scope["windows"].append({"window_id": "101", "process_id": 42, "process_start": "4201"})
    fixture.bridge._mapped_window = lambda node, _pid: None if node is target else "17"
    state = controller.observe()
    assert action["node_id"] not in {node["id"] for node in state["nodes"]}
    assert action["node_id"] not in fixture.bridge.handles


def test_target_permission_change_before_activation_is_rejected(monkeypatch):
    fixture, _target, _other, _mapping, activations, controller, _old = _switch_fixture(monkeypatch)
    owners, _transients, _starts = _restrict_current_fixture(fixture, monkeypatch)
    fixture.bridge.window_scope["windows"].append({"window_id": "101", "process_id": 42, "process_start": "4201"})
    state = controller.observe()
    step = {"id": "switch", "verb": "switch_window", "target": {"name": "Notes"},
            "expect": [{"kind": "active_window", "node_id": "$target"}]}
    action = next(item for item in controller.enumerate(step, state) if item["kind"] == "action")
    original = fixture.bridge._mapped_window
    count = [0]

    def changed(node, pid):
        count[0] += 1
        result = original(node, pid)
        if count[0] >= 2:
            owners["101"] = 99
        return result

    fixture.bridge._mapped_window = changed
    result = controller.execute(action["id"])
    assert activations == [] and result["status"] in {"rejected", "unresolved"}


def test_allowed_same_process_window_switch_retains_actual_verification(monkeypatch):
    fixture, _target, _other, _mapping, activations, controller, _old = _switch_fixture(monkeypatch)
    _restrict_current_fixture(fixture, monkeypatch)
    fixture.bridge.window_scope["windows"].append({"window_id": "101", "process_id": 42, "process_start": "4201"})
    state = controller.observe()
    step = {"id": "allowed-switch", "verb": "switch_window", "target": {"name": "Notes"},
            "expect": [{"kind": "active_window", "node_id": "$target"}]}
    action = next(item for item in controller.enumerate(step, state) if item["kind"] == "action")
    result = controller.execute(action["id"])
    assert activations == [["xdotool", "windowactivate", "--sync", "101"]]
    assert result["status"] == "verified" and result["receipt"]["status"] == "dispatched"


def test_scope_cannot_be_replaced_after_native_request_or_by_payload(monkeypatch):
    captured = []
    adapter = native.NativeUiAdapter("session", bridge_runner=lambda value: captured.append(value) or {})
    adapter.set_window_scope(SCOPE)
    adapter._request({"op": "observe", "window_scope": {"untrusted": True}})
    assert captured[0]["window_scope"] == SCOPE
    with pytest.raises(NativeUiUnavailable) as error:
        adapter.set_window_scope({**SCOPE, "windows": [{**SCOPE["windows"][0], "window_id": "18"}]})
    assert error.value.code == "native_task_scope_immutable"


def test_outside_foreground_never_reaches_visual_capture(monkeypatch):
    _identity_provider(monkeypatch)
    monkeypatch.setattr("portal.desktop.desktop_subprocess_environment", lambda: {"DISPLAY": ":fixture"})
    monkeypatch.setattr(native, "_x11_foreground", lambda *_: {"window_id": "101", "process_id": 42})
    monkeypatch.setattr(native, "_capture_window", lambda *_: pytest.fail("Out-of-scope image must not be captured"))
    adapter = native.NativeUiAdapter("session")
    adapter.set_window_scope(SCOPE)
    with pytest.raises(NativeUiUnavailable) as error:
        adapter._visual_capture({"scope": {"window_id": "101"}})
    assert error.value.code == "native_task_window_out_of_scope"


def test_worker_scope_is_fixed_on_first_private_protocol_request(monkeypatch):
    class Bridge:
        def __init__(self, _atspi, _glib, session_id, process_id, *, window_scope):
            self.session_id, self.process_id, self.window_scope = session_id, process_id, window_scope

        def observe(self, **_kwargs):
            return {"observed": True}

    gi = SimpleNamespace(require_version=lambda *_: None)
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", SimpleNamespace(Atspi=None, GLib=None))
    monkeypatch.setattr(native, "_AtspiBridge", Bridge)
    envelope = {"schema": "robit.omni.native-ui.v1", "session_id": "session", "process_id": None,
                "window_scope": SCOPE, "op": "observe"}
    altered = {**SCOPE, "windows": [{**SCOPE["windows"][0], "window_id": "18"}]}
    incoming = [{**envelope, "id": "one"}, {**envelope, "id": "two", "window_scope": altered},
                {**envelope, "id": "three"}]
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(("\n".join(map(json.dumps, incoming)) + "\n").encode())))
    monkeypatch.setattr(sys, "stdout", output)
    native._bridge_main()
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["ok"] for response in responses] == [True, False, True]
    assert responses[1]["error"] == "native_task_scope_mismatch"
