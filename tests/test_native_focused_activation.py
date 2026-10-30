"""CPU contract controls for provider-inactive, host-owned native dialogs.

All provider, process and X11 reads here are explicit doubles. No GI, GUI,
network, model or actual input command is invoked by these controls.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture, Node

from portal import native_ui as native

GEOMETRY = {"x": 10, "y": 20, "width": 100, "height": 80}


def attach(node, parent=None):
    node.get_parent = lambda: parent
    node.get_extents = lambda space: SimpleNamespace(**(
        GEOMETRY if space == "screen" else {**GEOMETRY, "x": 0, "y": 0}))
    for child in node.children:
        attach(child, node)


@pytest.fixture
def dialog(monkeypatch):
    fixture = Fixture()
    fixture.api.CoordType = SimpleNamespace(SCREEN="screen", WINDOW="window")
    fixture.window.role = "dialog"
    fixture.window.interfaces = ["Component"]
    fixture.window.state.values.discard("ACTIVE")
    fixture.title.state.values.add("FOCUSED")
    attach(fixture.desktop)
    fixture.bridge._credentials = lambda _bus: {"pid": 42, "uid": os.getuid()}
    monkeypatch.setattr(native, "_host_process_identity", lambda _pid: {
        "pid": 42, "start": "11", "uid": os.getuid()})
    monkeypatch.setattr(native, "_process_start", lambda _pid: "11")
    monkeypatch.setattr(native, "_x11_geometry", lambda *_args, **_kwargs: dict(GEOMETRY))
    monkeypatch.setattr(native, "_x11_frame_extents", lambda *_args, **_kwargs: {
        "left": 0, "right": 0, "top": 0, "bottom": 0})
    monkeypatch.setattr(native, "_x11_window_owner", lambda xid, *_args, **_kwargs: {
        "window_id": str(xid), "process_id": 42})

    def run(argv, **_kwargs):
        if argv[:3] == ["xdotool", "search", "--onlyvisible"]:
            return SimpleNamespace(returncode=0, stdout="17\n")
        if argv == ["xdotool", "getwindowname", "17"]:
            return SimpleNamespace(returncode=0, stdout="Fixture\n")
        if argv == ["xdotool", "getwindowpid", "17"]:
            return SimpleNamespace(returncode=0, stdout="42\n")
        raise AssertionError(f"unexpected command in CPU provider fixture: {argv}")

    monkeypatch.setattr(native.subprocess, "run", run)
    return fixture


def unavailable(fixture):
    with pytest.raises(native.NativeUiUnavailable) as error:
        fixture.bridge.observe()
    return error.value.code


def test_original_active_path_never_uses_fallback_or_credentials(monkeypatch):
    fixture = Fixture()
    fixture.bridge._activation_proof = lambda: pytest.fail("ACTIVE path used fallback")
    fixture.bridge._credentials = lambda *_: pytest.fail("ACTIVE path queried credentials")
    before = fixture.adapter.observe()
    assert before["nodes"] and fixture.bridge._alternate_activation is None
    assert "_namespace_provider_digest" not in before


def test_multiple_active_roots_do_not_use_fallback(monkeypatch):
    fixture = Fixture()
    fixture.application.children.append(Node("/second", "frame", "Other", states=["ACTIVE", "VISIBLE", "SHOWING", "SENSITIVE"]))
    fixture.bridge._activation_proof = lambda: pytest.fail("multiple ACTIVE roots must not fallback")
    assert unavailable(fixture) == "native_active_window_unavailable"


def test_owned_inactive_dialog_has_private_stable_witness_and_native_nodes(dialog):
    first = dialog.adapter.observe()
    witness = copy.deepcopy(dialog.bridge._alternate_activation)
    second = dialog.adapter.observe()
    assert first["revision"] == second["revision"]
    assert witness == dialog.bridge._alternate_activation
    assert witness["role"] == "dialog" and witness["host"]["identity"] == dialog.foreground
    assert not dialog.window.state.contains("ACTIVE")
    assert [row["focused"] for row in witness["focused_ancestry"]] == [False, True]
    assert next(n for n in first["nodes"] if n["name"] == "Title")["focused"] is True
    secret = next(n for n in first["nodes"] if n["sensitive"] is True)
    assert secret["sensitive"] is True and secret["operations"] == [] and "value" not in secret
    public = json.dumps(first)
    assert ":1.42" not in public and "/window" not in public and "focused_ancestry" not in public
    assert dialog.bridge.reads <= dialog.bridge.MAX_READS


@pytest.mark.parametrize("mutation", ["root", "ancestor"])
def test_focused_ancestor_chain_is_not_competing_focus(dialog, mutation):
    if mutation == "root":
        dialog.window.state.values.add("FOCUSED")
    else:
        box = Node("/box", "panel", children=[dialog.title], states=["VISIBLE", "SHOWING", "SENSITIVE", "FOCUSED"])
        dialog.window.children[0] = box
        attach(dialog.desktop)
    assert dialog.adapter.observe()["nodes"]


@pytest.mark.parametrize("change", ["missing", "sibling", "hidden", "defunct", "parent", "cycle", "alias", "foreign_bus"])
def test_invalid_or_incomplete_focused_ancestry_never_publishes_handles(dialog, change):
    if change == "missing":
        dialog.title.state.values.discard("FOCUSED")
    elif change == "sibling":
        dialog.button.state.values.add("FOCUSED")
    elif change == "hidden":
        dialog.window.state.values.discard("SHOWING")
    elif change == "defunct":
        dialog.title.state.values.add("DEFUNCT")
    elif change == "parent":
        dialog.title.get_parent = lambda: dialog.application
    elif change == "cycle":
        dialog.title.children.append(dialog.window)
    elif change == "alias":
        dialog.check.path = dialog.title.path
    else:
        dialog.title.app = SimpleNamespace(bus_name=":1.43")
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles and not dialog.bridge.generations
    assert dialog.bridge._alternate_activation is None


@pytest.mark.parametrize("limit", ["children", "depth", "nodes"])
def test_focus_coverage_must_be_complete_with_existing_bounds(dialog, limit):
    if limit == "children":
        dialog.title.children = [Node(f"/many/{i}", "label") for i in range(129)]
    elif limit == "depth":
        child = dialog.title
        for index in range(66):
            branch = Node(f"/deep/{index}", "panel", children=[child])
            child = branch
        dialog.window.children[0] = child
    else:
        dialog.title.children = [Node(f"/group/{i}", "panel", children=[
            Node(f"/leaf/{i}/{j}", "label") for j in range(110)]) for i in range(4)]
    attach(dialog.desktop)
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles


def test_cached_focus_is_invalidated_before_proof(dialog):
    clears = []
    def clear():
        clears.append(1)
        dialog.title.state.values.discard("FOCUSED")
    dialog.title.clear_cache_single = clear
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert clears and not dialog.bridge.handles


@pytest.mark.parametrize("change", ["pid", "uid", "no_credentials", "two_apps", "two_roots", "mapping", "geometry", "parent_pid"])
def test_same_title_pid_or_geometry_alone_never_proves_dialog(dialog, monkeypatch, change):
    if change == "pid":
        dialog.bridge._credentials = lambda _bus: {"pid": 43, "uid": os.getuid()}
    elif change == "uid":
        dialog.bridge._credentials = lambda _bus: {"pid": 42, "uid": os.getuid()+1}
    elif change == "no_credentials":
        dialog.bridge._credentials = lambda _bus: None
    elif change == "two_apps":
        dialog.desktop.children.append(Node("/app2", "application"))
    elif change == "two_roots":
        other = Node("/other", "dialog", "Fixture", interfaces=["Component"])
        dialog.application.children.append(other)
        attach(dialog.desktop)
    elif change == "mapping":
        dialog.bridge._mapped_window = lambda *_args, **_kwargs: "18"
    elif change == "geometry":
        dialog.window.get_extents = lambda _space: SimpleNamespace(x=30, y=40, width=70, height=60)
    else:
        monkeypatch.setattr(native, "_host_process_identity", lambda _pid: {
            "pid": 43, "uid": os.getuid(), "start": "11"})
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles


def test_other_root_focused_state_blocks_fallback(dialog):
    dialog.application.children.append(Node("/other", "frame", "Other", states=["FOCUSED", "VISIBLE", "SHOWING", "SENSITIVE"]))
    attach(dialog.desktop)
    assert unavailable(dialog) == "native_active_window_unavailable"


def test_host_lifetime_change_during_proof_rejects_without_publication(dialog, monkeypatch):
    calls = []
    def host(_pid):
        calls.append(1)
        return {"pid": 42, "uid": os.getuid(), "start": "11" if len(calls) == 1 else "12"}
    monkeypatch.setattr(native, "_host_process_identity", host)
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles


def test_late_witness_loss_rolls_back_previous_handles_generations_and_scratch(dialog):
    dialog.adapter.observe()
    old = (dict(dialog.bridge.handles), dict(dialog.bridge.generations), copy.deepcopy(dialog.bridge._alternate_activation))
    proof = dialog.bridge._activation_proof
    count = []
    def unstable():
        result = proof()
        count.append(1)
        if len(count) >= 2:
            result["focused_ancestry"][-1]["role"] = "different"
        return result
    dialog.bridge._activation_proof = unstable
    with pytest.raises(native.NativeUiUnavailable):
        dialog.bridge.observe()
    assert dialog.bridge.handles == old[0] and dialog.bridge.generations == old[1]
    assert dialog.bridge._alternate_activation == old[2]


def test_interface_lookup_cannot_reuse_lost_activation(dialog):
    action = dialog.action("Title", "set_value", {"value": "saved.txt"})
    def interface():
        dialog.title.state.values.discard("FOCUSED")
        return dialog.title
    dialog.title.get_editable_text_iface = interface
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected"
    assert not dialog.title.events and dialog.title.value == ""


def test_all_refreshes_share_original_deadline_and_read_budget(dialog, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    proof = dialog.bridge._activation_proof
    deadlines = []
    def measured():
        deadlines.append((dialog.bridge.deadline, dialog.bridge.reads))
        result = proof()
        now[0] += 1
        return result
    dialog.bridge._activation_proof = measured
    dialog.adapter.observe()
    assert len(deadlines) >= 2 and {deadline for deadline, _ in deadlines} == {15.0}
    assert [reads for _, reads in deadlines] == sorted(reads for _, reads in deadlines)
    assert deadlines[-1][1] > deadlines[0][1] > 0
    dialog.bridge.MAX_READS = dialog.bridge.reads - 1
    with pytest.raises(native.NativeUiUnavailable, match="budget"):
        dialog.bridge._require_activation(dialog.window)


def test_proof_returning_after_deadline_cannot_publish(dialog, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    real = native._x11_geometry
    def late(*args, **kwargs):
        value = real(*args, **kwargs)
        now[0] = 6.0
        return value
    monkeypatch.setattr(native, "_x11_geometry", late)
    assert unavailable(dialog) == "native_observation_budget_exceeded"
    assert not dialog.bridge.handles and not dialog.bridge.generations


def pointer_world(monkeypatch, *, start=(30, 40)):
    identity = {"window_id": "17", "process_id": 42}
    pointer, calls = list(start), []
    monkeypatch.setattr(native, "_x11_geometry", lambda *_args, **_kwargs: dict(GEOMETRY))
    def run(argv, **kwargs):
        calls.append((argv[1:], kwargs["timeout"]))
        command = argv[1:]
        if command == ["getmouselocation", "--shell"]:
            return SimpleNamespace(returncode=0, stdout=f"X={pointer[0]}\nY={pointer[1]}\nWINDOW=17\n")
        if command[0] == "mousemove":
            pointer[:] = [int(command[-2])+GEOMETRY["x"], int(command[-1])+GEOMETRY["y"]]
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(native.subprocess, "run", run)
    return identity, calls, run


@pytest.mark.parametrize("verb,args", [("click", {}), ("double_click", {}), ("right_click", {}),
    ("scroll", {"direction": "down"}), ("drag_drop", {"destination": (50, 50)})])
def test_final_activation_guard_blocks_every_first_button_after_arrival(monkeypatch, verb, args):
    identity, calls, _ = pointer_world(monkeypatch)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), verb,
        foreground=lambda: identity, activation=lambda: False, **args)
    assert result["status"] == "rejected"
    assert all(argv[0] not in {"click", "mousedown", "mouseup", "mousemove"} for argv, _ in calls)


def test_movement_then_focus_loss_reports_input_without_any_button(monkeypatch):
    identity, calls, _ = pointer_world(monkeypatch, start=(11, 21))
    checks = []
    def activation():
        checks.append(1)
        return len(checks) == 1
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click",
        foreground=lambda: identity, activation=activation)
    assert result["status"] == "dispatched" and result["input_attempted"] is True
    assert "unresolved" in result["reason"]
    assert sum(argv[0] == "mousemove" for argv, _ in calls) == 1
    assert all(argv[0] not in {"click", "mousedown", "mouseup"} for argv, _ in calls)


@pytest.mark.parametrize("failure", ["timeout", "focus", "deadline"])
def test_down_attempt_always_gets_one_bounded_release_without_replaying_gesture(monkeypatch, failure):
    identity, calls, run = pointer_world(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    down = [False]
    def input_run(argv, **kwargs):
        if argv[1] == "mousedown":
            down[0] = True
            if failure == "deadline":
                now[0] = 9.0
            if failure == "timeout":
                calls.append((argv[1:], kwargs["timeout"]))
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return run(argv, **kwargs)
    monkeypatch.setattr(native.subprocess, "run", input_run)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "drag_drop", destination=(50, 50),
        foreground=lambda: identity, activation=lambda: not down[0], deadline=5.0)
    assert result["status"] == "dispatched" and "unresolved" in result["reason"]
    assert sum(argv == ["mousedown", "1"] for argv, _ in calls) == 1
    releases = [(argv, timeout) for argv, timeout in calls if argv[0] == "mouseup"]
    assert releases == [(["mouseup", "1"], 2)]
    assert not any(argv[0] == "mousemove" for argv, _ in calls)


def test_legacy_pointer_dispatch_keeps_existing_no_callback_command_path(monkeypatch):
    identity, calls, _ = pointer_world(monkeypatch)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity)
    assert result == {"status": "dispatched", "reason": "native_pointer_invoked", "backend": "x11_guarded_xtest"}
    assert calls[-1] == (["click", "--repeat", "1", "--delay", "70", "1"], 2)


def scope(window_id="17", *, start="11"):
    return {"schema": "robit.omni.native-task-scope.v1", "windows": [
        {"window_id": window_id, "process_id": 42, "process_start": start}]}


def test_registered_exact_scope_is_bound_in_private_witness(dialog):
    dialog.bridge.window_scope = scope()
    dialog.adapter.observe()
    witness = dialog.bridge._alternate_activation
    assert witness["host"]["scope"] == {"kind": "registered", "digest": native.native_task_scope_sha256(scope()),
                                        "parent_window_id": None}


@pytest.mark.parametrize("changed", [False, True])
def test_exact_direct_transient_parent_is_revalidated_during_same_pid_scope(dialog, monkeypatch, changed):
    dialog.bridge.window_scope = scope("16")
    original = native.subprocess.run
    parents = ["0x10"]
    def run(argv, **kwargs):
        if argv == ["xprop", "-id", "17", "WM_TRANSIENT_FOR"]:
            return SimpleNamespace(returncode=0, stdout=f"WM_TRANSIENT_FOR(WINDOW): window id # {parents[0]}\n")
        return original(argv, **kwargs)
    monkeypatch.setattr(native.subprocess, "run", run)
    dialog.adapter.observe()
    assert dialog.bridge._alternate_activation["host"]["scope"]["kind"] == "direct_transient"
    if changed:
        parents[0] = "0x12"
        assert not dialog.bridge._require_activation(dialog.window)
    else:
        assert dialog.bridge._require_activation(dialog.window)


def test_scope_lifetime_mismatch_rejects_even_when_window_pid_is_identical(dialog):
    dialog.bridge.window_scope = scope(start="12")
    assert unavailable(dialog) == "native_task_window_out_of_scope"
    assert not dialog.bridge.handles


def test_provider_credential_change_invalidates_existing_witness(dialog):
    dialog.adapter.observe()
    dialog.bridge._credentials = lambda _bus: {"pid": 42, "uid": os.getuid()+1}
    assert not dialog.bridge._require_activation(dialog.window)


def test_active_root_membership_does_not_widen_to_inactive_same_pid_controls(dialog):
    other = Node("/other", "frame", "Other", children=[Node("/other-secret", "entry", "Other content", value="not-current")])
    dialog.application.children.append(other)
    attach(dialog.desktop)
    dialog.bridge._mapped_window = lambda node, *_args, **_kwargs: "17" if node is dialog.window else "18"
    snapshot = dialog.adapter.observe()
    assert not any(n["name"] == "Other content" for n in snapshot["nodes"])
    assert all(n["scope_id"] == snapshot["scope"]["id"] for n in snapshot["nodes"])
    assert dialog.bridge._alternate_activation["root"].endswith(":/window")


def test_already_current_switch_checks_witness_at_its_own_boundary(dialog):
    before = dialog.adapter.observe()
    window = next(n for n in before["nodes"] if n["id"] == before["scope"]["id"])
    action = {"session_id": "one", "kind": "action", "id": "switch", "node_id": window["id"],
        "generation": window["generation"], "scope_id": window["scope_id"], "revision": before["revision"],
        "verb": "switch_window", "args": {}, "expect": [{"kind": "active_window", "node_id": window["id"]}],
        "idempotency_key": "owned-switch"}
    # This native bridge include-windows path is not needed to validate this
    # already-current no-input branch; retain its real fresh observe result.
    original = dialog.bridge.observe
    dialog.bridge.observe = lambda **_kwargs: original()
    result = dialog.bridge.perform(action)
    assert result["status"] == "already_satisfied" and result["input_attempted"] is False
    assert not dialog.window.events


@pytest.mark.parametrize("verb,args", [("key_chord", {"keys": ["Return"]}), ("type_text", {"text": "name"}),
                                      ("select_text", {"start": 0, "end": 2})])
def test_physical_input_cannot_reuse_focus_lost_during_interface_lookup(dialog, verb, args):
    dialog.title.value = "abc"
    action = dialog.action("Title", verb, args)
    original = dialog.title.get_role_name
    observed, armed = dialog.bridge.observe, [False]
    def observe(*args, **kwargs):
        result = observed(*args, **kwargs)
        armed[0] = True
        return result
    dialog.bridge.observe = observe
    def role():
        # Arm from the completed snapshot, not a particular read count. Extra
        # fresh proof reads must not move this negative into snapshot setup.
        if armed[0]:
            dialog.title.state.values.discard("FOCUSED")
        return original()
    dialog.title.get_role_name = role
    dialog.api.Text.get_n_selections = lambda _node: 0
    dialog.api.Text.add_selection = lambda *_args: pytest.fail("selection must not be sent")
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected", result
    assert not dialog.title.events


def test_partial_selection_provider_count_cannot_move_focus_then_receive_input(dialog):
    dialog.title.value = "abc"
    dialog.api.Text.get_n_selections = lambda _node: 0
    action = dialog.action("Title", "select_text", {"start": 0, "end": 2})
    def count(_node):
        dialog.title.state.values.discard("FOCUSED")
        return 0
    dialog.api.Text.get_n_selections = count
    dialog.api.Text.add_selection = lambda *_args: pytest.fail("selection count is not dispatch authority")
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected"
    assert not dialog.title.events


def test_selection_clear_partial_effect_never_replays_or_selects_after_focus_loss(dialog):
    action = dialog.action("Items", "select", {"option": "B"})
    original = dialog.selection.clear_selection
    def clear():
        value = original()
        dialog.title.state.values.discard("FOCUSED")
        return value
    dialog.selection.clear_selection = clear
    result = dialog.bridge.perform(action)
    assert result["status"] == "dispatched" and result["reason"] == "native_pattern_effect_unresolved"
    assert dialog.selection.events == ["clear_selection"] and not dialog.selection.selected


def test_document_url_read_cannot_retain_source_credit_after_focus_loss(dialog):
    document = Node("/document", "document web", "Document", interfaces=["Document"])
    dialog.window.children.append(document)
    attach(dialog.desktop)
    document.get_document_iface = lambda: document
    dialog.api.Document = SimpleNamespace(get_document_attribute_value=lambda *_args: "https://example.com/")
    first = dialog.adapter.observe()
    assert first["document_evidence"]["status"] == "observed"
    count = []
    def url(*_args):
        count.append(1)
        dialog.title.state.values.discard("FOCUSED")
        return "https://example.com/"
    dialog.api.Document.get_document_attribute_value = url
    # Directly exercise the existing evidence collector against actual native
    # handles; observation publication itself independently rejects lost focus.
    result = dialog.bridge._document_evidence(dialog.window, first["scope"]["id"],
        dialog.bridge.handles, dialog.bridge.generations)
    assert result["status"] == "unavailable" and "url" not in result
    assert count


def test_mapped_window_stops_at_shared_deadline_instead_of_finishing_32_candidates(monkeypatch):
    fixture = Fixture()
    now, commands = [0.0], []
    fixture.bridge.reads, fixture.bridge.deadline = 0, .05
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    def run(argv, **kwargs):
        commands.append((argv, kwargs["timeout"]))
        now[0] += .02
        if "search" in argv:
            return SimpleNamespace(returncode=0, stdout="\n".join(str(i) for i in range(1, 33)))
        return SimpleNamespace(returncode=0, stdout="Fixture" if "getwindowname" in argv else "42")
    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeUiUnavailable, match="budget"):
        fixture.bridge._mapped_window(fixture.window, 42, deadline=.05)
    assert len(commands) == 3 and all(0 < timeout <= .05 for _, timeout in commands)
    assert fixture.bridge.reads == 4  # Three host reads and one exact title read.


@pytest.mark.parametrize("helper,stdout", [(native._x11_geometry,
    "xwininfo: Window id: 0x11 Fixture\n Absolute upper-left X: 10\n Absolute upper-left Y: 20\n Width: 100\n Height: 80\n"),
    (native._x11_frame_extents, "_GTK_FRAME_EXTENTS(CARDINAL) = 0, 0, 0, 0\n")])
def test_geometry_helpers_reject_late_successful_host_read(monkeypatch, helper, stdout):
    now, timeouts = [0.0], []
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    def run(_argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        now[0] = 2.0
        return SimpleNamespace(returncode=0, stdout=stdout)
    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeUiUnavailable, match="budget"):
        helper({"window_id": "17"}, deadline=.25)
    assert timeouts == [.25]


def test_expired_pointer_query_does_not_admit_a_button(monkeypatch):
    identity, calls, run = pointer_world(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    def late(argv, **kwargs):
        value = run(argv, **kwargs)
        now[0] = 5.0
        return value
    monkeypatch.setattr(native.subprocess, "run", late)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        activation=lambda: True, deadline=1.0)
    assert result["status"] == "rejected"
    assert not any(argv[0] in {"mousedown", "click", "mousemove"} for argv, _ in calls)


def test_frame_role_does_not_hide_competing_focused_descendant(dialog):
    other = Node("/other", "frame", "Fixture", interfaces=["Component"], children=[
        Node("/other-focus", "entry", "Other focus", states=["VISIBLE", "SHOWING", "SENSITIVE", "FOCUSED"])])
    dialog.application.children.append(other)
    attach(dialog.desktop)
    # Root FOCUSED=false, but the same actual mapped XID cannot be excluded
    # merely because this root has frame rather than dialog role.
    assert "FOCUSED" not in other.state.values
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles


def test_distinct_owned_xid_excludes_unrelated_frame_without_content_or_focus_reads(dialog):
    other = Node("/other", "frame", "Other", children=[Node("/other-entry", "entry", "Not current")])
    dialog.application.children.append(other)
    attach(dialog.desktop)
    dialog.bridge._mapped_window = lambda node, *_args, **_kwargs: "17" if node is dialog.window else "18"
    other.children[0].get_state_set = lambda: pytest.fail("excluded window content must not be scanned")
    result = dialog.adapter.observe()
    assert result["nodes"] and not any(n["name"] == "Not current" for n in result["nodes"])
    rows = dialog.bridge._alternate_activation["roots"]
    assert next(row for row in rows if row["identity"].endswith(":/other"))["window_id"] == "18"


def test_unknown_other_root_association_is_not_claimed_distinct_from_current(dialog):
    other = Node("/other", "frame", "No provable mapping", children=[Node("/other-focus", "entry", "Maybe current")])
    dialog.application.children.append(other)
    attach(dialog.desktop)
    dialog.bridge._mapped_window = lambda node, *_args, **_kwargs: "17" if node is dialog.window else None
    assert unavailable(dialog) == "native_active_window_unavailable"
    assert not dialog.bridge.handles


@pytest.mark.parametrize("lost_during_tail", [False, True])
def test_final_pointer_proof_rechecks_early_focus_after_later_sibling_reads(dialog, monkeypatch, lost_during_tail):
    dialog.adapter.observe()
    commands, tail_reads = [], []
    original = native.subprocess.run
    def run(argv, **kwargs):
        if argv[1:] == ["getmouselocation", "--shell"]:
            return SimpleNamespace(returncode=0, stdout="X=30\nY=40\nWINDOW=17\n")
        if argv[1] == "click":
            commands.append((argv, dialog.title.state.contains("FOCUSED")))
            return SimpleNamespace(returncode=0, stdout="")
        return original(argv, **kwargs)
    monkeypatch.setattr(native.subprocess, "run", run)
    def sibling_count():
        tail_reads.append(dialog.title.state.contains("FOCUSED"))
        if lost_during_tail:
            dialog.title.state.values.discard("FOCUSED")
        return 0
    dialog.button.get_child_count = sibling_count
    result = native._pointer_dispatch(dialog.foreground, GEOMETRY, (30, 40), "click",
        foreground=dialog.bridge.foreground, activation=lambda: dialog.bridge._require_activation(dialog.window),
        deadline=dialog.bridge.deadline)
    assert tail_reads
    if lost_during_tail:
        assert result["status"] == "rejected" and commands == []
        assert not dialog.title.state.contains("FOCUSED")
    else:
        assert result["status"] == "dispatched" and result["reason"] == "native_pointer_invoked"
        assert len(commands) == 1 and commands[0][1] is True


def test_large_owned_modal_reaches_setter_without_raising_original_read_budget(dialog):
    #184 current-root nodes, realistic accessibility breadth with Document API
    # available but no observed DocumentWeb. This exposed the repeated scans;
    # the invariant is exact input/readback under unchanged caps, not a scan count.
    dialog.window.children.extend(Node(f"/group/{i}", "panel", children=[
        Node(f"/leaf/{i}/{j}", "label", f"Visible item {i}:{j}") for j in range(21)]) for i in range(8))
    attach(dialog.desktop)
    dialog.api.Document = SimpleNamespace(get_document_attribute_value=lambda *_: pytest.fail("no document URI is admitted"))
    snapshot = dialog.adapter.observe()
    assert len(snapshot["nodes"]) == 184 and snapshot["truncated"] is False
    action = dialog.action("Title", "set_value", {"value": "saved.txt"})
    result = dialog.bridge.perform(action)
    assert result["status"] == "dispatched" and dialog.title.events == [("set_value", "saved.txt")]
    assert dialog.title.value == "saved.txt" and dialog.bridge.reads <= 12000
    assert dialog.bridge.MAX_READS == 12000 and dialog.bridge.MAX_NODES == 400


@pytest.mark.parametrize("verb,name,getter", [("invoke", "Preview", "get_action_iface"),
    ("focus", "Title", "get_component_iface"), ("select", "Items", "get_selection_iface")])
def test_coalesced_generic_final_proof_follows_native_interface_reads(dialog, verb, name, getter):
    args = {"option": "B"} if verb == "select" else {}
    action = dialog.action(name, verb, args)
    target = {"Preview": dialog.button, "Title": dialog.title, "Items": dialog.selection}[name]
    original = getattr(target, getter)
    observed, armed = dialog.bridge.observe, [False]
    def observe(*args, **kwargs):
        result = observed(*args, **kwargs)
        armed[0] = True
        return result
    dialog.bridge.observe = observe
    def interface():
        result = original()
        if armed[0]:
            dialog.title.state.values.discard("FOCUSED")
        return result
    setattr(target, getter, interface)
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected" and not target.events


def test_already_checked_noop_cannot_bypass_fresh_last_proof(dialog):
    dialog.check.state.values.add("CHECKED")
    action = dialog.action("Ready", "set_checked", {"checked": True})
    observed, armed = dialog.bridge.observe, [False]
    def observe(*args, **kwargs):
        result = observed(*args, **kwargs)
        armed[0] = True
        return result
    dialog.bridge.observe = observe
    dialog.check.state_hook = lambda: dialog.title.state.values.discard("FOCUSED") if armed[0] else None
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected" and not dialog.check.events


def test_document_interface_focus_loss_blocks_uri_reads_with_same_host_identity(dialog):
    document = Node("/document", "document web", "Document", interfaces=["Document"])
    dialog.window.children.append(document)
    attach(dialog.desktop)
    uri_calls = []
    dialog.api.Document = SimpleNamespace(get_document_attribute_value=lambda *_: uri_calls.append(1) or "https://example.com/")
    document.get_document_iface = lambda: document
    initial = dialog.adapter.observe()
    assert initial["document_evidence"]["status"] == "observed"
    uri_calls.clear()
    def interface():
        dialog.title.state.values.discard("FOCUSED")
        return document
    document.get_document_iface = interface
    evidence = dialog.bridge._document_evidence(dialog.window, initial["scope"]["id"],
        dialog.bridge.handles, dialog.bridge.generations)
    assert evidence["status"] == "unavailable" and "url" not in evidence
    assert uri_calls == []


def test_no_observed_current_document_is_unknown_without_uri_or_global_absence_claim(dialog):
    dialog.api.Document = SimpleNamespace(get_document_attribute_value=lambda *_: pytest.fail("no URI source exists"))
    snapshot = dialog.adapter.observe()
    evidence = snapshot["document_evidence"]
    assert evidence == {"status": "unavailable", "source": "atspi.Document.DocURL", "reason": "document_binding_unavailable"}
    assert not any(key in evidence for key in ("url", "absent", "task_progress", "resource_acquired", "input_authority"))


def test_inactive_switch_has_full_last_proof_after_operation_local_popup_query(dialog):
    other = Node("/other", "frame", "Other", interfaces=["Component"])
    dialog.application.children.append(other)
    attach(dialog.desktop)
    dialog.bridge._inactive_namespace_providers = lambda: []
    dialog.bridge._mapped_window = lambda node, *_args, **_kwargs: "17" if node is dialog.window else "18"
    snapshot = dialog.bridge.observe(include_windows=True)
    node = next(n for n in snapshot["nodes"] if n["name"] == "Other")
    action = {"session_id": "one", "kind": "action", "id": "switch-other", "node_id": node["id"],
        "generation": node["generation"], "scope_id": snapshot["scope"]["id"], "revision": snapshot["revision"],
        "verb": "switch_window", "args": {}, "expect": [{"kind": "active_window", "node_id": node["id"]}],
        "idempotency_key": "owned-switch-other"}
    original = dialog.bridge._open_popups
    def popups(*args):
        result = original(*args)
        # The observation also calls this helper before publication. Only the
        # actual operation-local call follows generic perform preparation.
        import inspect
        if inspect.currentframe().f_back.f_code.co_name == "operation":
            dialog.title.state.values.discard("FOCUSED")
        return result
    dialog.bridge._open_popups = popups
    result = dialog.bridge.perform(action)
    assert result["status"] == "rejected" and not other.events


@pytest.mark.parametrize("fault,phase,cause", [
    ("move_nonzero", "move_command", "command_nonzero"),
    ("move_timeout", "move_command", "command_timeout"),
    ("move_exception", "move_command", "command_error"),
    ("move_late", "move_post_command", "native_budget_exceeded"),
    ("arrival_window", "pointer_after_move", "pointer_window_mismatch"),
    ("arrival_coordinates", "pointer_after_move", "pointer_coordinates_mismatch"),
    ("arrival_query_nonzero", "pointer_after_move", "pointer_query_nonzero"),
    ("foreground_after_move", "foreground_after_move", "foreground_mismatch"),
    ("geometry_after_move", "geometry_after_move", "geometry_mismatch"),
    ("guard_after_move", "button_activation", "activation_rejected"),
    ("guard_error", "button_activation", "activation_error"),
    ("read_limit", "button_activation", "native_budget_exceeded"),
    ("deadline_guard", "button_activation", "native_budget_exceeded"),
    ("button_exception", "button_command", "command_error"),
    ("button_timeout", "button_command", "command_timeout"),
    ("button_late", "button_post_command", "native_budget_exceeded"),
    ("drag_guard_after_down", "drag_activation", "activation_rejected"),
    ("drag_guard_after_down_late_cleanup", "drag_activation", "activation_rejected"),
    ("drag_down_timeout_cleanup_failure", "button_command", "command_timeout"),
])
def test_pointer_diagnostic_distinguishes_causal_boundary_without_claiming_effect(monkeypatch, fault, phase, cause):
    moving = fault.startswith(("move_", "arrival_", "foreground_", "geometry_")) or fault == "guard_after_move"
    identity, commands, original_run = pointer_world(monkeypatch, start=(11, 21) if moving else (30, 40))
    now, moved, down, guard_calls, metadata_events = [0.0], [False], [False], [], []
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    def run(argv, **kwargs):
        kind = argv[1]
        if kind == "mousemove":
            moved[0] = True
            if fault in {"move_timeout", "move_exception"}:
                commands.append((argv[1:], kwargs["timeout"]))
                if fault == "move_timeout":
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                raise OSError("private-error-must-not-leak")
            value = original_run(argv, **kwargs)
            if fault == "move_late":
                now[0] = 6
            return SimpleNamespace(returncode=1, stdout="") if fault == "move_nonzero" else value
        if kind == "getmouselocation" and moved[0]:
            value = original_run(argv, **kwargs)
            if fault == "arrival_window":
                return SimpleNamespace(returncode=0, stdout="X=30\nY=40\nWINDOW=18\n")
            if fault == "arrival_coordinates":
                return SimpleNamespace(returncode=0, stdout="X=31\nY=40\nWINDOW=17\n")
            if fault == "arrival_query_nonzero":
                return SimpleNamespace(returncode=1, stdout="")
            return value
        if kind == "mousedown":
            down[0] = True
            if fault == "drag_down_timeout_cleanup_failure":
                commands.append((argv[1:], kwargs["timeout"]))
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if kind == "mouseup":
            value = original_run(argv, **kwargs)
            if fault == "drag_guard_after_down_late_cleanup":
                now[0] = 6
            return SimpleNamespace(returncode=1, stdout="") if fault == "drag_down_timeout_cleanup_failure" else value
        if kind == "click":
            if fault in {"button_exception", "button_timeout"}:
                commands.append((argv[1:], kwargs["timeout"]))
                if fault == "button_timeout":
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                raise OSError("private-error-must-not-leak")
            value = original_run(argv, **kwargs)
            if fault == "button_late":
                now[0] = 6
            return value
        return original_run(argv, **kwargs)
    monkeypatch.setattr(native.subprocess, "run", run)
    monkeypatch.setattr(native, "_x11_geometry", lambda *_args, **_kwargs: {
        **GEOMETRY, "width": 101 if fault == "geometry_after_move" and moved[0] else 100})
    def foreground():
        return {**identity, "process_id": 43} if fault == "foreground_after_move" and moved[0] else identity
    def activation():
        guard_calls.append(1)
        if fault == "guard_after_move" and moved[0]:
            return False
        if fault == "guard_error":
            raise RuntimeError("private-error-must-not-leak")
        if fault in {"read_limit", "deadline_guard"}:
            if fault == "deadline_guard":
                now[0] = 6
            raise native.NativeUiUnavailable("native_observation_budget_exceeded")
        return not (fault.startswith("drag_guard_") and down[0])
    def context():
        metadata_events.append(len(commands))
        return {"read_count": 12001 if fault == "read_limit" else 7000, "read_limit": 12000}
    drag = fault.startswith("drag_")
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "drag_drop" if drag else "click",
        destination=(50, 50) if drag else None, foreground=foreground, activation=activation,
        deadline=5, diagnostic_context=context)
    d = result["pointer_diagnostic"]
    assert (d["phase"], d["cause"]) == (phase, cause)
    assert d["schema"] == "robit.omni.native-pointer-diagnostic.v1"
    assert d["sample_point"] == "after_work_and_cleanup"
    assert metadata_events == [len(commands)]  # Not an input admission callback.
    assert d["input_authority"] is d["effect_verified"] is d["task_progress"] is False
    assert "private-error" not in json.dumps(result)
    assert d["read_limit_exceeded"] is (fault == "read_limit")
    late = fault in {"move_late", "deadline_guard", "button_late", "drag_guard_after_down_late_cleanup"}
    assert d["deadline_exceeded"] is late
    if drag:
        assert d["down_attempted"] and d["release_attempted"]
        assert sum(argv[0] == "mouseup" for argv, _ in commands) == 1
        failed_cleanup = fault == "drag_down_timeout_cleanup_failure"
        assert d["cleanup_failed"] is failed_cleanup
        assert d["release_acknowledged"] is (not failed_cleanup)
    if fault in {"button_exception", "button_timeout", "button_late"}:
        assert d["button_attempted"] and result["input_attempted"] is True
    assert result["reason"] != "native_pointer_invoked"


@pytest.mark.parametrize("bad", [None, {"read_count": True, "read_limit": 12000},
    {"read_count": 3, "read_limit": "private-token"}, {"read_count": 3, "read_limit": 12000, "private-token": "secret"}])
def test_invalid_pointer_diagnostic_context_is_unknown_and_does_not_change_rejection(monkeypatch, bad):
    identity, commands, _ = pointer_world(monkeypatch)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        activation=lambda: False, diagnostic_context=lambda: bad)
    d = result.pop("pointer_diagnostic")
    assert result == {"status": "rejected", "reason": "native_pointer_binding_changed"}
    assert not d["context_available"] and d["read_count"] is d["read_limit"] is d["read_limit_exceeded"] is None
    assert "private-token" not in json.dumps(d)
    assert not any(argv[0] == "click" for argv, _ in commands)


def test_diagnostic_callback_exception_cannot_replace_original_result(monkeypatch):
    identity, commands, _ = pointer_world(monkeypatch)
    def broken():
        raise RuntimeError("secret-input-text-never-return")
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        activation=lambda: False, diagnostic_context=broken)
    assert result["status"] == "rejected" and result["reason"] == "native_pointer_binding_changed"
    assert result["pointer_diagnostic"]["context_available"] is False
    assert "secret-input" not in json.dumps(result)
    assert not any(argv[0] == "click" for argv, _ in commands)


def test_budget_diagnostic_preserves_both_read_and_time_facts_without_exclusive_inference(monkeypatch):
    identity, _commands, _ = pointer_world(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: now[0])
    def activation():
        now[0] = 6
        raise native.NativeUiUnavailable("native_observation_budget_exceeded")
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        activation=activation, deadline=5, diagnostic_context=lambda: {"read_count": 12001, "read_limit": 12000})
    d = result["pointer_diagnostic"]
    assert d["cause"] == "native_budget_exceeded" and d["read_limit_exceeded"] and d["deadline_exceeded"]
    assert d["remaining_ms"] == -1000 and not d["movement_attempted"] and not d["button_attempted"]


@pytest.mark.parametrize("alternate", [False, True])
def test_success_does_not_invoke_diagnostic_callback_or_change_receipt(monkeypatch, alternate):
    identity, commands, _ = pointer_world(monkeypatch, start=(11, 21))
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        activation=(lambda: True) if alternate else None,
        diagnostic_context=lambda: pytest.fail("successful input must not collect diagnostics"))
    assert result == {"status": "dispatched", "reason": "native_pointer_invoked", "backend": "x11_guarded_xtest"}
    assert [argv[0] for argv, _ in commands if argv[0] != "getmouselocation"] == ["mousemove", "click"]


def test_raw_active_failure_shape_and_callback_behavior_are_unchanged(monkeypatch):
    identity, commands, original = pointer_world(monkeypatch)
    def run(argv, **kwargs):
        value = original(argv, **kwargs)
        return SimpleNamespace(returncode=1, stdout="") if argv[1] == "click" else value
    monkeypatch.setattr(native.subprocess, "run", run)
    result = native._pointer_dispatch(identity, GEOMETRY, (30, 40), "click", foreground=lambda: identity,
        diagnostic_context=lambda: pytest.fail("raw ACTIVE path must not collect diagnostics"))
    assert result == {"status": "dispatched", "reason": "native_pointer_effect_unresolved", "backend": "x11_guarded_xtest"}
    assert sum(argv[0] == "click" for argv, _ in commands) == 1


def dense_pointer_dialog(dialog, monkeypatch):
    """Controlled188-node sensitivity, not the actual provider interface list.

    All nodes advertise Component and a non-invoking Action;36 also expose
    Text. The26-app registry models the independently measured current count,
    not the unarchived registry during072. Real _hit_evidence logic is retained.
    """
    groups = []
    leaves = []
    for group, count in enumerate((60, 59, 59)):
        children = []
        for index in range(count):
            node = Node(f"/dense/{group}/{index}", "label", interfaces=["Component", "Action"])
            if len(leaves) < 35:
                node.interfaces.append("Text")
            children.append(node)
            leaves.append(node)
        groups.append(Node(f"/dense/group/{group}", "panel", children=children,
                           interfaces=["Component", "Action"]))
    dialog.button.name = "Cancel"
    dialog.button.interfaces = ["Component", "Action"]
    branch = Node("/dense/4", "panel", children=[dialog.title, dialog.button, *groups],
                  interfaces=["Component", "Action"])
    for index in range(3, 0, -1):
        branch = Node(f"/dense/{index}", "panel", children=[branch], interfaces=["Component", "Action"])
    dialog.window.children = [branch]
    dialog.window.interfaces = ["Component", "Action"]
    for index in range(25):
        app = Node(f"/foreign/{index}", "application")
        app.process_id = 43 + index
        app.app = SimpleNamespace(bus_name=f":1.{43+index}")
        dialog.desktop.children.append(app)
    attach(dialog.desktop)
    all_nodes = []
    def collect(node):
        all_nodes.append(node)
        for child in node.children:
            collect(child)
    collect(dialog.window)
    assert len(all_nodes) == 188
    for node in all_nodes:
        node.get_description = lambda: ""
        if node is not dialog.button:
            node.get_action_name = lambda _index: "non-invoking-provider-action"
    def credentials(_bus):
        for _ in range(5):
            dialog.bridge._read(lambda: None)
        return {"pid": 42, "uid": os.getuid()}
    dialog.bridge._credentials = credentials
    dialog.api.Document = SimpleNamespace(get_document_attribute_value=lambda *_args: None)
    # The deployed Text interface exposes selection readback; retained072 text
    # observations include selection_count, which the smaller old fixture lacked.
    dialog.api.Text.get_n_selections = lambda _node: 0
    dialog.api.Text.get_selection = lambda *_args: pytest.fail("no selected range in density fixture")
    dialog.api.Text.add_selection = lambda *_args: pytest.fail("no selection input permitted")
    dialog.api.Text.set_selection = lambda *_args: pytest.fail("no selection input permitted")
    monkeypatch.setattr(dialog.api.StateType, "HAS_POPUP", "HAS_POPUP", raising=False)
    dialog.api.RelationType = SimpleNamespace(CONTROLLER_FOR="CONTROLLER_FOR")
    # Native hit-testing returns successive real fixture ancestors, rather than
    # granting a blanket successful _hit result or using coordinates as proof.
    path = [dialog.button]
    while path[-1] is not dialog.window:
        path.append(path[-1].get_parent())
    path.reverse()
    for index, node in enumerate(path):
        child = path[index+1] if index+1 < len(path) else None
        node.get_accessible_at_point = lambda *_args, child=child: child
    pointer, commands = [11, 21], []
    original = native.subprocess.run
    def run(argv, **kwargs):
        if argv[1] == "getmouselocation":
            return SimpleNamespace(returncode=0, stdout=f"X={pointer[0]}\nY={pointer[1]}\nWINDOW=17\n")
        if argv[1] == "mousemove":
            commands.append("move")
            pointer[:] = [int(argv[-2])+GEOMETRY["x"], int(argv[-1])+GEOMETRY["y"]]
            return SimpleNamespace(returncode=0, stdout="")
        if argv[1] == "click":
            commands.append("click")
            return SimpleNamespace(returncode=0, stdout="")
        return original(argv, **kwargs)
    monkeypatch.setattr(native.subprocess, "run", run)
    monkeypatch.setattr(native.time, "monotonic", lambda: 0.0)
    return commands


def test_dense_moving_pointer_keeps_complete_per_input_proofs_under_original_limit(dialog, monkeypatch):
    commands = dense_pointer_dialog(dialog, monkeypatch)
    snapshot = dialog.adapter.observe()
    target = next(row for row in snapshot["nodes"] if row["name"] == "Cancel")
    proofs = []
    original = dialog.bridge._activation_proof
    def proof():
        before = dialog.bridge.reads
        result = original()
        proofs.append((before, dialog.bridge.reads))
        return result
    dialog.bridge._activation_proof = proof
    action = {"session_id": "one", "revision": snapshot["revision"], "scope_id": target["scope_id"],
              "node_id": target["id"], "generation": target["generation"], "verb": "click", "args": {}}
    result = dialog.bridge.perform(action)
    assert result == {"status": "dispatched", "reason": "native_pointer_invoked", "backend": "x11_guarded_xtest"}
    assert commands == ["move", "click"] and len(proofs) == 4
    assert all(after > before for before, after in proofs)
    assert all(proofs[index][1] <= proofs[index+1][0] for index in range(3))
    assert dialog.bridge.MAX_READS == 12000 and dialog.bridge.reads <= 12000
    assert dialog.bridge.deadline == 5 and len(snapshot["nodes"]) == 188


def test_identity_checks_provider_in_the_fresh_tuple_not_a_second_property_read(dialog):
    class ChangingBus:
        calls = 0
        @property
        def bus_name(self):
            self.calls += 1
            return ":1.foreign" if self.calls % 2 else ":1.42"
    changing = ChangingBus()
    dialog.title.app = changing
    dialog.bridge.reads = 0
    dialog.bridge.deadline = native.time.monotonic() + 5
    dialog.bridge.foreground_identity = dict(dialog.foreground)
    with pytest.raises(native.NativeUiUnavailable, match="native_activation_focus_identity_unproven"):
        dialog.bridge._activation_proof()
    assert changing.calls == 1
    assert not dialog.title.events and not dialog.bridge.handles


@pytest.mark.parametrize("foreign_bus", [":1.43", ":1.42-shadow"])
def test_final_focused_tail_rechecks_coherent_provider_identity_after_sibling_reads(dialog, foreign_bus):
    count = dialog.selection.get_child_count
    def late_count():
        dialog.title.app = SimpleNamespace(bus_name=foreign_bus)
        return count()
    dialog.selection.get_child_count = late_count
    dialog.bridge.reads = 0
    dialog.bridge.deadline = native.time.monotonic() + 5
    dialog.bridge.foreground_identity = dict(dialog.foreground)
    with pytest.raises(native.NativeUiUnavailable, match="native_activation_focused_ancestry_changed"):
        dialog.bridge._activation_proof()
    assert not dialog.title.events and not dialog.bridge.handles
