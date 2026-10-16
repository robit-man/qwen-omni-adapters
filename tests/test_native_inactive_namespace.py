from __future__ import annotations

import copy
import subprocess
import threading
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture, Node

from portal.native_ui import NativeUiUnavailable, normalize_native_task_scope


def inactive_fixture(monkeypatch, *, scoped=False):
    fixture = Fixture()
    app = Node("/flatpak-app", "application")
    frame = Node("/flatpak-frame", "frame", "Namespace browser", interfaces=["Component"])
    app.app.bus_name = frame.app.bus_name = ":1.99"
    app.process_id = frame.process_id = 99
    app.children = [frame]
    fixture.desktop.children.append(app)
    fixture.api.CoordType = SimpleNamespace(SCREEN="screen", WINDOW="window")
    geometry = {
        "17": {"x": 10, "y": 20, "width": 800, "height": 900},
        "21": {"x": 70, "y": 64, "width": 925, "height": 1016},
    }
    frame.get_extents = lambda _space: SimpleNamespace(**geometry["21"])
    windows, owners = (
        ["17", "21"],
        {"17": {"pid": 42, "start": "420"}, "21": {"pid": 77, "start": "770"}},
    )
    proofs, calls, activations = {}, [], []

    def proof(identity, bus, reported, **kwargs):
        calls.append(("proof", copy.deepcopy(identity), bus, reported))
        if identity["process_id"] != 77 or bus != ":1.99" or reported != 99:
            return None
        return copy.deepcopy(
            proofs.setdefault(
                identity["window_id"],
                {
                    "source": "flatpak_namespace_metadata",
                    "identity": dict(identity),
                    "reported_pid": 99,
                    "bus_name": ":1.99",
                    "digest": "a" * 64,
                    "values": {
                        "owner": {"pid": 77, "start": "770"},
                        "metadata": {"host": {"instance": "133"}},
                    },
                },
            )
        )

    def owner(xid, _environment, **kwargs):
        calls.append(("owner", xid))
        return copy.deepcopy(owners[xid])

    class IdentityError(RuntimeError):
        pass

    def host_process_identity(pid):
        if type(pid) is not int or pid not in {42, 77}:
            raise IdentityError("native_host_process_unavailable")
        return {"pid": pid, "start": {42: "420", 77: "770"}[pid]}

    helper = SimpleNamespace(
        IdentityUnavailable=IdentityError,
        host_process_identity=host_process_identity,
        relay_candidate=lambda pid, **kwargs: (
            {"application": "org.mozilla.firefox", "instance": "133"} if pid == 99 else None
        ),
        client_windows=lambda environment, **kwargs: list(windows),
        xres_owner=owner,
        provider_proof=proof,
    )
    monkeypatch.setattr("portal.native_ui._identity_helper", lambda: helper)
    monkeypatch.setattr(
        "portal.native_ui._x11_geometry",
        lambda identity, *args: copy.deepcopy(geometry[identity["window_id"]]),
    )
    monkeypatch.setattr(
        "portal.native_ui._process_start", lambda pid: "420" if pid == 42 else "770"
    )

    def run(command, **kwargs):
        if command[0] == "xprop":
            assert command[-1] == "WM_TRANSIENT_FOR"
            return SimpleNamespace(returncode=0, stdout="WM_TRANSIENT_FOR:  not found.")
        if command[1] == "getwindowpid":
            return SimpleNamespace(returncode=0, stdout=str(owners[command[2]]["pid"]))
        if command[1] == "search":
            return SimpleNamespace(
                returncode=1, stdout=""
            )  # Legacy provider-PID/title mapping cannot help.
        assert command == ["xdotool", "windowactivate", "--sync", "21"]
        activations.append(command)
        fixture.foreground.update(window_id="21", process_id=77)
        fixture.window.state.values.discard("ACTIVE")
        frame.state.values.add("ACTIVE")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    if scoped:
        fixture.bridge.window_scope = normalize_native_task_scope(
            {
                "schema": "robit.omni.native-task-scope.v1",
                "windows": [
                    {"window_id": "17", "process_id": 42, "process_start": "420"},
                    {"window_id": "21", "process_id": 77, "process_start": "770"},
                ],
            }
        )
    return SimpleNamespace(
        fixture=fixture,
        app=app,
        frame=frame,
        helper=helper,
        run=run,
        geometry=geometry,
        windows=windows,
        owners=owners,
        proofs=proofs,
        calls=calls,
        activations=activations,
    )


def test_positively_bound_inactive_namespace_window_is_offered_without_optional_scope(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    state = setup.fixture.adapter.planning_observe()
    matched = [node for node in state["nodes"] if node["name"] == "Namespace browser"]
    assert len(matched) == 1
    assert matched[0]["operations"] == ["switch_window"]
    assert setup.activations == []


def switch_step():
    return {
        "id": "switch-browser",
        "verb": "switch_window",
        "target": {"role": "window", "name": "Namespace browser"},
        "args": {},
        "expect": [{"kind": "active_window", "node_id": "$target"}],
    }


def offered_action(setup):
    from qwen_omni_adapters.computer_use import RapidUiController

    control = RapidUiController(
        setup.fixture.adapter, authorize=lambda *_args: True, max_observations=1
    )
    step = switch_step()
    setup.fixture.adapter.prepare_step(step)
    state = control.observe()
    action = next(item for item in control.enumerate(step, state) if item["kind"] == "action")
    return control, action


@pytest.mark.parametrize("scoped", [False, True])
def test_full_namespace_switch_dispatches_once_and_active_observer_verifies(monkeypatch, scoped):
    from qwen_omni_adapters.computer_use import RapidUiController

    setup = inactive_fixture(monkeypatch, scoped=scoped)
    control = RapidUiController(
        setup.fixture.adapter, authorize=lambda *_args: True, max_observations=1
    )
    result = control.run_step(switch_step())
    assert result["status"] == "verified"
    assert result["receipt"]["status"] == "dispatched"
    assert setup.activations == [["xdotool", "windowactivate", "--sync", "21"]]
    assert result["after"]["scope"]["window_id"] == "21"
    assert result["after"]["nodes"][0]["generation"] == "1"
    # Already-active requests retain b13's independent no-input path.
    again = control.run_step({**switch_step(), "id": "already-active"})
    assert again["status"] == "verified" and again["receipt"]["status"] == "already_satisfied"
    assert len(setup.activations) == 1


def test_optional_scope_narrows_exact_offers_and_handles_without_same_pid_enrollment(monkeypatch):
    setup = inactive_fixture(monkeypatch, scoped=True)
    setup.fixture.bridge.window_scope["windows"][1]["window_id"] = "22"
    setup.owners["22"] = {"pid": 77, "start": "770"}
    state = setup.fixture.adapter.planning_observe()
    assert not any(node["name"] == "Namespace browser" for node in state["nodes"])
    assert not any(record[0] is setup.frame for record in setup.fixture.bridge.handles.values())
    assert setup.activations == []


@pytest.mark.parametrize(
    "fault",
    [
        "same_geometry_client",
        "same_geometry_frame",
        "client_inventory",
        "inventory_changed",
        "xres_unavailable",
        "xres_changed",
        "wrong_peer",
        "wrong_owner",
        "wrong_start",
        "missing_namespace",
        "namespace_unavailable",
        "frame_bus",
        "frame_pid",
        "geometry_missing",
        "geometry_mismatch",
        "frame_hidden",
        "frame_disabled",
        "frame_defunct",
        "protected_ancestor",
        "frame_cycle",
        "frame_children_truncated",
        "provider_duplicate",
    ],
)
def test_unknown_ambiguous_or_ineligible_namespace_never_falls_back_or_offers_input(
    monkeypatch, fault
):
    setup = inactive_fixture(monkeypatch, scoped=True)
    if fault == "same_geometry_client":
        setup.windows.append("22")
        setup.owners["22"] = {"pid": 77, "start": "770"}
        setup.geometry["22"] = dict(setup.geometry["21"])
    elif fault == "same_geometry_frame":
        other = Node("/duplicate-frame", "frame", "Duplicate", interfaces=["Component"])
        other.app.bus_name = ":1.99"
        other.process_id = 99
        other.get_extents = setup.frame.get_extents
        setup.app.children.append(other)
    elif fault == "client_inventory":
        setup.helper.client_windows = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("native_client_inventory_incomplete")
        )
    elif fault == "inventory_changed":
        calls = 0

        def changed(*args, **kwargs):
            nonlocal calls
            calls += 1
            return list(setup.windows) if calls == 1 else list(reversed(setup.windows))

        setup.helper.client_windows = changed
    elif fault == "xres_unavailable":
        setup.helper.xres_owner = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("native_xres_owner_unavailable")
        )
    elif fault == "xres_changed":
        original = setup.helper.xres_owner
        calls = 0

        def changed(xid, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(xid, *args, **kwargs)
            if calls > 2 and xid == "21":
                result["start"] = "771"
            return result

        setup.helper.xres_owner = changed
    elif fault in {"wrong_peer", "wrong_owner", "wrong_start"}:
        setup.helper.provider_proof({"window_id": "21", "process_id": 77}, ":1.99", 99)
        proof = setup.proofs["21"]
        if fault == "wrong_peer":
            proof["reported_pid"] = 98
        elif fault == "wrong_owner":
            proof["values"]["owner"]["pid"] = 78
        else:
            proof["values"]["owner"]["start"] = "771"
    elif fault == "missing_namespace":
        setup.helper.provider_proof = lambda *args, **kwargs: None
    elif fault == "namespace_unavailable":
        setup.helper.provider_proof = lambda *args, **kwargs: (_ for _ in ()).throw(
            setup.helper.IdentityUnavailable("native_provider_identity_unverified")
        )
    elif fault == "frame_bus":
        setup.frame.app.bus_name = ":1.98"
    elif fault == "frame_pid":
        setup.frame.process_id = 98
    elif fault == "geometry_missing":
        setup.frame.get_extents = lambda _space: None
    elif fault == "geometry_mismatch":
        setup.frame.get_extents = lambda _space: SimpleNamespace(
            **{**setup.geometry["21"], "x": 71}
        )
    elif fault == "frame_hidden":
        setup.frame.state.values.discard("SHOWING")
    elif fault == "frame_disabled":
        setup.frame.state.values.discard("SENSITIVE")
    elif fault == "frame_defunct":
        setup.frame.state.values.add("DEFUNCT")
    elif fault == "protected_ancestor":
        protected = Node("/protected-parent", "password text", children=[setup.frame])
        protected.app.bus_name = ":1.99"
        protected.process_id = 99
        setup.app.children = [protected]
    elif fault == "frame_cycle":
        setup.app.children = [setup.app]
    elif fault == "frame_children_truncated":
        setup.app.get_child_count = lambda: 129
    else:
        other = Node("/duplicate-app", "application", children=[setup.frame])
        other.app.bus_name = ":1.99"
        other.process_id = 99
        setup.fixture.desktop.children.append(other)
    # A legacy title/PID positive is deliberately dangerous and must be unreachable.
    setup.fixture.bridge._mapped_window = lambda *_args: pytest.fail(
        "namespace identity cannot fall back to legacy title/PID"
    )
    if fault in {"frame_defunct", "frame_cycle", "frame_children_truncated"}:
        try:
            state = setup.fixture.adapter.planning_observe()
        except NativeUiUnavailable:
            state = {"nodes": []}
    else:
        state = setup.fixture.adapter.planning_observe()
    assert not any(node.get("name") == "Namespace browser" for node in state["nodes"])
    assert not any(record[0] is setup.frame for record in setup.fixture.bridge.handles.values())
    assert setup.activations == []


@pytest.mark.parametrize("fault", ["namespace", "geometry", "generation", "scope", "foreground"])
def test_offered_namespace_action_revalidates_target_and_scope_at_final_boundary(
    monkeypatch, fault
):
    setup = inactive_fixture(monkeypatch, scoped=True)
    control, action = offered_action(setup)
    original = setup.fixture.bridge._refresh_inactive_namespace_bindings
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1
        # execute observes, perform observes, then the operation refreshes.
        if refreshes == 3:
            if fault == "namespace":
                setup.proofs["21"]["digest"] = "b" * 64
            elif fault == "geometry":
                setup.geometry["21"]["x"] += 1
            elif fault == "generation":
                setup.fixture.bridge.generations[":1.99:/flatpak-frame"] = 2
            elif fault == "scope":
                setup.fixture.bridge.window_scope["windows"].pop()
            else:
                setup.fixture.foreground["window_id"] = "22"
                setup.owners["22"] = {"pid": 42, "start": "420"}
        return original()

    setup.fixture.bridge._refresh_inactive_namespace_bindings = refresh
    result = control.execute(action["id"])
    assert result["status"] in {"rejected", "unresolved"}
    assert result.get("receipt", {}).get("status") != "dispatched"
    assert setup.activations == []


def test_cancelled_or_unauthorized_namespace_action_has_no_input(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    control, action = offered_action(setup)
    cancel = threading.Event()
    cancel.set()
    assert control.execute(action["id"], cancel)["status"] == "cancelled"
    from qwen_omni_adapters.computer_use import RapidUiController

    denied = RapidUiController(setup.fixture.adapter, authorize=lambda *_args: False)
    assert denied.run_step(switch_step())["status"] == "rejected"
    assert setup.activations == []


def test_stale_revision_and_generation_reject_before_namespace_activation(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    _control, action = offered_action(setup)
    bad = copy.deepcopy(action)
    bad["revision"] = "b" * 64
    assert setup.fixture.adapter.perform(bad)["status"] == "rejected"
    bad = copy.deepcopy(action)
    bad["generation"] = "2"
    assert setup.fixture.adapter.perform(bad)["status"] == "rejected"
    assert setup.activations == []


def test_uncertain_namespace_activation_is_not_replayed(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    control, action = offered_action(setup)

    def run(command, **kwargs):
        if command[:2] == ["xdotool", "windowactivate"]:
            setup.activations.append(command)
            raise subprocess.TimeoutExpired(command, 2)
        return setup.run(command, **kwargs)

    monkeypatch.setattr("portal.native_ui.subprocess.run", run)
    first = control.execute(action["id"])
    second = control.execute(action["id"])
    assert first["status"] == "unresolved" and first["receipt"]["status"] == "dispatched"
    assert first == second and len(setup.activations) == 1


def test_failed_observation_rolls_back_private_namespace_mapping_and_generations(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    state = setup.fixture.adapter.planning_observe()
    bindings = setup.fixture.bridge._inactive_namespace_bindings
    handles = setup.fixture.bridge.handles
    generations = dict(setup.fixture.bridge.generations)
    original = setup.fixture.bridge._refresh_inactive_namespace_bindings

    def changed():
        setup.proofs["21"]["digest"] = "b" * 64
        return original()

    setup.fixture.bridge._refresh_inactive_namespace_bindings = changed
    with pytest.raises(NativeUiUnavailable, match="inactive_namespace_changed"):
        setup.fixture.adapter.planning_observe()
    assert setup.fixture.bridge._inactive_namespace_bindings is bindings
    assert setup.fixture.bridge.handles is handles
    assert setup.fixture.bridge.generations == generations
    assert state["revision"]


def test_namespace_read_budget_failure_propagates_without_input(monkeypatch):
    setup = inactive_fixture(monkeypatch)

    def exhausted(*args, **kwargs):
        raise NativeUiUnavailable("native_observation_budget_exceeded")

    setup.helper.provider_proof = exhausted
    with pytest.raises(NativeUiUnavailable, match="budget_exceeded"):
        setup.fixture.adapter.planning_observe()
    assert setup.activations == []


def test_private_namespace_proof_binds_revision_without_entering_public_state(monkeypatch):
    setup = inactive_fixture(monkeypatch)
    first = setup.fixture.adapter.planning_observe()
    setup.proofs["21"]["digest"] = "b" * 64
    second = setup.fixture.adapter.planning_observe()
    assert first["nodes"] == second["nodes"]
    assert first["revision"] != second["revision"]
    for state in (first, second):
        fields, strings, pending = set(), set(), [state]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                fields.update(value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str):
                strings.add(value)
        assert {"inactive_namespace_bindings", "metadata", "reported_pid", "bus_name"}.isdisjoint(fields)
        assert {"flatpak_namespace_metadata", "133", ":1.99"}.isdisjoint(strings)


def test_unknown_inactive_coverage_preserves_positive_current_control_authority(monkeypatch):
    from qwen_omni_adapters.computer_use import RapidUiController

    setup = inactive_fixture(monkeypatch)
    setup.helper.provider_proof = lambda *args, **kwargs: None
    control = RapidUiController(setup.fixture.adapter, authorize=lambda *_args: True)
    setup.fixture.adapter.prepare_step(switch_step())
    state = control.observe()
    assert state["truncated"] is True
    step = {
        "id": "current-preview",
        "verb": "invoke",
        "target": {"role": "button", "name": "Preview"},
        "args": {},
        "expect": [{"kind": "visible", "role": "button", "name": "Preview", "visible": True}],
    }
    assert any(item["kind"] == "action" for item in control.enumerate(step, state))
    assert not any(node["name"] == "Namespace browser" for node in state["nodes"])
    assert setup.activations == []
