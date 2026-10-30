"""Actual harness classes with explicit CPU process/input/transport doubles."""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest
import test_long_horizon_native as old

from runtime import long_horizon_desktop as desktop
from runtime import native_run_ledger as ledger
from runtime import verify_long_horizon_native as runner


def empty_inventory():
    return {"available": True, "coverage": "managed_client_windows_only", "windows": {}}


def real_ledger(tmp_path, monkeypatch):
    identities = {
        str(os.getpid()): {"pid": str(os.getpid()), "start": "100", "uid": os.getuid()},
        "12000": {"pid": "12000", "start": "200", "uid": os.getuid()},
        "12": {"pid": "12", "start": "34", "uid": os.getuid()},
    }
    original = ledger.NativeRunLedger.open
    roots = []

    def opening(environment, *, state_root=None):
        assert environment == {"DISPLAY": ":8765"}
        value = original(
            environment,
            state_root=tmp_path / "ledger-state",
            identity_reader=lambda pid: copy.deepcopy(identities.get(str(pid))),
        )
        roots.append(value)
        return value

    monkeypatch.setattr(runner, "NativeRunLedger", SimpleNamespace(open=opening))
    return roots, identities, opening


def fake_launch(monkeypatch, tmp_path, lifecycle):
    events = []

    class Process:
        pid = 12000

        def poll(self):
            return None

    def popen(*_args, **_kwargs):
        assert lifecycle.current["launches"][-1]["registered"] is False
        assert lifecycle.current["events"][-1]["phase"] == "launch_intent"
        saved = json.loads((tmp_path / "run/native-run-ownership.json").read_bytes())
        assert saved["body"] == lifecycle.current
        events.append("spawn")
        return Process()

    def command(argv, _env):
        if argv[1] == "getwindowpid":
            return "12000"
        if argv[1] == "windowactivate":
            saved = json.loads((tmp_path / "run/native-run-ownership.json").read_bytes())
            assert saved["body"]["events"][-1]["phase"] == "registered"
            assert saved["body"]["windows"][0]["owner"]["uid"] == os.getuid()
            events.append("activate")
            return ""
        pytest.fail("Unexpected GUI command in CPU fake")

    monkeypatch.setattr(desktop.shutil, "which", lambda _: "/installed/editor")
    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    monkeypatch.setattr(desktop, "process_start", lambda _: "200")
    monkeypatch.setattr(
        desktop, "class_windows", lambda _class, _env, **kw: {"111"} if kw.get("visible") else set()
    )
    monkeypatch.setattr(desktop, "command", command)
    # Preserve actual register method and its fresh xid/pid/start checks; only
    # the host process lookup is an explicit CPU stand-in.
    import runtime.acceptance_native_support as support

    monkeypatch.setattr(support, "_assert_host_process", lambda _pid, _start: None)
    monkeypatch.setattr(support, "process_start", lambda _: "200")
    monkeypatch.setattr(support, "command", command)
    return events


def test_actual_desktop_publishes_durable_intent_and_registration_before_activation(
    monkeypatch, tmp_path
):
    roots, _, opening = real_ledger(tmp_path, monkeypatch)
    value = opening({"DISPLAY": ":8765"})
    value.reconcile(empty_inventory())
    root = tmp_path / "run"
    root.mkdir(mode=0o700)
    value.begin(root, "LHN-047")
    events = fake_launch(monkeypatch, tmp_path, value)
    owned = desktop.TaskDesktop({"DISPLAY": ":8765"}, lifecycle=value)
    window = owned.open_application("editor", [], "editor", "Editor")
    assert window.window_id == "111" and events == ["spawn", "activate"]
    assert roots[0].current["windows"][0]["window_id"] == "111"
    value.close()


@pytest.mark.parametrize("failure_phase", ["launch_intent", "spawned", "registered"])
def test_checkpoint_failure_prevents_next_launch_or_activation(
    monkeypatch, tmp_path, failure_phase
):
    _, _, opening = real_ledger(tmp_path, monkeypatch)
    value = opening({"DISPLAY": ":8765"})
    value.reconcile(empty_inventory())
    root = tmp_path / "run"
    root.mkdir(mode=0o700)
    value.begin(root, "LHN-047")
    events = fake_launch(monkeypatch, tmp_path, value)
    real = value.publish_launches

    def publishing(launches, windows, *, phase):
        real(launches, windows, phase=phase)
        if phase == failure_phase:
            raise ledger.RunLedgerError("injected_checkpoint_failure")

    value.publish_launches = publishing
    owned = desktop.TaskDesktop({"DISPLAY": ":8765"}, lifecycle=value)
    with pytest.raises(ledger.RunLedgerError, match="injected_checkpoint"):
        owned.open_application("editor", [], "editor", "Editor")
    assert events == ([] if failure_phase == "launch_intent" else ["spawn"])
    assert value.index["pending"]
    value.close()


def setup_real_runner(monkeypatch, tmp_path, *, settled=True, setup_failure=False):
    case = old.cases.load_case("LHN-059")
    args, calls = old.mock_run(
        monkeypatch, tmp_path, case, settled=settled, setup_failure=setup_failure
    )
    roots, identities, opening = real_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "desktop_subprocess_environment", lambda: {"DISPLAY": ":8765"})
    monkeypatch.setattr(runner, "managed_inventory", lambda _env: empty_inventory())
    previous_factory = runner.TaskDesktop
    owned = previous_factory({})
    original_open = owned.open_blank

    def make_desktop(environment, *, editor, lifecycle):
        assert roots[-1].admitted and lifecycle.current is not None
        owned.environment = environment

        def open_blank(*args, **kwargs):
            launch = {"application": args[0], "registered": False, "before_xids": []}
            lifecycle.publish_launches(
                [*owned.launches, launch], owned.windows, phase="launch_intent"
            )
            result = original_open(*args, **kwargs)
            owned.launches[-1].update(
                application=args[0],
                before_xids=[],
                new_visible_xids=[result.window_id],
                pid=result.pid,
                process_start=result.process_start,
            )
            lifecycle.publish_launches(owned.launches, owned.windows, phase="registered")
            return result

        owned.open_blank = open_blank
        return owned

    monkeypatch.setattr(runner, "TaskDesktop", make_desktop)
    return args, calls, roots, identities, opening


@pytest.mark.parametrize("condition", ["normal", "setup_error", "cancel_drain_unresolved"])
def test_real_default_ledger_runner_normal_error_cancel_paths(monkeypatch, tmp_path, condition):
    args, calls, roots, _, _ = setup_real_runner(
        monkeypatch,
        tmp_path,
        settled=condition != "cancel_drain_unresolved",
        setup_failure=condition == "setup_error",
    )
    runner.run(args)
    report = json.loads((args.output / "summary.json").read_bytes())
    assert report["native_run_admission"]["status"] == "admitted"
    assert report["native_run_registration"]["run_id"]
    outcome = report["native_run_ledger_outcome"]
    assert outcome["state"] == ("pending" if condition == "cancel_drain_unresolved" else "resolved")
    assert roots[0].lock_fd is None
    if condition == "cancel_drain_unresolved":
        assert "close" not in calls and report["owned_windows_retained"] is True
    else:
        assert "close" in calls and report["owned_windows_closed"] is True
    packet = json.loads((args.output / "native-run-ownership.json").read_bytes())
    assert packet["body"]["state"] == outcome["state"]


def test_mandatory_default_admission_blocks_before_any_fake_setup(monkeypatch, tmp_path):
    args, calls, roots, _, opening = setup_real_runner(monkeypatch, tmp_path)
    prior = opening({"DISPLAY": ":8765"})
    prior.reconcile(empty_inventory())
    root = tmp_path / "prior"
    root.mkdir(mode=0o700)
    prior.begin(root, "LHN-047")
    prior.publish_launches([{"registered": False}], [], phase="launch_intent")
    prior.close()
    runner.run(args)
    report = json.loads((args.output / "summary.json").read_bytes())
    assert report["error"]["code"] == "native_run_prior_runner_active"
    assert not any(value.startswith("open:") or value == "model_attempt" for value in calls)
    assert not (args.output / "native-run-ownership.json").exists()
    assert all(value.lock_fd is None for value in roots)


def test_legacy_direct_desktop_remains_compatible_without_optional_publisher():
    assert desktop.TaskDesktop({})._lifecycle is None
