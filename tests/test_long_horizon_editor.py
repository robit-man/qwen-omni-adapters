"""CPU controls for explicit native editor setup and shared-process ownership."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from runtime import long_horizon_desktop as desktop
from runtime import verify_long_horizon_native as runner
from runtime.long_horizon_cases import DATA


def test_editor_cli_default_and_explicit_choice(tmp_path):
    args = ["--output", str(tmp_path / "run")]
    assert runner.parser().parse_args(args).editor == "gedit"
    assert runner.parser().parse_args([*args, "--editor", "gnome-text-editor"]).editor == "gnome-text-editor"
    with pytest.raises(SystemExit):
        runner.parser().parse_args([*args, "--editor", "arbitrary-app"])
    with pytest.raises(ValueError):
        desktop.TaskDesktop({}, editor="arbitrary-app")


def test_declared_editor_default_keeps_gedit_setup_unchanged(monkeypatch, tmp_path):
    owned = desktop.TaskDesktop({"DISPLAY": ":test"})
    marker = object()
    calls = []
    monkeypatch.setattr(owned, "open_editor", lambda **kwargs: calls.append(kwargs) or marker)
    assert owned.open_blank("gedit", tmp_path, browser="firefox", log_path=tmp_path / "log") is marker
    assert calls == [{"log_path": tmp_path / "log"}]


def test_selected_gnome_uses_installed_new_window_without_files_or_session_flags(monkeypatch, tmp_path):
    binary = tmp_path / "gnome-text-editor"
    binary.write_bytes(b"installed native binary")
    monkeypatch.setattr(desktop, "GNOME_TEXT_EDITOR", binary)
    owned = desktop.TaskDesktop({"DISPLAY": ":test"}, editor="gnome-text-editor")
    calls = []
    marker = object()
    monkeypatch.setattr(owned, "open_application", lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    assert owned.open_blank("gedit", tmp_path, browser="firefox", log_path=tmp_path / "log") is marker
    assert calls == [((str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$"),
                      {"log_path": tmp_path / "log", "expected_executable": binary.resolve()})]


def test_gnome_alias_to_gedit_is_not_native_editor(monkeypatch, tmp_path):
    gedit = tmp_path / "gedit"
    gedit.write_bytes(b"not GNOME Text Editor")
    alias = tmp_path / "gnome-text-editor"
    alias.symlink_to(gedit)
    monkeypatch.setattr(desktop, "GNOME_TEXT_EDITOR", alias)
    owned = desktop.TaskDesktop({}, editor="gnome-text-editor")
    monkeypatch.setattr(owned, "open_application", lambda *_args, **_kwargs: pytest.fail("No alias launch"))
    with pytest.raises(desktop.CapabilityBlocked, match="native GNOME"):
        owned.open_blank("gedit", tmp_path, browser="firefox", log_path=tmp_path / "log")


def process_fixture(monkeypatch, tmp_path):
    binary = tmp_path / "gnome-text-editor"
    binary.write_bytes(b"native executable")
    process = tmp_path / "proc" / "42"
    process.mkdir(parents=True)
    (process / "exe").symlink_to(binary)
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "100")
    return binary, process.parent


def test_native_editor_pid_uid_executable_start_are_verified(monkeypatch, tmp_path):
    binary, proc_root = process_fixture(monkeypatch, tmp_path)
    desktop._verify_native_editor_process("42", binary, "100", proc_root=proc_root)


@pytest.mark.parametrize("mutation", ["unknown_pid", "wrong_uid", "wrong_exe", "wrong_start", "reused_pid"])
def test_native_editor_process_unknown_or_changed_is_rejected(monkeypatch, tmp_path, mutation):
    binary, proc_root = process_fixture(monkeypatch, tmp_path)
    pid, start = "42", "100"
    if mutation == "unknown_pid":
        pid = "43"
    elif mutation == "wrong_uid":
        uid = os.getuid()
        monkeypatch.setattr(desktop.os, "getuid", lambda: uid + 1)
    elif mutation == "wrong_exe":
        other = tmp_path / "different-program"
        other.write_bytes(b"other")
        (proc_root / pid / "exe").unlink()
        (proc_root / pid / "exe").symlink_to(other)
    elif mutation == "wrong_start":
        start = "101"
    else:
        ticks = iter(["100", "101"])
        monkeypatch.setattr(desktop, "process_start", lambda _pid: next(ticks))
    with pytest.raises(desktop.CapabilityBlocked, match="identity"):
        desktop._verify_native_editor_process(pid, binary, start, proc_root=proc_root)


def launch_fixture(monkeypatch, tmp_path, *, pid="42", new=None, launch_code=0, verification_error=False):
    binary = tmp_path / "gnome-text-editor"
    binary.write_bytes(b"native")
    owned = desktop.TaskDesktop({"DISPLAY": ":test"}, editor="gnome-text-editor")
    calls, verified = [], []
    process = SimpleNamespace(pid=99, poll=lambda: launch_code)
    monkeypatch.setattr(desktop.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda argv, **kwargs: calls.append(("launch", argv, kwargs)) or process)
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "100")
    monkeypatch.setattr(desktop, "class_windows", lambda _cls, _env, *, visible=False: ({"10"} | (new or {"20"})) if visible else {"10"})
    def command(argv, environment):
        calls.append(("command", argv, environment))
        return pid if argv[1] == "getwindowpid" else ""
    monkeypatch.setattr(desktop, "command", command)
    monkeypatch.setattr(owned, "register", lambda window: owned.windows.append(window) or window)
    def verify(actual_pid, executable, expected_start):
        verified.append((actual_pid, executable, expected_start))
        if verification_error:
            raise desktop.CapabilityBlocked("Native editor process identity is unproven")
    monkeypatch.setattr(desktop, "_verify_native_editor_process", verify)
    return owned, binary, calls, verified


def test_shared_existing_editor_pid_owns_only_new_visible_xid(monkeypatch, tmp_path):
    owned, binary, calls, verified = launch_fixture(monkeypatch, tmp_path)
    window = owned.open_application(str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$",
                                    expected_executable=binary)
    assert (window.window_id, window.pid, window.process_start) == ("20", "42", "100")
    assert window.pid != "99" and owned.window_ids == {"20"}
    assert verified == [("42", binary, "100"), ("42", binary, "100")]
    assert calls[-1][1] == ["xdotool", "windowactivate", "--sync", "20"]
    assert owned.launches[0]["before_xids"] == ["10"]
    assert owned.is_owned({"window_id": "20", "process_id": 42})
    assert not owned.is_owned({"window_id": "10", "process_id": 42})


@pytest.mark.parametrize("mutation", ["unknown_pid", "unproven_exe_uid", "ambiguous", "nonzero_launch"])
def test_native_handoff_rejection_never_registers_activates_or_closes(monkeypatch, tmp_path, mutation):
    owned, binary, calls, _ = launch_fixture(monkeypatch, tmp_path,
        pid="unknown" if mutation == "unknown_pid" else "42",
        new={"20", "21"} if mutation == "ambiguous" else None,
        launch_code=2 if mutation == "nonzero_launch" else 0,
        verification_error=mutation == "unproven_exe_uid")
    with pytest.raises(desktop.CapabilityBlocked):
        owned.open_application(str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$",
                               expected_executable=binary)
    assert owned.windows == [] and not any(c[0] == "command" and "windowactivate" in c[1] for c in calls)
    assert owned.close()[0]["uncertain_unregistered_launch"] is True
    assert not any(c[0] == "command" and c[1][1] in {"windowclose", "windowkill"} for c in calls)


def test_editor_cleanup_closes_new_xid_only_and_preserves_old_shared_window(monkeypatch, tmp_path):
    owned, binary, _, _ = launch_fixture(monkeypatch, tmp_path)
    owned.open_application(str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$",
                           expected_executable=binary)
    closed = []
    monkeypatch.setattr(desktop, "class_windows", lambda *_args, **_kwargs: {"10"} if closed else {"10", "20"})
    monkeypatch.setattr(desktop, "request_owned_window_close", lambda xid, pid, start, env: closed.append((xid, pid, start)) or True)
    assert owned.close()[0]["closed"] is True
    assert closed == [("20", "42", "100")]
    assert "10" not in {item[0] for item in closed}


def test_editor_cleanup_refuses_changed_executable_without_kill(monkeypatch, tmp_path):
    owned, binary, _, _ = launch_fixture(monkeypatch, tmp_path)
    owned.open_application(str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$",
                           expected_executable=binary)
    monkeypatch.setattr(desktop, "class_windows", lambda *_args, **_kwargs: {"10", "20"})
    monkeypatch.setattr(desktop, "_verify_native_editor_process", lambda *_args: (_ for _ in ()).throw(desktop.CapabilityBlocked("identity changed")))
    monkeypatch.setattr(desktop, "request_owned_window_close", lambda *_args: pytest.fail("No close without proof"))
    receipt = owned.close()[0]
    assert receipt["closed"] is False and receipt["identity_or_close_unavailable"] is True


def test_editor_identity_changed_at_registration_never_activates_or_closes(monkeypatch, tmp_path):
    owned, binary, calls, _ = launch_fixture(monkeypatch, tmp_path)
    count = 0
    def verify(*_args):
        nonlocal count
        count += 1
        if count == 2:
            raise desktop.CapabilityBlocked("Identity changed after registration")
    monkeypatch.setattr(desktop, "_verify_native_editor_process", verify)
    with pytest.raises(desktop.CapabilityBlocked, match="Identity changed"):
        owned.open_application(str(binary), ["--new-window"], "gnome-text-editor", "^gnome-text-editor$",
                               expected_executable=binary)
    assert not owned.windows and not any(c[0] == "command" and "windowactivate" in c[1] for c in calls)
    assert owned.close()[0]["uncertain_unregistered_launch"]


def test_configured_editor_does_not_claim_setup_when_case_has_no_editor(monkeypatch, tmp_path):
    case = next(item for item in runner.read_json(DATA / "tasks.json")["tasks"]
                if "gedit" not in item["fixture"]["blank_apps"])
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "load_case", lambda _id: case)
    monkeypatch.setattr(runner, "source_state", lambda: {"revision": "frozen"})
    monkeypatch.setattr(runner, "worker_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner, "TaskDesktop", lambda *_args, **_kwargs: pytest.fail("No app during prepare-only"))
    args = runner.parser().parse_args(["--case", case["case_id"], "--prepare-only", "--editor", "gnome-text-editor",
                                      "--output", str(tmp_path / "runtime-data/run")])
    runner.run(args)
    metadata = runner.read_json(args.output / "summary.json")["editor"]
    assert metadata["configured_application"] == "gnome-text-editor" and not metadata["setup_slot_present"]
    assert metadata["declared_application"] is None and metadata["effective_application"] is None
