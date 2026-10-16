"""CPU setup-boundary controls; no browser or display is started."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime import long_horizon_desktop as desktop
from runtime import verify_long_horizon_native as runner


@pytest.fixture
def installed_snap(monkeypatch, tmp_path):
    dispatcher = tmp_path / "snap"
    dispatcher.write_bytes(b"\x7fELFfake CPU fixture; never executed")
    dispatcher.chmod(0o755)
    launcher = tmp_path / "chromium"
    launcher.symlink_to(dispatcher)
    monkeypatch.setattr(desktop, "SNAP_CHROMIUM", launcher, raising=False)
    monkeypatch.setattr(desktop, "SNAP_DISPATCHER", dispatcher, raising=False)
    original = Path.stat

    def root_owned(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path in {launcher, dispatcher}:
            fields = list(value)
            fields[4] = 0  # Explicit fixed installation boundary, not user ownership.
            return os.stat_result(fields, {key: getattr(value, key) for key in
                                   ("st_atime_ns", "st_mtime_ns", "st_ctime_ns")})
        return value

    monkeypatch.setattr(Path, "stat", root_owned)
    return launcher, dispatcher


def test_explicit_snap_choice_is_accepted_by_closed_cli(tmp_path):
    args = runner.parser().parse_args([
        "--output", str(tmp_path / "run"), "--browser", "chromium-snap"])
    assert args.browser == "chromium-snap"


def test_snap_choice_selects_fixed_alias_without_path_or_profile_change(monkeypatch, tmp_path, installed_snap):
    launcher, _ = installed_snap
    environment = {"PATH": "/shadowing/flatpak", "DISPLAY": ":cpu-fixture"}
    owned = desktop.TaskDesktop(environment)
    calls = []
    marker = object()
    monkeypatch.setattr(owned, "open_application", lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    assert owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None) is marker
    assert calls == [((str(launcher), ["--new-window", "about:blank"], "browser", "[Cc]hromium"),
                      {"log_path": None})]
    assert owned.environment == environment


@pytest.mark.parametrize("browser", ["/snap/bin/chromium", "/tmp/browser", "chromium-snap --profile=x", "arbitrary"])
def test_cli_does_not_accept_caller_selected_launcher_or_flags(tmp_path, browser):
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["--output", str(tmp_path / "run"), "--browser", browser])


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_existing_ordinary_choices_keep_path_selection(monkeypatch, tmp_path, browser):
    owned = desktop.TaskDesktop({})
    calls = []
    monkeypatch.setattr(desktop, "_snap_chromium_launcher", lambda: pytest.fail("No Snap selection"))
    monkeypatch.setattr(owned, "open_application", lambda *args, **kwargs: calls.append((args, kwargs)))
    owned.open_blank("browser", tmp_path, browser=browser, log_path=None)
    assert calls[0][0][0] == browser


@pytest.mark.parametrize("fault", ["missing", "regular_alias", "wrong_target", "symlink_loop", "non_ELF",
                                   "writable_dispatcher", "not_executable", "foreign_alias", "foreign_dispatcher"])
def test_invalid_snap_launcher_rejects_before_any_birth(monkeypatch, tmp_path, installed_snap, fault):
    launcher, dispatcher = installed_snap
    if fault == "missing":
        launcher.unlink()
    elif fault == "regular_alias":
        launcher.unlink()
        launcher.write_bytes(b"\x7fELF")
    elif fault == "wrong_target":
        other = tmp_path / "different-dispatcher"
        other.write_bytes(b"\x7fELF")
        launcher.unlink()
        launcher.symlink_to(other)
    elif fault == "symlink_loop":
        launcher.unlink()
        launcher.symlink_to(launcher)
    elif fault == "non_ELF":
        dispatcher.write_text("#!/bin/sh\nexec arbitrary-browser\n")
    elif fault == "writable_dispatcher":
        dispatcher.chmod(0o777)
    elif fault == "not_executable":
        dispatcher.chmod(0o644)
    else:
        original = Path.stat
        foreign = launcher if fault == "foreign_alias" else dispatcher
        def foreign_uid(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == foreign:
                fields = list(value)
                fields[4] = 1000
                return os.stat_result(fields, {key: getattr(value, key) for key in
                                   ("st_atime_ns", "st_mtime_ns", "st_ctime_ns")})
            return value
        monkeypatch.setattr(Path, "stat", foreign_uid)
    owned = desktop.TaskDesktop({})
    monkeypatch.setattr(owned, "open_application", lambda *_args, **_kwargs: pytest.fail("No application birth"))
    with pytest.raises(desktop.CapabilityBlocked, match="Snap Chromium launcher"):
        owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None)
    assert owned.launches == [] and owned.windows == []


def test_snap_dispatcher_change_during_validation_is_rejected(monkeypatch, tmp_path, installed_snap):
    _, dispatcher = installed_snap
    original = Path.stat
    samples = 0
    def changing(path, *args, **kwargs):
        nonlocal samples
        value = original(path, *args, **kwargs)
        if path == dispatcher:
            fields = list(value)
            fields[8] += samples
            changed_ns = value.st_mtime_ns + samples * 1_000_000_000
            samples += 1
            return os.stat_result(fields, {"st_atime_ns": value.st_atime_ns,
                                          "st_mtime_ns": changed_ns, "st_ctime_ns": value.st_ctime_ns})
        return value
    monkeypatch.setattr(Path, "stat", changing)
    owned = desktop.TaskDesktop({})
    monkeypatch.setattr(owned, "open_application", lambda *_args, **_kwargs: pytest.fail("No application birth"))
    with pytest.raises(desktop.CapabilityBlocked, match="Snap Chromium launcher"):
        owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None)


def ordinary_launch_fixture(monkeypatch, installed_snap, *, reject_host=False, ambiguous=False):
    from types import SimpleNamespace

    from runtime import acceptance_native_support as support
    launcher, _ = installed_snap
    owned = desktop.TaskDesktop({"PATH": "/shadowing/flatpak", "DISPLAY": ":cpu-fixture"})
    calls = []
    process = SimpleNamespace(pid=99)
    monkeypatch.setattr(desktop.shutil, "which", lambda binary: str(launcher) if binary == str(launcher) else None)
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda argv, **kwargs: calls.append(("launch", argv, kwargs)) or process)
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "100")
    monkeypatch.setattr(support, "process_start", lambda _pid: "100")
    monkeypatch.setattr(desktop, "class_windows", lambda _cls, _env, *, visible=False:
                        {"10", "20", "21"} if visible and ambiguous else {"10", "20"} if visible else {"10"})
    def command(argv, environment):
        calls.append(("command", argv, environment))
        return "42" if argv[1] == "getwindowpid" else ""
    monkeypatch.setattr(desktop, "command", command)
    monkeypatch.setattr(support, "command", command)
    def host_identity(pid, start):
        calls.append(("host_identity", pid, start))
        if reject_host or (pid, start) != ("42", "100"):
            raise RuntimeError("Host process identity unproved")
    monkeypatch.setattr(support, "_assert_host_process", host_identity)
    return owned, calls


def test_snap_uses_real_ordinary_registration_and_exact_cleanup(monkeypatch, tmp_path, installed_snap):
    owned, calls = ordinary_launch_fixture(monkeypatch, installed_snap)
    launcher, _ = installed_snap
    window = owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None)
    assert (window.window_id, window.pid, window.process_start) == ("20", "42", "100")
    assert owned.window_ids == {"20"} and ("host_identity", "42", "100") in calls
    argv = calls[0][1]
    assert argv == [str(launcher), "--new-window", "about:blank"]
    assert calls[0][2]["env"] == owned.environment
    closed = []
    monkeypatch.setattr(desktop, "class_windows", lambda *_args, **_kwargs: {"10"} if closed else {"10", "20"})
    monkeypatch.setattr(desktop, "request_owned_window_close", lambda xid, pid, start, env:
                        closed.append((xid, pid, start)) or True)
    assert owned.close()[0]["closed"] is True
    assert closed == [("20", "42", "100")]


@pytest.mark.parametrize("fault", ["host_unproved", "ambiguous"])
def test_snap_does_not_bypass_existing_ownership_rejection(monkeypatch, tmp_path, installed_snap, fault):
    owned, calls = ordinary_launch_fixture(monkeypatch, installed_snap,
        reject_host=fault == "host_unproved", ambiguous=fault == "ambiguous")
    with pytest.raises((RuntimeError, desktop.CapabilityBlocked)):
        owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None)
    assert owned.windows == []
    assert not any(x[0] == "command" and "windowactivate" in x[1] for x in calls)
    monkeypatch.setattr(desktop, "request_owned_window_close", lambda *_args: pytest.fail("No unproved close"))
    assert owned.close()[0]["uncertain_unregistered_launch"] is True


def test_own_ELF_read_may_update_actual_atime_without_invalidating_launcher(installed_snap):
    launcher, dispatcher = installed_snap
    original = dispatcher.stat()
    os.utime(dispatcher, ns=(1, original.st_mtime_ns))
    assert dispatcher.stat().st_atime_ns == 1
    assert desktop._snap_chromium_launcher() == str(launcher)
    after = dispatcher.stat()
    assert after.st_atime_ns > 1  # Actual filesystem update from the helper's own ELF read.
    assert after.st_mtime_ns == original.st_mtime_ns


@pytest.mark.parametrize("change", ["mtime", "inode"])
def test_actual_dispatcher_mutation_after_ELF_read_rejects_birth(monkeypatch, tmp_path, installed_snap, change):
    from contextlib import contextmanager

    _, dispatcher = installed_snap
    original = Path.open
    @contextmanager
    def altered(path, *args, **kwargs):
        with original(path, *args, **kwargs) as stream:
            yield stream
        if path == dispatcher:
            if change == "mtime":
                value = dispatcher.stat()
                os.utime(dispatcher, ns=(value.st_atime_ns, value.st_mtime_ns + 1))
            else:
                replacement = tmp_path / "replacement-snap"
                replacement.write_bytes(b"\x7fELFfake CPU fixture; never executed")
                replacement.chmod(0o755)
                replacement.replace(dispatcher)
    monkeypatch.setattr(Path, "open", altered)
    owned = desktop.TaskDesktop({})
    monkeypatch.setattr(owned, "open_application", lambda *_args, **_kwargs: pytest.fail("No application birth"))
    with pytest.raises(desktop.CapabilityBlocked, match="Snap Chromium launcher"):
        owned.open_blank("browser", tmp_path, browser="chromium-snap", log_path=None)
    assert owned.launches == [] and owned.windows == []
