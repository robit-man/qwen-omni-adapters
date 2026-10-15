"""Owned Snap launch delegation keeps the portal's service sandbox intact."""

from __future__ import annotations

import io
import queue
import stat
from pathlib import Path
from shutil import which as executable_on_path
from types import SimpleNamespace

import pytest

from portal import browser


@pytest.fixture
def launch_host(monkeypatch, tmp_path):
    captured = {}
    environment = {
        "DISPLAY": ":0", "XAUTHORITY": "/run/user/1000/gdm/Xauthority",
        "XDG_RUNTIME_DIR": "/run/user/1000", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "OMNI_PORTAL_TOKEN": "private-portal-credential",
    }
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(browser, "desktop_subprocess_environment", lambda: dict(environment))
    monkeypatch.setattr(browser, "_restricted_linux_caller", lambda: True)
    monkeypatch.setattr(browser, "_owned_profile_processes", lambda **kwargs: {})
    monkeypatch.setattr(browser, "_free_loopback_port", lambda: 42424)
    monkeypatch.setattr(browser.shutil, "which", lambda value: value)

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=4321, poll=lambda: None, stderr=io.BytesIO())

    monkeypatch.setattr(browser.subprocess, "Popen", popen)
    monkeypatch.setattr(browser.BrowserAutomationStore, "_page_socket", lambda *_: "ws://127.0.0.1/devtools/page/one")
    return captured, environment, tmp_path


def test_sandboxed_snap_launch_uses_own_user_unit_and_host_visible_private_profile(launch_host) -> None:
    captured, environment, tmp_path = launch_host
    store = browser.BrowserAutomationStore(chromium_bin="/snap/bin/chromium")

    session = store._launch()

    command = captured["command"]
    assert command[:6] == ["systemd-run", "--user", "--wait", "--pipe", "--collect", "--service-type=exec"]
    assert f"--unit={session.user_unit}" in command
    assert "--property=TimeoutStopSec=5" in command
    assert command[command.index("--") + 1] == "/snap/bin/chromium"
    assert "--remote-debugging-address=127.0.0.1" in command
    assert not any("--headless" in value for value in command)
    assert session.profile.parent == tmp_path / "home" / "snap" / "chromium" / "common"
    assert stat.S_IMODE(session.profile.stat().st_mode) == 0o700
    assert session.user_unit == browser._profile_user_unit(session.profile)
    assert "private-portal-credential" not in " ".join(command)
    assert "--setenv=DISPLAY=:0" in command
    assert "--setenv=OMNI_PORTAL_TOKEN=" not in " ".join(command)
    assert captured["env"] == environment


def test_restricted_path_discovers_real_snap_before_transitional_wrapper_and_delegates(launch_host, monkeypatch) -> None:
    captured, _environment, tmp_path = launch_host
    restricted_bin = tmp_path / "usr" / "bin"
    restricted_bin.mkdir(parents=True)
    wrapper = restricted_bin / "chromium-browser"
    wrapper.write_text('#!/bin/sh\nexec /snap/bin/chromium "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(restricted_bin))
    monkeypatch.delenv("OMNI_CHROMIUM_BIN", raising=False)
    probes = []

    def restricted_which(value):
        probes.append(value)
        if value == "/snap/bin/chromium":
            return value
        if value == "systemd-run":
            return "/usr/bin/systemd-run"
        if value == "/usr/local/bin/chromium":
            return None
        return executable_on_path(value)

    monkeypatch.setattr(browser.shutil, "which", restricted_which)
    assert executable_on_path("chromium") is None
    assert executable_on_path("chromium-browser") == str(wrapper)
    store = browser.BrowserAutomationStore()
    session = store._launch()

    assert store.chromium_bin == "/snap/bin/chromium"
    assert "/snap/bin/chromium" in probes
    assert captured["command"][:2] == ["systemd-run", "--user"]
    assert captured["command"][captured["command"].index("--") + 1] == "/snap/bin/chromium"
    assert session.user_unit.startswith("omni-browser-")
    assert session.profile.parent == tmp_path / "home" / "snap" / "chromium" / "common"


@pytest.mark.parametrize("name", ["/usr/local/bin/chromium", "chromium", "chromium-browser", "google-chrome"])
def test_native_browser_discovery_priority_is_preserved_when_snap_is_installed(launch_host, monkeypatch, name) -> None:
    captured, _environment, tmp_path = launch_host
    native = tmp_path / "native" / Path(name).name
    native.parent.mkdir()
    native.write_bytes(b'\x7fELF native Chromium\nexec /snap/bin/chromium "$@"\n')
    monkeypatch.delenv("OMNI_CHROMIUM_BIN", raising=False)
    available = {name: str(native), str(native): str(native), "/snap/bin/chromium": "/snap/bin/chromium"}
    monkeypatch.setattr(browser.shutil, "which", available.get)

    store = browser.BrowserAutomationStore()
    session = store._launch()

    assert store.chromium_bin == str(native)
    assert captured["command"][0] == str(native)
    assert session.user_unit == ""


def test_explicit_transitional_wrapper_is_recognized_as_snap_and_delegated(launch_host) -> None:
    captured, _environment, tmp_path = launch_host
    wrapper = tmp_path / "chromium-browser"
    wrapper.write_text('#!/bin/sh\nexec /snap/bin/chromium "$@"\n')
    store = browser.BrowserAutomationStore(chromium_bin=str(wrapper))
    session = store._launch()

    assert captured["command"][:2] == ["systemd-run", "--user"]
    assert session.user_unit.startswith("omni-browser-")
    assert session.profile.parent == tmp_path / "home" / "snap" / "chromium" / "common"


@pytest.mark.parametrize("configured", ["argument", "environment"])
def test_explicit_custom_native_browser_wins_over_snap_discovery(launch_host, monkeypatch, configured) -> None:
    captured, _environment, _tmp_path = launch_host
    monkeypatch.setenv("OMNI_CHROMIUM_BIN", "/custom/desktop/chromium")
    kwargs = {"chromium_bin": "/custom/argument/chromium"} if configured == "argument" else {}
    store = browser.BrowserAutomationStore(**kwargs)
    session = store._launch()

    assert captured["command"][0] == (
        "/custom/argument/chromium" if configured == "argument" else "/custom/desktop/chromium")
    assert session.user_unit == ""


@pytest.mark.parametrize("binary,restricted", [("/snap/bin/chromium", False), ("/usr/bin/google-chrome", True)])
def test_direct_desktop_and_native_chrome_launches_remain_direct(launch_host, monkeypatch, binary, restricted) -> None:
    captured, _environment, _tmp_path = launch_host
    monkeypatch.setattr(browser, "_restricted_linux_caller", lambda: restricted)
    session = browser.BrowserAutomationStore(chromium_bin=binary)._launch()

    assert captured["command"][0] == binary
    assert session.user_unit == ""
    assert session.visible_on_desktop is True


def test_snap_launch_requires_manager_instead_of_falling_back_to_restricted_child(launch_host, monkeypatch) -> None:
    captured, _environment, tmp_path = launch_host
    monkeypatch.setattr(browser.shutil, "which", lambda value: None if value == "systemd-run" else value)

    with pytest.raises(browser.BrowserDesktopUnavailable, match="signed-in user's systemd manager"):
        browser.BrowserAutomationStore(chromium_bin="/snap/bin/chromium")._launch()

    assert captured == {}
    assert list((tmp_path / "home" / "snap" / "chromium" / "common").iterdir()) == []


def test_startup_failure_exposes_bounded_redacted_diagnostic(launch_host, monkeypatch) -> None:
    _captured, _environment, _tmp_path = launch_host
    stderr = b"x" * 6000 + b" private-portal-credential https://private.example/#access=secret snap-confine cap_dac_override missing"
    monkeypatch.setattr(browser.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(
        pid=4321, poll=lambda: 1, stderr=io.BytesIO(stderr),
    ))
    terminated = []
    monkeypatch.setattr(browser.BrowserAutomationStore, "_terminate", lambda _self, session: terminated.append(session))

    with pytest.raises(browser.BrowserAutomationError) as failure:
        browser.BrowserAutomationStore(chromium_bin="/snap/bin/chromium")._launch()

    message = str(failure.value)
    assert "snap-confine cap_dac_override missing" in message
    assert "private-portal-credential" not in message
    assert "private.example" not in message and "#access=" not in message
    assert len(message) < 1100
    assert terminated[0].user_unit.startswith("omni-browser-")


def test_stderr_stops_retaining_browser_output_after_startup() -> None:
    blocks = queue.Queue()

    class Stream:
        def read(self, _size):
            return blocks.get(timeout=2)

        def close(self):
            pass

    output = browser._BrowserStartupLog(Stream())
    output.finish({})
    blocks.put(b"later page data and credentials")
    blocks.put(b"")
    output.thread.join(timeout=1)

    assert output.buffer == bytearray()
    assert output.finish({}) == ""


def test_browser_cleanup_stops_exact_unit_before_profile_removal(launch_host, monkeypatch) -> None:
    _captured, _environment, _tmp_path = launch_host
    store = browser.BrowserAutomationStore(chromium_bin="/snap/bin/chromium")
    session = store._launch()
    events = []

    def stop(unit):
        events.append(("stop", unit))
        session.process.poll = lambda: 0

    monkeypatch.setattr(browser, "_stop_browser_user_unit", stop)
    monkeypatch.setattr(browser, "_profile_process_ids", lambda _profile: [])
    monkeypatch.setattr(browser.shutil, "rmtree", lambda profile, **_kwargs: events.append(("remove", profile)))
    store._terminate(session)

    assert events == [("stop", session.user_unit), ("remove", session.profile)]


def test_profile_unit_owner_requires_actual_exact_user_cgroup(tmp_path) -> None:
    profile = tmp_path / "omni-visible-chromium-owned123"
    proc = tmp_path / "proc"
    for pid, cgroup in {
        10: "0::/user.slice/user-1000.slice/omni-browser-other.service\n",
        11: "0::/system.slice/omni-browser-owned123.service\n",
        12: "0::/user.slice/user-1000.slice/user@1000.service/app.slice/omni-browser-owned123.service\n",
    }.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cgroup").write_text(cgroup)

    assert browser._profile_unit_owner(profile, [10, 11], proc) == ""
    assert browser._profile_unit_owner(profile, [10, 11, 12], proc) == "omni-browser-owned123.service"


def test_snap_orphan_inventory_uses_common_root_and_preserves_live_profile(launch_host, monkeypatch) -> None:
    _captured, _environment, tmp_path = launch_host
    store = browser.BrowserAutomationStore(chromium_bin="/snap/bin/chromium")
    parent = tmp_path / "home" / "snap" / "chromium" / "common"
    live = parent / "omni-visible-chromium-live"
    stale = parent / "omni-visible-chromium-stale"
    events = []
    store._sessions["live"] = SimpleNamespace(profile=live, process=SimpleNamespace(poll=lambda: None))

    def inventory(**kwargs):
        assert kwargs["temp_root"] == parent
        return {live: [10], stale: [11]}

    monkeypatch.setattr(browser, "_owned_profile_processes", inventory)
    monkeypatch.setattr(browser, "_profile_unit_owner", lambda profile, _ids: browser._profile_user_unit(profile))
    monkeypatch.setattr(browser, "_stop_browser_user_unit", lambda unit: events.append(("stop", unit)))
    monkeypatch.setattr(browser.os, "kill", lambda pid, _signal: events.append(("kill", pid)))
    monkeypatch.setattr(browser, "_profile_process_ids", lambda _profile: [])
    monkeypatch.setattr(browser.shutil, "rmtree", lambda profile, **_kwargs: events.append(("remove", profile)))

    store._reap_orphan_browsers()

    assert events == [("stop", "omni-browser-stale.service"), ("remove", stale)]


@pytest.mark.parametrize("unit", ["*.service", "user@1000.service", "omni-browser-a.service other.service"])
def test_owned_unit_cleanup_rejects_foreign_or_wildcard_targets(unit) -> None:
    with pytest.raises(browser.BrowserAutomationError, match="Invalid owned Chromium user unit"):
        browser._stop_browser_user_unit(unit)


def test_restricted_caller_detects_no_new_privileges(tmp_path) -> None:
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_text("Name:\tpython\nNoNewPrivs:\t1\n")
    assert browser._restricted_linux_caller(tmp_path) is True
