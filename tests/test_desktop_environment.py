from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

from harness import location as location_module
from portal import browser as browser_module
from portal import desktop as desktop_module
from portal import gui as gui_module
from portal.desktop import desktop_subprocess_environment


def test_desktop_environment_recovers_session_bus_and_wayland(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with socket.socket(socket.AF_UNIX) as bus, socket.socket(socket.AF_UNIX) as wayland:
        bus.bind(str(runtime / "bus"))
        wayland.bind(str(runtime / "wayland-7"))

        environment = desktop_subprocess_environment(
            {}, runtime_dir=runtime, x11_dir=tmp_path / "x11", home=tmp_path
        )

    assert environment["XDG_RUNTIME_DIR"] == str(runtime)
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={runtime / 'bus'}"
    assert environment["WAYLAND_DISPLAY"] == "wayland-7"
    assert "DISPLAY" not in environment


def test_desktop_environment_preserves_explicit_session_values(tmp_path: Path) -> None:
    explicit = {
        "DISPLAY": ":44",
        "WAYLAND_DISPLAY": "wayland-explicit",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/chosen/bus",
        "XDG_RUNTIME_DIR": "/chosen/runtime",
        "XAUTHORITY": "/chosen/auth",
    }

    environment = desktop_subprocess_environment(
        explicit, runtime_dir=tmp_path, x11_dir=tmp_path, home=tmp_path
    )

    assert environment == explicit


def test_desktop_environment_recovers_one_x11_display(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    x11 = tmp_path / "x11"
    runtime.mkdir()
    x11.mkdir()
    with socket.socket(socket.AF_UNIX) as display:
        display.bind(str(x11 / "X0"))

        environment = desktop_subprocess_environment(
            {}, runtime_dir=runtime, x11_dir=x11, home=tmp_path
        )

    assert environment["DISPLAY"] == ":0"


def test_desktop_environment_prefers_signed_in_user_manager_display(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            "DISPLAY=:0\nXAUTHORITY=/run/user/4242/gdm/Xauthority\n"
            "UNRELATED_SECRET=never-copy-this\n",
            "",
        )

    monkeypatch.setattr(desktop_module.os, "getuid", lambda: 4242)
    monkeypatch.setattr(desktop_module.subprocess, "run", run)

    manager = desktop_module._systemd_user_desktop_environment(
        {
            "XDG_RUNTIME_DIR": "/run/user/4242",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/4242/bus",
        },
        Path("/run/user/4242"),
    )

    assert manager == {
        "DISPLAY": ":0",
        "XAUTHORITY": "/run/user/4242/gdm/Xauthority",
    }
    assert captured["command"] == ["systemctl", "--user", "show-environment"]


def test_gui_commands_receive_recovered_desktop_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "1280 720\n", "")

    monkeypatch.setattr(
        gui_module,
        "desktop_subprocess_environment",
        lambda: {"DISPLAY": ":0", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/bus"},
    )
    monkeypatch.setattr(gui_module.subprocess, "run", run)

    assert gui_module.GuiAutomation()._run(["xdotool", "getdisplaygeometry"]) == (
        "1280 720"
    )
    assert captured["env"] == {
        "DISPLAY": ":0",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/bus",
    }


def test_location_browser_is_isolated_from_web_search(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "<html>{}</html>", "")

    monkeypatch.setattr(
        location_module, "_find_location_browser", lambda: "/usr/bin/chromium"
    )
    monkeypatch.setattr(
        location_module,
        "desktop_subprocess_environment",
        lambda: {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
    )
    monkeypatch.setattr(location_module.subprocess, "run", run)

    assert "{}" in location_module._run_location_browser("https://example.test", 5)
    assert "--headless=new" in captured["command"]
    assert captured["env"] == {
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"
    }


def test_visible_browser_uses_the_recovered_display(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Process:
        pid = 4242

        @staticmethod
        def poll() -> None:
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(browser_module.shutil, "which", lambda _binary: "/chromium")
    monkeypatch.setattr(
        browser_module,
        "desktop_subprocess_environment",
        lambda: {"WAYLAND_DISPLAY": "wayland-0", "XDG_RUNTIME_DIR": "/run/user/1000"},
    )
    monkeypatch.setattr(browser_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        browser_module.BrowserAutomationStore,
        "_page_socket",
        lambda _self, _port: "ws://127.0.0.1/devtools/page/one",
    )
    store = browser_module.BrowserAutomationStore(chromium_bin="/chromium")

    session = store._launch()
    try:
        assert "--headless=new" not in captured["command"]
        assert session.visible_on_desktop is True
        assert captured["env"] == {
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
    finally:
        shutil.rmtree(session.profile, ignore_errors=True)


def test_visible_browser_refuses_to_fall_back_to_headless_without_a_display(
    monkeypatch,
) -> None:
    monkeypatch.setattr(browser_module.shutil, "which", lambda _binary: "/chromium")
    monkeypatch.setattr(
        browser_module,
        "desktop_subprocess_environment",
        lambda: {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
    )
    store = browser_module.BrowserAutomationStore(chromium_bin="/chromium")

    try:
        store._launch()
    except browser_module.BrowserDesktopUnavailable as error:
        assert "did not launch a headless substitute" in str(error)
    else:
        raise AssertionError("expected visible browser desktop requirement")


def test_browser_desktop_failure_requests_gui_capability() -> None:
    class MissingDesktop:
        def act(self, _session_id, _arguments):
            raise browser_module.BrowserDesktopUnavailable("no desktop")

        def clear(self, _session_id):
            pass

    from portal.documents import SessionDocumentStore
    from portal.tools import PortalToolHarness

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), browser_automation=MissingDesktop()
    )
    result = harness.execute("session", "browser_interact", {"action": "snapshot"})

    assert result["error"] == "BrowserDesktopUnavailable"
    assert result["disposition"] == "change_capability"
    assert result["task_blocked"] is False
    assert result["alternative_tools"] == ["gui_interact"]


def test_browser_profile_process_discovery_is_exact(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    owned_profile = Path("/tmp/omni-visible-chromium-owned")
    other_profile = Path("/tmp/omni-visible-chromium-other")
    for pid, arguments in {
        "101": [b"/app/chromium/chrome", f"--user-data-dir={owned_profile}".encode()],
        "102": [b"bwrap", f"--user-data-dir={other_profile}".encode()],
        "103": [b"python", f"text mentioning --user-data-dir={owned_profile}".encode()],
        "104": [
            b"/app/chromium/chrome --no-sandbox "
            + f"--user-data-dir={owned_profile} about:blank".encode()
        ],
    }.items():
        process = proc / pid
        process.mkdir()
        (process / "cmdline").write_bytes(b"\0".join(arguments) + b"\0")

    assert browser_module._profile_process_ids(owned_profile, proc) == [101, 104]
    assert browser_module._profile_process_ids(Path("/tmp/unowned"), proc) == []


def test_browser_keeps_the_controlled_tab_when_chromium_adds_a_blank_tab(
    monkeypatch,
) -> None:
    store = browser_module.BrowserAutomationStore()
    targets = [
        {
            "id": "reddit",
            "type": "page",
            "title": "Reddit - Prove your humanity",
            "url": "https://www.reddit.com/r/robots/",
            "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/reddit",
        },
        {
            "id": "blank",
            "type": "page",
            "title": "",
            "url": "about:blank",
            "webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/blank",
        },
    ]
    monkeypatch.setattr(store, "_json", lambda _port, _path: targets)

    assert store._page_socket(9222, "reddit").endswith("/reddit")
    assert store._page_socket(9222).endswith("/reddit")


def test_browser_challenge_hands_the_visible_window_to_gui() -> None:
    metadata = browser_module._challenge_metadata(
        "Reddit - Prove your humanity",
        "https://www.reddit.com/r/robots/",
        "",
    )

    assert metadata["challenge"] is True
    assert metadata["challenge_kind"] == "recaptcha"
    assert metadata["disposition"] == "change_capability"
    assert metadata["task_blocked"] is False
    assert metadata["alternative_tools"] == ["gui_interact"]
