"""Read actual GTK text after physical typing on an isolated X server."""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import time

import pytest

from portal.native_ui import _x11_text_command

_WIDGET = '''
import json, os, pathlib, sys, gi
gi.require_version("Gtk", "3.0")
gi.require_version("GdkX11", "3.0")
from gi.repository import Gtk, GdkX11, GLib
folder = pathlib.Path(sys.argv[1])
window = Gtk.Window(title="isolated-native-text-test")
view = Gtk.TextView()
window.add(view)
window.show_all()
view.grab_focus()
def tick():
    buffer = view.get_buffer()
    first, last = buffer.get_bounds()
    value = buffer.get_text(first, last, True)
    temp = folder / "state.tmp"
    temp.write_text(json.dumps({"pid": os.getpid(), "xid": window.get_window().get_xid(), "value": value}))
    temp.replace(folder / "state.json")
    if (folder / "stop").exists():
        Gtk.main_quit()
        return False
    return True
GLib.timeout_add(20, tick)
tick()
Gtk.main()
'''


@pytest.fixture(scope="module")
def private_text_widget(tmp_path_factory):
    if not all(shutil.which(name) for name in ("Xvfb", "xdotool", "dbus-run-session")):
        pytest.skip("Requires actual Xvfb, xdotool and a private D-Bus launcher")
    probe = subprocess.run(["/usr/bin/python3", "-c",
        "import gi; gi.require_version('Gtk','3.0'); gi.require_version('GdkX11','3.0')"],
        capture_output=True, timeout=5, check=False)
    if probe.returncode:
        pytest.skip("Requires system Python GTK3 bindings")
    folder = tmp_path_factory.mktemp("native-text-xtest")
    script = folder / "widget.py"
    script.write_text(_WIDGET)
    read_fd, write_fd = os.pipe()
    server = widget = None
    try:
        with (folder / "display.log").open("wb") as log:
            server = subprocess.Popen(["Xvfb", "-displayfd", str(write_fd), "-screen", "0",
                "1024x768x24", "-nolisten", "tcp", "-noreset"], pass_fds=(write_fd,), stdout=log, stderr=log)
        os.close(write_fd)
        write_fd = None
        assert select.select([read_fd], [], [], 5)[0], "Private display startup deadline"
        display = ":" + os.read(read_fd, 64).decode().strip()
        assert display[1:].isdigit()
        env = {**os.environ, "DISPLAY": display, "NO_AT_BRIDGE": "1"}
        env.pop("DBUS_SESSION_BUS_ADDRESS", None)
        with (folder / "widget.log").open("wb") as log:
            widget = subprocess.Popen(["dbus-run-session", "--", "/usr/bin/python3", str(script), str(folder)],
                env=env, stdout=log, stderr=log)

        def state():
            path = folder / "state.json"
            return json.loads(path.read_text()) if path.exists() else None

        deadline = time.monotonic() + 5
        while state() is None and time.monotonic() < deadline and widget.poll() is None:
            time.sleep(.01)
        ready = state()
        assert ready is not None, "Private widget startup deadline"
        subprocess.run(["xdotool", "windowfocus", "--sync", str(ready["xid"])],
            env=env, capture_output=True, check=True, timeout=3)
        yield env, state
    finally:
        (folder / "stop").write_text("stop")
        for process in (widget, server):
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.wait(3)
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


@pytest.mark.parametrize("text", [
    "ordinary ASCII", "literal\\nsequence", "first\nsecond\nthird\nfourth\nfifth\nsixth\nseventh",
    "\nleading", "trailing\n", "a\n\n\nb", "\n", "café\nnaïve",
    "--file /etc/passwd\nkey Return type\n$(printf literal); `literal`",
])
def test_actual_xtest_preserves_literal_text_and_line_breaks(private_text_widget, text):
    env, state = private_text_widget
    subprocess.run(["xdotool", "key", "--clearmodifiers", "Control_L+a", "BackSpace"],
        env=env, capture_output=True, check=True, timeout=3)
    deadline = time.monotonic() + 1
    while state()["value"] != "" and time.monotonic() < deadline:
        time.sleep(.01)
    assert state()["value"] == ""
    subprocess.run(_x11_text_command(text), env=env, capture_output=True, check=True, timeout=5)
    deadline = time.monotonic() + 1
    while state()["value"] != text and time.monotonic() < deadline:
        time.sleep(.01)
    assert state()["value"] == text
