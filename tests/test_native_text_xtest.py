"""Read actual GTK text after physical typing on an isolated X server."""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from portal.native_keymap import KeymapUnavailable, UnicodeKeymap
from portal.native_ui import _x11_text_command, _x11_text_segments

_WIDGET = '''
import json, os, pathlib, sys, time, gi
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
    if (folder / "stall").exists():
        (folder / "stall").unlink()
        (folder / "stalled").write_text("stalled")
        time.sleep(.25)
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
        state.directory = folder
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
    "\nleading", "trailing\n", "a\n\n\nb", "\n", "café\nnaïve", "Καλημέρα 中🙂",
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
    before = keyboard_mapping(env)
    with UnicodeKeymap(text, env) as mapping:
        prefix = ''
        for segment in _x11_text_segments(text) if mapping.codes else [text]:
            subprocess.run(_x11_text_command(segment, mapping.codes), env=env, capture_output=True, check=True, timeout=5)
            prefix += segment
            deadline = time.monotonic() + 1
            while state()["value"] != prefix and time.monotonic() < deadline:
                time.sleep(.01)
            assert state()["value"] == prefix
    assert keyboard_mapping(env) == before


def keyboard_mapping(env):
    mapping = UnicodeKeymap('', env)
    mapping._bind()
    try:
        return mapping._mapping()
    finally:
        mapping.close()


@pytest.mark.parametrize('text', ['café\nnaïve', 'éïéïéï', '中文🙂'])
def test_delayed_recipient_keeps_each_unicode_mapping_until_readback(private_text_widget, text):
    env, state = private_text_widget
    subprocess.run(['xdotool', 'key', '--clearmodifiers', 'Control_L+a', 'BackSpace'],
                   env=env, capture_output=True, check=True, timeout=3)
    deadline = time.monotonic() + 1
    while state()['value'] and time.monotonic() < deadline:
        time.sleep(.01)
    assert state()['value'] == ''
    before = keyboard_mapping(env)
    folder = state.directory
    (folder/'stalled').unlink(missing_ok=True)
    (folder/'stall').write_text('stall')
    deadline = time.monotonic() + 1
    while not (folder/'stalled').exists() and time.monotonic() < deadline:
        time.sleep(.001)
    assert (folder/'stalled').exists()
    with UnicodeKeymap(text, env) as mapping:
        assert len(set(mapping.codes.values())) == len({c for c in text if ord(c) >= 128})
        prefix = ''
        for segment in _x11_text_segments(text):
            subprocess.run(_x11_text_command(segment, mapping.codes), env=env, capture_output=True, check=True, timeout=5)
            prefix += segment
            deadline = time.monotonic() + 1
            while state()['value'] != prefix and time.monotonic() < deadline:
                time.sleep(.01)
            assert state()['value'] == prefix
    assert keyboard_mapping(env) == before


def test_keymap_capacity_failure_and_exception_restore_original_map(private_text_widget):
    env, _ = private_text_widget
    before = keyboard_mapping(env)
    with (pytest.raises(KeymapUnavailable, match='capacity'),
          UnicodeKeymap(''.join(chr(0x4E00+i) for i in range(256)), env)):
        pytest.fail('Capacity failure must precede dispatch')
    assert keyboard_mapping(env) == before
    with pytest.raises(RuntimeError, match='readback failed'), UnicodeKeymap('éï', env):
        raise RuntimeError('readback failed')
    assert keyboard_mapping(env) == before


def test_keymap_cleanup_preserves_a_foreign_reassignment(private_text_widget):
    env, _ = private_text_widget
    before = keyboard_mapping(env)
    mapping = UnicodeKeymap('é', env)
    with pytest.raises(KeymapUnavailable, match='keymap_changed'), mapping:
        code = mapping.codes['é']
        mapping._set(code, (0x01004E2D,) * len(before[code]))
        mapping.lib.XSync(mapping.display, 0)
    reader = UnicodeKeymap('', env)
    reader._bind()
    try:
        assert reader._mapping()[code][0] == 0x01004E2D
        # This fixture owns the foreign assignment and restores it separately.
        reader._set(code, before[code])
        reader.lib.XSync(reader.display, 0)
    finally:
        reader.close()
    assert keyboard_mapping(env) == before


def test_bridge_termination_unwinds_live_unicode_keymap(private_text_widget, tmp_path):
    env, _ = private_text_widget
    before = keyboard_mapping(env)
    ready = tmp_path/'mapping-ready.json'
    script = tmp_path/'held-bridge.py'
    native_source = Path(__file__).resolve().parents[1]/'portal/native_ui.py'
    script.write_text('''
import importlib.util, json, pathlib, sys, time
spec = importlib.util.spec_from_file_location('native', sys.argv[1])
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
class Bridge:
    def __init__(self, atspi, glib, session_id, process_id, window_scope=None):
        self.session_id, self.process_id, self.window_scope = session_id, process_id, window_scope
    def perform(self, action):
        with native._keymap_helper().UnicodeKeymap('éï', native.os.environ) as mapping:
            pathlib.Path(sys.argv[2]).write_text(json.dumps(mapping.codes))
            time.sleep(30)
native._AtspiBridge = Bridge
native._bridge_main()
''')
    child = subprocess.Popen(['/usr/bin/python3', str(script), str(native_source), str(ready)],
                             env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        child.stdin.write((json.dumps({'schema': 'robit.omni.native-ui.v1', 'id': 'terminate-held-map',
            'session_id': 'private-test', 'process_id': None, 'op': 'perform', 'action': {}})+'\n').encode())
        child.stdin.flush()
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic() < deadline and child.poll() is None:
            time.sleep(.01)
        assert ready.exists(), 'Bridge never held its temporary mapping'
        assert keyboard_mapping(env) != before
        child.terminate()
        child.wait(3)
        assert child.returncode == 0
        assert keyboard_mapping(env) == before
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(3)
