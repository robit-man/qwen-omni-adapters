"""Protocol errors on a private CPU X server must not become ownership evidence."""
from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def private_display():
    binary = shutil.which("Xvfb")
    if not binary:
        pytest.skip("Private CPU Xvfb is not installed")
    read, write = os.pipe()
    server = subprocess.Popen(
        [binary, "-displayfd", str(write), "-screen", "0", "64x64x24", "-nolisten", "tcp"],
        pass_fds=(write,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.close(write)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(read, selectors.EVENT_READ)
            assert selector.select(3), "Private X server did not become ready"
            number = os.read(read, 64).decode().strip()
        assert number.isdecimal()
        env = {**os.environ, "DISPLAY": ":" + number}
        env.pop("XAUTHORITY", None)
        yield env
    finally:
        os.close(read)
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=1)


PREFIX = """
import ctypes, json, time
from runtime import owned_flatpak_firefox as owned
x = owned._XDisplay()
lib = x.lib
lib.XCreateSimpleWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
    ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulong]
lib.XCreateSimpleWindow.restype = ctypes.c_ulong
lib.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
def create():
    result = lib.XCreateSimpleWindow(x.display, x.root, 0, 0, 8, 8, 0, 0, 0)
    lib.XSync(x.display, False)
    return result
def reject(fn):
    try:
        fn()
    except owned.OwnershipUnavailable as error:
        return error.code
    raise AssertionError('incomplete operation returned ownership evidence')
"""


def run_child(env, body):
    return subprocess.run(
        [sys.executable, "-c", PREFIX + body], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=5,
    )


def success(result):
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_healthy_complete_tree_and_real_create_events(private_display):
    value = success(run_child(private_display, """
tree = x.snapshot(time.monotonic() + 1)
x.arm(tree)
window = create()
events = x.events()
after = x.snapshot(time.monotonic() + 1)
print(json.dumps({'complete': after['complete'], 'window_in_tree': str(window) in after['windows'],
    'real_create': any(e['type'] == 16 and e['window_id'] == str(window) and not e['synthetic'] for e in events),
    'selected': str(window) in x.selected, 'poisoned': x._poisoned}))
"""))
    assert value == {"complete": True, "window_in_tree": True, "real_create": True,
                     "selected": True, "poisoned": False}


def test_maximum_complete_tree_arms_without_per_window_syncs(private_display):
    value = success(run_child(private_display, """
for _ in range(owned.MAX_WINDOWS - 1):
    lib.XCreateSimpleWindow(x.display, x.root, 0, 0, 1, 1, 0, 0, 0)
lib.XSync(x.display, False)
tree = x.snapshot(time.monotonic() + 3)
sync = lib.XSync
calls = []
def counted(*args):
    calls.append(1)
    return sync(*args)
lib.XSync = counted
x.arm(tree)
print(json.dumps({'nodes': len(tree['windows']), 'confirmed': len(x.selected),
    'armed': x.armed, 'syncs': len(calls)}))
"""))
    assert value["nodes"] == value["confirmed"] == 4096
    assert value["armed"] is True
    assert value["syncs"] <= 4


def test_window_destroyed_between_tree_and_attributes_is_typed_not_fatal(private_display):
    value = success(run_child(private_display, """
window = create()
original = lib.XGetWindowAttributes
def raced(display, xid, attrs):
    if xid == window:
        lib.XDestroyWindow(x.display, window)
        lib.XSync(x.display, False)
    return original(display, xid, attrs)
lib.XGetWindowAttributes = raced
code = reject(lambda: x.snapshot(time.monotonic() + 1))
again = reject(lambda: x.snapshot(time.monotonic() + 1))
print(json.dumps({'code': code, 'again': again, 'alive': True}))
"""))
    assert value == {"code": "flatpak_x11_protocol_error",
                     "again": "flatpak_x11_connection_unavailable", "alive": True}


@pytest.mark.parametrize("operation", ["select", "arm"])
def test_asynchronous_selection_error_cannot_confirm_subscription(private_display, operation):
    value = success(run_child(private_display, f"""
window = create()
tree = x.snapshot(time.monotonic() + 1)
lib.XDestroyWindow(x.display, window)
lib.XSync(x.display, False)
code = reject(lambda: x.select(str(window)) if {operation!r} == 'select' else x.arm(tree))
again = reject(x.events)
print(json.dumps({{'code': code, 'again': again, 'selected': str(window) in x.selected, 'armed': x.armed}}))
"""))
    assert value == {"code": "flatpak_x11_protocol_error",
                     "again": "flatpak_x11_connection_unavailable", "selected": False, "armed": False}


def test_destroyed_create_event_poison_prevents_later_complete_witness(private_display):
    value = success(run_child(private_display, """
x.arm(x.snapshot(time.monotonic() + 1))
window = create()
lib.XDestroyWindow(x.display, window)
lib.XSync(x.display, False)
code = reject(x.events)
again = reject(lambda: x.snapshot(time.monotonic() + 1))
print(json.dumps({'code': code, 'again': again, 'selected': str(window) in x.selected}))
"""))
    assert value == {"code": "flatpak_x11_protocol_error",
                     "again": "flatpak_x11_connection_unavailable", "selected": False}


def test_foreign_connection_custom_handler_delegated_and_restored(private_display):
    value = success(run_child(private_display, """
seen = []
@owned._X_ERROR_HANDLER
def previous(display, event):
    seen.append((display, event.contents.error_code))
    return 0
original = lib.XSetErrorHandler(previous)
foreign = owned._XDisplay()
with x._errors():
    attrs = owned._Attributes()
    lib.XGetWindowAttributes(foreign.display, 0, ctypes.byref(attrs))
    lib.XSync(foreign.display, False)
restored = lib.XSetErrorHandler(previous) == ctypes.cast(previous, ctypes.c_void_p).value
lib.XGetWindowAttributes(foreign.display, 0, ctypes.byref(attrs))
lib.XSync(foreign.display, False)
lib.XSetErrorHandler(original)
print(json.dumps({'delegated': len(seen), 'foreign_only': all(d == foreign.display and c == 3 for d,c in seen),
    'restored': restored, 'owned_poisoned': x._poisoned}))
"""))
    assert value == {"delegated": 2, "foreign_only": True, "restored": True, "owned_poisoned": False}


@pytest.mark.parametrize("after_owned_failure", [False, True])
def test_foreign_default_handler_keeps_fatal_behavior(private_display, after_owned_failure):
    result = run_child(private_display, f"""
foreign = owned._XDisplay()
attrs = owned._Attributes()
if {after_owned_failure!r}:
    code = reject(lambda: x.select('0'))
    assert code == 'flatpak_x11_protocol_error'
    print('owned_error_was_typed', flush=True)
    lib.XGetWindowAttributes(foreign.display, 0, ctypes.byref(attrs))
else:
    with x._errors():
        lib.XGetWindowAttributes(foreign.display, 0, ctypes.byref(attrs))
        lib.XSync(foreign.display, False)
print('foreign_error_was_swallowed', flush=True)
""")
    assert result.returncode == 1
    assert "BadWindow" in result.stderr
    assert "foreign_error_was_swallowed" not in result.stdout
    if after_owned_failure:
        assert "owned_error_was_typed" in result.stdout
