"""Politely request close of an exact owned X11 window without destroying it.

The isolated CPU helper inherits the recovered desktop environment so Xlib uses
that display's authentication. A successful request is not a cleanup receipt:
callers must separately observe their owned window disappear.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class _MessageData(ctypes.Union):
    _fields_ = [("b", ctypes.c_char * 20), ("s", ctypes.c_short * 10), ("l", ctypes.c_long * 5)]


class _ClientMessage(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
               ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
               ("format", ctypes.c_int), ("data", _MessageData)]


class _Event(ctypes.Union):
    _fields_ = [("client", _ClientMessage), ("pad", ctypes.c_long * 24)]


def _valid_identity(window_id: str, pid: str, start: str) -> bool:
    return (isinstance(window_id, str) and window_id.isascii() and window_id.isdecimal()
            and 0 < int(window_id) < 2**32 and isinstance(pid, str) and pid.isascii() and pid.isdecimal()
            and 0 < int(pid) < 2**31 and isinstance(start, str) and start.isascii() and start.isdecimal())


def _process_start(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""


def _identity_matches(window_id: str, pid: str, start: str) -> bool:
    if _process_start(pid) != start:
        return False
    result = subprocess.run(["xdotool", "getwindowpid", window_id], capture_output=True,
                            text=True, check=False, timeout=1)  # noqa: S603
    return result.returncode == 0 and result.stdout.strip() == pid and _process_start(pid) == start


def _xlib() -> Any:
    library = ctypes.CDLL("libX11.so.6")
    library.XOpenDisplay.argtypes, library.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
    library.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    library.XInternAtom.restype = ctypes.c_ulong
    library.XGetWMProtocols.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_int)]
    library.XGetWMProtocols.restype = ctypes.c_int
    library.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long,
                                  ctypes.POINTER(_Event)]
    library.XSendEvent.restype = ctypes.c_int
    library.XFlush.argtypes, library.XCloseDisplay.argtypes = [ctypes.c_void_p], [ctypes.c_void_p]
    library.XFree.argtypes = [ctypes.c_void_p]
    return library


def _send_request(window_id: str, pid: str, start: str) -> bool:
    if not _valid_identity(window_id, pid, start) or not _identity_matches(window_id, pid, start):
        return False
    library = _xlib()
    display = library.XOpenDisplay(None)
    if not display:
        return False
    protocols = ctypes.POINTER(ctypes.c_ulong)()
    try:
        protocol_atom = library.XInternAtom(display, b"WM_PROTOCOLS", True)
        delete_atom = library.XInternAtom(display, b"WM_DELETE_WINDOW", True)
        count = ctypes.c_int()
        if (not protocol_atom or not delete_atom
                or not library.XGetWMProtocols(display, int(window_id), ctypes.byref(protocols), ctypes.byref(count))
                or not protocols or not 0 < count.value <= 64
                or delete_atom not in [protocols[index] for index in range(count.value)]
                or not _identity_matches(window_id, pid, start)):
            return False
        event = _Event()
        event.client.type = 33  # ClientMessage, never DestroyWindow or KillClient.
        event.client.send_event = True
        event.client.display, event.client.window = display, int(window_id)
        event.client.message_type, event.client.format = protocol_atom, 32
        event.client.data.l[0], event.client.data.l[1] = delete_atom, 0  # CurrentTime.
        sent = library.XSendEvent(display, int(window_id), False, 0, ctypes.byref(event))
        library.XFlush(display)
        return bool(sent)
    finally:
        if protocols:
            library.XFree(protocols)
        library.XCloseDisplay(display)


def request_owned_window_close(window_id: str, pid: str, start: str, environment: Mapping[str, str]) -> bool:
    """Request WM_DELETE_WINDOW; the caller remains responsible for observing closure."""
    if not _valid_identity(window_id, pid, start) or not environment.get("DISPLAY"):
        return False
    try:
        result = subprocess.run([sys.executable, "-I", "-u", str(Path(__file__).resolve()),
                                 "--request-close", window_id, pid, start], env=dict(environment),
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                check=False, timeout=3)  # noqa: S603
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main(arguments: list[str]) -> int:
    if len(arguments) != 4 or arguments[0] != "--request-close":
        return 2
    try:
        return 0 if _send_request(*arguments[1:]) else 1
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
