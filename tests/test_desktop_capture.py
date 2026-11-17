from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from portal.desktop_capture import capture_wayland


def _compositor(monkeypatch, *, delivers_frame=True):
    output = io.BytesIO()
    Image.new("RGB", (48, 32), "orange").save(output, format="PNG")
    png = output.getvalue()
    calls, cleaned = [], []
    class Value:
        def __init__(self, *args):
            self.value = args[-1]
        def unpack(self):
            return self.value
    class Loop:
        def run(self):
            if not delivers_frame:
                timer[0]()
        def quit(self):
            pass
    timer = []
    glib = SimpleNamespace(MainLoop=Loop, Variant=Value,
        timeout_add=lambda _delay, callback: timer.append(callback) or 7,
        source_remove=lambda identity: cleaned.append(("timer", identity)))
    class Bus:
        notification = None
        def call_sync(self, _destination, path, _interface, method, arguments, *_args):
            calls.append((path, method, arguments.unpack() if arguments else None))
            if method == "CreateSession":
                return Value(("/owned-session",))
            if method == "RecordArea":
                return Value(("/owned-stream",))
            if method == "Start" and delivers_frame:
                self.notification(None, None, None, None, None, Value((42,)), None)
            return Value(())
        def signal_subscribe(self, *_args):
            self.notification = _args[-2]
            return 9
        def signal_unsubscribe(self, identity):
            cleaned.append(("signal", identity))
    bus = Bus()
    gio = SimpleNamespace(bus_get_sync=lambda *_args: bus, BusType=SimpleNamespace(SESSION=1),
        DBusCallFlags=SimpleNamespace(NONE=0), DBusSignalFlags=SimpleNamespace(NONE=0))
    class Buffer:
        def get_size(self):
            return len(png)
        def extract_dup(self, _offset, _size):
            return png
    class Sink:
        callback = None
        def emit(self, _event):
            return SimpleNamespace(get_buffer=lambda: Buffer())
        def connect(self, _event, callback):
            self.callback = callback
    sink = Sink()
    class Pipeline:
        def get_by_name(self, _name):
            return sink
        def get_bus(self):
            return self
        def add_signal_watch(self):
            pass
        def remove_signal_watch(self):
            cleaned.append(("watch", True))
        def connect(self, *_args):
            pass
        def set_state(self, state):
            if state == "playing":
                sink.callback(sink)
            else:
                cleaned.append(("pipeline", state))
    gst = SimpleNamespace(init=lambda _args: None, parse_launch=lambda _pipeline: Pipeline(),
        State=SimpleNamespace(PLAYING="playing", NULL="null"), FlowReturn=SimpleNamespace(OK=0), MessageType=SimpleNamespace(ERROR="error"))
    gi = SimpleNamespace(require_version=lambda *_args: None)
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", SimpleNamespace(Gio=gio, GLib=glib, Gst=gst))
    return png, calls, cleaned


def test_wayland_frame_uses_owned_stream_and_releases_every_resource(monkeypatch):
    expected, calls, cleaned = _compositor(monkeypatch)
    assert capture_wayland(width=48, height=32) == expected
    assert [method for _path, method, _args in calls] == ["CreateSession", "RecordArea", "Start", "Stop"]
    assert calls[-1][0] == "/owned-session"
    assert ("signal", 9) in cleaned and ("timer", 7) in cleaned and ("pipeline", "null") in cleaned


def test_wayland_timeout_stops_own_session_without_returning_placeholder_frame(monkeypatch):
    _expected, calls, cleaned = _compositor(monkeypatch, delivers_frame=False)
    with pytest.raises(ValueError, match="timed out"):
        capture_wayland(width=48, height=32)
    assert calls[-1][:2] == ("/owned-session", "Stop")
    assert ("signal", 9) in cleaned
