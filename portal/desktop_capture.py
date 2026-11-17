"""One explicitly requested desktop frame from the current compositor.

GNOME Wayland is captured through a short-lived, caller-owned Mutter/PipeWire
session. X11 uses Pillow's XCB reader. Neither path changes desktop permissions,
opens a physical camera, borrows another application's stream, or dispatches input.
"""
from __future__ import annotations

import io
import sys
from collections.abc import Mapping
from typing import Any

MAX_PNG_BYTES = 32 * 1024 * 1024


def capture_wayland(*, width: int = 0, height: int = 0, timeout_s: float = 8) -> bytes:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gio, GLib, Gst

    Gst.init(None)
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    loop = GLib.MainLoop()
    holder: dict[str, Any] = {"frame": None, "error": None, "pipeline": None}
    destination = "org.gnome.Mutter.ScreenCast"

    def call(path: str, interface: str, method: str, arguments: Any = None):
        return bus.call_sync(destination, path, interface, method, arguments, None,
            Gio.DBusCallFlags.NONE, 2500, None).unpack()

    session = call("/org/gnome/Mutter/ScreenCast", destination, "CreateSession", GLib.Variant("(a{sv})", ({},)))[0]
    subscription = 0
    timer = 0
    try:
        options = {"cursor-mode": GLib.Variant("u", 1)}
        if width > 0 and height > 0:
            if max(width, height) > 16384:
                raise ValueError("desktop geometry exceeds the capture bound")
            stream = call(session, destination + ".Session", "RecordArea", GLib.Variant("(iiiia{sv})", (0, 0, width, height, options)))[0]
        else:
            stream = call(session, destination + ".Session", "RecordMonitor", GLib.Variant("(sa{sv})", ("", options)))[0]

        def available(_bus, _sender, _path, _interface, _signal, parameters, _data):
            try:
                identity = int(parameters.unpack()[0])
                if not 0 < identity < 2**32:
                    raise ValueError("invalid compositor stream identity")
                pipeline = Gst.parse_launch(f"pipewiresrc path={identity} num-buffers=1 ! videoconvert ! pngenc snapshot=true ! appsink name=frame emit-signals=true sync=false max-buffers=1 drop=true")
                holder["pipeline"] = pipeline

                def sample(sink):
                    value = sink.emit("pull-sample")
                    buffer = value.get_buffer()
                    size = buffer.get_size()
                    if not 33 <= size <= MAX_PNG_BYTES:
                        holder["error"] = "desktop frame exceeds the capture bound"
                    else:
                        holder["frame"] = buffer.extract_dup(0, size)
                    loop.quit()
                    return Gst.FlowReturn.OK

                def message(_bus, event):
                    if event.type == Gst.MessageType.ERROR:
                        holder["error"] = "compositor stream could not produce a frame"
                        loop.quit()

                pipeline.get_by_name("frame").connect("new-sample", sample)
                pipeline.get_bus().add_signal_watch()
                pipeline.get_bus().connect("message", message)
                pipeline.set_state(Gst.State.PLAYING)
            except Exception:
                holder["error"] = "compositor capture setup failed"
                loop.quit()

        subscription = bus.signal_subscribe(destination, destination + ".Stream", "PipeWireStreamAdded", stream,
            None, Gio.DBusSignalFlags.NONE, available, None)
        def expired():
            holder["error"] = "desktop capture timed out"
            loop.quit()
            return False
        timer = GLib.timeout_add(max(100, int(timeout_s * 1000)), expired)
        call(session, destination + ".Session", "Start")
        loop.run()
        if not isinstance(holder["frame"], bytes):
            raise ValueError(holder["error"] or "no desktop frame was received")
        return holder["frame"]
    finally:
        if holder["pipeline"] is not None:
            holder["pipeline"].set_state(Gst.State.NULL)
            holder["pipeline"].get_bus().remove_signal_watch()
        if subscription:
            bus.signal_unsubscribe(subscription)
        if timer and holder["error"] != "desktop capture timed out":
            GLib.source_remove(timer)
        try:
            call(session, destination + ".Session", "Stop")
        except Exception:
            pass  # Caller disconnect also releases only this owned session.


def capture(width: int = 0, height: int = 0, *, environment: Mapping[str, str] | None = None) -> bytes:
    import os

    from PIL import Image, ImageGrab
    current = os.environ if environment is None else environment
    if current.get("WAYLAND_DISPLAY"):
        raw = capture_wayland(width=width, height=height)
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return raw
    image = ImageGrab.grab(xdisplay=current.get("DISPLAY"))
    output = io.BytesIO()
    image.save(output, format="PNG")
    if output.tell() > MAX_PNG_BYTES:
        raise ValueError("desktop frame exceeds the capture bound")
    return output.getvalue()


def main() -> int:
    try:
        width, height = (int(value) for value in sys.argv[1:3]) if len(sys.argv) == 3 else (0, 0)
        sys.stdout.buffer.write(capture(width, height))
        return 0
    except Exception:
        print("Current desktop capture backend unavailable", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
