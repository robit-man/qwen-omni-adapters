"""Owned GTK3 TextView fixture driven by command files in a private control directory."""
import os
import sys

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

control = sys.argv[1]
window = Gtk.Window(title=sys.argv[2])
window.set_default_size(600, 400)
box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
view = Gtk.TextView()
buffer = view.get_buffer()
buffer.set_text("alpha")
# GNOME Text Editor shows its load/save progress bar as an overlay of the view.
overlay = Gtk.Overlay()
overlay.add(view)
box.pack_start(overlay, True, True, 0)
window.add(box)
window.connect("destroy", Gtk.main_quit)
window.show_all()
view.grab_focus()
buffer.place_cursor(buffer.get_end_iter())
bars = []


def progress_bar():
    bar = Gtk.ProgressBar()
    bar.set_fraction(0.3)
    bars.append(bar)
    return bar


def poll():
    for name in sorted(os.listdir(control)):
        if not name.startswith("cmd-"):
            continue
        path = os.path.join(control, name)
        with open(path) as handle:
            command = handle.read().strip()
        os.remove(path)
        if command == "drift":  # Unrelated structure; recipient geometry unchanged.
            bar = progress_bar()
            bar.set_valign(Gtk.Align.START)
            overlay.add_overlay(bar)
            bar.show()
        elif command == "drift_resize":  # Packed bar shrinks the recipient.
            bar = progress_bar()
            box.pack_end(bar, False, False, 0)
            bar.show()
        elif command == "caret0":
            buffer.place_cursor(buffer.get_start_iter())
        text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        caret = buffer.get_iter_at_mark(buffer.get_insert()).get_offset()
        acknowledgement = os.path.join(control, "ack-" + name[4:])
        with open(acknowledgement + ".tmp", "w") as handle:
            handle.write(repr((text, caret, len(bars))))
        os.rename(acknowledgement + ".tmp", acknowledgement)
    return True


GLib.timeout_add(50, poll)
Gtk.main()
