"""Owned GTK4 TextView (the toolkit of GNOME Text Editor 4x) with an initial value."""
import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402


def activate(application):
    window = Gtk.ApplicationWindow(application=application, title=sys.argv[1])
    view = Gtk.TextView()
    view.get_buffer().set_text(sys.argv[2])
    window.set_child(view)
    window.present()
    view.grab_focus()


application = Gtk.Application(application_id="org.robit.OmniLiteral" + sys.argv[1][-8:])
application.connect("activate", activate)
application.run([])
