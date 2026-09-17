"""Top-bar indicator for the call harness.

An always-listening microphone should say so. This puts the harness's state in
the GNOME top bar, where it is visible without opening anything, and gives a
menu to mute, unmute, or quit -- so consent is one click away rather than a
terminal command.

AppIndicator is used because Ubuntu ships the extension that renders it; when
it is unavailable the harness still runs, silently, rather than refusing to
start over a status icon.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Glyph per state. Text rather than themed icons: these render identically on
# every theme, need no icon cache, and read at a glance.
LABELS: dict[str, str] = {
    "starting": "◌ Omni",
    "listening": "● Omni",
    "hearing": "◉ Hearing",
    "thinking": "◍ Thinking",
    "speaking": "▶ Speaking",
    "muted": "○ Muted",
    "offline": "✕ Offline",
}

TOOLTIPS: dict[str, str] = {
    "starting": "Connecting to the omni adapter",
    "listening": "Listening for speech",
    "hearing": "Hearing you speak",
    "thinking": "Working out a reply",
    "speaking": "Speaking",
    "muted": "Microphone muted",
    "offline": "The omni adapter is unreachable",
}


class NullIndicator:
    """What the harness uses when there is no desktop to show anything on."""

    def set_state(self, state: str, detail: str = "") -> None:  # noqa: D102
        logger.debug("state=%s detail=%s", state, detail)

    def run(self) -> None:  # noqa: D102
        pass

    def stop(self) -> None:  # noqa: D102
        pass


def build_indicator(
    *, on_mute: Callable[[bool], None], on_quit: Callable[[], None]
):
    """Return a top-bar indicator, or a no-op one if the desktop cannot host it."""

    try:
        import gi

        gi.require_version("Gtk", "3.0")
        try:
            gi.require_version("AyatanaAppIndicator3", "0.1")
            from gi.repository import AyatanaAppIndicator3 as AppIndicator
        except (ValueError, ImportError):
            gi.require_version("AppIndicator3", "0.1")
            from gi.repository import AppIndicator3 as AppIndicator
        from gi.repository import GLib, Gtk
    except Exception as error:  # noqa: BLE001 - a missing tray is not fatal
        logger.info("no top-bar indicator available (%s); running headless", error)
        return NullIndicator()

    class GtkIndicator:
        def __init__(self) -> None:
            self._indicator = AppIndicator.Indicator.new(
                "omni-call-harness",
                "audio-input-microphone",
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self._muted = False

            menu = Gtk.Menu()
            self._status_item = Gtk.MenuItem(label="Starting…")
            self._status_item.set_sensitive(False)
            menu.append(self._status_item)
            menu.append(Gtk.SeparatorMenuItem())

            self._mute_item = Gtk.CheckMenuItem(label="Mute microphone")
            self._mute_item.connect("toggled", self._toggled)
            menu.append(self._mute_item)

            quit_item = Gtk.MenuItem(label="Quit")
            quit_item.connect("activate", lambda *_: self._quit())
            menu.append(quit_item)

            menu.show_all()
            self._indicator.set_menu(menu)
            self._indicator.set_label(LABELS["starting"], "Omni")
            self._gtk = Gtk

        def _toggled(self, item) -> None:
            self._muted = bool(item.get_active())
            on_mute(self._muted)
            self.set_state("muted" if self._muted else "listening")

        def _quit(self) -> None:
            on_quit()
            self.stop()

        def set_state(self, state: str, detail: str = "") -> None:
            if self._muted and state not in {"muted", "offline"}:
                state = "muted"
            label = LABELS.get(state, LABELS["listening"])
            tooltip = TOOLTIPS.get(state, state)
            summary = f"{tooltip}: {detail}" if detail else tooltip

            def apply() -> bool:
                self._indicator.set_label(label, "Omni")
                self._status_item.set_label(summary[:80])
                return False

            # The loop runs on the main thread; the call loop does not.
            GLib.idle_add(apply)

        def run(self) -> None:
            self._gtk.main()

        def stop(self) -> None:
            GLib.idle_add(self._gtk.main_quit)

    return GtkIndicator()


class ThreadedIndicator:
    """Runs the desktop loop on the main thread with the call loop beside it.

    GTK insists on owning the main thread, and the call loop blocks on the
    microphone, so the two cannot share one. This keeps GTK where it demands to
    be and gives the conversation its own thread.
    """

    def __init__(self, indicator, worker: Callable[[], None]) -> None:
        self._indicator = indicator
        self._worker = worker
        self._thread: threading.Thread | None = None

    def run(self) -> None:
        self._thread = threading.Thread(
            target=self._worker, name="omni-call-loop", daemon=True
        )
        self._thread.start()
        self._indicator.run()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
