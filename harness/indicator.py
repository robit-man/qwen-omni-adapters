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
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

MAX_VISIBLE_TASKS = 8
MAX_VISIBLE_STEPS = 8

STATUS_MARKS = {
    "pending": "○",
    "running": "●",
    "completed": "✓",
    "blocked": "!",
    "cancelled": "×",
}

# Glyph per state. Text rather than themed icons: these render identically on
# every theme, need no icon cache, and read at a glance.
LABELS: dict[str, str] = {
    "starting": "Omni …",
    "listening": "Omni ●",
    "hearing": "Omni ◉",
    "thinking": "Omni ◍",
    "speaking": "Omni ▶",
    "muted": "Omni ○",
    "offline": "Omni ✕",
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


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def task_views(tasks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the bounded, deterministic task presentation used by the menu."""

    normalized = [task for task in tasks if str(task.get("task_id") or "")]
    normalized.sort(
        key=lambda task: (
            str(task.get("status") or "") not in {"pending", "running"},
            -float(task.get("updated_at") or 0),
        )
    )
    visible = normalized[:MAX_VISIBLE_TASKS]
    views: list[dict[str, Any]] = []
    for task in visible:
        status = str(task.get("status") or "unknown")
        objective = _short(task.get("objective"), 68) or "Untitled task"
        progress = task.get("progress")
        steps = (
            [_short(item, 110) for item in progress[-MAX_VISIBLE_STEPS:]]
            if isinstance(progress, list)
            else []
        )
        tools = task.get("tools_used")
        views.append(
            {
                "task_id": str(task["task_id"]),
                "status": status,
                "label": f"{STATUS_MARKS.get(status, '?')} {objective}",
                "current_stage": _short(task.get("current_stage"), 110),
                "steps": steps,
                "tools": [str(item) for item in tools] if isinstance(tools, list) else [],
                "result": _short(task.get("result"), 140),
                "error": _short(task.get("error"), 140),
            }
        )
    return views


class NullIndicator:
    """What the harness uses when there is no desktop to show anything on."""

    def set_state(self, state: str, detail: str = "") -> None:  # noqa: D102
        logger.debug("state=%s detail=%s", state, detail)

    def run(self) -> None:  # noqa: D102
        pass

    def stop(self) -> None:  # noqa: D102
        pass


def build_indicator(
    *,
    on_mute: Callable[[bool], None],
    on_quit: Callable[[], None],
    on_reload: Callable[[], None] | None = None,
    on_tools: Callable[[bool], None] | None = None,
    on_reasoning: Callable[[bool], None] | None = None,
    on_camera: Callable[[bool], None] | None = None,
    tools_enabled: bool = True,
    reasoning_enabled: bool = False,
    camera_enabled: bool = True,
    endpoint: Callable[[], str] | None = None,
    tasks: Callable[[], list[Mapping[str, Any]]] | None = None,
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
            # No icon: the label carries the state, and a microphone glyph
            # sitting permanently in the top bar reads as a warning rather
            # than a status.
            self._indicator = AppIndicator.Indicator.new(
                "omni-call-harness",
                "",
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self._muted = False
            self._gtk = Gtk
            self._expanded_tasks: set[str] = set()
            self._task_widgets: list[Any] = []
            self._task_signature: object = None
            self._tasks = tasks

            menu = Gtk.Menu()
            self._menu = menu
            self._status_item = Gtk.MenuItem(label="Starting…")
            self._status_item.set_sensitive(False)
            menu.append(self._status_item)
            menu.append(Gtk.SeparatorMenuItem())

            self._mute_item = Gtk.CheckMenuItem(label="Mute microphone")
            self._mute_item.connect("toggled", self._toggled)
            menu.append(self._mute_item)

            self._tools_item = Gtk.CheckMenuItem(label="Use tools")
            self._tools_item.set_active(tools_enabled)
            if on_tools is not None:
                self._tools_item.connect(
                    "toggled", lambda item: on_tools(bool(item.get_active()))
                )
            else:
                self._tools_item.set_sensitive(False)
            menu.append(self._tools_item)

            self._reasoning_item = Gtk.CheckMenuItem(label="Show reasoning (slower)")
            self._reasoning_item.set_active(reasoning_enabled)
            if on_reasoning is not None:
                self._reasoning_item.connect(
                    "toggled", lambda item: on_reasoning(bool(item.get_active()))
                )
            else:
                self._reasoning_item.set_sensitive(False)
            menu.append(self._reasoning_item)

            self._camera_item = Gtk.CheckMenuItem(label="Use cameras")
            self._camera_item.set_active(camera_enabled)
            if on_camera is not None:
                self._camera_item.connect(
                    "toggled", lambda item: on_camera(bool(item.get_active()))
                )
            else:
                self._camera_item.set_sensitive(False)
            menu.append(self._camera_item)

            menu.append(Gtk.SeparatorMenuItem())
            self._endpoint_item = Gtk.MenuItem(label="Copy public link")
            self._endpoint_item.connect("activate", lambda *_: self._copy_endpoint())
            self._endpoint_item.set_sensitive(endpoint is not None)
            menu.append(self._endpoint_item)
            self._endpoint = endpoint

            menu.append(Gtk.SeparatorMenuItem())
            self._reload_item = Gtk.MenuItem(label="Reload voice service")
            self._reload_item.connect("activate", lambda *_: self._reload())
            self._reload_item.set_sensitive(on_reload is not None)
            menu.append(self._reload_item)

            menu.append(Gtk.SeparatorMenuItem())
            self._tasks_header = Gtk.MenuItem(label="Tasks")
            self._tasks_header.set_sensitive(False)
            menu.append(self._tasks_header)

            self._quit_separator = Gtk.SeparatorMenuItem()
            menu.append(self._quit_separator)
            quit_item = Gtk.MenuItem(label="Quit")
            quit_item.connect("activate", lambda *_: self._quit())
            menu.append(quit_item)

            menu.show_all()
            self._indicator.set_menu(menu)
            self._indicator.set_label(LABELS["starting"], "Omni")
            self._refresh_tasks()
            if tasks is not None:
                GLib.timeout_add_seconds(1, self._refresh_tasks)

        def _toggled(self, item) -> None:
            self._muted = bool(item.get_active())
            on_mute(self._muted)
            self.set_state("muted" if self._muted else "listening")

        def _copy_endpoint(self) -> None:
            """Put the tunnel's URL, key included, on the clipboard."""

            if self._endpoint is None:
                return
            url = self._endpoint()
            if not url:
                self._status_item.set_label("No public link yet")
                return
            try:
                from gi.repository import Gdk

                clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
                clipboard.set_text(url, -1)
                clipboard.store()
                self._status_item.set_label("Public link copied")
            except Exception as error:  # noqa: BLE001 - clipboard is a nicety
                logger.info("public link: %s (clipboard unavailable: %s)", url, error)

        def _quit(self) -> None:
            on_quit()
            self.stop()

        def _reload(self) -> None:
            if on_reload is None:
                return
            self._reload_item.set_label("Reloading…")
            self._reload_item.set_sensitive(False)
            on_reload()
            self.stop()

        def _toggle_task(self, task_id: str) -> None:
            if task_id in self._expanded_tasks:
                self._expanded_tasks.remove(task_id)
            else:
                self._expanded_tasks.add(task_id)
            self._task_signature = None
            self._refresh_tasks()

        def _detail_item(self, text: str, *, spinning: bool = False):
            item = Gtk.MenuItem()
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=7)
            row.set_margin_start(18)
            if spinning:
                marker = Gtk.Spinner()
                marker.start()
            else:
                marker = Gtk.Label(label="✓")
            row.pack_start(marker, False, False, 0)
            label = Gtk.Label(label=text)
            label.set_xalign(0.0)
            label.set_line_wrap(True)
            label.set_max_width_chars(72)
            row.pack_start(label, True, True, 0)
            item.add(row)
            item.set_sensitive(False)
            return item

        def _refresh_tasks(self) -> bool:
            try:
                views = task_views(self._tasks()) if self._tasks is not None else []
            except Exception as error:  # noqa: BLE001 - a menu must not stop listening
                logger.warning("could not refresh indicator tasks: %s", error)
                views = []
            signature = (
                tuple(
                    (
                        view["task_id"],
                        view["status"],
                        view["label"],
                        view["current_stage"],
                        tuple(view["steps"]),
                        tuple(view["tools"]),
                        view["result"],
                        view["error"],
                    )
                    for view in views
                ),
                tuple(sorted(self._expanded_tasks)),
            )
            if signature == self._task_signature:
                return True
            self._task_signature = signature
            visible_ids = {view["task_id"] for view in views}
            self._expanded_tasks.intersection_update(visible_ids)
            for widget in self._task_widgets:
                self._menu.remove(widget)
            self._task_widgets.clear()
            self._tasks_header.set_label(
                f"Tasks — {len(views)}" if views else "Tasks — none"
            )
            position = self._menu.get_children().index(self._quit_separator)
            for view in views:
                task_id = view["task_id"]
                item = Gtk.CheckMenuItem(label=view["label"])
                item.set_active(task_id in self._expanded_tasks)
                item.connect("activate", lambda _item, value=task_id: self._toggle_task(value))
                self._menu.insert(item, position)
                self._task_widgets.append(item)
                position += 1
                if task_id not in self._expanded_tasks:
                    continue
                if view["status"] in {"pending", "running"}:
                    stage = view["current_stage"] or "Waiting for the next worker step"
                    detail = self._detail_item(stage, spinning=True)
                    self._menu.insert(detail, position)
                    self._task_widgets.append(detail)
                    position += 1
                tools = ", ".join(view["tools"]) or "none yet"
                detail = self._detail_item(f"Tools used: {tools}")
                self._menu.insert(detail, position)
                self._task_widgets.append(detail)
                position += 1
                for step in view["steps"]:
                    detail = self._detail_item(step)
                    self._menu.insert(detail, position)
                    self._task_widgets.append(detail)
                    position += 1
                terminal = view["error"] or view["result"]
                if terminal:
                    detail = self._detail_item(terminal)
                    self._menu.insert(detail, position)
                    self._task_widgets.append(detail)
                    position += 1
            self._menu.show_all()
            return True

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
