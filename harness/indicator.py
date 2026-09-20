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
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

MAX_VISIBLE_TASKS = 8
MAX_VISIBLE_STEPS = 8
MAX_VISIBLE_ACTIONS = 12

STATUS_MARKS = {
    "pending": "○",
    "running": "●",
    "completed": "✓",
    "blocked": "!",
    "cancelled": "×",
}

STATE_ICONS = {
    "starting": "process-working-symbolic",
    "listening": "audio-input-microphone-symbolic",
    "hearing": "media-record-symbolic",
    "thinking": "process-working-symbolic",
    "speaking": "audio-volume-high-symbolic",
    "muted": "microphone-sensitivity-muted-symbolic",
    "offline": "network-error-symbolic",
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
        tool_names = [str(item) for item in tools] if isinstance(tools, list) else []
        if not tool_names:
            # Older records predate exact tool auditing, but their checkpoint
            # labels still identify concrete calls without guessing.
            for step in steps:
                if step.startswith("Ran shell step"):
                    tool_names.append("shell")
                elif step.startswith("Ran ") and " and retained its result" in step:
                    tool_names.append(step[4:].split(" and retained", 1)[0])
        actions = task.get("actions")
        action_views: list[dict[str, Any]] = []
        if isinstance(actions, list):
            for action in actions[-MAX_VISIBLE_ACTIONS:]:
                if not isinstance(action, Mapping):
                    continue
                if action.get("at"):
                    try:
                        when = datetime.fromtimestamp(
                            float(action["at"])
                        ).astimezone().strftime("%H:%M:%S")
                    except (OSError, TypeError, ValueError):
                        when = "--:--:--"
                else:
                    when = "retained"
                name = _short(action.get("tool"), 40) or "unknown"
                arguments = _short(action.get("arguments"), 180) or "{}"
                outcome = _short(action.get("outcome"), 180)
                label = f"{when}  {name} {arguments}"
                if outcome:
                    label += f" → {outcome}"
                action_views.append(
                    {"label": label, "ok": action.get("ok") is True}
                )
        views.append(
            {
                "task_id": str(task["task_id"]),
                "status": status,
                "label": f"{STATUS_MARKS.get(status, '?')} {objective}",
                "current_stage": _short(task.get("current_stage"), 110),
                "steps": steps,
                "tools": list(dict.fromkeys(tool_names)),
                "actions": action_views,
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
    on_clear_tasks: Callable[[], int] | None = None,
    on_cancel_task: Callable[[str], bool] | None = None,
    on_clear_task: Callable[[str], bool] | None = None,
    on_open_archive: Callable[[], None] | None = None,
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
        from gi.repository import GLib, Gtk, Pango
    except Exception as error:  # noqa: BLE001 - a missing tray is not fatal
        logger.info("no top-bar indicator available (%s); running headless", error)
        return NullIndicator()

    class GtkIndicator:
        def __init__(self) -> None:
            self._indicator = AppIndicator.Indicator.new(
                "omni-call-harness",
                STATE_ICONS["starting"],
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self._muted = False
            self._gtk = Gtk
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

            self._clear_tasks_item = Gtk.MenuItem(label="Clear finished tasks")
            self._clear_tasks_item.connect("activate", lambda *_: self._clear_tasks())
            self._clear_tasks_item.set_sensitive(on_clear_tasks is not None)
            menu.append(self._clear_tasks_item)

            self._archive_item = Gtk.MenuItem(label="Open task archive")
            self._archive_item.connect("activate", lambda *_: self._open_archive())
            self._archive_item.set_sensitive(on_open_archive is not None)
            menu.append(self._archive_item)

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
            self._indicator.set_label("Omni", "Omni")
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

        def _clear_tasks(self) -> None:
            if on_clear_tasks is None:
                return
            count = on_clear_tasks()
            noun = "task" if count == 1 else "tasks"
            self._status_item.set_label(f"Archived {count} finished {noun}")
            self._task_signature = None
            self._refresh_tasks()

        def _open_archive(self) -> None:
            if on_open_archive is not None:
                on_open_archive()

        def _cancel_task(self, task_id: str) -> None:
            if on_cancel_task is None:
                return
            cancelled = on_cancel_task(task_id)
            self._status_item.set_label(
                "Task cancelled" if cancelled else "Task was not found"
            )
            self._task_signature = None
            self._refresh_tasks()

        def _clear_task(self, task_id: str) -> None:
            if on_clear_task is None:
                return
            removed = on_clear_task(task_id)
            self._status_item.set_label(
                "Task cleared" if removed else "Task was not found"
            )
            self._task_signature = None
            self._refresh_tasks()

        def _detail_item(
            self,
            text: str,
            *,
            spinning: bool = False,
            marker_text: str = "✓",
        ):
            # A wrapped Gtk.Label inside a custom box can receive a zero-width
            # allocation in an AppIndicator submenu: the marker remains visible
            # while the actual action becomes a blank line. A native menu label
            # gets a stable allocation and ellipsizes predictably.
            marker = "◌" if spinning else marker_text
            item = Gtk.MenuItem(label=f"{marker}  {text}")
            label = item.get_child()
            if isinstance(label, Gtk.Label):
                label.set_xalign(0.0)
                label.set_max_width_chars(96)
                label.set_ellipsize(Pango.EllipsizeMode.END)
            item.set_tooltip_text(text)
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
                        tuple(
                            (action["label"], action["ok"])
                            for action in view["actions"]
                        ),
                        view["result"],
                        view["error"],
                    )
                    for view in views
                ),
            )
            if signature == self._task_signature:
                return True
            self._task_signature = signature
            for widget in self._task_widgets:
                self._menu.remove(widget)
            self._task_widgets.clear()
            self._tasks_header.set_label(
                f"Tasks — {len(views)}" if views else "Tasks — none"
            )
            position = self._menu.get_children().index(self._quit_separator)
            for view in views:
                # AppIndicator closes its whole menu when an ordinary item is
                # activated. A native submenu stays open while the person
                # inspects stages and closes naturally when they leave it.
                item = Gtk.MenuItem(label=view["label"])
                details = Gtk.Menu()
                item.set_submenu(details)
                self._menu.insert(item, position)
                self._task_widgets.append(item)
                position += 1
                if view["status"] in {"pending", "running"}:
                    stage = view["current_stage"] or "Waiting for the next worker step"
                    detail = self._detail_item(stage, spinning=True)
                    details.append(detail)
                tools = ", ".join(view["tools"]) or "none recorded"
                detail = self._detail_item(f"Tools used: {tools}")
                details.append(detail)
                if view["actions"]:
                    action_header = Gtk.MenuItem(
                        label=f"Tool calls — latest {len(view['actions'])}"
                    )
                    action_menu = Gtk.Menu()
                    action_header.set_submenu(action_menu)
                    details.append(action_header)
                    for action in view["actions"]:
                        detail = self._detail_item(
                            action["label"],
                            marker_text="✓" if action["ok"] else "!",
                        )
                        action_menu.append(detail)
                    action_menu.show_all()
                for step in view["steps"]:
                    detail = self._detail_item(step)
                    details.append(detail)
                terminal = view["error"] or view["result"]
                if terminal:
                    detail = self._detail_item(terminal)
                    details.append(detail)
                details.append(Gtk.SeparatorMenuItem())
                if view["status"] in {"pending", "running"}:
                    cancel_item = Gtk.MenuItem(label="Cancel task")
                    cancel_item.set_sensitive(on_cancel_task is not None)
                    cancel_item.connect(
                        "activate",
                        lambda _item, task_id=view["task_id"]: self._cancel_task(
                            task_id
                        ),
                    )
                    details.append(cancel_item)
                clear_item = Gtk.MenuItem(label="Clear task record")
                clear_item.set_sensitive(on_clear_task is not None)
                clear_item.connect(
                    "activate",
                    lambda _item, task_id=view["task_id"]: self._clear_task(task_id),
                )
                details.append(clear_item)
                details.show_all()
            self._menu.show_all()
            return True

        def set_state(self, state: str, detail: str = "") -> None:
            if self._muted and state not in {"muted", "offline"}:
                state = "muted"
            tooltip = TOOLTIPS.get(state, state)
            summary = f"{tooltip}: {detail}" if detail else tooltip

            def apply() -> bool:
                icon = STATE_ICONS.get(state, STATE_ICONS["listening"])
                try:
                    self._indicator.set_icon_full(icon, tooltip)
                except AttributeError:
                    self._indicator.set_icon(icon)
                self._indicator.set_label("Omni", "Omni")
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
