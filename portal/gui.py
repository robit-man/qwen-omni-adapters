"""Direct, screenshot-driven control of the active Linux desktop."""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from portal.desktop import desktop_subprocess_environment
except ModuleNotFoundError:  # Direct script execution from portal/.
    from desktop import desktop_subprocess_environment


class GuiAutomationError(RuntimeError):
    """The desktop could not be observed or controlled."""


_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_+\-]+$")


class GuiAutomation:
    """Use X11-compatible desktop tools, with a fresh screenshot after each action."""

    def __init__(self, *, timeout_s: float = 10.0) -> None:
        self.timeout_s = max(2.0, float(timeout_s))

    def clear(self, _session_id: str) -> None:
        """Desktop state is real user state and is never destroyed with a web session."""

    def _run(self, command: list[str]) -> str:
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=desktop_subprocess_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuiAutomationError(f"Desktop command failed: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:500]
            raise GuiAutomationError(
                f"Desktop command exited {completed.returncode}: {detail}"
            )
        return completed.stdout.strip()

    def _integer(self, value: Any, name: str, minimum: int, maximum: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise GuiAutomationError(f"{name} must be an integer") from exc
        if not minimum <= result <= maximum:
            raise GuiAutomationError(f"{name} must be between {minimum} and {maximum}")
        return result

    def _active_window(self) -> dict[str, Any] | None:
        """Return one stable coordinate frame for the currently focused window."""

        try:
            window_id = self._run(["xdotool", "getactivewindow"])
            title = self._run(["xdotool", "getwindowname", window_id])
            geometry = self._run(
                ["xdotool", "getwindowgeometry", "--shell", window_id]
            )
        except GuiAutomationError:
            return None
        fields: dict[str, int] = {}
        for line in geometry.splitlines():
            key, separator, raw = line.partition("=")
            if not separator or key not in {"X", "Y", "WIDTH", "HEIGHT", "SCREEN"}:
                continue
            try:
                fields[key] = int(raw)
            except ValueError:
                return None
        if fields.get("WIDTH", 0) <= 0 or fields.get("HEIGHT", 0) <= 0:
            return None
        return {
            "id": window_id,
            "title": title[:500],
            "bounds": {
                "x": fields.get("X", 0),
                "y": fields.get("Y", 0),
                "width": fields["WIDTH"],
                "height": fields["HEIGHT"],
                "screen": fields.get("SCREEN", 0),
            },
        }

    @staticmethod
    def _space_name(arguments: dict[str, Any]) -> str:
        name = str(arguments.get("coordinate_space") or "active_window").strip().lower()
        if name not in {"active_window", "screen"}:
            raise GuiAutomationError(
                "coordinate_space must be active_window or screen"
            )
        return name

    def _point(
        self,
        arguments: dict[str, Any],
        x_name: str,
        y_name: str,
        *,
        coordinate_space: str,
        active_window: dict[str, Any] | None,
        display_width: int,
        display_height: int,
    ) -> tuple[int, int]:
        if coordinate_space == "active_window":
            if active_window is None:
                raise GuiAutomationError(
                    "No active window is available for window-relative coordinates; "
                    "take a screen snapshot or use coordinate_space=screen."
                )
            bounds = active_window["bounds"]
            x = self._integer(arguments.get(x_name), x_name, 0, bounds["width"] - 1)
            y = self._integer(arguments.get(y_name), y_name, 0, bounds["height"] - 1)
            return bounds["x"] + x, bounds["y"] + y
        x_max = max(0, display_width - 1) if display_width else 16_384
        y_max = max(0, display_height - 1) if display_height else 16_384
        return (
            self._integer(arguments.get(x_name), x_name, 0, x_max),
            self._integer(arguments.get(y_name), y_name, 0, y_max),
        )

    def _desktop_state(self) -> tuple[int, int, dict[str, Any] | None]:
        if not shutil.which("xdotool"):
            raise GuiAutomationError("xdotool is not installed")
        geometry = self._run(["xdotool", "getdisplaygeometry"]).split()
        width = int(geometry[0]) if len(geometry) >= 2 else 0
        height = int(geometry[1]) if len(geometry) >= 2 else 0
        return width, height, self._active_window()

    def _snapshot(self, coordinate_space: str = "active_window") -> dict[str, Any]:
        if not shutil.which("xdotool"):
            raise GuiAutomationError("xdotool is not installed")
        if not shutil.which("gnome-screenshot"):
            raise GuiAutomationError("gnome-screenshot is not installed")
        width, height, active_window = self._desktop_state()
        window_crop = coordinate_space == "active_window" and active_window is not None
        with tempfile.NamedTemporaryFile(
            prefix="omni-desktop-", suffix=".png", delete=False
        ) as temporary:
            path = Path(temporary.name)
        try:
            command = ["gnome-screenshot"]
            if window_crop:
                command.append("-w")
            command.extend(["-f", str(path)])
            self._run(command)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        finally:
            path.unlink(missing_ok=True)
        if window_crop:
            bounds = active_window["bounds"]
            frame = {
                "name": "active_window",
                "origin_x": bounds["x"],
                "origin_y": bounds["y"],
                "width": bounds["width"],
                "height": bounds["height"],
            }
        else:
            frame = {
                "name": "screen",
                "origin_x": 0,
                "origin_y": 0,
                "width": width,
                "height": height,
            }
        return {
            "rendered": True,
            "desktop_visible_to_user": True,
            "display": {"width": width, "height": height},
            "active_window": active_window or {"id": "", "title": ""},
            "coordinate_space": frame,
            "screenshot": {
                "mime_type": "image/png",
                "encoding": "base64",
                "data": encoded,
            },
        }

    def act(self, _session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip().lower()
        if action not in {
            "snapshot",
            "click",
            "drag",
            "type",
            "key",
            "hotkey",
            "scroll",
        }:
            raise GuiAutomationError(
                "action must be snapshot, click, drag, type, key, hotkey, or scroll"
            )
        coordinate_space = self._space_name(arguments)
        display_width = display_height = 0
        active_window: dict[str, Any] | None = None
        if action in {"click", "drag"}:
            display_width, display_height, active_window = self._desktop_state()
        if action == "click":
            x, y = self._point(
                arguments,
                "x",
                "y",
                coordinate_space=coordinate_space,
                active_window=active_window,
                display_width=display_width,
                display_height=display_height,
            )
            button = self._integer(arguments.get("button", 1), "button", 1, 5)
            self._run(
                [
                    "xdotool",
                    "mousemove",
                    "--sync",
                    str(x),
                    str(y),
                    "click",
                    str(button),
                ]
            )
        elif action == "drag":
            x, y = self._point(
                arguments,
                "x",
                "y",
                coordinate_space=coordinate_space,
                active_window=active_window,
                display_width=display_width,
                display_height=display_height,
            )
            to_x, to_y = self._point(
                arguments,
                "to_x",
                "to_y",
                coordinate_space=coordinate_space,
                active_window=active_window,
                display_width=display_width,
                display_height=display_height,
            )
            button = self._integer(arguments.get("button", 1), "button", 1, 5)
            if x == to_x and y == to_y:
                raise GuiAutomationError("drag start and destination must differ")
            self._run(
                [
                    "xdotool",
                    "mousemove",
                    "--sync",
                    str(x),
                    str(y),
                    "mousedown",
                    str(button),
                    "mousemove",
                    "--sync",
                    "--duration",
                    "600",
                    str(to_x),
                    str(to_y),
                    "mouseup",
                    str(button),
                ]
            )
        elif action == "type":
            text = str(arguments.get("text") or "")
            if len(text) > 8000:
                raise GuiAutomationError("text exceeds 8000 characters")
            self._run(["xdotool", "type", "--clearmodifiers", "--delay", "1", text])
        elif action in {"key", "hotkey"}:
            key = str(arguments.get("key") or "").strip()
            if not key or len(key) > 100 or not _KEY_PATTERN.fullmatch(key):
                raise GuiAutomationError("key must be a valid xdotool key name or chord")
            self._run(["xdotool", "key", "--clearmodifiers", key])
        elif action == "scroll":
            amount = self._integer(arguments.get("amount", 3), "amount", -20, 20)
            button = "5" if amount >= 0 else "4"
            for _ in range(abs(amount)):
                self._run(["xdotool", "click", button])
        wait_ms = self._integer(arguments.get("wait_ms", 250), "wait_ms", 0, 5000)
        if wait_ms:
            time.sleep(wait_ms / 1000.0)
        return self._snapshot(coordinate_space)
