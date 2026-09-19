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

    def _snapshot(self) -> dict[str, Any]:
        if not shutil.which("xdotool"):
            raise GuiAutomationError("xdotool is not installed")
        if not shutil.which("gnome-screenshot"):
            raise GuiAutomationError("gnome-screenshot is not installed")
        geometry = self._run(["xdotool", "getdisplaygeometry"]).split()
        width = int(geometry[0]) if len(geometry) >= 2 else 0
        height = int(geometry[1]) if len(geometry) >= 2 else 0
        try:
            active_window = self._run(["xdotool", "getactivewindow"])
            active_title = self._run(
                ["xdotool", "getwindowname", active_window]
            )
        except GuiAutomationError:
            active_window, active_title = "", ""
        with tempfile.NamedTemporaryFile(
            prefix="omni-desktop-", suffix=".png", delete=False
        ) as temporary:
            path = Path(temporary.name)
        try:
            self._run(["gnome-screenshot", "-f", str(path)])
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        finally:
            path.unlink(missing_ok=True)
        return {
            "rendered": True,
            "desktop_visible_to_user": True,
            "display": {"width": width, "height": height},
            "active_window": {"id": active_window, "title": active_title[:500]},
            "screenshot": {
                "mime_type": "image/png",
                "encoding": "base64",
                "data": encoded,
            },
        }

    def act(self, _session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip().lower()
        if action not in {"snapshot", "click", "type", "key", "hotkey", "scroll"}:
            raise GuiAutomationError(
                "action must be snapshot, click, type, key, hotkey, or scroll"
            )
        if action == "click":
            x = self._integer(arguments.get("x"), "x", 0, 16_384)
            y = self._integer(arguments.get("y"), "y", 0, 16_384)
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
        return self._snapshot()
