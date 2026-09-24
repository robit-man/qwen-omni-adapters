"""Direct, screenshot-driven control of the active Linux desktop."""

from __future__ import annotations

import base64
import hashlib
import io
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

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
        self._visual_states: dict[str, dict[str, Any]] = {}

    def clear(self, session_id: str) -> None:
        """Forget comparison state without mutating the user's real desktop."""

        self._visual_states.pop(session_id, None)

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
    def _space_name(
        arguments: dict[str, Any], *, observed_space: str = "active_window"
    ) -> str:
        # Pointer coordinates belong to the pixels most recently returned to this
        # session. If the caller omits the frame after a deliberate full-screen
        # snapshot, preserve that screen frame instead of silently reinterpreting
        # the same numbers relative to whichever window currently has focus.
        name = str(
            arguments.get("coordinate_space") or observed_space or "active_window"
        ).strip().lower()
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
            # Capture root-window pixels and crop them ourselves. `gnome-screenshot
            # -w` omits window-manager decorations on common X11 desktops, while
            # xdotool reports the outer window geometry. Advertising the latter as
            # coordinates for the former silently offsets every pointer action.
            self._run(["gnome-screenshot", "-f", str(path)])
            captured = Image.open(path).convert("RGB")
            if window_crop:
                bounds = active_window["bounds"]
                left = bounds["x"]
                top = bounds["y"]
                captured = captured.crop(
                    (
                        left,
                        top,
                        left + bounds["width"],
                        top + bounds["height"],
                    )
                )
            encoded_buffer = io.BytesIO()
            captured.save(encoded_buffer, format="PNG")
            encoded_bytes = encoded_buffer.getvalue()
            encoded = base64.b64encode(encoded_bytes).decode("ascii")
            image_width, image_height = captured.size
        finally:
            path.unlink(missing_ok=True)
        if window_crop:
            bounds = active_window["bounds"]
            frame = {
                "name": "active_window",
                "origin_x": bounds["x"],
                "origin_y": bounds["y"],
                "width": image_width,
                "height": image_height,
            }
        else:
            frame = {
                "name": "screen",
                "origin_x": 0,
                "origin_y": 0,
                "width": image_width,
                "height": image_height,
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
            "visual_fingerprint": self._visual_fingerprint(captured),
        }

    @staticmethod
    def _visual_fingerprint(image: Image.Image) -> dict[str, Any]:
        """Return a compact perceptual frame for causal action verification."""

        resampling = getattr(Image, "Resampling", Image).BILINEAR
        sample = image.convert("L").resize((32, 32), resampling)
        pixels = (
            sample.get_flattened_data()
            if hasattr(sample, "get_flattened_data")
            else sample.getdata()
        )
        quantized = bytes(value // 16 for value in pixels)
        return {
            "algorithm": "gray32-q16-v1",
            "digest": hashlib.sha256(quantized).hexdigest(),
            "sample": base64.b64encode(quantized).decode("ascii"),
        }

    def _annotate_visual_change(
        self, session_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Tell the planner whether its last GUI action materially changed pixels."""

        fingerprint = result.pop("visual_fingerprint", None)
        frame = result.get("coordinate_space")
        window = result.get("active_window")
        identity = (
            str(window.get("id") or "") if isinstance(window, dict) else "",
            str(frame.get("name") or "") if isinstance(frame, dict) else "",
            int(frame.get("width") or 0) if isinstance(frame, dict) else 0,
            int(frame.get("height") or 0) if isinstance(frame, dict) else 0,
        )
        previous = self._visual_states.get(session_id)
        change: dict[str, Any] = {
            "comparable": False,
            "materially_changed": None,
        }
        if isinstance(fingerprint, dict):
            encoded = fingerprint.pop("sample", "")
            try:
                sample = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                sample = b""
            if (
                previous is not None
                and previous.get("identity") == identity
                and sample
                and len(sample) == len(previous.get("sample") or b"")
            ):
                earlier = previous["sample"]
                changed = sum(
                    1 for left, right in zip(earlier, sample, strict=True) if left != right
                )
                fraction = changed / len(sample)
                change = {
                    "comparable": True,
                    "changed_sample_fraction": round(fraction, 4),
                    # A caret, cursor, or tiny animation must not turn a missed click
                    # into apparent progress. Two percent of the coarse frame is a
                    # conservative material-change floor, not a success assertion.
                    "materially_changed": fraction >= 0.02,
                }
            self._visual_states[session_id] = {
                "identity": identity,
                "sample": sample,
                "digest": fingerprint.get("digest"),
            }
        result["visual_change"] = change
        return result

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
        observed = self._visual_states.get(_session_id) or {}
        observed_identity = observed.get("identity")
        observed_space = (
            str(observed_identity[1])
            if isinstance(observed_identity, tuple) and len(observed_identity) >= 2
            else "active_window"
        )
        coordinate_space = self._space_name(
            arguments, observed_space=observed_space
        )
        display_width = display_height = 0
        active_window: dict[str, Any] | None = None
        if action in {"click", "drag"}:
            display_width, display_height, active_window = self._desktop_state()
            if (
                coordinate_space == "active_window"
                and isinstance(observed_identity, tuple)
                and len(observed_identity) >= 2
                and observed_identity[1] == "active_window"
                and observed_identity[0]
                and (
                    active_window is None
                    or str(active_window.get("id") or "") != observed_identity[0]
                )
            ):
                raise GuiAutomationError(
                    "The active window changed after the last screenshot; take a fresh "
                    "snapshot before using image-relative coordinates."
                )
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
        return self._annotate_visual_change(
            _session_id, self._snapshot(coordinate_space)
        )
