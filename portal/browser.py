"""Persistent Chromium automation backed by its native DevTools protocol.

The browser opens on the signed-in desktop so the person beside the machine can
see and take over the same page. It never silently substitutes a headless
session: GUI browser work without an attached graphical desktop is an explicit
capability failure.
CDP is used instead of ChromeDriver because the Flatpak Chromium build and the
distribution chromedriver are not version locked on the target Jetson.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from qwen_omni_adapters.memory import MemoryGovernor, MemoryPressure

try:
    from portal.desktop import desktop_subprocess_environment
except ModuleNotFoundError:  # Direct script execution from portal/.
    from desktop import desktop_subprocess_environment

logger = logging.getLogger(__name__)


class BrowserAutomationError(RuntimeError):
    """A visible-browser action could not be completed."""


class BrowserDesktopUnavailable(BrowserAutomationError):
    """The service could not join the signed-in graphical desktop."""


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _read_exact(connection: socket.socket, length: int) -> bytes:
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(length - len(output))
        if not chunk:
            raise BrowserAutomationError("Chromium closed the DevTools connection")
        output.extend(chunk)
    return bytes(output)


class _WebSocket:
    """Small RFC 6455 client sufficient for Chromium's local JSON CDP socket."""

    def __init__(self, url: str, timeout_s: float) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "ws" or not parsed.hostname:
            raise BrowserAutomationError("Chromium returned an invalid DevTools URL")
        self._socket = socket.create_connection(
            (parsed.hostname, parsed.port or 80), timeout=timeout_s
        )
        self._socket.settimeout(timeout_s)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port or 80}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response and len(response) < 16_384:
            response.extend(self._socket.recv(4096))
        if not response.startswith(b"HTTP/1.1 101"):
            self.close()
            raise BrowserAutomationError("Chromium rejected the DevTools connection")
        self._buffer = bytes(response.split(b"\r\n\r\n", 1)[1])

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def _receive(self, length: int) -> bytes:
        if len(self._buffer) >= length:
            value, self._buffer = self._buffer[:length], self._buffer[length:]
            return value
        initial, self._buffer = self._buffer, b""
        return initial + _read_exact(self._socket, length - len(initial))

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = bytes((first, 0x80 | length))
        elif length < 65_536:
            header = bytes((first, 0x80 | 126)) + struct.pack("!H", length)
        else:
            header = bytes((first, 0x80 | 127)) + struct.pack("!Q", length)
        mask = secrets.token_bytes(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self._socket.sendall(header + mask + masked)

    def send_json(self, value: dict[str, Any]) -> None:
        self._send_frame(0x1, json.dumps(value, separators=(",", ":")).encode())

    def receive_json(self) -> dict[str, Any]:
        combined = bytearray()
        message_opcode: int | None = None
        while True:
            head = self._receive(2)
            final = bool(head[0] & 0x80)
            opcode = head[0] & 0x0F
            masked = bool(head[1] & 0x80)
            length = head[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._receive(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._receive(8))[0]
            mask = self._receive(4) if masked else b""
            payload = self._receive(length)
            if masked:
                payload = bytes(
                    value ^ mask[index % 4] for index, value in enumerate(payload)
                )
            if opcode == 0x8:
                raise BrowserAutomationError("Chromium closed the DevTools connection")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x2}:
                message_opcode = opcode
                combined = bytearray(payload)
            elif opcode == 0x0 and message_opcode is not None:
                combined.extend(payload)
            else:
                continue
            if final:
                try:
                    value = json.loads(combined.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise BrowserAutomationError(
                        "Chromium returned malformed DevTools data"
                    ) from exc
                if isinstance(value, dict):
                    return value


class _Cdp:
    def __init__(self, url: str, timeout_s: float) -> None:
        self._socket = _WebSocket(url, timeout_s)
        self._next_id = 0

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._socket.send_json(
            {"id": request_id, "method": method, "params": params or {}}
        )
        while True:
            message = self._socket.receive_json()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise BrowserAutomationError(f"Chromium {method} failed: {detail}")
            result = message.get("result")
            return dict(result) if isinstance(result, dict) else {}


@dataclass
class _BrowserSession:
    process: subprocess.Popen[bytes]
    profile: Path
    port: int
    page_socket: str
    visible_on_desktop: bool
    last_seen: float = field(default_factory=time.monotonic)
    elements: dict[str, dict[str, Any]] = field(default_factory=dict)


_SNAPSHOT_SCRIPT = r"""
(() => {
  const visible = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.visibility !== 'hidden' && s.display !== 'none' &&
           Number(s.opacity || 1) > 0 && r.width > 1 && r.height > 1 &&
           r.bottom >= 0 && r.right >= 0 && r.top <= innerHeight && r.left <= innerWidth;
  };
  document.querySelectorAll('[data-omni-id]').forEach(el => el.removeAttribute('data-omni-id'));
  const candidates = [...document.querySelectorAll(
    'a[href],button,input,textarea,select,summary,[role="button"],[role="link"],[tabindex]'
  )].filter(visible).slice(0, 120);
  const elements = candidates.map((el, i) => {
    const id = `e${i + 1}`, r = el.getBoundingClientRect();
    el.setAttribute('data-omni-id', id);
    return {
      id,
      tag: el.tagName.toLowerCase(),
      type: (el.getAttribute('type') || '').slice(0, 40),
      text: (el.innerText || el.value || el.getAttribute('aria-label') ||
             el.getAttribute('title') || el.getAttribute('placeholder') || '').trim().slice(0, 240),
      href: (el.href || '').slice(0, 800),
      disabled: !!el.disabled,
      x: Math.round(r.left), y: Math.round(r.top),
      width: Math.round(r.width), height: Math.round(r.height)
    };
  });
  return {
    title: document.title,
    url: location.href,
    ready_state: document.readyState,
    viewport: {width: innerWidth, height: innerHeight, scroll_y: Math.round(scrollY)},
    visible_text: (document.body?.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 12000),
    elements
  };
})()
"""


class BrowserAutomationStore:
    """One persistent visible Chromium session per authenticated portal session."""

    def __init__(
        self,
        *,
        ttl_s: float = 900.0,
        chromium_bin: str | None = None,
        timeout_s: float = 15.0,
        memory_governor: MemoryGovernor | None = None,
        launch_reserve_gib: float | None = None,
    ) -> None:
        self.ttl_s = max(30.0, float(ttl_s))
        self.timeout_s = max(2.0, float(timeout_s))
        self.chromium_bin = chromium_bin or os.environ.get(
            "OMNI_CHROMIUM_BIN", "/usr/local/bin/chromium"
        )
        self.memory_governor = memory_governor
        self.launch_reserve_gib = launch_reserve_gib
        self._lock = threading.RLock()
        self._sessions: dict[str, _BrowserSession] = {}

    def _admit_single_window(self) -> None:
        live = [
            key
            for key, session in self._sessions.items()
            if session.process.poll() is None
        ]
        if live:
            raise BrowserAutomationError(
                "another portal session already owns the single rendered browser on "
                "this constrained runtime"
            )

    def _terminate(self, session: _BrowserSession) -> None:
        if session.process.poll() is None:
            try:
                os.killpg(session.process.pid, signal.SIGTERM)
                session.process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(session.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(session.profile, ignore_errors=True)

    def _expire_locked(self) -> None:
        now = time.monotonic()
        for key, session in list(self._sessions.items()):
            if now - session.last_seen >= self.ttl_s or session.process.poll() is not None:
                self._sessions.pop(key, None)
                self._terminate(session)
        if (
            self.memory_governor is not None
            and self.memory_governor.under_hard_pressure()
            and self._sessions
        ):
            logger.warning("closing visible browser sessions at the runtime memory floor")
            sessions, self._sessions = list(self._sessions.values()), {}
            for session in sessions:
                self._terminate(session)

    def clear(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(_session_key(session_id), None)
            if session is not None:
                self._terminate(session)

    def close(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
            for session in sessions:
                self._terminate(session)

    def _json(self, port: int, path: str) -> Any:
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            headers={"User-Agent": "omni-visible-browser/1"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            return json.loads(response.read(2 * 1024 * 1024))

    def _page_socket(self, port: int) -> str:
        targets = self._json(port, "/json/list")
        if not isinstance(targets, list):
            raise BrowserAutomationError("Chromium returned no browser targets")
        for target in targets:
            if (
                isinstance(target, dict)
                and target.get("type") == "page"
                and str(target.get("webSocketDebuggerUrl") or "")
            ):
                return str(target["webSocketDebuggerUrl"])
        raise BrowserAutomationError("Chromium has no visible page target")

    def _launch(self) -> _BrowserSession:
        if not shutil.which(self.chromium_bin):
            raise BrowserAutomationError(f"Chromium is unavailable at {self.chromium_bin}")
        environment = desktop_subprocess_environment()
        visible_on_desktop = bool(
            environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
        )
        if not visible_on_desktop:
            raise BrowserDesktopUnavailable(
                "No signed-in graphical desktop is available for visible Chromium; "
                "browser_interact did not launch a headless substitute."
            )
        self._admit_single_window()
        if self.memory_governor is not None:
            if self.launch_reserve_gib is None:
                self.memory_governor.require("visible browser")
            else:
                self.memory_governor.require_capacity(
                    "visible browser", self.launch_reserve_gib
                )
        port = _free_loopback_port()
        profile = Path(tempfile.mkdtemp(prefix="omni-visible-chromium-"))
        command = [
            self.chromium_bin,
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
            "--disable-background-networking",
            "--disable-gpu",
            "--remote-allow-origins=*",
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--window-size=1280,800",
            "about:blank",
        ]
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=environment,
        )
        deadline = time.monotonic() + self.timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline and process.poll() is None:
            try:
                socket_url = self._page_socket(port)
                return _BrowserSession(
                    process,
                    profile,
                    port,
                    socket_url,
                    visible_on_desktop,
                )
            except (BrowserAutomationError, OSError, URLError, ValueError) as exc:
                last_error = exc
                time.sleep(0.1)
        session = _BrowserSession(
            process,
            profile,
            port,
            "",
            visible_on_desktop,
        )
        self._terminate(session)
        raise BrowserAutomationError(
            f"Chromium did not become controllable: {last_error or 'process exited'}"
        )

    def _session_locked(self, session_id: str) -> _BrowserSession:
        self._expire_locked()
        key = _session_key(session_id)
        session = self._sessions.get(key)
        if session is None:
            session = self._launch()
            self._sessions[key] = session
        session.last_seen = time.monotonic()
        return session

    def _wait_rendered(self, cdp: _Cdp, wait_ms: int) -> None:
        deadline = time.monotonic() + max(0.2, min(5.0, wait_ms / 1000.0))
        while time.monotonic() < deadline:
            result = cdp.call(
                "Runtime.evaluate",
                {"expression": "document.readyState", "returnByValue": True},
            )
            value = result.get("result")
            if isinstance(value, dict) and value.get("value") in {"interactive", "complete"}:
                break
            time.sleep(0.05)
        if wait_ms:
            time.sleep(min(0.35, wait_ms / 1000.0))

    def _evaluate(self, cdp: _Cdp, expression: str) -> Any:
        result = cdp.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        remote = result.get("result")
        if isinstance(remote, dict):
            return remote.get("value")
        return None

    def _snapshot(self, session: _BrowserSession, cdp: _Cdp) -> dict[str, Any]:
        raw = self._evaluate(cdp, _SNAPSHOT_SCRIPT)
        if not isinstance(raw, dict):
            raise BrowserAutomationError("Could not inspect the rendered page")
        elements = raw.get("elements")
        element_items = elements if isinstance(elements, list) else []
        session.elements = {
            str(item.get("id")): dict(item)
            for item in element_items
            if isinstance(item, dict) and item.get("id")
        }
        shot = cdp.call(
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": False},
        ).get("data")
        if not isinstance(shot, str) or not shot:
            raise BrowserAutomationError("Chromium produced no screenshot")
        return {
            "rendered": True,
            "browser_visible_on_desktop": session.visible_on_desktop,
            "title": str(raw.get("title") or "")[:500],
            "url": str(raw.get("url") or "")[:4096],
            "ready_state": str(raw.get("ready_state") or ""),
            "viewport": raw.get("viewport") if isinstance(raw.get("viewport"), dict) else {},
            "visible_text": str(raw.get("visible_text") or "")[:12000],
            "elements": list(session.elements.values()),
            "screenshot": {
                "mime_type": "image/png",
                "encoding": "base64",
                "data": shot,
            },
        }

    def _element(self, session: _BrowserSession, element_id: Any) -> dict[str, Any]:
        normalized = str(element_id or "").strip()
        element = session.elements.get(normalized)
        if element is None:
            raise BrowserAutomationError(
                f"Unknown element {normalized or '(empty)'}; take a new snapshot and use its id"
            )
        return element

    def _click(self, cdp: _Cdp, element: dict[str, Any]) -> None:
        x = float(element.get("x") or 0) + float(element.get("width") or 0) / 2
        y = float(element.get("y") or 0) + float(element.get("height") or 0) / 2
        for event_type, buttons in (("mousePressed", 1), ("mouseReleased", 0)):
            cdp.call(
                "Input.dispatchMouseEvent",
                {
                    "type": event_type,
                    "x": x,
                    "y": y,
                    "button": "left",
                    "buttons": buttons,
                    "clickCount": 1,
                },
            )

    def _drag(
        self,
        cdp: _Cdp,
        element: dict[str, Any],
        delta_x: int,
        delta_y: int,
    ) -> None:
        start_x = float(element.get("x") or 0) + float(element.get("width") or 0) / 2
        start_y = float(element.get("y") or 0) + float(element.get("height") or 0) / 2
        end_x = start_x + delta_x
        end_y = start_y + delta_y
        cdp.call(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": start_x, "y": start_y, "buttons": 0},
        )
        cdp.call(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": start_x,
                "y": start_y,
                "button": "left",
                "buttons": 1,
                "clickCount": 1,
            },
        )
        for step in range(1, 13):
            fraction = step / 12
            cdp.call(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": start_x + delta_x * fraction,
                    "y": start_y + delta_y * fraction,
                    "button": "left",
                    "buttons": 1,
                },
            )
        cdp.call(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": end_x,
                "y": end_y,
                "button": "left",
                "buttons": 0,
                "clickCount": 1,
            },
        )

    def act(self, session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip().lower()
        if action not in {
            "navigate",
            "snapshot",
            "click",
            "drag",
            "type",
            "scroll",
            "back",
            "close",
        }:
            raise BrowserAutomationError(
                "action must be navigate, snapshot, click, drag, type, scroll, back, or close"
            )
        if action == "close":
            self.clear(session_id)
            return {"closed": True, "rendered": False}
        wait_value = arguments.get("wait_ms", 500)
        try:
            wait_ms = max(0, min(5000, int(wait_value)))
        except (TypeError, ValueError) as exc:
            raise BrowserAutomationError("wait_ms must be an integer") from exc

        with self._lock:
            session = self._session_locked(session_id)
            done = threading.Event()
            watcher = None
            tripped = threading.Event()
            if self.memory_governor is not None:
                watcher, tripped = self.memory_governor.watch(
                    "visible-browser", lambda: self._terminate(session), done
                )
            # Refresh the page target in case the user opened or closed a tab manually.
            try:
                session.page_socket = self._page_socket(session.port)
                cdp = _Cdp(session.page_socket, self.timeout_s)
                cdp.call("Page.enable")
                cdp.call("Runtime.enable")
                if action == "navigate":
                    url = str(arguments.get("url") or "").strip()
                    parsed = urlsplit(url)
                    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                        raise BrowserAutomationError(
                            "navigate requires an absolute HTTP(S) URL"
                        )
                    cdp.call("Page.navigate", {"url": url})
                elif action in {"click", "drag", "type"}:
                    element = self._element(session, arguments.get("element_id"))
                    if action == "drag":
                        try:
                            delta_x = max(-4000, min(4000, int(arguments.get("delta_x", 0))))
                            delta_y = max(-4000, min(4000, int(arguments.get("delta_y", 0))))
                        except (TypeError, ValueError) as exc:
                            raise BrowserAutomationError(
                                "delta_x and delta_y must be integers"
                            ) from exc
                        if delta_x == 0 and delta_y == 0:
                            raise BrowserAutomationError(
                                "drag requires a non-zero delta_x or delta_y"
                            )
                        self._drag(cdp, element, delta_x, delta_y)
                    else:
                        self._click(cdp, element)
                    if action == "type":
                        if arguments.get("clear") is True:
                            cdp.call(
                                "Input.dispatchKeyEvent",
                                {
                                    "type": "keyDown",
                                    "key": "a",
                                    "code": "KeyA",
                                    "modifiers": 2,
                                },
                            )
                            cdp.call(
                                "Input.dispatchKeyEvent",
                                {
                                    "type": "keyUp",
                                    "key": "a",
                                    "code": "KeyA",
                                    "modifiers": 2,
                                },
                            )
                            cdp.call(
                                "Input.dispatchKeyEvent",
                                {"type": "keyDown", "key": "Backspace", "code": "Backspace"},
                            )
                            cdp.call(
                                "Input.dispatchKeyEvent",
                                {"type": "keyUp", "key": "Backspace", "code": "Backspace"},
                            )
                        cdp.call("Input.insertText", {"text": str(arguments.get("text") or "")[:8000]})
                        if arguments.get("submit") is True:
                            for event_type in ("keyDown", "keyUp"):
                                cdp.call(
                                    "Input.dispatchKeyEvent",
                                    {"type": event_type, "key": "Enter", "code": "Enter"},
                                )
                elif action == "scroll":
                    try:
                        pixels = max(-4000, min(4000, int(arguments.get("pixels", 600))))
                    except (TypeError, ValueError) as exc:
                        raise BrowserAutomationError("pixels must be an integer") from exc
                    cdp.call(
                        "Input.dispatchMouseEvent",
                        {
                            "type": "mouseWheel",
                            "x": 640,
                            "y": 400,
                            "deltaX": 0,
                            "deltaY": pixels,
                        },
                    )
                elif action == "back":
                    self._evaluate(cdp, "history.back(); true")
                self._wait_rendered(cdp, wait_ms)
                result = self._snapshot(session, cdp)
                if tripped.is_set() and self.memory_governor is not None:
                    self._sessions.pop(_session_key(session_id), None)
                    raise MemoryPressure(
                        "visible browser",
                        self.memory_governor.available_gib(),
                        self.memory_governor.policy.hard_floor_gib,
                    )
                return result
            except (TimeoutError, OSError, URLError) as exc:
                if tripped.is_set() and self.memory_governor is not None:
                    self._sessions.pop(_session_key(session_id), None)
                    raise MemoryPressure(
                        "visible browser",
                        self.memory_governor.available_gib(),
                        self.memory_governor.policy.hard_floor_gib,
                    ) from exc
                raise BrowserAutomationError(f"Visible Chromium communication failed: {exc}") from exc
            finally:
                done.set()
                if watcher is not None:
                    watcher.join(timeout=1.0)
                if "cdp" in locals():
                    cdp.close()
