"""Bounded Linux AT-SPI semantics for the rapid computer-use controller.

The contracts and fail-closed dispatch discipline follow the demonstrated
Omnius rapid-computer-use native adapter. This Python implementation uses the
host's GI bindings through a fixed bridge, not model-authored code or global
keyboard/pointer injection. Password controls never expose text or operations.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import importlib.util
import json
import os
import re
import selectors
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_PROTOCOL = "robit.omni.native-ui.v1"
_NATIVE_LOCK = threading.RLock()
_POINTER_VERBS = {"click", "double_click", "right_click", "drag_drop", "scroll"}
_VISUAL_POINTER_VERBS = {"click", "double_click", "right_click"}
_OPERATIONS = {"invoke", "focus", "set_value", "set_checked", "select", "key_chord",
               "select_text", "switch_window", "type_text", *_POINTER_VERBS}
_MAX_REPLY_BYTES = 2 * 1024 * 1024
_IDENTITY_HELPER: Any = None


def _x11_text_command(text: str) -> list[str]:
    """One preselected XTEST sequence; every caller string stays a literal argv."""
    if "\n" not in text:
        return ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text]
    command = ["xdotool"]
    for index, line in enumerate(text.split("\n")):
        if index:
            command.extend(["key", "--clearmodifiers", "--delay", "1", "Return"])
        # --args limits this command to exactly one literal string. Without it,
        # xdotool type consumes the remainder of a chained command as text.
        command.extend(["type", "--clearmodifiers", "--delay", "1", "--args", "1", "--", line])
    return command


def _identity_helper() -> Any:
    global _IDENTITY_HELPER
    if _IDENTITY_HELPER is None:
        # The fixed system-Python bridge runs with -I, without a package path.
        source = Path(__file__).with_name("native_identity.py")
        spec = importlib.util.spec_from_file_location("_omni_native_identity", source)
        if spec is None or spec.loader is None:
            raise NativeUiUnavailable("native_provider_identity_unavailable")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        _IDENTITY_HELPER = helper
    return _IDENTITY_HELPER


def normalize_native_task_scope(value: Any) -> dict[str, Any]:
    """Pure validation of caller-owned restrictions; no window is granted here."""
    if (not isinstance(value, dict) or set(value) != {"schema", "windows"}
            or value["schema"] != "robit.omni.native-task-scope.v1"
            or not isinstance(value["windows"], list) or not 1 <= len(value["windows"]) <= 16):
        raise ValueError("Invalid native task window scope")
    windows, xids, starts = [], set(), {}
    for item in value["windows"]:
        if not isinstance(item, dict) or set(item) != {"window_id", "process_id", "process_start"}:
            raise ValueError("Invalid native task window identity")
        xid, pid, start = item["window_id"], item["process_id"], item["process_start"]
        if (not isinstance(xid, str) or re.fullmatch(r"[1-9][0-9]{0,9}", xid) is None
                or int(xid) >= 2**32 or type(pid) is not int or not 0 < pid < 2**31
                or not isinstance(start, str) or re.fullmatch(r"[1-9][0-9]{0,19}", start) is None
                or xid in xids or (pid in starts and starts[pid] != start)):
            raise ValueError("Invalid native task window identity")
        xids.add(xid)
        starts[pid] = start
        windows.append(dict(item))
    return {"schema": value["schema"], "windows": sorted(windows, key=lambda row: int(row["window_id"]))}


def native_task_scope_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(normalize_native_task_scope(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


class NativeUiUnavailable(RuntimeError):
    """Accessibility was unavailable; this does not establish target absence."""

    def __init__(self, code: str, message: str = "Native accessibility is unavailable") -> None:
        safe_code = (code if isinstance(code, str) and len(code) <= 128
                     and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) else None)
        super().__init__(f"{safe_code}: {message}" if safe_code else message)
        self.code = code


class _JsonBridge:
    def __init__(self, session_id: str, python_bin: str, timeout_s: float, process_id: int | None = None,
                 window_scope: dict[str, Any] | None = None) -> None:
        try:
            from portal.desktop import desktop_subprocess_environment
        except ModuleNotFoundError:
            from desktop import desktop_subprocess_environment

        environment = desktop_subprocess_environment()
        if not environment.get("DBUS_SESSION_BUS_ADDRESS") or not (
            environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
        ):
            raise NativeUiUnavailable(
                "desktop_session_unavailable", "No signed-in desktop bus is available"
            )
        self.timeout_s = timeout_s
        self.session_id = session_id
        self.process_id = process_id
        self.window_scope = normalize_native_task_scope(window_scope) if window_scope is not None else None
        self.process = subprocess.Popen(  # noqa: S603
            [python_bin, "-I", "-u", str(Path(__file__).resolve()), "--bridge"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        self.closed = False

    def request(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise NativeUiUnavailable("native_bridge_closed")
        request_id = uuid.uuid4().hex
        encoded = json.dumps(
            {"schema": _PROTOCOL, "id": request_id, "session_id": self.session_id,
             "process_id": self.process_id, **value, "window_scope": self.window_scope}
        ).encode() + b"\n"
        if len(encoded) > 65536:
            raise NativeUiUnavailable("native_request_too_large")
        remote_error = False
        try:
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
            data = bytearray()
            deadline = time.monotonic() + self.timeout_s
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while not data.endswith(b"\n"):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise NativeUiUnavailable("native_bridge_timeout")
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise NativeUiUnavailable("native_bridge_exited")
                    data.extend(chunk)
                    if len(data) > _MAX_REPLY_BYTES:
                        raise NativeUiUnavailable("native_reply_too_large")
            response = json.loads(data)
            if response.get("schema") != _PROTOCOL or response.get("id") != request_id:
                raise NativeUiUnavailable("native_bridge_protocol_error")
            if response.get("ok") is not True:
                remote_error = True  # A complete error reply preserves transport synchrony.
                raise NativeUiUnavailable(
                    str(response.get("error") or "native_observation_unavailable")
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise NativeUiUnavailable("native_bridge_protocol_error")
            return result
        except (OSError, ValueError, AssertionError, NativeUiUnavailable) as exc:
            if not remote_error:
                self.close()
            if isinstance(exc, NativeUiUnavailable):
                raise
            raise NativeUiUnavailable("native_bridge_unavailable") from None

    def close(self) -> None:
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


class NativeUiAdapter:
    """One isolated semantic session; callers retain authorization and verification."""

    def __init__(
        self,
        session_id: str,
        *,
        timeout_s: float = 10,
        bridge_runner: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        python_bin: str | None = None,
        process_id: int | None = None,
        pointing_url: str | None = None,
        pointing_model: str | None = None,
        pointing_revision: str | None = None,
        pointing_timeout_s: float = 60,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
            raise ValueError("A bounded native session ID is required")
        self.session_id = session_id
        if process_id is not None and (type(process_id) is not int or not 0 < process_id < 2**31):
            raise ValueError("A host-owned positive native process ID is required")
        # Some compositors also export accessibility projections of app windows.
        # A host allocation may scope the real application; model actions cannot.
        self.process_id = process_id
        self.timeout_s = max(1, min(30, float(timeout_s)))
        self.pointing_timeout_s = max(1, min(120, float(pointing_timeout_s)))
        self.python_bin = python_bin or (
            "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else sys.executable
        )
        self._runner = bridge_runner
        self._bridge: _JsonBridge | None = None
        self._provider_digest: str | None = None
        self._closed = False
        self.pointing_url = (pointing_url or "").rstrip("/")
        if self.pointing_url:
            parsed = urlsplit(self.pointing_url)
            if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Native pointing requires a host-selected loopback HTTP endpoint")
        self.pointing_model = pointing_model or os.environ.get("OMNI_POINTING_MODEL", "vikhyatk/moondream2")
        self.pointing_revision = pointing_revision or os.environ.get("OMNI_POINTING_REVISION", "9a7d4024050840e001defacec2b00727e89149e6")
        self._step: dict[str, Any] | None = None
        self._visual: dict[str, Any] | None = None
        self.point_receipt: dict[str, Any] | None = None
        if cancelled is not None and not callable(cancelled):
            raise ValueError("Native cancellation must be a host callable")
        self._cancelled = cancelled
        self._window_scope: dict[str, Any] | None = None
        self._scope_locked = False

    def set_window_scope(self, scope: dict[str, Any]) -> None:
        """Host-only setup of a fresh adapter; no tool argument can change it."""
        if self._scope_locked:
            raise NativeUiUnavailable("native_task_scope_immutable")
        self._window_scope = normalize_native_task_scope(scope)
        self._scope_locked = True

    def _scoped_foreground(self, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
        identity = _x11_foreground(environment)
        if not _window_scope_allows(self._window_scope, identity, environment):
            raise NativeUiUnavailable("native_task_window_out_of_scope")
        return identity

    def _check_cancelled(self) -> None:
        if self._closed:
            raise NativeUiUnavailable("native_session_closed")
        if self._cancelled is not None and self._cancelled():
            raise NativeUiUnavailable("native_operation_cancelled", "Native operation cancelled")

    def _probe_pointer(self, state: Mapping[str, Any], node: Mapping[str, Any], verb: str) -> dict[str, Any]:
        self._check_cancelled()
        if (not node.get("visible") or not node.get("enabled") or node.get("sensitive")
                or verb not in node.get("operations", []) or not node.get("bounds")):
            raise NativeUiUnavailable("native_pointer_probe_ineligible")
        binding = {"session_id": self.session_id, "revision": state["revision"], "scope_id": state["scope"]["id"],
                   "node_id": node["id"], "generation": node["generation"], "verb": verb, "bounds": node["bounds"]}
        result = self._request({"op": "probe_pointer", "binding": copy.deepcopy(binding)})
        fields = {"schema", "status", "reason", "complete", "input_attempted", "binding", "identity",
                  "process_start", "geometry", "coordinate_binding", "point"}
        if (not isinstance(result, dict) or set(result) != fields or result.get("schema") != "robit.omni.native-pointer-probe.v1"
                or result.get("binding") != binding or result.get("input_attempted") is not False
                or not isinstance(result.get("status"), str) or result["status"] not in {"target_proven", "blocked", "provider_unknown"}
                or type(result.get("complete")) is not bool
                or not isinstance(result.get("reason"), str)):
            raise NativeUiUnavailable("native_pointer_probe_protocol_error")
        self._check_cancelled()
        if result["status"] == "blocked" or not result["complete"]:
            raise NativeUiUnavailable("native_pointer_probe_unavailable", result["reason"])
        identity, geometry = result["identity"], result["geometry"]
        if (result["reason"] not in ({"native_hit_target"} if result["status"] == "target_proven" else
                                     {"native_hit_null", "native_hit_interface_unavailable"})
                or not isinstance(identity, dict) or set(identity) != {"window_id", "process_id"}
                or identity["window_id"] != state["scope"].get("window_id")
                or type(identity["process_id"]) is not int or identity["process_id"] <= 0
                or not isinstance(result["process_start"], str) or not result["process_start"].isascii()
                or not result["process_start"].isdecimal()
                or not isinstance(geometry, dict) or set(geometry) != {"x", "y", "width", "height"}
                or any(type(value) is not int for value in geometry.values())
                or min(geometry["width"], geometry["height"]) <= 0
                or result["coordinate_binding"] != state.get("coordinate_binding")
                or result["point"] != {"x": node["bounds"]["x"] + node["bounds"]["width"] // 2,
                                       "y": node["bounds"]["y"] + node["bounds"]["height"] // 2}):
            raise NativeUiUnavailable("native_pointer_probe_protocol_error")
        return result

    @staticmethod
    def _visual_key(capture: Mapping[str, Any], probe: Mapping[str, Any] | None) -> str:
        return hashlib.sha256(json.dumps([capture["key"], probe], sort_keys=True).encode()).hexdigest() if probe else capture["key"]

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise NativeUiUnavailable("native_session_closed")
        self._scope_locked = True
        if self._window_scope is not None:
            payload = {**payload, "window_scope": copy.deepcopy(self._window_scope)}
        if self._runner is not None:
            return self._runner(copy.deepcopy(dict(payload)))
        if sys.platform != "linux":
            raise NativeUiUnavailable(
                "native_platform_unsupported", "Native AT-SPI control requires Linux"
            )
        if self._bridge is None:
            try:
                self._bridge = _JsonBridge(self.session_id, self.python_bin, self.timeout_s, self.process_id,
                                           self._window_scope)
            except OSError:
                raise NativeUiUnavailable("native_python_unavailable") from None
        return self._bridge.request(payload)

    def _semantic_observe(self, *, include_windows: bool = False) -> dict[str, Any]:
        with _NATIVE_LOCK:
            target = (copy.deepcopy(self._step["target"])
                      if not include_windows and self._step and self._step["verb"] == "switch_window" else None)
            request = {"op": "observe", "include_windows": include_windows or target is not None}
            if target is not None:
                request["window_target"] = target
            result = self._request(request)
            if result.get("session_id") != self.session_id or not isinstance(
                result.get("nodes"), list
            ):
                raise NativeUiUnavailable("native_bridge_protocol_error")
            result = copy.deepcopy(result)
            digest = result.pop("_namespace_provider_digest", None)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise NativeUiUnavailable("native_bridge_protocol_error")
            self._provider_digest = digest
            for node in result["nodes"]:
                if node.get("sensitive"):
                    node["name"] = "[protected]"
                    node["operations"] = []
                    for key in ("value", "selected", "checked", "text_selection", "text_observation"):
                        node.pop(key, None)
            return result

    def planning_observe(self) -> dict[str, Any]:
        """Inspect active controls and mapped window identities without input/vision."""
        return self._semantic_observe(include_windows=True)

    def inspect_visual(self) -> dict[str, Any]:
        """Read one explicitly requested snapshot; never create input authority."""
        import httpx

        base = {"dispatch_status": "not_dispatched", "evidence_authority": "inspection",
                "task_progress": False}

        def failed(status: str, code: str) -> dict[str, Any]:
            safe = (code if isinstance(code, str) and len(code) <= 128
                    and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) else "native_visual_reader_unavailable")
            return {**base, "status": status, "error": safe}

        def binding(state: Mapping[str, Any]) -> tuple[Any, ...]:
            document = state.get("document_evidence")
            document_id = document.get("node_id") if isinstance(document, Mapping) else None
            nodes = [{key: node.get(key) for key in ("id", "generation", "scope_id", "role", "bounds", "visible", "enabled", "sensitive")}
                     for node in state["nodes"] if node.get("id") == document_id]
            return (state["session_id"], state["revision"], state["scope"],
                    document, state.get("window_scope_sha256"), nodes)

        def protected(state: Mapping[str, Any], image: Mapping[str, Any] | None = None) -> bool:
            for node in state["nodes"]:
                if node.get("visible") is not True or node.get("sensitive") is not True:
                    continue
                rect = node.get("bounds")
                if (not isinstance(rect, dict) or set(rect) != {"x", "y", "width", "height"}
                        or any(type(value) is not int for value in rect.values())
                        or min(rect["width"], rect["height"]) <= 0):
                    return True
                if image and (max(rect["x"], image["x"]) < min(rect["x"] + rect["width"], image["x"] + image["width"])
                              and max(rect["y"], image["y"]) < min(rect["y"] + rect["height"], image["y"] + image["height"])):
                    return True
            return False

        with _NATIVE_LOCK:
            try:
                self._check_cancelled()
                if not self.pointing_url:
                    return failed("unavailable", "native_visual_reader_not_configured")
                state = self.planning_observe()
                if protected(state):
                    return failed("unavailable", "native_visual_reading_protected")
                capture = self._visual_read_capture(state)
                captured_at = time.time()
                if protected(state, capture["image_geometry"]):
                    return failed("unavailable", "native_visual_reading_protected")
                self._check_cancelled()
                if binding(self.planning_observe()) != binding(state):
                    return failed("stale", "native_visual_inspection_changed")
                self._check_cancelled()
                response = httpx.post(self.pointing_url + "/observe", json={
                    "image": base64.b64encode(capture["image"]).decode()}, timeout=self.pointing_timeout_s)
                self._check_cancelled()
                response.raise_for_status()
                if len(response.content) > 65536:
                    return failed("unavailable", "native_visual_reader_reply_too_large")
                result = response.json()
                if (not isinstance(result, dict) or set(result) != {"model", "revision", "observation"}
                        or result.get("model") != self.pointing_model
                        or result.get("revision") != self.pointing_revision):
                    return failed("unavailable", "native_visual_reader_identity_changed")
                reading = result.get("observation")
                if not isinstance(reading, str) or not reading.strip() or len(reading) > 6000:
                    return failed("unavailable", "native_visual_reader_invalid_reading")
                fresh = self.planning_observe()
                if binding(fresh) != binding(state):
                    return failed("stale", "native_visual_inspection_changed")
                self._check_cancelled()
                checked = self._visual_read_capture(fresh)
                self._check_cancelled()
                def nonpixel(value):
                    return {key: item for key, item in value.items()
                            if key not in {"key", "sha256", "image", "environment", "source_frame_sha256"}}
                if nonpixel(checked) != nonpixel(capture) or protected(fresh, checked["image_geometry"]):
                    return failed("stale", "native_visual_inspection_changed")
                final = self.planning_observe()
                if binding(final) != binding(state) or protected(final, checked["image_geometry"]):
                    return failed("stale", "native_visual_inspection_changed")
                pixels_match = checked["sha256"] == capture["sha256"]
                if "document_region" not in capture and (not pixels_match or checked["key"] != capture["key"]):
                    return failed("stale", "native_visual_inspection_changed")
                self._check_cancelled()
                receipt = {"schema": "robit.omni.native-visual-inspection.v1", "id": uuid.uuid4().hex,
                    "source_type": "native_window_visual_inspection", "authority": "untrusted_tool_observation",
                    "snapshot_only": True, "temporal_scope": "captured_snapshot",
                    "current_ui_verified": False, "input_authority": False,
                    "post_capture_pixels_match": pixels_match,
                    "session_id": hashlib.sha256(self.session_id.encode()).hexdigest(),
                    "scope_id": state["scope"]["id"], "semantic_revision": state["revision"],
                    "captured_at_unix": captured_at, "checked_at_unix": time.time(),
                    "model": result["model"], "revision": result["revision"], "frame_sha256": capture["sha256"],
                    "checked_frame_sha256": checked["sha256"],
                    "checked_source_frame_sha256": checked.get("source_frame_sha256", checked["sha256"]),
                    "coverage": "unknown" if state.get("truncated") else "bounded_native_observation",
                    "source_truncated": bool(state.get("truncated")), "absence_claims_supported": False}
                for name in ("identity", "process_start", "geometry", "capture_source", "capture_root",
                             "image_geometry", "pixel_offset", "content_region", "document_region", "source_frame_sha256", "source_image_geometry"):
                    if name in capture:
                        receipt[name] = copy.deepcopy(capture[name])
                document = state.get("document_evidence", {})
                if (isinstance(document, Mapping) and document.get("status") == "observed" and document.get("source") == "atspi.Document.DocURL"
                        and document.get("url") == state["scope"].get("url")
                        and document.get("scope_id") == state["scope"]["id"]
                        and document.get("window_id") == capture["identity"]["window_id"]):
                    receipt.update(source_url=document["url"], document_evidence=copy.deepcopy(document))
                return {**base, "status": "observed" if pixels_match else "captured_snapshot", "reading": reading.strip(),
                        "resource": receipt.get("source_url", state["scope"]["id"]),
                        "visual_inspection_receipt": receipt}
            except NativeUiUnavailable as exc:
                if exc.code in ("native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"):
                    raise
                return failed("cancelled" if exc.code == "native_operation_cancelled" else "unavailable", exc.code)
            except (httpx.HTTPError, ValueError):
                return failed("unavailable", "native_visual_reader_unavailable")

    @staticmethod
    def _reader_document_region(state: Mapping[str, Any]) -> dict[str, Any] | None:
        document = state.get("document_evidence")
        if (not isinstance(document, Mapping) or document.get("status") != "observed"
                or document.get("source") != "atspi.Document.DocURL"):
            return None
        scope = state["scope"]
        nodes = [node for node in state["nodes"] if node.get("id") == document.get("node_id")]
        digest = document.get("provider_identity_sha256")
        if (document.get("provider") != "atspi2" or document.get("scope_id") != scope["id"]
                or document.get("window_id") != scope.get("window_id")
                or not isinstance(document.get("url"), str) or not document["url"]
                or document["url"] != scope.get("url") or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None or len(nodes) != 1):
            raise NativeUiUnavailable("native_visual_document_binding_invalid")
        node = nodes[0]
        bounds = node.get("bounds")
        if (node.get("role") != "document_web" or node.get("scope_id") != scope["id"]
                or not isinstance(node.get("generation"), str) or not node["generation"]
                or node["generation"] != document.get("generation")
                or node.get("visible") is not True or node.get("enabled") is not True
                or node.get("sensitive") is not False or not isinstance(bounds, dict)
                or set(bounds) != {"x", "y", "width", "height"}
                or any(type(value) is not int or abs(value) > 1_000_000 for value in bounds.values())
                or min(bounds["width"], bounds["height"]) <= 0):
            raise NativeUiUnavailable("native_visual_document_binding_invalid")
        return {"node_id": node["id"], "generation": node["generation"], "bounds": copy.deepcopy(bounds)}

    def _visual_read_capture(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Crop only positively bound document pixels; pointer capture is unchanged."""
        region = self._reader_document_region(state)
        capture = self._visual_capture(state)
        if region is None:
            return capture
        import io

        from PIL import Image

        source, geometry, bounds = capture["image_geometry"], capture["geometry"], region["bounds"]
        def rectangle(value: Any) -> bool:
            return (isinstance(value, dict) and set(value) == {"x", "y", "width", "height"}
                    and all(type(item) is int and abs(item) <= 1_000_000 for item in value.values())
                    and min(value["width"], value["height"]) > 0)
        root = capture.get("capture_root")
        if (capture["identity"]["window_id"] != state["scope"]["window_id"]
                or not rectangle(source) or not rectangle(geometry)
                or not isinstance(root, dict) or not rectangle(root.get("geometry"))
                or root["geometry"]["x"] != 0 or root["geometry"]["y"] != 0
                or bounds["x"] < geometry["x"] or bounds["y"] < geometry["y"]
                or bounds["x"] + bounds["width"] > geometry["x"] + geometry["width"]
                or bounds["y"] + bounds["height"] > geometry["y"] + geometry["height"]
                or source["x"] < 0 or source["y"] < 0
                or source["x"] + source["width"] > root["geometry"]["width"]
                or source["y"] + source["height"] > root["geometry"]["height"]):
            raise NativeUiUnavailable("native_visual_document_capture_invalid")
        left, top = max(bounds["x"], source["x"]), max(bounds["y"], source["y"])
        right = min(bounds["x"] + bounds["width"], source["x"] + source["width"])
        bottom = min(bounds["y"] + bounds["height"], source["y"] + source["height"])
        if right <= left or bottom <= top:
            raise NativeUiUnavailable("native_visual_document_capture_invalid")
        try:
            raw = capture["image"]
            if (not isinstance(raw, bytes) or not 33 <= len(raw) <= 8 * 1024 * 1024
                    or hashlib.sha256(raw).hexdigest() != capture["sha256"]):
                raise ValueError("Invalid source frame")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.size != (source["width"], source["height"]) or max(image.size) > 4096:
                    raise ValueError("Invalid source dimensions")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                pixels = image.convert("RGB").crop((left - source["x"], top - source["y"], right - source["x"], bottom - source["y"]))
                normalized = Image.frombytes("RGB", pixels.size, pixels.tobytes())
            encoded = io.BytesIO()
            normalized.save(encoded, format="PNG", optimize=False, compress_level=6)
        except Exception:
            raise NativeUiUnavailable("native_visual_document_capture_invalid") from None
        result = copy.deepcopy(capture)
        result.update(image=encoded.getvalue(), sha256=hashlib.sha256(encoded.getvalue()).hexdigest(),
                      document_region=region, image_geometry={"x": left, "y": top, "width": right-left, "height": bottom-top},
                      pixel_offset={"x": left-geometry["x"], "y": top-geometry["y"]},
                      source_frame_sha256=capture.get("source_frame_sha256", capture["sha256"]),
                      source_image_geometry=copy.deepcopy(capture.get("source_image_geometry", source)))
        return result

    def _visual_capture(self, state: Mapping[str, Any]) -> dict[str, Any]:
        try:
            from portal.desktop import desktop_subprocess_environment
        except ModuleNotFoundError:
            from desktop import desktop_subprocess_environment

        environment = desktop_subprocess_environment()
        identity = self._scoped_foreground(environment)
        if state["scope"].get("window_id") != identity["window_id"]:
            raise NativeUiUnavailable("native_visual_scope_changed")
        capture = _capture_window(identity, environment)
        if self._window_scope is not None and self._scoped_foreground(environment) != identity:
            raise NativeUiUnavailable("native_visual_scope_changed")
        if "coordinate_binding" in state:
            import io

            from PIL import Image

            def rectangle(value: Any) -> bool:
                return (isinstance(value, dict) and set(value) == {"x", "y", "width", "height"}
                        and all(type(item) is int and abs(item) <= 1_000_000 for item in value.values())
                        and min(value["width"], value["height"]) > 0)

            binding = state["coordinate_binding"]
            geometry, source = capture["geometry"], capture["image_geometry"]
            root = capture["capture_root"]
            if (not isinstance(binding, dict) or set(binding) != {"space", "window_id", "surface", "frame_extents", "content"}
                    or binding["space"] != "window" or binding["window_id"] != identity["window_id"]
                    or capture["identity"] != identity or not rectangle(geometry) or not rectangle(binding["surface"])
                    or binding["surface"] != geometry or not rectangle(binding["content"])
                    or not isinstance(root, dict) or set(root) != {"window_id", "geometry"}
                    or not isinstance(root["window_id"], str) or not root["window_id"].isascii() or not root["window_id"].isdecimal()
                    or not 0 < int(root["window_id"]) < 2**32 or not rectangle(root["geometry"])
                    or root["geometry"]["x"] != 0 or root["geometry"]["y"] != 0 or not rectangle(source)):
                raise NativeUiUnavailable("native_content_capture_binding_invalid")
            margins, content = binding["frame_extents"], binding["content"]
            if (not isinstance(margins, dict) or set(margins) != {"left", "right", "top", "bottom"}
                    or any(type(item) is not int or not 0 <= item <= 4096 for item in margins.values())
                    or content != {"x": geometry["x"] + margins["left"], "y": geometry["y"] + margins["top"],
                                   "width": geometry["width"] - margins["left"] - margins["right"],
                                   "height": geometry["height"] - margins["top"] - margins["bottom"]}
                    or source != {"x": max(0, geometry["x"]), "y": max(0, geometry["y"]),
                                  "width": min(root["geometry"]["width"], geometry["x"] + geometry["width"]) - max(0, geometry["x"]),
                                  "height": min(root["geometry"]["height"], geometry["y"] + geometry["height"]) - max(0, geometry["y"])}
                    or capture["pixel_offset"] != {"x": source["x"] - geometry["x"], "y": source["y"] - geometry["y"]}):
                raise NativeUiUnavailable("native_content_capture_binding_invalid")
            left, top = max(content["x"], source["x"]), max(content["y"], source["y"])
            right = min(content["x"] + content["width"], source["x"] + source["width"])
            bottom = min(content["y"] + content["height"], source["y"] + source["height"])
            if right <= left or bottom <= top:
                raise NativeUiUnavailable("native_content_capture_offscreen")
            try:
                raw = capture["image"]
                if (not isinstance(raw, bytes) or not 33 <= len(raw) <= 8 * 1024 * 1024
                        or hashlib.sha256(raw).hexdigest() != capture["sha256"]):
                    raise ValueError("Invalid source frame")
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.size != (source["width"], source["height"]) or max(image.size) > 4096:
                        raise ValueError("Invalid source dimensions")
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    pixels = image.convert("RGB").crop((left - source["x"], top - source["y"], right - source["x"], bottom - source["y"]))
                    # Fresh RGB pixels exclude source palette/global PNG metadata.
                    normalized = Image.frombytes("RGB", pixels.size, pixels.tobytes())
                encoded = io.BytesIO()
                normalized.save(encoded, format="PNG", optimize=False, compress_level=6)
                raw = encoded.getvalue()
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("Unbounded content frame")
            except Exception:
                raise NativeUiUnavailable("native_content_capture_image_invalid") from None
            capture["source_frame_sha256"] = capture["sha256"]
            capture["source_image_geometry"] = copy.deepcopy(source)
            capture["content_region"] = copy.deepcopy(binding)
            capture.update(image=raw, sha256=hashlib.sha256(raw).hexdigest(),
                           image_geometry={"x": left, "y": top, "width": right - left, "height": bottom - top},
                           pixel_offset={"x": left - geometry["x"], "y": top - geometry["y"]})
        capture["environment"] = environment
        key_binding = {
            "session_id": state["session_id"], "revision": state["revision"], "scope": state["scope"],
            "identity": identity, "geometry": capture["geometry"], "image_sha256": capture["sha256"],
            "process_start": capture["process_start"], "capture_source": capture["capture_source"],
            "capture_root": capture["capture_root"], "image_geometry": capture["image_geometry"],
            "pixel_offset": capture["pixel_offset"],
        }
        if "content_region" in capture:
            key_binding["content_region"] = capture["content_region"]
        capture["key"] = hashlib.sha256(json.dumps(key_binding, sort_keys=True).encode()).hexdigest()
        return capture

    def observe(self) -> dict[str, Any]:
        with _NATIVE_LOCK:
            self._check_cancelled()
            state = self._semantic_observe()
            step = self._step
            if not step or step["verb"] not in {"invoke", "click", "double_click", "right_click"} or not self.pointing_url:
                return state
            target = step["target"]
            if target.get("scope_id", state["scope"]["id"]) != state["scope"]["id"]:
                return state
            matching = [node for node in state["nodes"] if all(
                node.get("id" if key == "node_id" else key) == value for key, value in target.items())]
            known, probe = None, None
            if matching:
                # Count every exact match before eligibility; pixels cannot erase
                # ambiguity or a hidden/disabled/protected semantic identity.
                if (state.get("truncated") is True or len(matching) != 1 or step["verb"] not in _VISUAL_POINTER_VERBS
                        or not matching[0]["visible"] or not matching[0]["enabled"] or matching[0]["sensitive"]
                        or step["verb"] not in matching[0]["operations"] or not matching[0].get("bounds")
                        or not matching[0].get("name")):
                    self._visual, self.point_receipt = None, None
                    return state
                known = matching[0]
                probe = self._probe_pointer(state, known, step["verb"])
                if probe["status"] == "target_proven":
                    self._visual, self.point_receipt = None, None
                    return state
            elif target.get("node_id") or not target.get("name"):
                return state
            referent = known["name"] if known else target["name"]
            if len(referent) > 240:
                raise NativeUiUnavailable("native_visual_target_too_long")
            capture = self._visual_capture(state)
            if probe and (probe["identity"] != capture["identity"] or probe["process_start"] != capture["process_start"]
                          or probe["geometry"] != capture["geometry"]):
                raise NativeUiUnavailable("native_pointer_probe_capture_changed")
            key = self._visual_key(capture, probe)
            if self._visual and self._visual["key"] == key:
                return copy.deepcopy(self._visual["state"])
            self._visual, self.point_receipt = None, None
            import httpx

            self._check_cancelled()
            response = httpx.post(self.pointing_url + "/point", json={
                "image": base64.b64encode(capture["image"]).decode(), "target": referent,
            }, timeout=self.pointing_timeout_s)
            self._check_cancelled()
            response.raise_for_status()
            if len(response.content) > 65536:
                raise NativeUiUnavailable("native_point_reply_too_large")
            result = response.json()
            if result.get("model") != self.pointing_model or result.get("revision") != self.pointing_revision:
                raise NativeUiUnavailable("native_point_model_identity_changed")
            points = result.get("points")
            if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], dict):
                if known:
                    raise NativeUiUnavailable("native_point_target_unresolved")
                return state
            point = points[0]
            if any(type(point.get(key)) not in (int, float) or not 0 <= point[key] < 1 for key in ("x", "y")):
                raise NativeUiUnavailable("native_point_invalid_coordinates")
            image_geometry = capture["image_geometry"]
            position = (image_geometry["x"] + int(point["x"] * image_geometry["width"]),
                        image_geometry["y"] + int(point["y"] * image_geometry["height"]))
            if known and not (known["bounds"]["x"] <= position[0] < known["bounds"]["x"] + known["bounds"]["width"]
                              and known["bounds"]["y"] <= position[1] < known["bounds"]["y"] + known["bounds"]["height"]):
                raise NativeUiUnavailable("native_point_outside_bound_target")
            if any(node.get("sensitive") and node.get("bounds") and
                   node["bounds"]["x"] <= position[0] < node["bounds"]["x"] + node["bounds"]["width"] and
                   node["bounds"]["y"] <= position[1] < node["bounds"]["y"] + node["bounds"]["height"]
                   for node in state["nodes"]):
                raise NativeUiUnavailable("native_visual_point_over_protected_control")
            node_id = known["id"] if known else "visual:" + hashlib.sha256((key + json.dumps(target, sort_keys=True)).encode()).hexdigest()
            synthetic = copy.deepcopy(known) if known else {"id": node_id, "generation": capture["sha256"], "scope_id": state["scope"]["id"],
                         "role": target.get("role", "button"), "name": target["name"],
                         "visible": True, "enabled": True, "sensitive": False, "operations": [step["verb"]]}
            grounded = copy.deepcopy(state)
            if not known:
                grounded["nodes"].append(synthetic)
            grounded["base_semantic_revision"] = state["revision"]
            grounded["revision"] = hashlib.sha256(json.dumps([state["revision"], node_id, key, step], sort_keys=True).encode()).hexdigest()
            self.point_receipt = {"schema": "robit.omni.native-point.v1", "model": self.pointing_model,
                "revision": self.pointing_revision, "frame_sha256": capture["sha256"],
                "semantic_revision": state["revision"], "candidate_id": node_id,
                "point": dict(point), "geometry": dict(capture["geometry"]),
                "identity": dict(capture["identity"]), "process_start": capture["process_start"],
                "capture_source": capture["capture_source"], "capture_root": copy.deepcopy(capture["capture_root"]),
                "image_geometry": dict(image_geometry), "pixel_offset": dict(capture["pixel_offset"])}
            for name in ("content_region", "source_frame_sha256", "source_image_geometry"):
                if name in capture:
                    self.point_receipt[name] = copy.deepcopy(capture[name])
            if known:
                self.point_receipt["native_probe"] = copy.deepcopy(probe)
                self.point_receipt["semantic_target"] = {name: copy.deepcopy(known[name]) for name in
                                                       ("id", "generation", "scope_id", "bounds")}
            self._visual = {"key": key, "state": grounded, "capture": capture, "probe": probe, "step": copy.deepcopy(step),
                            "verb": step["verb"], "node": synthetic, "point": position, "receipt": copy.deepcopy(self.point_receipt)}
            return grounded

    def prepare_step(self, step: Mapping[str, Any]) -> None:
        from qwen_omni_adapters.computer_use import parse_step

        self._prepare(parse_step(dict(step)))

    def prepare_intent(self, intent: Mapping[str, Any]) -> None:
        from qwen_omni_adapters.computer_use import parse_intent

        self._prepare(parse_intent(dict(intent)))

    def _prepare(self, parsed: dict[str, Any]) -> None:
        with _NATIVE_LOCK:
            if parsed != self._step:
                self._visual = None
                self.point_receipt = None
            self._step = parsed

    def boundary_observe(self) -> dict[str, Any]:
        """One owned semantic read; prepared pointer intent never invokes MD."""
        with _NATIVE_LOCK:
            self._check_cancelled()
            return self._semantic_observe()

    def finish_intent(self) -> None:
        with _NATIVE_LOCK:
            self._step, self._visual = None, None

    def perform(self, action: Mapping[str, Any]) -> dict[str, Any]:
        with _NATIVE_LOCK:
            self._check_cancelled()
            if action.get("kind", "action") != "action" or action.get("verb") not in _OPERATIONS:
                return {"status": "rejected", "reason": "native_operation_unsupported"}
            if not _valid_arguments(action.get("verb"), action.get("args", {})):
                return {"status": "rejected", "reason": "invalid_native_arguments"}
            visual = self._visual
            if str(action.get("node_id", "")).startswith("visual:") or visual and action.get("node_id") == visual["node"]["id"]:
                if (not visual or action.get("session_id") != self.session_id
                        or action.get("node_id") != visual["node"]["id"]
                        or action.get("generation") != visual["node"]["generation"]
                        or action.get("revision") != visual["state"]["revision"]
                        or action.get("scope_id") != visual["state"]["scope"]["id"]
                        or action.get("verb") != visual["verb"] or action.get("args", {}) != visual["step"]["args"]):
                    return {"status": "rejected", "reason": "native_visual_candidate_stale"}
                try:
                    state = self._semantic_observe()
                    probe = None
                    if visual["probe"]:
                        matches = [node for node in state["nodes"] if all(
                            node.get("id" if name == "node_id" else name) == value for name, value in visual["step"]["target"].items())]
                        if len(matches) != 1:
                            return {"status": "rejected", "reason": "native_visual_target_changed"}
                        probe = self._probe_pointer(state, matches[0], visual["verb"])
                    fresh = self._visual_capture(state)
                    if self._visual_key(fresh, probe) != visual["key"]:
                        return {"status": "rejected", "reason": "native_visual_frame_changed"}
                    # Refresh private provider identity after capture, before input.
                    if self._provider_digest is not None and self._semantic_observe()["revision"] != state["revision"]:
                        return {"status": "rejected", "reason": "native_visual_provider_changed"}
                    self._check_cancelled()
                    result = _pointer_dispatch(fresh["identity"], fresh["geometry"], visual["point"],
                        "click" if action["verb"] == "invoke" else action["verb"], environment=fresh["environment"],
                        foreground=lambda: self._scoped_foreground(fresh["environment"]))
                    result["point_receipt"] = copy.deepcopy(self.point_receipt)
                    return result
                except NativeUiUnavailable as exc:
                    if exc.code in {"native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"}:
                        raise
                    return {"status": "rejected", "reason": exc.code}
                finally:
                    self._visual, self._step = None, None
            result = self._request({"op": "perform", "action": dict(action)})
            if result.get("status") == "already_satisfied" and (
                result.get("input_attempted") is not False or not _already_active_window_action(action)
            ):
                return {"status": "rejected", "reason": "invalid_native_already_satisfied_receipt"}
            if result.get("status") not in {"rejected", "dispatched", "already_satisfied"}:
                raise NativeUiUnavailable("native_dispatch_unresolved")
            return result

    def close(self) -> None:
        with _NATIVE_LOCK:
            self._closed = True
            self._visual, self._step, self.point_receipt = None, None, None
            if self._bridge is not None:
                self._bridge.close()

    clear = close


def _valid_arguments(verb: Any, args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    keys = {
        "invoke": set(),
        "focus": set(),
        "set_value": {"value"},
        "set_checked": {"checked"},
        "select": {"option"},
        "key_chord": {"keys"},
        "click": set(), "double_click": set(), "right_click": set(), "switch_window": set(),
        "drag_drop": {"to_node_id"}, "select_text": {"start", "end"}, "scroll": {"direction"},
        "type_text": {"text"},
    }
    if verb not in keys or set(args) != keys[verb]:
        return False
    if verb == "set_checked":
        return isinstance(args["checked"], bool)
    if verb == "select_text":
        return type(args["start"]) is int and type(args["end"]) is int and 0 <= args["start"] < args["end"] <= 1_000_000
    if verb == "drag_drop":
        return isinstance(args["to_node_id"], str) and 0 < len(args["to_node_id"]) <= 512
    if verb == "scroll":
        return args["direction"] in {"up", "down", "left", "right"}
    if verb == "key_chord":
        tokens = args["keys"]
        return (isinstance(tokens, list) and 1 <= len(tokens) <= 4 and all(
            isinstance(token, str) and 1 <= len(token) <= 30 and token[0].isascii()
            and token[0].isalpha() and all(char.isascii() and (char.isalnum() or char == "_") for char in token)
            for token in tokens))
    if verb == "type_text":
        text = args["text"]
        return isinstance(text, str) and 0 < len(text) <= 2048 and "\x00" not in text
    if verb in {"set_value", "select"}:
        text = args["value" if verb == "set_value" else "option"]
        return isinstance(text, str) and len(text) <= (8192 if verb == "set_value" else 2048)
    return True


def _absent_property_owner(window_id: str, environment: Mapping[str, str]) -> int:
    """XRes is available only after positively proving the WM PID is absent."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed read-only property query.
            ["xprop", "-id", window_id, "_NET_WM_PID"], capture_output=True,
            text=True, timeout=1, check=False, env={**environment, "LC_ALL": "C"})
        if result.returncode or result.stdout.strip() != "_NET_WM_PID:  not found.":
            raise NativeUiUnavailable("native_foreground_unmapped")
        return _identity_helper().xres_owner(window_id, environment)["pid"]
    except NativeUiUnavailable:
        raise
    except Exception:
        raise NativeUiUnavailable("native_foreground_unmapped") from None


def _host_process_identity(process_id: int) -> dict[str, Any]:
    return _identity_helper().host_process_identity(process_id)


def _x11_window_owner(window_id: str, environment: Mapping[str, str], *,
                      include_binding: bool = False, deadline: float | None = None) -> dict[str, Any]:
    """One typed owner path for foreground, allowed bases and direct transients."""
    if (not isinstance(window_id, str) or re.fullmatch(r"[1-9][0-9]{0,9}", window_id) is None
            or int(window_id) >= 2**32):
        raise NativeUiUnavailable("native_foreground_unmapped")

    deadline = deadline if deadline is not None else time.monotonic() + 3
    def query() -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        try:
            result = subprocess.run(["xdotool", "getwindowpid", window_id], capture_output=True,
                                    text=True, timeout=min(1, remaining), check=False, env=dict(environment))
        except (OSError, subprocess.TimeoutExpired):
            raise NativeUiUnavailable("native_foreground_unavailable") from None
        value = result.stdout.strip()
        if result.returncode or re.fullmatch(r"[1-9][0-9]{0,9}", value) is None or int(value) >= 2**31:
            raise NativeUiUnavailable("native_foreground_unmapped")
        return int(value)

    binding, absent = None, False
    try:
        process_id = query()
    except NativeUiUnavailable as exc:
        if exc.code != "native_foreground_unmapped":
            raise
        process_id = _absent_property_owner(window_id, environment)
        absent = True  # The existing fallback still requires positive property absence.
    try:
        _host_process_identity(process_id)
    except Exception:
        if absent:
            raise NativeUiUnavailable("native_foreground_unmapped") from None
        try:
            binding = _identity_helper().namespace_window_owner(window_id, process_id, environment, deadline=deadline)
            if query() != binding["reported_wm_pid"]:
                raise NativeUiUnavailable("native_namespace_owner_changed")
            process_id = binding["identity"]["process_id"]
        except NativeUiUnavailable:
            raise
        except Exception as exc:
            helper = _identity_helper()
            if (isinstance(exc, getattr(helper, "IdentityUnavailable", ()))
                    and str(exc) == "native_provider_identity_budget_exceeded"):
                raise NativeUiUnavailable("native_provider_identity_budget_exceeded") from None
            if time.monotonic() >= deadline:
                raise NativeUiUnavailable("native_observation_budget_exceeded") from None
            raise NativeUiUnavailable("native_foreground_unmapped") from None
    if time.monotonic() >= deadline:
        raise NativeUiUnavailable("native_observation_budget_exceeded")
    result = {"window_id": window_id, "process_id": process_id}
    if include_binding and binding is not None:
        result["_namespace_owner_binding"] = binding
    return result


def _x11_foreground(environment: Mapping[str, str] | None = None, *,
                    include_binding: bool = False, deadline: float | None = None) -> dict[str, Any]:
    """Bind native semantics to the window manager's actual foreground owner."""
    environment = dict(os.environ if environment is None else environment)
    if environment.get("XDG_SESSION_TYPE") == "wayland" or environment.get("WAYLAND_DISPLAY"):
        raise NativeUiUnavailable("native_foreground_wayland_unsupported")
    if not environment.get("DISPLAY"):
        raise NativeUiUnavailable("native_foreground_display_unavailable")

    deadline = deadline if deadline is not None else time.monotonic() + 3
    def active() -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        try:
            result = subprocess.run(["xdotool", "getactivewindow"], capture_output=True,
                                    text=True, timeout=min(1, remaining), check=False, env=environment)
        except (OSError, subprocess.TimeoutExpired):
            raise NativeUiUnavailable("native_foreground_unavailable") from None
        value = result.stdout.strip()
        if result.returncode or re.fullmatch(r"[1-9][0-9]{0,9}", value) is None or int(value) >= 2**32:
            raise NativeUiUnavailable("native_foreground_unmapped")
        return value

    window_id = active()
    result = _x11_window_owner(window_id, environment, include_binding=include_binding, deadline=deadline)
    if active() != window_id:
        raise NativeUiUnavailable("native_foreground_changed_during_observation")
    return result


def _native_foreground_binding() -> dict[str, Any]:
    return _x11_foreground(include_binding=True)


def _x11_geometry(identity: Mapping[str, Any], environment: Mapping[str, str] | None = None) -> dict[str, int]:
    # xdotool getwindowgeometry adds the parent-relative offset a second time
    # for reparented clients on this desktop. xwininfo's absolute client origin
    # agrees with native SCREEN extents and the drawable captured below.
    query_environment = dict(os.environ if environment is None else environment)
    query_environment["LC_ALL"] = "C"
    result = subprocess.run(  # noqa: S603 - fixed read-only X11 query.
        ["xwininfo", "-id", identity["window_id"]],
        capture_output=True, text=True, timeout=2, check=False, env=query_environment,
    )
    fields, window_ids = {}, []
    names = {"Absolute upper-left X": "x", "Absolute upper-left Y": "y", "Width": "width", "Height": "height"}
    for line in result.stdout.splitlines():
        if line.startswith("xwininfo: Window id: "):
            candidate = line.removeprefix("xwininfo: Window id: ").split(" ", 1)[0]
            try:
                window_ids.append(int(candidate, 16))
            except ValueError:
                raise NativeUiUnavailable("native_window_geometry_unavailable") from None
            continue
        key, separator, value = line.strip().partition(":")
        if key not in names:
            continue
        field, value = names[key], value.strip()
        if not separator or field in fields or not value.lstrip("-").isdecimal():
            raise NativeUiUnavailable("native_window_geometry_unavailable")
        fields[field] = int(value)
    if (result.returncode or window_ids != [int(identity["window_id"])] or set(fields) != set(names.values())
            or abs(fields["x"]) > 1_000_000 or abs(fields["y"]) > 1_000_000
            or not 0 < fields["width"] <= 16384 or not 0 < fields["height"] <= 16384):
        raise NativeUiUnavailable("native_window_geometry_unavailable")
    return fields


def _x11_frame_extents(identity: Mapping[str, Any], environment: Mapping[str, str] | None = None) -> dict[str, int]:
    query_environment = dict(os.environ if environment is None else environment)
    query_environment["LC_ALL"] = "C"
    result = subprocess.run(  # noqa: S603 - declared margins for the exact foreground X11 client.
        ["xprop", "-id", identity["window_id"], "_GTK_FRAME_EXTENTS"],
        capture_output=True, text=True, timeout=2, check=False, env=query_environment,
    )
    lines = result.stdout.strip().splitlines()
    prefix = "_GTK_FRAME_EXTENTS(CARDINAL) = "
    if result.returncode or len(lines) != 1 or not lines[0].startswith(prefix):
        raise NativeUiUnavailable("native_window_frame_extents_unavailable")
    values = [part.strip() for part in lines[0].removeprefix(prefix).split(",")]
    if len(values) != 4 or any(not value.isascii() or not value.isdecimal() or int(value) > 4096 for value in values):
        raise NativeUiUnavailable("native_window_frame_extents_unavailable")
    return dict(zip(("left", "right", "top", "bottom"), map(int, values), strict=True))


def _x11_capture_root(identity: Mapping[str, Any], environment: Mapping[str, str]) -> dict[str, Any]:
    """Prove the client's root is the default screen root used by IM6 import."""
    query_environment = dict(environment)
    query_environment["LC_ALL"] = "C"

    def query(arguments: list[str]) -> str:
        try:
            result = subprocess.run(  # noqa: S603 - fixed read-only X11 identity queries.
                ["xwininfo", *arguments], capture_output=True, text=True, timeout=2,
                check=False, env=query_environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise NativeUiUnavailable("native_capture_root_unavailable") from None
        if result.returncode or not isinstance(result.stdout, str) or len(result.stdout) > 65536:
            raise NativeUiUnavailable("native_capture_root_unavailable")
        return result.stdout

    def window_id(output: str, prefix: str) -> int:
        values = [line.strip().removeprefix(prefix).split(" ", 1)[0]
                  for line in output.splitlines() if line.strip().startswith(prefix)]
        try:
            if len(values) == 1 and values[0].startswith("0x") and 0 < int(values[0], 16) < 2**32:
                return int(values[0], 16)
        except ValueError:
            pass
        raise NativeUiUnavailable("native_capture_root_unavailable")

    client = query(["-id", identity["window_id"], "-children"])
    if window_id(client, "xwininfo: Window id: ") != int(identity["window_id"]):
        raise NativeUiUnavailable("native_capture_root_unavailable")
    client_root = window_id(client, "Root window id: ")
    default_root = query(["-root"])
    if window_id(default_root, "xwininfo: Window id: ") != client_root:
        raise NativeUiUnavailable("native_capture_root_unavailable")
    fields: dict[str, int] = {}
    names = {"Absolute upper-left X": "x", "Absolute upper-left Y": "y", "Width": "width", "Height": "height"}
    for line in default_root.splitlines():
        key, separator, value = line.strip().partition(":")
        if key in names:
            value = value.strip()
            if not separator or names[key] in fields or not value.isascii() or not value.isdecimal():
                raise NativeUiUnavailable("native_capture_root_unavailable")
            fields[names[key]] = int(value)
    if (set(fields) != set(names.values()) or fields["x"] != 0 or fields["y"] != 0
            or not 0 < fields["width"] <= 16384 or not 0 < fields["height"] <= 16384):
        raise NativeUiUnavailable("native_capture_root_unavailable")
    return {"window_id": str(client_root), "geometry": fields}


def _capture_window(identity: Mapping[str, Any], environment: Mapping[str, str]) -> dict[str, Any]:
    if _x11_foreground(environment) != identity:
        raise NativeUiUnavailable("native_capture_window_changed")
    process_start = _process_start(identity["process_id"])
    geometry = _x11_geometry(identity, environment)
    root = _x11_capture_root(identity, environment)
    image_geometry = {
        "x": max(0, geometry["x"]), "y": max(0, geometry["y"]),
        "width": min(root["geometry"]["width"], geometry["x"] + geometry["width"]) - max(0, geometry["x"]),
        "height": min(root["geometry"]["height"], geometry["y"] + geometry["height"]) - max(0, geometry["y"]),
    }
    if min(image_geometry["width"], image_geometry["height"]) <= 0:
        raise NativeUiUnavailable("native_capture_window_offscreen")
    if max(image_geometry["width"], image_geometry["height"]) > 4096:
        raise NativeUiUnavailable("native_capture_geometry_mismatch")
    if _x11_foreground(environment) != identity or _process_start(identity["process_id"]) != process_start:
        raise NativeUiUnavailable("native_capture_window_changed")
    # IM6 XImportImage's -screen path reads root pixels, then CropImage clips the
    # requested client rectangle to that root without scaling (6.9.12-98).
    result = subprocess.run(  # noqa: S603 - current screen pixels, cropped by the exact foreground XID.
        ["import", "+frame", "+border", "-screen", "-window", identity["window_id"], "png:-"], capture_output=True,
        timeout=5, check=False, env=dict(environment),
    )
    raw = result.stdout
    if result.returncode or not 33 <= len(raw) <= 8 * 1024 * 1024 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise NativeUiUnavailable("native_window_capture_unavailable")
    width, height = struct.unpack(">II", raw[16:24])
    if (width, height) != (image_geometry["width"], image_geometry["height"]):
        raise NativeUiUnavailable("native_capture_geometry_mismatch")
    if (_x11_capture_root(identity, environment) != root or _x11_geometry(identity, environment) != geometry
            or _x11_foreground(environment) != identity or _process_start(identity["process_id"]) != process_start):
        raise NativeUiUnavailable("native_capture_window_changed")
    return {"image": raw, "sha256": hashlib.sha256(raw).hexdigest(), "geometry": geometry, "identity": dict(identity),
            "process_start": process_start, "capture_source": "x11_root_screen", "capture_root": root,
            "image_geometry": image_geometry,
            "pixel_offset": {"x": image_geometry["x"] - geometry["x"], "y": image_geometry["y"] - geometry["y"]}}


def _pointer_dispatch(identity: Mapping[str, Any], geometry: Mapping[str, int], point: tuple[int, int],
                      verb: str, *, destination: tuple[int, int] | None = None,
                      direction: str | None = None, environment: Mapping[str, str] | None = None,
                      foreground: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Use XTEST only at code-owned points in one freshly bound foreground window."""
    current = foreground or (lambda: _x11_foreground(environment))
    rejected = {"status": "rejected", "reason": "native_pointer_binding_changed"}

    def inside(position: tuple[int, int]) -> bool:
        x, y = position
        return geometry["x"] <= x < geometry["x"] + geometry["width"] and geometry["y"] <= y < geometry["y"] + geometry["height"]

    if not inside(point) or destination is not None and not inside(destination):
        return {"status": "rejected", "reason": "native_pointer_outside_bound_window"}
    if current() != identity or _x11_geometry(identity, environment) != geometry:
        return rejected
    window_id = identity["window_id"]

    def command(arguments: list[str]) -> bool:
        completed = subprocess.run(["xdotool", *arguments], capture_output=True, timeout=2, check=False, env=environment)  # noqa: S603
        return completed.returncode == 0

    def at_target(position: tuple[int, int]) -> bool:
        location = subprocess.run(["xdotool", "getmouselocation", "--shell"], capture_output=True, text=True, timeout=2, check=False, env=environment)  # noqa: S603
        axes = dict(line.split("=", 1) for line in location.stdout.splitlines() if "=" in line)
        return (not location.returncode and axes.get("X") == str(position[0]) and axes.get("Y") == str(position[1])
                and axes.get("WINDOW") == window_id and current() == identity
                and _x11_geometry(identity, environment) == geometry)

    x, y = point[0] - geometry["x"], point[1] - geometry["y"]
    # --sync waits for a movement event that never arrives at an unchanged
    # position. Skip only when position, pointer window and live binding all
    # positively agree; retain the same final checks before any button input.
    if not at_target(point) and not command(["mousemove", "--sync", "--window", window_id, str(x), str(y)]):
        return rejected
    if current() != identity or _x11_geometry(identity, environment) != geometry:
        return rejected
    if not at_target(point):
        return {"status": "rejected", "reason": "native_pointer_did_not_reach_target"}
    # XTEST is the chosen native backend, never a retry after XSendEvent. The
    # actual pointer window, foreground owner and client geometry must agree.
    # Partial input stays dispatched until independent effects establish success.
    try:
        if verb == "drag_drop" and destination is not None:
            ok = command(["mousedown", "1"])
            try:
                ok = command(["mousemove", "--sync", "--window", window_id,
                              str(destination[0] - geometry["x"]), str(destination[1] - geometry["y"])]) and ok
                ok = at_target(destination) and ok
            finally:
                ok = command(["mouseup", "1"]) and ok
        elif verb == "scroll":
            button = {"up": "4", "down": "5", "left": "6", "right": "7"}[direction]
            ok = command(["click", "--repeat", "3", "--delay", "30", button])
        else:
            button, count = ("3", "1") if verb == "right_click" else ("1", "2" if verb == "double_click" else "1")
            ok = command(["click", "--repeat", count, "--delay", "70", button])
    except Exception:
        ok = False
    return {"status": "dispatched", "reason": "native_pointer_invoked" if ok else "native_pointer_effect_unresolved",
            "backend": "x11_guarded_xtest"}


def _process_start(process_id: int) -> str:
    if type(process_id) is not int or process_id <= 0:
        raise NativeUiUnavailable("native_window_process_identity_unavailable")
    try:
        value = Path(f"/proc/{process_id}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if value.isdecimal():
            return value
    except (OSError, IndexError):
        pass
    raise NativeUiUnavailable("native_window_process_identity_unavailable")


def _already_active_window_action(action: Mapping[str, Any]) -> bool:
    node_id = action.get("node_id")
    return (action.get("verb") == "switch_window" and action.get("args") == {}
            and isinstance(node_id, str) and node_id == action.get("scope_id")
            and action.get("expect") == [{"kind": "active_window", "node_id": node_id}])


def _window_query(target: Any) -> dict[str, str] | None:
    if target is None:
        return None
    if (not isinstance(target, dict) or set(target) - {"node_id", "role", "name", "scope_id"}
            or not {"node_id", "role", "name"} & target.keys()
            or any(not isinstance(value, str) or len(value) > 512 or (not value and key != "name")
                   for key, value in target.items())):
        raise NativeUiUnavailable("native_window_query_invalid")
    return copy.deepcopy(target)


def _window_scope_allows(scope: Mapping[str, Any] | None, identity: Mapping[str, Any],
                         environment: Mapping[str, str] | None = None, *,
                         deadline: float | None = None) -> bool:
    if scope is None:
        return True
    env = {**(environment or os.environ), "LC_ALL": "C"}
    deadline = deadline if deadline is not None else time.monotonic() + 3
    def owner(xid: str) -> int:
        return _x11_window_owner(xid, env, deadline=deadline)["process_id"]
    try:
        xid, pid = identity["window_id"], identity["process_id"]
        if not isinstance(xid, str) or type(pid) is not int:
            return False
        start = _process_start(pid)
        bases = [row for row in scope["windows"] if row["process_id"] == pid and row["process_start"] == start]
        if not bases or owner(xid) != pid:
            return False
        if any(row["window_id"] == xid for row in bases):
            return owner(xid) == pid and _process_start(pid) == start
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        result = subprocess.run(["xprop", "-id", xid, "WM_TRANSIENT_FOR"], capture_output=True,
                                text=True, timeout=min(1, remaining), check=False, env=env)
        match = re.fullmatch(r"WM_TRANSIENT_FOR\(WINDOW\): window id # (0x[0-9a-fA-F]+)\s*", result.stdout)
        if result.returncode or match is None:
            return False
        selected = [row for row in bases if int(row["window_id"]) == int(match[1], 16)]
        return (len(selected) == 1 and owner(selected[0]["window_id"]) == pid
                and owner(xid) == pid and _process_start(pid) == start)
    except NativeUiUnavailable as exc:
        if exc.code == "native_observation_budget_exceeded":
            raise
        return False
    except (KeyError, OSError, ValueError, subprocess.TimeoutExpired):
        return False


class _AtspiBridge:
    """Fixed native patterns with bounded reads and no arbitrary call surface."""

    MAX_NODES = 400
    MAX_CHILDREN = 128
    MAX_DEPTH = 64
    MAX_READS = 12000
    MAX_IDENTITIES = 4096
    MAX_RETAINED_WINDOWS = 16

    def __init__(self, atspi: Any, glib: Any, session_id: str, process_id: int | None = None,
                 foreground: Callable[[], dict[str, Any]] = _native_foreground_binding,
                 window_scope: dict[str, Any] | None = None) -> None:
        self.atspi, self.glib, self.session_id = atspi, glib, session_id
        self.process_id = process_id
        self.window_scope = normalize_native_task_scope(window_scope) if window_scope is not None else None
        self._foreground_provider = foreground
        self.key = os.urandom(32)
        self.previous: set[str] = set()
        self._identity_children: dict[str, set[str]] = {}
        self._window_target: dict[str, str] | None = None
        self.generations: dict[str, int] = {}
        self.handles: dict[str, tuple[Any, dict[str, Any], int | None]] = {}
        self.windows: dict[str, Any] = {}
        self._window_history: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._coordinate_binding: dict[str, Any] | None = None
        self._provider_binding: dict[str, Any] | None = None
        self._host_owner_binding = None
        self._host_owner_bound = False
        self._inactive_namespace_bindings: dict[str, dict[str, Any]] = {}
        self._inactive_namespace_error: str | None = None
        self._inactive_namespace_buses: set[str] = set()
        self.atspi.init()
        self.atspi.set_timeout(250, 250)

    def foreground(self) -> dict[str, Any]:
        raw = (_x11_foreground(include_binding=True, deadline=self.deadline)
               if self._foreground_provider is _native_foreground_binding else self._foreground_provider())
        binding = raw.get("_namespace_owner_binding")
        identity = {key: raw[key] for key in ("window_id", "process_id")}
        if self._host_owner_bound and binding != self._host_owner_binding:
            raise NativeUiUnavailable("native_namespace_owner_changed")
        self._host_owner_binding, self._host_owner_bound = copy.deepcopy(binding), True
        if not _window_scope_allows(self.window_scope, identity, deadline=getattr(self, "deadline", None)):
            raise NativeUiUnavailable("native_task_window_out_of_scope")
        return identity

    def _read(self, function: Callable[[], Any]) -> Any:
        self.reads += 1
        if self.reads > self.MAX_READS or time.monotonic() > self.deadline:
            raise NativeUiUnavailable("native_observation_budget_exceeded")
        try:
            return function()
        except Exception:
            raise NativeUiUnavailable("native_observation_unavailable") from None

    def _clear_cache(self, node: Any) -> None:
        # AT-SPI 2.44 exposes clear_cache (node plus cached descendants),
        # while newer versions also expose the narrower clear_cache_single.
        # Both invalidate local cached properties before the same fresh reads.
        method = getattr(node, "clear_cache_single", None)
        if not callable(method):
            method = getattr(node, "clear_cache", None)
        if not callable(method):
            raise NativeUiUnavailable("native_cache_invalidation_unavailable")
        self._read(method)  # Never retry an available method that raises.

    def _opaque(self, raw: str) -> str:
        return "atspi:" + hmac.new(self.key, raw.encode(), hashlib.sha256).hexdigest()[:32]

    def _count(self, node: Any) -> int:
        count = self._read(node.get_child_count)
        if not isinstance(count, int) or count < 0:
            raise NativeUiUnavailable("native_observation_unavailable")
        return count

    def _credentials(self, bus_name: str) -> dict[str, int]:
        from gi.repository import Gio

        cancellable, bus = Gio.Cancellable(), None
        timer = threading.Timer(max(.01, self.deadline - time.monotonic()), cancellable.cancel)
        timer.daemon = True
        timer.start()
        try:
            session = self._read(lambda: Gio.bus_get_sync(Gio.BusType.SESSION, cancellable))
            address = self._read(lambda: session.call_sync(
                "org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus", "GetAddress", None,
                self.glib.VariantType.new("(s)"), Gio.DBusCallFlags.NONE, 500, cancellable)).unpack()[0]
            bus = self._read(lambda: Gio.DBusConnection.new_for_address_sync(address,
                Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
                None, cancellable))
            return {key: self._read(lambda method=method: bus.call_sync(
                "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", method,
                self.glib.Variant("(s)", (bus_name,)), self.glib.VariantType.new("(u)"),
                Gio.DBusCallFlags.NONE, 500, cancellable)).unpack()[0]
                for key, method in (("pid", "GetConnectionUnixProcessID"), ("uid", "GetConnectionUnixUser"))}
        finally:
            try:
                if bus is not None:
                    bus.close_sync(cancellable)  # Never close Gio's shared session bus.
            finally:
                timer.cancel()

    def _provider_proof(self, application: Any, reported_pid: int, *, identity: Mapping[str, Any] | None = None) -> dict | None:
        helper = _identity_helper()
        bus_name = self._read(lambda: application.app.bus_name)
        try:
            return helper.provider_proof(self.foreground_identity if identity is None else identity, bus_name, reported_pid,
                credentials=self._credentials, read=self._read, deadline=self.deadline)
        except helper.IdentityUnavailable as exc:
            raise NativeUiUnavailable(str(exc)) from None

    def _refresh_provider_binding(self) -> None:
        binding = self._provider_binding
        if binding is None:
            return
        desktop = self._read(lambda: self.atspi.get_desktop(0))
        self._clear_cache(desktop)
        count = self._count(desktop)
        if count > self.MAX_CHILDREN:
            raise NativeUiUnavailable("native_provider_scan_incomplete")
        matches = []
        for index in range(count):
            application = self._read(lambda index=index: desktop.get_child_at_index(index))
            if application is None:
                raise NativeUiUnavailable("native_provider_scan_incomplete")
            self._clear_cache(application)
            reported = self._read(application.get_process_id)
            if reported == self.foreground_identity["process_id"]:
                raise NativeUiUnavailable("native_provider_identity_ambiguous")
            proof = self._provider_proof(application, reported)
            if proof is not None:
                matches.append((application, proof))
        if len(matches) != 1 or matches[0][1] != binding["proof"]:
            raise NativeUiUnavailable("native_provider_identity_changed")
        application = matches[0][0]
        if self._read(lambda: str(application.app.bus_name) + ":" + str(application.path)) != binding["application"]:
            raise NativeUiUnavailable("native_provider_identity_changed")
        frames, seen = [], set()

        def scan(node: Any, depth: int = 0) -> None:
            if node is None or depth > self.MAX_DEPTH or len(seen) >= self.MAX_NODES:
                raise NativeUiUnavailable("native_provider_scan_incomplete")
            raw = self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
            if raw in seen:
                raise NativeUiUnavailable("native_provider_scan_incomplete")
            seen.add(raw)
            self._clear_cache(node)
            state = self._read(node.get_state_set)
            if state is None or state.contains(self.atspi.StateType.DEFUNCT):
                raise NativeUiUnavailable("native_provider_scan_incomplete")
            role = self._read(node.get_role_name)
            if role in {"frame", "dialog", "window", "alert", "file chooser"}:
                if state.contains(self.atspi.StateType.ACTIVE):
                    if not all(state.contains(flag) for flag in (self.atspi.StateType.VISIBLE,
                            self.atspi.StateType.SHOWING, self.atspi.StateType.SENSITIVE)):
                        raise NativeUiUnavailable("native_provider_frame_ineligible")
                    frames.append((raw, self._native_rect(node)))
                return
            count = self._count(node)
            if count > self.MAX_CHILDREN:
                raise NativeUiUnavailable("native_provider_scan_incomplete")
            for index in range(count):
                scan(self._read(lambda index=index: node.get_child_at_index(index)), depth + 1)

        scan(application)
        if (len(frames) != 1 or frames[0][0] != binding["frame"]
                or frames[0][1] != _x11_geometry(self.foreground_identity)
                or self.foreground() != self.foreground_identity):
            raise NativeUiUnavailable("native_provider_frame_changed")

    def _provider_guard(self) -> bool:
        try:
            self._refresh_provider_binding()
            return True
        except NativeUiUnavailable as exc:
            if exc.code in {"native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"}:
                raise
            return False

    def _inactive_namespace_providers(self) -> list[tuple]:
        helper = _identity_helper()
        desktop = self._read(lambda: self.atspi.get_desktop(0))
        self._clear_cache(desktop)
        count = self._count(desktop)
        if count > self.MAX_CHILDREN:
            raise NativeUiUnavailable("native_inactive_provider_scan_incomplete")
        providers = []
        for index in range(count):
            app = self._read(lambda index=index: desktop.get_child_at_index(index))
            if app is None:
                raise NativeUiUnavailable("native_inactive_provider_scan_incomplete")
            self._clear_cache(app)
            reported = self._read(app.get_process_id)
            try:
                candidate = helper.relay_candidate(reported, read=self._read, deadline=self.deadline)
            except helper.IdentityUnavailable as exc:
                raise NativeUiUnavailable(str(exc)) from None
            if candidate:
                raw = self._read(lambda app=app: str(app.app.bus_name) + ":" + str(app.path))
                providers.append((app, reported, self._read(lambda app=app: app.app.bus_name), raw))
        return providers

    def _collect_inactive_namespace_bindings(self, providers: list[tuple] | None = None) -> dict[str, dict[str, Any]]:
        """Map complete native frames to server-owned clients; grant no scope."""
        helper = _identity_helper()
        if providers is None:
            providers = self._inactive_namespace_providers()
        if not providers:
            return {}
        client_ids = self._read(lambda: helper.client_windows(os.environ, timeout_s=self.deadline-time.monotonic()))
        physical, matched = [], {}
        for xid in client_ids:
            owner = self._read(lambda xid=xid: helper.xres_owner(xid, os.environ, timeout_s=self.deadline-time.monotonic()))
            identity = {"window_id": xid, "process_id": owner["pid"]}
            matches = []
            for app, reported, bus, raw in providers:
                proof = self._provider_proof(app, reported, identity=identity)
                if proof is not None:
                    if (proof["identity"] != identity or proof["reported_pid"] != reported
                            or proof["bus_name"] != bus or proof["values"]["owner"] != owner):
                        raise NativeUiUnavailable("native_inactive_provider_binding_changed")
                    matches.append((app, raw, proof, bus, reported))
            if len(matches) > 1:
                raise NativeUiUnavailable("native_inactive_provider_ambiguous")
            if matches:
                app, raw, proof, bus, reported = matches[0]
                matched[raw] = (app, bus, reported)
                physical.append({"identity": identity, "process_start": owner["start"],
                    "geometry": self._read(lambda identity=identity: _x11_geometry(identity)),
                    "application": raw, "proof": proof})
        if not matched:
            raise NativeUiUnavailable("native_inactive_provider_unavailable")
        frames = {}
        for application, (app, bus, reported) in matched.items():
            seen = set()

            def scan(node: Any, depth: int = 0, *, seen=seen, application=application, bus=bus, reported=reported) -> None:
                if node is None or depth > self.MAX_DEPTH or len(seen) >= self.MAX_NODES:
                    raise NativeUiUnavailable("native_inactive_frame_scan_incomplete")
                raw = self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
                if self._read(lambda: node.app.bus_name) != bus:
                    raise NativeUiUnavailable("native_inactive_frame_provider_changed")
                if raw in seen:
                    raise NativeUiUnavailable("native_inactive_frame_scan_incomplete")
                seen.add(raw)
                self._clear_cache(node)
                state, role = self._read(node.get_state_set), self._read(node.get_role_name)
                if state is None or state.contains(self.atspi.StateType.DEFUNCT):
                    raise NativeUiUnavailable("native_inactive_frame_state_unavailable")
                if role in {"frame", "dialog", "window", "alert", "file chooser"}:
                    if self._read(node.get_process_id) != reported:
                        raise NativeUiUnavailable("native_inactive_frame_provider_changed")
                    eligible = all(state.contains(flag) for flag in (self.atspi.StateType.VISIBLE,
                        self.atspi.StateType.SHOWING, self.atspi.StateType.SENSITIVE))
                    if eligible:
                        rect = self._native_rect(node)
                        if rect is None:
                            raise NativeUiUnavailable("native_inactive_frame_geometry_unavailable")
                        frames[raw] = {"application": application, "geometry": rect,
                            "active": state.contains(self.atspi.StateType.ACTIVE)}
                    return
                children = self._count(node)
                if role == "password text":
                    if children:
                        raise NativeUiUnavailable("native_inactive_frame_protected_ancestor")
                    return
                if children > self.MAX_CHILDREN:
                    raise NativeUiUnavailable("native_inactive_frame_scan_incomplete")
                for index in range(children):
                    scan(self._read(lambda index=index: node.get_child_at_index(index)), depth+1,
                         seen=seen, application=application, bus=bus, reported=reported)

            scan(app)
        bindings = {}
        for raw, frame in frames.items():
            matches = [window for window in physical if window["application"] == frame["application"]
                       and window["geometry"] == frame["geometry"]]
            if (len(matches) != 1 or sum(
                    item["application"] == frame["application"] and item["geometry"] == frame["geometry"]
                    for item in frames.values()) != 1):
                raise NativeUiUnavailable("native_inactive_window_ambiguous")
            if frame["active"]:
                continue
            bindings[raw] = {**matches[0], "frame": raw}
        if self._read(lambda: helper.client_windows(os.environ)) != client_ids:
            raise NativeUiUnavailable("native_client_inventory_changed")
        for window in physical:
            identity = window["identity"]
            owner = self._read(lambda identity=identity: helper.xres_owner(identity["window_id"], os.environ))
            if (owner != {"pid": identity["process_id"], "start": window["process_start"]}
                    or self._read(lambda identity=identity: _x11_geometry(identity)) != window["geometry"]):
                raise NativeUiUnavailable("native_inactive_window_changed")
        if self.foreground() != self.foreground_identity:
            raise NativeUiUnavailable("native_foreground_changed_during_observation")
        return bindings

    def _refresh_inactive_namespace_bindings(self) -> None:
        if self._inactive_namespace_bindings and self._collect_inactive_namespace_bindings() != self._inactive_namespace_bindings:
            raise NativeUiUnavailable("native_inactive_namespace_changed")

    def _document_evidence(self, window: Any, window_id: str, handles: Mapping[str, Any],
                           generations: Mapping[str, int]) -> dict[str, Any]:
        """Read committed top-level DocumentWeb evidence, never browser chrome text."""
        unavailable = {"status": "unavailable", "source": "atspi.Document.DocURL"}
        getter = getattr(getattr(self.atspi, "Document", None), "get_document_attribute_value", None)
        if not callable(getter):
            return {**unavailable, "reason": "document_interface_unavailable"}
        state_type = self.atspi.StateType
        seen, documents = set(), []

        def identity(node: Any) -> str:
            bus, path = self._read(lambda: (node.app.bus_name, node.path))
            if not isinstance(bus, str) or not bus or not isinstance(path, str) or not path:
                raise NativeUiUnavailable("native_document_identity_unavailable")
            return bus + ":" + path

        def visible(node: Any) -> bool:
            state = self._read(node.get_state_set)
            if state is None or state.contains(state_type.DEFUNCT):
                raise NativeUiUnavailable("native_document_state_unavailable")
            return all(
                state.contains(value) for value in (state_type.VISIBLE, state_type.SHOWING)
            )

        def scan(node: Any, depth: int = 0) -> None:
            if node is None or depth > self.MAX_DEPTH or len(seen) >= self.MAX_NODES:
                raise NativeUiUnavailable("native_document_root_scan_incomplete")
            raw = identity(node)
            if raw in seen:
                raise NativeUiUnavailable("native_document_root_scan_incomplete")
            seen.add(raw)
            self._clear_cache(node)
            role = self._read(node.get_role_name)
            if not isinstance(role, str) or not role:
                raise NativeUiUnavailable("native_document_root_scan_incomplete")
            showing = visible(node)
            if role == "document web":
                if showing:
                    documents.append((node, raw))
                return  # Nested frame documents can never replace the main document.
            count = self._count(node)
            if role == "password text":
                if count:
                    raise NativeUiUnavailable("native_document_root_scan_incomplete")
                return
            if count > self.MAX_CHILDREN:
                raise NativeUiUnavailable("native_document_root_scan_incomplete")
            for index in range(count):
                scan(self._read(lambda i=index: node.get_child_at_index(i)), depth + 1)

        def read_url(node: Any, raw: str) -> str | None:
            self._clear_cache(node)
            if (identity(node) != raw or self._read(node.get_role_name) != "document web"
                    or not visible(node) or "Document" not in self._read(node.get_interfaces)):
                return None
            interface = self._read(node.get_document_iface)
            uri = self._read(lambda: getter(interface, "DocURL"))
            if (not isinstance(uri, str) or not uri or len(uri) > 20000
                    or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in uri)):
                return None
            try:
                parsed = urlsplit(uri)
                if not parsed.scheme or not (parsed.netloc or parsed.path):
                    return None
                if parsed.scheme.lower() in {"http", "https", "ftp"}:
                    if not parsed.hostname:
                        return None
                    _ = parsed.port
            except ValueError:
                return None
            return uri

        try:
            window_raw = identity(window)
            window_state = self._read(window.get_state_set)
            if not visible(window) or not window_state.contains(state_type.ACTIVE):
                return {**unavailable, "reason": "document_window_not_active"}
            scan(window)
            if len(documents) != 1:
                return {**unavailable, "reason": "document_root_not_unique"}
            node, raw = documents[0]
            generation = generations.get(raw)
            record = handles.get(self._opaque(f"{raw}:{generation}")) if generation is not None else None
            if record is None:
                return {**unavailable, "reason": "document_not_in_current_observation"}
            item = record[1]
            if (item["scope_id"] != window_id or not item["visible"] or item["sensitive"]
                    or item["role"] != "document_web"):
                return {**unavailable, "reason": "document_binding_unavailable"}
            uri = read_url(node, raw)
            if uri is None or read_url(node, raw) != uri:
                return {**unavailable, "reason": "document_url_unavailable_or_changed"}
            self._clear_cache(window)
            state = self._read(window.get_state_set)
            if (identity(window) != window_raw or not visible(window) or not state.contains(state_type.ACTIVE)
                    or self.foreground() != self.foreground_identity):
                return {**unavailable, "reason": "document_window_binding_changed"}
            return {"status": "observed", "source": "atspi.Document.DocURL", "url": uri,
                    "node_id": item["id"], "generation": item["generation"], "scope_id": window_id,
                    "window_id": self.foreground_identity["window_id"], "provider": "atspi2",
                    "provider_identity_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        except NativeUiUnavailable as exc:
            if exc.code == "native_observation_budget_exceeded":
                raise
            return {**unavailable, "reason": exc.code}
        except (AttributeError, TypeError):
            return {**unavailable, "reason": "document_interface_unavailable"}

    def _native_rect(self, node: Any, space: Any = None) -> dict[str, int] | None:
        component = self._read(node.get_component_iface)
        if not hasattr(component, "get_extents"):
            return None
        if space is None:
            space = self.atspi.CoordType.SCREEN
        rect = self._read(lambda: component.get_extents(space))
        values = {key: getattr(rect, key, None) for key in ("x", "y", "width", "height")}
        if any(type(value) is not int or abs(value) > 1_000_000 for value in values.values()) or values["width"] <= 0 or values["height"] <= 0:
            return None
        return values

    def _coordinate_for(self, window: Any) -> dict[str, Any] | None:
        screen = self._native_rect(window)
        if screen is None or screen["x"] != 0 or screen["y"] != 0:
            return None
        # GTK4's SCREEN implementation can collapse every position to zero.
        # Bind its native WINDOW space only through exact declared CSD margins;
        # integer dimensions must agree, so no scale or offset is guessed.
        local = self._native_rect(window, self.atspi.CoordType.WINDOW)
        if local is None or local["x"] != 0 or local["y"] != 0:
            raise NativeUiUnavailable("native_window_coordinate_mapping_unavailable")
        identity = self.foreground_identity
        if self.foreground() != identity:
            raise NativeUiUnavailable("native_foreground_changed_during_observation")
        geometry, margins = _x11_geometry(identity), _x11_frame_extents(identity)
        width = geometry["width"] - margins["left"] - margins["right"]
        height = geometry["height"] - margins["top"] - margins["bottom"]
        if (width <= 0 or height <= 0 or local["width"] != width or local["height"] != height
                or screen["width"] != width or screen["height"] != height or self.foreground() != identity):
            raise NativeUiUnavailable("native_window_coordinate_mapping_unavailable")
        return {"space": "window", "window_id": identity["window_id"], "surface": geometry, "frame_extents": margins,
                "content": {"x": geometry["x"] + margins["left"], "y": geometry["y"] + margins["top"],
                            "width": width, "height": height}}

    def _rect(self, node: Any) -> dict[str, int] | None:
        binding = self._coordinate_binding
        rect = self._native_rect(node, self.atspi.CoordType.WINDOW if binding else None)
        if binding and rect:
            content = binding["content"]
            if (rect["x"] < 0 or rect["y"] < 0 or rect["x"] + rect["width"] > content["width"]
                    or rect["y"] + rect["height"] > content["height"]):
                # A clipped decoration/child does not invalidate its parent or
                # siblings. Preserve semantics while withholding pointer bounds;
                # never clamp a target or infer a hit from its visible portion.
                return None
            rect = {**rect, "x": rect["x"] + content["x"], "y": rect["y"] + content["y"]}
        return rect

    def _selected_page(self, notebook: Any, target: Any) -> Any:
        """Bridge GtkNotebook's missing native hit only through its selected page."""
        if self._read(notebook.get_role_name) != "page tab list" or "Selection" not in self._read(notebook.get_interfaces):
            return None
        selection = self._read(notebook.get_selection_iface)
        if self._read(selection.get_n_selected_children) != 1:
            return None
        page = self._read(lambda: selection.get_selected_child(0))
        if page is None or self._read(page.get_role_name) != "page tab":
            return None
        identity = self._read(lambda: (page.app.bus_name, page.path))
        parent = self._read(page.get_parent)
        if parent is None or self._read(lambda: (parent.app.bus_name, parent.path)) != self._read(
            lambda: (notebook.app.bus_name, notebook.path)
        ):
            return None
        self._clear_cache(page)
        state = self._read(page.get_state_set)
        if state.contains(self.atspi.StateType.DEFUNCT) or not all(state.contains(value) for value in (
            self.atspi.StateType.SELECTED, self.atspi.StateType.VISIBLE, self.atspi.StateType.SHOWING,
        )):
            return None
        ancestor, seen = target, set()
        for _ in range(32):
            if ancestor is None or self._read(ancestor.get_role_name) == "password text":
                return None
            current = self._read(lambda node=ancestor: (node.app.bus_name, node.path))
            if current == identity:
                return page
            if current in seen:
                return None
            seen.add(current)
            ancestor = self._read(ancestor.get_parent)
        return None

    def _hit(self, node: Any, point: tuple[int, int], window: Any) -> bool:
        return self._hit_evidence(node, point, window)["status"] == "target_proven"

    def _hit_evidence(self, node: Any, point: tuple[int, int], window: Any) -> dict[str, Any]:
        """Native proof only: NULL is unknown, never permission or occlusion proof."""
        def result(status: str, reason: str, complete: bool = True) -> dict[str, Any]:
            return {"status": status, "reason": reason, "complete": complete}

        space = self.atspi.CoordType.SCREEN
        if self._coordinate_binding:
            origin = self._coordinate_binding["content"]
            point, space = (point[0] - origin["x"], point[1] - origin["y"]), self.atspi.CoordType.WINDOW
        component = self._read(window.get_component_iface)
        if not hasattr(component, "get_accessible_at_point"):
            return result("provider_unknown", "native_hit_interface_unavailable")
        hit = self._read(lambda: component.get_accessible_at_point(*point, space))
        wanted = self._read(lambda: (node.app.bus_name, node.path))
        window_identity = self._read(lambda: (window.app.bus_name, window.path))
        visited = set()
        for _ in range(32):
            if hit is None:
                return result("provider_unknown", "native_hit_null")
            identity = self._read(lambda current=hit: (current.app.bus_name, current.path))
            if identity in visited:
                return result("provider_unknown", "native_hit_cycle", False)
            visited.add(identity)
            self._clear_cache(hit)
            hit_state = self._read(hit.get_state_set)
            if hit_state.contains(self.atspi.StateType.DEFUNCT) or not all(hit_state.contains(value) for value in (
                self.atspi.StateType.VISIBLE, self.atspi.StateType.SHOWING,
            )):
                return result("provider_unknown", "native_hit_unavailable", False)
            ancestor = hit
            ancestry, roles = set(), set()
            for _ in range(32):
                if ancestor is None:
                    break
                role = self._read(ancestor.get_role_name)
                current = self._read(lambda current=ancestor: (current.app.bus_name, current.path))
                if role == "password text":
                    return result("blocked", "native_hit_protected")
                if current in ancestry:
                    return result("provider_unknown", "native_hit_ancestry_cycle", False)
                ancestry.add(current)
                roles.add(role)
                if current == window_identity:
                    break
                if role in {"frame", "dialog", "window", "alert", "file chooser"}:
                    # A role alone does not prove a foreign XID/PID mapping.
                    return result("provider_unknown", "native_hit_foreign_ancestry_unbound", False)
                parent = self._read(ancestor.get_parent)
                if parent is ancestor:
                    return result("provider_unknown", "native_hit_ancestry_cycle", False)
                ancestor = parent
            else:
                return result("provider_unknown", "native_hit_ancestry_limit", False)
            if window_identity not in ancestry:
                return result("provider_unknown", "native_hit_ancestry_unbound", False)
            if wanted in ancestry:
                return result("target_proven", "native_hit_target")
            # AT-SPI returns an immediate hit child on GTK containers. Descend
            # each hit-tested Component rather than treating an ancestor as an
            # occluding sibling or accepting geometry without a native hit.
            if "Component" not in self._read(hit.get_interfaces):
                return result("provider_unknown", "native_hit_interface_unavailable")
            component = self._read(hit.get_component_iface)
            if not hasattr(component, "get_accessible_at_point"):
                return result("provider_unknown", "native_hit_interface_unavailable")
            child = self._read(lambda current=component: current.get_accessible_at_point(*point, space))
            if child is None and roles.intersection({"push button", "toggle button", "check box", "radio button",
                                                     "entry", "combo box", "menu item", "table cell", "list item"}):
                return result("blocked", "native_hit_other_control")
            hit = child if child is not None else self._selected_page(hit, node)
        return result("provider_unknown", "native_hit_depth_limit", False)

    def probe_pointer(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        """Closed read-only pre-selection proof; no coordinates come from a model."""
        fields = {"session_id", "revision", "scope_id", "node_id", "generation", "verb", "bounds"}
        if (not isinstance(binding, dict) or set(binding) != fields
                or any(not isinstance(binding.get(key), str) or not 0 < len(binding[key]) <= 512 for key in fields - {"bounds"})
                or binding["verb"] not in _VISUAL_POINTER_VERBS):
            raise NativeUiUnavailable("native_pointer_probe_invalid")
        bounds = binding["bounds"]
        if (not isinstance(bounds, dict) or set(bounds) != {"x", "y", "width", "height"}
                or any(type(value) is not int or abs(value) > 1_000_000 for value in bounds.values())
                or min(bounds["width"], bounds["height"]) <= 0):
            raise NativeUiUnavailable("native_pointer_probe_invalid")
        before = self.observe()
        record = self.handles.get(binding["node_id"])
        if (binding["session_id"] != self.session_id or binding["revision"] != before["revision"]
                or binding["scope_id"] != before["scope"]["id"] or record is None):
            raise NativeUiUnavailable("native_pointer_probe_stale")
        node, item, _index = record
        if (item["generation"] != binding["generation"] or item["scope_id"] != binding["scope_id"]
                or not item["visible"] or not item["enabled"] or item["sensitive"]
                or binding["verb"] not in item["operations"] or item.get("bounds") != bounds):
            raise NativeUiUnavailable("native_pointer_probe_ineligible")
        window = self.windows[binding["scope_id"]]
        role = self._read(node.get_role_name)
        identity = dict(self.foreground_identity)
        start = self._read(lambda: _process_start(identity["process_id"]))
        geometry = self._read(lambda: _x11_geometry(identity))
        mapping = copy.deepcopy(self._coordinate_binding)
        point = (bounds["x"] + bounds["width"] // 2, bounds["y"] + bounds["height"] // 2)
        evidence = self._hit_evidence(node, point, window)
        self._clear_cache(node)
        self._clear_cache(window)
        state = self._read(node.get_state_set)
        if (self.foreground() != identity or self._read(lambda: _process_start(identity["process_id"])) != start
                or self._read(lambda: _x11_geometry(identity)) != geometry
                or self._coordinate_for(window) != mapping or self._rect(node) != bounds
                or not self._read(window.get_state_set).contains(self.atspi.StateType.ACTIVE)
                or not all(state.contains(value) for value in (self.atspi.StateType.SENSITIVE,
                                                              self.atspi.StateType.VISIBLE, self.atspi.StateType.SHOWING))
                or state.contains(self.atspi.StateType.DEFUNCT) or self._read(node.get_role_name) != role
                or role == "password text" or self._opaque(self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
                                                           + ":" + binding["generation"]) != binding["node_id"]):
            raise NativeUiUnavailable("native_pointer_probe_changed")
        return {"schema": "robit.omni.native-pointer-probe.v1", **evidence, "input_attempted": False,
                "binding": copy.deepcopy(binding), "identity": identity, "process_start": start,
                "geometry": geometry, "coordinate_binding": mapping, "point": {"x": point[0], "y": point[1]}}

    def _pointer(self, action: Mapping[str, Any], node: Any, item: Mapping[str, Any], window: Any) -> dict[str, Any]:
        binding = self._coordinate_binding

        def binding_matches() -> bool:
            try:
                return not binding or self._coordinate_for(window) == binding
            except NativeUiUnavailable as exc:
                if exc.code == "native_observation_budget_exceeded":
                    raise
                return False

        if not binding_matches():
            return {"status": "rejected", "reason": "native_pointer_coordinate_binding_changed"}
        bounds = self._rect(node)
        if bounds is None or bounds != item.get("bounds"):
            return {"status": "rejected", "reason": "native_pointer_geometry_changed"}
        point = (bounds["x"] + bounds["width"] // 2, bounds["y"] + bounds["height"] // 2)
        if not self._hit(node, point, window):
            return {"status": "rejected", "reason": "native_pointer_target_occluded"}
        destination = None
        if action["verb"] == "drag_drop":
            record = self.handles.get(action["args"]["to_node_id"])
            if record is None:
                return {"status": "rejected", "reason": "native_drop_target_missing"}
            target, target_item, _index = record
            rect = self._rect(target)
            if rect is None or rect != target_item.get("bounds") or target_item["scope_id"] != item["scope_id"] or not target_item["visible"] or not target_item["enabled"] or target_item["sensitive"]:
                return {"status": "rejected", "reason": "native_drop_target_not_eligible"}
            destination = (rect["x"] + rect["width"] // 2, rect["y"] + rect["height"] // 2)
            if not self._hit(target, destination, window):
                return {"status": "rejected", "reason": "native_drop_target_occluded"}
        geometry = _x11_geometry(self.foreground_identity)
        if binding and (geometry != binding["surface"] or not binding_matches()):
            return {"status": "rejected", "reason": "native_pointer_coordinate_binding_changed"}
        if not self._provider_guard():
            return {"status": "rejected", "reason": "native_provider_changed_before_dispatch"}
        return _pointer_dispatch(self.foreground_identity, geometry, point, action["verb"],
                                 destination=destination, direction=action["args"].get("direction"), foreground=self.foreground)

    def _mapped_window(self, node: Any, process_id: int) -> str | None:
        """Resolve only an exactly named, uniquely owned visible X11 window."""
        result = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(process_id)],
                                capture_output=True, text=True, timeout=2, check=False)  # noqa: S603
        candidates = result.stdout.splitlines()
        if result.returncode or len(candidates) > 32 or any(not value.isdecimal() for value in candidates):
            return None
        name = self._read(node.get_name)
        matches = []
        for window_id in candidates:
            result = subprocess.run(["xdotool", "getwindowname", window_id],
                                    capture_output=True, text=True, timeout=1, check=False)  # noqa: S603
            if result.returncode == 0 and result.stdout.strip() == name:
                owner = subprocess.run(["xdotool", "getwindowpid", window_id],
                                       capture_output=True, text=True, timeout=1, check=False)  # noqa: S603
                if owner.returncode == 0 and owner.stdout.strip() == str(process_id):
                    matches.append(window_id)
        return matches[0] if len(matches) == 1 else None

    def _window_visibility(self, process_id: int, *, history: OrderedDict | None = None,
                           generations: dict[str, int] | None = None) -> list[dict[str, Any]]:
        """Fresh flags for known windows; retained handles never authorize input."""
        history = self._window_history if history is None else history
        generations = self.generations if generations is None else generations
        receipts = []
        for raw, record in list(history.items()):
            if record["process_id"] != process_id:
                continue
            node = record["node"]
            try:
                if generations.get(raw) != record["generation"] or self._read(
                    lambda current=node: str(current.app.bus_name) + ":" + str(current.path)
                ) != raw:
                    history.pop(raw, None)
                    continue
                self._clear_cache(node)
                state = self._read(node.get_state_set)
                if state is None or state.contains(self.atspi.StateType.DEFUNCT) or self._read(
                    node.get_role_name
                ) not in {"frame", "dialog", "window", "alert", "file chooser"}:
                    history.pop(raw, None)
                    continue
                receipts.append({"id": record["id"], "generation": str(record["generation"]),
                    "owner_window_id": record["owner_window_id"], "source": "retained_atspi_window",
                    "active": state.contains(self.atspi.StateType.ACTIVE),
                    "visible": state.contains(self.atspi.StateType.VISIBLE),
                    "showing": state.contains(self.atspi.StateType.SHOWING)})
            except NativeUiUnavailable as exc:
                if exc.code == "native_observation_budget_exceeded":
                    raise
                history.pop(raw, None)
        return receipts

    def observe(self, include_windows: bool = False, window_target: Any = None) -> dict[str, Any]:
        # A failed read is unknown evidence, never positive node disappearance.
        # Coordinate scratch state must also belong to the last successful read.
        missing = object()
        binding = self._coordinate_binding
        provider_binding = self._provider_binding
        host_owner = (self._host_owner_binding, self._host_owner_bound)
        inactive = (self._inactive_namespace_bindings, self._inactive_namespace_buses, self._inactive_namespace_error)
        target = self._window_target
        identity = getattr(self, "foreground_identity", missing)
        try:
            return self._observe(include_windows, _window_query(window_target))
        except Exception:
            self._coordinate_binding = binding
            self._provider_binding = provider_binding
            self._host_owner_binding, self._host_owner_bound = host_owner
            self._inactive_namespace_bindings, self._inactive_namespace_buses, self._inactive_namespace_error = inactive
            self._window_target = target
            if identity is missing:
                self.__dict__.pop("foreground_identity", None)
            else:
                self.foreground_identity = identity
            raise

    def _observe(self, include_windows: bool, window_target: dict[str, str] | None) -> dict[str, Any]:
        self.reads, self.deadline = 0, time.monotonic() + 5
        self._window_target = copy.deepcopy(window_target)
        self._coordinate_binding = None
        self._provider_binding = None
        self._host_owner_binding, self._host_owner_bound = None, False
        self._inactive_namespace_bindings, self._inactive_namespace_buses, self._inactive_namespace_error = {}, set(), None
        self.foreground_identity = self.foreground()
        process_id = self.foreground_identity["process_id"]
        if self.process_id is not None and process_id != self.process_id:
            raise NativeUiUnavailable("native_foreground_process_scope_changed")
        context = self.glib.MainContext.default()
        for _ in range(100):
            if not context.pending():
                break
            context.iteration(False)
        desktop = self._read(lambda: self.atspi.get_desktop(0))
        if desktop is None:
            raise NativeUiUnavailable("native_desktop_unavailable")
        self._clear_cache(desktop)
        application_count = self._count(desktop)
        applications = [self._read(lambda index=index: desktop.get_child_at_index(index))
                        for index in range(min(application_count, self.MAX_CHILDREN))]
        direct_provider = False
        for application in applications:
            if application is None:
                raise NativeUiUnavailable("native_observation_unavailable")
            self._clear_cache(application)
            direct_provider |= self._read(application.get_process_id) == process_id
        namespace_candidate = not direct_provider
        if namespace_candidate and application_count > self.MAX_CHILDREN:
            raise NativeUiUnavailable("native_provider_scan_incomplete")
        if include_windows:
            providers = self._inactive_namespace_providers()
            self._inactive_namespace_buses = {row[2] for row in providers}
            try:
                self._inactive_namespace_bindings = self._collect_inactive_namespace_bindings(providers)
            except NativeUiUnavailable as exc:
                if exc.code in {"native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"}:
                    raise
                self._inactive_namespace_error = exc.code  # Unknown windows grant no input handles.
        nodes, seen, active, handles, windows, window_ids = [], set(), set(), {}, {}, {}
        identity_root = "@desktop"
        observed_children: dict[str, set[str]] = {}
        complete_children: set[str] = set()
        coverage_uncertain = False
        active_records = {}
        generations = dict(self.generations)
        history = self._window_history.copy()
        truncated, text_remaining = self._inactive_namespace_error is not None, 32768
        state_type = self.atspi.StateType
        roles = {
            "push button": "button",
            "check box": "checkbox",
            "text": "textbox",
            "password text": "textbox",
            "entry": "textbox",
            "label": "text",
            "combo box": "combobox",
            "list box": "listbox",
            "menu item": "menuitem",
            "check menu item": "menuitemcheckbox",
            "radio button": "radio",
            "frame": "window",
        }

        def visit(node: Any, window: str = "", depth: int = 0, application_pid: int | None = None,
                  parent: str = identity_root, namespace_relay: bool = False) -> bool:
            nonlocal truncated, text_remaining, coverage_uncertain
            if len(seen) >= self.MAX_NODES or depth > self.MAX_DEPTH:
                truncated = True
                return False
            if node is None:
                truncated = True
                return False  # Unknown child coverage retains identities, never old handles.
            raw = self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
            if not raw or raw == ":":
                raise NativeUiUnavailable("native_identity_unavailable")
            observed_children.setdefault(parent, set()).add(raw)
            if raw in seen:
                coverage_uncertain = True  # Aliases/cycles cannot prove absent descendants.
                truncated = True
                return False
            seen.add(raw)
            self._clear_cache(node)
            state = self._read(node.get_state_set)
            if state is None or state.contains(state_type.DEFUNCT):
                raise NativeUiUnavailable("native_observation_unavailable")
            role = self._read(node.get_role_name)
            if not isinstance(role, str) or not role:
                raise NativeUiUnavailable("native_observation_unavailable")
            if role == "application":
                namespace_relay = self._read(lambda: node.app.bus_name) in self._inactive_namespace_buses
                application_pid = self._read(node.get_process_id)
                if application_pid != process_id and namespace_candidate:
                    proof = self._provider_proof(node, application_pid)
                    if proof is not None:
                        if self._provider_binding is not None:
                            raise NativeUiUnavailable("native_provider_identity_ambiguous")
                        self._provider_binding = {"proof": proof, "application": raw, "frame": None}
                        application_pid = process_id
                if application_pid != process_id and (not include_windows or self.process_id is not None):
                    return True
            generation = generations.get(raw, 0) + int(raw not in self.previous)
            generations[raw] = generation
            if len(generations) > self.MAX_IDENTITIES:
                raise NativeUiUnavailable("native_identity_budget_exceeded")
            node_id = self._opaque(f"{raw}:{generation}")
            if not window and role in {"frame", "dialog", "window", "alert", "file chooser"}:
                window = node_id
                windows[window] = node
                if state.contains(state_type.ACTIVE) and application_pid == process_id:
                    if self._provider_binding is not None:
                        if self._read(lambda: node.app.bus_name) != self._provider_binding["proof"]["bus_name"]:
                            raise NativeUiUnavailable("native_provider_identity_ambiguous")
                        self._provider_binding["frame"] = raw
                    active.add(window)
                    window_ids[window] = self.foreground_identity["window_id"]
                    active_records[raw] = {"node": node, "id": node_id, "generation": generation,
                        "process_id": process_id, "owner_window_id": self.foreground_identity["window_id"]}
                    if "Component" in self._read(node.get_interfaces):
                        self._coordinate_binding = self._coordinate_for(node)
                else:
                    namespace_binding = self._inactive_namespace_bindings.get(raw)
                    if (namespace_binding is None and (namespace_relay
                            or self._provider_binding is not None and self._read(lambda: node.app.bus_name)
                            == self._provider_binding["proof"]["bus_name"])):
                        return True  # Relay/executable discovery alone grants no inactive authority.
                    if include_windows and type(application_pid) is int:
                        owner = namespace_binding["identity"]["process_id"] if namespace_binding else application_pid
                        xid = namespace_binding["identity"]["window_id"] if namespace_binding else self._mapped_window(node, owner)
                        if xid and not _window_scope_allows(self.window_scope,
                                {"window_id": xid, "process_id": owner}):
                            windows.pop(window, None)
                            return True
                        if xid:
                            visible = all(state.contains(s) for s in (state_type.SHOWING, state_type.VISIBLE))
                            item = {"id": node_id, "generation": str(generation), "scope_id": node_id,
                                    "role": "window", "name": str(self._read(node.get_name) or "")[:500],
                                    "visible": visible, "enabled": state.contains(state_type.SENSITIVE),
                                    "sensitive": False, "operations": ["switch_window"]}
                            if namespace_binding is not None:
                                item["window_binding"] = {"window_id": xid, "process_id": owner,
                                    "process_start": namespace_binding["process_start"]}
                            elif window_target is not None and all(
                                item.get("id" if key == "node_id" else key) == value
                                for key, value in window_target.items()
                            ):
                                owner = self._read(node.get_process_id)
                                try:
                                    start = self._read(lambda: _process_start(owner))
                                except NativeUiUnavailable as exc:
                                    if exc.code == "native_observation_budget_exceeded":
                                        raise
                                    raise NativeUiUnavailable("native_switch_window_binding_unavailable") from None
                                if owner != application_pid:
                                    raise NativeUiUnavailable("native_switch_window_binding_unavailable")
                                item["window_binding"] = {"window_id": xid, "process_id": owner, "process_start": start}
                            nodes.append(item)
                            handles[node_id] = (node, item, None)
                            window_ids[window] = xid
                    return True  # Excluded contents provide no absence evidence.
            visible = all(state.contains(s) for s in (state_type.SHOWING, state_type.VISIBLE))
            enabled = state.contains(state_type.SENSITIVE)
            sensitive = role == "password text"
            interfaces = self._read(node.get_interfaces)
            operations, action_index = [], None
            if not sensitive and "Action" in interfaces:
                interface = self._read(node.get_action_iface)
                count = self._read(interface.get_n_actions)
                for index in range(min(max(0, count), 32)):
                    try:
                        name = self._read(lambda i=index: interface.get_action_name(i))
                    except NativeUiUnavailable as exc:
                        if exc.code != "native_observation_unavailable":
                            raise
                        # GTK4 may advertise actions whose names cannot be read.
                        # Withhold semantic invocation; independently observed
                        # Component/Text capabilities retain their own guards.
                        break
                    if name in {"click", "press", "activate", "toggle", "open", "default"}:
                        action_index = index
                        operations.append("invoke")
                        break
            checkable = role in {"check box", "check menu item", "toggle button"} or state.contains(
                state_type.CHECKABLE
            )
            if action_index is not None and checkable:
                operations.append("set_checked")
            if (
                not sensitive
                and "EditableText" in interfaces
                and state.contains(state_type.EDITABLE)
                and not state.contains(state_type.READ_ONLY)
            ):
                operations.append("set_value")
            if not sensitive and "Selection" in interfaces:
                operations.append("select")
            if not sensitive and "Component" in interfaces and state.contains(state_type.FOCUSABLE):
                operations.append("focus")
            bounds = self._rect(node) if visible and "Component" in interfaces else None
            if bounds and window and not sensitive:
                operations.extend(sorted(_POINTER_VERBS))
            item = {
                "id": node_id,
                "generation": str(generation),
                "scope_id": window,
                "role": "window" if node_id == window else roles.get(role, role.replace(" ", "_")),
                "name": "[protected]" if sensitive else str(self._read(node.get_name) or "")[:500],
                "visible": visible,
                "enabled": enabled,
                "sensitive": sensitive,
                "focused": state.contains(state_type.FOCUSED),
                "operations": operations if window else [],
            }
            if window and (node_id == window or item["focused"]) and visible and enabled and not sensitive:
                item["operations"].append("key_chord")
            if (node_id in active and visible and enabled and not sensitive
                    and window_ids.get(node_id) == self.foreground_identity["window_id"]):
                item["operations"].append("switch_window")
            if bounds:
                item["bounds"] = bounds
            if not sensitive and hasattr(node, "get_description"):
                description = str(self._read(node.get_description) or "")[:500]
                if description:
                    item["description"] = description
            if checkable and not sensitive:
                item["checked"] = state.contains(state_type.CHECKED) or (
                    role == "toggle button" and state.contains(state_type.PRESSED)
                )
            if not sensitive and "Text" in interfaces and text_remaining:
                interface = self._read(node.get_text_iface)
                character_count = self._read(lambda: self.atspi.Text.get_character_count(interface))
                count = min(
                    max(0, character_count),
                    2048,
                    text_remaining,
                )
                text = self._read(lambda: self.atspi.Text.get_text(interface, 0, count))
                item["value"] = str(text)[:count]
                text_remaining -= count
                text_observation = {}
                if type(character_count) is int and 0 <= character_count <= 2**31 - 1:
                    text_observation["character_count"] = character_count
                    # AT-SPI offsets count Unicode characters, not UTF-16 code
                    # units or key events. A short return is a captured prefix;
                    # a coerced or oversized return supplies no coverage proof.
                    if isinstance(text, str) and len(text) <= count:
                        text_observation.update(captured_range={"start": 0, "end": len(text)},
                                                truncated=len(text) < character_count)
                if all(hasattr(self.atspi.Text, method) for method in ("get_n_selections", "get_selection", "add_selection", "set_selection")):
                    item["operations"].append("select_text")
                    selection_count = self._read(lambda: self.atspi.Text.get_n_selections(interface))
                    if type(selection_count) is int and 0 <= selection_count <= 2**31 - 1:
                        text_observation["selection_count"] = selection_count
                    if selection_count > 0:
                        selected_range = self._read(lambda: self.atspi.Text.get_selection(interface, 0))
                        start, end = selected_range.start_offset, selected_range.end_offset
                        if type(start) is int and type(end) is int and 0 <= start < end <= 1_000_000:
                            item["text_selection"] = {"start": start, "end": end}
                if text_observation:
                    item["text_observation"] = text_observation
            if (window and visible and enabled and not sensitive and item["role"] in {"textbox", "combobox"}
                    and "value" in item and state.contains(state_type.EDITABLE) and not state.contains(state_type.READ_ONLY)):
                item["operations"].append("type_text")
            if not sensitive and "Selection" in interfaces:
                selection = self._read(node.get_selection_iface)
                count = min(max(0, self._read(selection.get_n_selected_children)), 20)
                selected = []
                for index in range(count):
                    child = self._read(lambda i=index: selection.get_selected_child(i))
                    selected.append(str(self._read(child.get_name) or "")[:100])
                item["selected"] = selected
            if window:
                nodes.append(item)
                handles[node_id] = (node, item, action_index)
            count = self._count(node)
            truncated |= count > self.MAX_CHILDREN
            child_list_complete = count <= self.MAX_CHILDREN
            observed_children.setdefault(raw, set())
            for index in range(min(count, self.MAX_CHILDREN)):
                if len(seen) >= self.MAX_NODES:
                    truncated = True
                    child_list_complete = False
                    break
                admitted = visit(self._read(lambda i=index: node.get_child_at_index(i)), window, depth + 1,
                                 application_pid, raw, namespace_relay)
                child_list_complete &= admitted
            if child_list_complete:
                complete_children.add(raw)
            return True

        count = application_count
        truncated |= count > self.MAX_CHILDREN
        desktop_complete = count <= self.MAX_CHILDREN
        observed_children.setdefault(identity_root, set())
        for application in applications:
            desktop_complete &= visit(application)
        if desktop_complete:
            complete_children.add(identity_root)
        if len(active) != 1:
            raise NativeUiUnavailable(
                "native_active_window_unavailable",
                "No unique active accessibility window is available",
            )
        window_id = next(iter(active))
        for raw, record in active_records.items():
            history[raw] = record
            history.move_to_end(raw)
        while len(history) > self.MAX_RETAINED_WINDOWS:
            history.popitem(last=False)
        nodes = [node for node in nodes if node["scope_id"] == window_id or "switch_window" in node["operations"]]
        if window_target is not None:
            nodes = [node for node in nodes if node["scope_id"] == window_id or all(
                node.get("id" if key == "node_id" else key) == value for key, value in window_target.items()
            )]
            allowed = {node["id"] for node in nodes}
            handles = {key: record for key, record in handles.items() if key in allowed}
            windows = {key: node for key, node in windows.items() if key in allowed}
            window_ids = {key: xid for key, xid in window_ids.items() if key in allowed}
        visibility = self._window_visibility(process_id, history=history, generations=generations)
        if window_target is not None:
            visibility = [record for record in visibility if record["id"] in windows]
        payload = {
            "session_id": self.session_id,
            "scope": {"id": window_id, "kind": "desktop", "provider": "atspi2",
                      "window_id": self.foreground_identity["window_id"]},
            "nodes": nodes,
            "window_visibility": visibility,
            "window_query_sha256": hashlib.sha256(json.dumps(window_target, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest() if window_target is not None else None,
            "truncated": truncated,
            "text": "\n".join(
                str(node.get("value", node["name"])) for node in nodes if not node["sensitive"]
            )[:32768],
        }
        payload["document_evidence"] = self._document_evidence(windows[window_id], window_id, handles, generations)
        if payload["document_evidence"]["status"] == "observed":
            payload["scope"]["url"] = payload["document_evidence"]["url"]
        if self._coordinate_binding:
            payload["coordinate_binding"] = copy.deepcopy(self._coordinate_binding)
        if self.window_scope is not None:
            payload["window_scope_sha256"] = native_task_scope_sha256(self.window_scope)
            payload["window_scope_count"] = len(self.window_scope["windows"])
        self._refresh_provider_binding()
        self._refresh_inactive_namespace_bindings()
        # Coverage/count facts are observation knowledge only. Clarification
        # cannot change action idempotency, mutation credit or input authority.
        revision_snapshot = {**payload, "nodes": [
            {key: value for key, value in node.items() if key != "text_observation"} for node in nodes]}
        revision_payload = revision_snapshot if self._provider_binding is None else {
            "snapshot": revision_snapshot, "provider_identity_digest": self._provider_binding["proof"]["digest"]}
        if self._host_owner_binding is not None:
            revision_payload = {"snapshot": revision_payload, "namespace_owner_digest": self._host_owner_binding["digest"]}
        if self._inactive_namespace_bindings:
            revision_payload = {"snapshot": revision_payload, "inactive_namespace_bindings": self._inactive_namespace_bindings}
        payload["revision"] = hashlib.sha256(
            json.dumps(revision_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        # Unknown/excluded subtrees retain identities, never old input handles.
        # Only complete native child lists prove an earlier child disappeared.
        children = {parent: set(values) for parent, values in self._identity_children.items()}
        for parent, values in observed_children.items():
            if parent in complete_children and not coverage_uncertain:
                children[parent] = values
            else:
                children.setdefault(parent, set()).update(values)
        present, pending = set(), [identity_root]
        while pending:
            raw = pending.pop()
            if raw not in present:
                present.add(raw)
                pending.extend(children.get(raw, ()))
        children = {parent: values for parent, values in children.items() if parent in present}
        present.discard(identity_root)
        if len(present) > self.MAX_IDENTITIES or sum(map(len, children.values())) > self.MAX_IDENTITIES:
            raise NativeUiUnavailable("native_identity_budget_exceeded")
        # Commit identities and handles only after the complete payload validates.
        self.generations, self._window_history = generations, history
        self._identity_children = children
        self.previous, self.handles, self.windows, self.window_ids = present, handles, windows, window_ids
        if self._provider_binding is not None:
            payload["_namespace_provider_digest"] = self._provider_binding["proof"]["digest"]
        if self._host_owner_binding is not None:
            payload["_namespace_provider_digest"] = hashlib.sha256(json.dumps({
                "provider": payload.get("_namespace_provider_digest"),
                "owner": self._host_owner_binding["digest"]}, sort_keys=True).encode()).hexdigest()
        return payload

    def perform(self, action: Mapping[str, Any]) -> dict[str, Any]:
        switching = action.get("verb") == "switch_window"
        before = self.observe(include_windows=switching,
                              window_target=copy.deepcopy(self._window_target) if switching else None)

        def reject(reason: str) -> dict[str, Any]:
            return {"status": "rejected", "reason": reason}

        if action.get("session_id") != self.session_id:
            return reject("native_session_mismatch")
        if (
            action.get("revision") != before["revision"]
            or action.get("scope_id") != before["scope"]["id"]
        ):
            return reject("stale_native_revision_or_scope")
        record = self.handles.get(action.get("node_id"))
        if record is None:
            return reject("native_target_missing")
        node, item, index = record
        verb, args = action.get("verb"), action.get("args", {})
        if (
            item["generation"] != action.get("generation")
            or item["scope_id"] != before["scope"]["id"] and verb != "switch_window"
        ):
            return reject("native_target_generation_or_window_changed")
        if (
            not item["visible"]
            or not item["enabled"]
            or item["sensitive"]
            or verb not in item["operations"]
        ):
            return reject("native_target_not_eligible")
        if not _valid_arguments(verb, args):
            return reject("invalid_native_arguments")
        window = self.windows[before["scope"]["id"]]
        self._clear_cache(window)
        if not self._read(window.get_state_set).contains(self.atspi.StateType.ACTIVE):
            return reject("native_focus_changed_before_dispatch")
        self._clear_cache(node)
        state = self._read(node.get_state_set)
        if not all(
            state.contains(s)
            for s in (
                self.atspi.StateType.SENSITIVE,
                self.atspi.StateType.SHOWING,
                self.atspi.StateType.VISIBLE,
            )
        ) or state.contains(self.atspi.StateType.DEFUNCT):
            return reject("native_target_changed_before_dispatch")
        operation: Callable[[], Any]
        if verb in {"invoke", "set_checked"}:
            if verb == "set_checked":
                checked = state.contains(self.atspi.StateType.CHECKED) or (
                    item["role"] == "toggle_button" and state.contains(self.atspi.StateType.PRESSED)
                )
                if checked == args["checked"]:
                    return {"status": "dispatched", "reason": "native_checked_already_satisfied"}
            interface = self._read(node.get_action_iface)

            def operation() -> Any:
                return interface.do_action(index)

        elif verb == "set_value":
            if not state.contains(self.atspi.StateType.EDITABLE) or state.contains(
                self.atspi.StateType.READ_ONLY
            ):
                return reject("native_target_not_editable")
            interface = self._read(node.get_editable_text_iface)

            def operation() -> Any:
                return interface.set_text_contents(args["value"])

        elif verb == "focus":
            interface = self._read(node.get_component_iface)
            operation = interface.grab_focus
        elif verb == "key_chord":
            geometry = _x11_geometry(self.foreground_identity)

            def operation() -> dict[str, Any]:
                if not self._provider_guard():
                    return reject("native_provider_changed_before_dispatch")
                if self.foreground() != self.foreground_identity or _x11_geometry(self.foreground_identity) != geometry:
                    return reject("native_key_binding_changed_before_dispatch")
                self._clear_cache(node)
                if item["id"] != before["scope"]["id"] and not self._read(node.get_state_set).contains(self.atspi.StateType.FOCUSED):
                    return reject("native_key_target_not_focused")
                try:
                    result = subprocess.run(  # noqa: S603 - fixed X11 input, never a shell.
                        ["xdotool", "key", "--clearmodifiers", "+".join(args["keys"])],
                        capture_output=True, timeout=2, check=False,
                    )
                    ok = result.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    ok = False
                return {"status": "dispatched", "backend": "x11_guarded_xtest",
                        "reason": "native_key_input_invoked" if ok else "native_key_input_effect_unresolved"}
        elif verb == "type_text":
            geometry = _x11_geometry(self.foreground_identity)

            def operation() -> dict[str, Any]:
                if not self._provider_guard():
                    return reject("native_provider_changed_before_dispatch")
                if (self.foreground() != self.foreground_identity or _x11_geometry(self.foreground_identity) != geometry
                        or self.foreground() != self.foreground_identity):
                    return reject("native_text_binding_changed_before_dispatch")
                self._clear_cache(node)
                current = self._read(node.get_state_set)
                if not current.contains(self.atspi.StateType.FOCUSED):
                    return reject("native_text_target_not_focused")
                if (not all(current.contains(flag) for flag in (
                    self.atspi.StateType.EDITABLE, self.atspi.StateType.VISIBLE,
                    self.atspi.StateType.SHOWING, self.atspi.StateType.SENSITIVE,
                )) or current.contains(self.atspi.StateType.READ_ONLY) or current.contains(self.atspi.StateType.DEFUNCT)
                        or self._read(node.get_role_name) not in {"text", "entry", "combo box"}):
                    return reject("native_text_target_not_editable")
                if "\n" in args["text"]:
                    multiline = getattr(self.atspi.StateType, "MULTI_LINE", None)
                    if multiline is None or not current.contains(multiline):
                        return reject("native_multiline_text_target_not_multiline")
                # Last host reads immediately precede the one chosen XTEST input.
                # A failed/partial type remains dispatched; no setter/clipboard retry.
                if (self.foreground() != self.foreground_identity or _x11_geometry(self.foreground_identity) != geometry
                        or self.foreground() != self.foreground_identity):
                    return reject("native_text_binding_changed_before_dispatch")
                try:
                    result = subprocess.run(  # noqa: S603 - bounded text is one literal argv value, never shell code.
                        _x11_text_command(args["text"]),
                        capture_output=True, timeout=5, check=False,
                    )
                    ok = result.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    ok = False
                return {"status": "dispatched", "backend": "x11_guarded_xtest",
                        "reason": "native_text_input_invoked" if ok else "native_text_input_effect_unresolved"}
        elif verb == "select_text":
            interface = self._read(node.get_text_iface)
            count = self._read(lambda: self.atspi.Text.get_character_count(interface))
            if args["end"] > count:
                return reject("native_text_selection_out_of_range")
            coverage, value = item.get("text_observation"), item.get("value")
            captured = coverage.get("captured_range") if isinstance(coverage, Mapping) else None
            complete = (
                type(count) is int and count > 0 and isinstance(value, str) and len(value) == count
                and args == {"start": 0, "end": count}
                and isinstance(coverage, Mapping) and type(coverage.get("character_count")) is int
                and coverage["character_count"] == count and coverage.get("truncated") is False
                and isinstance(captured, Mapping) and set(captured) == {"start", "end"}
                and type(captured["start"]) is int and type(captured["end"]) is int
                and captured == {"start": 0, "end": count}
                and item.get("focused") is True
                and {"select_text", "type_text", "key_chord"}.issubset(item["operations"])
                and state.contains(self.atspi.StateType.EDITABLE)
                and not state.contains(self.atspi.StateType.READ_ONLY)
            )
            if complete:
                geometry = _x11_geometry(self.foreground_identity)
                raw = self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))

                def operation() -> dict[str, Any]:
                    # Select the whole revision-bound value, never a guessed
                    # partial range or a retry after an uncertain native setter.
                    try:
                        if not self._provider_guard():
                            return reject("native_provider_changed_before_dispatch")
                        self._clear_cache(node)
                        current = self._read(node.get_state_set)
                        if (not all(current.contains(flag) for flag in (
                            self.atspi.StateType.EDITABLE, self.atspi.StateType.FOCUSED,
                            self.atspi.StateType.VISIBLE, self.atspi.StateType.SHOWING,
                            self.atspi.StateType.SENSITIVE,
                        )) or current.contains(self.atspi.StateType.READ_ONLY)
                                or current.contains(self.atspi.StateType.DEFUNCT)
                                or self._read(node.get_role_name) not in {"text", "entry", "combo box"}):
                            return reject("native_text_selection_target_changed")
                        if not self._provider_guard():
                            return reject("native_provider_changed_before_dispatch")
                        fresh = self._read(node.get_text_iface)
                        fresh_count = self._read(lambda: self.atspi.Text.get_character_count(fresh))
                        if (type(fresh_count) is not int or fresh_count != count
                                or self._read(lambda: self.atspi.Text.get_text(fresh, 0, count)) != value):
                            return reject("native_text_selection_value_changed")
                        if (self._read(lambda: str(node.app.bus_name) + ":" + str(node.path)) != raw
                                or str(self.generations.get(raw)) != item["generation"]
                                or self.handles.get(item["id"], (None,))[0] is not node):
                            return reject("native_text_selection_target_changed")
                        self._clear_cache(node)
                        current = self._read(node.get_state_set)
                        if (not all(current.contains(flag) for flag in (
                            self.atspi.StateType.EDITABLE, self.atspi.StateType.FOCUSED,
                            self.atspi.StateType.VISIBLE, self.atspi.StateType.SHOWING,
                            self.atspi.StateType.SENSITIVE,
                        )) or current.contains(self.atspi.StateType.READ_ONLY)
                                or current.contains(self.atspi.StateType.DEFUNCT)
                                or self._read(node.get_role_name) not in {"text", "entry", "combo box"}):
                            return reject("native_text_selection_target_changed")
                        if (self.foreground() != self.foreground_identity
                                or _x11_geometry(self.foreground_identity) != geometry
                                or self.foreground() != self.foreground_identity):
                            return reject("native_text_selection_binding_changed")
                    except Exception:
                        # None of these reads attempted input. In particular a
                        # budget/provider failure must not become 'dispatched'.
                        return reject("native_text_selection_preflight_unavailable")
                    try:
                        result = subprocess.run(  # noqa: S603 - fixed whole-text XTEST input, never shell code.
                            ["xdotool", "key", "--clearmodifiers", "Control_L+a"],
                            capture_output=True, timeout=2, check=False,
                        )
                        ok = result.returncode == 0
                    except (OSError, subprocess.TimeoutExpired):
                        ok = False
                    return {"status": "dispatched", "backend": "x11_guarded_xtest",
                            "reason": "native_key_input_invoked" if ok else "native_key_input_effect_unresolved"}
            else:
                def operation() -> Any:
                    if self.atspi.Text.get_n_selections(interface):
                        return self.atspi.Text.set_selection(interface, 0, args["start"], args["end"])
                    return self.atspi.Text.add_selection(interface, args["start"], args["end"])

        elif verb == "switch_window":
            if item["id"] == before["scope"]["id"]:
                if not _already_active_window_action(action):
                    return reject("native_already_active_expectation_invalid")

                def operation() -> dict[str, Any]:
                    # Current-root authority comes from the fresh foreground
                    # provider binding, including namespace proof when needed.
                    # Inactive title/PID mapping grants no authority here.
                    if (self.windows.get(item["id"]) is not node
                            or self.window_ids.get(item["id"]) != self.foreground_identity["window_id"]
                            or self.foreground() != self.foreground_identity
                            or not self._read(node.get_state_set).contains(self.atspi.StateType.ACTIVE)
                            or self.foreground() != self.foreground_identity):
                        return reject("native_active_window_binding_changed")
                    return {"status": "already_satisfied", "input_attempted": False,
                            "reason": "native_window_already_active"}

            else:
                xid = self.window_ids.get(item["id"])
                raw = self._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
                namespace_binding = self._inactive_namespace_bindings.get(raw)
                owner = namespace_binding["identity"]["process_id"] if namespace_binding else self._read(node.get_process_id)
                if not xid or (namespace_binding["identity"]["window_id"] if namespace_binding else self._mapped_window(node, owner)) != xid:
                    return reject("native_switch_window_mapping_changed")
                expected = item.get("window_binding")
                if expected is not None and expected != {"window_id": xid, "process_id": owner,
                                                         "process_start": self._read(lambda: _process_start(owner))}:
                    return reject("native_switch_window_mapping_changed")

                def operation() -> bool | dict[str, Any]:
                    if namespace_binding is not None:
                        try:
                            self._refresh_inactive_namespace_bindings()
                        except NativeUiUnavailable as exc:
                            if exc.code in {"native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"}:
                                raise
                            return reject("native_switch_namespace_changed")
                        if (self._inactive_namespace_bindings.get(raw) != namespace_binding
                                or self._read(lambda: str(node.app.bus_name) + ":" + str(node.path)) != raw
                                or str(self.generations.get(raw)) != item["generation"]
                                or self._read(node.get_process_id) != namespace_binding["proof"]["reported_pid"]):
                            return reject("native_switch_namespace_changed")
                    if not _window_scope_allows(self.window_scope, {"window_id": xid, "process_id": owner}):
                        return reject("native_task_window_out_of_scope")
                    if expected is not None and ((not namespace_binding and self._read(node.get_process_id) != expected["process_id"])
                            or self._read(lambda: _process_start(owner)) != expected["process_start"]
                            or (namespace_binding["identity"]["window_id"] if namespace_binding else self._mapped_window(node, owner)) != expected["window_id"]
                            or self.foreground() != self.foreground_identity):
                        return reject("native_switch_window_mapping_changed")
                    result = subprocess.run(["xdotool", "windowactivate", "--sync", xid],
                                            capture_output=True, timeout=2, check=False)  # noqa: S603
                    return result.returncode == 0

        elif verb in _POINTER_VERBS:
            # Pointer dispatch has its own immediate hit/geometry/foreground guard.
            def operation() -> Any:
                return self._pointer(action, node, item, window)
        else:
            count = self._count(node)
            if count > self.MAX_CHILDREN:
                return reject("native_selection_truncated")
            matches = []
            for child_index in range(count):
                child = self._read(lambda i=child_index: node.get_child_at_index(i))
                if self._read(child.get_name) == args["option"]:
                    matches.append(child_index)
            if len(matches) != 1:
                return reject("native_option_not_unique")
            interface = self._read(node.get_selection_iface)

            def operation() -> bool:
                return bool(interface.clear_selection() and interface.select_child(matches[0]))

        # Resolving an interface/selection can make further D-Bus round trips.
        # Recheck foreground, target state and role at the final input boundary.
        if self.foreground() != self.foreground_identity:
            return reject("native_foreground_changed_before_dispatch")
        self._clear_cache(window)
        if not self._read(window.get_state_set).contains(self.atspi.StateType.ACTIVE):
            return reject("native_focus_changed_before_dispatch")
        self._clear_cache(node)
        live = self._read(node.get_state_set)
        if (
            not all(
                live.contains(s)
                for s in (
                    self.atspi.StateType.SENSITIVE,
                    self.atspi.StateType.VISIBLE,
                    self.atspi.StateType.SHOWING,
                )
            )
            or live.contains(self.atspi.StateType.DEFUNCT)
            or self._read(node.get_role_name) == "password text"
        ):
            return reject("native_target_changed_before_dispatch")
        if verb in {"set_value", "type_text"} and (
            not live.contains(self.atspi.StateType.EDITABLE)
            or live.contains(self.atspi.StateType.READ_ONLY)
        ):
            return reject("native_target_not_editable")
        if verb == "type_text" and not live.contains(self.atspi.StateType.FOCUSED):
            return reject("native_text_target_not_focused")
        if verb == "key_chord" and item["id"] != before["scope"]["id"] and not live.contains(self.atspi.StateType.FOCUSED):
            return reject("native_key_target_not_focused")
        if verb == "set_checked":
            checked = live.contains(self.atspi.StateType.CHECKED) or (
                item["role"] == "toggle_button" and live.contains(self.atspi.StateType.PRESSED)
            )
            if checked == args["checked"]:
                return {"status": "dispatched", "reason": "native_checked_already_satisfied"}
        # Once a native pattern is attempted, failure/exception does not prove
        # absence of partial effects. Only the controller's fresh predicates can.
        if not self._provider_guard():
            return reject("native_provider_changed_before_dispatch")
        if self.window_scope is not None and self.foreground() != self.foreground_identity:
            return reject("native_foreground_changed_before_dispatch")
        try:
            ok = operation()
        except NativeUiUnavailable as exc:
            if exc.code in {"native_observation_budget_exceeded", "native_provider_identity_budget_exceeded"}:
                raise
            ok = False
        except Exception:
            ok = False
        if isinstance(ok, dict):
            return ok
        return {
            "status": "dispatched",
            "reason": "native_pattern_invoked" if ok else "native_pattern_effect_unresolved",
        }


def _bridge_main() -> None:
    bridge = None
    for line in iter(lambda: sys.stdin.buffer.readline(65537), b""):
        if len(line) > 65536:
            break
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("schema") != _PROTOCOL or not isinstance(request_id, str):
                raise NativeUiUnavailable("native_bridge_protocol_error")
            if bridge is None:
                session_id = request.get("session_id")
                if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
                    raise NativeUiUnavailable("native_session_invalid")
                process_id = request.get("process_id")
                if process_id is not None and (type(process_id) is not int or not 0 < process_id < 2**31):
                    raise NativeUiUnavailable("native_process_scope_invalid")
                try:
                    window_scope = (normalize_native_task_scope(request["window_scope"])
                                    if request.get("window_scope") is not None else None)
                except ValueError:
                    raise NativeUiUnavailable("native_task_scope_invalid") from None
                try:
                    import gi

                    gi.require_version("Atspi", "2.0")
                    from gi.repository import Atspi, GLib
                except (ImportError, ValueError):
                    raise NativeUiUnavailable("native_atspi_bindings_unavailable") from None
                bridge = _AtspiBridge(Atspi, GLib, session_id, process_id, window_scope=window_scope)
            if request.get("session_id") != bridge.session_id:
                raise NativeUiUnavailable("native_session_mismatch")
            if request.get("process_id") != bridge.process_id:
                raise NativeUiUnavailable("native_process_scope_mismatch")
            if request.get("window_scope") != bridge.window_scope:
                raise NativeUiUnavailable("native_task_scope_mismatch")
            if request.get("op") == "observe":
                result = bridge.observe(include_windows=request.get("include_windows") is True,
                                        window_target=request.get("window_target"))
            elif request.get("op") == "perform" and isinstance(request.get("action"), dict):
                result = bridge.perform(request["action"])
            elif request.get("op") == "probe_pointer" and isinstance(request.get("binding"), dict):
                result = bridge.probe_pointer(request["binding"])
            else:
                raise NativeUiUnavailable("native_operation_unsupported")
            response = {"schema": _PROTOCOL, "id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {
                "schema": _PROTOCOL,
                "id": request_id,
                "ok": False,
                "error": exc.code
                if isinstance(exc, NativeUiUnavailable)
                else "native_provider_unavailable",
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--bridge":
    _bridge_main()
