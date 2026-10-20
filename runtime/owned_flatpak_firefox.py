"""Private test-host ownership of one ordinary Flatpak Firefox blank window.

No model tool or public native bridge operation is added. A fixed system-Python
child retains its launch ledger; callers cannot submit an XID/ledger as authority.
The event witness supports a single-operator setup interval, not atomic proof
against an indistinguishable concurrent launch by the same application.
"""

from __future__ import annotations

import configparser
import copy
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import selectors
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

SCHEMA = "robit.omni.owned-flatpak-firefox.v1"
REPO = Path(__file__).resolve().parents[1]
MAX_WINDOWS = 4096
MAX_REPLY = 2 * 1024 * 1024
_X11_FIELDS = {"error_code", "request_code", "minor_code", "resource_id", "serial",
               "stage", "connection_poisoned"}
_X11_STAGES = {method + ":" + boundary for method in ("snapshot", "arm", "select", "events", "internal")
               for boundary in ("pre_sync", "post_sync")}


def _valid_x11_error(value):
    return (isinstance(value, dict) and set(value) == _X11_FIELDS
            and isinstance(value["stage"], str) and value["stage"] in _X11_STAGES
            and value["connection_poisoned"] is True
            and all(type(value[key]) is int and lower <= value[key] < upper
                    for key, lower, upper in (("error_code", 1, 256), ("request_code", 1, 256),
                        ("minor_code", 0, 256), ("resource_id", 0, 2**32), ("serial", 0, 2**64))))


class OwnershipUnavailable(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _XProtocolUnavailable(OwnershipUnavailable):
    def __init__(self, event, stage):
        super().__init__("flatpak_x11_protocol_error")
        self._event, self._stage = tuple(event), stage

    def diagnostic(self):
        return {**dict(zip(("error_code", "request_code", "minor_code", "resource_id", "serial"),
                           self._event, strict=True)), "stage": self._stage, "connection_poisoned": True}


def _source() -> dict:
    return {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "runtime/owned_flatpak_firefox.py",
            "portal/native_ui.py",
            "portal/native_identity.py",
            "runtime/owned_window_close.py",
        )
    }


def _tree(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"root", "windows", "complete"}
        or value["complete"] is not True
        or not isinstance(value["windows"], dict)
        or not 1 <= len(value["windows"]) <= MAX_WINDOWS
        or not isinstance(value["root"], str)
        or value["root"] not in value["windows"]
    ):
        raise OwnershipUnavailable("flatpak_window_tree_incomplete")
    for xid, row in value["windows"].items():
        if (
            not isinstance(xid, str)
            or not xid.isascii()
            or not xid.isdecimal()
            or not 0 < int(xid) < 2**32
            or not isinstance(row, dict)
            or set(row) != {"parent", "map_state", "override_redirect"}
            or type(row["map_state"]) is not int
            or row["map_state"] not in {0, 1, 2}
            or type(row["override_redirect"]) is not bool
            or not isinstance(row["parent"], str)
        ):
            raise OwnershipUnavailable("flatpak_window_tree_incomplete")
        current, seen = xid, set()
        while current != value["root"]:
            if current in seen or len(seen) > 64 or current not in value["windows"]:
                raise OwnershipUnavailable("flatpak_window_tree_incomplete")
            seen.add(current)
            current = value["windows"][current]["parent"]
        if value["windows"][value["root"]]["parent"] != "0":
            raise OwnershipUnavailable("flatpak_window_tree_incomplete")
    return value


def _check_proof(proof, xid, *, blank=False):
    window = proof.get("window", {})
    namespace = proof.get("namespace", {})
    if (
        window.get("window_id") != xid
        or type(window.get("process_id")) is not int
        or not 0 < window["process_id"] < 2**31
        or not isinstance(window.get("process_start"), str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", window["process_start"]) is None
        or namespace.get("source") != "flatpak_namespace_metadata"
        or namespace.get("identity") != {"window_id": xid, "process_id": window["process_id"]}
        or namespace.get("values", {}).get("owner")
        != {"pid": window["process_id"], "start": window["process_start"]}
        or not isinstance(proof.get("frame", {}).get("identity"), str)
        or not isinstance(proof.get("frame", {}).get("generation"), str)
        or blank
        and proof.get("blank_source") != {"source": "atspi.Document.DocURL", "url": "about:blank"}
    ):
        raise OwnershipUnavailable("flatpak_owned_identity_unproven")
    return proof


class _Transaction:
    """One closed setup transaction; its backend is never supplied over IPC."""

    def __init__(self, backend):
        self.backend, self.ledger = backend, None
        self.attempted = False

    def _preserved(self, before: dict, after: dict) -> bool:
        return before["root"] == after["root"] and all(
            after["windows"].get(xid) == row for xid, row in before["windows"].items()
        )

    def create(self) -> dict:
        if self.attempted:
            raise OwnershipUnavailable("flatpak_launch_already_attempted")
        self.attempted = True
        deployment = self.backend.deployment()
        before = _tree(self.backend.snapshot())
        clients = self.backend.clients()
        self.backend.arm(before)
        if (
            _tree(self.backend.snapshot()) != before
            or self.backend.clients() != clients
            or self.backend.events()
        ):
            raise OwnershipUnavailable("flatpak_prelaunch_inventory_changed")
        launch = self.backend.launch(deployment)
        observed = []
        deadline = self.backend.clock() + 15
        while self.backend.clock() < deadline:
            if self.backend.launch_record()["exit_code"] not in {None, 0}:
                raise OwnershipUnavailable("flatpak_launcher_failed")
            observed.extend(self.backend.events())
            after, current = _tree(self.backend.snapshot()), self.backend.clients()
            if not self._preserved(before, after) or not set(clients) <= set(current):
                raise OwnershipUnavailable("flatpak_preexisting_window_changed")
            new = set(current) - set(clients)
            if len(new) > 1:
                raise OwnershipUnavailable("flatpak_concurrent_new_windows")
            if len(new) == 1:
                xid = next(iter(new))
                if xid in before["windows"]:
                    raise OwnershipUnavailable("flatpak_preexisting_xid_cannot_be_owned")
                if not {16, 19} <= {e["type"] for e in observed if e["window_id"] == xid}:
                    self.backend.wait()
                    continue
                proof = _check_proof(
                    self.backend.proof(xid, deployment, blank=True), xid, blank=True
                )
                if (
                    self.backend.deployment() != deployment
                    or self.backend.snapshot() != after
                    or self.backend.clients() != current
                ):
                    raise OwnershipUnavailable("flatpak_launch_binding_changed")
                observed.extend(self.backend.events())
                self.backend.validate_events(observed, xid, before, after, proof)
                launch = self.backend.launch_record()
                if launch["exit_code"] not in {None, 0}:
                    raise OwnershipUnavailable("flatpak_launcher_failed")
                self.ledger = {
                    "schema": SCHEMA,
                    "source": _source(),
                    "deployment": deployment,
                    "before": before,
                    "clients_before": clients,
                    "window": proof["window"],
                    "frame": proof["frame"],
                    "namespace": proof["namespace"],
                    "launch": launch,
                    "create_map_events": observed,
                    "blank_source": proof["blank_source"],
                    "retired": False,
                }
                return copy.deepcopy(self.ledger)
            self.backend.wait()
        raise OwnershipUnavailable("flatpak_new_window_not_proven")

    def verify(self, *, blank: bool = False) -> dict:
        if self.ledger is None:
            raise OwnershipUnavailable("flatpak_launch_ledger_missing")
        ledger = self.ledger
        events = self.backend.events()
        if any(e["type"] == 17 and e["window_id"] in ledger["before"]["windows"] for e in events):
            raise OwnershipUnavailable("flatpak_preexisting_window_changed")
        if any(e["type"] == 17 and e["window_id"] == ledger["window"]["window_id"] for e in events):
            ledger["retired"] = True
        if (
            ledger["retired"]
            or _source() != ledger["source"]
            or self.backend.deployment() != ledger["deployment"]
        ):
            raise OwnershipUnavailable("flatpak_owned_identity_changed")
        tree = _tree(self.backend.snapshot())
        if not self._preserved(ledger["before"], tree) or not set(ledger["clients_before"]) <= set(
            self.backend.clients()
        ):
            raise OwnershipUnavailable("flatpak_preexisting_window_changed")
        proof = _check_proof(
            self.backend.proof(ledger["window"]["window_id"], ledger["deployment"], blank=blank),
            ledger["window"]["window_id"],
            blank=blank,
        )
        if (
            proof["window"] != ledger["window"]
            or proof["frame"]["identity"] != ledger["frame"]["identity"]
            or proof["frame"]["generation"] != ledger["frame"]["generation"]
            or proof["namespace"] != ledger["namespace"]
        ):
            raise OwnershipUnavailable("flatpak_owned_identity_changed")
        return proof

    def close(self) -> dict:
        if self.ledger is None:
            raise OwnershipUnavailable("flatpak_launch_ledger_missing")
        ledger = self.ledger
        if _source() != ledger["source"] or self.backend.deployment() != ledger["deployment"]:
            raise OwnershipUnavailable("flatpak_owned_identity_changed")
        events = self.backend.events()
        if any(e["type"] == 17 and e["window_id"] in ledger["before"]["windows"] for e in events):
            raise OwnershipUnavailable("flatpak_preexisting_window_changed")
        if any(e["type"] == 17 and e["window_id"] == ledger["window"]["window_id"] for e in events):
            ledger["retired"] = True
        tree = _tree(self.backend.snapshot())
        xid = ledger["window"]["window_id"]
        if xid not in tree["windows"]:
            if not self._preserved(ledger["before"], tree):
                raise OwnershipUnavailable("flatpak_preexisting_window_changed")
            return {
                "closed": True,
                "polite_close_requested": False,
                "absence": "complete_XQueryTree",
            }
        proof = self.verify()
        # The backend repeats the fixed full proof immediately before WM_DELETE.
        self.backend.polite_close(proof, ledger["deployment"])
        deadline = self.backend.clock() + 3
        while self.backend.clock() < deadline:
            tree = _tree(self.backend.snapshot())
            if not self._preserved(ledger["before"], tree):
                raise OwnershipUnavailable("flatpak_preexisting_window_changed")
            if xid not in tree["windows"]:
                ledger["retired"] = True
                return {
                    "closed": True,
                    "polite_close_requested": True,
                    "absence": "complete_XQueryTree",
                }
            self.backend.wait()
        return {
            "closed": False,
            "polite_close_requested": True,
            "reason": "flatpak_owned_window_still_present",
        }


class _Attributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("class_", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class _WindowEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("parent", ctypes.c_ulong),
        ("window", ctypes.c_ulong),
    ]


class _Event(ctypes.Union):
    _fields_ = [("window", _WindowEvent), ("pad", ctypes.c_long * 24)]


class _XError(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong), ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte), ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


_X_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_XError))
_X_ERROR_LOCK = threading.RLock()


def _checked_x(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        with self._errors(method.__name__):
            return method(self, *args, **kwargs)
    return checked


class _XDisplay:
    def __init__(self):
        self.lib = ctypes.CDLL("libX11.so.6")
        lib = self.lib
        lib.XOpenDisplay.argtypes, lib.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
        lib.XDefaultRootWindow.argtypes, lib.XDefaultRootWindow.restype = (
            [ctypes.c_void_p],
            ctypes.c_ulong,
        )
        lib.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.XGetWindowAttributes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_Attributes),
        ]
        lib.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]
        lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.XPending.argtypes = [ctypes.c_void_p]
        lib.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Event)]
        lib.XFree.argtypes, lib.XCloseDisplay.argtypes = [ctypes.c_void_p], [ctypes.c_void_p]
        lib.XSetErrorHandler.argtypes, lib.XSetErrorHandler.restype = [ctypes.c_void_p], ctypes.c_void_p
        self.display = lib.XOpenDisplay(None)
        if not self.display:
            raise OwnershipUnavailable("flatpak_x11_unavailable")
        self.root = lib.XDefaultRootWindow(self.display)
        self.armed, self.selected = False, set()
        self._poisoned, self._error_handlers = False, []

    @contextmanager
    def _errors(self, stage="internal"):
        # XSetErrorHandler is process-global. Only this connection's errors are
        # trapped; another Display retains its exact previous/default handler.
        with _X_ERROR_LOCK:
            if self._poisoned or len(self._error_handlers) >= MAX_WINDOWS:
                raise OwnershipUnavailable("flatpak_x11_connection_unavailable")
            errors, previous, published = [], None, threading.Event()
            installer = threading.get_ident()

            def handle(display, event):
                try:
                    if display == self.display:
                        if not errors:
                            error = event.contents
                            errors.append((error.error_code, error.request_code,
                                           error.minor_code, error.resourceid, error.serial))
                        return 0
                    # The setter sends no protocol requests. An unrelated
                    # thread may enter before its returned pointer is published.
                    if (threading.get_ident() == installer and not published.is_set()
                            or not published.wait(1) or previous is None):
                        os._exit(1)
                    return previous(display, event)
                except BaseException:
                    os._exit(1)  # Never propagate a Python exception into Xlib.

            callback = _X_ERROR_HANDLER(handle)
            # Keep C callback stubs alive even for a foreign callback already
            # entering at restoration. The private connection's lifetime is bounded.
            self._error_handlers.append(callback)
            old = None
            installed = False
            try:
                old = self.lib.XSetErrorHandler(callback)
                installed = True
                previous = _X_ERROR_HANDLER(old) if old else None
                published.set()
                if previous is None:
                    raise OwnershipUnavailable("flatpak_x11_error_handler_unavailable")
                self.lib.XSync(self.display, False)
                if errors:
                    self._poisoned = True
                    raise _XProtocolUnavailable(errors[0], stage + ":pre_sync")
                try:
                    yield
                finally:
                    # Includes asynchronous XSelectInput errors. Never return a
                    # complete tree/event witness with an unprocessed X error.
                    self.lib.XSync(self.display, False)
                    if errors:
                        self._poisoned = True
                        raise _XProtocolUnavailable(errors[0], stage + ":post_sync")
            except BaseException:
                # Subscriptions and consumed events cannot be rolled back. No
                # subsequent operation on this connection can establish ownership.
                self._poisoned = True
                raise
            finally:
                published.set()
                if installed:
                    replaced = self.lib.XSetErrorHandler(old)
                    if replaced != ctypes.cast(callback, ctypes.c_void_p).value:
                        self.lib.XSetErrorHandler(replaced)
                        self._poisoned = True
                        raise OwnershipUnavailable("flatpak_x11_error_handler_changed")

    @_checked_x
    def snapshot(self, deadline: float) -> dict:
        pending, rows = [(self.root, 0)], {}
        while pending:
            xid, depth = pending.pop()
            if (
                time.monotonic() >= deadline
                or len(rows) >= MAX_WINDOWS
                or depth > 64
                or str(xid) in rows
            ):
                raise OwnershipUnavailable("flatpak_window_tree_incomplete")
            attrs, root, parent = _Attributes(), ctypes.c_ulong(), ctypes.c_ulong()
            children, count = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
            if not self.lib.XGetWindowAttributes(self.display, xid, ctypes.byref(attrs)):
                raise OwnershipUnavailable("flatpak_window_tree_unavailable")
            try:
                if (
                    not self.lib.XQueryTree(
                        self.display,
                        xid,
                        ctypes.byref(root),
                        ctypes.byref(parent),
                        ctypes.byref(children),
                        ctypes.byref(count),
                    )
                    or root.value != self.root
                    or count.value > MAX_WINDOWS
                ):
                    raise OwnershipUnavailable("flatpak_window_tree_incomplete")
                pending.extend((children[index], depth + 1) for index in range(count.value))
            finally:
                if children:
                    self.lib.XFree(children)
            rows[str(xid)] = {
                "parent": str(parent.value),
                "map_state": attrs.map_state,
                "override_redirect": bool(attrs.override_redirect),
            }
        return {"root": str(self.root), "windows": rows, "complete": True}

    def arm(self, tree: dict) -> None:
        pending = set(tree["windows"]) - self.selected
        with self._errors("arm"):
            for xid in pending:
                self.lib.XSelectInput(self.display, int(xid), (1 << 19) | (1 << 17))
        self.selected.update(pending)
        self.armed = True

    def select(self, xid: str) -> None:
        with self._errors("select"):
            if xid not in self.selected:
                self.lib.XSelectInput(self.display, int(xid), (1 << 19) | (1 << 17))
        self.selected.add(xid)

    def events(self) -> list[dict]:
        with self._errors("events"):
            rows, selected = self._events()
        self.selected = selected
        return rows

    def _events(self):
        self.lib.XSync(self.display, False)
        rows, count = [], 0
        while self.lib.XPending(self.display):
            if count >= MAX_WINDOWS:
                raise OwnershipUnavailable("flatpak_window_events_incomplete")
            event = _Event()
            self.lib.XNextEvent(self.display, ctypes.byref(event))
            count += 1
            row = event.window
            if row.type in {16, 17, 18, 19, 21}:
                # StructureNotify and SubstructureNotify can report the same event.
                item = {
                    "type": row.type,
                    "window_id": str(row.window),
                    "parent": str(row.parent),
                    "synthetic": bool(row.send_event),
                }
                if item["synthetic"]:
                    # Any client may send these (ICCCM withdrawal sends a synthetic
                    # UnmapNotify to the root), so they are never evidence: they
                    # neither prove nor disturb a lifetime or a subscription.
                    continue
                if not (0 < row.window < 2**32 and 0 < row.parent < 2**32):
                    raise OwnershipUnavailable("flatpak_window_events_incomplete")
                rows.append(item)
        # Decide subscriptions only from a complete genuine batch. A Create
        # followed by Destroy is an ended lifetime, not a selectable window.
        # Removing that lifetime also permits a later Create reusing its XID.
        selected, pending = set(self.selected), {}
        for item in rows:
            xid = item["window_id"]
            if item["type"] == 17:
                selected.discard(xid)
                pending.pop(xid, None)
            elif item["type"] == 16 and xid not in selected:
                pending[xid] = None
        for xid in pending:
            self.lib.XSelectInput(self.display, int(xid), (1 << 19) | (1 << 17))
            selected.add(xid)
        return rows, selected


class _SystemBackend:
    def __init__(self, *, browser="firefox"):
        if browser not in {"firefox", "chromium"}:
            raise OwnershipUnavailable("flatpak_descriptor_unavailable")
        self.browser = browser
        self.application = "org.mozilla.firefox" if browser == "firefox" else "org.chromium.Chromium"
        self.executable_relative = "files/lib/firefox/firefox-bin" if browser == "firefox" else "files/chromium/chrome"
        spec = importlib.util.spec_from_file_location(
            "_omni_setup_native", REPO / "portal/native_ui.py"
        )
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi, GLib

        self.bridge = self.native._AtspiBridge(Atspi, GLib, "owned-flatpak-firefox-setup")
        self.identity = self.native._identity_helper()
        self.x = _XDisplay()
        self.launch_process = None
        self.launch_streams = []
        self.launch_identity = None
        self.deadline = time.monotonic() + 30
        self.lock_fd = None

    clock = staticmethod(time.monotonic)
    wait = staticmethod(lambda: time.sleep(0.1))

    def begin(self):
        self.deadline = time.monotonic() + 30
        name = hashlib.sha256(os.environ.get("DISPLAY", "").encode()).hexdigest()[:16]
        path = Path(f"/run/user/{os.getuid()}/omni-owned-flatpak-{name}.lock")
        self.lock_fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        info = os.fstat(self.lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise OwnershipUnavailable("flatpak_setup_lock_unavailable")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OwnershipUnavailable("flatpak_setup_lock_busy") from None

    def end(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def command(self, args):
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(0.05, min(3, self.deadline - time.monotonic())),
        )  # noqa: S603
        if result.returncode or len(result.stdout) > 65536:
            raise OwnershipUnavailable("flatpak_installed_identity_unavailable")
        return result.stdout.strip()

    def deployment(self) -> dict:
        ref = self.command(["/usr/bin/flatpak", "info", "--show-ref", self.application])
        match = re.fullmatch(r"app/" + re.escape(self.application) + r"/(aarch64|x86_64)/stable", ref)
        commit = self.command(["/usr/bin/flatpak", "info", "--show-commit", self.application])
        if match is None or re.fullmatch(r"[0-9a-f]{64}", commit) is None:
            raise OwnershipUnavailable("flatpak_deployment_unavailable")
        location = Path(
            self.command(["/usr/bin/flatpak", "info", "--show-location", self.application])
        )
        expected = Path("/var/lib/flatpak/app") / self.application / match[1] / "stable" / commit
        if location.resolve(strict=True) != expected.resolve(strict=True):
            raise OwnershipUnavailable("flatpak_deployment_mismatch")
        executable = location / self.executable_relative
        entry = location / ("files/share/applications/" + self.application + ".desktop")
        stamp = executable.stat()
        with entry.open("rb") as source:
            raw = source.read(65537)
        if (
            stamp.st_uid != 0
            or not stat.S_ISREG(stamp.st_mode)
            or stamp.st_mode & 0o022
            or len(raw) > 65536
        ):
            raise OwnershipUnavailable("flatpak_deployment_mismatch")
        launcher = None
        if self.browser == "chromium":
            installed = shutil.which("chromium")
            if not installed:
                raise OwnershipUnavailable("flatpak_installed_launcher_unavailable")
            path = Path(installed)
            info, data = path.stat(), path.read_bytes()
            try:
                lines = [line.strip() for line in data.decode("utf-8").splitlines()
                         if line.strip() and not line.lstrip().startswith("#")]
                words = shlex.split(lines[0]) if len(lines) == 1 else []
            except (UnicodeError, ValueError):
                raise OwnershipUnavailable("flatpak_installed_launcher_changed") from None
            if words[:1] == ["exec"]:
                words = words[1:]
            if (path.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022
                    or not stat.S_ISREG(info.st_mode) or len(data) > 4096 or path.stat() != info
                    or not data.startswith((b"#!/bin/bash\n", b"#!/bin/sh\n"))
                    or words[:1] not in (["flatpak"], ["/usr/bin/flatpak"])
                    or words[1:3] != ["run", self.application] or words[-1:] != ["$@"]
                    or words[3:-1] not in ([], ["--no-sandbox"])):
                raise OwnershipUnavailable("flatpak_installed_launcher_changed")
            launcher = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                        "identity": self.identity._stamp(info), "existing_flags": words[3:-1]}
        return {
            **({"launcher": launcher} if launcher else {}),
            "ref": ref,
            "arch": match[1],
            "commit": commit,
            "location": str(location),
            "executable": {
                "device": stamp.st_dev,
                "inode": stamp.st_ino,
                "uid": stamp.st_uid,
                "mode": stamp.st_mode,
                "size": stamp.st_size,
            },
            "desktop_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def snapshot(self):
        if time.monotonic() >= self.deadline:
            raise OwnershipUnavailable("flatpak_setup_deadline")
        return self.x.snapshot(self.deadline)

    def clients(self):
        return self.identity.client_windows(os.environ, timeout_s=self.deadline - time.monotonic())

    def arm(self, tree):
        self.x.arm(tree)

    def events(self):
        return self.x.events()

    def launch(self, deployment):
        if self.launch_process is not None:
            raise OwnershipUnavailable("flatpak_launch_already_attempted")
        # Kept for the dedicated helper's lifetime; closed at process exit.
        self.launch_streams = [tempfile.TemporaryFile(), tempfile.TemporaryFile()]  # noqa: SIM115
        command = ([deployment["launcher"]["path"], "--new-window", "about:blank"]
                   if self.browser == "chromium" else [
                "/usr/bin/flatpak",
                "run",
                "--branch=stable",
                "--arch=" + deployment["arch"],
                "--command=firefox",
                "org.mozilla.firefox",
                "--new-window",
                "about:blank",
            ])
        if self.deployment() != deployment:
            raise OwnershipUnavailable("flatpak_launch_binding_changed")
        self.launch_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.launch_streams[0],
            stderr=self.launch_streams[1],
            start_new_session=True,
        )  # noqa: S603 - fixed ordinary installed launcher only.
        pid = self.launch_process.pid
        try:
            start = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            start = None  # A short-lived remote launcher is never the owner.
        self.launch_identity = {
            "pid": pid,
            "start": start,
            "ordinary_profile": True,
            "requested_uri": "about:blank",
        }
        return self.launch_record()

    def launch_record(self):
        if self.launch_process is None:
            return {"attempted": False}
        result = {
            **(self.launch_identity or {}),
            "exit_code": self.launch_process.poll() if self.launch_process else None,
        }
        for name, stream in zip(("stdout", "stderr"), self.launch_streams, strict=True):
            stream.seek(0)
            data = stream.read(65537)
            result[name] = {
                "prefix_bytes": min(len(data), 65536),
                "prefix_sha256": hashlib.sha256(data[:65536]).hexdigest(),
                "truncated": len(data) > 65536,
                "raw_text_retained": False,
            }
        return result

    def proof(self, xid, deployment, *, blank):
        state = self.bridge.observe(include_windows=True)
        if self.bridge._inactive_namespace_error:
            raise OwnershipUnavailable("flatpak_native_mapping_incomplete")
        rows = [node for node in state["nodes"] if self.bridge.window_ids.get(node["id"]) == xid]
        if len(rows) != 1:
            raise OwnershipUnavailable("flatpak_native_frame_not_unique")
        node = rows[0]
        frame = self.bridge.windows[node["id"]]
        if (
            node.get("role") != "window"
            or not node.get("visible")
            or not node.get("enabled")
            or node.get("sensitive")
        ):
            raise OwnershipUnavailable("flatpak_native_frame_ineligible")
        raw = self.bridge._read(lambda: str(frame.app.bus_name) + ":" + str(frame.path))
        binding = self.bridge._inactive_namespace_bindings.get(raw)
        if (
            binding is None
            and self.bridge._provider_binding
            and self.bridge._provider_binding["frame"] == raw
        ):
            proof = self.bridge._provider_binding["proof"]
        elif binding:
            proof = binding["proof"]
        else:
            raise OwnershipUnavailable("flatpak_native_namespace_unavailable")
        owner = proof["values"]["owner"]
        if proof["identity"] != {"window_id": xid, "process_id": owner["pid"]}:
            raise OwnershipUnavailable("flatpak_native_owner_mismatch")
        host = Path(f"/proc/{owner['pid']}")
        stamp = self.bridge._read(lambda: (host / "exe").stat())
        actual = {
            "device": stamp.st_dev,
            "inode": stamp.st_ino,
            "uid": stamp.st_uid,
            "mode": stamp.st_mode,
            "size": stamp.st_size,
        }
        with (host / "root/.flatpak-info").open("rb") as metadata:
            raw_metadata = self.bridge._read(lambda: metadata.read(65537))
        if (
            len(raw_metadata) > 65536
            or hashlib.sha256(raw_metadata).hexdigest()
            != proof["values"]["metadata"]["host"]["sha256"]
        ):
            raise OwnershipUnavailable("flatpak_deployment_metadata_changed")
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(raw_metadata.decode())
        expected = {
            "app-path": deployment["location"] + "/files",
            "app-commit": deployment["commit"],
            "arch": deployment["arch"],
            "branch": "stable",
        }
        if (
            actual != deployment["executable"]
            or config.get("Application", "name") != self.application
            or any(config.get("Instance", key) != value for key, value in expected.items())
        ):
            raise OwnershipUnavailable("flatpak_running_deployment_mismatch")
        if self.bridge._read(lambda: self.identity.xres_owner(xid, os.environ)) != owner:
            raise OwnershipUnavailable("flatpak_native_owner_changed")
        geometry = self.bridge._native_rect(frame)
        # Exact surface, or exact surface minus declared client-side-decoration margins.
        if geometry is None or not self.bridge._read(
            lambda: self.native._frame_matches_surface(geometry, proof["identity"])
        ):
            raise OwnershipUnavailable("flatpak_native_geometry_unproven")
        # Revalidation is a separate bounded read pass, not a continuation of the
        # observation's 5 s budget; it stays inside this setup transaction's deadline.
        self.bridge.deadline = min(self.deadline, time.monotonic() + 5)
        blank_source = self.blank_document(frame) if blank else None
        self.bridge._refresh_provider_binding()
        self.bridge._refresh_inactive_namespace_bindings()
        if (
            self.bridge._native_rect(frame) != geometry
            or self.bridge._read(lambda: self.identity.xres_owner(xid, os.environ)) != owner
            or self.bridge._read(lambda: (host / "exe").stat()) != stamp
        ):
            raise OwnershipUnavailable("flatpak_owned_identity_changed")
        return {
            "window": {
                "window_id": xid,
                "process_id": owner["pid"],
                "process_start": owner["start"],
            },
            "frame": {
                "identity": raw,
                "generation": node["generation"],
                "screen_geometry": geometry,
            },
            "namespace": proof,
            "blank_source": blank_source,
        }

    def blank_document(self, frame):
        bridge = self.bridge
        seen, documents = set(), []

        def visit(node, depth=0):
            raw = bridge._read(lambda: str(node.app.bus_name) + ":" + str(node.path))
            if raw in seen or depth > bridge.MAX_DEPTH or len(seen) >= bridge.MAX_NODES:
                raise OwnershipUnavailable("flatpak_blank_document_incomplete")
            seen.add(raw)
            bridge._clear_cache(node)
            role, flags = bridge._read(node.get_role_name), bridge._read(node.get_state_set)
            if flags is None or flags.contains(bridge.atspi.StateType.DEFUNCT):
                raise OwnershipUnavailable("flatpak_blank_document_unavailable")
            if role == "document web":
                if all(
                    flags.contains(flag)
                    for flag in (bridge.atspi.StateType.VISIBLE, bridge.atspi.StateType.SHOWING)
                ):
                    documents.append(node)
                return
            count = bridge._count(node)
            if count > bridge.MAX_CHILDREN or role == "password text" and count:
                raise OwnershipUnavailable("flatpak_blank_document_incomplete")
            if role != "password text":
                for index in range(count):
                    visit(
                        bridge._read(lambda index=index: node.get_child_at_index(index)), depth + 1
                    )

        visit(frame)
        if len(documents) != 1:
            raise OwnershipUnavailable("flatpak_blank_document_not_unique")
        node = documents[0]
        interface = bridge._read(node.get_document_iface)
        getter = bridge.atspi.Document.get_document_attribute_value

        def uri():
            bridge._clear_cache(node)
            flags = bridge._read(node.get_state_set)
            if (
                bridge._read(node.get_role_name) != "document web"
                or flags is None
                or flags.contains(bridge.atspi.StateType.DEFUNCT)
                or not all(
                    flags.contains(flag)
                    for flag in (bridge.atspi.StateType.VISIBLE, bridge.atspi.StateType.SHOWING)
                )
            ):
                raise OwnershipUnavailable("flatpak_blank_document_changed")
            return bridge._read(lambda: getter(interface, "DocURL"))

        if uri() != "about:blank" or uri() != "about:blank":
            raise OwnershipUnavailable("flatpak_blank_document_mismatch")
        return {"source": "atspi.Document.DocURL", "url": "about:blank"}

    def validate_events(self, events, xid, before, after, proof):
        owner = {"pid": proof["window"]["process_id"], "start": proof["window"]["process_start"]}
        transient, client = set(), None
        for event in events:
            value = event["window_id"]
            if value in before["windows"]:
                raise OwnershipUnavailable("flatpak_concurrent_preexisting_event")
            if value not in after["windows"]:
                # A short-lived window created and destroyed outside the snapshot is
                # attributable only by X protocol: its XID lies in the resource range
                # of the still-connected client that owns the proven window.
                if client is None:
                    client = self.identity.xres_client_range(xid, os.environ)
                if int(value) & ~client["mask"] != client["base"]:
                    raise OwnershipUnavailable("flatpak_unknown_window_event")
                transient.add(value)
                continue
            related, current = value == xid, value
            for _ in range(65):
                current = after["windows"].get(current, {}).get("parent")
                if current == xid:
                    related = True
                if not current or current == after["root"]:
                    break
            current = xid
            for _ in range(65):
                current = after["windows"].get(current, {}).get("parent")
                if current == value:
                    related = True
                if not current or current == after["root"]:
                    break
            if not related and self.identity.xres_owner(value, os.environ) != owner:
                raise OwnershipUnavailable("flatpak_concurrent_unknown_window")
        if transient:
            current = _tree(self.snapshot())
            if (any(value in current["windows"] for value in transient)
                    or self.identity.xres_client_range(xid, os.environ) != client):
                raise OwnershipUnavailable("flatpak_unknown_window_event")

    def polite_close(self, proof, deployment):
        if self.proof(proof["window"]["window_id"], deployment, blank=False) != proof:
            raise OwnershipUnavailable("flatpak_close_identity_changed")
        # Reuse only the fixed WM_DELETE primitive; its ordinary PID resolver is
        # deliberately not bypassed by a caller-provided callback or raw XID.
        spec = importlib.util.spec_from_file_location(
            "_omni_setup_close", REPO / "runtime/owned_window_close.py"
        )
        close = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(close)
        lib, display = close._xlib(), None
        try:
            display = lib.XOpenDisplay(None)
            if not display:
                raise OwnershipUnavailable("flatpak_close_display_unavailable")
            protocols = ctypes.POINTER(ctypes.c_ulong)()
            count = ctypes.c_int()
            atom, delete = (
                lib.XInternAtom(display, b"WM_PROTOCOLS", True),
                lib.XInternAtom(display, b"WM_DELETE_WINDOW", True),
            )
            try:
                if (
                    not atom
                    or not delete
                    or not lib.XGetWMProtocols(
                        display,
                        int(proof["window"]["window_id"]),
                        ctypes.byref(protocols),
                        ctypes.byref(count),
                    )
                    or not 0 < count.value <= 64
                    or not protocols
                    or delete not in [protocols[i] for i in range(count.value)]
                ):
                    raise OwnershipUnavailable("flatpak_polite_close_unavailable")
                if self.proof(proof["window"]["window_id"], deployment, blank=False) != proof:
                    raise OwnershipUnavailable("flatpak_close_identity_changed")
                event = close._Event()
                event.client.type = 33
                event.client.send_event = True
                event.client.display = display
                event.client.window = int(proof["window"]["window_id"])
                event.client.message_type = atom
                event.client.format = 32
                event.client.data.l[0] = delete
                if not lib.XSendEvent(display, event.client.window, False, 0, ctypes.byref(event)):
                    raise OwnershipUnavailable("flatpak_polite_close_unavailable")
                lib.XFlush(display)
            finally:
                if protocols:
                    lib.XFree(protocols)
        finally:
            if display:
                lib.XCloseDisplay(display)


class OwnedFlatpakFirefox:
    """Host-only handle to the private ledger retained by one fixed child."""

    def __init__(self, environment, *, browser="firefox"):
        if browser not in {"firefox", "chromium"}:
            raise OwnershipUnavailable("flatpak_descriptor_unavailable")
        self.browser = browser
        self.environment, self.process, self.ledger, self.last_receipt = (
            dict(environment),
            None,
            None,
            None,
        )
        self._transport_lock = threading.RLock()
        self._poisoned, self._sequence = False, 0

    def _stop_helper(self):
        # This is the dedicated CPU protocol child, never the browser or its group.
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _poison(self, code):
        self._poisoned = True
        self.last_receipt = {
            "schema": SCHEMA,
            "ok": False,
            "code": code,
            "transport_terminal": True,
        }
        self._stop_helper()
        raise OwnershipUnavailable(code)

    def _request(self, op):
        with self._transport_lock:
            if self._poisoned:
                raise OwnershipUnavailable("flatpak_ownership_transport_closed")
            if not isinstance(op, str) or op not in {"create", "verify", "close"}:
                self._poison("flatpak_ownership_protocol_invalid")
            self._sequence += 1
            try:
                if self.process is None:
                    self.process = subprocess.Popen(
                        [
                            "/usr/bin/python3",
                            "-I",
                            "-u",
                            str(Path(__file__).resolve()),
                            "--owned-helper" if self.browser == "firefox" else "--owned-helper-chromium",
                        ],
                        env=self.environment,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )  # noqa: S603 - exactly one dedicated CPU helper; no respawn after failure.
                if self.process.poll() is not None:
                    self._poison("flatpak_ownership_helper_unavailable")
                request = {"schema": SCHEMA, "op": op, "request_id": self._sequence}
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                self.process.stdin.flush()
                deadline, data = time.monotonic() + 40, bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(self.process.stdout, selectors.EVENT_READ)
                    while time.monotonic() < deadline:
                        if not selector.select(max(0, deadline - time.monotonic())):
                            self._poison("flatpak_ownership_transport_timeout")
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                        if not chunk:
                            self._poison("flatpak_ownership_transport_eof")
                        data.extend(chunk)
                        if len(data) > MAX_REPLY:
                            self._poison("flatpak_ownership_reply_too_large")
                        if data.endswith(b"\n"):
                            result = json.loads(data)
                            key = {"create": "ledger", "verify": "proof", "close": "receipt"}[op]
                            common = {"schema", "op", "request_id", "ok"}
                            extra = {"x11_error"} if isinstance(result, dict) and "x11_error" in result else set()
                            if (
                                not isinstance(result, dict)
                                or result.get("schema") != SCHEMA
                                or result.get("op") != op
                                or type(result.get("request_id")) is not int
                                or result["request_id"] != self._sequence
                                or type(result.get("ok")) is not bool
                                or set(result)
                                != common | ({key} if result["ok"] else {"code", "launch"} | extra)
                                or result["ok"]
                                and not isinstance(result[key], dict)
                                or not result["ok"]
                                and (
                                    not isinstance(result["code"], str)
                                    or re.fullmatch(r"[a-z_][a-z0-9_]{0,127}", result["code"]) is None
                                    or result["launch"] is not None
                                    and not isinstance(result["launch"], dict)
                                    or extra and (result["code"] != "flatpak_x11_protocol_error"
                                                  or not _valid_x11_error(result["x11_error"]))
                                )
                            ):
                                self._poison("flatpak_ownership_protocol_invalid")
                            self.last_receipt = result
                            if not result["ok"]:
                                raise OwnershipUnavailable(result["code"])
                            return result
                self._poison("flatpak_ownership_transport_timeout")
            except OwnershipUnavailable:
                raise
            except (OSError, ValueError, TypeError, KeyError):
                self._poison("flatpak_ownership_protocol_invalid")

    def open_blank(self):
        self.ledger = self._request("create")["ledger"]
        return copy.deepcopy(self.ledger)

    def verify(self):
        return self._request("verify")["proof"]

    def close(self):
        try:
            return self._request("close")["receipt"]
        finally:
            self._poisoned = True
            self._stop_helper()


def _main():
    if sys.argv[1:] not in (["--owned-helper"], ["--owned-helper-chromium"]):
        return 2
    backend, transaction, sequence = None, None, 0
    for raw in iter(lambda: sys.stdin.buffer.readline(257), b""):
        request = None
        try:
            request = json.loads(raw) if len(raw) <= 256 else None
            if (
                not isinstance(request, dict)
                or set(request) != {"schema", "op", "request_id"}
                or request["schema"] != SCHEMA
                or not isinstance(request["op"], str)
                or request["op"] not in {"create", "verify", "close"}
                or type(request["request_id"]) is not int
                or request["request_id"] != sequence + 1
            ):
                raise OwnershipUnavailable("flatpak_helper_operation_invalid")
            sequence = request["request_id"]
            if backend is None:
                if request["op"] != "create":
                    raise OwnershipUnavailable("flatpak_launch_ledger_missing")
                backend = (_SystemBackend() if sys.argv[1:] == ["--owned-helper"]
                           else _SystemBackend(browser="chromium"))
                transaction = _Transaction(backend)
            backend.begin()
            op = request["op"]
            result = (
                {"ledger": transaction.create()}
                if op == "create"
                else {"proof": transaction.verify(blank=True)}
                if op == "verify"
                else {"receipt": transaction.close()}
            )
            output = {
                "schema": SCHEMA,
                "op": request["op"],
                "request_id": sequence,
                "ok": True,
                **result,
            }
        except Exception as exc:
            code = getattr(exc, "code", None)
            output = {
                "schema": SCHEMA,
                "op": request.get("op")
                if isinstance(request, dict) and isinstance(request.get("op"), str)
                else None,
                "request_id": request.get("request_id")
                if isinstance(request, dict) and type(request.get("request_id")) is int
                else None,
                "ok": False,
                "code": code
                if isinstance(code, str) and re.fullmatch(r"[a-z_][a-z0-9_]{0,127}", code)
                else "flatpak_ownership_unavailable",
                "launch": backend.launch_record() if backend else None,
            }
            if (isinstance(exc, _XProtocolUnavailable) and output["code"] == "flatpak_x11_protocol_error"
                    and _valid_x11_error(exc.diagnostic())):
                output["x11_error"] = exc.diagnostic()
        finally:
            if backend:
                backend.end()
        print(json.dumps(output, sort_keys=True, allow_nan=False), flush=True)
        if len(raw) > 256:
            break
        if isinstance(request, dict) and request.get("op") == "close":
            break
    # EOF never closes a browser: task drain/ownership may be unknown.
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
