"""Exact owned ordinary-app setup and polite cleanup for text task evaluation."""
from __future__ import annotations

import copy
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from runtime.acceptance_native_support import (
    CapabilityBlocked,
    OwnedDesktop,
    OwnedWindow,
    _spreadsheet_environment,
    _spreadsheet_process_ids,
    _verify_spreadsheet_process,
    class_windows,
    command,
    process_start,
)
from runtime.owned_window_close import request_owned_window_close

GNOME_TEXT_EDITOR = Path("/usr/bin/gnome-text-editor")
SNAP_CHROMIUM = Path("/snap/bin/chromium")
SNAP_DISPATCHER = Path("/usr/bin/snap")


def _snap_chromium_launcher() -> str:
    """Select the installed alias; browser ownership still requires normal guards."""
    def identity(value):
        return tuple(getattr(value, key) for key in (
            "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns"))

    try:
        resolved = SNAP_CHROMIUM.resolve(strict=True)
        link, executable = SNAP_CHROMIUM.lstat(), resolved.stat()
        if (resolved != SNAP_DISPATCHER or not SNAP_CHROMIUM.is_symlink()
                or link.st_uid != 0 or executable.st_uid != 0
                or not stat.S_ISREG(executable.st_mode) or executable.st_mode & 0o022
                or not os.access(SNAP_CHROMIUM, os.X_OK)):
            raise ValueError("Invalid installed Snap alias")
        with resolved.open("rb") as source:
            native = source.read(4) == b"\x7fELF"
        if (not native or SNAP_CHROMIUM.resolve(strict=True) != resolved
                or identity(SNAP_CHROMIUM.lstat()) != identity(link)
                or identity(resolved.stat()) != identity(executable)):
            raise ValueError("Installed Snap alias changed")
    except (OSError, ValueError, RuntimeError):
        raise CapabilityBlocked("Installed Snap Chromium launcher unavailable") from None
    return str(SNAP_CHROMIUM)


def _verify_native_editor_process(pid: str, executable: Path, expected_start: str,
                                  *, proc_root: Path = Path("/proc")) -> None:
    """Bind a native editor handoff by UID/executable/start, never title or argv."""
    try:
        start = process_start(pid)
        process = proc_root / pid
        if (start != expected_start or process.stat().st_uid != os.getuid()
                or (process / "exe").resolve(strict=True) != executable.resolve(strict=True)
                or process_start(pid) != start):
            raise CapabilityBlocked("Native editor process identity is unproven")
    except (OSError, ValueError):
        raise CapabilityBlocked("Native editor process identity is unavailable") from None


def owned_foreground(windows: list[OwnedWindow], identity: dict[str, Any], environment: dict[str, str]) -> bool:
    matches = [window for window in windows if window.pid == str(identity.get("process_id"))
               and process_start(window.pid) == window.process_start]
    if any(window.window_id == identity.get("window_id") for window in matches):
        return True
    if not matches:
        return False
    value = command(["xprop", "-id", identity["window_id"], "WM_TRANSIENT_FOR"], environment)
    match = re.fullmatch(r"WM_TRANSIENT_FOR\(WINDOW\): window id # (0x[0-9a-fA-F]+)\s*", value)
    return bool(match and any(window.window_id == str(int(match[1], 16)) for window in matches))


def managed_inventory(environment: dict[str, str]) -> dict[str, Any]:
    """Read managed XIDs/PID starts only, never titles or unrelated UI content."""
    try:
        value = command(["xprop", "-root", "_NET_CLIENT_LIST"], environment)
        match = re.fullmatch(r"_NET_CLIENT_LIST\(WINDOW\): window id #\s*(.*)", value)
        if not match:
            raise RuntimeError("Managed-window list unavailable")
        identifiers = match[1].split(", ") if match[1] else []
        if len(identifiers) > 256 or any(not re.fullmatch(r"0x[0-9a-fA-F]+", item) for item in identifiers):
            raise RuntimeError("Managed-window inventory invalid or oversized")
        windows = {}
        for item in identifiers:
            xid = str(int(item, 16))
            try:
                pid = command(["xdotool", "getwindowpid", xid], environment)
                if not pid.isdecimal():
                    raise RuntimeError("Managed window PID unavailable")
                windows[xid] = {"pid": pid, "start_ticks": process_start(pid)}
            except (OSError, RuntimeError, subprocess.SubprocessError):
                windows[xid] = {"identity_available": False}
        return {"available": True, "windows": windows, "coverage": "managed_client_windows_only"}
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {"available": False, "reason": "managed_window_inventory_unavailable"}


def unrelated_preservation(before: dict, after: dict) -> dict:
    if not before.get("available") or not after.get("available"):
        return {"status": "unavailable", "verified": False}
    changed = [xid for xid, identity in before["windows"].items()
               if xid not in after["windows"] or "pid" in identity and identity != after["windows"][xid]]
    unknown = [xid for xid, identity in before["windows"].items() if "pid" not in identity]
    return {"status": "failed" if changed else "unavailable" if unknown else "verified",
            "verified": not changed and not unknown, "changed_or_missing_xids": changed,
            "unknown_identity_xids": unknown, "coverage": "preexisting_managed_client_windows_only"}


class TaskDesktop(OwnedDesktop):
    """Retain launch identity before failure; never kill a browser/shared app."""

    def __init__(self, environment: dict[str, str], *, editor: str = "gedit", lifecycle=None) -> None:
        if editor not in {"gedit", "gnome-text-editor"}:
            raise ValueError("Unsupported explicit native editor")
        super().__init__(environment)
        self.editor = editor
        self.launches: list[dict[str, Any]] = []
        self._flatpak_owners: dict[int, Any] = {}
        self._controlled_close_attempts: set[tuple[str, str, str]] = set()
        self._lifecycle = lifecycle

    def _checkpoint(self, phase, *, registration=None):
        """A live runner supplies a synchronous durable publisher, never input."""
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is not None:
            launches = self.launches
            if registration is not None:
                launch, window = registration
                launches = [copy.deepcopy(row) for row in self.launches]
                index = next(number for number, row in enumerate(self.launches) if row is launch)
                launches[index].update(registered=True, pid=window.pid, process_start=window.process_start,
                                       window_id=window.window_id)
            lifecycle.publish_launches(launches, self.windows, phase=phase)

    def open_application(self, binary, args, application, window_class, *, stdin=None,
                         standalone=False, log_path=None, child_environment=None, expected_executable=None):
        executable = shutil.which(binary)
        if not executable:
            raise CapabilityBlocked("Installed application unavailable: " + application)
        before = class_windows(window_class, self.environment)
        launch = {"application": application, "window_class": window_class,
                  "before_xids": sorted(before), "registered": False}
        if expected_executable is not None:
            expected_executable = Path(expected_executable).resolve(strict=True)
            if Path(executable).resolve(strict=True) != expected_executable:
                raise CapabilityBlocked("Native editor launcher differs from installed executable")
            launch["expected_executable"] = str(expected_executable)
        self.launches.append(launch)
        self._checkpoint("launch_intent")
        log = log_path.open("ab") if log_path else subprocess.DEVNULL
        if log_path:
            self._logs.append(log)
        try:
            process = subprocess.Popen([executable, *args], env=child_environment or self.environment,
                stdout=subprocess.DEVNULL, stderr=log, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                start_new_session=True)  # noqa: S603
        except OSError:
            launch["spawn_failed_before_process"] = True
            self._checkpoint("spawn_failed")
            raise
        launch["launch_pid"] = process.pid
        try:
            launch["launch_start"] = process_start(str(process.pid))
        except OSError:
            launch["launch_start_unavailable"] = True
            self._checkpoint("spawned")
            if standalone:
                raise CapabilityBlocked("Standalone launch identity unavailable") from None
        self._checkpoint("spawned")
        if stdin is not None:
            process.stdin.write(stdin)
            process.stdin.close()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if expected_executable is not None and process.poll() not in {None, 0}:
                raise CapabilityBlocked("Native editor launch failed before ownership proof")
            new = class_windows(window_class, self.environment, visible=True) - before
            previous = launch.get("new_visible_xids")
            launch["new_visible_xids"] = sorted(new)
            if previous != launch["new_visible_xids"]:
                self._checkpoint("candidate")
            if len(new) > 1:
                raise CapabilityBlocked("Concurrent new windows prevent exact ownership")
            if len(new) == 1:
                xid = next(iter(new))
                launch["candidate_xid"] = xid
                self._checkpoint("candidate")
                pid = command(["xdotool", "getwindowpid", xid], self.environment)
                if not pid.isdecimal() or standalone and pid != str(process.pid):
                    raise CapabilityBlocked("New window has no exact host process identity")
                start = process_start(pid)
                if standalone and start != launch["launch_start"]:
                    raise CapabilityBlocked("Standalone process identity changed")
                if expected_executable is not None:
                    _verify_native_editor_process(pid, expected_executable, start)
                    if process.poll() not in {None, 0}:
                        raise CapabilityBlocked("Native editor launch failed before ownership proof")
                window = OwnedWindow(application, xid, pid, start)
                self.register(window)
                self._checkpoint("registered", registration=(launch, window))
                if expected_executable is not None:
                    try:
                        _verify_native_editor_process(pid, expected_executable, start)
                        if process.poll() not in {None, 0}:
                            raise CapabilityBlocked("Native editor launch failed before ownership proof")
                    except CapabilityBlocked:
                        self.windows.remove(window)
                        raise
                launch.update(registered=True, pid=pid, process_start=start, window_id=xid)
                command(["xdotool", "windowactivate", "--sync", xid], self.environment)
                return window
            time.sleep(.15)
        raise CapabilityBlocked("Application did not expose one owned visible window")

    def open_files(self, directory, *, log_path=None):
        return self.open_application("nautilus", ["--new-window", str(directory)], "files", "[Nn]autilus", log_path=log_path)

    def open_blank(self, application: str, work: Path, *, browser: str, log_path: Path):
        if application == "browser":
            if browser == "chromium-snap":
                return self.open_application(_snap_chromium_launcher(), ["--new-window", "about:blank"],
                    "browser", "[Cc]hromium", log_path=log_path)
            if browser in {"flatpak-firefox", "flatpak-chromium"}:
                from runtime.owned_flatpak_firefox import OwnedFlatpakFirefox
                owner = (OwnedFlatpakFirefox(self.environment) if browser == "flatpak-firefox"
                         else OwnedFlatpakFirefox(self.environment, browser="chromium"))
                launch = {"application": "browser", "ownership_kind": "flatpak_firefox" if browser == "flatpak-firefox" else "flatpak_chromium",
                          "registered": False}
                self.launches.append(launch)
                self._flatpak_owners[id(launch)] = owner
                self._checkpoint("launch_intent")
                try:
                    ledger = owner.open_blank()
                    identity = ledger["window"]
                    owner.verify()
                    window = OwnedWindow("browser", identity["window_id"], str(identity["process_id"]),
                                         identity["process_start"], owner=owner)
                    self.windows.append(window)  # Only the private launch ledger grants this registration.
                    launch.update(registered=True, window_id=window.window_id, pid=window.pid,
                                  process_start=window.process_start, ownership_evidence=ledger)
                    self._checkpoint("registered")
                    return window
                finally:
                    launch["last_ownership_receipt"] = owner.last_receipt
                    self._checkpoint("namespace_final")
            return self.open_application(browser, ["--new-window", "about:blank"], "browser",
                "[Ff]irefox" if browser == "firefox" else "[Cc]hromium", log_path=log_path)
        if application == "gedit":
            if self.editor == "gnome-text-editor":
                installed = GNOME_TEXT_EDITOR
                try:
                    executable = installed.resolve(strict=True)
                    if not installed.is_file() or executable.name != "gnome-text-editor":
                        raise CapabilityBlocked("Installed native GNOME Text Editor unavailable")
                except OSError:
                    raise CapabilityBlocked("Installed native GNOME Text Editor unavailable") from None
                return self.open_application(str(installed), ["--new-window"], "gnome-text-editor",
                    "^gnome-text-editor$", log_path=log_path, expected_executable=executable)
            return self.open_editor(log_path=log_path)
        if application == "files":
            return self.open_files(work, log_path=log_path)
        if application == "calc":
            return self.open_spreadsheet(log_path=log_path)
        if application == "writer":
            launcher = shutil.which("libreoffice")
            if not launcher:
                raise CapabilityBlocked("Installed LibreOffice Writer unavailable")
            executable = Path(launcher).resolve().with_name("soffice.bin")
            if not executable.is_file() or not executable.with_name("libswlo.so").is_file():
                raise CapabilityBlocked("Installed Writer module unavailable")
            for pid in _spreadsheet_process_ids():
                _verify_spreadsheet_process(pid, executable)
            window = self.open_application("libreoffice", ["--writer", "--norestore"], "writer",
                "^libreoffice-writer$", log_path=log_path, child_environment=_spreadsheet_environment(self.environment))
            _verify_spreadsheet_process(window.pid, executable, expected_start=window.process_start)
            return window
        commands = {"evince": ("evince", [], "[Ee]vince"), "vlc": ("vlc", ["--no-playlist-autostart"], "[Vv]lc")}
        if application not in commands:
            raise CapabilityBlocked("Unsupported declared real application")
        binary, args, window_class = commands[application]
        return self.open_application(binary, args, application, window_class, log_path=log_path)

    def is_owned(self, identity):
        return owned_foreground(self.windows, identity, self.environment)

    def verify_registered_window(self, identity):
        """Return None for ordinary launchers; namespace scopes require their own live ledger."""
        matches = [row for row in self.windows if row.window_id == identity["window_id"]
                   and row.pid == str(identity["process_id"]) and row.process_start == identity["process_start"]]
        if len(matches) == 1 and matches[0].owner is not None:
            ledgers = [launch for launch in self.launches if launch.get("ownership_kind") in {"flatpak_firefox", "flatpak_chromium"}
                       and launch.get("registered") is True and self._flatpak_owners.get(id(launch)) is matches[0].owner
                       and launch.get("window_id") == identity["window_id"]]
            if len(ledgers) != 1:
                return False
            try:
                return matches[0].owner.verify()["window"] == identity
            finally:
                for launch in self.launches:
                    if self._flatpak_owners.get(id(launch)) is matches[0].owner:
                        launch["last_ownership_receipt"] = matches[0].owner.last_receipt
        return None

    def controlled_window_status(self, target):
        from runtime.long_horizon_owned_close import window_observation
        return window_observation(target, self.controlled_window_observation(target))[0]

    def controlled_window_observation(self, target):
        return {"identity": dict(target), "inventory": managed_inventory(self.environment)}

    def reserve_controlled_close(self, target):
        key = tuple(target[k] for k in ("window_id", "process_id", "process_start"))
        if not any((row.window_id, row.pid, row.process_start) == key for row in self.windows):
            raise ValueError("Controlled close target is not positively registered")
        self._controlled_close_attempts.add(key)

    def close(self):
        receipts = []
        for launch in reversed(self.launches):
            receipt = {key: value for key, value in launch.items() if key not in {"before_xids", "new_visible_xids"}}
            receipt["closed"] = False
            key = (launch.get("window_id"), launch.get("pid"), launch.get("process_start"))
            if key in getattr(self, "_controlled_close_attempts", set()):
                target = dict(zip(("window_id", "process_id", "process_start"), key, strict=True))
                receipt["closed"] = self.controlled_window_status(target) == "absent"
                receipt["controlled_close_not_retried"] = True
                receipts.append(receipt)
                continue
            if launch.get("ownership_kind") in {"flatpak_firefox", "flatpak_chromium"}:
                owner = self._flatpak_owners.get(id(launch))
                try:
                    receipt.update(owner.close())
                except Exception:
                    receipt["identity_or_close_unavailable"] = True
                receipt["last_ownership_receipt"] = owner.last_receipt if owner else None
                receipts.append(receipt)
                continue
            if launch.get("registered"):
                try:
                    xid, pid, start = launch["window_id"], launch["pid"], launch["process_start"]
                    if xid not in class_windows(launch["window_class"], self.environment):
                        receipt["closed"] = True
                    elif process_start(pid) == start and command(["xdotool", "getwindowpid", xid], self.environment) == pid:
                        if launch.get("expected_executable"):
                            _verify_native_editor_process(pid, Path(launch["expected_executable"]), start)
                        receipt["polite_close_requested"] = request_owned_window_close(xid, pid, start, self.environment)
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline and xid in class_windows(launch["window_class"], self.environment):
                            time.sleep(.1)
                        receipt["closed"] = xid not in class_windows(launch["window_class"], self.environment)
                except (OSError, RuntimeError, subprocess.TimeoutExpired):
                    receipt["identity_or_close_unavailable"] = True
            else:
                receipt["uncertain_unregistered_launch"] = True
            receipts.append(receipt)
        for log in self._logs:
            log.close()
        return receipts
