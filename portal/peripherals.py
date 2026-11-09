"""Generic host-device observations and bounded transports; no product recognition.

Descriptors and bytes are untrusted evidence. Inventory does not open ports,
identify application protocols, transmit packets, or change radio state.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import os
import re
import secrets
import select
import shutil
import socket
import stat
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Any

MAX_BYTES = 16_384
MAX_WRITE = 4_096
MAX_ENDPOINTS = 64
_STORES: weakref.WeakSet = weakref.WeakSet()
_JANITOR_LOCK = threading.Lock()
_JANITOR: threading.Thread | None = None


def _expire_stores() -> None:
    # Keep strong snapshots local to one sweep, not across the janitor sleep.
    with _JANITOR_LOCK:
        stores = list(_STORES)
    for store in stores:
        with store._lock:
            store._reap()


class PeripheralError(ValueError):
    """A bounded transport or device-access failure, never fabricated evidence."""


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise PeripheralError(f"{name} must be between {low} and {high}")
    return float(value)


def _integer(value: Any, name: str, low: int, high: int) -> int:
    _number(value, name, low, high)
    if int(value) != value:
        raise PeripheralError(f"{name} must be an integer")
    return int(value)


def _text(value: Any, maximum: int = 200) -> str:
    return "".join(c for c in str(value) if c >= " " and c != "\x7f")[:maximum]


def _read(path: Path) -> str:
    try:
        return _text(path.read_text(errors="replace").strip())
    except OSError:
        return ""


def _group(number: int) -> str:
    try:
        import grp

        return grp.getgrgid(number).gr_name
    except (ImportError, KeyError):
        return str(number)


def _close_fd(fd: int) -> None:
    import fcntl
    import termios

    try:
        fcntl.ioctl(fd, termios.TIOCNXCL)
    except OSError:
        pass
    os.close(fd)


def _finalize_ports(ports: dict) -> None:
    for port in ports.values():
        try:
            _close_fd(port["fd"])
        except OSError:
            pass


def _start_janitor() -> None:
    global _JANITOR
    with _JANITOR_LOCK:
        if _JANITOR is None:

            def work():
                while True:
                    time.sleep(1)
                    _expire_stores()

            _JANITOR = threading.Thread(target=work, name="peripheral-expiry", daemon=True)
            _JANITOR.start()


def _decode(data: Any, encoding: str) -> bytes:
    if not isinstance(data, str) or len(data) > MAX_WRITE * 4:
        raise PeripheralError("data must be a bounded string")
    try:
        if encoding == "utf8":
            raw = data.encode()
        elif encoding == "hex":
            raw = bytes.fromhex(data)
        elif encoding == "base64":
            raw = base64.b64decode(data, validate=True)
        else:
            raise PeripheralError("encoding must be utf8, hex, or base64")
    except (ValueError, binascii.Error) as exc:
        raise PeripheralError("invalid encoded data") from exc
    if len(raw) > MAX_WRITE:
        raise PeripheralError(f"write exceeds {MAX_WRITE} bytes")
    return raw


def _bytes(raw: bytes) -> dict[str, Any]:
    return {
        "bytes_received": len(raw),
        "data_base64": base64.b64encode(raw).decode(),
        "text_preview": _text(raw.decode("utf8", errors="replace"), 1_024),
        "trust": "untrusted_peripheral_data",
        "protocol_identified": False,
    }


class PeripheralStore:
    def __init__(
        self,
        *,
        sys_root: Path = Path("/sys"),
        dev_root: Path = Path("/dev"),
        proc_root: Path = Path("/proc"),
        ttl_s: float = 300,
    ):
        self.sys = sys_root
        self.dev = dev_root
        self.proc = proc_root
        self.ttl_s = ttl_s
        self._lock = threading.RLock()
        self._ports: dict[str, dict[str, Any]] = {}
        self._bluetooth_lock = threading.Lock()
        with _JANITOR_LOCK:
            _STORES.add(self)
        self._finalizer = weakref.finalize(self, _finalize_ports, self._ports)

    def _command(self, argv: list[str], *, timeout: float = 5) -> dict[str, Any]:
        if not shutil.which(argv[0]):
            return {"status": "unavailable", "reason": "command_missing", "command": argv[0]}
        # Output never grows an unbounded pipe buffer, even on noisy machines.
        import tempfile

        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(
                    argv,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    env={**os.environ, "LC_ALL": "C", "NO_COLOR": "1"},
                )
            except subprocess.TimeoutExpired:
                return {"status": "timeout", "command": argv[0]}
            output.seek(0)
            raw = output.read(MAX_BYTES + 1)
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "output": raw[:MAX_BYTES].decode(errors="replace"),
            "truncated": len(raw) > MAX_BYTES,
        }

    def _owners(self, device: Path) -> list[dict[str, Any]]:
        owners = []
        # Only processes holding this device are reported, without argv or env.
        target = device.stat()
        for process in self.proc.iterdir():
            if not process.name.isdecimal():
                continue
            try:
                for fd in (process / "fd").iterdir():
                    try:
                        s = fd.stat()
                    except OSError:
                        continue
                    if stat.S_ISCHR(s.st_mode) and s.st_rdev == target.st_rdev:
                        unit = next(
                            (
                                v
                                for v in _read(process / "cgroup").split("/")
                                if v.endswith(".service")
                            ),
                            "",
                        )
                        owners.append(
                            {
                                "pid": int(process.name),
                                "name": _read(process / "comm"),
                                "service_unit": unit,
                            }
                        )
                        break
            except (OSError, PermissionError):
                continue
            if len(owners) >= 8:
                break
        return owners

    def _serial_path(self, value: Any) -> Path:
        if not isinstance(value, str) or len(value) > 512:
            raise PeripheralError("device must be a path from peripheral_inventory")
        path = Path(value).resolve(strict=True)
        if path.parent != self.dev.resolve() or not stat.S_ISCHR(path.stat().st_mode):
            raise PeripheralError("device must resolve to a physical serial character device")
        if not (self.sys / "class/tty" / path.name / "device").exists():
            raise PeripheralError("device is not a hardware TTY")
        try:
            consoles = {
                line.split()[0]
                for line in (self.proc / "consoles").read_text().splitlines()
                if line.split()
            }
        except OSError:
            consoles = set()
        consoles.update(_read(self.sys / "class/tty/console/active").split())
        if path.name in consoles:
            raise PeripheralError("system console access is excluded from serial I/O")
        return path

    def inventory(
        self, scope: str = "all", *, category: str | None = None, cursor: int = 0, limit: int = 32
    ) -> dict[str, Any]:
        if not isinstance(scope, str) or scope not in {
            "all",
            "serial",
            "usb",
            "wireless",
            "network",
            "system",
        }:
            raise PeripheralError("invalid inventory scope")
        result: dict[str, Any] = {
            "schema": "robit.omni.peripheral-inventory.v1",
            "observed_at": time.time(),
            "observer_uid": os.geteuid(),
            "observer_groups": [_group(g) for g in os.getgroups()]
            if hasattr(os, "getgroups")
            else [],
            "trust": "untrusted_device_metadata",
            "claim_limits": [
                "Descriptors are labels, not application-protocol identification.",
                "Visibility and OS permissions do not prove successful communication.",
                "Inventory opens no serial ports and makes no active radio scans.",
            ],
        }
        if scope in {"all", "serial"}:
            aliases: dict[str, list[str]] = {}
            for group in ["by-id", "by-path"]:
                folder = self.dev / "serial" / group
                if folder.is_dir():
                    for alias in folder.iterdir():
                        aliases.setdefault(str(alias.resolve()), []).append(str(alias))
            ports = []
            tty_root = self.sys / "class/tty"
            for entry in sorted(tty_root.iterdir()) if tty_root.is_dir() else []:
                path = self.dev / entry.name
                if not (entry / "device").exists() or not path.exists():
                    continue
                s = path.stat()
                driver = entry / "device/driver"
                owners = self._owners(path)
                try:
                    self._serial_path(str(path))
                    eligible = True
                except PeripheralError:
                    eligible = False
                ports.append(
                    {
                        "device": str(path),
                        "aliases": aliases.get(str(path.resolve()), []),
                        "driver": driver.resolve().name if driver.exists() else None,
                        "group": _group(s.st_gid),
                        "mode": stat.filemode(s.st_mode),
                        "readable": os.access(path, os.R_OK),
                        "writable": os.access(path, os.W_OK),
                        "serial_io_eligible": eligible,
                        "busy": bool(owners),
                        "owners": owners,
                    }
                )
                if len(ports) == MAX_ENDPOINTS:
                    break
            result["serial"] = ports
        if scope in {"all", "usb"}:
            devices = []
            root = self.sys / "bus/usb/devices"
            for entry in sorted(root.iterdir()) if root.is_dir() else []:
                if not (entry / "idVendor").exists():
                    continue
                interfaces = []
                for interface in sorted(root.glob(entry.name + ":*")):
                    driver = interface / "driver"
                    interfaces.append(
                        {
                            "interface": interface.name,
                            "class": _read(interface / "bInterfaceClass"),
                            "subclass": _read(interface / "bInterfaceSubClass"),
                            "protocol": _read(interface / "bInterfaceProtocol"),
                            "driver": driver.resolve().name if driver.exists() else None,
                            "modalias": _read(interface / "modalias"),
                            "driver_bound": driver.exists(),
                        }
                    )
                devices.append(
                    {
                        "usb_path": entry.name,
                        **{
                            name: _read(entry / name)
                            for name in [
                                "idVendor",
                                "idProduct",
                                "manufacturer",
                                "product",
                                "serial",
                            ]
                        },
                        "interfaces": interfaces,
                        "protocol_identified": False,
                    }
                )
                if len(devices) == MAX_ENDPOINTS:
                    break
            result["usb"] = devices
        if scope in {"all", "wireless"}:
            result["radio_blocks"] = self._command(["rfkill", "--json"])
            result["bluetooth_controllers"] = self._command(["bluetoothctl", "list"])
            result["bluetooth_state"] = self._command(["bluetoothctl", "show"])
            result["known_bluetooth_devices"] = self._command(["bluetoothctl", "devices"])
        if scope in {"all", "network"}:
            result["network"] = self._command(
                ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device", "status"]
                if scope == "all"
                else ["ip", "-json", "address", "show"]
            )
            result["wifi_radios"] = self._command(["nmcli", "-t", "-f", "WIFI-HW,WIFI", "radio"])
        if scope in {"all", "system"}:
            result["system_devices"] = self.system_devices(
                category=category, cursor=cursor, limit=limit, summary_only=scope == "all"
            )
        return result

    def system_devices(
        self,
        *,
        category: str | None = None,
        cursor: int = 0,
        limit: int = 32,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        cursor = _integer(cursor, "cursor", 0, 1_000_000)
        limit = _integer(limit, "limit", 1, 64)
        categories = {}
        for base, suffix in [("class", ""), ("bus", "devices")]:
            parent = self.sys / base
            for entry in sorted(parent.iterdir()) if parent.is_dir() else []:
                folder = entry / suffix if suffix else entry
                if folder.is_dir():
                    categories[f"{base}/{entry.name}"] = folder
        if category is not None and (not isinstance(category, str) or category not in categories):
            raise PeripheralError("category must be one of the observed system-device categories")
        counts = [
            {"category": name, "count": sum(1 for _ in folder.iterdir())}
            for name, folder in categories.items()
        ]
        entries = []
        for name, folder in categories.items():
            if category is None or name == category:
                entries.extend((name, entry) for entry in sorted(folder.iterdir()))
        devices = []
        fields = [
            "dev",
            "vendor",
            "device",
            "class",
            "subsystem_vendor",
            "subsystem_device",
            "name",
            "model",
            "manufacturer",
            "product",
            "type",
            "ifindex",
            "operstate",
            "address",
            "removable",
            "size",
            "ro",
            "state",
            "status",
            "modalias",
        ]
        for name, entry in [] if summary_only else entries[cursor : cursor + limit]:
            driver = entry / "driver"
            if not driver.exists():
                driver = entry / "device/driver"
            devname = ""
            try:
                values = dict(
                    line.split("=", 1)
                    for line in (entry / "uevent").read_text().splitlines()
                    if "=" in line
                )
                devname = values.get("DEVNAME", "")
            except OSError:
                pass
            node = self.dev / devname if devname else None
            devices.append(
                {
                    "device_ref": name + "/" + entry.name,
                    "sysfs_path": str(entry),
                    "driver": driver.resolve().name if driver.exists() else None,
                    "attributes": {
                        field: value for field in fields if (value := _read(entry / field))
                    },
                    "device_node": str(node) if node is not None else None,
                    "readable": os.access(node, os.R_OK) if node is not None else None,
                    "writable": os.access(node, os.W_OK) if node is not None else None,
                    "protocol_identified": False,
                }
            )
        return {
            "categories": counts,
            "selected_category": category,
            "devices": devices,
            "total_entries": len(entries),
            "cursor": cursor,
            "next_cursor": 0
            if summary_only and entries
            else cursor + limit
            if cursor + limit < len(entries)
            else None,
            "complete": not summary_only and cursor + limit >= len(entries),
            "next_action": "Choose scope=system and an observed category/cursor for device details.",
            "claim_limits": [
                "Sysfs exposes virtual and physical interfaces; entries may alias the same device.",
                "Descriptors and driver bindings do not identify application protocols.",
            ],
        }

    def _reap(self) -> None:
        for key, port in list(self._ports.items()):
            if time.monotonic() - port["touched"] > self.ttl_s:
                with port["lock"]:
                    _close_fd(port["fd"])
                del self._ports[key]

    def clear(self, session: str) -> None:
        with self._lock:
            for key, port in list(self._ports.items()):
                if port["owner"] == session:
                    with port["lock"]:
                        _close_fd(port["fd"])
                    del self._ports[key]

    def close(self) -> None:
        with self._lock:
            for port in self._ports.values():
                _close_fd(port["fd"])
            self._ports.clear()

    def serial(self, session: str, arguments: dict[str, Any]) -> dict[str, Any]:
        action = arguments.get("action", "status")
        if not isinstance(action, str):
            raise PeripheralError("action must be a string")
        if action == "open":
            return self._serial_open(session, arguments)
        if action not in {"read", "write", "exchange", "close", "status"}:
            raise PeripheralError("invalid serial action")
        with self._lock:
            self._reap()
            port = self._ports.get(str(arguments.get("handle") or ""))
            if port is None or port["owner"] != session:
                raise PeripheralError("unknown or expired serial handle for this session")
            port["lock"].acquire()
            port["touched"] = time.monotonic()
            if action == "close":
                del self._ports[str(arguments["handle"])]
        try:
            if action == "close":
                _close_fd(port["fd"])
                return {"status": "closed", "device": port["device"]}
            if action == "status":
                return {"status": "open", "device": port["device"], "baudrate": port["baudrate"]}
            timeout = _number(arguments.get("timeout_s", 2), "timeout_s", 0.05, 10)
            maximum = _integer(arguments.get("max_bytes", 4096), "max_bytes", 1, MAX_BYTES)
            deadline = time.monotonic() + timeout
            written = 0
            if action in {"write", "exchange"}:
                data = _decode(arguments.get("data"), arguments.get("encoding", "utf8"))
                while written < len(data) and time.monotonic() < deadline:
                    if select.select([], [port["fd"]], [], max(0, deadline - time.monotonic()))[1]:
                        written += os.write(port["fd"], data[written:])
            received = bytearray()
            if action in {"read", "exchange"}:
                while len(received) < maximum and time.monotonic() < deadline:
                    if not select.select([port["fd"]], [], [], max(0, deadline - time.monotonic()))[
                        0
                    ]:
                        break
                    data = os.read(port["fd"], maximum - len(received))
                    if not data:
                        break
                    received.extend(data)
            return {
                "status": "observed",
                "device": port["device"],
                "action": action,
                "os_bytes_written": written,
                **_bytes(bytes(received)),
                "claim_limits": ["Bytes accepted by the OS do not verify a remote command effect."],
            }
        finally:
            port["touched"] = time.monotonic()
            port["lock"].release()

    def _serial_open(self, session: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if os.name != "posix":
            raise PeripheralError("hardware TTY I/O requires a POSIX host")
        import fcntl
        import termios

        if "baudrate" not in arguments:
            raise PeripheralError(
                "open requires an explicit baudrate selected from device evidence"
            )
        path = self._serial_path(arguments.get("device"))
        baud = _integer(arguments.get("baudrate", 9600), "baudrate", 50, 4_000_000)
        speed = getattr(termios, f"B{baud}", None)
        if speed is None:
            raise PeripheralError("requested baudrate is not supported by this host")
        bits = _integer(arguments.get("data_bits", 8), "data_bits", 5, 8)
        stop = _integer(arguments.get("stop_bits", 1), "stop_bits", 1, 2)
        parity = arguments.get("parity", "none")
        if parity not in {"none", "even", "odd"}:
            raise PeripheralError("parity must be none, even or odd")
        with self._lock:
            self._reap()
            if len(self._ports) >= 8:
                raise PeripheralError("serial session capacity reached; close an unused handle")
            owners = self._owners(path)
            if owners or any(p["device"] == str(path) for p in self._ports.values()):
                return {
                    "status": "busy",
                    "device": str(path),
                    "owners": owners,
                    "opened": False,
                    "claim_limits": ["Do not compete with the existing device owner."],
                }
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK | os.O_CLOEXEC)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.ioctl(fd, termios.TIOCEXCL)
                attrs = termios.tcgetattr(fd)
                attrs[0] = attrs[1] = attrs[3] = 0
                attrs[2] &= ~(
                    termios.CSIZE | termios.PARENB | termios.PARODD | termios.CSTOPB | termios.HUPCL
                )
                attrs[2] |= termios.CLOCAL | termios.CREAD | getattr(termios, f"CS{bits}")
                if parity != "none":
                    attrs[2] |= termios.PARENB | (termios.PARODD if parity == "odd" else 0)
                if stop == 2:
                    attrs[2] |= termios.CSTOPB
                attrs[4] = attrs[5] = speed
                attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
            except BaseException:
                _close_fd(fd)
                raise
            handle = secrets.token_urlsafe(24)
            self._ports[handle] = {
                "owner": session,
                "device": str(path),
                "fd": fd,
                "baudrate": baud,
                "touched": time.monotonic(),
                "lock": threading.RLock(),
            }
            _start_janitor()
        return {
            "status": "open",
            "handle": handle,
            "device": str(path),
            "baudrate": baud,
            "dtr_rts_explicitly_changed": False,
            "close_hangup_disabled": True,
            "expires_after_idle_s": self.ttl_s,
            "protocol_identified": False,
        }

    def bluetooth(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = arguments.get("action", "scan")
        if not isinstance(action, str):
            raise PeripheralError("action must be a string")
        if action == "power":
            enabled = arguments.get("enabled")
            if not isinstance(enabled, bool):
                raise PeripheralError("enabled must be boolean")
            before = self._command(["bluetoothctl", "show"])
            attempt = self._command(["bluetoothctl", "power", "on" if enabled else "off"])
            after = self._command(["bluetoothctl", "show"])
            return {
                "before": before,
                "attempt": attempt,
                "after": after,
                "verified": after.get("status") == "ok"
                and bool(
                    re.search(
                        rf"^\s*Powered: {'yes' if enabled else 'no'}\s*$",
                        after.get("output", ""),
                        re.MULTILINE,
                    )
                ),
            }
        if action not in {"scan", "services", "read", "write", "listen"}:
            raise PeripheralError("invalid Bluetooth action")
        timeout = _number(arguments.get("timeout_s", 5), "timeout_s", 0.1, 15)
        try:
            from bleak.exc import BleakError
        except ImportError as exc:
            raise PeripheralError("Bluetooth GATT dependency unavailable; install bleak") from exc
        try:
            if not self._bluetooth_lock.acquire(blocking=False):
                return {
                    "status": "busy",
                    "reason": "another_bluetooth_operation",
                    "operation_started": False,
                }
            try:
                return asyncio.run(
                    asyncio.wait_for(self._bluetooth(action, arguments, timeout), timeout + 5)
                )
            finally:
                self._bluetooth_lock.release()
        except ImportError as exc:
            raise PeripheralError("Bluetooth GATT dependency unavailable; install bleak") from exc
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise PeripheralError("Bluetooth operation timed out; no result inferred") from exc
        except BleakError as exc:
            raise PeripheralError("Bluetooth operation failed: " + str(exc)[:300]) from exc

    async def _bluetooth(
        self, action: str, arguments: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        from bleak import BleakClient, BleakScanner

        if action == "scan":
            found = await BleakScanner.discover(timeout=timeout, return_adv=True)
            return {
                "observed_at": time.time(),
                "trust": "untrusted_radio_advertisements",
                "protocol_identified": False,
                "devices": [
                    {
                        "address": device.address,
                        "name": _text(adv.local_name or device.name or ""),
                        "rssi": adv.rssi,
                        "service_uuids": list(adv.service_uuids)[:32],
                        "manufacturer_data": {
                            str(k): bytes(v)[:128].hex()
                            for k, v in list(adv.manufacturer_data.items())[:8]
                        },
                        "service_data": {
                            str(k): bytes(v)[:128].hex()
                            for k, v in list(adv.service_data.items())[:8]
                        },
                    }
                    for device, adv in list(found.values())[:MAX_ENDPOINTS]
                ],
                "truncated": len(found) > MAX_ENDPOINTS,
            }
        address = arguments.get("address")
        if not isinstance(address, str) or not re.fullmatch(
            r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", address
        ):
            raise PeripheralError("address must be an observed Bluetooth address")
        info = self._command(["bluetoothctl", "info", address])
        if re.search(r"^\s*Connected: yes\s*$", info.get("output", ""), re.MULTILINE):
            return {
                "status": "busy",
                "reason": "device_already_connected",
                "device_state": info,
                "connection_taken": False,
            }
        device = await BleakScanner.find_device_by_address(address, timeout=timeout)
        if device is None:
            raise PeripheralError("requested Bluetooth device was not observed")
        async with BleakClient(device, timeout=timeout) as client:
            if action == "services":
                return {
                    "address": address,
                    "connected": client.is_connected,
                    "protocol_identified": False,
                    "services": [
                        {
                            "uuid": s.uuid,
                            "description": _text(s.description),
                            "characteristics": [
                                {
                                    "uuid": c.uuid,
                                    "handle": c.handle,
                                    "properties": list(c.properties),
                                }
                                for c in list(s.characteristics)[:64]
                            ],
                        }
                        for s in list(client.services)[:32]
                    ],
                }
            characteristic = arguments.get("characteristic")
            if not isinstance(characteristic, str) or not re.fullmatch(
                r"(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})",
                characteristic,
            ):
                raise PeripheralError("characteristic must be an observed UUID")
            if action == "read":
                return {
                    "address": address,
                    **_bytes(bytes(await client.read_gatt_char(characteristic))[:MAX_BYTES]),
                }
            if action == "write":
                raw = _decode(arguments.get("data"), arguments.get("encoding", "utf8"))
                await client.write_gatt_char(characteristic, raw, response=True)
                return {
                    "address": address,
                    "att_write_acknowledged": True,
                    "bytes_written": len(raw),
                    "claim_limits": [
                        "An ATT acknowledgement does not verify an application-level effect."
                    ],
                }
            notifications = []
            total = 0

            def notify(_sender, data):
                nonlocal total
                if len(notifications) < 64 and total < MAX_BYTES:
                    raw = bytes(data[: MAX_BYTES - total])
                    total += len(raw)
                    notifications.append(_bytes(raw))

            await client.start_notify(characteristic, notify)
            try:
                await asyncio.sleep(timeout)
            finally:
                await client.stop_notify(characteristic)
            return {
                "address": address,
                "notifications": notifications,
                "bytes_received": total,
                "bounded": True,
                "protocol_identified": False,
            }

    def network(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = arguments.get("action", "status")
        if not isinstance(action, str):
            raise PeripheralError("action must be a string")
        if action == "status":
            return {
                "interfaces": self._command(["ip", "-json", "address", "show"]),
                "devices": self._command(
                    ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device", "status"]
                ),
                "radios": self._command(["nmcli", "-t", "-f", "WIFI-HW,WIFI", "radio"]),
            }
        if action == "wifi_scan":
            return self._command(
                [
                    "nmcli",
                    "-t",
                    "-f",
                    "IN-USE,SSID,BSSID,CHAN,SIGNAL,SECURITY,DEVICE",
                    "device",
                    "wifi",
                    "list",
                    "--rescan",
                    "yes",
                ],
                timeout=15,
            )
        if action == "services":
            return self._command(
                ["avahi-browse", "--all", "--resolve", "--terminate", "--parsable"], timeout=10
            )
        if action == "wifi_power":
            enabled = arguments.get("enabled")
            if not isinstance(enabled, bool):
                raise PeripheralError("enabled must be boolean")
            before = self._command(["nmcli", "radio", "wifi"])
            attempt = self._command(["nmcli", "radio", "wifi", "on" if enabled else "off"])
            after = self._command(["nmcli", "radio", "wifi"])
            return {
                "before": before,
                "attempt": attempt,
                "after": after,
                "verified": after.get("status") == "ok"
                and after.get("output", "").strip() == ("enabled" if enabled else "disabled"),
            }
        if action not in {"tcp_exchange", "udp_exchange"}:
            raise PeripheralError("invalid network action")
        # Explicit IP literals prevent unbounded resolver work and make the
        # selected endpoint auditable. This is a local-peripheral transport,
        # separate from the public-only web fetcher.
        try:
            if not isinstance(arguments.get("address"), str):
                raise ValueError("IP address must be a string")
            address = ipaddress.ip_address(arguments.get("address"))
        except ValueError as exc:
            raise PeripheralError("address must be an explicit IP literal") from exc
        if address.is_unspecified or address.is_multicast:
            raise PeripheralError("unspecified and multicast targets are not exchange endpoints")
        port = _integer(arguments.get("port"), "port", 1, 65535)
        timeout = _number(arguments.get("timeout_s", 2), "timeout_s", 0.1, 10)
        maximum = _integer(arguments.get("max_bytes", 4096), "max_bytes", 1, MAX_BYTES)
        raw = _decode(arguments.get("data", ""), arguments.get("encoding", "utf8"))
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        kind = socket.SOCK_STREAM if action == "tcp_exchange" else socket.SOCK_DGRAM
        with socket.socket(family, kind) as connection:
            connection.settimeout(timeout)
            connection.connect((str(address), port))
            if raw:
                connection.sendall(raw)
            try:
                reply = connection.recv(maximum)
            except TimeoutError:
                reply = b""
        return {
            "endpoint": {"address": str(address), "port": port},
            "os_bytes_written": len(raw),
            **_bytes(reply),
            "claim_limits": [
                "A connected socket or successful send is not remote application verification."
            ],
        }
