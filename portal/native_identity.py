"""Private, read-only X11/Flatpak identity proofs; never an input authority.

Flatpak 1.12.7 gives its D-Bus proxy a namespace-local .flatpak-info copy.
xdg-desktop-portal 1.14.4 identifies authenticated peers through that file.
The proxy copy may be writable; the application copy must be a read-only mount.
We additionally bind the live registry, XRes owner, process starts, namespaces,
lineage, executable and cgroup. No proxy ancestry or socket peer is inferred.
"""

from __future__ import annotations

import configparser
import ctypes
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

Read = Callable[[Callable[[], Any]], Any]
Credentials = Callable[[str], dict[str, int]]
MAX_METADATA = 65536
PROXY = Path("/usr/bin/xdg-dbus-proxy")


class IdentityUnavailable(RuntimeError):
    pass


def _io(function: Callable[[], Any], read: Read, deadline: float) -> Any:
    if time.monotonic() >= deadline:
        raise IdentityUnavailable("native_provider_identity_budget_exceeded")
    value = read(function)
    if time.monotonic() >= deadline:
        raise IdentityUnavailable("native_provider_identity_budget_exceeded")
    return value


def _stamp(value: os.stat_result) -> dict[str, int]:
    return {"device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
            "mode": value.st_mode, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _open_directory(path: Path, *, optional: bool = False) -> int | None:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError:
        if optional:
            return None
        raise


def _metadata(root: Path, name: str, *, read: Read, deadline: float,
              optional: bool = False, nofollow_root: bool = False) -> dict | None:
    def open_root() -> int | None:
        if nofollow_root:
            return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        return _open_directory(root, optional=optional)

    root_fd = _io(open_root, read, deadline)
    if root_fd is None:
        return None
    fd = None
    try:
        root_stat = _io(lambda: os.fstat(root_fd), read, deadline)

        def open_file() -> int | None:
            try:
                return os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_fd)
            except FileNotFoundError:
                if optional:
                    return None
                raise

        fd = _io(open_file, read, deadline)
        if fd is None:
            return None
        before = _io(lambda: os.fstat(fd), read, deadline)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or not 0 < before.st_size <= MAX_METADATA or root_stat.st_uid != os.getuid()):
            raise IdentityUnavailable("native_provider_metadata_invalid")
        data = _io(lambda: os.read(fd, MAX_METADATA + 1), read, deadline)
        after = _io(lambda: os.fstat(fd), read, deadline)
        if len(data) != before.st_size or _stamp(before) != _stamp(after):
            raise IdentityUnavailable("native_provider_metadata_changed")
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(data.decode("utf-8"))
            application, instance = parser.get("Application", "name"), parser.get("Instance", "instance-id")
        except (UnicodeError, configparser.Error):
            raise IdentityUnavailable("native_provider_metadata_invalid") from None
        if (not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", application)
                or len(application) > 200 or not re.fullmatch(r"[0-9]{1,20}", instance)):
            raise IdentityUnavailable("native_provider_metadata_invalid")
        return {"application": application, "instance": instance,
                "sha256": hashlib.sha256(data).hexdigest(), "file": _stamp(before),
                "root": {"device": root_stat.st_dev, "inode": root_stat.st_ino,
                         "uid": root_stat.st_uid},
                "read_only": bool(_io(lambda: os.fstatvfs(fd), read, deadline).f_flag & os.ST_RDONLY)}
    finally:
        if fd is not None:
            os.close(fd)
        os.close(root_fd)


def _process(pid: int, *, read: Read, deadline: float) -> dict:
    if type(pid) is not int or not 0 < pid < 2**31:
        raise IdentityUnavailable("native_provider_process_invalid")
    root = Path("/proc") / str(pid)
    raw = _io(lambda: (root / "stat").read_text(), read, deadline).rsplit(")", 1)[1].split()
    status = _io(lambda: (root / "status").read_text(), read, deadline)
    fields = {key: value.strip() for line in status.splitlines() if ":" in line
              for key, _, value in [line.partition(":")] if key in {"Uid", "NSpid"}}
    uids = [int(value) for value in fields["Uid"].split()]
    if len(uids) != 4 or len(set(uids)) != 1:
        raise IdentityUnavailable("native_provider_uid_invalid")
    return {"pid": pid, "start": raw[19], "ppid": int(raw[1]), "uid": uids[0],
            "nspid": [int(value) for value in fields["NSpid"].split()],
            "namespaces": {key: _io(lambda key=key: os.readlink(root / "ns" / key), read, deadline)
                           for key in ("pid", "user", "mnt")},
            "cgroup": _io(lambda: (root / "cgroup").read_text(), read, deadline).splitlines()}


def _registry(*, read: Read, deadline: float) -> list[dict]:
    result = _io(lambda: subprocess.run(
        ["flatpak", "ps", "--columns=instance,pid,child-pid,application"],
        capture_output=True, text=True, timeout=max(.05, min(1, deadline - time.monotonic())),
        check=False), read, deadline)  # noqa: S603
    if result.returncode or len(result.stdout) > 65536:
        raise IdentityUnavailable("native_flatpak_registry_unavailable")
    lines = result.stdout.splitlines()
    if len(lines) > 128:
        raise IdentityUnavailable("native_flatpak_registry_incomplete")
    rows = []
    for line in lines:
        values = line.split("\t")
        if (len(values) != 4 or any(not re.fullmatch(r"[0-9]{1,20}", value) for value in values[:3])
                or len(values[3]) > 200):
            raise IdentityUnavailable("native_flatpak_registry_invalid")
        rows.append({"instance": values[0], "wrapper": int(values[1]),
                     "child": int(values[2]), "application": values[3]})
    return rows


def relay_candidate(pid: int, *, read: Read, deadline: float) -> bool:
    """Candidate discovery only; never authority to map, switch or close."""
    if type(pid) is not int or not 0 < pid < 2**31:
        return False

    def optional_stat(path: Path) -> os.stat_result | None:
        try:
            return path.stat()
        except OSError:
            return None

    installed = _io(lambda: optional_stat(PROXY), read, deadline)
    running = _io(lambda: optional_stat(Path(f"/proc/{pid}/exe")), read, deadline)
    return (installed is not None and running is not None
            and (installed.st_dev, installed.st_ino) == (running.st_dev, running.st_ino))


def client_windows(environment: Mapping[str, str], *, timeout_s: float = 1) -> list[str]:
    """Read the complete bounded WM-managed client list, never a title search."""
    result = subprocess.run(["xprop", "-root", "_NET_CLIENT_LIST"],  # noqa: S603
        env={**environment, "LC_ALL": "C"}, capture_output=True, text=True,
        timeout=max(.05, min(1, timeout_s)), check=False)
    raw = result.stdout.strip()
    if (result.returncode or len(raw) > 65536 or re.fullmatch(
            r"_NET_CLIENT_LIST\(WINDOW\): window id # 0x[0-9a-fA-F]+(?:, 0x[0-9a-fA-F]+)*", raw) is None):
        raise IdentityUnavailable("native_client_inventory_unavailable")
    values = [str(int(value, 16)) for value in re.findall(r"0x[0-9a-fA-F]+", raw)]
    if (not 1 <= len(values) <= 128 or len(set(values)) != len(values)
            or any(not 0 < int(value) < 2**32 for value in values)):
        raise IdentityUnavailable("native_client_inventory_incomplete")
    return sorted(values, key=int)


def _xres(window_id: str) -> dict:
    """Run only in the fixed isolated child with the recovered desktop env."""
    if not isinstance(window_id, str) or not window_id.isascii() or not window_id.isdecimal() or not 0 < int(window_id) < 2**32:
        raise IdentityUnavailable("native_xres_window_invalid")

    class Spec(ctypes.Structure):
        _fields_ = [("client", ctypes.c_ulong), ("mask", ctypes.c_ulong)]

    class Value(ctypes.Structure):
        _fields_ = [("spec", Spec), ("length", ctypes.c_long), ("value", ctypes.c_void_p)]

    library, extension = ctypes.CDLL("libX11.so.6"), ctypes.CDLL("libXRes.so.1")
    library.XOpenDisplay.argtypes, library.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
    library.XCloseDisplay.argtypes = [ctypes.c_void_p]
    extension.XResQueryExtension.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    extension.XResQueryVersion.argtypes = extension.XResQueryExtension.argtypes
    extension.XResQueryClientIds.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.POINTER(Spec), ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.POINTER(Value))]
    extension.XResGetClientPid.argtypes, extension.XResGetClientPid.restype = [ctypes.POINTER(Value)], ctypes.c_int
    extension.XResClientIdsDestroy.argtypes = [ctypes.c_long, ctypes.POINTER(Value)]
    display = library.XOpenDisplay(None)
    if not display:
        raise IdentityUnavailable("native_xres_display_unavailable")
    try:
        major, minor = ctypes.c_int(), ctypes.c_int()
        if (not extension.XResQueryExtension(display, ctypes.byref(major), ctypes.byref(minor))
                or not extension.XResQueryVersion(display, ctypes.byref(major), ctypes.byref(minor))
                or (major.value, minor.value) < (1, 2)):
            raise IdentityUnavailable("native_xres_extension_unavailable")

        def query() -> int:
            count, values, spec = ctypes.c_long(), ctypes.POINTER(Value)(), Spec(int(window_id), 2)
            try:
                status = extension.XResQueryClientIds(display, 1, ctypes.byref(spec), ctypes.byref(count), ctypes.byref(values))
                if status != 0 or count.value != 1 or not values or values[0].spec.mask != 2 or values[0].length != 4:
                    raise IdentityUnavailable("native_xres_owner_ambiguous")
                pid = extension.XResGetClientPid(ctypes.byref(values[0]))
                if not 0 < pid < 2**31:
                    raise IdentityUnavailable("native_xres_owner_unavailable")
                return pid
            finally:
                if values:
                    extension.XResClientIdsDestroy(count.value, values)

        pid = query()
        start = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if query() != pid or Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] != start:
            raise IdentityUnavailable("native_xres_owner_changed")
        return {"pid": pid, "start": start}
    finally:
        library.XCloseDisplay(display)


def xres_owner(window_id: str, environment: Mapping[str, str], *, timeout_s: float = 1) -> dict:
    result = subprocess.run(  # noqa: S603 - one fixed read-only helper, no global env mutation.
        [sys.executable, "-I", str(Path(__file__).resolve()), "--xres-owner", window_id],
        env=dict(environment), stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=max(.05, min(1, timeout_s)), check=False)
    if result.returncode or len(result.stdout) > 4096:
        raise IdentityUnavailable("native_xres_owner_unavailable")
    try:
        value = json.loads(result.stdout)
    except ValueError:
        raise IdentityUnavailable("native_xres_owner_invalid") from None
    if (not isinstance(value, dict) or set(value) != {"pid", "start"} or type(value["pid"]) is not int or not 0 < value["pid"] < 2**31
            or not isinstance(value["start"], str) or not value["start"].isdecimal()):
        raise IdentityUnavailable("native_xres_owner_invalid")
    return value


def _validate(value: dict) -> None:
    host, peer, child, wrapper = (value[key] for key in ("host", "peer", "child", "wrapper"))
    meta = value["metadata"]
    if (any(item["uid"] != os.getuid() for item in (host, peer, child, wrapper))
            or value["credentials"] != {"pid": peer["pid"], "uid": peer["uid"]}
            or value["owner"] != {"pid": host["pid"], "start": host["start"]}
            or value["installed_proxy"] != value["running_proxy"]
            or value["installed_proxy"]["uid"] != 0 or value["installed_proxy"]["mode"] & 0o022
            or not stat.S_ISREG(value["installed_proxy"]["mode"])
            or any(item["cgroup"] != host["cgroup"] for item in (peer, child, wrapper))
            or not host["cgroup"] or child["ppid"] != wrapper["pid"]
            or host["namespaces"]["pid"] != child["namespaces"]["pid"]
            or not value["lineage"] or value["lineage"][0] != host
            or value["lineage"][-1] != child):
        raise IdentityUnavailable("native_provider_identity_unverified")
    if any(first["ppid"] != second["pid"] for first, second in zip(value["lineage"], value["lineage"][1:], strict=False)):
        raise IdentityUnavailable("native_provider_lineage_unverified")
    expected = value["registry"]
    if (expected["wrapper"] != wrapper["pid"] or expected["child"] != child["pid"]
            or not meta["host"]["read_only"] or len({item["sha256"] for item in meta.values()}) != 1
            or any(item["application"] != expected["application"] or item["instance"] != expected["instance"]
                   or item["file"]["uid"] != os.getuid() or not stat.S_ISREG(item["file"]["mode"])
                   or not 0 < item["file"]["size"] <= MAX_METADATA for item in meta.values())):
        raise IdentityUnavailable("native_provider_metadata_unverified")


def _sample(identity: Mapping[str, Any], bus_name: str, reported_pid: int, *,
            credentials: Credentials, read: Read, deadline: float) -> dict | None:
    host_meta = _metadata(Path(f"/proc/{identity['process_id']}/root"), ".flatpak-info",
                          read=read, deadline=deadline, optional=True)
    if host_meta is None:
        return None
    installed = _io(PROXY.stat, read, deadline)
    running = _io(lambda: Path(f"/proc/{reported_pid}/exe").stat(), read, deadline)
    if (installed.st_dev, installed.st_ino) != (running.st_dev, running.st_ino):
        return None  # An unrelated ordinary provider is not this Flatpak relay.
    peer_meta = _metadata(Path(f"/proc/{reported_pid}/root"), ".flatpak-info", read=read, deadline=deadline)
    if peer_meta["instance"] != host_meta["instance"] or peer_meta["application"] != host_meta["application"]:
        return None
    rows = [row for row in _registry(read=read, deadline=deadline) if row["instance"] == host_meta["instance"]]
    if len(rows) != 1:
        raise IdentityUnavailable("native_flatpak_instance_not_unique")
    row = rows[0]
    records = {key: _process(pid, read=read, deadline=deadline) for key, pid in (
        ("host", identity["process_id"]), ("peer", reported_pid), ("child", row["child"]), ("wrapper", row["wrapper"]))}
    lineage, seen, current = [], set(), records["host"]
    for _ in range(12):
        if current["pid"] in seen:
            break
        seen.add(current["pid"])
        lineage.append(current)
        if current["pid"] == row["child"]:
            break
        current = _process(current["ppid"], read=read, deadline=deadline)
    authoritative = _metadata(Path(f"/run/user/{os.getuid()}/.flatpak/{row['instance']}"), "info",
                              read=read, deadline=deadline, nofollow_root=True)
    value = {**records, "lineage": lineage, "registry": row,
             # The native credential callback accounts for each D-Bus read.
             # Avoid nesting its budget exception inside another bridge read.
             "credentials": _io(lambda: credentials(bus_name), lambda function: function(), deadline),
             "bus_name": bus_name, "metadata": {"host": host_meta, "peer": peer_meta, "instance": authoritative},
             "installed_proxy": _stamp(installed), "running_proxy": _stamp(running),
             "owner": _io(lambda: xres_owner(identity["window_id"], os.environ,
                 timeout_s=deadline - time.monotonic()), read, deadline)}
    _validate(value)
    return value


def provider_proof(identity: Mapping[str, Any], bus_name: str, reported_pid: int, *,
                   credentials: Credentials, read: Read, deadline: float) -> dict | None:
    if not isinstance(bus_name, str) or not re.fullmatch(r":[0-9]+\.[0-9]+", bus_name) or len(bus_name) > 200:
        raise IdentityUnavailable("native_provider_bus_invalid")
    before = _sample(identity, bus_name, reported_pid, credentials=credentials, read=read, deadline=deadline)
    if before is None:
        return None
    after = _sample(identity, bus_name, reported_pid, credentials=credentials, read=read, deadline=deadline)
    if before != after:
        raise IdentityUnavailable("native_provider_identity_changed")
    return {"source": "flatpak_namespace_metadata", "identity": dict(identity), "reported_pid": reported_pid,
            "bus_name": bus_name, "values": before,
            "digest": hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def host_process_identity(pid: int, *, proc_root: Path = Path("/proc")) -> dict:
    """Positive ordinary host process identity; no XRes or namespace dependency."""
    if type(pid) is not int or not 0 < pid < 2**31:
        raise IdentityUnavailable("native_host_process_unavailable")
    try:
        root = proc_root / str(pid)
        start = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        uid = next(line.split()[1:] for line in root.joinpath("status").read_text().splitlines()
                   if line.startswith("Uid:"))
        executable = root.joinpath("exe").stat()
        if (root.stat().st_uid != os.getuid() or uid != [str(os.getuid())] * 4
                or not start.isascii() or not start.isdecimal() or int(start) <= 0
                or not stat.S_ISREG(executable.st_mode)
                or root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] != start
                or root.joinpath("exe").stat() != executable):
            raise IdentityUnavailable("native_host_process_unavailable")
        return {"pid": pid, "start": start, "uid": os.getuid(), "executable": _stamp(executable)}
    except (OSError, ValueError, IndexError, StopIteration):
        raise IdentityUnavailable("native_host_process_unavailable") from None


def _check_namespace_owner(value: dict) -> dict:
    try:
        if set(value) != {"identity", "reported_wm_pid", "owner", "host_process", "proof", "digest"}:
            raise ValueError
        proof, identity, host = value["proof"], value["identity"], value["host_process"]
        if set(proof) != {"source", "identity", "reported_pid", "bus_name", "values", "digest"}:
            raise ValueError
        values = proof["values"]
        if set(values) != {"host", "peer", "child", "wrapper", "lineage", "registry", "credentials",
                           "bus_name", "metadata", "installed_proxy", "running_proxy", "owner"}:
            raise ValueError
        process_fields = {"pid", "start", "ppid", "uid", "nspid", "namespaces", "cgroup"}
        stamp_fields = {"device", "inode", "uid", "mode", "size", "mtime_ns", "ctime_ns"}
        if (set(host) != {"pid", "start", "uid", "executable"}
                or set(host["executable"]) != stamp_fields
                or type(host["pid"]) is not int or not 0 < host["pid"] < 2**31
                or not isinstance(host["start"], str) or re.fullmatch(r"[1-9][0-9]{0,19}", host["start"]) is None
                or any(type(v) is not int for v in host["executable"].values())
                or any(set(values[key]) != process_fields for key in ("host", "peer", "child", "wrapper"))
                or any(set(item) != process_fields for item in values["lineage"])
                or set(value["owner"]) != {"pid", "start"}
                or set(values["credentials"]) != {"pid", "uid"}
                or set(values["registry"]) != {"instance", "wrapper", "child", "application"}
                or set(values["metadata"]) != {"host", "peer", "instance"}
                or any(set(item) != {"application", "instance", "sha256", "file", "root", "read_only"}
                       or set(item["file"]) != stamp_fields or set(item["root"]) != {"device", "inode", "uid"}
                       for item in values["metadata"].values())
                or any(set(values[key]) != stamp_fields for key in ("installed_proxy", "running_proxy"))
                or proof["reported_pid"] != values["peer"]["pid"] or proof["bus_name"] != values["bus_name"]
                or proof["digest"] != hashlib.sha256(json.dumps(values, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()):
            raise ValueError
        _validate(values)
        record = proof["values"]["host"]
        unsigned = {key: item for key, item in value.items() if key != "digest"}
        if (set(identity) != {"window_id", "process_id"}
                or proof["source"] != "flatpak_namespace_metadata" or proof["identity"] != identity
                or proof["values"]["owner"] != value["owner"]
                or host["pid"] != identity["process_id"] or host["pid"] != record["pid"]
                or host["start"] != record["start"] or host["uid"] != os.getuid()
                or value["owner"] != {"pid": host["pid"], "start": host["start"]}
                or type(value["reported_wm_pid"]) is not int or value["reported_wm_pid"] <= 0
                or host["pid"] == value["reported_wm_pid"] or len(record["nspid"]) < 2
                or record["nspid"][-1] != value["reported_wm_pid"]
                or not stat.S_ISREG(host["executable"]["mode"])
                or value["digest"] != hashlib.sha256(json.dumps(unsigned, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()):
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        raise IdentityUnavailable("native_namespace_owner_unverified") from None
    return value


def collect_namespace_owner(window_id: str, reported_wm_pid: int, bridge: Any) -> dict:
    """Complete bounded provider-root proof only; never recurse into observe."""
    def fresh_owner():
        remaining = bridge.deadline - time.monotonic()
        if remaining < .05:
            raise IdentityUnavailable("native_provider_identity_budget_exceeded")
        return bridge._read(lambda: xres_owner(window_id, os.environ, timeout_s=remaining))
    owner = fresh_owner()
    host = bridge._read(lambda: host_process_identity(owner["pid"]))
    if owner != {"pid": host["pid"], "start": host["start"]}:
        raise IdentityUnavailable("native_namespace_owner_changed")
    identity = {"window_id": window_id, "process_id": host["pid"]}
    providers = bridge._inactive_namespace_providers()
    matches = []
    for _app, pid, bus, _raw in providers:
        proof = provider_proof(identity, bus, pid, credentials=bridge._credentials,
                               read=bridge._read, deadline=bridge.deadline)
        if proof is not None:
            matches.append(proof)
    if len(matches) != 1:
        raise IdentityUnavailable("native_namespace_owner_not_unique")
    # A new provider cannot silently introduce another match after the scan.
    if ([(pid, bus, raw) for _, pid, bus, raw in bridge._inactive_namespace_providers()]
            != [(pid, bus, raw) for _, pid, bus, raw in providers]
            or bridge._read(lambda: host_process_identity(owner["pid"])) != host
            or fresh_owner() != owner):
        raise IdentityUnavailable("native_namespace_owner_changed")
    value = {"identity": identity, "reported_wm_pid": reported_wm_pid, "owner": owner,
             "host_process": host, "proof": matches[0]}
    value["digest"] = hashlib.sha256(json.dumps(value, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
    return _check_namespace_owner(value)


def _namespace_owner(window_id: str, reported_wm_pid: int, deadline: float) -> dict:
    import importlib.util

    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi, GLib
    spec = importlib.util.spec_from_file_location("_namespace_native", Path(__file__).with_name("native_ui.py"))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    bridge = native._AtspiBridge(Atspi, GLib, "namespace-owner-proof")
    bridge.deadline, bridge.reads = min(deadline - .05, time.monotonic() + 2), 0
    return collect_namespace_owner(window_id, reported_wm_pid, bridge)


def namespace_window_owner(window_id: str, reported_wm_pid: int, environment: Mapping[str, str], *,
                           deadline: float | None = None) -> dict:
    if (not isinstance(window_id, str) or not re.fullmatch(r"[1-9][0-9]{0,9}", window_id)
            or int(window_id) >= 2**32 or type(reported_wm_pid) is not int
            or not 0 < reported_wm_pid < 2**31):
        raise IdentityUnavailable("native_namespace_owner_invalid")
    deadline = min(deadline if deadline is not None else time.monotonic() + 3, time.monotonic() + 3)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise IdentityUnavailable("native_provider_identity_budget_exceeded")
    try:
        result = subprocess.run(["/usr/bin/python3", "-I", str(Path(__file__).resolve()),
            "--namespace-window-owner", window_id, str(reported_wm_pid), repr(deadline)], env=dict(environment),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=remaining, check=False)
        if len(result.stdout) > 65536:
            raise IdentityUnavailable("native_namespace_owner_unavailable")
        if result.returncode:
            failure = json.loads(result.stdout) if result.stdout else {}
            if (isinstance(failure, dict) and set(failure) == {"error"}
                    and isinstance(failure["error"], str) and failure["error"] in {"native_observation_budget_exceeded",
                                            "native_provider_identity_budget_exceeded"}):
                raise IdentityUnavailable(failure["error"])
            raise IdentityUnavailable("native_namespace_owner_unavailable")
        remaining = deadline - time.monotonic()
        if remaining < .05:
            raise IdentityUnavailable("native_provider_identity_budget_exceeded")
        value = _check_namespace_owner(json.loads(result.stdout))
        if (value["identity"]["window_id"] != window_id or value["reported_wm_pid"] != reported_wm_pid
                or host_process_identity(value["identity"]["process_id"]) != value["host_process"]
                or xres_owner(window_id, environment, timeout_s=remaining) != value["owner"]):
            raise IdentityUnavailable("native_namespace_owner_changed")
        if time.monotonic() >= deadline:
            raise IdentityUnavailable("native_provider_identity_budget_exceeded")
        return value
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise IdentityUnavailable("native_namespace_owner_unavailable") from None


if __name__ == "__main__":
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "--namespace-window-owner":
            if (re.fullmatch(r"[1-9][0-9]{0,9}", sys.argv[2]) is None or int(sys.argv[2]) >= 2**32
                    or re.fullmatch(r"[1-9][0-9]{0,9}", sys.argv[3]) is None
                    or int(sys.argv[3]) >= 2**31):
                raise IdentityUnavailable("native_namespace_owner_invalid")
            deadline = float(sys.argv[4])
            if not time.monotonic() < deadline <= time.monotonic() + 3:
                raise IdentityUnavailable("native_namespace_owner_invalid")
            print(json.dumps(_namespace_owner(sys.argv[2], int(sys.argv[3]), deadline)))
            raise SystemExit(0)
        if len(sys.argv) != 3 or sys.argv[1] != "--xres-owner":
            raise IdentityUnavailable("native_identity_operation_invalid")
        print(json.dumps(_xres(sys.argv[2])))
    except Exception as exc:
        if len(sys.argv) > 1 and sys.argv[1] == "--namespace-window-owner":
            code = getattr(exc, "code", str(exc) if isinstance(exc, IdentityUnavailable) else None)
            if not isinstance(code, str) or re.fullmatch(r"[a-z_][a-z0-9_]{0,127}", code) is None:
                code = "native_namespace_owner_unavailable"
            print(json.dumps({"error": code}))
        raise SystemExit(1) from None
