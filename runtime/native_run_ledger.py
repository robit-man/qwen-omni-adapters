"""Private, input-free cross-run admission for native desktop test ownership.

The ledger is evidence, never permission to close a window. Uncertain launches
remain pending. Releasing a lock or cancelling a task does not resolve them.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from itertools import islice
from pathlib import Path

MAX_PENDING = 32
MAX_LAUNCHES = 16
MAX_WINDOWS = 256
MAX_RECORD_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 65536
MAX_CLOSED = 128
MAX_EVENTS = 256
MAX_ENTRIES = 224
SCHEMA = "robit.omni.native-run-ledger.v1"
_ID = re.compile(r"[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")


_FLATPAK_APPLICATIONS = {"flatpak_firefox": "org.mozilla.firefox", "flatpak_chromium": "org.chromium.Chromium"}


class RunLedgerError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _json(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise RunLedgerError("native_run_ledger_invalid") from None


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def host_identity(pid):
    """Read host lifetime and actual UID; never interpret namespace PID as host."""
    path = Path("/proc") / str(pid)
    try:
        before = path.stat()
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        start = fields[19]
        after = path.stat()
        if before.st_uid != after.st_uid or before.st_ino != after.st_ino:
            raise OSError()
        return {"pid": str(pid), "start": start, "uid": after.st_uid}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError):
        raise RunLedgerError("native_run_process_identity_unavailable") from None


def desktop_session(environment):
    display = environment.get("DISPLAY", "")
    match = re.fullmatch(r":([0-9]{1,5})(?:\.[0-9]{1,5})?", display)
    if not match:
        raise RunLedgerError("native_run_desktop_session_unavailable")
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        raise RunLedgerError("native_run_desktop_session_unavailable") from None
    if not re.fullmatch(r"[0-9a-f-]{36}", boot):
        raise RunLedgerError("native_run_desktop_session_unavailable")
    return {"uid": os.getuid(), "boot_id": boot, "display": ":" + str(int(match[1]))}


def _directory(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        if part in {".", ".."}:
            raise RunLedgerError("native_run_ledger_unsafe_path")
        current /= part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = current.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid not in {0, os.getuid()}
            or info.st_mode & 0o022
            and not info.st_mode & stat.S_ISVTX
        ):
            raise RunLedgerError("native_run_ledger_unsafe_path")
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RunLedgerError("native_run_ledger_unsafe_path")
    return path


def _read(path, limit):
    try:
        path = Path(path).absolute()
        parent = Path(path.anchor)
        for component in path.parts[1:-1]:
            parent /= component
            info = parent.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, os.getuid()}:
                raise RunLedgerError("native_run_ledger_unsafe_path")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or not 0 < info.st_size <= limit
            ):
                raise RunLedgerError("native_run_ledger_unsafe_file")
            raw = source.read(limit + 1)
            if len(raw) != info.st_size or len(raw) > limit:
                raise RunLedgerError("native_run_ledger_oversized")
            return raw
    except OSError:
        raise RunLedgerError("native_run_ledger_read_unavailable") from None


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, UnicodeError, RecursionError):
        raise RunLedgerError("native_run_ledger_malformed") from None


def _atomic(path, raw, limit):
    if len(raw) > limit:
        raise RunLedgerError("native_run_ledger_oversized")
    if path.exists() or path.is_symlink():
        _read(path, limit)  # Refuse unsafe targets rather than replacing them.
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    published = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb", buffering=0) as target:
            if target.write(raw) != len(raw):
                raise RunLedgerError("native_run_ledger_short_write")
            os.fsync(target.fileno())
        os.replace(temporary, path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _read(path, limit) != raw:
            raise RunLedgerError("native_run_ledger_readback_mismatch")
    except OSError:
        raise RunLedgerError(
            "native_run_ledger_publication_indeterminate"
            if published
            else "native_run_ledger_write_failed"
        ) from None
    finally:
        if not published:
            temporary.unlink(missing_ok=True)


def _positive(value):
    return (
        isinstance(value, str) and value.isascii() and value.isdecimal() and 0 < int(value) < 2**63
    )


def _identity(value, uid):
    return (
        isinstance(value, dict)
        and set(value) == {"pid", "start", "uid"}
        and _positive(value["pid"])
        and _positive(value["start"])
        and type(value["uid"]) is int
        and value["uid"] == uid
    )


def _entries(directory):
    entries = list(islice(directory.iterdir(), MAX_ENTRIES + 1))
    if len(entries) > MAX_ENTRIES:
        raise RunLedgerError("native_run_ledger_oversized")
    return entries


def _verification_reference(reference):
    """Read one pinned private host artifact; never fetch or provision evidence."""
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not isinstance(reference["path"], str)
        or not 1 <= len(reference["path"]) <= 4096
        or not Path(reference["path"]).is_absolute()
        or not isinstance(reference["sha256"], str)
        or not _HASH.fullmatch(reference["sha256"])
    ):
        raise RunLedgerError("native_run_verified_settlement_reference_invalid")
    raw = _read(Path(reference["path"]), 65536)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise RunLedgerError("native_run_verified_settlement_hash_mismatch")
    return raw


def _verification_time(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


class NativeRunLedger:
    """One serialized desktop's bounded pending registry. No GUI operations."""

    @classmethod
    def open(cls, environment, *, state_root=None, identity_reader=host_identity):
        return cls(environment, state_root=state_root, identity_reader=identity_reader)

    def __init__(self, environment, *, state_root=None, identity_reader=host_identity):
        self.session = desktop_session(environment)
        self.identity_reader = identity_reader
        self.runner = identity_reader(os.getpid())
        if not _identity(self.runner, os.getuid()):
            raise RunLedgerError("native_run_runner_identity_unavailable")
        root = (
            Path(state_root)
            if state_root is not None
            else Path.home() / ".local/state/qwen-omni/native-test-runs"
        )
        self.directory = _directory(_directory(root) / _digest(self.session))
        self.lock_fd = None
        self.current = None
        self.admitted = False
        self.publication_failed = False
        try:
            fd = os.open(self.directory / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            self.lock_fd = fd
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise RunLedgerError("native_run_ledger_unsafe_file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunLedgerError("native_run_desktop_already_locked") from None
            path = self.directory / "index.json"
            if path.exists() or path.is_symlink():
                self.index = _decode(_read(path, MAX_INDEX_BYTES))
            else:
                if set(item.name for item in _entries(self.directory)) != {"lock"}:
                    raise RunLedgerError("native_run_ledger_orphaned")
                self.index = {
                    "schema": SCHEMA,
                    "session": self.session,
                    "pending": [],
                    "closed": [],
                    "migrations": [],
                }
                self._index()
            self._validate_index()
        except (KeyError, TypeError, ValueError, AttributeError):
            self.close()
            raise RunLedgerError("native_run_ledger_malformed") from None
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.lock_fd is None:
            raise RunLedgerError("native_run_ledger_closed")

    def _validate_index(self):
        value = self.index
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "session", "pending", "closed", "migrations"}
            or value["schema"] != SCHEMA
            or value["session"] != self.session
            or not isinstance(value["pending"], list)
            or len(value["pending"]) > MAX_PENDING
            or len(set(value["pending"])) != len(value["pending"])
            or not all(isinstance(item, str) and _ID.fullmatch(item) for item in value["pending"])
            or not isinstance(value["closed"], list)
            or len(value["closed"]) > MAX_CLOSED
            or not isinstance(value["migrations"], list)
            or len(value["migrations"]) > 8
            or not all(
                isinstance(item, str) and _HASH.fullmatch(item) for item in value["migrations"]
            )
        ):
            raise RunLedgerError("native_run_ledger_malformed")
        closed_ids = []
        for item in value["closed"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"run_id", "record_sha256", "retained_evidence_sha256"}
                or not isinstance(item["run_id"], str)
                or not _ID.fullmatch(item["run_id"])
                or not isinstance(item["record_sha256"], str)
                or not _HASH.fullmatch(item["record_sha256"])
                or not isinstance(item["retained_evidence_sha256"], str)
                or not _HASH.fullmatch(item["retained_evidence_sha256"])
            ):
                raise RunLedgerError("native_run_ledger_malformed")
            closed_ids.append(item["run_id"])
        if len(set(closed_ids)) != len(closed_ids) or set(closed_ids) & set(value["pending"]):
            raise RunLedgerError("native_run_ledger_malformed")
        entries = _entries(self.directory)
        allowed = {"lock", "index.json"} | {
            "record-" + item + ".json" for item in value["pending"] + closed_ids
        }
        if len(entries) > MAX_ENTRIES or any(item.name not in allowed for item in entries):
            raise RunLedgerError("native_run_ledger_orphaned")
        for item in entries:
            info = item.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise RunLedgerError("native_run_ledger_unsafe_file")
        for summary in value["closed"]:
            body = self._read_record(summary["run_id"])
            if body["state"] != "resolved" or _digest(body) != summary["record_sha256"]:
                raise RunLedgerError("native_run_ledger_archive_invalid")

    def _index(self):
        self._check()
        _atomic(self.directory / "index.json", _json(self.index), MAX_INDEX_BYTES)

    def _record_path(self, run_id):
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            raise RunLedgerError("native_run_ledger_malformed")
        return self.directory / ("record-" + run_id + ".json")

    def _validate_record(self, body):
        keys = {
            "schema",
            "session",
            "run_id",
            "runner",
            "run_root",
            "case_id",
            "launches",
            "windows",
            "scope_sha256",
            "events",
            "state",
            "outcome",
        }
        if (
            not isinstance(body, dict)
            or set(body) != keys
            or body["schema"] != SCHEMA
            or body["session"] != self.session
            or not isinstance(body["run_id"], str)
            or not _ID.fullmatch(body["run_id"])
            or not _identity(body["runner"], self.session["uid"])
            or not isinstance(body["run_root"], str)
            or len(body["run_root"]) > 4096
            or not Path(body["run_root"]).is_absolute()
            or not isinstance(body["case_id"], str)
            or not 1 <= len(body["case_id"]) <= 100
            or not isinstance(body["launches"], list)
            or len(body["launches"]) > MAX_LAUNCHES
            or not isinstance(body["windows"], list)
            or len(body["windows"]) > MAX_LAUNCHES
            or not isinstance(body["events"], list)
            or len(body["events"]) > MAX_EVENTS
            or body["scope_sha256"] != _digest(body["windows"])
            or body["state"] not in {"pending", "resolved"}
            or not isinstance(body["outcome"], dict)
        ):
            raise RunLedgerError("native_run_ledger_malformed")
        xids = set()
        for window in body["windows"]:
            if (
                not isinstance(window, dict)
                or set(window) != {"window_id", "owner", "launch_index"}
                or not _positive(window["window_id"])
                or int(window["window_id"]) >= 2**32
                or window["window_id"] in xids
                or not _identity(window["owner"], self.session["uid"])
                or type(window["launch_index"]) is not int
                or not 0 <= window["launch_index"] < len(body["launches"])
            ):
                raise RunLedgerError("native_run_ledger_malformed")
            xids.add(window["window_id"])
            launch = body["launches"][window["launch_index"]]
            if (
                not isinstance(launch, dict)
                or launch.get("registered") is not True
                or launch.get("window_id") != window["window_id"]
                or str(launch.get("pid")) != window["owner"]["pid"]
                or launch.get("process_start") != window["owner"]["start"]
            ):
                raise RunLedgerError("native_run_ledger_malformed")
            if launch.get("ownership_kind") in {"flatpak_firefox", "flatpak_chromium"}:
                proof = launch.get("ownership_evidence", {})
                if (
                    proof.get("window", {}).get("window_id") != window["window_id"]
                    or str(proof.get("window", {}).get("process_id")) != window["owner"]["pid"]
                    or proof.get("window", {}).get("process_start") != window["owner"]["start"]
                    or proof.get("namespace", {}).get("source") != "flatpak_namespace_metadata"
                ):
                    raise RunLedgerError("native_run_ledger_namespace_unproven")
            elif window["window_id"] in launch.get("before_xids", []) or window[
                "window_id"
            ] not in launch.get("new_visible_xids", []):
                raise RunLedgerError("native_run_ledger_registration_unproven")
        for launch in body["launches"]:
            if not isinstance(launch, dict) or type(launch.get("registered")) is not bool:
                raise RunLedgerError("native_run_ledger_malformed")
            for key in ("before_xids", "new_visible_xids"):
                if key in launch and (
                    not isinstance(launch[key], list)
                    or len(launch[key]) > MAX_WINDOWS
                    or not all(_positive(xid) and int(xid) < 2**32 for xid in launch[key])
                    or len(set(launch[key])) != len(launch[key])
                ):
                    raise RunLedgerError("native_run_ledger_malformed")
            if "spawn_failed_before_process" in launch and (
                launch["spawn_failed_before_process"] is not True
                or launch["registered"] is not False
                or any(key in launch for key in ("launch_pid", "launch_start", "candidate_xid"))
                or launch.get("new_visible_xids")
            ):
                raise RunLedgerError("native_run_ledger_malformed")
        for sequence, event in enumerate(body["events"], 1):
            if (
                not isinstance(event, dict)
                or set(event) != {"sequence", "phase", "launches_sha256", "scope_sha256"}
                or type(event["sequence"]) is not int
                or event["sequence"] != sequence
                or event["phase"]
                not in {
                    "launch_intent",
                    "spawned",
                    "spawn_failed",
                    "candidate",
                    "registered",
                    "namespace_final",
                    "cleanup",
                }
                or not isinstance(event["launches_sha256"], str)
                or not _HASH.fullmatch(event["launches_sha256"])
                or not isinstance(event["scope_sha256"], str)
                or not _HASH.fullmatch(event["scope_sha256"])
            ):
                raise RunLedgerError("native_run_ledger_malformed")
        registered = {window["launch_index"] for window in body["windows"]}
        if any(
            row.get("registered") is True and number not in registered
            for number, row in enumerate(body["launches"])
        ):
            raise RunLedgerError("native_run_ledger_registration_unproven")

    def _read_record(self, run_id):
        raw = _read(self._record_path(run_id), MAX_RECORD_BYTES)
        value = _decode(raw)
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "sha256"}
            or value["sha256"] != _digest(value["body"])
        ):
            raise RunLedgerError("native_run_ledger_seal_invalid")
        body = value["body"]
        try:
            self._validate_record(body)
        except (TypeError, KeyError, AttributeError):
            raise RunLedgerError("native_run_ledger_malformed") from None
        if body["run_id"] != run_id:
            raise RunLedgerError("native_run_ledger_malformed")
        return body

    def _publish(self, body, *, mirror=True):
        self._check()
        self._validate_record(body)
        raw = _json({"body": body, "sha256": _digest(body)})
        if len(raw) > MAX_RECORD_BYTES:
            raise RunLedgerError("native_run_ledger_oversized")
        # Registry first: failure of the evidence mirror still leaves a durable
        # pending intent; callers must not continue to launch or activate.
        _atomic(self._record_path(body["run_id"]), raw, MAX_RECORD_BYTES)
        if not mirror:
            return raw
        root = Path(body["run_root"])
        if not root.is_dir() or root.is_symlink():
            raise RunLedgerError("native_run_evidence_directory_unavailable")
        root = _directory(root)
        _atomic(root / "native-run-ownership.json", raw, MAX_RECORD_BYTES)
        return raw

    def _inventory(self, inventory):
        if (
            not isinstance(inventory, dict)
            or inventory.get("available") is not True
            or inventory.get("coverage") != "managed_client_windows_only"
            or not isinstance(inventory.get("windows"), dict)
            or len(inventory["windows"]) > MAX_WINDOWS
            or any(not _positive(xid) or int(xid) >= 2**32 for xid in inventory["windows"])
        ):
            raise RunLedgerError("native_run_inventory_unavailable")
        return inventory["windows"]

    def _absence(self, body, inventory, *, launches_absent=None):
        for index, launch in enumerate(body["launches"]):
            if (
                launch.get("registered") is not True
                and launch.get("spawn_failed_before_process") is not True
                and not self._launch_proven_absent(index, launch, launches_absent, inventory)
            ):
                raise RunLedgerError("native_run_uncertain_prior_launch")
        for window in body["windows"]:
            xid = window["window_id"]
            if xid not in inventory:
                continue
            row = inventory[xid]
            owner = window["owner"]
            if (
                not isinstance(row, dict)
                or row.get("pid") != owner["pid"]
                or row.get("start_ticks") != owner["start"]
            ):
                raise RunLedgerError("native_run_prior_identity_changed_or_unknown")
            live = self.identity_reader(owner["pid"])
            if live != owner:
                raise RunLedgerError("native_run_prior_identity_changed_or_unknown")
            raise RunLedgerError("native_run_prior_owned_window_residual")

    def _native_launch_proven_absent(self, index, launch, launches_absent, inventory):
        """A native launch runs in its own session (start_new_session); its proof is
        that the exact launch process is gone, no live process remains in that
        session, and every window it produced is absent from the inventory."""
        pid, start = launch.get("launch_pid"), launch.get("launch_start")
        if type(pid) is not int or not _positive(str(pid)) or not _positive(start):
            return False
        produced = sorted(set(launch.get("new_visible_xids", [])) | (
            {launch["candidate_xid"]} if "candidate_xid" in launch else set()))
        proof = [row for row in launches_absent if isinstance(row, dict) and row.get("index") == index]
        if (
            len(proof) != 1
            or proof[0] != {"index": index, "ownership_kind": "native_session",
                            "launch": {"pid": str(pid), "start": start, "present": False},
                            "session_processes": 0, "xids_absent": produced}
            or any(xid in inventory for xid in produced)
        ):
            return False
        live = self.identity_reader(str(pid))
        return live is None or live.get("start") != start

    def _launch_proven_absent(self, index, launch, launches_absent, inventory):
        if launches_absent is not None and "ownership_kind" not in launch:
            return self._native_launch_proven_absent(index, launch, launches_absent, inventory)
        """Only a verified host settlement may clear an unregistered Flatpak launch.

        A foreground ``flatpak run`` execs into the instance's bwrap process, so
        the recorded launch identity is that sandbox. It must be absent now
        (re-read here, not trusted) and the host verifier must have attributed
        every live window-capable instance of the same application elsewhere.
        """
        if launches_absent is None or launch.get("ownership_kind") not in _FLATPAK_APPLICATIONS:
            return False
        receipt = launch.get("last_ownership_receipt")
        started = receipt.get("launch") if isinstance(receipt, dict) else None
        if not isinstance(started, dict) or type(started.get("pid")) is not int:
            return False
        pid, start = str(started["pid"]), started.get("start")
        if not _positive(pid) or not _positive(start):
            return False
        proof = [row for row in launches_absent if isinstance(row, dict) and row.get("index") == index]
        if (
            len(proof) != 1
            or set(proof[0]) != {"index", "ownership_kind", "application_id", "launch", "live_instances"}
            or proof[0]["ownership_kind"] != launch["ownership_kind"]
            or proof[0]["application_id"] != _FLATPAK_APPLICATIONS[launch["ownership_kind"]]
            or proof[0]["launch"] != {"pid": pid, "start": start, "present": False}
            or not isinstance(proof[0]["live_instances"], list)
            or len(proof[0]["live_instances"]) > MAX_WINDOWS
        ):
            return False
        for row in proof[0]["live_instances"]:
            if (
                not isinstance(row, dict)
                or row.get("attribution") not in {"no_display_socket", "separate_main_instance"}
                or row.get("attribution") == "separate_main_instance"
                and (row.get("pid") == pid and row.get("start") == start or row.get("sandbox") is not False)
            ):
                return False
        live = self.identity_reader(pid)
        return live is None or live.get("start") != start

    def _archive(self, body, *, historical_evidence=None):
        body = copy.deepcopy(body)
        body["state"] = "resolved"
        raw = self._publish(body, mirror=historical_evidence is None)
        evidence_sha = historical_evidence or hashlib.sha256(raw).hexdigest()
        summary = {
            "run_id": body["run_id"],
            "record_sha256": _digest(body),
            "retained_evidence_sha256": evidence_sha,
        }
        index = copy.deepcopy(self.index)
        index["pending"].remove(body["run_id"])
        index["closed"].append(summary)
        evicted = index["closed"][:-MAX_CLOSED]
        index["closed"] = index["closed"][-MAX_CLOSED:]
        self.index = index
        self._index()
        # Only registry cache copies are pruned, never original run artifacts.
        for item in evicted:
            old = self._read_record(item["run_id"])
            if old["state"] != "resolved" or _digest(old) != item["record_sha256"]:
                raise RunLedgerError("native_run_ledger_archive_invalid")
            mirror = _read(Path(old["run_root"]) / "native-run-ownership.json", MAX_RECORD_BYTES)
            if hashlib.sha256(mirror).hexdigest() != item["retained_evidence_sha256"]:
                raise RunLedgerError("native_run_ledger_archive_invalid")
            self._record_path(item["run_id"]).unlink()
        return summary

    def _migrate(self, refs):
        if not isinstance(refs, (list, tuple)) or len(refs) > 8:
            raise RunLedgerError("native_run_migration_invalid")
        documents = []
        added = set()
        pending = set(self.index["pending"])
        closed = {row["run_id"] for row in self.index["closed"]}
        # Preflight ALL references before changing even the private import
        # index. Evidence roots are read-only, including already imported refs.
        for reference in refs:
            if (
                not isinstance(reference, dict)
                or set(reference) != {"path", "sha256"}
                or not isinstance(reference["sha256"], str)
                or not _HASH.fullmatch(reference["sha256"])
            ):
                raise RunLedgerError("native_run_migration_invalid")
            raw = _read(Path(reference["path"]), MAX_RECORD_BYTES)
            if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise RunLedgerError("native_run_migration_hash_mismatch")
            document = _decode(raw)
            if (
                not isinstance(document, dict)
                or set(document) != {"schema", "session", "source_journals", "records"}
                or document["schema"] != "robit.omni.native-run-migration.v1"
                or document["session"] != self.session
                or not isinstance(document["records"], list)
                or len(document["records"]) > MAX_PENDING
                or not isinstance(document["source_journals"], list)
                or not 1 <= len(document["source_journals"]) <= 8
            ):
                raise RunLedgerError("native_run_migration_invalid")
            for source in document["source_journals"]:
                if (
                    not isinstance(source, dict)
                    or set(source) != {"path", "sha256"}
                    or hashlib.sha256(_read(Path(source["path"]), MAX_RECORD_BYTES)).hexdigest()
                    != source["sha256"]
                ):
                    raise RunLedgerError("native_run_migration_hash_mismatch")
            for body in document["records"]:
                self._validate_record(body)
                if body["state"] != "pending":
                    raise RunLedgerError("native_run_migration_invalid")
                mirror = _read(
                    Path(body["run_root"]) / "native-run-ownership.json", MAX_RECORD_BYTES
                )
                if mirror != _json({"body": body, "sha256": _digest(body)}):
                    raise RunLedgerError("native_run_migration_evidence_mismatch")
            if reference["sha256"] in self.index["migrations"]:
                continue
            if reference["sha256"] in added:
                raise RunLedgerError("native_run_migration_invalid")
            added.add(reference["sha256"])
            if len(self.index["migrations"]) + len(added) > 8:
                raise RunLedgerError("native_run_migration_limit")
            for body in document["records"]:
                if body["run_id"] in pending | closed:
                    raise RunLedgerError("native_run_migration_invalid")
                pending.add(body["run_id"])
                if len(pending) > MAX_PENDING:
                    raise RunLedgerError("native_run_pending_limit")
            documents.append((reference["sha256"], document["records"]))
        for digest, records in documents:
            for body in records:
                self.index["pending"].append(body["run_id"])
                self._index()
                self._publish(body, mirror=False)
            self.index["migrations"].append(digest)
            self._index()

    def reconcile(self, inventory, *, migration_refs=()):
        self._check()
        current = self._inventory(inventory)
        self._migrate(migration_refs)
        resolved = []
        for run_id in list(self.index["pending"]):
            body = self._read_record(run_id)
            active = self.identity_reader(body["runner"]["pid"])
            if active == body["runner"]:
                raise RunLedgerError("native_run_prior_runner_active")
            if body["outcome"].get("cleanup_uncertain") is True:
                raise RunLedgerError("native_run_prior_cleanup_uncertain")
            # Reused runner PID is not the old run; window lifetimes remain
            # independently checked. An unavailable /proc read raises above.
            self._absence(body, current)
            mirror = _read(Path(body["run_root"]) / "native-run-ownership.json", MAX_RECORD_BYTES)
            if _decode(mirror) != {"body": body, "sha256": _digest(body)}:
                raise RunLedgerError("native_run_historical_evidence_mismatch")
            original_sha = hashlib.sha256(mirror).hexdigest()
            body["outcome"] = {
                "status": "reconciled_absent",
                "complete_inventory_sha256": _digest(inventory),
                "complete_inventory": copy.deepcopy(inventory),
                "retained_original_evidence_sha256": original_sha,
            }
            summary = self._archive(body, historical_evidence=original_sha)
            resolved.append({**summary, "resolution": copy.deepcopy(body["outcome"])})
        self.admitted = True
        return {
            "status": "admitted",
            "resolved": resolved,
            "desktop_session": copy.deepcopy(self.session),
            "unknown_preexisting_windows_preserved": True,
            "historical_ownership_complete": False,
        }

    def reconcile_verified_settlement(self, inventory, *, settlement_ref):
        """Resolve one pinned uncertain run using a trusted live host verifier.

        This is not authentication of arbitrary JSON or an input/close tool.
        The running host operator is responsible for its typed observations.
        Ordinary reconcile deliberately retains its sticky uncertainty guard.
        """
        self._check()
        if self.current is not None:
            raise RunLedgerError("native_run_admission_required")
        if self.publication_failed:
            raise RunLedgerError("native_run_checkpoint_publication_failed")
        current = self._inventory(inventory)
        receipt = _decode(_verification_reference(settlement_ref))
        fields = {
            "schema", "run_id", "session", "pending_body_sha256", "pending_record_sha256",
            "mirror_sha256", "scope_sha256", "launches_sha256", "inventory_sha256", "producer",
            "method", "observed_at_unix", "expires_at_unix", "settlement_evidence", "cleanup_evidence",
        }
        hashes = {
            "pending_body_sha256", "pending_record_sha256", "mirror_sha256", "scope_sha256",
            "launches_sha256", "inventory_sha256",
        }
        if (
            not isinstance(receipt, dict)
            or set(receipt) != fields
            or receipt["schema"] != "robit.omni.native-verified-settlement.v1"
            or not isinstance(receipt["run_id"], str)
            or not _ID.fullmatch(receipt["run_id"])
            or _json(receipt["session"]) != _json(self.session)
            or any(not isinstance(receipt[key], str) or not _HASH.fullmatch(receipt[key]) for key in hashes)
            or not _identity(receipt["producer"], self.session["uid"])
            or receipt["producer"] != self.runner
            or self.identity_reader(self.runner["pid"]) != self.runner
        ):
            raise RunLedgerError("native_run_verified_settlement_invalid")
        now = time.time()
        observed, expiry = receipt["observed_at_unix"], receipt["expires_at_unix"]
        if (
            not _verification_time(observed)
            or not _verification_time(expiry)
            or not observed <= now < expiry <= observed + 300
        ):
            raise RunLedgerError("native_run_verified_settlement_stale")
        method_raw = _verification_reference(receipt["method"])
        main = getattr(sys.modules.get("__main__"), "__file__", None)
        if (
            not isinstance(main, str)
            or receipt["method"]["path"] != str(Path(main).absolute())
            or hashlib.sha256(method_raw).hexdigest() != receipt["method"]["sha256"]
        ):
            raise RunLedgerError("native_run_verified_settlement_method_mismatch")
        run_id = receipt["run_id"]
        if run_id not in self.index["pending"]:
            raise RunLedgerError("native_run_verified_settlement_not_pending")
        body = self._read_record(run_id)
        record_raw = _read(self._record_path(run_id), MAX_RECORD_BYTES)
        mirror = _read(Path(body["run_root"]) / "native-run-ownership.json", MAX_RECORD_BYTES)
        if (
            body["state"] != "pending"
            or _decode(record_raw) != {"body": body, "sha256": _digest(body)}
            or _decode(mirror) != {"body": body, "sha256": _digest(body)}
            or receipt["pending_body_sha256"] != _digest(body)
            or receipt["pending_record_sha256"] != hashlib.sha256(record_raw).hexdigest()
            or receipt["mirror_sha256"] != hashlib.sha256(mirror).hexdigest()
            or receipt["scope_sha256"] != body["scope_sha256"]
            or receipt["launches_sha256"] != _digest(body["launches"])
            or receipt["inventory_sha256"] != _digest(inventory)
        ):
            raise RunLedgerError("native_run_verified_settlement_binding_mismatch")
        if body["outcome"].get("code") == "native_run_checkpoint_publication_failed":
            raise RunLedgerError("native_run_checkpoint_publication_failed")
        if body["outcome"].get("cleanup_uncertain") is not True:
            raise RunLedgerError("native_run_verified_settlement_not_uncertain")
        active = self.identity_reader(body["runner"]["pid"])
        if active == body["runner"]:
            raise RunLedgerError("native_run_prior_runner_active")
        if active is not None and not _identity(active, self.session["uid"]):
            raise RunLedgerError("native_run_prior_runner_unavailable")
        cleanup_document = _decode(_verification_reference(receipt["cleanup_evidence"]))
        observations = cleanup_document.get("observations") if isinstance(cleanup_document, dict) else None
        self._absence(body, current, launches_absent=(
            observations.get("unregistered_launches_absent", []) if isinstance(observations, dict) else []))
        evidence_fields = {
            "schema", "kind", "run_id", "session_sha256", "pending_record_sha256", "producer",
            "method", "observed_at_unix", "passed", "observations",
        }
        for key, kind in (("settlement_evidence", "downstream_settlement"), ("cleanup_evidence", "owned_cleanup")):
            evidence = _decode(_verification_reference(receipt[key]))
            if (
                not isinstance(evidence, dict)
                or set(evidence) != evidence_fields
                or evidence["schema"] != "robit.omni.native-host-verification.v1"
                or evidence["kind"] != kind
                or evidence["run_id"] != run_id
                or evidence["session_sha256"] != _digest(self.session)
                or evidence["pending_record_sha256"] != receipt["pending_record_sha256"]
                or not _identity(evidence["producer"], self.session["uid"])
                or evidence["producer"] != receipt["producer"]
                or evidence["method"] != receipt["method"]
                or evidence["passed"] is not True
                or not isinstance(evidence["observations"], dict)
                or not evidence["observations"]
                or not _verification_time(evidence["observed_at_unix"])
                or not 0 <= now - evidence["observed_at_unix"] <= 300
                or evidence["observed_at_unix"] > observed
            ):
                raise RunLedgerError("native_run_verified_settlement_evidence_invalid")
            if kind == "owned_cleanup" and (
                evidence["observations"].get("inventory_sha256") != _digest(inventory)
                or evidence["observations"].get("owned_window_ids_absent")
                != sorted(window["window_id"] for window in body["windows"])
            ):
                raise RunLedgerError("native_run_verified_settlement_evidence_invalid")
        original_sha = hashlib.sha256(mirror).hexdigest()
        body["outcome"] = {
            "status": "reconciled_verified_settlement",
            "prior_outcome": copy.deepcopy(body["outcome"]),
            "verified_settlement": {"reference": copy.deepcopy(settlement_ref), "receipt": receipt},
            "complete_inventory_sha256": _digest(inventory),
            "complete_inventory": copy.deepcopy(inventory),
            "retained_original_evidence_sha256": original_sha,
        }
        # Reject over-limit audit data before the first registry write.
        if len(_json({"body": {**body, "state": "resolved"}, "sha256": _digest({**body, "state": "resolved"})})) > MAX_RECORD_BYTES:
            raise RunLedgerError("native_run_ledger_oversized")
        summary = self._archive(body, historical_evidence=original_sha)
        return {"state": "resolved", **summary, "resolution": copy.deepcopy(body["outcome"])}

    def begin(self, run_root, case_id):
        self._check()
        if (
            not self.admitted
            or self.current is not None
            or len(self.index["pending"]) >= MAX_PENDING
        ):
            raise RunLedgerError("native_run_admission_required")
        run_id = uuid.uuid4().hex
        body = {
            "schema": SCHEMA,
            "session": self.session,
            "run_id": run_id,
            "runner": self.runner,
            "run_root": str(_directory(Path(run_root))),
            "case_id": case_id,
            "launches": [],
            "windows": [],
            "scope_sha256": _digest([]),
            "events": [],
            "state": "pending",
            "outcome": {},
        }
        self.index["pending"].append(run_id)
        self._index()
        self.current = body
        self._publish(body)
        return {
            "run_id": run_id,
            "desktop_session": copy.deepcopy(self.session),
            "state": "pending",
        }

    def publish_launches(self, launches, windows, *, phase):
        self._check()
        if self.current is None or phase not in {
            "launch_intent",
            "spawned",
            "spawn_failed",
            "candidate",
            "registered",
            "namespace_final",
            "cleanup",
        }:
            raise RunLedgerError("native_run_admission_required")
        if len(launches) > MAX_LAUNCHES or len(windows) > MAX_LAUNCHES:
            raise RunLedgerError("native_run_launch_limit")
        body = copy.deepcopy(self.current)
        body["launches"] = copy.deepcopy(launches)
        known = {item["window_id"]: item for item in body["windows"]}
        for window in windows:
            matches = [
                number
                for number, row in enumerate(launches)
                if row.get("registered") is True
                and row.get("window_id") == window.window_id
                and str(row.get("pid")) == window.pid
                and row.get("process_start") == window.process_start
            ]
            if len(matches) != 1:
                raise RunLedgerError("native_run_ledger_registration_unproven")
            identity = self.identity_reader(window.pid)
            if (
                not _identity(identity, self.session["uid"])
                or identity["start"] != window.process_start
            ):
                raise RunLedgerError("native_run_window_identity_unavailable")
            known[window.window_id] = {
                "window_id": window.window_id,
                "owner": identity,
                "launch_index": matches[0],
            }
        body["windows"] = list(known.values())
        body["scope_sha256"] = _digest(body["windows"])
        body["events"].append(
            {
                "sequence": len(body["events"]) + 1,
                "phase": phase,
                "launches_sha256": _digest(body["launches"]),
                "scope_sha256": body["scope_sha256"],
            }
        )
        if len(body["events"]) > MAX_EVENTS:
            raise RunLedgerError("native_run_checkpoint_limit")
        self.current = body
        try:
            self._publish(body)
        except Exception:
            self.publication_failed = True
            raise

    def finish(self, inventory, *, cleanup, settled):
        self._check()
        if self.current is None:
            return {"state": "not_started"}
        body = copy.deepcopy(self.current)
        body["outcome"] = {
            "cleanup_verified": cleanup.get("owned_windows_closed") is True,
            "settled": settled is True,
            "status": "pending",
        }
        no_spawn = not body["windows"] and all(
            row.get("spawn_failed_before_process") is True for row in body["launches"]
        )
        receipts = cleanup.get("cleanup", [])
        receipt_unknown = (
            not isinstance(receipts, list)
            or len(receipts) > MAX_LAUNCHES
            or any(
                not isinstance(row, dict)
                or any(
                    row.get(key) is True
                    for key in ("identity_or_close_unavailable", "uncertain_unregistered_launch")
                )
                for row in receipts
            )
        )
        body["outcome"]["cleanup_uncertain"] = (
            settled is not True
            or cleanup.get("owned_windows_retained") is True
            or cleanup.get("status") == "cleanup_deferred"
            or receipt_unknown
            and not no_spawn
            or self.publication_failed
        )
        try:
            current = self._inventory(inventory)
            if self.publication_failed:
                raise RunLedgerError("native_run_checkpoint_publication_failed")
            if body["outcome"]["cleanup_uncertain"] or (
                not body["outcome"]["cleanup_verified"] and not no_spawn
            ):
                raise RunLedgerError("native_run_cleanup_unresolved")
            self._absence(body, current)
        except RunLedgerError as exc:
            body["outcome"]["code"] = exc.code
            self._publish(body)
            self.current = body
            return {"state": "pending", "run_id": body["run_id"], "code": exc.code}
        body["outcome"].update(
            status="resolved_absent",
            complete_inventory_sha256=_digest(inventory),
            complete_inventory=copy.deepcopy(inventory),
        )
        summary = self._archive(body)
        self.current = None
        return {"state": "resolved", **summary}

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
