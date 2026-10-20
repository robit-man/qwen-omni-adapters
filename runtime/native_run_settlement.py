"""Host verifier that settles one pending, cleanup-uncertain native run from fresh evidence.

Usage (dry run prints the evidence; ``--apply`` records the settlement):

    .venv/bin/python -m runtime.native_run_settlement --run-id RUN_ID [--apply]

This is the pinned ``method`` of ``NativeRunLedger.reconcile_verified_settlement``.
It never sends input, closes windows or kills processes. It only reads current
host facts:

* the old runner and every recorded owned window are absent (the ledger
  re-checks both);
* each unregistered Flatpak browser launch's exact process identity is gone.
  A foreground ``flatpak run`` execs into the instance's bwrap process, so that
  identity is the launch's window-capable sandbox;
* an unregistered native launch (started in its own session) has no live
  process left in that session and none of its produced windows remains;
* every live instance of the same application is attributed elsewhere: either
  it has no display socket (cannot own a top-level window) or it is a separate
  main instance with a different process identity. A window-capable
  sub-sandbox, an unreadable instance or any other launch kind stays blocked.

The original failed record and its run-root mirror are never rewritten; the
ledger appends the verified settlement as the resolution.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from runtime.long_horizon_desktop import managed_inventory
from runtime.native_run_ledger import (
    _FLATPAK_APPLICATIONS,
    NativeRunLedger,
    RunLedgerError,
    _digest,
    _directory,
    _json,
    host_identity,
)

_DISPLAY_SOCKETS = {"x11", "fallback-x11", "wayland"}


class SettlementBlocked(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def flatpak_instances(application_id: str, *, runtime_dir: str | None = None,
                      environment: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Live instances of one application with their sandbox and display-socket facts."""
    runtime_dir = runtime_dir or os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    result = subprocess.run(["flatpak", "ps", "--columns=instance,pid,application"],  # noqa: S603, S607
                            capture_output=True, text=True, timeout=10, check=False, env=environment)
    if result.returncode != 0:
        raise SettlementBlocked("flatpak_instance_inventory_unavailable")
    rows = []
    for line in result.stdout.splitlines()[:256]:
        fields = line.split()
        if len(fields) != 3 or fields[2] != application_id:
            continue
        instance, pid = fields[0], fields[1]
        if not instance.isdecimal() or not pid.isdecimal():
            raise SettlementBlocked("flatpak_instance_inventory_invalid")
        info = configparser.RawConfigParser(strict=False, interpolation=None)
        try:
            with open(Path(runtime_dir) / ".flatpak" / instance / "info", encoding="utf-8") as handle:
                info.read_file(handle)
            name = info.get("Application", "name")
            sandbox = info.get("Instance", "sandbox", fallback="false").strip() == "true"
            sockets = {item for item in info.get("Context", "sockets", fallback="").split(";") if item}
        except (OSError, configparser.Error, UnicodeDecodeError):
            raise SettlementBlocked("flatpak_instance_metadata_unavailable") from None
        if name != application_id:
            raise SettlementBlocked("flatpak_instance_metadata_mismatch")
        identity = host_identity(pid)
        if identity is None:
            continue  # Exited since listing; it owns nothing now.
        rows.append({"instance": instance, "pid": pid, "start": identity["start"], "sandbox": sandbox,
                     "display_sockets": sorted(sockets & _DISPLAY_SOCKETS)})
    return rows


def session_processes(session_id: str) -> int:
    """Live host processes still in one session; unreadable rows count as present."""
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        except (OSError, IndexError):
            count += 1
            continue
        count += fields[3] == session_id
    return count


def native_launch_absence(index: int, launch: dict[str, Any], inventory: dict[str, Any], *,
                          identity_reader=host_identity, sessions=session_processes) -> dict[str, Any]:
    pid, start = launch.get("launch_pid"), launch.get("launch_start")
    if type(pid) is not int or not isinstance(start, str):
        raise SettlementBlocked("unregistered_launch_identity_unrecorded")
    live = identity_reader(str(pid))
    if live is not None and live.get("start") == start:
        raise SettlementBlocked("unregistered_launch_process_present")
    # Launches use start_new_session, so the session ID is the launch PID.
    if sessions(str(pid)) != 0:
        raise SettlementBlocked("unregistered_launch_session_present")
    produced = sorted(set(launch.get("new_visible_xids", [])) | (
        {launch["candidate_xid"]} if "candidate_xid" in launch else set()))
    if any(xid in inventory["windows"] for xid in produced):
        raise SettlementBlocked("unregistered_launch_window_present")
    return {"index": index, "ownership_kind": "native_session",
            "launch": {"pid": str(pid), "start": start, "present": False},
            "session_processes": 0, "xids_absent": produced}


def launch_absence(index: int, launch: dict[str, Any], *, instances=flatpak_instances,
                   identity_reader=host_identity) -> dict[str, Any]:
    kind = launch.get("ownership_kind")
    if kind not in _FLATPAK_APPLICATIONS:
        raise SettlementBlocked("unregistered_launch_kind_unsupported")
    started = (launch.get("last_ownership_receipt") or {}).get("launch") or {}
    if type(started.get("pid")) is not int or not isinstance(started.get("start"), str):
        raise SettlementBlocked("unregistered_launch_identity_unrecorded")
    pid, start = str(started["pid"]), started["start"]
    live = identity_reader(pid)
    if live is not None and live.get("start") == start:
        raise SettlementBlocked("unregistered_launch_process_present")
    application = _FLATPAK_APPLICATIONS[kind]
    attributed = []
    for row in instances(application):
        if not row["display_sockets"]:
            attribution = "no_display_socket"
        elif not row["sandbox"] and (row["pid"], row["start"]) != (pid, start):
            attribution = "separate_main_instance"
        else:
            raise SettlementBlocked("window_capable_instance_unattributed")
        attributed.append({**row, "attribution": attribution})
    return {"index": index, "ownership_kind": kind, "application_id": application,
            "launch": {"pid": pid, "start": start, "present": False}, "live_instances": attributed}


def _private(directory: Path, name: str, document: dict[str, Any]) -> dict[str, str]:
    path = directory / name
    raw = _json(document)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    return {"path": str(path.absolute()), "sha256": hashlib.sha256(raw).hexdigest()}


def settle(run_id: str, *, apply: bool, environment: dict[str, str] | None = None,
           state_root: str | None = None, directory: Path | None = None) -> dict[str, Any]:
    environment = dict(os.environ if environment is None else environment)
    ledger = NativeRunLedger.open(environment, state_root=state_root)
    try:
        if run_id not in ledger.index["pending"]:
            raise SettlementBlocked("run_not_pending")
        body = ledger._read_record(run_id)
        if body["outcome"].get("cleanup_uncertain") is not True:
            raise SettlementBlocked("run_not_cleanup_uncertain")
        runner = host_identity(body["runner"]["pid"])
        if runner == body["runner"]:
            raise SettlementBlocked("prior_runner_active")
        inventory = managed_inventory(environment)
        if inventory.get("available") is not True:
            raise SettlementBlocked("managed_inventory_unavailable")
        present = sorted(window["window_id"] for window in body["windows"] if window["window_id"] in inventory["windows"])
        if present:
            raise SettlementBlocked("recorded_owned_window_present")
        launches = [launch_absence(index, launch) if "ownership_kind" in launch
                    else native_launch_absence(index, launch, inventory)
                    for index, launch in enumerate(body["launches"])
                    if launch.get("registered") is not True and launch.get("spawn_failed_before_process") is not True]
        record_raw = ledger._record_path(run_id).read_bytes()
        mirror_raw = (Path(body["run_root"]) / "native-run-ownership.json").read_bytes()
        method_path = Path(sys.modules["__main__"].__file__).absolute()
        method = {"path": str(method_path), "sha256": hashlib.sha256(method_path.read_bytes()).hexdigest()}
        observed = time.time()
        shared = {"schema": "robit.omni.native-host-verification.v1", "run_id": run_id,
                  "session_sha256": _digest(ledger.session),
                  "pending_record_sha256": hashlib.sha256(record_raw).hexdigest(),
                  "producer": ledger.runner, "method": method, "observed_at_unix": observed, "passed": True}
        settlement = {**shared, "kind": "downstream_settlement", "observations": {
            "prior_runner": {**body["runner"], "present": False}}}
        cleanup = {**shared, "kind": "owned_cleanup", "observations": {
            "inventory_sha256": _digest(inventory),
            "owned_window_ids_absent": sorted(window["window_id"] for window in body["windows"]),
            "unregistered_launches_absent": launches}}
        report = {"run_id": run_id, "case_id": body["case_id"], "apply": apply,
                  "settlement": settlement, "cleanup": cleanup}
        if not apply:
            return {"status": "verified_dry_run", **report}
        if directory is None or method_path.parent != Path(directory).absolute():
            raise SettlementBlocked("private_method_copy_required")
        receipt = {"schema": "robit.omni.native-verified-settlement.v1", "run_id": run_id,
                   "session": ledger.session, "pending_body_sha256": _digest(body),
                   "pending_record_sha256": hashlib.sha256(record_raw).hexdigest(),
                   "mirror_sha256": hashlib.sha256(mirror_raw).hexdigest(), "scope_sha256": body["scope_sha256"],
                   "launches_sha256": _digest(body["launches"]), "inventory_sha256": _digest(inventory),
                   "producer": ledger.runner, "method": method, "observed_at_unix": observed,
                   "expires_at_unix": observed + 120,
                   "settlement_evidence": _private(directory, "settlement.json", settlement),
                   "cleanup_evidence": _private(directory, "cleanup.json", cleanup)}
        reference = _private(directory, "receipt.json", receipt)
        resolved = ledger.reconcile_verified_settlement(inventory, settlement_ref=reference)
        return {"status": "settled", **report, "receipt": reference, "resolution_state": resolved["state"]}
    finally:
        ledger.close()


def _private_copy(run_id: str) -> Path:
    """Owner-only evidence directory holding the exact verifier bytes that will run."""
    root = _directory(Path.home() / ".local/state/qwen-omni/native-run-settlements")
    directory = root / f"{run_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    directory.mkdir(mode=0o700)
    copy = directory / "native_run_settlement.py"
    descriptor = os.open(copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(Path(__file__).read_bytes())
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--private-copy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.apply and not args.private_copy:
        # The ledger pins __main__ by path and hash and reads only owner-only
        # files, so the settlement runs from a private copy of this source.
        try:
            directory = _private_copy(args.run_id)
        except (OSError, RunLedgerError) as exc:
            print(json.dumps({"status": "blocked", "code": getattr(exc, "code", "private_copy_unavailable")}))
            return 2
        root = str(Path(__file__).absolute().parents[1])
        return subprocess.run([sys.executable, "-I", "-c",  # noqa: S603 - fixed private copy, no shell.
            "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); sys.argv[0] = sys.argv[1]; "
            "runpy.run_path(sys.argv.pop(1), run_name='__main__')",
            root, str(directory / "native_run_settlement.py"), "--run-id", args.run_id, "--apply", "--private-copy"],
            check=False).returncode
    try:
        directory = Path(sys.modules["__main__"].__file__).absolute().parent if args.private_copy else None
        result = settle(args.run_id, apply=args.apply, directory=directory)
    except (SettlementBlocked, RunLedgerError) as exc:
        print(json.dumps({"status": "blocked", "code": getattr(exc, "code", str(exc))}))
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
