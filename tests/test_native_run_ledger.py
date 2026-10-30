"""CPU admission/durability controls; no desktop input, models or services."""

import copy
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from runtime import native_run_ledger as ledger
from runtime.acceptance_native_support import OwnedWindow


@pytest.fixture
def harness(tmp_path):
    identities = {
        str(os.getpid()): {"pid": str(os.getpid()), "start": "10", "uid": os.getuid()},
        "20000": {"pid": "20000", "start": "20", "uid": os.getuid()},
    }

    def reader(pid):
        return copy.deepcopy(identities.get(str(pid)))

    root = tmp_path / "state"

    def open_ledger():
        return ledger.NativeRunLedger.open(
            {"DISPLAY": ":8765.0"}, state_root=root, identity_reader=reader
        )

    return root, identities, open_ledger


def inventory(windows=None, **changes):
    return {
        "available": True,
        "coverage": "managed_client_windows_only",
        "windows": windows or {},
        **changes,
    }


def start(harness, tmp_path):
    _, _, opening = harness
    value = opening()
    value.reconcile(inventory({"999": {"identity_available": False}}))
    work = tmp_path / "run"
    work.mkdir(mode=0o700)
    value.begin(work, "LHN-047")
    return value, work


def register(value):
    launch = {
        "application": "editor",
        "window_class": "Editor",
        "registered": False,
        "before_xids": ["999"],
    }
    value.publish_launches([launch], [], phase="launch_intent")
    launch.update(launch_pid=20000, launch_start="20")
    value.publish_launches([launch], [], phase="spawned")
    launch.update(
        candidate_xid="111",
        new_visible_xids=["111"],
        registered=True,
        window_id="111",
        pid="20000",
        process_start="20",
    )
    value.publish_launches(
        [launch], [OwnedWindow("editor", "111", "20000", "20")], phase="registered"
    )
    return launch


def abandon(value, harness):
    value.close()
    harness[1].pop(str(os.getpid()))
    # A different test runner acquires the same desktop lock.
    harness[1][str(os.getpid())] = {"pid": str(os.getpid()), "start": "11", "uid": os.getuid()}


def test_registration_is_durable_before_cleanup_and_unknown_baseline_is_preserved(
    harness, tmp_path
):
    value, work = start(harness, tmp_path)
    register(value)
    raw = (work / "native-run-ownership.json").read_bytes()
    packet = json.loads(raw)
    assert packet["sha256"] == ledger._digest(packet["body"])
    assert packet["body"]["windows"] == [
        {"window_id": "111", "owner": harness[1]["20000"], "launch_index": 0}
    ]
    assert [event["phase"] for event in packet["body"]["events"]] == [
        "launch_intent",
        "spawned",
        "registered",
    ]
    assert (
        value.finish(
            inventory({"999": {"identity_available": False}}),
            cleanup={"owned_windows_closed": True},
            settled=True,
        )["state"]
        == "resolved"
    )
    assert (
        json.loads((work / "native-run-ownership.json").read_bytes())["body"]["state"] == "resolved"
    )
    value.close()


@pytest.mark.parametrize(
    "kind", ["residual", "pid_reused", "start_reused", "unknown", "foreign_uid"]
)
def test_crash_then_specific_residual_or_identity_drift_blocks_without_cleanup(
    harness, tmp_path, kind
):
    value, _ = start(harness, tmp_path)
    register(value)
    abandon(value, harness)
    row = {"pid": "20000", "start_ticks": "20"}
    if kind == "pid_reused":
        row["pid"] = "20001"
    if kind == "start_reused":
        row["start_ticks"] = "21"
    if kind == "unknown":
        row = {"identity_available": False}
    if kind == "foreign_uid":
        harness[1]["20000"]["uid"] += 1
    next_run = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="native_run_prior_"):
        next_run.reconcile(inventory({"111": row}))
    assert len(next_run.index["pending"]) == 1
    next_run.close()


def test_crashed_owner_absence_reconciles_without_requiring_shared_provider_exit(harness, tmp_path):
    value, work = start(harness, tmp_path)
    register(value)
    old_mirror = (work / "native-run-ownership.json").read_bytes()
    abandon(value, harness)
    next_run = harness[2]()
    result = next_run.reconcile(inventory({"222": {"pid": "20000", "start_ticks": "20"}}))
    assert len(result["resolved"]) == 1 and not next_run.index["pending"]
    assert harness[1]["20000"]  # Shared provider remains alive; no process-wide ownership.
    assert (work / "native-run-ownership.json").read_bytes() == old_mirror
    assert (
        result["resolved"][0]["retained_evidence_sha256"] == hashlib.sha256(old_mirror).hexdigest()
    )
    assert result["resolved"][0]["resolution"]["complete_inventory"]["windows"] == {
        "222": {"pid": "20000", "start_ticks": "20"}
    }
    assert next_run._read_record(result["resolved"][0]["run_id"])["state"] == "resolved"
    next_run.close()


@pytest.mark.parametrize("phase", ["launch_intent", "spawned", "candidate"])
def test_crash_without_registration_is_uncertain_even_with_empty_inventory(
    harness, tmp_path, phase
):
    value, _ = start(harness, tmp_path)
    row = {"application": "editor", "registered": False, "before_xids": []}
    if phase != "launch_intent":
        row.update(launch_pid=20000, launch_start="20")
    if phase == "candidate":
        row.update(candidate_xid="111", new_visible_xids=["111"])
    value.publish_launches([row], [], phase=phase)
    abandon(value, harness)
    next_run = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="uncertain_prior_launch"):
        next_run.reconcile(inventory())
    next_run.close()


def test_active_runner_conflict_survives_lock_release(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    register(value)
    value.close()
    next_run = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="prior_runner_active"):
        next_run.reconcile(inventory())
    next_run.close()


def test_flock_is_exclusive_across_instances_and_canonical_screens(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    with pytest.raises(ledger.RunLedgerError, match="already_locked"):
        ledger.NativeRunLedger.open(
            {"DISPLAY": ":8765.1"},
            state_root=harness[0],
            identity_reader=lambda pid: harness[1][str(pid)],
        )
    value.close()
    next_run = harness[2]()
    next_run.close()


@pytest.mark.parametrize(
    "bad",
    [
        {"available": False},
        {"available": True, "windows": {}},
        inventory({str(i + 1): {} for i in range(257)}),
        inventory({"not-an-xid": {}}),
    ],
)
def test_unavailable_or_unbounded_inventory_blocks_before_begin(harness, tmp_path, bad):
    value = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="inventory_unavailable"):
        value.reconcile(bad)
    with pytest.raises(ledger.RunLedgerError, match="admission_required"):
        value.begin(tmp_path, "LHN-047")
    assert not value.index["pending"]
    value.close()


@pytest.mark.parametrize("settled,closed", [(False, False), (False, True), (True, False)])
def test_error_cancel_or_deferred_cleanup_stays_pending(harness, tmp_path, settled, closed):
    value, _ = start(harness, tmp_path)
    register(value)
    result = value.finish(inventory(), cleanup={"owned_windows_closed": closed}, settled=settled)
    assert result["state"] == "pending" and len(value.index["pending"]) == 1
    value.close()


def test_close_does_not_clear_or_resolve_pending(harness, tmp_path):
    value, work = start(harness, tmp_path)
    register(value)
    before = (work / "native-run-ownership.json").read_bytes()
    value.close()
    assert (work / "native-run-ownership.json").read_bytes() == before
    with pytest.raises(ledger.RunLedgerError, match="ledger_closed"):
        value.finish(inventory(), cleanup={"owned_windows_closed": True}, settled=True)


@pytest.mark.parametrize(
    "corruption", ["malformed", "oversize", "symlink", "foreign_mode", "orphan"]
)
def test_unsafe_registry_never_becomes_an_empty_baseline(harness, tmp_path, corruption):
    value, _ = start(harness, tmp_path)
    directory = value.directory
    value.close()
    path = directory / "index.json"
    if corruption == "malformed":
        path.write_bytes(b'{"pending":false}')
    if corruption == "oversize":
        path.write_bytes(b" " * (ledger.MAX_INDEX_BYTES + 1))
    if corruption == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    if corruption == "foreign_mode":
        path.chmod(0o644)
    if corruption == "orphan":
        (directory / "unknown.json").write_text("{}")
    with pytest.raises(ledger.RunLedgerError):
        harness[2]()


def test_corrupt_seal_does_not_clear_pending(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    register(value)
    path = value._record_path(value.current["run_id"])
    packet = json.loads(path.read_bytes())
    packet["body"]["windows"] = []
    path.write_text(json.dumps(packet))
    abandon(value, harness)
    next_run = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="seal_invalid"):
        next_run.reconcile(inventory())
    assert len(next_run.index["pending"]) == 1
    next_run.close()


def test_short_write_preserves_old_atomic_file_and_rejects(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    ledger._atomic(path, b"old", 100)
    original = ledger.os.fdopen

    class ShortFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def write(self, data):
            return self.wrapped.write(data[:1])

        def fileno(self):
            return self.wrapped.fileno()

    monkeypatch.setattr(
        ledger.os,
        "fdopen",
        lambda fd, mode, **kw: (
            ShortFile(original(fd, mode, **kw)) if mode == "wb" else original(fd, mode, **kw)
        ),
    )
    with pytest.raises(ledger.RunLedgerError, match="short_write"):
        ledger._atomic(path, b"new-body", 100)
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_post_replace_directory_fsync_error_is_indeterminate(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    ledger._atomic(path, b"old", 100)
    original = ledger.os.fsync

    def fsync(fd):
        import stat

        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory sync")
        return original(fd)

    monkeypatch.setattr(ledger.os, "fsync", fsync)
    with pytest.raises(ledger.RunLedgerError, match="publication_indeterminate"):
        ledger._atomic(path, b"new", 100)
    assert path.read_bytes() == b"new"


def test_publication_failure_is_sticky_and_cannot_be_erased_by_finally(
    harness, tmp_path, monkeypatch
):
    value, _ = start(harness, tmp_path)
    original = ledger._atomic

    def fail(path, *args):
        if path.name == "native-run-ownership.json":
            raise ledger.RunLedgerError("injected_write_failure")
        return original(path, *args)

    monkeypatch.setattr(ledger, "_atomic", fail)
    with pytest.raises(ledger.RunLedgerError):
        value.publish_launches([{"registered": False}], [], phase="launch_intent")
    monkeypatch.setattr(ledger, "_atomic", original)
    outcome = value.finish(inventory(), cleanup={"owned_windows_closed": True}, settled=True)
    assert (
        outcome["state"] == "pending"
        and outcome["code"] == "native_run_checkpoint_publication_failed"
    )
    assert value.current["launches"] == [{"registered": False}]
    value.close()


def test_normalized_migration_requires_exact_pin_and_source_bytes(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    register(value)
    body = copy.deepcopy(value.current)
    # Separate private registry imports a pinned, explicitly host-normalized
    # record. The old source journal is retained as provenance, not parsed as code.
    source = tmp_path / "legacy.json"
    source.write_bytes(b"[]")
    source.chmod(0o600)
    normalized = tmp_path / "migration.json"
    document = {
        "schema": "robit.omni.native-run-migration.v1",
        "session": value.session,
        "source_journals": [
            {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
        "records": [body],
    }
    normalized.write_bytes(ledger._json(document))
    normalized.chmod(0o600)
    value.close()
    harness[1][str(os.getpid())]["start"] = "11"
    second = ledger.NativeRunLedger.open(
        {"DISPLAY": ":8765"},
        state_root=tmp_path / "other-state",
        identity_reader=lambda pid: harness[1].get(str(pid)),
    )
    with pytest.raises(ledger.RunLedgerError, match="prior_owned_window_residual"):
        second.reconcile(
            inventory({"111": {"pid": "20000", "start_ticks": "20"}}),
            migration_refs=[
                {
                    "path": str(normalized),
                    "sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
                }
            ],
        )
    assert len(second.index["pending"]) == 1
    second.close()


@pytest.mark.parametrize("display", ["", "remote:0", ":0;echo x", "localhost:0", ":abc"])
def test_missing_or_remote_unqualified_display_rejects_without_state(harness, display):
    with pytest.raises(ledger.RunLedgerError, match="desktop_session_unavailable"):
        ledger.NativeRunLedger.open({"DISPLAY": display}, state_root=harness[0])
    assert not harness[0].exists()


def test_actual_cpu_process_crash_keeps_durable_uncertain_intent(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "crashed-run"
    root.mkdir(mode=0o700)
    state = tmp_path / "crash-state"
    script = """import os,sys
from pathlib import Path
from runtime.native_run_ledger import NativeRunLedger
v=NativeRunLedger.open({'DISPLAY':':8765'},state_root=Path(sys.argv[1]))
v.reconcile({'available':True,'coverage':'managed_client_windows_only','windows':{}})
v.begin(Path(sys.argv[2]),'LHN-047')
v.publish_launches([{'registered':False}],[],phase='launch_intent')
os._exit(7)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(state), str(root)],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 7
    value = ledger.NativeRunLedger.open({"DISPLAY": ":8765"}, state_root=state)
    with pytest.raises(ledger.RunLedgerError, match="uncertain_prior_launch"):
        value.reconcile(inventory())
    assert len(value.index["pending"]) == 1
    value.close()


def test_known_popen_failure_is_not_misreported_as_a_started_process(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    row = {"registered": False, "before_xids": [], "spawn_failed_before_process": True}
    value.publish_launches([row], [], phase="spawn_failed")
    result = value.finish(inventory(), cleanup={"owned_windows_closed": False}, settled=True)
    assert result["state"] == "resolved"
    value.close()


@pytest.mark.parametrize("kind", ["missing", "wrong_hash", "symlink", "oversize"])
def test_migration_ref_failure_never_silently_skips_known_history(harness, tmp_path, kind):
    value = harness[2]()
    path = tmp_path / "migration.json"
    if kind != "missing":
        path.write_bytes(b"{}" if kind != "oversize" else b" " * (ledger.MAX_RECORD_BYTES + 1))
        path.chmod(0o600)
    if kind == "symlink":
        other = tmp_path / "original.json"
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile(inventory(), migration_refs=[{"path": str(path), "sha256": "0" * 64}])
    assert not value.admitted and not value.index["pending"]
    value.close()


def test_registered_scope_rejects_before_existing_and_unqualified_namespace(harness, tmp_path):
    value, _ = start(harness, tmp_path)
    row = {
        "registered": True,
        "window_id": "111",
        "pid": "20000",
        "process_start": "20",
        "before_xids": ["111"],
        "new_visible_xids": ["111"],
    }
    with pytest.raises(ledger.RunLedgerError, match="registration_unproven"):
        value.publish_launches(
            [row], [OwnedWindow("editor", "111", "20000", "20")], phase="registered"
        )
    row.update(
        ownership_kind="flatpak_firefox",
        ownership_evidence={"window": {"window_id": "111", "process_id": 2, "process_start": "0"}},
    )
    with pytest.raises(ledger.RunLedgerError, match="namespace_unproven"):
        value.publish_launches(
            [row], [OwnedWindow("editor", "111", "20000", "20")], phase="registered"
        )
    value.close()


@pytest.mark.parametrize(
    "cleanup",
    [
        {
            "owned_windows_closed": False,
            "owned_windows_retained": True,
            "status": "cleanup_deferred",
        },
        {
            "owned_windows_closed": False,
            "cleanup": [{"closed": False, "identity_or_close_unavailable": True}],
        },
    ],
)
def test_explicit_unsettled_or_unknown_cleanup_does_not_resolve_from_parent_absence(
    harness, tmp_path, cleanup
):
    value, _ = start(harness, tmp_path)
    register(value)
    value.finish(
        inventory({"111": {"pid": "20000", "start_ticks": "20"}}),
        cleanup=cleanup,
        settled=cleanup.get("status") != "cleanup_deferred",
    )
    abandon(value, harness)
    next_run = harness[2]()
    with pytest.raises(ledger.RunLedgerError, match="prior_cleanup_uncertain"):
        next_run.reconcile(inventory())
    assert next_run.index["pending"]
    next_run.close()


@pytest.mark.parametrize("kind", ["conflicting", "missing", "later_reference_missing"])
def test_migration_preflights_all_old_evidence_before_any_import_or_root_write(
    harness, tmp_path, kind
):
    value, work = start(harness, tmp_path)
    register(value)
    body = copy.deepcopy(value.current)
    mirror = work / "native-run-ownership.json"
    prior = {item.name: item.read_bytes() for item in work.iterdir() if item.is_file()}
    value.close()
    source = tmp_path / "legacy-source.json"
    source.write_bytes(b"[]")
    source.chmod(0o600)
    if kind == "conflicting":
        body["case_id"] = "host-normalized-other-case"
    if kind == "missing":
        mirror.unlink()
        prior = {}
    document = {
        "schema": "robit.omni.native-run-migration.v1",
        "session": body["session"],
        "source_journals": [
            {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
        "records": [body],
    }
    path = tmp_path / "normalized.json"
    path.write_bytes(ledger._json(document))
    path.chmod(0o600)
    refs = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    if kind == "later_reference_missing":
        refs.append({"path": str(tmp_path / "missing.json"), "sha256": "0" * 64})
    second = ledger.NativeRunLedger.open(
        {"DISPLAY": ":8765"},
        state_root=tmp_path / "migration-state",
        identity_reader=lambda pid: harness[1].get(str(pid)),
    )
    before = (second.directory / "index.json").read_bytes()
    with pytest.raises(ledger.RunLedgerError):
        second.reconcile(inventory(), migration_refs=refs)
    assert (second.directory / "index.json").read_bytes() == before
    assert not second.index["pending"] and not second.index["migrations"]
    assert {item.name: item.read_bytes() for item in work.iterdir() if item.is_file()} == prior
    second.close()


def settlement_case(harness, tmp_path, monkeypatch, *, extra_pending=False):
    """Actual-shaped retained uncertain cleanup; all host observations are CPU doubles."""
    value, work = start(harness, tmp_path)
    register(value)
    failed = value.finish(
        inventory({"111": {"pid": "20000", "start_ticks": "20"}}),
        cleanup={"owned_windows_closed": False, "owned_windows_retained": True},
        settled=False,
    )
    assert failed["state"] == "pending"
    run_id = value.current["run_id"]
    abandon(value, harness)
    value = harness[2]()
    current = inventory({"999": {"identity_available": False}})
    method = tmp_path / "host-operator.py"
    method.write_text("# Explicit CPU host-verifier source stand-in; never executed.\n")
    method.chmod(0o600)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(method))
    monkeypatch.setattr(ledger, "time", SimpleNamespace(time=lambda: 1000.0))
    if extra_pending:
        other = copy.deepcopy(value._read_record(run_id))
        other["run_id"] = "f" * 32
        other_root = tmp_path / "other-run"
        other_root.mkdir(mode=0o700)
        other["run_root"] = str(other_root)
        value.index["pending"].append(other["run_id"])
        value._index()
        value._publish(other)
    return value, work, run_id, current, method


def private_ref(path, document):
    path.write_bytes(ledger._json(document))
    path.chmod(0o600)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def settlement_proof(value, work, run_id, current, method, tmp_path):
    body = value._read_record(run_id)
    raw = value._record_path(run_id).read_bytes()
    method_ref = {"path": str(method), "sha256": hashlib.sha256(method.read_bytes()).hexdigest()}
    shared = {
        "schema": "robit.omni.native-host-verification.v1",
        "run_id": run_id,
        "session_sha256": ledger._digest(value.session),
        "pending_record_sha256": hashlib.sha256(raw).hexdigest(),
        "producer": copy.deepcopy(value.runner),
        "method": method_ref,
        "observed_at_unix": 990.0,
        "passed": True,
    }
    settlement = {**shared, "kind": "downstream_settlement", "observations": {"idle": True}}
    cleanup = {
        **shared, "kind": "owned_cleanup", "observations": {
            "inventory_sha256": ledger._digest(current),
            "owned_window_ids_absent": sorted(window["window_id"] for window in body["windows"]),
        },
    }
    receipt = {
        "schema": "robit.omni.native-verified-settlement.v1",
        "run_id": run_id,
        "session": copy.deepcopy(value.session),
        "pending_body_sha256": ledger._digest(body),
        "pending_record_sha256": hashlib.sha256(raw).hexdigest(),
        "mirror_sha256": hashlib.sha256((work / "native-run-ownership.json").read_bytes()).hexdigest(),
        "scope_sha256": body["scope_sha256"],
        "launches_sha256": ledger._digest(body["launches"]),
        "inventory_sha256": ledger._digest(current),
        "producer": copy.deepcopy(value.runner),
        "method": method_ref,
        "observed_at_unix": 995.0,
        "expires_at_unix": 1100.0,
        "settlement_evidence": private_ref(tmp_path / "settlement.json", settlement),
        "cleanup_evidence": private_ref(tmp_path / "cleanup.json", cleanup),
    }
    return receipt, settlement, cleanup


def ledger_bytes(value, work):
    return ({path.name: path.read_bytes() for path in value.directory.iterdir()},
            (work / "native-run-ownership.json").read_bytes(), copy.deepcopy(value.index))


def test_verified_settlement_default_still_red_explicit_green_preserves_original(
    harness, tmp_path, monkeypatch
):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    original = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError, match="prior_cleanup_uncertain"):
        value.reconcile(current)
    assert ledger_bytes(value, work) == original
    prior = copy.deepcopy(value._read_record(run_id)["outcome"])
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    reference = private_ref(tmp_path / "receipt.json", receipt)
    resolved = value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert resolved["state"] == "resolved" and not value.index["pending"]
    assert value.admitted is False and value.current is None
    assert resolved["resolution"]["prior_outcome"] == prior
    assert resolved["resolution"]["verified_settlement"]["reference"] == reference
    assert resolved["retained_evidence_sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert (work / "native-run-ownership.json").read_bytes() == original[1]
    assert harness[1]["20000"]  # Shared provider survives; no process-wide authority.
    after = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError, match="not_pending"):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == after
    assert value.reconcile(current)["status"] == "admitted"
    value.close()
    reopened = harness[2]()
    assert reopened._read_record(run_id)["outcome"]["prior_outcome"] == prior
    reopened.close()


@pytest.mark.parametrize("field", [
    "schema", "run_id", "session", "pending_body_sha256", "pending_record_sha256", "mirror_sha256",
    "scope_sha256", "launches_sha256", "inventory_sha256", "producer", "method", "extra", "missing",
])
def test_verified_settlement_closed_bindings_reject_without_mutation(
    harness, tmp_path, monkeypatch, field
):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    if field == "session":
        receipt[field]["display"] = ":8766"
    elif field == "producer":
        receipt[field]["start"] = "12"
    elif field == "method":
        other = tmp_path / "other-method.py"
        other.write_bytes(method.read_bytes())
        other.chmod(0o600)
        receipt[field]["path"] = str(other)
    elif field == "extra":
        receipt[field] = True
    elif field == "missing":
        del receipt["settlement_evidence"]
    else:
        receipt[field] = "0" * 32 if field == "run_id" else "0" * 64
    reference = private_ref(tmp_path / "receipt.json", receipt)
    before = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == before
    value.close()


@pytest.mark.parametrize("field,bad", [
    ("observed_at_unix", True), ("observed_at_unix", 1001),
    ("observed_at_unix", -1), ("observed_at_unix", 10**400),
    ("expires_at_unix", 1000), ("expires_at_unix", 1296), ("expires_at_unix", False),
])
def test_verified_settlement_time_contract(harness, tmp_path, monkeypatch, field, bad):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    receipt[field] = bad
    reference = private_ref(tmp_path / "receipt.json", receipt)
    before = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == before
    value.close()


@pytest.mark.parametrize("kind", ["settlement", "cleanup"])
@pytest.mark.parametrize("fault", [
    "empty", "false", "extra", "missing", "kind", "run_id", "session", "record", "producer",
    "method", "future", "stale", "stacked_age", "bool", "inventory", "owned_ids",
])
def test_verified_settlement_evidence_contract_no_mutation(
    harness, tmp_path, monkeypatch, kind, fault
):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    receipt, settlement, cleanup = settlement_proof(value, work, run_id, current, method, tmp_path)
    evidence = settlement if kind == "settlement" else cleanup
    if fault == "empty":
        evidence["observations"] = {}
    elif fault == "false":
        evidence["passed"] = False
    elif fault == "extra":
        evidence["extra"] = 1
    elif fault == "missing":
        del evidence["passed"]
    elif fault in {"kind", "run_id"}:
        evidence[fault] = "wrong"
    elif fault == "session":
        evidence["session_sha256"] = "0" * 64
    elif fault == "record":
        evidence["pending_record_sha256"] = "0" * 64
    elif fault == "producer":
        evidence["producer"] = {**value.runner, "start": "12"}
    elif fault == "method":
        evidence["method"] = {**receipt["method"], "sha256": "0" * 64}
    elif fault in {"future", "stale", "stacked_age", "bool"}:
        evidence["observed_at_unix"] = {"future": 996, "stale": 699, "stacked_age": 600, "bool": False}[fault]
        if fault == "stacked_age":
            receipt.update(observed_at_unix=800, expires_at_unix=1050)
    elif fault in {"inventory", "owned_ids"}:
        # Only owned_cleanup interprets these two observation fields. Other
        # downstream details remain host audit data, never generic verification.
        evidence = cleanup
        kind = "cleanup"
        evidence["observations"]["inventory_sha256" if fault == "inventory" else "owned_window_ids_absent"] = "wrong"
    receipt[kind + "_evidence"] = private_ref(tmp_path / (kind + ".json"), evidence)
    reference = private_ref(tmp_path / "receipt.json", receipt)
    before = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == before
    value.close()


@pytest.mark.parametrize("fault", [
    "live_runner", "unknown_runner", "unregistered", "publication_failed", "residual",
    "reused_xid", "unknown_xid", "inventory_unavailable", "inventory_substituted", "current_run",
])
def test_verified_settlement_preserves_existing_admission_guards(
    harness, tmp_path, monkeypatch, fault
):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    if fault in {"unregistered", "publication_failed"}:
        body = value._read_record(run_id)
        if fault == "unregistered":
            body["launches"].append({"registered": False})
        else:
            body["outcome"]["code"] = "native_run_checkpoint_publication_failed"
        value._publish(body)
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    reference = private_ref(tmp_path / "receipt.json", receipt)
    old = value._read_record(run_id)["runner"]
    if fault == "live_runner":
        # Distinct current verifier and original runner, both positively live.
        value.runner = {"pid": "30000", "start": "30", "uid": os.getuid()}
        harness[1]["30000"] = value.runner
        harness[1][old["pid"]] = old
        receipt["producer"] = value.runner
        # Reissue evidence with the actual current verifier identity.
        receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
        reference = private_ref(tmp_path / "receipt.json", receipt)
    elif fault == "unknown_runner":
        original_reader = value.identity_reader
        value.identity_reader = lambda pid: original_reader(pid) if str(pid) == value.runner["pid"] else {}
        # The prior PID is reused by this CPU runner; distinguish it so the
        # original lifetime lookup alone is an explicit unknown test double.
        body = value._read_record(run_id)
        body["runner"] = {"pid": "30000", "start": "30", "uid": os.getuid()}
        value._publish(body)
        receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
        reference = private_ref(tmp_path / "receipt.json", receipt)
    elif fault in {"residual", "reused_xid", "unknown_xid"}:
        current["windows"]["111"] = {"pid": "20000", "start_ticks": "20"}
        if fault == "reused_xid":
            current["windows"]["111"]["start_ticks"] = "21"
        elif fault == "unknown_xid":
            current["windows"]["111"] = {"identity_available": False}
        receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
        reference = private_ref(tmp_path / "receipt.json", receipt)
    elif fault == "inventory_unavailable":
        current["available"] = False
    elif fault == "inventory_substituted":
        current["windows"]["222"] = {"identity_available": False}
    elif fault == "current_run":
        value.current = value._read_record(run_id)
    before = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == before
    value.close()


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "mode", "relative", "oversize", "wrong_hash"])
def test_verified_settlement_private_reference_safety(harness, tmp_path, monkeypatch, fault):
    value, work, run_id, current, method = settlement_case(harness, tmp_path, monkeypatch)
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    reference = private_ref(tmp_path / "receipt.json", receipt)
    path = tmp_path / "receipt.json"
    if fault == "symlink":
        path.rename(tmp_path / "real.json")
        path.symlink_to(tmp_path / "real.json")
    elif fault == "hardlink":
        os.link(path, tmp_path / "other-link.json")
    elif fault == "mode":
        path.chmod(0o644)
    elif fault == "relative":
        reference["path"] = "receipt.json"
    elif fault == "oversize":
        path.write_bytes(b"x" * 65537)
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "wrong_hash":
        reference["sha256"] = "0" * 64
    before = ledger_bytes(value, work)
    with pytest.raises(ledger.RunLedgerError):
        value.reconcile_verified_settlement(current, settlement_ref=reference)
    assert ledger_bytes(value, work) == before
    value.close()


def test_verified_settlement_resolves_only_pinned_run(harness, tmp_path, monkeypatch):
    value, work, run_id, current, method = settlement_case(
        harness, tmp_path, monkeypatch, extra_pending=True
    )
    other_raw = value._record_path("f" * 32).read_bytes()
    receipt, _, _ = settlement_proof(value, work, run_id, current, method, tmp_path)
    value.reconcile_verified_settlement(
        current, settlement_ref=private_ref(tmp_path / "receipt.json", receipt)
    )
    assert value.index["pending"] == ["f" * 32] and not value.admitted
    assert value._record_path("f" * 32).read_bytes() == other_raw
    with pytest.raises(ledger.RunLedgerError, match="prior_cleanup_uncertain"):
        value.reconcile(current)
    value.close()
