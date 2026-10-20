"""Verified settlement of an unregistered Flatpak launch from fresh host attribution (#10)."""
import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest
from test_native_run_ledger import abandon, harness, inventory, private_ref, start  # noqa: F401

from runtime import native_run_ledger as ledger
from runtime import native_run_settlement as settlement

LAUNCH = {"pid": 30000, "start": "30"}


def flatpak_case(harness, tmp_path, monkeypatch, *, kind="flatpak_chromium"):  # noqa: F811
    value, work = start(harness, tmp_path)
    launch = {"application": "browser", "ownership_kind": kind, "registered": False}
    value.publish_launches([launch], [], phase="launch_intent")
    launch["last_ownership_receipt"] = {"ok": False, "code": "flatpak_x11_protocol_error",
                                        "launch": dict(LAUNCH)}
    value.publish_launches([launch], [], phase="namespace_final")
    failed = value.finish(inventory({"999": {"identity_available": False}}),
                          cleanup={"owned_windows_closed": False}, settled=False)
    assert failed["state"] == "pending"
    run_id = value.current["run_id"]
    abandon(value, harness)
    value = harness[2]()
    method = tmp_path / "host-verifier.py"
    method.write_text("# CPU stand-in for runtime/native_run_settlement.py\n")
    method.chmod(0o600)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(method))
    monkeypatch.setattr(ledger, "time", SimpleNamespace(time=lambda: 1000.0))
    return value, work, run_id, inventory({"999": {"identity_available": False}}), method


def absent(**changes):
    row = {"index": 0, "ownership_kind": "flatpak_chromium", "application_id": "org.chromium.Chromium",
           "launch": {"pid": "30000", "start": "30", "present": False}, "live_instances": [
               {"instance": "1", "pid": "40000", "start": "40", "sandbox": True, "display_sockets": [],
                "attribution": "no_display_socket"},
               {"instance": "2", "pid": "40001", "start": "41", "sandbox": False, "display_sockets": ["x11"],
                "attribution": "separate_main_instance"}]}
    row.update(changes)
    return row


def receipt(value, work, run_id, current, method, tmp_path, launches):
    body = value._read_record(run_id)
    raw = value._record_path(run_id).read_bytes()
    method_ref = {"path": str(method), "sha256": hashlib.sha256(method.read_bytes()).hexdigest()}
    shared = {"schema": "robit.omni.native-host-verification.v1", "run_id": run_id,
              "session_sha256": ledger._digest(value.session), "pending_record_sha256": hashlib.sha256(raw).hexdigest(),
              "producer": copy.deepcopy(value.runner), "method": method_ref, "observed_at_unix": 990.0, "passed": True}
    observations = {"inventory_sha256": ledger._digest(current), "owned_window_ids_absent": []}
    if launches is not None:
        observations["unregistered_launches_absent"] = launches
    document = {"schema": "robit.omni.native-verified-settlement.v1", "run_id": run_id,
        "session": copy.deepcopy(value.session), "pending_body_sha256": ledger._digest(body),
        "pending_record_sha256": hashlib.sha256(raw).hexdigest(),
        "mirror_sha256": hashlib.sha256((work / "native-run-ownership.json").read_bytes()).hexdigest(),
        "scope_sha256": body["scope_sha256"], "launches_sha256": ledger._digest(body["launches"]),
        "inventory_sha256": ledger._digest(current), "producer": copy.deepcopy(value.runner),
        "method": method_ref, "observed_at_unix": 995.0, "expires_at_unix": 1100.0,
        "settlement_evidence": private_ref(tmp_path / "settlement.json", {
            **shared, "kind": "downstream_settlement", "observations": {"prior_runner_present": False}}),
        "cleanup_evidence": private_ref(tmp_path / "cleanup.json", {
            **shared, "kind": "owned_cleanup", "observations": observations})}
    return private_ref(tmp_path / "receipt.json", document)


def test_unregistered_flatpak_launch_settles_only_through_verified_attribution(harness, tmp_path, monkeypatch):  # noqa: F811
    value, work, run_id, current, method = flatpak_case(harness, tmp_path, monkeypatch)
    original = (work / "native-run-ownership.json").read_bytes()
    with pytest.raises(ledger.RunLedgerError, match="prior_cleanup_uncertain"):
        value.reconcile(current)  # Ordinary admission keeps its sticky guard.
    resolved = value.reconcile_verified_settlement(
        current, settlement_ref=receipt(value, work, run_id, current, method, tmp_path, [absent()]))
    assert resolved["state"] == "resolved" and not value.index["pending"]
    assert resolved["resolution"]["prior_outcome"]["cleanup_uncertain"] is True
    assert (work / "native-run-ownership.json").read_bytes() == original  # Original attempt intact.
    assert value.reconcile(current)["status"] == "admitted"
    value.close()


@pytest.mark.parametrize("fault", ["missing", "launch_present", "wrong_pid", "wrong_kind", "window_sandbox",
                                   "same_identity_main", "unknown_attribution", "duplicate", "extra_field"])
def test_unproven_launch_absence_keeps_the_run_pending(harness, tmp_path, monkeypatch, fault):  # noqa: F811
    value, work, run_id, current, method = flatpak_case(harness, tmp_path, monkeypatch)
    launches = [absent()]
    if fault == "missing":
        launches = None
    elif fault == "launch_present":
        harness[1]["30000"] = {"pid": "30000", "start": "30", "uid": 0}
    elif fault == "wrong_pid":
        launches = [absent(launch={"pid": "30001", "start": "30", "present": False})]
    elif fault == "wrong_kind":
        launches = [absent(application_id="org.mozilla.firefox")]
    elif fault == "window_sandbox":
        launches[0]["live_instances"][0].update(display_sockets=["x11"], attribution="separate_main_instance")
    elif fault == "same_identity_main":
        launches[0]["live_instances"][1].update(pid="30000", start="30")
    elif fault == "unknown_attribution":
        launches[0]["live_instances"][1]["attribution"] = "probably_unrelated"
    elif fault == "duplicate":
        launches = [absent(), absent()]
    else:
        launches = [absent(note="x")]
    with pytest.raises(ledger.RunLedgerError, match="uncertain_prior_launch"):
        value.reconcile_verified_settlement(
            current, settlement_ref=receipt(value, work, run_id, current, method, tmp_path, launches))
    assert value.index["pending"] == [run_id]
    value.close()


def test_native_unregistered_launches_are_never_cleared_by_flatpak_attribution(harness, tmp_path, monkeypatch):  # noqa: F811
    value, work, run_id, current, method = flatpak_case(harness, tmp_path, monkeypatch, kind="native_editor")
    with pytest.raises(ledger.RunLedgerError, match="uncertain_prior_launch"):
        value.reconcile_verified_settlement(
            current, settlement_ref=receipt(value, work, run_id, current, method, tmp_path, [absent()]))
    value.close()


def instances(*rows):
    return lambda _application: [dict(row) for row in rows]


def launch_record(kind="flatpak_chromium"):
    return {"ownership_kind": kind, "registered": False, "last_ownership_receipt": {"launch": dict(LAUNCH)}}


def test_verifier_attributes_live_instances_or_blocks():
    gone = lambda _pid: None  # noqa: E731
    proof = settlement.launch_absence(0, launch_record(), identity_reader=gone, instances=instances(
        {"instance": "1", "pid": "40000", "start": "40", "sandbox": True, "display_sockets": []},
        {"instance": "2", "pid": "40001", "start": "41", "sandbox": False, "display_sockets": ["wayland", "x11"]}))
    assert [row["attribution"] for row in proof["live_instances"]] == ["no_display_socket", "separate_main_instance"]
    assert proof["launch"] == {"pid": "30000", "start": "30", "present": False}
    blocked = {
        "unregistered_launch_process_present": dict(identity_reader=lambda pid: {"pid": pid, "start": "30", "uid": 1},
                                                    instances=instances()),
        "window_capable_instance_unattributed": dict(identity_reader=gone, instances=instances(
            {"instance": "3", "pid": "40002", "start": "42", "sandbox": True, "display_sockets": ["x11"]})),
    }
    for code, kwargs in blocked.items():
        with pytest.raises(settlement.SettlementBlocked, match=code):
            settlement.launch_absence(0, launch_record(), **kwargs)
    with pytest.raises(settlement.SettlementBlocked, match="kind_unsupported"):
        settlement.launch_absence(0, launch_record("native_editor"), identity_reader=gone, instances=instances())
    with pytest.raises(settlement.SettlementBlocked, match="identity_unrecorded"):
        settlement.launch_absence(0, {"ownership_kind": "flatpak_chromium", "registered": False},
                                  identity_reader=gone, instances=instances())


def test_flatpak_instance_metadata_drives_display_capability(tmp_path, monkeypatch):
    runtime = tmp_path / "run"
    for instance, extra in (("11", "sandbox=true\n[Context]\nshared=network;\n"),
                            ("12", "[Context]\nsockets=x11;wayland;pulseaudio;\n")):
        (runtime / ".flatpak" / instance).mkdir(parents=True)
        (runtime / ".flatpak" / instance / "info").write_text(
            f"[Application]\nname=org.chromium.Chromium\n[Instance]\ninstance-id={instance}\n{extra}")
    listing = "11\t501\torg.chromium.Chromium\n12\t502\torg.chromium.Chromium\n13\t503\torg.other.App\n"
    monkeypatch.setattr(settlement.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing))
    monkeypatch.setattr(settlement, "host_identity", lambda pid: {"pid": pid, "start": "9" + pid, "uid": 1})
    rows = settlement.flatpak_instances("org.chromium.Chromium", runtime_dir=str(runtime))
    assert rows == [{"instance": "11", "pid": "501", "start": "9501", "sandbox": True, "display_sockets": []},
                    {"instance": "12", "pid": "502", "start": "9502", "sandbox": False,
                     "display_sockets": ["wayland", "x11"]}]


def native_case(harness, tmp_path, monkeypatch):  # noqa: F811
    value, work = start(harness, tmp_path)
    launch = {"application": "browser", "window_class": "[Ff]irefox", "before_xids": [], "registered": False}
    value.publish_launches([launch], [], phase="launch_intent")
    launch.update(launch_pid=30000, launch_start="30", new_visible_xids=["222"], candidate_xid="222")
    value.publish_launches([launch], [], phase="spawned")
    assert value.finish(inventory({"222": {"identity_available": False}}),
                        cleanup={"owned_windows_closed": False}, settled=False)["state"] == "pending"
    run_id = value.current["run_id"]
    abandon(value, harness)
    value = harness[2]()
    method = tmp_path / "host-verifier.py"
    method.write_text("# CPU stand-in for runtime/native_run_settlement.py\n")
    method.chmod(0o600)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(method))
    monkeypatch.setattr(ledger, "time", SimpleNamespace(time=lambda: 1000.0))
    return value, work, run_id, method


def native_proof(**changes):
    return {"index": 0, "ownership_kind": "native_session",
            "launch": {"pid": "30000", "start": "30", "present": False},
            "session_processes": 0, "xids_absent": ["222"], **changes}


@pytest.mark.parametrize("fault", [None, "window_present", "launch_present", "session_left", "xids_incomplete",
                                   "wrong_start"])
def test_unregistered_native_launch_settles_only_when_session_and_windows_are_gone(
        harness, tmp_path, monkeypatch, fault):  # noqa: F811
    value, work, run_id, method = native_case(harness, tmp_path, monkeypatch)
    current = inventory({"999": {"identity_available": False}})
    proof = native_proof()
    if fault == "window_present":
        current = inventory({"222": {"identity_available": False}})
    elif fault == "launch_present":
        harness[1]["30000"] = {"pid": "30000", "start": "30", "uid": 0}
    elif fault == "session_left":
        proof = native_proof(session_processes=1)
    elif fault == "xids_incomplete":
        proof = native_proof(xids_absent=[])
    elif fault == "wrong_start":
        proof = native_proof(launch={"pid": "30000", "start": "31", "present": False})
    reference = receipt(value, work, run_id, current, method, tmp_path, [proof])
    if fault is None:
        assert value.reconcile_verified_settlement(current, settlement_ref=reference)["state"] == "resolved"
        assert value.reconcile(current)["status"] == "admitted"
    else:
        with pytest.raises(ledger.RunLedgerError, match="uncertain_prior_launch|owned_window_residual"):
            value.reconcile_verified_settlement(current, settlement_ref=reference)
        assert value.index["pending"] == [run_id]
    value.close()


def test_native_verifier_requires_empty_session_and_absent_windows():
    launch = {"launch_pid": 30000, "launch_start": "30", "new_visible_xids": ["222"], "candidate_xid": "222"}
    gone, empty = (lambda _pid: None), (lambda _sid: 0)
    assert settlement.native_launch_absence(0, launch, inventory({}), identity_reader=gone, sessions=empty) == \
        native_proof()
    for code, kwargs, windows in (
        ("process_present", dict(identity_reader=lambda pid: {"pid": pid, "start": "30", "uid": 1}, sessions=empty), {}),
        ("session_present", dict(identity_reader=gone, sessions=lambda _sid: 3), {}),
        ("window_present", dict(identity_reader=gone, sessions=empty), {"222": {"identity_available": False}}),
    ):
        with pytest.raises(settlement.SettlementBlocked, match=code):
            settlement.native_launch_absence(0, launch, inventory(windows), **kwargs)
