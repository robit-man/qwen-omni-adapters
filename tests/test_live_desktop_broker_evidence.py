from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from runtime import verify_live_desktop_matrix as matrix

OTHER = "GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0"
THIRD = "GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a"
SECRET = "SECRET_CANARY_TOKEN_OWNER_PURPOSE_ENV_ERROR"


def discovery():
    return {"available": True, "gpu_health": {"admission_blocked": False, "missing_selected_gpu_ids": []},
            "unregistered_gpu_quarantine": [], "selected_gpu_ids": [matrix.GPU, OTHER],
            "gpus": [{"uuid": matrix.GPU}, {"uuid": OTHER}], "warnings": [], "inference_hold": {"active": False},
            "active_leases": [{"owner": SECRET, "token": SECRET, "justification": SECRET}]}


def mock_discover(monkeypatch, value):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv == ["docker", "gpu", "discover"] and kwargs["timeout"] == 15
        return SimpleNamespace(stdout=json.dumps(value))
    monkeypatch.setattr(matrix.subprocess, "run", run)
    monkeypatch.setattr(matrix, "process_is_gpu_resident", lambda *_a: pytest.fail("No residency after blocked health"))
    return calls


@pytest.mark.parametrize("guard,code", [
    ("available", "discovery_unavailable"), ("admission", "admission_not_clear"),
    ("missing", "missing_selected_gpu_ids"), ("quarantine", "unregistered_gpu_quarantine")])
def test_actual_guard_retains_exact_same_sanitized_snapshot_without_retry(monkeypatch, guard, code):
    value = discovery()
    if guard == "available":
        value["available"] = False
    elif guard == "admission":
        value["gpu_health"]["admission_blocked"] = True
    elif guard == "missing":
        value["gpu_health"]["missing_selected_gpu_ids"] = [OTHER]
    else:
        value["unregistered_gpu_quarantine"] = [{"uuid": OTHER, "reason": SECRET}]
    value["gpu_health"]["error"] = SECRET
    calls = mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    error = caught.value
    assert len(calls) == 1 and error.code == code and error.evidence["blocker_codes"] == [code]
    assert SECRET not in str(error) and SECRET not in json.dumps(matrix.gpu_acceptance_failure_evidence(error))
    assert error.evidence["selected_gpu_ids"] == [matrix.GPU, OTHER]
    assert error.evidence["missing_selected_gpu_ids"] == ([OTHER] if guard == "missing" else [])
    assert error.evidence["unregistered_gpu_quarantine_present"] is (guard == "quarantine")
    assert error.evidence["unregistered_gpu_quarantine_count"] == (1 if guard == "quarantine" else 0)
    captured = error.evidence
    value["available"] = not value["available"]
    returned = error.evidence
    returned["token"] = SECRET
    assert error.evidence == captured


def test_all_actual_blockers_retained_together_and_output_is_bounded(monkeypatch):
    value = discovery()
    value.update(available=False, unregistered_gpu_quarantine=[{"secret": SECRET}] * 100)
    value["gpu_health"].update(admission_blocked=True, missing_selected_gpu_ids=[OTHER] * 100, error=SECRET)
    value["selected_gpu_ids"] = [matrix.GPU] * 100 + [SECRET]
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    e = caught.value.evidence
    assert caught.value.code == "multiple_admission_blockers"
    assert e["blocker_codes"] == ["discovery_unavailable", "admission_not_clear", "missing_selected_gpu_ids", "unregistered_gpu_quarantine"]
    assert e["missing_selected_gpu_ids_count"] == e["unregistered_gpu_quarantine_count"] == 100
    assert len(e["missing_selected_gpu_ids"]) == len(e["selected_gpu_ids"]) == 16
    assert len(str(caught.value)) < 4096 and SECRET not in str(caught.value)


@pytest.mark.parametrize("value,code", [
    ([], "discovery_malformed"), ({"gpu_health": SECRET}, "gpu_health_malformed"),
    ({"available": True, "gpu_health": {"admission_blocked": 0}, "unregistered_gpu_quarantine": []}, "admission_not_clear"),
    ({"available": True, "gpu_health": {}, "unregistered_gpu_quarantine": []}, "admission_not_clear"),
    ({"available": False, "gpu_health": {"admission_blocked": False}, "unregistered_gpu_quarantine": SECRET}, "multiple_admission_blockers")])
def test_malformed_or_unproven_health_still_fails_without_raw_values(monkeypatch, value, code):
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == code and SECRET not in str(caught.value)
    if isinstance(value, dict) and isinstance(value.get("gpu_health"), dict):
        assert caught.value.evidence["admission_confirmed_clear"] is (value["gpu_health"].get("admission_blocked") is False)


def test_gpu_selection_guard_unchanged_with_typed_evidence(monkeypatch):
    value = discovery()
    value["selected_gpu_ids"] = [OTHER]
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == "gpu_scope_not_selected"


def test_malformed_json_failure_never_echoes_unparsed_response(monkeypatch):
    monkeypatch.setattr(matrix.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout=SECRET))
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == "discovery_malformed" and SECRET not in str(caught.value)


def allow_residency(monkeypatch, *, visibility=None, outside=False):
    original = matrix.Path.read_bytes
    def environment(path):
        if str(path) == "/proc/42/environ":
            return b"UNRELATED_SECRET=" + SECRET.encode() + b"\0CUDA_VISIBLE_DEVICES=" + (visibility or matrix.GPU).encode() + b"\0"
        return original(path)
    monkeypatch.setattr(matrix.Path, "read_bytes", environment)
    monkeypatch.setattr(matrix, "process_is_gpu_resident", lambda pid, gpu: gpu == matrix.GPU or outside)


@pytest.mark.parametrize("quarantine", [[OTHER], [THIRD], [OTHER, THIRD]])
def test_actual_outside_scope_quarantine_keeps_exact_gpu1_admission(monkeypatch, quarantine):
    value = discovery()
    value["selected_gpu_ids"].append(THIRD)
    value["gpus"].append({"uuid": THIRD})
    value["unregistered_gpu_quarantine"] = quarantine
    before = copy.deepcopy(value)
    calls = mock_discover(monkeypatch, value)
    allow_residency(monkeypatch)
    result = matrix.broker_evidence([42])
    assert len(calls) == 1 and value == before
    assert result["gpu_uuid"] == matrix.GPU
    assert result["unregistered_gpu_quarantine"] == sorted(quarantine)
    assert result["processes"] == [{"pid": 42, "cuda_visible_devices": matrix.GPU,
                                   "resident_on_gpu1": True, "other_selected_residency": []}]


@pytest.mark.parametrize("fault", ["visibility", "outside_residency"])
def test_residency_and_exact_visibility_checks_remain_fail_closed(monkeypatch, fault):
    mock_discover(monkeypatch, discovery())
    allow_residency(monkeypatch, visibility=SECRET if fault == "visibility" else None, outside=fault == "outside_residency")
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == "gpu_scope_residency_unproven" and SECRET not in str(caught.value)


def test_success_shape_and_single_discovery_are_unchanged(monkeypatch):
    value = discovery()
    value["active_leases"] = [{"owner": "public-owner"}]
    original = copy.deepcopy(value)
    calls = mock_discover(monkeypatch, value)
    allow_residency(monkeypatch)
    result = matrix.broker_evidence([42])
    assert len(calls) == 1 and value == original
    assert set(result) == {"at", "gpu_uuid", "gpu_health", "unregistered_gpu_quarantine", "warnings", "inference_hold", "active_leases", "processes"}
    assert result["active_leases"] == value["active_leases"] and result["gpu_health"] == value["gpu_health"]
    assert result["processes"] == [{"pid": 42, "cuda_visible_devices": matrix.GPU,
                                   "resident_on_gpu1": True, "other_selected_residency": []}]


def test_watchdog_extractor_rejects_generic_exception_attributes_and_message():
    error = RuntimeError(SECRET)
    error.code, error.evidence = SECRET, {"token": SECRET}
    assert matrix.gpu_acceptance_failure_evidence(error) == {}
    trusted = matrix.GpuAcceptanceBlocked(["discovery_unavailable"], {"available": False, "secret": SECRET})
    assert json.loads(json.dumps(matrix.gpu_acceptance_failure_evidence(trusted))) == {
        "code": "discovery_unavailable", "evidence": trusted.evidence}
    assert SECRET not in str(trusted)


@pytest.mark.parametrize("quarantine", [[matrix.GPU], [OTHER, matrix.GPU], [matrix.GPU, THIRD]])
def test_owned_or_mixed_quarantine_still_blocks_before_residency(monkeypatch, quarantine):
    value = discovery()
    value["selected_gpu_ids"].append(THIRD)
    value["unregistered_gpu_quarantine"] = quarantine
    calls = mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    error = caught.value
    assert len(calls) == 1 and error.code == "unregistered_gpu_quarantine"
    assert error.evidence["unregistered_gpu_quarantine_ids"] == quarantine
    assert error.evidence["unregistered_gpu_quarantine_scope_verified"] is True
    assert error.evidence["leased_gpu_quarantined"] is True


@pytest.mark.parametrize("quarantine", [None, False, "", SECRET, {}, {"uuid": OTHER},
    [SECRET], [True], [42], [{"uuid": OTHER, "reason": SECRET}], [OTHER, OTHER],
    ["GPU-00000000-0000-0000-0000-000000000001"], [matrix.GPU.upper()],
    [matrix.GPU.replace("ee", "eE", 1)], [OTHER.upper()], [OTHER] * 17,
    [OTHER] * 16 + [matrix.GPU]])
def test_unknown_malformed_or_truncated_quarantine_never_admits(monkeypatch, quarantine):
    value = discovery()
    value["unregistered_gpu_quarantine"] = quarantine
    calls = mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    error = caught.value
    assert len(calls) == 1 and error.code == "unregistered_gpu_quarantine"
    assert error.evidence["unregistered_gpu_quarantine_scope_verified"] is False
    assert error.evidence["leased_gpu_quarantined"] is None
    assert error.evidence["unregistered_gpu_quarantine_ids_truncated"] is (isinstance(quarantine, list) and len(quarantine) > 16)
    assert len(error.evidence["unregistered_gpu_quarantine_ids"]) <= 16
    assert SECRET not in str(error) and SECRET not in json.dumps(matrix.gpu_acceptance_failure_evidence(error))


def test_missing_quarantine_scope_fails_closed(monkeypatch):
    value = discovery()
    del value["unregistered_gpu_quarantine"]
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == "unregistered_gpu_quarantine"
    assert caught.value.evidence["unregistered_gpu_quarantine_scope_verified"] is False


@pytest.mark.parametrize("selected", [None, SECRET, [matrix.GPU, SECRET], [matrix.GPU, OTHER, OTHER],
                                      [matrix.GPU.upper(), OTHER], [matrix.GPU, matrix.GPU.upper(), OTHER],
                                      [matrix.GPU.replace("ee", "eE", 1), OTHER],
                                      [matrix.GPU, OTHER] * 9])
def test_outside_quarantine_requires_complete_authoritative_inventory(monkeypatch, selected):
    value = discovery()
    value.update(selected_gpu_ids=selected, unregistered_gpu_quarantine=[OTHER])
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    assert caught.value.code == "unregistered_gpu_quarantine"
    assert caught.value.evidence["unregistered_gpu_quarantine_scope_verified"] is False
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("guard", ["available", "admission", "missing", "visibility", "outside_residency"])
def test_outside_quarantine_does_not_bypass_other_admission_guards(monkeypatch, guard):
    value = discovery()
    value["unregistered_gpu_quarantine"] = [OTHER]
    if guard == "available":
        value["available"] = False
    elif guard == "admission":
        value["gpu_health"]["admission_blocked"] = True
    elif guard == "missing":
        value["gpu_health"]["missing_selected_gpu_ids"] = [OTHER]
    mock_discover(monkeypatch, value)
    if guard in {"visibility", "outside_residency"}:
        allow_residency(monkeypatch, visibility=SECRET if guard == "visibility" else None,
                        outside=guard == "outside_residency")
    with pytest.raises(matrix.GpuAcceptanceBlocked):
        matrix.broker_evidence([42])


def test_trusted_serializer_retains_scoped_evidence_without_secrets(monkeypatch):
    value = discovery()
    value.update(unregistered_gpu_quarantine=[matrix.GPU, OTHER], error=SECRET)
    mock_discover(monkeypatch, value)
    with pytest.raises(matrix.GpuAcceptanceBlocked) as caught:
        matrix.broker_evidence([42])
    diagnostic = matrix.gpu_acceptance_failure_evidence(caught.value)
    assert diagnostic["evidence"]["unregistered_gpu_quarantine_ids"] == [matrix.GPU, OTHER]
    assert diagnostic["evidence"]["leased_gpu_quarantined"] is True
    assert SECRET not in json.dumps(diagnostic)
