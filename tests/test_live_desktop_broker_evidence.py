from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from runtime import verify_live_desktop_matrix as matrix

OTHER = "GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0"
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
    ({"available": True, "gpu_health": {"admission_blocked": 0}}, "admission_not_clear"),
    ({"available": True, "gpu_health": {}}, "admission_not_clear"),
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
    assert set(result) == {"at", "gpu_uuid", "gpu_health", "warnings", "inference_hold", "active_leases", "processes"}
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
