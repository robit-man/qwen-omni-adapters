"""Aggregation integrity tests; these fake child reports are never live evidence."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from runtime import verify_real_desktop_suite as suite

MODEL = "selected-exact-model"


def artifact(root, case, *, model=MODEL, passed=True):
    root.mkdir(parents=True)
    report = {"schema": case.schema, "model": model, "scenario": case.scenario, "policy_required": case.policy,
              "source": "generated_tts_audio", "physical_microphone": False, "typed_goal_substituted": False,
              "authoritative_final_received": True, "owned_window_closed": True, "criteria": {"runner_check": True},
              "passed": passed, "think": False}
    if case.turns > 1:
        report["turns"] = [{"authoritative_final_received": True} for _ in range(case.turns)]
    (root / "summary.json").write_text(json.dumps(report))
    for index in range(case.turns):
        turn = root / f"turn-{index + 1}" if case.turns > 1 else root
        turn.mkdir(exist_ok=True)
        raw = b"unit-test-source-artifact"
        digest = hashlib.sha256(raw).hexdigest()
        (turn / "instruction.wav").write_bytes(raw)
        (turn / "tts-evidence.json").write_text(json.dumps({"source": "generated_tts_audio", "model": model, "wav_sha256": digest}))
        (turn / "speech-evidence.json").write_text(json.dumps({"source": "generated_tts_audio", "model": model,
            "input_wav_sha256": digest, "asr_transcript": "fixture transcript", "typed_goal_substituted": False, "think": False}))
    return report


@pytest.mark.parametrize("exit_code,passed", [(1, True), (0, False), (2, False)])
def test_nonzero_runner_or_false_json_pass_never_counts_as_success(tmp_path, exit_code, passed):
    case = suite.CASES[0]
    root = tmp_path / "case"
    artifact(root, case, passed=passed)
    result = suite.case_result(root, case, MODEL, exit_code)
    assert result["status"] == "failed"
    assert "runner_nonzero" in result["failure_codes"] or "runner_failed" in result["failure_codes"]


@pytest.mark.parametrize("field,value,code", [
    ("criteria", {}, "criteria_not_all_verified"), ("criteria", {"check": False}, "criteria_not_all_verified"),
    ("criteria", {"check": 1}, "criteria_not_all_verified"), ("model", "other-model", "wrong_model"),
    ("typed_goal_substituted", True, "wrong_speech_provenance"),
    ("authoritative_final_received", False, "authoritative_final_not_confirmed"),
    ("owned_window_closed", False, "owned_window_cleanup_not_confirmed"),
])
def test_manifest_rejects_bad_identity_provenance_criteria_and_cleanup(tmp_path, field, value, code):
    case, root = suite.CASES[0], tmp_path / "case"
    report = artifact(root, case)
    report[field] = value
    (root / "summary.json").write_text(json.dumps(report))
    result = suite.case_result(root, case, MODEL, 0)
    assert result["status"] == "failed" and code in result["failure_codes"]


def test_speech_hash_mismatch_or_missing_second_turn_cannot_be_promoted_to_pass(tmp_path):
    case, root = suite.CASES[-1], tmp_path / "firefox"
    artifact(root, case)
    (root / "turn-2" / "instruction.wav").write_bytes(b"changed-after-report")
    result = suite.case_result(root, case, MODEL, 0)
    assert result["status"] == "failed"
    assert result["speech"]["tts_verified_turns"] == 1 and result["speech"]["asr_verified_turns"] == 1
    assert "speech_evidence_not_verified" in result["failure_codes"]


def test_stdout_claim_does_not_replace_authoritative_summary(tmp_path):
    (tmp_path / "stdout.log").write_text('{"passed":true}')
    result = suite.case_result(tmp_path, suite.CASES[0], MODEL, 0)
    assert result["status"] == "failed" and "missing_or_invalid_summary" in result["failure_codes"]


def arguments(tmp_path, **overrides):
    token = tmp_path / "token.txt"
    token.write_text("private-token-sentinel")
    return SimpleNamespace(model=MODEL, endpoint="http://127.0.0.1:8920", token_file=token,
        timeout=180, case_timeout=600, case=None, output=tmp_path / "runtime-data" / "suite", keep_going=False, think=False, **overrides)


def test_default_runs_strictly_sequentially_and_stops_at_first_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(suite, "REPO", tmp_path)
    launched = []
    def launch(_args, case, root, logs):
        launched.append(case.name)
        artifact(root, case, passed=False)
        (logs / "private.stdout.log").write_text("private-ui-sentinel private-token-sentinel")
        return 1, False
    monkeypatch.setattr(suite, "launch", launch)
    args = arguments(tmp_path)
    assert suite.run(args) == 1
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert launched == [suite.CASES[0].name]
    assert manifest["cases"][0]["status"] == "failed"
    assert all(item["status"] == "unrun" and item["reason"] == "stopped_after_failure" for item in manifest["cases"][1:])
    assert manifest["selected_scope_passed"] is False and manifest["full_scope_passed"] is False
    printed = capsys.readouterr().out
    assert "private-ui-sentinel" not in printed and "private-token-sentinel" not in printed


def test_explicit_case_resume_keeps_full_suite_unrun_and_preserves_old_run(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "REPO", tmp_path)
    launched = []
    def launch(_args, case, root, _logs):
        launched.append(case.name)
        artifact(root, case)
        return 0, False
    monkeypatch.setattr(suite, "launch", launch)
    args = arguments(tmp_path)
    args.case = ["calculator-arithmetic"]
    assert suite.run(args) == 0
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert launched == ["calculator-arithmetic"]
    assert manifest["selected_scope_passed"] is True and manifest["full_scope_passed"] is False
    assert sum(item["status"] == "unrun" for item in manifest["cases"]) == len(suite.CASES) - 1
    original = (args.output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        suite.run(args)
    assert (args.output / "manifest.json").read_bytes() == original


def test_timeout_is_failed_even_if_runner_wrote_passing_json(tmp_path):
    case, root = suite.CASES[0], tmp_path / "case"
    artifact(root, case)
    result = suite.case_result(root, case, MODEL, 0, timed_out=True)
    assert result["status"] == "failed" and "suite_case_timeout" in result["failure_codes"]


def test_keep_going_still_stops_when_window_cleanup_is_unconfirmed(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "REPO", tmp_path)
    launched = []
    def launch(_args, case, root, _logs):
        launched.append(case.name)
        report = artifact(root, case)
        report["owned_window_closed"] = False
        (root / "summary.json").write_text(json.dumps(report))
        return 0, False
    monkeypatch.setattr(suite, "launch", launch)
    args = arguments(tmp_path)
    args.keep_going = True
    assert suite.run(args) == 1
    assert len(launched) == 1


def test_child_command_forwards_exact_selected_identity_and_existing_case_flags(tmp_path):
    args = arguments(tmp_path)
    command = suite.child_command(args, suite.CASES[7], tmp_path / "case")
    assert command[:3] == [suite.sys.executable, "-m", "runtime.verify_real_calculator"]
    for flag, value in [("--model", MODEL), ("--endpoint", args.endpoint), ("--token-file", str(args.token_file.resolve()))]:
        assert command[command.index(flag) + 1] == value
    assert "--require-policy" in command
    assert "private-token-sentinel" not in command
    assert "--think" not in command
    args.think = True
    assert "--think" in suite.child_command(args, suite.CASES[7], tmp_path / "case")


def test_suite_think_must_match_runner_and_actual_speech_evidence(tmp_path):
    case, root = suite.CASES[0], tmp_path / "case"
    report = artifact(root, case)
    mismatch = suite.case_result(root, case, MODEL, 0, think=True)
    assert mismatch["status"] == "failed" and "wrong_think" in mismatch["failure_codes"]
    report["think"] = True
    (root / "summary.json").write_text(json.dumps(report))
    assert suite.case_result(root, case, MODEL, 0, think=True)["speech"]["asr"] == "not_confirmed"
    heard = suite.read_json(root / "speech-evidence.json")
    heard["think"] = True
    (root / "speech-evidence.json").write_text(json.dumps(heard))
    assert suite.case_result(root, case, MODEL, 0, think=True)["status"] == "passed"


def test_capability_fragment_rejected_before_any_child_or_artifact_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "REPO", tmp_path)
    args = arguments(tmp_path)
    args.endpoint += "/#private-capability"
    with pytest.raises(ValueError, match="fragment"):
        suite.run(args)
    assert not args.output.exists()
