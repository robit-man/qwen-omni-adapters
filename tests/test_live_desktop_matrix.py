"""Evidence guards only; these source tests are not live desktop results."""

import hashlib
import json
import wave

from runtime import verify_live_desktop_matrix as matrix


def write(path, value):
    path.write_text(json.dumps(value))


def test_campaign_keeps_failed_attempt_when_a_fresh_repeat_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "digest", lambda _path: "plan-hash")
    plan = {"models": [{"tag": "model"}], "tests": [{"id": "CU-001", "title": "Select"}]}
    campaign = matrix.open_campaign(tmp_path, plan)
    case = plan["tests"][0]
    matrix.update_campaign(tmp_path, campaign, case, "model", {"status": "failed", "artifact_directory": "first"})
    matrix.update_campaign(tmp_path, campaign, case, "model", {"status": "passed", "artifact_directory": "second"})
    retained = json.loads((tmp_path / "campaign.json").read_text())["cases"][0]
    assert retained["status"] == "passed"
    assert [attempt["status"] for attempt in retained["attempts"]] == ["failed", "passed"]
    assert retained["attempts"][0]["artifact_directory"] == "first"


def test_capture_boolean_alone_does_not_establish_physical_mic_evidence(tmp_path):
    assert matrix.capture_confirmed(tmp_path, {"input_source": "physical_capture"},
                                    {"capture_provenance_verified": True}) is False


def test_physical_mic_capture_requires_bound_wav_and_vad_trace(tmp_path):
    with wave.open(str(tmp_path / "capture.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 320)
    (tmp_path / "vad-events.jsonl").write_text('{"type":"captured_frame"}\n')
    write(tmp_path / "capture-evidence.json", {
        "source": "physical_microphone", "device": "case-owned-test-source",
        "wav_sha256": matrix.digest(tmp_path / "capture.wav"),
        "vad_trace_sha256": matrix.digest(tmp_path / "vad-events.jsonl"),
    })
    case, report = {"input_source": "physical_capture"}, {"capture_provenance_verified": True}
    assert matrix.capture_confirmed(tmp_path, case, report) is True
    (tmp_path / "vad-events.jsonl").write_text('{"type":"different_capture"}\n')
    assert matrix.capture_confirmed(tmp_path, case, report) is False


def test_loopback_cannot_be_labeled_physical_microphone(tmp_path):
    write(tmp_path / "capture-evidence.json", {"source": "pulse_capture", "device": "monitor"})
    assert matrix.capture_confirmed(tmp_path, {"input_source": "physical_capture"},
                                    {"capture_provenance_verified": True}) is False


def successful_report():
    return {"schema": "robit.omni.extended-desktop.v1", "case_id": "CU-011", "model": "exact",
            "think": False, "passed": True, "criteria": {"independent_selection": True},
            "source": "generated_tts_audio", "physical_microphone": False,
            "typed_goal_substituted": False, "authoritative_final_received": True, "owned_window_closed": True}


def speech_files(root):
    with wave.open(str(root / "instruction.wav"), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 480)
    raw = (root / "instruction.wav").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    write(root / "tts-evidence.json", {"source": "generated_tts_audio", "model": "exact", "wav_sha256": sha})
    write(root / "speech-evidence.json", {"source": "generated_tts_audio", "model": "exact", "think": False,
        "input_wav_sha256": sha, "typed_goal_substituted": False, "asr_transcript": "Select alpha"})


def test_reported_success_cannot_replace_missing_speech_provenance(tmp_path):
    write(tmp_path / "summary.json", successful_report())
    result = matrix.extended_result(tmp_path, {"id": "CU-011", "input_source": "omni_tts_to_asr"}, "exact", False, 0)
    assert result["status"] == "failed"
    assert "genuine_speech_evidence_unconfirmed" in result["failure_codes"]


def test_exact_model_and_native_boolean_are_required_even_if_runner_claims_pass(tmp_path):
    speech_files(tmp_path)
    report = successful_report()
    report.update(model="another-model", think=0)
    write(tmp_path / "summary.json", report)
    result = matrix.extended_result(tmp_path, {"id": "CU-011", "input_source": "omni_tts_to_asr"}, "exact", False, 0)
    assert result["status"] == "failed"
    assert "wrong_model_or_think" in result["failure_codes"]


def test_numeric_truthy_criteria_do_not_count_as_verified(tmp_path):
    speech_files(tmp_path)
    report = successful_report()
    report["criteria"] = {"independent_selection": 1}
    write(tmp_path / "summary.json", report)
    result = matrix.extended_result(tmp_path, {"id": "CU-011", "input_source": "omni_tts_to_asr"}, "exact", False, 0)
    assert result["status"] == "failed"
    assert "criteria_not_all_verified" in result["failure_codes"]


def test_changing_wav_after_asr_cannot_leave_a_pass(tmp_path):
    speech_files(tmp_path)
    write(tmp_path / "summary.json", successful_report())
    (tmp_path / "instruction.wav").write_bytes(b"changed audio")
    result = matrix.extended_result(tmp_path, {"id": "CU-011", "input_source": "omni_tts_to_asr"}, "exact", False, 0)
    assert result["status"] == "failed"
    assert "genuine_speech_evidence_unconfirmed" in result["failure_codes"]


def test_blocked_capability_remains_blocked_even_when_other_booleans_are_true(tmp_path):
    speech_files(tmp_path)
    report = successful_report()
    report.update(status="blocked_capability", passed=False)
    write(tmp_path / "summary.json", report)
    result = matrix.extended_result(tmp_path, {"id": "CU-011", "input_source": "omni_tts_to_asr"}, "exact", False, 2)
    assert result["status"] == "blocked_capability"


def test_after_test_note_names_failed_oracle_and_preserves_revision(tmp_path):
    case = {"id": "CU-011", "title": "Single click", "independent_success_check": "Only alpha is selected"}
    matrix.write_note(tmp_path, case, "exact", {"status": "failed", "source_revision": "abc123",
        "criteria": {"selection": False, "unchanged_file": True}, "failure_codes": ["criteria_not_all_verified"],
        "owned_window_closed": True})
    note = json.loads((tmp_path / "after-test-note.json").read_text())
    assert note["failed_criteria"] == ["selection"]
    assert note["source_revision"] == "abc123"
    assert note["independent_success_check"] == "Only alpha is selected"


def test_unrun_model_case_prevents_full_scope_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "digest", lambda _path: "plan-hash")
    plan = {"models": [{"tag": "qwen"}, {"tag": "ornith"}], "tests": [{"id": "CU-001", "title": "Select"}]}
    campaign = matrix.open_campaign(tmp_path, plan)
    matrix.update_campaign(tmp_path, campaign, plan["tests"][0], "qwen", {"status": "passed"})
    assert campaign["full_scope_passed"] is False
    assert campaign["cases"][1]["status"] == "unrun"
