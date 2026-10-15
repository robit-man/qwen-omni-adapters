#!/usr/bin/env python3
"""Serial, resumable live acceptance with a fresh note after every attempt.

This runner never loads models or manages GPU leases. A single live operator
selects the already-running exact model, its token file, and CUDA process IDs.
The campaign retains failed attempts and pending cases across profile switches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import secrets
import signal
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from qwen_omni_adapters.accelerator import process_is_gpu_resident
from runtime.verify_real_desktop_suite import CASES, case_result, read_json

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / ".aiwg/testing/computer-use-100-cases.json"
GPU = "GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0"
SCHEMA = "robit.omni.live-desktop-campaign.v1"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_plan() -> dict[str, Any]:
    plan = read_json(PLAN)
    tests = plan.get("tests", [])
    if (plan.get("schema") != "robit.omni.computer-use-acceptance-plan.v1"
            or [case.get("id") for case in tests] != [f"CU-{number:03d}" for number in range(1, 101)]
            or plan.get("gpu", {}).get("uuid") != GPU):
        raise ValueError("The exact 100-case GPU 1 plan is required")
    return plan


def campaign_path(path: Path) -> Path:
    root = path.resolve()
    if not root.is_relative_to((REPO / "runtime-data").resolve()):
        raise ValueError("Campaign artifacts must remain under runtime-data")
    return root


def open_campaign(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = root / "campaign.json"
    if path.exists():
        result = read_json(path)
        if (result.get("schema") != SCHEMA or result.get("plan_sha256") != digest(PLAN)
                or result.get("gpu_uuid") != GPU):
            raise ValueError("Existing campaign differs from the exact plan or GPU scope")
        return result
    root.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "created_at": stamp(), "plan_sha256": digest(PLAN),
              "gpu_uuid": GPU, "status": "pending", "full_scope_passed": False,
              "cases": [{"case_id": case["id"], "title": case["title"],
                         "model": model["tag"], "status": "unrun", "attempts": []}
                        for model in plan["models"] for case in plan["tests"]]}
    save(path, result)
    return result


def source_revision() -> str:
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True, timeout=5)
    return process.stdout.strip()


def broker_evidence(pids: list[int]) -> dict[str, Any]:
    process = subprocess.run(["docker", "gpu", "discover"], capture_output=True,
                             text=True, check=True, timeout=15)
    discovery = json.loads(process.stdout)
    health = discovery.get("gpu_health", {})
    if (not discovery.get("available") or health.get("admission_blocked") is not False
            or health.get("missing_selected_gpu_ids") or discovery.get("unregistered_gpu_quarantine")):
        raise RuntimeError("GPU health or quarantine blocks live acceptance")
    if GPU not in discovery.get("selected_gpu_ids", []):
        raise RuntimeError("GPU 1 is no longer broker-selected")
    others = [item["uuid"] for item in discovery.get("gpus", []) if item.get("uuid") != GPU]
    residents = []
    for pid in pids:
        # Inspect only CUDA visibility, never dump the process environment.
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        visibility = next((item.partition(b"=")[2].decode()
                           for item in environment if item.startswith(b"CUDA_VISIBLE_DEVICES=")), "")
        on_scope = process_is_gpu_resident(pid, GPU)
        outside = [uuid for uuid in others if process_is_gpu_resident(pid, uuid)]
        residents.append({"pid": pid, "cuda_visible_devices": visibility,
                          "resident_on_gpu1": on_scope, "other_selected_residency": outside})
        if visibility != GPU or not on_scope or outside:
            raise RuntimeError("An owned CUDA process has not proved exact GPU 1 residency")
    if not residents:
        raise ValueError("Supply the already-running trunk and required resident policy CUDA PIDs")
    # Discovery excludes tokens; retain only coordination and health fields.
    return {"at": stamp(), "gpu_uuid": GPU, "gpu_health": health,
            "warnings": discovery.get("warnings", []),
            "inference_hold": discovery.get("inference_hold", {}),
            "active_leases": discovery.get("active_leases", []), "processes": residents}


def runner_module(case: dict[str, Any]) -> str:
    if case.get("existing_suite_case"):
        return case["existing_runner"]["module"]
    number = int(case["id"].removeprefix("CU-"))
    if number <= 40:
        return "runtime.verify_extended_native"
    if number <= 70 or number >= 91:
        return "runtime.verify_extended_browser"
    return "runtime.verify_extended_perception_voice"


def child_command(args: argparse.Namespace, case: dict[str, Any], artifact: Path) -> list[str]:
    command = [sys.executable, "-m", runner_module(case), "--model", args.model,
               "--endpoint", args.endpoint, "--token-file", str(args.token_file.resolve()),
               "--output", str(artifact), "--timeout", str(args.timeout)]
    if case.get("existing_suite_case"):
        existing = case["existing_runner"]
        if existing.get("scenario"):
            command.extend(["--scenario", existing["scenario"]])
        if existing.get("require_policy"):
            command.append("--require-policy")
    else:
        command.extend(["--case", case["id"]])
    if args.think:
        command.append("--think")
    return command


def extended_result(root: Path, case: dict[str, Any], model: str, think: bool,
                    returncode: int) -> dict[str, Any]:
    report = read_json(root / "summary.json")
    criteria = report.get("criteria", {})
    failures = []
    for condition, code in (
        (report.get("schema") != "robit.omni.extended-desktop.v1", "wrong_runner_schema"),
        (report.get("case_id") != case["id"], "wrong_case"),
        (report.get("model") != model or report.get("think") is not think, "wrong_model_or_think"),
        (report.get("passed") is not True or returncode != 0, "runner_failed"),
        (not isinstance(criteria, dict) or not criteria or any(item is not True for item in criteria.values()),
         "criteria_not_all_verified"),
        (report.get("typed_goal_substituted") is not False, "speech_provenance_unconfirmed"),
        (report.get("owned_window_closed") is not True, "cleanup_unconfirmed"),
    ):
        if condition:
            failures.append(code)
    if case["input_source"] == "omni_tts_to_asr":
        speech = read_json(root / "speech-evidence.json")
        source = read_json(root / "tts-evidence.json")
        wav_hash = digest(root / "instruction.wav")
        try:
            with wave.open(str(root / "instruction.wav"), "rb") as audio:
                valid_wav = ((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
                             and 0 < audio.getnframes() <= 24000 * 180
                             and bool(any(audio.readframes(audio.getnframes()))))
        except (OSError, EOFError, wave.Error):
            valid_wav = False
        # Multi-turn runners additionally verify and report every required turn.
        speech_ok = bool(valid_wav and wav_hash and source.get("model") == model
                         and source.get("source") == "generated_tts_audio"
                         and source.get("wav_sha256") == wav_hash
                         and speech.get("model") == model and speech.get("think") is think
                         and speech.get("source") == "generated_tts_audio"
                         and speech.get("input_wav_sha256") == wav_hash
                         and speech.get("typed_goal_substituted") is False
                         and str(speech.get("asr_transcript", "")).strip())
        if not speech_ok:
            failures.append("genuine_speech_evidence_unconfirmed")
        if report.get("source") != "generated_tts_audio" or report.get("physical_microphone") is not False:
            failures.append("wrong_speech_source")
        if report.get("authoritative_final_received") is not True:
            failures.append("authoritative_final_not_confirmed")
    else:
        if not capture_confirmed(root, case, report):
            failures.append("capture_provenance_unconfirmed")
        if case["id"] != "CU-086" and report.get("authoritative_final_received") is not True:
            failures.append("authoritative_final_not_confirmed")
    status = "passed" if not failures else "failed"
    if report.get("status") in {"blocked_capability", "blocked_environment", "blocked_operator"}:
        status = report["status"]
    return {"status": status, "failure_codes": failures,
            "criteria": criteria if isinstance(criteria, dict) else {},
            "failure_stage": report.get("failure_stage"),
            "owned_window_closed": report.get("owned_window_closed") is True,
            "hard_fault": report.get("hard_fault") is True,
            "summary_sha256": digest(root / "summary.json")}


def capture_confirmed(root: Path, case: dict[str, Any], report: dict[str, Any]) -> bool:
    """A boolean claim without the actual bound capture cannot pass mic cases."""
    capture = read_json(root / "capture-evidence.json")
    expected = {"physical_capture": "physical_microphone", "captured_non_speech": "pulse_capture",
                "omni_tts_through_call_capture": "pulse_capture"}[case["input_source"]]
    if (report.get("capture_provenance_verified") is not True or capture.get("source") != expected
            or not capture.get("device") or capture.get("wav_sha256") != digest(root / "capture.wav")
            or not capture.get("wav_sha256") or capture.get("vad_trace_sha256") != digest(root / "vad-events.jsonl")
            or not capture.get("vad_trace_sha256")):
        return False
    try:
        with wave.open(str(root / "capture.wav"), "rb") as audio:
            return ((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
                    and 0 < audio.getnframes() <= 16000 * 300)
    except (OSError, EOFError, wave.Error):
        return False


def evaluate(root: Path, case: dict[str, Any], model: str, think: bool,
             returncode: int) -> dict[str, Any]:
    if not case.get("existing_suite_case"):
        return extended_result(root, case, model, think, returncode)
    definition = next(item for item in CASES if item.name == case["existing_suite_case"])
    result = case_result(root, definition, model, returncode, think=think)
    report = read_json(root / "summary.json")
    result["criteria"] = report.get("criteria", {})
    result["hard_fault"] = report.get("hard_fault") is True
    result["failure_stage"] = report.get("failure_stage")
    return result


def write_note(root: Path, case: dict[str, Any], model: str, result: dict[str, Any]) -> None:
    failures = [name for name, value in result.get("criteria", {}).items() if value is not True]
    summary = read_json(root / "summary.json")
    actions = []
    try:
        outcome_path = root / "native-outcomes.json"
        if outcome_path.stat().st_size <= 16 * 1024 * 1024:
            outcomes = json.loads(outcome_path.read_text())
            if isinstance(outcomes, list):
                actions = [{"verb": item.get("action", {}).get("verb"), "status": item.get("status"),
                            "dispatch_status": item.get("receipt", {}).get("status"),
                            "backend": item.get("receipt", {}).get("backend")}
                           for item in outcomes if isinstance(item, dict)]
    except (OSError, ValueError):
        pass
    note = {"at": stamp(), "case_id": case["id"], "title": case["title"], "model": model,
            "gpu_uuid": GPU, "status": result["status"], "source_revision": result["source_revision"],
            "elapsed_s": result.get("elapsed_s"), "failure_stage": result.get("failure_stage"),
            "failed_criteria": failures, "failure_codes": result.get("failure_codes", []),
            "independent_success_check": case["independent_success_check"],
            "observed_actions": actions,
            "tts_wav_sha256": digest(root / "instruction.wav"),
            "capture_wav_sha256": digest(root / "capture.wav"),
            "asr_transcript_sha256": summary.get("asr_transcript_sha256", ""),
            "file_receipt": summary.get("file", {}),
            "tool_names": summary.get("tool_names", []),
            "owned_window_closed": result.get("owned_window_closed", False),
            "artifact_directory": str(root), "summary_sha256": result.get("summary_sha256", "")}
    save(root / "after-test-note.json", note)
    lines = [f"# {case['id']}: {case['title']}", "", f"Model: `{model}`.",
             f"Outcome: **{result['status']}**. GPU: `{GPU}`.",
             f"Source revision: `{result['source_revision']}`.", "",
             "Independent check: " + case["independent_success_check"], "",
             "Failed criteria: " + (", ".join(failures) or "none recorded"),
             "Failure codes: " + (", ".join(result.get("failure_codes", [])) or "none"),
             "Observed inputs: " + (", ".join(str(item["verb"]) + "/" + str(item["dispatch_status"])
                                               for item in actions) or "see captured stream evidence"),
             "Owned-window cleanup confirmed: " + str(note["owned_window_closed"]), "",
             "See the retained speech, action, fresh-state and independent-oracle artifacts in this directory."]
    (root / "after-test-note.md").write_text("\n".join(lines) + "\n")


def update_campaign(root: Path, campaign: dict[str, Any], case: dict[str, Any],
                    model: str, result: dict[str, Any]) -> None:
    slot = next(item for item in campaign["cases"] if item["case_id"] == case["id"] and item["model"] == model)
    slot["attempts"].append(result)
    slot["status"] = result["status"]
    campaign["updated_at"] = stamp()
    campaign["full_scope_passed"] = all(item["status"] == "passed" for item in campaign["cases"])
    campaign["status"] = "passed" if campaign["full_scope_passed"] else "incomplete"
    save(root / "campaign.json", campaign)
    rows = ["# Live desktop acceptance progress", "",
            "Every row retains all attempts. Unrun and blocked cases are not passes.", "",
            "| Case | Model | Latest result | Attempts |", "| --- | --- | --- | --- |"]
    rows.extend(f"| {item['case_id']} | {item['model']} | {item['status']} | {len(item['attempts'])} |"
                for item in campaign["cases"])
    (root / "progress.md").write_text("\n".join(rows) + "\n")


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(args: argparse.Namespace) -> int:
    plan = load_plan()
    if args.model not in {item["tag"] for item in plan["models"]}:
        raise ValueError("Select one exact Qwen or Ornith model from the plan")
    endpoint = urlsplit(args.endpoint)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username
            or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("Use a plain endpoint with no credentials, query or capability fragment")
    selected = args.case or [case["id"] for case in plan["tests"]]
    if len(set(selected)) != len(selected) or set(selected) - {case["id"] for case in plan["tests"]}:
        raise ValueError("Select distinct exact case IDs")
    root = campaign_path(args.campaign)
    campaign = open_campaign(root, plan)
    with (REPO / "runtime-data" / "gpu1-acceptance" / "desktop-input.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with httpx.Client(headers={"Authorization": "Bearer " + args.token_file.read_text().strip()},
                          timeout=10, trust_env=False) as client:
            response = client.get(args.endpoint.rstrip("/") + "/api/status")
            response.raise_for_status()
            if response.json().get("model") != args.model:
                raise ValueError("Endpoint belongs to another model")
        for case in plan["tests"]:
            if case["id"] not in selected:
                continue
            # Completed results are not silently re-run; explicit --repeat retains
            # a new attempt with a new artifact directory instead of overwriting.
            slot = next(item for item in campaign["cases"]
                        if item["case_id"] == case["id"] and item["model"] == args.model)
            if slot["attempts"] and not args.repeat:
                continue
            artifact = root / "attempts" / (case["id"] + "-" + args.model.split("/")[-1].split(":")[0]
                                           + "-" + secrets.token_hex(4))
            artifact.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            before = broker_evidence(args.cuda_pid)
            # Application runners own creation of their output directory.
            log_root = root / "logs"
            log_root.mkdir(exist_ok=True)
            stdout_path, stderr_path = log_root / (artifact.name + ".stdout.log"), log_root / (artifact.name + ".stderr.log")
            timed_out, interrupted, returncode = False, False, 1
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(child_command(args, case, artifact), cwd=REPO, stdout=stdout, stderr=stderr)
                try:
                    returncode = process.wait(timeout=case["case_deadline_seconds"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop_child(process)
                    returncode = process.returncode
                except KeyboardInterrupt:
                    stop_child(process)
                    interrupted = True
                    returncode = process.returncode
            artifact.mkdir(parents=True, exist_ok=True)
            result = evaluate(artifact, case, args.model, args.think, returncode)
            result.update(at=stamp(), source_revision=source_revision(), elapsed_s=time.monotonic() - started,
                          artifact_directory=str(artifact), runner_sha256=digest(REPO / (runner_module(case).replace(".", "/") + ".py")),
                          before_gpu=before, timed_out=timed_out, returncode=returncode)
            if timed_out:
                result["status"] = "failed"
                result.setdefault("failure_codes", []).append("case_deadline_exceeded")
            if interrupted:
                result["status"] = "failed"
                result.setdefault("failure_codes", []).append("interrupted")
            try:
                result["after_gpu"] = broker_evidence(args.cuda_pid)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result["status"] = "failed"
                result["hard_fault"] = True
                result["failure_stage"] = "gpu_health_or_residency"
                result.setdefault("failure_codes", []).append(type(error).__name__)
            write_note(artifact, case, args.model, result)
            update_campaign(root, campaign, case, args.model, result)
            print(json.dumps({"case": case["id"], "model": args.model, "status": result["status"],
                              "note": str(artifact / "after-test-note.md")}), flush=True)
            if (interrupted or result.get("hard_fault") or result.get("owned_window_closed") is not True
                    or result["status"] != "passed" and not args.keep_going):
                return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--cuda-pid", type=int, action="append", required=True,
                        help="Already resident owned trunk/policy PID; repeat per resident worker")
    parser.add_argument("--case", action="append")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Continue only after confirmed cleanup and healthy GPU")
    parser.add_argument("--repeat", action="store_true", help="Retain a fresh attempt for already-attempted selections")
    args = parser.parse_args()
    if not 0 < args.timeout <= 600 or any(pid <= 0 for pid in args.cuda_pid):
        parser.error("Use a positive finite timeout <=600 and valid owned CUDA PIDs")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
