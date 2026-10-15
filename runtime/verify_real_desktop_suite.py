#!/usr/bin/env python3
"""Sequential real desktop TTS→ASR acceptance for one already-deployed model.

No services or models are started by this wrapper. Each existing application
runner owns its windows, task logic, and evidence. Default: stop on first failure.
--case selects a fresh subset run; omitted cases remain unrun, never inherited
as passes from an older manifest. Child stdout/stderr stay in runtime-data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    module: str
    schema: str
    scenario: str | None = None
    policy: bool = False
    turns: int = 1


CASES = (
    *(Case("gedit-" + scenario, "runtime.verify_real_desktop", "robit.omni.real-desktop.v1", scenario)
      for scenario in ("selection-only", "double-click-word", "replace-save", "context-menu", "full-spelled")),
    Case("calculator-arithmetic", "runtime.verify_real_calculator", "robit.omni.real-calculator.v1", "click-arithmetic"),
    Case("calculator-visual-result", "runtime.verify_real_calculator", "robit.omni.real-calculator.v1", "visual-result"),
    Case("calculator-require-policy", "runtime.verify_real_calculator", "robit.omni.real-calculator.v1", "click-arithmetic", True),
    Case("files", "runtime.verify_real_files", "robit.omni.real-files.v1"),
    Case("firefox-docs-memory", "runtime.verify_real_firefox", "robit.omni.real-firefox-speech.v1", "python-numbers", turns=2),
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def file_hash(path: Path) -> str:
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def speech_status(root: Path, case: Case, model: str, *, think: bool = False) -> dict[str, Any]:
    tts, asr = [], []
    for index in range(case.turns):
        turn = root / f"turn-{index + 1}" if case.turns > 1 else root
        source, heard = read_json(turn / "tts-evidence.json"), read_json(turn / "speech-evidence.json")
        wav_hash = file_hash(turn / "instruction.wav")
        tts.append(bool(wav_hash and source.get("source") == "generated_tts_audio"
                        and source.get("model") == model and source.get("wav_sha256") == wav_hash))
        asr.append(bool(tts[-1] and heard.get("source") == "generated_tts_audio"
                        and heard.get("model") == model and heard.get("input_wav_sha256") == wav_hash
                        and heard.get("typed_goal_substituted") is False
                        and heard.get("think") is think
                        and isinstance(heard.get("asr_transcript"), str) and heard["asr_transcript"].strip()))
    return {"expected_turns": case.turns, "think": think, "tts_verified_turns": sum(tts), "asr_verified_turns": sum(asr),
            "tts": "verified" if all(tts) else "not_confirmed", "asr": "verified" if all(asr) else "not_confirmed"}


def case_result(root: Path, case: Case, model: str, returncode: int, *, timed_out: bool = False,
                think: bool = False) -> dict[str, Any]:
    report = read_json(root / "summary.json")
    speech = speech_status(root, case, model, think=think)
    criteria = report.get("criteria")
    turns = report.get("turns")
    live = (isinstance(turns, list) and len(turns) == case.turns and all(
        isinstance(turn, dict) and turn.get("authoritative_final_received") is True for turn in turns)) if case.turns > 1 else report.get("authoritative_final_received") is True
    cleanup = report.get("owned_editor_closed", report.get("owned_calculator_closed", report.get("owned_window_closed"))) is True
    errors = []
    for condition, code in (
        (returncode != 0, "runner_nonzero"), (timed_out, "suite_case_timeout"),
        (not report, "missing_or_invalid_summary"), (report.get("schema") != case.schema, "wrong_runner_schema"),
        (report.get("model") != model, "wrong_model"), (report.get("passed") is not True, "runner_failed"),
        (report.get("think") is not think, "wrong_think"),
        (not isinstance(criteria, dict) or not criteria or any(value is not True for value in criteria.values()), "criteria_not_all_verified"),
        (report.get("source") != "generated_tts_audio" or report.get("physical_microphone") is not False
         or report.get("typed_goal_substituted") is not False, "wrong_speech_provenance"),
        (speech["tts"] != "verified" or speech["asr"] != "verified", "speech_evidence_not_verified"),
        (not live, "authoritative_final_not_confirmed"), (not cleanup, "owned_window_cleanup_not_confirmed"),
        (case.scenario is not None and report.get("scenario") != case.scenario, "wrong_scenario"),
        (case.policy and report.get("policy_required") is not True, "required_policy_not_requested"),
    ):
        if condition:
            errors.append(code)
    return {"name": case.name, "think": think, "status": "failed" if errors else "passed", "artifact_directory": str(root),
            "returncode": returncode, "failure_codes": errors, "speech": speech,
            "live": "authoritative_final" if live else "not_confirmed", "owned_window_closed": cleanup,
            "criteria_passed": sum(value is True for value in criteria.values()) if isinstance(criteria, dict) else 0,
            "criteria_total": len(criteria) if isinstance(criteria, dict) else 0,
            "summary_sha256": file_hash(root / "summary.json")}


def child_command(args: argparse.Namespace, case: Case, root: Path) -> list[str]:
    command = [sys.executable, "-m", case.module, "--model", args.model, "--endpoint", args.endpoint,
               "--token-file", str(args.token_file.resolve()), "--timeout", str(args.timeout), "--output", str(root)]
    if case.scenario:
        command.extend(["--scenario", case.scenario])
    if case.policy:
        command.append("--require-policy")
    if args.think:
        command.append("--think")
    return command


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def launch(args: argparse.Namespace, case: Case, root: Path, logs: Path) -> tuple[int, bool]:
    with (logs / f"{case.name}.stdout.log").open("wb") as stdout, (logs / f"{case.name}.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(child_command(args, case, root), cwd=REPO,
                                   stdout=stdout, stderr=stderr)  # noqa: S603
        try:
            return process.wait(timeout=args.case_timeout), False
        except subprocess.TimeoutExpired:
            stop_child(process)
            return process.returncode, True
        except KeyboardInterrupt:
            stop_child(process)
            raise


def save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    temporary = root / "manifest.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")


def run(args: argparse.Namespace) -> int:
    if type(args.think) is not bool:
        raise ValueError("think must be a native boolean")
    selected = set(args.case or [case.name for case in CASES])
    if selected - {case.name for case in CASES}:
        raise ValueError("Unknown suite case")
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Endpoint must have no credentials, query, or capability fragment")
    root = (args.output or REPO / "runtime-data" / ("real-desktop-suite-" + secrets.token_hex(6))).resolve()
    if not root.is_relative_to(REPO / "runtime-data"):
        raise ValueError("Suite evidence must remain in this checkout's runtime-data")
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    (root / "cases").mkdir()
    manifest: dict[str, Any] = {"schema": "robit.omni.real-desktop-suite.v1", "model": args.model, "think": args.think,
        "endpoint": args.endpoint, "source": "generated_tts_audio", "physical_microphone": False,
        "typed_goal_substituted": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "stop_on_failure": not args.keep_going, "selected_cases": [case.name for case in CASES if case.name in selected],
        "selected_scope_passed": False, "full_scope_passed": False,
        "cases": [{"name": case.name, "think": args.think, "status": "unrun", "reason": "pending" if case.name in selected else "not_selected"}
                  for case in CASES]}
    save_manifest(root, manifest)
    started, stopped = time.monotonic(), False
    for index, case in enumerate(CASES):
        if case.name not in selected:
            continue
        if stopped:
            manifest["cases"][index]["reason"] = "stopped_after_failure"
            continue
        artifact = root / "cases" / case.name
        try:
            code, timed_out = launch(args, case, artifact, logs)
            result = case_result(artifact, case, args.model, code, timed_out=timed_out, think=args.think)
        except (OSError, KeyboardInterrupt) as error:
            result = {"name": case.name, "think": args.think, "status": "failed", "artifact_directory": str(artifact),
                      "failure_codes": ["interrupted" if isinstance(error, KeyboardInterrupt) else "runner_launch_failed"],
                      "speech": {"tts": "not_confirmed", "asr": "not_confirmed"}, "live": "not_confirmed"}
            stopped = True
        manifest["cases"][index] = result
        stopped = stopped or (result["status"] == "failed" and not args.keep_going)
        # Do not continue into another app if the owned window cleanup is uncertain.
        stopped = stopped or result.get("owned_window_closed") is not True
        save_manifest(root, manifest)
    manifest["selected_scope_passed"] = all(item["status"] == "passed" for item in manifest["cases"] if item["name"] in selected)
    manifest["full_scope_passed"] = all(item["status"] == "passed" for item in manifest["cases"])
    manifest["elapsed_s"] = time.monotonic() - started
    save_manifest(root, manifest)
    counts = {status: sum(item["status"] == status for item in manifest["cases"]) for status in ("passed", "failed", "unrun")}
    print(json.dumps({"manifest": str(root / "manifest.json"), "model": args.model, "think": args.think, "counts": counts,
        "selected_scope_passed": manifest["selected_scope_passed"], "full_scope_passed": manifest["full_scope_passed"]}))
    return 0 if manifest["selected_scope_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact model tag of an already-running deployment")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--think", action="store_true", help="Forward native think=true to each runner; default is false")
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES], help="Select one or more cases for a fresh subset/resume run")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed case only if its owned window was closed")
    parser.add_argument("--timeout", type=float, default=180, help="Each runner's spoken-turn deadline in seconds")
    parser.add_argument("--case-timeout", type=float, default=600, help="Bounded child-run deadline, followed by graceful cleanup")
    parser.add_argument("--output", type=Path, help="New directory under runtime-data; existing runs are never overwritten")
    args = parser.parse_args()
    if not (0 < args.timeout <= 600 and 0 < args.case_timeout <= 1800):
        parser.error("timeout must be in (0,600] and case-timeout in (0,1800]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
