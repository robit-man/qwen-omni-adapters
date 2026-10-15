#!/usr/bin/env python3
"""Genuine capture, speech and grounded native acceptance cases CU071–CU090.

Importing never arms a microphone, plays audio, loads models or opens an app.
The live operator alone runs this module. Capture cases use the production call
loop and immutable evidence hooks; Pulse monitor capture is never labeled mic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from harness.call import CallConfig, run_call_loop
from portal.desktop import desktop_subprocess_environment
from runtime.acceptance_native_support import (
    CapabilityBlocked,
    NativeEvidenceObserver,
    OwnedDesktop,
)
from runtime.computer_use_speech import synthesize_instruction
from runtime.verify_real_desktop import (
    PointerObserver,
    actual_pointer_movement,
    dispatched_actions,
    outcomes_from_events,
    physical_pointer_action,
)

REPO = Path(__file__).resolve().parents[1]
CASE_IDS = tuple(f"CU-{number:03}" for number in range(71, 91))


def case_spec(case_id: str) -> dict[str, Any]:
    if case_id not in CASE_IDS:
        raise ValueError("Case is outside perception/voice acceptance scope")
    return next(item for item in json.loads((REPO / ".aiwg/testing/computer-use-100-cases.json").read_text())["tests"] if item["id"] == case_id)


def pulse_command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)  # noqa: S603
    if result.returncode:
        raise CapabilityBlocked("PulseAudio capability command failed: " + argv[0])
    return result.stdout.strip()


class OwnedPulse:
    """One explicitly named null sink; never changes system defaults."""
    def __init__(self):
        self.sink = "omni_acceptance_" + secrets.token_hex(8)
        self.module = ""

    def open(self):
        self.module = pulse_command(["pactl", "load-module", "module-null-sink", "sink_name=" + self.sink,
                                     "rate=16000", "channels=1"])
        if not self.module.isdecimal():
            raise CapabilityBlocked("Pulse did not return an owned module identity")
        return self

    @property
    def source(self):
        return self.sink + ".monitor"

    def close(self) -> bool:
        if not self.module:
            return True
        try:
            pulse_command(["pactl", "unload-module", self.module])
            self.module = ""
            return True
        except CapabilityBlocked:
            return False


class CaptureRecorder:
    """Persist real frames, VAD decisions and exact submitted WAVs unchanged."""
    def __init__(self, root: Path, *, source: str, device: str, microphone_ready: bool = False,
                 device_metadata: dict[str, Any] | None = None):
        if source not in {"physical_microphone", "pulse_capture"} or not device:
            raise ValueError("Capture needs a distinct actual source and device")
        self.root, self.source, self.device = root, source, device
        self.microphone_ready = microphone_ready
        self.listening = False
        self.device_metadata = dict(device_metadata or {})
        self.ready, self.lock = threading.Event(), threading.RLock()
        self.started, self.frames, self.turns = time.monotonic(), 0, []
        self.events, self.vad, self.rms = [], [], []
        self.capture = wave.open(str(root / "capture.wav"), "wb")  # noqa: SIM115 - lifecycle spans callback invocations
        self.capture.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        self.closed = False

    def __call__(self, kind: str, value: dict[str, Any]):
        with self.lock:
            if self.closed:
                return
            elapsed = time.monotonic() - self.started
            if kind == "capture_frame":
                samples = np.asarray(value.get("samples"))
                if value.get("sample_rate_hz") != 16000 or samples.ndim != 1 or not len(samples):
                    raise ValueError("Actual capture hook returned an invalid mono 16 kHz frame")
                pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                self.capture.writeframesraw(pcm)
                self.frames += len(samples)
                self.rms.append({"elapsed_s": elapsed, "rms": float(np.sqrt(np.mean(samples ** 2))),
                                 "peak": float(np.max(np.abs(samples)))})
                if not self.ready.is_set() and not self.microphone_ready:
                    self.ready.set()
                self._publish_ready()
            elif kind == "vad":
                item = {"elapsed_s": elapsed, **value}
                self.vad.append(item)
                with (self.root / "vad-events.jsonl").open("a") as journal:
                    journal.write(json.dumps(item) + "\n")
            elif kind == "turn_audio":
                raw = value.get("wav")
                if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != value.get("wav_sha256"):
                    raise ValueError("Submitted turn WAV does not match its actual transport hash")
                name = f"submitted-turn-{len(self.turns)+1:03}.wav"
                (self.root / name).write_bytes(raw)
                self.turns.append({"path": name, "wav_sha256": value["wav_sha256"], "segments": value.get("segments"),
                                   "sample_rate_hz": value.get("sample_rate_hz"), "elapsed_s": elapsed})
            elif kind == "stream_event":
                event = value.get("event")
                if isinstance(event, dict):
                    self.events.append(event)
                    with (self.root / "speech-events.jsonl").open("a") as journal:
                        journal.write(json.dumps({"elapsed_s": elapsed, **event}) + "\n")

    def on_state(self, state: str, _detail: str):
        with self.lock:
            if state == "listening":
                self.listening = True
                self._publish_ready()

    def _publish_ready(self):
        if self.microphone_ready and self.listening and self.frames and not self.ready.is_set():
            (self.root / "microphone-ready.json").write_text(json.dumps({"ready": True,
                "source": self.source, "device": self.device, "actual_frame_samples": self.frames,
                "listening_observed": True, "sample_rate_hz": 16000,
                "physical_device_confirmed": self.device_metadata.get("physical_device_confirmed") is True,
                "cue": "Capture is live; wait for the attention blip and speech-command-ready before speaking."}) + "\n")
            self.ready.set()

    def close(self) -> dict[str, Any]:
        with self.lock:
            if not self.closed:
                self.capture.close()
                self.closed = True
            for path in (self.root / "vad-events.jsonl", self.root / "speech-events.jsonl"):
                path.touch(exist_ok=True)
            evidence = {"source": self.source, "device": self.device, "sample_rate_hz": 16000,
                        "channels": 1, "sample_width_bits": 16, "frames": self.frames,
                        "wav_sha256": hashlib.sha256((self.root / "capture.wav").read_bytes()).hexdigest(),
                        "vad_trace_sha256": hashlib.sha256((self.root / "vad-events.jsonl").read_bytes()).hexdigest(),
                        "submitted_turns": self.turns, "captured_frame_count": len(self.rms)}
            evidence.update(self.device_metadata)
            (self.root / "capture-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            (self.root / "capture-levels.json").write_text(json.dumps(self.rms, indent=2) + "\n")
            return evidence


def non_speech_fixture(path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(86)
    quiet = np.zeros(16000 * 2, dtype=np.float32)
    click = quiet.copy()
    click[8000:8080] = .35
    noise = rng.normal(0, .004, 16000 * 3).astype(np.float32)
    samples = np.concatenate((quiet, click, quiet, noise, quiet))
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes((samples * 32767).astype("<i2").tobytes())
    return {"source": "generated_non_speech_fixture", "components": ["quiet", "isolated_click", "quiet", "steady_noise", "quiet"],
            "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "duration_s": len(samples)/16000}


def capture_device_metadata(device: str, *, physical: bool) -> dict[str, Any]:
    sources = json.loads(pulse_command(["pactl", "--format=json", "list", "sources"]))
    matches = [source for source in sources if source.get("name") == device]
    if len(matches) != 1:
        raise CapabilityBlocked("The exact selected Pulse capture source is not unique")
    source = matches[0]
    properties = source.get("properties", {})
    monitor = source.get("monitor_of_sink")
    is_monitor = ".monitor" in device or monitor not in {None, "n/a", "N/A", 4294967295}
    hardware = not is_monitor and (properties.get("device.api") == "alsa" or properties.get("device.class") == "sound"
        and ("alsa.card" in properties or "api.alsa.path" in properties))
    if physical and not hardware:
        raise CapabilityBlocked("Selected input is not proven physical ALSA microphone hardware")
    return {"physical_device_confirmed": bool(hardware), "pulse_source_name": source["name"],
            "pulse_source_index": source.get("index"), "monitor_of_sink": monitor,
            "device_properties": {key: properties[key] for key in ("device.api", "device.class", "device.bus", "alsa.card", "api.alsa.path") if key in properties}}


def attention_cue_wav(path: Path) -> dict[str, Any]:
    rate, frequency, duration, amplitude = 48000, 8000, .08, .05
    samples = np.sin(2*np.pi*frequency*np.arange(round(rate*duration))/rate) * amplitude
    fade = round(rate*.005)
    samples[:fade] *= np.linspace(0, 1, fade)
    samples[-fade:] *= np.linspace(1, 0, fade)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes((samples*32767).astype("<i2").tobytes())
    return {"frequency_hz": frequency, "sample_rate_hz": rate, "duration_s": duration, "peak_amplitude": amplitude,
            "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def play_attention_cue(root: Path, recorder: CaptureRecorder, output_device: str | None) -> dict[str, Any]:
    if not recorder.ready.is_set() or not recorder.listening or not recorder.frames:
        raise RuntimeError("Attention cue requires actual microphone frames and listening readiness")
    sink = output_device or pulse_command(["pactl", "get-default-sink"])
    sinks = json.loads(pulse_command(["pactl", "--format=json", "list", "sinks"]))
    matches = [item for item in sinks if item.get("name") == sink]
    if len(matches) != 1 or matches[0].get("properties", {}).get("device.api") not in {"alsa", "bluez", "bluez5"}:
        raise CapabilityBlocked("Attention cue requires a positively identified physical output sink")
    cue = attention_cue_wav(root / "attention-cue.wav")
    cue.update(output_device=sink, started_at_unix=time.time())
    played = subprocess.run(["paplay", "--device=" + sink, str(root / "attention-cue.wav")],
                            capture_output=True, timeout=8, check=False)  # noqa: S603
    if played.returncode:
        raise CapabilityBlocked("Physical attention cue playback failed")
    cue["completed_at_unix"] = time.time()
    finished = time.monotonic() - recorder.started
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        with recorder.lock:
            quiet = [item for item in recorder.rms if item["elapsed_s"] >= finished]
            admitted = bool(recorder.turns) or any(item.get("event") == "utterance" for item in recorder.vad)
        if admitted:
            raise RuntimeError("Speech was admitted before the attention cue/quiet readiness boundary")
        if quiet and quiet[-1]["elapsed_s"] - quiet[0]["elapsed_s"] >= 1 and all(item["rms"] < .008 for item in quiet):
            cue["quiet_capture_confirmed"] = True
            (root / "attention-cue.json").write_text(json.dumps(cue, indent=2) + "\n")
            (root / "speech-command-ready.json").write_text(json.dumps({"ready": True,
                "attention_cue_sha256": cue["wav_sha256"], "quiet_capture_confirmed": True,
                "spoken_instruction": case_spec("CU-087")["spoken_instruction"]}) + "\n")
            return cue
        time.sleep(.05)
    raise CapabilityBlocked("Microphone did not settle quietly after the attention blip")


def capture_checks(case_id: str, recorder: CaptureRecorder, fresh: dict[str, Any], *,
                   initial_document: dict[str, Any] | None = None) -> dict[str, bool]:
    outcomes = outcomes_from_events(recorder.events)
    checks = {"actual_capture_frames": recorder.frames >= 16000,
              "actual_vad_trace": bool(recorder.vad),
              "capture_source_not_substituted": recorder.source == ("pulse_capture" if case_id in {"CU-086", "CU-090"} else "physical_microphone")}
    if case_id == "CU-086":
        checks.update(zero_submitted_turns=not recorder.turns, zero_stream_events=not recorder.events,
                      zero_gui_dispatch=not dispatched_actions(outcomes),
                      quiet_click_and_noise_captured=any(item["peak"] > .2 for item in recorder.rms)
                      and any(.001 < item["rms"] < .01 for item in recorder.rms)
                      and any(item["rms"] < .0001 for item in recorder.rms))
    else:
        expected = "copper finch ready" if case_id == "CU-087" else "15" if case_id == "CU-088" else "42"
        checks.update(actual_submitted_audio=bool(recorder.turns), actual_speech_admission=any(item.get("event") == "utterance" for item in recorder.vad),
                      actual_asr_transcript=any(item.get("type") == "observation" and bool(item.get("transcript")) for item in recorder.events),
                      exact_fresh_visible_result=any(node.get("visible") is True and node.get("role") in {"textbox", "text", "label"}
                          and node.get("value", node.get("name")) == expected for node in fresh.get("nodes", [])),
                      only_native_task_tools=all(call.get("name") in {"native_ui", "tool_search"} for event in recorder.events for call in event.get("tools", [])))
        if recorder.source == "physical_microphone":
            checks["physical_device_metadata_confirmed"] = recorder.device_metadata.get("physical_device_confirmed") is True
        if case_id == "CU-087":
            initial = initial_document or {}
            document = next((node for node in fresh.get("nodes", []) if node.get("id") == initial.get("id")), {})
            checks["exact_fresh_visible_result"] = (bool(initial.get("id") and initial.get("generation") and initial.get("scope_id") and initial.get("window_id"))
                and fresh.get("scope", {}).get("id") == initial["scope_id"]
                and fresh.get("scope", {}).get("window_id") == initial["window_id"]
                and document.get("generation") == initial["generation"] and document.get("scope_id") == initial["scope_id"]
                and document.get("role") == "textbox" and document.get("visible") is True and not document.get("sensitive")
                and document.get("value") == expected)
            checks["one_actual_microphone_turn"] = len(recorder.turns) == 1
            checks["actual_editor_keyboard_dispatch"] = (any(item.get("action", {}).get("verb") in {"type_text", "key_chord"} for item in dispatched_actions(outcomes))
                and not any(item.get("action", {}).get("verb") == "set_value" for item in dispatched_actions(outcomes)))
        if case_id == "CU-090":
            checks["actual_multi_segment_single_turn"] = len(recorder.turns) == 1 and recorder.turns[0].get("segments", 0) >= 2
    return checks


def run_capture(args, root: Path, report: dict[str, Any]) -> None:
    number = int(args.case[-3:])
    for binary in ("pactl", "paplay", "parecord", "xdotool"):
        if not shutil.which(binary):
            raise CapabilityBlocked("Actual audio capture dependency unavailable: " + binary)
    desktop = OwnedDesktop(desktop_subprocess_environment())
    observer = pulse = recorder = pointer = None
    stop, results, errors = threading.Event(), [], []
    worker = None
    stage, fresh, initial_document = "owned_setup", {}, {}
    try:
        if number == 87:
            desktop.open_editor(initial_text="")
        elif number in {88, 90}:
            desktop.open_calculator()
        observer = NativeEvidenceObserver(root, desktop)
        if number == 87:
            initial = observer.fresh().get("state", {})
            documents = [node for node in initial.get("nodes", []) if node.get("role") == "textbox" and node.get("visible") is True
                         and node.get("enabled") is True and not node.get("sensitive") and "set_value" in node.get("operations", [])]
            if len(documents) != 1 or documents[0].get("value") != "":
                raise CapabilityBlocked("Physical command needs one uniquely observed empty owned editor document")
            initial_document = {key: documents[0].get(key) for key in ("id", "generation", "scope_id")}
            initial_document["window_id"] = initial.get("scope", {}).get("window_id")
            (root / "initial-document.json").write_text(json.dumps({"binding": initial_document, "state": initial}, indent=2) + "\n")
        observer.start()
        observer.task_started.set()
        if number in {88, 90}:
            pointer = PointerObserver(root, desktop.environment, next(iter(desktop.window_ids)))
            pointer.thread.start()
        if number in {86, 90}:
            pulse = OwnedPulse().open()
            device, source = pulse.source, "pulse_capture"
        else:
            device = getattr(args, "input_device", None) or pulse_command(["pactl", "get-default-source"])
            if not device or ".monitor" in device:
                raise CapabilityBlocked("A physical microphone source is required, not a monitor")
            source = "physical_microphone"
        metadata = capture_device_metadata(device, physical=source == "physical_microphone")
        recorder = CaptureRecorder(root, source=source, device=device, microphone_ready=number == 87, device_metadata=metadata)
        config = CallConfig(portal_url=args.endpoint, token=args.token_file.read_text().strip(), model=args.model,
            portal_session_id=secrets.token_hex(32), input_device=device, output_device=getattr(args, "output_device", None),
            portal_tool_allowlist=("native_ui",),
            tools_enabled=True, camera_enabled=False, reasoning_enabled=args.think, request_timeout_s=args.timeout,
            barge_in_enabled=False, evidence_trace=recorder)
        def on_turn(result):
            results.append(result)
            stop.set()
        def listen():
            try:
                run_call_loop(config, stop=stop, on_turn=on_turn, on_state=recorder.on_state)
            except Exception as error:
                errors.append(error)
                stop.set()
        # Speech sources are synthesized before arming capture, so generation
        # latency cannot split a rapid multi-segment acoustic test into turns.
        prepared_sources = []
        if number in {88, 90}:
            instructions = [case_spec(args.case)["spoken_instruction"]] if number == 88 else ["Clear Calculator.", "Then calculate seventeen plus twenty five.", "Verify the answer."]
            with httpx.Client(headers={"Authorization": "Bearer " + config.token}, timeout=180, trust_env=False) as client:
                for index, instruction in enumerate(instructions):
                    target = root / f"source-{index+1}"
                    target.mkdir()
                    _raw, tts = synthesize_instruction(client, args.endpoint, args.model, instruction, target)
                    prepared_sources.append(target / "instruction.wav")
                    if index == 0:
                        shutil.copyfile(target / "instruction.wav", root / "instruction.wav")
                        shutil.copyfile(target / "tts-evidence.json", root / "tts-evidence.json")
                    report.setdefault("speaker_sources", []).append(tts)
        stage = "actual_capture"
        worker = threading.Thread(target=listen, daemon=True)
        worker.start()
        if not recorder.ready.wait(15):
            raise CapabilityBlocked("The production capture loop produced no actual frame")
        deadline = time.monotonic() + args.timeout
        if pointer:
            pointer.task_started.set()
        if number == 87:
            stage = "attention_cue"
            report["attention_cue"] = play_attention_cue(root, recorder, config.output_device)
            stage = "physical_user_speech"
        if number == 86:
            fixture = non_speech_fixture(root / "non-speech-fixture.wav")
            report["fixture"] = fixture
            playback = subprocess.run(["paplay", "--device=" + pulse.sink, str(root / "non-speech-fixture.wav")],
                timeout=fixture["duration_s"] + 10, capture_output=True, check=False)  # noqa: S603
            if playback.returncode:
                raise CapabilityBlocked("Owned Pulse fixture playback failed")
            stop.wait(2)
            stop.set()
        elif number in {88, 90}:
            for index, path in enumerate(prepared_sources):
                argv = ["paplay", str(path)]
                if pulse:
                    argv.insert(1, "--device=" + pulse.sink)
                elif config.output_device:
                    argv.insert(1, "--device=" + config.output_device)
                played = subprocess.run(argv, timeout=180, capture_output=True, check=False)  # noqa: S603
                if played.returncode:
                    raise CapabilityBlocked("Actual speaker instruction playback failed")
                if number == 90 and index < len(prepared_sources)-1:
                    stop.wait(.82)
        while not stop.is_set() and time.monotonic() < deadline:
            stop.wait(.2)
        stop.set()
        worker.join(timeout=min(args.timeout, 15))
        if worker.is_alive():
            raise RuntimeError("Production call loop did not release its owned acceptance worker")
        if errors:
            raise errors[0]
        stage = "fresh_oracles"
        observer.stop_sampling()
        fresh = observer.fresh().get("state", {})
        report["authoritative_final_received"] = any(event.get("type") == "final" for event in recorder.events)
        report["criteria"] = capture_checks(args.case, recorder, fresh, initial_document=initial_document)
        outcomes = outcomes_from_events(recorder.events)
        observed_owned = {record["window_id"] for record in observer.observations if record.get("state") and record.get("window_id")}
        report["criteria"]["all_inputs_in_owned_application"] = all(str(item.get("before", {}).get("scope", {}).get("window_id"))
            in desktop.window_ids | observed_owned for item in dispatched_actions(outcomes))
        if number in {88, 90}:
            report["criteria"]["actual_guarded_pointer_input"] = any(item.get("receipt", {}).get("backend") == "x11_guarded_xtest"
                and physical_pointer_action(item) for item in outcomes)
            report["criteria"]["actual_task_pointer_movement"] = actual_pointer_movement(root / "pointer-events.jsonl", outcomes)
        report["turns"] = [{"transcript": result.transcript, "error": result.error} for result in results]
        if any(result.error for result in results):
            raise RuntimeError("Production call turn returned an actual error")
    except Exception:
        report["failure_stage"] = stage
        raise
    finally:
        stop.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=5)
            report["call_worker_released"] = not worker.is_alive()
        else:
            report["call_worker_released"] = True
        if observer:
            observer.close()
        if pointer:
            pointer.close()
        if recorder:
            report["capture"] = recorder.close()
            report["capture_provenance_verified"] = recorder.frames > 0 and bool(recorder.vad)
            report["source"] = recorder.source
            report["physical_microphone"] = recorder.source == "physical_microphone"
            (root / "speech-evidence.json").write_text(json.dumps({"source": recorder.source, "model": args.model,
                "think": args.think, "typed_goal_substituted": False,
                "submitted_turns": recorder.turns, "asr_transcripts": [result.transcript for result in results]}, indent=2) + "\n")
        (root / "fresh-ui.json").write_text(json.dumps(fresh, indent=2) + "\n")
        report["cleanup"] = desktop.close()
        report["owned_window_closed"] = all(item.get("closed") is True for item in report["cleanup"])
        report["pulse_scope_released"] = pulse.close() if pulse else True
        report.setdefault("criteria", {})["call_worker_released"] = report["call_worker_released"]


def run(args: argparse.Namespace) -> int:
    spec = case_spec(args.case)
    if type(args.think) is not bool or not isinstance(args.timeout, (int, float)) or not 0 < args.timeout <= 1800:
        raise ValueError("A native think boolean and bounded positive timeout are required")
    root = (args.output or REPO / "runtime-data" / ("extended-voice-" + secrets.token_hex(6))).resolve()
    if not root.is_relative_to(REPO / "runtime-data"):
        raise ValueError("Acceptance evidence must remain under runtime-data")
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema": "robit.omni.extended-desktop.v1", "case_id": spec["id"], "model": args.model,
        "think": args.think, "passed": False, "source": "physical_microphone" if args.case == "CU-087" else "generated_tts_audio",
        "physical_microphone": args.case == "CU-087", "typed_goal_substituted": False,
        "authoritative_final_received": False, "owned_window_closed": False, "criteria": {}}
    started = time.monotonic()
    try:
        if args.case in {"CU-086", "CU-087", "CU-088", "CU-090"}:
            run_capture(args, root, report)
        else:
            raise CapabilityBlocked("This case requires its explicitly controlled native setup contract")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "detail": str(error)[:1000]}
        report["status"] = "blocked_capability" if isinstance(error, CapabilityBlocked) else "failed"
        report.setdefault("failure_stage", "owned_setup")
    report["criteria"]["owned_cleanup_verified"] = report["owned_window_closed"] and report.get("pulse_scope_released", True)
    report["passed"] = "error" not in report and bool(report["criteria"]) and all(report["criteria"].values())
    report["status"] = "passed" if report["passed"] else report.get("status", "failed")
    report["elapsed_s"] = time.monotonic() - started
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case_id": args.case, "model": args.model, "passed": report["passed"],
                      "status": report["status"], "artifact_directory": str(root)}, indent=2))
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
