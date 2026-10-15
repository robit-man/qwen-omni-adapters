from __future__ import annotations

import hashlib
import io
import json
import wave

import numpy as np
import pytest

from runtime import verify_extended_perception_voice as runner


def wav(samples):
    raw = io.BytesIO()
    with wave.open(raw, "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes((samples * 32767).astype("<i2").tobytes())
    return raw.getvalue()


@pytest.mark.parametrize("order", ["frame_first", "listening_first"])
def test_physical_ready_requires_both_real_frame_and_listening(tmp_path, order):
    recorder = runner.CaptureRecorder(tmp_path, source="physical_microphone", device="alsa_input.test",
                                      microphone_ready=True, device_metadata={"physical_device_confirmed": True})
    def frame():
        recorder("capture_frame", {"samples": np.zeros(320, dtype=np.float32), "sample_rate_hz": 16000})
    def listening():
        recorder.on_state("listening", "")
    first, second = (frame, listening) if order == "frame_first" else (listening, frame)
    first()
    assert not recorder.ready.is_set() and not (tmp_path / "microphone-ready.json").exists()
    second()
    ready = json.loads((tmp_path / "microphone-ready.json").read_text())
    assert ready["actual_frame_samples"] == 320 and ready["listening_observed"] is True
    assert ready["physical_device_confirmed"] is True
    recorder.close()


def test_whole_capture_and_submitted_turn_are_separate_hash_bound_artifacts(tmp_path):
    recorder = runner.CaptureRecorder(tmp_path, source="pulse_capture", device="owned.monitor")
    samples = np.linspace(-.2, .2, 320, dtype=np.float32)
    original = samples.copy()
    recorder("capture_frame", {"samples": samples, "sample_rate_hz": 16000})
    recorder("vad", {"event": "utterance", "active": False})
    submitted = wav(np.ones(640, dtype=np.float32) * .1)
    recorder("turn_audio", {"wav": submitted, "wav_sha256": hashlib.sha256(submitted).hexdigest(), "segments": 2, "sample_rate_hz": 16000})
    evidence = recorder.close()
    assert np.array_equal(samples, original)
    assert evidence["frames"] == 320 and evidence["submitted_turns"][0]["segments"] == 2
    assert (tmp_path / "submitted-turn-001.wav").read_bytes() == submitted
    assert evidence["wav_sha256"] != evidence["submitted_turns"][0]["wav_sha256"]
    assert evidence["vad_trace_sha256"] == hashlib.sha256((tmp_path / "vad-events.jsonl").read_bytes()).hexdigest()
    with wave.open(str(tmp_path / "capture.wav")) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()) == (16000, 1, 2, 320)
    assert not (tmp_path / "microphone-ready.json").exists()


def test_submitted_turn_hash_mismatch_is_not_silently_repaired(tmp_path):
    recorder = runner.CaptureRecorder(tmp_path, source="pulse_capture", device="owned.monitor")
    with pytest.raises(ValueError, match="transport hash"):
        recorder("turn_audio", {"wav": b"actual", "wav_sha256": "invented"})
    assert recorder.turns == []
    recorder.close()


def test_negative_capture_cannot_pass_when_any_turn_or_stream_event_was_submitted(tmp_path):
    recorder = runner.CaptureRecorder(tmp_path, source="pulse_capture", device="owned.monitor")
    recorder.frames = 16000
    recorder.vad = [{"event": "idle"}]
    recorder.rms = [{"peak": .35, "rms": .05}, {"peak": .01, "rms": .004}, {"peak": 0, "rms": 0}]
    # These are collector guard unit controls, not an actual VAD/capture run.
    assert all(runner.capture_checks("CU-086", recorder, {}).values())
    recorder.turns.append({"path": "unit.wav"})
    assert not runner.capture_checks("CU-086", recorder, {})["zero_submitted_turns"]
    recorder.turns.clear()
    recorder.events.append({"type": "observation", "transcript": "unexpected speech"})
    assert not runner.capture_checks("CU-086", recorder, {})["zero_stream_events"]
    recorder.close()


@pytest.mark.parametrize("properties,monitor,accepted", [
    ({"device.api": "alsa", "device.class": "sound"}, None, True),
    ({"device.api": "alsa"}, 4294967295, True),
    ({"device.api": "alsa"}, 3, False),
    ({"device.class": "filter"}, None, False),
    ({"device.api": "pipewire"}, None, False),
])
def test_physical_source_proof_requires_actual_hardware_metadata(monkeypatch, properties, monitor, accepted):
    monkeypatch.setattr(runner, "pulse_command", lambda _argv: json.dumps([{"name": "selected", "index": 12,
        "properties": properties, "monitor_of_sink": monitor}]))
    if accepted:
        assert runner.capture_device_metadata("selected", physical=True)["physical_device_confirmed"] is True
    else:
        with pytest.raises(runner.CapabilityBlocked):
            runner.capture_device_metadata("selected", physical=True)


def test_source_identity_cannot_be_substituted_by_another_physical_device(monkeypatch):
    monkeypatch.setattr(runner, "pulse_command", lambda _argv: json.dumps([{"name": "another", "properties": {"device.api": "alsa"}}]))
    with pytest.raises(runner.CapabilityBlocked, match="exact selected"):
        runner.capture_device_metadata("selected", physical=True)


def test_null_sink_cleanup_uses_only_its_exact_owned_module(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "pulse_command", lambda argv: calls.append(argv) or ("123" if argv[1] == "load-module" else ""))
    scope = runner.OwnedPulse().open()
    assert scope.source == scope.sink + ".monitor"
    assert scope.close() is True
    assert calls[-1] == ["pactl", "unload-module", "123"]
    assert not any("set-default" in " ".join(argv) for argv in calls)


def test_physical_keyboard_oracle_rejects_accessibility_setter_and_unbound_text_fields(tmp_path):
    recorder = runner.CaptureRecorder(tmp_path, source="physical_microphone", device="alsa_input.test")
    binding = {"id": "document", "generation": "g1", "scope_id": "scope", "window_id": "123"}
    document = {"id": "document", "generation": "g1", "scope_id": "scope", "role": "textbox", "visible": True,
                "value": "copper finch ready"}
    fresh = {"scope": {"id": "scope", "window_id": "123"}, "nodes": [document]}
    def event(verb):
        return {"type": "tool", "tools": [{"name": "native_ui", "result": json.dumps({"status": "verified",
            "action": {"id": "action", "verb": verb}, "receipt": {"status": "dispatched"}})}]}
    recorder.events = [event("set_value")]
    checks = runner.capture_checks("CU-087", recorder, fresh, initial_document=binding)
    assert checks["actual_editor_keyboard_dispatch"] is False
    recorder.events = [event("type_text")]
    assert runner.capture_checks("CU-087", recorder, fresh, initial_document=binding)["actual_editor_keyboard_dispatch"] is True
    assert runner.capture_checks("CU-087", recorder, fresh, initial_document=binding)["exact_fresh_visible_result"] is True
    for key, bad in (("id", "find-field"), ("generation", "g2"), ("scope_id", "another-scope")):
        changed = {"scope": fresh["scope"], "nodes": [{**document, key: bad}]}
        assert runner.capture_checks("CU-087", recorder, changed, initial_document=binding)["exact_fresh_visible_result"] is False
    changed = {"scope": {"id": "scope", "window_id": "another-window"}, "nodes": [document]}
    assert runner.capture_checks("CU-087", recorder, changed, initial_document=binding)["exact_fresh_visible_result"] is False
    assert runner.capture_checks("CU-087", recorder, fresh)["exact_fresh_visible_result"] is False
    recorder.close()


def test_attention_blip_is_actual_eight_khz_at_forty_eight_khz_with_modest_gain(tmp_path):
    path = tmp_path / "cue.wav"
    evidence = runner.attention_cue_wav(path)
    with wave.open(str(path)) as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (48000, 1, 2)
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(float)/32767
    dominant = np.fft.rfftfreq(len(samples), 1/48000)[np.argmax(np.abs(np.fft.rfft(samples)))]
    assert dominant == 8000 and .04 < np.max(np.abs(samples)) <= .05
    assert len(samples) == 3840 and samples[0] == 0 and samples[-1] == 0
    assert evidence["wav_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_attention_cue_cannot_play_before_real_readiness(tmp_path):
    recorder = runner.CaptureRecorder(tmp_path, source="physical_microphone", device="alsa_input.test", microphone_ready=True)
    with pytest.raises(RuntimeError, match="actual microphone frames"):
        runner.play_attention_cue(tmp_path, recorder, None)
    assert not (tmp_path / "attention-cue.wav").exists()
    recorder.close()
