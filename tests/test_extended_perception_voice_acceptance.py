from __future__ import annotations

import copy
import hashlib
import io
import json
import os
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


def node(identity='doc', value='copper finch ready', **extra):
    return {'id': identity, 'generation': 'g1', 'scope_id': 'scope', 'role': 'textbox', 'visible': True,
            'enabled': True, 'sensitive': False, 'value': value, 'operations': ['set_value'], **extra}


def state(nodes):
    return {'revision': 'r1', 'scope': {'id': 'scope', 'window_id': '100'}, 'nodes': nodes}


def binding():
    return {'id': 'doc', 'generation': 'g1', 'scope_id': 'scope', 'window_id': '100'}


def distribution():
    candidates = [{'id': 'a', 'kind': 'action'}, {'id': 'b', 'kind': 'action'},
                  {'id': 'observe', 'kind': 'observe'}, {'id': 'abstain', 'kind': 'abstain'}]
    receipt = {'status': 'accepted', 'mode': 'active', 'model_confirmed': True, 'probability_kind': 'native',
               'model': 'jaredpalmer/kev-4b@pinned', 'candidate_ids': [item['id'] for item in candidates],
               'policy_choice': 'a', 'selected_action_id': 'abstain', 'state_revision': 'r1',
               'probabilities': {'a': .6, 'b': .2, 'observe': .1, 'abstain': .1}}
    return {'state': {'revision': 'r1'}, 'candidates': candidates, 'policy_receipt': receipt}


def event(body):
    return {'type': 'tool', 'tools': [{'name': 'native_ui', 'result': json.dumps(body)}]}


def action(verb, *, name='', args=None):
    return event({'action': {'id': verb+name, 'node_id': 'target', 'verb': verb, 'args': args or {}},
                  'before': {'nodes': [{'id': 'target', 'name': name}]}, 'receipt': {'status': 'dispatched'}})


def test_negative_policy_requires_actual_active_bound_finite_distribution():
    body = distribution()
    assert len(runner.policy_abstentions([event(body)], low_confidence=True)) == 1
    assert not runner.policy_abstentions([event({'selected_action_id': 'abstain'})], low_confidence=False)
    for mutate in (
        lambda x: x['policy_receipt'].update(mode='shadow'),
        lambda x: x['policy_receipt'].update(model_confirmed=False),
        lambda x: x['policy_receipt'].update(probability_kind='synthetic'),
        lambda x: x['policy_receipt'].update(state_revision='old'),
        lambda x: x['policy_receipt'].update(selected_action_id='a'),
        lambda x: x['policy_receipt']['probabilities'].update(a=.99),
        lambda x: x['candidates'].pop(),
    ):
        bad = copy.deepcopy(body)
        mutate(bad)
        assert not runner.policy_abstentions([event(bad)], low_confidence=True)


def test_confident_abstention_does_not_prove_low_confidence_gate():
    body = distribution()
    body['policy_receipt'].update(policy_choice='abstain', probabilities={'a': .01, 'b': .01, 'observe': .01, 'abstain': .97})
    assert runner.policy_abstentions([event(body)], low_confidence=False)
    assert not runner.policy_abstentions([event(body)], low_confidence=True)


def test_positional_target_is_independently_geometry_derived_not_digit_answer():
    digits = [{'id': f'n{i}', 'name': str(i), 'role': 'button', 'visible': True,
               'bounds': {'x': (i%3)*20, 'y': (i//3)*20, 'width': 18, 'height': 18}} for i in range(10)]
    assert runner.positional_digit(state(digits))['name'] == '0'
    # Swapping positions changes the oracle, rather than hard-coding seven.
    digits[0]['bounds'], digits[7]['bounds'] = digits[7]['bounds'], digits[0]['bounds']
    assert runner.positional_digit(state(digits))['name'] == '7'
    digits[1]['bounds'] = digits[2]['bounds']
    assert runner.positional_digit(state(digits)) == {}


@pytest.mark.parametrize('key,value', [('id', 'find'), ('generation', 'g2'), ('scope_id', 'other')])
def test_spoken_text_oracle_rejects_unbound_document(key, value):
    context = {'document_binding': binding()}
    fresh = state([node(**{key: value})]) if key != 'id' else state([node(identity=value)])
    checks = runner.native_checks('CU-081', context, [action('type_text')], fresh, [], [])
    assert checks['exact_bound_document'] is False


def test_spoken_correction_requires_real_first_turn_state_and_final_same_document():
    context = {'document_binding': binding()}
    good = state([node(value='copper robin ready')])
    checks = runner.native_checks('CU-084', context, [action('type_text')], good,
                                 [state([node()]), good], [])
    assert all(checks.values())
    assert not runner.native_checks('CU-084', context, [action('type_text')], good,
                                   [state([node(value='unrelated')]), good], [])['first_turn_exact_intermediate']
    assert not runner.native_checks('CU-084', context, [action('set_value')], good,
                                   [state([node()]), good], [])['actual_keyboard_input']


def test_negated_save_requires_exact_selection_hash_mtime_and_no_mutation(tmp_path):
    path = tmp_path/'source.txt'
    path.write_text('alpha copper ready')
    context = {'document_binding': binding(), 'source': path, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'source_mtime_ns': path.stat().st_mtime_ns, 'initial_text': path.read_text()}
    fresh = state([node(value=context['initial_text'], text_selection={'start': 0, 'end': len(context['initial_text'])})])
    assert all(runner.native_checks('CU-083', context, [action('key_chord', args={'keys': ['Control_L', 'a']})], fresh, [], []).values())
    assert not runner.native_checks('CU-083', context, [action('key_chord', args={'keys': ['Control_L', 's']})], fresh, [], [])['zero_typing_or_save']
    os.utime(path, ns=(context['source_mtime_ns']+1000000, context['source_mtime_ns']+1000000))  # Same bytes cannot conceal a changed timestamp.
    assert not runner.native_checks('CU-083', context, [], fresh, [], [])['file_hash_and_mtime_unchanged']


def test_referent_move_requires_exact_initial_selection_and_whole_manifest():
    before = [{'path': 'Destination', 'kind': 'directory'}, {'path': 'alpha.txt', 'kind': 'file', 'sha256': 'alpha'},
              {'path': 'beta.txt', 'kind': 'file', 'sha256': 'beta'}]
    context = {'manifest_before': before}
    selected = state([{'id': 'list', 'visible': True, 'selected': ['alpha']},
                      {'id': 'alpha', 'visible': True, 'name': 'alpha.txt'}])
    after = [before[0], before[2], {**before[1], 'path': 'Destination/alpha.txt'}]
    checks = runner.native_checks('CU-085', context, [action('key_chord', args={'keys': ['Control_L', 'v']})], {}, [selected, {}], after)
    assert all(checks.values())
    assert not runner.native_checks('CU-085', context, [], {}, [{}, {}], after)['first_turn_exact_selection']
    after[1] = {**after[1], 'sha256': 'wrong-file-mutated'}
    assert not runner.native_checks('CU-085', context, [action('key_chord', args={'keys': ['Control_L', 'v']})], {}, [selected, {}], after)['exact_selected_file_move']


def test_all_perception_cases_resolve_and_missing_fault_barriers_remain_blocked():
    assert len(runner.CASE_IDS) == 20
    assert all(runner.case_spec(case)['id'] == case for case in runner.CASE_IDS)
    assert {'CU-073', 'CU-074', 'CU-077', 'CU-080', 'CU-089'} == set(runner.BLOCKED_CASES)


def test_independent_terminal_oracle_can_use_real_unresolved_frame_bound_input_without_rewriting_it():
    frame = 'a'*64
    point = {'model': runner.POINT_MODEL, 'revision': runner.POINT_REVISION, 'candidate_id': 'visual:target', 'frame_sha256': frame}
    outcome = {'status': 'unresolved', 'action': {'id': 'a', 'node_id': 'visual:target', 'verb': 'click', 'generation': frame},
               'receipt': {'status': 'dispatched', 'backend': 'x11_guarded_xtest'}, 'point_receipt': point}
    assert runner.grounding_receipts([outcome]) == [point]
    assert outcome['status'] == 'unresolved'
    for field, bad in [('frame_sha256', 'b'*64), ('revision', 'wrong'), ('candidate_id', 'other')]:
        broken = copy.deepcopy(outcome)
        broken['point_receipt'][field] = bad
        assert runner.grounding_receipts([broken]) == []
    outcome['receipt']['status'] = 'rejected'
    assert runner.grounding_receipts([outcome]) == []
