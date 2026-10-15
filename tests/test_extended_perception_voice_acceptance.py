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


def bluez_source(**extra):
    return {"name": "selected", "index": 410, "flags": ["HARDWARE"], "monitor_of_sink": None,
        "active_port": "headset-input", "ports": [{"name": "headset-input", "availability": "available"}],
        "properties": {"device.api": "bluez5", "device.bus": "bluetooth", "factory.name": "api.bluez5.sco.source",
                       "media.class": "Audio/Source", "api.bluez5.address": "AA:BB:CC:DD:EE:FF",
                       "api.bluez5.connection": "disconnected", "api.bluez5.profile": "off"}, **extra}


def test_bluetooth_mic_uses_fresh_exact_connected_device_despite_stale_pipewire_cache(monkeypatch):
    calls = []
    def query(argv):
        calls.append(argv)
        return json.dumps([bluez_source()]) if argv[0] == "pactl" else "Device AA:BB:CC:DD:EE:FF Headset\n\tConnected: yes"
    monkeypatch.setattr(runner, "pulse_command", query)
    evidence = runner.capture_device_metadata("selected", physical=True)
    assert evidence["physical_device_confirmed"] is True
    assert evidence["bluetooth_connection"]["connected"] is True
    assert calls[-1] == ["bluetoothctl", "--timeout", "4", "info", "AA:BB:CC:DD:EE:FF"]
    assert "AA:BB:CC:DD:EE:FF" not in json.dumps(evidence)


@pytest.mark.parametrize("change", ["monitor", "virtual", "nonhardware", "unavailable_port", "wrong_factory", "wrong_class"])
def test_bluetooth_lookalikes_reject_before_any_connection_query(monkeypatch, change):
    source = bluez_source()
    if change == "monitor":
        source["monitor_of_sink"] = 2
    elif change == "virtual":
        source["properties"]["node.virtual"] = "true"
    elif change == "nonhardware":
        source["flags"] = []
    elif change == "unavailable_port":
        source["ports"][0]["availability"] = "not available"
    elif change == "wrong_factory":
        source["properties"]["factory.name"] = "api.bluez5.sco.sink"
    else:
        source["properties"]["media.class"] = "Audio/Sink"
    calls = []
    monkeypatch.setattr(runner, "pulse_command", lambda argv: calls.append(argv) or json.dumps([source]))
    with pytest.raises(runner.CapabilityBlocked):
        runner.capture_device_metadata("selected", physical=True)
    assert len(calls) == 1 and calls[0][0] == "pactl"


@pytest.mark.parametrize("reply", ["Device AA:BB:CC:DD:EE:FF Test\nConnected: no",
    "Device 11:22:33:44:55:66 Other\nConnected: yes", "Device AA:BB:CC:DD:EE:FF Test\nPaired: yes"])
def test_cached_connected_or_wrong_device_cannot_replace_live_connection_proof(monkeypatch, reply):
    monkeypatch.setattr(runner, "pulse_command", lambda argv: json.dumps([bluez_source()]) if argv[0] == "pactl" else reply)
    with pytest.raises(runner.CapabilityBlocked, match="freshly confirmed connected"):
        runner.capture_device_metadata("selected", physical=True)


def test_bluetooth_identity_is_typed_and_never_inferred_from_an_unbound_name(monkeypatch):
    monkeypatch.setattr(runner, "pulse_command", lambda _argv: pytest.fail("No query for an unbound identity"))
    with pytest.raises(runner.CapabilityBlocked, match="typed BlueZ"):
        runner._bluez_connection({"device.string": "Headset; injected"})


@pytest.mark.parametrize("rate,accepted", [(8000, False), (16000, False), (32000, True), (48000, True)])
def test_true_eight_khz_cue_requires_output_bandwidth_above_nyquist(rate, accepted):
    sink = {"properties": {"device.api": "bluez5"}, "sample_specification": f"s16le 1ch {rate}Hz"}
    if accepted:
        assert runner.cue_output_metadata(sink)["output_sample_rate_hz"] == rate
    else:
        with pytest.raises(runner.CapabilityBlocked, match="sample rate"):
            runner.cue_output_metadata(sink)


def test_unknown_or_virtual_output_cannot_admit_the_attention_cue():
    for sink in ({"properties": {"device.api": "alsa"}},
                 {"properties": {"device.api": "alsa", "node.virtual": "true"}, "sample_specification": "s16le 2ch 48000Hz"}):
        with pytest.raises(runner.CapabilityBlocked):
            runner.cue_output_metadata(sink)


def test_user_selected_headset_blip_is_actual_one_khz_without_changing_eight_khz_default(tmp_path):
    path = tmp_path/"cue.wav"
    metadata = runner.attention_cue_wav(path, frequency_hz=1000)
    with wave.open(str(path)) as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (48000, 1, 2)
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(float)/32767
    frequencies = np.fft.rfftfreq(len(samples), 1/48000)
    assert frequencies[np.argmax(np.abs(np.fft.rfft(samples)))] == 1000
    assert metadata["frequency_hz"] == 1000 and len(samples) == 3840 and samples[0] == samples[-1] == 0
    sink = {"properties": {"device.api": "bluez5"}, "sample_specification": "s16le 1ch 8000Hz"}
    assert runner.cue_output_metadata(sink, frequency_hz=1000)["frequency_hz"] == 1000
    with pytest.raises(runner.CapabilityBlocked):
        runner.cue_output_metadata(sink)  # Eight kHz still does not fit this headset.
    sink["sample_specification"] = "s16le 1ch 2000Hz"
    with pytest.raises(runner.CapabilityBlocked):
        runner.cue_output_metadata(sink, frequency_hz=1000)


@pytest.mark.parametrize("frequency", [True, 1000.0, 500, 4000, 16000])
def test_cue_frequency_rejects_unapproved_or_noninteger_values(tmp_path, frequency):
    with pytest.raises(ValueError, match="exactly"):
        runner.attention_cue_wav(tmp_path/"cue.wav", frequency_hz=frequency)
    assert not (tmp_path/"cue.wav").exists()


@pytest.mark.parametrize("case_id", ["CU-088", "CU-090"])
@pytest.mark.parametrize("substitution", ["set_value", "type_text", "key_chord", "invoke"])
def test_acoustic_button_method_rejects_incidental_click_masking_another_input(tmp_path, case_id, substitution):
    recorder = runner.CaptureRecorder(tmp_path, source="pulse_capture" if case_id == "CU-090" else "physical_microphone", device="unit-source")
    click = {"action": {"id": "click", "verb": "click"},
             "receipt": {"status": "dispatched", "backend": "x11_guarded_xtest"}}
    other = {"action": {"id": "substitution", "verb": substitution},
             "receipt": {"status": "dispatched", "backend": "x11_guarded_xtest"}}
    recorder.events = [event(click)]
    fresh = {"nodes": [{"role": "textbox", "visible": True, "value": "42" if case_id == "CU-090" else "15"}]}
    baseline = runner.capture_checks(case_id, recorder, fresh)
    assert baseline["all_mutating_task_inputs_guarded_pointer"] is True
    recorder.events = [event(other), event(click)]
    actual = runner.capture_checks(case_id, recorder, fresh)
    assert actual["exact_fresh_visible_result"] is True
    assert actual["all_mutating_task_inputs_guarded_pointer"] is False
    if substitution != "invoke":
        assert actual["no_keyboard_or_value_arithmetic_substitution"] is False
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
