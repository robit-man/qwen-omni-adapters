"""Combined fixture wiring; CPU assets are real, GUI/model callbacks controlled."""
import json
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest
from test_long_horizon_amendments import REVISION, WINDOW, probe, source_pack, workbook, write_setup
from test_long_horizon_native import mock_run, proof_pack

from runtime import acceptance_native_support as support
from runtime import long_horizon_amendments as amendments
from runtime import long_horizon_cases as cases
from runtime import long_horizon_media as media
from runtime import verify_long_horizon_native as runner


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('CPU decoders unavailable; live media readiness remains blocked')
    case = cases.load_case('LHN-083')
    cache = tmp_path_factory.mktemp('integrated-owned-media') / 'cache'
    return case, cache, media.build_media_cache(case, cache)


@pytest.mark.parametrize('kind', ['owned', 'foreign', 'provider_error', 'identity_error'])
def test_actual_native_observer_emits_owned_false_only_after_owned_provider_success(tmp_path, monkeypatch, kind):
    calls = []
    identity = {'window_id': '123', 'process_id': 456}
    def foreground(_environment):
        calls.append('foreground')
        if kind == 'identity_error':
            raise RuntimeError('No current foreground identity')
        return identity
    def is_owned(current):
        assert current == identity
        calls.append('is_owned')
        return kind != 'foreign'
    def observe():
        calls.append('observe')
        if kind == 'provider_error':
            raise RuntimeError('No fresh native observation')
        return {'scope': {'id': 'native', 'window_id': '123'}, 'nodes': [{'id': 'observed'}]}
    monkeypatch.setattr(support, '_x11_foreground', foreground)
    monkeypatch.setattr(support, 'NativeUiAdapter', lambda *_args, **_kwargs: SimpleNamespace(observe=observe))
    owned = SimpleNamespace(environment={}, metadata=lambda: [], is_owned=is_owned)
    result = support.NativeEvidenceObserver(tmp_path, owned).fresh()
    if kind == 'owned':
        assert result['outside_owned_foreground'] is False and result['state']['scope']['window_id'] == '123'
        assert calls == ['foreground', 'is_owned', 'observe']
    elif kind == 'foreign':
        assert result['outside_owned_foreground'] is True and 'state' not in result
        assert calls == ['foreground', 'is_owned']
    else:
        assert 'outside_owned_foreground' not in result and 'state' not in result and result['error']


@pytest.mark.parametrize('number', range(77, 91))
def test_all_fourteen_cases_still_reject_missing_actual_fixture_capabilities(tmp_path, number):
    case = cases.load_case(f'LHN-{number:03}')
    work = tmp_path / 'Workbench'
    cases.materialize(case, work)
    proof = proof_pack(case, tmp_path)
    assert cases.source_readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path) == []
    blockers = cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path)
    assert blockers and any('not_implemented' in value for value in blockers)
    assert not any((work / name).exists() for name in case['required_outputs'])
    payload = (case.get('controlled_perturbation') or {}).get('payload')
    if payload:
        assert not (work / payload['path']).exists()


@pytest.mark.parametrize('number', range(77, 83))
def test_each_calendar_capability_clears_only_its_bound_readiness_without_releasing(tmp_path, number):
    case = cases.load_case(f'LHN-{number:03}')
    work = tmp_path / 'Workbench'
    initial = cases.materialize(case, work)
    write_setup(case, work)
    capability = amendments.prepare_capability(case, work=work, revision=REVISION,
        initial_hashes=initial, calc_window=WINDOW, document_probe=probe())
    path = tmp_path / 'amendment-capability.json'
    path.write_text(json.dumps(capability))
    proof = source_pack(case, tmp_path)
    options = {'revision': REVISION, 'work': work, 'evidence_root': tmp_path}
    assert cases.readiness(case, proof, amendment_capability=path, **options) == []
    assert not (work / amendments.RELEASE_PATH).exists()
    assert not any((work / name).exists() for name in case['required_outputs'])
    wrong = {**capability, 'case_id': 'LHN-089'}
    path.write_text(json.dumps(wrong))
    assert cases.readiness(case, proof, amendment_capability=path, **options) == ['amendment_capability_unverified']


@pytest.mark.parametrize('number', range(83, 89))
def test_all_media_cases_prepare_exact_inputs_and_capture_sources_with_real_decoder_proof(generated, tmp_path, number):
    _, cache, receipt = generated
    case = cases.load_case(f'LHN-{number:03}')
    work = tmp_path / 'Workbench'
    media_options = {'owned_media_cache': cache, 'owned_media_manifest_sha256': receipt['cache_manifest_sha256']}
    initial = cases.materialize(case, work, **media_options)
    assert len(initial) == 9 and all(initial[p] == h for p, h in receipt['asset_hashes'].items())
    proof = proof_pack(case, tmp_path)
    assert cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path, **media_options) == []
    proof_path = tmp_path / 'readiness.json'
    proof_path.write_text(json.dumps(proof))
    captured = cases.capture_source_proofs(case, proof_path, tmp_path / 'captured',
        revision='frozen', work=work, **media_options)
    assert {row['id'] for row in captured['sources']} == cases.readiness_requirements(case)
    assert {p.relative_to(work).as_posix() for p in work.rglob('*') if p.is_file()} == set(initial)
    assert not any((work / name).exists() for name in case['required_outputs'])
    assert json.loads((work / 'Input/clip-manifest.json').read_text())[0]['materialized'] is False
    capability = cases.verified_media_inputs(case, work, cache, receipt['cache_manifest_sha256'])
    assert capability['materialized_inputs_verified'] is True and capability['task_progress'] is False


@pytest.mark.parametrize('missing', ['pin', 'cache', 'wrong_case'])
def test_unpaired_or_undeclared_media_options_fail_before_workbench_creation(tmp_path, missing):
    case = cases.load_case('LHN-047' if missing == 'wrong_case' else 'LHN-083')
    options = {'owned_media_cache': tmp_path / 'cache', 'owned_media_manifest_sha256': 'a' * 64}
    if missing == 'pin':
        options.pop('owned_media_manifest_sha256')
    elif missing == 'cache':
        options.pop('owned_media_cache')
    work = tmp_path / 'Workbench'
    with pytest.raises(ValueError):
        cases.materialize(case, work, **options)
    assert not work.exists()


@pytest.mark.parametrize('fault', ['wrong_pin', 'changed_copy', 'changed_inline', 'decoder_timeout'])
def test_media_readiness_never_substitutes_manifest_claims_for_exact_readback(generated, tmp_path, monkeypatch, fault):
    _, cache, receipt = generated
    case = cases.load_case('LHN-083')
    work = tmp_path / 'Workbench'
    options = {'owned_media_cache': cache, 'owned_media_manifest_sha256': receipt['cache_manifest_sha256']}
    cases.materialize(case, work, **options)
    proof = proof_pack(case, tmp_path)
    if fault == 'wrong_pin':
        options['owned_media_manifest_sha256'] = '0' * 64
    elif fault in {'changed_copy', 'changed_inline'}:
        target = work / ('Input/Clips/clip-1.mp4' if fault == 'changed_copy' else 'Input/clip-manifest.json')
        target.chmod(0o600)
        target.write_bytes(b'Changed independently of the frozen cache')
    else:
        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired('bounded CPU decoder', 1)
        monkeypatch.setattr(media, 'verify_media_cache', timeout)
    assert cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path,
                           **options) == ['owned_media_capability_unverified']
    assert not any((work / name).exists() for name in case['required_outputs'])


@pytest.mark.parametrize('protocol_available', [True, False])
def test_combined_runner_calendar_capability_and_original_first_save_release_are_wired(
        tmp_path, monkeypatch, protocol_available):
    case = cases.load_case('LHN-077')
    args, calls = mock_run(monkeypatch, tmp_path, case)
    monkeypatch.setattr(runner, 'source_state', lambda: {'revision': REVISION})
    args.readiness_file.write_text(json.dumps(source_pack(case, args.readiness_file.parent)))
    owned = runner.TaskDesktop({})
    started, url = threading.Event(), ['']
    observer = SimpleNamespace(task_started=started, start=lambda: calls.append('sampling'),
        close=lambda: calls.append('stop_sampling'), fresh=lambda: {
            'phase': 'model_task' if started.is_set() else 'setup', 'outside_owned_foreground': False,
            'window_id': owned.window.window_id,
            'state': {'scope': {'window_id': owned.window.window_id}, 'nodes': [{'id': 'native'}]}})
    monkeypatch.setattr(runner, 'NativeEvidenceObserver', lambda *_: observer)
    def document_probe(window, _environment):
        return {**probe(url[0]), 'window': window,
                **({} if protocol_available else {'docurl_query_supported': False})}
    monkeypatch.setattr(runner, 'probe_calendar_document', document_probe)
    monkeypatch.setattr(amendments, 'probe_calendar_document', document_probe)
    def stream(_client, _base, payload, root, *_args, milestone, **_kwargs):
        calls.append('model_attempt')  # Controlled callback; no model or GUI request.
        capability = cases.read_json(root / 'amendment-capability.json')
        assert payload['messages'][0]['content'] == case['goal_text'].replace('WORKBENCH', str(root / 'Workbench'))
        assert not (root / 'Workbench' / amendments.RELEASE_PATH).exists()
        native = {'phase': 'model_task', 'outside_owned_foreground': False,
            'window_id': capability['calc_window']['window_id'],
            'state': {'scope': {'window_id': capability['calc_window']['window_id']}}}
        milestone({'round': 1}, native)
        assert not (root / 'Workbench' / amendments.RELEASE_PATH).exists()
        saved = root / 'Workbench' / 'first-not-final.ods'
        saved.write_bytes(workbook('Initial incomplete schedule'))
        url[0] = saved.as_uri()
        milestone({'round': 2}, native)
        release = (root / 'Workbench' / amendments.RELEASE_PATH).read_bytes()
        assert json.loads(release) == case['controlled_perturbation']['payload']['content']
        milestone({'round': 3}, native)
        assert (root / 'Workbench' / amendments.RELEASE_PATH).read_bytes() == release
        events = [{'type': 'final', 'response': {'message': {'content': 'No independent task proof'}}}]
        (root / 'events.ndjson').write_text(json.dumps(events[0]) + '\n')
        guard = SimpleNamespace(failure=None, records=[], positions=[], close=lambda: calls.append('guard_close'))
        return {'events': events, 'final': events[0]['response'], 'error': None}, guard
    monkeypatch.setattr(runner, 'stream_turn', stream)
    result = runner.run(args)
    report = cases.read_json(args.output / 'summary.json')
    assert report['passed'] is False and result != 0
    if protocol_available:
        assert calls.count('model_attempt') == 1 and report['controlled_perturbation']['status'] == 'released'
        assert amendments.verified_event(case, args.output, report)
        assert report['released_input_hashes'] == {amendments.RELEASE_PATH:
            cases.sha((args.output / 'Workbench' / amendments.RELEASE_PATH).read_bytes())}
        assert amendments.RELEASE_PATH not in report['initial_input_hashes']
        assert not report['method']['verified']
    else:
        assert result == 2 and calls.count('model_attempt') == 0
        assert not (args.output / 'Workbench' / amendments.RELEASE_PATH).exists()


@pytest.mark.parametrize('correct_pin', [True, False])
def test_combined_runner_actual_media_copy_and_pin_check_precede_any_solver_callback(generated, tmp_path, monkeypatch, correct_pin):
    _, cache, receipt = generated
    case = cases.load_case('LHN-083')
    args, calls = mock_run(monkeypatch, tmp_path, case)
    args.owned_media_cache = cache
    args.owned_media_manifest_sha256 = receipt['cache_manifest_sha256'] if correct_pin else '0' * 64
    def stream(_client, _base, _payload, root, *_args, **_kwargs):
        calls.append('model_attempt')  # No real GUI, playback or model submission.
        capability = cases.read_json(root / 'owned-media-capability.json')
        assert capability['cache_manifest_sha256'] == receipt['cache_manifest_sha256']
        assert capability['materialized_inputs_verified'] and capability['task_progress'] is False
        assert len(list((root / 'Workbench/Input/Clips').glob('*.mp4'))) == 6
        assert not any((root / 'Workbench' / name).exists() for name in case['required_outputs'])
        events = [{'type': 'final', 'response': {'message': {'content': 'Not a passing result'}}}]
        (root / 'events.ndjson').write_text(json.dumps(events[0]) + '\n')
        guard = SimpleNamespace(failure=None, records=[], positions=[], close=lambda: calls.append('guard_close'))
        return {'events': events, 'final': events[0]['response'], 'error': None}, guard
    monkeypatch.setattr(runner, 'stream_turn', stream)
    result = runner.run(args)
    report = cases.read_json(args.output / 'summary.json')
    assert result != 0 and report['passed'] is False
    if correct_pin:
        assert calls.count('model_attempt') == 1 and len(report['initial_input_hashes']) == 9
        assert all(report['initial_input_hashes'][p] == h for p, h in receipt['asset_hashes'].items())
    else:
        assert calls.count('model_attempt') == 0 and result == 2
        assert not any(call.startswith('open:') for call in calls)
