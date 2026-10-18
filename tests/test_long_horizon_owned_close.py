"""Frozen-case setup/event contracts with controlled OS callbacks, never GUI calls."""
import json
from types import SimpleNamespace

import pytest
from test_long_horizon_native import mock_run, proof_pack

from runtime import long_horizon_cases as cases
from runtime import long_horizon_desktop as desktop
from runtime import long_horizon_owned_close as close
from runtime import verify_long_horizon_native as runner


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


def supported(target):
    return {'identity': target, 'status': 'supported', 'input_attempted': False}


def native(target, text=''):
    scope = 'scope-' + target['window_id']
    return {'window_id': target['window_id'], 'outside_owned_foreground': False,
        'state': {'scope': {'id': scope, 'window_id': target['window_id']}, 'nodes': [{
            'id': 'field-' + target['window_id'], 'generation': '1', 'scope_id': scope,
            'role': 'textbox', 'visible': True, 'enabled': True, 'sensitive': False,
            'operations': ['select_text', 'type_text'], 'value': text,
            'text_observation': {'character_count': len(text), 'truncated': False,
                'captured_range': {'start': 0, 'end': len(text)}}}]}}


def browser_native(target, url='https://docs.python.org/3/library/pathlib.html'):
    observed = native(target)
    state = observed['state']
    scope = state['scope']['id']
    state['scope']['url'] = url
    state['nodes'].append({'id': 'document', 'generation': 'doc1', 'scope_id': scope,
        'role': 'document_web', 'visible': True, 'enabled': True, 'sensitive': False})
    state['document_evidence'] = {'status': 'observed', 'source': 'atspi.Document.DocURL',
        'url': url, 'node_id': 'document', 'generation': 'doc1', 'scope_id': scope,
        'window_id': target['window_id'], 'provider': 'atspi2', 'provider_identity_sha256': 'b' * 64}
    return observed


def observe_research(controller, cap, tmp_path, *, index=-1, observed=None):
    browser = close.identity([row for row in cap['windows'] if row['application'] == 'browser'][index])
    observed = observed or browser_native(browser)
    controller.observe(observed)
    with (tmp_path / 'milestones.jsonl').open('a') as output:
        output.write(json.dumps({'native': observed}) + '\n')
    return browser


def archive_trigger(root, observed, case, work, initial):
    with (root / 'milestones.jsonl').open('a') as output:
        output.write(json.dumps({'native': observed, 'files': cases.file_evidence(work, initial, case)}) + '\n')


def fixture(tmp_path, case_id='LHN-089'):
    case = cases.load_case(case_id)
    work = tmp_path / 'Workbench'
    initial = cases.materialize(case, work)
    setup = []
    for app, count in case['fixture']['blank_app_window_counts'].items():
        for _ in range(count):
            number = len(setup) + 10
            target = {'window_id': str(number), 'process_id': str(100 + number), 'process_start': str(500 + number)}
            setup.append({'application': app, **target, 'setup_origin': 'declared_fixture',
                          'native': native(target)})
    setup_path = dump(tmp_path / 'setup-native-observations.json', setup)
    cap = close.capability(case, setup_path, work, revision='frozen', probe=supported)
    cap_path = dump(tmp_path / 'owned-close-capability.json', cap)
    return case, work, initial, cap, cap_path


class Effects:
    def __init__(self, *, sent=True, observation='absent', error=False):
        self.sent, self.observation, self.error = sent, observation, error
        self.requests, self.reserved, self.polls = [], [], []
        self.known_windows = []
        self.now = 0.

    def request(self, target):
        self.requests.append(target)
        assert self.reserved == [target]
        if self.error:
            raise TimeoutError('Controlled lost request acknowledgement')
        return self.sent

    def observe(self, target):
        self.polls.append(target)
        windows = {row['window_id']: {'pid': row['process_id'], 'start_ticks': row['process_start']}
                   for row in self.known_windows}
        if not self.requests:
            return {'identity': target, 'inventory': {'available': True, 'windows': windows}}
        if self.observation == 'absent':
            windows.pop(target['window_id'])
        elif self.observation == 'all_absent':
            windows = {}
        elif self.observation == 'reused':
            windows[target['window_id']] = {'pid': '999', 'start_ticks': '999'}
        return {'identity': target, 'inventory': {'available': self.observation != 'unavailable', 'windows': windows}}

    def sleep(self, delay):
        self.now += delay


def event(tmp_path, case_id='LHN-089', *, effects=None, probe=supported):
    case, work, initial, cap, path = fixture(tmp_path, case_id)
    effects = effects or Effects()
    effects.known_windows = cap['windows']
    controller = close.OwnedCloseEvent(case, path, work, initial, revision='frozen',
        journal=tmp_path / 'owned-close-events.jsonl', probe=probe, request_close=effects.request,
        window_status=effects.observe, reserve_close=effects.reserved.append,
        clock=lambda: effects.now, sleep=effects.sleep)
    editor = close.identity(next(row for row in cap['windows'] if row['application'] == 'gedit'))
    return controller, case, work, initial, cap, editor, effects


@pytest.mark.parametrize('case_id', ['LHN-089', 'LHN-090'])
def test_final_readiness_needs_actual_case_bound_setup_and_protocol_capability(tmp_path, case_id):
    case, work, _, cap, path = fixture(tmp_path, case_id)
    proof = proof_pack(case, tmp_path)
    assert cases.source_readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path) == []
    assert cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path)
    assert cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path,
                           owned_close_capability=path) == []
    assert cap['event_executed'] is False and cap['input_attempted'] is False
    assert len(cap['targets']) == 2
    assert len([row for row in cap['windows'] if row['application'] == 'browser']) == (1 if case_id == 'LHN-089' else 2)
    proof_path = dump(tmp_path / 'source-readiness.json', proof)
    copied = cases.capture_source_proofs(case, proof_path, tmp_path / 'captured', revision='frozen',
                                        work=work, owned_close_capability=path)
    assert len(copied['sources']) == 2
    assert not (work / 'Delivery').exists()


@pytest.mark.parametrize('fault', ['case', 'signature', 'revision', 'workroot', 'helper', 'implementation',
    'window', 'window_count', 'probe', 'claimed_event', 'authority', 'setup_bytes', 'setup_native', 'workbench'])
def test_capability_forgery_staleness_and_solver_owned_proofs_stay_blocked(tmp_path, fault):
    case, work, _, cap, path = fixture(tmp_path)
    proof = proof_pack(case, tmp_path)
    fields = {'case': 'case_id', 'signature': 'task_signature_sha256', 'revision': 'source_revision',
              'workroot': 'work_root', 'helper': 'close_helper_sha256', 'implementation': 'implementation_sha256',
              'authority': 'authority'}
    if fault in fields:
        cap[fields[fault]] = 'changed'
    elif fault == 'window':
        cap['targets'][0]['process_start'] = '999'
    elif fault == 'window_count':
        cap['windows'].pop()
    elif fault == 'probe':
        cap['protocol_probes'][0]['status'] = 'unavailable'
    elif fault == 'claimed_event':
        cap['event_executed'] = True
    elif fault == 'setup_bytes':
        (tmp_path / cap['setup_path']).write_text('[]')
    elif fault == 'setup_native':
        setup = json.loads((tmp_path / cap['setup_path']).read_text())
        setup[0]['native']['outside_owned_foreground'] = True
        dump(tmp_path / cap['setup_path'], setup)
        cap['setup_sha256'] = cases.sha((tmp_path / cap['setup_path']).read_bytes())
    else:
        path = work / 'forged-capability.json'
    dump(path, cap)
    assert cases.readiness(case, proof, revision='frozen', work=work, evidence_root=tmp_path,
                           owned_close_capability=path)


def test_missing_or_negative_protocol_probe_cannot_be_called_a_capability(tmp_path):
    case, work, _, _, _ = fixture(tmp_path)
    for probe in (lambda target: {**supported(target), 'status': 'unavailable'},
                  lambda target: {'identity': target, 'status': 'supported', 'input_attempted': True},
                  lambda target: {'status': 'supported'}):
        with pytest.raises(ValueError):
            close.capability(case, tmp_path / 'setup-native-observations.json', work, revision='frozen', probe=probe)


@pytest.mark.parametrize('fault', ['extra_application', 'unknown_foreground'])
def test_frozen_setup_cannot_gain_extra_applications_or_unknown_native_ownership(tmp_path, fault):
    case, work, _, _, _ = fixture(tmp_path)
    setup = tmp_path / 'setup-native-observations.json'
    rows = json.loads(setup.read_text())
    if fault == 'extra_application':
        target = {'window_id': '999', 'process_id': '1999', 'process_start': '2999'}
        rows.append({'application': 'unexpected_application', **target,
                     'setup_origin': 'declared_fixture', 'native': native(target)})
    else:
        rows[0]['native'].pop('outside_owned_foreground')
    dump(setup, rows)
    with pytest.raises(ValueError):
        close.capability(case, setup, work, revision='frozen', probe=supported)


@pytest.mark.parametrize('case_id', ['LHN-089', 'LHN-090'])
def test_first_real_saved_draft_triggers_one_exact_owned_close_without_output_edits(tmp_path, case_id):
    controller, case, work, initial, cap, editor, effects = event(tmp_path, case_id)
    browser = observe_research(controller, cap, tmp_path) if case_id == 'LHN-090' else None
    # An intermediate draft qualifies: final output filename/correctness is
    # deliberately not substituted for the frozen first saved-draft trigger.
    path = work / 'Drafts' / 'first.txt'
    path.parent.mkdir()
    text = 'A partial source-bound draft, still unfinished.'
    before = dict(initial)
    assert controller.observe(native(editor, text))['status'] == 'armed'
    assert not effects.requests
    path.write_text(text)
    archive_trigger(tmp_path, native(editor, text), case, work, initial)
    receipt = controller.observe(native(editor, text))
    target = editor if case_id == 'LHN-089' else browser
    assert effects.requests == effects.reserved == [target]
    assert receipt['status'] == 'observed_closed' and receipt['trigger_verified'] is True
    assert receipt['preservation_verified'] is True and path.read_text() == text
    assert receipt['surviving_owned_window_verified'] is True
    assert all(cases.sha((work / name).read_bytes()) == digest for name, digest in before.items())
    controller.observe(native(editor, text))
    assert effects.requests == [target]
    assert receipt['input_authority'] is False and receipt['solver_progress_credit'] is False
    report = {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()}
    assert close.verified_event(case, tmp_path, report)
    assert not (work / case['required_outputs'][-1]).exists()


@pytest.mark.parametrize('condition', ['not_saved', 'input_only', 'wrong_text', 'clipped', 'outside_scope',
    'other_window', 'empty', 'duplicate_files', 'different_scope', 'foreign_node_scope', 'disabled'])
def test_observation_or_save_words_cannot_replace_independent_current_saved_draft(tmp_path, condition):
    controller, _, work, _, _, editor, effects = event(tmp_path)
    text = 'Save complete. This text itself proves nothing.'
    path = work / 'draft.txt'
    if condition != 'not_saved':
        path.write_text(text)
    observed = native(editor, text)
    if condition == 'input_only':
        path.unlink()
        observed = native(editor, (work / 'Input/source-note.txt').read_text())
    elif condition == 'wrong_text':
        path.write_text('Different file bytes')
    elif condition == 'clipped':
        observed['state']['nodes'][0]['text_observation']['truncated'] = True
    elif condition == 'outside_scope':
        observed['outside_owned_foreground'] = True
    elif condition == 'other_window':
        observed['window_id'] = '999'
    elif condition == 'empty':
        path.write_text('')
        observed = native(editor, '')
    elif condition == 'duplicate_files':
        (work / 'duplicate.txt').write_text(text)
    elif condition == 'different_scope':
        observed['state']['scope']['window_id'] = '999'
    elif condition == 'foreign_node_scope':
        observed['state']['nodes'][0]['scope_id'] = 'foreign'
    elif condition == 'disabled':
        observed['state']['nodes'][0]['enabled'] = False
    assert controller.observe(observed)['status'] == 'armed'
    assert not effects.requests and not effects.reserved


@pytest.mark.parametrize('observation,sent,error', [('present_same', True, False), ('unavailable', True, False),
    ('reused', True, False), ('absent', False, False), ('absent', True, True)])
def test_unknown_close_is_durable_terminal_and_never_replayed_or_called_verified(tmp_path, observation, sent, error):
    effects = Effects(observation=observation, sent=sent, error=error)
    controller, case, work, _, _, editor, effects = event(tmp_path, effects=effects)
    (work / 'first.txt').write_text('Saved partial draft')
    observed = native(editor, 'Saved partial draft')
    receipt = controller.observe(observed)
    assert receipt['status'] == 'unknown' and receipt['dispatch_attempted'] is True
    assert len(effects.requests) == len(effects.reserved) == 1
    records = [json.loads(line) for line in controller.journal.read_text().splitlines()]
    assert records[-2]['status'] == 'request_pending' and records[-1]['status'] == 'unknown'
    controller.observe(observed)
    assert len(effects.requests) == 1
    assert not close.verified_event(case, tmp_path,
        {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()})
    assert len(effects.polls) <= 62


@pytest.mark.parametrize('case_id', ['LHN-089', 'LHN-090'])
def test_closing_other_owned_windows_is_not_the_declared_single_window_perturbation(tmp_path, case_id):
    effects = Effects(observation='all_absent')
    controller, case, work, _, cap, editor, effects = event(tmp_path, case_id, effects=effects)
    if case_id == 'LHN-090':
        observe_research(controller, cap, tmp_path)
    (work / 'first.txt').write_text('Saved partial draft')
    observed = native(editor, 'Saved partial draft')
    receipt = controller.observe(observed)
    assert receipt['window_closed_observed'] is True
    assert receipt['surviving_owned_window_verified'] is False and receipt['status'] == 'unknown'
    controller.observe(observed)
    assert len(effects.requests) == 1
    assert not close.verified_event(case, tmp_path,
        {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()})


@pytest.mark.parametrize('fault', ['none', 'address_only', 'foreign_scope', 'wrong_generation', 'bad_provider',
    'outside', 'wrong_domain', 'path_lookalike', 'both_used'])
def test_browser_target_requires_unique_actual_declared_source_document_before_first_save(tmp_path, fault):
    controller, _, work, _, cap, editor, effects = event(tmp_path, 'LHN-090')
    browser = close.identity(cap['targets'][-1])
    observed = browser_native(browser)
    if fault == 'address_only':
        observed['state'].pop('document_evidence')
        observed['state']['nodes'][0]['value'] = 'https://docs.python.org/'
    elif fault == 'foreign_scope':
        observed['state']['document_evidence']['scope_id'] = 'foreign'
    elif fault == 'wrong_generation':
        observed['state']['document_evidence']['generation'] = 'stale'
    elif fault == 'bad_provider':
        observed['state']['document_evidence']['provider_identity_sha256'] = 'invalid'
    elif fault == 'outside':
        observed['outside_owned_foreground'] = True
    elif fault == 'wrong_domain':
        observed = browser_native(browser, 'https://docs.python.org.evil.test/3/')
    elif fault == 'path_lookalike':
        observed = browser_native(browser, 'https://git-scm.com/docs-impostor/')
    if fault != 'none':
        observe_research(controller, cap, tmp_path, observed=observed)
    if fault == 'both_used':
        observe_research(controller, cap, tmp_path, index=0)
    (work / 'partial.txt').write_text('First saved partial draft')
    receipt = controller.observe(native(editor, 'First saved partial draft'))
    assert receipt['status'] == 'blocked_before_input' and receipt['trigger_verified'] is True
    assert receipt['reason'] == 'owned_research_browser_unproven_or_ambiguous'
    assert not effects.requests and not effects.reserved
    # Later source discovery cannot move the frozen first-save trigger.
    observe_research(controller, cap, tmp_path)
    controller.observe(native(editor, 'First saved partial draft'))
    assert not effects.requests


def test_source_bound_browser_event_requires_the_archived_actual_native_observation(tmp_path):
    controller, case, work, initial, cap, editor, effects = event(tmp_path, 'LHN-090')
    browser = observe_research(controller, cap, tmp_path)
    (work / 'partial.txt').write_text('Draft')
    archive_trigger(tmp_path, native(editor, 'Draft'), case, work, initial)
    assert controller.observe(native(editor, 'Draft'))['status'] == 'observed_closed'
    assert effects.requests == [browser]
    report = {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()}
    assert close.verified_event(case, tmp_path, report)
    (tmp_path / 'milestones.jsonl').write_text(json.dumps({'native': native(editor, 'Draft')}) + '\n')
    assert not close.verified_event(case, tmp_path, report)


@pytest.mark.parametrize('fault', ['source_after_trigger', 'saved_bytes', 'survivor_after', 'before_target', 'malformed_files'])
def test_readback_evaluator_recomputes_trigger_order_saved_bytes_and_non_target_preservation(tmp_path, fault):
    controller, case, work, initial, cap, editor, _ = event(tmp_path, 'LHN-090')
    observe_research(controller, cap, tmp_path)
    (work / 'draft.txt').write_text('Saved draft')
    archive_trigger(tmp_path, native(editor, 'Saved draft'), case, work, initial)
    controller.observe(native(editor, 'Saved draft'))
    report = {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()}
    assert close.verified_event(case, tmp_path, report)
    milestones = [json.loads(row) for row in (tmp_path / 'milestones.jsonl').read_text().splitlines()]
    if fault == 'source_after_trigger':
        milestones.reverse()
    elif fault == 'saved_bytes':
        milestones[-1]['files']['files']['draft.txt']['sha256'] = '0' * 64
    elif fault == 'malformed_files':
        milestones[-1]['files'] = None
    else:
        records = [json.loads(row) for row in controller.journal.read_text().splitlines()]
        final = records[-1]
        if fault == 'survivor_after':
            final['window_observation']['inventory']['windows'].clear()
        else:
            final['window_before']['inventory']['windows'].pop(final['target']['window_id'])
        controller.journal.write_text(''.join(json.dumps(row) + '\n' for row in records))
        controller.receipt = final
        report['controlled_perturbation'] = controller.snapshot()
    (tmp_path / 'milestones.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in milestones))
    assert not close.verified_event(case, tmp_path, report)


def test_an_existing_pending_journal_cannot_be_rearmed_as_a_new_close(tmp_path):
    controller, _, _, _, _, _, effects = event(tmp_path)
    controller.journal.write_text('{"status":"request_pending"}\n')
    # Recreating the same run cannot erase evidence of a potentially sent input.
    with pytest.raises(FileExistsError):
        close.OwnedCloseEvent(controller.case, tmp_path / 'owned-close-capability.json',
            controller.work, controller.initial, revision='frozen', journal=controller.journal,
            probe=supported, request_close=effects.request, window_status=effects.observe,
            reserve_close=effects.reserved.append)
    assert not effects.requests


@pytest.mark.parametrize('fault', ['changed_input', 'lost_identity', 'symlink_output'])
def test_changed_protected_input_or_ownership_stops_before_any_close(tmp_path, fault):
    probe = (lambda target: {**supported(target), 'status': 'unavailable'}) if fault == 'lost_identity' else supported
    controller, _, work, _, _, editor, effects = event(tmp_path, probe=probe)
    draft = work / 'first.txt'
    draft.write_text('Saved draft')
    if fault == 'changed_input':
        (work / 'Input/source-note.txt').write_text('User input changed')
    elif fault == 'symlink_output':
        draft.unlink()
        (tmp_path / 'outside.txt').write_text('Saved draft')
        draft.symlink_to(tmp_path / 'outside.txt')
    if fault == 'symlink_output':
        with pytest.raises(ValueError):
            controller.observe(native(editor, 'Saved draft'))
    else:
        assert controller.observe(native(editor, 'Saved draft'))['status'] == 'blocked_before_input'
    assert not effects.requests and not effects.reserved


@pytest.mark.parametrize('fault', ['journal', 'report_boolean', 'capability_only'])
def test_actual_event_journal_cannot_be_replaced_by_readiness_or_completion_claim(tmp_path, fault):
    controller, case, work, initial, _, editor, _ = event(tmp_path)
    (work / 'first.txt').write_text('Saved draft')
    archive_trigger(tmp_path, native(editor, 'Saved draft'), case, work, initial)
    controller.observe(native(editor, 'Saved draft'))
    report = {'source_before': {'revision': 'frozen'}, 'controlled_perturbation': controller.snapshot()}
    assert close.verified_event(case, tmp_path, report)
    if fault == 'journal':
        controller.journal.write_text('{}\n')
    elif fault == 'report_boolean':
        report['controlled_perturbation']['preservation_verified'] = False
    else:
        report['controlled_perturbation'] = {'status': 'verified', 'event_executed': True}
    assert not close.verified_event(case, tmp_path, report)


@pytest.mark.parametrize('present', [True, False])
def test_cleanup_only_observes_prior_controlled_close_and_cannot_retry_unknown_request(monkeypatch, present):
    owned = desktop.TaskDesktop.__new__(desktop.TaskDesktop)
    target = {'window_id': '11', 'process_id': '111', 'process_start': '511'}
    owned._controlled_close_attempts = {tuple(target.values())}
    owned.launches = [{'application': 'gedit', 'window_class': 'unused', 'registered': True,
                      'window_id': '11', 'pid': '111', 'process_start': '511'}]
    owned._logs = []
    owned.controlled_window_status = lambda _: 'present_same' if present else 'absent'
    monkeypatch.setattr(desktop, 'request_owned_window_close', lambda *_: pytest.fail('Cleanup replayed close'))
    receipts = owned.close()
    assert receipts[0]['controlled_close_not_retried'] is True
    assert receipts[0]['closed'] is not present


def test_close_reservation_requires_positive_current_setup_registration():
    owned = desktop.TaskDesktop.__new__(desktop.TaskDesktop)
    owned.windows = [SimpleNamespace(window_id='11', pid='111', process_start='511')]
    owned._controlled_close_attempts = set()
    target = {'window_id': '11', 'process_id': '111', 'process_start': '511'}
    with pytest.raises(ValueError):
        owned.reserve_controlled_close({**target, 'process_start': '999'})
    owned.reserve_controlled_close(target)
    assert owned._controlled_close_attempts == {('11', '111', '511')}


@pytest.mark.parametrize('false_presence', [None, {'identity_available': False}, {'pid': 'different'}])
def test_malformed_or_foreign_presence_cannot_be_called_window_absence(false_presence):
    target = {'window_id': '11', 'process_id': '111', 'process_start': '511'}
    status, _ = close.window_observation(target, {'identity': target,
        'inventory': {'available': True, 'windows': {'11': false_presence}}})
    assert status == 'changed_or_unknown'


def test_remaining_twelve_materializer_gaps_are_not_removed_by_close_support(tmp_path):
    for number in range(77, 89):
        case = cases.load_case(f'LHN-{number:03}')
        assert not close.supported(case)
        assert any('not_implemented' in value for value in cases.readiness(case, None,
            revision='frozen', work=tmp_path / 'Workbench', evidence_root=tmp_path))


@pytest.mark.parametrize('case_id', ['LHN-089', 'LHN-090'])
@pytest.mark.parametrize('protocol_available', [True, False])
def test_live_runner_wiring_requires_fresh_capability_then_fires_only_after_observed_save(
        tmp_path, monkeypatch, case_id, protocol_available):
    case = cases.load_case(case_id)
    args, calls = mock_run(monkeypatch, tmp_path, case)
    owned = runner.TaskDesktop({})
    effects = Effects()
    owned.reserve_controlled_close = effects.reserved.append
    owned.controlled_window_observation = effects.observe
    monkeypatch.setattr(runner, 'probe_owned_close', lambda target, _environment:
        supported(target) if protocol_available else {**supported(target), 'status': 'unavailable'})
    monkeypatch.setattr(runner, 'request_owned_window_close',
        lambda window, pid, start, _environment: effects.request(
            {'window_id': window, 'process_id': pid, 'process_start': start}))
    observer = SimpleNamespace(task_started=SimpleNamespace(set=lambda: None),
        fresh=lambda: native({'window_id': owned.window.window_id,
            'process_id': owned.window.pid, 'process_start': owned.window.process_start}),
        start=lambda: calls.append('sampling'), close=lambda: calls.append('stop_sampling'))
    monkeypatch.setattr(runner, 'NativeEvidenceObserver', lambda *_: observer)

    def stream(_client, _base, payload, root, *_args, milestone, **_kwargs):
        calls.append('model_attempt')  # Controlled callback; no model/network request.
        cap = close.validate_capability(case, root / 'owned-close-capability.json', root / 'Workbench', revision='frozen')
        effects.known_windows = cap['windows']
        assert len(payload['portal_native_task_scope']['windows']) == sum(
            case['fixture']['blank_app_window_counts'].values())
        assert 'host_independent_evaluator' not in json.dumps(payload)
        assert cases.read_json(root / 'host-source-evidence/index.json')['sources']
        if case_id == 'LHN-090':
            milestone({'round': 1}, browser_native(close.identity(cap['targets'][-1])))
        editor = close.identity(next(row for row in cap['windows'] if row['application'] == 'gedit'))
        state = native(editor, 'Partial draft')
        milestone({'round': 2}, state)
        assert effects.requests == []
        (root / 'Workbench/intermediate.txt').write_text('Partial draft')
        milestone({'round': 3}, state)
        assert len(effects.requests) == 1
        milestone({'round': 4}, state)
        assert len(effects.requests) == 1
        events = [{'type': 'final', 'response': {'message': {'content': 'Unverified model claim'}}}]
        (root / 'events.ndjson').write_text(json.dumps(events[0]) + '\n')
        guard = SimpleNamespace(failure=None, records=[], positions=[], close=lambda: calls.append('guard_close'))
        return {'events': events, 'final': events[0]['response'], 'error': None}, guard

    monkeypatch.setattr(runner, 'stream_turn', stream)
    result = runner.run(args)
    report = cases.read_json(args.output / 'summary.json')
    if not protocol_available:
        assert result == 2 and calls.count('model_attempt') == 0 and not effects.requests
        assert report['status'] == 'blocked_precondition' and not report['passed']
    else:
        assert calls.count('model_attempt') == 1 and len(effects.requests) == 1
        assert report['controlled_perturbation']['status'] == 'observed_closed'
        assert close.verified_event(case, args.output, report)
        assert not report['method']['verified'] and not report['passed'] and result != 0
