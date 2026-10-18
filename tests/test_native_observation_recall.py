"""Historical native text survives switches without becoming current authority."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import packet, reply
from test_native_working_state import envelope
from test_ui_task import Fixture, completed_tools, final

from portal import ui_decision
from portal.app import _model_ui_observation
from portal.ui_observations import NativeObservationArchive, _encoded
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError


def observed(text='Exact condition: FileNotFoundError.', *, scope='browser', revision='r1', session='owned'):
    raw = {'session_id': session, 'scope': {'kind': 'desktop', 'id': scope}, 'revision': revision,
        'truncated': True, 'nodes': [
            {'id': scope, 'scope_id': scope, 'role': 'window', 'name': 'Documentation', 'generation': '1',
             'visible': True, 'enabled': True, 'sensitive': False, 'operations': ['key_chord']},
            {'id': scope + '-paragraph', 'scope_id': scope, 'role': 'paragraph', 'name': text, 'generation': '1',
             'visible': True, 'enabled': True, 'sensitive': False, 'operations': []}],
        'document_evidence': {'status': 'observed', 'source': 'atspi.Document.DocURL',
            'scope_id': scope, 'url': 'https://docs.example/version/guide', 'node_id': 'document', 'generation': '1'}}
    return {'state': _model_ui_observation(raw)}


def test_exact_source_retained_across_switch_and_recall_is_not_current_input():
    archive = NativeObservationArchive('task')
    source = observed()
    original = copy.deepcopy(source)
    archive.remember(1, source)
    reference = archive.view()['entries'][0]['reference']
    archive.remember(2, observed('Empty editor', scope='editor', revision='r2'))
    result = archive.recall(reference)
    body = result['recalled_observation']['observation']
    assert body['native_content_excerpt'] == original['state']['native_content_excerpt']
    assert body['source_revision'] == 'r1' and body['source_scope_id'] == 'browser'
    assert body['document_evidence']['url'] == 'https://docs.example/version/guide'
    assert result['input_attempted'] is False and result['task_progress'] is False
    assert result['observation_recall_receipt']['current_ui_verified'] is False
    assert result['observation_recall_receipt']['input_authority'] is False
    assert result['evidence_authority'] == 'untrusted_historical_native_observation'
    source['state']['native_content_excerpt']['text'] = 'Changed after capture'
    result['recalled_observation']['observation']['native_content_excerpt']['text'] = 'Changed by consumer'
    assert archive.recall(reference)['recalled_observation']['observation'] == body | {
        'native_content_excerpt': original['state']['native_content_excerpt']}
    assert archive.recall(reference)['observation_recall_receipt']['first_expansion_in_task'] is False


@pytest.mark.parametrize('fault', ['unprojected', 'stale_projection', 'stale_excerpt', 'wrong_authority', 'oversized'])
def test_unbound_or_oversized_observations_are_not_archived(fault):
    source = observed()
    state = source['state']
    if fault == 'unprojected':
        del state['model_projection']
    elif fault == 'stale_projection':
        state['model_projection']['source_revision'] = 'stale'
    elif fault == 'stale_excerpt':
        state['native_content_excerpt']['source_revision'] = 'stale'
    elif fault == 'wrong_authority':
        state['native_content_excerpt']['input_authority'] = True
    else:
        state['native_content_excerpt']['text'] = 'x' * 4001
    archive = NativeObservationArchive('task')
    archive.remember(1, source)
    assert not archive.view()['entries']


def test_cross_task_and_cross_session_recall_cannot_resolve():
    first, second = NativeObservationArchive('first'), NativeObservationArchive('second')
    first.remember(1, observed())
    second.remember(1, observed())
    ref = first.view()['entries'][0]['reference']
    with pytest.raises(ComputerUseError):
        second.recall(ref)
    with pytest.raises(ComputerUseError):
        first.remember(2, observed(session='other'))
    assert len(first.records) == 1


def test_repeated_excerpt_deduplicates_without_rebinding_original_source():
    archive = NativeObservationArchive('task')
    archive.remember(1, observed())
    ref = archive.view()['entries'][0]['reference']
    archive.remember(12, observed(revision='new-capture'))
    assert len(archive.records) == 1
    assert archive.records[ref]['last_observed_phase'] == 12
    assert archive.records[ref]['observation']['source_revision'] == 'r1'
    assert archive.bytes == sum(len(_encoded(r)) for r in archive.records.values())


def test_record_and_byte_eviction_remove_expansion_and_account_exactly():
    archive = NativeObservationArchive('task')
    archive.MAX_RECORDS = 2
    archive.remember(1, observed('first'))
    first = archive.view()['entries'][0]['reference']
    archive.recall(first)
    for phase in range(2, 12):
        archive.remember(phase, observed(f'Content {phase}'))
    assert len(archive.records) == 2 and first not in archive.expanded
    with pytest.raises(ComputerUseError):
        archive.recall(first)
    archive.MAX_BYTES = archive.bytes - 1
    archive.remember(20, observed('Content 11'))
    assert len(archive.records) == 1 and archive.bytes <= archive.MAX_BYTES
    assert archive.bytes == sum(len(_encoded(r)) for r in archive.records.values())
    assert archive.view()['evicted_count'] == 10


def test_foreign_document_identity_never_becomes_source_url():
    source = observed()
    source['state']['document_evidence']['scope_id'] = 'other-window'
    archive = NativeObservationArchive('task')
    archive.remember(1, source)
    result = archive.recall(archive.view()['entries'][0]['reference'])
    assert result['recalled_observation']['observation']['document_evidence'] == {'status': 'unavailable'}


def test_recall_is_offered_in_both_contracts_with_only_current_task_references():
    p = packet(memory=False)
    state = _NativeWorkingState(p['goal'])
    state.observations.remember(1, observed())
    p['working_state'] = state.view(2, 0)
    p['response_schema'] = _working_phase_schema(['native_ui'],
        _grounded_phase_schema(['native_ui'], p['current_state'], intent_choice=True), p['working_state'])
    compact, binding = ui_decision.project(p)
    ref = state.observations.view()['entries'][0]['reference']
    decision = reply(p, {'family': 'recall_observation', 'reference': ref})
    expanded = ui_decision.expand(decision, p, binding)
    assert expanded['phase']['kind'] == 'recall_observation'
    assert Draft202012Validator(p['response_schema']).is_valid(expanded)
    assert compact['observation'].get('native_content_excerpt') is None
    decision['action']['reference'] = 'foreign-reference'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, p, binding)


def test_recall_runs_privately_without_input_and_repeating_it_exhausts_budget():
    fixture = Fixture([])
    observations = iter([observed(), observed('Editor', scope='editor'), observed('Editor', scope='editor'),
                         observed('Editor', scope='editor'), observed('Editor', scope='editor')])
    fixture.observe = lambda: next(observations)
    def compile_phase(p):
        fixture.packets.append(copy.deepcopy(p))
        return envelope(p, {'kind': 'recall_observation',
            'reference': p['working_state']['observations']['entries'][0]['reference']})
    fixture.compile = compile_phase
    events = list(fixture.run(working_state=True))
    assert not fixture.executions and not fixture.adapter.calls
    # The planner already received these exact retained values. Recalling one
    # again cannot manufacture novel acquisition or renew the failure budget.
    assert [t['ok'] for t in completed_tools(events)] == [False, False, False]
    assert final(events)['portal']['native_task']['status'] == 'blocked'
    assert final(events)['portal']['native_task']['task_verified'] is False
    second = fixture.packets[1]
    assert second['current_state']['state']['scope']['id'] == 'editor'
    assert second['last_result']['recalled_observation']['observation']['source_scope_id'] == 'browser'
    assert all(row[2] == 0 for row in final(events)['portal']['native_task']['working_state']['phase_refs'])


def test_full_archive_and_forty_phase_state_fit_existing_working_budget():
    state = _NativeWorkingState('Unchanged original task')
    for phase in range(1, 41):
        source = observed('😀' * 3990 + str(phase), revision=str(phase))
        source['state']['document_evidence']['url'] = 'https://example/' + '😀' * 1000 + str(phase)
        state.observations.remember(phase, source)
        state.record(phase, 'recall_observation', {'status': 'historical_observation',
            'input_attempted': False, 'dispatch_status': 'not_dispatched'}, recover=True)
    view = state.view(40, 2)
    assert len(json.dumps(view, ensure_ascii=False).encode()) <= 12288
    assert 0 < len(view['observations']['entries']) <= 16
    assert state.observations.bytes <= state.observations.MAX_BYTES
