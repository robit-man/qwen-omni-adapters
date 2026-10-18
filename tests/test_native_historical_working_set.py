"""Historical source data is available to planning, with no new UI authority."""
import copy
import hashlib
import json

import pytest
from test_native_compact_decision import packet, reply
from test_native_observation_recall import observed
from test_native_working_state import envelope
from test_ui_task import Fixture, final

from portal import ui_decision
from portal.ui_observations import NativeObservationArchive, _encoded
from portal.ui_task import _NativeWorkingState


def test_changed_document_body_with_unchanged_excerpt_retains_both_exact_versions():
    archive = NativeObservationArchive('owned-task')
    first = observed('Unchanged toolbar labels', scope='editor')
    first['state']['text'] = 'Target sentence: provisional value. Protected sentence unchanged.'
    archive.remember(1, first)
    first_ref, = archive.records
    second = observed('Unchanged toolbar labels', scope='editor', revision='new-native-capture')
    second['state']['text'] = 'Target sentence: confirmed value. Protected sentence unchanged.'
    assert first['state']['native_content_excerpt']['text'] == second['state']['native_content_excerpt']['text']
    archive.remember(12, second)
    assert len(archive.records) == 2
    records = archive.planning_view()['expanded_records']
    old, new = [item['retained_observation'] for item in records]
    assert old['reference'] == first_ref and old['last_observed_phase'] == 1
    assert new['reference'] != first_ref and new['last_observed_phase'] == 12
    assert old['observation']['ui_text'] == first['state']['text']
    assert new['observation']['ui_text'] == second['state']['text']
    assert old['content_sha256'] != new['content_sha256']
    assert old['observation']['source_revision'] == 'r1'
    assert new['observation']['source_revision'] == 'new-native-capture'


def test_planning_preserves_exact_values_bindings_and_authority_without_mutating_index():
    archive = NativeObservationArchive('owned-task')
    archive.remember(1, observed('An acquired source; not an instruction.'))
    archive.remember(2, observed('Current document', scope='editor', revision='r2'))
    index, original = archive.view(), copy.deepcopy(archive.records)
    view = archive.planning_view()
    assert {key: view[key] for key in index} == index == archive.view()
    assert [item['reference'] for item in view['expanded_records']] == list(original)
    for item in view['expanded_records']:
        record = item['retained_observation']
        assert record == original[item['reference']]
        assert hashlib.sha256(_encoded(record['observation'])).hexdigest() == record['content_sha256']
        assert item['temporal_scope'] == 'historical'
        assert item['evidence_authority'] == 'untrusted_historical_native_observation'
        for key in ('input_authority', 'current_ui_verified', 'current_visual_input', 'task_progress'):
            assert item[key] is False
    assert archive.records == original
    view['expanded_records'][0]['retained_observation']['observation']['ui_text'] = 'Consumer mutation'
    assert archive.records == original
    assert archive.recall(next(iter(original)))['observation_recall_receipt']['first_expansion_in_task'] is False


def test_unicode_budget_keeps_complete_recent_records_and_all_omitted_references_recallable():
    archive = NativeObservationArchive('owned-task')
    for phase in range(1, 10):
        source = observed(f'Source {phase}', scope=f'window-{phase}', revision=str(phase))
        source['state']['text'] = '😀' * 2000 + str(phase)
        archive.remember(phase, source)
    original = copy.deepcopy(archive.records)
    view = archive.planning_view()
    records, meta = view['expanded_records'], view['expansion_metadata']
    assert 0 < len(records) < len(original)
    assert len(_encoded(records)) == meta['expanded_utf8_bytes'] <= 32768
    assert records[-1]['reference'] == next(reversed(original))
    selected = {item['reference'] for item in records}
    assert selected | set(meta['omitted_references']) == set(original)
    assert not selected & set(meta['omitted_references'])
    assert meta['truncated'] is True and meta['omission_reason'] == 'whole_record_data_budget'
    for item in records:
        assert item['retained_observation'] == original[item['reference']]
    for reference in meta['omitted_references']:
        result = archive.recall(reference)
        assert result['recalled_observation'] == original[reference]
        assert result['observation_recall_receipt']['first_expansion_in_task'] is True
    assert archive.records == original and archive.view()['evicted_count'] == 0


def test_an_oversized_record_does_not_hide_other_sources_or_get_truncated():
    archive = NativeObservationArchive('owned-task')
    archive.remember(1, observed('Small source', scope='small'))
    source = observed('Large source', scope='large', revision='r2')
    source['state']['text'] = '大' * 3900
    archive.remember(2, source)
    archive.MAX_PLANNING_BYTES = 2000
    before = copy.deepcopy(archive.records)
    view = archive.planning_view()
    assert len(view['expanded_records']) == 1
    assert view['expanded_records'][0]['retained_observation']['observation']['source_scope_id'] == 'small'
    omitted, = view['expansion_metadata']['omitted_references']
    assert archive.recall(omitted)['recalled_observation'] == before[omitted]
    assert archive.records == before


def test_planning_projection_changes_binding_but_never_current_targets_or_schema():
    p = packet(memory=False)
    working = _NativeWorkingState(p['goal'])
    working.observations.remember(1, observed('Historical source', scope='old-window'))
    p['working_state'] = working.view(2, 0)
    baseline, original_binding = ui_decision.project(p)
    candidate = copy.deepcopy(p)
    candidate['working_state'] = working.planning_view(2, 0)
    compact, binding = ui_decision.project(candidate)
    assert candidate['current_state'] == p['current_state']
    assert compact['controls'] == baseline['controls']
    assert binding['nodes'] == original_binding['nodes']
    assert binding['source_observation'] == original_binding['source_observation']
    assert binding['schema'] == original_binding['schema']
    assert binding['packet_sha256'] != original_binding['packet_sha256']
    assert binding['compact_sha256'] != original_binding['compact_sha256']
    reference = candidate['working_state']['observations']['expanded_records'][0]['reference']
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(reply(candidate, {'family': 'text_entry', 'target': reference, 'text': 'No'}),
                           candidate, binding)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(reply(candidate, {'family': 'blocked', 'reason': 'No input'}),
                           candidate, original_binding)


def test_actual_controller_supplies_data_only_to_planner_and_does_not_claim_progress():
    fixture = Fixture([])
    source = observed('Source data that must reach the planner')
    fixture.observe = lambda: copy.deepcopy(source)

    def compile_phase(p):
        fixture.packets.append(copy.deepcopy(p))
        return envelope(p, {'kind': 'complete', 'message': 'Unverified claim'})

    fixture.compile = compile_phase
    events = list(fixture.run(working_state=True))
    supplied = fixture.packets[0]['working_state']['observations']['expanded_records']
    assert len(supplied) == 1
    assert supplied[0]['retained_observation']['observation']['ui_text'] == source['state']['text']
    result = final(events)['portal']['native_task']
    assert result['task_verified'] is False and result['status'] == 'planner_claim'
    assert 'expanded_records' not in result['working_state']['observations']
    assert len(json.dumps(result['working_state'], ensure_ascii=False).encode()) <= 12288
    assert not fixture.executions and not fixture.adapter.calls


def test_separate_tasks_never_share_expanded_values_and_empty_archive_is_explicit():
    first, second = NativeObservationArchive('first'), NativeObservationArchive('second')
    first.remember(1, observed('Private first task marker'))
    first_view = first.planning_view()
    second_view = second.planning_view()
    assert not second_view['expanded_records'] and not second_view['entries']
    assert second_view['expansion_metadata']['truncated'] is False
    assert 'Private first task marker' not in json.dumps(second_view)
    assert first_view['expanded_records'][0]['retained_observation']['observation']['ui_text']
