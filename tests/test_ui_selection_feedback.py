from __future__ import annotations

import copy
import hashlib
import json

import httpx
import pytest
from test_canonical_computer_use import Adapter, step

from portal.computer_use import ComputerUseStore, SystemOneUiProvider
from portal.ui_task import _compact_result, run_native_goal


class CurrentWindow(Adapter):
    def __init__(self, session):
        super().__init__(session)
        self.nodes = [{'id': 'owned-window', 'generation': 'g1', 'scope_id': 'owned-window',
                       'role': 'window', 'name': 'Input', 'value': '', 'visible': True,
                       'enabled': True, 'sensitive': False, 'focused': False,
                       'operations': ['click', 'double_click', 'right_click', 'scroll', 'key_chord']}]


def _switch_request():
    return {'operation': 'workflow', 'surface': 'desktop', 'steps': [
        {'id': 'switch-to-input', 'verb': 'switch_window',
         'target': {'node_id': 'owned-window', 'role': 'window', 'name': 'Input'},
         'args': {}, 'expect': [{'kind': 'active_window', 'node_id': '$target'}]}],
        'final_predicates': [{'kind': 'active_window', 'node_id': 'owned-window'}]}


def test_current_window_unsupported_switch_retains_actual_zero_candidate_selector_feedback():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: CurrentWindow(session))
    result = store.act('session', _switch_request())
    assert result['status'] == 'needs_replan'
    receipt = result['selection_receipts'][0]
    assert receipt['selection_status'] == 'no_matching_control'
    assert receipt['selected_kind'] == 'abstain' and receipt['executable_candidate_count'] == 0
    assert receipt['current_scope_id'] == 'owned-window'
    matched = receipt['matched_controls'][0]
    assert matched['id'] == 'owned-window' and matched['is_current_scope'] is True
    assert matched['focused'] is False and 'switch_window' not in matched['operations']
    assert store._owned('session', 'desktop').adapter.calls == []
    assert result['phase_handoff']['new_effect_count'] == 0
    assert result['phase_handoff']['terminal_verified'] is False


def _write_request(steps=None):
    return {'operation': 'workflow', 'surface': 'desktop', 'steps': steps or [step()],
            'final_predicates': [{'kind': 'value', 'node_id': 'draft', 'value': 'Reviewed draft'}]}


@pytest.mark.parametrize(('ambiguous', 'status', 'count', 'kind'), [
    (False, 'semantic_unique', 1, 'action'), (True, 'not_configured', 2, 'abstain'),
])
def test_actual_unique_or_missing_policy_status_is_retained_without_changing_dispatch(ambiguous, status, count, kind):
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session, ambiguous=ambiguous))
    result = store.act('session', _write_request())
    receipt = result['selection_receipts'][0]
    assert (receipt['selection_status'], receipt['executable_candidate_count'], receipt['selected_kind']) == (status, count, kind)
    assert receipt['session_id'] != 'session' and len(receipt['session_id']) == 64
    assert len(store._owned('session', 'desktop').adapter.calls) == (0 if ambiguous else 1)
    assert result['phase_handoff']['new_effect_count'] == (0 if ambiguous else 1)


@pytest.mark.parametrize('low_confidence', [False, True])
def test_finite_policy_response_and_threshold_abstention_keep_original_policy_receipt(low_confidence):
    def respond(request):
        body = json.loads(request.content)
        legal = body['questions']['action']['criteria']
        choice = next(key for key, contract in legal.items() if 'node_ref' in json.loads(contract))
        probabilities = {key: 1 / len(legal) if low_confidence else .01 for key in legal}
        if not low_confidence:
            probabilities[choice] = 1 - .01 * (len(legal) - 1)
        return httpx.Response(200, json={'model': body['model'], 'answers': {'action': {
            'choice': choice, 'probabilities': probabilities}}})
    policy = SystemOneUiProvider('http://127.0.0.1:8941', mode='active', threshold=.95,
                                 client=httpx.Client(transport=httpx.MockTransport(respond)))
    store = ComputerUseStore(None, provider=policy, adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    result = store.act('session', _write_request())
    receipt = result['selection_receipts'][0]
    # "accepted" is the provider response status even when its threshold
    # chooses abstain; feedback must not invent a different causal label.
    assert receipt['selection_status'] == 'accepted' and receipt['executable_candidate_count'] == 2
    assert receipt['selected_kind'] == ('abstain' if low_confidence else 'action')
    assert result['policy_receipts'] == [policy.last_decision]
    assert len(store._owned('session', 'desktop').adapter.calls) == (0 if low_confidence else 1)


def test_cancelled_finite_selection_is_retained_but_never_dispatched():
    class Policy:
        def decide(self, _state, actions, *, context):
            store.act('session', {'operation': 'cancel', 'surface': 'desktop'})
            return {'choice': next(item['id'] for item in actions if item['kind'] == 'action'), 'receipt': {'status': 'accepted'}}
    store = ComputerUseStore(None, provider=Policy(), adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    result = store.act('session', _write_request())
    assert result['status'] == 'cancelled'
    assert result['selection_receipts'][0]['selection_status'] == 'cancelled'
    assert result['selection_receipts'][0]['selected_kind'] == 'abstain'
    assert store._owned('session', 'desktop').adapter.calls == []


def test_invalid_policy_choice_is_unresolved_without_fabricating_a_valid_selection_receipt():
    class Policy:
        def decide(self, *_args, **_kwargs):
            return {'choice': 'invented-choice', 'receipt': {'status': 'accepted'}}
    store = ComputerUseStore(None, provider=Policy(), adapter_factory=lambda session, _surface: Adapter(session, ambiguous=True))
    result = store.act('session', _write_request())
    assert result['status'] == 'needs_replan' and result['selection_receipts'] == []
    assert 'unknown candidate' in result['outcomes'][0]['reason']
    assert store._owned('session', 'desktop').adapter.calls == []


@pytest.mark.parametrize(('field', 'value', 'matches'), [
    ('name', 'Input', 1), ('name', 'input', 0), ('role', 'frame', 0),
    ('scope_id', 'other', 0), ('node_id', 'owned', 0),
])
def test_zero_offer_control_feedback_uses_exact_canonical_target_fields(field, value, matches):
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: CurrentWindow(session))
    request = _switch_request()
    request['steps'][0]['target'][field] = value
    result = store.act('session', request)
    receipt = result['selection_receipts'][0]
    assert receipt['observed_matching_control_count'] == matches
    assert len(receipt['matched_controls']) == matches
    assert receipt['absence_claims_supported'] is False
    assert 'reason' not in receipt


def test_observed_match_feedback_is_bounded_redacted_and_preserves_incomplete_coverage():
    class Crowded(CurrentWindow):
        def __init__(self, session):
            super().__init__(session)
            self.nodes = [{**self.nodes[0], 'id': f'window-{index}', 'sensitive': index == 0,
                           'name': 'private-name' if index == 0 else 'Input',
                           'description': 'private-description', 'value': 'private-value'} for index in range(12)]
        def observe(self):
            state = super().observe()
            state['revision'] = hashlib.sha256(state['revision'].encode()).hexdigest()
            return {**state, 'truncated': True}
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Crowded(session))
    request = _switch_request()
    request['steps'][0]['target'] = {'role': 'window'}
    result = store.act('session', request)
    receipt = result['selection_receipts'][0]
    assert receipt['source_truncated'] is True and receipt['absence_claims_supported'] is False
    assert receipt['observed_matching_control_count'] == 12 and receipt['omitted_matching_controls'] == 4
    assert len(receipt['matched_controls']) == 8
    assert receipt['matched_controls'][0]['name'] == '[protected]'
    assert receipt['matched_controls'][0]['operations'] == []
    assert 'private-' not in json.dumps(receipt)
    assert all('value' not in node and 'description' not in node for node in receipt['matched_controls'])


def test_selection_revision_binds_host_base_semantics_instead_of_frame_projection_metadata():
    class Grounded(CurrentWindow):
        def observe(self):
            return {**super().observe(), 'revision': 'frame-bound', 'base_semantic_revision': 'host-semantic'}
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Grounded(session))
    receipt = store.act('session', _switch_request())['selection_receipts'][0]
    assert receipt['semantic_revision'] == 'host-semantic'


def test_duplicate_step_positions_and_cached_outcomes_keep_idempotency_and_effect_counts():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: Adapter(session))
    request = _write_request([step(), step()])
    first = store.act('session', request)
    second = store.act('session', request)
    assert [receipt['step_index'] for receipt in first['selection_receipts']] == [0, 1]
    assert [receipt['step_id'] for receipt in first['selection_receipts']] == ['update-draft'] * 2
    assert len(store._owned('session', 'desktop').adapter.calls) == 2
    assert first['phase_handoff']['new_effect_count'] == 1
    assert second['phase_handoff']['new_effect_count'] == 0
    assert all(item['effect_receipt']['effect_state'] == 'replayed' for item in second['outcomes'])
    assert second['outcomes'][0]['action'] == first['outcomes'][1]['action']


def test_actual_next_compiler_packet_retains_selector_facts_and_raw_tool_result_is_unchanged():
    store = ComputerUseStore(None, adapter_factory=lambda session, _surface: CurrentWindow(session))
    request = _switch_request()
    plans = iter([{'kind': 'ui', 'steps': request['steps'], 'final_predicates': request['final_predicates']},
                  {'kind': 'blocked', 'reason': 'Planner stops after reviewing factual feedback.'}])
    packets, raw_results = [], []
    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        return next(plans)
    def execute(name, arguments):
        assert name == 'native_ui'
        result = store.act('session', {**arguments, 'surface': 'desktop'})
        raw_results.append(copy.deepcopy(result))
        return result
    events = list(run_native_goal('Exact original user goal',
        observe=lambda: store.act('session', {'operation': 'observe', 'surface': 'desktop'}),
        compile_phase=compile_phase, execute=execute, cancel=lambda: None, allowed_tools={'native_ui'}))
    feedback = packets[1]['last_result']['selection_receipts']
    assert feedback == raw_results[0]['selection_receipts']
    assert feedback[0]['selection_status'] == 'no_matching_control'
    assert feedback[0]['matched_controls'][0]['is_current_scope'] is True
    assert packets[1]['goal'] == 'Exact original user goal'
    complete = next(event for event in events if event['type'] == 'tool' and event['phase'] == 'complete')
    assert json.loads(complete['tools'][0]['result']) == raw_results[0]
    compact = _compact_result(raw_results[0])
    compact['selection_receipts'][0]['selection_status'] = 'changed-copy'
    assert raw_results[0]['selection_receipts'][0]['selection_status'] == 'no_matching_control'
    assert events[-1]['response']['portal']['native_task']['task_verified'] is False
