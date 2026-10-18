"""Typed UI predictions survive policy/recovery and remain target-bound observations."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_action_handoff import _deferred
from test_native_compact_decision import packet, reply
from test_native_effect_integration import GOAL, OBJECTIVE, _claim, _immediate, _store
from test_native_subgoal_handoff import proposal
from test_native_text_entry import Editor
from test_native_text_entry import intent as text_intent
from test_ui_intent_choice import configured, subgoal
from test_ui_observation_boundary import CELL, ROW, intent, node

from portal import ui_decision
from portal.ui_recovery import NativeStrategyLedger
from portal.ui_task import _grounded_phase_schema, _working_phase_schema


def _descriptor(state, kind='popup_opened'):
    return {'kind': kind, 'source_ref': {'session_id': state['session_id'],
        'scope_id': state['scope']['id'], 'revision': state['revision']}}


def _production_packet():
    p = packet()
    p['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, p['current_state'], intent_choice=True),
        p['working_state'], observed_state=p['current_state']['state'], action_handoffs=True)
    return p


def _popup_observer(adapter, *, unrelated=False, contradictory=False):
    original = adapter.observe
    def observe():
        state = original()
        state['scope']['provider'] = 'atspi2'
        state['truncated'] = False
        if adapter.selected:
            target = next(n for n in state['nodes'] if n['id'] == (ROW if unrelated else CELL))
            ancestor = {k: target[k] for k in ('id', 'generation', 'role', 'name', 'visible', 'enabled')}
            if contradictory:
                ancestor['role'] = 'unrelated_role'
            state['nodes'].append(node('new-popup', 'popup_menu', 'Documents', [], structure={
                'parent_id': target['id'], 'child_index': 0, 'ancestors': [ancestor]}))
        state['revision'] = ui_decision._digest({k: v for k, v in state.items() if k != 'revision'})
        return state
    adapter.observe = observe


@pytest.mark.parametrize('family,extra', [('pointer_or_activation', {}),
    ('key_chord', {'keys': ['Control_L', 'l']}), ('scroll', {'direction': 'down'})])
def test_compact_transition_choice_expands_to_exact_captured_source_without_predicates(family, extra):
    p = _production_packet()
    compact, binding = ui_decision.project(p)
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal())
    decision = reply(p, {'family': family, 'target': 'c0', 'transition_kind': 'popup_opened', **extra})
    decision['action']['objective'] = proposal()['next_subgoal']
    result = ui_decision.expand(decision, p, binding)
    assert result['transition'] == _descriptor(p['current_state']['state'])
    assert result['action_handoff']['expected_transition']['transition_kind'] == 'popup_opened'
    assert Draft202012Validator(p['response_schema']).is_valid(result)
    assert result['action_handoff']['input_authority'] is False
    assert 'predicates' not in result['action_handoff']['expected_transition']


@pytest.mark.parametrize('fault', ['unsupported_kind', 'literal_oracle', 'wrong_family', 'changed_source'])
def test_compact_transition_cannot_invent_literals_operations_or_rebind_source(fault):
    p = _production_packet()
    compact, binding = ui_decision.project(p)
    action = {'family': 'pointer_or_activation', 'target': 'c0', 'transition_kind': 'popup_opened'}
    if fault == 'unsupported_kind':
        action['transition_kind'] = 'file_saved'
    elif fault == 'literal_oracle':
        action['transition'] = {'kind': 'text', 'text': 'Narrative explanation'}
    elif fault == 'wrong_family':
        action = {'family': 'visual_inspection', 'transition_kind': 'popup_opened'}
    else:
        p['current_state']['state']['revision'] = 'other-source'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(reply(p, action), p, binding)


def test_recovery_preserves_original_kind_and_rebinds_only_the_new_host_captured_source():
    p = _deferred()
    claim = p['last_result']['action_handoff']
    claim['expected_transition']['transition_kind'] = 'popup_opened'
    original = _descriptor(p['current_state']['state'])
    p['last_result']['pointer_recovery']['transition'] = original
    p['current_state']['state']['revision'] = 'new-planning-projection'
    p['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, p['current_state'], intent_choice=True), p['working_state'],
        pointer_recovery=p['last_result']['pointer_recovery'], observed_state=p['current_state']['state'], action_handoffs=True)
    compact, binding = ui_decision.project(p)
    view, schema, _ = ui_decision.pointer_resolution_context(compact, binding)
    assert view['requested_transition'] == original
    assert schema['properties']['transition_kind'] == {'const': 'popup_opened'}
    resolved, _, _ = ui_decision.resolve_pointer_proposal(compact, binding, p,
        {'choice': 'double_click_c0', 'expected_observation': 'A chooser may open.'})
    assert resolved['transition'] == _descriptor(p['current_state']['state'])
    assert resolved['transition']['kind'] == original['kind']
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.resolve_pointer_proposal(compact, binding, p,
            {'choice': 'double_click_c0', 'expected_observation': 'Another transition.', 'transition_kind': 'document_changed'})


@pytest.mark.parametrize('unrelated,contradictory', [(False, False), (True, False), (False, True)])
def test_actual_selected_pointer_comparison_requires_exact_target_linkage(unrelated, contradictory):
    adapter, store, cancel, requests = configured(selected_verb='double_click')
    _popup_observer(adapter, unrelated=unrelated, contradictory=contradictory)
    before = adapter.observe()
    claim = _claim(before)
    claim['expected_transition']['transition_kind'] = 'popup_opened'
    descriptor = _descriptor(before)
    result = _immediate(store, cancel, intent('click'), action_handoff=claim, transition=descriptor)
    assert len(requests) == len(adapter.calls) == 1
    assert adapter.calls[0]['verb'] == 'double_click'
    assert result['immediate_effect_contract']['operation'] == 'double_click'
    comparison = result['immediate_transition_receipt']['comparison']
    assert comparison['status'] == ('unknown' if unrelated or contradictory else 'observed')
    if unrelated or contradictory:
        assert comparison['reason'] == 'transition_unrelated_to_action_target'
    assert result['transition'] == descriptor
    assert result['immediate_transition_receipt']['original_source_ref'] == descriptor['source_ref']
    assert result['immediate_transition_receipt']['source_target_binding']['node_id'] == CELL
    assert result['task_progress'] is False and comparison['resource_acquired'] is False
    ledger = NativeStrategyLedger()
    facts = ledger.record(result)
    assert facts[0]['classification'] == ('input_effect_uncertain' if unrelated or contradictory else 'observed_transition')


def test_deferral_and_pointer_recovery_keep_transition_and_original_source_intact():
    adapter, store, cancel, _ = configured(probability=.6)
    _popup_observer(adapter)
    before = adapter.observe()
    claim = _claim(before)
    claim['expected_transition']['transition_kind'] = 'popup_opened'
    descriptor = _descriptor(before)
    with store.task_scope('session', 'desktop', cancel):
        deferred = store.intent_boundary('session', subgoal(), goal=GOAL,
            action_handoff=claim, transition=descriptor)
        offer = deferred['pointer_recovery']
        assert not adapter.calls
        assert offer['transition'] == descriptor
        assert deferred['immediate_transition_receipt']['comparison']['status'] == 'unknown'
        rebound = _claim(adapter.observe(), family='pointer_recovery', phase=2)
        rebound['expected_transition']['transition_kind'] = 'popup_opened'
        rebound['predecessor'] = {'handoff_sha256': ui_decision._digest(claim), 'source_phase': 1,
            'recovery_reference': offer['reference']}
        result = store.resolve_pointer('session', offer['reference'], intent('click'), goal=GOAL,
            objective=OBJECTIVE, action_handoff=rebound)
    assert len(adapter.calls) == 1
    assert result['immediate_transition_receipt']['comparison']['status'] == 'observed'
    assert result['transition'] == descriptor
    assert result['immediate_transition_receipt']['original_source_ref'] == descriptor['source_ref']
    assert result['task_progress'] is False


@pytest.mark.parametrize('replacement', ['kind', 'foreign_source'])
def test_recovery_cannot_replace_captured_kind_or_source(replacement):
    adapter, store, cancel, _ = configured(probability=.6)
    before = adapter.observe()
    descriptor = _descriptor(before)
    with store.task_scope('session', 'desktop', cancel):
        deferred = store.intent_boundary('session', subgoal(), goal=GOAL, transition=descriptor)
        changed = copy.deepcopy(descriptor)
        if replacement == 'kind':
            changed['kind'] = 'document_changed'
        else:
            changed['source_ref']['session_id'] = 'another-session'
        result = store.resolve_pointer('session', deferred['pointer_recovery']['reference'], intent('click'),
            goal=GOAL, objective=OBJECTIVE, transition=changed)
    assert not adapter.calls
    assert result['status'] != 'observed_unverified'


@pytest.mark.parametrize('verb,args,target', [('focus', {}, 'draft'), ('set_value', {'value': 'existing'}, 'draft'),
    ('set_checked', {'checked': True}, 'draft'), ('select_text', {'start': 0, 'end': 3}, 'draft'),
    ('switch_window', {}, 'editor')])
def test_already_satisfied_host_predicate_gets_independent_no_input_verification(verb, args, target):
    editor = Editor('existing', setter=True)
    editor.focused, editor.selection = True, {'start': 0, 'end': 3}
    original = editor.observe
    def observed():
        state = original()
        state['nodes'][0].update(checked=True)
        state['nodes'][0]['operations'].append('set_checked')
        window = node('editor', 'window', 'Editor', ['switch_window'])
        window['scope_id'] = 'editor'
        state['nodes'].append(window)
        return state
    editor.observe = observed
    _, store, cancel = _store(editor)
    result = _immediate(store, cancel, {'id': 'verify-current', 'verb': verb, 'target': {'node_id': target}, 'args': args})
    assert result['status'] == 'complete' and result['outcomes'] == [] and not editor.calls
    receipt = result['verification_only_receipt']
    assert receipt['schema'] == 'robit.omni.native-verification-only.v1' and receipt['status'] == 'verified'
    assert receipt['input_attempted'] is False and receipt['task_progress'] is False
    assert receipt['target']['node_id'] == target
    assert receipt['predicates'][0]['node_id'] == target
    assert result['terminal_receipt']['status'] == 'verified'
    assert result['phase_handoff']['goal_verified'] is False
    assert 'dispatch_record' not in result and result['task_progress'] is False


def test_changed_independent_no_input_readback_does_not_skip_required_write():
    editor = Editor('existing', setter=True)
    original = editor.observe
    reads = 0
    def changed():
        nonlocal reads
        reads += 1
        if reads == 2:
            editor.value = 'external edit'
            editor.revision += 1
        return original()
    editor.observe = changed
    _, store, cancel = _store(editor)
    result = _immediate(store, cancel, {'id': 'write', 'verb': 'set_value', 'target': {'node_id': 'draft'},
        'args': {'value': 'existing'}})
    assert result['status'] == 'complete' and editor.value == 'existing'
    assert [a['verb'] for a in editor.calls] == ['set_value']
    assert 'verification_only_receipt' not in result


def test_type_text_is_not_reclassified_as_no_input_even_when_predicted_value_is_already_equal():
    editor = Editor('x')
    editor.focused, editor.selection = True, {'start': 0, 'end': 1}
    _, store, cancel = _store(editor)
    result = _immediate(store, cancel, {'id': 'type', 'verb': 'type_text', 'target': {'node_id': 'draft'},
        'args': {'text': 'x'}})
    assert 'verification_only_receipt' not in result
    assert [a['verb'] for a in editor.calls] == ['type_text']


def test_composite_text_transition_is_retained_as_unknown_without_claiming_a_ui_effect():
    editor = Editor('existing', setter=True)
    _, store, cancel = _store(editor)
    descriptor = _descriptor(editor.observe(), 'focus_changed_within_scope')
    with store.task_scope('session', 'desktop', cancel):
        result = store.intent_boundary('session', text_intent('existing'), goal=GOAL, transition=descriptor)
    assert result['transition'] == descriptor
    assert result['immediate_transition_receipt']['comparison']['status'] == 'unknown'
    assert result['immediate_transition_receipt']['comparison']['reason'] == 'transition_not_supported_for_composite_text_action'
    assert result['task_progress'] is False
