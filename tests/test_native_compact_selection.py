"""Compact selection reaches the existing native executor with honest readback.

These controlled adapter faults exercise the real private planner/controller
path. They are not loaded-model or physical desktop acceptance.
"""
import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from portal import ui_decision
from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_recovery import NativeStrategyLedger
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError, parse_step
from qwen_omni_adapters.ui_effects import compile_immediate_transition, immediate_effect_schema

GOAL = 'Select the specified passage in the editor and preserve every character.'
TEXT = 'First café 中 passage.\nSecond protected passage.'


class SelectionAdapter:
    def __init__(self, value=TEXT, *, fault=None, coverage='complete'):
        self.value, self.selection, self.focused = value, None, False
        self.generation, self.scope, self.fault = 'g1', 'window', fault
        self.coverage = coverage
        self.calls, self.prepared = [], []

    def observe(self):
        field = {'id': 'editor', 'generation': self.generation, 'scope_id': self.scope,
            'role': 'textbox', 'name': 'Document', 'visible': True, 'enabled': True,
            'sensitive': False, 'focused': self.focused, 'value': self.value,
            'operations': ['select_text', 'focus', 'key_chord', 'type_text'],
            'text_observation': {'character_count': len(self.value),
                'captured_range': {'start': 0, 'end': len(self.value)},
                'truncated': False, 'selection_count': int(self.selection is not None)}}
        if self.selection is not None:
            field['text_selection'] = copy.deepcopy(self.selection)
        if self.coverage == 'missing':
            field.pop('text_observation')
        elif self.coverage == 'clipped_large_document':
            field['text_observation'].update(character_count=50_000, truncated=True)
        state = {'session_id': 'session', 'scope': {'kind': 'desktop', 'id': self.scope},
            'nodes': [field, {'id': 'save', 'generation': 'b1', 'scope_id': self.scope,
                'role': 'button', 'name': 'Save', 'visible': True, 'enabled': True,
                'sensitive': False, 'focused': False, 'operations': ['invoke']}],
            'text': self.value}
        state['revision'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        return state

    def prepare_step(self, step):
        self.prepared.append(parse_step(step))

    def perform(self, action):
        self.calls.append(copy.deepcopy(action))
        assert action['verb'] == 'select_text'
        self.focused = True
        if self.fault != 'focus_only':
            self.selection = copy.deepcopy(action['args'])
        if self.fault == 'wrong_range':
            self.selection['end'] -= 1
        if self.fault == 'changed_text':
            self.value = 'X' + self.value[1:]
        return {'status': 'dispatched'}


def source(adapter=None, *, handoffs=True):
    state = (adapter or SelectionAdapter()).observe()
    working = _NativeWorkingState(GOAL).view(1, 0)
    observed = {'state': state}
    return {'goal': GOAL, 'current_state': observed, 'ledger': [], 'last_result': None,
        'working_state': working, 'remaining_seconds': 100,
        'response_schema': _working_phase_schema({'native_ui'},
            _grounded_phase_schema({'native_ui'}, observed, intent_choice=True, effect_contracts=True),
            working, observed_state=state, action_handoffs=handoffs)}


def selection(p, start=6, end=12, *, atomic=False):
    compact, binding = ui_decision.project(p)
    target = next(ref for ref, node in binding['nodes'].items() if node['id'] == 'editor')
    objective = 'Select only the requested observed character range without editing it.'
    compact, binding = ui_decision.bind_subgoal(compact, binding, {
        'current_view': 'An observed editor is available; selection is not yet verified.',
        'needed_evidence': ['Independent exact selection and unchanged-text readback'],
        'next_subgoal': objective, 'next_operation': 'Select the observed passage.',
        **({'immediate_action': {'family': 'select_text', 'target': target}} if atomic else {})},
        atomic_anchor=atomic)
    decision = {'action': {'family': 'select_text', 'target': target,
        'start': start, 'end': end, 'objective': objective},
        'source_epoch': p['working_state']['source_epoch'], 'source_phase': 1,
        'assessment': 'The requested range still needs native verification.',
        'intended_resource': 'Current editor', 'remaining_criteria': ['Preserve the text']}
    return compact, binding, decision


def execute(adapter, envelope):
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=True)
    controller = store._sessions[('session', 'desktop')].controller
    controller.max_observations, controller.verification_timeout_s = 1, .01
    controller.observation_interval_s = .001
    ledger = NativeStrategyLedger()
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger):
        result = run_verified_prefix(envelope['phase']['steps'],
            execute_step=lambda step, index: store.immediate_transition('session', step,
                goal=GOAL, objective=envelope['commitment']['objective'],
                action_handoff=envelope.get('action_handoff')),
            cancelled=cancel.is_set, strategy_ledger=ledger, source_phase=1)
    return result


def test_advertised_native_selection_reaches_compact_and_original_canonical_contract():
    p = source()
    before = copy.deepcopy(p)
    compact, binding, decision = selection(p, atomic=True)
    control = next(c for c in compact['controls'] if c['ref'] == decision['action']['target'])
    assert 'select_text' in control['families']
    assert control['value'] == TEXT
    assert control['text_observation']['character_count'] == len(TEXT)
    assert 'select_text' not in next(c for c in compact['controls'] if c['ref'] != control['ref'])['families']
    envelope = ui_decision.expand(decision, p, binding)
    Draft202012Validator(p['response_schema']).validate(envelope)
    step = envelope['phase']['steps'][0]
    assert step['verb'] == 'select_text' and step['target'] == {'node_id': 'editor'}
    assert step['args'] == {'start': 6, 'end': 12}
    assert step['expect'] == [{'kind': 'text_selection', 'node_id': '$target', 'start': 6, 'end': 12}]
    assert envelope['action_handoff']['operation_family'] == 'select_text'
    assert envelope['action_handoff']['input_authority'] is False
    assert envelope['action_handoff']['expected_transition']['task_progress'] is False
    assert p == before


@pytest.mark.parametrize('fault', ['not_advertised', 'hidden', 'disabled', 'sensitive', 'canonical_no_ui',
                                  'canonical_other_operation'])
def test_selection_never_gains_unadvertised_or_canonically_excluded_capability(fault):
    p = source()
    node = p['current_state']['state']['nodes'][0]
    if fault == 'not_advertised':
        node['operations'].remove('select_text')
    elif fault in {'hidden', 'disabled', 'sensitive'}:
        node[{'hidden': 'visible', 'disabled': 'enabled', 'sensitive': 'sensitive'}[fault]] = fault == 'sensitive'
    elif fault == 'canonical_no_ui':
        p['response_schema']['$defs']['native_working_phase']['oneOf'] = [
            branch for branch in p['response_schema']['$defs']['native_working_phase']['oneOf']
            if branch['properties']['kind']['const'] != 'ui']
    else:
        p['response_schema']['$defs']['grounded_first_step']['oneOf'] = [
            branch for branch in p['response_schema']['$defs']['grounded_first_step']['oneOf']
            if branch['properties']['verb']['const'] != 'select_text']
    compact, _ = ui_decision.project(p)
    assert all('select_text' not in c['families'] for c in compact['controls'])
    assert all(b['properties']['family']['const'] != 'select_text'
        for b in compact['decision_schema']['properties']['action']['oneOf'])


@pytest.mark.parametrize('start,end', [(0, 5), (6, 12), (0, len(TEXT))])
def test_selection_executes_exact_unicode_range_and_preserves_full_text_without_task_credit(start, end):
    adapter = SelectionAdapter()
    p = source(adapter)
    _, binding, decision = selection(p, start, end, atomic=True)
    envelope = ui_decision.expand(decision, p, binding)
    result = execute(adapter, envelope)
    assert len(adapter.calls) == 1
    assert adapter.calls[0]['args'] == {'start': start, 'end': end}
    assert adapter.selection == {'start': start, 'end': end} and adapter.value == TEXT
    assert adapter.calls[0]['expect'] == [
        {'kind': 'text_selection', 'node_id': 'editor', 'start': start, 'end': end},
        {'kind': 'value', 'node_id': 'editor', 'value': TEXT}]
    assert result['status'] == 'complete'
    assert result['verified_prefix_receipt']['all_immediate_readbacks_verified'] is True
    assert result['phase_handoff']['terminal_verified'] is False
    assert result['phase_handoff']['goal_verified'] is False and result['task_progress'] is False
    Draft202012Validator(immediate_effect_schema()).validate(result['immediate_effect_contract'])


@pytest.mark.parametrize('fault', ['focus_only', 'wrong_range', 'changed_text'])
def test_dispatch_focus_or_even_correct_range_cannot_verify_the_wrong_selection_effect(fault):
    adapter = SelectionAdapter(fault=fault)
    p = source(adapter)
    _, binding, decision = selection(p)
    result = execute(adapter, ui_decision.expand(decision, p, binding))
    assert len(adapter.calls) == 1 and adapter.focused is True
    if fault == 'changed_text':
        assert adapter.selection == {'start': 6, 'end': 12} and adapter.value != TEXT
    assert result['status'] == 'needs_replan'
    assert result['verified_prefix_receipt']['verified_prefix_count'] == 0
    assert result['verified_prefix_receipt']['all_immediate_readbacks_verified'] is False
    assert result['task_progress'] is False


@pytest.mark.parametrize('fault', ['generation', 'scope', 'value'])
def test_captured_selection_target_must_still_match_before_any_native_input(fault):
    adapter = SelectionAdapter()
    p = source(adapter)
    _, binding, decision = selection(p)
    envelope = ui_decision.expand(decision, p, binding)
    setattr(adapter, fault, 'changed')
    result = execute(adapter, envelope)
    assert not adapter.calls
    assert result['error'] == 'ui_action_handoff_stale'
    assert result['task_progress'] is False


@pytest.mark.parametrize('start,end', [(1, 1), (5, 2), (-1, 4), (0, 1_000_001), (False, 4), (0, 2.5)])
def test_invalid_ranges_do_not_expand_into_native_work(start, end):
    p = source()
    _, binding, decision = selection(p, start, end)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, p, binding)


@pytest.mark.parametrize('fault', ['packet', 'private_target', 'private_source', 'injected_predicate', 'authority'])
def test_modified_private_source_or_model_oracle_is_rejected(fault):
    p = source()
    _, binding, decision = selection(p, atomic=True)
    if fault == 'packet':
        p['current_state']['state']['revision'] = 'changed'
    elif fault == 'private_target':
        binding['nodes'][decision['action']['target']]['generation'] = 'changed'
    elif fault == 'private_source':
        binding['source_observation']['scope_id'] = 'changed'
    elif fault == 'injected_predicate':
        decision['action']['expect'] = [{'kind': 'focused', 'node_id': '$target'}]
    else:
        decision['action']['input_authority'] = True
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, p, binding)


def test_existing_non_handoff_client_envelope_stays_valid_and_selection_is_not_public_boundary():
    p = source(handoffs=False)
    _, binding, decision = selection(p)
    envelope = ui_decision.expand(decision, p, binding)
    assert 'action_handoff' not in envelope
    Draft202012Validator(p['response_schema']).validate(envelope)
    store = ComputerUseStore(None, adapter_factory=lambda *_: SelectionAdapter())
    cancel = store.begin_task('session', private_boundaries=True)
    with pytest.raises(ComputerUseError):
        store.immediate_transition('session', envelope['phase']['steps'][0], goal=GOAL, objective='Selection')
    with store.task_scope('session', 'desktop', cancel), pytest.raises(ComputerUseError):
        store.act('session', {'operation': 'immediate_transition', 'surface': 'desktop',
            'step': envelope['phase']['steps'][0]}, goal=GOAL)


def test_unchanged_text_predicate_uses_native_pretext_even_if_model_supplies_another_value():
    before = SelectionAdapter().observe()
    step = {'id': 'select', 'verb': 'select_text', 'target': {'node_id': 'editor'},
        'args': {'start': 0, 'end': 5},
        'expect': [{'kind': 'value', 'node_id': '$target', 'value': 'A model claim'}]}
    effect = compile_immediate_transition(step, before)
    assert effect.bind(step)['expect'][-1] == {'kind': 'value', 'node_id': '$target', 'value': TEXT}
    assert effect.as_dict()['task_progress'] is False


@pytest.mark.parametrize('coverage', ['missing', 'clipped_large_document'])
def test_unknown_document_preservation_keeps_native_range_selection_available(coverage):
    adapter = SelectionAdapter(coverage=coverage)
    p = source(adapter)
    _, binding, decision = selection(p)
    result = execute(adapter, ui_decision.expand(decision, p, binding))
    assert len(adapter.calls) == 1 and adapter.selection == {'start': 6, 'end': 12}
    assert adapter.calls[0]['expect'] == [
        {'kind': 'text_selection', 'node_id': 'editor', 'start': 6, 'end': 12}]
    assert result['status'] == 'complete'
    contract = result['immediate_effect_contract']
    assert contract['mode'] == 'verified_step'
    assert contract['reason'] == 'selection_range_only_text_preservation_unknown'
    assert all(p['kind'] != 'value' for p in contract['predicates'])
    assert result['phase_handoff']['goal_verified'] is False and result['task_progress'] is False


@pytest.mark.parametrize('fault', ['no_source', 'missing_coverage', 'clipped', 'too_large'])
def test_selection_compiler_does_not_invent_document_preservation_when_pretext_is_unknown(fault):
    before = SelectionAdapter().observe()
    if fault == 'no_source':
        before = None
    elif fault == 'missing_coverage':
        before['nodes'][0].pop('text_observation')
    elif fault == 'clipped':
        before['nodes'][0]['text_observation']['truncated'] = True
    else:
        before = SelectionAdapter('x' * 5000).observe()
    step = {'id': 'select', 'verb': 'select_text', 'target': {'node_id': 'editor'},
        'args': {'start': 0, 'end': 5}}
    effect = compile_immediate_transition(step, before)
    assert effect.mode == 'verified_step'
    assert effect.as_dict()['reason'] == 'selection_range_only_text_preservation_unknown'
    assert effect.bind(step)['expect'] == [
        {'kind': 'text_selection', 'node_id': '$target', 'start': 0, 'end': 5}]
