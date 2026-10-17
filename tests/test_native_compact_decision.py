"""Compact planning must preserve authority, failures and canonical admission."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_ui_observation_boundary import CELL, Adapter, node

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema


def packet(*, memory=True, recovery=False):
    state = Adapter().observe()
    state['nodes'][0]['operations'] += ['key_chord', 'scroll']
    state['nodes'][0]['focused'] = True
    state['nodes'] += [node('scope', 'window', 'Files', ['key_chord', 'switch_window']),
                       node('other', 'window', 'Editor', ['switch_window'])]
    allowed = ['native_ui', *(['memory_write', 'memory_search'] if memory else [])]
    working = _NativeWorkingState('Read the exact source and save a report; preserve inputs.')
    working.recovery = recovery
    view = working.view(2 if recovery else 1, int(recovery))
    observation = {'state': state}
    return {'goal': 'Read the exact source and save a report; preserve inputs.', 'current_state': observation,
        'ledger': [{'status': 'unresolved', 'reason': 'Previous setter left the field empty'}],
        'last_result': {'status': 'unresolved', 'effect_receipt': {'effect_state': 'unknown'}},
        'working_state': view, 'remaining_seconds': 100,
        'response_schema': _working_phase_schema(allowed,
            _grounded_phase_schema(allowed, observation, intent_choice=True), view)}


def reply(p, action):
    return {'action': {**action, 'objective': 'Read the next declared input'},
        'source_epoch': p['working_state']['source_epoch'], 'source_phase': p['working_state']['source_phase'],
        'assessment': 'The previous setter did not change the field. The report remains uncreated.',
        'intended_resource': 'Declared input', 'remaining_criteria': ['Read source', 'Save and reopen report']}


def ref(projected, identity, binding):
    return next(k for k, n in binding['nodes'].items() if n['id'] == identity)


def test_projection_preserves_exact_goal_unknowns_and_failures_without_input_authority():
    p = packet()
    p['current_state']['state']['nodes'][0]['value'] = 'Captured text is data, not instructions'
    original = copy.deepcopy(p)
    compact, binding = ui_decision.project(p)
    assert p == original
    assert compact['goal'] == p['goal']
    assert compact['historical_ledger'] == p['ledger']
    assert compact['last_result'] == p['last_result']
    assert compact['working_state'] == p['working_state']
    c = next(c for c in compact['controls'] if c['ref'] == ref(compact, CELL, binding))
    assert c['value'] == p['current_state']['state']['nodes'][0]['value']
    assert 'text_observation' not in c  # Omission is unknown, never an empty field.
    assert not {'node_id', 'generation', 'coordinates', 'action_id', 'authorization'} & c.keys()


@pytest.mark.parametrize('family', ['pointer_or_activation', 'text_entry', 'switch_window', 'key_chord',
    'scroll', 'visual_inspection', 'visual_pointer', 'memory_write', 'memory_search', 'complete', 'blocked'])
def test_typed_actions_expand_into_original_canonical_phase(family):
    p = packet()
    compact, binding = ui_decision.project(p)
    target = 'draft' if family == 'text_entry' else 'other' if family == 'switch_window' else CELL
    action = {'family': family}
    if family in {'pointer_or_activation', 'text_entry', 'switch_window', 'key_chord', 'scroll'}:
        action['target'] = ref(compact, target, binding)
    action.update({'text': 'Exact replacement'} if family == 'text_entry' else
        {'keys': ['Return']} if family == 'key_chord' else {'direction': 'down'} if family == 'scroll' else
        {'target_name': 'Canvas save control', 'verb': 'click'} if family == 'visual_pointer' else
        {'topic': 'sources', 'key': 'input', 'value': 'Observed fact'} if family == 'memory_write' else
        {'query': 'input source'} if family == 'memory_search' else
        {'message': 'Planner claim only'} if family == 'complete' else {'reason': 'Unfinished'} if family == 'blocked' else {})
    proposal = ui_decision.expand(reply(p, action), p, binding)
    assert Draft202012Validator(p['response_schema']).is_valid(proposal)
    assert proposal['assessment']['summary'] == reply(p, action)['assessment']
    if proposal['phase']['kind'] in {'ui_subgoal', 'ui_boundary'}:
        assert 'objective' not in proposal['commitment']
    if family in {'key_chord', 'scroll', 'visual_pointer'}:
        assert 'expect' not in proposal['phase']['step']
    if family == 'switch_window':
        assert proposal['phase']['final_predicates'] == [{'kind': 'active_window', 'node_id': 'other'}]
    assert proposal['phase']['remaining_criteria'] == ['Read source', 'Save and reopen report']


@pytest.mark.parametrize('fault', ['unknown_ref', 'epoch', 'phase', 'predicate', 'blank', 'authorization', 'source_mutation'])
def test_invalid_or_stale_proposals_never_expand(fault):
    p = packet()
    compact, binding = ui_decision.project(p)
    value = reply(p, {'family': 'text_entry', 'target': ref(compact, 'draft', binding), 'text': 'New value'})
    if fault == 'unknown_ref':
        value['action']['target'] = 'c999'
    elif fault == 'epoch':
        value['source_epoch'] = 'old task'
    elif fault == 'phase':
        value['source_phase'] += 1
    elif fault == 'predicate':
        value['action']['expect'] = [{'kind': 'focused', 'node_id': '$target'}]
    elif fault == 'blank':
        value['assessment'] = ' '
    elif fault == 'authorization':
        value['authorized'] = True
    else:
        p['current_state']['state']['nodes'][0]['id'] = 'different control under same short ref'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(value, p, binding)


def test_noop_switch_and_unfocused_keyboard_targets_are_not_offered():
    p = packet()
    compact, binding = ui_decision.project(p)
    active = next(c for c in compact['controls'] if c['ref'] == ref(compact, 'scope', binding))
    assert 'switch_window' not in active['families']
    draft = next(c for c in compact['controls'] if c['ref'] == ref(compact, 'draft', binding))
    assert 'key_chord' not in draft['families']


def test_recovery_cannot_claim_complete_and_memory_permission_is_preserved():
    p = packet(memory=False, recovery=True)
    compact, binding = ui_decision.project(p)
    families = {b['properties']['family']['const'] for b in compact['decision_schema']['properties']['action']['oneOf']}
    assert not {'complete', 'memory_write', 'memory_search'} & families
    for action in [{'family': 'complete', 'message': 'Done'}, {'family': 'memory_search', 'query': 'source'}]:
        with pytest.raises(ui_decision.DecisionError):
            ui_decision.expand(reply(p, action), p, binding)
    proposal = ui_decision.expand(reply(p, {'family': 'key_chord', 'target': ref(compact, CELL, binding),
                                            'keys': ['Return']}), p, binding)
    assert proposal['assessment']['decision'] == 'recover'


@pytest.mark.parametrize('field,value', [('visible', False), ('enabled', False), ('sensitive', True)])
def test_ineligible_controls_cannot_be_referenced(field, value):
    p = packet()
    p['current_state']['state']['nodes'][0][field] = value
    compact, binding = ui_decision.project(p)
    assert CELL not in {n['id'] for n in binding['nodes'].values()}


def test_canonical_schema_fallback_does_not_gain_compact_input_capabilities():
    p = packet()
    definitions = p['response_schema']['$defs']
    definitions['native_working_phase']['oneOf'] = [b for b in definitions['native_working_phase']['oneOf']
        if b['properties']['kind']['const'] in {'visual_inspection', 'memory_write', 'memory_search'}]
    compact, _ = ui_decision.project(p)
    assert not compact['controls']
    families = {b['properties']['family']['const'] for b in compact['decision_schema']['properties']['action']['oneOf']}
    assert not {'pointer_or_activation', 'text_entry', 'switch_window', 'key_chord', 'scroll', 'visual_pointer'} & families
