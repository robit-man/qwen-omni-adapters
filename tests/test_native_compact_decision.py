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


def _scope_pointer_packet(*, visual_boundary=True, pointer_only=False):
    value = packet()
    root = next(n for n in value['current_state']['state']['nodes'] if n['id'] == 'scope')
    root['operations'] = ['click', 'double_click', 'right_click', 'invoke']
    if not pointer_only:
        root['operations'] += ['key_chord', 'scroll', 'switch_window']
    root['focused'] = False  # Keyboard focus is distinct from current window scope.
    if not visual_boundary:
        allowed = ['native_ui', 'memory_write', 'memory_search']
        phases = _grounded_phase_schema(allowed, value['current_state'], intent_choice=True)
        phases['oneOf'] = [b for b in phases['oneOf'] if b['properties']['kind']['const'] != 'ui_boundary']
        value['response_schema'] = _working_phase_schema(allowed, phases, value['working_state'])
    return value


def _pointer_targets(compact):
    return [target for branch in compact['decision_schema']['properties']['action']['oneOf']
        if branch['properties']['family']['const'] == 'pointer_or_activation'
        for target in branch['properties']['target']['enum']]


@pytest.mark.parametrize('role,name', [('window', 'Files'), ('dialog', 'Σ same label'),
                                       ('custom_surface', '名前と🙂')])
def test_current_scope_pointer_is_not_an_immediate_activation_target(role, name):
    value = _scope_pointer_packet()
    state = value['current_state']['state']
    root = next(n for n in state['nodes'] if n['id'] == state['scope']['id'])
    root.update(role=role, name=name)
    state['nodes'][0]['name'] = name
    original = copy.deepcopy(value)
    compact, binding = ui_decision.project(value)
    root_ref = ref(compact, root['id'], binding)
    control = next(c for c in compact['controls'] if c['ref'] == root_ref)
    assert value == original and binding['nodes'][root_ref] == root
    assert control['active_window'] is True and control['focused'] is False
    assert control['role'] == role and control['name'] == name
    assert control['families'] == ['key_chord', 'scroll']
    assert root_ref not in _pointer_targets(compact)
    assert ref(compact, CELL, binding) in _pointer_targets(compact)
    context, schema = ui_decision.subgoal_context(compact, atomic_anchor=True)
    proposal = {'current_view': 'Current source scope is observed.', 'needed_evidence': [],
        'next_subgoal': 'Use a supported operation on this source.', 'next_operation': 'Choose an offered target.',
        'immediate_action': {'family': 'pointer_or_activation', 'target': root_ref}}
    assert context['active_window'] == control
    assert not Draft202012Validator(schema).is_valid(proposal)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.bind_subgoal(compact, binding, proposal, atomic_anchor=True)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(reply(value, proposal['immediate_action']), value, binding)


def test_pointer_only_current_scope_retains_observation_without_input_family():
    value = _scope_pointer_packet(pointer_only=True)
    compact, binding = ui_decision.project(value)
    root_ref = ref(compact, 'scope', binding)
    context, _ = ui_decision.subgoal_context(compact, atomic_anchor=True)
    assert context['active_window']['ref'] == root_ref
    assert context['active_window']['families'] == []
    assert binding['nodes'][root_ref] == value['current_state']['state']['nodes'][-2]
    assert root_ref not in _pointer_targets(compact)
    assert compact['goal'] == value['goal'] and compact['last_result'] == value['last_result']


def test_without_visual_boundary_legacy_current_scope_pointer_remains():
    value = _scope_pointer_packet(visual_boundary=False, pointer_only=True)
    compact, binding = ui_decision.project(value)
    root_ref = ref(compact, 'scope', binding)
    assert root_ref in _pointer_targets(compact)
    envelope = ui_decision.expand(reply(value, {'family': 'pointer_or_activation', 'target': root_ref}),
                                  value, binding)
    assert envelope['phase']['subgoal']['target'] == {'node_id': 'scope'}


def test_changing_scope_changes_only_the_exact_root_pointer_applicability():
    value = _scope_pointer_packet()
    other = next(n for n in value['current_state']['state']['nodes'] if n['id'] == 'other')
    other['operations'] += ['click', 'right_click']
    compact, binding = ui_decision.project(value)
    assert ref(compact, 'other', binding) in _pointer_targets(compact)
    value['current_state']['state']['scope']['id'] = 'other'
    compact, binding = ui_decision.project(value)
    assert ref(compact, 'other', binding) not in _pointer_targets(compact)
    assert ref(compact, 'scope', binding) in _pointer_targets(compact)
    switch = next(b for b in compact['decision_schema']['properties']['action']['oneOf']
                  if b['properties']['family']['const'] == 'switch_window')
    assert switch['properties']['target']['enum'] == [ref(compact, 'scope', binding)]


@pytest.mark.parametrize('verb', ['click', 'double_click', 'right_click', 'invoke'])
def test_each_child_primitive_and_explicit_visual_pointer_route_remains(verb):
    from portal.computer_use import _intent_candidates

    value = _scope_pointer_packet()
    state = value['current_state']['state']
    state['nodes'][0]['operations'] = [verb]
    compact, binding = ui_decision.project(value)
    child_ref = ref(compact, CELL, binding)
    native = ui_decision.expand(reply(value, {'family': 'pointer_or_activation', 'target': child_ref}),
                               value, binding)
    target = native['phase']['subgoal']['target']
    candidates = [c for c in _intent_candidates(state, target) if c['kind'] == 'intent']
    assert [(c['verb'], c['node']['id']) for c in candidates] == [(verb, CELL)]
    visual = ui_decision.expand(reply(value, {'family': 'visual_pointer',
        'target_name': 'Observed body or title region', 'verb': verb}), value, binding)
    assert visual['phase']['kind'] == 'ui_boundary'
    assert visual['phase']['step']['verb'] == verb
    assert visual['phase']['step']['target'] == {'name': 'Observed body or title region'}
    assert 'expect' not in visual['phase']['step'] and 'action_handoff' not in visual
    assert Draft202012Validator(value['response_schema']).is_valid(visual)
