"""An immediate proposal cannot silently become another family or native target."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import packet, ref, reply
from test_native_subgoal_handoff import proposal
from test_ui_observation_boundary import CELL, node

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _working_phase_schema


def _source(*, memory=True):
    value = packet(memory=memory)
    state = value['current_state']['state']
    next(item for item in state['nodes'] if item['id'] == 'draft')['operations'].append('key_chord')
    allowed = ['native_ui', *(['memory_write', 'memory_search'] if memory else [])]
    value['response_schema'] = _working_phase_schema(allowed,
        _grounded_phase_schema(allowed, value['current_state'], intent_choice=True), value['working_state'],
        observed_state=state, action_handoffs=True)
    return value


def _bound(selection=None, *, source=None, objective=None):
    source = source or _source()
    compact, binding = ui_decision.project(source)
    value = proposal()
    value['immediate_action'] = selection or {'family': 'pointer_or_activation',
                                            'target': ref(compact, CELL, binding)}
    if objective is not None:
        value['next_subgoal'] = objective
    bound, private = ui_decision.bind_subgoal(compact, binding, value, atomic_anchor=True)
    return source, bound, private, value


def _decision(source, value, action=None):
    result = reply(source, action or value['immediate_action'])
    result['action']['objective'] = value['next_subgoal']
    return result


def test_proposal_schema_derives_only_current_finite_family_target_options():
    source = _source(memory=False)
    compact, binding = ui_decision.project(source)
    _, schema = ui_decision.subgoal_context(compact, atomic_anchor=True)
    validator = Draft202012Validator(schema)
    assert 'immediate_action' in schema['required']
    value = {**proposal(), 'immediate_action': {'family': 'switch_window',
                                             'target': ref(compact, 'other', binding)}}
    validator.validate(value)
    for bad in ({'family': 'switch_window', 'target': ref(compact, CELL, binding)},
                {'family': 'pointer_or_activation', 'target': ref(compact, 'other', binding)},
                {'family': 'switch_window', 'target': 'c999'}, {'family': 'invented'},
                {'family': 'memory_write'}, {'family': 'switch_window'},
                {'family': 'pointer_or_activation', 'target': {'node_id': CELL}}):
        assert not validator.is_valid({**value, 'immediate_action': bad})
    assert not validator.is_valid(proposal())


@pytest.mark.parametrize('change', ['family', 'target', 'both'])
def test_second_stage_cannot_turn_an_editor_switch_into_a_workbench_pointer(change):
    source = _source()
    compact, mapping = ui_decision.project(source)
    selection = {'family': 'switch_window', 'target': ref(compact, 'other', mapping)}
    source, _, binding, value = _bound(selection, source=source)
    good = _decision(source, value)
    assert ui_decision.expand(good, source, binding)['phase']['steps'][0]['verb'] == 'switch_window'
    bad = copy.deepcopy(good)
    if change in {'family', 'both'}:
        bad['action']['family'] = 'pointer_or_activation'
    if change in {'target', 'both'}:
        bad['action']['target'] = ref(compact, CELL, mapping)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(bad, source, binding)


def test_same_family_cannot_substitute_a_breadcrumb_for_the_proposed_input_row():
    source = _source()
    source['current_state']['state']['nodes'].append(node('breadcrumb', 'button', 'tmp', ['click', 'double_click']))
    source, compact, binding, value = _bound(source=source)
    action = _decision(source, value)
    action['action']['target'] = next(key for key, item in binding['nodes'].items() if item['id'] == 'breadcrumb')
    assert action['action']['target'] != value['immediate_action']['target']
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(action, source, binding)
    assert compact['planning_handoff']['proposal']['next_operation'] == value['next_operation']


@pytest.mark.parametrize('family,identity,arguments,kind', [
    ('pointer_or_activation', CELL, {}, 'ui_subgoal'),
    ('text_entry', 'draft', {'text': 'Exact authored content'}, 'ui_subgoal'),
    ('text_submit', 'draft', {'text': 'Exact authored content'}, 'ui_subgoal'),
    ('switch_window', 'other', {}, 'ui'),
    ('key_chord', CELL, {'keys': ['Control_L', 'l']}, 'ui_boundary'),
    ('scroll', CELL, {'direction': 'down'}, 'ui_boundary')])
def test_permitted_arguments_expand_only_within_anchored_native_family(family, identity, arguments, kind):
    source = _source()
    compact, mapping = ui_decision.project(source)
    selection = {'family': family, 'target': ref(compact, identity, mapping)}
    source, _, binding, value = _bound(selection, source=source)
    envelope = ui_decision.expand(_decision(source, value, {**selection, **arguments}), source, binding)
    assert envelope['phase']['kind'] == kind
    if kind == 'ui_subgoal':
        assert envelope['phase']['subgoal']['family'] == family
        assert envelope['phase']['subgoal']['target'] == {'node_id': identity}
        if 'text' in arguments:
            assert envelope['phase']['subgoal']['text'] == arguments['text']
    elif kind == 'ui':
        assert envelope['phase']['steps'][0]['target'] == {'node_id': identity}
    else:
        assert envelope['phase']['step']['verb'] == family
        assert envelope['phase']['step']['args'] == arguments
    assert envelope['action_handoff']['input_authority'] is False
    assert 'task_progress' not in envelope


def test_inactive_window_anchor_preserves_its_distinct_observed_scope():
    source = _source()
    target = next(item for item in source['current_state']['state']['nodes'] if item['id'] == 'other')
    target['scope_id'] = 'other'
    compact, mapping = ui_decision.project(source)
    selection = {'family': 'switch_window', 'target': ref(compact, 'other', mapping)}
    source, _, binding, value = _bound(selection, source=source)
    anchor = binding['atomic_action_anchor']
    assert anchor['target']['scope_id'] != anchor['source_observation']['scope_id']
    envelope = ui_decision.expand(_decision(source, value), source, binding)
    assert envelope['phase']['steps'][0]['target']['node_id'] == 'other'


def test_targetless_family_retains_authored_payload_without_inventing_native_target():
    source, _, binding, value = _bound({'family': 'memory_write'})
    anchor = binding['atomic_action_anchor']
    assert anchor['selection'] == {'family': 'memory_write'} and 'target' not in anchor
    action = {'family': 'memory_write', 'topic': 'sources', 'key': 'source', 'value': 'Unverified draft note'}
    envelope = ui_decision.expand(_decision(source, value, action), source, binding)
    assert envelope['phase']['kind'] == 'memory_write'
    assert envelope['phase']['value'] == action['value']
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(_decision(source, value, {**action, 'target': 'c0'}), source, binding)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(_decision(source, value, {'family': 'memory_search', 'query': 'source'}), source, binding)


def test_explicit_replan_is_a_distinct_no_canonical_phase_control_event():
    source, compact, binding, value = _bound()
    before = copy.deepcopy((source, compact, binding))
    decision = _decision(source, value, {'family': 'replan', 'reason': 'The anchored operation needs another prerequisite.'})
    Draft202012Validator(binding['schema']).validate(decision)
    with pytest.raises(ui_decision.DecisionReplan) as caught:
        ui_decision.expand(decision, source, binding)
    assert caught.value.reason == decision['action']['reason']
    assert not isinstance(caught.value, ui_decision.DecisionError)
    assert (source, compact, binding) == before
    for reason in (' ', 'x' * 501):
        decision['action']['reason'] = reason
        with pytest.raises(ui_decision.DecisionError):
            ui_decision.expand(decision, source, binding)


@pytest.mark.parametrize('fault', ['packet_goal', 'packet_revision', 'packet_epoch', 'packet_phase',
    'binding_generation', 'binding_role', 'binding_source', 'anchor_hash', 'anchor_epoch', 'anchor_authority',
    'anchor_deleted', 'proposal_changed', 'reply_epoch', 'reply_phase'])
def test_stale_source_or_private_anchor_rejects_even_an_explicit_replan(fault):
    source, _, binding, value = _bound()
    decision = _decision(source, value, {'family': 'replan', 'reason': 'Request a fresh plan.'})
    if fault == 'packet_goal':
        source['goal'] = 'Different user task'
    elif fault == 'packet_revision':
        source['current_state']['state']['revision'] = 'new-revision'
    elif fault == 'packet_epoch':
        source['working_state']['source_epoch'] = 'new-epoch'
    elif fault == 'packet_phase':
        source['working_state']['source_phase'] += 1
    elif fault in {'binding_generation', 'binding_role'}:
        field = 'generation' if fault == 'binding_generation' else 'role'
        binding['nodes'][value['immediate_action']['target']][field] = 'changed'
    elif fault == 'binding_source':
        binding['source_observation']['revision'] = 'changed'
    elif fault == 'anchor_hash':
        binding['atomic_action_anchor_sha256'] = '0' * 64
    elif fault in {'anchor_epoch', 'anchor_authority'}:
        field = 'source_epoch' if fault == 'anchor_epoch' else 'input_authority'
        binding['atomic_action_anchor'][field] = 'other-epoch' if fault == 'anchor_epoch' else True
        binding['atomic_action_anchor_sha256'] = ui_decision._digest(binding['atomic_action_anchor'])
    elif fault == 'anchor_deleted':
        del binding['atomic_action_anchor']
    elif fault == 'proposal_changed':
        binding['planning_handoff']['proposal']['next_operation'] = 'Do something different.'
    elif fault == 'reply_epoch':
        decision['source_epoch'] = 'different'
    else:
        decision['source_phase'] += 1
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, source, binding)


def test_duplicate_native_identity_is_rejected_but_duplicate_labels_are_not_ambiguous_refs():
    source = _source()
    source['current_state']['state']['nodes'].append(copy.deepcopy(source['current_state']['state']['nodes'][0]))
    with pytest.raises(ui_decision.DecisionError):
        _bound(source=source)
    source = _source()
    duplicate = copy.deepcopy(source['current_state']['state']['nodes'][0])
    duplicate['id'] = 'different-identity-with-same-label'
    source['current_state']['state']['nodes'].append(duplicate)
    source, _, binding, value = _bound(source=source)
    envelope = ui_decision.expand(_decision(source, value), source, binding)
    assert envelope['phase']['subgoal']['target']['node_id'] == CELL


def test_anchor_is_checked_independently_of_a_broadened_downstream_schema():
    source, _, binding, value = _bound()
    original, _ = ui_decision.project(source)
    binding['schema'] = original['decision_schema']
    decision = _decision(source, value, {'family': 'switch_window',
        'target': next(key for key, item in binding['nodes'].items() if item['id'] == 'other')})
    assert Draft202012Validator(binding['schema']).is_valid(decision)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, source, binding)


@pytest.mark.parametrize('field,value', [('id', 'recycled-native-id'), ('generation', '2'),
                                        ('role', 'window')])
def test_refreshing_a_packet_hash_cannot_rebind_an_old_anchor_to_changed_native_facts(field, value):
    source, _, binding, proposal_value = _bound()
    source['current_state']['state']['nodes'][0][field] = value
    binding['packet_sha256'] = ui_decision._digest(source)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(_decision(source, proposal_value), source, binding)


def test_repeated_recovery_branches_do_not_make_the_anchor_schema_ambiguous():
    source = _source()
    offer = {'schema': 'robit.omni.pointer-recovery-proposal.v1', 'reference': 'a' * 32,
             'input_authority': False, 'choices': [{'node_id': CELL, 'verb': verb}
                for verb in ('click', 'double_click', 'right_click')]}
    source['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, source['current_state'], intent_choice=True),
        source['working_state'], pointer_recovery=offer, action_handoffs=True)
    compact, mapping = ui_decision.project(source)
    selection = {'family': 'pointer_recovery', 'target': ref(compact, CELL, mapping)}
    _, schema = ui_decision.subgoal_context(compact, atomic_anchor=True)
    Draft202012Validator(schema).validate({**proposal(), 'immediate_action': selection})
    _, _, binding, value = _bound(selection, source=source)
    for verb in ('click', 'double_click', 'right_click'):
        result = ui_decision.expand(_decision(source, value, {**selection, 'reference': offer['reference'],
            'verb': verb}), source, binding)
        assert result['phase']['step']['verb'] == verb


def test_legacy_subgoal_and_public_phase_contracts_remain_unchanged():
    source = _source()
    compact, binding = ui_decision.project(source)
    _, schema = ui_decision.subgoal_context(compact)
    assert 'immediate_action' not in schema['properties']
    _, bound = ui_decision.bind_subgoal(compact, binding, proposal())
    assert 'atomic_action_anchor' not in bound
    assert all(branch['properties']['family']['const'] != 'replan'
               for branch in bound['schema']['properties']['action']['oneOf'])
    for action in ({'family': 'pointer_or_activation', 'target': ref(compact, CELL, binding)},
                   {'family': 'switch_window', 'target': ref(compact, 'other', binding)}):
        value = _decision(source, proposal(), action)
        assert Draft202012Validator(source['response_schema']).is_valid(ui_decision.expand(value, source, bound))
    assert 'native-action-anchor' not in str(source['response_schema'])


def test_goal_words_and_unverified_screen_prose_do_not_change_anchor_selectors():
    source, compact, binding, value = _bound(objective='A deliberately unrelated wording with no UI label.')
    assert compact['planning_handoff']['proposal']['current_view'] == value['current_view']
    assert 'current_view' not in binding['atomic_action_anchor']
    assert binding['atomic_action_anchor']['input_authority'] is False
    decision = _decision(source, value)
    assert ui_decision.expand(decision, source, binding)['phase']['subgoal']['target']['node_id'] == CELL
