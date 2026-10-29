"""Immediate intent survives escalation without becoming observed evidence."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import packet, reply
from test_native_pointer_recovery import recovery_packet
from test_native_subgoal_handoff import proposal

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _working_phase_schema
from qwen_omni_adapters.ui_effects import compile_immediate_transition


def _admit_handoff(p):
    # The production private working envelope owns this additive field. Public
    # operation and target schemas remain unchanged.
    for branch in p['response_schema']['oneOf']:
        branch['properties']['action_handoff'] = ui_decision.action_handoff_schema()
    return p


def _bound_action():
    p = _admit_handoff(packet())
    compact, binding = ui_decision.project(p)
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal())
    decision = reply(p, {'family': 'pointer_or_activation', 'target': 'c0'})
    decision['action']['objective'] = proposal()['next_subgoal']
    return p, ui_decision.expand(decision, p, binding)


def _deferred():
    before, envelope = _bound_action()
    p, offer = recovery_packet()
    p['working_state']['source_epoch'] = before['working_state']['source_epoch']
    p['working_state']['source_phase'] = 2
    offer['source_revision'] = p['current_state']['state']['revision']
    p['last_result'] = {'status': 'abstained', 'reason': 'intent_policy_abstained',
        'ui_boundary_receipt': {'input_attempted': False}, 'effect_receipt': {'effect_state': 'absent'},
        'intent_selection_receipt': {'objective': envelope['action_handoff']['objective'],
            'state_revision': offer['source_revision']}, 'pointer_recovery': offer,
        'action_handoff': envelope['action_handoff']}
    p['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, p['current_state'], intent_choice=True),
        p['working_state'], pointer_recovery=offer, action_handoffs=True)
    return _admit_handoff(p)


def test_expansion_retains_exact_prerequisite_operation_as_a_source_bound_claim():
    p, envelope = _bound_action()
    handoff = envelope['action_handoff']
    assert Draft202012Validator(p['response_schema']).is_valid(envelope)
    assert handoff['objective'] == proposal()['next_subgoal']
    assert handoff['next_operation'] == proposal()['next_operation']
    assert handoff['needed_evidence'] == proposal()['needed_evidence']
    assert handoff['target']['node_id'] == p['current_state']['state']['nodes'][0]['id']
    assert handoff['authority'] == 'planner_claim_not_effect_or_input_authority'
    assert handoff['input_authority'] is False
    assert handoff['expected_transition']['mode'] == 'observe_after_input'
    assert handoff['expected_transition']['task_progress'] is False
    assert 'predicates' not in handoff['expected_transition']
    assert 'expect' not in envelope['phase']['subgoal']
    checked = ui_decision.validate_action_handoff(handoff, goal=p['goal'],
        source_epoch=p['working_state']['source_epoch'], source_phase=1,
        state=p['current_state']['state'])
    checked['needed_evidence'].clear()
    assert handoff['needed_evidence'] == proposal()['needed_evidence']


def test_private_handoff_does_not_change_unchanged_client_envelopes_or_fabricate_a_target():
    p = packet()
    for branch in p['response_schema']['oneOf']:
        branch['properties'].pop('action_handoff', None)
    compact, binding = ui_decision.project(p)
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal())
    decision = reply(p, {'family': 'pointer_or_activation', 'target': 'c0'})
    decision['action']['objective'] = proposal()['next_subgoal']
    assert 'action_handoff' not in ui_decision.expand(decision, p, binding)
    p = _admit_handoff(packet())
    compact, binding = ui_decision.project(p)
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal())
    decision = reply(p, {'family': 'visual_pointer', 'target_name': 'Absent control', 'verb': 'click'})
    decision['action']['objective'] = proposal()['next_subgoal']
    assert 'action_handoff' not in ui_decision.expand(decision, p, binding)


@pytest.mark.parametrize('fault', ['goal', 'epoch', 'phase', 'scope', 'revision', 'target',
    'generation', 'node_effect', 'authority', 'transition', 'extra', 'blank'])
def test_modified_or_misattributed_handoff_is_rejected(fault):
    p, envelope = _bound_action()
    handoff = copy.deepcopy(envelope['action_handoff'])
    state = copy.deepcopy(p['current_state']['state'])
    kwargs = {'goal': p['goal'], 'source_epoch': p['working_state']['source_epoch'],
        'source_phase': 1, 'state': state}
    if fault == 'goal':
        kwargs['goal'] = 'A different task'
    elif fault == 'epoch':
        kwargs['source_epoch'] = 'different-epoch'
    elif fault == 'phase':
        kwargs['source_phase'] = 2
    elif fault == 'scope':
        state['scope']['id'] = 'different-scope'
    elif fault == 'revision':
        state['revision'] = 'different-revision'
    elif fault == 'target':
        handoff['target']['node_id'] = 'other'
    elif fault == 'generation':
        state['nodes'][0]['generation'] = 'another-generation'
    elif fault == 'node_effect':
        state['nodes'][0]['selected'] = ['Already selected']
    elif fault == 'authority':
        handoff['input_authority'] = True
    elif fault == 'transition':
        handoff['expected_transition']['operation_family'] = 'text_entry'
    elif fault == 'extra':
        handoff['authorized_action'] = 'invented'
    else:
        handoff['next_operation'] = ' '
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.validate_action_handoff(handoff, **kwargs)


def test_pointer_escalation_receives_immediate_operation_not_just_eventual_source_objective():
    p = _deferred()
    compact, binding = ui_decision.project(p)
    before = copy.deepcopy((p, compact, binding))
    view, _, _ = ui_decision.pointer_resolution_context(compact, binding)
    claim = view['action_handoff']
    assert claim['next_operation'] == proposal()['next_operation']
    assert claim['needed_evidence'] == ['Contents of the requested source']
    assert claim['source_phase'] == 1 and compact['working_state']['source_phase'] == 2
    assert view['objective'] == proposal()['next_subgoal']
    assert view['previous_attempt']['effect_receipt']['effect_state'] == 'absent'
    assert (p, compact, binding) == before
    result, _, _ = ui_decision.resolve_pointer_proposal(compact, binding, p,
        {'choice': 'double_click_c0', 'expected_observation': 'A chooser may open before the source can be read.'})
    assert result['phase']['step']['verb'] == 'double_click'
    assert 'expect' not in result['phase']['step']
    assert 'task_progress' not in result
    rebound = result['action_handoff']
    assert rebound['source_phase'] == 2 and rebound['operation_family'] == 'pointer_recovery'
    assert rebound['next_operation'] == claim['next_operation']
    assert rebound['needed_evidence'] == claim['needed_evidence']
    assert rebound['objective'] == claim['objective']
    assert rebound['predecessor'] == {'handoff_sha256': ui_decision._digest(claim),
        'source_phase': 1, 'recovery_reference': p['last_result']['pointer_recovery']['reference']}
    assert rebound['source_observation'] == binding['source_observation']
    assert rebound['input_authority'] is False


@pytest.mark.parametrize('fault', ['older_phase', 'different_epoch', 'changed_target', 'changed_scope',
    'wrong_offer', 'unknown_effect', 'already_input', 'other_objective', 'missing_binding'])
def test_stale_or_unrelated_handoff_is_not_used_for_pointer_escalation(fault):
    p = _deferred()
    if fault == 'older_phase':
        p['last_result']['action_handoff']['source_phase'] = 0
    elif fault == 'different_epoch':
        p['last_result']['action_handoff']['source_epoch'] = 'other-epoch'
    elif fault == 'changed_target':
        p['current_state']['state']['nodes'][0]['name'] = 'Different item'
    elif fault == 'changed_scope':
        p['current_state']['state']['scope']['id'] = 'other-scope'
    elif fault == 'wrong_offer':
        p['last_result']['pointer_recovery']['reference'] = 'b' * 32
    elif fault == 'unknown_effect':
        p['last_result']['effect_receipt']['effect_state'] = 'unknown'
    elif fault == 'already_input':
        p['last_result']['ui_boundary_receipt']['input_attempted'] = True
    elif fault == 'other_objective':
        p['last_result']['intent_selection_receipt']['objective'] = 'A different local objective'
    compact, binding = ui_decision.project(p)
    view, _, _ = ui_decision.pointer_resolution_context(compact, None if fault == 'missing_binding' else binding)
    assert 'action_handoff' not in view
    assert view['choices']['double_click_c0']['verb'] == 'double_click'


def test_pointer_claim_survives_planning_only_window_projection_with_explicit_revision_provenance():
    p = _deferred()
    policy_revision = p['last_result']['pointer_recovery']['source_revision']
    p['current_state']['state']['revision'] = 'planning-with-inactive-window-roots'
    compact, binding = ui_decision.project(p)
    view, _, _ = ui_decision.pointer_resolution_context(compact, binding)
    assert view['action_handoff'] == p['last_result']['action_handoff']
    assert view['action_handoff_revalidation']['policy_revision'] == policy_revision
    assert view['action_handoff_revalidation']['current_planning_revision'] == 'planning-with-inactive-window-roots'
    assert view['action_handoff_revalidation']['input_authority'] is False


@pytest.mark.parametrize('fault', ['older_predecessor', 'future_predecessor', 'bad_reference', 'other_family',
    'changed_predecessor_hash', 'different_valid_reference', 'changed_operation_explanation'])
def test_rebound_handoff_cannot_claim_a_stale_or_nonrecovery_predecessor(fault):
    p = _deferred()
    compact, binding = ui_decision.project(p)
    result, _, _ = ui_decision.resolve_pointer_proposal(compact, binding, p,
        {'choice': 'double_click_c0', 'expected_observation': 'A chooser may open.'})
    handoff = result['action_handoff']
    if fault == 'older_predecessor':
        handoff['predecessor']['source_phase'] = 0
    elif fault == 'future_predecessor':
        handoff['predecessor']['source_phase'] = 2
    elif fault == 'bad_reference':
        handoff['predecessor']['recovery_reference'] = 'not-a-host-reference'
    elif fault == 'other_family':
        handoff['operation_family'] = 'pointer_or_activation'
        handoff['expected_transition']['operation_family'] = 'pointer_or_activation'
    elif fault == 'changed_predecessor_hash':
        handoff['predecessor']['handoff_sha256'] = '0' * 64
    elif fault == 'different_valid_reference':
        handoff['predecessor']['recovery_reference'] = 'b' * 32
    else:
        handoff['next_operation'] = 'An unrelated immediate operation'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.validate_action_handoff(handoff, goal=p['goal'], source_phase=2,
            state=p['current_state']['state'], predecessor=p['last_result']['action_handoff'],
            recovery_reference=p['last_result']['pointer_recovery']['reference'])


def test_mutating_captured_planning_handoff_cannot_change_the_dispatched_claim():
    p = _admit_handoff(packet())
    compact, binding = ui_decision.project(p)
    compact, binding = ui_decision.bind_subgoal(compact, binding, proposal())
    binding['planning_handoff']['proposal']['next_operation'] = 'Invent a different operation'
    decision = reply(p, {'family': 'pointer_or_activation', 'target': 'c0'})
    decision['action']['objective'] = proposal()['next_subgoal']
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(decision, p, binding)


@pytest.mark.parametrize('fault', [None, 'operation', 'scope', 'revision', 'session', 'transition_source'])
def test_compiled_immediate_effect_metadata_matches_the_handoff_family_and_actual_source(fault):
    p, envelope = _bound_action()
    state = p['current_state']['state']
    value = envelope['action_handoff']
    effect = compile_immediate_transition({'id': 'one', 'verb': 'double_click',
        'target': {'node_id': value['target']['node_id']}, 'args': {}}, state).as_dict()
    value['expected_transition'] = effect
    if fault == 'operation':
        effect['operation'] = 'set_value'
    elif fault == 'scope':
        effect['source_ref']['scope_id'] = 'other-scope'
    elif fault == 'revision':
        effect['source_ref']['revision'] = 'other-revision'
    elif fault == 'session':
        effect['source_ref']['session_id'] = 'other-session'
    elif fault == 'transition_source':
        source = copy.deepcopy(effect['source_ref'])
        source['revision'] = 'other-revision'
        effect['transition'] = {'kind': 'popup_opened', 'source_ref': source}
    if fault is None:
        assert ui_decision.validate_action_handoff(value, goal=p['goal'], state=state) == value
    else:
        with pytest.raises(ui_decision.DecisionError):
            ui_decision.validate_action_handoff(value, goal=p['goal'], state=state)


def test_projection_and_recovery_preserve_observed_parent_context_without_inventing_controls():
    p = _deferred()
    node = p['current_state']['state']['nodes'][0]
    node['description'] = 'A source document row'
    node['structure'] = {'parent_id': 'scope', 'child_index': 3, 'ancestors': [
        {'id': 'scope', 'generation': '1', 'role': 'window', 'name': 'Files', 'visible': True, 'enabled': True},
        {'id': 'popup', 'generation': '1', 'role': 'popup_menu', 'name': 'Recent files',
         'visible': True, 'enabled': True}]}
    compact, binding = ui_decision.project(p)
    control = compact['controls'][0]
    assert control['description'] == node['description']
    assert control['structure']['parent_ref'] == next(c['ref'] for c in compact['controls'] if c['active_window'])
    assert control['structure']['ancestors'] == node['structure']['ancestors']
    view, _, actions = ui_decision.pointer_resolution_context(compact, binding)
    assert view['choices']['double_click_c0']['control']['structure'] == control['structure']
    assert set(actions) == {'double_click_c0'}
    assert 'popup' not in actions
    assert 'parent_ref' not in node['structure']
