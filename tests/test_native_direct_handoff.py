"""Direct semantic replies cannot invent observed provenance or input authority."""
import copy
import hashlib
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_action_handoff import _deferred
from test_native_atomic_anchor import _source
from test_native_pending_control_projection import actual_deferral
from test_portal_native_task_routes import (
    SESSION_COOKIE_NAME,
    _config,
    _install_store,
    _phase,
    _post,
    _request,
    _working_reply,
)
from test_ui_observation_boundary import CELL, node

from portal import ui_decision
from portal.app import _model_ui_observation, create_app
from portal.ui_task import (
    _compact_result,
    _grounded_phase_schema,
    _NativeWorkingState,
    _working_phase_schema,
)

INTENT = {'next_operation': 'Open the observed source — 阅读原文',
          'needed_evidence': ['Its exact source text', 'Then the unchanged input bytes']}
OBJECTIVE = 'Acquire the next input, without claiming it is already open.'


def _reply(packet, phase=None, intent=INTENT):
    phase = phase or {'kind': 'ui_subgoal', 'subgoal': {'id': 'read-source',
        'family': 'pointer_or_activation', 'objective': OBJECTIVE, 'target': {'node_id': CELL}}}
    working = packet['working_state']
    result = {'assessment': {'source_epoch': working['source_epoch'], 'source_phase': working['source_phase'],
            'decision': 'recover' if working['recovery_mode'] else 'start',
            'summary': 'The source remains unacquired; this is only a proposed next operation.'},
        'phase': copy.deepcopy(phase),
        'commitment': {**({} if phase['kind'] in {'ui_subgoal', 'ui_boundary', 'ui_pointer_recovery'}
                           else {'objective': OBJECTIVE}),
            'intended_resource': 'Observed input', 'unmet_constraints': ['Read the source', 'Save the report']}}
    operation = _operation(result)
    if operation is not None:
        operation['immediate_intent'] = copy.deepcopy(intent)
    return result


def _operation(reply):
    phase = reply['phase']
    return (phase['subgoal'] if phase['kind'] == 'ui_subgoal' else phase['step']
            if phase['kind'] in {'ui_boundary', 'ui_pointer_recovery'} else
            phase['steps'][0] if phase['kind'] == 'ui' else None)


def _canonical_phase(reply):
    value = copy.deepcopy(reply)
    operation = _operation(value)
    if operation is not None:
        operation.pop('immediate_intent')
    return value['phase']


def _expand(packet, reply=None):
    _, binding = ui_decision.project_direct(packet)
    return ui_decision.expand_direct(reply or _reply(packet), packet, binding)


def test_direct_model_schema_only_replaces_opaque_proof_fields_and_preserves_source():
    packet = _source()
    original = copy.deepcopy(packet)
    view, binding = ui_decision.project_direct(packet)
    assert packet == original
    assert binding['packet_sha256'] == ui_decision._digest(packet)
    assert {k: v for k, v in view.items() if k != 'response_schema'} == {
        k: v for k, v in packet.items() if k not in {'response_schema', 'remaining_seconds'}}
    rebuilt = copy.deepcopy(view['response_schema'])
    for branch, original_branch in zip(rebuilt['oneOf'], packet['response_schema']['oneOf'], strict=True):
        assert 'action_handoff' not in branch['properties'] and 'transition' not in branch['properties']
        assert 'immediate_intent' not in branch['properties']
        for key in ('action_handoff', 'transition'):
            if key in original_branch['properties']:
                branch['properties'][key] = original_branch['properties'][key]
    rebuilt['$defs'].update({key: packet['response_schema']['$defs'][key] for key in
                            ('native_action_handoff', 'native_immediate_transition')})
    assert rebuilt['oneOf'] == packet['response_schema']['oneOf']
    for key, value in packet['response_schema']['$defs'].items():
        if key not in {'native_working_phase', 'grounded_first_step'}:
            assert rebuilt['$defs'][key] == value
    assert {'direct_native_intent', 'direct_transition_intent'} <= rebuilt['$defs'].keys()


def test_unique_native_reply_preserves_semantics_and_host_computes_exact_target_hash():
    packet = _source()
    selected = packet['current_state']['state']['nodes'][0]
    reply = _reply(packet)
    reply['phase']['subgoal']['target'] = {'role': selected['role'], 'name': selected['name']}
    expanded = _expand(packet, reply)
    assert expanded['phase'] == _canonical_phase(reply) and expanded['commitment'] == reply['commitment']
    assert 'immediate_intent' not in expanded
    claim = expanded['action_handoff']
    assert claim['objective'] == OBJECTIVE
    assert claim['next_operation'] == INTENT['next_operation']
    assert claim['needed_evidence'] == INTENT['needed_evidence']
    fields = ('id', 'generation', 'scope_id', 'role', 'name', 'visible', 'enabled', 'sensitive',
              'operations', 'bounds', 'value', 'checked', 'selected', 'focused', 'structure')
    fingerprint = hashlib.sha256(json.dumps({k: selected[k] for k in fields if k in selected},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert claim['target']['observed_target_sha256'] == fingerprint
    assert claim['target']['node_id'] == selected['id']
    assert claim['input_authority'] is False and claim['expected_transition']['task_progress'] is False
    working = _NativeWorkingState(packet['goal'])
    working.epoch = packet['working_state']['source_epoch']
    working.accept(expanded, state=packet['current_state']['state'])
    assert working.action_handoff == claim and working.commitment['objective'] == OBJECTIVE


@pytest.mark.parametrize('field', ['action_handoff', 'transition', 'forged_hash'])
def test_model_supplied_host_fields_fail_the_actual_closed_direct_schema(field):
    packet = _source()
    view, binding = ui_decision.project_direct(packet)
    value = _reply(packet)
    value[field] = {'input_authority': True, 'sha256': '0' * 64}
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('fault', ['goal', 'epoch', 'phase', 'session', 'scope', 'revision',
    'generation', 'role', 'name', 'operations', 'binding', 'schema'])
def test_changed_captured_source_or_binding_cannot_be_rebound(fault):
    packet = _source()
    _, binding = ui_decision.project_direct(packet)
    value = _reply(packet)
    state = packet['current_state']['state']
    if fault == 'goal':
        packet['goal'] += ' changed'
    elif fault in {'epoch', 'phase'}:
        packet['working_state']['source_' + fault] = 'changed' if fault == 'epoch' else 2
    elif fault == 'session':
        state['session_id'] = 'other session'
    elif fault == 'scope':
        state['scope']['id'] = 'other scope'
    elif fault == 'revision':
        state['revision'] = 'other revision'
    elif fault in {'generation', 'role', 'name'}:
        state['nodes'][0][fault] = 'changed'
    elif fault == 'operations':
        state['nodes'][0]['operations'] = []
    else:
        binding['packet_sha256' if fault == 'binding' else 'schema_sha256'] = '0' * 64
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('fault', ['extra', 'empty', 'blank', 'missing', 'too_many', 'transition', 'objective_twice'])
def test_invalid_model_semantics_or_duplicate_objective_never_compile(fault):
    packet = _source()
    value = _reply(packet)
    if fault == 'extra':
        _operation(value)['immediate_intent']['input_authority'] = True
    elif fault == 'empty':
        _operation(value)['immediate_intent']['next_operation'] = ''
    elif fault == 'blank':
        _operation(value)['immediate_intent']['needed_evidence'] = [' ']
    elif fault == 'missing':
        _operation(value).pop('immediate_intent')
    elif fault == 'too_many':
        _operation(value)['immediate_intent']['needed_evidence'] = ['x'] * 7
    elif fault == 'transition':
        _operation(value)['immediate_intent']['transition_kind'] = 'file_saved'
    else:
        value['commitment']['objective'] = OBJECTIVE
    with pytest.raises(ui_decision.DecisionError):
        _expand(packet, value)


def test_ambiguous_selector_never_silently_selects_first_and_exact_id_stays_available():
    packet = _source()
    state = packet['current_state']['state']
    selected = state['nodes'][0]
    duplicate = {**copy.deepcopy(selected), 'id': 'second-same-label'}
    state['nodes'].append(duplicate)
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True),
        packet['working_state'], observed_state=state, action_handoffs=True)
    value = _reply(packet)
    value['phase']['subgoal']['target'] = {'role': selected['role'], 'name': selected['name']}
    with pytest.raises(ui_decision.DecisionReplan):
        _expand(packet, value)
    assert _expand(packet)['action_handoff']['target']['node_id'] == CELL


@pytest.mark.parametrize('field,value', [('visible', False), ('enabled', False), ('sensitive', True), ('operations', [])])
def test_nonpositive_selected_control_cannot_gain_handoff(field, value):
    packet = _source()
    # Even a stale grounded schema cannot make nonpositive evidence eligible.
    packet['current_state']['state']['nodes'][0][field] = value
    with pytest.raises(ui_decision.DecisionReplan):
        _expand(packet)


def test_switch_preserves_inactive_window_identity_and_its_distinct_scope():
    packet = _source()
    selected = next(n for n in packet['current_state']['state']['nodes'] if n['id'] == 'other')
    selected['scope_id'] = 'other'
    phase = {'kind': 'ui', 'steps': [{'id': 'switch', 'verb': 'switch_window',
        'target': {'role': 'window', 'name': 'Editor'}, 'args': {},
        'expect': [{'kind': 'active_window', 'node_id': '$target'}]}],
        'final_predicates': [{'kind': 'active_window', 'node_id': 'other'}]}
    result = _expand(packet, _reply(packet, phase))
    assert result['phase'] == phase
    assert result['action_handoff']['target']['scope_id'] == 'other'
    assert result['action_handoff']['operation_family'] == 'switch_window'


@pytest.mark.parametrize('verb,identity,args,predicate', [
    ('focus', 'draft', {}, {'kind': 'focused', 'node_id': '$target'}),
    ('select', 'selection', {'option': 'Input'}, {'kind': 'selected', 'node_id': '$target', 'values': ['Input']}),
    ('set_checked', 'check', {'checked': True}, {'kind': 'checked', 'node_id': '$target', 'checked': True})])
def test_operations_without_existing_handoff_family_keep_canonical_route(verb, identity, args, predicate):
    packet = _source()
    state = packet['current_state']['state']
    state['nodes'] += [node('selection', 'list', 'Selection', ['select']),
                       node('check', 'checkbox', 'Option', ['set_checked'])]
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True),
        packet['working_state'], observed_state=state, action_handoffs=True)
    phase = {'kind': 'ui', 'steps': [{'id': verb, 'verb': verb, 'target': {'node_id': identity},
        'args': args, 'expect': [predicate]}], 'final_predicates': [{**predicate, 'node_id': identity}]}
    result = _expand(packet, _reply(packet, phase, None))
    assert result['phase'] == phase and 'action_handoff' not in result


def test_visual_missing_name_retains_existing_discovery_without_invented_identity():
    packet = _source()
    phase = {'kind': 'ui_boundary', 'objective': OBJECTIVE,
        'step': {'id': 'discover', 'verb': 'click', 'target': {'name': 'Unobserved canvas feature'}, 'args': {}}}
    result = _expand(packet, _reply(packet, phase, None))
    assert result['phase'] == phase and 'action_handoff' not in result


def test_transition_is_semantic_kind_then_host_binds_source_without_claimed_effect():
    packet = _source()
    result = _expand(packet, _reply(packet, intent={**INTENT, 'transition_kind': 'popup_opened'}))
    state = packet['current_state']['state']
    assert result['transition'] == {'kind': 'popup_opened', 'source_ref': {
        'session_id': state['session_id'], 'scope_id': state['scope']['id'], 'revision': state['revision']}}
    assert result['action_handoff']['expected_transition']['task_progress'] is False


def _recovery():
    packet = _deferred()
    packet['working_state']['recovery_mode'] = True
    prior = packet['last_result']['action_handoff']
    offer = packet['last_result']['pointer_recovery']
    choice = offer['choices'][0]
    phase = {'kind': 'ui_pointer_recovery', 'reference': offer['reference'], 'objective': prior['objective'],
        'step': {'id': 'recover', 'verb': choice['verb'], 'target': {'node_id': choice['node_id']}, 'args': {}}}
    intent = {k: copy.deepcopy(prior[k]) for k in ('next_operation', 'needed_evidence')}
    return packet, _reply(packet, phase, intent)


def test_recovery_preserves_exact_predecessor_and_never_renews_original_intent():
    packet, value = _recovery()
    result = _expand(packet, value)
    prior = packet['last_result']['action_handoff']
    current = result['action_handoff']
    assert current['predecessor'] == {'handoff_sha256': ui_decision._digest(prior),
        'source_phase': prior['source_phase'], 'recovery_reference': value['phase']['reference']}
    for key in ('objective', 'next_operation', 'needed_evidence'):
        assert current[key] == prior[key]


def _transition_recovery(kind=None):
    packet, value = _recovery()
    state = packet['current_state']['state']
    offer = packet['last_result']['pointer_recovery']
    if kind is not None:
        packet['last_result']['action_handoff']['expected_transition']['transition_kind'] = kind
        offer['transition'] = {'kind': kind, 'source_ref': {
            'session_id': state['session_id'], 'scope_id': state['scope']['id'], 'revision': state['revision']}}
        _operation(value)['immediate_intent']['transition_kind'] = kind
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True, effect_contracts=True),
        packet['working_state'], pointer_recovery=offer, observed_state=state, action_handoffs=True)
    return packet, value


@pytest.mark.parametrize('kind', [None, *ui_decision._TRANSITION_KINDS])
def test_recovery_schema_matches_full_canonical_expansion_and_working_acceptance(kind):
    packet, value = _transition_recovery(kind)
    original = copy.deepcopy(packet)
    view, binding = ui_decision.project_direct(packet)
    Draft202012Validator(view['response_schema']).validate(value)
    result = ui_decision.expand_direct(value, packet, binding)
    Draft202012Validator(packet['response_schema']).validate(result)
    working = _NativeWorkingState(packet['goal'])
    working.epoch = packet['working_state']['source_epoch']
    working.accept(result, state=packet['current_state']['state'], previous_result=packet['last_result'])
    claim, prior = working.action_handoff, packet['last_result']['action_handoff']
    assert claim['expected_transition'].get('transition_kind') == kind
    assert claim['input_authority'] is claim['expected_transition']['task_progress'] is False
    assert result['phase'] == _canonical_phase(value)
    for key in ('objective', 'next_operation', 'needed_evidence', 'target'):
        assert claim[key] == prior[key]
    assert packet == original


@pytest.mark.parametrize('prior_kind', [None, *ui_decision._TRANSITION_KINDS])
@pytest.mark.parametrize('supplied', ['omitted', None, 'unknown', *ui_decision._TRANSITION_KINDS])
def test_recovery_kind_schema_rejects_every_predecessor_incompatible_value(prior_kind, supplied):
    packet, value = _transition_recovery(prior_kind)
    intent = _operation(value)['immediate_intent']
    if supplied == 'omitted':
        intent.pop('transition_kind', None)
    else:
        intent['transition_kind'] = supplied
    view, binding = ui_decision.project_direct(packet)
    compatible = supplied == 'omitted' if prior_kind is None else supplied == prior_kind
    assert Draft202012Validator(view['response_schema']).is_valid(value) is compatible
    if not compatible:
        with pytest.raises(ui_decision.DecisionError):
            ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('malformed', [None, 'unknown', False, 1, {}, []])
def test_malformed_retained_recovery_kind_cannot_become_known_absence(malformed):
    packet, _ = _transition_recovery()
    packet['last_result']['action_handoff']['expected_transition']['transition_kind'] = malformed
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.project_direct(packet)


def test_recovery_specific_definition_isolated_and_missing_predecessor_stays_guarded():
    plain = _source()
    plain_view, plain_binding = ui_decision.project_direct(plain)
    for kind in [None, *ui_decision._TRANSITION_KINDS, None]:
        packet, _ = _transition_recovery(kind)
        view, _ = ui_decision.project_direct(packet)
        for name in ('direct_native_intent', 'direct_transition_intent'):
            assert view['response_schema']['$defs'][name] == plain_view['response_schema']['$defs'][name]
        assert ui_decision.project_direct(plain) == (plain_view, plain_binding)
    packet, value = _transition_recovery()
    packet['last_result'].pop('action_handoff')
    view, binding = ui_decision.project_direct(packet)
    assert 'direct_recovery_intent' not in view['response_schema']['$defs']
    Draft202012Validator(view['response_schema']).validate(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


def pending_direct_packet():
    from test_portal_native_task_routes import GOAL
    adapter, result, handoff = actual_deferral()
    raw = adapter.observe()
    pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    priority, receipt = pending.consume(goal=GOAL, state=raw)
    observed = {'state': _model_ui_observation(raw, priority)}
    receipt['full_binding_retained'] = CELL in {n['id'] for n in observed['state']['nodes']}
    observed['pending_recovery_projection_receipt'] = receipt
    working = _NativeWorkingState(GOAL)
    working.epoch = handoff['source_epoch']
    working.record(1, 'ui_subgoal', result, recover=True)
    view = working.planning_view(2, 0)
    packet = {'goal': GOAL, 'current_state': observed, 'last_result': _compact_result(result),
        'ledger': [], 'working_state': view, 'remaining_seconds': 30}
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, observed, intent_choice=True, effect_contracts=True), view,
        pointer_recovery=result['pointer_recovery'], observed_state=observed['state'], action_handoffs=True)
    return packet, adapter, result


def test_actual_pending_delegation_is_not_another_fast_choice_but_all_recoveries_compile():
    packet, adapter, result = pending_direct_packet()
    original = copy.deepcopy(packet)
    view, binding = ui_decision.project_direct(packet)
    assert binding['pending_delegation']['node_id'] == CELL
    prior = result['action_handoff']
    repeated = _reply(packet, {'kind': 'ui_subgoal', 'subgoal': {'id': 'repeat',
        'family': 'pointer_or_activation', 'target': {'node_id': CELL}, 'objective': prior['objective']}},
        {k: prior[k] for k in ('next_operation', 'needed_evidence')})
    assert not Draft202012Validator(view['response_schema']).is_valid(repeated)
    for choice in result['pointer_recovery']['choices']:
        phase = {'kind': 'ui_pointer_recovery', 'reference': result['pointer_recovery']['reference'],
            'objective': prior['objective'], 'step': {'id': 'resolve', 'verb': choice['verb'],
                'target': {'node_id': CELL}, 'args': {}}}
        value = _reply(packet, phase, {k: prior[k] for k in ('next_operation', 'needed_evidence')})
        expanded = ui_decision.expand_direct(value, packet, binding)
        Draft202012Validator(packet['response_schema']).validate(expanded)
        assert expanded['phase']['step']['verb'] == choice['verb']
        assert expanded['action_handoff']['input_authority'] is False
    assert packet == original and not adapter.calls


@pytest.mark.parametrize('alias', ['exact_boundary', 'named_boundary', 'suffix', 'predicted_mutation_suffix'])
def test_pending_alias_and_every_pointer_suffix_replan_before_any_prefix(alias):
    packet, adapter, _ = pending_direct_packet()
    target = {'node_id': CELL} if alias == 'exact_boundary' else {'role': 'table_cell', 'name': 'Input'}
    pointer = {'id': 'again', 'verb': 'double_click', 'target': target, 'args': {}}
    if alias in {'suffix', 'predicted_mutation_suffix'}:
        first = {'id': 'prefix', 'verb': 'set_value', 'target': {'node_id': 'draft'},
            'args': {'value': 'new' if alias == 'predicted_mutation_suffix' else ''},
            'expect': [{'kind': 'value', 'node_id': '$target', 'value': 'new' if alias == 'predicted_mutation_suffix' else ''}]}
        phase = {'kind': 'ui', 'steps': [first, {**pointer, 'expect': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]}],
            'final_predicates': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]}
    else:
        phase = {'kind': 'ui_boundary', 'objective': OBJECTIVE, 'step': pointer}
    value = _reply(packet, phase)
    # A positive exact-ID boundary was never offered by grounded intent choice;
    # the named forms and generated suffix must be caught by phase admission.
    if alias == 'exact_boundary':
        with pytest.raises(ui_decision.DecisionError):
            _expand(packet, value)
    else:
        view, binding = ui_decision.project_direct(packet)
        Draft202012Validator(view['response_schema']).validate(value)
        with pytest.raises(ui_decision.DecisionReplan):
            ui_decision.expand_direct(value, packet, binding)
    assert not adapter.calls


@pytest.mark.parametrize('fault', ['missing_marker', 'bank', 'revision', 'reference', 'phase', 'scope',
    'target', 'confirmed', 'extra_marker', 'already_input'])
def test_only_current_host_pending_bank_can_restrict_the_next_direct_phase(fault):
    packet, _, _ = pending_direct_packet()
    receipt = packet['current_state']['pending_recovery_projection_receipt']
    if fault == 'missing_marker':
        receipt.pop('pending_delegation')
    elif fault in {'bank', 'revision'}:
        receipt['pending_delegation']['intent_bank_sha256' if fault == 'bank' else 'semantic_revision'] = 'f' * 64
    elif fault == 'reference':
        receipt['reference'] = 'f' * 32
    elif fault == 'phase':
        packet['working_state']['source_phase'] += 1
    elif fault == 'scope':
        packet['current_state']['state']['scope']['id'] = 'foreign'
    elif fault == 'target':
        packet['current_state']['state']['nodes'][0]['generation'] = 'replacement'
    elif fault == 'confirmed':
        packet['last_result']['policy_receipts'][0]['model_confirmed'] = False
    elif fault == 'extra_marker':
        receipt['pending_delegation']['force_verb'] = 'double_click'
    else:
        packet['last_result']['ui_boundary_receipt']['input_attempted'] = True
    assert 'pending_delegation' not in ui_decision.project_direct(packet)[1]


def test_pending_keeps_other_targets_nonpointer_suffix_and_block_inspection_available():
    packet, adapter, _ = pending_direct_packet()
    other = node('unrelated-control', 'button', 'Other', ['click', 'double_click'])
    packet['current_state']['state']['nodes'].append(other)
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True, effect_contracts=True),
        packet['working_state'], pointer_recovery=packet['last_result']['pointer_recovery'],
        observed_state=packet['current_state']['state'], action_handoffs=True)
    view, binding = ui_decision.project_direct(packet)
    assert 'pending_delegation' in binding
    other_reply = _reply(packet, {'kind': 'ui_subgoal', 'subgoal': {'id': 'other',
        'family': 'pointer_or_activation', 'target': {'node_id': other['id']}, 'objective': OBJECTIVE}})
    assert ui_decision.expand_direct(other_reply, packet, binding)['phase']['subgoal']['target'] == {'node_id': other['id']}
    for phase in [{'kind': 'visual_inspection'}, {'kind': 'blocked', 'reason': 'Still uncertain'}]:
        reply = _working_reply(view, phase)
        assert ui_decision.expand_direct(reply, packet, binding)['phase'] == phase
    for suffix in [{'id': 'focus-later', 'verb': 'focus', 'target': {'node_id': 'draft'}, 'args': {},
                    'expect': [{'kind': 'focused', 'node_id': '$target'}]},
                   {'id': 'other-later', 'verb': 'click', 'target': {'node_id': other['id']}, 'args': {},
                    'expect': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]}]:
        phase = {'kind': 'ui', 'steps': [{'id': 'empty', 'verb': 'set_value', 'target': {'node_id': 'draft'},
            'args': {'value': ''}, 'expect': [{'kind': 'value', 'node_id': '$target', 'value': ''}]}, suffix],
            'final_predicates': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]}
        value = _reply(packet, phase)
        assert ui_decision.expand_direct(value, packet, binding)['phase'] == _canonical_phase(value)
    assert not adapter.calls


def test_pending_only_pointer_target_removes_phase_without_dangling_refs_or_empty_choices():
    packet, _, result = pending_direct_packet()
    state = packet['current_state']['state']
    state['nodes'] = [node for node in state['nodes'] if node['id'] == CELL]
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True, effect_contracts=True),
        packet['working_state'], pointer_recovery=result['pointer_recovery'], observed_state=state, action_handoffs=True)
    view, binding = ui_decision.project_direct(packet)
    kinds = [phase['properties']['kind']['const'] for phase in view['response_schema']['$defs']['native_working_phase']['oneOf']]
    assert 'ui_subgoal' not in kinds
    assert {'ui_boundary', 'ui_pointer_recovery', 'visual_inspection'} <= set(kinds)
    for phase in [{'kind': 'visual_inspection'}, {'kind': 'blocked', 'reason': 'Uncertain'}]:
        value = _working_reply(view, phase)
        assert ui_decision.expand_direct(value, packet, binding)['phase'] == phase
    assert len(json.dumps(view['response_schema'], ensure_ascii=False).encode()) < 56 * 1024


@pytest.mark.parametrize('fault', ['intent', 'predecessor', 'epoch', 'reference', 'target', 'generation', 'input'])
def test_foreign_stale_or_rewritten_recovery_rejects(fault):
    packet, value = _recovery()
    last = packet['last_result']
    if fault == 'intent':
        _operation(value)['immediate_intent']['next_operation'] = 'A newly invented operation'
    elif fault == 'predecessor':
        last.pop('action_handoff')
    elif fault == 'epoch':
        last['action_handoff']['source_epoch'] = 'foreign'
    elif fault == 'reference':
        last['pointer_recovery']['reference'] = '0' * 32
    elif fault == 'target':
        last['action_handoff']['target']['node_id'] = 'other'
    elif fault == 'generation':
        packet['current_state']['state']['nodes'][0]['generation'] = 'new'
    else:
        last['ui_boundary_receipt']['input_attempted'] = True
    with pytest.raises(ui_decision.DecisionError):
        _expand(packet, value)


@pytest.mark.parametrize('stream', [False, True])
def test_actual_direct_portal_sends_semantic_schema_then_preserves_canonical_execution(tmp_path, monkeypatch, stream):
    store, adapters = _install_store(monkeypatch)
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '0')
    packets = []
    def upstream(request):
        body = json.loads(request.content)
        packet = json.loads(body['messages'][1]['content'])
        packets.append(packet)
        assert packet['response_schema'] == body['response_format']['json_schema']['schema']
        assert 'immediate_intent' not in packet['response_schema']['oneOf'][0]['required']
        assert 'direct_native_intent' in packet['response_schema']['$defs']
        assert 'native_action_handoff' not in packet['response_schema']['$defs']
        assert body['think'] is False and body['options'] == _request(stream)['options']
        phase = _phase() if len(packets) == 1 else {'kind': 'blocked', 'reason': 'Independent verification still owed'}
        return httpx.Response(200, json={'message': {'content': json.dumps(_working_reply(request, phase))}})
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, 'direct-host-handoff-owned-session-12345')
    result, _ = _post(client, _request(stream), stream)
    assert len(packets) == 2 and adapters[0].value == 'Reference note'
    assert [action['verb'] for action in adapters[0].calls] == ['set_value', 'invoke']
    assert result['portal']['native_task']['task_verified'] is False
    assert store._owned('direct-host-handoff-owned-session-12345', 'desktop').cancel.is_set()


def test_targetless_direct_phase_is_canonical_without_input_claim():
    packet = _source()
    value = _reply(packet, {'kind': 'visual_inspection'}, None)
    assert _expand(packet, value) == value


@pytest.mark.parametrize('fault', ['legacy_proof', 'missing_intent', 'unknown_target'])
def test_actual_portal_rejects_invalid_semantic_reply_at_unchanged_three_failure_boundary(tmp_path, monkeypatch, fault):
    _, adapters = _install_store(monkeypatch)
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '0')
    requests = []
    def upstream(request):
        packet = json.loads(json.loads(request.content)['messages'][1]['content'])
        requests.append(packet)
        value = _working_reply(request, _phase())
        if fault == 'legacy_proof':
            value['action_handoff'] = {'target': {'observed_target_sha256': '0' * 64}}
        elif fault == 'missing_intent':
            _operation(value).pop('immediate_intent')
        else:
            value['phase']['steps'][0]['target'] = {'node_id': 'invented'}
        return httpx.Response(200, json={'message': {'content': json.dumps(value)}})
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, 'direct-invalid-handoff-owned-session-12345')
    result, _ = _post(client, _request(False), False)
    assert len(requests) == 3 and not adapters[0].calls
    assert result['portal']['native_task']['status'] == 'blocked'
    assert all(p['last_result']['error'] == 'invalid_native_phase_json' for p in requests[1:])
