"""One genuine no-input deferral, exact negative routing, guarded recovery."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import ref, reply
from test_native_direct_handoff import pending_direct_packet
from test_native_family_planner import FULL_PHASES, selector
from test_native_pending_control_projection import actual_deferral
from test_native_subgoal_handoff import proposal
from test_portal_native_task_routes import GOAL, _working_reply
from test_ui_intent_choice import configured
from test_ui_observation_boundary import CELL, node

from portal import ui_decision
from portal.app import _model_ui_observation
from portal.ui_task import (
    _compact_result,
    _grounded_phase_schema,
    _NativeWorkingState,
    _working_phase_schema,
)
from qwen_omni_adapters.computer_use import VERBS, ComputerUseError


def packet_after(adapter, result, handoff, raw=None):
    raw = adapter.observe() if raw is None else raw
    retained = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    assert retained is not None
    priority, receipt = retained.consume(goal=GOAL, state=raw)
    observed = {'state': _model_ui_observation(raw, priority)}
    receipt['full_binding_retained'] = bool(priority) and priority[0] in {
        item['id'] for item in observed['state']['nodes']}
    observed['pending_recovery_projection_receipt'] = receipt
    working = _NativeWorkingState(GOAL)
    working.epoch = handoff['source_epoch']
    working.record(1, 'ui_subgoal', result, recover=True)
    view = working.planning_view(2, 0)
    packet = {'goal': GOAL, 'current_state': observed, 'last_result': _compact_result(result),
              'ledger': [], 'working_state': view, 'remaining_seconds': 30}
    rebuild_schema(packet)
    return packet


def rebuild_schema(packet, allowed=('native_ui',)):
    packet['response_schema'] = _working_phase_schema(allowed,
        _grounded_phase_schema(allowed, packet['current_state'], intent_choice=True, effect_contracts=True),
        packet['working_state'], pointer_recovery=packet['last_result']['pointer_recovery'],
        observed_state=packet['current_state']['state'], action_handoffs=True)


def expand_recovery(packet, verb, route):
    compact, binding = ui_decision.project(packet)
    prior = packet['last_result']['action_handoff']
    action = {'family': 'pointer_recovery', 'target': ref(compact, CELL, binding), 'verb': verb,
              'reference': packet['last_result']['pointer_recovery']['reference']}
    plan = {**proposal(), 'next_subgoal': prior['objective'],
            'next_operation': prior['next_operation'], 'needed_evidence': prior['needed_evidence'],
            'immediate_action': {'family': 'pointer_recovery', 'target': action['target']}}
    if route == 'family':
        selection = {**plan, 'route': 'recipe',
                     **{k: packet['working_state'][k] for k in ('source_epoch', 'source_phase')}}
        compact, binding = ui_decision.bind_family(packet, compact, binding, selection)
    else:
        compact, binding = ui_decision.bind_subgoal(compact, binding, plan, atomic_anchor=True)
    answer = reply(packet, action)
    answer['action']['objective'] = prior['objective']
    Draft202012Validator(compact['decision_schema']).validate(answer)
    return ui_decision.expand(answer, packet, binding)


def accept_recovery(packet, expanded):
    previous = packet['last_result']
    checked = ui_decision.validate_action_handoff(expanded['action_handoff'], goal=GOAL,
        state=packet['current_state']['state'], source_epoch=packet['working_state']['source_epoch'],
        source_phase=2, predecessor=previous['action_handoff'],
        recovery_reference=previous['pointer_recovery']['reference'])
    working = _NativeWorkingState(GOAL)
    working.epoch = packet['working_state']['source_epoch']
    working.accept(expanded, state=packet['current_state']['state'], previous_result=previous)
    assert working.action_handoff == checked
    assert checked['input_authority'] is checked['expected_transition']['task_progress'] is False
    return checked


def test_genuine_deferral_is_no_input_and_inactive_window_metadata_preserves_negative_marker():
    adapter, result, handoff = actual_deferral()
    assert result['ui_boundary_receipt']['input_attempted'] is False
    assert result['effect_receipt']['effect_state'] == 'absent'
    assert result['policy_receipts'][0]['policy_choice'] != 'intent-abstain'
    assert result['policy_receipts'][0]['selected_action_id'] == 'intent-abstain'
    raw = adapter.observe()
    target = copy.deepcopy(next(n for n in raw['nodes'] if n['id'] == CELL))
    raw['nodes'].append(node('inactive-editor', 'window', 'Other editor', ['switch_window']))
    raw['nodes'][-1]['scope_id'] = 'inactive-editor'
    raw['revision'] = ui_decision._digest({k: v for k, v in raw.items() if k != 'revision'})
    assert raw['revision'] != result['intent_selection_receipt']['semantic_revision']
    packet = packet_after(adapter, result, handoff, raw)
    receipt = packet['current_state']['pending_recovery_projection_receipt']
    assert receipt['pending_delegation']['semantic_revision'] == result['intent_selection_receipt']['semantic_revision']
    assert receipt['fresh_source']['revision'] == raw['revision']
    assert next(n for n in raw['nodes'] if n['id'] == CELL) == target
    assert ui_decision.project_direct(packet)[1]['pending_delegation']['node_id'] == CELL
    assert not adapter.calls


@pytest.mark.parametrize('fault', ['target', 'scope', 'session', 'goal', 'phase', 'raw_bank'])
def test_drift_or_wrong_source_cannot_admit_a_pending_restriction(fault):
    adapter, result, handoff = actual_deferral()
    raw = adapter.observe()
    if fault == 'target':
        raw['nodes'][0]['generation'] = 'replacement'
    elif fault == 'scope':
        raw['scope']['id'] = 'other-scope'
    elif fault == 'session':
        raw['session_id'] = 'other-session'
    elif fault == 'raw_bank':
        # A strategy-filtered bank may differ from the raw candidate bank.
        result['intent_selection_receipt']['intent_bank_sha256'] = 'f' * 64
        result['policy_receipts'][0]['intent_bank_sha256'] = 'f' * 64
    if fault in {'target', 'scope', 'session'}:
        pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
        assert pending is not None
        assert pending.consume(goal=GOAL, state=raw)[0] == ()
    else:
        packet = packet_after(adapter, result, handoff, raw)
        if fault == 'goal':
            packet['goal'] = 'A different goal'
        elif fault == 'phase':
            packet['working_state']['source_phase'] += 1
        assert ui_decision._pending_direct_delegation(packet) is None
    assert not adapter.calls


def test_first_family_offer_removes_only_the_pending_fast_target_and_invalid_selection_cannot_bind():
    packet, adapter, _ = pending_direct_packet()
    state = packet['current_state']['state']
    state['nodes'].append(node('other-target', 'button', 'Another observed target', ['click']))
    rebuild_schema(packet)
    compact, binding = ui_decision.project(packet)
    target_ref = ref(compact, CELL, binding)
    target = next(c for c in compact['controls'] if c['ref'] == target_ref)
    assert 'pointer_or_activation' not in target['families']
    assert 'pointer_recovery' in target['families']
    other_ref = ref(compact, 'other-target', binding)
    assert 'pointer_or_activation' in next(c for c in compact['controls'] if c['ref'] == other_ref)['families']
    context, schema = ui_decision.family_context(packet, compact)
    invalid = selector(packet, 'recipe', {'family': 'pointer_or_activation', 'target': target_ref})
    assert not Draft202012Validator(schema).is_valid(invalid)
    assert ui_decision.bind_family(packet, compact, binding, invalid) is None
    assert context['goal'] == GOAL and not adapter.calls


@pytest.mark.parametrize('phase', FULL_PHASES)
def test_full_escape_retains_original_capability_surface_under_pending_restriction(phase):
    packet, adapter, _ = pending_direct_packet()
    draft = next(n for n in packet['current_state']['state']['nodes'] if n['id'] == 'draft')
    draft.update(operations=sorted(VERBS), focused=True)
    packet['current_state']['state']['nodes'].append(node('other', 'button', 'Other', ['click']))
    rebuild_schema(packet, ('native_ui', 'memory_write', 'memory_search'))
    view, binding = ui_decision.project_direct(packet)
    # Compare with the identical source when no host pending marker qualifies.
    baseline = copy.deepcopy(packet)
    baseline['current_state']['pending_recovery_projection_receipt'].pop('pending_delegation')
    original_view, _ = ui_decision.project_direct(baseline)
    assert ui_decision.family_capabilities(view['response_schema'])['phases'] == \
        ui_decision.family_capabilities(original_view['response_schema'])['phases']
    value = _working_reply(view, phase)
    assert Draft202012Validator(view['response_schema']).is_valid(value) == \
        Draft202012Validator(original_view['response_schema']).is_valid(value)
    if Draft202012Validator(view['response_schema']).is_valid(value):
        expanded = ui_decision.expand_direct(value, packet, binding)
        assert expanded['phase'] == phase
    assert not adapter.calls


@pytest.mark.parametrize('route', ['compact', 'family'])
@pytest.mark.parametrize('verb', ['click', 'double_click', 'right_click', 'invoke'])
def test_every_actual_offered_recovery_keeps_predecessor_and_passes_existing_working_guard(route, verb):
    packet, adapter, result = pending_direct_packet()
    assert {'node_id': CELL, 'verb': verb} in result['pointer_recovery']['choices']
    expanded = expand_recovery(packet, verb, route)
    checked = accept_recovery(packet, expanded)
    prior = result['action_handoff']
    assert checked['predecessor'] == {'handoff_sha256': ui_decision._digest(prior),
        'source_phase': 1, 'recovery_reference': result['pointer_recovery']['reference']}
    assert all(checked[k] == prior[k] for k in ('objective', 'next_operation', 'needed_evidence'))
    assert expanded['phase']['step']['verb'] == verb and not adapter.calls


@pytest.mark.parametrize('field,value', [('objective', 'Another objective'),
    ('next_operation', 'A substituted operation'), ('needed_evidence', ['Different evidence']),
    ('transition', 'document_changed')])
def test_changed_recovery_intent_cannot_pass_the_existing_predecessor_guard(field, value):
    packet, adapter, _ = pending_direct_packet()
    expanded = expand_recovery(packet, 'double_click', 'family')
    if field == 'transition':
        expanded['action_handoff']['expected_transition']['transition_kind'] = value
    else:
        expanded['action_handoff'][field] = value
    with pytest.raises(ui_decision.DecisionError):
        accept_recovery(packet, expanded)
    assert not adapter.calls


def test_expanded_family_recovery_dispatches_once_through_existing_real_store_guards():
    adapter, store, cancel, requests = configured(probability=.6)
    raw = adapter.observe()
    working = _NativeWorkingState(GOAL)
    view = working.view(1, 0)
    first = {'goal': GOAL, 'current_state': {'state': raw}, 'last_result': None,
             'ledger': [], 'working_state': view, 'remaining_seconds': 30}
    first['response_schema'] = _working_phase_schema(['native_ui'],
        _grounded_phase_schema(['native_ui'], first['current_state'], intent_choice=True),
        view, observed_state=raw, action_handoffs=True)
    compact, binding = ui_decision.project(first)
    selected = {'family': 'pointer_or_activation', 'target': ref(compact, CELL, binding)}
    plan = {**proposal(), 'immediate_action': selected}
    _, bound = ui_decision.bind_subgoal(compact, binding, plan, atomic_anchor=True)
    answer = reply(first, selected)
    answer['action']['objective'] = plan['next_subgoal']
    initial = ui_decision.expand(answer, first, bound)
    with store.task_scope('session', 'desktop', cancel):
        deferred = store.intent_boundary('session', initial['phase']['subgoal'], goal=GOAL,
                                        action_handoff=initial['action_handoff'])
        assert deferred['status'] == 'abstained' and not adapter.calls
        packet = packet_after(adapter, deferred, initial['action_handoff'])
        expanded = expand_recovery(packet, 'double_click', 'family')
        accept_recovery(packet, expanded)
        phase = expanded['phase']
        result = store.resolve_pointer('session', phase['reference'], phase['step'], goal=GOAL,
            objective=phase['objective'], action_handoff=expanded['action_handoff'])
        assert result['status'] == 'observed_unverified'
        assert result['effect_receipt']['effect_state'] == 'unknown'
        assert result['task_progress'] is False and result['new_effect_count'] == 0
        with pytest.raises(ComputerUseError):
            store.resolve_pointer('session', phase['reference'], phase['step'], goal=GOAL,
                objective=phase['objective'], action_handoff=expanded['action_handoff'])
    assert len(requests) == len(adapter.calls) == 1
    assert adapter.calls[0]['verb'] == 'double_click'
