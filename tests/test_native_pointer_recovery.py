"""A slow-planner proposal resolves uncertainty, never guard or effect checks."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import packet, reply
from test_ui_intent_choice import configured, run
from test_ui_observation_boundary import CELL, intent

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _working_phase_schema, run_native_goal
from qwen_omni_adapters.computer_use import ComputerUseError

GOAL = 'Original exact user task'


def resolve(store, cancel, offer, **kw):
    with store.task_scope('session', 'desktop', cancel):
        return store.resolve_pointer('session', offer['reference'], kw.pop('step', intent('double_click')),
            goal=kw.pop('goal', GOAL), objective='Open the observed folder', **kw)


def test_one_abstention_one_slow_choice_fresh_input_unknown_effect_no_second_kev_call():
    adapter, store, cancel, requests = configured(probability=.6)
    result = run(store, cancel)
    offer = result['pointer_recovery']
    assert result['status'] == 'abstained' and not adapter.calls
    assert offer['input_authority'] is False
    fresh = resolve(store, cancel, offer)
    assert len(requests) == len(adapter.calls) == 1
    assert adapter.calls[0]['verb'] == 'double_click'
    assert fresh['status'] == 'observed_unverified'
    assert fresh['effect_receipt']['effect_state'] == 'unknown'
    assert fresh['new_effect_count'] == 0 and fresh['task_progress'] is False
    assert fresh['pointer_recovery_receipt']['policy_receipt_id'] == offer['policy_receipt_id']
    with pytest.raises(ComputerUseError):
        resolve(store, cancel, offer)
    assert len(adapter.calls) == 1


@pytest.mark.parametrize('fault', ['goal', 'reference', 'provider', 'threshold', 'scope', 'unoffered_node', 'unoffered_verb', 'task'])
def test_binding_changes_consume_offer_without_dispatch(fault):
    adapter, store, cancel, _ = configured(probability=.6)
    offer = run(store, cancel)['pointer_recovery']
    kw = {}
    if fault == 'goal':
        kw['goal'] = 'Different task'
    elif fault == 'reference':
        offer['reference'] = '0' * 32
    elif fault == 'provider':
        store.provider = copy.copy(store.provider)
    elif fault == 'threshold':
        store.provider.threshold = .5
    elif fault == 'scope':
        store._sessions[('session', 'desktop')].window_scope = {'changed': True}
    elif fault == 'unoffered_node':
        kw['step'] = {**intent('double_click'), 'target': {'node_id': 'draft'}}
    elif fault == 'unoffered_verb':
        kw['step'] = intent('focus')
    else:
        cancel = store.begin_task('session', private_boundaries=True)
    with pytest.raises(ComputerUseError):
        resolve(store, cancel, offer, **kw)
    assert not adapter.calls


@pytest.mark.parametrize('field,value', [('generation', '2'), ('bounds', {'x': 0, 'y': 0, 'width': 1, 'height': 1}),
    ('name', 'Other folder'), ('role', 'button'), ('visible', False), ('enabled', False), ('sensitive', True), ('operations', [])])
def test_target_reobservation_rejects_changed_or_ineligible_controls(field, value):
    adapter, store, cancel, _ = configured(probability=.6)
    offer = run(store, cancel)['pointer_recovery']
    observe = adapter.observe
    def changed():
        state = observe()
        state['nodes'][0][field] = value
        return state
    adapter.observe = changed
    result = resolve(store, cancel, offer)
    assert not adapter.calls and result['ui_boundary_receipt']['input_attempted'] is False
    with pytest.raises(ComputerUseError):
        resolve(store, cancel, offer)


def test_cancellation_and_public_tool_route_cannot_dispatch():
    adapter, store, cancel, _ = configured(probability=.6)
    offer = run(store, cancel)['pointer_recovery']
    with pytest.raises(ComputerUseError):
        store.act('session', {'surface': 'desktop', 'operation': 'resolve_pointer', 'reference': offer['reference']})
    cancel.set()
    assert resolve(store, cancel, offer)['status'] == 'cancelled'
    assert not adapter.calls


def test_wrong_model_and_success_do_not_offer_escalation():
    for options in [{'probability': .6, 'alter': lambda b: b.update(model='other')}, {'probability': 1.0}]:
        adapter, store, cancel, _ = configured(**options)
        assert 'pointer_recovery' not in run(store, cancel)


def recovery_packet():
    p = packet(recovery=True)
    offer = {'schema': 'robit.omni.pointer-recovery-proposal.v1', 'input_authority': False,
             'reference': 'a' * 32, 'choices': [{'node_id': CELL, 'verb': 'double_click'}]}
    p['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, p['current_state'], intent_choice=True),
        p['working_state'], pointer_recovery=offer)
    return p, offer


def test_compact_recovery_only_exposes_offered_current_pairs_and_preserves_objective():
    p, offer = recovery_packet()
    compact, binding = ui_decision.project(p)
    action = {'family': 'pointer_recovery', 'reference': offer['reference'], 'verb': 'double_click', 'target': 'c0'}
    result = ui_decision.expand(reply(p, action), p, binding)
    assert result['phase']['kind'] == 'ui_pointer_recovery' and result['phase']['step']['target'] == {'node_id': CELL}
    assert 'objective' not in result['commitment']
    for field, value in [('reference', 'b' * 32), ('verb', 'right_click'), ('target', 'c1')]:
        with pytest.raises(ui_decision.DecisionError):
            ui_decision.expand(reply(p, {**action, field: value}), p, binding)
    p['current_state']['state']['nodes'][0]['visible'] = False
    compact, _ = ui_decision.project(p)
    assert not any(b['properties']['family']['const'] == 'pointer_recovery'
                   for b in compact['decision_schema']['properties']['action']['oneOf'])


def test_controller_offers_recovery_after_abstention_and_keeps_unknown_effect_in_ledger():
    adapter, store, cancel, requests = configured(probability=.6)
    packets = []
    def compile_phase(p):
        packets.append(copy.deepcopy(p))
        compact, binding = ui_decision.project(p)
        phase = p['working_state']['source_phase']
        if phase == 1:
            action = {'family': 'pointer_or_activation', 'target': 'c0'}
        elif phase == 2:
            action = {'family': 'pointer_recovery', 'target': 'c0', 'verb': 'double_click',
                      'reference': p['last_result']['pointer_recovery']['reference']}
        else:
            action = {'family': 'blocked', 'reason': 'Only the input was observed; task not complete.'}
        proposal = ui_decision.expand(reply(p, action), p, binding)
        Draft202012Validator(p['response_schema']).validate(proposal)
        return proposal
    def execute(name, args):
        if args['operation'] == 'intent_boundary':
            return store.intent_boundary('session', args['subgoal'], goal=GOAL, feedback=args['feedback'])
        return store.resolve_pointer('session', args['reference'], args['intent'], goal=GOAL, objective=args['objective'])
    with store.task_scope('session', 'desktop', cancel):
        events = list(run_native_goal(goal=GOAL, allowed_tools={'native_ui'},
            compile_phase=compile_phase, execute=execute, observe=lambda: store.planning_observation('session'),
            cancel=cancel.set, cancelled=cancel.is_set, intent_choice=True, working_state=True))
    assert len(requests) == len(adapter.calls) == 1
    assert packets[2]['last_result']['effect_receipt']['effect_state'] == 'unknown'
    assert packets[2]['ledger'][-1]['result']['pointer_recovery_receipt']['input_authority'] is False
    assert packets[2]['working_state']['recovery_mode'] is True
    assert events[-1]['response']['portal']['native_task']['task_verified'] is False
