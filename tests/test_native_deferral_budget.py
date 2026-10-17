"""Fast uncertainty is one pending attempt, never an unlimited retry exemption."""
import copy

import pytest
from test_native_compact_decision import reply
from test_ui_intent_choice import configured

from portal import ui_decision
from portal.ui_task import run_native_goal


@pytest.mark.parametrize('mode', ['resolved_chain', 'ignore_recovery', 'failed_recovery', 'invalid_policy'])
def test_controller_distinguishes_pending_slow_resolution_from_failed_attempt(mode):
    adapter, store, cancel, requests = configured(probability=.6,
        alter=(lambda body: body.update(model='wrong')) if mode == 'invalid_policy' else None)
    # The adapter exposes a different actual native generation after each input;
    # no objective text or random offer token can establish this state change.
    original = adapter.observe
    def observe():
        state = original()
        state['revision'] += str(len(adapter.calls))
        state['nodes'][0]['generation'] = str(len(adapter.calls))
        return state
    adapter.observe = observe
    packets = []
    def compile_phase(p):
        packets.append(copy.deepcopy(p))
        compact, binding = ui_decision.project(p)
        phase = p['working_state']['source_phase']
        offer = (p.get('last_result') or {}).get('pointer_recovery')
        if phase == 9:
            action = {'family': 'blocked', 'reason': 'Observe chain only; original task remains incomplete.'}
        elif offer and mode in {'resolved_chain', 'failed_recovery'}:
            action = {'family': 'pointer_recovery', 'reference': offer['reference'], 'target': 'c0', 'verb': 'double_click'}
        else:
            action = {'family': 'pointer_or_activation', 'target': 'c0'}
        return ui_decision.expand(reply(p, action), p, binding)
    def execute(name, args):
        if args['operation'] == 'intent_boundary':
            return store.intent_boundary('session', args['subgoal'], goal='Original task', feedback=args['feedback'])
        if mode == 'failed_recovery':
            saved = adapter.observe
            def stale():
                state = saved()
                state['nodes'][0]['generation'] += 'stale'
                return state
            adapter.observe = stale
            try:
                return store.resolve_pointer('session', args['reference'], args['intent'], goal='Original task', objective=args['objective'])
            finally:
                adapter.observe = saved
        return store.resolve_pointer('session', args['reference'], args['intent'], goal='Original task', objective=args['objective'])
    with store.task_scope('session', 'desktop', cancel):
        events = list(run_native_goal(goal='Original task', allowed_tools={'native_ui'}, compile_phase=compile_phase,
            execute=execute, observe=lambda: store.planning_observation('session'), cancel=cancel.set,
            cancelled=cancel.is_set, working_state=True, intent_choice=True))
    final = events[-1]['response']['portal']['native_task']
    assert final['task_verified'] is False
    if mode == 'resolved_chain':
        assert len(adapter.calls) == len(requests) == 4 and final['phases'] == 9
        assert all(p['working_state']['consecutive_unverified_count'] == 0 for p in packets)
        assert all(row['result'].get('ui_boundary_receipt', {}).get('task_progress') is False for row in final['ledger'])
        assert [r['phase'] for r in final['ledger'] if r['deferred_to_planner']] == [1, 3, 5, 7]
    else:
        assert not adapter.calls and final['status'] == 'blocked'
        assert final['phases'] == {'ignore_recovery': 4, 'failed_recovery': 6, 'invalid_policy': 3}[mode]
