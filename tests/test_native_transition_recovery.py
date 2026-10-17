"""Successful physical transitions cannot hide a repeated-state task cycle."""
import copy
import json

from test_native_working_state import envelope
from test_ui_task import Fixture, WindowCycleAdapter, completed_tools, final


def switch(target, index):
    return {'kind': 'ui', 'steps': [{'id': f'switch-{index}', 'verb': 'switch_window',
        'target': {'node_id': target}, 'args': {},
        'expect': [{'kind': 'active_window', 'node_id': '$target'}]}],
        'final_predicates': [{'kind': 'active_window', 'node_id': target}],
        'remaining_criteria': ['Read input', 'Produce and verify deliverables']}


def cycle_fixture(targets):
    fixture = Fixture([])
    fixture.adapter = WindowCycleAdapter()
    plans = iter(switch(target, i) for i, target in enumerate(targets))
    def compile_phase(packet):
        fixture.packets.append(copy.deepcopy(packet))
        return envelope(packet, next(plans))
    fixture.compile = compile_phase
    return fixture


def test_complete_window_cycle_enters_recovery_and_exhausts_bounded_repeats():
    fixture = cycle_fixture(['right', 'left', 'right', 'left', 'right'])
    events = list(fixture.run(working_state=True))
    tools = completed_tools(events)
    assert [tool['ok'] for tool in tools] == [True, True, False, False, False]
    assert final(events)['portal']['native_task']['status'] == 'blocked'
    assert final(events)['portal']['native_task']['task_verified'] is False
    assert len(fixture.adapter.calls) == 5
    assert fixture.packets[3]['working_state']['recovery_mode'] is True
    assert fixture.packets[3]['working_state']['consecutive_unverified_count'] == 1
    assert fixture.packets[4]['working_state']['consecutive_unverified_count'] == 2
    for tool in tools[2:]:
        result = json.loads(tool['result'])
        assert result['status'] == 'complete'  # Preserve the true input/effect result.
        assert result['error'] == 'ui_phase_repeated_transition'
        assert result['outcomes'][0]['status'] == 'verified'
        assert result['outcomes'][0]['effect_receipt']['effect_state'] == 'applied'


def test_new_observed_state_after_repeat_allows_recovery_without_erasing_criteria():
    fixture = cycle_fixture(['right', 'left', 'right', 'left', 'right'])
    fixture.adapter.version = 0
    observe = fixture.adapter.observe
    def changed_observation():
        state = observe()
        state['revision'] += ':' + str(fixture.adapter.version)
        return state
    fixture.adapter.observe = changed_observation
    execute = fixture.execute
    def new_external_state(name, arguments):
        result = execute(name, arguments)
        if len(fixture.executions) == 3:
            # A real external UI change becomes observable after the repeat.
            fixture.adapter.version += 1
        return result
    fixture.execute = new_external_state
    events = list(fixture.run(working_state=True, max_phases=5))
    assert [tool['ok'] for tool in completed_tools(events)] == [True, True, False, True, True]
    assert fixture.packets[3]['working_state']['recovery_mode'] is True
    assert fixture.packets[4]['working_state']['recovery_mode'] is False
    assert fixture.packets[4]['working_state']['consecutive_unverified_count'] == 0
    assert final(events)['portal']['native_task']['remaining_criteria'] == ['Read input', 'Produce and verify deliverables']
    assert final(events)['portal']['native_task']['task_verified'] is False


def test_repeated_result_carries_host_feedback_to_the_next_planner():
    fixture = cycle_fixture(['right', 'left', 'right', 'left'])
    events = list(fixture.run(working_state=True, max_phases=4))
    last = fixture.packets[3]['last_result']
    assert last['error'] == 'ui_phase_repeated_transition'
    assert 'previously verified' in last['reason']
    assert fixture.packets[3]['ledger'][-1]['ok'] is False
    assert final(events)['portal']['native_task']['status'] == 'phase_budget_exceeded'
