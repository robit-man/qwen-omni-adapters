"""Narrow planner recovery keeps exact task authority and host execution guards."""
import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import reply
from test_native_pointer_recovery import recovery_packet
from test_native_subgoal_handoff import proposal
from test_portal_native_task_routes import _config, _post, _request
from test_ui_intent_choice import configured

import portal.tools as tools_module
from portal import ui_decision
from portal.app import create_app


def source():
    p, offer = recovery_packet()
    p['last_result']['intent_selection_receipt'] = {'objective': 'Open the observed Input folder'}
    p['working_state']['commitment'] = {'unmet_constraints': ['Read source', 'Write and verify report']}
    return p, *ui_decision.project(p)


def test_small_context_keeps_goal_local_objective_and_offered_input_pairs():
    p, compact, binding = source()
    view, schema, actions = ui_decision.pointer_resolution_context(compact)
    assert view['goal'] == p['goal'] and view['objective'] == p['last_result']['intent_selection_receipt']['objective']
    assert 'historical_ledger' not in view and 'working_state' not in view and 'controls' not in view
    assert view['active_window'] == {'name': 'Files', 'already_foreground': True}
    assert set(actions) == {'double_click_c0'} and schema['properties']['choice']['enum'] == ['double_click_c0', 'replan']
    value = {'choice': 'double_click_c0', 'expected_observation': 'The folder contents may appear.'}
    phase, _, _ = ui_decision.resolve_pointer_proposal(compact, binding, p, value)
    Draft202012Validator(p['response_schema']).validate(phase)
    assert phase['phase']['kind'] == 'ui_pointer_recovery'
    assert phase['phase']['objective'] == view['objective']
    assert 'Unverified planner prediction' in phase['assessment']['summary']
    assert phase['phase']['remaining_criteria'] == ['Read source', 'Write and verify report']
    assert 'expect' not in phase['phase']['step']


@pytest.mark.parametrize('fault', ['choice', 'blank', 'extra', 'compact_changed', 'packet_changed'])
def test_resolution_cannot_change_binding_or_invent_an_operation(fault):
    p, compact, binding = source()
    value = {'choice': 'double_click_c0', 'expected_observation': 'The folder contents may appear.'}
    if fault == 'choice':
        value['choice'] = 'click_c0'
    elif fault == 'blank':
        value['expected_observation'] = ' '
    elif fault == 'extra':
        value['input_authority'] = True
    elif fault == 'compact_changed':
        compact['goal'] = 'Replacement'
    else:
        p['goal'] = 'Replacement'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.resolve_pointer_proposal(compact, binding, p, value)


def test_replan_feedback_does_not_erase_failures_or_repeat_the_declined_resolution():
    p, compact, binding = source()
    original = copy.deepcopy((p, compact, binding))
    phase, view, bound = ui_decision.resolve_pointer_proposal(compact, binding, p,
        {'choice': 'replan', 'expected_observation': 'No input; this target cannot open the requested source.'})
    assert phase is None and (p, compact, binding) == original
    assert view['working_state'] == compact['working_state'] and view['historical_ledger'] == compact['historical_ledger']
    assert view['last_result']['slow_pointer_resolution']['authority'] == 'planner_prediction_not_observed_effect'
    assert ui_decision.pointer_resolution_context(view) is None
    _, bound = ui_decision.bind_subgoal(view, bound, proposal())
    decision = reply(p, {'family': 'blocked', 'reason': 'No applicable action'})
    decision['action']['objective'] = proposal()['next_subgoal']
    ui_decision.expand(decision, p, bound)


@pytest.mark.parametrize('mode', ['resolve', 'replan', 'cancel'])
def test_portal_uses_one_narrow_call_preserves_policy_and_only_dispatches_valid_resolution(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '1')
    adapter, store, _, kev_requests = configured(probability=.6)
    monkeypatch.setattr(tools_module, 'ComputerUseStore', lambda *a, **kw: store)
    body = _request(True, think=True)
    seen = []
    def upstream(request):
        payload = json.loads(request.content)
        schema_name = payload['response_format']['json_schema']['name']
        context = json.loads(payload['messages'][1]['content'])
        seen.append(schema_name)
        assert payload['think'] is True and payload['options'] == body['options']
        assert 'Preserve this exact caller policy.' in payload['messages'][0]['content']
        assert context['goal'] == body['messages'][-1]['content']
        if schema_name == 'native_pointer_resolution':
            assert seen == ['native_ui_subgoal', 'native_ui_phase', 'native_pointer_resolution']
            assert 'historical_ledger' not in context
            answer = {'choice': 'double_click_c0' if mode == 'resolve' else 'replan',
                      'expected_observation': 'The folder contents may appear.'}
            if mode == 'cancel':
                for owned in store._sessions.values():
                    owned.cancel.set()
        elif schema_name == 'native_ui_subgoal':
            if mode == 'replan' and len(seen) > 3:
                assert context['last_result']['slow_pointer_resolution']['status'] == 'replan'
            answer = proposal()
        else:
            action = ({'family': 'pointer_or_activation', 'target': 'c0'} if len(seen) == 2
                      else {'family': 'blocked', 'reason': 'Only a fixture input was tested.'})
            answer = {'action': {**action, 'objective': proposal()['next_subgoal']},
                'source_epoch': context['working_state']['source_epoch'],
                'source_phase': context['working_state']['source_phase'], 'assessment': 'No task completion claim.',
                'intended_resource': 'Observed input', 'remaining_criteria': ['Write and verify report']}
        return httpx.Response(200, json={'message': {'content': json.dumps(answer)}})
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), body, True)
    assert len(kev_requests) == 1 and len(adapter.calls) == int(mode == 'resolve')
    assert seen.count('native_pointer_resolution') == 1
    assert result['portal']['native_task']['task_verified'] is False
    if mode == 'cancel':
        assert result['portal']['native_task']['status'] == 'cancelled' and len(seen) == 3
    else:
        assert len(seen) == 5
