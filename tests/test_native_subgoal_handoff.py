"""Subgoal planning is a bounded proposal, not new task or input authority."""
import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import packet, reply
from test_portal_native_task_routes import _config, _install_store, _post, _request

from portal import ui_decision
from portal.app import create_app
from portal.computer_use import SystemOneUiProvider


def proposal():
    return {'current_view': 'An empty editor is active.', 'needed_evidence': ['Contents of the requested source'],
        'next_subgoal': 'Open and read the declared source before authoring the report.',
        'next_operation': 'Use the observed Open control.'}


def test_prerequisites_precede_one_objective_and_binding_preserves_original_authority():
    p = packet()
    compact, binding = ui_decision.project(p)
    original = copy.deepcopy((p, compact, binding))
    context, schema = ui_decision.subgoal_context(compact)
    assert list(schema['properties']) == ['current_view', 'needed_evidence', 'next_subgoal', 'next_operation']
    assert context['goal'] == p['goal'] and context['last_result'] == p['last_result']
    assert 'historical_ledger' not in context
    bound, b = ui_decision.bind_subgoal(compact, binding, proposal())
    assert (p, compact, binding) == original
    assert bound['goal'] == p['goal'] and bound['controls'] == compact['controls']
    assert bound['last_result'] == compact['last_result'] and bound['working_state'] == compact['working_state']
    assert bound['planning_handoff']['authority'] == 'planner_claim_not_effect_or_input_authority'
    value = reply(p, {'family': 'blocked', 'reason': 'No applicable offered operation'})
    value['action']['objective'] = proposal()['next_subgoal']
    expanded = ui_decision.expand(value, p, b)
    assert expanded['commitment']['objective'] == proposal()['next_subgoal']
    assert Draft202012Validator(p['response_schema']).is_valid(expanded)
    value['action']['objective'] = 'Silently change to another task'
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(value, p, b)


def test_subgoal_retains_unmet_work_without_turning_plans_into_effects():
    p = packet()
    commitment = {'authority': 'planner_claim', 'input_authority': False,
        'objective': 'Read the second source after opening Files.',
        'intended_resource': 'second-source.txt',
        'unmet_constraints': ['Acquire the second source', 'Save and reopen the report']}
    p['working_state']['commitment'] = copy.deepcopy(commitment)
    compact, _ = ui_decision.project(p)
    context, _ = ui_decision.subgoal_context(compact)
    assert context['retained_plan'] == commitment
    assert context['last_result']['effect_receipt']['effect_state'] == 'unknown'
    assert context['goal'] == p['goal']
    context['retained_plan']['unmet_constraints'].clear()
    assert compact['working_state']['commitment'] == commitment
    assert p['working_state']['commitment'] == commitment


def test_fresh_task_cannot_inherit_another_tasks_retained_plan():
    previous = packet()
    previous['working_state']['commitment'] = {'objective': 'Other task private source'}
    ui_decision.subgoal_context(ui_decision.project(previous)[0])
    fresh, _ = ui_decision.subgoal_context(ui_decision.project(packet())[0])
    assert fresh['retained_plan'] is None
    assert 'Other task private source' not in json.dumps(fresh)


@pytest.mark.parametrize('fault', ['blank', 'extra', 'missing', 'too_long', 'source_changed', 'none'])
def test_invalid_subgoal_cannot_bind_an_action(fault):
    compact, binding = ui_decision.project(packet())
    plan = proposal()
    if fault == 'blank':
        plan['next_subgoal'] = '   '
    elif fault == 'extra':
        plan['input_authority'] = True
    elif fault == 'missing':
        del plan['needed_evidence']
    elif fault == 'too_long':
        plan['next_subgoal'] = 'x' * 501
    elif fault == 'source_changed':
        compact['goal'] = 'Replacement task'
    else:
        plan = None
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.bind_subgoal(compact, binding, plan)


@pytest.mark.parametrize('mode', ['normal', 'invalid_proposal', 'cancel_after_proposal'])
def test_real_portal_compiler_orders_two_passes_preserves_options_and_prevents_input(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '1')
    store, adapters = _install_store(monkeypatch)
    store.provider = SystemOneUiProvider('http://127.0.0.1:18959', mode='active')
    seen = []
    body = _request(True, think=True)
    def upstream(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert payload['options'] == body['options'] and payload['think'] is True
        assert 'Preserve this exact caller policy.' in payload['messages'][0]['content']
        context = json.loads(payload['messages'][1]['content'])
        if payload['response_format']['json_schema']['name'] == 'native_ui_subgoal':
            result = {**proposal(), 'immediate_action': {'family': 'blocked'}}
            if mode == 'invalid_proposal':
                result['input_authority'] = True
            elif mode == 'cancel_after_proposal':
                next(iter(store._sessions.values())).cancel.set()
        else:
            assert len(seen) == 2 and context['planning_handoff']['proposal'] == {
                **proposal(), 'immediate_action': {'family': 'blocked'}}
            result = {'action': {'family': 'blocked', 'reason': 'No applicable offered operation',
                'objective': proposal()['next_subgoal']}, 'source_epoch': context['working_state']['source_epoch'],
                'source_phase': context['working_state']['source_phase'], 'assessment': 'No completion claim.',
                'intended_resource': 'Declared input', 'remaining_criteria': ['Read source', 'Verify report']}
        return httpx.Response(200, json={'message': {'content': json.dumps(result)}})
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), body, True)
    task = result['portal']['native_task']
    assert not adapters[0].calls and task['task_verified'] is False
    if mode == 'invalid_proposal':
        assert len(seen) == 3 and task['status'] == 'blocked'
        assert all(r['response_format']['json_schema']['name'] == 'native_ui_subgoal' for r in seen)
    elif mode == 'cancel_after_proposal':
        assert len(seen) == 1 and task['status'] == 'cancelled'
    else:
        assert len(seen) == 2 and task['status'] == 'blocked'
