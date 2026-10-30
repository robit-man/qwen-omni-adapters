"""CPU native deferrals keep a fresh referent without minting input rights."""
import copy
import json

import httpx
import pytest
from test_native_compact_decision import reply
from test_native_control_catalogue import node
from test_native_subgoal_handoff import proposal
from test_portal_native_task_routes import GOAL, _config, _post, _request, _working_reply
from test_ui_intent_choice import configured
from test_ui_observation_boundary import CELL

import portal.tools as tools_module
from portal import ui_decision
from portal.app import create_app
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema


def actual_deferral(goal=GOAL):
    adapter, store, cancel, requests = configured(probability=.6)
    raw = adapter.observe()
    working = _NativeWorkingState(goal)
    view = working.view(1, 0)
    observation = {'state': raw}
    packet = {'goal': goal, 'current_state': observation, 'ledger': [], 'last_result': None,
        'working_state': view, 'remaining_seconds': 30,
        'response_schema': _working_phase_schema(['native_ui'],
            _grounded_phase_schema(['native_ui'], observation, intent_choice=True), view,
            observed_state=raw, action_handoffs=True)}
    compact, binding = ui_decision.project(packet)
    reference = next(ref for ref, value in binding['nodes'].items() if value['id'] == CELL)
    selection = {'family': 'pointer_or_activation', 'target': reference}
    plan = {**proposal(), 'immediate_action': selection}
    _, bound = ui_decision.bind_subgoal(compact, binding, plan, atomic_anchor=True)
    decision = reply(packet, selection)
    decision['action']['objective'] = plan['next_subgoal']
    expanded = ui_decision.expand(decision, packet, bound)
    with store.task_scope('session', 'desktop', cancel):
        result = store.intent_boundary('session', expanded['phase']['subgoal'], goal=goal,
                                       action_handoff=expanded['action_handoff'])
    assert len(requests) == 1 and result['status'] == 'abstained' and not adapter.calls
    return adapter, result, expanded['action_handoff']


def test_actual_accepted_policy_deferral_is_one_use_and_unrelated_revision_is_not_target_drift():
    adapter, result, handoff = actual_deferral()
    pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    assert pending is not None
    raw = adapter.observe()
    raw['revision'] = 'independent later capture; exact target remains unchanged'
    raw['nodes'].append(node('unrelated-observation', scope_id='scope'))
    priority, receipt = pending.consume(goal=GOAL, state=raw)
    assert priority == (CELL,)
    assert receipt['original_source']['revision'] != receipt['fresh_source']['revision']
    assert receipt['policy_source_revision'] == result['pointer_recovery']['source_revision']
    assert receipt['input_authority'] is receipt['task_progress'] is False
    assert receipt['new_effect_count'] == 0 and not adapter.calls
    assert pending.consume(goal=GOAL, state=raw) == ((), None)


@pytest.mark.parametrize('field,value', [('generation', 'replacement'), ('name', 'Other item'),
    ('role', 'button'), ('scope_id', 'different-scope'), ('visible', False), ('enabled', False),
    ('sensitive', True), ('operations', []), ('bounds', {'x': 1, 'y': 2, 'width': 3, 'height': 4}),
    ('focused', True), ('selected', ['Different selection'])])
def test_changed_fresh_target_never_gets_a_priority(field, value):
    adapter, result, handoff = actual_deferral()
    pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    assert pending is not None
    raw = adapter.observe()
    raw['nodes'][0][field] = value
    assert pending.consume(goal=GOAL, state=raw)[0] == ()
    assert pending.consume(goal=GOAL, state=adapter.observe()) == ((), None)
    assert not adapter.calls


@pytest.mark.parametrize('fault', ['session', 'scope', 'missing', 'goal', 'clear', 'tampered_record'])
def test_foreign_or_invalidated_pending_source_cannot_be_reused(fault):
    adapter, result, handoff = actual_deferral()
    pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    assert pending is not None
    raw, goal = adapter.observe(), GOAL
    if fault == 'session':
        raw['session_id'] = 'other-native-session'
    elif fault == 'scope':
        raw['scope']['id'] = 'other-scope'
    elif fault == 'missing':
        raw['nodes'].pop(0)
    elif fault == 'goal':
        goal = 'A different user goal'
    elif fault == 'clear':
        pending.clear()
    else:
        pending._record['pointer_recovery']['reference'] = '0' * 32
    assert pending.consume(goal=goal, state=raw)[0] == ()
    assert not adapter.calls


@pytest.mark.parametrize('fault', ['handoff', 'source_revision', 'policy_id', 'policy_model',
    'policy_mode', 'policy_confirmed', 'policy_status', 'scope_receipt', 'session_receipt',
    'candidate_ids', 'bank_hash', 'foreign_choice', 'unknown_verb', 'duplicate_choice',
    'input_attempted', 'effect', 'goal', 'missing_handoff'])
def test_malformed_or_foreign_returned_offer_cannot_mint_or_expand(fault):
    adapter, original, handoff = actual_deferral()
    result = copy.deepcopy(original)
    goal = GOAL
    if fault == 'handoff':
        result['action_handoff']['next_operation'] = 'A substituted operation'
    elif fault == 'source_revision':
        result['pointer_recovery']['source_revision'] = 'another policy source'
    elif fault == 'policy_id':
        result['pointer_recovery']['policy_receipt_id'] = 'another policy id'
    elif fault.startswith('policy_'):
        field = {'policy_model': 'model', 'policy_mode': 'mode', 'policy_confirmed': 'model_confirmed',
                 'policy_status': 'status'}[fault]
        result['policy_receipts'][0][field] = '' if fault == 'policy_model' else False
    elif fault in {'scope_receipt', 'session_receipt'}:
        result['intent_selection_receipt']['scope_sha256' if fault == 'scope_receipt' else 'session_sha256'] = 'f' * 64
    elif fault == 'candidate_ids':
        result['intent_selection_receipt']['candidate_ids'] = []
    elif fault == 'bank_hash':
        result['intent_selection_receipt']['intent_bank_sha256'] = 'f' * 64
    elif fault == 'foreign_choice':
        result['pointer_recovery']['choices'][0]['node_id'] = 'another native control'
    elif fault == 'unknown_verb':
        result['pointer_recovery']['choices'][0]['verb'] = 'focus'
    elif fault == 'duplicate_choice':
        result['pointer_recovery']['choices'].append(copy.deepcopy(result['pointer_recovery']['choices'][0]))
    elif fault == 'input_attempted':
        result['ui_boundary_receipt']['input_attempted'] = True
    elif fault == 'effect':
        result['effect_receipt']['effect_state'] = 'unknown'
    elif fault == 'goal':
        goal = 'Different task'
    else:
        result.pop('action_handoff')
    pending = ui_decision.retain_pointer_recovery(result, goal=goal, action_handoff=handoff)
    assert pending is None or pending.consume(goal=GOAL, state=adapter.observe())[0] == ()
    assert not adapter.calls


def test_full_portal_discovery_deferral_next_projection_and_normal_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '1')
    adapter, store, _, kev_requests = configured(probability=.6)
    original = adapter.observe

    def dense_observe():
        raw = original()
        raw['nodes'][0]['description'] = 'Source-grounded complete control context ' + 'z' * 900
        raw['nodes'][:0] = [{**node(f'pad-{i}', name='Unrelated ' + 'x' * 150, scope_id='scope'),
                            'description': 'Observed source ' + 'y' * 400} for i in range(30)]
        return raw

    adapter.observe = dense_observe
    monkeypatch.setattr(tools_module, 'ComputerUseStore', lambda *a, **kw: store)
    seen, actions = [], {}

    def upstream(request):
        wire = json.loads(request.content)
        schema_name = wire['response_format']['json_schema']['name']
        context = json.loads(wire['messages'][1]['content'])
        seen.append(schema_name)
        if schema_name == 'native_pointer_resolution':
            choices = [key for key in context['choices'] if key.startswith('double_click_')]
            assert len(choices) == 1 and len(kev_requests) == 1 and not adapter.calls
            assert context['action_handoff']['target']['node_id'] == CELL
            answer = {'choice': choices[0], 'expected_observation': 'Observe the actual resulting fixture state.'}
        elif schema_name == 'native_ui_subgoal':
            count = seen.count(schema_name)
            catalogue = context['observation']['model_projection']['control_catalogue']
            if count == 1:
                assert not any(control.get('name') == 'Input' for control in context['controls'])
                selected = {'family': 'inspect_controls', 'reference': next(
                    row['reference'] for row in catalogue['entries'] if row.get('role') == 'table_cell')}
            elif count == 2:
                selected = {'family': 'inspect_controls', 'reference': next(
                    row['reference'] for row in catalogue['entries'] if row.get('name') == 'Input')}
            elif count == 3:
                selected = {'family': 'pointer_or_activation', 'target': next(
                    control['ref'] for control in context['controls'] if control.get('name') == 'Input')}
            else:
                selected = {'family': 'blocked'}
            actions[count] = selected
            answer = {**proposal(), 'immediate_action': selected}
        else:
            count = seen.count('native_ui_subgoal')
            action = actions[count]
            if action['family'] == 'blocked':
                action = {**action, 'reason': 'One CPU fixture recovery occurred; task remains unverified.'}
            answer = {'action': {**action, 'objective': proposal()['next_subgoal']},
                'source_epoch': context['working_state']['source_epoch'],
                'source_phase': context['working_state']['source_phase'], 'assessment': 'No task completion claim.',
                'intended_resource': 'Observed fixture', 'remaining_criteria': ['Independent task verification']}
        return httpx.Response(200, json={'message': {'content': json.dumps(answer)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(True), True)
    assert seen.count('native_pointer_resolution') == 1, result
    assert len(kev_requests) == len(adapter.calls) == 1
    assert adapter.calls[0]['node_id'] == CELL and adapter.calls[0]['verb'] == 'double_click'
    assert result['portal']['native_task']['task_verified'] is False
    assert result['portal']['native_task']['status'] == 'blocked'


def test_memory_result_cannot_retain_even_a_genuine_other_native_offer(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '0')
    adapter, foreign, _ = actual_deferral()
    _, store, _, _ = configured(probability=.6)
    monkeypatch.setattr(tools_module, 'ComputerUseStore', lambda *a, **kw: store)
    actual_execute = tools_module.PortalToolHarness.execute
    def fake_memory(self, session, name, arguments):
        return copy.deepcopy(foreign) if name == 'memory_search' else actual_execute(self, session, name, arguments)
    monkeypatch.setattr(tools_module.PortalToolHarness, 'execute', fake_memory)
    captures = []
    original_retain = ui_decision.retain_pointer_recovery
    def capture(*args, **kwargs):
        captures.append((args, kwargs))
        return original_retain(*args, **kwargs)
    monkeypatch.setattr(ui_decision, 'retain_pointer_recovery', capture)
    packets = []
    def upstream(request):
        packet = json.loads(json.loads(request.content)['messages'][1]['content'])
        packets.append(packet)
        plan = {'kind': 'memory_search', 'query': 'a CPU fixture'} if len(packets) == 1 else {
            'kind': 'blocked', 'reason': 'Memory evidence cannot become a native pending action.'}
        return httpx.Response(200, json={'message': {'content': json.dumps(_working_reply(packet, plan))}})
    body = _request(True)
    body['portal_tool_allowlist'].append('memory_search')
    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), body, True)
    assert len(packets) == 2 and not captures and not adapter.calls
    assert 'pending_recovery_projection_receipt' not in packets[1]['current_state']
    assert result['portal']['native_task']['task_verified'] is False
