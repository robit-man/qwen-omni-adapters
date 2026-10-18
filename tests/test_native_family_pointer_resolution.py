"""Pending family planning uses one finite question; no model or UI is executed."""
import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_direct_handoff import pending_direct_packet
from test_native_family_planner import body_context, enabled, response
from test_native_pending_family import accept_recovery, rebuild_schema
from test_portal_native_task_routes import _config, _install_store, _post, _request, _working_reply

import portal.app as app_module
from portal import ui_decision
from portal.app import create_app
from portal.computer_use import SystemOneUiProvider
from portal.ui_task import NativePhaseCompilationError


def pending_source():
    packet, adapter, result = pending_direct_packet()
    assert result['ui_boundary_receipt']['input_attempted'] is False and not adapter.calls
    return packet


def answer(context, verb='double_click'):
    choice = next(key for key, item in context['choices'].items() if item.get('verb') == verb)
    return {'choice': choice, 'expected_observation': 'Observe the selected control after one guarded operation.'}


def replan():
    return {'choice': 'replan', 'expected_observation': 'These local operations do not resolve the prerequisite.'}


def probe(tmp_path, monkeypatch, packet, handler, *, stream=False):
    """Reach the actual create_app compile closure without dispatching its result."""
    enabled(monkeypatch)
    store, adapters = _install_store(monkeypatch)
    store.provider = SystemOneUiProvider('http://127.0.0.1:18959', mode='active')
    original = copy.deepcopy(packet)
    direct, binding = ui_decision.project_direct(packet)
    seen, compiled, errors = [], [], []
    request_body = _request(stream, think=True)
    request_body['messages'][-1]['content'] = packet['goal']

    def run(goal, *, compile_phase, **_kwargs):
        assert goal == packet['goal']
        try:
            compiled.append(compile_phase(packet))
        except Exception as error:
            errors.append(error)
        yield {'type': 'final', 'response': {'message': {'content': 'CPU compiler boundary only.'},
            'portal': {'native_task': {'task_verified': False}}}}

    monkeypatch.setattr(app_module, 'run_native_goal', run)

    def upstream(request):
        body, context = body_context(request)
        seen.append(body)
        assert body['options'] == request_body['options'] and body['think'] is True
        assert body['model'] == request_body['model'] and body['speech_mode'] == 'never'
        assert 'Preserve this exact caller policy.' in body['messages'][0]['content']
        return handler(body, context, len(seen), store)

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), request_body, stream)
    assert packet == original and ui_decision.project_direct(packet) == (direct, binding)
    assert not adapters[0].calls and result['portal']['native_task']['task_verified'] is False
    return seen, compiled, errors, result, direct


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('verb', ['click', 'double_click', 'right_click', 'invoke'])
def test_genuine_pending_runs_finite_first_and_accepts_each_offered_verb_once(tmp_path, monkeypatch, stream, verb):
    packet = pending_source()
    compact, binding = ui_decision.project(packet)
    expected, schema, _ = ui_decision.pointer_resolution_context(compact, binding)

    def upstream(body, context, count, _store):
        assert count == 1 and body['response_format']['json_schema']['name'] == 'native_pointer_resolution'
        assert context == {**expected, 'response_schema': schema}
        value = answer(context, verb)
        Draft202012Validator(schema).validate(value)
        return response(value)

    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet, upstream, stream=stream)
    assert len(seen) == len(compiled) == 1 and not errors
    expanded = compiled[0]
    Draft202012Validator(packet['response_schema']).validate(expanded)
    checked = accept_recovery(packet, expanded)
    assert expanded['phase']['step']['verb'] == verb
    assert checked['target'] == packet['last_result']['action_handoff']['target']
    assert 'planner_rejections' not in result['portal']['native_task']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('kind', ['blocked', 'memory_search'])
def test_explicit_replan_goes_direct_once_with_complete_source_and_separate_untrusted_feedback(
        tmp_path, monkeypatch, stream, kind):
    packet = pending_source()
    rebuild_schema(packet, ('native_ui', 'memory_search'))
    packet['working_state']['consecutive_unverified_count'] = 2
    direct, private = ui_decision.project_direct(packet)
    phase = ({'kind': 'blocked', 'reason': 'Independent goal criteria remain unmet.'} if kind == 'blocked' else
             {'kind': 'memory_search', 'query': 'Retained source evidence', 'max_results': 8})

    def upstream(body, context, count, _store):
        if count == 1:
            assert body['response_format']['json_schema']['name'] == 'native_pointer_resolution'
            return response(replan())
        assert count == 2 and body['response_format']['json_schema']['name'] == 'native_ui_phase'
        assert context == direct and body['response_format']['json_schema']['schema'] == direct['response_schema']
        assert len(body['messages']) == 3 and body['messages'][1]['role'] == 'user'
        feedback = json.loads(body['messages'][1]['content'])
        assert feedback == {'authority': 'untrusted_planner_claim', 'slow_pointer_resolution': {
            'status': 'replan', 'input_authority': False, 'authority': 'planner_prediction_not_observed_effect',
            'expected_observation': replan()['expected_observation']}}
        assert 'You are the Omni task planner' in body['messages'][0]['content']
        assert body['messages'][-1]['role'] == 'user'
        assert not {'planning_handoff', 'atomic_action_anchor', 'slow_pointer_resolution'} & context.keys()
        assert ui_decision.project_direct(packet) == (direct, private)
        return response(_working_reply(context, phase))

    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet, upstream, stream=stream)
    assert len(seen) == 2 and not errors and compiled[0]['phase'] == phase
    claim, = result['portal']['native_task']['planner_rejections']
    assert claim['authority'] == 'untrusted_planner_claim' and claim['next_route'] == 'full_direct'
    assert claim['reason'] == 'pending_pointer_resolution_replan' and claim['generation_calls'] == 2
    assert claim['source_epoch'] == packet['working_state']['source_epoch'] and claim['source_phase'] == 2
    assert all(claim[key] is False for key in ('input_attempted', 'task_progress', 'verified'))
    assert packet['working_state']['consecutive_unverified_count'] == 2


@pytest.mark.parametrize('fault', ['json', 'extra', 'unknown_choice', 'blank', 'oversize', 'foreign_transition'])
def test_invalid_finite_reply_never_becomes_direct_fallback(tmp_path, monkeypatch, fault):
    packet = pending_source()

    def upstream(_body, context, count, _store):
        assert count == 1
        if fault == 'json':
            return httpx.Response(200, json={'message': {'content': '{"private-invalid-reply"'}})
        value = replan()
        if fault == 'extra':
            value['input_authority'] = True
        elif fault == 'unknown_choice':
            value['choice'] = 'invoke_another_target'
        elif fault == 'blank':
            value['expected_observation'] = '  '
        elif fault == 'oversize':
            value['expected_observation'] = 'x' * 301
        else:
            value['transition_kind'] = 'document_changed'
        return response(value)

    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet, upstream)
    assert len(seen) == len(errors) == 1 and not compiled
    assert isinstance(errors[0], NativePhaseCompilationError)
    assert 'planner_rejections' not in result['portal']['native_task']


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('fault', ['http', 'protocol', 'timeout'])
def test_upstream_failure_never_adds_a_question(tmp_path, monkeypatch, stage, fault):
    packet = pending_source()

    def upstream(_body, _context, count, _store):
        assert count <= stage
        if count < stage:
            return response(replan())
        if fault == 'http':
            return httpx.Response(503, json={'error': 'Controlled transport failure'})
        if fault == 'timeout':
            raise httpx.ReadTimeout('Controlled timeout')
        return httpx.Response(200, json={'message': {'content': ['not', 'JSON text']}})

    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet, upstream)
    assert len(seen) == stage and len(errors) == 1 and not compiled
    claims = result['portal']['native_task'].get('planner_rejections', [])
    assert len(claims) == stage - 1
    if claims:
        assert claims[0]['generation_calls'] == 2 and claims[0]['verified'] is False


@pytest.mark.parametrize('failure', ['cancel', 'deadline'])
def test_replan_escape_rechecks_same_stop_and_deadline_without_budget_reset(tmp_path, monkeypatch, failure):
    packet = pending_source()
    now, actual_time = [100.0], app_module.time

    class Clock:
        def monotonic(self):
            return now[0]

        def __getattr__(self, key):
            return getattr(actual_time, key)

    monkeypatch.setattr(app_module, 'time', Clock())

    def upstream(_body, _context, count, store):
        assert count == 1
        if failure == 'cancel':
            next(iter(store._sessions.values())).cancel.set()
        else:
            now[0] += 1000
        return response(replan())

    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet, upstream)
    assert len(seen) == len(errors) == 1 and not compiled
    claim, = result['portal']['native_task']['planner_rejections']
    assert claim['generation_calls'] == 1 and claim['verified'] is False


def test_private_source_binding_error_is_not_a_replan_escape(tmp_path, monkeypatch):
    packet = pending_source()
    original = ui_decision.resolve_pointer_proposal

    def corrupted(compact, binding, packet, value):
        binding['packet_sha256'] = '0' * 64
        return original(compact, binding, packet, value)

    monkeypatch.setattr(ui_decision, 'resolve_pointer_proposal', corrupted)
    seen, compiled, errors, result, _ = probe(tmp_path, monkeypatch, packet,
        lambda _body, context, _count, _store: response(answer(context)))
    assert len(seen) == len(errors) == 1 and not compiled
    assert isinstance(errors[0], NativePhaseCompilationError)
    assert 'planner_rejections' not in result['portal']['native_task']
