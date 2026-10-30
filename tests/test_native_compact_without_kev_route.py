"""Portal route: compact planning without Kev activates one exact control and calls no policy."""
import json

import httpx
from test_native_subgoal_handoff import proposal
from test_portal_native_task_routes import _config, _install_store, _post, _request

from portal.app import create_app


def test_compact_without_selector_dispatches_exact_activation_once(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '1')
    store, adapters = _install_store(monkeypatch)
    assert store.provider is None and not store.intent_choice_enabled
    seen = []

    def upstream(request):
        payload = json.loads(request.content)
        seen.append(payload['response_format']['json_schema']['name'])
        context = json.loads(payload['messages'][1]['content'])
        commit = next((c['ref'] for c in context.get('controls', []) if c.get('name') == 'Commit'), None)
        if seen[-1] == 'native_ui_subgoal':
            done = any(call['verb'] == 'invoke' for call in adapters[0].calls) if adapters else False
            result = {**proposal(), 'immediate_action': {'family': 'blocked'} if done else
                      {'family': 'activate', 'target': commit}}
        elif adapters and adapters[0].calls:
            result = {'action': {'family': 'blocked', 'reason': 'Stop after one observed activation.',
                'objective': proposal()['next_subgoal']}, 'source_epoch': context['working_state']['source_epoch'],
                'source_phase': context['working_state']['source_phase'], 'assessment': 'No completion claim.',
                'intended_resource': 'Form', 'remaining_criteria': ['Verify receipt']}
        else:
            assert commit is not None
            result = {'action': {'family': 'activate', 'verb': 'invoke', 'target': commit,
                'objective': proposal()['next_subgoal']}, 'source_epoch': context['working_state']['source_epoch'],
                'source_phase': context['working_state']['source_phase'], 'assessment': 'Activate Commit.',
                'intended_resource': 'Form', 'remaining_criteria': ['Verify receipt']}
        return httpx.Response(200, json={'message': {'content': json.dumps(result)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(True), True)
    task = result['portal']['native_task']
    invokes = [call for call in adapters[0].calls if call['verb'] == 'invoke']
    assert len(invokes) == 1 and invokes[0]['node_id'] == 'commit', adapters[0].calls
    assert task['task_verified'] is False
    assert 'native_ui_phase' in seen and 'native_ui_subgoal' in seen
