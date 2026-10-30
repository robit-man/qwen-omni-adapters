"""CPU-only real planner/controller routing; transports and desktop are doubles."""
import copy
import json

import httpx
from test_native_control_catalogue import node, state
from test_portal_native_task_routes import (
    Adapter,
    _config,
    _install_store,
    _post,
    _request,
    _working_reply,
)

from portal import ui_decision
from portal.app import _model_ui_observation, create_app
from portal.computer_use import SystemOneUiProvider
from portal.ui_control_catalogue import NativeControlCatalogue
from portal.ui_task import NativePhaseCompilationError, run_native_goal


def test_discovery_keeps_prior_failures_and_only_normal_executor_can_write(tmp_path, monkeypatch):
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '1')
    store, adapters = _install_store(monkeypatch)
    store.provider = SystemOneUiProvider('http://127.0.0.1:18959', mode='active')
    old_state = Adapter._state

    def dense_state(self, include_windows=False):
        value = old_state(self, include_windows)
        value['nodes'][0]['description'] = 'Observed field context ' + 'z' * 900
        value['nodes'][0]['text_observation'] = {'character_count': len(self.value),
            'selection_count': 0, 'captured_range': {'start': 0, 'end': len(self.value)}, 'truncated': False}
        value['nodes'][:0] = [{**node(f'pad-{i}', name='Unrelated control ' + 'x' * 150,
            scope_id='form'), 'description': 'Observed source ' + 'y' * 400} for i in range(30)]
        return value

    monkeypatch.setattr(Adapter, '_state', dense_state)
    seen = []
    wire_systems = []
    actions = {}

    def upstream(request):
        wire = json.loads(request.content)
        context = json.loads(wire['messages'][1]['content'])
        name = wire['response_format']['json_schema']['name']
        wire_systems.append((name, wire['messages'][0]['content']))
        if name == 'native_ui_subgoal':
            phase = len(seen) + 1
            seen.append(copy.deepcopy(context))
            assert not adapters[0].calls if phase < 4 else adapters[0].calls
            catalogue = context['observation']['model_projection']['control_catalogue']
            if phase == 1:
                assert not any(control.get('name') == 'Draft' for control in context['controls'])
                reference = next(row['reference'] for row in catalogue['entries'] if row.get('role') == 'textbox')
                action = {'family': 'inspect_controls', 'reference': reference}
            elif phase == 2:
                reference = next(row['reference'] for row in catalogue['entries'] if row.get('name') == 'Draft')
                action = {'family': 'inspect_controls', 'reference': reference}
            elif phase == 3:
                draft = next(control for control in context['controls'] if control.get('name') == 'Draft')
                action = {'family': 'text_entry', 'target': draft['ref'], 'text': 'CPU observed exact value'}
            else:
                action = {'family': 'blocked', 'reason': 'Only the fixture write was verified; task remains unmet.'}
            actions[phase] = action
            selected = {key: value for key, value in action.items() if key in {'family', 'target', 'reference'}}
            answer = {'current_view': 'Observed fixture controls; no inferred task result.',
                'needed_evidence': [], 'next_subgoal': 'Inspect or edit the observed fixture field.',
                'next_operation': 'Use only the offered readonly or native operation.', 'immediate_action': selected}
        else:
            phase = context['working_state']['source_phase']
            answer = {'action': {**actions[phase], 'objective': context['planning_handoff']['proposal']['next_subgoal']},
                'source_epoch': context['working_state']['source_epoch'], 'source_phase': phase,
                'assessment': 'The original task remains unverified.', 'intended_resource': 'Observed fixture',
                'remaining_criteria': ['Independent task verification']}
        return httpx.Response(200, json={'message': {'content': json.dumps(answer)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(True), True)
    assert len(seen) == 4, result
    assert len(adapters[0].calls) == 1, json.dumps(seen[-1].get('last_result'), indent=2)
    assert adapters[0].calls[0]['node_id'] == 'draft'
    assert adapters[0].value == 'CPU observed exact value'
    task = result['portal']['native_task']
    assert task['task_verified'] is False and task['status'] == 'blocked'
    traces = result['portal']['safe_tools_executed']
    for trace in traces[:2]:
        evidence = json.loads(trace['result'])
        assert evidence['input_attempted'] is evidence['task_progress'] is False
        assert evidence['new_effect_count'] == 0
    assert json.loads(traces[2]['result'])['text_entry_receipt']['exact_value_verified'] is True
    assert set(traces[0]['arguments']) == {'operation', 'reference'}
    assert all('nc-' not in str(call['node_id']) for call in adapters[0].calls)
    # This asserts actual first-stage and downstream HTTP payloads, not source
    # presence: both capability-selecting layers receive the identical contract.
    definition = (
        'observation.model_projection.control_catalogue offers read-only native role groups and controls. '
        'inspect_controls expands one offered reference without input; an exact control requests a fresh '
        'full binding on the next observation. Catalogue references are never action targets. ')
    assert {name for name, _ in wire_systems} == {'native_ui_subgoal', 'native_ui_phase'}
    assert all(system.count(definition) == 1 for _, system in wire_systems)
    assert definition == ui_decision.CONTROL_DISCOVERY_SYSTEM


def test_new_knowledge_does_not_erase_a_preexisting_action_failure():
    catalogue = NativeControlCatalogue('cpu-task')
    observed = state([node('one')])
    packets = []
    calls = []

    def observe():
        view, priority, _ = catalogue.capture(observed)
        result = _model_ui_observation({'state': copy.deepcopy(observed)}, priority)
        result['state']['model_projection']['control_catalogue'] = view
        return result

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        phase = packet['working_state']['source_phase']
        if phase in {1, 4, 5}:
            raise NativePhaseCompilationError()
        catalogue_view = packet['current_state']['state']['model_projection']['control_catalogue']
        kind = 'group' if phase == 2 else 'control'
        reference = next(row['reference'] for row in catalogue_view['entries'] if row['kind'] == kind)
        return _working_reply(packet, {'kind': 'inspect_controls', 'reference': reference})

    def execute(name, arguments):
        calls.append((name, copy.deepcopy(arguments)))
        assert arguments['operation'] == 'inspect_controls'
        return catalogue.inspect(arguments['reference'])

    events = list(run_native_goal('Inspect the CPU fixture without pretending it completed.',
        observe=observe, compile_phase=compile_phase, execute=execute, cancel=catalogue.clear,
        allowed_tools=['native_ui'], max_phases=8, max_seconds=30,
        intent_choice=True, working_state=True, effect_contracts=True))
    assert [p['working_state']['consecutive_unverified_count'] for p in packets] == [0, 1, 1, 1, 2]
    assert events[-1]['response']['portal']['native_task']['status'] == 'blocked'
    assert len(calls) == 2
    assert events[-1]['response']['portal']['native_task']['task_verified'] is False


def test_readonly_refs_cannot_pass_the_ordinary_action_family_schema():
    # This exercises native compact schema validation, not string-prefix guards.
    from test_native_compact_decision import packet, reply
    value = packet()
    compact, binding = ui_decision.project(value)
    reference = NativeControlCatalogue('task').capture(state())[0]['entries'][0]['reference']
    try:
        ui_decision.expand(reply(value, {'family': 'pointer_or_activation', 'target': reference}), value, binding)
    except ui_decision.DecisionError:
        pass
    else:
        raise AssertionError('A catalogue reference became an ordinary native target')
    assert all(control['ref'] != reference for control in compact['controls'])


def test_repeated_unchanged_expansion_spends_the_existing_three_failure_limit():
    catalogue = NativeControlCatalogue('repeat-task')
    observed = state([node('one')])
    packets = []

    def observe():
        view, priority, _ = catalogue.capture(observed)
        result = _model_ui_observation({'state': copy.deepcopy(observed)}, priority)
        result['state']['model_projection']['control_catalogue'] = view
        return result

    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        kind = 'group' if len(packets) == 1 else 'control'
        view = packet['current_state']['state']['model_projection']['control_catalogue']
        reference = next(row['reference'] for row in view['entries'] if row['kind'] == kind)
        return _working_reply(packet, {'kind': 'inspect_controls', 'reference': reference})

    events = list(run_native_goal('Inspect a control; do not claim completion.', observe=observe,
        compile_phase=compile_phase, execute=lambda _, args: catalogue.inspect(args['reference']),
        cancel=catalogue.clear, allowed_tools=['native_ui'], max_phases=10, max_seconds=30,
        intent_choice=True, working_state=True, effect_contracts=True))
    assert [p['working_state']['consecutive_unverified_count'] for p in packets] == [0, 0, 0, 1, 2]
    assert events[-1]['response']['portal']['native_task']['phases'] == 5
    assert events[-1]['response']['portal']['native_task']['status'] == 'blocked'
