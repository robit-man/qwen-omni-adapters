"""Generated drags use real host routes with explicitly controlled CPU providers.

These observations and input acknowledgments are doubles, not live desktop or
model evidence. The production store/controller still owns every offered input.
"""
from __future__ import annotations

import copy
import hashlib
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_atomic_anchor import _source
from test_native_direct_handoff import _reply
from test_native_manipulation import _pointer_fixture
from test_portal_native_task_routes import (
    SESSION_COOKIE_NAME,
    _config,
    _post,
    _request,
    _working_reply,
)
from test_ui_observation_boundary import node

import portal.tools as tools_module
from portal import ui_decision
from portal.app import create_app
from portal.computer_use import ComputerUseStore
from portal.ui_prefix import run_verified_prefix
from portal.ui_task import _grounded_phase_schema, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError, parse_intent, parse_step
from qwen_omni_adapters.ui_effects import compile_immediate_transition

OBJECTIVE = 'Move the observed item to the observed destination, then inspect the result.'
GOAL = 'Use the owned surface to arrange its items and independently verify the requested result.'
LABEL = 'Already visible — 原文'


def drag():
    return {'id': 'move-item', 'verb': 'drag_drop', 'target': {'node_id': 'item'},
            'args': {'to_node_id': 'destination'}}


def boundary():
    return {'kind': 'ui_boundary', 'objective': OBJECTIVE, 'step': drag()}


class DragAdapter:
    """No native queries: finite source/destination facts and a recorded fake input."""

    def __init__(self, session='session'):
        self.session = session
        self.nodes = [
            node('item', 'table_cell', 'Arbitrary source', ['drag_drop'],
                 bounds={'x': 10, 'y': 20, 'width': 30, 'height': 20}),
            node('destination', 'list', 'Arbitrary destination', [],
                 bounds={'x': 70, 'y': 20, 'width': 30, 'height': 20}),
            node('draft', 'textbox', 'Draft', ['focus', 'set_value'], focused=True, value=''),
        ]
        self.calls, self.events, self.prepared = [], [], []
        self.changed = False
        self.mutate = True
        self.after_error = False
        self.reply = {'status': 'dispatched', 'input_attempted': True}

    def observe(self):
        self.events.append('observe')
        nodes = copy.deepcopy(self.nodes)
        nodes[0]['selected'] = ['Arbitrary source'] if self.changed else []
        value = {'session_id': self.session, 'scope': {'kind': 'desktop', 'id': 'scope'},
                 'nodes': nodes, 'text': LABEL}
        value['revision'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value

    planning_observe = observe

    def prepare_intent(self, intent):
        self.events.append('prepare_intent')
        self.prepared.append(parse_intent(intent))

    def prepare_step(self, step):
        self.prepared.append(parse_step(step))

    def finish_intent(self):
        self.events.append('finish_intent')

    def boundary_observe(self):
        self.events.append('boundary_observe')
        if self.after_error:
            raise RuntimeError('Controlled after-observation failure')
        return self.observe()

    def perform(self, action):
        self.events.append('perform')
        self.calls.append(copy.deepcopy(action))
        if self.mutate:
            self.changed = True
        return copy.deepcopy(self.reply)


def drag_packet():
    packet = _source(memory=False)
    state = DragAdapter().observe()
    packet['current_state']['state'] = state
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], effect_contracts=True),
        packet['working_state'], observed_state=state, action_handoffs=True)
    return packet


def store_fixture(*, private=True):
    adapter = DragAdapter()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=private)
    controller = store._sessions[('session', 'desktop')].controller
    controller.max_observations = 1
    controller.verification_timeout_s = .02
    controller.observation_interval_s = .001
    return adapter, store, cancel, controller


def immediate(store, cancel, step=None):
    with store.task_scope('session', 'desktop', cancel):
        return store.immediate_transition('session', step or drag(), goal=GOAL, objective=OBJECTIVE)


def test_full_working_and_direct_schemas_admit_only_unpromised_null_intent_drag():
    packet = drag_packet()
    view, binding = ui_decision.project_direct(packet)
    reply = _reply(packet, boundary(), None)
    Draft202012Validator(view['response_schema']).validate(reply)
    expanded = ui_decision.expand_direct(reply, packet, binding)
    assert expanded['phase'] == boundary()
    assert 'action_handoff' not in expanded and 'transition' not in expanded
    canonical = Draft202012Validator(packet['response_schema'])
    canonical.validate(expanded)
    effect = compile_immediate_transition(expanded['phase']['step'], packet['current_state']['state'])
    assert effect.mode == 'observation_boundary' and effect.as_dict()['predicates'] == []
    assert effect.bind(drag()) == drag()
    assert effect.as_dict()['non_authorizing'] is True
    assert effect.as_dict()['task_progress'] is False


@pytest.mark.parametrize('fault', ['expect', 'missing_destination', 'extra_args', 'nonnull_intent', 'transition'])
def test_emitted_direct_schema_rejects_effect_promises_and_inapplicable_drag_intent(fault):
    packet = drag_packet()
    view, binding = ui_decision.project_direct(packet)
    value = _reply(packet, boundary(), None)
    step = value['phase']['step']
    if fault == 'expect':
        step['expect'] = [{'kind': 'text', 'text': LABEL}]
    elif fault == 'missing_destination':
        step['args'] = {}
    elif fault == 'extra_args':
        step['args']['x'] = 10
    elif fault == 'nonnull_intent':
        step['immediate_intent'] = {'next_operation': 'Move', 'needed_evidence': []}
    else:
        value['transition'] = {'kind': 'document_changed', 'source_ref': {}}
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


def test_legacy_grounding_keeps_predicate_drag_while_generated_first_drag_is_a_boundary():
    observation = {'state': DragAdapter().observe()}
    legacy = {'kind': 'ui', 'steps': [{**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]}],
              'final_predicates': [{'kind': 'text', 'text': LABEL}]}
    for intent_choice in (False, True):
        old = _grounded_phase_schema({'native_ui'}, observation, intent_choice=intent_choice)
        generated = _grounded_phase_schema({'native_ui'}, observation,
            intent_choice=intent_choice, effect_contracts=True)
        assert Draft202012Validator(old).is_valid(legacy)
        assert not Draft202012Validator(old).is_valid(boundary())
        assert Draft202012Validator(generated).is_valid(boundary())
        assert not Draft202012Validator(generated).is_valid(legacy)


@pytest.mark.parametrize('args', [{}, {'to_node_id': ''}, {'to_node_id': 7},
    {'to_node_id': 'destination', 'x': 12}, {'to_node_id': 'destination', 'force': True}])
def test_private_parser_keeps_closed_destination_arguments(args):
    with pytest.raises(ComputerUseError):
        parse_intent({**drag(), 'args': args})


def test_boundary_parser_rejects_model_effect_even_when_already_true():
    with pytest.raises(ComputerUseError):
        parse_intent({**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]})


@pytest.mark.parametrize('prefix,stream', [(False, False), (False, True), (True, False)])
def test_actual_portal_direct_route_dispatches_one_drag_then_replans_without_effect_credit(
        tmp_path, monkeypatch, prefix, stream):
    adapters, packets, replies = [], [], []

    def factory(session, surface):
        assert surface == 'desktop'
        adapter = DragAdapter(session)
        adapters.append(adapter)
        return adapter

    store = ComputerUseStore(None, adapter_factory=factory)
    monkeypatch.setattr(tools_module, 'ComputerUseStore', lambda *_args, **_kwargs: store)
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '0')
    suffix = {'id': 'unexecuted-write', 'verb': 'set_value', 'target': {'node_id': 'draft'},
              'args': {'value': 'Must not be written'}, 'expect': [{'kind': 'text', 'text': LABEL}]}
    def upstream(request):
        body = json.loads(request.content)
        packet = json.loads(body['messages'][1]['content'])
        packets.append(packet)
        assert packet['goal'] == GOAL and body['think'] is False
        assert body['response_format']['json_schema']['schema'] == packet['response_schema']
        if len(packets) == 1:
            phase = boundary()
            if prefix:
                phase = {'kind': 'ui', 'steps': [
                    {'id': 'already-focused', 'verb': 'focus', 'target': {'node_id': 'draft'},
                     'expect': [{'kind': 'focused', 'node_id': '$target'}]},
                    {**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]}, suffix],
                    'final_predicates': [{'kind': 'text', 'text': LABEL}]}
        elif prefix and len(packets) == 2:
            assert not adapters[0].calls and not adapters[0].prepared
            feedback = packet['last_result']
            receipt = feedback['verified_prefix_receipt']
            assert receipt['stop'] == 'unannotated_observation_boundary'
            assert receipt['attempted_step_count'] == receipt['verified_prefix_count'] == 1
            assert receipt['unattempted_step_count'] == 2
            declined = replies[0]['phase']['steps'][1]
            assert receipt['continuation']['step_index'] == 1 and receipt['continuation']['step'] == declined
            assert receipt['continuation']['objective'] == replies[0]['commitment']['objective']
            assert feedback['planner_workflow_claim']['remaining_steps'] == [declined, suffix]
            assert not feedback['policy_receipts'] and not feedback['point_receipts']
            assert receipt['task_progress'] is False and feedback['phase_handoff']['new_effect_count'] == 0
            phase = boundary()
        else:
            phase = {'kind': 'blocked', 'reason': 'Independent task criteria are still unverified.'}
        reply = _working_reply(request, phase)
        Draft202012Validator(packet['response_schema']).validate(reply)
        replies.append(copy.deepcopy(reply))
        return httpx.Response(200, json={'message': {'content': json.dumps(reply)}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, 'generated-drag-cpu-session-12345')
    body = _request(stream)
    body['messages'][-1]['content'] = GOAL
    result, _ = _post(client, body, stream)
    assert len(packets) == (3 if prefix else 2)
    adapter = adapters[0]
    assert len(adapter.calls) == 1 and adapter.calls[0]['verb'] == 'drag_drop'
    assert adapter.calls[0]['node_id'] == 'item'
    assert adapter.calls[0]['args'] == {'to_node_id': 'destination'}
    assert adapter.calls[0]['expect'] == []
    assert adapter.events.index('perform') < adapter.events.index('boundary_observe')
    if not prefix:
        assert replies[0]['phase']['step']['immediate_intent'] is None
    traces = [json.loads(row['result']) for row in result['portal']['safe_tools_executed']]
    feedback = next(row for row in traces if 'ui_boundary_receipt' in row)
    assert feedback['status'] == 'observed_unverified'
    assert feedback['ui_boundary_receipt']['verified'] is False
    assert feedback['ui_boundary_receipt']['input_attempted'] is True
    assert feedback['new_effect_count'] == 0 and feedback['task_progress'] is False
    assert feedback['effect_receipt']['effect_state'] == 'unknown'
    assert feedback['immediate_effect_contract']['predicates'] == []
    actual = feedback.get('dispatch_record', feedback.get('outcomes', [None])[-1])
    assert actual['before']['session_id'] == actual['after']['session_id'] == adapter.session
    assert actual['before']['revision'] == actual['action']['revision']
    assert actual['after']['revision'] != actual['before']['revision']
    assert 'verified_prefix_receipt' not in feedback
    if prefix:
        receipt = traces[0]['verified_prefix_receipt']
        assert receipt['stop'] == 'unannotated_observation_boundary'
        assert receipt['unattempted_step_count'] == 2
        assert receipt['model_terminal_predicates_verified'] is False
        assert packets[1]['last_result']['planner_workflow_claim']['remaining_steps'] == [
            replies[0]['phase']['steps'][1], suffix]
    assert result['portal']['native_task']['task_verified'] is False
    assert adapter.nodes[-1]['value'] == ''


@pytest.mark.parametrize('endpoint', ['item', 'destination'])
@pytest.mark.parametrize('fault', ['missing', 'hidden', 'disabled', 'protected', 'cross_scope'])
def test_current_ineligible_or_missing_endpoint_cannot_offer_a_drag(endpoint, fault):
    adapter, store, cancel, _ = store_fixture()
    target = next(n for n in adapter.nodes if n['id'] == endpoint)
    if fault == 'missing':
        adapter.nodes.remove(target)
    else:
        key, value = {'hidden': ('visible', False), 'disabled': ('enabled', False),
                      'protected': ('sensitive', True), 'cross_scope': ('scope_id', 'foreign')}[fault]
        target[key] = value
    result = immediate(store, cancel)
    assert not adapter.calls
    assert result['ui_boundary_receipt']['input_attempted'] is False


@pytest.mark.parametrize('endpoint', ['item', 'destination'])
@pytest.mark.parametrize('fault', ['generation', 'bounds', 'visibility', 'scope'])
def test_fresh_offer_rejects_endpoint_drift_without_boundary_retry(endpoint, fault):
    adapter, store, cancel, _ = store_fixture()
    select = store._select
    selected = []
    def change_after_offer(*args, **kwargs):
        result = select(*args, **kwargs)
        selected.append(result)
        target = next(n for n in adapter.nodes if n['id'] == endpoint)
        if fault == 'generation':
            target['generation'] = '2'
        elif fault == 'bounds':
            target['bounds']['x'] += 1
        elif fault == 'visibility':
            target['visible'] = False
        else:
            target['scope_id'] = 'foreign'
        return result
    store._select = change_after_offer
    result = immediate(store, cancel)
    assert len(selected) == 1 and not adapter.calls
    assert result['dispatch_record'].get('dispatch_attempted') is not True
    assert 'receipt' not in result['dispatch_record']
    assert result['dispatch_record'].get('pre_input_refresh') is None


@pytest.mark.parametrize('fault', ['denied', 'cancelled', 'input_busy', 'untrusted'])
def test_existing_authority_cancellation_and_input_exclusion_cannot_be_bypassed(fault):
    adapter, store, cancel, controller = store_fixture(private=fault != 'untrusted')
    if fault == 'denied':
        controller.authorize = lambda *_: False
    elif fault == 'cancelled':
        cancel.set()
    elif fault == 'input_busy':
        controller._input.acquire()
    try:
        if fault == 'untrusted':
            with pytest.raises(ComputerUseError):
                immediate(store, cancel)
        else:
            immediate(store, cancel)
    finally:
        if fault == 'input_busy':
            controller._input.release()
    assert not adapter.calls


@pytest.mark.parametrize('after_error', [False, True])
def test_unknown_drag_outcome_is_not_replayed_or_promoted_by_already_visible_text(after_error):
    adapter, store, cancel, _ = store_fixture()
    adapter.mutate = False
    adapter.after_error = after_error
    step = {**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]}
    first = immediate(store, cancel, step)
    second = immediate(store, cancel, {**step, 'id': 'Rewording is not renewed input authority'})
    assert len(adapter.calls) == 1
    assert first['effect_receipt']['effect_state'] == 'unknown'
    assert second['effect_receipt']['effect_state'] == 'replayed'
    assert first['task_progress'] is second['task_progress'] is False
    assert first['new_effect_count'] == second['new_effect_count'] == 0
    assert first['ui_boundary_receipt']['input_attempted'] is True
    assert second['ui_boundary_receipt']['cached'] is True


@pytest.mark.parametrize('terminal_present', [False, True])
def test_explicit_public_drag_workflow_keeps_client_predicate_semantics(terminal_present):
    adapter, store, _cancel, _ = store_fixture(private=False)
    step = {**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]}
    terminal = LABEL if terminal_present else 'Independent missing evidence'
    result = store.act('session', {'surface': 'desktop', 'operation': 'workflow', 'steps': [step],
        'final_predicates': [{'kind': 'text', 'text': terminal}]}, goal=GOAL)
    assert len(adapter.calls) == 1 and adapter.calls[0]['expect'] == step['expect']
    assert result['status'] == ('complete' if terminal_present else 'needs_replan')
    assert 'immediate_effect_contract' not in result and 'ui_boundary_receipt' not in result


@pytest.mark.parametrize('fault', [None, 'bounds', 'hit', 'scope', 'hidden', 'disabled', 'protected', 'missing'])
def test_real_native_worker_preserves_live_destination_guards_with_cpu_providers(monkeypatch, fault):
    fixture, commands = _pointer_fixture(monkeypatch)
    state = fixture.adapter.observe()
    destination = next(n for n in state['nodes'] if n['name'] == 'Title')
    action = fixture.action('Preview', 'drag_drop', {'to_node_id': destination['id']})
    pointer = fixture.bridge._pointer
    def change_at_native_pointer(actual, source, item, window):
        record = fixture.bridge.handles[destination['id']]
        if fault == 'bounds':
            original = fixture.title.get_extents
            def changed(space):
                value = original(space)
                value.x += 1
                return value
            fixture.title.get_extents = changed
        elif fault == 'hit':
            fixture.window.get_accessible_at_point = lambda *_: fixture.button
        elif fault == 'missing':
            fixture.bridge.handles.pop(destination['id'])
        elif fault is not None:
            key, value = {'scope': ('scope_id', 'foreign'), 'hidden': ('visible', False),
                          'disabled': ('enabled', False), 'protected': ('sensitive', True)}[fault]
            record[1][key] = value
        return pointer(actual, source, item, window)
    fixture.bridge._pointer = change_at_native_pointer
    result = fixture.adapter.perform(action)
    if fault is None:
        assert result['status'] == 'dispatched'
        assert [command[1] for command in commands] == ['getmouselocation', 'mousemove',
            'getmouselocation', 'mousedown', 'mousemove', 'getmouselocation', 'mouseup']
    else:
        assert result['status'] == 'rejected' and not commands


def test_legacy_prefix_stops_at_drag_without_using_model_text_or_running_suffix():
    adapter, store, cancel, _ = store_fixture()
    steps = [{**drag(), 'expect': [{'kind': 'text', 'text': LABEL}]},
             {'id': 'suffix', 'verb': 'set_value', 'target': {'node_id': 'draft'},
              'args': {'value': 'Never executed'}, 'expect': [{'kind': 'text', 'text': LABEL}]}]
    with store.task_scope('session', 'desktop', cancel):
        result = run_verified_prefix(steps, execute_step=lambda step, _: immediate(store, cancel, step))
    assert len(adapter.calls) == 1 and adapter.calls[0]['verb'] == 'drag_drop'
    assert result['verified_prefix_receipt']['stop'] == 'opaque_input_boundary'
    assert result['verified_prefix_receipt']['unattempted_step_count'] == 1
    assert result['verified_prefix_receipt']['model_terminal_predicates_verified'] is False
    assert result['phase_handoff']['terminal_verified'] is False
    assert result['task_progress'] is False
