"""Navigation input has observed consequences, never invented effect predicates."""
import copy
import threading

import pytest
from jsonschema import Draft202012Validator
from test_ui_observation_boundary import CELL, Adapter, control

from portal.computer_use import ComputerUseStore
from portal.ui_task import _grounded_phase_schema, run_native_goal
from qwen_omni_adapters.computer_use import ComputerUseError, parse_intent, parse_step


class Navigation(Adapter):
    def observe(self):
        state = super().observe()
        state['nodes'][0]['operations'] += ['key_chord', 'scroll']
        return state


def intent(verb='key_chord'):
    return {'id': 'navigate', 'verb': verb, 'target': {'node_id': CELL},
            'args': {'keys': ['Control_L', 'l']} if verb == 'key_chord' else {'direction': 'down'}}


def phase(verb='key_chord'):
    return {'kind': 'ui_boundary', 'objective': 'Reveal the next navigation control', 'step': intent(verb)}


@pytest.mark.parametrize('verb', ['key_chord', 'scroll'])
def test_active_schema_exposes_input_without_an_invented_effect(verb):
    observation = {'state': Navigation().observe()}
    active = Draft202012Validator(_grounded_phase_schema({'native_ui'}, observation, intent_choice=True))
    assert active.is_valid(phase(verb))
    assert not active.is_valid({**phase(verb), 'step': {**intent(verb), 'expect': [{'kind': 'focused', 'node_id': '$target'}]}})
    assert not active.is_valid({**phase(verb), 'step': {**intent(verb), 'target': {'node_id': 'unobserved'}}})
    assert not Draft202012Validator(_grounded_phase_schema({'native_ui'}, observation)).is_valid(phase(verb))
    assert parse_intent(intent(verb)) == intent(verb)
    with pytest.raises(ComputerUseError):
        parse_step(intent(verb))


@pytest.mark.parametrize('verb', ['key_chord', 'scroll'])
def test_input_observes_once_and_retains_unknown_effect_without_progress(verb):
    adapter = Navigation()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=True)
    with store.task_scope('session', 'desktop', cancel):
        result = store.observation_boundary('session', intent(verb), goal='Read the requested file', objective=phase()['objective'])
    assert len(adapter.calls) == 1 and adapter.semantic_reads == 1
    assert result['status'] == 'observed_unverified'
    assert result['effect_receipt']['effect_state'] == 'unknown'
    assert result['task_progress'] is False and result['new_effect_count'] == 0
    assert result['ui_boundary_receipt']['verified'] is False
    assert result['ui_boundary_receipt']['input_attempted'] is True
    assert result['selection_receipts'][0]['selection_status'] == 'semantic_unique'
    assert not result['policy_receipts']


@pytest.mark.parametrize('change', ['generation', 'focus', 'capability', 'scope', 'revision'])
def test_changed_target_cannot_dispatch_or_refresh_input(change):
    adapter = Navigation()
    c = control(adapter)
    original = adapter.observe
    reads = 0
    def observe():
        nonlocal reads
        reads += 1
        state = original()
        if reads == 2:
            if change == 'generation':
                state['nodes'][0]['generation'] = '2'
            elif change == 'focus':
                state['nodes'][0]['focused'] = True
                state['revision'] = 'changed-focus'
            elif change == 'capability':
                state['nodes'][0]['operations'].remove('key_chord')
            elif change == 'scope':
                state['scope']['id'] = 'other'
            else:
                state['revision'] = 'changed'
        return state
    adapter.observe = observe
    result = c.run_boundary(intent())
    assert result['status'] == 'rejected'
    assert not adapter.calls and not result.get('pre_input_refresh')


@pytest.mark.parametrize('fault', ['objective', 'goal', 'task_binding', 'cancel', 'no_protocol'])
def test_private_input_preconditions_fail_before_dispatch(fault):
    adapter = Navigation()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=fault != 'no_protocol')
    if fault == 'cancel':
        cancel.set()
    kwargs = {'goal': '' if fault == 'goal' else 'Read the requested file',
              'objective': ' ' if fault == 'objective' else 'Open the location field'}
    if fault == 'task_binding':
        with pytest.raises(ComputerUseError):
            store.observation_boundary('session', intent(), **kwargs)
    else:
        with store.task_scope('session', 'desktop', cancel):
            if fault in {'objective', 'goal', 'no_protocol'}:
                with pytest.raises(ComputerUseError):
                    store.observation_boundary('session', intent(), **kwargs)
            else:
                assert store.observation_boundary('session', intent(), **kwargs)['status'] == 'cancelled'
    assert not adapter.calls


def test_transport_uncertainty_prevents_repeating_key_input_with_a_new_id():
    adapter = Navigation()
    def uncertain(action):
        adapter.calls.append(copy.deepcopy(action))
        raise RuntimeError('lost acknowledgement after key input')
    adapter.perform = uncertain
    c = control(adapter)
    first = c.run_boundary(intent())
    assert first['status'] == 'unresolved'
    assert c.run_boundary({**intent(), 'id': 'fresh-story'}) == first
    assert len(adapter.calls) == 1


def test_task_loop_admits_input_objective_and_preserves_observed_unknown_result():
    adapter = Navigation()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=True)
    packets = []
    def compile_phase(packet):
        packets.append(copy.deepcopy(packet))
        return phase() if len(packets) == 1 else {'kind': 'blocked', 'reason': 'No whole-task claim'}
    def execute(name, arguments):
        assert name == 'native_ui' and arguments['operation'] == 'observation_boundary'
        with store.task_scope('session', 'desktop', cancel):
            return store.observation_boundary('session', arguments['intent'], goal='Read the file',
                                             objective=arguments['objective'], feedback=arguments['feedback'])
    events = list(run_native_goal('Read the file', observe=lambda: {'state': adapter.observe()},
        compile_phase=compile_phase, execute=execute, cancel=threading.Event().set,
        allowed_tools={'native_ui'}, intent_choice=True))
    assert len(adapter.calls) == 1
    assert packets[1]['last_result']['effect_receipt']['effect_state'] == 'unknown'
    assert events[-1]['response']['portal']['native_task']['task_verified'] is False
