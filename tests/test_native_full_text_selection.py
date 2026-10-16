"""Exact complete text selection chooses one guarded physical backend before native."""
from __future__ import annotations

import copy
import subprocess
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture

from portal.native_ui import NativeUiUnavailable
from qwen_omni_adapters.computer_use import RapidUiController


@pytest.fixture
def selection(monkeypatch):
    fixture = Fixture('full-text-selection')
    fixture.title.value = 'alpha beta'
    fixture.title.state.values.add('FOCUSED')
    fixture.title.selection_range = None
    fixture.commands, fixture.setters = [], []
    fixture.api.Text.get_n_selections = lambda node: int(node.selection_range is not None)
    fixture.api.Text.get_selection = lambda node, _index: SimpleNamespace(
        start_offset=node.selection_range[0], end_offset=node.selection_range[1])

    def uncertain(*args):
        fixture.setters.append(args[1:])
        raise RuntimeError('CPU provider setter failure; historical class not inferred')

    fixture.api.Text.add_selection = fixture.api.Text.set_selection = uncertain
    fixture.geometry = {'x': 10, 'y': 20, 'width': 800, 'height': 900}
    monkeypatch.setattr('portal.native_ui._x11_geometry', lambda *_: dict(fixture.geometry))

    def run(argv, **kwargs):
        assert argv == ['xdotool', 'key', '--clearmodifiers', 'Control_L+a']
        assert kwargs.get('timeout') == 2 and kwargs.get('check') is False
        fixture.commands.append(argv)
        fixture.title.selection_range = (0, len(fixture.title.value))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('portal.native_ui.subprocess.run', run)
    return fixture


def select(fixture, *, start=0, end=None):
    end = len(fixture.title.value) if end is None else end
    controller = RapidUiController(fixture.adapter, authorize=lambda *_: True,
        max_observations=1, verification_timeout_s=.01)
    step = {'id': 'select-declared-range', 'verb': 'select_text', 'target': {'name': 'Title'},
        'args': {'start': start, 'end': end},
        'expect': [{'kind': 'text_selection', 'node_id': '$target', 'start': start, 'end': end}]}
    return controller, step, controller.run_step(step)


@pytest.mark.parametrize('value', ['alpha beta', 'e\u0301🙂文\n', 'a' * 1077, '文' * 2048],
    ids=['plain', 'unicode', 'count-matched-1077', 'complete-budget-2048'])
def test_complete_range_uses_one_physical_chord_and_verifies_exact_native_range(selection, value):
    selection.title.value = value
    _controller, _step, outcome = select(selection)
    assert outcome['status'] == 'verified'
    assert outcome['receipt']['backend'] == 'x11_guarded_xtest'
    assert selection.setters == [] and len(selection.commands) == 1
    assert outcome['action']['verb'] == 'select_text'
    assert outcome['action']['args'] == {'start': 0, 'end': len(value)}
    node = next(n for n in outcome['after']['nodes'] if n['name'] == 'Title')
    assert node['value'] == value and node['text_selection'] == {'start': 0, 'end': len(value)}


def test_partial_range_native_uncertainty_never_becomes_select_all(selection):
    controller, step, outcome = select(selection, end=1)
    assert outcome['status'] == 'unresolved' and selection.commands == []
    assert selection.setters and outcome['receipt']['reason'] == 'native_pattern_effect_unresolved'
    replay = controller.run_step(step)
    assert replay['action']['idempotency_key'] == outcome['action']['idempotency_key']
    assert selection.commands == [] and len(selection.setters) == 1


def direct_action(fixture):
    return fixture.action('Title', 'select_text', {'start': 0, 'end': len(fixture.title.value)})


@pytest.mark.parametrize('fault', [
    'missing', 'empty', 'unknown-count', 'boolean-count', 'wrong-count', 'unknown-range',
    'boolean-start', 'boolean-end', 'short-range', 'nonzero-start', 'extra-range',
    'unknown-truncation', 'boolean-like-truncation', 'truncated',
    'no-type-text', 'no-key-chord',
])
def test_incomplete_or_malformed_knowledge_cannot_choose_physical_input(selection, fault):
    original = selection.bridge.observe

    def observe(**kwargs):
        result = original(**kwargs)
        item = next(item for node, item, _index in selection.bridge.handles.values()
                    if node is selection.title)
        metadata = item['text_observation']
        if fault == 'missing':
            item.pop('text_observation')
        elif fault == 'empty':
            item['text_observation'] = {}
        elif fault in {'no-type-text', 'no-key-chord'}:
            item['operations'].remove(fault[3:].replace('-', '_'))
        elif fault == 'unknown-count':
            metadata.pop('character_count')
        elif fault == 'boolean-count':
            metadata['character_count'] = True
        elif fault == 'wrong-count':
            metadata['character_count'] -= 1
        elif fault == 'unknown-range':
            metadata.pop('captured_range')
        elif fault == 'boolean-start':
            metadata['captured_range']['start'] = False
        elif fault == 'boolean-end':
            selection.title.value = 'a'
            metadata['character_count'] = 1
            metadata['captured_range']['end'] = True
        elif fault == 'short-range':
            metadata['captured_range']['end'] -= 1
        elif fault == 'nonzero-start':
            metadata['captured_range']['start'] = 1
        elif fault == 'extra-range':
            metadata['captured_range']['unknown'] = 0
        elif fault == 'unknown-truncation':
            metadata.pop('truncated')
        elif fault == 'boolean-like-truncation':
            metadata['truncated'] = 0
        elif fault == 'truncated':
            metadata['truncated'] = True
        return result

    action = direct_action(selection)
    selection.bridge.observe = observe
    result = selection.bridge.perform(action)
    assert result['status'] in {'dispatched', 'rejected'}
    assert result.get('backend') is None and selection.commands == []


@pytest.mark.parametrize('fault', ['unfocused', 'readonly', 'noneditable', 'protected', 'disabled', 'hidden'])
def test_unqualified_control_never_uses_physical_selection(selection, fault):
    if fault == 'unfocused':
        selection.title.state.values.remove('FOCUSED')
    elif fault == 'readonly':
        selection.title.state.values.add('READ_ONLY')
    elif fault == 'noneditable':
        selection.title.state.values.remove('EDITABLE')
    elif fault == 'protected':
        selection.title.role = 'password text'
    elif fault == 'disabled':
        selection.title.state.values.remove('SENSITIVE')
    elif fault == 'hidden':
        selection.title.state.values.remove('SHOWING')
    action = direct_action(selection) if fault not in {'protected', 'disabled', 'hidden'} else None
    if action is not None:
        assert selection.adapter.perform(action).get('backend') is None
    else:
        assert not any(node.get('operations') for node in selection.adapter.observe()['nodes']
                       if node['id'] == selection.bridge._opaque(':1.42:/title'))
    assert selection.commands == []


@pytest.mark.parametrize('fault', ['count', 'boolean-count', 'value', 'short-text', 'non-string',
    'focus', 'editable', 'readonly', 'hidden', 'protected', 'generation', 'identity',
    'provider', 'foreground', 'geometry', 'read-error', 'budget'])
def test_late_change_or_precommand_read_failure_rejects_without_backend_or_native_retry(selection, monkeypatch, fault):
    action = direct_action(selection)
    original_text = selection.api.Text.get_text
    original_count = selection.api.Text.get_character_count
    original_state = selection.title.get_state_set
    original_provider = selection.bridge._provider_guard
    armed = False
    reads = 0

    def geometry(_identity):
        nonlocal armed
        if not armed:
            armed = True
            return dict(selection.geometry)
        return {**selection.geometry, 'x': 11 if fault == 'geometry' else 10}

    def state():
        if armed:
            if fault in {'focus', 'editable', 'hidden'}:
                selection.title.state.values.discard({'focus': 'FOCUSED', 'editable': 'EDITABLE', 'hidden': 'SHOWING'}[fault])
            if fault == 'readonly':
                selection.title.state.values.add('READ_ONLY')
            if fault == 'protected':
                selection.title.role = 'password text'
            if fault == 'generation':
                selection.bridge.generations[':1.42:/title'] += 1
            if fault == 'identity':
                selection.title.path = '/replacement'
            if fault == 'read-error' and reads >= 2:
                raise RuntimeError('private provider error')
            if fault == 'budget' and reads >= 2:
                selection.bridge.reads = selection.bridge.MAX_READS
        return original_state()

    def text(node, start, end):
        if armed:
            if fault == 'value':
                return 'ALPHA BETA'
            if fault == 'short-text':
                return 'alpha'
            if fault == 'non-string':
                return None
        return original_text(node, start, end)

    def provider():
        nonlocal reads
        reads += 1
        if armed and fault == 'provider' and reads > 1:
            return False
        if armed and fault == 'foreground' and reads > 1:
            selection.foreground['window_id'] = '18'
        return original_provider()

    selection.api.Text.get_character_count = lambda node: (
        True if armed and fault == 'boolean-count' else
        11 if armed and fault == 'count' else original_count(node))
    selection.api.Text.get_text = text
    selection.title.get_state_set = state
    selection.bridge._provider_guard = provider
    monkeypatch.setattr('portal.native_ui._x11_geometry', geometry)
    result = selection.adapter.perform(action)
    assert result['status'] == 'rejected' and result.get('backend') is None
    assert selection.commands == [] and selection.setters == []


@pytest.mark.parametrize('failure', ['nonzero', 'oserror', 'timeout'])
def test_uncertain_physical_attempt_never_falls_back_or_replays(selection, monkeypatch, failure):
    def run(argv, **_kwargs):
        selection.commands.append(argv)
        if failure == 'oserror':
            raise OSError('uncertain launch')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(argv, 2)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr('portal.native_ui.subprocess.run', run)
    controller, step, outcome = select(selection)
    assert outcome['status'] == 'unresolved'
    assert outcome['receipt']['status'] == 'dispatched'
    assert outcome['receipt']['backend'] == 'x11_guarded_xtest'
    assert selection.setters == [] and len(selection.commands) == 1
    replay = controller.run_step(copy.deepcopy(step))
    assert replay['action'] == outcome['action']
    assert len(selection.commands) == 1 and selection.setters == []


def test_metadata_clarification_cannot_retry_an_uncertain_cached_partial_action(selection):
    controller, step, outcome = select(selection, end=1)
    selection.api.Text.get_n_selections = lambda _node: 1
    selection.api.Text.get_selection = lambda _node, _index: SimpleNamespace(start_offset=-1, end_offset=-1)
    replay = controller.run_step(step)
    assert replay['action'] == outcome['action'] and selection.commands == []
    assert len(selection.setters) == 1


def test_adapter_cancellation_prevents_full_range_dispatch(selection):
    action = direct_action(selection)
    selection.adapter._cancelled = lambda: True
    with pytest.raises(NativeUiUnavailable) as error:
        selection.adapter.perform(action)
    assert error.value.code == 'native_operation_cancelled'
    assert selection.commands == [] and selection.setters == []


@pytest.mark.parametrize('trigger', ['text', 'provider'])
@pytest.mark.parametrize('change', ['unfocused', 'readonly', 'noneditable', 'defunct'])
def test_final_state_read_follows_text_and_provider_roundtrips(selection, trigger, change):
    action = direct_action(selection)
    original_text, original_guard = selection.api.Text.get_text, selection.bridge._provider_guard
    text_reads = provider_reads = 0

    def mutate():
        if change in {'unfocused', 'noneditable'}:
            selection.title.state.values.remove('FOCUSED' if change == 'unfocused' else 'EDITABLE')
        else:
            selection.title.state.values.add('READ_ONLY' if change == 'readonly' else 'DEFUNCT')

    def text(node, start, end):
        nonlocal text_reads
        text_reads += 1
        result = original_text(node, start, end)
        if text_reads == 2 and trigger == 'text':
            mutate()
        return result

    def guard():
        nonlocal provider_reads
        provider_reads += 1
        result = original_guard()
        if provider_reads == 3 and trigger == 'provider':
            mutate()
        return result

    selection.api.Text.get_text = text
    selection.bridge._provider_guard = guard
    result = selection.adapter.perform(action)
    assert result == {'status': 'rejected', 'reason': 'native_text_selection_target_changed'}
    assert selection.commands == [] and selection.setters == []


@pytest.mark.parametrize('count', [2049, 4096])
def test_clipped_value_never_authorizes_select_all(selection, count):
    selection.title.value = 'x' * count
    action = selection.action('Title', 'select_text', {'start': 0, 'end': 2048})
    result = selection.adapter.perform(action)
    assert result.get('backend') is None and selection.commands == []
    assert selection.setters == [(0, 2048)]


@pytest.mark.parametrize('change', ['same-pid-other-window', 'pid-reused', 'different-owner'])
def test_exact_task_scope_rejects_late_host_identity_change(selection, monkeypatch, change):
    import portal.native_ui as native

    start, owner = ['4201'], [42]
    monkeypatch.setattr(native, '_process_start', lambda _pid: start[0])
    monkeypatch.setattr(native, '_x11_window_owner', lambda *_args, **_kwargs: {'process_id': owner[0]})
    command = native.subprocess.run
    monkeypatch.setattr(native.subprocess, 'run', lambda argv, **kwargs:
        SimpleNamespace(returncode=1, stdout='') if argv[0] == 'xprop' else command(argv, **kwargs))
    scope = {'schema': 'robit.omni.native-task-scope.v1', 'windows': [
        {'window_id': '17', 'process_id': 42, 'process_start': '4201'}]}
    selection.bridge.window_scope = copy.deepcopy(scope)
    selection.adapter._window_scope = copy.deepcopy(scope)
    action = direct_action(selection)
    original, calls = selection.bridge._provider_guard, 0

    def guard():
        nonlocal calls
        calls += 1
        result = original()
        if calls == 3:
            if change == 'same-pid-other-window':
                selection.foreground['window_id'] = '18'
            elif change == 'pid-reused':
                start[0] = '4202'
            else:
                owner[0] = 43
        return result

    selection.bridge._provider_guard = guard
    assert selection.adapter.perform(action)['status'] == 'rejected'
    assert selection.commands == [] and selection.setters == []


@pytest.mark.parametrize('field', ['session_id', 'revision', 'scope_id', 'generation'])
def test_original_action_authority_is_required(selection, field):
    action = direct_action(selection)
    action[field] = 'wrong-authority'
    assert selection.adapter.perform(action)['status'] == 'rejected'
    assert selection.commands == [] and selection.setters == []


def test_already_selected_full_text_does_not_claim_new_effect(selection):
    selection.title.selection_range = (0, len(selection.title.value))
    _controller, _step, outcome = select(selection)
    assert outcome['status'] == 'verified' and outcome['before'] == outcome['after']
    assert selection.setters == [] and len(selection.commands) == 1


def test_new_metadata_cannot_authorize_select_all_for_partial_bound_value(selection):
    action = direct_action(selection)
    original = selection.bridge.observe

    def clipped(**kwargs):
        result = original(**kwargs)
        item = next(item for node, item, _index in selection.bridge.handles.values()
                    if node is selection.title)
        item['value'] = item['value'][:-1]
        return result

    selection.bridge.observe = clipped
    result = selection.bridge.perform(action)
    assert result.get('backend') is None and selection.commands == []
    assert len(selection.setters) == 1


def test_provider_refresh_same_length_value_change_rejects_before_keyboard(selection):
    action = direct_action(selection)
    original, calls = selection.bridge._provider_guard, 0

    def guard():
        nonlocal calls
        calls += 1
        result = original()
        if calls == 3:
            selection.title.value = 'ALPHA BETA'
        return result

    selection.bridge._provider_guard = guard
    result = selection.adapter.perform(action)
    assert result == {'status': 'rejected', 'reason': 'native_text_selection_value_changed'}
    assert selection.commands == [] and selection.setters == []
