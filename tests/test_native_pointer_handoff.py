from __future__ import annotations

import copy
import threading
from types import SimpleNamespace

import httpx
import pytest
from test_native_manipulation import _pointer_fixture
from test_native_ui import Node

from portal.computer_use import ComputerUseStore, _effect_result
from portal.native_ui import NativeUiAdapter, NativeUiUnavailable
from qwen_omni_adapters.computer_use import RapidUiController


def _known_target(monkeypatch):
    fixture, commands = _pointer_fixture(monkeypatch)
    panel = Node('/provider-panel', 'panel', interfaces=['Component'])
    panel.get_parent = lambda: fixture.window
    panel.get_accessible_at_point = lambda *_args: None
    fixture.window.get_accessible_at_point = lambda *_args: panel
    monkeypatch.setattr('portal.native_ui._process_start', lambda _pid: '101')
    def request(value):
        if value['op'] == 'probe_pointer':
            return fixture.bridge.probe_pointer(value['binding'])
        return fixture.request(value)
    adapter = NativeUiAdapter('one', bridge_runner=request, pointing_url='http://127.0.0.1:8940',
                              pointing_model='fixture/moondream', pointing_revision='pinned')
    frame = {'key': 'frame-key', 'sha256': 'a' * 64, 'image': b'fixture-png',
             'geometry': {'x': 100, 'y': 200, 'width': 600, 'height': 400},
             'image_geometry': {'x': 100, 'y': 200, 'width': 600, 'height': 400},
             'capture_root': {'window_id': '1', 'geometry': {'x': 0, 'y': 0, 'width': 1920, 'height': 1200}},
             'pixel_offset': {'x': 0, 'y': 0}, 'capture_source': 'x11_root_screen', 'process_start': '101',
             'identity': dict(fixture.foreground), 'environment': {}}
    monkeypatch.setattr(adapter, '_visual_capture', lambda _state: copy.deepcopy(frame))
    requests = []
    def point(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(200, json={'model': 'fixture/moondream', 'revision': 'pinned',
            'points': [{'x': 1 / 6, 'y': .2}]}, request=httpx.Request('POST', url))
    monkeypatch.setattr('httpx.post', point)
    target = next(node for node in fixture.adapter.observe()['nodes'] if node['name'] == 'Preview')
    step = {'id': 'open', 'verb': 'double_click', 'target': {'node_id': target['id']}, 'args': {},
            'expect': [{'kind': 'focused', 'node_id': '$target'}]}
    adapter.prepare_step(step)
    return fixture, adapter, frame, requests, commands, target, step


def test_provider_unknown_known_id_gets_one_visual_capability_without_duplicate_semantic_node(monkeypatch):
    _fixture, adapter, _frame, requests, commands, target, step = _known_target(monkeypatch)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    assert len(requests) == 1
    assert len({node['id'] for node in state['nodes']}) == len(state['nodes'])
    observed = next(node for node in state['nodes'] if node['id'] == target['id'])
    assert observed['generation'] == target['generation']
    assert observed['operations'] == target['operations']
    actions = [item for item in controller.enumerate(step, state) if item['kind'] == 'action']
    assert len(actions) == 1 and actions[0]['node_id'] == target['id']
    assert actions[0]['expect'][0]['node_id'] == target['id']
    assert commands == []


def test_frame_binding_revision_alone_is_not_a_semantic_mutation():
    result = _effect_result({'status': 'verified', 'action': {'kind': 'action', 'id': 'a', 'verb': 'click', 'expect': []},
        'receipt': {'status': 'dispatched'},
        'before': {'session_id': 'one', 'scope': {'id': 'w'}, 'revision': 'frame-bound', 'base_semantic_revision': 'semantic'},
        'after': {'session_id': 'one', 'scope': {'id': 'w'}, 'revision': 'semantic'}})
    assert result['effect_receipt']['effect_state'] == 'unchanged'
    assert result['evidence_authority'] == 'unchanged_effect'


@pytest.mark.parametrize('selector', ['node_id', 'name'])
def test_guarded_known_target_dispatch_keeps_original_effect_target_and_reuses_one_point(monkeypatch, selector):
    fixture, adapter, _frame, requests, _commands, target, step = _known_target(monkeypatch)
    step['target'] = {selector: target['id'] if selector == 'node_id' else target['name']}
    dispatched = []
    def dispatch(identity, geometry, point, verb, **_kwargs):
        dispatched.append((identity, geometry, point, verb))
        fixture.button.state.values.add('FOCUSED')
        return {'status': 'dispatched', 'backend': 'x11_guarded_xtest'}
    monkeypatch.setattr('portal.native_ui._pointer_dispatch', dispatch)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    outcome = controller.run_step(step)
    assert outcome['status'] == 'verified'
    assert outcome['action']['node_id'] == target['id']
    assert outcome['action']['generation'] == target['generation']
    assert outcome['receipt']['point_receipt']['native_probe']['status'] == 'provider_unknown'
    assert outcome['receipt']['point_receipt']['native_probe']['input_attempted'] is False
    assert dispatched[0][2:] == ((200, 280), 'double_click')
    assert len(dispatched) == len(requests) == 1
    assert _effect_result(outcome)['effect_receipt']['effect_state'] == 'applied'
    assert controller.execute(outcome['action']['id']) == outcome
    assert len(dispatched) == len(requests) == 1


def test_complete_native_center_proof_skips_vision_and_uses_existing_native_dispatch(monkeypatch):
    fixture, adapter, _frame, requests, commands, _target, step = _known_target(monkeypatch)
    fixture.window.get_accessible_at_point = lambda *_args: fixture.button
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    assert 'base_semantic_revision' not in state and requests == []
    action = next(item for item in controller.enumerate(step, state) if item['kind'] == 'action')
    assert adapter.perform(action)['status'] == 'dispatched'
    assert requests == [] and commands[-1][1] == 'click'


@pytest.mark.parametrize('cause', ['protected', 'other_control', 'other_window', 'cycle', 'depth', 'budget', 'read_error'])
def test_blocked_incomplete_or_failed_native_proof_never_calls_vision_or_input(monkeypatch, cause):
    fixture, adapter, _frame, requests, commands, _target, _step = _known_target(monkeypatch)
    if cause in {'protected', 'other_control', 'other_window'}:
        node = fixture.secret if cause == 'protected' else fixture.title if cause == 'other_control' else Node('/other', 'frame')
        node.get_parent = lambda: fixture.window
        node.get_accessible_at_point = lambda *_args: None
        fixture.window.get_accessible_at_point = lambda *_args: node
    elif cause == 'cycle':
        panel = Node('/cycle', 'panel', interfaces=['Component'])
        panel.get_parent = lambda: fixture.window
        panel.get_accessible_at_point = lambda *_args: panel
        fixture.window.get_accessible_at_point = lambda *_args: panel
    elif cause == 'depth':
        panels = [Node(f'/depth-{index}', 'panel', interfaces=['Component']) for index in range(33)]
        for index, panel in enumerate(panels):
            panel.get_parent = lambda: fixture.window
            panel.get_accessible_at_point = lambda *_args, i=index: panels[min(i + 1, 32)]
        fixture.window.get_accessible_at_point = lambda *_args: panels[0]
    else:
        def fail(*_args):
            if cause == 'budget':
                fixture.bridge.reads = fixture.bridge.MAX_READS
                fixture.bridge._read(lambda: None)
            raise NativeUiUnavailable('native_observation_unavailable')
        monkeypatch.setattr(fixture.bridge, '_hit_evidence', fail)
    with pytest.raises(NativeUiUnavailable):
        adapter.observe()
    assert requests == commands == []


@pytest.mark.parametrize('cause', ['hidden', 'disabled', 'protected', 'duplicate', 'hidden_duplicate', 'invoke', 'no_bounds'])
def test_exact_matching_semantic_ineligibility_or_ambiguity_cannot_be_erased_by_vision(monkeypatch, cause):
    fixture, adapter, _frame, requests, commands, target, step = _known_target(monkeypatch)
    step['target'] = {'name': target['name']}
    if cause == 'hidden':
        fixture.button.state.values.remove('SHOWING')
    elif cause == 'disabled':
        fixture.button.state.values.remove('SENSITIVE')
    elif cause == 'protected':
        fixture.button.role = 'password text'
        step['target'] = {'node_id': target['id']}
    elif cause in {'duplicate', 'hidden_duplicate'}:
        duplicate = Node('/other-preview', 'push button', 'Preview', interfaces=['Component', 'Action'])
        duplicate.get_parent = lambda: fixture.window
        duplicate.get_extents = fixture.button.get_extents
        if cause == 'hidden_duplicate':
            duplicate.state.values.remove('SHOWING')
        fixture.window.children.append(duplicate)
    elif cause == 'invoke':
        step['verb'] = 'invoke'
    else:
        fixture.button.get_extents = lambda _space: SimpleNamespace(x=150, y=250, width=0, height=60)
    adapter.prepare_step(step)
    state = adapter.observe()
    assert 'base_semantic_revision' not in state and requests == commands == []


@pytest.mark.parametrize('fault', ['extra_field', 'binding', 'input_attempted', 'incomplete', 'reason', 'point', 'start', 'mapping'])
def test_untrusted_or_malformed_private_probe_response_is_not_visual_eligibility(monkeypatch, fault):
    fixture, adapter, _frame, requests, commands, _target, _step = _known_target(monkeypatch)
    original = adapter._runner
    def tamper(value):
        result = original(value)
        if value['op'] == 'probe_pointer':
            if fault == 'extra_field':
                result['claimed_safe'] = True
            elif fault == 'binding':
                result['binding']['generation'] = 'old'
            elif fault == 'input_attempted':
                result['input_attempted'] = True
            elif fault == 'incomplete':
                result['complete'] = False
            elif fault == 'reason':
                result['reason'] = 'native_hit_depth_limit'
            elif fault == 'point':
                result['point']['x'] += 1
            elif fault == 'start':
                result['process_start'] = 101
            else:
                result['coordinate_binding'] = {'claimed': 'mapping'}
        return result
    adapter._runner = tamper
    with pytest.raises(NativeUiUnavailable):
        adapter.observe()
    assert requests == commands == [] and fixture.button.events == []


@pytest.mark.parametrize('fault', ['empty', 'outside', 'model', 'revision', 'nan'])
def test_point_must_be_pinned_and_inside_the_same_observed_control(monkeypatch, fault):
    _fixture, adapter, _frame, _requests, commands, _target, _step = _known_target(monkeypatch)
    result = {'model': 'fixture/moondream', 'revision': 'pinned', 'points': [{'x': 1 / 6, 'y': .2}]}
    if fault == 'empty':
        result['points'] = []
    elif fault == 'outside':
        result['points'] = [{'x': .9, 'y': .9}]
    elif fault in {'model', 'revision'}:
        result[fault] = 'wrong'
    else:
        result['points'] = [{'x': float('nan'), 'y': .2}]
    monkeypatch.setattr('httpx.post', lambda url, **_kwargs: SimpleNamespace(
        content=b'bounded', raise_for_status=lambda: None, json=lambda: result))
    with pytest.raises(NativeUiUnavailable):
        adapter.observe()
    assert commands == []


@pytest.mark.parametrize('fault', ['frame', 'bounds', 'generation', 'hidden', 'positive_block', 'owner_start'])
def test_perform_refresh_rejects_changed_bindings_without_another_point_call(monkeypatch, fault):
    fixture, adapter, frame, requests, commands, target, step = _known_target(monkeypatch)
    controller = RapidUiController(adapter, authorize=lambda *_args: True)
    state = controller.observe()
    action = next(item for item in controller.enumerate(step, state) if item['kind'] == 'action')
    if fault == 'frame':
        frame['key'] = 'new-frame'
    elif fault == 'bounds':
        fixture.button.get_extents = lambda _space: SimpleNamespace(x=151, y=250, width=100, height=60)
    elif fault == 'generation':
        fixture.window.children.remove(fixture.button)
        fixture.bridge.observe()
        fixture.window.children.append(fixture.button)
    elif fault == 'hidden':
        fixture.button.state.values.remove('SHOWING')
    elif fault == 'positive_block':
        fixture.window.get_accessible_at_point = lambda *_args: fixture.secret
    else:
        monkeypatch.setattr('portal.native_ui._process_start', lambda _pid: '102')
    monkeypatch.setattr('portal.native_ui._pointer_dispatch', lambda *_args, **_kwargs:
                        pytest.fail('A stale bound visual capability cannot dispatch'))
    assert adapter.perform(action)['status'] == 'rejected'
    assert len(requests) == 1 and commands == []
    assert target['generation'] == '1'


@pytest.mark.parametrize('boundary', ['before_probe', 'during_point', 'before_perform'])
def test_owned_cancellation_prevents_new_visual_authority_or_input(monkeypatch, boundary):
    _fixture, adapter, _frame, requests, commands, _target, step = _known_target(monkeypatch)
    cancelled = threading.Event()
    adapter._cancelled = cancelled.is_set
    if boundary == 'before_probe':
        cancelled.set()
    elif boundary == 'during_point':
        original = httpx.post
        def cancel_after_point(*args, **kwargs):
            result = original(*args, **kwargs)
            cancelled.set()
            return result
        monkeypatch.setattr('httpx.post', cancel_after_point)
    if boundary != 'before_perform':
        with pytest.raises(NativeUiUnavailable) as exc:
            adapter.observe()
        assert exc.value.code == 'native_operation_cancelled'
    else:
        controller = RapidUiController(adapter, authorize=lambda *_args: True)
        state = controller.observe()
        action = next(item for item in controller.enumerate(step, state) if item['kind'] == 'action')
        cancelled.set()
        with pytest.raises(NativeUiUnavailable):
            adapter.perform(action)
    assert len(requests) == (0 if boundary == 'before_probe' else 1) and commands == []


def test_store_native_cancellation_callback_belongs_to_exact_controller_epoch():
    store = ComputerUseStore(SimpleNamespace(pointing_url=None))
    first = store._owned('one', 'desktop')
    first.cancel.set()
    with pytest.raises(NativeUiUnavailable):
        first.adapter._check_cancelled()
    event = store.begin_task('one')
    second = store._owned('one', 'desktop')
    assert second.cancel is event and not event.is_set()
    second.adapter._check_cancelled()
    first.cancel.set()
    assert not event.is_set()
    store.clear('one')


@pytest.mark.parametrize('ancestry', ['orphan', 'foreign_role', 'embedded_provider', 'protected_above_target', 'defunct_hit'])
def test_only_complete_owned_native_ancestry_can_admit_visual_handoff(monkeypatch, ancestry):
    fixture, adapter, _frame, requests, commands, _target, _step = _known_target(monkeypatch)
    panel = Node('/ancestry-panel', 'panel', interfaces=['Component'])
    panel.get_accessible_at_point = lambda *_args: None
    panel.get_parent = lambda: fixture.window
    fixture.window.get_accessible_at_point = lambda *_args: panel
    if ancestry == 'orphan':
        panel.get_parent = lambda: None
    elif ancestry == 'foreign_role':
        panel.role = 'frame'
    elif ancestry == 'embedded_provider':
        panel.app.bus_name = ':1.99'
    elif ancestry == 'protected_above_target':
        fixture.window.get_accessible_at_point = lambda *_args: fixture.button
        fixture.button.get_parent = lambda: fixture.secret
        fixture.secret.get_parent = lambda: fixture.window
    else:
        panel.state.values.add('DEFUNCT')
    if ancestry == 'embedded_provider':
        state = adapter.observe()
        assert 'base_semantic_revision' in state and len(requests) == 1
    else:
        with pytest.raises(NativeUiUnavailable):
            adapter.observe()
        assert requests == []
    assert commands == []


@pytest.mark.parametrize('fault', ['extra', 'scope', 'revision', 'generation', 'bounds', 'input_verb'])
def test_private_probe_request_is_closed_and_cannot_reuse_stale_or_different_target_binding(monkeypatch, fault):
    fixture, adapter, _frame, requests, commands, target, step = _known_target(monkeypatch)
    state = adapter._semantic_observe()
    binding = {'session_id': 'one', 'revision': state['revision'], 'scope_id': state['scope']['id'],
               'node_id': target['id'], 'generation': target['generation'], 'verb': step['verb'], 'bounds': target['bounds']}
    if fault == 'extra':
        binding['coordinates'] = [200, 280]
    elif fault == 'scope':
        binding['scope_id'] = 'other'
    elif fault in {'revision', 'generation'}:
        binding[fault] = 'old'
    elif fault == 'bounds':
        binding['bounds'] = {**binding['bounds'], 'width': 999}
    else:
        binding['verb'] = 'type_text'
    with pytest.raises(NativeUiUnavailable):
        fixture.bridge.probe_pointer(binding)
    assert requests == commands == []


def test_point_over_another_protected_control_is_rejected_even_inside_known_target(monkeypatch):
    fixture, adapter, _frame, requests, commands, _target, _step = _known_target(monkeypatch)
    fixture.secret.get_extents = fixture.button.get_extents
    with pytest.raises(NativeUiUnavailable) as exc:
        adapter.observe()
    assert exc.value.code == 'native_visual_point_over_protected_control'
    assert len(requests) == 1 and commands == []


def test_successful_visual_dispatch_with_unchanged_semantics_does_not_claim_mutation(monkeypatch):
    fixture, adapter, _frame, requests, commands, _target, step = _known_target(monkeypatch)
    fixture.button.state.values.add('FOCUSED')
    monkeypatch.setattr('portal.native_ui._pointer_dispatch', lambda *_args, **_kwargs:
                        {'status': 'dispatched', 'backend': 'x11_guarded_xtest'})
    outcome = RapidUiController(adapter, authorize=lambda *_args: True).run_step(step)
    assert outcome['status'] == 'verified'
    effect = _effect_result(outcome)
    assert effect['effect_receipt']['effect_state'] == 'unchanged'
    assert effect['evidence_authority'] == 'unchanged_effect'
    assert effect['effect_receipt']['before_semantic_revision'] == effect['effect_receipt']['after_semantic_revision']
    assert len(requests) == 1 and commands == []


def test_truncated_semantic_coverage_does_not_authorize_known_target_visual_handoff(monkeypatch):
    _fixture, adapter, _frame, requests, commands, _target, _step = _known_target(monkeypatch)
    original = adapter._runner
    def truncated(request):
        assert request['op'] == 'observe', 'Incomplete semantic coverage must not start a visual probe'
        state = original(request)
        state['truncated'] = True
        return state
    adapter._runner = truncated
    state = adapter.observe()
    assert state['truncated'] is True and 'base_semantic_revision' not in state
    assert requests == commands == []
