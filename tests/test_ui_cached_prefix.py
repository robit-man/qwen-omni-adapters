"""Historical workflow receipts cannot substitute for current prerequisites."""
from __future__ import annotations

import copy
import threading

import pytest

from qwen_omni_adapters.computer_use import ComputerUseError, RapidUiController, _digest


class TextAdapter:
    def __init__(self):
        self.value = 'ABCDEF'
        self.selection = None
        self.focused = True
        self.checked = True
        self.selected = ['one']
        self.marker = True
        self.session = 'prefix-session'
        self.scope = 'window'
        self.url = 'https://example.com/source'
        self.generation = '1'
        self.sensitive = False
        self.missing = False
        self.inputs = []
        self.reads = 0
        self.on_read = None
        self.metadata = None
        self.truncated = False

    def observe(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self.reads)
        nodes = [{'id': 'window', 'scope_id': self.scope, 'generation': '1',
                  'role': 'window', 'name': 'Fixture', 'visible': True, 'enabled': True,
                  'sensitive': False, 'operations': ['switch_window']}]
        if not self.missing:
            nodes.append({'id': 'text', 'scope_id': self.scope, 'generation': self.generation,
                'role': 'textbox', 'name': '', 'visible': True, 'enabled': True,
                'sensitive': self.sensitive, 'focused': self.focused, 'value': self.value,
                'checked': self.checked, 'selected': self.selected,
                **({'text_selection': self.selection} if self.selection else {}),
                **({'text_observation': self.metadata} if self.metadata else {}),
                'operations': ['focus', 'key_chord', 'type_text']})
        nodes.append({'id': 'marker', 'scope_id': self.scope, 'generation': '1',
            'role': 'button', 'name': 'marker' if self.marker else 'other',
            'visible': self.marker, 'enabled': True, 'sensitive': False, 'operations': []})
        return {'session_id': self.session, 'revision': _digest([self.value, self.selection]),
            'scope': {'id': self.scope, 'kind': 'desktop', 'url': self.url},
            'nodes': nodes, 'text': 'marker' if self.marker else '', 'truncated': self.truncated}

    def perform(self, action):
        self.inputs.append(copy.deepcopy(action))
        if action['verb'] == 'key_chord':
            self.selection = {'start': 0, 'end': len(self.value)}
        elif action['verb'] == 'type_text':
            text = action['args']['text'].replace('\n', '')  # Saved042 backend fault shape.
            self.value = text if self.selection else self.value + text
            self.selection = None
            if self.metadata:
                self.metadata.update(character_count=len(self.value),
                                     captured_range={'start': 0, 'end': len(self.value)})
        return {'status': 'dispatched'}


def controller(adapter):
    return RapidUiController(adapter, authorize=lambda *_: True, max_observations=1,
                             verification_timeout_s=.05)


def focus(expect=None):
    return {'id': 'focus-textbox', 'verb': 'focus', 'target': {'node_id': 'text'},
            'args': {}, 'expect': expect or [{'kind': 'focused', 'node_id': 'text'}]}


def select():
    return {'id': 'select-all-text', 'verb': 'key_chord', 'target': {'node_id': 'text'},
            'args': {'keys': ['Control_L', 'a']},
            'expect': [{'kind': 'text_selection', 'node_id': 'text', 'start': 0, 'end': 6}]}


def write(text):
    return {'id': 'type-selection-index', 'verb': 'type_text', 'target': {'node_id': 'text'},
            'args': {'text': text}, 'expect': [{'kind': 'value', 'node_id': 'text', 'value': text}]}


def workflow(rapid, prefix, text='Z', cancel=None):
    return rapid.run_workflow([prefix, write(text)],
        final_predicates=[{'kind': 'value', 'node_id': 'text', 'value': text}], cancel=cancel)


def test_saved042_shape_cached_selection_cannot_permit_changed_following_write():
    adapter = TextAdapter()
    rapid = controller(adapter)
    first = rapid.run_workflow([select(), write('ABC\nDEF')],
        final_predicates=[{'kind': 'value', 'node_id': 'text', 'value': 'ABC\nDEF'}])
    assert first['status'] == 'needs_replan' and adapter.value == 'ABCDEF'
    assert adapter.selection is None
    historical = copy.deepcopy(first['outcomes'][0])
    calls = len(adapter.inputs)
    result = workflow(rapid, select(), 'NEW\nTEXT')
    assert len(adapter.inputs) == calls  # Never append to the unselected current value.
    assert result['status'] == 'needs_replan' and adapter.value == 'ABCDEF'
    assert result['outcomes'][0]['status'] == 'unresolved'
    assert result['outcomes'][0]['current_prefix_verification']['status'] == 'unresolved'
    assert rapid.execute(historical['action']['id']) == historical


PREDICATES = [
    ('focused', {'kind': 'focused', 'node_id': 'text'}),
    ('value', {'kind': 'value', 'node_id': 'text', 'value': 'ABCDEF'}),
    ('checked', {'kind': 'checked', 'node_id': 'text', 'checked': True}),
    ('selected', {'kind': 'selected', 'node_id': 'text', 'values': ['one']}),
    ('selection', {'kind': 'text_selection', 'node_id': 'text', 'start': 0, 'end': 6}),
    ('active', {'kind': 'active_window', 'node_id': 'window'}),
    ('visible', {'kind': 'visible', 'role': 'button', 'name': 'marker', 'visible': True}),
    ('text', {'kind': 'text', 'text': 'marker'}),
    ('url', {'kind': 'url', 'url': 'https://example.com/source'}),
]


@pytest.mark.parametrize(('kind', 'predicate'), PREDICATES)
def test_each_cached_predicate_requires_fresh_positive_evidence(kind, predicate):
    adapter = TextAdapter()
    if kind == 'selection':
        adapter.selection = {'start': 0, 'end': 6}
    rapid = controller(adapter)
    step = focus([predicate])
    original = rapid.run_step(step)
    assert original['status'] == 'verified'
    trigger = adapter.reads + 2  # After run_step's fresh enumeration, during cached proof.

    def change(read):
        if read != trigger:
            return
        if kind == 'focused':
            adapter.focused = False
        elif kind == 'value':
            adapter.value = 'different'
        elif kind == 'checked':
            adapter.checked = False
        elif kind == 'selected':
            adapter.selected = []
        elif kind == 'selection':
            adapter.selection = None
        elif kind == 'active':
            adapter.scope = 'other-window'
        elif kind in {'visible', 'text'}:
            adapter.marker = False
        else:
            adapter.url = 'https://example.com/other'

    adapter.on_read = change
    calls = len(adapter.inputs)
    result = workflow(rapid, step)
    assert result['status'] == 'needs_replan' and len(adapter.inputs) == calls
    assert rapid.execute(original['action']['id']) == original


@pytest.mark.parametrize('change', ['session', 'scope', 'generation', 'missing', 'sensitive', 'read_failure', 'cancel'])
def test_late_identity_unknown_and_cancellation_block_cached_prefix(change):
    adapter = TextAdapter()
    rapid = controller(adapter)
    original = rapid.run_step(focus())
    trigger = adapter.reads + 2
    cancel = threading.Event()

    def late(read):
        if read != trigger:
            return
        if change in {'session', 'scope', 'generation'}:
            setattr(adapter, change, 'changed')
        elif change in {'missing', 'sensitive'}:
            setattr(adapter, change, True)
        elif change == 'read_failure':
            raise ComputerUseError('bounded_read_failure_secret_canary')
        else:
            cancel.set()

    adapter.on_read = late
    calls = len(adapter.inputs)
    result = workflow(rapid, focus(), cancel=cancel)
    assert result['status'] == ('cancelled' if change == 'cancel' else 'needs_replan')
    assert len(adapter.inputs) == calls
    assert 'secret_canary' not in str(result)
    assert rapid.execute(original['action']['id']) == original


def test_current_cached_predicate_allows_the_real_following_step_without_redispatch():
    adapter = TextAdapter()
    rapid = controller(adapter)
    original = rapid.run_step(focus())
    following = write('Z')
    following['expect'][0]['value'] = 'ABCDEFZ'
    result = rapid.run_workflow([focus(), following],
        final_predicates=[{'kind': 'value', 'node_id': 'text', 'value': 'ABCDEFZ'}])
    assert result['status'] == 'complete' and adapter.value == 'ABCDEFZ'
    assert [a['verb'] for a in adapter.inputs] == ['focus', 'type_text']
    assert result['outcomes'][0]['current_prefix_verification']['status'] == 'verified'
    assert rapid.execute(original['action']['id']) == original


def test_fresh_prefix_uses_existing_verification_without_cached_extra_read():
    adapter = TextAdapter()
    rapid = controller(adapter)
    result = workflow(rapid, select(), 'Z')
    assert result['status'] == 'complete'
    assert [a['verb'] for a in adapter.inputs] == ['key_chord', 'type_text']
    assert 'current_prefix_verification' not in result['outcomes'][0]


def test_final_cached_step_uses_the_existing_independent_terminal_observation():
    adapter = TextAdapter()
    rapid = controller(adapter)
    original = rapid.run_step(focus())
    adapter.focused = False
    result = rapid.run_workflow([focus()], final_predicates=[{'kind': 'focused', 'node_id': 'text'}])
    assert result['status'] == 'needs_replan' and len(adapter.inputs) == 1
    assert result['outcomes'][0] == original


def test_candidate_identity_distinguishes_fresh_execution_from_historical_cache():
    rapid = controller(TextAdapter())
    first = rapid.run_step(focus())
    assert rapid.candidate(first['action']['id']) == first['action']
    repeated = rapid.run_step(focus())
    assert repeated == first and rapid.candidate(repeated['action']['id']) is None
    assert len(rapid.adapter.inputs) == 1 and rapid._execution_epoch == 0


def test_replaced_candidate_set_conservatively_revalidates_without_redispatch():
    adapter = TextAdapter()
    rapid = controller(adapter)
    original = rapid.run_step

    def interleaved(step, provider=None, cancel=None):
        result = original(step, provider, cancel)
        if step['verb'] == 'key_chord':
            rapid.enumerate(focus(), rapid.observe())  # Legal intervening enumeration.
        return result

    rapid.run_step = interleaved
    result = workflow(rapid, select(), 'Z')
    assert result['status'] == 'complete' and adapter.value == 'Z'
    assert [a['verb'] for a in adapter.inputs] == ['key_chord', 'type_text']
    assert result['outcomes'][0]['current_prefix_verification']['status'] == 'verified'


def test_metadata_clarification_does_not_invalidate_true_cached_predicate_or_grant_progress():
    adapter = TextAdapter()
    rapid = controller(adapter)
    rapid.run_step(focus())
    adapter.metadata = {'character_count': 6, 'selection_count': 1,
                        'captured_range': {'start': 0, 'end': 6}, 'truncated': False}
    following = write('Z')
    following['expect'][0]['value'] = 'ABCDEFZ'
    result = rapid.run_workflow([focus(), following],
        final_predicates=[{'kind': 'value', 'node_id': 'text', 'value': 'ABCDEFZ'}])
    assert result['status'] == 'complete'
    proof = result['outcomes'][0]['current_prefix_verification']
    assert proof['task_progress'] is False and proof['input_attempted'] is False
    assert [a['verb'] for a in adapter.inputs] == ['focus', 'type_text']


def test_truncated_view_cannot_prove_cached_negative_visibility():
    adapter = TextAdapter()
    adapter.marker = False
    rapid = controller(adapter)
    step = focus([{'kind': 'visible', 'role': 'button', 'name': 'marker', 'visible': False}])
    assert rapid.run_step(step)['status'] == 'verified'
    trigger = adapter.reads + 2
    adapter.on_read = lambda read: setattr(adapter, 'truncated', True) if read == trigger else None
    calls = len(adapter.inputs)
    result = workflow(rapid, step)
    assert result['status'] == 'needs_replan' and len(adapter.inputs) == calls
