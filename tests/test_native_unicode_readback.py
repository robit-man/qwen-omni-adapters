"""Recipient acknowledgement cannot confuse a selected old value with delivery."""
from types import SimpleNamespace

import pytest

from portal.native_ui import NativeUiUnavailable, _AtspiBridge


def bridge_and_node(value, caret, selection=None):
    state = {'value': value, 'caret': caret, 'selection': selection}
    text = SimpleNamespace(
        get_character_count=lambda _: len(state['value']),
        get_caret_offset=lambda _: state['caret'],
        get_n_selections=lambda _: int(state['selection'] is not None),
        get_selection=lambda _, index: SimpleNamespace(start_offset=state['selection'][0], end_offset=state['selection'][1]),
        get_text=lambda _, first, last: state['value'][first:last])
    bridge = object.__new__(_AtspiBridge)
    bridge.atspi = SimpleNamespace(Text=text)
    bridge._read = lambda function: function()
    bridge._clear_cache = lambda _: None
    bridge.foreground_identity = {'window_id': '123', 'process_id': 456}
    bridge.foreground = lambda: bridge.foreground_identity
    return bridge, SimpleNamespace(get_text_iface=lambda: object()), state


def test_identical_selected_value_does_not_acknowledge_queued_unicode_input():
    bridge, node, state = bridge_and_node('é', 1, (0, 1))
    acknowledged = bridge._unicode_delivery_probe(node, 'é')
    assert not acknowledged()
    state['selection'] = None
    assert acknowledged()


@pytest.mark.parametrize('fault', ['partial_text', 'wrong_text', 'wrong_caret', 'focus_changed'])
def test_unicode_readback_requires_complete_inserted_span_and_bound_window(fault):
    bridge, node, state = bridge_and_node('prefix suffix', 7)
    acknowledged = bridge._unicode_delivery_probe(node, 'éï')
    state.update(value='prefix éïsuffix', caret=9)
    assert acknowledged()
    if fault == 'partial_text':
        state.update(value='prefix ésuffix', caret=8)
    elif fault == 'wrong_text':
        state['value'] = 'prefix eïsuffix'
    elif fault == 'wrong_caret':
        state['caret'] = 8
    else:
        bridge.foreground = lambda: {'window_id': '789', 'process_id': 456}
    assert not acknowledged()


def test_unavailable_insertion_range_cannot_create_a_delivery_probe():
    bridge, node, _ = bridge_and_node('', -1)
    with pytest.raises(NativeUiUnavailable, match='native_unicode_selection_unavailable'):
        bridge._unicode_delivery_probe(node, 'é')


def test_unicode_acknowledgement_tracks_prefix_before_allowing_next_segment():
    bridge, node, state = bridge_and_node('old', 3, (0, 3))
    acknowledge = bridge._unicode_delivery_probe(node, 'aéz')
    state.update(value='a', caret=1, selection=None)
    assert acknowledge('a')
    assert not acknowledge('aé')
    state.update(value='aé', caret=2)
    assert acknowledge('aé')
    assert not acknowledge()
    with pytest.raises(NativeUiUnavailable, match='prefix_invalid'):
        acknowledge('wrong')


@pytest.mark.parametrize('fault', [None, 'no_ack', 'command_failure', 'focus_lost', 'map_unavailable'])
def test_native_unicode_segments_stop_after_uncertain_input_and_restore_mapping(monkeypatch, fault):
    from test_native_typing import typing_fixture

    fixture, _, _ = typing_fixture(monkeypatch)
    trace = []

    class Mapping:
        codes = {'é': 8}

        def __init__(self, *_):
            pass

        def __enter__(self):
            if fault == 'map_unavailable':
                raise RuntimeError('no unused mapping')
            trace.append('map')
            return self

        def __exit__(self, *_):
            trace.append('restore')

    def run(argv, **kwargs):
        assert 0 < kwargs['timeout'] <= 5 and 'shell' not in kwargs
        trace.append(('input', argv[-1]))
        if fault == 'focus_lost':
            fixture.title.state.values.remove('FOCUSED')
        return SimpleNamespace(returncode=int(fault == 'command_failure'))

    def acknowledged(prefix):
        trace.append(('ack', prefix))
        return fault != 'no_ack'

    monkeypatch.setattr('portal.native_ui._keymap_helper', lambda: SimpleNamespace(UnicodeKeymap=Mapping))
    monkeypatch.setattr('portal.native_ui.subprocess.run', run)
    monkeypatch.setattr(_AtspiBridge, '_unicode_delivery_probe', lambda *_: acknowledged)
    result = fixture.adapter.perform(fixture.action('Title', 'type_text', {'text': 'aéz'}))
    inputs = [row for row in trace if isinstance(row, tuple) and row[0] == 'input']
    if fault == 'map_unavailable':
        assert result['status'] == 'rejected' and result['input_attempted'] is False
        assert not trace
    else:
        assert trace[0] == 'map' and trace[-1] == 'restore'
        assert result['status'] == 'dispatched'
        if fault:
            assert inputs == [('input', 'a')]
            assert result['reason'] == 'native_text_input_effect_unresolved'
        else:
            assert trace == ['map', ('input', 'a'), ('ack', 'a'), ('input', '008'),
                             ('ack', 'aé'), ('input', 'z'), ('ack', 'aéz'), 'restore']
            assert result['reason'] == 'native_text_input_invoked'
