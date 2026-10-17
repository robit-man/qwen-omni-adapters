"""A popup grab is a prerequisite, not a successful application switch."""
from types import SimpleNamespace

import pytest
from test_native_ui import STATES, Fixture, Node

from portal.native_ui import NativeUiUnavailable


def popup_fixture():
    f = Fixture()
    f.api.StateType = SimpleNamespace(**vars(STATES), HAS_POPUP='HAS_POPUP')
    f.api.RelationType = SimpleNamespace(CONTROLLER_FOR='controller_for')
    f.button.state.values.add('HAS_POPUP')
    popup = Node('/popup', 'panel')
    wrapper = Node('/popup-wrapper', 'panel', children=[popup])
    f.button.children.append(wrapper)
    popup.get_parent = lambda: wrapper
    wrapper.get_parent = lambda: f.button
    f.button.get_parent = lambda: f.window
    for node, parent in [(popup, wrapper), (wrapper, f.button), (f.button, f.window)]:
        node.get_index_in_parent = lambda n=node, p=parent: p.children.index(n) if n in p.children else -1
    relation = SimpleNamespace(get_relation_type=lambda: 'controller_for',
        get_n_targets=lambda: 1, get_target=lambda _i: popup)
    f.button.get_relation_set = lambda: [relation]
    return f, popup, wrapper, relation


def test_visible_native_popup_withholds_switch_but_retains_keyboard_and_popup_controls():
    f, popup, _, _ = popup_fixture()
    before = f.adapter.observe()
    assert len(before['open_popups']) == 1
    root = next(n for n in before['nodes'] if n['id'] == before['scope']['id'])
    assert 'key_chord' in root['operations']
    assert all('switch_window' not in n['operations'] for n in before['nodes'])
    assert any(n['role'] == 'panel' and n['visible'] for n in before['nodes'])
    popup.state.values.remove('SHOWING')
    after = f.adapter.observe()
    assert 'open_popups' not in after and after['revision'] != before['revision']
    assert 'switch_window' in next(n for n in after['nodes'] if n['id'] == after['scope']['id'])['operations']
    assert f.button.events == []


def test_popup_opened_after_offer_rejects_switch_without_input(monkeypatch):
    f, popup, _, _ = popup_fixture()
    popup.state.values.remove('SHOWING')
    action = f.action('Fixture', 'switch_window')
    popup.state.values.add('SHOWING')
    monkeypatch.setattr('portal.native_ui.subprocess.run', lambda *_a, **_kw: pytest.fail('No input or X11 activation'))
    result = f.adapter.perform(action)
    assert result == {'status': 'rejected', 'reason': 'native_popup_requires_dismissal', 'input_attempted': False}


@pytest.mark.parametrize('condition', ['hidden-parent', 'hidden-popup', 'foreign-bus', 'foreign-frame', 'cycle', 'no-popup-flag', 'non-popup-relation', 'retained-hidden-popup'])
def test_unrelated_or_hidden_relation_never_claims_open_popup(condition):
    f, popup, wrapper, relation = popup_fixture()
    if condition == 'hidden-parent':
        wrapper.state.values.remove('SHOWING')
    elif condition == 'hidden-popup':
        popup.state.values.remove('VISIBLE')
    elif condition == 'foreign-bus':
        popup.app = SimpleNamespace(bus_name=':99.1')
    elif condition == 'foreign-frame':
        foreign = Node('/foreign', 'frame')
        foreign.get_parent = lambda: None
        wrapper.get_parent = lambda: foreign
    elif condition == 'cycle':
        wrapper.get_parent = lambda: popup
    elif condition == 'no-popup-flag':
        f.button.state.values.remove('HAS_POPUP')
    elif condition == 'retained-hidden-popup':
        f.button.children.remove(wrapper)
    else:
        relation.get_relation_type = lambda: 'label_for'
    assert 'open_popups' not in f.adapter.observe()


def test_relation_failure_retains_previous_observation_and_does_not_authorize_switch():
    f, _, _, _ = popup_fixture()
    f.adapter.observe()
    previous, handles = set(f.bridge.previous), dict(f.bridge.handles)
    def unavailable():
        raise RuntimeError('provider unavailable')
    f.button.get_relation_set = unavailable
    with pytest.raises(NativeUiUnavailable, match='native_observation_unavailable'):
        f.adapter.observe()
    assert f.bridge.previous == previous and f.bridge.handles == handles


def test_popup_relation_budget_is_bounded():
    f, _, _, relation = popup_fixture()
    relation.get_n_targets = lambda: 17
    with pytest.raises(NativeUiUnavailable, match='native_popup_relations_unavailable'):
        f.adapter.observe()


def test_popup_evidence_survives_compaction_and_escape_is_canonical():
    from test_native_compact_decision import packet, reply

    from portal import ui_decision
    p = packet()
    state = p['current_state']['state']
    state['open_popups'] = [{'controller_id': 'scope', 'authority': 'native_popup_relation'}]
    for n in state['nodes']:
        n['operations'] = [v for v in n['operations'] if v != 'switch_window']
    compact, binding = ui_decision.project(p)
    assert compact['observation']['open_popups'] == state['open_popups']
    assert all('switch_window' not in c['families'] for c in compact['controls'])
    root = next(c['ref'] for c in compact['controls'] if c['active_window'])
    expanded = ui_decision.expand(reply(p, {'family': 'key_chord', 'target': root, 'keys': ['Escape']}), p, binding)
    assert expanded['phase']['kind'] == 'ui_boundary'
    assert expanded['phase']['step']['args'] == {'keys': ['Escape']}


def test_popup_appearing_after_fresh_snapshot_still_blocks_final_activation(monkeypatch):
    from test_native_ui import _switch_fixture
    f, _, _, _, activations, _, action = _switch_fixture(monkeypatch)
    reads = []
    def appeared(*_args):
        reads.append(True)
        return [] if len(reads) == 1 else [{'controller_id': 'new', 'authority': 'native_popup_relation'}]
    f.bridge._open_popups = appeared
    result = f.adapter.perform(action)
    assert result['reason'] == 'native_popup_requires_dismissal'
    assert result['input_attempted'] is False
    assert len(reads) == 2 and not activations
