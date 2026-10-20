"""Compact planning without a delegated selector can still activate an exact observed control."""
import pytest
from test_native_compact_decision import reply
from test_ui_observation_boundary import CELL, Adapter, node

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema

GOAL = 'Read the exact source and save a report; preserve inputs.'


def packet(*, intent_choice=False, effect_contracts=True):
    state = Adapter().observe()
    state['nodes'][0]['operations'] += ['key_chord', 'scroll']
    state['nodes'] += [node('scope', 'window', 'Files', ['key_chord', 'switch_window', 'double_click']),
                       node('other', 'window', 'Editor', ['switch_window'])]
    working = _NativeWorkingState(GOAL)
    view = working.view(1, 0)
    observation = {'state': state}
    phases = _grounded_phase_schema(['native_ui'], observation, intent_choice=intent_choice,
                                    effect_contracts=effect_contracts)
    return {'goal': GOAL, 'current_state': observation, 'ledger': [], 'last_result': None,
            'working_state': view, 'remaining_seconds': 100,
            'response_schema': _working_phase_schema(['native_ui'], phases, view, observed_state=state,
                                                     action_handoffs=effect_contracts)}


def families(compact):
    return [(branch['properties']['family']['const'], branch['properties'].get('verb', {}).get('const'),
             branch['properties'].get('target', {}).get('enum'))
            for branch in compact['decision_schema']['properties']['action']['oneOf']]


def ref(binding, identity):
    return next(key for key, value in binding['nodes'].items() if value['id'] == identity)


def test_exact_control_activation_is_offered_and_expands_to_one_guarded_boundary():
    p = packet()  # The portal always plans with effect contracts (portal/app.py).
    compact, binding = ui_decision.project(p)
    cell = ref(binding, CELL)
    offered = {(verb, target) for family, verb, targets in families(compact) if family == 'activate'
               for target in targets}
    operations = set(p['current_state']['state']['nodes'][0]['operations'])
    assert {verb for verb, target in offered if target == cell} == operations & {'invoke', 'double_click', 'click',
                                                                                   'right_click'}
    assert all(binding['nodes'][target]['id'] != p['current_state']['state']['scope']['id']
               for _verb, target in offered)  # The active scope itself is never an activation target.
    verb = next(verb for verb, target in sorted(offered) if target == cell)
    envelope = ui_decision.expand(reply(p, {'family': 'activate', 'verb': verb, 'target': cell}), p, binding)
    assert envelope['phase']['kind'] == 'ui_boundary'
    assert envelope['phase']['step'] == {'id': 'phase-1-activate', 'verb': verb, 'target': {'node_id': CELL},
                                         'args': {}}


def test_delegated_selector_interface_is_unchanged():
    compact, _binding = ui_decision.project(packet(intent_choice=True))
    assert not any(family == 'activate' for family, _verb, _targets in families(compact))
    assert any(family == 'pointer_or_activation' for family, _verb, _targets in families(compact))


@pytest.mark.parametrize('action', [
    {'family': 'activate', 'verb': 'double_click', 'target': 'c999'},
    {'family': 'activate', 'verb': 'type_text', 'target': 'c0'},
    {'family': 'activate', 'verb': 'invoke', 'node_id': CELL},
])
def test_unoffered_targets_verbs_or_raw_identities_are_rejected(action):
    p = packet()
    compact, binding = ui_decision.project(p)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand(reply(p, action), p, binding)


@pytest.mark.parametrize('family', ['scroll', 'switch_window', 'key_chord'])
def test_existing_boundary_families_also_expand_without_a_selector(family):
    p = packet()
    compact, binding = ui_decision.project(p)
    target = next(targets[0] for name, _verb, targets in families(compact) if name == family)
    extra = {'scroll': {'direction': 'down'}, 'key_chord': {'keys': ['Escape']}}.get(family, {})
    envelope = ui_decision.expand(reply(p, {'family': family, 'target': target, **extra}), p, binding)
    assert envelope['phase']['kind'] in {'ui_boundary', 'ui'}
