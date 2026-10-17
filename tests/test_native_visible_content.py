"""Current native document text survives traversal and planner compression."""
import copy
import json

import pytest
from test_native_ui import Fixture, Node

from portal import ui_decision
from portal.app import _model_ui_observation
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema


def test_offscreen_subtree_does_not_exhaust_visible_document_budget():
    fixture = Fixture()
    hidden = Node('/offscreen', 'section', children=[
        Node(f'/old-{i}', 'paragraph', f'Offscreen paragraph {i}') for i in range(500)], states=[])
    visible = Node('/current', 'paragraph', 'The current document condition and its exact error.')
    fixture.window.children = [hidden, visible]
    fixture.bridge.MAX_NODES = 10
    state = fixture.bridge.observe()
    assert any(n['name'] == visible.name and n['visible'] for n in state['nodes'])
    assert not any(n['name'].startswith('Offscreen paragraph') for n in state['nodes'])
    assert state['truncated'] is True
    assert ':1.42:/offscreen' in fixture.bridge.previous
    assert len(state['nodes']) < fixture.bridge.MAX_NODES


def test_omitted_subtree_retains_identity_but_never_previous_input_handles():
    fixture = Fixture()
    child = Node('/child', 'push button', 'Nested control', interfaces=['Action'])
    parent = Node('/container', 'section', children=[child])
    fixture.window.children = [parent]
    first = next(n for n in fixture.bridge.observe()['nodes'] if n['name'] == child.name)
    parent.state.values.remove('SHOWING')
    hidden = fixture.bridge.observe()
    assert first['id'] not in fixture.bridge.handles
    assert not any(n['id'] == first['id'] for n in hidden['nodes'])
    assert ':1.42:/child' in fixture.bridge.previous
    parent.state.values.add('SHOWING')
    restored = next(n for n in fixture.bridge.observe()['nodes'] if n['name'] == child.name)
    assert (restored['id'], restored['generation']) == (first['id'], first['generation'])
    parent.children.clear()
    fixture.bridge.observe()  # A complete list can now prove disappearance.
    parent.children.append(child)
    renewed = next(n for n in fixture.bridge.observe()['nodes'] if n['name'] == child.name)
    assert renewed['id'] != first['id'] and renewed['generation'] == '2'


def crowded():
    fixture = Fixture()
    fixture.window.children = [Node(f'/control-{i}', 'push button', f'Control {i}', interfaces=['Action'])
                              for i in range(70)]
    fixture.window.children += [Node('/signature', 'description term', 'Path.mkdir(mode=0o777, parents=False, exist_ok=False)'),
        Node('/paragraph', 'paragraph', 'If parents is false, a missing parent raises FileNotFoundError.')]
    return fixture.bridge.observe()


def test_reading_is_separate_from_control_budget_and_source_bound():
    state = crowded()
    original = copy.deepcopy(state)
    projection = _model_ui_observation(state)
    assert not any(n['role'] == 'paragraph' for n in projection['nodes'])
    assert projection['nodes'][0]['id'] == state['scope']['id']
    excerpt = projection['native_content_excerpt']
    assert 'FileNotFoundError' in excerpt['text']
    assert excerpt['source_revision'] == state['revision']
    assert excerpt['coverage'] == 'partial'
    assert excerpt['authority'] == 'untrusted_native_accessibility'
    assert excerpt['input_authority'] is False and excerpt['pixel_visibility_verified'] is False
    assert state == original
    assert sum(len(json.dumps(n, ensure_ascii=False)) for n in projection['nodes']) <= 12000


@pytest.mark.parametrize('exclusion', ['hidden', 'protected', 'other_window', 'non_content'])
def test_content_excerpt_never_expands_authority_or_exposes_excluded_text(exclusion):
    state = crowded()
    node = state['nodes'][-1]
    node['name'] = 'EXCLUDED-CONTENT'
    if exclusion == 'hidden':
        node['visible'] = False
    elif exclusion == 'protected':
        node['sensitive'] = True
    elif exclusion == 'other_window':
        node['scope_id'] = 'different-window'
    else:
        node['role'] = 'button'
    projection = _model_ui_observation(state)
    assert 'EXCLUDED-CONTENT' not in projection['native_content_excerpt']['text']


def test_excerpt_is_bounded_and_rebuilt_from_fresh_nodes():
    state = crowded()
    state['nodes'][-1]['name'] = 'long source ' * 1000
    first = _model_ui_observation(state)
    assert len(first['native_content_excerpt']['text']) == 4000
    assert first['native_content_excerpt']['text_truncated'] is True
    first['nodes'] = [n for n in first['nodes'] if n['role'] not in {'paragraph', 'description_term'}]
    second = _model_ui_observation(first)
    assert 'native_content_excerpt' not in second  # No replay as a new observation.


def test_compact_planner_retains_native_excerpt_without_new_action_authority():
    observation = {'state': _model_ui_observation(crowded())}
    working = _NativeWorkingState('Read current documentation.').view(1, 0)
    canonical = _working_phase_schema(['native_ui'], _grounded_phase_schema(['native_ui'], observation, intent_choice=True), working)
    p = {'goal': 'Read current documentation.', 'current_state': observation, 'working_state': working,
         'ledger': [], 'last_result': None, 'response_schema': canonical}
    compact, _ = ui_decision.project(p)
    assert compact['observation']['native_content_excerpt'] == observation['state']['native_content_excerpt']
    assert 'FileNotFoundError' in compact['observation']['native_content_excerpt']['text']
