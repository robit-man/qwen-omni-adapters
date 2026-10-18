"""Observed tree context survives native flattening without input authority."""
import copy

import pytest
from test_native_ui import Fixture, Node, _named_node

from portal import ui_decision
from portal.app import _model_ui_observation
from portal.ui_task import _grounded_phase_schema, _NativeWorkingState, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError, _snapshot


def test_native_rows_keep_observed_parent_and_sibling_relationships():
    fixture = Fixture()
    first = Node('/row-one', 'list item', children=[Node('/label-one', 'label', 'First document')])
    second = Node('/row-two', 'list item', children=[Node('/label-two', 'label', 'Second document')])
    popup = Node('/popup', 'panel', 'Documents', children=[first, second])
    fixture.window.children.append(popup)

    state = fixture.adapter.observe()

    first_label = _named_node(state, 'First document')
    second_label = _named_node(state, 'Second document')
    first_row = next(n for n in state['nodes'] if n['id'] == first_label['structure']['parent_id'])
    second_row = next(n for n in state['nodes'] if n['id'] == second_label['structure']['parent_id'])
    assert first_row['structure']['parent_id'] == second_row['structure']['parent_id']
    assert (first_row['structure']['child_index'], second_row['structure']['child_index']) == (0, 1)
    assert first_label['structure']['ancestors'][-1]['id'] == first_row['id']
    assert first_label['structure']['ancestors'][-2]['name'] == 'Documents'
    assert len(first_label['structure']['ancestors']) <= 3
    assert first_label['operations'] == []
    assert first_label['id'] not in {first_row['id'], second_row['id']}


def test_native_structure_uses_existing_reads_and_protects_sensitive_ancestor_name():
    fixture = Fixture()
    protected = Node('/protected-parent', 'password text', 'private ancestor name',
                     children=[Node('/protected-child', 'label', 'Child')])
    fixture.window.children.append(protected)
    state = fixture.adapter.observe()
    child = _named_node(state, 'Child')
    assert child['structure']['ancestors'][-1]['name'] == '[protected]'
    assert 'private ancestor name' not in str(state)
    assert 'native_structure' not in str(child['operations'])


def test_actual_portal_projection_preserves_hierarchy_and_exact_target_binding_into_compact_planner():
    raw = _snapshot(Fixture().adapter.observe())
    projected = _model_ui_observation(raw)
    raw_target = _named_node(raw, 'Title')
    target = _named_node(projected, 'Title')
    assert target['structure'] == raw_target['structure']
    assert ui_decision._target_binding(target) == ui_decision._target_binding(raw_target)
    working = _NativeWorkingState('Edit the observed title field')
    observed = {'state': projected}
    schema = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, observed, intent_choice=True, effect_contracts=True),
        working.view(1, 0), observed_state=projected, action_handoffs=True)
    packet = {'goal': working.goal, 'current_state': observed, 'working_state': working.view(1, 0),
        'response_schema': schema, 'ledger': [], 'last_result': None}
    compact, binding = ui_decision.project(packet)
    control = next(c for c in compact['controls'] if c.get('name') == 'Title')
    assert control['structure']['ancestors'] == raw_target['structure']['ancestors']
    assert ui_decision._target_binding(binding['nodes'][control['ref']]) == ui_decision._target_binding(raw_target)


@pytest.mark.parametrize('fault', ['too_many', 'cycle', 'parent_mismatch', 'negative_index', 'bad_state'])
def test_observed_structure_rejects_invalid_relationships(fault):
    state = Fixture().adapter.observe()
    node = _named_node(state, 'Title')
    structure = node['structure']
    if fault == 'too_many':
        structure['ancestors'] *= 4
    elif fault == 'cycle':
        structure['ancestors'][-1]['id'] = node['id']
        structure['parent_id'] = node['id']
    elif fault == 'parent_mismatch':
        structure['parent_id'] = 'other'
    elif fault == 'negative_index':
        structure['child_index'] = -1
    else:
        structure['ancestors'][-1]['visible'] = 'yes'
    with pytest.raises(ComputerUseError):
        _snapshot(copy.deepcopy(state))
