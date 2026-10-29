"""Dense observed rows retain usable bindings within the existing UI budget."""
import copy
import json

import pytest

from portal.app import _model_ui_observation


def dense_rows(names):
    scope = 'active-window'
    state = {'session_id': 'observed-session', 'revision': 'observed-revision',
        'scope': {'kind': 'desktop', 'id': scope}, 'truncated': False, 'nodes': []}
    for index, name in enumerate(('File manager', 'Editor', 'Browser')):
        state['nodes'].append({'id': scope if index == 0 else f'window-{index}',
            'generation': '1', 'scope_id': scope if index == 0 else f'window-{index}',
            'role': 'window', 'name': name, 'visible': True, 'enabled': True,
            'sensitive': False, 'operations': ['key_chord'] if index == 0 else ['switch_window']})
    for row_index, name in enumerate(names):
        parent = f'row-{row_index}-' + 'r' * 30
        ancestors = [{'id': f'{role}-' + 'a' * 32, 'generation': '1', 'role': role,
            'name': 'Observed accessible container', 'visible': True, 'enabled': True}
            for role in ('tree_table', 'list')]
        ancestors.append({'id': parent, 'generation': '1', 'role': 'table_row',
            'name': name, 'visible': True, 'enabled': True})
        for column, value in enumerate((name, '510 bytes', 'Today 03:08')):
            state['nodes'].append({'id': f'row-{row_index}-cell-{column}', 'generation': '1',
                'scope_id': scope, 'role': 'table_cell', 'name': value, 'visible': True,
                'enabled': True, 'sensitive': False, 'focused': False,
                'operations': ['click', 'double_click', 'right_click', 'scroll'],
                'bounds': {'x': 120 + column * 200, 'y': 80 + row_index * 40, 'width': 190, 'height': 40},
                'structure': {'ancestors': copy.deepcopy(ancestors), 'parent_id': parent,
                    'child_index': column}})
    return state


@pytest.mark.parametrize('names', [
    ['corrections.json', 'draft-1.txt', 'draft-2.txt', 'draft-3.txt', 'task-brief.txt'],
    ['Delta', 'Alpha', 'Beta', 'Gamma', 'Epsilon'],
    ['café', '中', 'café', 'Δοκιμή', 'مرحبا'],
])
def test_dense_rows_keep_each_observed_primary_binding_without_spending_more_budget(names):
    state = dense_rows(names)
    original = copy.deepcopy(state)
    references = ('active-window', 'window-1', 'window-2')
    projected = _model_ui_observation(state, references)
    by_id = {node['id']: node for node in projected['nodes']}
    assert all(f'row-{row}-cell-0' in by_id for row in range(len(names)))
    assert set(references) <= by_id.keys()
    assert sum(len(json.dumps(node, ensure_ascii=False)) for node in projected['nodes']) <= 12000
    assert len(projected['nodes']) <= 80
    assert projected['model_projection']['omitted_visible_nodes'] > 0
    assert projected['model_projection']['absence_claims_supported'] is False
    for node in state['nodes']:
        if node['id'] in by_id:
            for key in ('id', 'generation', 'scope_id', 'operations', 'bounds', 'structure'):
                if key in node:
                    assert by_id[node['id']][key] == node[key]
    assert state == original


def test_receipt_references_and_keyboard_focus_keep_precedence_over_row_diversity():
    state = dense_rows([f'Entry {n}' for n in range(5)])
    last = state['nodes'][-1]
    last['focused'] = True
    referenced = state['nodes'][-2]
    projected = _model_ui_observation(state, (referenced['id'],))
    ids = [node['id'] for node in projected['nodes']]
    assert ids[0] == referenced['id']
    assert ids.index('active-window') < ids.index(last['id'])
    assert ids.index(last['id']) < ids.index('row-0-cell-0')


@pytest.mark.parametrize('fault', ['no_structure', 'unbound_parent', 'non_row_parent', 'boolean_index'])
def test_row_priority_requires_actual_well_formed_observed_row_relationship(fault):
    state = dense_rows([f'Entry {n}' for n in range(5)])
    for node in state['nodes']:
        if node.get('role') != 'table_cell':
            continue
        if fault == 'no_structure':
            node.pop('structure')
        elif fault == 'unbound_parent':
            node['structure']['parent_id'] = 'unobserved-parent'
        elif fault == 'non_row_parent':
            node['structure']['ancestors'][-1]['role'] = 'panel'
        else:
            node['structure']['child_index'] = False
    projected = _model_ui_observation(state)
    retained = [node['id'] for node in projected['nodes']]
    original_order = [node['id'] for node in state['nodes'] if node.get('role') == 'table_cell']
    actual_order = [node_id for node_id in retained if node_id in original_order]
    assert actual_order == [node_id for node_id in original_order if node_id in retained]


def test_sensitive_or_disabled_cells_cannot_become_actionable_row_representatives():
    state = dense_rows([f'Entry {n}' for n in range(5)])
    protected, disabled = state['nodes'][3], state['nodes'][6]
    protected.update(sensitive=True, name='private-original', value='private-text')
    disabled['enabled'] = False
    projected = _model_ui_observation(state)
    cells = [node for node in projected['nodes'] if node['role'] == 'table_cell']
    assert cells[0]['id'] == 'row-0-cell-1'
    protected_projection = next((node for node in projected['nodes'] if node['id'] == protected['id']), None)
    if protected_projection:
        assert protected_projection['name'] == '[protected]'
        assert protected_projection['operations'] == []
        assert 'value' not in protected_projection and 'structure' not in protected_projection
    disabled_projection = next((node for node in projected['nodes'] if node['id'] == disabled['id']), None)
    if disabled_projection:
        assert disabled_projection['enabled'] is False
