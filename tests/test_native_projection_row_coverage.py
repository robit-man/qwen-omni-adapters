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


@pytest.mark.parametrize('change', [None, 'bank_metadata', 'semantic_revision'])
def test_pending_bank_fact_uses_complete_raw_state_without_changing_projection_priority(change):
    from test_native_pending_control_projection import actual_deferral
    from test_portal_native_task_routes import GOAL
    from test_ui_observation_boundary import CELL

    from portal import ui_decision
    adapter, result, handoff = actual_deferral()
    pending = ui_decision.retain_pointer_recovery(result, goal=GOAL, action_handoff=handoff)
    raw = adapter.observe()
    if change == 'bank_metadata':
        raw['nodes'][0]['description'] = 'New native fact changes the complete bank'
    elif change == 'semantic_revision':
        raw['revision'] = 'New actual semantic snapshot'
    priority, receipt = pending.consume(goal=GOAL, state=raw)
    assert priority == (CELL,)
    # A whole-snapshot revision is not a change to the exact target/raw bank.
    assert ('pending_delegation' in receipt) is (change != 'bank_metadata')
    projected = _model_ui_observation(raw, priority)
    assert CELL in {node['id'] for node in projected['nodes']}
    assert sum(len(json.dumps(node, ensure_ascii=False)) for node in projected['nodes']) <= 12000
    assert len(projected['nodes']) <= 80
    assert pending.consume(goal=GOAL, state=raw) == ((), None)
    assert not adapter.calls


@pytest.mark.parametrize('route', ['named_boundary', 'generated_suffix'])
def test_actual_direct_callback_refuses_pending_alias_or_suffix_before_any_new_policy_or_input(
        tmp_path, monkeypatch, route):
    import httpx
    from test_portal_native_task_routes import _config, _post, _request, _working_reply
    from test_ui_intent_choice import configured
    from test_ui_observation_boundary import CELL

    import portal.tools as tools_module
    from portal.app import create_app
    monkeypatch.setenv('OMNI_NATIVE_COMPACT_PLANNER', '0')
    adapter, store, _, policy_requests = configured(probability=.6)
    monkeypatch.setattr(tools_module, 'ComputerUseStore', lambda *_a, **_kw: store)
    packets = []

    def upstream(request):
        body = json.loads(request.content)
        packet = json.loads(body['messages'][1]['content'])
        packets.append(packet)
        if len(packets) == 1:
            phase = {'kind': 'ui_subgoal', 'subgoal': {'id': 'delegate', 'family': 'pointer_or_activation',
                'objective': 'Open the observed control', 'target': {'node_id': CELL}}}
        elif len(packets) == 2:
            assert len(policy_requests) == 1 and not adapter.calls
            assert packet['current_state']['pending_recovery_projection_receipt']['pending_delegation']
            pointer = {'id': 'repeat', 'verb': 'double_click',
                'target': {'role': 'table_cell', 'name': 'Input'}, 'args': {}}
            phase = ({'kind': 'ui_boundary', 'objective': 'Open the observed control', 'step': pointer}
                if route == 'named_boundary' else {'kind': 'ui', 'steps': [
                    {'id': 'already-empty', 'verb': 'set_value', 'target': {'node_id': 'draft'},
                        'args': {'value': ''}, 'expect': [{'kind': 'value', 'node_id': '$target', 'value': ''}]},
                    {**pointer, 'expect': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]}],
                    'final_predicates': [{'kind': 'value', 'node_id': 'draft', 'value': ''}]})
        else:
            assert packet['last_result']['error'] == 'native_planner_requested_replan'
            phase = {'kind': 'blocked', 'reason': 'The controlled re-delegation was refused; task remains unverified.'}
        return httpx.Response(200, json={'message': {'content': json.dumps(_working_reply(packet, phase))}})

    app = create_app(_config(tmp_path), httpx.Client(transport=httpx.MockTransport(upstream)))
    result, _ = _post(app.test_client(), _request(True), True)
    assert len(packets) == 3, result
    assert len(policy_requests) == 1 and not adapter.calls
    assert result['portal']['native_task']['task_verified'] is False
    assert result['portal']['native_task']['status'] == 'blocked'


def gtk_tree_table(names, chrome=12):
    """Shape observed from GTK3 Files on the Egg: no table_row nodes; each row's first
    column is a composite table_cell holding an unnamed icon cell and the named cell."""
    scope = 'files-window'
    state = {'session_id': 'observed-session', 'revision': 'observed-revision',
        'scope': {'kind': 'desktop', 'id': scope}, 'truncated': True, 'nodes': [
            {'id': scope, 'generation': '1', 'scope_id': scope, 'role': 'window', 'name': 'Workbench',
             'visible': True, 'enabled': True, 'sensitive': False, 'operations': ['key_chord']}]}
    def ancestor(identity, role, name=''):
        return {'id': identity, 'generation': '1', 'role': role, 'name': name, 'visible': True, 'enabled': True}
    ops = ['invoke', 'focus', 'click', 'double_click', 'right_click', 'scroll']
    for index in range(chrome):  # Path-bar buttons precede the content in document order.
        state['nodes'].append({'id': f'crumb-{index}-' + 'c' * 30, 'generation': '1', 'scope_id': scope,
            'role': 'button', 'name': f'folder-{index}', 'visible': True, 'enabled': True, 'sensitive': False,
            'operations': ops, 'bounds': {'x': 10 * index, 'y': 0, 'width': 60, 'height': 30},
            'structure': {'ancestors': [ancestor('bar-' + 'b' * 32, 'panel'), ancestor('box-' + 'x' * 32, 'filler')],
                          'parent_id': 'box-' + 'x' * 32, 'child_index': index}})
    table = ancestor('table-' + 't' * 32, 'tree_table', 'List View')
    for row, name in enumerate(names):
        composite = f'row-{row}-composite-' + 'r' * 24
        state['nodes'].append({'id': composite, 'generation': '1', 'scope_id': scope, 'role': 'table_cell',
            'name': '', 'visible': True, 'enabled': True, 'sensitive': False, 'operations': ops,
            'structure': {'ancestors': [ancestor('scroll-' + 's' * 32, 'scroll_pane'), table],
                          'parent_id': table['id'], 'child_index': 4 + row * 4}})
        for column, cell_name in enumerate(('', name)):
            state['nodes'].append({'id': f'row-{row}-cell-{column}', 'generation': '1', 'scope_id': scope,
                'role': 'table_cell', 'name': cell_name, 'visible': True, 'enabled': True, 'sensitive': False,
                'operations': ops, 'bounds': {'x': 100, 'y': 100 + row * 30, 'width': 300, 'height': 30},
                'structure': {'ancestors': [table, ancestor(composite, 'table_cell')],
                              'parent_id': composite, 'child_index': column}})
    return state


@pytest.mark.parametrize('names', [['Input'], ['Input', 'Research', 'Archive'], ['café', '中']])
def test_gtk_tree_rows_without_table_row_keep_their_named_cell_ahead_of_chrome(names):
    state = gtk_tree_table(names)
    projected = _model_ui_observation(state, ('files-window',))
    ids = [node['id'] for node in projected['nodes']]
    for row in range(len(names)):
        assert f'row-{row}-cell-1' in ids  # The named cell, not the unnamed icon renderer.
        assert ids.index(f'row-{row}-cell-1') < ids.index('crumb-0-' + 'c' * 30)
    assert sum(len(json.dumps(node, ensure_ascii=False)) for node in projected['nodes']) <= 12000


@pytest.mark.parametrize('fault', ['no_table_ancestor', 'unbound_parent'])
def test_composite_cell_rows_still_require_an_observed_table_relationship(fault):
    state = gtk_tree_table(['Input'])
    for node in state['nodes']:
        if node['id'].startswith('row-0-cell-'):
            if fault == 'no_table_ancestor':
                node['structure']['ancestors'][0]['role'] = 'panel'
            else:
                node['structure']['parent_id'] = 'unobserved-parent'
    ids = [node['id'] for node in _model_ui_observation(state, ('files-window',))['nodes']]
    if 'row-0-cell-1' in ids:
        assert ids.index('row-0-cell-1') > ids.index('crumb-0-' + 'c' * 30)
