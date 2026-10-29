"""The host task ledger constrains actual choices and the final input barrier."""
import copy
import json

import pytest
from test_native_effect_integration import GOAL, OBJECTIVE, _store
from test_native_text_entry import Editor
from test_native_text_entry import intent as text_intent
from test_ui_intent_choice import configured, subgoal
from test_ui_observation_boundary import CELL, intent, write

from portal.ui_recovery import NativeStrategyLedger
from qwen_omni_adapters.computer_use import ComputerUseError


def _uncertain(ledger, state, verb='click', target=CELL, args=None):
    node = next(n for n in state['nodes'] if n['id'] == target)
    record = {'status': 'unresolved', 'dispatch_attempted': True, 'before': copy.deepcopy(state),
        'action': {'id': 'previous-action', 'idempotency_key': 'previous-input', 'kind': 'action',
            'node_id': node['id'], 'generation': node['generation'], 'session_id': state['session_id'],
            'scope_id': state['scope']['id'], 'revision': state['revision'], 'verb': verb, 'args': args or {}},
        'receipt': {'status': 'dispatched', 'input_attempted': True}}
    ledger.record(record, source_phase=1)


def _blocked(receipts):
    return [item for receipt in receipts for item in receipt['blocked']]


def test_exact_same_ledger_filters_failed_primitive_before_kev_and_keeps_alternatives():
    adapter, store, cancel, requests = configured(selected_verb='double_click')
    ledger = NativeStrategyLedger()
    _uncertain(ledger, adapter.observe(), 'click')
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.intent_boundary('session', subgoal(), goal=GOAL)
    assert len(requests) == len(adapter.calls) == 1
    proposals = [json.loads(v) for v in requests[0]['questions']['action']['criteria'].values()]
    assert 'click' not in {p.get('verb') for p in proposals}
    assert {'double_click', 'right_click', 'invoke'} <= {p.get('verb') for p in proposals}
    assert {'observe', 'abstain'} <= {p['kind'] for p in proposals}
    assert adapter.calls[0]['verb'] == 'double_click'
    assert any(b['repeat_status'] == 'uncertain_input_requires_reconciliation' for b in _blocked(receipts))
    assert result['task_progress'] is False
    # Store reads this object; only the root records actual returned outcomes.
    assert ledger.view()['retained_entry_count'] == 1


def test_uncertain_target_remains_blocked_after_geometry_or_focus_changes_and_new_wording():
    adapter, store, cancel = _store()
    ledger = NativeStrategyLedger()
    _uncertain(ledger, adapter.observe())
    original = adapter.observe
    def changed():
        state = original()
        state['revision'] = 'fresh-window-revision'
        state['nodes'][0]['focused'] = True
        state['nodes'][0]['bounds']['x'] += 25
        return state
    adapter.observe = changed
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.immediate_transition('session', intent(identity='a fresh alternative'), goal=GOAL,
            objective='Try another better new approach')
    assert not adapter.calls
    assert result['ui_boundary_receipt']['input_attempted'] is False
    assert any(b['repeat_status'] == 'uncertain_input_requires_reconciliation' for b in _blocked(receipts))


def test_scope_capacity_exhaustion_keeps_abstention_and_never_dispatches_policy():
    adapter, store, cancel, requests = configured()
    ledger = NativeStrategyLedger(max_entries=1)
    _uncertain(ledger, adapter.observe(), 'click')
    _uncertain(ledger, adapter.observe(), 'double_click')
    assert ledger.capacity_exhausted
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.intent_boundary('session', subgoal(), goal=GOAL)
    assert not requests and not adapter.calls
    assert result['ui_boundary_receipt']['input_attempted'] is False
    assert len(_blocked(receipts)) == 4


def test_policy_deferral_without_input_does_not_lock_legal_choices():
    adapter, store, cancel, requests = configured()
    ledger = NativeStrategyLedger()
    ledger.record({'status': 'abstained', 'ui_boundary_receipt': {'input_attempted': False,
        'adapter_called': False}, 'intent_selection_receipt': {'objective': OBJECTIVE}})
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger):
        result = store.intent_boundary('session', subgoal(), goal=GOAL)
    assert len(requests) == len(adapter.calls) == 1
    assert result['status'] == 'observed_unverified'


def test_policy_returning_after_history_changes_cannot_dispatch_stale_selected_primitive():
    ledger = NativeStrategyLedger()
    adapter = None
    def history_changed():
        _uncertain(ledger, adapter.observe(), 'double_click')
    adapter, store, cancel, requests = configured(hook=history_changed)
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.intent_boundary('session', subgoal(), goal=GOAL)
    assert len(requests) == 1 and not adapter.calls
    assert result['ui_boundary_receipt']['input_attempted'] is False
    assert any(receipt['stage'] == 'action_offers' and receipt['blocked'] for receipt in receipts)


def test_private_selection_callback_cannot_bypass_postselection_strategy_check():
    adapter, store, cancel = _store()
    ledger = NativeStrategyLedger()
    def choose(state, offers):
        action = next(a for a in offers if a['kind'] == 'action')
        _uncertain(ledger, state, action['verb'])
        return {'selected_action_id': action['id'], 'selection_status': 'forced_callback'}
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.observation_boundary('session', intent(), goal=GOAL, objective=OBJECTIVE, _selection=choose)
    assert not adapter.calls and result['ui_boundary_receipt']['input_attempted'] is False
    assert result['selection_receipts'][0]['selection_status'] == 'strategy_history_blocked'
    assert any(receipt['stage'] == 'selected_offer' and receipt['blocked'] for receipt in receipts)


def test_final_authorizer_blocks_direct_controller_choice_that_skips_store_filtering():
    adapter, store, cancel = _store()
    ledger = NativeStrategyLedger()
    _uncertain(ledger, adapter.observe(), 'set_value', target='draft', args={'value': 'unsafe replay'})
    controller = store._sessions[('session', 'desktop')].controller
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = controller.run_step(write('unsafe replay'), lambda state, actions:
            {'choice': next(a['id'] for a in actions if a['kind'] == 'action')}, cancel)
    assert result['status'] == 'rejected' and not adapter.calls
    assert any(receipt['stage'] == 'authorize' and receipt['blocked'] for receipt in receipts)


def test_internal_text_entry_physical_typing_cannot_bypass_final_authorizer():
    editor = Editor('')
    original = editor.observe
    def physical_only():
        state = original()
        state['nodes'][0]['operations'].remove('set_value')
        return state
    editor.observe = physical_only
    _, store, cancel = _store(editor)
    ledger = NativeStrategyLedger()
    _uncertain(ledger, editor.observe(), 'type_text', target='draft', args={'text': 'New value'})
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        result = store.text_entry('session', text_intent(), goal=GOAL)
    assert [a['verb'] for a in editor.calls] == ['focus']
    assert editor.value == '' and result['status'] != 'complete'
    assert any(receipt['stage'] == 'authorize' and receipt['blocked'] for receipt in receipts)


@pytest.mark.parametrize('path', ['intent', 'direct'])
def test_strategy_assessment_error_fails_closed_before_input(path):
    adapter, store, cancel, requests = configured()
    ledger = NativeStrategyLedger()
    def broken(*args):
        raise RuntimeError('controlled ledger failure')
    ledger.assess = broken
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        if path == 'intent':
            result = store.intent_boundary('session', subgoal(), goal=GOAL)
            assert result['ui_boundary_receipt']['input_attempted'] is False
        else:
            result = store._sessions[('session', 'desktop')].controller.run_step(write('New value'),
                lambda state, actions: {'choice': next(a['id'] for a in actions if a['kind'] == 'action')}, cancel)
            assert result['status'] == 'rejected'
    assert not requests and not adapter.calls
    assert all(b['repeat_status'] == 'strategy_assessment_unavailable' for b in _blocked(receipts))


def test_nested_same_task_inherits_same_ledger_and_receipt_list_but_cannot_replace_it():
    adapter, store, cancel = _store()
    ledger = NativeStrategyLedger()
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as outer:
        with store.task_scope('session', 'desktop', cancel) as inner:
            assert inner is outer and store._strategy_binding.get().ledger is ledger
            store.immediate_transition('session', write('New value'), goal=GOAL, objective=OBJECTIVE)
        with pytest.raises(ComputerUseError), store.task_scope('session', 'desktop', cancel,
                strategy_ledger=NativeStrategyLedger()):
            pass
    assert adapter.value == 'New value' and outer
    assert store._strategy_binding.get() is None


def test_ordinary_portable_workflow_without_bound_ledger_retains_existing_behavior():
    adapter, store, _ = _store()
    unused_ledger = NativeStrategyLedger()
    _uncertain(unused_ledger, adapter.observe(), 'set_value', target='draft', args={'value': 'Allowed write'})
    result = store.act('session', {'surface': 'desktop', 'operation': 'workflow', 'steps': [write('Allowed write')],
        'final_predicates': [{'kind': 'value', 'node_id': 'draft', 'value': 'Allowed write'}]}, goal=GOAL)
    assert result['status'] == 'complete' and adapter.value == 'Allowed write'


def test_old_epoch_ledger_cannot_authorize_a_new_controller_and_new_scope_does_not_inherit_it():
    adapter, store, old_cancel = _store()
    old_ledger = NativeStrategyLedger()
    _uncertain(old_ledger, adapter.observe())
    with store.task_scope('session', 'desktop', old_cancel, strategy_ledger=old_ledger) as old_receipts:
        new_cancel = store.begin_task('session', private_boundaries=True)
        new_owned = store._sessions[('session', 'desktop')]
        state = new_owned.controller.observe()
        offers = new_owned.controller._enumerate(intent(), state, boundary=True)
        action = next(a for a in offers if a['kind'] == 'action')
        assert new_owned.controller.authorize(action, state) is False
        assert old_cancel.is_set() and not adapter.calls
        with store.task_scope('session', 'desktop', new_cancel) as new_receipts:
            assert store._strategy_binding.get() is None and new_receipts is not old_receipts
        assert store._strategy_binding.get().ledger is old_ledger
    assert any(b['repeat_status'] == 'strategy_assessment_unavailable' for b in _blocked(old_receipts))


def test_another_session_scope_does_not_inherit_outer_task_strategy_history():
    _, store, first_cancel = _store()
    second_cancel = store.begin_task('other-session', private_boundaries=True)
    ledger = NativeStrategyLedger()
    with store.task_scope('session', 'desktop', first_cancel, strategy_ledger=ledger) as first:
        with store.task_scope('other-session', 'desktop', second_cancel) as second:
            assert store._strategy_binding.get() is None and second is not first
        assert store._strategy_binding.get().ledger is ledger


def test_model_arguments_cannot_supply_a_strategy_ledger_or_disable_bound_filter():
    _, store, cancel = _store()
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=NativeStrategyLedger()):
        for extra in [{'strategy_ledger': {}}, {'disable_strategy_filter': True}]:
            with pytest.raises(ComputerUseError):
                store.act('session', {'surface': 'desktop', 'operation': 'observe', **extra})


def test_filter_receipt_collection_is_bounded_and_does_not_forget_ledger_locks():
    adapter, store, cancel = _store()
    owned = store._sessions[('session', 'desktop')]
    ledger = NativeStrategyLedger()
    _uncertain(ledger, adapter.observe())
    with store.task_scope('session', 'desktop', cancel, strategy_ledger=ledger) as receipts:
        state = owned.controller.observe()
        offers = owned.controller._enumerate(intent(), state, boundary=True)
        for _ in range(60):
            store._strategy_filter(owned, state, offers, stage='action_offers')
        assert len(receipts) <= 32
        assert len(json.dumps(receipts, ensure_ascii=False).encode()) <= 32768
        assert receipts[-1]['omitted_prior_receipts'] > 0
        assert ledger.assess(state, intent())['blocked']
