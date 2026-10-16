"""Verified state cycles renew explicit intent; uncertain effects never do."""
from __future__ import annotations

import copy
import threading

import pytest

from portal.computer_use import _effect_result
from qwen_omni_adapters.computer_use import RapidUiController, _digest


class CyclingAdapter:
    def __init__(self, session='epoch-fixture'):
        self.session, self.active, self.document = session, 'editor', 'about:blank'
        self.switches, self.inputs = [], []
        self.observations = 0

    def observe(self):
        self.observations += 1
        nodes = [{'id': name, 'scope_id': name, 'generation': '1', 'role': 'window',
            'name': name, 'visible': True, 'enabled': True, 'sensitive': False,
            'operations': ['switch_window']} for name in ('editor', 'firefox')]
        nodes += [{'id': 'address', 'scope_id': self.active, 'generation': '1', 'role': 'textbox',
            'name': 'address', 'visible': True, 'enabled': True, 'sensitive': False,
            'value': self.document, 'operations': ['set_value']},
            {'id': 'uncertain', 'scope_id': self.active, 'generation': '1', 'role': 'button',
            'name': 'uncertain', 'visible': True, 'enabled': True, 'sensitive': False,
            'focused': False, 'operations': ['focus']}]
        return {'session_id': self.session, 'revision': _digest([self.active, self.document]),
            'scope': {'kind': 'desktop', 'id': self.active}, 'nodes': nodes}

    def perform(self, action):
        self.inputs.append(copy.deepcopy(action))
        if action['verb'] == 'switch_window':
            if self.active == action['node_id']:
                return {'status': 'already_satisfied', 'input_attempted': False}
            self.active = action['node_id']
            self.switches.append(self.active)
        elif action['verb'] == 'set_value':
            self.document = action['args']['value']
        return {'status': 'dispatched'}


def controller(adapter):
    return RapidUiController(adapter, authorize=lambda *_: True, max_observations=1,
        verification_timeout_s=.05)


def switch(name):
    return {'id': 'activate-' + ('text-editor' if name == 'editor' else 'firefox'),
        'verb': 'switch_window', 'target': {'node_id': name}, 'args': {},
        'expect': [{'kind': 'active_window', 'node_id': '$target'}]}


def set_document(value):
    return {'id': 'navigate-official-source', 'verb': 'set_value', 'target': {'node_id': 'address'},
        'args': {'value': value}, 'expect': [{'kind': 'value', 'node_id': '$target', 'value': value}]}


def uncertain():
    return {'id': 'uncertain-focus', 'verb': 'focus', 'target': {'node_id': 'uncertain'}, 'args': {},
        'expect': [{'kind': 'focused', 'node_id': '$target'}]}


def test_actual072_shape_returns_to_editor_after_verified_browser_mutations():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    outcomes = [rapid.run_step(switch('firefox')),
        rapid.run_step(set_document('https://git-scm.com/docs/git-status')),
        rapid.run_step(switch('editor')), rapid.run_step(switch('firefox')),
        rapid.run_step(switch('editor')), rapid.run_step(switch('firefox'))]
    assert adapter.switches == ['firefox', 'editor', 'firefox', 'editor', 'firefox']
    assert all(outcome['status'] == 'verified' for outcome in outcomes)
    assert outcomes[2]['action']['idempotency_key'] != outcomes[4]['action']['idempotency_key']
    assert outcomes[4]['before']['scope']['id'] == 'firefox'
    assert outcomes[4]['after']['scope']['id'] == 'editor'


@pytest.mark.parametrize('clarification', ['observe', 'metadata', 'already_satisfied'])
def test_own_completion_without_later_verified_mutation_cannot_renew(clarification):
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    first = rapid.run_step(switch('firefox'))
    assert first['status'] == 'verified' and rapid._execution_epoch == 1
    adapter.active = 'editor'  # External reset is not a controller-verified transition.
    if clarification == 'metadata':
        original = adapter.observe

        def metadata_only():
            state = original()
            state['nodes'][2]['text_observation'] = {'selection_count': 0}
            return state

        adapter.observe = metadata_only
    if clarification == 'already_satisfied':
        assert rapid.run_step(switch('editor'))['receipt']['status'] == 'already_satisfied'
    else:
        rapid.observe()
    calls = len(adapter.inputs)
    repeat = rapid.run_step(switch('firefox'))
    assert repeat == first and len(adapter.inputs) == calls
    assert adapter.active == 'editor' and rapid._execution_epoch == 1
    assert rapid.run_step(set_document('temporary'))['status'] == 'verified'
    assert rapid.run_step(set_document('about:blank'))['status'] == 'verified'
    renewed = rapid.run_step(switch('firefox'))
    assert renewed['action']['idempotency_key'] != first['action']['idempotency_key']
    assert renewed['status'] == 'verified' and len(adapter.inputs) == calls + 3


def test_uncertain_legacy_intent_stays_locked_after_unrelated_verified_state_cycle():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    first = rapid.run_step(uncertain())
    assert first['status'] == 'unresolved'
    assert rapid.run_step(set_document('temporary'))['status'] == 'verified'
    assert rapid.run_step(set_document('about:blank'))['status'] == 'verified'
    repeat = rapid.run_step(uncertain())
    assert repeat == first
    assert sum(action['verb'] == 'focus' for action in adapter.inputs) == 1


def test_missing_completion_epoch_keeps_verified_binding_locked():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    first = rapid.run_step(switch('firefox'))
    key = first['action']['idempotency_key']
    assert rapid._completion_epochs[key] == rapid._execution_epoch == 1
    assert rapid.run_step(switch('editor'))['status'] == 'verified'
    del rapid._completion_epochs[key]  # No independent proof of a later mutation.
    calls = len(adapter.inputs)
    assert rapid.run_step(switch('firefox')) == first
    assert len(adapter.inputs) == calls and adapter.active == 'editor'


@pytest.mark.parametrize('failure', ['unverified', 'exception-after-mutation', 'cancelled-after-mutation'])
def test_possible_real_effect_without_verified_receipt_stays_locked_after_recovery(failure):
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    request = set_document('possibly-applied')
    if failure == 'unverified':
        request['expect'][0]['value'] = 'not-observed'
    cancel = threading.Event()
    original = adapter.perform

    def perform(action):
        receipt = original(action)
        if action['args'].get('value') == 'possibly-applied':
            if failure == 'exception-after-mutation':
                raise RuntimeError('provider effect unknown')
            if failure == 'cancelled-after-mutation':
                cancel.set()
        return receipt

    adapter.perform = perform
    first = rapid.run_step(request, cancel=cancel if failure == 'cancelled-after-mutation' else None)
    assert first['status'] in {'unresolved', 'cancelled'} and rapid._execution_epoch == 0
    assert adapter.document == 'possibly-applied'
    cancel.clear()
    assert rapid.run_step(set_document('recovery-state'))['status'] == 'verified'
    assert rapid.run_step(set_document('about:blank'))['status'] == 'verified'
    before = len(adapter.inputs)
    repeat = rapid.run_step(request)
    assert repeat == first and len(adapter.inputs) == before


def offer(rapid, step):
    state = rapid.observe()
    return next(action for action in rapid.enumerate(step, state) if action['kind'] == 'action')


def test_original_recorded_token_stays_historical_after_later_verified_cycles():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    action = offer(rapid, switch('firefox'))
    first = rapid.execute(action['id'])
    assert rapid.run_step(switch('editor'))['status'] == 'verified'
    later = rapid.run_step(switch('firefox'))
    calls, reads = len(adapter.inputs), adapter.observations
    assert later['action']['idempotency_key'] != first['action']['idempotency_key']
    assert rapid.execute(action['id']) == first
    assert len(adapter.inputs) == calls and adapter.observations == reads
    assert adapter.switches == ['firefox', 'editor', 'firefox']
    settled = set()
    assert _effect_result(first, settled)['effect_receipt']['effect_state'] == 'applied'
    assert _effect_result(later, settled)['effect_receipt']['effect_state'] == 'applied'
    assert _effect_result(first, settled)['effect_receipt']['effect_state'] == 'replayed'


@pytest.mark.parametrize('outcome', ['noop', 'already_satisfied', 'rejected', 'cancelled', 'exception'])
def test_non_applied_intent_cannot_renew_after_other_verified_mutations(outcome):
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    request = set_document('about:blank') if outcome == 'noop' else switch('editor') if outcome == 'already_satisfied' else uncertain()
    original = adapter.perform
    cancel = threading.Event()

    def perform(action):
        if action['verb'] != 'focus':
            return original(action)
        adapter.inputs.append(copy.deepcopy(action))
        if outcome == 'rejected':
            return {'status': 'rejected'}
        if outcome == 'exception':
            raise RuntimeError('provider effect uncertain')
        if outcome == 'cancelled':
            cancel.set()
        return {'status': 'dispatched'}

    adapter.perform = perform
    first = rapid.run_step(request, cancel=cancel if outcome == 'cancelled' else None)
    assert rapid._execution_epoch == 0
    cancel.clear()
    assert rapid.run_step(set_document('temporary'))['status'] == 'verified'
    assert rapid.run_step(set_document('about:blank'))['status'] == 'verified'
    assert rapid._execution_epoch == 2
    before = len(adapter.inputs)
    repeat = rapid.run_step(request)
    assert repeat == first and len(adapter.inputs) == before


def test_new_offer_after_metadata_clarification_keeps_uncertain_binding_and_epoch():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    first = rapid.run_step(uncertain())
    original = adapter.observe

    def metadata_only():
        state = original()
        node = next(node for node in state['nodes'] if node['id'] == 'address')
        node['text_observation'] = {'character_count': len(node['value']) + 1, 'selection_count': 1,
            'captured_range': {'start': 0, 'end': len(node['value'])}, 'truncated': True}
        return state

    adapter.observe = metadata_only
    repeat = rapid.run_step(uncertain())
    assert repeat == first and rapid._execution_epoch == 0
    assert len(adapter.inputs) == 1


def test_raw_revision_change_without_semantic_change_does_not_advance_epoch():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    original = adapter.observe

    def visual_only():
        state = original()
        state['base_semantic_revision'] = 'same-semantic-view'
        return state

    adapter.observe = visual_only
    outcome = rapid.run_step(set_document('new-render-only-fixture'))
    assert outcome['status'] == 'verified' and rapid._execution_epoch == 0
    assert _effect_result(outcome)['effect_receipt']['effect_state'] == 'unchanged'


def test_session_change_cannot_advance_local_execution_epoch():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    original = adapter.perform

    def change_session(action):
        receipt = original(action)
        adapter.session = 'different-provider-session'
        return receipt

    adapter.perform = change_session
    rapid.run_step(switch('firefox'))
    assert rapid._execution_epoch == 0


def test_controller_sessions_do_not_share_epoch_or_historical_tokens():
    first, second = controller(CyclingAdapter('one')), controller(CyclingAdapter('two'))
    result = first.run_step(switch('firefox'))
    assert first._execution_epoch == 1 and second._execution_epoch == 0
    assert second.execute(result['action']['id'])['status'] == 'rejected'
    assert second.adapter.inputs == []


def test_pending_same_legacy_offer_remains_cached_across_concurrent_completion():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    entered, release = threading.Event(), threading.Event()
    original = adapter.perform

    def held(action):
        entered.set()
        assert release.wait(2)
        return original(action)

    adapter.perform = held
    action = offer(rapid, switch('firefox'))
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(rapid.execute(action['id'])))
    worker.start()
    assert entered.wait(2)
    try:
        pending = offer(rapid, switch('firefox'))
        assert pending['idempotency_key'] == action['idempotency_key']
        assert rapid.execute(pending['id'])['status'] == 'unresolved'
    finally:
        release.set()
        worker.join(2)
    assert outcomes[0]['status'] == 'verified' and adapter.switches == ['firefox']
    assert rapid.execute(pending['id']) == outcomes[0]
    assert adapter.switches == ['firefox'] and rapid._execution_epoch == 1


def test_never_dispatched_offer_from_old_epoch_rejects_after_state_cycles_back():
    adapter = CyclingAdapter()
    rapid = controller(adapter)
    entered, release = threading.Event(), threading.Event()
    original = adapter.perform

    def held(action):
        entered.set()
        assert release.wait(2)
        return original(action)

    adapter.perform = held
    first = offer(rapid, switch('firefox'))
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(rapid.execute(first['id'])))
    worker.start()
    assert entered.wait(2)
    try:
        stale = offer(rapid, set_document('new-document'))
    finally:
        release.set()
        worker.join(2)
    assert outcomes[0]['status'] == 'verified' and rapid._execution_epoch == 1
    adapter.active = 'editor'  # External state cycles back; no controller epoch reset.
    before = len(adapter.inputs)
    result = rapid.execute(stale['id'])
    assert result['status'] == 'rejected' and 'epoch' in result['reason'].lower()
    assert len(adapter.inputs) == before
    adapter.perform = original
    fresh = rapid.run_step(set_document('new-document'))
    assert fresh['status'] == 'verified' and len(adapter.inputs) == before + 1


@pytest.mark.parametrize('token', [None, False, [], {}, 'unknown-token'])
def test_malformed_or_foreign_token_never_reaches_adapter(token):
    rapid = controller(CyclingAdapter())
    assert rapid.execute(token)['status'] == 'rejected'
    assert rapid.adapter.inputs == []
