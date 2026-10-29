"""Private generated actions retain intent and verify only host-derived effects."""
import hashlib
import json

import pytest
from test_native_text_entry import Editor
from test_ui_intent_choice import configured, subgoal
from test_ui_observation_boundary import CELL, ROW, Adapter, intent, node, write

from portal import ui_decision
from portal.computer_use import ComputerUseStore
from qwen_omni_adapters.computer_use import ComputerUseError

GOAL = 'Original exact user task'
OBJECTIVE = 'Open the observed Input folder'


def _claim(state, *, family='pointer_or_activation', target=CELL, objective=OBJECTIVE, phase=1):
    observed = next(n for n in state['nodes'] if n['id'] == target)
    return {'schema': ui_decision.HANDOFF_CONTRACT,
        'authority': 'planner_claim_not_effect_or_input_authority', 'input_authority': False,
        'goal_sha256': hashlib.sha256(GOAL.encode()).hexdigest(), 'source_epoch': 'a' * 32,
        'source_phase': phase, 'source_observation': ui_decision._observation_binding(state),
        'target': ui_decision._target_binding(observed), 'operation_family': family,
        'objective': objective, 'next_operation': 'Open the chooser as a prerequisite to acquiring the source.',
        'needed_evidence': ['The declared source contents'], 'expected_transition': {
            'schema': 'robit.omni.native-expected-transition.v1', 'mode': 'observe_after_input',
            'operation_family': family, 'authority': 'host_operation_descriptor_not_observed_effect',
            'input_authority': False, 'task_progress': False}}


def _store(adapter=None):
    adapter = adapter or Adapter()
    store = ComputerUseStore(None, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=True)
    control = store._sessions[('session', 'desktop')].controller
    control.max_observations, control.verification_timeout_s, control.observation_interval_s = 1, .02, .001
    return adapter, store, cancel


def _immediate(store, cancel, step, **kwargs):
    with store.task_scope('session', 'desktop', cancel):
        return store.immediate_transition('session', step, goal=GOAL, objective=OBJECTIVE, **kwargs)


def _assert_model_handoff(compact, claim):
    context = compact['task']
    projected = context['action_handoff']
    assert context['goal'] == GOAL and context['active_objective'] == OBJECTIVE
    alias = projected['target']['node_ref']
    assert ui_decision._target_binding(compact['nodes'][alias]['observed']) == claim['target']
    assert projected == {'schema': 'robit.omni.ui-intent-handoff-projection.v1',
        **{key: claim[key] for key in ('authority', 'input_authority', 'operation_family',
            'next_operation', 'needed_evidence', 'expected_transition')},
        'target': {'node_ref': alias}, 'binding_metadata_retained_privately': True,
        'omission_is_absence': False}


def test_policy_gets_exact_immediate_operation_and_retains_claim_on_no_input_deferral():
    adapter, store, cancel, requests = configured(probability=.6)
    claim = _claim(adapter.observe())
    with store.task_scope('session', 'desktop', cancel):
        result = store.intent_boundary('session', subgoal(), goal=GOAL, action_handoff=claim)
    compact = json.loads(requests[0]['state'])
    context = compact['task']
    _assert_model_handoff(compact, claim)
    assert context['action_handoff']['needed_evidence'] == ['The declared source contents']
    assert result['action_handoff'] == claim
    assert result['intent_selection_receipt']['action_handoff'] == claim
    assert store._sessions[('session', 'desktop')].pointer_recovery['action_handoff'] == claim
    assert result['status'] == 'abstained' and not adapter.calls
    assert result['task_progress'] is False


@pytest.mark.parametrize('fault', ['goal', 'scope', 'generation', 'name', 'bounds', 'focus', 'family'])
def test_changed_claim_source_or_target_replans_before_policy_or_input(fault):
    adapter, store, cancel, requests = configured()
    claim = _claim(adapter.observe())
    goal = GOAL
    if fault == 'goal':
        goal = 'A different user request'
    elif fault == 'family':
        claim['operation_family'] = claim['expected_transition']['operation_family'] = 'scroll'
    else:
        original = adapter.observe
        def changed():
            state = original()
            if fault == 'scope':
                state['scope']['id'] = 'other-scope'
            elif fault == 'bounds':
                state['nodes'][0]['bounds']['x'] += 1
            elif fault == 'focus':
                state['nodes'][0]['focused'] = True
            else:
                state['nodes'][0][fault] = 'changed'
            return state
        adapter.observe = changed
    with store.task_scope('session', 'desktop', cancel):
        result = store.intent_boundary('session', subgoal(), goal=goal, action_handoff=claim)
    assert result['error'] == 'ui_action_handoff_stale'
    assert result['ui_boundary_receipt']['input_attempted'] is False
    assert not requests and not adapter.calls


def test_planning_only_inactive_roots_do_not_invalidate_same_current_target():
    adapter, store, cancel, requests = configured(probability=.6)
    planning = adapter.observe()
    planning['nodes'].append(node('inactive', 'window', 'Unrelated editor', ['switch_window']))
    planning['revision'] = 'planning-with-inactive-window'
    claim = _claim(planning)
    with store.task_scope('session', 'desktop', cancel):
        result = store.intent_boundary('session', subgoal(), goal=GOAL, action_handoff=claim)
    assert len(requests) == 1 and not adapter.calls
    assert result['action_handoff'] == claim
    proof = result['action_handoff_revalidation']
    assert proof['source_revision'] == 'planning-with-inactive-window'
    assert proof['current_revision'] == adapter.observe()['revision']
    assert proof['binding'] == 'exact_target_fingerprint_and_session_scope'
    assert proof['input_authority'] is False


@pytest.mark.parametrize('fault', [None, 'omitted', 'hash', 'reference', 'operation'])
def test_pointer_recovery_requires_the_exact_captured_predecessor(fault):
    adapter, store, cancel, requests = configured(probability=.6)
    claim = _claim(adapter.observe())
    with store.task_scope('session', 'desktop', cancel):
        deferred = store.intent_boundary('session', subgoal(), goal=GOAL, action_handoff=claim)
        offer = deferred['pointer_recovery']
        rebound = _claim(adapter.observe(), family='pointer_recovery', phase=2)
        rebound['predecessor'] = {'handoff_sha256': ui_decision._digest(claim), 'source_phase': 1,
            'recovery_reference': offer['reference']}
        if fault == 'hash':
            rebound['predecessor']['handoff_sha256'] = '0' * 64
        elif fault == 'reference':
            rebound['predecessor']['recovery_reference'] = 'b' * 32
        elif fault == 'operation':
            rebound['next_operation'] = 'A substituted operation'
        result = store.resolve_pointer('session', offer['reference'], intent('double_click'), goal=GOAL,
            objective=OBJECTIVE, action_handoff=None if fault == 'omitted' else rebound)
    assert len(requests) == 1
    if fault is None:
        assert len(adapter.calls) == 1 and result['status'] == 'observed_unverified'
        assert result['action_handoff'] == rebound and result['task_progress'] is False
    else:
        assert not adapter.calls and result['error'] == 'ui_action_handoff_stale'
    assert store._sessions[('session', 'desktop')].pointer_recovery is None


def test_generated_focus_replaces_narrative_oracle_with_host_focus_predicate():
    adapter, store, cancel = _store()
    step = {'id': 'focus', 'verb': 'focus', 'target': {'node_id': ROW}, 'args': {},
        'expect': [{'kind': 'text', 'text': 'Workbench file manager surface text'}]}
    result = _immediate(store, cancel, step)
    assert result['status'] == 'complete' and len(adapter.calls) == 1
    assert adapter.calls[0]['expect'] == [{'kind': 'focused', 'node_id': ROW}]
    assert result['immediate_effect_contract']['mode'] == 'verified_step'
    assert result['phase_handoff']['goal_verified'] is False
    assert result['task_progress'] is False


def test_generated_write_checks_exact_value_instead_of_already_focused_field():
    adapter, store, cancel = _store()
    adapter.mutate = False
    step = write('The exact URL')
    step['expect'] = [{'kind': 'focused', 'node_id': '$target'}]
    result = _immediate(store, cancel, step)
    assert result['status'] == 'needs_replan'
    assert adapter.calls[0]['expect'] == [{'kind': 'value', 'node_id': 'draft', 'value': 'The exact URL'}]
    assert adapter.value == ''
    assert result['error'] == 'ui_immediate_effect_unresolved'
    assert result['task_progress'] is False


def test_generated_opaque_pointer_remains_usable_without_choice_and_gets_no_effect_credit():
    adapter, store, cancel = _store()
    step = {**intent('double_click'), 'expect': [{'kind': 'text', 'text': 'Narrative is not an oracle'}]}
    result = _immediate(store, cancel, step)
    assert len(adapter.calls) == 1 and adapter.calls[0]['verb'] == 'double_click'
    assert result['status'] == 'observed_unverified'
    assert result['effect_receipt']['effect_state'] == 'unknown'
    assert result['immediate_effect_contract']['predicates'] == []
    assert result['immediate_transition_receipt']['comparison']['status'] == 'unknown'
    assert result['new_effect_count'] == 0 and result['task_progress'] is False


def test_generated_opaque_pointer_still_uses_active_choice_and_preserves_claim():
    adapter, store, cancel, requests = configured(selected_verb='double_click')
    claim = _claim(adapter.observe())
    result = _immediate(store, cancel, intent('click'), action_handoff=claim)
    assert len(requests) == len(adapter.calls) == 1
    assert adapter.calls[0]['verb'] == 'double_click'
    _assert_model_handoff(json.loads(requests[0]['state']), claim)
    assert result['action_handoff'] == claim and result['task_progress'] is False


def test_unsupported_immediate_effect_and_unobserved_typing_caret_do_not_dispatch():
    for step in [dict(id='select', verb='select', target={'node_id': 'draft'}, args={'option': 'value'}),
                 dict(id='type', verb='type_text', target={'node_id': 'draft'}, args={'text': 'value'})]:
        adapter, store, cancel = _store()
        result = _immediate(store, cancel, step)
        assert result['error'] == 'ui_immediate_effect_requires_replan'
        assert result['input_attempted'] is False and not adapter.calls


def test_typed_transition_planning_source_is_explicitly_rebound_after_target_proof():
    adapter, store, cancel = _store()
    state = adapter.observe()
    state['revision'] = 'planning-only-window-list'
    state['nodes'][0]['operations'].append('key_chord')
    original = adapter.observe
    def keyboard_state():
        current = original()
        current['nodes'][0]['operations'].append('key_chord')
        return current
    adapter.observe = keyboard_state
    claim = _claim(state, family='key_chord')
    source = {'session_id': state['session_id'], 'scope_id': state['scope']['id'], 'revision': state['revision']}
    transition = {'kind': 'focus_changed_within_scope', 'source_ref': source}
    result = _immediate(store, cancel, {'id': 'focus-location', 'verb': 'key_chord',
        'target': {'node_id': CELL}, 'args': {'keys': ['Control_L', 'l']}},
        action_handoff=claim, transition=transition)
    assert len(adapter.calls) == 1
    assert result['immediate_transition_receipt']['original_source_ref'] == source
    assert result['immediate_effect_contract']['source_ref']['revision'] != source['revision']
    assert result['action_handoff'] == claim
    assert result['task_progress'] is False


def test_transition_source_without_matching_claim_cannot_be_rebound():
    adapter, store, cancel = _store()
    result = _immediate(store, cancel, intent(), transition={'kind': 'popup_opened',
        'source_ref': {'session_id': 'session', 'scope_id': 'scope', 'revision': 'stale'}})
    assert not adapter.calls and result['error'] == 'ui_action_handoff_stale'


def test_public_workflow_keeps_its_original_requested_terminal_conditions():
    adapter, store, _ = _store()
    step = write('Exact write')
    result = store.act('session', {'surface': 'desktop', 'operation': 'workflow', 'steps': [step],
        'final_predicates': [{'kind': 'text', 'text': 'Not present in this application'}]}, goal=GOAL)
    assert result['status'] == 'needs_replan'
    assert result['error'] == 'ui_workflow_unresolved'
    assert 'immediate_effect_contract' not in result


def test_immediate_transition_is_private_and_not_a_new_public_operation():
    _, store, cancel = _store()
    with pytest.raises(ComputerUseError):
        store.immediate_transition('session', intent(), goal=GOAL, objective=OBJECTIVE)
    with store.task_scope('session', 'desktop', cancel), pytest.raises(ComputerUseError):
        store.act('session', {'surface': 'desktop', 'operation': 'immediate_transition', 'step': intent()}, goal=GOAL)


def test_typing_selection_change_between_compilation_and_choice_stops_before_input():
    editor = Editor('old')
    editor.focused = True
    editor.selection = {'start': 0, 'end': 3}
    _, store, cancel = _store(editor)
    original = editor.observe
    reads = 0
    def changed():
        nonlocal reads
        reads += 1
        if reads >= 2:
            editor.selection = None
        return original()
    editor.observe = changed
    result = _immediate(store, cancel, {'id': 'type', 'verb': 'type_text',
        'target': {'node_id': 'draft'}, 'args': {'text': 'new'}})
    assert not editor.calls and editor.value == 'old'
    assert result['status'] == 'needs_replan' and result['task_progress'] is False
