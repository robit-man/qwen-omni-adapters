"""A reusable text/Return sequence requires observed preparation before input."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_text_entry import Editor

from portal import ui_decision
from portal.computer_use import ComputerUseStore, parse_subgoal
from portal.ui_task import (
    _compact_result,
    _grounded_phase_schema,
    _NativeWorkingState,
    _working_phase_schema,
)
from qwen_omni_adapters.computer_use import ComputerUseError, parse_intent


class SubmitEditor(Editor):
    def __init__(self, value='old'):
        super().__init__(value)
        self.submits = 0
        self.after_submission_empty = True
        self.supports_submit = True

    def observe(self):
        state = super().observe()
        if self.supports_submit:
            state['nodes'][0]['operations'].append('key_chord')
        return state

    def prepare_intent(self, intent):
        parse_intent(intent)

    def finish_intent(self):
        pass

    def boundary_observe(self):
        return self.observe()

    def perform(self, action):
        if action['verb'] == 'key_chord':
            self.calls.append(copy.deepcopy(action))
            assert self.focused
            assert action['args']['keys'] == ['Return']
            self.submits += 1
            if self.after_submission_empty:
                self.value = ''
                self.revision += 1
            return {'status': 'dispatched'}
        return super().perform(action)


def subgoal(text='Exact navigation text'):
    return {'id': 'submit-text', 'family': 'text_submit', 'objective': 'Submit the supplied location',
            'target': {'node_id': 'draft'}, 'mode': 'replace', 'text': text}


def fixture():
    editor = SubmitEditor()
    store = ComputerUseStore(None, adapter_factory=lambda *_: editor)
    cancel = store.begin_task('session', private_boundaries=True)
    return editor, store, cancel


def run(store, cancel, goal=None):
    with store.task_scope('session', 'desktop', cancel):
        return store.intent_boundary('session', goal or subgoal(), goal='Navigate to the supplied resource')


def test_exact_replacement_precedes_one_return_and_does_not_prove_navigation():
    editor, store, cancel = fixture()
    result = run(store, cancel)
    assert [row['verb'] for row in editor.calls] == ['focus', 'select_text', 'type_text', 'key_chord']
    assert editor.submits == 1 and editor.value == ''
    assert result['status'] == 'observed_unverified'
    assert result['text_entry_receipt']['exact_value_verified'] is True
    assert result['text_submit_receipt']['text_verified'] is True
    assert result['text_submit_receipt']['submit_attempted'] is True
    assert result['text_submit_receipt']['post_submit_effect'] == 'unknown'
    assert result['effect_receipt']['effect_state'] == 'unknown'
    assert result['task_progress'] is False and result['new_effect_count'] == 0
    assert result['text_submit_receipt']['goal_verified'] is False
    assert len(result['outcomes']) == 4
    assert _compact_result(result)['text_submit_receipt'] == result['text_submit_receipt']


@pytest.mark.parametrize('fault', ['partial_write', 'lost_write_ack', 'cancel_after_write'])
def test_failed_or_uncertain_preparation_never_submits(fault):
    editor, store, cancel = fixture()
    def after(editor, action):
        if action['verb'] == 'type_text':
            if fault == 'partial_write':
                editor.value = editor.value[:-2]
            elif fault == 'lost_write_ack':
                raise RuntimeError('lost write acknowledgement')
            else:
                cancel.set()
    editor.hook = after
    result = run(store, cancel)
    assert editor.submits == 0 and all(a['verb'] != 'key_chord' for a in editor.calls)
    assert result['text_submit_receipt']['submit_attempted'] is False


@pytest.mark.parametrize('change', ['focus', 'value', 'generation', 'capability', 'scope', 'sensitive'])
def test_pre_submit_guard_rejects_changes_after_verified_preparation(change):
    editor, store, cancel = fixture()
    original = store.observation_boundary
    def changed(*args, **kwargs):
        if change == 'focus':
            editor.focused = False
        elif change == 'value':
            editor.value = 'Unexpected content'
        elif change == 'generation':
            editor.generation = 'new'
        elif change == 'capability':
            editor.supports_submit = False
        else:
            observed = editor.observe
            def different():
                state = observed()
                if change == 'scope':
                    state['scope']['id'] = 'other-window'
                else:
                    state['nodes'][0]['sensitive'] = True
                return state
            editor.observe = different
        return original(*args, **kwargs)
    store.observation_boundary = changed
    result = run(store, cancel)
    assert editor.submits == 0
    assert result['status'] != 'observed_unverified'
    assert result['text_submit_receipt']['text_verified'] is True


def test_capability_missing_before_preparation_does_not_write():
    editor, store, cancel = fixture()
    editor.supports_submit = False
    assert run(store, cancel)['status'] == 'abstained'
    assert not editor.calls


def test_repeat_after_unknown_unchanged_submission_does_not_press_return_twice():
    editor, store, cancel = fixture()
    editor.after_submission_empty = False
    first = run(store, cancel)
    assert first['status'] == 'observed_unverified'
    second = run(store, cancel, {**subgoal(), 'id': 'a-new-plan-id'})
    assert editor.submits == 1
    assert second['ui_boundary_receipt']['cached'] is True


@pytest.mark.parametrize('bad', [
    {'target': {'role': 'textbox'}}, {'mode': 'append'}, {'text': ''}, {'objective': ' '},
    {'keys': ['Return']}, {'expect': []},
])
def test_submission_is_one_closed_explicit_sequence(bad):
    with pytest.raises(ComputerUseError):
        parse_subgoal({**subgoal(), **bad})


def test_maximum_identity_length_does_not_fail_after_writing():
    editor, store, cancel = fixture()
    result = run(store, cancel, {**subgoal(), 'id': 'x' * 200})
    assert result['status'] == 'observed_unverified' and editor.submits == 1


def test_compact_and_canonical_schema_admit_same_explicit_sequence():
    editor, _, _ = fixture()
    observation = {'state': editor.observe()}
    working = _NativeWorkingState('Navigate to the supplied resource').view(1, 0)
    canonical = _working_phase_schema(['native_ui'], _grounded_phase_schema(['native_ui'], observation, intent_choice=True), working)
    packet = {'goal': 'Navigate to the supplied resource', 'current_state': observation, 'working_state': working,
              'ledger': [], 'last_result': None, 'response_schema': canonical}
    compact, binding = ui_decision.project(packet)
    target = next(k for k, n in binding['nodes'].items() if n['id'] == 'draft')
    reply = {'action': {'family': 'text_submit', 'target': target, 'text': 'Exact navigation text',
                       'objective': 'Submit the supplied location'},
             'source_epoch': working['source_epoch'], 'source_phase': 1, 'assessment': 'The document has not loaded.',
             'intended_resource': 'Requested document', 'remaining_criteria': ['Read the loaded document']}
    assert Draft202012Validator(compact['decision_schema']).is_valid(reply)
    result = ui_decision.expand(reply, packet, binding)
    assert result['phase']['subgoal']['family'] == 'text_submit'
    assert Draft202012Validator(canonical).is_valid(result)
