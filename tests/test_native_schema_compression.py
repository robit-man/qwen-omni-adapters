"""Real Firefox capability loss at the grounded/working-schema boundary."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_native_compact_decision import reply

from portal import ui_decision, ui_task

FIXTURE = Path(__file__).parent / 'fixtures/native_ui/firefox-address-schema-budget.json'


def packet():
    observation = json.loads(FIXTURE.read_text())
    goal = 'Read the official documentation and save a sourced report.'
    working = ui_task._NativeWorkingState(goal).view(10, 0)
    allowed = ['native_ui', 'memory_write', 'memory_search']
    grounded = ui_task._grounded_phase_schema(allowed, observation, intent_choice=True)
    return {'goal': goal, 'current_state': observation, 'working_state': working,
        'ledger': [], 'last_result': None, 'remaining_seconds': 90,
        'response_schema': ui_task._working_phase_schema(allowed, grounded, working)}


def test_captured_firefox_keeps_actions_after_working_state_wrap(monkeypatch):
    # The original wrapper discards all UI kinds at 58,434 bytes, despite the
    # address bar advertising typing and Return. Exercise that exact state.
    with monkeypatch.context() as m:
        m.setattr(ui_task, '_compact_grounded_targets', lambda schema: schema)
        original = packet()
    kinds = {b['properties']['kind']['const'] for b in original['response_schema']['$defs']['native_working_phase']['oneOf']}
    assert not {'ui', 'ui_boundary', 'ui_subgoal'} & kinds
    fixed = packet()
    assert len(json.dumps(fixed['response_schema'], ensure_ascii=False, separators=(',', ':')).encode()) <= 56 * 1024
    compact, binding = ui_decision.project(fixed)
    address = next(n for n in compact['controls'] if n['role'] == 'combobox' and n.get('focused'))
    assert {'text_entry', 'text_submit', 'key_chord'} <= set(address['families'])
    families = {b['properties']['family']['const'] for b in compact['decision_schema']['properties']['action']['oneOf']}
    assert {'switch_window', 'scroll', 'visual_pointer', 'pointer_or_activation'} <= families
    proposal = ui_decision.expand(reply(fixed, {'family': 'text_submit', 'target': address['ref'],
        'text': 'https://docs.python.org/3.10/library/pathlib.html#pathlib.Path.mkdir'}), fixed, binding)
    assert Draft202012Validator(fixed['response_schema']).is_valid(proposal)


@pytest.mark.parametrize('fault', ['none', 'stale_epoch', 'unknown_target', 'missing_text', 'extra_field', 'wrong_mode'])
def test_factoring_preserves_validation_of_real_proposal(monkeypatch, fault):
    fixed = packet()
    compact, binding = ui_decision.project(fixed)
    target = next(n['ref'] for n in compact['controls'] if 'text_submit' in n['families'])
    proposal = ui_decision.expand(reply(fixed, {'family': 'text_submit', 'target': target, 'text': 'Literal text'}), fixed, binding)
    if fault == 'stale_epoch':
        proposal['assessment']['source_epoch'] = 'another-task'
    elif fault == 'unknown_target':
        proposal['phase']['subgoal']['target']['node_id'] = 'unobserved'
    elif fault == 'missing_text':
        del proposal['phase']['subgoal']['text']
    elif fault == 'extra_field':
        proposal['phase']['subgoal']['authorized'] = True
    elif fault == 'wrong_mode':
        proposal['phase']['subgoal']['mode'] = 'append'
    # Reference is the exact unfactored contract, with the size guard disabled
    # only for this validator comparison. Production limit remains unchanged.
    with monkeypatch.context() as m:
        m.setattr(ui_task, '_MAX_GROUNDED_SCHEMA_BYTES', 1_000_000)
        reference = ui_task._working_phase_schema(['native_ui', 'memory_write', 'memory_search'],
            ui_task._grounded_phase_schema(['native_ui', 'memory_write', 'memory_search'],
                fixed['current_state'], intent_choice=True), fixed['working_state'])
    assert Draft202012Validator(reference).is_valid(proposal) == (fault == 'none')
    assert Draft202012Validator(fixed['response_schema']).is_valid(proposal) == (fault == 'none')


def test_sharing_preserves_source_property_order_and_existing_definitions():
    target = {'type': 'object', 'properties': {'role': {'const': 'button'}, 'name': {'const': 'Save'}},
              'required': ['role'], 'additionalProperties': False}
    schema = {'$defs': {'shared_grounded_target_1': {'const': 'existing'}}, 'anyOf': [
        {'properties': {'target': copy.deepcopy(target)}}, {'properties': {'target': copy.deepcopy(target)}}]}
    original = copy.deepcopy(schema)
    result = ui_task._compact_grounded_targets(schema)
    assert schema == original
    assert result['$defs']['shared_grounded_target_1'] == {'const': 'existing'}
    ref = result['anyOf'][0]['properties']['target']['$ref'].split('/')[-1]
    assert list(result['$defs'][ref]['properties']) == ['role', 'name']
    assert result['anyOf'][1]['properties']['target'] == {'$ref': '#/$defs/' + ref}
    assert ui_task._compact_grounded_targets(result) == result


def test_oversized_unknown_state_still_grants_no_ui():
    p = packet()
    p['current_state']['state']['nodes'] *= 3
    grounded = ui_task._grounded_phase_schema(['native_ui'], p['current_state'], intent_choice=True)
    assert not {'ui', 'ui_boundary', 'ui_subgoal'} & {b['properties']['kind']['const'] for b in grounded['oneOf']}
