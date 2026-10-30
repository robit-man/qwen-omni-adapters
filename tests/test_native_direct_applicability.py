"""Operation-local intent applicability does not create a native capability."""
import copy

import pytest
from jsonschema import Draft202012Validator
from test_native_atomic_anchor import _source
from test_native_direct_handoff import (
    INTENT,
    OBJECTIVE,
    _canonical_phase,
    _expand,
    _operation,
    _reply,
)
from test_ui_observation_boundary import node

from portal import ui_decision
from portal.ui_task import _grounded_phase_schema, _working_phase_schema
from qwen_omni_adapters.computer_use import VERBS

NO_FAMILY = {'focus', 'select', 'set_checked', 'drag_drop'}
PLAIN = {'set_value', 'type_text', 'select_text', 'switch_window'}
POINTER = {'invoke', 'click', 'double_click', 'right_click'}
MATRIX = {
    'focus': ({}, None), 'select': ({'option': 'Input'}, None),
    'set_checked': ({'checked': True}, None), 'drag_drop': ({'to_node_id': 'draft'}, None),
    'set_value': ({'value': 'Exact — 原文'}, 'text_entry'),
    'type_text': ({'text': 'Exact — 原文'}, 'text_entry'),
    'select_text': ({'start': 0, 'end': 1}, 'select_text'), 'switch_window': ({}, 'switch_window'),
    'key_chord': ({'keys': ['Control_L', 'l']}, 'key_chord'),
    'scroll': ({'direction': 'down'}, 'scroll'),
    **{verb: ({}, 'pointer_or_activation') for verb in POINTER}}


def matrix_case(verb):
    packet = _source()
    state = packet['current_state']['state']
    target = node('matrix', 'textbox', 'Matrix target', sorted(VERBS))
    target['focused'] = True
    state['nodes'].append(target)
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True),
        packet['working_state'], observed_state=state, action_handoffs=True)
    arguments, family = MATRIX[verb]
    step = {'id': 'matrix-' + verb, 'verb': verb,
            'target': {'name': target['name']} if verb in POINTER else {'node_id': target['id']},
            'args': copy.deepcopy(arguments)}
    if verb in POINTER | {'key_chord', 'scroll'}:
        phase = {'kind': 'ui_boundary', 'step': step, 'objective': OBJECTIVE}
    else:
        step['expect'] = [{'kind': 'active_window' if verb == 'switch_window' else 'focused', 'node_id': '$target'}]
        phase = {'kind': 'ui', 'steps': [step], 'final_predicates': [
            {'kind': step['expect'][0]['kind'], 'node_id': target['id']}]}
    return packet, _reply(packet, phase, None if family is None else INTENT)


def test_all_existing_verbs_are_covered_without_new_families():
    assert MATRIX.keys() == VERBS
    assert {family for _, family in MATRIX.values() if family} <= set(ui_decision._NATIVE_ACTION_FAMILIES)


@pytest.mark.parametrize('verb', sorted(MATRIX))
def test_every_offered_verb_retains_its_valid_canonical_route_and_exact_semantics(verb):
    packet, value = matrix_case(verb)
    view, _ = ui_decision.project_direct(packet)
    Draft202012Validator(view['response_schema']).validate(value)
    result = _expand(packet, value)
    assert result['phase'] == _canonical_phase(value)
    assert result['commitment'] == value['commitment']
    family = MATRIX[verb][1]
    if family is None:
        assert 'action_handoff' not in result and 'transition' not in result
    else:
        assert result['action_handoff']['operation_family'] == family
        assert result['action_handoff']['next_operation'] == INTENT['next_operation']
        assert result['action_handoff']['needed_evidence'] == INTENT['needed_evidence']


@pytest.mark.parametrize('verb', sorted(NO_FAMILY))
@pytest.mark.parametrize('intent', [INTENT, {**INTENT, 'transition_kind': 'document_changed'}])
def test_legacy_verb_nonnull_intent_rejects_in_emitted_schema_before_expansion(verb, intent):
    packet, value = matrix_case(verb)
    _operation(value)['immediate_intent'] = copy.deepcopy(intent)
    view, binding = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('verb', sorted(PLAIN))
def test_plain_native_intent_forbids_transition_in_schema_without_silently_discarding_it(verb):
    packet, value = matrix_case(verb)
    _operation(value)['immediate_intent']['transition_kind'] = 'focus_changed_within_scope'
    view, binding = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('verb', sorted(VERBS - NO_FAMILY - POINTER))
@pytest.mark.parametrize('fault', ['null', 'missing'])
def test_supported_operation_has_required_nonnull_local_intent(verb, fault):
    packet, value = matrix_case(verb)
    if fault == 'missing':
        _operation(value).pop('immediate_intent')
    else:
        _operation(value)['immediate_intent'] = None
    view, binding = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('verb', sorted(POINTER | {'key_chord', 'scroll'}))
def test_existing_transition_families_keep_optional_transition_and_host_source_binding(verb):
    packet, value = matrix_case(verb)
    _operation(value)['immediate_intent']['transition_kind'] = 'popup_opened'
    result = _expand(packet, value)
    assert result['transition']['kind'] == 'popup_opened'
    assert result['action_handoff']['expected_transition']['task_progress'] is False


@pytest.mark.parametrize('verb', sorted(POINTER))
def test_named_boundary_remains_nullable_but_cannot_fabricate_or_drop_native_claim(verb):
    packet, value = matrix_case(verb)
    _operation(value)['immediate_intent'] = None
    view, binding = ui_decision.project_direct(packet)
    assert Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)
    _operation(value)['target'] = {'name': 'Genuinely absent visual feature'}
    assert 'action_handoff' not in ui_decision.expand_direct(value, packet, binding)
    _operation(value)['immediate_intent'] = copy.deepcopy(INTENT)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


def test_first_only_annotation_preserves_causal_suffix_and_cannot_borrow_later_intent():
    packet, value = matrix_case('focus')
    suffix = {'id': 'later-write', 'verb': 'set_value', 'target': {'node_id': 'draft'},
              'args': {'value': 'Later original instruction'},
              'expect': [{'kind': 'value', 'node_id': '$target', 'value': 'Later original instruction'}]}
    value['phase']['steps'].append(copy.deepcopy(suffix))
    result = _expand(packet, value)
    assert result['phase']['steps'][1] == suffix and 'action_handoff' not in result
    value['phase']['steps'][1]['immediate_intent'] = copy.deepcopy(INTENT)
    view, _ = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    _operation(value)['immediate_intent'] = copy.deepcopy(INTENT)
    with pytest.raises(ui_decision.DecisionError):
        _expand(packet, value)


def test_shared_current_operation_ref_is_cloned_and_suffix_ref_stays_unmodified():
    packet = _source()
    definitions = packet['response_schema']['$defs']
    definitions['causal_step'] = {'$ref': '#/$defs/grounded_first_step'}
    original = copy.deepcopy(packet)
    view, _ = ui_decision.project_direct(packet)
    projected = view['response_schema']['$defs']
    assert packet == original
    assert projected['causal_step'] == definitions['causal_step']
    assert projected['grounded_first_step'] == definitions['grounded_first_step']
    assert 'immediate_intent' not in str(projected['grounded_first_step'])
    ui = next(p for p in projected['native_working_phase']['oneOf'] if p['properties']['kind']['const'] == 'ui')
    assert ui['properties']['steps']['anyOf'][0]['prefixItems'][0]['$ref'] != '#/$defs/grounded_first_step'


def test_mixed_typed_enum_splits_only_applicability_profiles_and_preserves_other_metadata():
    original = {'type': 'object', 'additionalProperties': False, 'properties': {
        'verb': {'type': 'string', 'enum': sorted(VERBS), 'description': 'Existing typed operation'},
        'target': {'type': 'string'}}, 'required': ['verb', 'target']}
    schema = {'$defs': {'direct_native_intent': ui_decision._direct_intent_schema(transitions=False),
                        'direct_transition_intent': ui_decision._direct_intent_schema(transitions=True)}}
    result = ui_decision._direct_operation_schema(original, schema, 'ui', {})
    assert len(result['oneOf']) == 3
    names = []
    for branch in result['oneOf']:
        assert branch['properties']['verb']['type'] == 'string'
        assert branch['properties']['verb']['description'] == 'Existing typed operation'
        assert branch['properties']['target'] == original['properties']['target']
        assert branch['additionalProperties'] is False
        names += branch['properties']['verb']['enum']
    assert set(names) == VERBS and len(names) == len(VERBS)


def test_referenced_operation_rejects_ignored_sibling_constraints():
    packet = _source()
    phase = next(p for p in packet['response_schema']['$defs']['native_working_phase']['oneOf']
                 if p['properties']['kind']['const'] == 'ui')
    phase['properties']['steps']['anyOf'][0]['prefixItems'][0]['minProperties'] = 1
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.project_direct(packet)


def test_readonly_schema_variants_are_exact_and_do_not_accept_supplied_native_intent():
    packet = _source()
    view, binding = ui_decision.project_direct(packet)
    before = packet['response_schema']['$defs']['native_working_phase']['oneOf']
    after = view['response_schema']['$defs']['native_working_phase']['oneOf']
    for original, projected in zip(before, after, strict=True):
        if original['properties']['kind']['const'] not in {'ui', 'ui_subgoal', 'ui_boundary', 'ui_pointer_recovery'}:
            assert original == projected
    value = _reply(packet, {'kind': 'visual_inspection'})
    assert ui_decision.expand_direct(value, packet, binding) == value
    value['phase']['immediate_intent'] = copy.deepcopy(INTENT)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


@pytest.mark.parametrize('family', ['text_entry', 'text_submit'])
def test_text_subgoal_requires_plain_local_intent_and_preserves_text_and_objective(family):
    packet = _source()
    phase = {'kind': 'ui_subgoal', 'subgoal': {'id': family, 'family': family,
        'target': {'node_id': 'draft'}, 'mode': 'replace', 'text': 'Exact text — 原文', 'objective': OBJECTIVE}}
    value = _reply(packet, phase)
    result = _expand(packet, value)
    assert result['phase'] == phase
    assert result['action_handoff']['operation_family'] == family
    assert result['action_handoff']['next_operation'] == INTENT['next_operation']
    _operation(value)['immediate_intent']['transition_kind'] = 'document_changed'
    view, binding = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


def test_all25_workflow_lengths_and_every_suffix_schema_remain_exact():
    packet = _source()
    view, _ = ui_decision.project_direct(packet)
    def arrays(schema):
        return next(p for p in schema['$defs']['native_working_phase']['oneOf']
                    if p['properties']['kind']['const'] == 'ui')['properties']['steps']['anyOf']
    before, after = arrays(packet['response_schema']), arrays(view['response_schema'])
    assert len(before) == len(after) == 25
    for source, projected in zip(before, after, strict=True):
        assert source['minItems'] == projected['minItems'] and source['maxItems'] == projected['maxItems']
        assert source['prefixItems'][1:] == projected['prefixItems'][1:]
    assert packet['response_schema']['$defs']['causal_step'] == view['response_schema']['$defs']['causal_step']


def test_old_outer_semantic_representation_is_invalid_rather_than_automatically_repaired():
    packet = _source()
    value = _reply(packet)
    value['immediate_intent'] = _operation(value).pop('immediate_intent')
    view, binding = ui_decision.project_direct(packet)
    assert not Draft202012Validator(view['response_schema']).is_valid(value)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.expand_direct(value, packet, binding)


def test_private_model_schema_keeps_existing56kib_ceiling_without_dropping_operations():
    packet = _source()
    packet['response_schema']['description'] = 'x' * (56 * 1024)
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.project_direct(packet)


@pytest.mark.parametrize('fault', ['missing', 'cyclic', 'malformed_escape', 'external', 'array_alias'])
def test_operation_alias_uses_existing_bounded_local_resolver_and_fails_closed(fault):
    packet = _source()
    definitions = packet['response_schema']['$defs']
    bad = {'missing': '#/$defs/not_present', 'cyclic': '#/$defs/grounded_first_step',
           'malformed_escape': '#/$defs/invalid~2escape', 'external': 'https://example.invalid/schema',
           'array_alias': '#/$defs/step/oneOf/0'}[fault]
    definitions['grounded_first_step'] = {'$ref': bad}
    with pytest.raises(ui_decision.DecisionError):
        ui_decision.project_direct(packet)
