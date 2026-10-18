"""Offline candidate views require exact captured sources and isolate ablations."""
import copy
import hashlib
import json

import pytest

from portal import ui_decision
from portal.ui_task import _compact_grounded_targets, _grounded_phase_schema, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError
from runtime import reconstruct_orchestration_candidates as reconstruction
from runtime.verify_orchestration_contracts import digest, load_panel, render_request


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_bundle(root, files):
    manifest = {'files': []}
    for name, rows in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        path.write_bytes(content)
        manifest['files'].append({'path': name, 'bytes': len(content), 'sha256': _sha(content)})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return {row['path']: row['sha256'] for row in manifest['files']}


@pytest.fixture
def captured(tmp_path):
    def make(*, hierarchy=False):
        root = tmp_path / ('hierarchy' if hierarchy else 'plain')
        root.mkdir()
        nodes = [{'id': 'window', 'generation': '1', 'scope_id': 'window', 'role': 'window',
                  'name': 'Files', 'visible': True, 'enabled': True, 'sensitive': False,
                  'operations': ['key_chord', 'switch_window']},
                 {'id': 'folder', 'generation': '7', 'scope_id': 'window', 'role': 'table_cell',
                  'name': 'Inputs', 'visible': True, 'enabled': True, 'sensitive': False,
                  'operations': ['click', 'double_click', 'right_click'], 'focused': False,
                  'text_observation': {'complete': False}}]
        if hierarchy:
            nodes[1]['structure'] = {'parent_id': 'window', 'child_index': 0, 'ancestors': [
                {key: nodes[0][key] for key in ('id', 'generation', 'role', 'name', 'visible', 'enabled')}]}
        raw = {'session_id': 'captured-session', 'scope': {'kind': 'native', 'id': 'window'},
               'nodes': nodes, 'revision': 'a' * 64, 'text': 'Files\nInputs', 'truncated': False}
        state = reconstruction._project_observed(raw)
        working = {'source_epoch': 'e' * 32, 'source_phase': 1, 'recovery_mode': False,
                   'commitment': None, 'observations': {'entries': []}}
        previous = {'goal': 'Read the declared source and save a report.', 'working_state': working,
                    'historical_ledger': [], 'last_result': {'status': 'not_started'}}
        packet = reconstruction._canonical_packet(previous, state, working)
        compact, binding = ui_decision.project(packet)
        proposal = {'current_view': 'Files shows Inputs.', 'needed_evidence': ['The declared source text'],
                    'next_subgoal': 'Read the declared source.', 'next_operation': 'Open Inputs, then the declared source.'}
        compact, binding = ui_decision.bind_subgoal(compact, binding, proposal)
        # The historical solver view had no native hierarchy, even when the raw
        # archive retained it. This lets the two ablations remain independent.
        for control in compact['controls']:
            control.pop('structure', None)
            control.pop('description', None)
        binding['compact_sha256'] = ui_decision._digest(compact)
        action = {'action': {'family': 'pointer_or_activation', 'target': 'c1', 'objective': proposal['next_subgoal']},
                  'source_epoch': working['source_epoch'], 'source_phase': 1, 'assessment': 'Source is unread.',
                  'intended_resource': 'Inputs', 'remaining_criteria': ['Read source', 'Save report']}
        envelope = ui_decision.expand(action, packet, binding)
        offer = {'schema': 'robit.omni.pointer-recovery-proposal.v1', 'reference': 'b' * 32,
                 'source_revision': raw['revision'], 'input_authority': False,
                 'choices': [{'node_id': 'folder', 'verb': verb} for verb in ('click', 'double_click', 'right_click')]}
        deferral = {'status': 'abstained', 'reason': 'intent_policy_abstained',
                    'ui_boundary_receipt': {'input_attempted': False},
                    'effect_receipt': {'effect_state': 'absent'},
                    'intent_selection_receipt': {'objective': proposal['next_subgoal'], 'state_revision': raw['revision']},
                    'pointer_recovery': offer}
        recovery_working = {**working, 'source_phase': 2, 'recovery_mode': True, 'commitment': envelope['commitment']}
        rp = reconstruction._canonical_packet(previous, state, recovery_working, last_result=deferral, offer=offer)
        rc, rb = ui_decision.project(rp)
        for control in rc['controls']:
            control.pop('structure', None)
            control.pop('description', None)
        pointer_view, pointer_schema, _ = ui_decision.pointer_resolution_context(rc, rb)
        pointer_view['response_schema'] = pointer_schema

        def request(context):
            return json.dumps({'messages': [{'role': 'user', 'content': json.dumps(context)}]}, separators=(',', ':'))

        action_wire, pointer_wire = request(compact), request(pointer_view)
        action_hash, pointer_hash = _sha(action_wire.encode()), _sha(pointer_wire.encode())
        content = json.dumps(action, ensure_ascii=False)
        files = {reconstruction._PACKETS: [
            {'at': '2026-10-17T00:00:00+00:00', 'request_body_utf8': action_wire,
             'request_body_sha256': action_hash, 'request_schema_name': 'native_ui_phase'},
            {'at': '2026-10-17T00:00:03+00:00', 'request_body_utf8': pointer_wire,
             'request_body_sha256': pointer_hash, 'request_schema_name': 'native_ui_pointer_resolution'}],
            reconstruction._RESPONSES: [{'request_body_sha256': action_hash, 'request_schema_name': 'native_ui_phase',
                'at': '2026-10-17T00:00:01+00:00',
                'status_code': 200, 'response_capture_truncated': False,
                'response_content_sha256': reconstruction.raw_state_sha256(content),
                'response_json': {'message': {'content': content}}},
                # A poison historical pointer answer must never enter a candidate.
                {'request_body_sha256': pointer_hash, 'request_schema_name': 'native_ui_pointer_resolution',
                 'response_json': {'message': {'content': 'HISTORICAL-POINTER-ANSWER-POISON'}}}],
            reconstruction._OBSERVATIONS: [
                {'at': '2026-10-16T23:59:59+00:00', 'planning': True, 'state': copy.deepcopy(raw)},
                {'at': '2026-10-17T00:00:02+00:00', 'planning': False, 'state': copy.deepcopy(raw)},
                {'at': '2026-10-17T00:00:02.5+00:00', 'planning': True, 'state': copy.deepcopy(raw)}],
            reconstruction._EVENTS: [{'type': 'tool', 'phase': 'complete', 'round': 1, 'tools': [
                {'name': 'native_ui', 'arguments': {'operation': 'intent_boundary', 'subgoal': envelope['phase']['subgoal']},
                 'result': json.dumps(deferral)}]}]}
        hashes = _write_bundle(root, files)
        case_input = {'pointer_view': pointer_view, 'preceding_action_view': compact,
                      'raw_wire_utf8': pointer_wire, 'wire_request': json.loads(pointer_wire)}
        case = {'case_id': 'SYNTHETIC.010', 'input': case_input, 'input_sha256': digest(case_input),
                'oracle': {'secret': 'ORACLE-POISON'}, 'source': {'file_sha256': hashes[reconstruction._PACKETS],
                    'jsonl_row_zero_based': 1, 'preceding_action_row_zero_based': 0,
                    'request_body_sha256': pointer_hash, 'preceding_action_request_body_sha256': action_hash,
                    'actual_baseline_reply': 'BASELINE-ANSWER-POISON'}}
        return root, case, files
    return make


def test_handoff_is_compiled_from_actual_reply_with_no_oracle_or_pointer_answer(captured):
    root, case, _ = captured()
    before = copy.deepcopy(case)
    cell = reconstruction.reconstruct_case(case, archive_root=root)
    assert cell['status'] == 'constructed', cell
    claim = cell['view']['action_handoff']
    assert claim['next_operation'] == case['input']['preceding_action_view']['planning_handoff']['proposal']['next_operation']
    assert claim['target']['node_id'] == 'folder' and claim['target']['generation'] == '7'
    assert claim['input_authority'] is False and cell['whole_task_success_credit'] is False
    assert cell['private_binding_provenance'] == 'counterfactual_reconstruction_not_retained_original_binding'
    assert cell['provenance']['observed_state_rows_zero_based'] == [0, 2]
    assert cell['provenance']['pointer_planning_state_row_zero_based'] == 2
    assert cell['provenance']['policy_observation_row_zero_based'] == 1
    assert cell['provenance']['actual_action_reply_sha256'] != cell['provenance']['captured_reply_json_string_sha256']
    assert 'POISON' not in json.dumps(cell)
    assert case == before
    retained = {key: value for key, value in cell['view'].items()
                if key not in ('action_handoff', 'action_handoff_revalidation')}
    assert retained == case['input']['pointer_view']


def test_hierarchy_only_uses_observed_structure_and_never_adds_handoff(captured):
    root, case, _ = captured(hierarchy=True)
    cell = reconstruction.reconstruct_case(case, 'hierarchy_only', archive_root=root)
    assert cell['status'] == 'constructed', cell
    assert 'action_handoff' not in cell['view']
    structure = cell['view']['choices']['double_click_c1']['control']['structure']
    assert structure['parent_id'] == 'window' and structure['parent_ref'] == 'c0'
    assert structure['ancestors'][0]['id'] == 'window'
    handoff = reconstruction.reconstruct_case(case, 'handoff_only', archive_root=root)
    assert handoff['status'] == 'constructed'
    assert 'structure' not in handoff['view']['choices']['double_click_c1']['control']


def test_missing_hierarchy_is_explicit_and_not_inferred_from_flat_node_order(captured):
    root, case, _ = captured()
    cell = reconstruction.reconstruct_case(case, 'hierarchy_only', archive_root=root)
    assert cell['status'] == 'unconstructible'
    assert cell['unconstructible_reason'] == 'missing_observed_native_hierarchy'
    assert cell['view'] is None


@pytest.mark.parametrize('fault', ['wire_hash', 'reply_hash', 'reply_epoch', 'raw_generation',
                                  'raw_revision', 'current_generation', 'policy_generation',
                                  'pointer_time', 'missing_policy', 'deferral_target',
                                  'deferral_input', 'missing_proposal'])
def test_changed_source_relationship_never_produces_candidate(captured, fault):
    root, case, files = captured()
    if fault == 'wire_hash':
        files[reconstruction._PACKETS][0]['request_body_sha256'] = '0' * 64
    elif fault.startswith('reply_'):
        row = files[reconstruction._RESPONSES][0]
        if fault == 'reply_hash':
            row['response_content_sha256'] = '0' * 64
        else:
            reply = json.loads(row['response_json']['message']['content'])
            reply['source_epoch'] = 'foreign'
            row['response_json']['message']['content'] = json.dumps(reply)
            row['response_content_sha256'] = reconstruction.raw_state_sha256(row['response_json']['message']['content'])
    elif fault.startswith('raw_'):
        for row in files[reconstruction._OBSERVATIONS]:
            if fault == 'raw_generation':
                row['state']['nodes'][1]['generation'] = '8'
            else:
                row['state']['revision'] = 'different'
    elif fault == 'current_generation':
        files[reconstruction._OBSERVATIONS][2]['state']['nodes'][1]['generation'] = '8'
    elif fault == 'policy_generation':
        files[reconstruction._OBSERVATIONS][1]['state']['nodes'][1]['generation'] = '8'
    elif fault == 'pointer_time':
        files[reconstruction._PACKETS][1]['at'] = '2026-10-16T23:59:00+00:00'
    elif fault == 'missing_policy':
        files[reconstruction._OBSERVATIONS].pop(1)
    elif fault.startswith('deferral_'):
        row = files[reconstruction._EVENTS][0]['tools'][0]
        value = json.loads(row['result'])
        if fault == 'deferral_target':
            value['pointer_recovery']['choices'][0]['node_id'] = 'unobserved'
        else:
            value['ui_boundary_receipt']['input_attempted'] = True
        row['result'] = json.dumps(value)
    else:
        case['input']['preceding_action_view'].pop('planning_handoff')
        case['input_sha256'] = digest(case['input'])
    hashes = _write_bundle(root, files)
    case['source']['file_sha256'] = hashes[reconstruction._PACKETS]
    cell = reconstruction.reconstruct_case(case, archive_root=root)
    assert cell['status'] == 'unconstructible'
    assert cell['view'] is None and cell['unconstructible_reason']


def test_file_tampering_is_detected_even_if_request_row_is_unchanged(captured):
    root, case, _ = captured()
    with (root / reconstruction._OBSERVATIONS).open('a') as file:
        file.write('{}\n')
    cell = reconstruction.reconstruct_case(case, archive_root=root)
    assert cell['unconstructible_reason'] == 'archive_source_integrity_failed'


def test_negative_derivative_view_is_preserved_and_never_overwritten_by_positive_archive(captured):
    root, case, _ = captured()
    case['input']['pointer_view']['choices']['double_click_c1']['control']['name'] = 'Remove'
    case['source']['derived_from_case_id'] = 'SYNTHETIC.POSITIVE'
    case['input_sha256'] = digest(case['input'])
    before = copy.deepcopy(case)
    cell = reconstruction.reconstruct_case(case, archive_root=root)
    assert cell['unconstructible_reason'] == 'controlled_derivative_has_no_exact_captured_source'
    assert cell['frozen_pointer_view_sha256'] == digest(case['input']['pointer_view'])
    assert case == before and cell['view'] is None


def test_actual_target_schema_deduplication_keeps_every_recovery_choice():
    from test_native_pointer_recovery import recovery_packet

    packet, offer = recovery_packet()
    identity = offer['choices'][0]['node_id']
    offer['choices'].extend({'node_id': identity, 'verb': verb} for verb in ('click', 'right_click'))
    packet['response_schema'] = _working_phase_schema({'native_ui'},
        _grounded_phase_schema({'native_ui'}, packet['current_state'], intent_choice=True),
        packet['working_state'], pointer_recovery=offer)
    packet['last_result']['intent_selection_receipt'] = {'objective': 'Open the observed source.'}
    compact, binding = ui_decision.project(packet)
    before = ui_decision.pointer_resolution_context(compact, binding)
    packet['response_schema'] = _compact_grounded_targets(packet['response_schema'])
    recovery = next(branch for branch in packet['response_schema']['$defs']['native_working_phase']['oneOf']
                    if branch['properties']['kind']['const'] == 'ui_pointer_recovery')
    assert all('$ref' in step['properties']['target'] for step in recovery['properties']['step']['oneOf'])
    compact, binding = ui_decision.project(packet)
    assert ui_decision.pointer_resolution_context(compact, binding) == before
    original = recovery['properties']['step']['oneOf'][0]['properties']['target']
    packet['response_schema']['$defs']['recovery/target~alias'] = copy.deepcopy(original)
    recovery['properties']['step']['oneOf'][0]['properties']['target'] = {
        '$ref': '#/$defs/recovery~1target~0alias'}
    compact, binding = ui_decision.project(packet)
    assert ui_decision.pointer_resolution_context(compact, binding) == before


@pytest.mark.parametrize('reference', ['https://example.invalid/target.json', '#/$defs/missing',
                                     '#/$defs/cycle', '#/$defs/bad~', '#/$defs/bad~2', '#/$defs/nonobject'])
def test_malformed_or_external_local_target_alias_fails_closed_without_keyerror(reference):
    from test_native_pointer_recovery import recovery_packet

    packet, _ = recovery_packet()
    packet['response_schema'] = _compact_grounded_targets(packet['response_schema'])
    definitions = packet['response_schema']['$defs']
    definitions.update(cycle={'$ref': '#/$defs/cycle'}, nonobject='not a target schema')
    recovery = next(branch for branch in definitions['native_working_phase']['oneOf']
                    if branch['properties']['kind']['const'] == 'ui_pointer_recovery')
    recovery['properties']['step']['oneOf'][0]['properties']['target'] = {'$ref': reference}
    with pytest.raises(ComputerUseError):
        ui_decision.project(packet)


def test_missing_archive_still_retains_every_panel_cell(tmp_path):
    panel = reconstruction.reconstruct_panel(archive_root=tmp_path)
    assert panel['planned_cells'] == len(panel['cells']) == 22
    assert panel['constructed_cells'] == 0
    assert all(cell['status'] == 'unconstructible' and cell['view'] is None for cell in panel['cells'])


@pytest.mark.skipif(not reconstruction.ARCHIVE.exists(), reason='Actual preserved archive is not installed')
def test_actual_frozen_panel_preserves_every_missing_cell_and_renders_actual_candidate_schema():
    panel = reconstruction.reconstruct_panel()
    assert panel['planned_cells'] == len(panel['cells']) == 22
    assert panel['constructed_cells'] == 6
    assert len({(cell['case_id'], cell['condition']) for cell in panel['cells']}) == 22
    _, cases = load_panel()
    for cell in panel['cells']:
        case = cases[cell['case_id']]
        if cell['condition'] == 'hierarchy_only':
            assert cell['status'] == 'unconstructible'
        if '.NEGATIVE.' in cell['case_id']:
            assert cell['unconstructible_reason'] == 'controlled_derivative_has_no_exact_captured_source'
        if cell['status'] == 'constructed':
            rendered = render_request(case, 'robit/ornith-1.5-omni-audio-bridge:q4km', condition='candidate',
                                      replacement_view=cell['view'], candidate_sources=cell['candidate_sources'])
            assert rendered['request']['response_format']['json_schema']['schema'] == cell['response_schema']
            assert rendered['execution_status'] == 'not_executed'
    assert any(cell['status'] == 'constructed' for cell in panel['cells'])
