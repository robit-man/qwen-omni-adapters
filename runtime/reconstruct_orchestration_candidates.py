"""Offline, explicitly counterfactual reconstruction of archived candidate views.

Only captured planner proposals, their actual action replies, no-input deferrals,
and exact observed native states are inputs. Historical pointer answers and panel
oracles are never consulted. A missing source is an unconstructible cell, not an
excuse to infer native identities or omit a comparison from its denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from portal import ui_decision
from portal.app import _model_ui_observation
from portal.ui_task import _grounded_phase_schema, _working_phase_schema
from qwen_omni_adapters.computer_use import ComputerUseError
from runtime.verify_orchestration_contracts import PANEL, ROOT, digest, load_panel

ARCHIVE = Path('/home/roko/Documents/qwen-omni-research/2026-10-17/observation-whole-task-local')
CONDITIONS = ('handoff_only', 'hierarchy_only')
_PACKETS = 'live-frontend/adapter-planner-packets.jsonl'
_RESPONSES = 'live-frontend/adapter-phase-responses.jsonl'
_OBSERVATIONS = 'live-frontend/controller-observations.jsonl'
_EVENTS = 'live-case/events.ndjson'
_SOURCES = ('portal/ui_decision.py', 'portal/app.py', 'portal/ui_task.py',
            'runtime/reconstruct_orchestration_candidates.py')
_MAX_BYTES = 16 * 1024 * 1024


class ReconstructionError(ValueError):
    """A source relationship is absent, changed, or cannot be established."""


def _require(condition, reason):
    if not condition:
        raise ReconstructionError(reason)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def raw_state_sha256(state):
    """The archived projector used ASCII-escaped JSON for this specific hash."""
    return _sha(json.dumps(state, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode())


class ArchiveBundle:
    """Read-only archive with measured file provenance and bounded JSON input."""

    def __init__(self, root=ARCHIVE):
        self.root = Path(root)
        content = (self.root / 'manifest.json').read_bytes()
        _require(len(content) <= _MAX_BYTES, 'archive_manifest_exceeds_bound')
        manifest = json.loads(content)
        self.manifest_sha256 = _sha(content)
        self.files = {row['path']: row for row in manifest['files']}
        self.measured = {}
        self._rows = {}

    def rows(self, name):
        if name not in self._rows:
            _require(name in {_PACKETS, _RESPONSES, _OBSERVATIONS, _EVENTS}, 'unsupported_archive_source')
            expected = self.files.get(name)
            _require(isinstance(expected, dict), 'source_missing_from_archive_manifest')
            content = (self.root / name).read_bytes()
            _require(len(content) <= _MAX_BYTES, 'archive_source_exceeds_bound')
            measured = _sha(content)
            _require(measured == expected['sha256'] and len(content) == expected['bytes'],
                     'archive_source_integrity_failed')
            self.measured[name] = measured
            self._rows[name] = [json.loads(line) for line in content.splitlines() if line.strip()]
        return self._rows[name]


def _wire_context(row, expected_hash):
    raw = row['request_body_utf8']
    _require(_sha(raw.encode()) == expected_hash == row['request_body_sha256'],
             'wire_request_binding_changed')
    return json.loads(json.loads(raw)['messages'][-1]['content'])


def _captured_time(value):
    _require(isinstance(value, str), 'missing_captured_time_binding')
    timestamp = datetime.fromisoformat(value)
    _require(timestamp.tzinfo is not None, 'missing_captured_time_timezone')
    return timestamp


def _project_observed(raw):
    references = tuple(node['id'] for node in raw['nodes'] if node.get('role') == 'window'
        and 'switch_window' in node.get('operations', []) and node.get('visible') is True
        and node.get('enabled') is True and node.get('sensitive') is False)
    return _model_ui_observation(copy.deepcopy(raw), references=references)


def _canonical_packet(previous, state, working, *, last_result=None, offer=None):
    # The candidate private binding was not archived. This is a reconstruction
    # using current pure compiler functions, never adoption of old input authority.
    allowed = ('native_ui',)
    phase = _grounded_phase_schema(allowed, {'state': state}, intent_choice=True,
                                   effect_contracts=True)
    schema = _working_phase_schema(allowed, phase, working, pointer_recovery=offer,
                                   observed_state=state, action_handoffs=True)
    return {'goal': previous['goal'], 'current_state': {'state': copy.deepcopy(state)},
        'working_state': copy.deepcopy(working), 'response_schema': schema,
        'ledger': copy.deepcopy(previous['historical_ledger']),
        'last_result': copy.deepcopy(previous['last_result'] if last_result is None else last_result)}


def _source_material(case, archive):
    source, frozen = case['source'], case['input']
    rows = archive.rows(_PACKETS)
    _require(archive.measured[_PACKETS] == source['file_sha256'], 'panel_archive_file_binding_changed')
    pointer_index = source['jsonl_row_zero_based']
    action_index = source['preceding_action_row_zero_based']
    _require(type(pointer_index) is int and type(action_index) is int
             and 0 <= action_index < pointer_index < len(rows), 'invalid_source_row_binding')
    pointer = _wire_context(rows[pointer_index], source['request_body_sha256'])
    _require(pointer == frozen['pointer_view'], 'controlled_derivative_has_no_exact_captured_source')
    previous = _wire_context(rows[action_index], source['preceding_action_request_body_sha256'])
    _require(previous == frozen['preceding_action_view'], 'preceding_action_context_changed')
    _require(rows[action_index]['request_schema_name'] == 'native_ui_phase', 'not_a_native_action_request')
    action_time, pointer_time = (_captured_time(rows[index].get('at')) for index in (action_index, pointer_index))
    _require(action_time < pointer_time, 'captured_action_pointer_order_changed')

    response_rows = [(index, row) for index, row in enumerate(archive.rows(_RESPONSES))
        if row.get('request_body_sha256') == source['preceding_action_request_body_sha256']
        and row.get('request_schema_name') == 'native_ui_phase']
    _require(len(response_rows) == 1, 'missing_or_ambiguous_actual_action_reply')
    response_index, response = response_rows[0]
    content = response['response_json']['message']['content']
    # The capture hashes JSON values, including the quotes/escapes of a string;
    # it does not hash the decoded message's UTF-8 bytes. Measure both explicitly.
    captured_content_hash = _sha(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode())
    _require(response.get('status_code') == 200 and response.get('response_capture_truncated') is False
             and captured_content_hash == response['response_content_sha256'], 'actual_action_reply_integrity_failed')
    _require(action_time <= _captured_time(response.get('at')) < pointer_time, 'actual_action_reply_time_changed')
    reply = json.loads(content)
    Draft202012Validator(previous['decision_schema']).validate(reply)

    projection = previous['observation']['model_projection']
    matching = [(index, row['state']) for index, row in enumerate(archive.rows(_OBSERVATIONS))
        if row.get('planning') is True and row.get('state', {}).get('revision') == projection['source_revision']
        and raw_state_sha256(row['state']) == projection['raw_source_sha256']]
    _require(bool(matching), 'missing_exact_observed_source_state')
    raw = matching[0][1]
    current_rows = [(index, row) for index, row in enumerate(archive.rows(_OBSERVATIONS))
        if row.get('planning') is True and action_time < _captured_time(row.get('at')) < pointer_time]
    _require(bool(current_rows), 'missing_captured_pointer_planning_state')
    current_index, current_row = max(current_rows, key=lambda item: _captured_time(item[1]['at']))
    _require(raw_state_sha256(current_row['state']) == projection['raw_source_sha256'],
             'pointer_planning_source_changed_after_action')
    state = _project_observed(raw)
    # cN denotes the Nth *projected* node, never the raw accessibility-tree row.
    for control in previous['controls']:
        reference = control['ref']
        _require(reference.startswith('c') and reference[1:].isdigit(), 'invalid_archived_control_alias')
        index = int(reference[1:])
        _require(index < len(state['nodes']), 'archived_control_alias_not_observed')
        node = state['nodes'][index]
        keys = ('role', 'name', 'value', 'focused', 'checked', 'selected', 'text_observation', 'text_selection')
        _require(all(node.get(key) == control.get(key) for key in keys), 'archived_control_alias_changed')

    packet = _canonical_packet(previous, state, previous['working_state'])
    compact, binding = ui_decision.project(packet)
    compact, binding = ui_decision.bind_subgoal(compact, binding, previous['planning_handoff']['proposal'])
    expanded = ui_decision.expand(reply, packet, binding)
    _require(expanded['phase']['kind'] == 'ui_subgoal'
             and expanded['phase']['subgoal']['family'] == 'pointer_or_activation', 'unsupported_original_action_family')

    deferrals = []
    for event_index, event in enumerate(archive.rows(_EVENTS)):
        if event.get('type') != 'tool' or event.get('phase') != 'complete' or event.get('round') != reply['source_phase']:
            continue
        for tool_index, tool in enumerate(event.get('tools', [])):
            if (tool.get('name') == 'native_ui' and tool.get('arguments', {}).get('operation') == 'intent_boundary'
                    and tool['arguments'].get('subgoal') == {
                        key: value for key, value in expanded['phase']['subgoal'].items()}):
                deferrals.append((event_index, tool_index, json.loads(tool['result'])))
    _require(len(deferrals) == 1, 'missing_or_ambiguous_actual_policy_deferral')
    event_index, tool_index, result = deferrals[0]
    _require(result.get('status') == 'abstained' and result.get('reason') == 'intent_policy_abstained'
        and result.get('ui_boundary_receipt', {}).get('input_attempted') is False
        and result.get('effect_receipt', {}).get('effect_state') == 'absent', 'not_a_no_input_policy_deferral')
    _require(result['intent_selection_receipt']['objective'] == reply['action']['objective'], 'deferral_objective_changed')
    offer = result['pointer_recovery']
    _require(offer.get('source_revision') == result['intent_selection_receipt']['state_revision'],
             'deferral_source_revision_changed')
    source_target = expanded['phase']['subgoal']['target']['node_id']
    _require(all(choice['node_id'] == source_target for choice in offer['choices']), 'deferral_target_changed')
    policy_rows = [(index, row) for index, row in enumerate(archive.rows(_OBSERVATIONS))
        if row.get('planning') is False and row.get('state', {}).get('revision') == offer['source_revision']
        and action_time < _captured_time(row.get('at')) < pointer_time]
    _require(bool(policy_rows), 'missing_exact_policy_observation')
    policy_index, policy_row = max(policy_rows, key=lambda item: _captured_time(item[1]['at']))
    policy_state = policy_row['state']
    _require(policy_state['session_id'] == raw['session_id'] and policy_state['scope'] == raw['scope'],
             'policy_observation_session_or_scope_changed')
    policy_targets = [node for node in policy_state['nodes'] if node.get('id') == source_target]
    _require(len(policy_targets) == 1, 'policy_target_not_uniquely_observed')
    ui_decision.validate_action_handoff(expanded['action_handoff'], goal=previous['goal'], target=policy_targets[0])
    return previous, raw, state, reply, expanded, result, {
        'action_request_row_zero_based': action_index, 'pointer_request_row_zero_based': pointer_index,
        'action_response_row_zero_based': response_index, 'actual_action_reply_sha256': _sha(content.encode()),
        'captured_reply_json_string_sha256': captured_content_hash,
        'observed_state_rows_zero_based': [index for index, _ in matching],
        'pointer_planning_state_row_zero_based': current_index,
        'policy_observation_row_zero_based': policy_index,
        'policy_state_sha256': raw_state_sha256(policy_state),
        'raw_state_sha256': projection['raw_source_sha256'], 'source_revision': raw['revision'],
        'source_epoch': reply['source_epoch'], 'source_phase': reply['source_phase'],
        'proposal_sha256': digest(previous['planning_handoff']['proposal']),
        'deferral_event_row_zero_based': event_index, 'deferral_tool_index': tool_index,
        'deferral_result_sha256': digest(result)}


def reconstruct_case(case, condition='handoff_only', *, archive_root=ARCHIVE, _archive=None):
    """Return one candidate or an explicit unconstructible cell; never execute it."""
    result = {'schema': 'robit.omni.counterfactual-orchestration-candidate.v1',
        'case_id': case.get('case_id'), 'condition': condition, 'status': 'unconstructible',
        'view': None, 'response_schema': None, 'unconstructible_reason': None,
        'input_sha256': case.get('input_sha256'), 'frozen_pointer_view_sha256': digest(case['input']['pointer_view']),
        'private_binding_provenance': 'counterfactual_reconstruction_not_retained_original_binding',
        'input_authority': False, 'whole_task_success_credit': False, 'execution_status': 'not_executed',
        'candidate_sources': {name: _sha((ROOT / name).read_bytes()) for name in _SOURCES}, 'provenance': {}}
    try:
        _require(condition in CONDITIONS, 'unknown_ablation_condition')
        _require(digest(case['input']) == case['input_sha256'], 'frozen_case_input_changed')
        archive = _archive or ArchiveBundle(archive_root)
        previous, raw, state, reply, expanded, deferral, provenance = _source_material(case, archive)
        result['provenance'] = {**provenance, 'archive_manifest_sha256': archive.manifest_sha256,
                                'archive_files': copy.deepcopy(archive.measured)}
        if condition == 'hierarchy_only':
            _require(any(isinstance(node.get('structure'), dict) for node in raw['nodes']),
                     'missing_observed_native_hierarchy')
        working = copy.deepcopy(previous['working_state'])
        working.update(source_phase=reply['source_phase'] + 1, recovery_mode=True,
                       commitment=copy.deepcopy(expanded['commitment']))
        last = copy.deepcopy(deferral)
        if condition == 'handoff_only':
            last['action_handoff'] = copy.deepcopy(expanded['action_handoff'])
        packet = _canonical_packet(previous, state, working, last_result=last, offer=last['pointer_recovery'])
        compact, binding = ui_decision.project(packet)
        if condition == 'handoff_only':
            # Isolate the retained intent claim from the separately evaluated
            # native hierarchy/description projection. Source nodes stay intact.
            for control in compact['controls']:
                control.pop('description', None)
                control.pop('structure', None)
            binding['compact_sha256'] = ui_decision._digest(compact)
        view, schema, _ = ui_decision.pointer_resolution_context(compact, binding)
        view['response_schema'] = copy.deepcopy(schema)
        baseline = case['input']['pointer_view']
        if condition == 'handoff_only':
            _require('action_handoff' in view, 'candidate_handoff_not_source_bound')
            _require({key: value for key, value in view.items() if key not in {
                'action_handoff', 'action_handoff_revalidation'}} == baseline, 'handoff_ablation_changed_baseline_fields')
        else:
            _require('action_handoff' not in view, 'hierarchy_ablation_leaked_handoff')
            stripped = copy.deepcopy(view)
            for choice in stripped['choices'].values():
                choice.get('control', {}).pop('structure', None)
                choice.get('control', {}).pop('description', None)
            _require(stripped == baseline, 'hierarchy_ablation_changed_baseline_fields')
        result.update(status='constructed', view=view, response_schema=schema,
                      candidate_view_sha256=digest(view))
        result['provenance']['working_state_derivation'] = 'original_state_plus_immediately_deferred_phase_and_actual_proposal_commitment'
    except ReconstructionError as exc:
        result['unconstructible_reason'] = str(exc)
    except (OSError, ValueError, TypeError, KeyError, IndexError, ValidationError, ComputerUseError) as exc:
        result['unconstructible_reason'] = 'source_contract_unconstructible'
        result['source_error_type'] = type(exc).__name__
    return result


def reconstruct_panel(panel=PANEL, *, archive_root=ARCHIVE):
    """Keep every case/ablation cell, including absent-source cells, in output."""
    manifest, cases = load_panel(panel)
    try:
        archive = ArchiveBundle(archive_root)
    except (OSError, ValueError, TypeError, KeyError):
        archive = None
    cells = [reconstruct_case(case, condition, archive_root=archive_root, _archive=archive)
             for case in cases.values() for condition in CONDITIONS]
    return {'schema': 'robit.omni.counterfactual-orchestration-panel.v1',
        'panel_case_ids': manifest['case_ids'], 'conditions': list(CONDITIONS),
        'planned_cells': len(cases) * len(CONDITIONS), 'cells': cells,
        'constructed_cells': sum(cell['status'] == 'constructed' for cell in cells),
        'execution_status': 'not_executed', 'whole_task_success_credit': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(reconstruct_panel(args.panel, archive_root=args.archive_root),
                      indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
