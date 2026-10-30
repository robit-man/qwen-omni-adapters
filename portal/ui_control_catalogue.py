"""Bounded task-owned discovery metadata; no input or executor authority."""
from __future__ import annotations

import copy
import hashlib
import json

from qwen_omni_adapters.computer_use import ComputerUseError

CATALOGUE_SCHEMA = 'robit.omni.native-control-catalogue.v1'
EXPANSION_SCHEMA = 'robit.omni.native-control-expansion.v1'


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _excerpt(value, limit):
    result = value.encode()[:limit].decode(errors='ignore')
    return result, result != value


def _identity(node):
    return {key: node[key] for key in ('id', 'generation', 'scope_id', 'role', 'name')}


class NativeControlCatalogue:
    MAX_BYTES = 4096
    MAX_ENTRIES = 16
    MAX_RECORDS = 400
    MAX_EXPANSIONS = 40

    def __init__(self, task_epoch):
        if not isinstance(task_epoch, str) or not task_epoch:
            raise ComputerUseError('Control catalogue requires a task epoch')
        self._epoch = task_epoch
        self._session = None
        self._state = None
        self._records = []
        self._offers = {}
        self._selection = (None, 0)
        self._pending = None
        self._seen = set()
        self._closed = False

    def clear(self):
        self._state, self._pending = None, None
        self._records, self._offers, self._seen = [], {}, set()
        self._session, self._selection = None, (None, 0)
        self._closed = True

    def _source(self, state=None):
        state = self._state if state is None else state
        return {'task_epoch_sha256': _digest(self._epoch),
            'session_sha256': _digest(state['session_id']),
            'scope_id': state['scope']['id'], 'scope_sha256': _digest(state['scope']),
            'revision': state['revision'], 'raw_source_sha256': _digest(state)}

    @staticmethod
    def _eligible(node, scope):
        return (node.get('scope_id') == scope and node.get('visible') is True
            and node.get('enabled') is True and node.get('sensitive') is False
            and isinstance(node.get('operations'), list) and bool(node['operations']))

    def capture(self, state):
        """Retain one fresh host snapshot and consume at most one requested hint."""
        if self._closed:
            raise ComputerUseError('Control catalogue is closed')
        if (not isinstance(state, dict) or not isinstance(state.get('session_id'), str)
                or not state['session_id'] or not isinstance(state.get('revision'), str)
                or not state['revision'] or not isinstance(state.get('scope'), dict)
                or state['scope'].get('kind') not in {'desktop', 'native'}
                or not isinstance(state['scope'].get('id'), str) or not state['scope']['id']
                or not isinstance(state.get('nodes'), list)):
            raise ComputerUseError('Invalid bounded native catalogue source')
        if self._session is not None and state['session_id'] != self._session:
            raise ComputerUseError('Control catalogue crossed its native session')
        if len(state['nodes']) > self.MAX_RECORDS:
            # Planning may append inactive roots to a bounded native traversal.
            # Catalogue overflow cannot suppress an otherwise valid native UI.
            source = self._source(state)
            pending, self._pending = self._pending, None
            self._session, self._state = state['session_id'], None
            self._records, self._offers, self._selection = [], {}, (None, 0)
            view = {'schema': CATALOGUE_SCHEMA, 'status': 'unavailable',
                'error': 'native_control_catalogue_source_limit', 'source': source,
                'source_node_count': len(state['nodes']), 'maximum_source_records': self.MAX_RECORDS,
                'entries': [], 'input_authority': False, 'task_progress': False,
                'absence_claims_supported': False}
            receipt = ({'schema': EXPANSION_SCHEMA, 'status': 'unavailable',
                'original_source': pending['source'], 'fresh_source': source,
                'reference': pending['reference'], 'fresh_identity_verified': False,
                'input_authority': False, 'task_progress': False} if pending else None)
            return view, (), receipt
        identities = set()
        for node in state['nodes']:
            if (not isinstance(node, dict) or any(not isinstance(node.get(key), str)
                    or len(node[key]) > 512 or (not node[key] and key != 'name')
                    for key in ('id', 'generation', 'scope_id', 'role', 'name'))
                    or node['id'] in identities
                    or any(type(node.get(key)) is not bool for key in ('visible', 'enabled', 'sensitive'))
                    or not isinstance(node.get('operations'), list)
                    or any(not isinstance(verb, str) or len(verb) > 64 for verb in node['operations'])):
                raise ComputerUseError('Invalid native catalogue node identity')
            identities.add(node['id'])
        _encoded(state)  # Reject nonfinite/non-JSON source before retaining it.
        self._session = state['session_id']
        self._state = copy.deepcopy(state)
        self._records = [node for node in self._state['nodes']
                         if self._eligible(node, state['scope']['id'])]
        pending, self._pending = self._pending, None
        priority, receipt = (), None
        if pending is not None:
            matches = [node for node in self._records if _identity(node) == pending['identity']]
            coherent = (pending['source']['session_sha256'] == _digest(state['session_id'])
                and pending['source']['scope_sha256'] == _digest(state['scope']) and len(matches) == 1)
            if coherent:
                priority = (matches[0]['id'],)
            receipt = {'schema': EXPANSION_SCHEMA, 'status': 'fresh_control_observed' if coherent else 'unavailable',
                'original_source': pending['source'], 'fresh_source': self._source(),
                'reference': pending['reference'], 'identity_sha256': _digest(pending['identity']),
                'fresh_identity_verified': coherent, 'input_authority': False, 'task_progress': False}
        return self._view(), priority, receipt

    def _reference(self, selection):
        return 'nc-' + _digest([self._source(), selection])[:32]

    def _base(self):
        return {'schema': CATALOGUE_SCHEMA, 'source': self._source(),
            'temporal_scope': 'captured_native_observation', 'input_authority': False,
            'evidence_authority': 'untrusted_native_accessibility', 'pixel_visibility_verified': False,
            'task_progress': False, 'absence_claims_supported': False,
            'source_truncated': self._state.get('truncated', True),
            'eligible_source_records': len(self._records), 'maximum_source_records': self.MAX_RECORDS,
            'maximum_utf8_bytes': self.MAX_BYTES, 'maximum_entries': self.MAX_ENTRIES,
            'entries': [], 'omitted_entries': 0}

    def _view(self):
        self._offers = {}
        role, start = self._selection
        groups = sorted({node['role'] for node in self._records})
        if role is not None and role not in groups:
            role, start = None, 0
        rows = groups if role is None else [node for node in self._records if node['role'] == role]
        if start >= len(rows):
            start = 0
        self._selection = role, start
        view = self._base()
        view['view'] = {'kind': 'role_groups' if role is None else 'controls',
                        'role': _excerpt(role, 64)[0] if role is not None else None,
                        'offset': start, 'total_entries': len(rows)}

        def append(selection, fields):
            reference = self._reference(selection)
            item = {'reference': reference, **fields}
            trial = {**view, 'entries': [*view['entries'], item]}
            if len(trial['entries']) > self.MAX_ENTRIES or len(_encoded(trial)) > self.MAX_BYTES:
                return False
            view['entries'].append(item)
            self._offers[reference] = selection
            return True

        # Navigation choices count toward the same entry and byte budgets.
        if role is not None:
            append({'kind': 'page', 'role': None, 'offset': 0}, {'kind': 'back'})
        if start:
            append({'kind': 'page', 'role': role, 'offset': 0}, {'kind': 'first_page'})
        end = start
        # Reserve one slot and bytes for a next-page descriptor.
        for index in range(start, len(rows)):
            if len(view['entries']) >= self.MAX_ENTRIES - 1 or len(_encoded(view)) > self.MAX_BYTES - 512:
                break
            row = rows[index]
            if role is None:
                label, clipped = _excerpt(row, 64)
                fields = {'kind': 'group', 'role': label, 'role_truncated': clipped,
                          'count': sum(node['role'] == row for node in self._records)}
                selection = {'kind': 'page', 'role': row, 'offset': 0}
            else:
                label, clipped = _excerpt(row['name'], 120)
                fields = {'kind': 'control', 'name': label, 'name_truncated': clipped,
                          'name_sha256': _digest(row['name']), 'role': _excerpt(row['role'], 64)[0]}
                selection = {'kind': 'control', 'index': self._state['nodes'].index(row)}
            if not append(selection, fields):
                break
            end = index + 1
        if end < len(rows):
            append({'kind': 'page', 'role': role, 'offset': end}, {'kind': 'next_page'})
        view['omitted_entries'] = len(rows) - (end - start)
        view['coverage'] = 'partial'
        if len(_encoded(view)) > self.MAX_BYTES:
            raise ComputerUseError('Control catalogue exceeded its bounded presentation')
        return self._name_examples(view) if role is None else view

    def _name_examples(self, view):
        """Add partial source-name evidence only after all original offers fit."""
        enriched = copy.deepcopy(view)
        enriched['name_examples_scope'] = 'partial_accessible_name_excerpts_not_control_bindings'
        if len(_encoded(enriched)) > self.MAX_BYTES:
            return view
        for index, entry in enumerate(enriched['entries']):
            if entry['kind'] != 'group':
                continue
            role = self._offers[entry['reference']]['role']
            examples = []
            for node in self._records:
                if node['role'] != role or not node['name'].strip():
                    continue
                prospective = [*examples, _excerpt(node['name'], 120)[0]]
                if len(_encoded(prospective)) > 256:
                    break
                trial = copy.deepcopy(enriched)
                trial['entries'][index]['name_examples'] = prospective
                if len(_encoded(trial)) > self.MAX_BYTES:
                    break
                examples = prospective
                entry['name_examples'] = examples
        return enriched

    def inspect(self, reference):
        """Select only an actually offered evidence expansion, never an input."""
        base = {'dispatch_status': 'not_dispatched', 'input_attempted': False,
                'input_authority': False, 'task_progress': False, 'new_effect_count': 0,
                'evidence_authority': 'untrusted_native_accessibility'}
        if self._closed or not isinstance(reference, str) or reference not in self._offers:
            return {**base, 'status': 'unavailable', 'error': 'native_control_reference_unavailable'}
        selection = self._offers[reference]
        content = ([_identity(node) for node in self._records if selection['role'] is None
                    or node['role'] == selection['role']] if selection['kind'] == 'page'
                   else _identity(self._state['nodes'][selection['index']]))
        novelty = _digest([self._epoch, self._session, self._state['scope'], selection, content])
        first = novelty not in self._seen
        if first and len(self._seen) >= self.MAX_EXPANSIONS:
            return {**base, 'status': 'unavailable', 'error': 'native_control_expansion_limit'}
        self._seen.add(novelty)
        receipt = {'schema': EXPANSION_SCHEMA, 'reference': reference,
            'original_source': self._source(), 'expansion_kind': selection['kind'],
            'first_expansion_in_task': first, 'input_authority': False, 'task_progress': False}
        if selection['kind'] == 'page':
            self._selection = selection['role'], selection['offset']
        else:
            self._pending = {'reference': reference, 'source': self._source(),
                             'identity': _identity(self._state['nodes'][selection['index']])}
        return {**base, 'status': 'control_discovery', 'control_expansion_receipt': receipt,
                'control_catalogue': self._view()}
