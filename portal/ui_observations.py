"""Task-local expansion of historical native text, without input authority."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict

from qwen_omni_adapters.computer_use import ComputerUseError


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class NativeObservationArchive:
    MAX_RECORDS = 16
    MAX_BYTES = 512 * 1024

    def __init__(self, epoch: str):
        self.epoch = epoch
        self.session = None
        self.records = OrderedDict()
        self.expanded = set()
        self.evicted = 0
        self.bytes = 0

    def remember(self, phase: int, observed: dict) -> None:
        state = observed.get('state', {})
        projection = state.get('model_projection', {})
        scope = state.get('scope', {})
        if (scope.get('kind') not in {'desktop', 'native'}
                or projection.get('schema') != 'robit.omni.ui-model-projection.v1'
                or projection.get('source_revision') != state.get('revision')
                or projection.get('text_authority') != 'untrusted_visible_ui'
                or not isinstance(state.get('session_id'), str) or not state['session_id']):
            return
        if self.session is not None and self.session != state['session_id']:
            raise ComputerUseError('Historical observation crossed native task session')
        self.session = state['session_id']
        text = state.get('text', '')
        if not isinstance(text, str) or len(text) > 4000:
            return
        excerpt = state.get('native_content_excerpt')
        if excerpt is not None and (not isinstance(excerpt, dict)
                or excerpt.get('schema') != 'robit.omni.native-content-excerpt.v1'
                or excerpt.get('source_revision') != state['revision']
                or excerpt.get('authority') != 'untrusted_native_accessibility'
                or excerpt.get('input_authority') is not False
                or excerpt.get('pixel_visibility_verified') is not False
                or not isinstance(excerpt.get('text'), str) or len(excerpt['text']) > 4000):
            return
        if not text.strip() and not (excerpt and excerpt['text'].strip()):
            return
        title = next((n.get('name', '') for n in state.get('nodes', [])
                      if n.get('id') == scope.get('id') and not n.get('sensitive')), '')
        document = state.get('document_evidence', {})
        source = {'status': 'unavailable'}
        if (document.get('status') == 'observed' and document.get('source') == 'atspi.Document.DocURL'
                and document.get('scope_id') == scope.get('id')
                and isinstance(document.get('url'), str) and len(document['url']) <= 2048):
            source = {key: copy.deepcopy(document[key]) for key in
                      ('status', 'source', 'url', 'node_id', 'generation', 'scope_id', 'window_id') if key in document}
        body = {'window_title': str(title)[:500], 'document_evidence': source, 'ui_text': text,
                'native_content_excerpt': copy.deepcopy(excerpt), 'source_revision': state['revision'],
                'source_scope_id': scope.get('id'), 'source_truncated': state.get('truncated', True)}
        # Focus/capture revisions and toolbar changes do not create another copy
        # of the same document excerpt. The original source binding stays exact.
        identity = {'session': self.session, 'scope': scope.get('id'), 'document': source,
                    'content': excerpt['text'] if excerpt else text}
        reference = 'obs-' + hashlib.sha256(_encoded([self.epoch, identity])).hexdigest()[:24]
        if reference in self.records:
            self.bytes -= len(_encoded(self.records[reference]))
            self.records[reference]['last_observed_phase'] = phase
            self.bytes += len(_encoded(self.records[reference]))
            self.records.move_to_end(reference)
        else:
            record = {'reference': reference, 'captured_phase': phase, 'last_observed_phase': phase,
                      'content_sha256': hashlib.sha256(_encoded(body)).hexdigest(), 'observation': body}
            size = len(_encoded(record))
            if size > self.MAX_BYTES:
                return
            self.records[reference] = record
            self.bytes += size
        while len(self.records) > self.MAX_RECORDS or self.bytes > self.MAX_BYTES:
            key, removed = self.records.popitem(last=False)
            self.bytes -= len(_encoded(removed))
            self.expanded.discard(key)
            self.evicted += 1

    def view(self) -> dict:
        entries = []
        for reference, record in self.records.items():
            body = record['observation']
            resource = body['document_evidence'].get('url') or body['window_title']
            label = resource.encode()[:160].decode(errors='ignore')
            entries.append({'reference': reference, 'resource': label, 'resource_truncated': label != resource,
                            'captured_phase': record['captured_phase'], 'last_observed_phase': record['last_observed_phase']})
        return {'schema': 'robit.omni.native-observation-index.v1', 'temporal_scope': 'historical',
                'input_authority': False, 'current_ui_verified': False, 'task_lifetime_only': True,
                'evicted_count': self.evicted, 'entries': entries}

    def recall(self, reference: str) -> dict:
        if not isinstance(reference, str) or reference not in self.records:
            raise ComputerUseError('Historical observation is not retained in this native task')
        first = reference not in self.expanded
        self.expanded.add(reference)
        record = copy.deepcopy(self.records[reference])
        return {'status': 'historical_observation', 'dispatch_status': 'not_dispatched',
                'input_attempted': False, 'task_progress': False, 'new_effect_count': 0,
                'evidence_authority': 'untrusted_historical_native_observation',
                'recalled_observation': record, 'observation_recall_receipt': {
                    'schema': 'robit.omni.native-observation-recall.v1', 'reference': reference,
                    'temporal_scope': 'historical', 'current_ui_verified': False,
                    'input_authority': False, 'first_expansion_in_task': first}}
