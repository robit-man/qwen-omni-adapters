"""Durable source retention, actual SQLite retrieval, temporal semantics and isolation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from harness.history import ConversationHistory, time_window
from harness.memory import MemoryStore, PassiveMemory
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness

ZONE = 'America/Los_Angeles'


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


class Encoder:
    def __call__(self, text):
        if text in {'Our pet is Biscuit.', 'What is my puppy called?'}:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def close(self):
        pass


@pytest.mark.parametrize('day,hours', [('2026-11-02T12:00:00-08:00', 25),
                                      ('2026-03-09T12:00:00-07:00', 23)])
def test_yesterday_is_a_calendar_day_across_dst(day, hours):
    w = time_window('yesterday', now=stamp(day), zone=ZONE)
    assert w['end'] - w['start'] == hours * 3600
    assert datetime.fromtimestamp(w['start'], ZoneInfo(ZONE)).hour == 0
    assert datetime.fromtimestamp(w['end'], ZoneInfo(ZONE)).hour == 0


def test_week_and_elapsed_hour_windows_are_distinct():
    now = stamp('2026-11-08T12:00:00-08:00')
    week = time_window('last_week', now=now, zone=ZONE)
    assert week['start_at'] == '2026-10-26T00:00:00-07:00'
    assert week['end_at'] == '2026-11-02T00:00:00-08:00'
    assert week['end'] - week['start'] == 169 * 3600
    rolling = time_window('last_7_days', now=now, zone=ZONE)
    assert rolling['end'] - rolling['start'] == 168 * 3600
    hour = time_window('hour_ago', now=now, zone=ZONE)
    assert (hour['start'], hour['end']) == (now - 5400, now - 1800)
    recent = time_window('past_hour', now=now, zone=ZONE)
    assert (recent['start'], recent['end']) == (now - 3600, now)


@pytest.mark.parametrize('fields', [
    {'period': 'bad'}, {'period': 'today', 'zone': 'not/a/zone'},
    {'period': 'range', 'start_at': '2026-01-01', 'end_at': '2026-02-01'},
    {'period': 'range', 'start_at': '2026-02-01T00:00:00Z', 'end_at': '2026-01-01T00:00:00Z'},
    {'period': 'range'}, {'period': 'yesterday', 'start_at': '2026-01-01T00:00:00Z'},
])
def test_invalid_time_constraints_are_explicit_errors(fields):
    with pytest.raises(ValueError):
        time_window(**fields)


def test_time_filter_precedes_similarity_and_preserves_repeated_events(tmp_path):
    h = ConversationHistory(tmp_path / 'archive.db')
    now = stamp('2026-11-08T12:00:00-08:00')
    old = h.append('Our pet is Biscuit.', occurred_at=now - 8 * 86400)
    wanted = h.append('Our pet is Biscuit.', occurred_at=now - 86400)
    future = h.append('Our pet is Biscuit.', occurred_at=now + 86400)
    for event in (old, wanted, future):
        h.vector(event, [1.0, 0.0])
    r = h.search('What is my puppy called?', vector=[1.0, 0.0], period='yesterday', now=now, zone=ZONE)
    assert [x['event_id'] for x in r['results']] == [wanted]
    assert r['events_in_window'] == 1
    assert r['results'][0]['content_sha256'] == hashlib.sha256(b'Our pet is Biscuit.').hexdigest()
    assert h._db.execute('SELECT count(*) FROM history_events').fetchone()[0] == 3
    h.close()


def test_boundaries_are_start_inclusive_end_exclusive_and_pages_follow_event_time(tmp_path):
    h = ConversationHistory(tmp_path / 'archive.db')
    now = stamp('2026-11-08T12:00:00-08:00')
    w = time_window('yesterday', now=now, zone=ZONE)
    start = h.append('At the start.', occurred_at=w['start'])
    h.append('At the end, outside yesterday.', occurred_at=w['end'])
    late = h.append('Late in the day.', occurred_at=w['end'] - 1)
    middle = h.append('Earlier in the day but inserted later.', occurred_at=w['start'] + 3600)
    first = h.search(period='yesterday', now=now, zone=ZONE, limit=1)
    assert [r['event_id'] for r in first['results']] == [late]
    second = h.search(period='yesterday', now=now, zone=ZONE, limit=1,
                      before_id=first['next_before_id'])
    assert [r['event_id'] for r in second['results']] == [middle]
    third = h.search(period='yesterday', now=now, zone=ZONE, limit=1,
                     before_id=second['next_before_id'])
    assert [r['event_id'] for r in third['results']] == [start]
    assert third['next_before_id'] is None
    h.close()


def test_encoder_failure_preserves_source_and_lexical_recall_after_restart(tmp_path):
    class Dead(Encoder):
        def __call__(self, _text):
            return None
    p = tmp_path / 'archive.db'
    s = MemoryStore(p, embedder=Dead())
    assert s.remember('The calibration reference is ultraviolet quartz.') is None
    s.close()
    s = MemoryStore(p, embedder=Dead())
    r = s.search_history('ultraviolet quartz')
    assert len(r['results']) == 1
    assert not r['semantic_query_available']
    assert s.history.pending() is None  # Failed work has a bounded retry delay.
    s.close()


def test_raw_journal_survives_cache_merge_decay_and_idempotent_reopen(tmp_path):
    p = tmp_path / 'archive.db'
    s = MemoryStore(p, embedder=Encoder())
    s.remember('Our pet is Biscuit.')
    s.remember('Our pet is Biscuit.')
    assert s.stats()['memories'] == 1
    s._db.execute('UPDATE memories SET strength=0.01,last_used_at=0,uses=0')
    s._db.commit()
    s.decay()
    assert s.stats()['memories'] == 0
    s.close()
    s = MemoryStore(p, embedder=Encoder())
    assert len(s.search_history('What is my puppy called?')['results']) == 2
    s.close()


def test_consolidation_and_rebuild_keep_immutable_sources_and_links(tmp_path):
    h = ConversationHistory(tmp_path / 'archive.db')
    event = h.append('The discussion concerned a violet telescope.', source={'type': 'voice_exchange',
                    'user': 'Which instrument?', 'assistant': 'A violet telescope.'})
    before = [tuple(r) for r in h._db.execute('SELECT id,text,occurred_at,content_sha256 FROM history_events')]
    h.consolidate()
    row = h._db.execute('SELECT * FROM history_buckets LIMIT 1').fetchone()
    assert json.loads(row['source_ids']) == [event]
    assert f'[event:{event}]' in row['excerpt']
    h._db.execute('DELETE FROM history_buckets')
    h._db.commit()
    h.rebuild_views()
    h.consolidate()
    assert [tuple(r) for r in h._db.execute('SELECT id,text,occurred_at,content_sha256 FROM history_events')] == before
    assert h.search('violet telescope')['results'][0]['event_id'] == event
    assert h._db.execute('SELECT count(*) FROM history_buckets').fetchone()[0] == 2
    h.close()


def test_passive_capture_survives_admission_denial_queue_saturation_and_shutdown(tmp_path):
    worker = PassiveMemory(tmp_path / 'archive.db', max_pending=1)
    worker.set_admission(lambda _size: False)
    for i in range(6):
        worker.remember(f'Durable exchange number {i}.')
    h = ConversationHistory(worker.path)
    assert h._db.execute('SELECT count(*) FROM history_events').fetchone()[0] == 6
    h.close()
    worker.close()
    h = ConversationHistory(worker.path)
    assert len(h.search(limit=8)['results']) == 6
    h.close()


def test_decay_applies_each_elapsed_interval_only_once(tmp_path):
    s = MemoryStore(tmp_path / 'archive.db', embedder=Encoder())
    s.remember('Keep this old calibration reference.')
    s._db.execute('UPDATE memories SET last_used_at=last_used_at-86400*10,uses=1')
    s._db.commit()
    s.decay()
    first = s.stats()['mean_strength']
    s.decay()
    assert s.stats()['mean_strength'] == first
    s.close()


def test_owner_archive_never_leaks_into_another_session_and_returns_typed_bad_time(tmp_path):
    p = tmp_path / 'archive.db'
    h = ConversationHistory(p)
    h.append('Private owner marker: violet telescope.')
    h.close()
    tools = PortalToolHarness(SessionDocumentStore(), conversation_history_path=p,
                              conversation_history_owner='owner-session')
    denied = tools.execute('other-session', 'conversation_search', {'query': '', 'period': 'today'})
    assert denied['error'] == 'ToolInputError' and 'results' not in denied
    allowed = tools.execute('owner-session', 'conversation_search', {'query': '', 'period': 'today'})
    assert len(allowed['results']) == 1
    assert allowed['source_is_current_visual_evidence'] is False
    bad = tools.execute('owner-session', 'conversation_search', {'query': '', 'period': 'range'})
    assert bad['error'] == 'ToolInputError'


def test_legacy_migration_is_exact_and_runs_once_without_importing_new_cache_rows(tmp_path):
    import sqlite3
    p = tmp_path / 'legacy.db'
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE memories(id INTEGER PRIMARY KEY,text TEXT,kind TEXT,vector BLOB,'
              'created_at REAL,last_used_at REAL,uses INTEGER,strength REAL)')
    c.execute('INSERT INTO memories VALUES(7,?,?,?,?,?,?,?)',
              ('Original calibration reference.', 'exchange', b'[0.0,1.0]', 1780000000.0,
               1780000001.0, 4, 0.4))
    c.commit()
    c.close()
    s = MemoryStore(p, embedder=Encoder())
    r = s.history._db.execute('SELECT * FROM history_events').fetchone()
    assert r['event_key'] == 'legacy:7' and r['occurred_at'] == 1780000000.0
    assert r['vector'] == b'[0.0,1.0]'
    assert json.loads(r['source_json'])['timestamp_uncertain'] is True
    s.remember('Our pet is Biscuit.')
    s.close()
    s = MemoryStore(p, embedder=Encoder())
    assert s.history._db.execute('SELECT count(*) FROM history_events').fetchone()[0] == 2
    s.close()


def test_long_sources_remain_durable_while_model_evidence_is_bounded(tmp_path):
    h = ConversationHistory(tmp_path / 'archive.db')
    for i in range(8):
        h.append(f'{i} ' + 'calibration ' * 1000, source={'type': 'voice_exchange',
                 'user': 'u' * 10000, 'assistant': 'a' * 10000})
    result = h.search(limit=8)
    assert len(json.dumps(result['results'])) <= 12000
    assert result['results_truncated'] and result['next_before_id']
    assert h._db.execute('SELECT min(length(text)) FROM history_events').fetchone()[0] > 10000
    h.close()


def test_real_portal_http_scope_and_token_rotation_keep_archive_private(tmp_path, monkeypatch):
    from portal.app import PortalConfig, create_app
    p = tmp_path / 'archive.db'
    h = ConversationHistory(p)
    h.append('Archive owner marker: violet telescope.')
    h.close()
    monkeypatch.setenv('OMNI_CALL_MEMORY', str(p))
    monkeypatch.setenv('OMNI_REPO_ROOT', str(tmp_path))
    for token in ('owner-token-before-restart-123456', 'owner-token-after-restart-654321'):
        config = PortalConfig(adapter_url='http://adapter/api/chat', adapter_health_url='http://adapter/healthz',
                              comprehension_health_url='http://comprehension/health', tts_health_url='http://tts/healthz',
                              ollama_health_url='http://ollama/api/tags', model='test', access_token=token,
                              timeout_s=30, max_body_bytes=1000000, session_log_dir=tmp_path / 'sessions')
        client = create_app(config).test_client()
        header = {'Authorization': 'Bearer ' + token}
        client.set_cookie('omni_portal_session', 'a' * 64)
        denied = client.post('/api/tools/conversation_search/call', headers=header,
                             json={'arguments': {'query': '', 'period': 'today'}})
        assert denied.status_code == 200 and denied.json['result']['error'] == 'ToolInputError'
        owner = hashlib.sha256(b'omni-voice-session\0' + token.encode()).hexdigest()
        client.set_cookie('omni_portal_session', owner)
        accepted = client.post('/api/tools/conversation_search/call', headers=header,
                               json={'arguments': {'query': '', 'period': 'today'}})
        assert accepted.status_code == 200
        assert accepted.json['result']['results'][0]['text'] == 'Archive owner marker: violet telescope.'


def test_spoken_recall_excludes_ambient_observations_without_deleting_them(tmp_path):
    h = ConversationHistory(tmp_path / 'archive.db')
    spoken = h.append('We discussed the calibration schedule.', kind='exchange')
    sound = h.append('Ambient hissing noise.', kind='sound')
    assert [r['event_id'] for r in h.search()['results']] == [spoken]
    assert [r['event_id'] for r in h.search(kind='sound')['results']] == [sound]
    assert h.search(kind='all')['events_in_window'] == 2
    h.close()


def test_concurrent_open_and_replayed_event_key_commit_one_source(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    p = tmp_path / 'archive.db'
    h = ConversationHistory(p)
    h.close()
    def write(_index):
        h = ConversationHistory(p)
        try:
            return h.append('Same delivery receipt after a producer retry.', event_key='retry-identity')
        finally:
            h.close()
    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(write, range(8)))
    h = ConversationHistory(p)
    assert len(set(ids)) == 1
    assert h._db.execute('SELECT count(*) FROM history_events').fetchone()[0] == 1
    h.close()
