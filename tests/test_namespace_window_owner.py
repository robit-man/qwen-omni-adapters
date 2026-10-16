from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_native_identity import mutate, valid_sample

from portal import native_identity as identity
from portal import native_ui
from runtime import acceptance_native_support as support


def test_registration_rejects_reported_pid_with_foreign_uid_and_zero_start(monkeypatch):
    monkeypatch.setattr(support, 'process_start', lambda pid: '0')
    monkeypatch.setattr(support, 'command', lambda *args: '2')
    check = Mock(side_effect=support.CapabilityBlocked('Host process ownership unavailable'))
    monkeypatch.setattr(support, '_assert_host_process', check, raising=False)
    desktop = support.OwnedDesktop({'DISPLAY': ':fixture'})
    with pytest.raises((support.CapabilityBlocked, RuntimeError)):
        desktop.register(support.OwnedWindow('browser', '10', '2', '0'))
    assert desktop.windows == []


def test_foreground_can_publish_only_the_authenticated_namespace_host(monkeypatch):
    monkeypatch.setattr(native_ui.subprocess, 'run', lambda args, **kwargs: SimpleNamespace(
        returncode=0, stdout='10\n' if args[1] == 'getactivewindow' else '2\n'))
    monkeypatch.setattr(native_ui, '_host_process_identity', Mock(side_effect=RuntimeError('foreign uid')), raising=False)
    proof = {'identity': {'window_id': '10', 'process_id': 77}, 'digest': 'a' * 64,
             'reported_wm_pid': 2, 'owner': {'pid': 77, 'start': '770'}}
    helper = SimpleNamespace(namespace_window_owner=Mock(return_value=proof))
    monkeypatch.setattr(native_ui, '_identity_helper', lambda: helper)
    value = native_ui._x11_foreground({'DISPLAY': ':fixture'}, include_binding=True)
    assert value['process_id'] == 77
    assert value['_namespace_owner_binding'] == proof
    helper.namespace_window_owner.assert_called_once()


def signed(value):
    value = copy.deepcopy(value)
    value.pop('digest', None)
    value['digest'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value


def binding():
    values = valid_sample()
    proof = {'source': 'flatpak_namespace_metadata', 'identity': {'window_id': '10', 'process_id': 20},
             'reported_pid': 17, 'bus_name': values['bus_name'], 'values': values,
             'digest': hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    return signed({'identity': proof['identity'], 'reported_wm_pid': 2,
                   'owner': {'pid': 20, 'start': '200'},
                   'host_process': {'pid': 20, 'start': '200', 'uid': os.getuid(),
                                    'executable': copy.deepcopy(values['installed_proxy'])}, 'proof': proof})


def test_full_namespace_proof_and_wrapper_digests_are_both_required():
    value = binding()
    assert identity._check_namespace_owner(value) == value
    value['proof']['values']['metadata']['host']['sha256'] = 'e' * 64
    with pytest.raises(identity.IdentityUnavailable):
        identity._check_namespace_owner(signed(value))


@pytest.mark.parametrize(('path', 'replacement'), [
    ('reported_wm_pid', 3), ('reported_wm_pid', True), ('owner.start', '201'),
    ('host_process.uid', -1), ('host_process.start', '0'), ('host_process.executable.mode', stat.S_IFIFO),
    ('proof.values.credentials.pid', 999), ('proof.values.registry.instance', 'wrong'),
    ('proof.values.host.uid', -1), ('proof.values.host.nspid', [20]),
    ('proof.values.host.nspid', [20, 3]), ('proof.values.wrapper.ppid', 999),
])
def test_incomplete_or_spoofed_namespace_proof_never_returns_owner(path, replacement):
    value = binding()
    mutate(value, path, replacement)
    with pytest.raises(identity.IdentityUnavailable):
        identity._check_namespace_owner(signed(value))


@pytest.mark.parametrize('level', ['top', 'proof', 'host_process', 'record', 'executable', 'metadata'])
def test_extraneous_output_fields_reject(level):
    value = binding()
    target = {'top': value, 'proof': value['proof'], 'host_process': value['host_process'],
              'record': value['proof']['values']['host'], 'executable': value['host_process']['executable'],
              'metadata': value['proof']['values']['metadata']['host']}[level]
    target['unexpected'] = 'not authority'
    with pytest.raises(identity.IdentityUnavailable):
        identity._check_namespace_owner(signed(value))


@pytest.mark.parametrize('fault', ['wrong_xid', 'wrong_reported_pid', 'exe_changed', 'xres_changed', 'oversize', 'timeout'])
def test_parent_revalidates_exact_request_and_fresh_host_process(monkeypatch, fault):
    value = binding()
    if fault == 'wrong_xid':
        value['identity']['window_id'] = '11'
    if fault == 'wrong_reported_pid':
        value['reported_wm_pid'] = 3
        value['proof']['values']['host']['nspid'][-1] = 3
        values = value['proof']['values']
        value['proof']['digest'] = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    value = signed(value)
    raw = json.dumps(value)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=raw + (' ' * 65536 if fault == 'oversize' else '')))
    if fault == 'timeout':
        run.side_effect = subprocess.TimeoutExpired('fixed-helper', 1)
    monkeypatch.setattr(identity.subprocess, 'run', run)
    host = copy.deepcopy(value['host_process'])
    if fault == 'exe_changed':
        host['executable']['inode'] += 1
    monkeypatch.setattr(identity, 'host_process_identity', lambda pid: host)
    monkeypatch.setattr(identity, 'xres_owner', lambda *a, **kw: {'pid': 21, 'start': '210'} if fault == 'xres_changed' else value['owner'])
    with pytest.raises(identity.IdentityUnavailable):
        identity.namespace_window_owner('10', 2, {'DISPLAY': ':fixture'})


def test_ordinary_valid_host_does_not_query_namespace_or_xres(monkeypatch):
    helper = SimpleNamespace(host_process_identity=Mock(return_value={'pid': 20, 'start': '200'}),
                             namespace_window_owner=Mock(side_effect=AssertionError('not ordinary')),
                             xres_owner=Mock(side_effect=AssertionError('not ordinary')))
    monkeypatch.setattr(native_ui, '_identity_helper', lambda: helper)
    monkeypatch.setattr(native_ui.subprocess, 'run', lambda args, **kw: SimpleNamespace(
        returncode=0, stdout='10\n' if args[1] == 'getactivewindow' else '20\n'))
    assert native_ui._x11_foreground({'DISPLAY': ':fixture'}) == {'window_id': '10', 'process_id': 20}
    assert helper.namespace_window_owner.call_count == helper.xres_owner.call_count == 0


def test_collector_uses_complete_roots_and_existing_full_authentication_without_observe(monkeypatch):
    value = binding()
    bridge = SimpleNamespace(_read=lambda fn: fn(), _credentials=Mock(), deadline=time.monotonic() + 3,
        _inactive_namespace_providers=Mock(return_value=[(object(), 17, ':1.237', ':1.237:/app')]),
        observe=Mock(side_effect=AssertionError('recursive observe')),
        foreground=Mock(side_effect=AssertionError('recursive foreground')))
    monkeypatch.setattr(identity, 'xres_owner', Mock(return_value=value['owner']))
    monkeypatch.setattr(identity, 'host_process_identity', Mock(return_value=value['host_process']))
    def sample(*args, **kwargs):
        identity._validate(value['proof']['values'])
        return copy.deepcopy(value['proof']['values'])
    monkeypatch.setattr(identity, '_sample', sample)
    result = identity.collect_namespace_owner('10', 2, bridge)
    assert result == value
    assert bridge._inactive_namespace_providers.call_count == 2
    assert identity.xres_owner.call_count == identity.host_process_identity.call_count == 2
    bridge.observe.assert_not_called()
    bridge.foreground.assert_not_called()


@pytest.mark.parametrize('fault', ['ambiguous', 'roots_changed', 'xres_changed', 'host_changed'])
def test_collector_changes_withhold_identity(monkeypatch, fault):
    value = binding()
    providers = [(object(), 17, ':1.237', ':1.237:/app')]
    bridge = SimpleNamespace(_read=lambda fn: fn(), _credentials=Mock(), deadline=time.monotonic() + 3,
                             _inactive_namespace_providers=Mock(return_value=providers))
    if fault == 'ambiguous':
        bridge._inactive_namespace_providers.return_value = providers * 2
    if fault == 'roots_changed':
        bridge._inactive_namespace_providers.side_effect = [providers, []]
    monkeypatch.setattr(identity, 'provider_proof', lambda *a, **kw: value['proof'])
    monkeypatch.setattr(identity, 'xres_owner', Mock(side_effect=[value['owner'], {'pid': 21, 'start': '210'}] if fault == 'xres_changed' else None,
                                                   return_value=value['owner']))
    monkeypatch.setattr(identity, 'host_process_identity', Mock(side_effect=[value['host_process'], {}] if fault == 'host_changed' else None,
                                                               return_value=value['host_process']))
    with pytest.raises(identity.IdentityUnavailable):
        identity.collect_namespace_owner('10', 2, bridge)


def test_exact_allowed_xid_does_not_query_unrelated_bases_or_transients(monkeypatch):
    calls = []
    monkeypatch.setattr(native_ui, '_process_start', lambda pid: '200')
    monkeypatch.setattr(native_ui, '_x11_window_owner', lambda xid, *a, **kw: calls.append(xid) or {'process_id': 20})
    monkeypatch.setattr(native_ui.subprocess, 'run', Mock(side_effect=AssertionError('not a transient')))
    scope = {'windows': [{'window_id': xid, 'process_id': 20, 'process_start': '200'} for xid in ['9', '10', '11']]}
    assert native_ui._window_scope_allows(scope, {'window_id': '10', 'process_id': 20})
    assert calls == ['10', '10']


@pytest.mark.parametrize('transient,expected', [(False, ['10', '10', '10']),
                                               (True, ['12', '12', '10', '12'])])
def test_scoped_foreground_refresh_count_shares_one_absolute_deadline(monkeypatch, transient, expected):
    xid, calls = ('12' if transient else '10'), []
    deadline = time.monotonic() + 3
    def namespace(window_id, wm_pid, environment, *, deadline):
        calls.append((window_id, deadline))
        value = binding()
        value['identity']['window_id'] = window_id
        return value
    helper = SimpleNamespace(host_process_identity=Mock(side_effect=identity.IdentityUnavailable('foreign')),
                             namespace_window_owner=namespace,
                             xres_owner=Mock(side_effect=AssertionError('not absent-property fallback')))
    monkeypatch.setattr(native_ui, '_identity_helper', lambda: helper)
    monkeypatch.setattr(native_ui, '_process_start', lambda pid: '200')
    def run(args, **kwargs):
        assert 0 < kwargs['timeout'] <= 1
        value = ('WM_TRANSIENT_FOR(WINDOW): window id # 0xa' if args[0] == 'xprop'
                 else xid if args[1] == 'getactivewindow' else '2')
        return SimpleNamespace(returncode=0, stdout=value)
    monkeypatch.setattr(native_ui.subprocess, 'run', run)
    monkeypatch.setenv('DISPLAY', ':fixture')
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.setenv('XDG_SESSION_TYPE', 'x11')
    fixture = SimpleNamespace(_foreground_provider=native_ui._native_foreground_binding, deadline=deadline,
        _host_owner_bound=False, _host_owner_binding=None,
        window_scope={'windows': [{'window_id': '10', 'process_id': 20, 'process_start': '200'}]})
    assert native_ui._AtspiBridge.foreground(fixture) == {'window_id': xid, 'process_id': 20}
    assert [window for window, _ in calls] == expected
    assert all(bound == deadline for _, bound in calls)
    helper.xres_owner.assert_not_called()


def test_expired_deadline_does_not_spawn_any_namespace_helper(monkeypatch):
    run = Mock(side_effect=AssertionError('expired'))
    monkeypatch.setattr(identity.subprocess, 'run', run)
    with pytest.raises(identity.IdentityUnavailable, match='budget'):
        identity.namespace_window_owner('10', 2, {}, deadline=time.monotonic()-1)
    run.assert_not_called()


def fake_proc(tmp_path, *, uid=None, start='200', executable=True):
    root = tmp_path / 'proc'
    process = root / '20'
    process.mkdir(parents=True)
    words = ['S'] + ['0'] * 19
    words[19] = start
    (process / 'stat').write_text('20 (fixture) ' + ' '.join(words))
    (process / 'status').write_text('Uid:\t' + '\t'.join([str(os.getuid() if uid is None else uid)] * 4) + '\n')
    if executable:
        image = tmp_path / 'binary'
        image.write_bytes(b'\x7fELFfixture')
        (process / 'exe').symlink_to(image)
    return root


@pytest.mark.parametrize('fault', [None, 'foreign_uid', 'zero_start', 'missing_exe'])
def test_registration_consumes_real_positive_host_uid_exe_and_start_metadata(tmp_path, monkeypatch, fault):
    proc = fake_proc(tmp_path, uid=os.getuid()+1 if fault == 'foreign_uid' else None,
                     start='0' if fault == 'zero_start' else '200', executable=fault != 'missing_exe')
    monkeypatch.setattr(native_ui, '_host_process_identity', lambda pid: identity.host_process_identity(pid, proc_root=proc))
    monkeypatch.setattr(support, 'process_start', lambda pid: '200')
    command = Mock(return_value='20')
    monkeypatch.setattr(support, 'command', command)
    desktop = support.OwnedDesktop({'DISPLAY': ':fixture'})
    window = support.OwnedWindow('browser', '10', '20', '200')
    if fault:
        with pytest.raises(support.CapabilityBlocked):
            desktop.register(window)
        assert desktop.windows == []
        command.assert_not_called()
    else:
        assert desktop.register(window) is window
        assert desktop.windows == [window]


def test_real_bounded_provider_root_collector_rejects_incomplete_scan_without_observe(monkeypatch):
    fixture = SimpleNamespace(MAX_CHILDREN=2, deadline=time.monotonic()+3,
        atspi=SimpleNamespace(get_desktop=lambda i: object()), _read=lambda fn: fn(),
        _clear_cache=lambda node: None, _count=lambda node: 3,
        foreground=Mock(side_effect=AssertionError('not foreground')), observe=Mock(side_effect=AssertionError('not observe')))
    with pytest.raises(native_ui.NativeUiUnavailable, match='scan_incomplete'):
        native_ui._AtspiBridge._inactive_namespace_providers(fixture)
    fixture.foreground.assert_not_called()
    fixture.observe.assert_not_called()


def test_namespace_helper_timeout_uses_remaining_absolute_budget(monkeypatch):
    value = binding()
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(value))
    monkeypatch.setattr(identity.subprocess, 'run', run)
    monkeypatch.setattr(identity, 'host_process_identity', lambda pid: value['host_process'])
    monkeypatch.setattr(identity, 'xres_owner', lambda *a, **kw: value['owner'])
    deadline = time.monotonic()+.2
    assert identity.namespace_window_owner('10', 2, {}, deadline=deadline) == value
    assert 0 < calls[0][1]['timeout'] <= .2
    assert float(calls[0][0][-1]) == deadline


@pytest.mark.parametrize('args', [('0', '2'), ('10', '0'), ('bad', '2'), ('10', '2', 'nan')])
def test_actual_private_cli_rejects_bad_args_before_native_import_or_query(args):
    from pathlib import Path
    argv = list(args)
    if len(argv) == 2:
        argv.append(str(time.monotonic()+1))
    process = subprocess.run([subprocess.sys.executable, '-I', str(Path(identity.__file__)),
                              '--namespace-window-owner', *argv], capture_output=True, text=True, timeout=3)
    assert process.returncode == 1
    assert json.loads(process.stdout) == {'error': 'native_namespace_owner_invalid'}
    assert process.stderr == ''
