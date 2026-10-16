from __future__ import annotations

import io
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from portal import native_identity
from runtime import owned_flatpak_firefox as owned
from runtime.verify_long_horizon_native import parser


def descriptor(monkeypatch, text=b'#!/bin/bash\nflatpak run org.chromium.Chromium --no-sandbox "$@"\n', uid=0):
    base = '/var/lib/flatpak/app/org.chromium.Chromium/aarch64/stable/' + 'a' * 64
    class InstalledPath(type(Path())):
        def resolve(self, strict=False):
            return self
        def stat(self, **kwargs):
            return SimpleNamespace(st_uid=uid, st_mode=stat.S_IFREG | 0o755, st_dev=1, st_ino=2,
                                   st_size=40, st_mtime_ns=1, st_ctime_ns=2)
        def read_text(self, *args, **kwargs):
            if str(self).startswith("/proc/"):
                raise FileNotFoundError
            return text.decode()
        def is_symlink(self):
            return False
        def read_bytes(self):
            return text
        def open(self, *args, **kwargs):
            return io.BytesIO(b'desktop entry')
    monkeypatch.setattr(owned, 'Path', InstalledPath)
    monkeypatch.setattr(owned.shutil, 'which', lambda name: '/usr/local/bin/chromium')
    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.browser, backend.application = 'chromium', 'org.chromium.Chromium'
    backend.executable_relative = 'files/chromium/chrome'
    backend.identity = native_identity
    backend.command = lambda args: {'--show-ref':'app/org.chromium.Chromium/aarch64/stable',
                                   '--show-commit':'a'*64, '--show-location':base}[args[2]]
    return backend


def test_root_installed_launcher_pin_is_per_deployment_not_global_hash(monkeypatch):
    before = descriptor(monkeypatch).deployment()
    after = descriptor(monkeypatch, b'#!/bin/sh\n# legitimate packaging update\nexec /usr/bin/flatpak run org.chromium.Chromium --no-sandbox "$@"\n').deployment()
    assert before['launcher']['sha256'] != after['launcher']['sha256']
    assert before['launcher']['existing_flags'] == after['launcher']['existing_flags'] == ['--no-sandbox']
    assert before['executable'] == after['executable']


@pytest.mark.parametrize('text', [
    b'#!/bin/sh\nflatpak run org.other.Browser "$@"\n',
    b'#!/bin/sh\nflatpak run org.chromium.Chromium --user-data-dir=/tmp/private "$@"\n',
    b'#!/bin/sh\nflatpak run org.chromium.Chromium --remote-debugging-port=9000 "$@"\n',
    b'#!/bin/sh\nflatpak run org.chromium.Chromium "$@"\nfalse\n',
    b'#!/bin/sh\nflatpak run org.chromium.Chromium\n',
])
def test_closed_descriptor_never_accepts_other_app_profile_or_injection(monkeypatch, text):
    with pytest.raises(owned.OwnershipUnavailable):
        descriptor(monkeypatch, text).deployment()


def test_non_root_installed_launcher_is_not_authority(monkeypatch):
    with pytest.raises(owned.OwnershipUnavailable):
        descriptor(monkeypatch, uid=1000).deployment()


def test_chromium_launch_reuses_exact_pinned_launcher_and_ordinary_arguments(monkeypatch):
    backend = descriptor(monkeypatch)
    deployment = backend.deployment()
    backend.launch_process, backend.launch_identity, backend.launch_streams = None, None, []
    popen = Mock(return_value=SimpleNamespace(pid=2147483000, poll=lambda: 0))
    monkeypatch.setattr(owned.subprocess, 'Popen', popen)
    backend.launch(deployment)
    assert popen.call_args.args[0] == ['/usr/local/bin/chromium', '--new-window', 'about:blank']
    assert popen.call_args.kwargs['start_new_session'] is True
    assert '--no-sandbox' not in popen.call_args.args[0]  # Existing launcher supplies it, unchanged.
    for stream in backend.launch_streams:
        stream.close()


def test_changed_deployment_before_launch_grants_no_browser_creation(monkeypatch):
    backend = descriptor(monkeypatch)
    deployment = backend.deployment()
    backend.launch_process, backend.launch_streams = None, []
    backend.deployment = lambda: {**deployment, 'commit':'b'*64}
    popen = Mock(side_effect=AssertionError('unbound launch'))
    monkeypatch.setattr(owned.subprocess, 'Popen', popen)
    with pytest.raises(owned.OwnershipUnavailable, match='binding_changed'):
        backend.launch(deployment)
    popen.assert_not_called()
    for stream in backend.launch_streams:
        stream.close()


def test_private_browser_selector_is_closed_and_cli_condition_explicit():
    with pytest.raises(owned.OwnershipUnavailable, match='descriptor'):
        owned.OwnedFlatpakFirefox({}, browser='arbitrary-app')
    args = parser().parse_args(['--case', 'LHN-001', '--browser', 'flatpak-chromium', '--output', '/tmp/never-created'])
    assert args.browser == 'flatpak-chromium'
