"""Actual private pipe protocol, with CPU-only failure-producing backends."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from runtime import owned_flatpak_firefox as owned

MAIN_SERVER = """import json,sys
from runtime import owned_flatpak_firefox as owned
code=json.loads(sys.argv[1]);unexpected=sys.argv[2]=='unexpected'
class Backend:
 def begin(self):
  if unexpected:raise RuntimeError('private provider detail must stay redacted')
  raise owned.OwnershipUnavailable(code)
 def end(self):pass
 def launch_record(self):return {'attempted':False}
owned._SystemBackend=Backend
sys.argv=['cpu-fixture','--owned-helper']
raise SystemExit(owned._main())
"""

REPLY_SERVER = """import json,sys
code=json.loads(sys.argv[1])
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps(dict(schema=r['schema'],op=r['op'],request_id=r['request_id'],
  ok=False,code=code,launch=None)),flush=True)
"""

INVALID_CODES = [
    "",
    "1flatpak_error",
    "a" * 129,
    "flatpak private_detail",
    "flatpak\nprivate_detail",
    "flatpak\x00private_detail",
    "flatpak_é",
    "flatpak_١",
    "FLATPAK_error",
    None,
]


def child(script, code, *args):
    return subprocess.Popen(
        [sys.executable, "-u", "-c", script, json.dumps(code), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.parametrize(
    "code",
    [
        "flatpak_x11_protocol_error",
        "flatpak_x11_unavailable",
        "flatpak_launch_ledger_missing",
        "flatpak_" + "1" * 120,
    ],
)
def test_real_main_to_parent_preserves_typed_failure_without_ownership_or_poison(code):
    process = child(MAIN_SERVER, code, "typed")
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = process
    try:
        for sequence, operation in enumerate(["create", "close"], 1):
            with pytest.raises(owned.OwnershipUnavailable) as caught:
                owner._request(operation)
            assert caught.value.code == code
            assert owner.last_receipt == {
                "schema": owned.SCHEMA,
                "op": operation,
                "request_id": sequence,
                "ok": False,
                "code": code,
                "launch": {"attempted": False},
            }
            assert owner._poisoned is False and owner.ledger is None
    finally:
        owner._stop_helper()


def test_parent_accepts_valid_digit_code_from_real_pipe():
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = child(REPLY_SERVER, "flatpak_x11_protocol_error")
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == "flatpak_x11_protocol_error"
        assert owner._poisoned is False and owner.last_receipt["request_id"] == 1
        assert owner.ledger is None
    finally:
        owner._stop_helper()


@pytest.mark.parametrize("code", INVALID_CODES)
def test_main_redacts_invalid_provider_code_before_actual_parent_receives_it(code):
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = child(MAIN_SERVER, code, "typed")
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == "flatpak_ownership_unavailable"
        assert owner.last_receipt["code"] == "flatpak_ownership_unavailable"
        assert set(owner.last_receipt) == {"schema", "op", "request_id", "ok", "code", "launch"}
        assert owner._poisoned is False and owner.ledger is None
    finally:
        owner._stop_helper()


def test_main_does_not_expose_untyped_provider_exception_text():
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = child(MAIN_SERVER, None, "unexpected")
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == "flatpak_ownership_unavailable"
        assert "private provider detail" not in json.dumps(owner.last_receipt)
        assert owner.ledger is None
    finally:
        owner._stop_helper()


@pytest.mark.parametrize("code", INVALID_CODES)
def test_parent_terminally_rejects_invalid_code_without_replay_or_respawn(monkeypatch, code):
    owner = owned.OwnedFlatpakFirefox({})
    process = owner.process = child(REPLY_SERVER, code)
    monkeypatch.setattr(
        owned.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("No helper respawn")
    )
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == "flatpak_ownership_protocol_invalid"
        assert owner._poisoned is True and process.poll() is not None
        assert owner.ledger is None
        with pytest.raises(owned.OwnershipUnavailable, match="transport_closed"):
            owner._request("create")
        assert owner._sequence == 1
    finally:
        owner._stop_helper()
