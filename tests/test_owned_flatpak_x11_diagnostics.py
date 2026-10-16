"""Real private X errors must survive the actual correlated CPU pipe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_owned_flatpak_xlib import private_display as private_display

from runtime import owned_flatpak_firefox as owned

X_MAIN_SERVER = """
import ctypes,sys,time
from runtime import owned_flatpak_firefox as owned
mode=sys.argv[1]
class Backend:
 def __init__(self):
  self.x=owned._XDisplay();lib=self.x.lib
  lib.XCreateSimpleWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,
   ctypes.c_int,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_ulong,ctypes.c_ulong]
  lib.XCreateSimpleWindow.restype=ctypes.c_ulong
  lib.XDestroyWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
  if mode=='events':self.x.arm(self.x.snapshot(time.monotonic()+1))
  self.window=lib.XCreateSimpleWindow(self.x.display,self.x.root,0,0,8,8,0,0,0)
  lib.XSync(self.x.display,False)
  if mode in {'select','events'}:
   lib.XDestroyWindow(self.x.display,self.window);lib.XSync(self.x.display,False)
  if mode=='snapshot':
   original=lib.XGetWindowAttributes
   def raced(display,xid,attrs):
    if xid==self.window:
     lib.XDestroyWindow(self.x.display,self.window);lib.XSync(self.x.display,False)
    return original(display,xid,attrs)
   lib.XGetWindowAttributes=raced
 def begin(self):
  if mode=='select':self.x.select(str(self.window))
  elif mode=='events':self.x.events()
  else:self.x.snapshot(time.monotonic()+1)
 def end(self):pass
 def launch(self,*args):raise AssertionError('No browser may launch')
 def launch_record(self):return {'cpu_fixture_resource':self.window,'attempted':False}
owned._SystemBackend=Backend
sys.argv=['cpu-fixture','--owned-helper']
raise SystemExit(owned._main())
"""


@pytest.mark.parametrize("operation,request_code", [("snapshot", 3), ("select", 2), ("events", 2)])
def test_actual_xvfb_error_survives_main_pipe_and_parent(private_display, operation, request_code):
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", X_MAIN_SERVER, operation],
        cwd=Path(__file__).resolve().parents[1], env=private_display,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    owner = owned.OwnedFlatpakFirefox(private_display)
    owner.process = process
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == "flatpak_x11_protocol_error"
        reply = owner.last_receipt
        assert "x11_error" in reply, "Actual native diagnostic was discarded by IPC"
        error = reply["x11_error"]
        assert set(error) == {"error_code", "request_code", "minor_code", "resource_id", "serial",
                              "stage", "connection_poisoned"}
        assert error["error_code"] == 3  # BadWindow: local primary X11/X.h.
        assert error["request_code"] == request_code  # Xproto.h: attributes3 / change2.
        assert error["minor_code"] == 0
        assert error["resource_id"] == reply["launch"]["cpu_fixture_resource"]
        assert type(error["serial"]) is int and 0 < error["serial"] < 2**64
        assert error["stage"] == operation + ":post_sync"
        assert error["connection_poisoned"] is True
        assert reply["request_id"] == 1 and reply["op"] == "create" and reply["ok"] is False
        assert reply["launch"]["attempted"] is False and owner.ledger is None
        assert owner._poisoned is False  # Native-connection poison is distinct from transport.
        with pytest.raises(owned.OwnershipUnavailable) as again:
            owner._request("close")
        assert again.value.code == "flatpak_x11_connection_unavailable"
        assert "x11_error" not in owner.last_receipt and owner.last_receipt["request_id"] == 2
        assert owner.ledger is None
    finally:
        owner._stop_helper()
        process.stderr.close()


REPLY_SERVER = """import json,sys
extra=json.loads(sys.argv[1]);success=sys.argv[2]=='success';code=sys.argv[3]
for line in sys.stdin:
 request=json.loads(line)
 payload={'ledger':{}}if success else{'code':code,'launch':None}
 print(json.dumps(dict(schema=request['schema'],op=request['op'],request_id=request['request_id'],
  ok=success,**payload,x11_error=extra)),flush=True)
"""


def diagnostic():
    return {"error_code": 3, "request_code": 2, "minor_code": 0, "resource_id": 0,
            "serial": 0, "stage": "select:post_sync", "connection_poisoned": True}


def pipe_owner(value, *, success=False, code="flatpak_x11_protocol_error"):
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", REPLY_SERVER, json.dumps(value),
         "success" if success else "failure", code],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return owner


@pytest.mark.parametrize("upper", [False, True])
def test_exact_numeric_boundaries_are_accepted_only_as_negative_evidence(upper):
    value = diagnostic()
    if upper:
        value.update(error_code=255, request_code=255, minor_code=255,
                     resource_id=2**32 - 1, serial=2**64 - 1, stage="internal:pre_sync")
    owner = pipe_owner(value)
    try:
        with pytest.raises(owned.OwnershipUnavailable, match="flatpak_x11_protocol_error"):
            owner._request("create")
        assert owner.last_receipt["x11_error"] == value
        assert owner.ledger is None and owner._poisoned is False
    finally:
        owner._stop_helper()


@pytest.mark.parametrize("key,bad", [
    ("error_code", 0), ("error_code", 256), ("error_code", True), ("error_code", "3"),
    ("request_code", 0), ("request_code", 256),
    ("minor_code", -1), ("minor_code", 256),
    ("resource_id", -1), ("resource_id", 2**32),
    ("serial", -1), ("serial", 2**64), ("serial", 1.0),
    ("stage", "provider-private-detail"), ("stage", []),
    ("connection_poisoned", False), ("connection_poisoned", 1),
    ("private_message", "must not be exported"), ("missing_serial", None),
    ("whole_value", None), ("whole_value", []),
])
def test_invalid_diagnostics_terminally_close_pipe_without_authority_or_respawn(monkeypatch, key, bad):
    value = diagnostic()
    if key == "whole_value":
        value = bad
    elif key == "missing_serial":
        value.pop("serial")
    else:
        value[key] = bad
    owner = pipe_owner(value)
    monkeypatch.setattr(owned.subprocess, "Popen", lambda *a, **kw: pytest.fail("No helper respawn"))
    try:
        with pytest.raises(owned.OwnershipUnavailable, match="flatpak_ownership_protocol_invalid"):
            owner._request("create")
        assert owner._poisoned is True and owner.process.poll() is not None and owner.ledger is None
        with pytest.raises(owned.OwnershipUnavailable, match="transport_closed"):
            owner._request("verify")
        assert owner._sequence == 1
        assert "must not be exported" not in json.dumps(owner.last_receipt)
    finally:
        owner._stop_helper()


@pytest.mark.parametrize("success,code", [(True, "flatpak_x11_protocol_error"), (False, "other_error")])
def test_diagnostics_on_success_or_wrong_error_code_never_become_ownership(success, code):
    owner = pipe_owner(diagnostic(), success=success, code=code)
    try:
        with pytest.raises(owned.OwnershipUnavailable, match="flatpak_ownership_protocol_invalid"):
            owner._request("create")
        assert owner._poisoned is True and owner.ledger is None and owner.process.poll() is not None
    finally:
        owner._stop_helper()


ORIGIN_SERVER = """import sys
from runtime import owned_flatpak_firefox as owned
mode=sys.argv[1]
class Backend:
 def begin(self):
  if mode=='untyped':
   error=RuntimeError('private provider text');error.code='flatpak_x11_protocol_error'
   error.x11_error={'private_message':'private provider text'}
  else:
   error=owned._XProtocolUnavailable((3,2,0,0,1),'select:post_sync')
   error.code='other_error'
  raise error
 def end(self):pass
 def launch_record(self):return None
owned._SystemBackend=Backend
sys.argv=['cpu-fixture','--owned-helper']
raise SystemExit(owned._main())
"""


@pytest.mark.parametrize("mode,code", [("untyped", "flatpak_x11_protocol_error"), ("wrong_code", "other_error")])
def test_main_emits_diagnostic_only_from_private_error_with_exact_code(mode, code):
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", ORIGIN_SERVER, mode],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        with pytest.raises(owned.OwnershipUnavailable) as caught:
            owner._request("create")
        assert caught.value.code == code and "x11_error" not in owner.last_receipt
        assert "private provider text" not in json.dumps(owner.last_receipt)
        assert owner.ledger is None and owner._poisoned is False
    finally:
        owner._stop_helper()
