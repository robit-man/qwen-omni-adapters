from __future__ import annotations

import ctypes
import subprocess
from types import SimpleNamespace

import pytest

from runtime import owned_window_close as close


class X11:
    def __init__(self, protocols=(102,), *, open_display=True, send=True):
        self.calls = []
        self.atoms = (ctypes.c_ulong * len(protocols))(*protocols)
        self.open_display, self.send = open_display, send
        self.message = None

    def XOpenDisplay(self, name):
        self.calls.append(("open", name))
        return 1234 if self.open_display else None

    def XInternAtom(self, display, name, only_existing):
        self.calls.append(("atom", display, name, only_existing))
        return {b"WM_PROTOCOLS": 101, b"WM_DELETE_WINDOW": 102}[name]

    def XGetWMProtocols(self, display, window, protocols, count):
        self.calls.append(("protocols", display, window))
        ctypes.cast(protocols, ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)))[0] = ctypes.cast(
            self.atoms, ctypes.POINTER(ctypes.c_ulong))
        ctypes.cast(count, ctypes.POINTER(ctypes.c_int))[0] = len(self.atoms)
        return 1

    def XSendEvent(self, display, window, propagate, mask, event):
        self.calls.append(("send", display, window, propagate, mask))
        message = ctypes.cast(event, ctypes.POINTER(close._Event)).contents.client
        self.message = {"type": message.type, "window": message.window, "message_type": message.message_type,
                        "format": message.format, "delete_atom": message.data.l[0], "time": message.data.l[1],
                        "send_event": message.send_event}
        return self.send

    def XFlush(self, display):
        self.calls.append(("flush", display))

    def XFree(self, pointer):
        self.calls.append(("free", bool(pointer)))

    def XCloseDisplay(self, display):
        self.calls.append(("close_display", display))


def test_exact_client_message_after_two_identity_checks_without_destructive_api(monkeypatch):
    library = X11()
    identities = []
    monkeypatch.setattr(close, "_xlib", lambda: library)
    monkeypatch.setattr(close, "_identity_matches", lambda *args: identities.append(args) or True)
    assert close._send_request("222", "333", "444") is True
    assert identities == [("222", "333", "444"), ("222", "333", "444")]
    assert library.message == {"type": 33, "window": 222, "message_type": 101, "format": 32,
                               "delete_atom": 102, "time": 0, "send_event": 1}
    assert ("send", 1234, 222, False, 0) in library.calls
    assert library.calls[-2:] == [("free", True), ("close_display", 1234)]
    # The fake has no destroy, kill, focus or input methods; no fallback is allowed.


@pytest.mark.parametrize("fault", ["wrong_identity", "identity_changes", "unsupported_close", "no_display", "send_refused"])
def test_rejection_never_falls_back_to_destroy_or_kill(monkeypatch, fault):
    library = X11(protocols=(999,) if fault == "unsupported_close" else (102,),
                  open_display=fault != "no_display", send=fault != "send_refused")
    states = iter([fault != "wrong_identity", fault != "identity_changes"])
    monkeypatch.setattr(close, "_identity_matches", lambda *_: next(states))
    monkeypatch.setattr(close, "_xlib", lambda: library)
    assert close._send_request("222", "333", "444") is False
    sends = [call for call in library.calls if call[0] == "send"]
    assert len(sends) == (1 if fault == "send_refused" else 0)
    if fault not in {"wrong_identity", "no_display"}:
        assert library.calls[-1] == ("close_display", 1234)


@pytest.mark.parametrize("identity", [("", "2", "3"), ("-1", "2", "3"), ("2;kill", "2", "3"),
                                      (str(2**32), "2", "3"), ("1", "0", "3"), ("1", "2", "old"),
                                      ("١", "2", "3")])
def test_invalid_identity_never_starts_transport(monkeypatch, identity):
    monkeypatch.setattr(close.subprocess, "run", lambda *_a, **_k: pytest.fail("Transport started"))
    assert close.request_owned_window_close(*identity, {"DISPLAY": ":test"}) is False


def test_fixed_isolated_helper_inherits_recovered_env_without_mutating_global_env(monkeypatch):
    captured = []
    environment = {"DISPLAY": ":owned", "XAUTHORITY": "/owned/authority", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/owned/bus"}
    monkeypatch.setattr(close.subprocess, "run", lambda argv, **kwargs: captured.append((argv, kwargs))
                        or SimpleNamespace(returncode=0))
    assert close.request_owned_window_close("222", "333", "444", environment)
    argv, kwargs = captured[0]
    assert argv == [close.sys.executable, "-I", "-u", str(close.Path(close.__file__).resolve()),
                    "--request-close", "222", "333", "444"]
    assert kwargs["env"] == environment and kwargs["env"] is not environment
    assert kwargs["timeout"] == 3 and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert "shell" not in kwargs


def test_transport_failure_reports_failure_without_retry(monkeypatch):
    calls = []
    def timed_out(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 3)
    monkeypatch.setattr(close.subprocess, "run", timed_out)
    assert not close.request_owned_window_close("222", "333", "444", {"DISPLAY": ":owned"})
    assert len(calls) == 1


def test_pid_reuse_after_window_pid_probe_is_rejected(monkeypatch):
    starts = iter(["444", "changed"])
    monkeypatch.setattr(close, "_process_start", lambda _pid: next(starts))
    monkeypatch.setattr(close.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="333\n"))
    assert close._identity_matches("222", "333", "444") is False


def test_window_reassigned_to_another_process_is_rejected(monkeypatch):
    monkeypatch.setattr(close, "_process_start", lambda _pid: "444")
    monkeypatch.setattr(close.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="999\n"))
    assert close._identity_matches("222", "333", "444") is False


def test_shared_app_cleanup_reports_refusal_without_destroy_or_kill(monkeypatch):
    from runtime import acceptance_native_support as support

    owner = support.OwnedDesktop({"DISPLAY": ":owned"})
    owner.windows.append(support.OwnedWindow("shared-app", "222", "333", "444"))
    calls, requests = [], []
    monkeypatch.setattr(support.subprocess, "run", lambda argv, **_kwargs: calls.append(argv)
                        or SimpleNamespace(returncode=0, stdout="333\n"))
    monkeypatch.setattr(support, "process_start", lambda _pid: "444")
    monkeypatch.setattr(support, "request_owned_window_close", lambda *args: requests.append(args) or False)
    monkeypatch.setattr(support.os, "killpg", lambda *_args: pytest.fail("Shared client killed"))
    assert owner.close() == [{"application": "shared-app", "window_id": "222", "closed": False}]
    assert calls == [["xdotool", "getwindowpid", "222"]]
    assert requests == [("222", "333", "444", {"DISPLAY": ":owned"})]


def test_firefox_close_request_is_not_claimed_as_closed_while_window_remains(monkeypatch):
    from runtime import verify_real_firefox as firefox

    owner = firefox.OwnedFirefox({"DISPLAY": ":owned"})
    owner.window, owner.pid, owner.start, owner.before = "222", "333", "444", {"111"}
    calls = []
    monkeypatch.setattr(firefox, "firefox_windows", lambda _env: {"111", "222"})
    monkeypatch.setattr(firefox, "process_start", lambda _pid: "444")
    monkeypatch.setattr(firefox, "command", lambda argv, _env: calls.append(argv) or "333")
    monkeypatch.setattr(firefox, "request_owned_window_close", lambda *_args: True)
    ticks = iter([0, 4])
    monkeypatch.setattr(firefox.time, "monotonic", lambda: next(ticks))
    assert owner.close() is False
    assert calls == [["xdotool", "getwindowpid", "222"]]
