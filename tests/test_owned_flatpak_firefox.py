from __future__ import annotations

import copy
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime import owned_flatpak_firefox as owned


def row(parent="1", mapped=2):
    return {"parent": parent, "map_state": mapped, "override_redirect": False}


class Backend:
    """CPU fixture with real transaction validation; it never touches X11."""

    def __init__(self):
        self.tree = {
            "root": "1",
            "complete": True,
            "windows": {"1": row("0"), "17": row(), "18": row(mapped=0)},
        }
        self.managed = ["17"]
        self.queue = []
        self.elapsed, self.launches, self.closes = 0, 0, 0
        self.installed = {"ref": "app/org.mozilla.firefox/aarch64/stable", "commit": "a" * 64}
        self.identity = SimpleNamespace(
            xres_owner=lambda xid, env: (
                {"pid": 77, "start": "770"} if xid == "21" else {"pid": 88, "start": "880"}
            )
        )
        self.proof_row = {
            "window": {"window_id": "21", "process_id": 77, "process_start": "770"},
            "frame": {
                "identity": ":1.99:/frame",
                "generation": "1",
                "screen_geometry": {"x": 10, "y": 20, "width": 800, "height": 900},
            },
            "namespace": {
                "source": "flatpak_namespace_metadata",
                "identity": {"window_id": "21", "process_id": 77},
                "values": {"owner": {"pid": 77, "start": "770"}},
                "digest": "b" * 64,
            },
            "blank_source": {"source": "atspi.Document.DocURL", "url": "about:blank"},
        }
        self.on_launch = None
        self.exit_code = 0

    def clock(self):
        return self.elapsed

    def wait(self):
        self.elapsed += 1

    def snapshot(self):
        return copy.deepcopy(self.tree)

    def clients(self):
        return list(self.managed)

    def deployment(self):
        return copy.deepcopy(self.installed)

    def arm(self, _tree):
        pass

    def events(self):
        queued, self.queue = self.queue, []
        return queued

    def launch(self, _deployment):
        self.launches += 1
        self.tree["windows"]["21"] = row()
        self.managed.append("21")
        self.queue = [
            {"type": kind, "window_id": "21", "parent": "1", "synthetic": False}
            for kind in [16, 19]
        ]
        if self.on_launch:
            self.on_launch()
        return self.launch_record()

    def launch_record(self):
        return {"pid": 999, "start": None, "exit_code": self.exit_code}

    def proof(self, _xid, _deployment, *, blank):
        result = copy.deepcopy(self.proof_row)
        if not blank:
            result["blank_source"] = None
        return result

    validate_events = owned._SystemBackend.validate_events

    def polite_close(self, _proof, _deployment):
        self.closes += 1
        self.tree["windows"].pop("21")
        self.managed.remove("21")


def test_one_fixed_launch_private_ledger_recheck_and_exact_close():
    backend = Backend()
    transaction = owned._Transaction(backend)
    ledger = transaction.create()
    assert ledger["window"]["window_id"] == "21" and backend.launches == 1
    assert "18" in ledger["before"]["windows"]  # Old hidden resource is excluded, not dropped.
    assert transaction.verify(blank=True)["blank_source"]["url"] == "about:blank"
    assert transaction.close() == {
        "closed": True,
        "polite_close_requested": True,
        "absence": "complete_XQueryTree",
    }
    assert backend.closes == 1 and set(backend.tree["windows"]) == {"1", "17", "18"}
    with pytest.raises(owned.OwnershipUnavailable, match="already_attempted"):
        transaction.create()
    assert backend.launches == 1


@pytest.mark.parametrize(
    "fault", ["incomplete", "cycle", "missing_parent", "wrong_root", "oversize"]
)
def test_invalid_complete_tree_withholds_launch(fault):
    backend = Backend()
    if fault == "incomplete":
        backend.tree["complete"] = False
    elif fault == "cycle":
        backend.tree["windows"]["18"]["parent"] = "18"
    elif fault == "missing_parent":
        backend.tree["windows"]["18"]["parent"] = "99"
    elif fault == "wrong_root":
        backend.tree["root"] = "99"
    else:
        backend.tree["windows"].update({str(i): row() for i in range(30, 4200)})
    with pytest.raises(owned.OwnershipUnavailable, match="tree_incomplete"):
        owned._Transaction(backend).create()
    assert backend.launches == 0


@pytest.mark.parametrize(
    "fault",
    [
        "old_hidden",
        "concurrent",
        "old_state",
        "no_create",
        "no_map",
        "wrong_owner",
        "wrong_window",
        "not_blank",
        "foreign_event",
        "launcher_error",
        "deployment_drift",
    ],
)
def test_new_window_requires_complete_nonconcurrent_bound_blank_launch(fault):
    backend = Backend()
    if fault == "old_hidden":
        backend.tree["windows"]["21"] = row()

    def change():
        if fault == "concurrent":
            backend.managed.append("22")
            backend.tree["windows"]["22"] = row()
        elif fault == "old_state":
            backend.tree["windows"]["18"]["map_state"] = 2
        elif fault in {"no_create", "no_map"}:
            backend.queue = [
                e for e in backend.queue if e["type"] != (16 if fault == "no_create" else 19)
            ]
        elif fault == "wrong_owner":
            backend.proof_row["namespace"]["values"]["owner"]["pid"] = 88
        elif fault == "wrong_window":
            backend.proof_row["window"]["window_id"] = "17"
        elif fault == "not_blank":
            backend.proof_row["blank_source"]["url"] = "https://example.org/"
        elif fault == "foreign_event":
            backend.tree["windows"]["22"] = row()
            backend.queue.append({"type": 16, "window_id": "22", "parent": "1", "synthetic": False})
        elif fault == "launcher_error":
            backend.exit_code = 1
        elif fault == "deployment_drift":
            backend.installed["commit"] = "c" * 64

    backend.on_launch = change
    transaction = owned._Transaction(backend)
    with pytest.raises(owned.OwnershipUnavailable):
        transaction.create()
    assert backend.launches == 1 and transaction.ledger is None and backend.closes == 0


@pytest.mark.parametrize(
    "fault", ["namespace", "frame", "generation", "source", "recycled", "preimage_destroyed"]
)
def test_refresh_and_cleanup_fail_closed_for_changed_private_binding(monkeypatch, fault):
    backend = Backend()
    transaction = owned._Transaction(backend)
    transaction.create()
    if fault == "namespace":
        backend.proof_row["namespace"]["digest"] = "d" * 64
    elif fault == "frame":
        backend.proof_row["frame"]["identity"] = ":1.99:/different"
    elif fault == "generation":
        backend.proof_row["frame"]["generation"] = "2"
    elif fault == "source":
        monkeypatch.setattr(owned, "_source", lambda: {"changed": "yes"})
    else:
        backend.queue = [{"type": 17, "window_id": "21" if fault == "recycled" else "18"}]
    with pytest.raises(owned.OwnershipUnavailable):
        transaction.close()
    assert backend.closes == 0


def test_actual_xid_absence_needed_class_search_is_never_consulted():
    backend = Backend()
    transaction = owned._Transaction(backend)
    transaction.create()
    backend.tree["windows"].pop("21")
    backend.managed.remove("21")
    assert transaction.close()["closed"] is True and backend.closes == 0
    empty = owned._Transaction(Backend())
    with pytest.raises(owned.OwnershipUnavailable, match="ledger_missing"):
        empty.close()


def test_task_may_navigate_but_pre_submit_still_requires_original_blank_document():
    backend = Backend()
    transaction = owned._Transaction(backend)
    transaction.create()
    backend.proof_row["blank_source"]["url"] = "https://docs.python.org/"
    with pytest.raises(owned.OwnershipUnavailable):
        transaction.verify(blank=True)
    assert transaction.close()["closed"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": owned.SCHEMA, "op": "close", "window_id": "17"},
        {"schema": owned.SCHEMA, "op": "create", "ledger": {}},
        {"schema": owned.SCHEMA, "op": []},
        None,
    ],
)
def test_helper_never_accepts_caller_window_or_ledger_as_ownership(monkeypatch, capsys, payload):
    factory = Mock(side_effect=AssertionError("Native backend must not be initialized"))
    monkeypatch.setattr(owned, "_SystemBackend", factory)
    monkeypatch.setattr(owned.sys, "argv", ["owned_flatpak_firefox.py", "--owned-helper"])
    monkeypatch.setattr(
        owned.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO((__import__("json").dumps(payload) + "\n").encode())),
    )
    assert owned._main() == 0
    assert '"ok": false' in capsys.readouterr().out
    factory.assert_not_called()


def test_explicit_task_branch_registers_only_private_proof_and_scope_refresh(monkeypatch):
    from runtime.long_horizon_desktop import TaskDesktop
    from runtime.verify_long_horizon_native import registered_native_scope

    backend = Backend()
    transaction = owned._Transaction(backend)
    owner = SimpleNamespace(
        last_receipt=None,
        open_blank=transaction.create,
        verify=lambda: transaction.verify(blank=True),
        close=transaction.close,
    )
    monkeypatch.setattr(owned, "OwnedFlatpakFirefox", lambda _environment: owner)
    monkeypatch.setattr(
        "runtime.long_horizon_desktop.class_windows",
        lambda *_a, **_kw: pytest.fail("Flatpak setup/close must not use class ownership"),
    )
    monkeypatch.setattr(
        "runtime.verify_long_horizon_native.command",
        lambda *_a, **_kw: pytest.fail("Namespace pre-submit must use the launch ledger"),
    )
    desktop = TaskDesktop({"DISPLAY": ":cpu"})
    window = desktop.open_blank("browser", None, browser="flatpak-firefox", log_path=None)
    assert window.owner is owner
    assert registered_native_scope(desktop)["windows"] == [backend.proof_row["window"]]
    backend.proof_row["namespace"]["digest"] = "f" * 64
    from runtime.acceptance_native_support import CapabilityBlocked

    with pytest.raises(CapabilityBlocked):
        registered_native_scope(desktop)
    assert desktop.close()[0]["closed"] is False and backend.closes == 0


@pytest.mark.parametrize(
    "fault", [None, "ref", "commit", "location", "writable", "uid", "entry_size"]
)
def test_installed_system_ref_commit_and_executable_are_checked_without_launch(monkeypatch, fault):
    import stat
    from pathlib import Path

    base = "/var/lib/flatpak/app/org.mozilla.firefox/aarch64/stable/" + "a" * 64

    class InstalledPath(type(Path())):
        def resolve(self, strict=False):
            return self

        def stat(self, **_kwargs):
            return SimpleNamespace(
                st_uid=1000 if fault == "uid" else 0,
                st_mode=stat.S_IFREG | (0o777 if fault == "writable" else 0o755),
                st_dev=1,
                st_ino=2,
                st_size=123,
            )

        def open(self, *_args, **_kwargs):
            return io.BytesIO(b"x" * (65537 if fault == "entry_size" else 40))

    monkeypatch.setattr(owned, "Path", InstalledPath)
    commands = []

    def command(argv):
        commands.append(argv)
        return {
            "--show-ref": "app/org.mozilla.firefox/aarch64/other"
            if fault == "ref"
            else "app/org.mozilla.firefox/aarch64/stable",
            "--show-commit": "not-a-commit" if fault == "commit" else "a" * 64,
            "--show-location": "/user/unproven" if fault == "location" else base,
        }[argv[2]]

    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.command = command
    if fault:
        with pytest.raises(owned.OwnershipUnavailable):
            backend.deployment()
    else:
        assert backend.deployment()["location"] == base
    assert all(argv[:2] == ["/usr/bin/flatpak", "info"] for argv in commands)


@pytest.mark.parametrize(
    "fault", [None, "wrong_url", "two_roots", "hidden_only", "protected_ancestor", "changed_url"]
)
def test_blank_proof_reads_committed_document_attribute_without_typing_or_active_window_inference(
    monkeypatch, fault
):
    from test_native_ui import Fixture, Node

    fixture = Fixture()
    document = Node("/document", "document web", interfaces=["Document"])
    document.get_document_iface = lambda: document
    fixture.window.children = [document]
    calls = []

    def get_url(interface, attribute):
        calls.append((interface, attribute))
        return (
            "https://example.org/"
            if fault == "wrong_url" or fault == "changed_url" and len(calls) > 1
            else "about:blank"
        )

    fixture.api.Document = SimpleNamespace(get_document_attribute_value=get_url)
    if fault == "two_roots":
        other = Node("/other-document", "document web", interfaces=["Document"])
        fixture.window.children.append(other)
    if fault == "hidden_only":
        document.state.values.discard("SHOWING")
    if fault == "protected_ancestor":
        protected = Node("/protected", "password text")
        protected.children = [document]
        fixture.window.children = [protected]
    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.bridge = fixture.bridge
    fixture.bridge.reads = 0
    fixture.bridge.deadline = __import__("time").monotonic() + 5
    if fault:
        with pytest.raises(owned.OwnershipUnavailable):
            backend.blank_document(fixture.window)
    else:
        result = backend.blank_document(fixture.window)
        assert result == {"source": "atspi.Document.DocURL", "url": "about:blank"}
        assert len(calls) == 2 and all(attribute == "DocURL" for _, attribute in calls)


def test_prelaunch_failure_has_safe_diagnostic_without_launch_streams():
    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.launch_process = None
    backend.launch_identity = None
    backend.launch_streams = []
    assert backend.launch_record() == {"attempted": False}


@pytest.mark.parametrize("operation", ["verify", "close"])
def test_private_helper_without_its_own_launch_never_initializes_native_backend(
    monkeypatch, capsys, operation
):
    import json

    constructor = Mock(side_effect=AssertionError("No ledger; no native access"))
    monkeypatch.setattr(owned, "_SystemBackend", constructor)
    monkeypatch.setattr(owned.sys, "argv", ["helper", "--owned-helper"])
    monkeypatch.setattr(
        owned.sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(
                (
                    json.dumps({"schema": owned.SCHEMA, "op": operation, "request_id": 1}) + "\n"
                ).encode()
            )
        ),
    )
    assert owned._main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["code"] == "flatpak_launch_ledger_missing"
    constructor.assert_not_called()


def test_external_owner_object_cannot_enroll_a_namespace_window(monkeypatch):
    from runtime.acceptance_native_support import CapabilityBlocked, OwnedWindow
    from runtime.long_horizon_desktop import TaskDesktop
    from runtime.verify_long_horizon_native import registered_native_scope

    callback = Mock(side_effect=AssertionError("Unregistered callbacks cannot create authority"))
    desktop = TaskDesktop({"DISPLAY": ":cpu"})
    desktop.windows.append(
        OwnedWindow("browser", "21", "77", "770", owner=SimpleNamespace(verify=callback))
    )
    monkeypatch.setattr(
        "runtime.verify_long_horizon_native.command",
        Mock(side_effect=AssertionError("No PID fallback")),
    )
    with pytest.raises(CapabilityBlocked):
        registered_native_scope(desktop)
    callback.assert_not_called()


def test_actual_launcher_arguments_are_closed_ordinary_about_blank(monkeypatch):
    import subprocess

    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.launch_process = None
    backend.launch_identity = None
    backend.launch_streams = []
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=2147483000, poll=lambda: 0)

    monkeypatch.setattr(owned.subprocess, "Popen", popen)
    result = backend.launch({"arch": "aarch64"})
    assert calls[0][0] == [
        "/usr/bin/flatpak",
        "run",
        "--branch=stable",
        "--arch=aarch64",
        "--command=firefox",
        "org.mozilla.firefox",
        "--new-window",
        "about:blank",
    ]
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert result["ordinary_profile"] is True and result["exit_code"] == 0
    for stream in backend.launch_streams:
        stream.close()


def test_namespace_polite_close_rechecks_proof_immediately_before_one_wm_delete(monkeypatch):
    import ctypes

    from runtime import owned_window_close

    proof = Backend().proof_row
    proof["blank_source"] = None
    sent = []
    protocols = (ctypes.c_ulong * 1)(101)

    class Library:
        def XOpenDisplay(self, _):
            return 1

        def XCloseDisplay(self, _):
            pass

        def XInternAtom(self, _display, atom, _only):
            return 100 if atom == b"WM_PROTOCOLS" else 101

        def XGetWMProtocols(self, display, xid, output, count):
            assert xid == 21
            ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)))[0] = ctypes.cast(
                protocols, ctypes.POINTER(ctypes.c_ulong)
            )
            ctypes.cast(count, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 1

        def XFree(self, _):
            pass

        def XSendEvent(self, _display, xid, _propagate, _mask, _event):
            sent.append(xid)
            return 1

        def XFlush(self, _):
            pass

    def load(module):
        module._xlib = lambda: Library()
        module._Event = owned_window_close._Event

    monkeypatch.setattr(
        owned.importlib.util,
        "spec_from_file_location",
        lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=load)),
    )
    monkeypatch.setattr(owned.importlib.util, "module_from_spec", lambda _spec: SimpleNamespace())
    backend = owned._SystemBackend.__new__(owned._SystemBackend)
    backend.proof = Mock(return_value=proof)
    backend.polite_close(proof, {})
    assert sent == [21] and backend.proof.call_count == 2
    sent.clear()
    changed = copy.deepcopy(proof)
    changed["frame"]["generation"] = "2"
    backend.proof = Mock(side_effect=[proof, changed])
    with pytest.raises(owned.OwnershipUnavailable):
        backend.polite_close(proof, {})
    assert sent == []


def test_actual_ownership_journal_is_json_before_scope_submission_and_after_close(
    monkeypatch, tmp_path
):
    import json

    from runtime.long_horizon_desktop import TaskDesktop
    from runtime.verify_long_horizon_native import registered_native_scope, save

    backend = Backend()
    transaction = owned._Transaction(backend)
    owner = SimpleNamespace(
        last_receipt=None,
        open_blank=transaction.create,
        verify=lambda: transaction.verify(blank=True),
        close=transaction.close,
    )
    monkeypatch.setattr(owned, "OwnedFlatpakFirefox", lambda _env: owner)
    desktop = TaskDesktop({"DISPLAY": ":cpu"})
    desktop.open_blank("browser", None, browser="flatpak-firefox", log_path=None)
    journal = tmp_path / "ownership-journal.json"
    save(journal, desktop.launches)
    saved = json.loads(journal.read_bytes())
    assert saved[0]["ownership_evidence"]["window"] == backend.proof_row["window"]
    assert registered_native_scope(desktop)["windows"] == [backend.proof_row["window"]]
    assert desktop.close()[0]["closed"] is True
    save(journal, desktop.launches)


def test_actual_parent_transport_never_consumes_late_create_reply_as_verify(monkeypatch):
    import subprocess
    import sys
    import time

    script = """import json,sys,time
r=json.loads(sys.stdin.readline());time.sleep(.12)
print(json.dumps(dict(schema=r['schema'],ok=True,op=r['op'],request_id=r.get('request_id'),ledger={'late_create':True})),flush=True)
sys.stdin.readline();time.sleep(2)
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = child
    real_selector = owned.selectors.DefaultSelector

    class ShortDeadline:
        def __enter__(self):
            self.selector = real_selector()
            return self

        def __exit__(self, *_):
            self.selector.close()

        def register(self, *args):
            return self.selector.register(*args)

        def select(self, timeout):
            return self.selector.select(min(timeout, 0.02))

    try:
        monkeypatch.setattr(owned.selectors, "DefaultSelector", ShortDeadline)
        with pytest.raises(owned.OwnershipUnavailable):
            owner._request("create")
        monkeypatch.setattr(owned.selectors, "DefaultSelector", real_selector)
        time.sleep(0.15)
        with pytest.raises(owned.OwnershipUnavailable):
            owner._request("verify")
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)
        child.stdin.close()
        child.stdout.close()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "malformed",
        "oversize",
        "eof",
        "unknown_key",
        "wrong_op",
        "wrong_id",
        "missing_op",
        "wrong_payload",
        "wrong_schema",
    ],
)
def test_actual_parent_pipe_protocol_correlates_and_terminally_poisons_invalid_reply(
    monkeypatch, fault
):
    import subprocess
    import sys

    script = """import json,sys
fault=sys.argv[1]
r=json.loads(sys.stdin.readline())
if fault=='eof':sys.exit(0)
if fault=='oversize':
 print('x'*(2*1024*1024+1),flush=True)
elif fault=='malformed':print('{broken-json',flush=True)
else:
 out=dict(schema=r['schema'],op=r['op'],request_id=r['request_id'],ok=True,proof={'transport_only':True})
 if fault=='unknown_key':out['unexpected']=True
 if fault=='wrong_op':out['op']='create'
 if fault=='wrong_id':out['request_id']+=1
 if fault=='missing_op':del out['op']
 if fault=='wrong_payload':out['proof']='not-an-object'
 if fault=='wrong_schema':out['schema']='unknown'
 print(json.dumps(out),flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(fault)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    owner = owned.OwnedFlatpakFirefox({})
    owner.process = child
    respawn = Mock(side_effect=AssertionError("Poisoned transport may not respawn"))
    monkeypatch.setattr(owned.subprocess, "Popen", respawn)
    try:
        if fault is None:
            assert owner._request("verify")["proof"] == {"transport_only": True}
            assert owner._poisoned is False and owner._sequence == 1
        else:
            with pytest.raises(owned.OwnershipUnavailable):
                owner._request("verify")
            assert owner._poisoned is True and child.poll() is not None
            with pytest.raises(owned.OwnershipUnavailable):
                owner._request("close")
            respawn.assert_not_called()
    finally:
        owner._stop_helper()


def test_helper_echoes_sequence_and_operation_without_exposing_a_supplied_ledger(
    monkeypatch, capsys
):
    import json

    backend = Backend()
    backend.begin = lambda: None
    backend.end = lambda: None
    monkeypatch.setattr(owned, "_SystemBackend", lambda: backend)
    monkeypatch.setattr(owned.sys, "argv", ["helper", "--owned-helper"])
    commands = [
        {"schema": owned.SCHEMA, "op": op, "request_id": index}
        for index, op in enumerate(["create", "verify", "close"], 1)
    ]
    monkeypatch.setattr(
        owned.sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(("\n".join(json.dumps(c) for c in commands) + "\n").encode())
        ),
    )
    assert owned._main() == 0
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["request_id"], r["op"], r["ok"]) for r in results] == [
        (1, "create", True),
        (2, "verify", True),
        (3, "close", True),
    ]
    assert results[-1]["receipt"]["closed"] is True and backend.launches == backend.closes == 1
