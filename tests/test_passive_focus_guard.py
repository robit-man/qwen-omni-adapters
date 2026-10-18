"""Read-only controlled focus races; these do not prove actual desktop recovery."""
import os
import subprocess
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from portal import native_ui
from runtime import long_horizon_desktop as desktop
from runtime import passive_focus_guard as focus
from runtime import verify_long_horizon_native as runner

A = focus.OwnerBinding("17", 42, "420", os.getuid())
B = focus.OwnerBinding("18", 43, "430", os.getuid())
D = focus.OwnerBinding("19", 42, "420", os.getuid())
E = focus.OwnerBinding("20", 42, "420", os.getuid())
F = focus.OwnerBinding("21", 99, "990", os.getuid())
SCOPE = {"schema": "robit.omni.native-task-scope.v1", "windows": [
    {"window_id": row.window_id, "process_id": row.process_id, "process_start": row.process_start} for row in (A, B)]}


class Reader:
    def __init__(self, ids=("17", "18", "18"), *, stable=B, owners=None, parents=None, advance=0):
        self.ids, self.last_id, self.stable_value = iter(ids), ids[-1], stable
        self.owners = {row.window_id: row for row in (A, B, D, E, F)}
        self.owners.update(owners or {})
        self.parents = {"19": "17", "20": "17", **(parents or {})}
        self.calls, self.now, self.advance = [], 0., advance

    def _record(self, name, deadline):
        self.calls.append((name, deadline))
        self.now += self.advance

    def active(self, deadline):
        self._record("active", deadline)
        return next(self.ids, self.last_id)

    def owner(self, xid, deadline):
        self._record("owner:" + xid, deadline)
        value = self.owners[xid]
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Exception):
            raise value
        return value

    def stable(self, deadline):
        self._record("strict_stable", deadline)
        if isinstance(self.stable_value, Exception):
            raise self.stable_value
        return self.stable_value

    def parent(self, xid, deadline):
        self._record("parent:" + xid, deadline)
        value = self.parents[xid]
        return value.pop(0) if isinstance(value, list) else value


def sample(reader, **options):
    return focus.sample_owned_focus({}, SCOPE, reader=reader, clock=lambda: reader.now, **options)


@pytest.mark.parametrize("ids,stable,status", [
    (("17", "17"), A, "stable"), (("19", "19"), D, "stable"),
    (("17", "18", "18"), B, "owned_transition"), (("17", "19", "19"), D, "owned_transition"),
    (("19", "17", "17"), A, "owned_transition"), (("19", "18", "18"), B, "owned_transition"),
    (("19", "20", "20"), E, "owned_transition"),
])
def test_positive_owned_bases_and_direct_dialog_transitions(ids, stable, status):
    reader = Reader(ids, stable=stable)
    result = sample(reader)
    receipt = result.receipt()
    assert result.status == status
    assert result.identity == (stable.identity if status == "stable" else None)
    assert all(receipt[k] is False for k in ("input_authority", "pointer_credit", "method_credit", "task_progress"))
    assert len({deadline for _, deadline in reader.calls}) == 1
    if stable in (D, E):
        assert result.after.kind == "direct_transient" and result.after.registered_base == A
        assert result.after.transient_parent_window_id == "17"


@pytest.mark.parametrize("reader", [
    Reader(("21", "18", "18")), Reader(("17", "21", "18")),
    Reader(stable=A), Reader(("17", "18", "17")),
    Reader(stable=native_ui.NativeUiUnavailable("native_foreground_changed_during_observation")),
    Reader(owners={"17": replace(A, process_start="999")}),
    Reader(owners={"18": replace(B, process_start="999")}),
    Reader(owners={"17": replace(A, uid=os.getuid() + 1)}),
    Reader(owners={"18": replace(B, uid=os.getuid() + 1)}),
    Reader(owners={"18": None}), Reader(owners={"17": {}}),
    Reader(owners={"18": native_ui.NativeUiUnavailable("native_foreground_unmapped")}),
    Reader(("19", "18", "18"), parents={"19": "21"}),
    Reader(("19", "18", "18"), parents={"19": None}),
    Reader(("19", "18", "18"), parents={"19": ["17", "18"]}),
    Reader(("17", "19", "19"), stable=D, parents={"19": ["17", "17", "18", "18"]}),
    Reader(("19", "18", "18"), owners={"17": replace(A, process_start="999")}),
    Reader(("19", "18", "18"), owners={"17": replace(A, uid=os.getuid() + 1)}),
    Reader(("19", "18", "18"), owners={"19": [D, replace(D, process_start="999")]}),
    Reader(("17", "17"), owners={"17": [A, replace(A, process_start="999")]}),
    Reader(("17", "17"), owners={"17": [A, replace(A, uid=os.getuid() + 1)]}),
    Reader(("19", "19"), parents={"19": ["17", "17", "18", "18"]}),
    Reader(("19", "18", "18"), parents={"19": "20"}),  # Indirect ancestry is not authority.
])
def test_foreign_unknown_repeated_or_changing_ownership_never_recovers(reader):
    with pytest.raises(focus.FocusUnavailable) as error:
        sample(reader)
    assert error.value.evidence["input_authority"] is False
    assert error.value.evidence["status"] == "unavailable"
    assert reader.calls and len(reader.calls) <= 20


@pytest.mark.parametrize("value", [0, -.1, 2.6, float("nan"), float("inf")])
def test_invalid_total_deadline_is_rejected_before_queries(value):
    reader = Reader()
    with pytest.raises(ValueError):
        sample(reader, budget_s=value)
    assert not reader.calls


def test_total_deadline_does_not_reset_between_reads():
    reader = Reader(advance=.6)
    with pytest.raises(focus.FocusUnavailable) as error:
        sample(reader)
    assert error.value.code == "native_observation_budget_exceeded"
    assert len({deadline for _, deadline in reader.calls}) == 1 and reader.now <= 3


@pytest.mark.parametrize("during", [False, True])
def test_stop_before_or_during_proof_returns_no_positive_sample(during):
    reader = Reader()
    stopped = threading.Event()
    if during:
        original = reader.owner
        def owner(*args):
            result = original(*args)
            stopped.set()
            return result
        reader.owner = owner
    else:
        stopped.set()
    with pytest.raises(focus.FocusStopped) as error:
        sample(reader, stop_requested=stopped.is_set)
    assert error.value.evidence["status"] == "stopped"
    assert error.value.evidence["pointer_credit"] is False
    if not during:
        assert not reader.calls


@pytest.mark.parametrize("binding,parent,expected", [(A, None, True), (D, "17", True),
    (D, "18", False), (D, "21", False), (replace(A, process_start="999"), None, False)])
def test_typed_policy_matches_existing_exact_base_and_direct_transient_fixtures(monkeypatch, binding, parent, expected):
    windows = [SimpleNamespace(window_id=r.window_id, pid=str(r.process_id), process_start=r.process_start) for r in (A, B)]
    monkeypatch.setattr(desktop, "process_start", lambda _pid: binding.process_start)
    monkeypatch.setattr(desktop, "command", lambda *_: "WM_TRANSIENT_FOR(WINDOW): window id # " + hex(int(parent or "21")))
    assert desktop.owned_foreground(windows, binding.identity, {}) is expected
    reader = Reader((binding.window_id, binding.window_id), owners={binding.window_id: binding}, parents={binding.window_id: parent})
    if expected:
        assert sample(reader).status == "stable"
    else:
        with pytest.raises(focus.FocusUnavailable):
            sample(reader)


@pytest.mark.parametrize("host", [{}, {"pid": 42, "start": "420"},
    {"pid": True, "start": "420", "uid": os.getuid()},
    {"pid": 42, "start": "0", "uid": os.getuid()},
    {"pid": 42, "start": "420", "uid": True},
    {"pid": 42, "start": "420", "uid": os.getuid() + 1}])
def test_actual_reader_rejects_missing_or_malformed_host_proof(monkeypatch, host):
    monkeypatch.setattr(native_ui, "_host_process_identity", lambda _pid: host)
    with pytest.raises(native_ui.NativeUiUnavailable):
        focus.X11FocusReader._binding({"window_id": "17", "process_id": 42})


def test_actual_reader_keeps_fixed_queries(monkeypatch):
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout="17" if command[0] == "xdotool" else
            "WM_TRANSIENT_FOR(WINDOW): window id # 0x11\n")
    monkeypatch.setattr(focus.subprocess, "run", run)
    reader = focus.X11FocusReader({"DISPLAY": ":fixture"})
    deadline = time.monotonic() + 2.5
    assert reader.active(deadline) == "17" and reader.parent("19", deadline) == "17"
    assert calls[0][0] == ["xdotool", "getactivewindow"] and calls[1][0] == ["xprop", "-id", "19", "WM_TRANSIENT_FOR"]
    assert all(0 < options["timeout"] <= 1 for _, options in calls)


@pytest.mark.parametrize("cancel", [False, True])
def test_private_helper_kills_its_cpu_process_group_on_deadline_or_cancel(monkeypatch, tmp_path, cancel):
    original = subprocess.Popen
    processes, commands = [], []
    child_file = tmp_path / "child-pid"
    script = ("import subprocess,time,pathlib; "
        "p=subprocess.Popen([" + repr(__import__("sys").executable) + ",'-c','import time; time.sleep(30)']); "
        "pathlib.Path(" + repr(str(child_file)) + ").write_text(str(p.pid)); time.sleep(30)")
    def popen(command, **options):
        commands.append((command, options))
        process = original([__import__("sys").executable, "-c", script], **options)
        processes.append(process)
        return process
    monkeypatch.setattr(focus, "subprocess", SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired))
    assert subprocess.Popen is original
    stop = threading.Event()
    reader = focus.X11FocusReader({"DISPLAY": ":fixture"}, stop_requested=stop.is_set)
    timer = threading.Timer(.15, stop.set) if cancel else None
    if timer:
        timer.start()
    started = time.monotonic()
    with pytest.raises(focus.FocusStopped if cancel else subprocess.TimeoutExpired):
        reader.owner("17", started + (2.5 if cancel else .15))
    if timer:
        timer.join()
    assert time.monotonic() - started < .8 and processes[0].poll() is not None
    assert commands[0][0][-1] == "--read-identity" and commands[0][1]["start_new_session"] is True
    assert commands[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert child_file.exists()
    child = __import__("pathlib").Path("/proc", child_file.read_text(), "stat")
    def alive():
        try:
            return child.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False
    for _ in range(50):
        if not alive():
            break
        time.sleep(.01)
    assert not alive()



def test_strict_reader_cannot_be_silently_replaced_by_inprocess_wrapper():
    with pytest.raises(ValueError):
        focus.X11FocusReader({"DISPLAY": ":fixture"}, strict_foreground=lambda *_a, **_k: A.identity)


class StopAfterWait:
    def __init__(self): self.stopped = False
    def is_set(self): return self.stopped
    def wait(self, _): self.stopped = True
    def set(self): self.stopped = True


def guard_fixture(tmp_path):
    owned = SimpleNamespace(environment={}, windows=[SimpleNamespace(window_id=r.window_id,
        pid=str(r.process_id), process_start=r.process_start) for r in (A, B)])
    deadline = SimpleNamespace(calls=0)
    def expire(): deadline.calls += 1
    deadline._expire = expire
    guard = runner.Guard(owned, deadline, tmp_path)
    return guard, deadline


@pytest.mark.parametrize("stopped", [False, True])
def test_guard_transition_or_normal_stop_never_collects_pointer_or_expires(tmp_path, monkeypatch, stopped):
    guard, deadline = guard_fixture(tmp_path)
    guard.stop = StopAfterWait()
    def observe(*_args, **_kwargs):
        if stopped:
            error = focus.FocusStopped()
            error.evidence = {"schema": focus.SCHEMA, "status": "stopped", "pointer_credit": False}
            raise error
        return sample(Reader())
    monkeypatch.setattr(runner, "sample_owned_focus", observe)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("Transition/stop cannot sample pointer"))
    guard.run()
    assert guard.failure is None and deadline.calls == 0 and not guard.positions and not guard.records
    assert guard.focus_sample_count == 1


def test_guard_stop_after_owned_sample_does_not_start_mouse_query(tmp_path, monkeypatch):
    guard, deadline = guard_fixture(tmp_path)
    def observe(*_args, **_kwargs):
        guard.stop.set()
        return sample(Reader(("17", "17"), stable=A))
    monkeypatch.setattr(runner, "sample_owned_focus", observe)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("Stopped guard cannot query mouse"))
    guard.run()
    assert not guard.positions and guard.failure is None and deadline.calls == 0


def test_guard_foreign_focus_still_expires_and_retains_typed_proof(tmp_path, monkeypatch):
    guard, deadline = guard_fixture(tmp_path)
    monkeypatch.setattr(runner, "sample_owned_focus", lambda *_a, **_k: sample(Reader(("21", "18", "18"))))
    guard.run()
    assert deadline.calls == 1 and guard.failure["code"] == "guard_focus_outside_owned_scope"
    assert guard.focus_samples[-1]["active_ids"] == ["21"] and not guard.positions


def test_guard_close_during_stalled_read_is_bounded_without_false_failure(tmp_path, monkeypatch):
    guard, deadline = guard_fixture(tmp_path)
    entered = threading.Event()
    class Stalled(Reader):
        def active(self, deadline):
            entered.set()
            # Model a read that honors its deadline but cannot be interrupted
            # before its blocking operation returns.
            time.sleep(max(0, deadline - time.monotonic()))
            raise subprocess.TimeoutExpired("controlled read", 2.5)
    def observe(environment, registered, **options):
        return focus.sample_owned_focus(environment, registered, reader=Stalled(), **options)
    monkeypatch.setattr(runner, "sample_owned_focus", observe)
    guard.thread.start()
    assert entered.wait(1)
    started = time.monotonic()
    guard.close()
    assert time.monotonic() - started < 3 and not guard.thread.is_alive()
    assert guard.shutdown == {"requested": True, "joined": True}
    assert guard.failure is None and deadline.calls == 0 and not guard.positions
    assert guard.focus_samples[-1]["status"] == "stopped"


@pytest.mark.parametrize("status", ["cleanup_deferred", "failed_model_request"])
def test_finalizer_retains_typed_focus_on_early_error_or_undrained_exit(tmp_path, monkeypatch, status):
    guard, _ = guard_fixture(tmp_path)
    guard.failure = {"code": "guard_focus_repeated_transition"}
    guard.focus_samples = [{"schema": focus.SCHEMA, "status": "unavailable", "active_ids": ["17", "18"]}]
    guard.focus_sample_count = 1
    monkeypatch.setattr(runner, "source_state", lambda: {})
    monkeypatch.setattr(runner, "worker_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner, "freeze_evidence", lambda *_a: None)
    report = {"status": status, "source_before": {}}
    runner.finalize_run(report, tmp_path, {}, desktop=None, observer=None, guard=guard,
        client=None, request_started=True, settled=False, started=time.monotonic())
    proof = report["ownership_guard"]
    assert proof["focus_samples"] == guard.focus_samples and proof["failure"] == guard.failure
    assert proof["shutdown"] == {"requested": True, "joined": True}
    assert proof["focus_sample_count"] == 1 and not report.get("passed")
