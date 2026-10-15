"""CPU-only owned Calc launch guards; no app/desktop/model is started."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from runtime import acceptance_native_support as support


@pytest.fixture
def office(tmp_path, monkeypatch):
    program = tmp_path/"program"
    program.mkdir()
    executable, launcher = program/"soffice.bin", program/"soffice"
    executable.touch()
    launcher.touch()
    proc = tmp_path/"proc"
    proc.mkdir()
    monkeypatch.setattr(support, "process_start", lambda pid: (proc/pid/"start").read_text())
    monkeypatch.setattr(support.shutil, "which", lambda name: str(launcher) if name == "libreoffice" else None)
    def add(pid="101", *, env=b"SAL_DISABLE_OPENCL=1\0", binary=None, comm="soffice.bin", start="stable"):
        path = proc/pid
        path.mkdir()
        (path/"comm").write_text(comm+"\n")
        (path/"start").write_text(start)
        (path/"exe").symlink_to(binary or executable)
        (path/"environ").write_bytes(env)
        return path
    return executable, launcher, proc, add


def test_spreadsheet_environment_is_per_child_and_removes_force_override(monkeypatch):
    monkeypatch.setenv("SC_FORCE_CALCULATION", "opencl")
    original = {"DISPLAY": ":0", "SAL_DISABLE_OPENCL": "0", "SC_FORCE_CALCULATION": "opencl", "OTHER": "retained"}
    child = support._spreadsheet_environment(original)
    assert child == {"DISPLAY": ":0", "SAL_DISABLE_OPENCL": "1", "OTHER": "retained"}
    assert original["SAL_DISABLE_OPENCL"] == "0" and original["SC_FORCE_CALCULATION"] == "opencl"
    assert os.environ["SC_FORCE_CALCULATION"] == "opencl"
    child["OTHER"] = "child-only"
    assert original["OTHER"] == "retained"


def test_office_kernel_filter_reads_no_unrelated_environment(office):
    _exe, _launcher, proc, add = office
    add("101")
    unrelated = add("102", comm="other")
    (unrelated/"exe").unlink()
    (unrelated/"environ").unlink()
    assert support._spreadsheet_process_ids(proc) == ["101"]


def test_exact_office_executable_and_only_relevant_safe_env_are_verified(office):
    executable, _launcher, proc, add = office
    add(env=b"UNRELATED_SECRET=never surfaced\0SAL_DISABLE_OPENCL=1\0DISPLAY=:0\0")
    support._verify_spreadsheet_process("101", executable, expected_start="stable", proc_root=proc)


@pytest.mark.parametrize("environment", [
    b"DISPLAY=:0\0", b"SAL_DISABLE_OPENCL=0\0", b"SAL_DISABLE_OPENCL=\0",
    b"SAL_DISABLE_OPENCL=1\0SC_FORCE_CALCULATION=opencl\0",
    b"SAL_DISABLE_OPENCL=1\0SC_FORCE_CALCULATION=\0",
    b"SAL_DISABLE_OPENCL=1\0SAL_DISABLE_OPENCL=1\0",
])
def test_unsafe_existing_office_environment_fails_closed_without_dumping_secrets(office, environment):
    executable, _launcher, proc, add = office
    add(env=environment+b"UNRELATED_SECRET=hidden-data\0")
    with pytest.raises(support.CapabilityBlocked) as error:
        support._verify_spreadsheet_process("101", executable, proc_root=proc)
    assert "hidden-data" not in str(error.value) and "UNRELATED_SECRET" not in str(error.value)


def test_wrong_executable_or_bound_process_start_rejects(office):
    executable, _launcher, proc, add = office
    impostor = executable.with_name("other.bin")
    impostor.touch()
    add(binary=impostor)
    with pytest.raises(support.CapabilityBlocked, match="executable or identity"):
        support._verify_spreadsheet_process("101", executable, proc_root=proc)
    with pytest.raises(support.CapabilityBlocked, match="executable or identity"):
        support._verify_spreadsheet_process("101", executable, expected_start="reused", proc_root=proc)


def test_process_start_change_during_read_rejects(office, monkeypatch):
    executable, _launcher, proc, add = office
    add()
    reads = iter(("initial", "reused"))
    monkeypatch.setattr(support, "process_start", lambda _pid: next(reads))
    with pytest.raises(support.CapabilityBlocked, match="identity changed"):
        support._verify_spreadsheet_process("101", executable, proc_root=proc)


@pytest.mark.parametrize("missing", ["exe", "environ"])
def test_unreadable_matching_process_cannot_authorize_handoff(office, missing):
    executable, _launcher, proc, add = office
    process = add()
    (process/missing).unlink()
    with pytest.raises(support.CapabilityBlocked, match="proof is unavailable"):
        support._verify_spreadsheet_process("101", executable, proc_root=proc)


def test_environment_read_budget_is_bounded(office):
    executable, _launcher, proc, add = office
    add(env=b"SAL_DISABLE_OPENCL=1\0OTHER="+b"x"*131072)
    with pytest.raises(support.CapabilityBlocked, match="bounded read"):
        support._verify_spreadsheet_process("101", executable, proc_root=proc)


def prepare_launcher(office, monkeypatch, *, existing=(), child_env=b"SAL_DISABLE_OPENCL=1\0"):
    _executable, _launcher, proc, add = office
    add("101", env=child_env)
    calls = []
    original_verify = support._verify_spreadsheet_process
    monkeypatch.setattr(support, "_spreadsheet_process_ids", lambda: list(existing))
    def verify(pid, executable, **kwargs):
        calls.append(("verify", pid, kwargs))
        return original_verify(pid, executable, proc_root=proc, **kwargs)
    monkeypatch.setattr(support, "_verify_spreadsheet_process", verify)
    owned = support.OwnedDesktop({"DISPLAY": ":test", "SC_FORCE_CALCULATION": "opencl", "OTHER": "retained"})
    def launch(*args, **kwargs):
        calls.append(("launch", args, kwargs))
        window = support.OwnedWindow("libreoffice-calc", "202", "101", "stable")
        owned.windows.append(window)
        return window
    monkeypatch.setattr(owned, "open_application", launch)
    return owned, calls, add


def test_spreadsheet_launcher_uses_exact_class_and_safe_child_then_bound_process_check(office, monkeypatch):
    owned, calls, _add = prepare_launcher(office, monkeypatch)
    assert owned.open_spreadsheet().window_id == "202"
    assert calls[0][0] == "launch" and calls[0][1] == (
        "libreoffice", ["--calc", "--norestore"], "libreoffice-calc", "^libreoffice-calc$")
    child = calls[0][2]["child_environment"]
    assert child["SAL_DISABLE_OPENCL"] == "1" and "SC_FORCE_CALCULATION" not in child
    assert owned.environment["SC_FORCE_CALCULATION"] == "opencl"
    assert calls[1] == ("verify", "101", {"expected_start": "stable"})


def test_unsafe_existing_process_blocks_before_any_application_launch(office, monkeypatch):
    owned, calls, add = prepare_launcher(office, monkeypatch, existing=("99",))
    add("99", env=b"SC_FORCE_CALCULATION=opencl\0")
    with pytest.raises(support.CapabilityBlocked, match="OpenCL-disabled"):
        owned.open_spreadsheet()
    assert all(call[0] != "launch" for call in calls) and owned.windows == []


def test_safe_existing_process_is_verified_before_launch(office, monkeypatch):
    owned, calls, add = prepare_launcher(office, monkeypatch, existing=("99",))
    add("99")
    owned.open_spreadsheet()
    assert [call[0] for call in calls] == ["verify", "launch", "verify"]


def test_unsafe_actual_window_process_stays_registered_for_exact_cleanup(office, monkeypatch):
    owned, calls, _add = prepare_launcher(office, monkeypatch, child_env=b"DISPLAY=:0\0")
    with pytest.raises(support.CapabilityBlocked, match="OpenCL-disabled"):
        owned.open_spreadsheet()
    assert [call[0] for call in calls] == ["launch", "verify"]
    assert owned.window_ids == {"202"}  # The caller's existing finally closes only this new window.


def test_generic_application_launch_copies_explicit_child_environment(monkeypatch):
    launched, queries = [], []
    parent = {"DISPLAY": ":parent", "UNCHANGED": "yes"}
    child = {"DISPLAY": ":parent", "SAL_DISABLE_OPENCL": "1"}
    owned = support.OwnedDesktop(parent)
    monkeypatch.setattr(support.shutil, "which", lambda _name: "/usr/bin/app")
    monkeypatch.setattr(support, "process_start", lambda _pid: "stable")
    def command(args, environment, **_kwargs):
        queries.append((args, environment))
        if args[1] == "search":
            return "202" if "--onlyvisible" in args else ""
        return "101" if args[1] == "getwindowpid" else ""
    monkeypatch.setattr(support, "command", command)
    def launch(args, **kwargs):
        launched.append(kwargs["env"])
        return SimpleNamespace(pid=101, poll=lambda: None)
    monkeypatch.setattr(support.subprocess, "Popen", launch)
    owned.open_application("app", [], "app", "class", child_environment=child)
    assert launched[0] == child and launched[0] is not child and launched[0] is not parent
    assert all(environment is parent for _args, environment in queries)
    launched[0]["SAL_DISABLE_OPENCL"] = "changed"
    assert child["SAL_DISABLE_OPENCL"] == "1" and "SAL_DISABLE_OPENCL" not in parent


def test_missing_installed_office_binary_blocks_before_launch(monkeypatch):
    monkeypatch.setattr(support.shutil, "which", lambda _name: None)
    owned = support.OwnedDesktop({"DISPLAY": ":test"})
    monkeypatch.setattr(owned, "open_application", lambda *_a, **_k: pytest.fail("No app may launch"))
    with pytest.raises(support.CapabilityBlocked, match="unavailable"):
        owned.open_spreadsheet()


def test_cu030_setup_uses_guarded_spreadsheet_launcher(tmp_path):
    from runtime import verify_extended_native as runner

    calls = []
    def open_spreadsheet(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(window_id="202")
    owned = SimpleNamespace(open_spreadsheet=open_spreadsheet)
    observer = SimpleNamespace(fresh=lambda: {
        "window_id": "202", "state": {"scope": {"id": "owned", "window_id": "202"}, "nodes": []}})
    result = runner.setup("CU-030", tmp_path, owned, observer)
    assert result["primary_window"] == "202" and calls == [{"log_path": tmp_path/"calc-stderr.log"}]
    assert result["manifest_before"] == []
