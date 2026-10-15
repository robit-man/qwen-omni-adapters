from __future__ import annotations

import pytest

from runtime import verify_real_files
from runtime.verify_real_files import CONTENTS, OwnedFiles, criteria, expected_tree, tree_receipt


@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--think"], True)])
def test_files_cli_preserves_native_think_default_and_explicit_flag(monkeypatch, flags, expected) -> None:
    captured = []
    monkeypatch.setattr(verify_real_files, "run", lambda args: captured.append(args) or 17)
    monkeypatch.setattr("sys.argv", ["verify_real_files", *flags])
    assert verify_real_files.main() == 17
    assert captured[0].think is expected


def test_files_readback_requires_exact_move_rename_and_unchanged_bytes(tmp_path) -> None:
    workbench = tmp_path / "Workbench"
    destination = workbench / "Destination"
    destination.mkdir(parents=True)
    (workbench / "alpha.txt").write_bytes(CONTENTS["alpha.txt"])
    (workbench / "beta.txt").write_bytes(CONTENTS["beta.txt"])
    assert tree_receipt(tmp_path) != expected_tree()
    (workbench / "alpha.txt").rename(destination / "copper.txt")
    assert tree_receipt(tmp_path) == expected_tree()
    # Perfect file bytes alone cannot replace actual selection/menu/drag evidence.
    checks = criteria([], {}, tree_receipt(tmp_path), "owned", tmp_path / "missing-pointer")
    assert checks["exact_filesystem_readback"]
    assert not all(checks.values())
    (destination / "copper.txt").write_bytes(b"silently different content\n")
    assert tree_receipt(tmp_path) != expected_tree()


def test_readback_never_reads_a_symlink_destination(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"unrelated data")
    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "link").symlink_to(outside)
    assert tree_receipt(owned) == [{"path": "link", "kind": "symlink"}]


def test_files_cleanup_rejects_pid_reuse_and_preexisting_window(monkeypatch) -> None:
    owned = OwnedFiles({})
    owned.before, owned.window, owned.pid, owned.start = {"old"}, "new", "20", "old-start"
    calls = []
    monkeypatch.setattr("runtime.verify_real_files.files_windows", lambda *_args, **_kwargs: {"old", "new"})
    monkeypatch.setattr("runtime.verify_real_files.command", lambda argv, _environment: calls.append(argv) or "20")
    monkeypatch.setattr("runtime.verify_real_files.process_start", lambda _pid: "reused-start")
    assert not owned.close()
    assert all("windowclose" not in call for call in calls)
    owned.window = "old"
    calls.clear()
    assert not owned.close()
    assert calls == []


def test_files_cleanup_closes_only_exact_owned_window_in_shared_dbus_process(monkeypatch) -> None:
    owned = OwnedFiles({})
    owned.before, owned.window, owned.pid, owned.start = {"old"}, "new", "20", "same-start"
    windows, calls = {"old", "new"}, []
    monkeypatch.setattr("runtime.verify_real_files.files_windows", lambda *_args, **_kwargs: set(windows))
    monkeypatch.setattr("runtime.verify_real_files.process_start", lambda _pid: "same-start")
    def command(argv, _environment):
        calls.append(argv)
        if argv[1] == "windowclose":
            windows.remove(argv[2])
        return "20"
    monkeypatch.setattr("runtime.verify_real_files.command", command)
    assert owned.close()
    assert windows == {"old"}
    assert calls == [["xdotool", "getwindowpid", "new"], ["xdotool", "windowclose", "new"]]
