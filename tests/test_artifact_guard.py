"""Exercise destructive-operation refusals and storage admission before execution."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "artifact_guard.py"
spec = importlib.util.spec_from_file_location("artifact_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def plan_for(tmp_path, names=("old.f32",)):
    source = tmp_path / "retained-reconstruction.json"
    source.write_text('{"raw_audio_and_checkpoint":"retained"}')
    entries = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"derived disposable features")
        value = path.stat()
        entries.append({"path": str(path), "kind": "derived_feature_cache", "reason": "completed trial",
                        "sha256": guard.sha256(path), "dev": value.st_dev, "ino": value.st_ino,
                        "size": value.st_size, "mtime_ns": value.st_mtime_ns,
                        "allocated": value.st_blocks * 512,
                        "provenance": [{"path": str(source), "sha256": guard.sha256(source)}]})
    plan = {"schema": guard.SCHEMA, "owner": "test-completed-experiment",
            "allowed_roots": [str(tmp_path)], "protected": [], "entries": entries}
    return plan, source


def invoke(tmp_path, plan, apply=True):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    receipt = tmp_path / "receipt.json"
    args = ["reclaim", "--plan", str(path), "--receipt", str(receipt)]
    if apply:
        args.append("--apply")
    return guard.main(args), receipt


def test_dry_run_preserves_every_file(tmp_path):
    plan, source = plan_for(tmp_path)
    code, receipt = invoke(tmp_path, plan, apply=False)
    assert code == 0 and source.exists() and (tmp_path / "old.f32").exists()
    assert json.loads(receipt.read_text())["removed_files"] == 0


def test_apply_unlinks_only_inventoried_artifact_and_keeps_receipt(tmp_path):
    plan, source = plan_for(tmp_path)
    unlisted = tmp_path / "keep.f32"
    unlisted.write_bytes(b"keep")
    code, receipt = invoke(tmp_path, plan)
    assert code == 0 and not (tmp_path / "old.f32").exists()
    assert source.exists() and unlisted.read_bytes() == b"keep"
    assert json.loads(receipt.read_text())["removed_files"] == 1
    assert json.loads(receipt.with_suffix(".jsonl").read_text())["path"] == plan["entries"][0]["path"]


def test_one_changed_target_prevents_deleting_any_target(tmp_path):
    plan, _ = plan_for(tmp_path, ("first.f32", "second.f32"))
    (tmp_path / "second.f32").write_bytes(b"a later job changed this")
    code, _ = invoke(tmp_path, plan)
    assert code == 1 and (tmp_path / "first.f32").exists() and (tmp_path / "second.f32").exists()


@pytest.mark.parametrize("failure", ["protected", "shared", "symlink", "source_changed", "source_missing", "source_target", "weights", "broad_root", "busy"])
def test_unsafe_reclamation_is_refused(tmp_path, failure):
    plan, source = plan_for(tmp_path)
    target = tmp_path / "old.f32"
    opened = None
    if failure == "protected":
        plan["protected"] = [str(target)]
    elif failure == "shared":
        (tmp_path / "shared.f32").hardlink_to(target)
    elif failure == "symlink":
        target.unlink()
        target.symlink_to(source)
    elif failure == "source_changed":
        source.write_text("changed")
    elif failure == "source_missing":
        source.unlink()
    elif failure == "source_target":
        plan["entries"][0]["provenance"] = [{"path": str(target), "sha256": guard.sha256(target)}]
    elif failure == "weights":
        renamed = tmp_path / "adapter_model.safetensors"
        target.rename(renamed)
        plan["entries"][0]["path"] = str(renamed)
        target = renamed
    elif failure == "broad_root":
        plan["allowed_roots"] = ["/"]
    elif failure == "busy":
        opened = target.open("rb")
    try:
        code, _ = invoke(tmp_path, plan)
        assert code == 1 and target.exists()
    finally:
        if opened:
            opened.close()


def test_refuses_to_overwrite_a_receipt(tmp_path):
    plan, _ = plan_for(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("previous evidence")
    code, _ = invoke(tmp_path, plan)
    assert code == 1 and receipt.read_text() == "previous evidence"
    assert (tmp_path / "old.f32").exists()


def test_low_disk_budget_never_executes_workload(tmp_path):
    marker = tmp_path / "workload-started"
    code = guard.main(["check", "--path", str(tmp_path), "--reserve-gib", "1000000000",
                       "--expected-write-gib", "1", "--execute", sys.executable, "-c",
                       f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    assert code == 1 and not marker.exists()


def test_admitted_budget_executes_workload(tmp_path):
    marker = tmp_path / "started"
    result = subprocess.run([sys.executable, str(SCRIPT), "check", "--path", str(tmp_path),
                             "--reserve-gib", "0", "--expected-write-gib", "0", "--execute",
                             sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and marker.exists()


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_disk_estimates_cannot_admit_work(tmp_path, value):
    assert guard.main(["check", "--path", str(tmp_path), "--expected-write-gib", value]) == 1


def test_owned_artifact_budget_includes_forecast_writes(tmp_path):
    (tmp_path / "retained.bin").write_bytes(b"x" * 8192)
    assert guard.main(["check", "--path", str(tmp_path), "--reserve-gib", "0",
                       "--expected-write-gib", "0", "--owned-root", str(tmp_path),
                       "--max-owned-gib", "0"]) == 1


def test_ram_budget_prevents_workload_execution(tmp_path):
    marker = tmp_path / "memory-hungry-work-started"
    assert guard.main(["check", "--path", str(tmp_path), "--reserve-gib", "0",
                       "--expected-write-gib", "0", "--expected-ram-gib", "1000000000",
                       "--execute", sys.executable, "-c",
                       f"from pathlib import Path; Path({str(marker)!r}).touch()"])==1
    assert not marker.exists()


def test_ram_budget_uses_available_memory_instead_of_low_free_counter(tmp_path, capsys):
    assert guard.main(["check", "--path", str(tmp_path), "--reserve-gib", "0",
                       "--expected-write-gib", "0", "--expected-ram-gib", "0",
                       "--ram-reserve-gib", "0"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ram_available_bytes"] > 0 and result["ram_required_bytes"] == 0
