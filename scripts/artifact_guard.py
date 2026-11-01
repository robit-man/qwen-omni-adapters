"""Bound build disk use and reclaim only explicitly inventoried disposable files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import time
from pathlib import Path

GIB = 1024**3
SCHEMA = "robit.omni.disposable-artifacts.v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def owned_bytes(roots: list[Path]) -> int:
    seen = set()
    total = 0
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Missing inventory root: {root}")
        for parent, directories, names in os.walk(root, followlinks=False):
            directories[:] = [n for n in directories if not (Path(parent) / n).is_symlink()]
            for name in names:
                value = (Path(parent) / name).lstat()
                identity = value.st_dev, value.st_ino
                if stat.S_ISREG(value.st_mode) and identity not in seen:
                    seen.add(identity)
                    total += value.st_blocks * 512
    return total


def check_budget(args: argparse.Namespace) -> None:
    for value in [args.reserve_gib, args.expected_write_gib, args.max_owned_gib,
                  args.expected_ram_gib, args.ram_reserve_gib]:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Storage budgets must be finite nonnegative GiB")
    free = shutil.disk_usage(existing_parent(args.path.absolute())).free
    required = (args.reserve_gib + args.expected_write_gib) * GIB
    result = {"path": str(args.path.absolute()), "free_bytes": free,
              "reserve_bytes": int(args.reserve_gib * GIB),
              "expected_write_bytes": int(args.expected_write_gib * GIB)}
    if args.expected_ram_gib is not None:
        meminfo = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
                   for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemAvailable:")}
        result["ram_available_bytes"] = meminfo["MemAvailable"]
        result["ram_required_bytes"] = int((args.expected_ram_gib + args.ram_reserve_gib) * GIB)
        if result["ram_available_bytes"] < result["ram_required_bytes"]:
            raise ValueError(f"Insufficient available RAM: {json.dumps(result)}")
    if args.owned_root:
        if args.max_owned_gib is None:
            raise ValueError("--owned-root requires --max-owned-gib")
        result["owned_allocated_bytes"] = owned_bytes(args.owned_root)
        result["owned_limit_bytes"] = int(args.max_owned_gib * GIB)
        if result["owned_allocated_bytes"] + args.expected_write_gib * GIB > args.max_owned_gib * GIB:
            raise ValueError(f"Owned artifact budget exceeded: {json.dumps(result)}")
    elif args.max_owned_gib is not None:
        raise ValueError("--max-owned-gib requires --owned-root")
    if free < required:
        raise ValueError(f"Insufficient disk headroom: {json.dumps(result)}")
    print(json.dumps({"storage_admitted": True, **result}), flush=True)
    if args.execute:
        os.execvp(args.execute[0], args.execute)


def checked_file(path: Path) -> os.stat_result:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f"Noncanonical or symlink path: {path}")
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError(f"Expected a regular unshared file: {path}")
    return value


def identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def busy_paths(paths: set[str]) -> set[str]:
    """Check visible descriptors and mappings without collecting process content."""
    busy = set()
    for process in Path("/proc").glob("[0-9]*"):
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if target in paths:
                    busy.add(target)
            with (process / "maps").open() as stream:
                for line in stream:
                    fields = line.rstrip().split(maxsplit=5)
                    if len(fields) == 6 and fields[5] in paths:
                        busy.add(fields[5])
        except (OSError, PermissionError):
            continue
    return busy


def verify_plan(plan: dict) -> list[tuple[Path, tuple[int, int, int, int]]]:
    if plan.get("schema") != SCHEMA or not plan.get("owner"):
        raise ValueError("Missing plan schema or owner")
    forbidden = {Path("/"), Path("/srv"), Path("/home"), Path.home(), Path.home() / ".cache"}
    roots = [Path(p) for p in plan["allowed_roots"]]
    if not roots or any(r in forbidden or len(r.parts) < 4 or not r.is_dir()
                        or r.resolve() != r for r in roots):
        raise ValueError("Cleanup roots must be explicit canonical task directories")
    entries = plan["entries"]
    targets = [Path(e["path"]) for e in entries]
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate cleanup target")
    protected = {Path(p) for p in plan.get("protected", [])}
    references = {Path(p["path"]) for e in entries for p in e["provenance"]}
    if set(targets) & (protected | references):
        raise ValueError("A cleanup target is protected or needed for reconstruction")
    checked = []
    verified = {}
    for entry, path in zip(entries, targets, strict=True):
        if not any(path != root and path.is_relative_to(root) for root in roots):
            raise ValueError(f"Target outside owned roots: {path}")
        if "blobs" in path.parts or ".git" in path.parts or "checkpoints" in path.parts:
            raise ValueError(f"Protected storage: {path}")
        kind = entry["kind"]
        allowed = (
            kind == "derived_feature_cache" and path.suffix in {".f32", ".npy", ".npz"}
            or kind == "materialized_component_view"
            and path.name in {"tts-model.gguf", "tts-projector.gguf", "tts_model.gguf",
                              "tts_projector.gguf", "tts_reference_encoder.gguf", "audio_ctc_head.gguf"}
            or kind == "rejected_projector_export" and path.name.endswith("-projector.gguf")
        )
        if not allowed or not entry.get("reason") or not entry["provenance"]:
            raise ValueError(f"Not an approved disposable artifact: {path}")
        value = checked_file(path)
        expected = entry["dev"], entry["ino"], entry["size"], entry["mtime_ns"]
        if identity(value) != expected or sha256(path) != entry["sha256"]:
            raise ValueError(f"Cleanup target changed: {path}")
        if identity(checked_file(path)) != expected:
            raise ValueError(f"Cleanup target changed while hashing: {path}")
        for proof in entry["provenance"]:
            source = Path(proof["path"])
            source_value = source.stat()
            source_identity = identity(source_value)
            key = source, source_identity
            if key not in verified:
                verified[key] = sha256(source)
            if verified[key] != proof["sha256"]:
                raise ValueError(f"Reconstruction evidence changed: {source}")
        checked.append((path, expected))
    active = busy_paths({str(p) for p in targets})
    if active:
        raise ValueError(f"Artifacts still open or mapped: {sorted(active)[:10]}")
    return checked


def reclaim(args: argparse.Namespace) -> None:
    if args.receipt.exists() or args.receipt.suffix != ".json":
        raise ValueError("Use a new .json receipt path")
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    checked = verify_plan(plan)
    report = {"schema": "robit.omni.artifact-reclamation.v1", "owner": plan["owner"],
              "plan": str(args.plan.absolute()), "plan_sha256": hashlib.sha256(raw).hexdigest(),
              "mode": "apply" if args.apply else "dry_run", "files": len(checked),
              "planned_allocated_bytes": sum(e["allocated"] for e in plan["entries"]),
              "removed_files": 0, "removed_allocated_bytes": 0, "complete": False,
              "free_before": {r: shutil.disk_usage(r).free for r in plan["allowed_roots"]}}
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if args.apply:
            journal = args.receipt.with_suffix(".jsonl")
            with journal.open("x") as stream:
                for entry, (path, expected) in zip(plan["entries"], checked, strict=True):
                    if identity(checked_file(path)) != expected:
                        raise ValueError(f"Target changed after verification: {path}")
                    path.unlink()
                    report["removed_files"] += 1
                    report["removed_allocated_bytes"] += entry["allocated"]
                    stream.write(json.dumps({"path": str(path), "sha256": entry["sha256"],
                                             "allocated": entry["allocated"]}) + "\n")
                    if report["removed_files"] % 128 == 0:
                        stream.flush()
                stream.flush()
                os.fsync(stream.fileno())
        report["complete"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        report["free_after"] = {r: shutil.disk_usage(r).free for r in plan["allowed_roots"]}
        report["finished_unix"] = time.time()
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Admit a build only within disk and artifact budgets")
    check.add_argument("--path", type=Path, required=True)
    check.add_argument("--reserve-gib", type=float, default=10)
    check.add_argument("--expected-write-gib", type=float, required=True)
    check.add_argument("--owned-root", type=Path, action="append", default=[])
    check.add_argument("--max-owned-gib", type=float)
    check.add_argument("--expected-ram-gib", type=float)
    check.add_argument("--ram-reserve-gib", type=float, default=32)
    check.add_argument("--execute", nargs=argparse.REMAINDER)
    clean = sub.add_parser("reclaim", help="Verify an explicit plan; default is dry run")
    clean.add_argument("--plan", type=Path, required=True)
    clean.add_argument("--receipt", type=Path, required=True)
    clean.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        check_budget(args) if args.command == "check" else reclaim(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Artifact guard refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
