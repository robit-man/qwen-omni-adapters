"""Offline suite loading, declared-input setup and independent artifact evidence.

No function launches desktop applications, networks or models. Explicit owned
media verification uses CPU decoders. The evaluator pack remains outside the
solver workbench and is never a user goal.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

DATA = Path(__file__).resolve().parents[1] / "tests/live/native_long_horizon"
MAX_FILE = 16 * 1024 * 1024
MAX_BUNDLE_FILE = 512 * 1024 * 1024
BUNDLE_DERIVED = {"evidence-manifest.json", "independent-verdict.json"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Expected a bounded regular JSON evidence file")
    return json.loads(path.read_bytes())


def bounded_bytes(path: Path, limit: int = MAX_FILE) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or not 0 <= before.st_size <= limit:
            raise ValueError("Expected bounded owner-owned regular evidence")
        content = source.read(limit + 1)
        after = os.fstat(source.fileno())
        if (len(content) > limit or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("Evidence changed during its bounded read")
        return content


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError("Invalid owned relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in {".", ".."} for part in relative.split("/")):
        raise ValueError("Owned paths cannot escape or use dot components")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError("Owned path contains a symlink or escape")
    return target


def load_case(case_id: str, data_root: Path = DATA) -> dict[str, Any]:
    manifest = read_json(data_root / "manifest.json")
    for name, expected in manifest["files"].items():
        path = safe_path(data_root, name)
        if sha(path.read_bytes()) != expected:
            raise ValueError("Packaged suite digest mismatch")
    suite = read_json(data_root / "tasks.json")
    cases = suite.get("tasks", [])
    if len(cases) != 100 or len({case["case_id"] for case in cases}) != 100:
        raise ValueError("Expected exactly 100 unique packaged tasks")
    case = next((case for case in cases if case["case_id"] == case_id), None)
    if case is None:
        raise ValueError("Unknown long-horizon case")
    return case


def materialize(case: dict[str, Any], work: Path, *, owned_media_cache: Path | None = None,
                owned_media_manifest_sha256: str | None = None) -> dict[str, str]:
    """Create declared input bytes only, in a brand-new owned workbench."""
    if ((owned_media_cache is not None or owned_media_manifest_sha256 is not None)
            and (owned_media_cache is None or owned_media_manifest_sha256 is None
                 or not case["fixture"].get("additional_asset_requirements"))):
        raise ValueError("Media materialization requires a declared case, cache and external manifest pin")
    work.mkdir(mode=0o700, exist_ok=False)
    inventory = {}
    for item in case["fixture"]["input_files"]:
        relative = item["path"]
        if not relative.startswith("Input/"):
            raise ValueError("A fixture may write declared inputs only")
        path = safe_path(work, relative)
        data = item["content_utf8"].encode()
        if len(data) > MAX_FILE or sha(data) != item["content_sha256"] or relative in inventory:
            raise ValueError("Invalid or duplicate declared fixture content")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as output:
            output.write(data)
        path.chmod(0o444 if item.get("mode_after_materialization") == "0444" else 0o600)
        inventory[relative] = sha(data)
    if owned_media_cache is not None:
        from runtime.long_horizon_media import materialize_media
        media, _ = materialize_media(case, work, owned_media_cache,
            expected_manifest_sha256=owned_media_manifest_sha256)
        if set(media) & set(inventory):
            raise ValueError("Media inputs overlap declared inline inputs")
        inventory.update(media)
    return inventory


def verified_media_inputs(case: dict, work: Path, cache: Path, manifest_sha256: str) -> dict:
    """Actual decoder capability plus independent copies; never a boolean waiver."""
    from runtime.long_horizon_media import verify_media_cache
    if cache.resolve().is_relative_to(work.resolve()):
        raise ValueError("Solver workbench cannot confer media capability")
    capability = verify_media_cache(case, cache, expected_manifest_sha256=manifest_sha256)
    for relative, asset in capability["assets"].items():
        target = safe_path(work, relative)
        source = safe_path(cache, Path(relative).name)
        if (sha(bounded_bytes(target)) != asset["sha256"] or target.stat().st_nlink != 1
                or (target.stat().st_dev, target.stat().st_ino) == (source.stat().st_dev, source.stat().st_ino)):
            raise ValueError("Media inputs are not independent exact materialized copies")
    for item in case["fixture"]["input_files"]:
        if sha(bounded_bytes(safe_path(work, item["path"]))) != item["content_sha256"]:
            raise ValueError("Declared inline media-case inputs changed")
    return {**capability, "work_root": str(work.resolve()), "materialized_inputs_verified": True}


def readiness_requirements(case: dict[str, Any]) -> set[str]:
    # Apps are verified from fresh owned native observations during setup.
    # Rubrics are already predeclared data; grading them is a post-run step.
    return {"source:" + item["source_id"] for item in case["sources_to_acquire"]}


def rubric_blockers(case: dict[str, Any]) -> list[str]:
    criteria = case.get("mandatory_criteria")
    if not isinstance(criteria, list) or not criteria:
        return ["missing_predeclared_criterion_rubric"]
    seen, blockers = set(), []
    for item in criteria:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]
                or item["id"] in seen or item.get("mandatory") is not True
                or not isinstance(item.get("assertion"), str) or not item["assertion"].strip()
                or not isinstance(item.get("independent_evaluator"), str) or not item["independent_evaluator"].strip()
                or not isinstance(item.get("expected"), dict)):
            blockers.append("invalid_predeclared_criterion_rubric")
        else:
            seen.add(item["id"])
    return blockers


def _source_artifact(case: dict[str, Any], artifact: dict[str, Any], key: str,
                     evidence_root: Path, work: Path) -> None:
    source = artifact.get("source")
    declared = next(item for item in case["sources_to_acquire"] if "source:" + item["source_id"] == key)
    if (not isinstance(source, dict) or set(source) != {"url", "topic_pin", "content_path", "content_sha256"}
            or not isinstance(source["topic_pin"], str) or not source["topic_pin"].strip()):
        raise ValueError("Missing frozen primary-source evidence")
    actual, expected = urlsplit(source["url"]), urlsplit(declared["url"])
    if (actual.scheme != "https" or actual.hostname != expected.hostname
            or actual.username or actual.password or actual.fragment):
        raise ValueError("Source authority differs from the declared primary origin")
    content = safe_path(evidence_root, source["content_path"])
    if (content.resolve().is_relative_to(work.resolve()) or content.stat().st_uid != os.getuid()
            or not content.is_file() or not 0 < content.stat().st_size <= MAX_FILE
            or sha(content.read_bytes()) != source["content_sha256"]):
        raise ValueError("Missing or changed frozen source bytes")


def source_readiness(case: dict[str, Any], proof: dict[str, Any] | None, *, revision: str,
                     work: Path, evidence_root: Path) -> list[str]:
    """Validate source prerequisites only; this does not establish fixture readiness."""
    blockers = rubric_blockers(case)
    if not isinstance(proof, dict):
        return [*blockers, "missing_case_bound_host_readiness_pack"]
    if (proof.get("schema") != "robit.omni.long-native-readiness.v1"
            or proof.get("case_id") != case["case_id"]
            or proof.get("task_signature_sha256") != case["task_signature_sha256"]
            or proof.get("source_revision") != revision):
        return [*blockers, "readiness_identity_or_source_mismatch"]
    required = readiness_requirements(case)
    receipts = proof.get("receipts")
    if not isinstance(receipts, list) or len(receipts) > 64:
        return [*blockers, "invalid_readiness_receipts"]
    seen = set()
    for item in receipts:
        if not isinstance(item, dict) or set(item) != {"id", "path", "sha256", "authority"}:
            blockers.append("invalid_readiness_receipt_shape")
            continue
        key = item["id"]
        if key not in required or key in seen or item["authority"] != "host_independent_evaluator":
            blockers.append("unbound_or_duplicate_readiness_receipt")
            continue
        try:
            path = safe_path(evidence_root, item["path"])
            if path.resolve().is_relative_to(work.resolve()) or path.stat().st_uid != os.getuid():
                raise ValueError("Solver input cannot confer evaluator authority")
            if path.stat().st_size > MAX_FILE or sha(path.read_bytes()) != item["sha256"]:
                raise ValueError("Readiness evidence changed")
            artifact = read_json(path)
            if (artifact.get("schema") != "robit.omni.long-native-prerequisite-proof.v1"
                    or artifact.get("case_id") != case["case_id"]
                    or artifact.get("task_signature_sha256") != case["task_signature_sha256"]
                    or artifact.get("source_revision") != revision or artifact.get("receipt_id") != key
                    or artifact.get("authority") != "host_independent_evaluator"
                    or artifact.get("status") != "verified"):
                raise ValueError("Unbound prerequisite artifact")
            _source_artifact(case, artifact, key, evidence_root, work)
        except (OSError, ValueError, TypeError):
            blockers.append("missing_or_changed_readiness_artifact:" + str(key))
            continue
        seen.add(key)
    blockers.extend("missing_readiness:" + key for key in sorted(required - seen))
    return blockers


def readiness(case: dict[str, Any], proof: dict[str, Any] | None, *, revision: str,
              work: Path, evidence_root: Path, owned_close_capability: Path | None = None,
              amendment_capability: Path | None = None, owned_media_cache: Path | None = None,
              owned_media_manifest_sha256: str | None = None) -> list[str]:
    """Check every technical prerequisite, including actual fixture capability."""
    blockers = []
    if case.get("controlled_perturbation"):
        from runtime.long_horizon_amendments import supported as amendment_case
        from runtime.long_horizon_amendments import validate_capability as valid_amendment
        from runtime.long_horizon_owned_close import supported as close_case
        from runtime.long_horizon_owned_close import validate_capability as valid_close
        if close_case(case) and owned_close_capability is not None:
            try:
                valid_close(case, owned_close_capability, work, revision=revision)
            except (OSError, ValueError, TypeError, KeyError):
                blockers.append("owned_close_capability_unverified")
        elif amendment_case(case) and amendment_capability is not None:
            try:
                valid = (not amendment_capability.resolve().is_relative_to(work.resolve())
                    and valid_amendment(case, json.loads(bounded_bytes(amendment_capability)),
                        work=work, revision=revision, initial_hashes={
                            item["path"]: item["content_sha256"] for item in case["fixture"]["input_files"]}))
            except (OSError, ValueError, TypeError, KeyError):
                valid = False
            if not valid:
                blockers.append("amendment_capability_unverified")
        else:
            blockers.append("controlled_perturbation_injector_not_implemented")
    if case["fixture"].get("additional_asset_requirements"):
        if owned_media_cache is None or owned_media_manifest_sha256 is None:
            blockers.append("declared_media_materializer_not_implemented")
        else:
            try:
                verified_media_inputs(case, work, owned_media_cache, owned_media_manifest_sha256)
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
                blockers.append("owned_media_capability_unverified")
    return blockers + source_readiness(case, proof, revision=revision, work=work, evidence_root=evidence_root)


def capture_source_proofs(case: dict, proof_path: Path, destination: Path, *, revision: str, work: Path,
                          owned_close_capability: Path | None = None, amendment_capability: Path | None = None,
                          owned_media_cache: Path | None = None,
                          owned_media_manifest_sha256: str | None = None) -> dict:
    """Copy already validated public-reference proof bytes outside Workbench."""
    destination.mkdir(mode=0o700, exist_ok=False)
    pack_bytes = bounded_bytes(proof_path)
    pack = json.loads(pack_bytes)
    if readiness(case, pack, revision=revision, work=work, evidence_root=proof_path.parent,
                 owned_close_capability=owned_close_capability, amendment_capability=amendment_capability,
                 owned_media_cache=owned_media_cache, owned_media_manifest_sha256=owned_media_manifest_sha256):
        raise ValueError("Source prerequisites changed before capture")
    captured = {"schema": "robit.omni.long-native-captured-sources.v1", "case_id": case["case_id"],
                "task_signature_sha256": case["task_signature_sha256"], "pack_sha256": sha(pack_bytes), "sources": []}
    (destination / "readiness-pack.json").write_bytes(pack_bytes)
    for index, receipt in enumerate(pack["receipts"]):
        artifact_bytes = bounded_bytes(safe_path(proof_path.parent, receipt["path"]))
        if sha(artifact_bytes) != receipt["sha256"]:
            raise ValueError("Source proof changed after prerequisite validation")
        artifact = json.loads(artifact_bytes)
        source = artifact["source"]
        content = bounded_bytes(safe_path(proof_path.parent, source["content_path"]))
        if sha(content) != source["content_sha256"]:
            raise ValueError("Frozen primary source changed before capture")
        proof_name, content_name = f"proof-{index:02}.json", f"source-{index:02}.bin"
        (destination / proof_name).write_bytes(artifact_bytes)
        (destination / content_name).write_bytes(content)
        captured["sources"].append({"id": receipt["id"], "url": source["url"], "topic_pin": source["topic_pin"],
            "proof_path": proof_name, "proof_sha256": sha(artifact_bytes),
            "content_path": content_name, "content_sha256": sha(content)})
    (destination / "index.json").write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")
    return captured


def evidence_manifest(root: Path, case_id: str, signature: str) -> dict:
    """Bind all immutable run evidence; only derived manifest/verdict are omitted."""
    paths = list(islice(root.rglob("*"), 4097))
    if root.is_symlink() or len(paths) > 4096:
        raise ValueError("Invalid or oversized evidence bundle")
    files, total = [], 0
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if path.is_dir():
            continue
        if relative in BUNDLE_DERIVED:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Invalid derived evidence artifact")
            continue
        content = bounded_bytes(path, MAX_BUNDLE_FILE)
        total += len(content)
        if total > 1536 * 1024 * 1024:
            raise ValueError("Evidence bundle exceeded its total byte bound")
        files.append({"path": relative, "bytes": len(content), "sha256": sha(content)})
    body = {"schema": "robit.omni.long-native-evidence-manifest.v1", "case_id": case_id,
            "task_signature_sha256": signature, "excluded_derived_artifacts": sorted(BUNDLE_DERIVED), "files": files}
    return {**body, "bundle_sha256": sha(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())}


def freeze_evidence(root: Path, case: dict) -> dict:
    manifest = evidence_manifest(root, case["case_id"], case["task_signature_sha256"])
    (root / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def file_evidence(work: Path, initial: dict[str, str], case: dict[str, Any]) -> dict[str, Any]:
    files, archives = {}, {}
    for index, path in enumerate(sorted(work.rglob("*"))):
        if index >= 1024:
            raise ValueError("Owned artifact inventory exceeded its bound")
        relative = path.relative_to(work).as_posix()
        safe_path(work, relative)
        if path.is_file():
            if path.stat().st_size > MAX_FILE:
                raise ValueError("Owned artifact exceeded its byte bound")
            files[relative] = {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
            if path.suffix == ".zip":
                try:
                    with ZipFile(path) as archive:
                        members = archive.infolist()
                        if len(members) > 256 or sum(member.file_size for member in members) > MAX_FILE:
                            raise ValueError("Archive exceeded inspection bounds")
                        for member in members:
                            safe_path(work, member.filename.rstrip("/"))
                        if any(stat.S_ISLNK(member.external_attr >> 16) for member in members):
                            raise ValueError("Archive symlink members are forbidden")
                        archives[relative] = {"files": [{"path": member.filename, "bytes": member.file_size,
                                               "sha256": sha(archive.read(member))} for member in members if not member.is_dir()],
                                              "directories": [member.filename.rstrip("/") for member in members if member.is_dir()]}
                except BadZipFile:
                    archives[relative] = {"error": "invalid_zip"}
    automatic = {}
    for criterion in case["mandatory_criteria"]:
        if criterion["kind"] == "archive_membership":
            contract = criterion["expected"]
            observed = archives.get(contract["archive_path"], {})
            entries = observed.get("files", [])
            expected = {item["member_path"]: files.get(item["source_path"], {}).get("sha256")
                        for item in contract["entries"]}
            parents = {parent.as_posix() for name in expected for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
            actual = {item["path"]: item["sha256"] for item in entries}
            valid = (bool(expected) and all(expected.values()) and actual == expected and len(actual) == len(entries)
                     and set(observed.get("directories", [])) <= parents and "error" not in observed)
            automatic[criterion["id"]] = {"passed": bool(valid), "evaluator": "prefrozen_exact_archive_members_and_bytes"}
            continue
        if criterion["kind"] == "version_authority" and "current_file_bindings" in criterion["expected"]:
            expected = {item["output_path"]: files.get(item["source_path"], {}).get("sha256")
                        for item in criterion["expected"]["current_file_bindings"]}
            allowed = set(expected) | {name for name in case["required_outputs"] if name.startswith("Current/")}
            actual = {name for name in files if name.startswith("Current/")}
            automatic[criterion["id"]] = {"passed": bool(expected) and all(expected.values()) and actual == allowed
                and all(files.get(name, {}).get("sha256") == value for name, value in expected.items()),
                "evaluator": "prefrozen_authoritative_current_set_and_exact_bytes"}
            continue
        if criterion["kind"] != "exact_edit":
            continue
        checks = {}
        for item in case["fixture"]["input_files"]:
            if not item["path"].startswith("Input/draft-"):
                continue
            draft = PurePosixPath(item["path"]).name
            expected = item["content_utf8"]
            for correction in criterion["expected"]["corrections"]:
                if correction["draft"] == draft:
                    before, after = correction["exact_before"], correction["exact_after"]
                    if expected.count(before) != 1:
                        raise ValueError("Exact-edit rubric lacks a unique declared source span")
                    expected = expected.replace(before, after, 1)
            name = "Edited/" + draft
            checks[name] = {"exact_bytes_match": files.get(name, {}).get("sha256") == sha(expected.encode())}
        automatic[criterion["id"]] = {"passed": bool(checks) and all(item["exact_bytes_match"] for item in checks.values()),
                                      "evaluator": "exact_declared_source_full_text_diff", "files": checks}
    return {"files": files, "archives": archives, "automatic_criterion_evidence": automatic,
            "inputs_unchanged": all(files.get(name, {}).get("sha256") == value for name, value in initial.items()),
            "required_outputs_present": {name: name in files for name in case["required_outputs"]},
            "semantic_source_layout_and_reopen_verdict": "pending_independent_evaluator"}


def _observed_states(root: Path) -> list[dict[str, Any]]:
    states = []
    journal = root / "ui-observations.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8", errors="replace").splitlines()[:600]:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("state"), dict):
                states.append(record["state"])
    final = root / "final-native.json"
    if final.is_file():
        try:
            record = json.loads(final.read_text(encoding="utf-8"))
        except ValueError:
            record = None
        if isinstance(record, dict) and isinstance(record.get("state"), dict):
            states.append(record["state"])
    return states


def progress_milestones(case: dict[str, Any], root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Prefrozen partial-progress facts from recorded independent observations.

    These never create completion credit and never enter independent_verdict.
    An input counts as read back only when its complete declared text appears in
    an observed control value; a source counts as loaded only when an observed
    document URL's host is a declared source host.
    """
    states = _observed_states(root)
    values = [node["value"] for state in states for node in state.get("nodes", [])
              if isinstance(node, dict) and isinstance(node.get("value"), str) and not node.get("sensitive")]
    inputs = {}
    for item in case["fixture"]["input_files"]:
        text = item["content_utf8"].strip()
        inputs[item["path"]] = bool(text) and any(text in value for value in values)
    hosts = {urlsplit(source["url"]).hostname for source in case.get("sources_to_acquire", [])
             if isinstance(source.get("url"), str)}
    urls = {state.get("document_evidence", {}).get("url") for state in states
            if state.get("document_evidence", {}).get("status") == "observed"}
    loaded = sorted(url for url in urls if isinstance(url, str) and urlsplit(url).hostname in hosts)
    outputs = evidence.get("required_outputs_present", {})
    return {"schema": "robit.omni.long-native-progress-milestones.v1",
            "authority": "progress_only_no_completion_credit", "observed_states": len(states),
            "inputs_read_back": inputs, "official_source_urls_loaded": loaded[:16],
            "required_outputs_created": sum(value is True for value in outputs.values()),
            "required_outputs_total": len(outputs)}


def independent_verdict(case: dict[str, Any], evidence: dict[str, Any], *,
                        method_verified: bool, review: dict[str, Any] | None = None,
                        bundle_sha256: str = "") -> dict[str, Any]:
    """No model completion claim, filename or exit status can create a pass."""
    outputs = evidence.get("required_outputs_present", {})
    structural = (evidence.get("inputs_unchanged") is True and isinstance(outputs, dict)
                  and set(outputs) == set(case["required_outputs"]) and all(value is True for value in outputs.values())
                  and all(name in evidence.get("files", {}) for name in case["required_outputs"]))
    required = {item["id"] for item in case["mandatory_criteria"]}
    if not structural or not method_verified:
        return {"status": "failed_independent_checks", "passed": False}
    automatic = evidence.get("automatic_criterion_evidence", {})
    if any(item.get("passed") is not True for item in automatic.values()):
        return {"status": "failed_independent_checks", "passed": False, "automatic_criterion_evidence": automatic}
    if (not isinstance(review, dict) or review.get("schema") != "robit.omni.long-native-independent-review.v1"
            or review.get("case_id") != case["case_id"] or not bundle_sha256
            or review.get("bundle_sha256") != bundle_sha256
            or review.get("authority") != "host_independent_evaluator"):
        return {"status": "pending_independent_review", "passed": False}
    grades = review.get("criteria")
    if (not isinstance(grades, dict) or set(grades) != required
            or any(type(value) is not bool for value in grades.values())):
        return {"status": "invalid_independent_review", "passed": False}
    passed = all(grades.values())
    return {"status": "passed_independent_review" if passed else "failed_independent_review", "passed": passed,
            "criterion_grades": grades, "review_authority": "host_independent_evaluator"}
