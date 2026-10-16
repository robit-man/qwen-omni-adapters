"""CPU data, ownership, cancellation and independent-verification controls."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from zipfile import ZipFile

import httpx
import pytest

from runtime import long_horizon_cases as cases
from runtime import long_horizon_desktop as desktop
from runtime import verify_long_horizon_native as runner


@pytest.fixture
def case():
    return cases.load_case("LHN-047")


def test_exact_packaged_suite_is_unrun_and_unique():
    suite = cases.read_json(cases.DATA / "tasks.json")
    assert len(suite["tasks"]) == 100
    assert len({item["family_id"] for item in suite["tasks"]}) == 16
    assert len({item["task_signature_sha256"] for item in suite["tasks"]}) == 100
    assert all(not item["ready"] and not item["passed"] for item in suite["tasks"])
    assert all(cell["attempts"] == 0 for item in suite["tasks"] for cell in item["model_cells"].values())
    assert all(cases.rubric_blockers(item) == [] for item in suite["tasks"])


def test_every_case_supplies_public_urls_paths_and_bound_task_brief():
    suite = cases.read_json(cases.DATA / "tasks.json")
    assert sum(len(case["fixture"]["input_files"]) for case in suite["tasks"]) == 454
    for case in suite["tasks"]:
        brief = next(item for item in case["fixture"]["input_files"] if item["path"] == "Input/task-brief.txt")
        assert brief["mode_after_materialization"] == "0444"
        assert cases.sha(brief["content_utf8"].encode()) == brief["content_sha256"]
        assert "Input/task-brief.txt" in case["goal_text"]
        assert all(source["url"] in brief["content_utf8"] and source["authority"] in brief["content_utf8"]
                   for source in case["sources_to_acquire"])
        assert all(name in brief["content_utf8"] for name in case["required_outputs"])
        expected = cases.sha(json.dumps({key: case[key] for key in
            ("goal_text", "fixture", "mandatory_criteria")}, sort_keys=True, ensure_ascii=False).encode())
        assert expected == case["task_signature_sha256"]
        if case["family_id"] == "LHT-07":
            assert all(item["output_path"] not in brief["content_utf8"]
                       for item in case["sealed_evaluator_only"]["current_file_bindings"])


def test_all_archive_rubrics_are_prefrozen_without_recursive_or_private_release_members():
    count = 0
    for case in cases.read_json(cases.DATA / "tasks.json")["tasks"]:
        if not (contract := case.get("archive_contract")):
            continue
        count += 1
        entries = contract["entries"]
        members = [item["member_path"] for item in entries]
        assert len(members) == len(set(members)) and contract["archive_path"] not in members
        criterion = next(item for item in case["mandatory_criteria"] if item["kind"] == "archive_membership")
        assert criterion["expected"] == contract
        assert "freeze exact" not in json.dumps(contract)
        if case["family_id"] == "LHT-06":
            assert set(members) == {"Public/public-register.csv", "Public/public-summary.pdf"}
        if case["family_id"] == "LHT-16":
            assert "Archive/handoff.odt" not in members and len(members) == 10
        if case["family_id"] == "LHT-07":
            assert len(members) == 12 and all(name.startswith("Superseded/") for name in members)
    assert count == 75


def test_implementation_metadata_is_precise_without_readiness_or_pass_claims():
    suite = cases.read_json(cases.DATA / "tasks.json")
    implemented = []
    for case in suite["tasks"]:
        assert case["fixture"]["materializer_implementation"]["declared_inline_inputs"] == "implemented"
        assert case["status"] == "unrun_specification" and not case["ready"] and not case["passed"]
        assert "Fixture materializer and independent oracle not implemented/frozen" not in case["prerequisite_readiness"]["shared_blockers"]
        for item in case["mandatory_criteria"]:
            if item["oracle_implemented"]:
                implemented.append(item["kind"])
                assert item["kind"] in {"exact_edit", "archive_membership", "version_authority"}
            assert not item["verified"]
    assert implemented.count("exact_edit") == 6 and implemented.count("archive_membership") == 75
    assert implemented.count("version_authority") == 6


def test_fixture_setup_contains_only_exact_declared_inputs(case, tmp_path):
    work = tmp_path / "Workbench"
    inventory = cases.materialize(case, work)
    assert len(inventory) == len(case["fixture"]["input_files"])
    assert {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()} == set(inventory)
    assert all(not (work / name).exists() for name in case["required_outputs"])
    for item in case["fixture"]["input_files"]:
        assert (work / item["path"]).read_bytes() == item["content_utf8"].encode()
    with pytest.raises(FileExistsError):
        cases.materialize(case, work)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "Input/../output", "Input/./name", "Input\\name", "Input/\0name", ""])
def test_owned_paths_reject_escapes(tmp_path, path):
    with pytest.raises(ValueError):
        cases.safe_path(tmp_path, path)


def test_owned_paths_and_evidence_reject_symlinks(tmp_path, case):
    work = tmp_path / "Workbench"
    initial = cases.materialize(case, work)
    (work / "linked").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        cases.file_evidence(work, initial, case)


@pytest.mark.parametrize("mutation", ["digest", "output", "duplicate"])
def test_invalid_seed_cannot_create_goal_output(case, tmp_path, mutation):
    bad = copy.deepcopy(case)
    item = bad["fixture"]["input_files"][0]
    if mutation == "digest":
        item["content_sha256"] = "0" * 64
    elif mutation == "output":
        item["path"] = "Research/solved.txt"
    else:
        bad["fixture"]["input_files"].append(item)
    with pytest.raises(ValueError):
        cases.materialize(bad, tmp_path / "Workbench")
    assert not (tmp_path / "Research/solved.txt").exists()


def proof_pack(case, root, revision="frozen"):
    ids = cases.readiness_requirements(case)
    receipts = []
    for index, key in enumerate(sorted(ids)):
        declared = next(item for item in case["sources_to_acquire"] if key == "source:" + item["source_id"])
        content = root / (str(index) + ".html")
        content.write_text("Mock pinned primary-source bytes; no network access")
        path = root / (str(index) + ".json")
        path.write_text(json.dumps({"schema": "robit.omni.long-native-prerequisite-proof.v1", "case_id": case["case_id"],
            "task_signature_sha256": case["task_signature_sha256"], "source_revision": revision,
            "receipt_id": key, "authority": "host_independent_evaluator", "status": "verified",
            "source": {"url": declared["url"], "topic_pin": "Explicit mock rubric source section/version",
                       "content_path": content.name, "content_sha256": cases.sha(content.read_bytes())}}))
        receipts.append({"id": key, "path": path.name, "sha256": cases.sha(path.read_bytes()), "authority": "host_independent_evaluator"})
    return {"schema": "robit.omni.long-native-readiness.v1", "case_id": case["case_id"],
            "task_signature_sha256": case["task_signature_sha256"], "source_revision": revision, "receipts": receipts}


def test_readiness_requires_case_source_bound_host_artifacts(case, tmp_path):
    work = tmp_path / "Workbench"
    assert cases.readiness(case, None, revision="frozen", work=work, evidence_root=tmp_path)
    proof = proof_pack(case, tmp_path)
    assert cases.readiness(case, proof, revision="frozen", work=work, evidence_root=tmp_path) == []
    assert all(key.startswith("source:") for key in cases.readiness_requirements(case))
    assert any(not item["oracle_implemented"] for item in case["mandatory_criteria"])
    assert cases.readiness(case, proof, revision="other", work=work, evidence_root=tmp_path)
    proof["receipts"][0]["authority"] = "model_claim"
    assert cases.readiness(case, proof, revision="frozen", work=work, evidence_root=tmp_path)


def test_pending_semantic_grading_does_not_block_technically_ready_transport(case, tmp_path):
    proof = proof_pack(case, tmp_path)
    assert cases.rubric_blockers(case) == []
    assert cases.readiness(case, proof, revision="frozen", work=tmp_path / "Workbench", evidence_root=tmp_path) == []
    assert all(not item["verified"] for item in case["mandatory_criteria"])


@pytest.mark.parametrize("mutation", ["missing_rubric", "blank_assertion", "duplicate_criterion", "missing_expected"])
def test_incomplete_predeclared_rubric_remains_precise_blocker(case, tmp_path, mutation):
    bad = copy.deepcopy(case)
    if mutation == "missing_rubric":
        bad["mandatory_criteria"] = []
    elif mutation == "blank_assertion":
        bad["mandatory_criteria"][0]["assertion"] = " "
    elif mutation == "duplicate_criterion":
        bad["mandatory_criteria"].append(bad["mandatory_criteria"][0])
    else:
        bad["mandatory_criteria"][0].pop("expected")
    proof = proof_pack(bad, tmp_path)
    assert any("criterion_rubric" in item for item in cases.readiness(bad, proof, revision="frozen",
               work=tmp_path / "Workbench", evidence_root=tmp_path))


@pytest.mark.parametrize("mutation", ["source_bytes", "source_origin", "source_pin", "source_in_workbench"])
def test_source_readiness_requires_actual_frozen_bytes_and_declared_origin(case, tmp_path, mutation):
    proof = proof_pack(case, tmp_path)
    path = tmp_path / proof["receipts"][0]["path"]
    artifact = cases.read_json(path)
    if mutation == "source_bytes":
        (tmp_path / artifact["source"]["content_path"]).write_text("changed")
    elif mutation == "source_origin":
        artifact["source"]["url"] = "https://unrelated.example/"
    elif mutation == "source_pin":
        artifact["source"]["topic_pin"] = " "
    else:
        (tmp_path / "Workbench").mkdir()
        content = tmp_path / artifact["source"]["content_path"]
        content.rename(tmp_path / "Workbench" / content.name)
        artifact["source"]["content_path"] = "Workbench/" + content.name
    path.write_text(json.dumps(artifact))
    proof["receipts"][0]["sha256"] = cases.sha(path.read_bytes())
    assert cases.readiness(case, proof, revision="frozen", work=tmp_path / "Workbench", evidence_root=tmp_path)


def test_changed_or_arbitrary_readiness_file_is_not_proof(case, tmp_path):
    proof = proof_pack(case, tmp_path)
    path = tmp_path / proof["receipts"][0]["path"]
    path.write_text('{"status":"verified"}')
    proof["receipts"][0]["sha256"] = cases.sha(path.read_bytes())
    assert cases.readiness(case, proof, revision="frozen", work=tmp_path / "Workbench", evidence_root=tmp_path)


def test_readiness_artifacts_must_be_outside_solver_inputs(case, tmp_path):
    work = tmp_path / "Workbench"
    work.mkdir()
    proof = proof_pack(case, work)
    assert cases.readiness(case, proof, revision="frozen", work=work, evidence_root=work)


@pytest.mark.parametrize("case_id,blocker", [("LHN-077", "controlled_perturbation"), ("LHN-083", "media_materializer")])
def test_unimplemented_media_and_injectors_remain_explicit_blockers(tmp_path, case_id, blocker):
    case = cases.load_case(case_id)
    proof = proof_pack(case, tmp_path)
    assert blocker in cases.readiness(case, proof, revision="frozen", work=tmp_path / "Workbench", evidence_root=tmp_path)[0]


def complete_files(case, tmp_path):
    work = tmp_path / "Workbench"
    initial = cases.materialize(case, work)
    for name in case["required_outputs"]:
        path = cases.safe_path(work, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".zip":
            continue
        else:
            path.write_text("Not a semantically verified answer")
    # CPU oracle controls only: host fixture construction is never a live task
    # completion path. Deliberately keep the other deliverables ungraded.
    for item in case["fixture"]["input_files"]:
        if not item["path"].startswith("Input/draft-"):
            continue
        text = item["content_utf8"]
        name = item["path"].split("/")[-1]
        for correction in case["sealed_evaluator_only"]["corrections"]:
            if correction["draft"] == name:
                text = text.replace(correction["exact_before"], correction["exact_after"])
        (work / "Edited" / name).write_text(text)
    if contract := case.get("archive_contract"):
        with ZipFile(work / contract["archive_path"], "w") as archive:
            for item in contract["entries"]:
                archive.write(work / item["source_path"], item["member_path"])
    return work, initial


def test_file_existence_and_model_done_do_not_pass(case, tmp_path):
    work, initial = complete_files(case, tmp_path)
    evidence = cases.file_evidence(work, initial, case)
    result = cases.independent_verdict(case, evidence, method_verified=True,
                                      review={"message": "Done", "passed": True}, bundle_sha256="bundle")
    assert result == {"status": "pending_independent_review", "passed": False}
    assert cases.independent_verdict(case, evidence, method_verified=False)["passed"] is False


def test_review_must_bind_all_criteria_and_actual_bundle(case, tmp_path):
    work, initial = complete_files(case, tmp_path)
    evidence = cases.file_evidence(work, initial, case)
    review = {"schema": "robit.omni.long-native-independent-review.v1", "case_id": case["case_id"],
              "authority": "host_independent_evaluator", "bundle_sha256": "exact",
              "criteria": {item["id"]: True for item in case["mandatory_criteria"]}}
    assert cases.independent_verdict(case, evidence, method_verified=True, review=review, bundle_sha256="exact")["passed"]
    assert not cases.independent_verdict(case, evidence, method_verified=True, review=review, bundle_sha256="changed")["passed"]
    review["criteria"].pop(next(iter(review["criteria"])))
    assert not cases.independent_verdict(case, evidence, method_verified=True, review=review, bundle_sha256="exact")["passed"]


@pytest.mark.parametrize("mutation", ["protected_quote", "similar_distractor", "one_unapplied_edit", "extra_newline"])
def test_editorial_oracle_rejects_changes_outside_exact_ten_spans(case, tmp_path, mutation):
    work, initial = complete_files(case, tmp_path)
    evidence = cases.file_evidence(work, initial, case)
    criterion = case["mandatory_criteria"][0]["id"]
    assert evidence["automatic_criterion_evidence"][criterion]["passed"]
    path = work / "Edited/draft-1.txt"
    text = path.read_text()
    if mutation == "protected_quote":
        text = text.replace("colour", "color")
    elif mutation == "similar_distractor":
        text = text.replace("Non-target similar sentence 1: provisional value 101.", "Accidentally edited similar line.")
    elif mutation == "one_unapplied_edit":
        text = text.replace("Target sentence 1: confirmed value 201.", "Target sentence 1: provisional value 101.")
    else:
        text += "\n"
    path.write_text(text)
    evidence = cases.file_evidence(work, initial, case)
    assert not evidence["automatic_criterion_evidence"][criterion]["passed"]
    review = {"schema": "robit.omni.long-native-independent-review.v1", "case_id": case["case_id"],
              "authority": "host_independent_evaluator", "bundle_sha256": "exact",
              "criteria": {item["id"]: True for item in case["mandatory_criteria"]}}
    assert not cases.independent_verdict(case, evidence, method_verified=True, review=review, bundle_sha256="exact")["passed"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrapped_folder", "wrong_bytes", "unrelated_directory"])
def test_prefrozen_archive_oracle_rejects_wrong_membership_or_bytes(case, tmp_path, mutation):
    work, initial = complete_files(case, tmp_path)
    contract = case["archive_contract"]
    criterion = next(item["id"] for item in case["mandatory_criteria"] if item["kind"] == "archive_membership")
    assert cases.file_evidence(work, initial, case)["automatic_criterion_evidence"][criterion]["passed"]
    with ZipFile(work / contract["archive_path"], "w") as archive:
        for index, item in enumerate(contract["entries"]):
            if mutation == "missing" and index == 0:
                continue
            name = item["member_path"]
            if mutation == "wrapped_folder":
                name = "Unexpected/" + name
            content = (work / item["source_path"]).read_bytes()
            archive.writestr(name, content + b"changed" if mutation == "wrong_bytes" and index == 0 else content)
            if mutation == "duplicate" and index == 0:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, content)
        if mutation == "extra":
            archive.writestr("unrelated.txt", b"extra")
        if mutation == "unrelated_directory":
            archive.writestr("Unrelated/", b"")
    assert not cases.file_evidence(work, initial, case)["automatic_criterion_evidence"][criterion]["passed"]


def test_authoritative_current_set_is_checked_without_revealing_it_in_public_brief(tmp_path):
    case = cases.load_case("LHN-041")
    work, initial = complete_files(case, tmp_path)
    criterion = next(item for item in case["mandatory_criteria"] if item["kind"] == "version_authority")
    for item in criterion["expected"]["current_file_bindings"]:
        (work / item["output_path"]).write_bytes((work / item["source_path"]).read_bytes())
    assert cases.file_evidence(work, initial, case)["automatic_criterion_evidence"][criterion["id"]]["passed"]
    (work / "Current/not-selected.txt").write_text("extra")
    assert not cases.file_evidence(work, initial, case)["automatic_criterion_evidence"][criterion["id"]]["passed"]


def test_empty_structural_receipt_cannot_vacuously_pass(case):
    assert not cases.independent_verdict(case, {"inputs_unchanged": True, "required_outputs_present": {}}, method_verified=True)["passed"]


def test_changed_inputs_and_unsafe_archive_are_independently_detected(case, tmp_path):
    work, initial = complete_files(case, tmp_path)
    (work / next(iter(initial))).write_text("changed")
    evidence = cases.file_evidence(work, initial, case)
    assert not evidence["inputs_unchanged"]
    assert not cases.independent_verdict(case, evidence, method_verified=True)["passed"]
    archive = work / next(name for name in case["required_outputs"] if name.endswith(".zip"))
    with ZipFile(archive, "w") as file:
        file.writestr("../foreign.txt", "unsafe")
    with pytest.raises(ValueError):
        cases.file_evidence(work, initial, case)


@pytest.mark.parametrize("value", ["https://127.0.0.1", "http://user:secret@localhost", "http://localhost/?token=secret", "http://remote.example:8920", "http://localhost/other"])
def test_endpoint_cannot_embed_secrets_or_silently_select_remote(value):
    with pytest.raises(ValueError):
        runner.endpoint(value)


def test_text_payload_contains_only_goal_and_pinned_method():
    value = runner.task_payload("exact/model", "Exact outcome, not a click plan")
    assert value["messages"] == [{"role": "user", "content": "Exact outcome, not a click plan"}]
    assert value["portal_execution_profile"] == "native_task"
    assert value["portal_tool_allowlist"] == list(runner.ALLOWED)
    assert value["think"] is False and value["speech_mode"] == "never"
    assert value["response_modalities"] == ["text"] and value["portal_shell_bridge"] is False
    recall = runner.task_payload("exact/model", "Recall", recall=True)
    assert recall["portal_tool_allowlist"] == ["memory_search"] and "portal_execution_profile" not in recall
    with pytest.raises(ValueError):
        runner.task_payload("model", "goal", think="false")


def test_drain_waits_for_real_idle_without_canceling_other_work():
    clock, queries = [0.0], []
    def handle(request):
        queries.append(request.method)
        return httpx.Response(200, json={"active": 1 if clock[0] < 2 else 0, "inflight": 0, "queued": 0})
    def sleep(seconds):
        clock[0] += seconds
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert not runner.drain(client, "http://localhost", 0, clock=lambda: clock[0], sleep=sleep)
        assert runner.drain(client, "http://localhost", 3, clock=lambda: clock[0], sleep=sleep)
    assert set(queries) == {"GET"}


def test_cancellation_is_one_cookie_scoped_native_cancel():
    calls = []
    def handle(request):
        calls.append((request.url.path, json.loads(request.content), request.headers.get("cookie")))
        return httpx.Response(200, json={"result": {"status": "cancelled"}})
    with httpx.Client(transport=httpx.MockTransport(handle), cookies={"omni_portal_session": "owned-fixture"}) as client:
        assert runner.cancel_owned(client, "http://localhost")["http_status"] == 200
    assert calls == [("/api/tools/native_ui/call", {"arguments": {"operation": "cancel"}}, "omni_portal_session=owned-fixture")]


def test_raw_stream_retains_resets_and_final_claim_without_executing_code(tmp_path):
    raw = b'{"type":"reset","reason":"correction"}\n{"type":"final","response":{"message":{"content":"Done"}}}\n'
    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=raw))) as client:
        result, guard = runner.stream_turn(client, "http://localhost", runner.task_payload("model", "goal"), tmp_path, 2)
    assert (tmp_path / "events.ndjson").read_bytes() == raw
    assert result["final"]["message"]["content"] == "Done" and result["error"] is None and guard is None


def test_truncated_stream_is_not_an_authoritative_completion(tmp_path):
    raw = b'{"type":"final","response":'
    with httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=raw))) as client:
        result, _ = runner.stream_turn(client, "http://localhost", runner.task_payload("model", "goal"), tmp_path, 2)
    assert result["final"] is None and result["error"] is not None
    assert (tmp_path / "events.ndjson").read_bytes() == raw


def test_observed_large_legitimate_final_event_is_accepted_unchanged(tmp_path):
    raw = (json.dumps({"type": "final", "response": {"message": {"content": "x" * (4 * 1024 * 1024 + 1000)}}}) + "\n").encode()
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw))) as client:
        result, _ = runner.stream_turn(client, "http://localhost", runner.task_payload("model", "goal"), tmp_path, 3)
    assert result["error"] is None and result["final"] is not None
    assert (tmp_path / "events.ndjson").read_bytes() == raw


@pytest.mark.parametrize("bound", ["fragment", "complete", "total"])
def test_stream_bounds_reject_overlimit_data_and_accept_multiple_small_lines(tmp_path, monkeypatch, bound):
    monkeypatch.setattr(runner, "MAX_EVENT_BYTES", 80)
    monkeypatch.setattr(runner, "MAX_STREAM_BYTES", 200)
    small = b'{"type":"progress"}\n'
    raw = b"x" * 81 if bound == "fragment" else (b"x" * 81 + b"\n" if bound == "complete" else small * 11)
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw))) as client:
        result, _ = runner.stream_turn(client, "http://localhost", runner.task_payload("model", "goal"), tmp_path, 3)
    assert result["error"] and result["final"] is None
    raw = small * 8
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw))) as client:
        result, _ = runner.stream_turn(client, "http://localhost", runner.task_payload("model", "goal"), tmp_path, 3)
    assert not result["error"] and len(result["events"]) == 8
    assert (tmp_path / "events.ndjson").read_bytes() == raw


def test_same_pid_is_not_unlimited_window_authority(monkeypatch):
    window = SimpleNamespace(pid="12", process_start="34", window_id="50")
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "34")
    monkeypatch.setattr(desktop, "command", lambda *_: "WM_TRANSIENT_FOR(WINDOW): window id # 0x32")
    assert desktop.owned_foreground([window], {"process_id": 12, "window_id": "51"}, {})
    monkeypatch.setattr(desktop, "command", lambda *_: "WM_TRANSIENT_FOR(WINDOW): window id # 0x99")
    assert not desktop.owned_foreground([window], {"process_id": 12, "window_id": "51"}, {})
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "reused")
    assert not desktop.owned_foreground([window], {"process_id": 12, "window_id": "50"}, {})


def test_unregistered_launch_cleanup_is_not_empty_list_success():
    owned = desktop.TaskDesktop({})
    owned.launches = [{"application": "browser", "registered": False, "launch_pid": 123, "launch_start": "456"}]
    assert owned.close() == [{"application": "browser", "registered": False, "launch_pid": 123,
                             "launch_start": "456", "closed": False, "uncertain_unregistered_launch": True}]


def test_active_or_uncertain_work_keeps_owned_windows_open():
    owned = SimpleNamespace(close=lambda: pytest.fail("Never close while downstream activity is unknown"))
    assert runner.cleanup_owned(owned, request_started=True, settled=False) == {
        "owned_windows_closed": False, "status": "cleanup_deferred", "owned_windows_retained": True}


def test_confirmed_idle_permits_only_recorded_cleanup():
    owned = SimpleNamespace(close=lambda: [{"window_id": "50", "closed": True}, {"window_id": None, "closed": False}])
    assert not runner.cleanup_owned(owned, request_started=True, settled=True)["owned_windows_closed"]


def test_no_launch_is_distinct_from_uncertain_unregistered_launch():
    owned = SimpleNamespace(close=lambda: [])
    assert runner.cleanup_owned(owned, request_started=False, settled=False)["owned_windows_closed"]


def test_missing_readiness_blocks_before_any_network_or_app(case, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "load_case", lambda _id: case)
    monkeypatch.setattr(runner, "source_state", lambda: {"revision": "frozen"})
    monkeypatch.setattr(runner, "worker_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner.httpx, "Client", lambda **_kw: pytest.fail("No network before readiness"))
    monkeypatch.setattr(runner, "TaskDesktop", lambda _env, **_kwargs: pytest.fail("No GUI before readiness"))
    args = runner.parser().parse_args(["--case", case["case_id"], "--output", str(tmp_path / "runtime-data/run")])
    assert runner.run(args) == 2
    report = cases.read_json(args.output / "summary.json")
    assert report["status"] == "blocked_precondition" and report["portal_turn_attempts"] == 0 and not report["passed"]
    assert report["owned_windows_closed"] and not any((args.output / "Workbench" / path).exists() for path in case["required_outputs"])


def test_unknown_id_and_semantic_invocation_cannot_satisfy_physical_method():
    events = [{"type": "tool", "phase": "complete", "tools": [{"name": "native_ui", "result": {
        "outcomes": [{"status": "verified", "action": {"id": "one", "verb": "invoke", "kind": "action"},
            "receipt": {"status": "dispatched", "reason": "native_pattern_invoked"},
            "before": {"scope": {"window_id": "50"}}, "after": {"scope": {"window_id": "50"}}}]}}]}]
    guard = SimpleNamespace(records=[{"window_id": "50"}], positions=[{"x": 1, "y": 2}, {"x": 3, "y": 4}], failure=None)
    result = runner.method_evidence(events, guard)
    assert result["same_owned_scope"] and not result["actual_pointer"] and not result["verified"]


def test_every_new_blank_app_requires_independent_exact_owned_observation(case, tmp_path):
    calls, windows = [], []
    def open_blank(app, *_args, **_kwargs):
        window = SimpleNamespace(window_id=str(50 + len(windows)), pid="12", process_start="34")
        windows.append(window)
        calls.append("open:" + app)
        return window
    def fresh():
        calls.append("observe:" + windows[-1].window_id)
        return {"window_id": windows[-1].window_id,
                "state": {"scope": {"window_id": windows[-1].window_id}, "nodes": [{"id": "native-control"}]}}
    receipts = runner.prepare_blank_apps(SimpleNamespace(open_blank=open_blank), SimpleNamespace(fresh=fresh),
        case, tmp_path / "Workbench", tmp_path, "firefox")
    assert len(receipts) == sum(case["fixture"]["blank_app_window_counts"].values())
    assert calls == [value for app, window in zip(case["fixture"]["blank_apps"], windows, strict=True)
                     for value in ("open:" + app, "observe:" + window.window_id)]


@pytest.mark.parametrize("state", [{}, {"outside_owned_foreground": True},
    {"window_id": "foreign", "state": {"scope": {"window_id": "50"}, "nodes": [{"id": "x"}]}},
    {"window_id": "50", "state": {"scope": {"window_id": "foreign"}, "nodes": [{"id": "x"}]}},
    {"window_id": "50", "state": {"scope": {"window_id": "50"}, "nodes": []}}])
def test_unobserved_or_mismatched_owned_app_blocks_before_next_launch(case, tmp_path, state):
    calls = []
    def open_blank(app, *_args, **_kwargs):
        calls.append(app)
        return SimpleNamespace(window_id="50", pid="12", process_start="34")
    with pytest.raises(desktop.CapabilityBlocked, match="owned_native_setup_observation_unavailable"):
        runner.prepare_blank_apps(SimpleNamespace(open_blank=open_blank), SimpleNamespace(fresh=lambda: state),
                                  case, tmp_path / "Workbench", tmp_path, "firefox")
    assert calls == [case["fixture"]["blank_apps"][0]]
    assert (tmp_path / "setup-native-observations.json").is_file()


def mock_run(monkeypatch, tmp_path, case, *, settled=True, setup_failure=False, deployed_model="exact/model",
             observer_close_error=False, recall_settled=True, memory=False):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "load_case", lambda _id: case)
    monkeypatch.setattr(runner, "source_state", lambda: {"revision": "frozen"})
    monkeypatch.setattr(runner, "worker_snapshot", lambda: {"available": False})
    monkeypatch.setattr(runner, "desktop_subprocess_environment", lambda: {})
    monkeypatch.setattr(runner, "managed_inventory", lambda _env: {"available": True, "windows": {}})
    calls = []
    class FakeDesktop:
        environment = {}
        launches = []
        def open_blank(self, app, *_args, **_kwargs):
            calls.append("open:" + app)
            self.window = SimpleNamespace(window_id=str(50 + len(calls)), pid="12", process_start="34")
            self.launches.append({"registered": True, "window_id": self.window.window_id})
            return self.window
        def close(self):
            calls.append("close")
            return [{"closed": True}]
    owned = FakeDesktop()
    class FakeObserver:
        task_started = SimpleNamespace(set=lambda: None)
        def __init__(self, *_args):
            pass
        def fresh(self):
            if setup_failure:
                return {"error": {"code": "native_observation_unavailable"}}
            return {"window_id": owned.window.window_id, "state": {
                "scope": {"window_id": owned.window.window_id}, "nodes": [{"id": "control"}]}}
        def start(self):
            calls.append("sampling")
        def close(self):
            calls.append("stop_sampling")
            if observer_close_error:
                raise RuntimeError("observer shutdown not confirmed")
    monkeypatch.setattr(runner, "TaskDesktop", lambda _env, **_kwargs: owned)
    monkeypatch.setattr(runner, "NativeEvidenceObserver", FakeObserver)
    factory = httpx.Client
    def transport(request):
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"model": deployed_model})
        assert request.url.path == "/api/activity"
        return httpx.Response(200, json={"active": 0, "inflight": 0, "queued": 0})
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: factory(transport=httpx.MockTransport(transport), **kwargs))
    def stream(_client, _base, payload, root, *_args, **_kwargs):
        calls.append("model_attempt")
        assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
        events = [{"type": "final", "response": {"message": {"content": "All completed"}}}]
        if memory:
            events.insert(0, {"type": "tool", "phase": "complete", "tools": [{"name": "memory_write",
                          "ok": True, "result": {"stored": True}}]})
        (root / "events.ndjson").write_text("\n".join(json.dumps(event) for event in events) + "\n")
        guard = SimpleNamespace(failure=None, records=[], positions=[], close=lambda: calls.append("guard_close"))
        return {"events": events, "final": events[-1]["response"], "error": None}, guard
    monkeypatch.setattr(runner, "stream_turn", stream)
    def drain(*_args):
        result = settled and (recall_settled if calls.count("model_attempt") > 1 else True)
        calls.append("drain_idle" if result else "drain_uncertain")
        return result
    monkeypatch.setattr(runner, "drain", drain)
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    pack = proof_pack(case, proofs)
    (proofs / "pack.json").write_text(json.dumps(pack))
    token = tmp_path / "token.txt"
    token.write_text("must-never-appear-in-artifacts")
    args = runner.parser().parse_args(["--case", case["case_id"], "--output", str(tmp_path / "runtime-data/run"),
        "--readiness-file", str(proofs / "pack.json"), "--model", "exact/model", "--token-file", str(token)])
    return args, calls


def test_model_done_cannot_pass_and_cleanup_follows_drain(case, tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path, case)
    assert runner.run(args) == 1
    report = cases.read_json(args.output / "summary.json")
    assert report["authoritative_final_received"] and not report["passed"]
    assert report["status"] == "failed_independent_checks" and not report["method"]["verified"]
    assert report["portal_turn_attempts"] == 1 and report["upstream_model_request_count"] is None
    assert report["owned_windows_closed"] and calls.index("close") > calls.index("drain_idle")
    assert all("must-never-appear-in-artifacts" not in path.read_text()
               for path in args.output.rglob("*.json"))
    goal = (args.output / "goal.txt").read_text()
    assert goal.strip() == case["goal_text"].replace("WORKBENCH", str(args.output / "Workbench"))


def test_run_preserves_windows_when_downstream_drain_unconfirmed(case, tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path, case, settled=False)
    assert runner.run(args) == 3
    report = cases.read_json(args.output / "summary.json")
    assert report["status"] == "cleanup_deferred" and not report["downstream_idle"]
    assert report["owned_windows_retained"] and not report["owned_windows_closed"]
    assert "close" not in calls and calls.count("model_attempt") == 1


def test_actual_app_setup_failure_is_blocked_not_a_model_failure(case, tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path, case, setup_failure=True)
    assert runner.run(args) == 2
    report = cases.read_json(args.output / "summary.json")
    assert report["status"] == "blocked_precondition" and report["portal_turn_attempts"] == 0
    assert report["error"]["code"].startswith("owned_native_setup_observation_unavailable:")
    assert report["owned_windows_closed"] and "model_attempt" not in calls


def test_observer_shutdown_error_keeps_windows_and_still_retains_summary_note(case, tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path, case, observer_close_error=True)
    assert runner.run(args) == 1
    report = cases.read_json(args.output / "summary.json")
    assert report["teardown_errors"] == [{"type": "RuntimeError", "stage": "observer_shutdown"}]
    assert not report["owned_windows_closed"] and report["owned_windows_retained"]
    assert "close" not in calls and (args.output / "after-test-note.json").is_file()
    assert (args.output / "evidence-manifest.json").is_file()


def test_recall_drain_failure_returns_cleanup_deferred_exit_and_retains_windows(case, tmp_path, monkeypatch):
    assert case["memory_followup"]["required"]
    args, calls = mock_run(monkeypatch, tmp_path, case, memory=True, recall_settled=False)
    assert runner.run(args) == 3
    report = cases.read_json(args.output / "summary.json")
    assert report["status"] == "cleanup_deferred" and report["portal_turn_attempts"] == 2
    assert report["owned_windows_retained"] and "close" not in calls


def test_wrong_deployed_model_blocks_before_opening_app_or_submitting(case, tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path, case, deployed_model="different/model")
    assert runner.run(args) == 2
    report = cases.read_json(args.output / "summary.json")
    assert report["error"]["code"] == "exact_deployed_model_mismatch"
    assert report["portal_turn_attempts"] == 0 and not any(value.startswith("open:") for value in calls)


def test_safe_transport_error_does_not_echo_capability_or_authentication():
    error = httpx.ReadError("Bearer private-token https://host/private-capability")
    assert runner.safe_error(error, stage="task") == {"type": "ReadError", "stage": "task"}


@pytest.mark.parametrize("mutation", [None, "candidate", "frame", "semantic_node"])
def test_visual_invoke_method_requires_actual_candidate_frame_binding(mutation):
    point = {"candidate_id": "visual:one", "frame_sha256": "frame"}
    click = {"status": "unresolved", "action": {"id": "click", "kind": "action", "verb": "invoke",
             "node_id": "visual:one", "generation": "frame"}, "point_receipt": point,
             "receipt": {"status": "dispatched", "backend": "x11_guarded_xtest"},
             "before": {"scope": {"window_id": "50"}}, "after": {"scope": {"window_id": "50"}}}
    keyboard = copy.deepcopy(click)
    keyboard["action"] = {"id": "key", "verb": "type_text", "kind": "action"}
    keyboard.pop("point_receipt")
    if mutation == "candidate":
        point["candidate_id"] = "other"
    elif mutation == "frame":
        point["frame_sha256"] = "wrong"
    elif mutation == "semantic_node":
        click["action"]["node_id"] = "semantic"
    events = [{"type": "tool", "phase": "complete", "tools": [{"name": "native_ui", "result": {"outcomes": [click, keyboard]}}]}]
    guard = SimpleNamespace(records=[{"window_id": "50"}], positions=[{"x": 1, "y": 2}, {"x": 2, "y": 3}], failure=None)
    method = runner.method_evidence(events, guard)
    assert method["actual_pointer"] is (mutation is None)
    assert method["verified"] is (mutation is None)


def test_delegation_metrics_deduplicate_receipts_without_inferred_model_calls():
    policy = {"id": "policy-one", "status": "accepted", "model": "kev@revision", "reply_sha256": "reply"}
    point = {"model": "moondream", "revision": "pinned", "frame_sha256": "frame", "candidate_id": "visual:one"}
    outcome = {"action": {"id": "one", "idempotency_key": "only-input", "kind": "action"},
               "receipt": {"status": "dispatched"}, "policy_receipt": policy, "point_receipt": point}
    result = {"outcomes": [outcome], "policy_receipts": [policy], "point_receipts": [point],
              "capabilities": {"policy": {"status": "configured"}}}
    replay = copy.deepcopy(result)
    replay["outcomes"][0]["effect_receipt"] = {"effect_state": "replayed"}
    events = [{"type": "tool", "phase": "complete", "round": index,
               "tools": [{"name": "native_ui", "result": item}]} for index, item in enumerate([result, replay], 1)]
    metrics = runner.measured_metrics(events, elapsed_s=1, method={"unique_dispatched_inputs": 1})
    assert metrics["delegation"]["policy"]["unique_receipts"] == 1
    assert metrics["delegation"]["point"]["unique_receipts"] == 1
    assert [phase["new_unique_dispatched_inputs"] for phase in metrics["inner_inputs_per_phase"]] == [1, 0]
    assert metrics["semantic_unique_count"] is None and metrics["upstream_planner_request_count"] is None
    empty = runner.measured_metrics([], elapsed_s=0, method={"unique_dispatched_inputs": 0})
    assert empty["delegation"]["point"]["unique_receipts"] == 0


def test_final_cumulative_trace_is_only_retained_raw_not_duplicated_in_summary():
    final = {"message": {"content": "Claim"}, "portal": {"safe_tools_executed": [{"large": "raw"}],
                                                              "native_task": {"status": "planner_claim"}}}
    result = runner.turn_summary({"events": [{"type": "final", "response": final}], "final": final, "error": None})
    assert "safe_tools_executed" not in result["final"]["portal"]
    assert final["portal"]["safe_tools_executed"] == [{"large": "raw"}]
    assert result["raw_final_and_full_cumulative_trace"] == "events.ndjson"


def test_managed_inventory_is_metadata_only_and_binds_pid_start(monkeypatch):
    calls = []
    def command(argv, _env):
        calls.append(argv)
        return "_NET_CLIENT_LIST(WINDOW): window id # 0x32, 0x33" if argv[0] == "xprop" else "12"
    monkeypatch.setattr(desktop, "command", command)
    monkeypatch.setattr(desktop, "process_start", lambda _pid: "34")
    evidence = desktop.managed_inventory({})
    assert evidence == {"available": True, "windows": {"50": {"pid": "12", "start_ticks": "34"},
                                                       "51": {"pid": "12", "start_ticks": "34"}},
                        "coverage": "managed_client_windows_only"}
    assert calls == [["xprop", "-root", "_NET_CLIENT_LIST"], ["xdotool", "getwindowpid", "50"],
                     ["xdotool", "getwindowpid", "51"]]


@pytest.mark.parametrize("after", [{}, {"50": {"pid": "12", "start_ticks": "reused"}},
                                  {"50": {"pid": "other", "start_ticks": "34"}}])
def test_preexisting_window_disappearance_or_reuse_is_preservation_failure(after):
    before = {"available": True, "windows": {"50": {"pid": "12", "start_ticks": "34"}}}
    result = desktop.unrelated_preservation(before, {"available": True, "windows": after})
    assert result["status"] == "failed" and not result["verified"]


def test_unavailable_window_identity_is_not_preservation_proof():
    unknown = {"available": True, "windows": {"50": {"identity_available": False}}}
    assert desktop.unrelated_preservation(unknown, unknown)["status"] == "unavailable"
    assert not desktop.unrelated_preservation({"available": False}, unknown)["verified"]


@pytest.mark.parametrize("changed", [None, "output", "recall", "summary", "pointer", "source_proof", "native", "setup", "added", "deleted"])
def test_offline_grading_is_bundle_bound_and_keeps_original_summary(case, tmp_path, monkeypatch, changed):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "load_case", lambda _id: case)
    root = tmp_path / "runtime-data/run"
    root.mkdir(parents=True)
    work, initial = complete_files(case, root)
    evidence = cases.file_evidence(work, initial, case)
    report = {"case_id": case["case_id"], "task_signature_sha256": case["task_signature_sha256"],
              "initial_input_hashes": initial, "final_file_evidence": evidence,
              "method": {"verified": True}, "downstream_idle": True, "owned_windows_closed": True,
              "source_unchanged": True, "unrelated_window_preservation": {"verified": True},
              "passed": False, "status": "pending_independent_review"}
    raw = b'{"type":"final","response":{"message":{"content":"Claim only"}}}\n'
    (root / "events.ndjson").write_bytes(raw)
    runner.save(root / "summary.json", report)
    extra = {"recall": "recall/events.ndjson", "pointer": "pointer-observations.jsonl",
             "native": "final-native.json", "setup": "setup-native-observations.json"}
    for name in extra.values():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("Owned independent evidence")
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    runner.save(proofs / "pack.json", proof_pack(case, proofs))
    cases.capture_source_proofs(case, proofs / "pack.json", root / "host-source-evidence", revision="frozen", work=work)
    original = (root / "summary.json").read_bytes()
    bundle = cases.freeze_evidence(root, case)["bundle_sha256"]
    review = {"schema": "robit.omni.long-native-independent-review.v1", "case_id": case["case_id"],
              "authority": "host_independent_evaluator", "bundle_sha256": bundle,
              "criteria": {item["id"]: True for item in case["mandatory_criteria"]}}
    runner.save(tmp_path / "review.json", review)
    if changed == "output":
        (work / "Edited/draft-1.txt").write_text("Post-run alteration")
    elif changed == "summary":
        altered = cases.read_json(root / "summary.json")
        altered["untrusted_added_claim"] = "Should invalidate previous review"
        runner.save(root / "summary.json", altered)
        original = (root / "summary.json").read_bytes()
    elif changed == "source_proof":
        (root / "host-source-evidence/source-00.bin").write_text("Changed source snapshot")
    elif changed == "added":
        (root / "extra-evidence.json").write_text("New evidence changes manifest")
    elif changed == "deleted":
        (root / extra["setup"]).unlink()
    elif changed:
        (root / extra[changed]).write_text("Changed evidence")
    args = runner.parser().parse_args(["--output", str(root), "--evaluate", "--review-file", str(tmp_path / "review.json")])
    assert runner.run(args) == (4 if changed else 0)
    assert cases.read_json(root / "independent-verdict.json")["passed"] is (not changed)
    assert (root / "summary.json").read_bytes() == original


def test_frozen_source_capture_keeps_verified_bytes_outside_solver_workbench(case, tmp_path):
    work = tmp_path / "Workbench"
    cases.materialize(case, work)
    proofs = tmp_path / "proofs"
    proofs.mkdir()
    pack = proof_pack(case, proofs)
    runner.save(proofs / "pack.json", pack)
    captured = cases.capture_source_proofs(case, proofs / "pack.json", tmp_path / "host-evidence", revision="frozen", work=work)
    assert len(captured["sources"]) == len(case["sources_to_acquire"])
    assert not (work / "host-evidence").exists()
    for item in captured["sources"]:
        assert cases.sha((tmp_path / "host-evidence" / item["content_path"]).read_bytes()) == item["content_sha256"]
    (proofs / pack["receipts"][0]["path"]).write_text("Changed since validation")
    with pytest.raises(ValueError):
        cases.capture_source_proofs(case, proofs / "pack.json", tmp_path / "second-capture", revision="frozen", work=work)


@pytest.mark.parametrize("mutation", ["symlink", "derived_symlink", "oversize"])
def test_manifest_refuses_unbounded_or_symlinked_evidence(tmp_path, monkeypatch, mutation):
    (tmp_path / "ordinary.txt").write_text("valid")
    if mutation == "symlink":
        (tmp_path / "link.txt").symlink_to(tmp_path / "ordinary.txt")
    elif mutation == "derived_symlink":
        (tmp_path / "independent-verdict.json").symlink_to(tmp_path / "ordinary.txt")
    else:
        monkeypatch.setattr(cases, "MAX_BUNDLE_FILE", 4)
    with pytest.raises((ValueError, OSError)):
        cases.evidence_manifest(tmp_path, "case", "signature")
