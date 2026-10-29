"""Real filesystem fixture controls; injected native evidence has no live credit."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from runtime import long_horizon_amendments as amendments
from runtime import long_horizon_cases as cases
from runtime import verify_long_horizon_native as runner
from runtime.long_horizon_cases import load_case, materialize, sha

REVISION = "24f1691767e61c94350fe0610c20036d0eec560b"
WINDOW = {"window_id": "12345", "process_id": 123, "process_start": "456"}


def workbook(value="first draft", *, body=None):
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        archive.writestr("mimetype", amendments.MIME)
        archive.writestr("META-INF/manifest.xml", '<manifest:manifest '
            'xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
            '<manifest:file-entry manifest:full-path="/" manifest:media-type="'
            + amendments.MIME.decode() + '"/></manifest:manifest>')
        archive.writestr("content.xml", body or (
            '<office:document-content xmlns:office="' + amendments.NS["office"] + '" '
            'xmlns:table="' + amendments.NS["table"] + '" xmlns:text="' + amendments.NS["text"] + '">'
            '<office:body><office:spreadsheet><table:table table:name="Calendar">'
            '<table:table-row><table:table-cell office:value-type="string"><text:p>' + value
            + '</text:p></table:table-cell></table:table-row></table:table>'
            '</office:spreadsheet></office:body></office:document-content>'))
    return output.getvalue()


def probe(url=""):
    return {"status": "observed", "authority": "host_read_only_atspi", "source": "atspi.Document.DocURL",
            "window": dict(WINDOW), "role": "document spreadsheet", "url": url,
            "docurl_query_supported": True, "provider_identity_sha256": "a" * 64}


def write_setup(case, work):
    rows = []
    for index, app in enumerate(case["fixture"]["blank_apps"]):
        window = dict(WINDOW) if app == "calc" else {"window_id": str(20000 + index),
                  "process_id": 200 + index, "process_start": str(900 + index)}
        native = {"phase": "setup", "window_id": window["window_id"], "outside_owned_foreground": False,
                  "state": {"scope": {"window_id": window["window_id"]}, "nodes": [{"id": "setup"}]}}
        rows.append({"application": app, **window, "native": native,
                     "setup_origin": "declared_fixture", **({"calendar_document_probe": probe()} if app == "calc" else {})})
    (work.parent / "setup-native-observations.json").write_text(json.dumps(rows))


@pytest.fixture
def fixture(tmp_path):
    case = load_case("LHN-077")
    work = tmp_path / "Workbench"
    initial = materialize(case, work)
    write_setup(case, work)
    capability = amendments.prepare_capability(case, work=work, revision=REVISION,
        initial_hashes=initial, calc_window=WINDOW, document_probe=probe())
    native = {"phase": "model_task", "window_id": WINDOW["window_id"], "outside_owned_foreground": False,
              "state": {"scope": {"window_id": WINDOW["window_id"]}}}
    return case, work, initial, capability, native


def injector(fixture, tmp_path, *, probe_value=None, probe_function=None):
    case, work, initial, capability, _ = fixture
    return amendments.AmendmentRelease(case, capability=capability, work=work, revision=REVISION,
        initial_hashes=initial, event_root=tmp_path / "amendment-events",
        probe=probe_function or (lambda window, environment: copy.deepcopy(probe_value or probe())))


def save_first(fixture, relative="Drafts/early-calendar.ods", content=None):
    path = fixture[1] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else workbook())
    return path


@pytest.mark.parametrize("number", range(77, 83))
def test_each_exact_frozen_payload_is_absent_until_real_persisted_calendar(number, tmp_path):
    case = load_case(f"LHN-{number:03}")
    work = tmp_path / "Workbench"
    initial = materialize(case, work)
    write_setup(case, work)
    capability = amendments.prepare_capability(case, work=work, revision=REVISION,
        initial_hashes=initial, calc_window=WINDOW, document_probe=probe())
    native = {"phase": "model_task", "window_id": WINDOW["window_id"], "outside_owned_foreground": False,
              "state": {"scope": {"window_id": WINDOW["window_id"]}}}
    fixture = case, work, initial, capability, native
    engine = injector(fixture, tmp_path)
    assert engine.observe(native) is None
    assert not (work / amendments.RELEASE_PATH).exists()
    saved = save_first(fixture)
    original = saved.read_bytes()
    engine = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    event = engine.observe(native)
    assert event["status"] == "released" and event["release_attempts"] == 1
    actual = (work / amendments.RELEASE_PATH).read_bytes()
    assert json.loads(actual) == case["controlled_perturbation"]["payload"]["content"]
    assert json.loads(actual)["task_id"] == f"T{number - 67}"
    assert event["release"]["readback_verified"] and event["release"]["sha256"] == sha(actual)
    assert event["unchanged_prior_files"] and event["host_solver_output_mutation"] is False
    assert event["whole_task_success_credit"] is False
    assert saved.read_bytes() == original
    assert {name: sha((work / name).read_bytes()) for name in initial} == initial
    assert not any((work / name).exists() for name in case["required_outputs"])


def test_first_save_does_not_require_final_filename_or_a_correct_schedule(fixture, tmp_path):
    saved = save_first(fixture, "not-the-final-path/initial-incorrect-schedule.ods", workbook("T01 only, not solved"))
    event = injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4])
    assert event["status"] == "released"
    assert event["first_saved_workbook"]["path"] == "not-the-final-path/initial-incorrect-schedule.ods"
    assert not (fixture[1] / "Calendar/production-calendar.ods").exists()


def test_duplicate_observation_and_object_restart_never_release_again(fixture, tmp_path):
    saved = save_first(fixture)
    first = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    assert first.observe(fixture[4])["status"] == "released"
    payload = fixture[1] / amendments.RELEASE_PATH
    inode, content = payload.stat().st_ino, payload.read_bytes()
    saved.write_bytes(workbook("second save"))
    assert first.observe(fixture[4]) is None
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None
    assert payload.stat().st_ino == inode and payload.read_bytes() == content


@pytest.mark.parametrize("content", [b"", b"partial zip", b"<xml/>", workbook(""),
    workbook(body='<office:document-content xmlns:office="' + amendments.NS["office"]
             + '"><office:body><office:text/></office:body></office:document-content>')])
def test_empty_partial_wrong_format_and_empty_workbooks_cannot_trigger(fixture, tmp_path, content):
    saved = save_first(fixture, content=content)
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None
    assert not (fixture[1] / amendments.RELEASE_PATH).exists()


@pytest.mark.parametrize("delta", [
    {"phase": "setup"}, {"window_id": "999"}, {"state": {"scope": {"window_id": "999"}}},
    {"error": {"type": "read failed"}}, {"outside_owned_foreground": True},
])
def test_wrong_phase_or_window_has_no_trigger_even_with_save_prose(fixture, tmp_path, delta):
    saved = save_first(fixture)
    native = {**fixture[4], "text": "Save succeeded, production-calendar.ods", **delta}
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(native) is None


@pytest.mark.parametrize("change", [
    {"authority": "model"}, {"window": {**WINDOW, "process_start": "999"}},
    {"role": "table cell"}, {"docurl_query_supported": False}, {"source": "window title"},
])
def test_unbound_native_document_evidence_cannot_trigger(fixture, tmp_path, change):
    saved = save_first(fixture)
    assert injector(fixture, tmp_path, probe_value={**probe(saved.as_uri()), **change}).observe(fixture[4]) is None


@pytest.mark.parametrize("uri", ["https://example.com/calendar.ods", "file://otherhost/tmp/calendar.ods",
    "file:///tmp/unrelated-calendar.ods", "file:///tmp/x%00.ods", "file:relative.ods"])
def test_foreign_unrelated_or_invalid_document_url_cannot_trigger(fixture, tmp_path, uri):
    save_first(fixture)
    assert injector(fixture, tmp_path, probe_value=probe(uri)).observe(fixture[4]) is None


def test_current_document_must_match_disk_not_an_unrelated_valid_ods(fixture, tmp_path):
    save_first(fixture)
    assert injector(fixture, tmp_path, probe_value=probe((fixture[1] / "missing.ods").as_uri())).observe(fixture[4]) is None


def test_native_identity_or_saved_bytes_changing_during_probe_remains_armed(fixture, tmp_path):
    saved = save_first(fixture)
    calls = 0
    def changing(window, environment):
        nonlocal calls
        calls += 1
        if calls == 2:
            saved.write_bytes(workbook("different save"))
        return probe(saved.as_uri())
    assert injector(fixture, tmp_path, probe_function=changing).observe(fixture[4]) is None
    assert not (fixture[1] / amendments.RELEASE_PATH).exists()


def test_original_input_change_prevents_release(fixture, tmp_path):
    saved = save_first(fixture)
    (fixture[1] / "Input/tasks.csv").write_text("changed")
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None


@pytest.mark.parametrize("as_link", [False, True])
def test_concurrent_existing_release_is_never_overwritten_and_unknown_is_sticky(fixture, tmp_path, as_link):
    saved = save_first(fixture)
    destination = fixture[1] / amendments.RELEASE_PATH
    destination.parent.mkdir()
    if as_link:
        victim = tmp_path / "outside.json"
        victim.write_text("preserve me")
        destination.symlink_to(victim)
    else:
        destination.write_text("racing writer")
    engine = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    # Existing paths are noticed before any attempt; no overwrite authority.
    result = engine.observe(fixture[4])
    if as_link:
        assert result is None
    else:
        assert result["status"] == "unknown"
    assert destination.read_text() == ("preserve me" if as_link else "racing writer")
    if not as_link:
        assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None


def test_failure_after_publication_does_not_retry_or_replace(fixture, tmp_path, monkeypatch):
    saved = save_first(fixture)
    real_release = amendments._release
    def publish_then_fail(work, payload):
        real_release(work, payload)
        raise OSError("readback became unavailable")
    monkeypatch.setattr(amendments, "_release", publish_then_fail)
    engine = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    event = engine.observe(fixture[4])
    assert event["status"] == "unknown" and event["retry_permitted"] is False
    assert event["release_attempts"] == 1
    content = (fixture[1] / amendments.RELEASE_PATH).read_bytes()
    assert engine.observe(fixture[4]) is None
    assert (fixture[1] / amendments.RELEASE_PATH).read_bytes() == content


@pytest.mark.parametrize("field,value", [
    ("case_id", "LHN-078"), ("source_revision", "0" * 40), ("payload_sha256", "0" * 64),
    ("implementation_sha256", "0" * 64), ("trigger", "after final correct schedule"),
    ("task_signature_sha256", "0" * 64), ("event_occurred", True),
])
def test_capability_is_case_source_implementation_and_trigger_bound(fixture, field, value):
    case, work, initial, capability, _ = fixture
    assert not amendments.validate_capability(case, {**capability, field: value},
        work=work, revision=REVISION, initial_hashes=initial)


def test_mutating_frozen_case_or_declared_payload_is_rejected(fixture):
    case = copy.deepcopy(fixture[0])
    case["controlled_perturbation"]["payload"]["content"]["new_duration_minutes"] = 91
    with pytest.raises(ValueError, match="unchanged frozen"):
        amendments._declared(case)


def test_capability_requires_actual_document_interface_not_registration_alone(fixture):
    case, work, initial, _, _ = fixture
    with pytest.raises(ValueError, match="Document evidence unavailable"):
        amendments.prepare_capability(case, work=work, revision=REVISION, initial_hashes=initial,
            calc_window=WINDOW, document_probe={"registered": True})


def test_no_capability_if_amendment_exists_at_setup(fixture):
    case, work, initial, _, _ = fixture
    destination = work / amendments.RELEASE_PATH
    destination.parent.mkdir()
    destination.write_text("must not exist")
    with pytest.raises(ValueError, match="exactly unchanged"):
        amendments.prepare_capability(case, work=work, revision=REVISION, initial_hashes=initial,
            calc_window=WINDOW, document_probe=probe())


def test_directory_symlink_cannot_publish_outside_workbench(fixture, tmp_path):
    saved = save_first(fixture)
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture[1] / "Input/Amendments").symlink_to(outside, target_is_directory=True)
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None
    assert list(outside.iterdir()) == []


def test_same_directory_different_inode_breaks_workroot_binding(fixture, tmp_path):
    case, work, initial, capability, native = fixture
    engine = injector(fixture, tmp_path)
    work.rename(tmp_path / "old-Workbench")
    materialize(case, work)
    assert not amendments.validate_capability(case, capability, work=work, revision=REVISION, initial_hashes=initial)
    with pytest.raises(ValueError, match="Workbench identity changed"):
        engine.observe(native)


def test_durable_attempt_without_final_receipt_remains_unknown_and_never_retries(fixture, tmp_path):
    saved = save_first(fixture)
    engine = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    state = json.loads(engine.state_path.read_text())
    state.update(status="attempted", release_attempts=1)
    with engine.journal_path.open("a") as output:
        output.write(json.dumps(state) + "\n")
    # Simulate crash after durable attempted journal and before atomic snapshot.
    assert json.loads(engine.state_path.read_text())["status"] == "armed"
    assert injector(fixture, tmp_path, probe_value=probe(saved.as_uri())).observe(fixture[4]) is None
    assert not (fixture[1] / amendments.RELEASE_PATH).exists()


def test_production_probe_launch_is_fixed_read_only_and_cuda_disabled(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": json.dumps(probe()).encode()})()
    monkeypatch.setattr(amendments.subprocess, "run", run)
    assert amendments.probe_calendar_document(WINDOW, {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "bad"}) == probe()
    argv, kwargs = calls[0]
    assert argv == ["/usr/bin/python3", "-m", "runtime.long_horizon_amendments", "--probe"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads(kwargs["input"]) == WINDOW and kwargs["timeout"] == 9
    assert Path(kwargs["cwd"]) == Path(amendments.__file__).resolve().parents[1]


def completed_receipt(fixture, tmp_path):
    case, work, initial, capability, native = fixture
    saved = save_first(fixture)
    before = amendments.inventory(work)
    (tmp_path / "amendment-capability.json").write_text(json.dumps(capability))
    (tmp_path / "milestones.jsonl").write_text(json.dumps({"native": native,
        "files": {"files": before, "inputs_unchanged": True}}) + "\n")
    engine = injector(fixture, tmp_path, probe_value=probe(saved.as_uri()))
    assert engine.observe(native)["status"] == "released"
    report = {"source_before": {"revision": REVISION}, "initial_input_hashes": initial,
              "controlled_perturbation": engine.snapshot()}
    return report


@pytest.mark.parametrize("stage", ["setup", "trigger"])
@pytest.mark.parametrize("ownership", ["missing", None, 0, True, False])
def test_setup_and_trigger_require_explicit_owned_false_independently(fixture, tmp_path, stage, ownership):
    case, work, initial, capability, native = fixture
    def set_ownership(observation):
        if ownership == "missing":
            observation.pop("outside_owned_foreground", None)
        else:
            observation["outside_owned_foreground"] = ownership
    if stage == "setup":
        path = tmp_path / "setup-native-observations.json"
        rows = json.loads(path.read_bytes())
        for row in rows:
            set_ownership(row["native"])
        path.write_text(json.dumps(rows))
        if ownership is False:
            observed = amendments.prepare_capability(case, work=work, revision=REVISION,
                initial_hashes=initial, calc_window=WINDOW, document_probe=probe())
            assert amendments.validate_capability(case, observed, work=work, revision=REVISION, initial_hashes=initial)
        else:
            with pytest.raises(ValueError):
                amendments.prepare_capability(case, work=work, revision=REVISION,
                    initial_hashes=initial, calc_window=WINDOW, document_probe=probe())
        assert not (work / amendments.RELEASE_PATH).exists()
        return
    # Keep setup positively owned so a failed setup cannot mask a trigger fault.
    assert amendments.validate_capability(case, capability, work=work, revision=REVISION, initial_hashes=initial)
    saved = save_first(fixture)
    before = amendments.inventory(work)
    calls = []
    def document_probe(window, _environment):
        calls.append(window)
        return probe(saved.as_uri())
    engine = injector(fixture, tmp_path, probe_function=document_probe)
    set_ownership(native)
    event = engine.observe(native)
    if ownership is False:
        assert event["status"] == "released" and len(calls) == 2
        assert json.loads((work / amendments.RELEASE_PATH).read_bytes()) == case["controlled_perturbation"]["payload"]["content"]
    else:
        assert event is None and not calls
        assert engine.snapshot()["status"] == "armed" and engine.snapshot()["release_attempts"] == 0
        assert amendments.inventory(work) == before and not (work / amendments.RELEASE_PATH).exists()


@pytest.mark.parametrize("ownership", ["missing", None, 0, True, False])
def test_independent_verifier_requires_owned_trigger_even_with_self_consistent_rewritten_hashes(fixture, tmp_path, ownership):
    report = completed_receipt(fixture, tmp_path)
    assert amendments.verified_event(fixture[0], tmp_path, report)
    milestone_path = tmp_path / "milestones.jsonl"
    milestone = json.loads(milestone_path.read_bytes())
    if ownership == "missing":
        milestone["native"].pop("outside_owned_foreground")
    else:
        milestone["native"]["outside_owned_foreground"] = ownership
    milestone_path.write_text(json.dumps(milestone) + "\n")
    native_hash = sha(amendments._json(milestone["native"]))
    journal = tmp_path / "amendment-events/amendment-events.jsonl"
    records = [json.loads(line) for line in journal.read_bytes().splitlines()]
    for row in records[1:]:
        row["native_sha256"] = native_hash
    journal.write_bytes(b"".join(amendments._json(row) + b"\n" for row in records))
    (tmp_path / "amendment-events/amendment-event.json").write_bytes(amendments._json(records[-1]))
    report["controlled_perturbation"] = {**records[-1], "journal_sha256": sha(journal.read_bytes())}
    assert records[-1]["native_sha256"] == sha(amendments._json(milestone["native"]))
    assert amendments.verified_event(fixture[0], tmp_path, report) is (ownership is False)


def test_independent_event_verifier_requires_actual_bound_trigger_and_payload(fixture, tmp_path):
    report = completed_receipt(fixture, tmp_path)
    assert amendments.verified_event(fixture[0], tmp_path, report)


@pytest.mark.parametrize("damage", ["wrong_native", "wrong_file", "wrong_case", "extra_output",
    "missing_journal", "only_capability", "changed_release", "changed_original", "changed_setup",
    "missing_milestone", "state_snapshot", "changed_source", "repeated_attempt"])
def test_independent_verifier_rejects_changed_or_missing_event_evidence(fixture, tmp_path, damage):
    report = completed_receipt(fixture, tmp_path)
    case, work, _, _, _ = fixture
    if damage == "wrong_native":
        path = tmp_path / "milestones.jsonl"
        value = json.loads(path.read_text())
        value["native"]["window_id"] = "999"
        path.write_text(json.dumps(value) + "\n")
    elif damage == "wrong_file":
        path = tmp_path / "milestones.jsonl"
        value = json.loads(path.read_text())
        value["files"]["files"]["Drafts/early-calendar.ods"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value) + "\n")
    elif damage in {"wrong_case", "extra_output", "repeated_attempt"}:
        path = tmp_path / "amendment-events/amendment-events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "wrong_case":
            rows[-1]["case_id"] = "LHN-078"
        elif damage == "extra_output":
            rows[-1]["files_after"]["Calendar/host-written.txt"] = {"sha256": "0" * 64, "bytes": 1}
        else:
            rows.append(rows[-1])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif damage == "missing_journal":
        (tmp_path / "amendment-events/amendment-events.jsonl").unlink()
    elif damage == "only_capability":
        report["controlled_perturbation"] = fixture[3]
    elif damage == "changed_release":
        path = work / amendments.RELEASE_PATH
        path.chmod(0o600)
        path.write_text("changed")
    elif damage == "changed_original":
        (work / "Input/tasks.csv").write_text("changed")
    elif damage == "changed_setup":
        (tmp_path / "setup-native-observations.json").write_text("[]")
    elif damage == "missing_milestone":
        (tmp_path / "milestones.jsonl").unlink()
    elif damage == "state_snapshot":
        (tmp_path / "amendment-events/amendment-event.json").write_text("{}")
    elif damage == "changed_source":
        report["source_before"]["revision"] = "0" * 40
    assert not amendments.verified_event(case, tmp_path, report)


def test_blank_supported_query_does_not_require_a_saved_document_uri(fixture):
    case, work, initial, capability, _ = fixture
    assert capability["blank_document_probe"]["url"] == ""
    assert amendments.validate_capability(case, capability, work=work, revision=REVISION, initial_hashes=initial)


def source_pack(case, root):
    receipts = []
    for index, source in enumerate(case["sources_to_acquire"]):
        content = root / f"source-{index}.html"
        content.write_text("Unit fixture only; no live source acquisition credit")
        artifact = {"schema": "robit.omni.long-native-prerequisite-proof.v1", "case_id": case["case_id"],
            "task_signature_sha256": case["task_signature_sha256"], "source_revision": REVISION,
            "receipt_id": "source:" + source["source_id"], "authority": "host_independent_evaluator", "status": "verified",
            "source": {"url": source["url"], "topic_pin": "Unit fixture", "content_path": content.name,
                       "content_sha256": sha(content.read_bytes())}}
        path = root / f"source-{index}.json"
        path.write_text(json.dumps(artifact))
        receipts.append({"id": artifact["receipt_id"], "path": path.name,
                         "sha256": sha(path.read_bytes()), "authority": "host_independent_evaluator"})
    return {"schema": "robit.omni.long-native-readiness.v1", "case_id": case["case_id"],
            "task_signature_sha256": case["task_signature_sha256"], "source_revision": REVISION, "receipts": receipts}


def test_source_preflight_does_not_remove_final_capability_gate_or_release(fixture, tmp_path):
    case, work, _, capability, _ = fixture
    pack = source_pack(case, tmp_path)
    options = {"revision": REVISION, "work": work, "evidence_root": tmp_path}
    assert cases.source_readiness(case, pack, **options) == []
    assert cases.readiness(case, pack, **options) == ["controlled_perturbation_injector_not_implemented"]
    path = tmp_path / "amendment-capability.json"
    path.write_text(json.dumps(capability))
    assert cases.readiness(case, pack, amendment_capability=path, **options) == []
    assert not (work / amendments.RELEASE_PATH).exists()
    path = work / "solver-capability.json"
    path.write_text(json.dumps(capability))
    assert cases.readiness(case, pack, amendment_capability=path, **options)


def test_capture_sources_keeps_injector_capability_separate_from_primary_source_receipts(fixture, tmp_path):
    case, work, _, capability, _ = fixture
    pack = source_pack(case, tmp_path)
    pack_path, cap_path = tmp_path / "readiness.json", tmp_path / "amendment-capability.json"
    pack_path.write_text(json.dumps(pack))
    cap_path.write_text(json.dumps(capability))
    result = cases.capture_source_proofs(case, pack_path, tmp_path / "captured", revision=REVISION,
                                         work=work, amendment_capability=cap_path)
    assert len(result["sources"]) == len(case["sources_to_acquire"])
    assert all(row["id"].startswith("source:") for row in result["sources"])
    assert not (work / amendments.RELEASE_PATH).exists()


def test_setup_probe_runs_only_while_fresh_blank_calc_is_current(fixture, tmp_path, monkeypatch):
    case, work, _, _, _ = fixture
    active, probe_calls = [], []
    class Desktop:
        environment = {"PATH": "/usr/bin"}
        def open_blank(self, app, *args, **kwargs):
            active[:] = [app]
            window = WINDOW if app == "calc" else {"window_id": str(500 + len(active)), "process_id": 1000, "process_start": "42"}
            return type("Window", (), {"window_id": window["window_id"], "pid": str(window["process_id"]),
                                        "process_start": window["process_start"]})()
    class Observer:
        def fresh(self):
            xid = WINDOW["window_id"] if active == ["calc"] else "501"
            return {"phase": "setup", "window_id": xid, "outside_owned_foreground": False,
                    "state": {"scope": {"window_id": xid}, "nodes": [{"id": "native"}]}}
    def document_probe(window, environment):
        assert active == ["calc"]
        probe_calls.append(window)
        return probe()
    monkeypatch.setattr(runner, "probe_calendar_document", document_probe)
    rows = runner.prepare_blank_apps(Desktop(), Observer(), case, work, tmp_path, "firefox")
    assert probe_calls == [WINDOW]
    assert rows[1]["application"] == "calc" and rows[1]["calendar_document_probe"]["url"] == ""
    assert active == ["files"]


def test_first_saved_bytes_are_frozen_outside_solver_workbench_before_revision(fixture, tmp_path):
    report = completed_receipt(fixture, tmp_path)
    saved = fixture[1] / "Drafts/early-calendar.ods"
    first = saved.read_bytes()
    saved.write_bytes(workbook("revised after amendment"))
    assert (tmp_path / "amendment-events/first-saved-workbook.ods").read_bytes() == first
    assert amendments.verified_event(fixture[0], tmp_path, report)
    frozen = tmp_path / "amendment-events/first-saved-workbook.ods"
    frozen.chmod(0o600)
    frozen.write_bytes(workbook("substituted"))
    assert not amendments.verified_event(fixture[0], tmp_path, report)
