"""Partial-progress milestones come only from recorded observations and grant no pass."""
import json

import pytest

from runtime.long_horizon_cases import independent_verdict, progress_milestones


def case():
    with open("tests/live/native_long_horizon/tasks.json") as handle:
        tasks = json.load(handle)
    return (tasks["tasks"] if isinstance(tasks, dict) else tasks)[0]


def bundle(tmp_path, nodes=(), url=None):
    state = {"nodes": list(nodes), "document_evidence": (
        {"status": "observed", "url": url} if url else {"status": "unavailable"})}
    (tmp_path / "ui-observations.jsonl").write_text(json.dumps({"elapsed_s": 1, "state": state}) + "\n")
    return tmp_path


def brief():
    return next(item for item in case()["fixture"]["input_files"] if item["path"] == "Input/task-brief.txt")


def test_complete_input_text_and_declared_source_host_count_as_progress(tmp_path):
    root = bundle(tmp_path, [{"id": "editor", "value": "header\n" + brief()["content_utf8"]}],
                  url="https://docs.python.org/3.12/library/pathlib.html")
    result = progress_milestones(case(), root, {"required_outputs_present": {"Research/migration-dossier.txt": True}})
    assert result["inputs_read_back"]["Input/task-brief.txt"] is True
    assert result["official_source_urls_loaded"] == ["https://docs.python.org/3.12/library/pathlib.html"]
    assert result["required_outputs_created"] == 1
    assert result["authority"] == "progress_only_no_completion_credit"


@pytest.mark.parametrize("nodes,url", [
    ([{"id": "editor", "value": brief()["content_utf8"][:40]}], None),          # partial text
    ([{"id": "editor", "value": brief()["content_utf8"], "sensitive": True}], None),
    ([], "https://docs.python.org.example.net/3.12/"),                          # foreign host
])
def test_partial_sensitive_or_foreign_evidence_is_not_progress(tmp_path, nodes, url):
    result = progress_milestones(case(), bundle(tmp_path, nodes, url), {"required_outputs_present": {}})
    assert not any(result["inputs_read_back"].values()) and result["official_source_urls_loaded"] == []


def test_milestones_never_change_the_independent_verdict(tmp_path):
    evidence = {"inputs_unchanged": True, "required_outputs_present": {name: False for name in case()["required_outputs"]},
                "files": {}, "automatic_criterion_evidence": {}}
    before = independent_verdict(case(), evidence, method_verified=True)
    progress_milestones(case(), bundle(tmp_path, [{"id": "e", "value": brief()["content_utf8"]}]), evidence)
    assert independent_verdict(case(), evidence, method_verified=True) == before
    assert before["passed"] is False
