"""Offline orchestration panel integrity, request rendering and response grading.

This module never contacts a model, sends desktop input or changes a service.
Rendering exposes only source-bound model inputs, never the acceptance oracle.
Finite choice correctness is not an effect receipt or a whole-task result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "tests/live/native_orchestration_acceptance"
MODELS = (
    "robit/ornith-1.5-omni-audio-bridge:q4km",
    "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km",
)
MAX_BYTES = 8 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    content = Path(path).read_bytes()
    if len(content) > MAX_BYTES:
        raise ValueError("Panel artifact exceeds byte bound")
    return json.loads(content)


def load_panel(root=PANEL):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "robit.omni.orchestration-acceptance-panel.v1":
        raise ValueError("Unknown panel schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Panel manifest path must be a local filename")
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Frozen panel artifact changed: {name}")
    for name, expected in manifest.get("evaluator_files", {}).items():
        if name != "runtime/verify_orchestration_contracts.py":
            raise ValueError("Unknown evaluator path")
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen evaluator changed; version the panel before comparison")
    cases = {}
    for case_id in manifest["case_ids"]:
        if Path(case_id).name != case_id:
            raise ValueError("Case identifier must be a local filename")
        case = read_json(root / (case_id + ".json"))
        if case["case_id"] != case_id or digest(case["input"]) != case["input_sha256"]:
            raise ValueError("Case input binding changed")
        wire = case["input"].get("raw_wire_utf8")
        if wire is not None and (
                hashlib.sha256(wire.encode()).hexdigest() != case["source"]["request_body_sha256"]
                or json.loads(wire) != case["input"]["wire_request"]):
            raise ValueError("Archived raw request binding changed")
        cases[case_id] = case
    return manifest, cases


def render_request(case, model, *, condition="baseline", replacement_view=None,
                   replacement_system=None, candidate_sources=None):
    """Render one unexecuted proposal; candidate inputs must identify code sources.

    The root executor obtains replacement_view from the candidate's actual
    pointer_resolution_context function. No expected choice or historical reply
    is copied into the request. Sampler settings and media policy stay pinned.
    """
    if model not in MODELS or condition not in {"baseline", "candidate"}:
        raise ValueError("Unknown model or comparison condition")
    if condition == "baseline" and (replacement_view is not None or replacement_system is not None):
        raise ValueError("Baseline input cannot change")
    if condition == "candidate" and (not isinstance(candidate_sources, dict) or not candidate_sources
            or any(not isinstance(path, str) or not path or not isinstance(value, str)
                   or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                   for path, value in candidate_sources.items())):
        raise ValueError("Candidate rendering requires exact source hashes")
    request = copy.deepcopy(case["input"]["wire_request"])
    original_settings = {key: copy.deepcopy(request.get(key)) for key in
                         ("options", "think", "stream", "omni", "speech_mode", "response_modalities")}
    request["model"] = model
    if replacement_system is not None:
        request["messages"][0]["content"] = replacement_system
    if replacement_view is not None:
        request["messages"][-1]["content"] = json.dumps(replacement_view, ensure_ascii=False,
                                                        separators=(",", ":"))
        schema = replacement_view.get("response_schema")
        if not isinstance(schema, dict):
            raise ValueError("Candidate view must carry its actual response schema")
        request["response_format"]["json_schema"]["schema"] = copy.deepcopy(schema)
    if any(request.get(key) != value for key, value in original_settings.items()):
        raise ValueError("Comparison changed model sampler or execution settings")
    return {"schema": "robit.omni.orchestration-rendered-request.v1", "case_id": case["case_id"],
            "input_sha256": case["input_sha256"], "condition": condition, "model": model,
            "source_request_body_sha256": case["source"]["request_body_sha256"],
            "candidate_sources": copy.deepcopy(candidate_sources), "request": request,
            "request_sha256": digest(request), "execution_status": "not_executed",
            "whole_task_success_credit": False}


def grade_response(case, response, *, response_schema=None):
    """Grade finite choices; explanatory predictions require separate review."""
    if isinstance(response, dict) and "message" in response:
        response = response["message"].get("content")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except ValueError:
            return {"status": "invalid_response", "passed": False}
    schema = response_schema or case["input"]["pointer_view"]["response_schema"]
    try:
        Draft202012Validator(schema).validate(response)
    except (ValidationError, TypeError):
        return {"status": "invalid_response", "passed": False}
    if case["oracle"]["kind"] != "finite_choice":
        return {"status": "pending_independent_review", "passed": None,
                "choice": response.get("choice"), "whole_task_success_credit": False}
    passed = response.get("choice") in case["oracle"]["accepted_choices"]
    return {"status": "choice_correct" if passed else "choice_incorrect", "passed": passed,
            "choice": response.get("choice"), "prediction_status": "pending_independent_review",
            "whole_task_success_credit": False}


def grade_records(cases, records):
    """Reject missing/duplicate bindings; report unrun cells in the denominator.

    Trace files are measured, not accepted from self-declared hash fields.
    Their presence does not independently prove GPU residency; the executor
    must attach broker/worker evidence in the separate run manifest.
    """
    expected = {(case_id, model, condition) for case_id in cases for model in MODELS
                for condition in ("baseline", "candidate")}
    seen, grades = set(), []
    for record in records:
        key = (record.get("case_id"), record.get("model"), record.get("condition"))
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate comparison cell")
        seen.add(key)
        case = cases[key[0]]
        rendered = read_json(record["rendered_request_path"])
        request = rendered.get("request")
        if (rendered.get("case_id"), rendered.get("model"), rendered.get("condition")) != key:
            raise ValueError("Rendered request cell changed")
        if (rendered.get("input_sha256") != case["input_sha256"]
                or rendered.get("request_sha256") != digest(request)):
            raise ValueError("Rendered request binding changed")
        baseline = render_request(case, key[1])["request"]
        if key[2] == "baseline" and request != baseline:
            raise ValueError("Baseline request changed")
        if key[2] == "candidate":
            # Check provenance syntax and immutable model/settings independently
            # of any fields declared by the renderer record.
            render_request(case, key[1], condition="candidate",
                           candidate_sources=rendered.get("candidate_sources"))
            if (set(request) != set(baseline)
                    or [m.get("role") for m in request.get("messages", [])]
                    != [m.get("role") for m in baseline["messages"]]):
                raise ValueError("Candidate changed request structure")
            for field in baseline.keys() - {"messages", "response_format"}:
                if request.get(field) != baseline[field]:
                    raise ValueError("Candidate changed immutable request settings")
        response_bytes = Path(record["response_path"]).read_bytes()
        if len(response_bytes) > MAX_BYTES:
            raise ValueError("Response exceeds byte bound")
        response = json.loads(response_bytes)
        schema = request["response_format"]["json_schema"]["schema"]
        result = grade_response(case, response, response_schema=schema)
        grades.append({"case_id": key[0], "model": key[1], "condition": key[2], **result,
                       "request_sha256": digest(request),
                       "response_file_sha256": hashlib.sha256(response_bytes).hexdigest()})
    finite_expected = sum(c["oracle"]["kind"] == "finite_choice" for c in cases.values()) * 4
    return {"schema": "robit.omni.orchestration-panel-grades.v1", "grades": grades,
            "planned_cells": len(expected), "recorded_cells": len(seen),
            "unrun_cells": [{"case_id": key[0], "model": key[1], "condition": key[2]}
                            for key in sorted(expected - seen)],
            "finite_choice_denominator": finite_expected,
            "finite_choices_correct": sum(g.get("passed") is True for g in grades),
            "pending_review": sum(g.get("passed") is None for g in grades),
            "whole_task_success_credit": False, "GPU_residency_verified_by_this_tool": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--records", type=Path, help="JSON list of independently captured response files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest, cases = load_panel(args.panel)
    if args.render_dir:
        if not args.model:
            parser.error("--render-dir requires --model")
        args.render_dir.mkdir(parents=True, exist_ok=False)
        for case_id, case in cases.items():
            value = render_request(case, args.model)
            (args.render_dir / (case_id + ".json")).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    result = (grade_records(cases, read_json(args.records)) if args.records else
              {"panel_integrity": True, "case_count": len(cases),
               "finite_choice_cases": sum(c["oracle"]["kind"] == "finite_choice" for c in cases.values()),
               "review_only_cases": manifest["review_only_case_ids"], "whole_task_success_credit": False})
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
