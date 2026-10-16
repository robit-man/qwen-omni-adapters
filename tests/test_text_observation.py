"""Native text knowledge stays explicit, bounded and separate from input authority."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_ui import Fixture, Node
from test_ui_task import completed_tools, final, no_effect_fixture, no_effect_phase

from clients.python_client import printable_response
from portal.app import _model_ui_observation
from portal.computer_use import _digest, _effect_result, _policy_projection
from portal.native_ui import NativeUiAdapter, NativeUiUnavailable
from portal.ui_history import _digest as history_digest
from portal.ui_history import _retired
from portal.ui_task import _compact_result, _new_verified_ui_prefix, _zero_effect_ui_phase_key
from qwen_omni_adapters.computer_use import (
    ComputerUseError,
    RapidUiController,
    _snapshot,
    parse_predicates,
)
from qwen_omni_adapters.contract import ADAPTER_SCHEMA, parse_adapter_request
from runtime.adapter_server import build_language_payload


@pytest.fixture
def native(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU provider fixture must not call host processes")

    monkeypatch.setattr("portal.native_ui.subprocess.run", forbidden)
    fixture = Fixture("text-knowledge")
    fixture.title.value = "alpha beta"
    fixture.title.selection_range = None
    fixture.title.selection_count = 0
    fixture.reads = []
    text = fixture.api.Text

    def read(name, node, *args):
        fixture.reads.append((name, node.path, args))
        if name == "count":
            return len(node.value)
        if name == "text":
            return node.value[args[0]:args[1]]
        if name == "selections":
            return getattr(node, "selection_count", 0)
        selection = node.selection_range
        return SimpleNamespace(start_offset=selection[0], end_offset=selection[1])

    text.get_character_count = lambda node: read("count", node)
    text.get_text = lambda node, start, end: read("text", node, start, end)
    text.get_n_selections = lambda node: read("selections", node)
    text.get_selection = lambda node, index: read("range", node, index)
    text.add_selection = lambda *_args: True
    text.set_selection = lambda *_args: True
    return fixture


def title(state):
    return next(node for node in state["nodes"] if node["name"] == "Title")


def coverage(text, *, total=None, selections=0):
    return {"character_count": len(text) if total is None else total,
            "captured_range": {"start": 0, "end": len(text)},
            "truncated": total is not None and len(text) < total, "selection_count": selections}


def test_zero_unknown_and_invalid_range_keep_distinct_positive_facts(native):
    zero = native.adapter.observe()
    native.title.selection_count, native.title.selection_range = 1, (0, 0)
    unknown_range = native.adapter.observe()
    native.title.selection_count = -1
    invalid_count = native.adapter.observe()
    assert title(zero)["text_observation"] == coverage("alpha beta")
    assert title(unknown_range)["text_observation"]["selection_count"] == 1
    assert "selection_count" not in title(invalid_count)["text_observation"]
    assert all("text_selection" not in title(state) for state in (zero, unknown_range, invalid_count))
    assert zero["revision"] == unknown_range["revision"] == invalid_count["revision"]
    assert all("select_text" in title(state)["operations"] for state in (zero, unknown_range, invalid_count))


@pytest.mark.parametrize("value", ["", "e\u0301🙂\r\n文", "a" * 2048, "🙂" * 2049])
def test_native_unicode_prefix_coverage_uses_actual_character_return(native, value):
    native.title.value = value
    state = native.adapter.observe()
    node = title(state)
    assert node["value"] == value[:2048]
    assert node["text_observation"] == coverage(value[:2048], total=len(value))
    _snapshot(state)
    assert [row for row in native.reads if row[1] == "/title"] == [
        ("count", "/title", ()), ("text", "/title", (0, min(2048, len(value)))),
        ("selections", "/title", ())]


def test_short_native_return_is_prefix_not_complete_or_grapheme_claim(native):
    native.title.value = "e\u0301🙂more"
    native.api.Text.get_text = lambda node, start, end: node.value[start:min(end, 1)]
    state = native.adapter.observe()
    assert title(state)["text_observation"] == coverage("e", total=len(native.title.value))
    _snapshot(state)


@pytest.mark.parametrize("returned", [None, 42, b"alpha", "returned more than the requested ten characters"])
def test_coerced_or_oversized_return_never_proves_text_coverage(native, returned):
    native.api.Text.get_text = lambda *_args: returned
    metadata = title(native.adapter.observe())["text_observation"]
    assert metadata == {"character_count": 10, "selection_count": 0}


@pytest.mark.parametrize("count", [-1, False, True, 2**31])
def test_invalid_original_character_count_is_not_clamped_into_metadata(native, count):
    native.api.Text.get_character_count = lambda _node: count
    metadata = title(native.adapter.observe())["text_observation"]
    assert metadata == {"selection_count": 0}


@pytest.mark.parametrize("count", [-1, False, True, 2**31])
def test_invalid_selection_count_stays_unknown(native, count):
    native.title.selection_count, native.title.selection_range = count, (0, 1)
    metadata = title(native.adapter.observe())["text_observation"]
    assert "selection_count" not in metadata
    assert metadata["character_count"] == 10


@pytest.mark.parametrize("method", ["get_character_count", "get_text", "get_n_selections", "get_selection"])
def test_existing_native_read_errors_are_not_recovered_as_empty_evidence(native, method):
    native.title.selection_count, native.title.selection_range = 1, (0, 1)

    def unavailable(*_args):
        raise RuntimeError("PRIVATE_PROVIDER_ERROR")

    setattr(native.api.Text, method, unavailable)
    with pytest.raises(NativeUiUnavailable) as error:
        native.adapter.observe()
    assert error.value.code == "native_observation_unavailable"


def test_missing_interface_methods_and_exhausted_text_budget_remain_unknown(native):
    del native.api.Text.add_selection
    assert "selection_count" not in title(native.adapter.observe())["text_observation"]
    assert not any(name == "selections" for name, *_ in native.reads)
    native.window.children = [Node(f"/prefix-{index}", "text", interfaces=["Text"], value="a" * 2048)
                              for index in range(16)] + [native.title]
    native.reads.clear()
    node = title(native.adapter.observe())
    assert "text_observation" not in node and "value" not in node
    assert not any(path == "/title" for _, path, _ in native.reads)
    native.title.interfaces.remove("Text")
    native.window.children = [native.title]
    assert "text_observation" not in title(native.adapter.observe())


def test_count_multiple_reads_only_the_existing_first_selection(native):
    native.title.selection_count, native.title.selection_range = 3, (1, 4)
    node = title(native.adapter.observe())
    assert node["text_observation"]["selection_count"] == 3
    assert node["text_selection"] == {"start": 1, "end": 4}
    assert [row for row in native.reads if row[1] == "/title"] == [
        ("count", "/title", ()), ("text", "/title", (0, 10)),
        ("selections", "/title", ()), ("range", "/title", (0,))]


def test_sensitive_and_failed_budget_observation_export_no_count_certificate(native):
    state = native.adapter.observe()
    secret = next(node for node in state["nodes"] if node["sensitive"])
    assert "text_observation" not in secret
    assert not any(path == native.secret.path for _, path, _ in native.reads)
    native.bridge.MAX_READS = 1
    with pytest.raises(NativeUiUnavailable, match="budget"):
        native.adapter.observe()


@pytest.mark.parametrize("bad", [
    None, {}, {"extra": 0}, {"character_count": -1}, {"character_count": True},
    {"character_count": None}, {"character_count": "0"}, {"character_count": 1.0},
    {"character_count": 2**31}, {"selection_count": False}, {"selection_count": "0"},
    {"selection_count": 0.0}, {"selection_count": None},
    {"character_count": 10, "captured_range": {"start": 0, "end": 10}},
    {"character_count": 10, "truncated": False},
    {"character_count": 10, "captured_range": {"start": False, "end": 10}, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 1, "end": 10}, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 0, "end": 11}, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 0, "end": 9}, "truncated": True},
    {"character_count": 10, "captured_range": {"start": 0, "end": 10}, "truncated": True},
    {"character_count": 10, "captured_range": {"start": 0, "end": 10}, "truncated": 0},
    {"character_count": 10, "captured_range": None, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 0}, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 0, "end": True}, "truncated": False},
    {"character_count": 10, "captured_range": {"start": 0, "end": 10, "extra": 0}, "truncated": False},
])
def test_shared_snapshot_rejects_false_or_malformed_certificates(native, bad):
    state = native.adapter.observe()
    title(state)["text_observation"] = bad
    with pytest.raises(ComputerUseError):
        _snapshot(state)


@pytest.mark.parametrize("metadata", [{"selection_count": 0}, {"selection_count": 3},
                                      {"character_count": 10}, coverage("alpha beta")])
def test_partial_known_fields_and_legacy_absence_are_valid_without_mutation(native, metadata):
    state = native.adapter.observe()
    title(state)["text_observation"] = metadata
    original = copy.deepcopy(state)
    assert _snapshot(state) == original and state == original
    title(state).pop("text_observation")
    assert "text_observation" not in title(_snapshot(state))


def test_positive_provider_limit_and_unseen_length_change_remain_knowledge(native):
    native.title.value = "a" * 2048
    native.api.Text.get_character_count = lambda _node: 2**31 - 1
    before = native.adapter.observe()
    assert title(before)["text_observation"] == coverage(native.title.value, total=2**31 - 1)
    _snapshot(before)
    native.api.Text.get_character_count = lambda _node: 2**31 - 2
    after = native.adapter.observe()
    assert title(before)["text_observation"] != title(after)["text_observation"]
    assert before["revision"] == after["revision"]


def test_contradictory_selection_and_protected_metadata_are_not_public(native):
    state = native.adapter.observe()
    title(state)["text_selection"] = {"start": 0, "end": 1}
    with pytest.raises(ComputerUseError, match="Contradictory"):
        _snapshot(state)
    title(state)["sensitive"] = True
    public = _snapshot(state)
    assert "text_observation" not in title(public) and "value" not in title(public)
    adapter = NativeUiAdapter("text-knowledge", bridge_runner=lambda _request: copy.deepcopy(state))
    assert all("text_observation" not in node for node in adapter.observe()["nodes"] if node["sensitive"])
    assert all("text_observation" not in node for node in _model_ui_observation(state)["nodes"] if node["sensitive"])
    title(state)["sensitive"] = False
    title(state)["text_observation"] = {"character_count": 0, "selection_count": 1}
    with pytest.raises(ComputerUseError, match="Contradictory"):
        _snapshot(state)


def test_projection_compaction_and_history_preserve_zero_but_not_executable_authority(native):
    state = native.adapter.observe()
    node = title(state)
    original = copy.deepcopy(state)
    projected = _model_ui_observation(state)
    assert title(projected)["text_observation"] == node["text_observation"]
    outcome = {"action": {"node_id": node["id"]}, "after": state, "receipt": {"status": "dispatched"}}
    compact = _compact_result({"outcomes": [outcome]})
    assert compact["outcomes"][0]["observed_effects"][0]["text_observation"] == node["text_observation"]
    historical = _retired({"state": state}, (history_digest(state["session_id"]), "desktop"),
                          {"target": {"node_id": node["id"]}})
    evidence = historical["state"]["historical_control_evidence"][0]
    assert evidence["text_observation"] == node["text_observation"] and "operations" not in evidence
    assert historical["state"]["history_projection"]["current_observation"] is False
    actions = [{"id": "a", "kind": "action", "node_id": node["id"]}, {"id": "abstain", "kind": "abstain"}]
    policy, _ = _policy_projection(state, actions, None)
    assert policy["node_defaults"]["text_observation"] == node["text_observation"]
    assert state == original


def test_policy_shared_defaults_never_impute_count_to_unknown_node(native):
    state = native.adapter.observe()
    known = title(state)
    unknown = copy.deepcopy(known)
    unknown["id"] = "unknown-text"
    unknown.pop("text_observation")
    state["nodes"].append(unknown)
    actions = [{"id": "known", "kind": "action", "node_id": known["id"]},
               {"id": "unknown", "kind": "action", "node_id": unknown["id"]},
               {"id": "abstain", "kind": "abstain"}]
    policy, _ = _policy_projection(state, actions, None)
    assert "text_observation" not in policy["node_defaults"]
    assert policy["nodes"]["n0"]["text_observation"]["selection_count"] == 0
    assert "text_observation" not in policy["nodes"]["n1"]


def test_model_node_omission_and_aggregate_text_clipping_do_not_change_source_coverage(native):
    native.title.value = "a" * 2049
    state = native.adapter.observe()
    expected = title(state)["text_observation"]
    for index in range(5):
        node = copy.deepcopy(title(state))
        node.update(id=f"extra-{index}", value="b" * 2048)
        state["nodes"].append(node)
    projected = _model_ui_observation(state, references=(title(state)["id"],))
    assert title(projected)["text_observation"] == expected
    assert projected["model_projection"]["text_truncated"] is True
    assert projected["model_projection"]["omitted_visible_nodes"] > 0
    assert all(node["text_observation"] == expected for node in projected["nodes"] if "text_observation" in node)


def test_metadata_clarification_does_not_reset_uncertain_idempotency_or_grant_progress(native):
    native.title.state.values.add("FOCUSED")
    setter_calls = []

    def failed(*_args):
        setter_calls.append(1)
        raise RuntimeError("native setter uncertain")

    native.api.Text.add_selection = failed
    controller = RapidUiController(native.adapter, authorize=lambda *_args: True, max_observations=1)
    controller.observe()
    step = {"id": "range", "verb": "select_text", "target": {"name": "Title"},
            "args": {"start": 0, "end": 1},
            "expect": [{"kind": "text_selection", "node_id": "$target", "start": 0, "end": 1}]}
    candidate = next(item for item in controller.enumerate(step) if item["kind"] == "action")
    first = controller.execute(candidate["id"])
    native.title.selection_count = -1
    controller.observe()
    next_candidate = next(item for item in controller.enumerate(step) if item["kind"] == "action")
    assert next_candidate["idempotency_key"] == candidate["idempotency_key"]
    replay = controller.execute(next_candidate["id"])
    assert replay == first and setter_calls == [1] and first["status"] == "unresolved"
    after = native.adapter.observe()
    outcome = {"status": "verified", "action": candidate, "before": first["before"], "after": after,
               "receipt": {"status": "dispatched"}}
    effect = _effect_result(outcome)
    assert effect["effect_receipt"]["effect_state"] == "unchanged"
    assert _new_verified_ui_prefix({"outcomes": [effect]}, set()) is False
    with pytest.raises(ComputerUseError):
        parse_predicates([{"kind": "text_observation", "node_id": title(after)["id"], "selection_count": 0}])


def test_real_value_and_range_changes_still_change_semantic_revision(native):
    initial = native.adapter.observe()
    native.title.selection_count, native.title.selection_range = 1, (0, 1)
    selected = native.adapter.observe()
    native.title.value = "different text"
    changed = native.adapter.observe()
    assert len({initial["revision"], selected["revision"], changed["revision"]}) == 3


@pytest.mark.parametrize("project", [False, True])
def test_metadata_only_changes_cannot_renew_actual_zero_effect_phase_budget(project):
    fixture = no_effect_fixture([no_effect_phase()] * 4 + [{"kind": "complete", "message": "Unverified claim"}])
    observe = fixture.adapter.observe

    def clarified():
        state = observe()
        node = state["nodes"][0]
        node["value"] = "a" * 18
        count = 18 + len(fixture.packets)
        node["text_observation"] = coverage(node["value"], total=count, selections=1)
        return state

    fixture.adapter.observe = clarified
    if project:
        observe_receipt = fixture.observe
        fixture.observe = lambda: _model_ui_observation(observe_receipt())
    events = list(fixture.run())
    assert final(events)["portal"]["native_task"]["status"] == "blocked"
    assert [tool["ok"] for tool in completed_tools(events)] == [True, False, False, False]
    assert len(fixture.adapter.calls) == 2 and fixture.cancel_count == 1
    feedback = fixture.packets[2]["last_result"]
    assert feedback["error"] == "ui_phase_repeated_without_new_effect"
    assert feedback["phase_handoff"]["new_effect_count"] == 0
    assert [item["effect_receipt"]["effect_state"] for item in feedback["outcomes"]] == ["absent", "replayed", "replayed"]
    assert all(item["status"] == "verified" for item in feedback["outcomes"])
    assert feedback["outcomes"][1]["observed_effects"][0]["text_observation"]["character_count"] == 19
    draft = next(node for node in fixture.packets[1]["current_state"]["state"]["nodes"] if node["id"] == "draft")
    assert draft["text_observation"]["character_count"] == 19


def test_metadata_cannot_displace_legacy_nodes_at_exact_projection_budget():
    node = {"id": "one", "generation": "1", "scope_id": "form", "role": "textbox", "name": "Field",
            "value": "x", "operations": ["focus"], "visible": True, "enabled": True, "sensitive": False}
    second = {**node, "id": "two"}
    node["description"] = ""
    node["description"] = "d" * (12000 - len(json.dumps(node, ensure_ascii=False)) - len(json.dumps(second, ensure_ascii=False)))
    state = {"session_id": "s", "scope": {"id": "form", "kind": "native"}, "revision": "same", "nodes": [node, second]}
    legacy = _model_ui_observation(state)
    known = copy.deepcopy(state)
    for item in known["nodes"]:
        item["text_observation"] = coverage("x")
    original = copy.deepcopy(known)
    actual = _model_ui_observation(known)
    assert [item["id"] for item in actual["nodes"]] == ["one", "two"]
    assert actual["nodes"] == legacy["nodes"]  # No spare space, knowledge remains unknown.
    assert sum(len(json.dumps(item, ensure_ascii=False)) for item in actual["nodes"]) == 12000
    assert actual["model_projection"]["returned_node_count"] == legacy["model_projection"]["returned_node_count"] == 2
    assert actual["model_projection"]["omitted_visible_nodes"] == 0 and known == original


def _zero_effect_evidence():
    fixture = no_effect_fixture([])
    original = fixture.adapter.observe

    def known():
        state = original()
        state["nodes"][0]["value"] = "a" * 18
        state["nodes"][0]["text_observation"] = coverage("a" * 18, selections=1)
        state["nodes"].append({"id": "hidden", "role": "label", "name": "Hidden", "value": "legacy",
                               "visible": False, "enabled": True, "sensitive": False,
                               "generation": "1", "scope_id": "form", "operations": []})
        return state

    fixture.adapter.observe = known
    observed = fixture.observe()
    phase = no_effect_phase()
    result = fixture.execute("native_ui", {"operation": "workflow", "steps": phase["steps"],
                                         "final_predicates": phase["final_predicates"]})
    assert result["status"] == "complete"
    return observed, result


def test_projected_repeat_hash_removes_only_knowledge_and_verified_derived_hash():
    observed, result = _zero_effect_evidence()
    changed = copy.deepcopy(observed)
    changed["state"]["nodes"][0]["text_observation"] = coverage("a" * 18, total=19, selections=1)
    changed["content_sha256"] = _digest(changed["state"])
    before, after = _model_ui_observation(observed), _model_ui_observation(changed)
    originals = copy.deepcopy((before, after, result))
    assert _digest(before) != _digest(after) and observed["content_sha256"] != changed["content_sha256"]
    assert _zero_effect_ui_phase_key(result, before) == _zero_effect_ui_phase_key(result, after)
    assert _zero_effect_ui_phase_key(result, before) is not None
    assert (before, after, result) == originals
    # The original legacy-only signature is unchanged, including its raw hash.
    legacy = copy.deepcopy(observed)
    legacy["state"]["nodes"][0].pop("text_observation")
    legacy["content_sha256"] = _digest(legacy["state"])
    assert _zero_effect_ui_phase_key(result, before) == _zero_effect_ui_phase_key(result, _model_ui_observation(legacy))


@pytest.mark.parametrize("change", ["hidden_value", "revision", "scope", "generation", "operations", "top_authority", "action_args", "unmatched_hash"])
def test_repeat_normalization_preserves_all_legacy_bindings_and_hidden_changes(change):
    observed, result = _zero_effect_evidence()
    before = _model_ui_observation(observed)
    changed, changed_result = copy.deepcopy(observed), copy.deepcopy(result)
    if change == "hidden_value":
        changed["state"]["nodes"][-1]["value"] = "other legacy"
    elif change == "revision":
        changed["state"]["revision"] = "different"
    elif change == "scope":
        changed["state"]["scope"]["window_id"] = "123"
    elif change == "generation":
        changed["state"]["nodes"][0]["generation"] = "2"
    elif change == "operations":
        changed["state"]["nodes"][0]["operations"].append("type_text")
    elif change == "top_authority":
        changed["state"]["text_observation"] = {"authority": "legacy field, never stripped"}
    elif change == "action_args":
        changed_result["outcomes"][0]["action"]["args"]["text_observation"] = "legacy action data"
    changed["content_sha256"] = _digest(changed["state"])
    after = _model_ui_observation(changed)
    if change == "unmatched_hash":
        after["content_sha256"] = "0" * 64
    assert _zero_effect_ui_phase_key(result, before) != _zero_effect_ui_phase_key(changed_result, after)


@pytest.mark.parametrize("fault", ["missing_raw", "missing_semantic", "invalid_raw", "invalid_semantic", "unmatched_raw"])
def test_unproven_projection_anchors_cannot_establish_phase_equivalence(fault):
    observed, result = _zero_effect_evidence()
    projected = _model_ui_observation(observed)
    anchors = projected["state"]["model_projection"]
    if fault.startswith("missing"):
        anchors.pop("raw_source_sha256" if fault.endswith("raw") else "text_observation_semantic_sha256")
    else:
        anchors["raw_source_sha256" if fault.endswith("raw") else "text_observation_semantic_sha256"] = "0" * 64 if fault.startswith("unmatched") else "not-a-digest"
    assert _zero_effect_ui_phase_key(result, projected) is None


def test_portable_adapter_and_python_client_keep_ui_evidence_opaque(native):
    state = native.adapter.observe()
    tool_content = json.dumps({"state": state}, ensure_ascii=False)
    body = {"model": "fixture", "messages": [{"role": "user", "content": "Continue"},
            {"role": "tool", "content": tool_content, "tool_name": "native_ui"}],
            "omni": {"schema": ADAPTER_SCHEMA, "task": "chat", "tool_routing": "client"},
            "stream": False, "think": False, "speech_mode": "never", "response_modalities": ["text"]}
    upstream = build_language_payload(parse_adapter_request(body), None, "fixture", "ollama")
    assert next(message["content"] for message in upstream["messages"] if message["role"] == "tool") == tool_content
    response = {"model": "fixture", "message": {"role": "assistant", "content": "Fixture reply"},
                "adapter": {"schema": ADAPTER_SCHEMA, "task": "chat", "route": ["language"],
                            "tool_evidence": {"state": state}}, "done": True}
    original = copy.deepcopy(response)
    assert printable_response(response, include_audio_base64=False) == original and response == original
    assert not {"text_observation", "selection_count", "character_count"} & upstream.keys()


def test_javascript_example_accepts_opaque_additive_ui_evidence(tmp_path):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"CPU fetch double; never sent to a service")
    response = {"model": "fixture", "message": {"role": "assistant", "content": "Fixture reply"},
                "adapter": {"schema": ADAPTER_SCHEMA, "task": "chat", "route": ["language"],
                            "tool_evidence": {"text_observation": {"selection_count": 0}}}, "done": True}
    preload = tmp_path / "mock-fetch.mjs"
    preload.write_text("import assert from 'node:assert/strict';\n"
        "globalThis.fetch = async (_url, request) => { const body=JSON.parse(request.body);\n"
        "assert.equal(body.omni.schema,'robit.ollama.omni-adapter.v1');\n"
        "assert.equal(body.omni.tool_routing,'client');\n"
        "assert.equal('text_observation' in body,false);\n"
        "return {ok:true,json:async()=>(" + json.dumps(response) + ")}; };\n")
    client = Path(__file__).resolve().parents[1] / "clients/javascript_client.mjs"
    process = subprocess.run(["node", "--import", str(preload), str(client), str(audio)],
                             capture_output=True, text=True, timeout=10, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "Fixture reply"
