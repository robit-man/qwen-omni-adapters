from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from runtime import verify_real_firefox as runner

SCENARIO = runner.SCENARIOS["python-numbers"]
MARKER = "copper finch willow north"
QUOTE = "The interpreter acts as a simple calculator: you can type an expression at it and it will write the value."
URL = SCENARIO["source"]
RECORD = {"marker": MARKER, "source_url": URL, "quote": QUOTE}


def test_setup_preserves_native_capability_error_instead_of_missing_state():
    def transport(request):
        return httpx.Response(200, json={"result": {"error": "native_ui_unavailable",
                                                   "code": "native_foreground_unavailable"}})
    with (httpx.Client(transport=httpx.MockTransport(transport)) as client,
          pytest.raises(RuntimeError, match="native_foreground_unavailable")):
        runner.tool(client, "http://127.0.0.1:9020", "native_ui", {"operation": "observe"})


def state(*, window="new-window", visible=True):
    return {"revision": "fresh-revision", "scope": {"id": "scope-1", "kind": "desktop", "window_id": window},
            "text": "untrusted aggregate must not establish visibility",
            "nodes": [{"id": "address", "role": "textbox", "name": "Search or enter address", "value": URL,
                       "visible": True, "enabled": True, "operations": ["set_value"]},
                      {"id": "fact", "role": "paragraph", "name": QUOTE, "visible": visible, "operations": []}]}


def event(name, arguments, result):
    return {"type": "tool", "tools": [{"name": name, "arguments": arguments, "result": json.dumps(result)}]}


def outcome(verb, *, window="new-window", backend="x11_guarded_xtest", revision="r2"):
    before, after = state(window=window), state(window=window)
    before["revision"], after["revision"] = "r1", revision
    after["nodes"].append({"id": "find", "role": "textbox", "name": "Find in page", "value": "Numbers", "visible": True})
    return {"status": "verified", "receipt": {"status": "dispatched", "backend": backend},
            "action": {"id": verb, "verb": verb}, "before": before, "after": after}


def evidence():
    memory = {"scope": "browser_session", "results": [{"value": json.dumps(RECORD)}]}
    first = [event("native_ui", {"operation": "workflow"}, {"outcomes": [outcome("click"), outcome("scroll")]}),
             event("memory_write", {"topic": "docs", "key": MARKER, "value": json.dumps(RECORD)}, {"stored": True})]
    second = [event("memory_search", {"query": MARKER}, memory)]
    return first, memory, second


def checks(first=None, current=None, memory=None, second=None, reply=None):
    initial, stored, recalled = evidence()
    return runner.acceptance(initial if first is None else first, state() if current is None else current,
                             stored if memory is None else memory, recalled if second is None else second,
                             QUOTE + " " + URL if reply is None else reply, MARKER, "new-window", SCENARIO)


def test_exact_native_browser_actions_visible_source_and_two_turn_memory_pass():
    assert all(checks().values())


@pytest.mark.parametrize("field,bad", [("marker", "another session marker"), ("source_url", "https://example.com/"),
                                       ("quote", "The interpreter acts as a simple calculator but this sentence is invented.")])
def test_wrong_marker_source_or_unseen_sentence_fails(field, bad):
    first, memory, second = evidence()
    changed = {**RECORD, field: bad}
    memory["results"][0]["value"] = json.dumps(changed)
    result = checks(first=first, memory=memory, second=second)
    assert result["fresh_address_and_visible_source_fact"] is False
    assert not all(result.values())


def test_hidden_source_text_or_page_text_mimicking_an_addressbar_cannot_prove_source():
    hidden = state(visible=False)
    hidden["text"] = QUOTE
    assert checks(current=hidden)["fresh_address_and_visible_source_fact"] is False
    mimicked = state()
    mimicked["nodes"][0]["role"] = "paragraph"
    assert checks(current=mimicked)["fresh_address_and_visible_source_fact"] is False


@pytest.mark.parametrize("role,operations", [("textbox", ["set_value"]), ("toolbar", []), ("paragraph", ["set_value"])])
def test_quote_in_find_or_chrome_cannot_substitute_for_rendered_document(role, operations):
    current = state(visible=False)
    current["nodes"].append({"id": "model-entered-quote", "role": role, "name": "Find in page",
                             "value": QUOTE, "visible": True, "operations": operations})
    assert checks(current=current)["fresh_address_and_visible_source_fact"] is False


def test_actual_readonly_atspi_document_text_remains_valid_source():
    current = state(visible=False)
    current["nodes"].append({"id": "actual-web-document", "role": "document_web", "name": "Python tutorial",
                             "value": QUOTE, "visible": True, "operations": ["scroll", "click"]})
    assert checks(current=current)["fresh_address_and_visible_source_fact"] is True


def test_final_prose_and_memory_alone_do_not_prove_native_find_scroll_or_click():
    first, _, _ = evidence()
    result = checks(first=first[1:])
    assert result["actual_mouse_click_verified"] is False
    assert result["actual_mouse_scroll_verified"] is False
    assert result["firefox_find_value_observed"] is False
    assert not all(result.values())


def test_semantic_invoke_and_unchanged_scroll_do_not_count_as_real_pointer_progress():
    first, _, _ = evidence()
    result = json.loads(first[0]["tools"][0]["result"])
    result["outcomes"] = [outcome("invoke", backend="atspi"), outcome("scroll", revision="r1")]
    first[0]["tools"][0]["result"] = json.dumps(result)
    checked = checks(first=first)
    assert checked["actual_mouse_click_verified"] is False
    assert checked["actual_mouse_scroll_verified"] is False


@pytest.mark.parametrize("name", ["browser_interact", "computer_use", "shell", "web_fetch", "web_search", "workspace_file"])
def test_alternative_execution_methods_are_retained_as_failed_acceptance(name):
    first, _, _ = evidence()
    first.append(event(name, {}, {}))
    assert checks(first=first)["only_native_and_memory_method"] is False


def test_discovery_is_limited_to_the_two_declared_capabilities():
    first, _, _ = evidence()
    first.extend([event("tool_search", {"family": "desktop"}, {}), event("tool_search", {"family": "memory"}, {})])
    assert checks(first=first)["only_native_and_memory_method"] is True
    first.append(event("tool_search", {"family": "shell"}, {}))
    assert checks(first=first)["only_native_and_memory_method"] is False


def test_recall_needs_actual_search_and_exact_stored_fact_not_prior_final_prose():
    result = checks(second=[])
    assert result["second_spoken_turn_searched_same_note"] is False
    assert checks(reply="I remember the answer.")["second_reply_recalls_quote_and_source"] is False
    first, _, second = evidence()
    second.append(first[0])
    assert checks(second=second)["recall_turn_did_not_browse"] is False


def test_write_must_have_actual_storage_receipt_and_match_observed_note():
    first, _, _ = evidence()
    first[1]["tools"][0]["result"] = json.dumps({"error": "not stored"})
    assert checks(first=first)["actual_model_memory_write"] is False
    first, _, _ = evidence()
    first[1]["tools"][0]["arguments"]["value"] = json.dumps({**RECORD, "quote": "invented"})
    assert checks(first=first)["actual_model_memory_write"] is False


def test_any_dispatched_action_outside_owned_window_fails_even_if_unverified():
    first, _, _ = evidence()
    other = outcome("other-click", window="preexisting-window")
    other["status"] = "unresolved"
    first.append(event("native_ui", {"operation": "execute"}, other))
    assert checks(first=first)["verified_actions_in_owned_firefox"] is False
    assert checks(current=state(window="preexisting-window"))["fresh_source_in_owned_window"] is False


def test_default_profile_launch_has_no_cdp_private_or_profile_override(monkeypatch):
    launches, commands = [], []
    windows = iter([{"old-window"}, {"old-window", "123"}])
    monkeypatch.setattr(runner, "firefox_windows", lambda _environment, **_kwargs: next(windows))
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/firefox")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **_kwargs: launches.append(argv))
    monkeypatch.setattr(runner, "process_start", lambda _pid: "start-42")
    monkeypatch.setattr(runner, "command", lambda argv, _environment: commands.append(argv) or ("42" if argv[1] == "getwindowpid" else ""))
    owner = runner.OwnedFirefox({})
    owner.open()
    assert launches == [["/usr/bin/firefox", "--new-window", "about:blank"]]
    assert owner.before == {"old-window"} and owner.window == "123"
    assert owner.pid == "42" and owner.start == "start-42"
    assert commands[-1] == ["xdotool", "windowactivate", "--sync", "123"]


@pytest.mark.parametrize("fault", ["preexisting", "pid_changed", "process_reused"])
def test_cleanup_never_closes_preexisting_or_reassigned_windows(monkeypatch, fault):
    commands = []
    owner = runner.OwnedFirefox({})
    owner.window, owner.pid, owner.start = "123", "42", "start-42"
    owner.before = {"123"} if fault == "preexisting" else {"old-window"}
    monkeypatch.setattr(runner, "firefox_windows", lambda _environment: {"123", "old-window"})
    monkeypatch.setattr(runner, "process_start", lambda _pid: "changed" if fault == "process_reused" else "start-42")
    monkeypatch.setattr(runner, "command", lambda argv, _environment: commands.append(argv) or ("99" if fault == "pid_changed" else "42"))
    monkeypatch.setattr(runner, "request_owned_window_close", lambda *_args: pytest.fail("Unowned close requested"))
    assert owner.close() is False
    assert not any("windowclose" in argv for argv in commands)


def test_cleanup_closes_only_exact_owned_window_and_waits_for_disappearance(monkeypatch):
    commands = []
    windows = iter([{"123", "old-window"}, {"old-window"}])
    owner = runner.OwnedFirefox({})
    owner.window, owner.pid, owner.start, owner.before = "123", "42", "start-42", {"old-window"}
    monkeypatch.setattr(runner, "firefox_windows", lambda _environment: next(windows))
    monkeypatch.setattr(runner, "process_start", lambda _pid: "start-42")
    monkeypatch.setattr(runner, "command", lambda argv, _environment: commands.append(argv) or "42")
    requests = []
    monkeypatch.setattr(runner, "request_owned_window_close", lambda *args: requests.append(args) or True)
    assert owner.close() is True
    assert commands == [["xdotool", "getwindowpid", "123"]]
    assert requests == [("123", "42", "start-42", {})]


def test_partial_journals_survive_truncated_final_event(tmp_path):
    journal = tmp_path / "events.jsonl"
    journal.write_text(json.dumps(event("native_ui", {"operation": "observe"}, {})) + '\n{"type":')
    assert len(runner.read_events(tmp_path)) == 1


@pytest.mark.parametrize("speech_failure", [False, True])
@pytest.mark.parametrize("think", [False, True])
def test_generated_speech_is_only_goal_and_partial_failures_survive(monkeypatch, tmp_path, speech_failure, think):
    """The source-only fake ensures the runner never supplies a typed task to chat."""
    roots, inputs, native = [], [], []
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "desktop_subprocess_environment", lambda: {})
    monkeypatch.setattr(runner.OwnedFirefox, "open", lambda self: (
        setattr(self, "window", "new-window"), setattr(self, "pid", "42"), setattr(self, "start", "start-42")))
    monkeypatch.setattr(runner.OwnedFirefox, "close", lambda _self: True)
    monkeypatch.setattr(runner, "PointerObserver", lambda *_args: SimpleNamespace(
        thread=SimpleNamespace(start=lambda: None), close=lambda: None, count=0,
        task_started=SimpleNamespace(set=lambda: None, clear=lambda: None)))
    # A real context manager is needed, while all network/GUI seams remain fake.
    class Client:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return None
    monkeypatch.setattr(runner.httpx, "Client", lambda **_kwargs: Client())
    monkeypatch.setattr(runner.secrets, "choice", lambda values: values[0])
    def speech(_client, _endpoint, _model, instruction, root):
        roots.append(root)
        root.mkdir()
        inputs.append(instruction)
        return b"actual-wav-seam", {"wav_sha256": "wav-hash"}
    monkeypatch.setattr(runner, "synthesize_instruction", speech)
    def audio(_client, _endpoint, _model, raw, **kwargs):
        assert raw == b"actual-wav-seam"
        assert "context" not in kwargs
        assert kwargs["system"] == runner.METHOD
        assert kwargs["think"] is think
        assert all(instruction not in kwargs["system"] for instruction in inputs)
        if speech_failure:
            raise RuntimeError("bounded simulated speech failure")
        first, memory, second = evidence()
        record = {**RECORD, "marker": "copper finch cedar north"}
        first[1]["tools"][0]["arguments"]["value"] = json.dumps(record)
        memory["results"][0]["value"] = json.dumps(record)
        second[0]["tools"][0]["result"] = json.dumps(memory)
        events = first if len(inputs) == 1 else second
        kwargs["evidence_path"].write_text("\n".join(json.dumps(item) for item in events) + "\n")
        return {"message": {"content": QUOTE + " " + URL}}, {"asr_transcript": "actual ASR seam", "elapsed_s": 1}
    monkeypatch.setattr(runner, "audio_turn", audio)
    def tool(_client, _endpoint, name, arguments):
        native.append((name, arguments))
        if name == "memory_search":
            return {"results": [{"value": json.dumps({**RECORD, "marker": "copper finch cedar north"})}]}
        if arguments.get("operation") == "verify":
            return {"state": state(), "status": "verified", "verification_receipt": {"status": "verified"}}
        return {"state": state()}
    monkeypatch.setattr(runner, "tool", tool)
    token = tmp_path / "token"
    token.write_text("test-token-not-an-actual-credential")
    args = SimpleNamespace(output=tmp_path / "runtime-data" / "speech", scenario="python-numbers",
                           model="test-model", token_file=token, endpoint="http://127.0.0.1:8920", timeout=1, think=think)
    assert runner.run(args) == (1 if speech_failure else 0)
    report = json.loads((args.output / "summary.json").read_text())
    assert report["source"] == "generated_tts_audio" and report["typed_goal_substituted"] is False
    assert report["physical_microphone"] is False and report["owned_window_closed"] is True
    assert report["think"] is think
    if speech_failure:
        assert report["error"]["detail"] == "bounded simulated speech failure"
        assert native == [("native_ui", {"operation": "observe"})]
        assert len(roots) == len(inputs) == 1
    else:
        assert all(report["criteria"].values())
        assert len(roots) == len(inputs) == len(report["turns"]) == 2
        assert inputs[0] != inputs[1]
