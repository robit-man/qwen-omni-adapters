from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.background_agent import (
    AGENT_SYSTEM_PROMPT,
    MAX_CHECKPOINT_REPORT_CHARS,
    MAX_RETAINED_TASK_MESSAGES,
    MAX_TOOL_RESULT_CHARS,
    TASK_CHECKPOINT_TOOL,
    TASK_COMPACT_TOOL,
    TASK_RECOVERY_TOOL,
    BackgroundAgent,
    _audit_json,
    _bounded_tool_result,
    _call_fingerprint,
    _checkpoint_available,
    _compact_task_messages,
    _compaction_available,
    _compaction_receipt,
    _freshest_evidence_id,
    _inference_diagnostics,
    _latest_tool_fingerprint,
    _MalformedToolCall,
    _NonRetryableBackgroundError,
    _seen_tool_fingerprints,
    _stream_error,
    _task_system_prompt,
    _tool_evidence,
)
from portal.background_tasks import BackgroundTaskStore
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness


def test_task_system_prompt_pins_objective_and_latest_directions() -> None:
    prompt = _task_system_prompt(
        {
            "objective": "Open the requested application.",
            "completion_criteria": "Its window is visible.",
            "guidance": [
                {"content": "Use the desktop session."},
                {"content": "Verify the active window."},
            ],
        }
    )

    assert prompt.startswith("<current_task>\nObjective: Open the requested application.")
    assert "Completion criteria: Its window is visible." in prompt
    assert "- Verify the active window." in prompt
    assert "Ignore unrelated topics" in prompt
    assert "every qualifier in the completion criteria as a constraint" in prompt
    assert prompt.endswith(AGENT_SYSTEM_PROMPT)


def test_background_inference_diagnostics_report_budget_without_reasoning_text() -> None:
    diagnostic = _inference_diagnostics(
        {
            "done_reason": "length",
            "prompt_eval_count": 7916,
            "eval_count": 768,
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "private reasoning that must not enter logs",
            },
        },
        768,
    )

    assert diagnostic == {
        "classification": "output_budget_exhausted_without_action",
        "done_reason": "length",
        "prompt_eval_count": 7916,
        "eval_count": 768,
        "step_token_limit": 768,
        "thinking_chars": 42,
        "content_chars": 0,
        "tool_call_count": 0,
    }
    assert "private reasoning" not in json.dumps(diagnostic)


def _checkpoint_response(
    action: str, report: str, evidence_ids: list[str]
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"checkpoint-{action}",
                        "function": {
                            "name": "task_checkpoint",
                            "arguments": {
                                "action": action,
                                "report": report,
                                "criteria_assessment": (
                                    "The cited evidence was checked against the "
                                    "completion criteria; no required work remains."
                                ),
                                "evidence_ids": evidence_ids,
                            },
                        },
                    }
                ],
            }
        },
    )


def test_checkpoint_schema_stays_below_llama_grammar_repetition_limit() -> None:
    report = TASK_CHECKPOINT_TOOL["function"]["parameters"]["properties"]["report"]
    criteria = TASK_CHECKPOINT_TOOL["function"]["parameters"]["properties"][
        "criteria_assessment"
    ]

    assert report["maxLength"] == MAX_CHECKPOINT_REPORT_CHARS
    assert criteria["maxLength"] == MAX_CHECKPOINT_REPORT_CHARS
    assert MAX_CHECKPOINT_REPORT_CHARS < 2_000
    assert TASK_RECOVERY_TOOL["function"]["name"] == "task_recovery"
    assert TASK_COMPACT_TOOL["function"]["name"] == "task_compact"


def test_deterministic_client_error_is_not_retryable() -> None:
    error = _stream_error("language returned HTTP 400: invalid grammar")

    assert isinstance(error, _NonRetryableBackgroundError)
    assert isinstance(
        _stream_error(
            "language returned HTTP 500: Failed to parse tool call arguments as JSON"
        ),
        _MalformedToolCall,
    )
    assert type(_stream_error("language returned HTTP 429: busy")) is RuntimeError
    assert type(_stream_error("language returned HTTP 503: unavailable")) is RuntimeError


def test_nonretryable_worker_request_quiesces_until_restart(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Complete the accepted task.")
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            content=(
                json.dumps(
                    {
                        "type": "error",
                        "error": "language returned HTTP 400: invalid request",
                    }
                )
                + "\n"
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        retry_initial_s=0.01,
        retry_max_s=0.01,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("current_stage") == (
            "Waiting for corrected worker code and restart"
        ):
            break
        time.sleep(0.01)
    time.sleep(0.1)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "pending"
    assert current["current_stage"] == "Waiting for corrected worker code and restart"
    assert requests == 1


def test_background_task_store_checkpoints_and_recovers_expired_work(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Create and verify a small application.", "Tests pass.")
    claimed = store.claim_next("first", lease_s=5)

    assert claimed is not None
    assert claimed["task_id"] == created["task_id"]
    assert claimed["status"] == "running"
    assert store.update_stage(
        created["task_id"], "first", "Running shell"
    )["current_stage"] == "Running shell"  # type: ignore[index]
    checkpoint = store.checkpoint(
        created["task_id"],
        "first",
        messages=[{"role": "tool", "content": "created"}],
        tools_used=["shell"],
        progress="Created the files.",
        current_stage="Assessing the result",
    )
    assert checkpoint is not None
    assert checkpoint["progress"][-1] == "Created the files."
    assert checkpoint["current_stage"] == "Assessing the result"
    assert checkpoint["tools_used"] == ["shell"]

    # A second instance sees the same cross-process checkpoint.
    reopened = BackgroundTaskStore(tmp_path / "tasks.json")
    assert reopened.get(created["task_id"])["round"] == 1  # type: ignore[index]


def test_background_task_store_compaction_is_control_not_progress(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Keep the objective exact.", "Verify the artifact.")
    claimed = store.claim_next("worker")
    assert claimed is not None

    compacted = store.compact_context(
        created["task_id"],
        "worker",
        messages=[
            {"role": "system", "content": "current policy"},
            {"role": "user", "content": "exact objective"},
        ],
        receipt={
            "schema": "robit.omni.task-compaction.v1",
            "before": {"messages": 40, "bytes": 90000},
            "after": {"messages": 2, "bytes": 1000},
        },
    )

    assert compacted is not None
    assert compacted["round"] == 0
    assert compacted["progress"] == ["Accepted from the live conversation."]
    assert compacted["compaction"]["before"]["messages"] == 40
    raw = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    assert raw["tasks"][0]["messages"][1]["content"] == "exact objective"


def test_background_task_blocks_after_three_expired_worker_leases(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "tasks.json"
    store = BackgroundTaskStore(task_path)
    created = store.create("Finish one durable task.")
    assert store.claim_next("initial") is not None

    for attempt in range(1, 4):
        state = json.loads(task_path.read_text(encoding="utf-8"))
        state["tasks"][0]["lease_until"] = 0
        task_path.write_text(json.dumps(state), encoding="utf-8")
        claimed = store.claim_next(f"worker-{attempt}")
        if attempt < 3:
            assert claimed is not None
            assert claimed["resume_count"] == attempt
        else:
            assert claimed is None

    blocked = store.get(created["task_id"])
    assert blocked is not None
    assert blocked["status"] == "blocked"
    assert blocked["resume_count"] == 3
    assert "three expired worker leases" in blocked["progress"][-1]


def test_background_task_claims_rotate_between_pending_work(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    first = store.create("First task")
    second = store.create("Second task")

    claimed_first = store.claim_next("worker")
    assert claimed_first is not None
    assert claimed_first["task_id"] == first["task_id"]
    store.checkpoint(first["task_id"], "worker", status="pending")

    claimed_second = store.claim_next("worker")
    assert claimed_second is not None
    assert claimed_second["task_id"] == second["task_id"]


def test_background_task_store_records_bounded_action_audit(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Inspect a file.")
    claimed = store.claim_next("worker")
    assert claimed is not None

    recorded = store.record_action(
        created["task_id"],
        "worker",
        call_id="call-1",
        tool="shell",
        arguments='{"command": "ls -l Desktop"}',
        outcome='{"exit_code": 0}',
        ok=True,
    )

    assert recorded is not None
    assert recorded["actions"] == [
        {
            "call_id": "call-1",
            "at": recorded["actions"][0]["at"],
            "tool": "shell",
            "arguments": '{"command": "ls -l Desktop"}',
            "outcome": '{"exit_code": 0}',
            "ok": True,
        }
    ]


def test_action_audit_redacts_credentials_and_bulk_payloads() -> None:
    rendered = _audit_json(
        {
            "command": "play Desktop/song.mp3",
            "access_token": "do-not-retain",
            "stdin": "x" * 4000,
            "screenshot": {"data": "also-do-not-retain"},
        },
        2000,
    )

    assert "play Desktop/song.mp3" in rendered
    assert "do-not-retain" not in rendered
    assert "also-do-not-retain" not in rendered
    assert "[redacted]" in rendered
    assert "[omitted 4000 characters]" in rendered


def test_checkpoint_requires_new_concrete_action_after_every_attempt() -> None:
    messages = [
        {
            "role": "tool",
            "tool_name": "tool_search",
            "tool_call_id": "search-1",
            "content": '{"available_tools": ["shell"]}',
        }
    ]
    assert _checkpoint_available(messages) is False

    messages.append(
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "shell-1",
            "content": '{"exit_code": 0}',
        }
    )
    assert _checkpoint_available(messages) is True
    assert _freshest_evidence_id(messages) == "shell-1"

    messages.append(
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "shell-duplicate",
            "content": '{"error": "duplicate_tool_call"}',
        }
    )
    assert _checkpoint_available(messages) is False
    assert "shell-duplicate" not in _tool_evidence(messages)
    assert _freshest_evidence_id(messages) == "shell-1"

    messages.append(
        {
            "role": "tool",
            "tool_name": "task_checkpoint",
            "tool_call_id": "checkpoint-1",
            "content": '{"error": "unsupported_checkpoint"}',
        }
    )
    assert _checkpoint_available(messages) is False

    messages.append(
        {
            "role": "tool",
            "tool_name": "tool_search",
            "tool_call_id": "search-2",
            "content": '{"error": "duplicate_tool_call"}',
        }
    )
    assert _checkpoint_available(messages) is False

    messages.append(
        {
            "role": "user",
            "content": '<task_update id="new">Use the corrected target.</task_update>',
        }
    )
    assert _checkpoint_available(messages) is False
    messages.append(
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "shell-2",
            "content": '{"exit_code": 0}',
        }
    )
    assert _checkpoint_available(messages) is True


def test_long_task_context_compacts_to_a_fresh_complete_checkpoint_chain() -> None:
    objective = {"role": "user", "content": "<objective>Build it.</objective>"}
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "rules"},
        objective,
    ]
    for index in range(80):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": f"step-{index}"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "shell",
                    "tool_call_id": f"call-{index}",
                    "content": '{"exit_code": 0}',
                },
            ]
        )
    task = {
        "objective": "Build the requested artifact.",
        "completion_criteria": "The final probe passes.",
        "progress": ["Created the workspace.", "Verified the latest artifact."],
        "guidance": [{"content": "Make the final version blue."}],
        "tools_used": ["shell"],
        "actions": [
            {
                "call_id": "failed-write",
                "tool": "shell",
                "ok": False,
                "outcome": "directory missing",
            },
            {
                "call_id": "fixed-write",
                "tool": "shell",
                "ok": True,
                "outcome": "file written and read back",
            },
        ],
    }

    seen = _seen_tool_fingerprints(messages)  # type: ignore[arg-type]
    compacted = _compact_task_messages(messages, task)  # type: ignore[arg-type]

    assert len(seen) == 80
    assert len(compacted) < 20
    assert compacted[:2] == [{"role": "system", "content": "rules"}, objective]
    checkpoint = compacted[2]["content"]
    assert "Verified the latest artifact" in checkpoint
    assert "Make the final version blue" in checkpoint
    assert "failed-write | shell | failed" in checkpoint
    assert "fixed-write | shell | succeeded" in checkpoint
    assert compacted[3]["role"] == "assistant"
    assert compacted[4]["role"] == "tool"

    receipt = _compaction_receipt(  # type: ignore[arg-type]
        messages,
        compacted,
        task,
        reason="memory_pressure_pre_admission",
    )
    assert receipt["before"]["messages"] == len(messages)
    assert receipt["after"]["messages"] == len(compacted)
    assert receipt["retained"]["objective"] is True
    assert receipt["retained"]["completion_criteria"] is True
    assert receipt["retained"]["latest_guidance_count"] == 1


def test_duplicate_guard_is_scoped_to_the_immediately_preceding_external_call() -> None:
    build = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "build-1",
                "function": {
                    "name": "shell",
                    "arguments": {"command": "npm run build"},
                },
            }
        ],
    }
    repair = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "repair-1",
                "function": {
                    "name": "shell",
                    "arguments": {"command": "sed -i s/bad/good/ app.js"},
                },
            }
        ],
    }

    build_fingerprint = _call_fingerprint("shell", {"command": "npm run build"})
    assert _latest_tool_fingerprint([build]) == build_fingerprint
    assert _latest_tool_fingerprint([build, repair]) != build_fingerprint
    assert _latest_tool_fingerprint(
        [
            build,
            repair,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "checkpoint-1",
                        "function": {
                            "name": "task_checkpoint",
                            "arguments": {"action": "progress"},
                        },
                    }
                ],
            },
        ]
    ) == _latest_tool_fingerprint([build, repair])


def test_a_single_tool_result_cannot_balloon_the_durable_task_context() -> None:
    result = {
        "exit_code": 0,
        "stdout": "x" * (2 * MAX_TOOL_RESULT_CHARS),
        "stderr": "",
    }

    bounded = _bounded_tool_result(result)
    rendered = json.dumps(bounded)

    assert len(rendered) <= MAX_TOOL_RESULT_CHARS + 100
    assert bounded["exit_code"] == 0
    assert bounded["truncated"] is True
    assert len(bounded["original_sha256"]) == 64


def test_compaction_control_waits_for_new_external_evidence_after_receipt() -> None:
    messages = [
        {"role": "user", "content": f"retained-{index}"}
        for index in range(MAX_RETAINED_TASK_MESSAGES + 4)
    ]
    assert _compaction_available(messages) is True

    messages.append(
        {
            "role": "tool",
            "tool_name": "task_compact",
            "tool_call_id": "compact-1",
            "content": '{"compacted": true}',
        }
    )
    assert _compaction_available(messages) is False

    messages.append(
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "verify-1",
            "content": '{"exit_code": 0}',
        }
    )
    assert _compaction_available(messages) is True


def test_background_agent_can_invoke_deterministic_compaction(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Continue the exact task.", "Verify the final state.")
    seeded = store.claim_next("seed")
    assert seeded is not None
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "exact task"},
    ]
    for index in range(8):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"evidence-{index}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": f"step-{index}"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "shell",
                    "tool_call_id": f"evidence-{index}",
                    "content": '{"exit_code": 0}',
                },
            ]
        )
    store.checkpoint(
        task["task_id"],
        "seed",
        messages=messages,  # type: ignore[arg-type]
        progress="Eight concrete steps ran.",
        status="pending",
    )

    stop = threading.Event()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("the compaction control is local")
            )
        )
    )
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=stop,
        client=client,
    )
    claimed = store.claim_next(agent.owner)
    assert claimed is not None
    seen_tools: list[str] = []
    persist_compaction = store.compact_context

    def persist_and_stop(*args, **kwargs):
        result = persist_compaction(*args, **kwargs)
        stop.set()
        return result

    store.compact_context = persist_and_stop  # type: ignore[method-assign]

    def compact(payload: dict[str, object]) -> dict[str, object]:
        seen_tools.extend(
            item["function"]["name"]  # type: ignore[index]
            for item in payload["tools"]  # type: ignore[union-attr]
        )
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "compact-1",
                        "function": {
                            "name": "task_compact",
                            "arguments": {"reason": "context_noise"},
                        },
                    }
                ],
            }
        }

    agent._chat = compact  # type: ignore[method-assign]
    agent._execute(claimed)
    client.close()

    assert "task_compact" in seen_tools
    current = store.get(task["task_id"])
    assert current is not None
    assert current["compaction"]["reason"] == "agent_requested"
    assert current["compaction"]["after"]["messages"] < len(messages)
    assert current["actions"][-1]["tool"] == "task_compact"


def test_terminal_announcement_survives_restart_until_marked_spoken(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Finish something and report it.")
    claimed = store.claim_next("worker")
    assert claimed is not None

    store.checkpoint(
        created["task_id"],
        "worker",
        result="I finished the work and verified it.",
        status="completed",
    )

    reopened = BackgroundTaskStore(tmp_path / "tasks.json")
    pending = reopened.pending_announcements()
    assert [item["task_id"] for item in pending] == [created["task_id"]]
    assert reopened.mark_announced(created["task_id"]) is True
    assert reopened.pending_announcements() == []
    assert reopened.mark_announced(created["task_id"]) is False


def test_finished_tasks_move_to_a_human_readable_archive(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    finished = store.create("Create a verified artifact.")
    live = store.create("Keep working on this.")
    claimed = store.claim_next("worker")
    assert claimed is not None and claimed["task_id"] == finished["task_id"]
    store.checkpoint(
        finished["task_id"],
        "worker",
        tools_used=["shell"],
        progress="Verified the artifact.",
        result="The artifact is ready.",
        status="completed",
    )
    archive = tmp_path / "task-archive.log"

    count = store.archive_terminal(archive)

    assert count == 1
    assert store.get(finished["task_id"]) is None
    assert store.get(live["task_id"])["status"] == "pending"  # type: ignore[index]
    content = archive.read_text(encoding="utf-8")
    assert f"Task {finished['task_id']}" in content
    assert "Tools used: shell" in content
    assert "Verified the artifact." in content
    assert "The artifact is ready." in content


def test_task_store_can_remove_a_live_task_record(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("A task the user wants to clear immediately.")
    claimed = store.claim_next("worker")

    assert claimed is not None
    assert store.remove(task["task_id"]) is True
    assert store.get(task["task_id"]) is None
    assert store.remove(task["task_id"]) is False


def test_portal_background_tool_starts_and_controls_persistent_work(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), background_tasks=store
    )

    started = harness.execute(
        "voice",
        "background_task",
        {
            "action": "start",
            "objective": "Build the requested application.",
            "completion_criteria": "The production build succeeds.",
        },
    )

    assert started["accepted"] is True
    task_id = started["task_id"]
    assert harness.execute(
        "voice", "background_task", {"action": "status", "task_id": task_id}
    )["task"]["status"] == "pending"
    assert harness.execute(
        "voice", "background_task", {"action": "list"}
    )["tasks"][0]["task_id"] == task_id
    updated = harness.execute(
        "voice",
        "background_task",
        {
            "action": "update",
            "task_id": task_id,
            "guidance": "Use TypeScript and include a production build check.",
        },
    )
    assert updated["accepted"] is True
    assert updated["task"]["guidance"][0]["content"].startswith("Use TypeScript")
    assert harness.execute(
        "voice", "background_task", {"action": "cancel", "task_id": task_id}
    )["task"]["status"] == "cancelled"


def test_new_background_start_must_reconcile_unfinished_work(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    existing = store.create("Finish the current request.")
    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), background_tasks=store
    )

    rejected = harness.execute(
        "voice",
        "background_task",
        {"action": "start", "objective": "Do another thing."},
    )

    assert rejected["accepted"] is False
    assert rejected["error"] == "unfinished_task_requires_relationship"
    assert rejected["active_tasks"][0]["task_id"] == existing["task_id"]
    assert len(store.list()) == 1

    independent = harness.execute(
        "voice",
        "background_task",
        {
            "action": "start",
            "objective": "Do a separate concurrent thing.",
            "independent": True,
        },
    )

    assert independent["accepted"] is True
    assert len(store.list()) == 2


def test_background_agent_yields_between_inference_and_tool_steps(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Write a marker file and verify it.")
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/chat/stream":
            payload = json.loads(request.content)
            assert payload["tool_choice"] == "required"
            tool_results = [
                item for item in payload["messages"] if item.get("role") == "tool"
            ]
            if not tool_results:
                assert payload["think"] is True
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "write-1",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": {
                                            "command": "printf ready > marker.txt"
                                        },
                                    },
                                }
                            ],
                        }
                    },
                )
            assert payload["think"] is False
            assert any(
                "<task_self_check" in str(item.get("content") or "")
                and "write-1" in str(item.get("content") or "")
                for item in payload["messages"]
            )
            return _checkpoint_response(
                "complete",
                "I created marker.txt and verified the write succeeded.",
                ["write-1"],
            )
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(
                200,
                json={"result": {"stdout": "", "stderr": "", "exit_code": 0}},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    stop = threading.Event()
    completed: list[dict[str, object]] = []
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=stop,
        on_complete=completed.append,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["result"].startswith("I created marker.txt")
    assert "TASK_COMPLETE" not in current["result"]
    assert current["actions"][0]["tool"] == "shell"
    assert "printf ready" in current["actions"][0]["arguments"]
    assert '"exit_code": 0' in current["actions"][0]["outcome"]
    assert current["actions"][0]["ok"] is True
    assert requests == [
        "/api/chat/stream",
        "/api/tools/shell/call",
        "/api/chat/stream",
    ]
    assert completed[0]["task_id"] == task["task_id"]


def test_background_agent_repeats_verification_after_an_intervening_repair(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Repair the application and verify its production build.")
    seeded = store.claim_next("seed")
    assert seeded is not None
    store.checkpoint(
        task["task_id"],
        "seed",
        active_tools=["shell"],
        status="pending",
    )
    chat_round = 0
    commands: list[str] = []

    def call(call_id: str, command: str) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "shell",
                                "arguments": {"command": command},
                            },
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        body = json.loads(request.content)
        if request.url.path == "/api/tools/shell/call":
            command = body["arguments"]["command"]
            commands.append(command)
            if command == "npm run build" and commands.count(command) == 1:
                return httpx.Response(
                    200,
                    json={"result": {"exit_code": 1, "stderr": "bad import"}},
                )
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round == 1:
            return call("build-before", "npm run build")
        if chat_round == 2:
            return call("repair", "sed -i s/bad/good/ app.js")
        if chat_round == 3:
            return call("build-after", "npm run build")
        return _checkpoint_response(
            "complete",
            "I repaired the application and verified its production build.",
            ["build-after"],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert commands == [
        "npm run build",
        "sed -i s/bad/good/ app.js",
        "npm run build",
    ]
    assert not any(
        "duplicate_tool_call" in str(item.get("outcome"))
        for item in current["actions"]
    )


def test_capability_failure_triggers_generic_recovery_and_headed_browser(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Research a current topic from public sources.")
    chat_round = 0

    def tool_call(call_id: str, name: str, arguments: dict[str, object]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/tool_search/call":
            query = payload["arguments"]["query"]
            selected = (
                "browser_interact" if "interactive rendered" in query else "web_search"
            )
            return httpx.Response(
                200,
                json={
                    "result": {
                        "available_tools": [selected],
                        "results": [{"name": selected}],
                    }
                },
            )
        if request.url.path == "/api/tools/web_search/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "error": "rendered_page_required",
                        "retryable": False,
                        "failure_scope": "capability",
                        "task_blocked": False,
                        "disposition": "change_capability",
                    }
                },
            )
        if request.url.path == "/api/tools/browser_interact/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "rendered": True,
                        "title": "Verified source",
                        "url": "https://example.com/source",
                    }
                },
            )
        chat_round += 1
        exposed = [item["function"]["name"] for item in payload["tools"]]
        if chat_round == 1:
            assert exposed == ["tool_search"]
            return tool_call(
                "discover-web",
                "tool_search",
                {"query": "public web research"},
            )
        if chat_round == 2:
            assert "web_search" in exposed
            return tool_call(
                "challenged-search",
                "web_search",
                {"query": "current topic", "mode": "discover"},
            )
        if chat_round == 3:
            assert exposed == ["task_recovery"]
            assert any(
                "Call task_recovery now" in str(message.get("content") or "")
                for message in payload["messages"]
            )
            return tool_call(
                "assess-recovery",
                "task_recovery",
                {
                    "evidence_id": "challenged-search",
                    "failure_scope": "capability",
                    "unmet_requirement": "retrieve and verify current public information",
                    "capability_query": "interactive rendered public web navigation and inspection",
                },
            )
        if chat_round == 4:
            assert exposed == ["tool_search"]
            assert any(
                "interactive rendered public web navigation and inspection"
                in str(message.get("content") or "")
                for message in payload["messages"]
            )
            return tool_call(
                "discover-alternative",
                "tool_search",
                {"query": "interactive rendered public web navigation and inspection"},
            )
        if chat_round == 5:
            assert "browser_interact" in exposed
            return tool_call(
                "headed-browser",
                "browser_interact",
                {"action": "navigate", "url": "https://example.com/source"},
            )
        return _checkpoint_response(
            "complete",
            "I researched the topic and verified the rendered source.",
            ["headed-browser"],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert [item["tool"] for item in current["actions"]] == [
        "tool_search",
        "web_search",
        "task_recovery",
        "tool_search",
        "browser_interact",
        "task_checkpoint",
    ]


def test_background_agent_discovers_before_exposing_tools_and_acts_without_runaway_thinking(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Open the rendered browser and inspect the page.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/tool_search/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "available_tools": ["browser_interact"],
                        "results": [{"name": "browser_interact"}],
                    }
                },
            )
        if request.url.path == "/api/tools/browser_interact/call":
            return httpx.Response(
                200,
                json={"result": {"rendered": True, "url": "http://example.test/"}},
            )
        chat_round += 1
        tool_names = [item["function"]["name"] for item in payload["tools"]]
        assert payload["options"]["num_predict"] == 256
        if chat_round == 1:
            assert payload["think"] is True
            assert payload["tool_choice"] == "required"
            assert tool_names == ["tool_search"]
            call_name = "tool_search"
            arguments = {"query": "visual rendered browser navigation"}
        elif chat_round == 2:
            assert payload["think"] is False
            assert tool_names == [
                "tool_search",
                "browser_interact",
            ]
            call_name = "browser_interact"
            arguments = {"action": "navigate", "url": "http://example.test/"}
        else:
            assert payload["think"] is False
            assert tool_names == [
                "task_checkpoint",
                "tool_search",
                "browser_interact",
            ]
            return _checkpoint_response(
                "complete",
                "I opened and verified the rendered page.",
                ["call-2"],
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{chat_round}",
                            "function": {"name": call_name, "arguments": arguments},
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        step_token_limit=256,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["tools_used"] == ["tool_search", "browser_interact"]


def test_background_agent_recovers_from_backend_outage_without_a_retry_storm(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create a marker after the backend recovers.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round <= 3:
            return httpx.Response(502, json={"error": "temporarily unavailable"})
        if chat_round == 4:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "recovered-action",
                                "function": {
                                    "name": "shell",
                                    "arguments": {"command": "touch marker"},
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Created and verified the marker.", ["recovered-action"]
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        retry_initial_s=0.01,
        retry_max_s=0.04,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.01)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["tools_used"] == ["shell"]
    assert not any("transient failure" in item for item in current["progress"])


def test_malformed_tool_json_replans_with_a_smaller_call_instead_of_replaying(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create and verify a small application.", "The marker exists.")
    chat_round = 0
    repair_seen = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round, repair_seen
        if request.url.path == "/api/tools/shell/call":
            arguments = json.loads(request.content).get("arguments", {})
            if not arguments.get("command"):
                return httpx.Response(
                    200,
                    json={
                        "result": {
                            "error": "ToolInputError",
                            "message": "command is required",
                        }
                    },
                )
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        body = json.loads(request.content)
        if chat_round == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "inspect",
                                "function": {
                                    "name": "shell",
                                    "arguments": (
                                        '{"command":"cat > package.json <<\'EOF\'\\n'
                                        '{\\n  "name": "unfinished'
                                    ),
                                },
                            }
                        ],
                    }
                },
            )
        if chat_round == 2:
            repair_messages = [
                message
                for message in body["messages"]
                if "structured_call_recovery"
                in str(message.get("content") or "")
            ]
            repair_seen = len(repair_messages) == 1
            malformed = next(
                message
                for message in body["messages"]
                if message.get("role") == "assistant" and message.get("tool_calls")
            )
            assert malformed["tool_calls"][0]["function"]["arguments"] == {}
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "recovered-action",
                                "function": {
                                    "name": "shell",
                                    "arguments": {"command": "touch marker"},
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Created and verified the marker.", ["recovered-action"]
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        retry_initial_s=0.01,
        retry_max_s=0.02,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.01)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert repair_seen is True
    assert chat_round == 3
    assert [item["tool"] for item in current["actions"]] == [
        "shell",
        "shell",
        "task_checkpoint",
    ]
    assert [item["ok"] for item in current["actions"]] == [False, True, True]


def test_browser_screenshot_is_seen_once_but_not_persisted_as_base64(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Inspect the rendered page and report the visible fact.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/browser_interact/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "url": "http://example.test/",
                        "visible_text": "The answer is cobalt.",
                        "elements": [],
                        "rendered": True,
                        "screenshot": {
                            "mime_type": "image/png",
                            "encoding": "base64",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
                        },
                    }
                },
            )
        chat_round += 1
        payload = json.loads(request.content)
        if chat_round == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "browser-1",
                                "function": {
                                    "name": "browser_interact",
                                    "arguments": {
                                        "action": "navigate",
                                        "url": "http://example.test/",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        assert any(message.get("images") for message in payload["messages"])
        return _checkpoint_response(
            "complete",
            "I inspected the page and verified cobalt.",
            ["browser-1"],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["tools_used"] == ["browser_interact"]
    persisted = (tmp_path / "tasks.json").read_text(encoding="utf-8")
    assert '"images"' not in persisted
    assert "iVBORw0KGgo" not in persisted
    assert "rendered screenshot was inspected" in persisted


def test_background_agent_rejects_a_completion_with_no_action_evidence(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create a marker using the shell.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round == 1:
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "Done."}},
            )
        if chat_round == 2:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "real-action",
                                "function": {
                                    "name": "shell",
                                    "arguments": {"command": "touch marker"},
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Created and verified the marker.", ["real-action"]
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert chat_round == 3
    assert "Done." in (tmp_path / "tasks.json").read_text(encoding="utf-8")


def test_background_agent_speaks_a_sparse_checkpoint_then_resumes(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Build and verify the requested artifact.")
    chat_round = 0
    progress_updates: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round == 1:
            content = "touch artifact"
        elif chat_round == 2:
            return _checkpoint_response(
                "progress",
                "I created the artifact. I’m verifying its contents now.",
                ["step-1"],
            )
        elif chat_round == 3:
            payload = json.loads(request.content)
            assert '"accepted": true' in str(payload["messages"][-1]).lower()
            content = "test -f artifact"
        else:
            return _checkpoint_response(
                "complete",
                "I finished the artifact and verified that the file exists.",
                ["step-3"],
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"step-{chat_round}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": content},
                            },
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        on_progress=progress_updates.append,
        progress_after_s=0,
        progress_min_interval_s=0,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["result"].startswith("I finished the artifact")
    assert len(progress_updates) == 1
    assert progress_updates[0]["result"] == (
        "I created the artifact. I’m verifying its contents now."
    )


def test_background_agent_rejects_completion_after_latest_action_failed(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create and verify an artifact.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            command = json.loads(request.content)["arguments"]["command"]
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 1 if command == "bad verification" else 0,
                        "stderr": "failed" if command == "bad verification" else "",
                    }
                },
            )
        chat_round += 1
        if chat_round == 1:
            command = "create artifact"
        elif chat_round == 2:
            command = "bad verification"
        elif chat_round == 3:
            return _checkpoint_response(
                "complete", "I finished it.", ["action-1"]
            )
        elif chat_round == 4:
            payload = json.loads(request.content)
            assert "unsupported_checkpoint" in str(payload["messages"][-1])
            command = "different successful verification"
        else:
            return _checkpoint_response(
                "complete", "I finished it and the new check passed.", ["action-4"]
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"action-{chat_round}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": command},
                            },
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert current["result"] == "I finished it and the new check passed."
    assert chat_round == 5
    assert "unsupported_checkpoint" in (tmp_path / "tasks.json").read_text()


def test_new_guidance_wins_over_an_inflight_completion(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create the requested project.")
    stale_final_started = threading.Event()
    release_stale_final = threading.Event()
    chat_round = 0
    commands: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/shell/call":
            commands.append(payload["arguments"]["command"])
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round == 1:
            content = "touch base"
        elif chat_round == 2:
            stale_final_started.set()
            release_stale_final.wait(2)
            return _checkpoint_response(
                "complete", "Created the base project.", ["action-1"]
            )
        elif chat_round == 3:
            assert any(
                "Use TypeScript" in str(item.get("content") or "")
                for item in payload["messages"]
            )
            content = "touch typescript"
        else:
            return _checkpoint_response(
                "complete", "Applied the TypeScript update.", ["action-3"]
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"action-{chat_round}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": content},
                            },
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    assert stale_final_started.wait(2)
    store.add_guidance(task["task_id"], "Use TypeScript instead.")
    release_stale_final.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["result"] == "Applied the TypeScript update."
    assert commands == ["touch base", "touch typescript"]
    assert any("redirected" in item.lower() for item in current["progress"])


def test_human_interjection_redirects_before_a_stale_action_and_then_resumes(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Create the requested artifact.")
    inference_started = threading.Event()
    release_inference = threading.Event()
    foreground = threading.Event()
    chat_round = 0
    commands: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/shell/call":
            commands.append(payload["arguments"]["command"])
            return httpx.Response(200, json={"result": {"exit_code": 0}})
        chat_round += 1
        if chat_round == 1:
            inference_started.set()
            release_inference.wait(2)
            command = "touch stale-artifact"
        elif chat_round == 2:
            assert any(
                "make the redirected artifact" in str(item.get("content") or "")
                for item in payload["messages"]
            )
            assert not any(
                item.get("role") == "assistant" and item.get("tool_calls")
                for item in payload["messages"]
            )
            command = "touch redirected-artifact"
        else:
            return _checkpoint_response(
                "complete",
                "I created and verified the redirected artifact.",
                ["action-2"],
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"action-{chat_round}",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": command},
                            },
                        }
                    ],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=foreground,
        stop=threading.Event(),
        client=client,
    )
    agent.start()
    assert inference_started.wait(2)
    foreground.set()
    store.add_guidance(task["task_id"], "Instead, make the redirected artifact.")
    release_inference.set()
    time.sleep(0.1)
    assert commands == []
    foreground.clear()
    agent.wake()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = store.get(task["task_id"])
        if current and current.get("status") == "completed":
            break
        time.sleep(0.02)
    agent.close()
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["status"] == "completed"
    assert commands == ["touch redirected-artifact"]
    assert not any("stale-artifact" in item for item in current["progress"])
