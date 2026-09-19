from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.background_agent import (
    BackgroundAgent,
    _compact_task_messages,
    _seen_tool_fingerprints,
)
from portal.background_tasks import BackgroundTaskStore
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness


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
        "progress": ["Created the workspace.", "Verified the latest artifact."],
        "guidance": [{"content": "Make the final version blue."}],
        "tools_used": ["shell"],
    }

    seen = _seen_tool_fingerprints(messages)  # type: ignore[arg-type]
    compacted = _compact_task_messages(messages, task)  # type: ignore[arg-type]

    assert len(seen) == 80
    assert len(compacted) < 20
    assert compacted[:2] == [{"role": "system", "content": "rules"}, objective]
    checkpoint = compacted[2]["content"]
    assert "Verified the latest artifact" in checkpoint
    assert "Make the final version blue" in checkpoint
    assert compacted[3]["role"] == "assistant"
    assert compacted[4]["role"] == "tool"


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


def test_background_agent_yields_between_inference_and_tool_steps(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Write a marker file and verify it.")
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            assert payload["think"] is True
            tool_results = [
                item for item in payload["messages"] if item.get("role") == "tool"
            ]
            if not tool_results:
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
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I created marker.txt and verified the write succeeded.\n"
                            "TASK_COMPLETE"
                        ),
                    }
                },
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
    assert requests == ["/api/chat", "/api/tools/shell/call", "/api/chat"]
    assert completed[0]["task_id"] == task["task_id"]


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
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "TASK_COMPLETE\nCreated and verified the marker.",
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
    failures = [
        item for item in current["progress"] if "transient failure" in item
    ]
    assert len(failures) == 2


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
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "TASK_COMPLETE\nI inspected the page and verified cobalt.",
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
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "TASK_COMPLETE\nCreated and verified the marker.",
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
    assert chat_round == 3
    assert any(
        "Rejected an unsupported completion" in item
        for item in current["progress"]
    )


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
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": (
                            "TASK_PROGRESS\nI created the artifact. I’m verifying its "
                            "contents now."
                        ),
                    }
                },
            )
        elif chat_round == 3:
            payload = json.loads(request.content)
            assert "milestone update was delivered" in str(payload["messages"][-1])
            content = "test -f artifact"
        else:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": (
                            "TASK_COMPLETE\nI finished the artifact and verified that "
                            "the file exists."
                        ),
                    }
                },
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
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "I finished it.\nTASK_COMPLETE",
                    }
                },
            )
        elif chat_round == 4:
            payload = json.loads(request.content)
            assert "latest concrete action failed" in str(payload["messages"][-1])
            command = "different successful verification"
        else:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "I finished it and the new check passed.\nTASK_COMPLETE",
                    }
                },
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
    assert any(
        "latest concrete action failed" in item.lower()
        for item in current["progress"]
    )


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
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "TASK_COMPLETE\nCreated the base project.",
                    }
                },
            )
        elif chat_round == 3:
            assert any(
                "Use TypeScript" in str(item.get("content") or "")
                for item in payload["messages"]
            )
            content = "touch typescript"
        else:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "TASK_COMPLETE\nApplied the TypeScript update.",
                    }
                },
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
    assert any(
        "newer spoken update" in item.lower() for item in current["progress"]
    )


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
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": (
                            "TASK_COMPLETE\nI created and verified the redirected artifact."
                        ),
                    }
                },
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
    assert any(
        "redirected the task before its pending action" in item
        for item in current["progress"]
    )
