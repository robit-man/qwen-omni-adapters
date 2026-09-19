from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.background_agent import BackgroundAgent
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
    checkpoint = store.checkpoint(
        created["task_id"],
        "first",
        messages=[{"role": "tool", "content": "created"}],
        progress="Created the files.",
    )
    assert checkpoint is not None
    assert checkpoint["progress"][-1] == "Created the files."

    # A second instance sees the same cross-process checkpoint.
    reopened = BackgroundTaskStore(tmp_path / "tasks.json")
    assert reopened.get(created["task_id"])["round"] == 1  # type: ignore[index]


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
                        "content": "Created marker.txt and verified the write succeeded.",
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
    assert current["result"].startswith("Created marker.txt")
    assert requests == ["/api/chat", "/api/tools/shell/call", "/api/chat"]
    assert completed[0]["task_id"] == task["task_id"]


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
