from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from harness.background_agent import BackgroundAgent
from portal.background_tasks import BackgroundTaskStore
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness


def _tool_response(call_id: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": dict(arguments)},
                }
            ],
        }
    }


def _run_scripted_task(
    tmp_path: Path,
    *,
    objective: str,
    completion_criteria: str,
    steps: list[tuple[str, str, dict[str, Any]]],
    harness: PortalToolHarness,
    evidence_ids: list[str],
) -> dict[str, Any]:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create(objective, completion_criteria)
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/chat/stream":
            payload = json.loads(request.content)
            system = str(payload["messages"][0]["content"])
            assert objective in system
            assert completion_criteria in system
            if chat_round < len(steps):
                call_id, name, arguments = steps[chat_round]
                exposed = {
                    item["function"]["name"] for item in payload.get("tools", [])
                }
                assert name in exposed
                chat_round += 1
                return httpx.Response(200, json=_tool_response(call_id, name, arguments))
            chat_round += 1
            return httpx.Response(
                200,
                json=_tool_response(
                    "checkpoint-complete",
                    "task_checkpoint",
                    {
                        "action": "complete",
                        "report": "I finished the requested work and verified the result.",
                        "criteria_assessment": (
                            "The cited results satisfy every stated completion "
                            "criterion; no required work remains."
                        ),
                        "evidence_ids": evidence_ids,
                    },
                ),
            )
        if request.url.path.startswith("/api/tools/"):
            name = request.url.path.split("/")[-2]
            arguments = json.loads(request.content).get("arguments", {})
            result = harness.execute("asr-workflow", name, arguments)
            return httpx.Response(200, json={"result": result})
        return httpx.Response(404)

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
    deadline = time.monotonic() + 5
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
    assert chat_round == len(steps) + 1
    return current


def test_asr_style_file_creation_editing_and_coding_workflow(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text('print("broken")\n', encoding="utf-8")
    harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    steps = [
        (
            "discover-files",
            "tool_search",
            {"family": "filesystem"},
        ),
        (
            "write-markdown",
            "workspace_file",
            {
                "action": "write",
                "path": str(tmp_path / "plan.md"),
                "content": "# Plan\n\nCreated from a spoken request.\n",
            },
        ),
        (
            "write-json",
            "workspace_file",
            {
                "action": "write",
                "path": str(tmp_path / "data.json"),
                "content": '{"name":"voice","enabled":true}\n',
            },
        ),
        (
            "write-csv",
            "workspace_file",
            {
                "action": "write",
                "path": str(tmp_path / "table.csv"),
                "content": "label,value\ntwo,2\n",
            },
        ),
        (
            "edit-python",
            "workspace_file",
            {
                "action": "write",
                "path": str(tmp_path / "app.py"),
                "content": (
                    "def add(left: int, right: int) -> int:\n"
                    "    return left + right\n\n"
                    "print(f\"sum={add(2, 3)}\")\n"
                ),
            },
        ),
        (
            "discover-verification",
            "tool_search",
            {"family": "shell"},
        ),
            (
                "verify-files",
                "shell",
            {
                "command": (
                    "python3 -m py_compile app.py "
                    "&& test \"$(python3 app.py)\" = \"sum=5\" "
                    "&& test \"$(jq -r .name data.json)\" = \"voice\" "
                    "&& test \"$(awk -F, 'NR==2 {print $2}' table.csv)\" = \"2\" "
                    "&& grep -q '# Plan' plan.md"
                ),
                "cwd": str(tmp_path),
            },
        ),
    ]

    task = _run_scripted_task(
        tmp_path,
        objective=(
            "Hey, um, in this workspace make me a Markdown plan, a JSON settings file, "
            "and a CSV, then fix that little Python program—actually verify all of it too."
        ),
        completion_criteria=(
            "The Markdown, JSON, CSV, and Python files exist; the edited Python compiles "
            "and prints sum=5; structured file contents are checked."
        ),
        steps=steps,
        harness=harness,
        evidence_ids=[
            "write-markdown",
            "write-json",
            "write-csv",
            "edit-python",
            "verify-files",
        ],
    )

    assert (tmp_path / "plan.md").read_text(encoding="utf-8").startswith("# Plan")
    assert json.loads((tmp_path / "data.json").read_text())["enabled"] is True
    assert (tmp_path / "table.csv").read_text().splitlines()[-1] == "two,2"
    assert "def add" in (tmp_path / "app.py").read_text()
    assert [item["tool"] for item in task["actions"]] == [
        "tool_search",
        "workspace_file",
        "workspace_file",
        "workspace_file",
        "workspace_file",
        "tool_search",
        "shell",
        "task_checkpoint",
    ]


def test_asr_style_browser_navigation_and_form_filling_workflow(tmp_path: Path) -> None:
    browser_calls: list[dict[str, Any]] = []

    class Browser:
        def act(self, _session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            browser_calls.append(dict(arguments))
            submitted = arguments.get("action") == "click"
            return {
                "rendered": True,
                "url": "http://example.test/form" + ("/thanks" if submitted else ""),
                "visible_text": "Thanks, your form was submitted." if submitted else "Contact form",
                "elements": [
                    {"id": "e1", "tag": "input", "text": "Name"},
                    {"id": "e2", "tag": "input", "text": "Email"},
                    {"id": "e3", "tag": "button", "text": "Submit"},
                ],
            }

        def clear(self, _session_id: str) -> None:
            pass

    harness = PortalToolHarness(
        SessionDocumentStore(ttl_s=300), browser_automation=Browser()
    )
    steps = [
        (
            "discover-browser",
            "tool_search",
            {"family": "browser"},
        ),
        (
            "navigate-form",
            "browser_interact",
            {"action": "navigate", "url": "http://example.test/form"},
        ),
        (
            "fill-name",
            "browser_interact",
            {"action": "type", "element_id": "e1", "text": "Ada Lovelace", "clear": True},
        ),
        (
            "fill-email",
            "browser_interact",
            {
                "action": "type",
                "element_id": "e2",
                "text": "ada@example.test",
                "clear": True,
            },
        ),
        (
            "submit-form",
            "browser_interact",
            {"action": "click", "element_id": "e3"},
        ),
    ]

    task = _run_scripted_task(
        tmp_path,
        objective=(
            "Could you, uh, open the contact form in the browser, put Ada Lovelace and "
            "ada at example dot test in there, then submit it and make sure it went through?"
        ),
        completion_criteria="The page is navigated, both fields are filled, and the submitted confirmation is observed.",
        steps=steps,
        harness=harness,
        evidence_ids=["navigate-form", "fill-name", "fill-email", "submit-form"],
    )

    assert browser_calls == [arguments for _call_id, _name, arguments in steps[1:]]
    assert task["tools_used"] == ["tool_search", "browser_interact"]
    assert "Thanks" in task["actions"][-2]["outcome"]
