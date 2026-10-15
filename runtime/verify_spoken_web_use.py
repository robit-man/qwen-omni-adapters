#!/usr/bin/env python3
"""Live TTS→ASR→Omni browser navigation and memory gate on public documentation.

This runner never supplies the spoken goal as typed input to the action loop.
It checks fresh browser state and actual session memory after the model acts.
All artifacts are run-local and generated speech is not labeled a human mic.
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Any

import httpx

from runtime.computer_use_speech import audio_turn, synthesize_instruction

DOC_URL = "https://docs.python.org/3/library/pathlib.html"


def _result(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Tool returned no structured evidence")
    return value


def run(args: argparse.Namespace) -> int:
    root = args.output.parent / (args.output.stem + "-artifacts")
    root.mkdir(parents=True, exist_ok=True)
    marker = "Copper Finch " + secrets.choice(["Willow", "Cedar", "Maple", "Birch"])
    report: dict[str, Any] = {"schema": "robit.omni.spoken-web-use.v1", "model": args.model,
                              "source": "generated_tts_audio", "marker": marker, "passed": False, "turns": []}
    started = time.monotonic()
    token = args.token_file.read_text().strip()
    headers = {"Authorization": f"Bearer {token}"}
    cookie = secrets.token_hex(32)
    try:
        with httpx.Client(headers=headers, cookies={"omni_portal_session": cookie},
                          timeout=httpx.Timeout(300, connect=10), trust_env=False) as client:
            def tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                response = client.post(args.endpoint.rstrip("/") + f"/api/tools/{name}/call", json={"arguments": arguments})
                response.raise_for_status()
                value = _result(response.json().get("result"))
                if "error" in value:
                    raise ValueError(f"Independent {name} observation failed")
                return value

            instructions = [
                "Use the visible Chromium browser to find the official Python documentation for working with file system paths. "
                "Explore the page and open the method for making directories. Read what the parents option does. "
                f"Remember that explanation together with the exact source URL and the label {marker}. "
                "Use browser controls to navigate the documentation, and store the memory with the memory tool.",
                f"Retrieve the documentation memory labeled {marker}. Reopen its saved source in the visible browser. "
                "Navigate to the section about pure paths using the page's own links, then scroll and inspect its documentation. "
                "Update the same memory with that source URL and what you learned."
            ]
            memory_identity: tuple[str, str] | None = None
            for index, instruction in enumerate(instructions):
                turn_root = root / f"turn-{index + 1}"
                raw, tts = synthesize_instruction(client, args.endpoint, args.model, instruction, turn_root)
                body, evidence = audio_turn(client, args.endpoint, args.model, raw, tools=True,
                                             max_seconds=args.timeout,
                                             evidence_path=turn_root / "speech-events.jsonl")
                turn: dict[str, Any] = {"tts": tts, "speech": evidence,
                                       "reply": str(body.get("message", {}).get("content") or "")}
                report["turns"].append(turn)
                (turn_root / "action-evidence.json").write_text(json.dumps(turn, indent=2) + "\n")
                traces = evidence["tools"]
                writes = [item for item in traces if item.get("name") == "memory_write" and item.get("ok")]
                browser_calls = [item for item in traces if item.get("name") in {"browser_interact", "computer_use"}
                                 and item.get("ok")]
                if not writes or not browser_calls:
                    raise ValueError("Spoken task did not produce both real browser and memory-write tool receipts")
                write = writes[-1]
                arguments = write.get("arguments", {})
                memory_identity = (str(arguments.get("topic") or ""), str(arguments.get("key") or ""))
                if not all(memory_identity):
                    raise ValueError("Memory receipt omitted its actual session-owned identity")
                memory = tool("memory_read", {"topic": memory_identity[0], "key": memory_identity[1]})
                memory_text = json.dumps(memory.get("value"), ensure_ascii=False)
                if memory.get("found") is not True or "docs.python.org/3/library/pathlib.html" not in memory_text:
                    raise ValueError("Independent memory readback did not contain the real documentation URL")
                page = tool("browser_interact", {"action": "snapshot"})
                snapshot = page.get("snapshot", page)
                page_text = json.dumps(snapshot, ensure_ascii=False)
                if "docs.python.org/3/library/pathlib.html" not in page_text:
                    raise ValueError("Fresh browser state did not verify the documentation destination")
                expected_fragment = "Path.mkdir" if index == 0 else "PurePath"
                if expected_fragment not in page_text:
                    raise ValueError("Fresh rendered evidence did not contain the requested method or class")
                if index == 1 and not any(item.get("name") in {"memory_read", "memory_search"} and item.get("ok") for item in traces):
                    raise ValueError("Follow-up did not actually retrieve its remembered action context")
                turn.update(memory_readback=memory,
                            browser_verification={"source_url_present": True, "target": expected_fragment})
                (turn_root / "action-evidence.json").write_text(json.dumps(turn, indent=2) + "\n")
            assert memory_identity is not None
            with httpx.Client(headers=headers, cookies={"omni_portal_session": secrets.token_hex(32)},
                              timeout=30, trust_env=False) as isolated:
                response = isolated.post(args.endpoint.rstrip("/") + "/api/tools/memory_read/call",
                                         json={"arguments": {"topic": memory_identity[0], "key": memory_identity[1]}})
                response.raise_for_status()
                if _result(response.json().get("result")).get("found") is not False:
                    raise ValueError("Another session could retrieve the test memory")
            report["memory_session_isolation"] = True
            tool("browser_interact", {"action": "close"})
            report["owned_browser_closed"] = True
            report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "detail": str(exc)[:1000]}
    finally:
        # The authenticated Trash operation removes this runner's session only.
        try:
            with httpx.Client(headers=headers, cookies={"omni_portal_session": cookie}, timeout=30,
                              trust_env=False) as cleaner:
                report["owned_session_cleared"] = cleaner.delete(args.endpoint.rstrip("/") + "/api/diagnostics").status_code == 204
        except httpx.HTTPError:
            report["owned_session_cleared"] = False
        report["elapsed_s"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "turns"}, indent=2))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
