from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.background_agent import (
    AGENT_SYSTEM_PROMPT,
    COMPACT_AGENT_SYSTEM_PROMPT,
    MAX_CHECKPOINT_REPORT_CHARS,
    MAX_PHASE_ACTIONS,
    MAX_RETAINED_TASK_MESSAGES,
    MAX_TOOL_RESULT_CHARS,
    TASK_CHECKPOINT_TOOL,
    TASK_COMPACT_TOOL,
    TASK_RECOVERY_TOOL,
    TASK_START_REQUEST,
    BackgroundAgent,
    _action_audit_report,
    _action_family,
    _apply_capability_retry_budget,
    _audit_for_evidence,
    _audit_json,
    _background_discovery_preflight,
    _background_portal_session,
    _background_step_token_limit,
    _background_tool_contract,
    _bounded_tool_result,
    _call_fingerprint,
    _checkpoint_available,
    _checkpoint_has_milestone,
    _checkpoint_retry_pending,
    _compact_task_messages,
    _compact_tool_schema,
    _compaction_available,
    _compaction_evidence_records,
    _compaction_receipt,
    _compaction_tool_available,
    _completion_is_audited,
    _computer_action_messages,
    _context_metrics,
    _direct_alternative_tools,
    _discard_visual_frames,
    _durable_task_messages,
    _evidence_authority,
    _filter_background_discovery,
    _focus_memory,
    _ForegroundPreempted,
    _fresh_executor_messages,
    _freshest_evidence_id,
    _ground_visual_click,
    _guard_repeated_unchanged_result,
    _inference_diagnostics,
    _latest_external_result_digest,
    _latest_receipt_requires_checkpoint,
    _latest_result_requires_replan,
    _latest_tool_fingerprint,
    _MalformedToolCall,
    _milestone_evidence_ids,
    _NonRetryableBackgroundError,
    _normalize_progress_evidence,
    _recovery_required,
    _retired_action_families,
    _sanitize_checkpoint_history,
    _search_task_evidence,
    _seen_tool_fingerprints,
    _source_receipt_has_evidence,
    _stream_error,
    _structured_action_phase,
    _successor_tools,
    _task_allows_physical_camera,
    _task_context_limits,
    _task_expand_available,
    _task_system_prompt,
    _task_virtual_query,
    _tool_evidence,
    _trailing_capability_failures,
    _uncheckpointed_action_count,
    _web_fetch_preflight,
)
from portal.background_tasks import BackgroundTaskStore
from portal.documents import SessionDocumentStore
from portal.tools import PortalToolHarness
from qwen_omni_adapters.virtual_memory.packer import conservative_token_estimate


def test_task_system_prompt_pins_objective_and_latest_directions() -> None:
    prompt = _task_system_prompt(
        {
            "objective": "Open the requested application.",
            "completion_criteria": "Its window is visible.",
            "guidance": [
                {"content": "Use the desktop session."},
                {"content": "Verify the active window."},
            ],
            "progress": [
                "Ran tool_search and retained its result.",
                "The target folder is verified; research is the next unmet milestone.",
            ],
            "actions": [
                {
                    "call_id": "source-1",
                    "tool": "web_fetch",
                    "arguments": json.dumps(
                        {"url": "https://example.test/field-service"}
                    ),
                    "outcome": json.dumps({"content": "retrieved"}),
                    "ok": True,
                }
            ],
        }
    )

    assert prompt.startswith("<current_task>\nObjective: Open the requested application.")
    assert "Completion criteria: Its window is visible." in prompt
    assert "- Verify the active window." in prompt
    assert "<current_plan_state>" not in prompt
    assert "research is the next unmet milestone" not in prompt
    assert "Ran tool_search" not in prompt
    assert '<focus_memory schema="robit.omni.background-focus.v2"' in prompt
    assert "https://example.test/field-service" in prompt
    assert "task_expand" not in prompt
    assert "no paging control is available or needed" in prompt
    assert "Advance the earliest unmet prerequisite" in prompt
    assert "downstream target absent" in prompt
    assert "Ignore unrelated topics" in prompt
    assert "every qualifier in the completion criteria as a constraint" in prompt
    assert prompt.endswith(AGENT_SYSTEM_PROMPT)


def test_background_discovery_requires_an_exact_typed_family() -> None:
    rejected = _background_discovery_preflight({"family": "not-a-family"})

    assert rejected is not None
    assert rejected["error"] == "invalid_tool_family"
    assert rejected["task_progress"] is False
    assert _background_discovery_preflight({"family": "filesystem"}) is None
    assert _background_discovery_preflight({"family": "browser"}) is None
    assert _background_discovery_preflight({"family": "web"}) is None
    assert _background_discovery_preflight({"family": "uncertain"}) is None


def test_physical_camera_scope_is_distinct_from_browser_and_desktop_vision() -> None:
    assert not _task_allows_physical_camera(
        {
            "objective": "Build a polished SaaS dashboard and inspect it visually.",
            "completion_criteria": "Verify the rendered browser and desktop window.",
        }
    )
    assert not _task_allows_physical_camera(
        {
            "objective": "Build a camera settings page.",
            "completion_criteria": "Do not use the physical camera.",
        }
    )
    assert _task_allows_physical_camera(
        {"objective": "Use the webcam to describe what I am holding."}
    )
    assert _task_allows_physical_camera(
        {
            "objective": "Continue the application build.",
            "guidance": [{"content": "Now watch the room for motion."}],
        }
    )


def test_background_discovery_filters_unauthorized_physical_camera() -> None:
    task = {"objective": "Create and test a Next.js application."}
    result = _filter_background_discovery(
        {
            "available_tools": ["request_camera_view", "workspace_file"],
            "suggested_tools": ["request_camera_view", "workspace_file"],
            "results": [
                {"name": "request_camera_view"},
                {"name": "workspace_file"},
            ],
        },
        task,
    )

    assert result["available_tools"] == ["workspace_file"]
    assert result["suggested_tools"] == ["workspace_file"]
    assert result["results"] == [{"name": "workspace_file"}]
    assert result["scope_filtered_tools"] == ["request_camera_view"]

    rejected = _filter_background_discovery(
        {
            "available_tools": ["request_camera_view"],
            "results": [{"name": "request_camera_view"}],
        },
        task,
    )
    assert rejected["error"] == "physical_camera_outside_task_scope"


def test_background_worker_replans_generic_discovery_without_camera_capture(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create(
        "Create a small Next.js application in the workspace.",
        "The requested file exists and is verified.",
    )
    chat_round = 0
    discovery_posts = 0
    camera_posts = 0

    def call(call_id: str, name: str, arguments: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round, discovery_posts, camera_posts
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/tool_search/call":
            discovery_posts += 1
            return httpx.Response(
                200,
                json={
                    "result": {
                        "available_tools": [
                            "request_camera_view",
                            "workspace_file",
                        ],
                        "suggested_tools": [
                            "request_camera_view",
                            "workspace_file",
                        ],
                        "results": [
                            {"name": "request_camera_view"},
                            {"name": "workspace_file"},
                        ],
                    }
                },
            )
        if request.url.path == "/api/tools/request_camera_view/call":
            camera_posts += 1
            return httpx.Response(200, json={"result": {"camera_capture_requested": True}})
        if request.url.path == "/api/tools/workspace_file/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "operation": "write",
                        "path": "/workspace/app/page.tsx",
                        "evidence_authority": "mutation",
                    }
                },
            )
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": "verification",
                    }
                },
            )

        chat_round += 1
        if chat_round == 1:
            return call(
                "generic-discovery",
                "tool_search",
                {"family": "not-a-family"},
            )
        if chat_round == 2:
            result = json.loads(payload["messages"][-1]["content"])
            assert result["error"] == "invalid_tool_family"
            return call(
                "specific-discovery",
                "tool_search",
                {"family": "filesystem"},
            )
        if chat_round == 3:
            exposed = {item["function"]["name"] for item in payload["tools"]}
            assert "workspace_file" in exposed
            assert "request_camera_view" not in exposed
            return call(
                "write-app",
                "workspace_file",
                {
                    "action": "write",
                    "path": "/workspace/app/page.tsx",
                    "content": "export default function Page() { return <main>Ready</main> }",
                },
            )
        if chat_round == 4:
            return _checkpoint_response(
                "progress",
                "I created the application file; verification remains.",
                ["write-app"],
            )
        if chat_round == 5:
            return call(
                "discover-verification",
                "tool_search",
                {"family": "shell"},
            )
        if chat_round == 6:
            return call(
                "verify-app",
                "shell",
                {"command": "test -f /workspace/app/page.tsx", "intent": "verify"},
            )
        return _checkpoint_response(
            "complete",
            "I created and verified the requested application file.",
            ["verify-app"],
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
    assert discovery_posts == 2
    assert camera_posts == 0


def test_task_virtual_query_reserves_space_for_latest_direction() -> None:
    latest_direction = "Use the corrected source and write the retained plan."
    query = _task_virtual_query(
        {
            "objective": "long objective token " * 500,
            "guidance": [
                {"content": "An older direction."},
                {"content": latest_direction},
            ],
        }
    )

    assert query.startswith("Advance and verify the pinned task. Objective: ")
    assert f"Latest user direction: {latest_direction}" in query
    assert len(query) <= 1_200


def test_task_context_limits_follow_the_live_resident_window(
    tmp_path: Path, monkeypatch,
) -> None:
    state = tmp_path / "resident-context"
    state.write_text("4096\n", encoding="utf-8")
    monkeypatch.setenv("OMNI_COMPREHENSION_CONTEXT_TOKENS", "65536")
    monkeypatch.setenv("OMNI_COMPREHENSION_CONTEXT_FILE", str(state))
    monkeypatch.delenv("OMNI_REPO_ROOT", raising=False)

    constrained = _task_context_limits()
    assert constrained == {
        "resident_context_tokens": 4_096,
        "context_bytes": 32 * 1_024,
        "retained_messages": 4,
        "focus_chars": 1_228,
        "replay_chars": 4_096,
        "compaction_high_water_tokens": 2_949,
    }

    state.write_text("8192\n", encoding="utf-8")
    expanded = _task_context_limits()
    assert expanded["resident_context_tokens"] == 8_192
    assert expanded["context_bytes"] == 65_536
    assert expanded["retained_messages"] == 6
    assert expanded["focus_chars"] == 2_457
    assert expanded["replay_chars"] == 8_192

    monkeypatch.setenv("OMNI_COMPREHENSION_CONTEXT_FILE", str(tmp_path / "missing"))
    assert _task_context_limits()["resident_context_tokens"] == 4_096


def test_background_output_budget_tracks_selected_action_and_kv_tier() -> None:
    assert (
        _background_step_token_limit(
            768, [], 16_384, allow_expansion=True
        )
        == 768
    )
    assert (
        _background_step_token_limit(
            768, ["workspace_file"], 16_384, allow_expansion=True
        )
        == 3_072
    )
    assert (
        _background_step_token_limit(
            768, ["shell"], 16_384, allow_expansion=True
        )
        == 3_072
    )
    assert (
        _background_step_token_limit(
            768, ["workspace_file"], 4_096, allow_expansion=True
        )
        == 819
    )
    assert (
        _background_step_token_limit(
            256, ["workspace_file"], 16_384, allow_expansion=False
        )
        == 256
    )


def test_constrained_task_contract_and_query_fit_the_resident_tier() -> None:
    task = {
        "objective": "Build and verify the requested application. " * 35,
        "completion_criteria": "The application exists and its tests pass. " * 12,
        "guidance": [{"content": "Keep the current workspace and repair it in place."}],
        "actions": [
            {
                "call_id": "write-current",
                "tool": "workspace_file",
                "arguments": json.dumps({"action": "read", "path": "/tmp/app/page.tsx"}),
                "outcome": json.dumps({"content": "export default function Page() {}"}),
                "ok": True,
            }
        ],
    }

    prompt = _task_system_prompt(
        task,
        resident_context_tokens=4_096,
        expand_available=True,
    )
    query = _task_virtual_query(task, resident_context_tokens=4_096)

    assert prompt.endswith(COMPACT_AGENT_SYSTEM_PROMPT)
    assert AGENT_SYSTEM_PROMPT not in prompt
    assert "select exactly one typed capability family by meaning" in prompt
    assert len(query) <= 420
    assert "Build and verify" in prompt
    assert "tests pass" in prompt
    assert "write-current" in prompt


def test_constrained_action_contract_keeps_json_rules_without_prose_bloat() -> None:
    schemas = _background_tool_contract(
        ["browser_interact", "shell"],
        recovery_required=False,
        phase_boundary=False,
        expand_available=True,
        can_checkpoint=True,
        resident_context_tokens=4_096,
    )
    names = [item["function"]["name"] for item in schemas]

    assert names == ["browser_interact", "task_expand", "task_checkpoint"]
    serialized = json.dumps(schemas, sort_keys=True)
    assert '"required"' in serialized
    assert '"enum"' in serialized
    assert '"additionalProperties": false' in serialized
    assert '"description"' not in serialized
    assert conservative_token_estimate(serialized) < 850

    discovery = _background_tool_contract(
        [],
        recovery_required=False,
        phase_boundary=False,
        expand_available=True,
        can_checkpoint=False,
        resident_context_tokens=4_096,
    )
    assert [item["function"]["name"] for item in discovery] == [
        "tool_search",
        "task_expand",
    ]

    original = {"description": "omit", "enum": ["a"], "required": ["x"]}
    assert _compact_tool_schema(original) == {
        "enum": ["a"],
        "required": ["x"],
    }


def test_audited_stagnation_retires_only_the_typed_transition() -> None:
    task = {
        "task_state": {
            "controller": {"stagnation": {"fingerprint": "same", "count": 2}},
            "audit_reports": [
                {
                    "action_family": "shell:inspect",
                    "epistemic_progress": False,
                    "environmental_progress": False,
                }
            ],
        }
    }
    retired = _retired_action_families(task)

    assert retired == {"shell:inspect"}
    assert _action_family("shell", {"intent": "inspect"}) in retired
    assert _action_family("shell", {"intent": "mutate_filesystem"}) not in retired

    schemas = _background_tool_contract(
        ["shell"],
        recovery_required=False,
        phase_boundary=False,
        expand_available=False,
        can_checkpoint=False,
        resident_context_tokens=16_384,
        retired_action_families=retired,
    )
    shell = next(
        item for item in schemas if item["function"]["name"] == "shell"
    )
    intents = shell["function"]["parameters"]["properties"]["intent"]["enum"]
    assert "inspect" not in intents
    assert intents == ["mutate_filesystem", "mutate_runtime", "verify"]

    task["task_state"]["controller"]["stagnation"]["count"] = 0
    assert _retired_action_families(task) == set()

def test_evidence_paging_cannot_loop_or_replace_capability_discovery() -> None:
    checkpoint = [
        {
            "role": "tool",
            "tool_name": "task_checkpoint",
            "content": '{"accepted": true}',
        }
    ]
    assert _task_expand_available(checkpoint, compacted=False) is False
    assert _task_expand_available(checkpoint, compacted=True) is True

    expanded = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "task_expand",
            "content": '{"expanded": [{"evidence_id": "source-1"}]}',
        },
    ]
    assert _task_expand_available(expanded, compacted=True) is False

    missing = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "task_expand",
            "content": '{"error": "evidence_not_found"}',
        },
    ]
    assert _task_expand_available(missing, compacted=True) is False

    routed = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "tool_search",
            "content": '{"available_tools": ["workspace_file"]}',
        },
    ]
    assert _task_expand_available(routed, compacted=True) is False

    changed = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "workspace_file",
            "content": '{"action": "mkdir", "path": "/tmp/app"}',
        },
    ]
    assert _task_expand_available(changed, compacted=True) is True

    inspected = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "shell",
            "content": json.dumps(
                {
                    "exit_code": 0,
                    "stdout": "target is empty",
                    "task_progress": False,
                    "evidence_authority": "inspection",
                }
            ),
        },
    ]
    assert _task_expand_available(inspected, compacted=True) is False

    failed = [
        *checkpoint,
        {
            "role": "tool",
            "tool_name": "workspace_file",
            "content": '{"error": "duplicate_tool_call"}',
        },
    ]
    assert _task_expand_available(failed, compacted=True) is False


def test_only_typed_nonprogress_result_reenables_bounded_planning() -> None:
    inspection = [
        {
            "role": "tool",
            "tool_name": "shell",
            "content": json.dumps(
                {"exit_code": 0, "stdout": "empty", "task_progress": False}
            ),
        }
    ]
    assert _latest_result_requires_replan(inspection) is True
    assert (
        _latest_result_requires_replan(
            [
                *inspection,
                {"role": "assistant", "content": "I should change approach."},
                {"role": "user", "content": "Make one structured call now."},
            ]
        )
        is False
    )
    assert (
        _latest_result_requires_replan(
            [
                *inspection,
                {
                    "role": "tool",
                    "tool_name": "tool_search",
                    "content": json.dumps(
                        {
                            "family": "web",
                            "available_tools": ["web_fetch", "web_search"],
                        }
                    ),
                },
            ]
        )
        is False
    )
    assert (
        _latest_result_requires_replan(
            [
                *inspection,
                {
                    "role": "tool",
                    "tool_name": "workspace_file",
                    "content": json.dumps(
                        {"action": "write", "path": "/tmp/app/page.tsx"}
                    ),
                },
            ]
        )
        is False
    )
    assert (
        _latest_result_requires_replan(
            [
                {
                    "role": "tool",
                    "tool_name": "shell",
                    "content": json.dumps(
                        {
                            "error": "permission denied",
                            "task_progress": False,
                        }
                    ),
                }
            ]
        )
        is False
    )


def test_nonprogress_replan_can_select_a_different_typed_family() -> None:
    schemas = _background_tool_contract(
        ["shell"],
        recovery_required=False,
        phase_boundary=False,
        expand_available=False,
        can_checkpoint=True,
        resident_context_tokens=16_384,
        recovery_exploration=True,
    )

    assert [schema["function"]["name"] for schema in schemas] == [
        "shell",
        "tool_search",
        "task_checkpoint",
    ]

    routed_messages = [
        {
            "role": "tool",
            "tool_name": "shell",
            "content": json.dumps(
                {"exit_code": 0, "task_progress": False, "stdout": "empty"}
            ),
        },
        {
            "role": "tool",
            "tool_name": "tool_search",
            "content": json.dumps(
                {"family": "web", "available_tools": ["web_fetch", "web_search"]}
            ),
        },
    ]
    routed = _background_tool_contract(
        ["web_fetch", "web_search"],
        recovery_required=False,
        phase_boundary=False,
        expand_available=False,
        can_checkpoint=True,
        resident_context_tokens=16_384,
        recovery_exploration=_latest_result_requires_replan(routed_messages),
    )
    assert [schema["function"]["name"] for schema in routed] == [
        "web_fetch",
        "web_search",
        "task_checkpoint",
    ]


def test_nonsticky_milestone_closes_before_discovery_reopens() -> None:
    task = {
        "task_state": {
            "audit_reports": [
                {
                    "evidence_id": "source-1",
                    "action_family": "web_fetch:execute",
                    "milestone_progress": True,
                }
            ]
        }
    }
    source_messages = [
        {
            "role": "tool",
            "tool_name": "web_fetch",
            "tool_call_id": "source-1",
            "content": '{"content":"exact acquired source"}',
        },
        {"role": "user", "content": "Assess the result."},
    ]
    assert _latest_receipt_requires_checkpoint(task, source_messages) is True
    schemas = _background_tool_contract(
        [],
        recovery_required=False,
        phase_boundary=True,
        expand_available=False,
        can_checkpoint=True,
        resident_context_tokens=16_384,
    )
    assert [schema["function"]["name"] for schema in schemas] == [
        "task_checkpoint"
    ]

    later_probe = [
        *source_messages,
        {
            "role": "tool",
            "tool_name": "get_system_snapshot",
            "tool_call_id": "snapshot-1",
            "content": '{"task_progress":false}',
        },
    ]
    assert _latest_receipt_requires_checkpoint(task, later_probe) is False


def test_web_fetch_preflight_allows_a_user_supplied_url_but_not_self_authorization() -> None:
    task = {
        "objective": "Inspect https://example.test/direct/ and summarize it.",
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "invented",
                    "function": {
                        "name": "web_fetch",
                        "arguments": {"url": "https://invented.test/source"},
                    },
                }
            ],
        }
    ]

    assert (
        _web_fetch_preflight(
            messages,
            task,
            {"url": "https://example.test/direct"},
        )
        is None
    )
    rejected = _web_fetch_preflight(
        messages,
        task,
        {"url": "https://invented.test/source"},
    )
    assert rejected is not None
    assert rejected["error"] == "undiscovered_url"
    assert rejected["allowed_urls"] == ["https://example.test/direct"]


def test_phase_budget_counts_only_actions_after_guidance_or_accepted_progress() -> None:
    task = {
        "guidance": [{"content": "Use the corrected target.", "received_at": 20.0}],
        "actions": [
            {"at": 10.0, "tool": "shell", "outcome": '{"exit_code": 0}'},
            {"at": 21.0, "tool": "tool_search", "outcome": "{}"},
            {"at": 22.0, "tool": "shell", "outcome": '{"exit_code": 0}'},
            {
                "at": 23.0,
                "tool": "task_checkpoint",
                "outcome": '{"accepted": true, "action": "progress"}',
            },
            {"at": 24.0, "tool": "web_search", "outcome": "{}"},
        ],
    }

    assert _uncheckpointed_action_count(task) == 1
    assert MAX_PHASE_ACTIONS == 8


def test_only_progress_checkpoint_evidence_can_be_normalized_to_freshest() -> None:
    evidence = {
        "fresh": {"name": "shell", "result": {"exit_code": 0}},
        "failed": {"name": "shell", "result": {"exit_code": 1}},
        "discovery": {
            "name": "web_search",
            "result": {
                "mode": "discover",
                "provenance": {
                    "authority": "discovery_only",
                    "citation_ready": False,
                },
            },
        },
    }

    assert _normalize_progress_evidence(
        "progress", ["invented"], evidence, "fresh"
    ) == (["fresh"], True)
    assert _normalize_progress_evidence(
        "complete", ["invented"], evidence, "fresh"
    ) == (["invented"], False)
    assert _normalize_progress_evidence(
        "progress", ["invented"], evidence, "failed"
    ) == (["invented"], False)
    assert _normalize_progress_evidence(
        "progress", ["invented"], evidence, "discovery"
    ) == (["invented"], False)


def test_checkpoint_evidence_authority_separates_discovery_inspection_and_progress() -> None:
    assert _evidence_authority(
        {
            "name": "web_search",
            "result": {
                "mode": "discover",
                "provenance": {"authority": "discovery_only", "citation_ready": False},
            },
        }
    ) == "discovery"
    assert _evidence_authority(
        {"name": "workspace_file", "result": {"task_progress": False}}
    ) == "inspection"
    assert _evidence_authority(
        {"name": "shell", "result": {"exit_code": 0}}
    ) == "concrete"
    assert _evidence_authority(
        {
            "name": "shell",
            "result": {
                "exit_code": 0,
                "task_progress": True,
                "evidence_authority": "mutation",
            },
        }
    ) == "mutation"
    assert _evidence_authority(
        {
            "name": "shell",
            "result": {
                "exit_code": 0,
                "task_progress": False,
                "evidence_authority": "verification",
            },
        }
    ) == "verification"
    assert _evidence_authority(
        {"name": "shell", "result": {"exit_code": 1}}
    ) == "failed"


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

    active = _inference_diagnostics(
        {
            "message": {"role": "assistant", "tool_calls": []},
            "portal": {
                "virtual_context": {
                    "mode": "active",
                    "physical_context_tokens": 4096,
                    "working_tokens": 4012,
                }
            },
        },
        768,
    )
    assert active["virtual_context"] == {
        "mode": "active",
        "physical_context_tokens": 4096,
        "working_tokens": 4012,
    }


def test_strict_current_visual_target_overrides_language_coordinate_guess() -> None:
    arguments, receipt = _ground_visual_click(
        {
            "action": "visual_click",
            "coordinate_unit": "normalized_1000",
            "x": 163,
            "y": 119,
        },
        (
            "<visual_observation>target=blue triangle point=(818,494) "
            "bbox=(768,444,868,544)</visual_observation>"
        ),
    )

    assert arguments["x"] == 818
    assert arguments["y"] == 494
    assert arguments["target"] == "blue triangle"
    assert receipt == {
        "source": "strict_current_visual_observation",
        "target": "blue triangle",
        "proposed": {"x": 163, "y": 119},
        "executed": {"x": 818, "y": 494},
    }


def test_ambiguous_visual_targets_stay_with_language_reasoning() -> None:
    proposed = {
        "action": "visual_click",
        "coordinate_unit": "normalized_1000",
        "x": 400,
        "y": 500,
    }
    observation = (
        "<visual_observation>target=first point=(100,200) bbox=(50,150,150,250); "
        "target=second point=(700,800) bbox=(650,750,750,850)</visual_observation>"
    )

    assert _ground_visual_click(proposed, observation) == (proposed, None)


def test_natural_current_visual_instruction_supplies_point_head_referent() -> None:
    proposed = {
        "action": "visual_click",
        "coordinate_unit": "normalized_1000",
        "x": 840,
        "y": 366,
    }
    observation = (
        '<visual_observation>Screen shows a game titled "Stage 1 of 3 — click '
        'the BLUE TRIANGLE". The instruction identifies one target — the blue '
        "triangle at row 2, column 4.</visual_observation>"
    )

    grounded, receipt = _ground_visual_click(proposed, observation)

    assert grounded == {**proposed, "target": "BLUE TRIANGLE"}
    assert receipt == {
        "source": "current_visual_referring_expression",
        "target": "BLUE TRIANGLE",
        "proposed": {"x": 840, "y": 366},
    }


def test_computer_action_scope_keeps_durable_state_and_two_fresh_motor_cycles() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "long-horizon objective"},
    ]
    for index in range(3):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"gui-{index}",
                            "function": {
                                "name": "gui_interact",
                                "arguments": {"action": "snapshot"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "gui_interact",
                    "tool_call_id": f"gui-{index}",
                    "content": json.dumps(
                        {
                            "rendered": True,
                            "coordinate_space": {"name": "active_window"},
                            "marker": f"result-{index}",
                        }
                    ),
                },
                {
                    "role": "user",
                    "content": f"visual-{index}",
                    "images": [
                        {
                            "mime_type": "image/png",
                            "encoding": "base64",
                            "data": f"image-{index}",
                        }
                    ],
                },
            ]
        )
    original = json.loads(json.dumps(messages))
    scoped = _computer_action_messages(
        messages,
        {
            "progress": ["Reached the site."],
            "actions": [
                {
                    "call_id": "gui-2",
                    "tool": "gui_interact",
                    "arguments": '{"action":"click","x":730,"y":527}',
                    "outcome": '{"rendered":true}',
                    "ok": True,
                    "receipt": {
                        "action": "visual_click",
                        "target": "BLUE TRIANGLE",
                    },
                }
            ],
        },
        ["gui_interact"],
        recovery_required=False,
    )

    rendered = json.dumps(scoped)
    assert messages == original
    assert scoped[:2] == messages[:2]
    assert "<computer_action_state>" in scoped[2]["content"]
    assert "Reached the site" not in scoped[2]["content"]
    assert "BLUE TRIANGLE" in scoped[2]["content"]
    assert '"x":730' not in scoped[2]["content"]
    assert "action=click" in scoped[2]["content"]
    assert '"target": "BLUE TRIANGLE"' in scoped[2]["content"]
    assert "gui-0" not in rendered
    assert "image-0" not in rendered
    assert "gui-1" not in rendered and "gui-2" in rendered
    assert "image-1" not in rendered and "image-2" in rendered


def test_computer_action_scope_is_disabled_while_capability_recovery_is_required() -> None:
    messages = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "objective"},
    ]

    assert (
        _computer_action_messages(
            messages,
            {},
            ["gui_interact"],
            recovery_required=True,
        )
        is messages
    )


def test_computer_action_scope_retains_completed_form_control_ledger() -> None:
    messages = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "complete the rendered form"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "browser-1",
                    "function": {
                        "name": "browser_interact",
                        "arguments": {"action": "snapshot"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "browser_interact",
            "tool_call_id": "browser-1",
            "content": '{"rendered":true}',
        },
    ]
    actions = [
        {
            "call_id": f"field-{index}",
            "tool": "browser_interact",
            "arguments": '{"action":"type","element_id":"expired"}',
            "outcome": '{"rendered":true}',
            "ok": True,
            "receipt": {"action": "type", "target": f"field {index}"},
        }
        for index in range(28)
    ]

    scoped = _computer_action_messages(
        messages,
        {"actions": actions},
        ["browser_interact"],
        recovery_required=False,
    )
    state = str(scoped[2]["content"])

    assert '"target": "field 27"' in state
    assert '"target": "field 4"' in state
    assert '"target": "field 3"' not in state
    assert "element_id" not in state


def test_context_metrics_charge_visual_tokens_not_raw_base64_transport() -> None:
    messages = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "inspect the current frame"},
        {
            "role": "user",
            "content": "fresh visual evidence",
            "images": [
                {
                    "mime_type": "image/png",
                    "encoding": "base64",
                    "data": "x" * 500_000,
                }
            ],
        },
    ]

    metrics = _context_metrics(messages)

    assert metrics["messages"] == 3
    assert 8 * 1024 < metrics["bytes"] < 12 * 1024
    assert _compaction_available(messages) is False


def test_manual_compaction_is_hidden_during_scoped_computer_action_loop() -> None:
    small_messages = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "objective"},
        *(
            {"role": "user", "content": f"old result {index}"}
            for index in range(20)
        ),
    ]
    assert _compaction_available(small_messages) is False

    messages = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "objective"},
        *(
            {"role": "user", "content": f"old result {index} " + "x" * 6_000}
            for index in range(20)
        ),
    ]

    assert _compaction_available(messages) is True
    assert _compaction_tool_available(messages, ["browser_interact"]) is False
    assert _compaction_tool_available(messages, ["gui_interact"]) is False
    assert _compaction_tool_available(messages, ["shell"]) is True


def test_visual_frame_is_discarded_only_for_a_real_replacement() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "task policy"},
        {"role": "user", "content": "objective"},
        {
            "role": "user",
            "content": "fresh frame",
            "images": [{"encoding": "base64", "data": "current-image"}],
        },
    ]

    # A locally rejected duplicate executes nothing, so its caller deliberately
    # keeps the only current frame instead of invoking this replacement helper.
    assert messages[-1]["images"][0]["data"] == "current-image"  # type: ignore[index]

    discarded = _discard_visual_frames(messages, replacement_note="Superseded by a newer frame.")

    assert discarded == 1
    assert "images" not in messages[-1]
    assert "Superseded by a newer frame." in str(messages[-1]["content"])


def test_allowlisted_executor_alternative_skips_generative_rediscovery() -> None:
    result = {
        "disposition": "change_capability",
        "task_blocked": False,
        "alternative_tools": ["gui_interact", "rm_rf", "gui_interact"],
    }
    messages = [
        {
            "role": "tool",
            "tool_name": "browser_interact",
            "content": json.dumps(result),
        }
    ]

    assert _direct_alternative_tools(result) == ["gui_interact"]
    assert _recovery_required(messages) is False


def test_foreground_preemption_releases_a_blocked_stream_reader_promptly(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    closed = threading.Event()

    class BlockingStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            started.set()
            yield b'{"type":"thinking","content":"working"}\n'
            closed.wait(5)

        def close(self) -> None:
            closed.set()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            stream=BlockingStream(),
        )

    foreground = threading.Event()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=BackgroundTaskStore(tmp_path / "tasks.json"),
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=foreground,
        stop=threading.Event(),
        client=client,
    )
    errors: list[Exception] = []

    def chat() -> None:
        try:
            agent._chat({"model": "model", "messages": [], "stream": False})
        except Exception as error:  # noqa: BLE001 - asserted below
            errors.append(error)

    thread = threading.Thread(target=chat)
    thread.start()
    assert started.wait(1)
    foreground.set()
    thread.join(timeout=1)
    client.close()

    assert not thread.is_alive()
    assert closed.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], _ForegroundPreempted)


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
                                    "completion criteria; "
                                    + (
                                        "no required work remains."
                                        if action == "complete"
                                        else "further required work remains."
                                    )
                                ),
                                "remaining_requirements": (
                                    []
                                    if action == "complete"
                                    else ["Continue the next unmet requirement."]
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
    assert "remaining_requirements" in TASK_CHECKPOINT_TOOL["function"][
        "parameters"
    ]["required"]


def test_background_portal_sessions_are_stable_and_task_isolated() -> None:
    first = _background_portal_session("foreground-seed", "task-one")

    assert first == _background_portal_session("foreground-seed", "task-one")
    assert first != _background_portal_session("foreground-seed", "task-two")
    assert first.startswith("background-")
    assert 16 <= len(first) <= 128


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
        evidence_records=[
            {
                "evidence_id": "fetch-1",
                "tool": "web_fetch",
                "arguments": '{"url":"https://example.test/source"}',
                "result": '{"content":"exact retained source"}',
                "result_sha256": "abc123",
            }
        ],
    )

    assert compacted is not None
    assert compacted["round"] == 0
    assert compacted["progress"] == ["Accepted from the live conversation."]
    assert compacted["compaction"]["before"]["messages"] == 40
    raw = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    assert raw["tasks"][0]["messages"][1]["content"] == "exact objective"
    assert store.expand_evidence(created["task_id"], ["fetch-1"]) == [
        {
            "evidence_id": "fetch-1",
            "tool": "web_fetch",
            "arguments": '{"url":"https://example.test/source"}',
            "result": '{"content":"exact retained source"}',
            "result_sha256": "abc123",
        }
    ]

    # Evidence is append-only across later compactions with the same ID.
    store.compact_context(
        created["task_id"],
        "worker",
        messages=[{"role": "system", "content": "new working set"}],
        receipt={"schema": "robit.omni.task-compaction.v1"},
        evidence_records=[
            {
                "evidence_id": "fetch-1",
                "tool": "web_fetch",
                "result": "replacement must not overwrite",
            }
        ],
    )
    assert store.expand_evidence(created["task_id"], ["fetch-1"])[0][
        "result"
    ] == '{"content":"exact retained source"}'


def test_background_task_store_archives_evidence_with_action_before_compaction(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Fetch and retain a source.", "Use it later.")
    assert store.claim_next("worker") is not None

    updated = store.record_action(
        created["task_id"],
        "worker",
        call_id="fetch-live-1",
        tool="web_fetch",
        arguments='{"url":"https://example.test/source"}',
        outcome='{"content":"bounded audit"}',
        ok=True,
        evidence_record={
            "evidence_id": "fetch-live-1",
            "tool": "web_fetch",
            "arguments": '{"url":"https://example.test/source"}',
            "result": '{"content":"exact expandable passage"}',
            "result_sha256": "feedface",
        },
    )

    assert updated is not None
    assert updated["evidence_record_count"] == 1
    assert store.expand_evidence(created["task_id"], ["fetch-live-1"]) == [
        {
            "evidence_id": "fetch-live-1",
            "tool": "web_fetch",
            "arguments": '{"url":"https://example.test/source"}',
            "result": '{"content":"exact expandable passage"}',
            "result_sha256": "feedface",
        }
    ]


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

    resumed = store.resume_after_review(
        created["task_id"], "The restarts were controlled deployment updates."
    )
    assert resumed is not None
    assert resumed["status"] == "pending"
    assert resumed.get("resume_count") in {None, 0}
    assert "error" not in resumed
    assert "controlled deployment updates" in resumed["progress"][-1]


def test_healthy_checkpoint_resets_expired_lease_streak(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.json"
    store = BackgroundTaskStore(task_path)
    created = store.create("Continue through isolated worker restarts.")
    assert store.claim_next("first") is not None

    state = json.loads(task_path.read_text(encoding="utf-8"))
    state["tasks"][0]["lease_until"] = 0
    task_path.write_text(json.dumps(state), encoding="utf-8")
    resumed = store.claim_next("second")
    assert resumed is not None
    assert resumed["resume_count"] == 1

    checkpoint = store.checkpoint(
        created["task_id"], "second", progress="Verified a healthy milestone."
    )
    assert checkpoint is not None
    assert checkpoint.get("resume_count") in {None, 0}

    state = json.loads(task_path.read_text(encoding="utf-8"))
    state["tasks"][0]["lease_until"] = 0
    task_path.write_text(json.dumps(state), encoding="utf-8")
    resumed_again = store.claim_next("third")
    assert resumed_again is not None
    assert resumed_again["resume_count"] == 1


def test_orderly_release_does_not_count_as_an_expired_lease(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Survive a managed harness restart.")
    assert store.claim_next("old-worker") is not None

    assert store.release_owner("old-worker") == 1
    pending = store.get(created["task_id"])
    assert pending is not None
    assert pending["status"] == "pending"
    assert pending["current_stage"] == "Paused for an orderly worker restart"

    claimed = store.claim_next("new-worker")
    assert claimed is not None
    assert claimed.get("resume_count") in {None, 0}


def test_agent_releases_lease_before_waiting_for_worker_join(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Keep this task durable during deployment.")
    stop = threading.Event()
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=stop,
        client=client,
    )
    assert store.claim_next(agent.owner) is not None

    class JoinProbe:
        def join(self, timeout: float) -> None:
            assert timeout == 5
            released = store.get(task["task_id"])
            assert released is not None
            assert released["status"] == "pending"
            assert released.get("owner") is None

    agent._thread = JoinProbe()  # type: ignore[assignment]
    agent.close()
    client.close()


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


def test_audited_task_state_tracks_knowledge_environment_and_stagnation(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create("Create and verify an artifact.", "The artifact passes checks.")
    current = store.claim_next("worker")
    assert current is not None
    assert current["task_state"]["environment"]["version"] == 0
    clock_audit = _action_audit_report(
        current,
        call_id="clock-1",
        name="get_current_time",
        arguments={},
        result={"date": "2026-09-26", "time": "12:00:00"},
    )
    assert clock_audit["executor_succeeded"] is True
    assert clock_audit["milestone_progress"] is False
    assert not _checkpoint_has_milestone(
        {"task_state": {"audit_reports": [clock_audit]}}, ["clock-1"]
    )
    assert not _completion_is_audited(
        {"task_state": {"environment": {"version": 0}, "audit_reports": [clock_audit]}},
        ["clock-1"],
    )

    empty_fetch = _action_audit_report(
        current,
        call_id="empty-source",
        name="web_fetch",
        arguments={"url": "https://example.test/client-rendered"},
        result={"content": "", "receipt": {"status": 200}},
    )
    assert empty_fetch["executor_succeeded"] is True
    assert empty_fetch["milestone_progress"] is False
    assert not _source_receipt_has_evidence(
        "web_fetch", {"content": "", "receipt": {"status": 200}}
    )
    assert _source_receipt_has_evidence(
        "web_fetch", {"content": "Exact source-bearing text."}
    )

    def record(
        call_id: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal current
        report = _action_audit_report(
            current,
            call_id=call_id,
            name="shell",
            arguments=arguments,
            result=result,
        )
        updated = store.record_action(
            created["task_id"],
            "worker",
            call_id=call_id,
            tool="shell",
            arguments=json.dumps(arguments),
            outcome=json.dumps(result),
            ok=result.get("exit_code") == 0,
            audit_report=report,
        )
        assert updated is not None
        current = updated
        return report

    inspection = {
        "intent": "inspect",
        "cwd": str(tmp_path),
    }
    inspection_result = {
        "exit_code": 0,
        "evidence_authority": "inspection",
        "task_progress": False,
    }
    first = record("inspect-1", inspection, inspection_result)
    assert current["task_state"]["knowledge"]["version"] == 1
    assert current["task_state"]["audit_reports"][-1]["epistemic_progress"] is True

    second = record("inspect-2", inspection, inspection_result)
    assert second["evidence_slot"] == first["evidence_slot"]
    assert second["state_fingerprint"] == first["state_fingerprint"]
    assert current["task_state"]["knowledge"]["version"] == 1
    assert current["task_state"]["controller"]["stagnation"]["count"] == 1
    record("inspect-3", inspection, inspection_result)
    assert current["task_state"]["controller"]["stagnation"]["count"] == 2

    record(
        "mutate-1",
        {"intent": "mutate_filesystem", "cwd": str(tmp_path)},
        {
            "exit_code": 0,
            "evidence_authority": "mutation",
            "effect_receipt": {"changed_paths": [str(tmp_path / "artifact.txt")]},
        },
    )
    assert current["task_state"]["environment"]["version"] == 1
    assert current["task_state"]["controller"]["stagnation"]["count"] == 0
    assert not _completion_is_audited(current, ["mutate-1"])

    record(
        "verify-1",
        {"intent": "verify", "cwd": str(tmp_path)},
        {"exit_code": 0, "evidence_authority": "verification"},
    )
    assert _completion_is_audited(current, ["verify-1"])

    checkpoint = store.checkpoint(
        created["task_id"],
        "worker",
        progress="Artifact exists; perform the final acceptance check.",
        controller_transition={
            "action": "progress",
            "evidence_ids": ["verify-1"],
            "remaining_requirements": ["Run the final acceptance check."],
        },
    )
    assert checkpoint is not None
    controller = checkpoint["task_state"]["controller"]
    assert controller["executor_generation"] == 1
    assert controller["current_subtask"] == "Run the final acceptance check."
    assert controller["next_transition"] == "prethink"

    renewed = store.renew_executor_context(
        created["task_id"],
        "worker",
        messages=[
            {"role": "system", "content": "fresh audited frontier"},
            {"role": "user", "content": "choose the next transition"},
        ],
    )
    assert renewed is not None
    assert renewed["round"] == checkpoint["round"]
    raw = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    assert raw["tasks"][0]["messages"][0]["content"] == "fresh audited frontier"


def test_control_call_cannot_inherit_a_previous_receipt_audit() -> None:
    task = {
        "task_state": {
            "audit_reports": [
                {
                    "audit_id": "audit-inspect-3",
                    "evidence_id": "inspect-3",
                    "stagnation_count": 2,
                    "milestone_progress": False,
                }
            ]
        }
    }

    assert _audit_for_evidence(task, "inspect-3")["stagnation_count"] == 2
    assert _audit_for_evidence(task, "router-1") == {}


def test_phase_handoffs_replay_cumulative_exact_milestone_evidence(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    created = store.create(
        "Research a product and build an application from that evidence.",
        "Research and application artifacts are complete and verified.",
    )
    current = store.claim_next("worker")
    assert current is not None

    source_result = {
        "url": "https://example.test/features",
        "content": "UNIQUE_SOURCE_DETAIL: intake automation reduces manual triage.",
    }
    source_audit = _action_audit_report(
        current,
        call_id="source-1",
        name="web_fetch",
        arguments={"url": "https://example.test/features"},
        result=source_result,
    )
    current = store.record_action(
        created["task_id"],
        "worker",
        call_id="source-1",
        tool="web_fetch",
        arguments='{"url":"https://example.test/features"}',
        outcome='{"content":"bounded source audit"}',
        ok=True,
        evidence_record={
            "evidence_id": "source-1",
            "tool": "web_fetch",
            "arguments": '{"url":"https://example.test/features"}',
            "result": json.dumps(source_result),
            "result_sha256": "source-digest",
        },
        audit_report=source_audit,
    )
    assert current is not None
    current = store.checkpoint(
        created["task_id"],
        "worker",
        controller_transition={
            "action": "progress",
            "evidence_ids": ["source-1"],
            "remaining_requirements": ["Write the research artifact."],
        },
    )
    assert current is not None

    mutation_result = {
        "path": str(tmp_path / "application"),
        "action": "mkdir",
        "evidence_authority": "mutation",
        "effect_receipt": {"changed_paths": [str(tmp_path / "application")]},
    }
    mutation_audit = _action_audit_report(
        current,
        call_id="mkdir-1",
        name="workspace_file",
        arguments={"action": "mkdir", "path": str(tmp_path / "application")},
        result=mutation_result,
    )
    current = store.record_action(
        created["task_id"],
        "worker",
        call_id="mkdir-1",
        tool="workspace_file",
        arguments=json.dumps(
            {"action": "mkdir", "path": str(tmp_path / "application")}
        ),
        outcome='{"action":"mkdir"}',
        ok=True,
        evidence_record={
            "evidence_id": "mkdir-1",
            "tool": "workspace_file",
            "arguments": json.dumps(
                {"action": "mkdir", "path": str(tmp_path / "application")}
            ),
            "result": json.dumps(mutation_result),
            "result_sha256": "mkdir-digest",
        },
        audit_report=mutation_audit,
    )
    assert current is not None
    current = store.checkpoint(
        created["task_id"],
        "worker",
        controller_transition={
            "action": "progress",
            "evidence_ids": ["mkdir-1"],
            "remaining_requirements": ["Write RESEARCH.md from acquired evidence."],
        },
    )
    assert current is not None

    root = current["task_state"]["requirements"][0]
    assert root["evidence_ids"] == ["source-1", "mkdir-1"]
    replay_ids = _milestone_evidence_ids(current)
    assert replay_ids == ["source-1", "mkdir-1"]
    records = store.expand_evidence(created["task_id"], replay_ids)
    messages = _fresh_executor_messages(
        current,
        reason="verified_phase_checkpoint",
        evidence_records=records,
    )
    handoff = messages[1]["content"]
    assert "UNIQUE_SOURCE_DETAIL: intake automation reduces manual triage." in handoff
    assert '"action": "mkdir"' in handoff
    assert '"replayed_evidence_ids": ["source-1", "mkdir-1"]' in handoff
    assert "use it instead of repeating acquisition" in handoff


def test_audited_stagnation_renews_context_and_recovers_to_completion(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create(
        "Create artifact.txt, then verify it exists.",
        "artifact.txt exists and a current verification succeeds.",
    )
    tool_harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    chat_round = 0
    artifact = tmp_path / "artifact.txt"

    def tool_response(call_id: str, name: str, arguments: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path.startswith("/api/tools/"):
            name = request.url.path.split("/")[-2]
            arguments = json.loads(request.content).get("arguments", {})
            result = tool_harness.execute("stagnation-recovery", name, arguments)
            return httpx.Response(200, json={"result": result})

        chat_round += 1
        payload = json.loads(request.content)
        if chat_round == 1:
            return tool_response("discover-shell-1", "tool_search", {"family": "shell"})
        if chat_round in {2, 3, 4}:
            return tool_response(
                f"inspect-{chat_round - 1}",
                "shell",
                {
                    "command": f"printf inspect-{chat_round - 1}",
                    "intent": "inspect",
                    "cwd": str(tmp_path),
                },
            )
        if chat_round == 5:
            last_result = json.loads(payload["messages"][-2]["content"])
            assert last_result["error"] == "audited_state_stagnation"
            assert last_result["stagnation_count"] == 2
            return tool_response("discover-shell-2", "tool_search", {"family": "shell"})
        if chat_round == 6:
            return tool_response(
                "inspect-4",
                "shell",
                {
                    "command": "printf inspect-4",
                    "intent": "inspect",
                    "cwd": str(tmp_path),
                },
            )
        if chat_round == 7:
            assert len(payload["messages"]) == 2
            assert "audited_stagnation_reset" in payload["messages"][1]["content"]
            assert "printf inspect" not in json.dumps(payload["messages"])
            persisted = json.loads(
                (tmp_path / "tasks.json").read_text(encoding="utf-8")
            )
            assert len(persisted["tasks"][0]["messages"]) == 2
            return tool_response(
                "discover-files", "tool_search", {"family": "filesystem"}
            )
        if chat_round == 8:
            return tool_response(
                "write-after-reset",
                "workspace_file",
                {
                    "action": "write",
                    "path": str(artifact),
                    "content": "recovered\n",
                },
            )
        if chat_round == 9:
            return _checkpoint_response(
                "progress",
                "I created the artifact after changing strategy.",
                ["write-after-reset"],
            )
        if chat_round == 10:
            assert len(payload["messages"]) == 2
            assert "verified_phase_checkpoint" in payload["messages"][1]["content"]
            return tool_response("discover-shell-3", "tool_search", {"family": "shell"})
        if chat_round == 11:
            return tool_response(
                "verify-after-reset",
                "shell",
                {
                    "command": "test -f artifact.txt",
                    "intent": "verify",
                    "cwd": str(tmp_path),
                },
            )
        return _checkpoint_response(
            "complete",
            "I created artifact.txt and verified it in the current state.",
            ["verify-after-reset"],
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
    assert artifact.read_text(encoding="utf-8") == "recovered\n"
    assert current["task_state"]["environment"]["version"] == 1
    assert current["task_state"]["controller"]["executor_generation"] == 1
    assert chat_round == 12


def test_selected_alternative_survives_audited_context_reset(tmp_path: Path) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create(
        "Create recovered.txt, then verify it exists.",
        "recovered.txt exists and a current verification succeeds.",
    )
    tool_harness = PortalToolHarness(SessionDocumentStore(ttl_s=300))
    artifact = tmp_path / "recovered.txt"
    chat_round = 0

    def call(call_id: str, name: str, arguments: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path.startswith("/api/tools/"):
            name = request.url.path.split("/")[-2]
            arguments = json.loads(request.content).get("arguments", {})
            result = tool_harness.execute("alternative-reset", name, arguments)
            return httpx.Response(200, json={"result": result})

        chat_round += 1
        payload = json.loads(request.content)
        if chat_round in {1, 5, 7, 9, 13}:
            return call(
                f"discover-{chat_round}", "tool_search", {"family": "shell"}
            )
        if chat_round in {2, 3, 4, 6, 8, 10}:
            return call(
                f"inspect-{chat_round}",
                "shell",
                {
                    "command": f"printf inspect-{chat_round}",
                    "intent": "inspect",
                    "cwd": str(tmp_path),
                },
            )
        if chat_round == 11:
            offered = {
                item["function"]["name"] for item in payload.get("tools", [])
            }
            assert "workspace_file" in offered
            assert "shell" not in offered
            return call(
                "write-after-alternative",
                "workspace_file",
                {
                    "action": "write",
                    "path": str(artifact),
                    "content": "recovered\n",
                },
            )
        if chat_round == 12:
            return _checkpoint_response(
                "progress",
                "I created the requested artifact through the selected alternative.",
                ["write-after-alternative"],
            )
        if chat_round == 14:
            return call(
                "verify-alternative",
                "shell",
                {
                    "command": "test -f recovered.txt",
                    "intent": "verify",
                    "cwd": str(tmp_path),
                },
            )
        return _checkpoint_response(
            "complete",
            "I created recovered.txt and verified it in the current state.",
            ["verify-alternative"],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    agent = BackgroundAgent(
        store=store,
        portal_url="http://portal.test",
        token="token",
        model="model",
        foreground_active=threading.Event(),
        stop=threading.Event(),
        max_slice_rounds=30,
        max_slice_stalls=30,
        client=client,
    )
    agent.start()
    deadline = time.monotonic() + 6
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
    assert artifact.read_text(encoding="utf-8") == "recovered\n"
    assert chat_round == 15


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


def test_checkpoint_allows_one_bounded_id_correction_before_another_action() -> None:
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
    assert _checkpoint_available(messages) is True
    assert "shell-duplicate" not in _tool_evidence(messages)
    assert _freshest_evidence_id(messages) == "shell-1"

    messages.append(
        {
            "role": "tool",
            "tool_name": "task_checkpoint",
            "tool_call_id": "checkpoint-1",
            "content": '{"error": "unsupported_checkpoint", "retryable": true}',
        }
    )
    assert _checkpoint_available(messages) is True
    assert _checkpoint_retry_pending(messages) is True

    messages.append(
        {
            "role": "tool",
            "tool_name": "task_checkpoint",
            "tool_call_id": "checkpoint-2",
            "content": '{"error": "unsupported_checkpoint", "retryable": false}',
        }
    )
    assert _checkpoint_available(messages) is False
    assert _checkpoint_retry_pending(messages) is False

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
        "progress": [
            "Ran tool_search and retained its result.",
            "Created the workspace.",
            "Verified the latest artifact.",
        ],
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
            {
                "call_id": "compact-control",
                "tool": "task_compact",
                "ok": True,
                "outcome": "context compacted",
            },
        ],
    }

    seen = _seen_tool_fingerprints(messages)  # type: ignore[arg-type]
    compacted = _compact_task_messages(  # type: ignore[arg-type]
        messages, task, resident_context_tokens=4_096
    )

    assert len(seen) == 80
    assert len(compacted) <= 8
    assert compacted[0]["role"] == "system"
    assert compacted[1] == objective
    system = compacted[0]["content"]
    assert "Make the final version blue" in system
    assert "fixed-write" in system
    assert "failed-write" not in system
    assert 'omitted_failures="1"' in system
    assert system.count("<focus_memory") == 1
    checkpoint = compacted[2]["content"]
    assert "Verified the latest artifact" not in checkpoint
    assert "Make the final version blue" not in checkpoint
    assert "failed-write" not in checkpoint
    assert "fixed-write" not in checkpoint
    assert "Ran tool_search" not in checkpoint
    assert "compact-control" not in checkpoint
    assert "task_expand evidence pointers" in checkpoint
    assert "task_expand is never an action" in system
    assert "page_in_evidence_id" in system
    assert compacted[3]["role"] == "assistant"
    assert compacted[4]["role"] == "tool"
    assert _context_metrics(compacted)["bytes"] < _context_metrics(messages)["bytes"]

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


def test_shell_fingerprint_ignores_cwd_overridden_by_absolute_cd() -> None:
    command = "cd /tmp/exact-target && pwd && ls -la"

    assert _call_fingerprint("shell", {"command": command}) == _call_fingerprint(
        "shell",
        {"command": command, "cwd": "/tmp/exact-target"},
    )
    assert _call_fingerprint(
        "shell",
        {"command": "pwd && ls -la", "cwd": "/tmp/one"},
    ) != _call_fingerprint(
        "shell",
        {"command": "pwd && ls -la", "cwd": "/tmp/two"},
    )


def test_compaction_retains_typed_expandable_focus_records() -> None:
    task = {
        "actions": [
            {
                "call_id": "source-1",
                "tool": "web_fetch",
                "arguments": json.dumps(
                    {"url": "https://example.test/field-service"}
                ),
                "outcome": json.dumps({"content": "retrieved"}),
                "ok": True,
            },
            {
                "call_id": "list-1",
                "tool": "workspace_file",
                "arguments": json.dumps(
                    {"action": "list", "path": "/tmp/project"}
                ),
                "outcome": json.dumps(
                    {"action": "list", "path": "/tmp/project", "entries": []}
                ),
                "ok": True,
            },
            {
                "call_id": "file-1",
                "tool": "workspace_file",
                "arguments": json.dumps(
                    {"action": "write", "path": "/tmp/project/docs/research.md"}
                ),
                "outcome": json.dumps(
                    {
                        "action": "write",
                        "path": "/tmp/project/docs/research.md",
                        "sha256": "deadbeef",
                        "validation": "text",
                    }
                ),
                "ok": True,
            },
            {
                "call_id": "checkpoint-1",
                "tool": "task_checkpoint",
                "arguments": json.dumps(
                    {
                        "action": "progress",
                        "report": "Research is written.",
                        "criteria_assessment": "Research satisfied; plan remains.",
                        "remaining_requirements": ["Write docs/plan.md."],
                        "evidence_ids": ["file-1"],
                    }
                ),
                "outcome": json.dumps({"accepted": True}),
                "ok": True,
            },
            {
                "call_id": "shell-inspection",
                "tool": "shell",
                "arguments": json.dumps(
                    {"command": "node -v && ls -la", "cwd": "/tmp/project"}
                ),
                "outcome": json.dumps(
                    {
                        "command": "node -v && ls -la",
                        "cwd": "/tmp/project",
                        "exit_code": 0,
                        "stdout": "v20.1.0\ntotal 0\n",
                        "stderr": "",
                        "task_progress": False,
                        "evidence_authority": "inspection",
                    }
                ),
                "ok": True,
            },
        ]
    }

    focus = _focus_memory(task, expand_available=True)

    assert 'schema="robit.omni.background-focus.v2"' in focus
    assert "<phase_checkpoints>" in focus
    assert "<acquired_sources>" in focus
    assert "<artifacts>" in focus
    assert "<inspections>" in focus
    assert "https://example.test/field-service" in focus
    assert "/tmp/project/docs/research.md" in focus
    assert "&quot;task_progress&quot;:false" in focus
    assert "list-1" not in focus.split("<artifacts>", 1)[1].split(
        "</artifacts>", 1
    )[0]
    assert "list-1" in focus.split("<inspections>", 1)[1].split(
        "</inspections>", 1
    )[0]
    inspections = focus.split("<inspections>", 1)[1].split(
        "</inspections>", 1
    )[0]
    assert "shell-inspection" in inspections
    assert "v20.1.0" in inspections
    assert "<other_successes>" not in focus
    assert "Research satisfied; plan remains." not in focus
    assert "declared_remaining_requirements" not in focus
    assert "Write docs/plan.md." not in focus
    assert "model_checkpoint_control_not_task_evidence" in focus
    assert "Phase checkpoints are control boundaries, not proof" in focus
    assert "task_expand(source-1)" not in focus
    assert "task_expand is never an action" in focus
    assert "&quot;page_in_evidence_id&quot;:&quot;source-1&quot;" in focus
    assert "Do not redo an acquired source" in focus


def test_focus_memory_is_bounded_and_keeps_latest_artifact_versions() -> None:
    actions: list[dict[str, object]] = []
    for index in range(20):
        path = f"/tmp/app/file-{index}.py"
        actions.append(
            {
                "call_id": f"file-{index}",
                "tool": "workspace_file",
                "arguments": json.dumps({"action": "write", "path": path}),
                "outcome": json.dumps(
                    {"action": "write", "path": path, "sha256": f"hash-{index}"}
                ),
                "ok": True,
            }
        )
    for index in range(10):
        path = f"/tmp/app/missing-{index}.py"
        actions.append(
            {
                "call_id": f"missing-{index}",
                "tool": "workspace_file",
                "arguments": json.dumps({"action": "read", "path": path}),
                "outcome": json.dumps(
                    {
                        "error": "ToolInputError",
                        "message": f"path is not a file: {path}",
                    }
                ),
                "ok": False,
            }
        )
    actions.extend(
        [
            {
                "call_id": "shared-old",
                "tool": "workspace_file",
                "arguments": json.dumps(
                    {"action": "write", "path": "/tmp/app/shared.py"}
                ),
                "outcome": json.dumps(
                    {
                        "action": "write",
                        "path": "/tmp/app/shared.py",
                        "sha256": "old-hash",
                    }
                ),
                "ok": True,
            },
            {
                "call_id": "shared-new",
                "tool": "workspace_file",
                "arguments": json.dumps(
                    {"action": "replace", "path": "/tmp/app/shared.py"}
                ),
                "outcome": json.dumps(
                    {
                        "action": "replace",
                        "path": "/tmp/app/shared.py",
                        "sha256": "new-hash",
                    }
                ),
                "ok": True,
            },
        ]
    )
    task = {
        "objective": "Build and validate the application.",
        "completion_criteria": "Research, plan, implementation, and tests pass.",
        "actions": actions,
        "compaction": {"schema": "robit.omni.task-compaction.v1"},
    }

    focus = _focus_memory(task, expand_available=True, max_chars=2_400)
    system = _task_system_prompt(
        task, resident_context_tokens=4_096, expand_available=True
    )

    assert len(focus) <= 2_400
    assert 'omitted_artifacts="0"' not in focus
    assert 'omitted_failures="0"' not in focus
    assert "shared-new" in focus
    assert "new-hash" in focus
    assert "shared-old" not in focus
    assert "old-hash" not in focus
    assert system.count("<focus_memory") == 1
    # A 4K tier reserves 768 tokens for output and still has ample room for
    # the current query/tool schema after pinning this system contract.
    assert conservative_token_estimate(system) < 2_200


def test_focus_memory_does_not_promote_discovery_or_blank_browser_state() -> None:
    task = {
        "actions": [
            {
                "call_id": "search-discovery",
                "tool": "web_search",
                "arguments": json.dumps({"query": "field service SaaS"}),
                "outcome": json.dumps(
                    {
                        "mode": "discover",
                        "provenance": {
                            "authority": "discovery_only",
                            "citation_ready": False,
                        },
                        "results": [
                            {"url": "https://example.test/field-service"}
                        ],
                    }
                ),
                "ok": True,
            },
            {
                "call_id": "blank-snapshot",
                "tool": "browser_interact",
                "arguments": json.dumps({"action": "snapshot"}),
                "outcome": json.dumps(
                    {
                        "rendered": True,
                        "url": "about:blank",
                        "visible_text": "",
                    }
                ),
                "ok": True,
            },
            {
                "call_id": "rendered-source",
                "tool": "browser_interact",
                "arguments": json.dumps(
                    {
                        "action": "navigate",
                        "url": "https://example.test/field-service",
                    }
                ),
                "outcome": json.dumps(
                    {
                        "rendered": True,
                        "url": "https://example.test/field-service",
                        "title": "Field Service",
                        "visible_text": "Dispatch board and technician scheduling",
                    }
                ),
                "ok": True,
            },
        ]
    }

    focus = _focus_memory(task, expand_available=True)

    assert "search-discovery" not in focus
    assert "blank-snapshot" in focus.split("<inspections>", 1)[1].split(
        "</inspections>", 1
    )[0]
    sources = focus.split("<acquired_sources>", 1)[1].split(
        "</acquired_sources>", 1
    )[0]
    assert "rendered-source" in sources
    assert "https://example.test/field-service" in sources
    assert "blank-snapshot" not in sources


def test_nonresident_task_evidence_can_be_found_without_an_evidence_id() -> None:
    task = {
        "evidence_records": [
            {
                "evidence_id": "old-plan",
                "tool": "workspace_file",
                "arguments": '{"path":"/tmp/app/docs/plan.md"}',
                "result": '{"sha256":"old"}',
            },
            {
                "evidence_id": "auth-test",
                "tool": "shell",
                "arguments": '{"command":"pytest tests/test_auth.py"}',
                "result": '{"exit_code":0,"stdout":"4 passed"}',
            },
            {
                "evidence_id": "new-plan",
                "tool": "workspace_file",
                "arguments": '{"path":"/tmp/app/docs/plan.md"}',
                "result": '{"sha256":"new"}',
            },
        ]
    }

    plan = _search_task_evidence(task, "/tmp/app/docs/plan.md")
    auth = _search_task_evidence(task, "test_auth.py 4 passed")

    assert [item["evidence_id"] for item in plan] == ["new-plan", "old-plan"]
    assert [item["evidence_id"] for item in auth] == ["auth-test"]
    assert _search_task_evidence(task, "unrelated missing phrase") == []


def test_checkpoint_prose_is_removed_from_recurrent_and_durable_history() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "I have definitely created every file and passed every test.",
            "tool_calls": [
                {
                    "id": "checkpoint-1",
                    "function": {
                        "name": "task_checkpoint",
                        "arguments": {
                            "action": "progress",
                            "report": "All application files exist.",
                            "criteria_assessment": "Seven criteria are complete.",
                            "remaining_requirements": ["Only open the browser."],
                            "evidence_ids": ["real-1"],
                        },
                    },
                }
            ],
        }
    ]

    durable = _durable_task_messages(messages)

    assert "All application files exist" in json.dumps(messages)
    assert _sanitize_checkpoint_history(durable) == 1
    rendered = json.dumps(durable)
    assert "definitely created" not in rendered
    assert "All application files exist" not in rendered
    assert "Seven criteria" not in rendered
    assert "Only open the browser" not in rendered
    assert durable[0]["tool_calls"][0]["function"]["arguments"] == {
        "action": "progress",
        "evidence_ids": ["real-1"],
    }


def test_compaction_archives_model_visible_evidence_for_expansion() -> None:
    exact_passage = "exact page excerpt " * 80
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "fetch-1",
                    "function": {
                        "name": "web_fetch",
                        "arguments": {"url": "https://example.test/source"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "web_fetch",
            "tool_call_id": "fetch-1",
            "content": json.dumps(
                {
                    "content": exact_passage,
                    "provenance": {"source_url": "https://example.test/source"},
                }
            ),
        },
    ]

    records = _compaction_evidence_records(messages)

    assert len(records) == 1
    assert records[0]["evidence_id"] == "fetch-1"
    assert records[0]["tool"] == "web_fetch"
    assert exact_passage in records[0]["result"]
    assert len(records[0]["result_sha256"]) == 64


def test_discovery_after_checkpoint_does_not_erase_last_concrete_call() -> None:
    inspect = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "inspect-1",
                "function": {
                    "name": "shell",
                    "arguments": {"command": "find app -maxdepth 2 -type f"},
                },
            }
        ],
    }
    checkpoint = {
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
    }
    discovery = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "discovery-1",
                "function": {
                    "name": "tool_search",
                    "arguments": {"family": "shell"},
                },
            }
        ],
    }

    expected = _call_fingerprint(
        "shell", {"command": "find app -maxdepth 2 -type f"}
    )
    assert _latest_tool_fingerprint([inspect, checkpoint, discovery]) == expected


def test_unchanged_fetch_result_routes_back_to_discovery() -> None:
    fetched = {
        "content": "same page",
        "provenance": {"source_url": "https://a.test"},
    }

    first, last_digest, first_digest, repeated = _guard_repeated_unchanged_result(
        "web_fetch", {"url": "https://a.test"}, fetched, ""
    )
    second, retained_digest, second_digest, repeated_again = (
        _guard_repeated_unchanged_result(
            "web_fetch", {"url": "https://a.test"}, fetched, last_digest
        )
    )

    assert first == fetched
    assert repeated is False
    assert last_digest == first_digest
    assert second_digest == first_digest
    assert retained_digest == last_digest
    assert repeated_again is True
    assert second["error"] == "repeated_unchanged_result"
    assert second["disposition"] == "change_capability"
    assert second["alternative_tools"] == ["web_search", "browser_interact"]


def test_shell_failure_guard_ignores_cosmetic_argument_and_stream_changes() -> None:
    first_result = {
        "command": "cd /missing && python3 --version",
        "cwd": "/repo",
        "exit_code": 1,
        "stdout": "",
        "stderr": "/bin/bash: line 1: cd: /missing: No such file or directory\n",
        "timed_out": False,
    }
    retry_result = {
        "command": "cd /missing 2>&1 && python3 --version",
        "cwd": "/repo",
        "exit_code": 1,
        "stdout": "/bin/bash: line 1: cd: /missing: No such file or directory\n",
        "stderr": "",
        "timed_out": False,
        "stdin_bytes": 85,
    }

    _, last_digest, _, repeated = _guard_repeated_unchanged_result(
        "shell", {"command": first_result["command"]}, first_result, ""
    )
    guarded, retained, retry_digest, repeated_retry = _guard_repeated_unchanged_result(
        "shell",
        {"command": retry_result["command"], "timeout_seconds": 60},
        retry_result,
        last_digest,
    )

    assert repeated is False
    assert repeated_retry is True
    assert retry_digest == last_digest
    assert retained == last_digest
    assert guarded["error"] == "repeated_unchanged_result"
    assert guarded["disposition"] == "change_capability"
    assert guarded["alternative_tools"] == ["workspace_file"]


def test_failed_outcome_cannot_recur_after_an_intervening_failed_probe() -> None:
    missing_target = {
        "command": "cd /missing && ls",
        "cwd": "/repo",
        "exit_code": 1,
        "stdout": "",
        "stderr": "cd: /missing: No such file or directory",
        "timed_out": False,
    }
    unavailable_runtime = {
        "command": "missing-runtime --version",
        "cwd": "/repo",
        "exit_code": 127,
        "stdout": "",
        "stderr": "missing-runtime: command not found",
        "timed_out": False,
    }
    _first, first_last, first_digest, first_repeated = (
        _guard_repeated_unchanged_result(
            "shell", {"command": missing_target["command"]}, missing_target, ""
        )
    )
    _second, second_last, second_digest, second_repeated = (
        _guard_repeated_unchanged_result(
            "shell",
            {"command": unavailable_runtime["command"]},
            unavailable_runtime,
            first_last,
            {first_digest},
        )
    )
    guarded, retained, retry_digest, retry_repeated = (
        _guard_repeated_unchanged_result(
            "shell",
            {"command": missing_target["command"], "stdin": "irrelevant"},
            {**missing_target, "stdin_bytes": 10},
            second_last,
            {first_digest, second_digest},
        )
    )

    assert first_repeated is False
    assert second_repeated is False
    assert retry_repeated is True
    assert retry_digest == first_digest
    assert retained == second_last
    assert guarded["error"] == "repeated_unchanged_result"


def test_trailing_capability_failures_survive_discovery_actions() -> None:
    task = {
        "actions": [
            {"tool": "shell", "ok": False},
            {"tool": "tool_search", "ok": True},
            {"tool": "shell", "ok": False},
            {"tool": "web_search", "ok": True},
            {"tool": "shell", "ok": False},
        ]
    }

    assert _trailing_capability_failures(task) == {"shell": 3}


def test_capability_retry_budget_requires_a_different_action_space() -> None:
    failures: dict[str, int] = {}
    failed = {"exit_code": 2, "stderr": "target does not exist"}

    assert _apply_capability_retry_budget("shell", failed, failures) == failed
    assert _apply_capability_retry_budget("shell", failed, failures) == failed
    exhausted = _apply_capability_retry_budget("shell", failed, failures)

    assert exhausted["error"] == "capability_retry_exhausted"
    assert exhausted["disposition"] == "change_capability"
    assert exhausted["alternative_tools"] == ["workspace_file"]
    assert exhausted["task_blocked"] is False
    assert exhausted["last_result"] == failed


def test_successful_shell_observations_do_not_consume_capability_failures() -> None:
    failures: dict[str, int] = {}
    inspection = {
        "exit_code": 0,
        "stdout": "target absent",
        "task_progress": False,
        "evidence_authority": "inspection",
    }

    assert _apply_capability_retry_budget("shell", inspection, failures) == inspection
    assert _apply_capability_retry_budget("shell", inspection, failures) == inspection
    assert _apply_capability_retry_budget("shell", inspection, failures) == inspection
    assert failures == {}


def test_concrete_success_resets_capability_retry_budget() -> None:
    failures = {"shell": 3}
    success = {"action": "mkdir", "path": "/tmp/project", "ok": True}

    assert _apply_capability_retry_budget("workspace_file", success, failures) == success
    assert failures == {}
    assert _apply_capability_retry_budget(
        "shell", {"exit_code": 1}, failures
    ) == {"exit_code": 1}
    assert failures == {"shell": 1}


def test_workspace_inspection_guard_ignores_cosmetic_defaults() -> None:
    result = {
        "action": "list",
        "path": "/tmp/project",
        "depth": 2,
        "entries": [],
        "truncated": False,
    }
    _first, last_digest, _digest, repeated = _guard_repeated_unchanged_result(
        "workspace_file",
        {"action": "list", "path": "/tmp/project"},
        result,
        "",
    )
    assert repeated is False

    guarded, retained, retry_digest, repeated_retry = (
        _guard_repeated_unchanged_result(
            "workspace_file",
            {"action": "list", "path": "/tmp/project", "depth": 2},
            result,
            last_digest,
        )
    )

    assert repeated_retry is True
    assert retry_digest == last_digest
    assert retained == last_digest
    assert guarded["error"] == "repeated_unchanged_result"


def test_causal_result_boundary_survives_discovery_and_worker_restart() -> None:
    result = {
        "command": "cd /missing && ls",
        "cwd": "/repo",
        "exit_code": 1,
        "stdout": "",
        "stderr": "cd: /missing: No such file or directory",
        "timed_out": False,
    }
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "shell-1",
                    "function": {
                        "name": "shell",
                        "arguments": {"command": "cd /missing && ls"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "shell-1",
            "content": json.dumps(result),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "function": {
                        "name": "tool_search",
                        "arguments": {"family": "shell"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "tool_search",
            "tool_call_id": "search-1",
            "content": json.dumps({"available_tools": ["shell"]}),
        },
    ]

    expected = _guard_repeated_unchanged_result(
        "shell", {"command": "cd /missing && ls"}, result, ""
    )[1]
    assert _latest_external_result_digest(messages) == expected


def test_query_results_transition_without_pinning_the_completed_query_tool() -> None:
    assert _successor_tools(
        "web_search",
        {"alternative_tools": ["web_fetch", "browser_interact"]},
    ) == ["web_fetch", "browser_interact"]
    assert _successor_tools("web_fetch", {"content": "fetched"}) == []
    assert _successor_tools("get_portal_capabilities", {"output": ["text"]}) == []
    assert _successor_tools("shell", {"exit_code": 0}) == ["shell"]
    assert _successor_tools(
        "workspace_file", {"error": "duplicate_tool_call"}
    ) == []
    assert _successor_tools(
        "workspace_file", {"error": "repeated_unchanged_result"}
    ) == []
    assert _successor_tools(
        "browser_interact", {"error": "browser_navigation_error"}
    ) == []
    assert _successor_tools(
        "web_fetch", {"error": "undiscovered_url", "allowed_urls": []}
    ) == []
    assert _successor_tools(
        "web_fetch",
        {
            "error": "undiscovered_url",
            "allowed_urls": ["https://example.test/discovered"],
        },
    ) == []


def test_native_thinking_is_not_reenabled_when_a_query_tool_becomes_non_sticky() -> None:
    assert _structured_action_phase(
        [{"role": "user", "content": "Start."}], []
    ) is False
    assert _structured_action_phase(
        [
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_fetch"}}]},
            {
                "role": "tool",
                "tool_name": "web_fetch",
                "tool_call_id": "fetch-1",
                "content": '{"content": "page"}',
            },
        ],
        [],
    ) is True


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
        {"role": "user", "content": f"retained-{index} " + "x" * 5_000}
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

    messages.insert(
        -1,
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "real-evidence",
            "content": '{"exit_code": 0, "stdout": "verified"}',
        },
    )
    assert _freshest_evidence_id(messages) == "real-evidence"
    assert _checkpoint_available(messages) is True

    messages.append(
        {
            "role": "tool",
            "tool_name": "shell",
            "tool_call_id": "verify-1",
            "content": '{"exit_code": 0}',
        }
    )
    assert _compaction_available(messages) is True


def test_background_agent_compacts_deterministically_before_inference(
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
                        "content": json.dumps(
                            {"exit_code": 0, "stdout": "x" * 8_000}
                        ),
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
    persist_compaction = store.compact_context

    def persist_and_stop(*args, **kwargs):
        result = persist_compaction(*args, **kwargs)
        stop.set()
        return result

    store.compact_context = persist_and_stop  # type: ignore[method-assign]

    agent._execute(claimed)
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert current["compaction"]["reason"] == "automatic_context_limit"
    assert current["compaction"]["after"]["messages"] < len(messages)
    assert not any(
        action["tool"] == "task_compact" for action in current.get("actions", [])
    )


def test_background_agent_rejects_tool_absent_from_current_action_contract(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Continue through the current browser page.")
    seeded = store.claim_next("seed")
    assert seeded is not None
    store.checkpoint(
        task["task_id"],
        "seed",
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "browser task"},
            *(
                {"role": "user", "content": f"retained result {index}"}
                for index in range(20)
            ),
        ],
        active_tools=["browser_interact"],
        status="pending",
    )

    stop = threading.Event()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("an off-contract control call must remain local")
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
    record_action = agent._record_action

    def record_and_stop(*args, **kwargs) -> None:
        record_action(*args, **kwargs)
        stop.set()

    agent._record_action = record_and_stop  # type: ignore[method-assign]

    def hallucinate_compaction(payload: dict[str, object]) -> dict[str, object]:
        offered = {
            item["function"]["name"]  # type: ignore[index]
            for item in payload["tools"]  # type: ignore[union-attr]
        }
        assert "browser_interact" in offered
        assert "task_compact" not in offered
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "off-contract-compact",
                        "function": {
                            "name": "task_compact",
                            "arguments": {"reason": "long_task"},
                        },
                    }
                ],
            }
        }

    agent._chat = hallucinate_compaction  # type: ignore[method-assign]
    agent._execute(claimed)
    client.close()

    current = store.get(task["task_id"])
    assert current is not None
    assert "compaction" not in current
    assert current["actions"][-1]["tool"] == "task_compact"
    assert '"error": "tool_not_offered"' in current["actions"][-1]["outcome"]


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
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        requests.append(request.url.path)
        if request.url.path == "/api/chat/stream":
            chat_round += 1
            payload = json.loads(request.content)
            assert payload["tool_choice"] == "required"
            if chat_round == 1:
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
                                            "command": "printf ready > marker.txt",
                                            "intent": "mutate_filesystem",
                                            "mutation_paths": ["marker.txt"],
                                        },
                                    },
                                }
                            ],
                        }
                    },
                )
            if chat_round == 2:
                assert payload["think"] is False
                assert any(
                    "<task_self_check" in str(item.get("content") or "")
                    and "write-1" in str(item.get("content") or "")
                    for item in payload["messages"]
                )
                return _checkpoint_response(
                    "progress",
                    "I created marker.txt; verification remains.",
                    ["write-1"],
                )
            if chat_round == 3:
                assert "verified_phase_checkpoint" in payload["messages"][1]["content"]
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "verify-1",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": {
                                            "command": "test -f marker.txt",
                                            "intent": "verify",
                                        },
                                    },
                                }
                            ],
                        }
                    },
                )
            return _checkpoint_response(
                "complete",
                "I created marker.txt and verified the write succeeded.",
                ["verify-1"],
            )
        if request.url.path == "/api/tools/shell/call":
            arguments = json.loads(request.content)["arguments"]
            mutation = arguments.get("intent") == "mutate_filesystem"
            return httpx.Response(
                200,
                json={
                    "result": {
                        "stdout": "",
                        "stderr": "",
                        "exit_code": 0,
                        "evidence_authority": "mutation" if mutation else "verification",
                        "effect_receipt": {
                            "changed_paths": ["marker.txt"] if mutation else []
                        },
                    }
                },
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
        repair = command.startswith("sed ")
        arguments: dict[str, Any] = {
            "command": command,
            "intent": "mutate_filesystem" if repair else "verify",
        }
        if repair:
            arguments["mutation_paths"] = ["app.js"]
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
                                "arguments": arguments,
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
            repair = command.startswith("sed ")
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": (
                            "mutation" if repair else "verification"
                        ),
                        "effect_receipt": {
                            "changed_paths": ["app.js"] if repair else []
                        }
                    }
                },
            )
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
            family = payload["arguments"]["family"]
            selected = {"browser": "browser_interact", "web": "web_search"}[family]
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
                {"family": "web"},
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
                {"family": "browser"},
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


def test_background_web_fetch_must_follow_user_or_tool_evidence(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Research the requested public workflow source.")
    chat_round = 0
    external_fetches: list[str] = []
    grounded_url = "https://example.test/grounded-source"

    def model_call(call_id: str, name: str, arguments: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
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
            return httpx.Response(
                200,
                json={"result": {"available_tools": ["web_search"]}},
            )
        if request.url.path == "/api/tools/web_search/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "alternative_tools": ["web_fetch"],
                        "mode": "discover",
                        "results": [
                            {
                                "title": "Grounded source",
                                "url": grounded_url,
                            }
                        ],
                    }
                },
            )
        if request.url.path == "/api/tools/web_fetch/call":
            external_fetches.append(payload["arguments"]["url"])
            return httpx.Response(
                200,
                json={
                    "result": {
                        "url": grounded_url,
                        "content": "Verified workflow evidence.",
                    }
                },
            )

        chat_round += 1
        if chat_round == 1:
            return model_call(
                "discover-web",
                "tool_search",
                {"family": "web"},
            )
        if chat_round == 2:
            return model_call(
                "search-web",
                "web_search",
                {"query": "workflow source", "mode": "discover"},
            )
        if chat_round == 3:
            return model_call(
                "invented-fetch",
                "web_fetch",
                {"url": "https://invented.test/not-in-evidence"},
            )
        if chat_round == 4:
            tool_result = next(
                message
                for message in reversed(payload["messages"])
                if message.get("role") == "tool"
            )
            rejection = json.loads(tool_result["content"])
            assert rejection["error"] == "undiscovered_url"
            assert rejection["allowed_urls"] == [grounded_url]
            return model_call(
                "grounded-fetch",
                "web_fetch",
                {"url": grounded_url},
            )
        return _checkpoint_response(
            "complete",
            "I fetched and verified the discovered source.",
            ["grounded-fetch"],
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
    assert external_fetches == [grounded_url]
    invented = next(
        item for item in current["actions"] if item["call_id"] == "invented-fetch"
    )
    assert "undiscovered_url" in str(invented["outcome"])


def test_background_agent_discovers_before_exposing_tools_and_acts_without_runaway_thinking(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Open the rendered browser and inspect the page.")
    chat_round = 0
    residency_prepared = threading.Event()

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
            assert payload["portal_background_worker"] is True
            assert payload["portal_virtual_query"].startswith(
                "Advance and verify the pinned task. Objective: Open the rendered browser"
            )
            assert len(payload["portal_virtual_query"]) <= 1_200
            assert payload["messages"][1] == {
                "role": "user",
                "content": TASK_START_REQUEST,
            }
            assert payload["think"] is True
            assert payload["tool_choice"] == "required"
            assert tool_names == ["tool_search"]
            call_name = "tool_search"
            arguments = {"family": "browser"}
        elif chat_round == 2:
            assert payload["think"] is False
            assert tool_names == ["browser_interact"]
            call_name = "browser_interact"
            arguments = {"action": "navigate", "url": "http://example.test/"}
        else:
            assert payload["think"] is False
            assert tool_names == [
                "browser_interact",
                "task_checkpoint",
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
        prepare_action_residency=residency_prepared.set,
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
    assert residency_prepared.is_set()


def test_background_agent_executes_one_external_action_before_self_check(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Verify exactly the first marker.")
    chat_round = 0
    discovery_families: list[str] = []
    commands: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        payload = json.loads(request.content)
        if request.url.path == "/api/tools/tool_search/call":
            discovery_families.append(payload["arguments"]["family"])
            return httpx.Response(
                200,
                json={"result": {"available_tools": ["shell"], "results": []}},
            )
        if request.url.path == "/api/tools/shell/call":
            commands.append(payload["arguments"]["command"])
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "stdout": "first marker verified",
                        "evidence_authority": "verification",
                    }
                },
            )
        chat_round += 1
        if chat_round == 1:
            calls = [
                {
                    "id": "discover-first",
                    "function": {
                        "name": "tool_search",
                        "arguments": {"family": "shell"},
                    },
                },
                {
                    "id": "discover-stale",
                    "function": {
                        "name": "tool_search",
                        "arguments": {"family": "browser"},
                    },
                },
            ]
        elif chat_round == 2:
            calls = [
                {
                    "id": "write-first",
                    "function": {
                        "name": "shell",
                        "arguments": {
                            "command": "test -f first-marker",
                            "intent": "verify",
                        },
                    },
                },
                {
                    "id": "write-stale",
                    "function": {
                        "name": "shell",
                        "arguments": {
                            "command": "test -f stale-second-marker",
                            "intent": "verify",
                        },
                    },
                },
            ]
        else:
            return _checkpoint_response(
                "complete", "I verified the first marker.", ["write-first"]
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": calls,
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
    assert discovery_families == ["shell"]
    assert commands == ["test -f first-marker"]
    assert [item["tool"] for item in current["actions"]] == [
        "tool_search",
        "shell",
        "task_checkpoint",
    ]
    assert [item["call_id"] for item in current["actions"]] == [
        "discover-first",
        "write-first",
        "checkpoint-complete",
    ]


def test_background_agent_recovers_from_backend_outage_without_a_retry_storm(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Verify a marker after the backend recovers.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": "verification",
                    }
                },
            )
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
                                    "arguments": {
                                        "command": "test -f marker",
                                        "intent": "verify",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Verified the marker.", ["recovered-action"]
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
    task = store.create("Verify a small application marker.", "The marker exists.")
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
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": "verification",
                    }
                },
            )
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
                                    "arguments": {
                                        "command": "test -f marker",
                                        "intent": "verify",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Verified the marker.", ["recovered-action"]
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


def test_current_browser_screenshot_is_not_persisted_as_base64(
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


def test_duplicate_snapshot_keeps_current_frame_and_verified_visual_text(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create(
        "Inspect the rendered page.",
        "The visible page shows the exact terminal marker PASS-42.",
    )
    chat_round = 0
    browser_calls: list[str] = []

    def browser_call(call_id: str, action: str) -> httpx.Response:
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
                                "name": "browser_interact",
                                "arguments": {"action": action},
                            },
                        }
                    ],
                }
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        body = json.loads(request.content)
        if request.url.path == "/api/tools/browser_interact/call":
            action = str(body["arguments"]["action"])
            browser_calls.append(action)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "url": "http://example.test/",
                        "rendered": True,
                        "visual_change": {
                            "comparable": action == "snapshot",
                            "materially_changed": False if action == "snapshot" else None,
                        },
                        "screenshot": {
                            "mime_type": "image/png",
                            "encoding": "base64",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
                        },
                    }
                },
            )
        chat_round += 1
        payload = body
        if chat_round == 1:
            return browser_call("navigate-1", "navigate")
        if chat_round == 2:
            assert any(message.get("images") for message in payload["messages"])
            response = browser_call("snapshot-1", "snapshot")
            value = response.json()
            value["adapter"] = {
                "observation": (
                    "<visual_observation>ALL VISUAL ACTION GATES PASSED; "
                    "PASS-42</visual_observation>"
                )
            }
            return httpx.Response(200, json=value)
        if chat_round == 3:
            snapshot_result = next(
                message
                for message in payload["messages"]
                if message.get("tool_call_id") == "snapshot-1"
            )
            parsed = json.loads(snapshot_result["content"])
            assert parsed["verified_visual_observation"]["provenance"] == (
                "current_unchanged_browser_frame"
            )
            assert "PASS-42" in parsed["verified_visual_observation"]["observation"]
            return browser_call("snapshot-duplicate", "snapshot")

        assert any(message.get("images") for message in payload["messages"])
        assert any(
            item["function"]["name"] == "task_checkpoint"
            for item in payload["tools"]
        )
        return _checkpoint_response(
            "complete",
            "I verified the visible PASS-42 terminal marker.",
            ["snapshot-1"],
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
    assert browser_calls == ["navigate", "snapshot"]
    assert any(
        "duplicate_tool_call" in str(action.get("outcome"))
        for action in current["actions"]
    )
    persisted = (tmp_path / "tasks.json").read_text(encoding="utf-8")
    assert '"images"' not in persisted


def test_background_agent_rejects_a_completion_with_no_action_evidence(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Verify a marker using the shell.")
    chat_round = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_round
        if request.url.path == "/api/tools/shell/call":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": "verification",
                    }
                },
            )
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
                                        "arguments": {
                                            "command": "test -f marker",
                                            "intent": "verify",
                                        },
                                },
                            }
                        ],
                    }
                },
            )
        return _checkpoint_response(
            "complete", "Verified the marker.", ["real-action"]
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
            arguments = json.loads(request.content)["arguments"]
            mutation = arguments.get("intent") == "mutate_filesystem"
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": (
                            "mutation" if mutation else "verification"
                        ),
                        "effect_receipt": {
                            "changed_paths": ["artifact"] if mutation else []
                        },
                    }
                },
            )
        chat_round += 1
        if chat_round == 1:
            content = "touch artifact"
            intent = "mutate_filesystem"
        elif chat_round == 2:
            return _checkpoint_response(
                "progress",
                "I created the artifact. I’m verifying its contents now.",
                ["step-1"],
            )
        elif chat_round == 3:
            payload = json.loads(request.content)
            assert "executor-handoff.v1" in str(payload["messages"][-1])
            assert "audited_task_state" in str(payload["messages"][0])
            assert "I created the artifact" not in json.dumps(payload["messages"])
            assert "further required work remains" not in json.dumps(
                payload["messages"]
            )
            content = "test -f artifact"
            intent = "verify"
        else:
            return _checkpoint_response(
                "complete",
                "I finished the artifact and verified that the file exists.",
                ["step-1", "step-3"],
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
                                "arguments": {
                                    "command": content,
                                    "intent": intent,
                                    **(
                                        {"mutation_paths": ["artifact"]}
                                        if intent == "mutate_filesystem"
                                        else {}
                                    ),
                                },
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
        "Phase checkpoint retained against concrete evidence step-1; work remains."
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
            if command == "create artifact":
                receipt = {
                    "exit_code": 0,
                    "evidence_authority": "mutation",
                    "effect_receipt": {"changed_paths": ["artifact.txt"]},
                }
            elif command == "different successful verification":
                receipt = {
                    "exit_code": 0,
                    "evidence_authority": "verification",
                }
            else:
                receipt = {
                    "exit_code": 1,
                    "stderr": "failed",
                }
            return httpx.Response(
                200,
                json={"result": receipt},
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
            assert "I finished it." not in json.dumps(payload["messages"])
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
            mutation = payload["arguments"].get("intent") == "mutate_filesystem"
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": (
                            "mutation" if mutation else "verification"
                        ),
                        "effect_receipt": {
                            "changed_paths": [
                                payload["arguments"]["mutation_paths"][0]
                            ]
                            if mutation
                            else []
                        },
                    }
                },
            )
        chat_round += 1
        if chat_round == 1:
            content = "touch base"
            intent = "mutate_filesystem"
            mutation_path = "base"
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
            intent = "mutate_filesystem"
            mutation_path = "typescript"
        elif chat_round == 4:
            content = "test -f typescript"
            intent = "verify"
            mutation_path = ""
        else:
            return _checkpoint_response(
                "complete", "Applied the TypeScript update.", ["action-4"]
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
                                    "arguments": {
                                        "command": content,
                                        "intent": intent,
                                        **(
                                            {"mutation_paths": [mutation_path]}
                                            if mutation_path
                                            else {}
                                        ),
                                    },
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
    assert commands == ["touch base", "touch typescript", "test -f typescript"]
    assert any("redirected" in item.lower() for item in current["progress"])


def test_human_interjection_redirects_before_a_stale_action_and_then_resumes(
    tmp_path: Path,
) -> None:
    store = BackgroundTaskStore(tmp_path / "tasks.json")
    task = store.create("Verify the requested artifact.")
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
            return httpx.Response(
                200,
                json={
                    "result": {
                        "exit_code": 0,
                        "evidence_authority": "verification",
                    }
                },
            )
        chat_round += 1
        if chat_round == 1:
            inference_started.set()
            release_inference.wait(2)
            command = "test -f stale-artifact"
        elif chat_round == 2:
            assert any(
                "verify the redirected artifact" in str(item.get("content") or "")
                for item in payload["messages"]
            )
            assert not any(
                item.get("role") == "assistant" and item.get("tool_calls")
                for item in payload["messages"]
            )
            command = "test -f redirected-artifact"
        else:
            return _checkpoint_response(
                "complete",
                "I verified the redirected artifact.",
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
                                "arguments": {
                                    "command": command,
                                    "intent": "verify",
                                },
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
    store.add_guidance(task["task_id"], "Instead, verify the redirected artifact.")
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
    assert commands == ["test -f redirected-artifact"]
    assert not any("stale-artifact" in item for item in current["progress"])
