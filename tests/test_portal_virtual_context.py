from __future__ import annotations

import json
from pathlib import Path

from portal.virtual_context import SessionVirtualContext, _adaptive_output_headroom


def test_output_headroom_tracks_the_resident_context_tier() -> None:
    assert _adaptive_output_headroom(4_096) == 768
    assert _adaptive_output_headroom(8_192) == 1_024
    assert _adaptive_output_headroom(16_384) == 2_048
    assert _adaptive_output_headroom(65_536) == 2_384


def test_new_session_uses_adaptive_headroom_after_a_live_downshift(
    tmp_path: Path,
) -> None:
    state = tmp_path / "resident-context"
    state.write_text("4096\n", encoding="utf-8")
    manager = SessionVirtualContext(
        tmp_path / "virtual",
        mode="shadow",
        physical_context_tokens=16_384,
        physical_context_state_file=state,
    )

    session = manager._session("downshifted")

    assert session.engine.packer.budget.max_tokens == 4_096
    assert session.engine.packer.budget.output_headroom == 768


def test_status_does_not_create_an_empty_session_database(tmp_path: Path) -> None:
    root = tmp_path / "virtual"
    manager = SessionVirtualContext(root, mode="shadow")

    status = manager.stats("unused-session")

    assert status["documents"] == 0
    assert not root.exists()


def test_shadow_mode_persists_lossless_turns_without_rewriting_payload(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="shadow")
    messages = [
        {"role": "user", "content": "The actuator code is cobalt-771."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What was the actuator code?"},
    ]
    payload = {"messages": list(messages)}

    assert manager.observe_messages("session-one", messages) == 3
    prepared = manager.prepare(
        "session-one", messages, system_contract="Answer from exact evidence."
    )
    manager.apply_active(payload, prepared)

    assert payload["messages"] == messages
    assert "cobalt-771" in prepared.context.text
    assert manager.stats("session-one")["documents"] == 3
    recurrent = [
        item for item in prepared.context.items if item.category == "recurrent_memory"
    ]
    assert len(recurrent) == 1
    assert recurrent[0].provenance
    assert 'authority="derived_unverified"' in recurrent[0].text
    assert any(event["operation"] == "MERGE" for event in prepared.context.trace)


def test_background_index_excludes_unverified_assistant_narration(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="shadow")
    messages = [
        {"role": "user", "content": "Build the empty current workspace."},
        {
            "role": "assistant",
            "content": "Imaginary artifact mercury-884 already passed every test.",
        },
        {
            "role": "tool",
            "content": '{"path":"/tmp/current","entries":[]}',
        },
    ]

    assert (
        manager.observe_messages(
            "background-task",
            messages,
            include_assistant=False,
        )
        == 2
    )
    session = manager._session("background-task")
    assert session.store.exact_search("mercury-884") == []
    assert session.store.exact_search("/tmp/current")


def test_user_constraint_is_promoted_and_deterministically_pinned(tmp_path: Path) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="shadow")
    messages = [
        {
            "role": "user",
            "content": "MUST NOT create a new asset for every edit.",
        },
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Apply the next edit."},
    ]
    manager.observe_messages("session-constraint", messages)

    prepared = manager.prepare(
        "session-constraint", messages, system_contract="Follow active constraints."
    )

    assert "MUST NOT create a new asset for every edit." in prepared.context.text
    pinned = [item for item in prepared.context.items if item.category == "constraint"]
    assert len(pinned) == 1
    assert pinned[0].pinned is True


def test_active_mode_replaces_history_with_bounded_pack_and_current_media(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [
        {"role": "user", "content": "The bus value is quartz-991."},
        {"role": "assistant", "content": "Noted."},
        {
            "role": "user",
            "content": "What is the bus value?",
            "images": [{"data": "current-frame", "mime_type": "image/jpeg"}],
        },
    ]
    manager.observe_messages("session-two", messages)
    prepared = manager.prepare(
        "session-two", messages, system_contract="Answer from exact evidence."
    )
    payload = {"messages": [{"role": "system", "content": "old"}, *messages]}

    manager.apply_active(payload, prepared)

    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "Answer from exact evidence."
    assert "quartz-991" in payload["messages"][1]["content"]
    assert "<current_query>" in payload["messages"][1]["content"]
    assert payload["messages"][1]["content"].count("What is the bus value?") == 1
    assert payload["messages"][1]["images"][0]["data"] == "current-frame"
    assert prepared.context.total_tokens <= prepared.context.max_tokens


def test_current_user_question_is_stored_but_never_self_replayed_as_evidence(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [{"role": "user", "content": "What is the unrecorded bus value?"}]

    manager.observe_messages("session-current", messages)
    prepared = manager.prepare(
        "session-current",
        messages,
        system_contract="Use exact historical evidence when it exists.",
    )

    assert manager.stats("session-current")["documents"] == 1
    assert prepared.context.evidence_chunk_ids == ()
    assert prepared.context.text.count("What is the unrecorded bus value?") == 1
    assert prepared.answer_allowed is False
    assert any(
        event["operation"] == "EVICT"
        and event["detail"].get("reason") == "current_query_is_not_evidence"
        for event in prepared.controller.trace
    )


def test_explicit_task_query_overrides_a_retained_checkpoint_as_retrieval_intent(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [
        {"role": "user", "content": "Begin the pinned task."},
        {
            "role": "user",
            "content": (
                "<retained_checkpoint>Old browser failures and environment probes."
                "</retained_checkpoint>"
            ),
        },
    ]
    query = "Advance and verify the pinned task. Objective: Build the care app."
    manager.observe_messages("background-task", messages)

    prepared = manager.prepare(
        "background-task",
        messages,
        system_contract="<current_task>Build the care app.</current_task>",
        query_override=query,
    )
    payload = {"messages": list(messages)}
    manager.apply_active(payload, prepared)

    assert prepared.controller.queries[0] == query
    assert prepared.context.text.count(query) == 1
    assert f"<current_query>\n{query}\n</current_query>" in payload["messages"][1][
        "content"
    ]


def test_active_mode_replaces_inline_text_but_keeps_current_media_parts(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in the current frame?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
            ],
        }
    ]
    manager.observe_messages("session-inline-media", messages)
    prepared = manager.prepare(
        "session-inline-media",
        messages,
        system_contract="Use only current visual evidence.",
    )
    payload = {"messages": list(messages)}

    manager.apply_active(payload, prepared)

    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"].count("What is in the current frame?") == 1
    assert content[1] == messages[0]["content"][1]


def test_active_mode_recalls_early_fact_beyond_physical_history_window(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [
        {"role": "user", "content": "The immutable actuator code is quartz-991."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "assistant", "content": "irrelevant padding " * 12_000},
        {
            "role": "user",
            "content": "What is the immutable actuator code?",
            "images": [{"data": "current-frame", "mime_type": "image/jpeg"}],
        },
    ]
    manager.observe_messages("session-long", messages)

    prepared = manager.prepare(
        "session-long",
        messages,
        system_contract="Answer from exact historical evidence.",
    )
    payload = {"messages": [{"role": "system", "content": "old"}, *messages]}
    manager.apply_active(payload, prepared)

    assert prepared.answer_allowed is True
    assert "quartz-991" in prepared.context.text
    assert prepared.context.evidence_chunk_ids
    assert prepared.context.total_tokens <= prepared.context.max_tokens
    assert prepared.context.text.count("irrelevant padding") < 100
    assert any(
        event["operation"] == "final_evidence_sufficiency"
        and event["detail"]["sufficient"] is True
        for event in prepared.context.trace
    )
    assert len(payload["messages"]) == 2
    assert payload["messages"][1]["images"][0]["data"] == "current-frame"


def test_active_tool_followup_is_repacked_with_result_and_original_query(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    messages = [{"role": "user", "content": "Find the actuator status."}]
    manager.observe_messages("session-tool", messages)
    initial = manager.prepare(
        "session-tool", messages, system_contract="Use current tool evidence."
    )
    payload = {
        "messages": [
            {"role": "system", "content": initial.context.text},
            {"role": "user", "content": "Act on <current_query>."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "status-call",
                        "type": "function",
                        "function": {
                            "name": "inspect_status",
                            "arguments": {},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "inspect_status",
                "tool_call_id": "status-call",
                "content": '{"actuator_status":"cobalt-ready"}',
            },
        ]
    }

    followup = manager.repack_followup(
        "session-tool",
        payload,
        query="Find the actuator status.",
        system_contract="Use current tool evidence.",
    )

    assert followup is not None
    assert "cobalt-ready" not in followup.context.text
    assert "Find the actuator status." in followup.context.text
    assert followup.context.text.count("Find the actuator status.") == 1
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert "<current_query>" not in payload["messages"][0]["content"]
    assert "cobalt-ready" not in payload["messages"][1]["content"]
    assert "cobalt-ready" in payload["messages"][3]["content"]
    assert payload["messages"][2]["tool_calls"][0]["id"] == "status-call"
    assert payload["messages"][3]["tool_call_id"] == "status-call"
    assert "Act on <current_query>." not in payload["messages"][1]["content"]
    assert payload["messages"][1]["content"].count("Find the actuator status.") == 1
    assert json.dumps(payload["messages"]).count("cobalt-ready") == 1
    assert followup.context.total_tokens <= followup.context.max_tokens


def test_tool_followup_media_is_not_charged_as_base64_text(tmp_path: Path) -> None:
    manager = SessionVirtualContext(tmp_path / "virtual", mode="active")
    query = "Inspect the current desktop state."
    messages = [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "desktop-snapshot",
                    "type": "function",
                    "function": {"name": "gui_interact", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "gui_interact",
            "tool_call_id": "desktop-snapshot",
            "content": '{"rendered":true,"screenshot":{"data":"attached"}}',
            "images": [
                {
                    "mime_type": "image/png",
                    "encoding": "base64",
                    "data": "A" * 500_000,
                }
            ],
        },
    ]
    payload = {"messages": messages, "tools": []}

    followup = manager.repack_followup(
        "session-media-tool",
        payload,
        query=query,
        system_contract="Use the newest current-frame evidence.",
    )

    assert followup is not None
    assert followup.context.total_tokens <= followup.context.max_tokens
    assert payload["messages"][-1]["images"][0]["data"] == "A" * 500_000
    assert followup.context.token_usage["request_envelope"] < 5000


def test_live_tool_envelope_is_reserved_from_physical_context(tmp_path: Path) -> None:
    manager = SessionVirtualContext(
        tmp_path / "virtual", mode="shadow", token_counter=lambda value: len(value.split())
    )
    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_motor",
                    "description": "Inspect the motor state with a verbose schema.",
                },
            }
        ],
        "tool_choice": "auto",
    }

    reserve = manager.request_envelope_tokens(payload)

    assert reserve > 96


def test_live_resident_context_state_resizes_existing_session_budget(
    tmp_path: Path,
) -> None:
    state = tmp_path / "comprehension-context-tokens"
    state.write_text("8192\n", encoding="utf-8")
    manager = SessionVirtualContext(
        tmp_path / "virtual",
        mode="active",
        physical_context_tokens=16_384,
        physical_context_state_file=state,
    )
    messages = [
        {"role": "user", "content": "The bus token is heliotrope-7319."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "assistant", "content": "distractor " * 10_000},
        {"role": "user", "content": "What is the bus token?"},
    ]
    manager.observe_messages("dynamic-window", messages)

    first = manager.prepare(
        "dynamic-window", messages, system_contract="Answer from exact evidence."
    )
    state.write_text("4096\n", encoding="utf-8")
    second = manager.prepare(
        "dynamic-window", messages, system_contract="Answer from exact evidence."
    )

    assert first.context.max_tokens == 8192
    assert first.context.total_tokens <= 8192
    assert second.context.max_tokens == 4096
    assert second.context.total_tokens <= 4096
    assert manager.stats("dynamic-window")["physical_context_tokens"] == 4096
    assert manager.stats("dynamic-window")["configured_physical_context_tokens"] == 16_384


def test_missing_resident_context_state_fails_closed_to_supported_floor(
    tmp_path: Path,
) -> None:
    manager = SessionVirtualContext(
        tmp_path / "virtual",
        mode="shadow",
        physical_context_tokens=16_384,
        physical_context_state_file=tmp_path / "not-written-yet",
    )

    status = manager.stats("unused")

    assert status["physical_context_tokens"] == 4096
    assert status["physical_context_source"] == "resident_state_fallback"


def test_document_text_is_indexed_and_trash_destroys_the_session_corpus(
    tmp_path: Path,
) -> None:
    root = tmp_path / "virtual"
    manager = SessionVirtualContext(root, mode="shadow")
    manager.observe_documents(
        "session-three",
        [
            {
                "id": "document-1",
                "digest": "a" * 64,
                "name": "controller.md",
                "mime_type": "text/markdown",
                "text": "# Controller\n\nExact phase offset: 17.25 degrees.",
            }
        ],
    )
    database = next(root.glob("*.sqlite3"))
    prepared = manager.prepare(
        "session-three",
        [{"role": "user", "content": "What is the exact phase offset?"}],
        system_contract="Use evidence.",
    )

    assert "17.25 degrees" in prepared.context.text
    assert database.is_file()

    manager.clear("session-three")

    assert not database.exists()
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
