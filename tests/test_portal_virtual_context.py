from __future__ import annotations

from pathlib import Path

from portal.virtual_context import SessionVirtualContext


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
    assert "quartz-991" in payload["messages"][0]["content"]
    assert "<current_query>" not in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"] == "What is the bus value?"
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
            {"role": "assistant", "content": ""},
            {
                "role": "tool",
                "tool_name": "inspect_status",
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
    assert "cobalt-ready" in followup.context.text
    assert "Find the actuator status." in followup.context.text
    assert followup.context.text.count("Find the actuator status.") == 1
    assert len(payload["messages"]) == 2
    assert "<current_query>" not in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"] == "Find the actuator status."
    assert followup.context.total_tokens <= followup.context.max_tokens


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
