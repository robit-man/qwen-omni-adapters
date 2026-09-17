"""An over-long prompt must be made to fit, not handed back as a 400.

llama.cpp refuses a prompt that will not fit rather than truncating it, so a
conversation that outgrows the worker's window stops answering entirely:
"request (4267 tokens) exceeds the available context size (4096 tokens)".
Every client would otherwise have to implement this for itself, and none of
them reliably knows the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from adapter_server import _shed_language_context  # noqa: E402


def conversation() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "RULES\n" + "context line\n" * 400},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "live question"},
        ]
    }


def test_the_oldest_exchange_goes_first() -> None:
    payload = conversation()
    assert _shed_language_context(payload) is True
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "assistant",
        "user",
    ]


def test_the_instructions_and_the_live_question_are_never_dropped() -> None:
    payload = conversation()
    for _ in range(4):
        if not _shed_language_context(payload):
            break
    roles = [message["role"] for message in payload["messages"]]
    assert roles[0] == "system"
    assert payload["messages"][-1]["content"] == "live question"


def test_the_system_tail_is_cut_once_history_is_exhausted() -> None:
    """Its head carries the instructions; its tail carries accumulated context."""

    payload = conversation()
    while len(payload["messages"]) > 2:
        assert _shed_language_context(payload) is True

    before = len(payload["messages"][0]["content"])
    assert _shed_language_context(payload) is True
    after = payload["messages"][0]["content"]

    assert len(after) < before
    assert after.startswith("RULES")
    assert after.rstrip().endswith("[Earlier context omitted to fit the context window.]")


def test_a_prompt_with_nothing_left_to_give_up_says_so() -> None:
    """Otherwise the retry loop would spin against a prompt that cannot fit."""

    payload = {
        "messages": [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "hi"},
        ]
    }
    assert _shed_language_context(payload) is False


def test_a_long_conversation_is_fitted_before_it_is_sent() -> None:
    """No extra round trip: the prompt is sized before the request goes out."""

    from types import SimpleNamespace

    from adapter_server import _estimated_prompt_tokens, _fit_language_context

    config = SimpleNamespace(comprehension_context_tokens=4096)

    payload = {
        "messages": [
            {"role": "system", "content": "RULES\n" + "world context line\n" * 900},
            *[
                {"role": "user", "content": "a question " * 120}
                for _ in range(12)
            ],
            {"role": "user", "content": "the live question"},
        ],
        "max_tokens": 160,
    }
    assert _estimated_prompt_tokens(payload) > 4096

    _fit_language_context(payload, config)

    assert _estimated_prompt_tokens(payload) <= 4096 - 160
    assert payload["messages"][-1]["content"] == "the live question"
    assert payload["messages"][0]["role"] == "system"


def test_tool_schemas_are_counted_toward_the_window() -> None:
    """The template renders them into the prompt, so they take up room."""

    from adapter_server import _estimated_prompt_tokens

    messages = [{"role": "user", "content": "hi"}]
    bare = _estimated_prompt_tokens({"messages": messages})
    withtools = _estimated_prompt_tokens(
        {
            "messages": messages,
            "tools": [{"type": "function", "function": {"name": "t", "description": "d" * 3000}}],
        }
    )
    assert withtools > bare + 500
