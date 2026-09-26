"""An over-long prompt must be made to fit, not handed back as a 400.

llama.cpp refuses a prompt that will not fit rather than truncating it, so a
conversation that outgrows the worker's window stops answering entirely:
"request (4267 tokens) exceeds the available context size (4096 tokens)".
Every client would otherwise have to implement this for itself, and none of
them reliably knows the window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from adapter_server import (  # noqa: E402
    _post_language_with_context_retries,
    _shed_language_context,
    _stream_language_with_context_retries,
)


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


def test_context_fitting_has_no_fixed_conversation_length_ceiling() -> None:
    from types import SimpleNamespace

    from adapter_server import _estimated_prompt_tokens, _fit_language_context

    payload = {
        "messages": [
            {"role": "system", "content": "Keep the current objective."},
            *[
                {"role": "user", "content": f"old tool round {index} " * 20}
                for index in range(300)
            ],
            {"role": "user", "content": "the live task state"},
        ],
        "max_tokens": 256,
    }

    _fit_language_context(payload, SimpleNamespace(comprehension_context_tokens=4096))

    assert _estimated_prompt_tokens(payload) <= 4096 - 256
    assert payload["messages"][-1]["content"] == "the live task state"


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


def test_tool_schemas_are_shed_when_messages_cannot_free_enough() -> None:
    """Two dozen tools are ~3000 tokens of a 4096 window.

    Dropping messages cannot reach them, so a tool-using turn failed with
    "request (4179 tokens) exceeds the available context size" while the
    shedding loop reported nothing left to give up.
    """

    from adapter_server import _shed_language_context

    payload = {
        "messages": [
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "find me the news"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "subagent_delegate"}},
        ],
    }

    assert _shed_language_context(payload) is True
    # Shed from the back: the generally useful ones come first in the suite.
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["web_search"]


def test_shedding_preserves_the_capability_selected_by_tool_discovery() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "Open the site and use the visible form."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": {"family": "browser"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "tool_search",
                "content": '{"available_tools":["browser_interact"]}',
            },
        ],
        "tools": [
            {"type": "function", "function": {"name": "tool_search"}},
            {"type": "function", "function": {"name": "request_camera_view"}},
            {"type": "function", "function": {"name": "background_task"}},
            {"type": "function", "function": {"name": "browser_interact"}},
        ],
    }

    assert _shed_language_context(payload) is True
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert "browser_interact" in names
    assert "background_task" not in names


def test_exact_backend_overflow_retries_after_generic_context_shedding() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "message": "request (19759 tokens) exceeds the available context size (16384 tokens)",
                        "type": "exceed_context_size_error",
                    }
                },
            )
        return httpx.Response(200, json={"choices": []})

    payload = {
        "messages": [
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ]
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = _post_language_with_context_retries(
            client, "http://language.test", payload
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert [message["content"] for message in requests[1]["messages"]] == [
        "RULES",
        "old answer",
        "current",
    ]


def test_streaming_backend_overflow_is_read_then_retried() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                400,
                text="request (19759 tokens) exceeds the available context size (16384 tokens)",
            )
        return httpx.Response(200, text='data: {"choices":[]}\n\n')

    payload = {
        "messages": [
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ]
    }
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        _stream_language_with_context_retries(
            client, "http://language.test", payload
        ) as response,
    ):
        assert response.status_code == 200
        assert b"choices" in response.read()

    assert len(requests) == 2


def test_the_last_tool_is_kept_rather_than_leaving_none() -> None:
    from adapter_server import _shed_language_context

    payload = {
        "messages": [
            {"role": "system", "content": "RULES " * 400},
            {"role": "user", "content": "hello"},
        ],
        "tools": [{"type": "function", "function": {"name": "web_search"}}],
    }

    # Falls through to trimming the system message instead of emptying tools.
    assert _shed_language_context(payload) is True
    assert len(payload["tools"]) == 1


def test_large_shell_result_compacts_without_losing_current_user_intent() -> None:
    from types import SimpleNamespace

    from adapter_server import _estimated_prompt_tokens, _fit_language_context

    command = "printf x" + "y" * 12_000
    payload = {
        "messages": [
            {"role": "system", "content": "RULES"},
            {"role": "user", "content": "Run the command and tell me what happened."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": {"command": command}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "shell",
                "tool_call_id": "call-1",
                "content": "stdout-start\n" + "z" * 30_000 + "\nstdout-end",
            },
        ],
        "tools": [{"type": "function", "function": {"name": "tool_search"}}],
        "max_tokens": 160,
    }

    _fit_language_context(
        payload,
        SimpleNamespace(comprehension_context_tokens=4096),
    )

    assert _estimated_prompt_tokens(payload) <= 4096 - 160
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Run the command and tell me what happened."
        for message in payload["messages"]
    )
    assistant = next(
        message for message in payload["messages"] if message.get("role") == "assistant"
    )
    assert assistant["tool_calls"][0]["function"]["arguments"] == {
        "omitted": "arguments compacted after execution"
    }
    tool = next(message for message in payload["messages"] if message.get("role") == "tool")
    assert "Tool result compacted" in tool["content"]
    assert "stdout-start" in tool["content"]
    assert "stdout-end" in tool["content"]
