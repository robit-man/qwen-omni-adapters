"""Exact model-tokenizer adapter tests."""

from __future__ import annotations

import json

import httpx

from qwen_omni_adapters.virtual_memory import LlamaCppTokenCounter


def test_llama_cpp_counter_uses_tokenize_endpoint_and_caches() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"tokens": [11, 12, 13]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    counter = LlamaCppTokenCounter(
        "http://llama/tokenize", client=client, cache_entries=64
    )

    assert counter("identifier:::without spaces") == 3
    assert counter("identifier:::without spaces") == 3
    assert requests == [
        {
            "content": "identifier:::without spaces",
            "add_special": False,
            "with_pieces": False,
        }
    ]
    counter.close()


def test_llama_cpp_counter_rejects_malformed_response() -> None:
    counter = LlamaCppTokenCounter(
        "http://llama/tokenize",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={"count": 4})
            )
        ),
    )

    try:
        counter("value")
    except ValueError as exc:
        assert "no token list" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("malformed tokenizer response was accepted")
    finally:
        counter.close()
