"""Exact tokenizer adapters for enforcing the physical working-set ceiling."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import httpx


class LlamaCppTokenCounter:
    """Count with the active llama.cpp model's `/tokenize` endpoint."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 15.0,
        cache_entries: int = 4096,
        client: httpx.Client | None = None,
    ) -> None:
        endpoint = str(url or "").strip()
        if not endpoint:
            raise ValueError("tokenizer URL is required")
        self.url = endpoint
        self.client = client or httpx.Client(timeout=timeout)
        self.cache_entries = max(64, cache_entries)
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.RLock()

    def __call__(self, text: str) -> int:
        value = str(text or "")
        with self._lock:
            cached = self._cache.get(value)
            if cached is not None:
                self._cache.move_to_end(value)
                return cached
        response = self.client.post(
            self.url,
            json={"content": value, "add_special": False, "with_pieces": False},
        )
        response.raise_for_status()
        body: Any = response.json()
        tokens = body.get("tokens") if isinstance(body, dict) else None
        if not isinstance(tokens, list):
            raise ValueError("tokenizer endpoint returned no token list")
        count = len(tokens)
        with self._lock:
            self._cache[value] = count
            self._cache.move_to_end(value)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return count

    def close(self) -> None:
        self.client.close()
