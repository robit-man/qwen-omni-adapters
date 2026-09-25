"""Shared bounded-context benchmark inference client helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import httpx


class EndpointResponder:
    """Small deterministic client for OpenAI- and Ollama-compatible endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        endpoint_style: str,
        model: str,
        api_key: str | None,
        timeout: float,
        max_tokens: int,
        think: bool,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        self.endpoint = endpoint
        self.endpoint_style = endpoint_style
        self.model = model
        self.max_tokens = max_tokens
        self.think = think
        self.last_metadata: dict[str, Any] = {}

    def _payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.endpoint_style == "openai":
            payload.update(
                {
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    # Never let one benchmark condition inherit another
                    # condition's slot KV state.
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": self.think},
                }
            )
        else:
            payload["think"] = self.think
            payload["options"] = {
                "temperature": 0.0,
                "num_predict": self.max_tokens,
            }
        return payload

    def __call__(self, prompt: str) -> str:
        response = self.client.post(self.endpoint, json=self._payload(prompt))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise RuntimeError(
                f"inference endpoint returned HTTP {response.status_code}: {detail}"
            ) from exc
        body = response.json()
        if self.endpoint_style == "openai":
            choice = body["choices"][0]
            self.last_metadata = {
                "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
                "finish_reason": choice.get("finish_reason"),
            }
            return str(choice["message"]["content"])
        self.last_metadata = {
            "usage": {
                "prompt_tokens": body.get("prompt_eval_count"),
                "completion_tokens": body.get("eval_count"),
            },
            "finish_reason": body.get("done_reason"),
        }
        return str(body["message"]["content"])

    def close(self) -> None:
        self.client.close()


def default_tokenize_endpoint(endpoint: str | None, endpoint_style: str) -> str | None:
    if not endpoint or endpoint_style != "openai":
        return None
    normalized = endpoint.rstrip("/")
    suffix = "/v1/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)] + "/tokenize"
    return None


def resident_physical_context(configured: int, state_file: Path) -> int:
    """Resolve a live worker window while retaining the CLI ceiling."""

    try:
        selected = int(state_file.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise ValueError(f"cannot read physical-context state file: {state_file}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid physical-context state file: {state_file}") from exc
    if selected < 4096:
        raise ValueError("resident physical context must be at least 4096 tokens")
    return min(configured, selected)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]
