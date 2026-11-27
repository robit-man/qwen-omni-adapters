"""Bound an HTTP/1.1 stream to the lifetime of its downstream request."""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx


def post_until_disconnect(
    url: str,
    payload: Mapping[str, Any],
    timeout: httpx.Timeout,
    disconnected: Callable[[], bool],
) -> httpx.Response | None:
    """Cancel a local, non-streaming request before its response headers arrive."""

    async def post() -> httpx.Response | None:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            request = asyncio.create_task(client.post(url, json=payload))
            try:
                while not disconnected():
                    done, _ = await asyncio.wait({request}, timeout=0.05)
                    if request in done:
                        return request.result()
                return None
            finally:
                if not request.done():
                    request.cancel()
                # Wait for HTTPX to close the connection on cancellation.
                await asyncio.gather(request, return_exceptions=True)

    if disconnected():
        return None
    return asyncio.run(post())


def abort_response(response: httpx.Response) -> None:
    if response.is_closed or response.http_version != "HTTP/1.1":
        return
    stream = response.extensions.get("network_stream")
    connection = stream.get_extra_info("socket") if stream is not None else None
    if isinstance(connection, socket.socket):
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def watch_for_disconnect(
    disconnected: Callable[[], bool] | None,
    upstream: Callable[[], httpx.Response],
) -> threading.Event:
    finished = threading.Event()
    if disconnected is None:
        return finished

    def watch() -> None:
        while not finished.wait(0.05):
            if disconnected():
                abort_response(upstream())
                return

    threading.Thread(target=watch, name="omni-stream-disconnect", daemon=True).start()
    return finished


@contextmanager
def abort_on_disconnect(
    disconnected: Callable[[], bool] | None,
    upstream: Callable[[], httpx.Response],
) -> Iterator[None]:
    finished = watch_for_disconnect(disconnected, upstream)
    try:
        yield
    finally:
        finished.set()
