"""Bound an HTTP/1.1 stream to the lifetime of its downstream request."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx


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
