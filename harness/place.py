"""Where this machine is, for the same reason it needs to know when it is.

A model asked about the weather, the time somewhere, what is nearby, or what
"local" means has nothing to go on and will invent something. One lookup gives
it a city and a timezone to reason from.

Everything here is best-effort. The lookup is cached, refreshed rarely, and
never blocks a turn: a machine with no internet still has a microphone, and a
conversation must not wait on a geolocation service to say hello.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENDPOINT = "http://ip-api.com/json/"
# The address is only ever as current as the network, and the service asks not
# to be polled. Once an hour is far more often than a machine moves.
REFRESH_S = 3600.0
LOOKUP_TIMEOUT_S = 4.0


@dataclass(frozen=True)
class Place:
    """A coarse, IP-derived location. Never precise, and never claimed to be."""

    city: str = ""
    region: str = ""
    country: str = ""
    timezone: str = ""

    def __bool__(self) -> bool:
        return bool(self.city or self.region or self.country)

    def describe(self) -> str:
        """How a person would say where this is."""

        parts = [part for part in (self.city, self.region, self.country) if part]
        return ", ".join(parts)


class PlaceLookup:
    """Remembers where this machine is, and re-checks it rarely.

    The first lookup happens on a background thread at startup so the first
    spoken turn does not pay for it, and a failure simply leaves the location
    unknown rather than raising.
    """

    def __init__(
        self,
        endpoint: str = ENDPOINT,
        *,
        refresh_s: float = REFRESH_S,
        timeout_s: float = LOOKUP_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.refresh_s = refresh_s
        self.timeout_s = timeout_s
        self._place = Place()
        self._checked_at = 0.0
        self._lock = threading.Lock()
        self._warned = False

    @property
    def place(self) -> Place:
        """The last known location, without waiting for a new lookup."""

        with self._lock:
            stale = time.monotonic() - self._checked_at > self.refresh_s
            known = self._place
        if stale:
            self.refresh_async()
        return known

    def refresh_async(self) -> None:
        threading.Thread(
            target=self.refresh, name="omni-place", daemon=True
        ).start()

    def refresh(self) -> Place:
        """Look the location up now. Returns the last known value on failure."""

        with self._lock:
            # Claim the slot before the request, so a slow or failing lookup
            # cannot be retried by every turn that asks in the meantime.
            self._checked_at = time.monotonic()
        try:
            request = urllib.request.Request(
                self.endpoint,
                headers={"User-Agent": "qwen-omni-adapters-harness"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except Exception as error:  # noqa: BLE001 - offline is a normal state
            if not self._warned:
                logger.info("location is unknown (%s); continuing without it", error)
                self._warned = True
            return self._place

        if not isinstance(payload, dict) or payload.get("status") != "success":
            return self._place

        found = Place(
            city=str(payload.get("city") or "").strip(),
            region=str(payload.get("regionName") or "").strip(),
            country=str(payload.get("country") or "").strip(),
            timezone=str(payload.get("timezone") or "").strip(),
        )
        with self._lock:
            self._place = found
        self._warned = False
        if found:
            logger.info("location: %s (%s)", found.describe(), found.timezone or "no tz")
        return found
