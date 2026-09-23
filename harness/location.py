"""Privacy-bounded approximate location for the local voice client."""

from __future__ import annotations

import html
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from portal.tools import _run_local_browser

logger = logging.getLogger(__name__)

LOCATION_ENDPOINT = "https://ipwho.is/"
LOCATION_TTL_S = 240.0
LOCATION_RETRY_S = 30.0


def _text(value: Any, maximum: int = 120) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:maximum]


def _coordinate(value: Any, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return round(number, 2)


def _json_from_browser_dom(dom: str) -> Mapping[str, Any]:
    text = re.sub(r"<[^>]+>", " ", dom)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("location browser returned no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, Mapping):
        raise ValueError("location browser returned a non-object")
    return value


def sanitize_browser_location(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Allowlist approximate geography and discard raw IP/provider metadata."""

    if value.get("success") is False:
        return None
    timezone = value.get("timezone")
    timezone = timezone if isinstance(timezone, Mapping) else {}
    location: dict[str, Any] = {
        "city": _text(value.get("city")),
        "region": _text(value.get("region")),
        "region_code": _text(value.get("region_code"), 16),
        "country": _text(value.get("country")),
        "country_code": _text(value.get("country_code"), 8),
        "continent": _text(value.get("continent")),
        "continent_code": _text(value.get("continent_code"), 8),
        "latitude": _coordinate(value.get("latitude"), -90.0, 90.0),
        "longitude": _coordinate(value.get("longitude"), -180.0, 180.0),
        "timezone": {
            "id": _text(timezone.get("id")),
            "abbreviation": _text(timezone.get("abbreviation"), 24),
            "utc_offset": _text(timezone.get("utc"), 16),
        },
    }
    if not any(
        location.get(key) not in (None, "", {})
        for key in ("city", "region", "country", "latitude", "longitude")
    ):
        return None
    return location


class BrowserLocationProvider:
    """Refresh location in Chromium without delaying a spoken turn."""

    def __init__(
        self,
        runner: Callable[[str, float], str] | None = None,
        *,
        ttl_s: float = LOCATION_TTL_S,
        retry_s: float = LOCATION_RETRY_S,
    ) -> None:
        self._runner = runner or _run_local_browser
        self._ttl_s = max(30.0, float(ttl_s))
        self._retry_s = max(5.0, float(retry_s))
        self._lock = threading.Lock()
        self._value: dict[str, Any] | None = None
        self._updated_at = 0.0
        self._retry_at = 0.0
        self._thread: threading.Thread | None = None

    def _refresh(self) -> None:
        value: dict[str, Any] | None = None
        try:
            dom = self._runner(LOCATION_ENDPOINT, 10.0)
            value = sanitize_browser_location(_json_from_browser_dom(dom))
        except Exception as error:  # noqa: BLE001 - location is optional evidence
            logger.info("browser location refresh unavailable: %s", error)
        now = time.monotonic()
        with self._lock:
            if value is not None:
                self._value = value
                self._updated_at = now
            self._retry_at = now + (self._ttl_s if value is not None else self._retry_s)
            self._thread = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or time.monotonic() < self._retry_at:
                return
            self._thread = threading.Thread(
                target=self._refresh,
                name="omni-browser-location",
                daemon=True,
            )
            self._thread.start()

    def get(self) -> dict[str, Any] | None:
        self.start()
        with self._lock:
            return dict(self._value) if self._value is not None else None

    def refresh_now(self) -> dict[str, Any] | None:
        """Synchronously refresh for diagnostics and deterministic tests."""

        self._refresh()
        with self._lock:
            return dict(self._value) if self._value is not None else None
