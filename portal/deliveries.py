"""Session-scoped, authenticated file delivery for the phone portal.

Files are copied into an immutable staging area before they are exposed.  The
portal never serves an arbitrary path supplied in a URL, and a delivery ID is
valid only for the browser session that created it.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import secrets
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileDeliveryError(ValueError):
    """A safe file-delivery validation or lookup failure."""


@dataclass
class _Delivery:
    delivery_id: str
    session_key: str
    path: Path
    download_name: str
    media_type: str
    size: int
    sha256: str
    expires_monotonic: float
    expires_at: str
    timer: threading.Timer | None = None


class SessionFileDeliveryStore:
    """Stage bounded files and resolve them only for their owning session."""

    def __init__(
        self,
        directory: Path,
        *,
        allowed_roots: tuple[Path, ...],
        ttl_s: float = 300.0,
        max_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if not allowed_roots:
            raise ValueError("file delivery requires at least one allowed root")
        self.directory = directory.expanduser().resolve(strict=False)
        self.allowed_roots = tuple(
            root.expanduser().resolve(strict=False) for root in allowed_roots
        )
        self.ttl_s = max(1.0, float(ttl_s))
        self.max_bytes = max(1, int(max_bytes))
        self._lock = threading.Lock()
        self._deliveries: dict[str, _Delivery] = {}

    @staticmethod
    def _session_key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _allowed_source(self, source: Path) -> bool:
        return any(source.is_relative_to(root) for root in self.allowed_roots)

    @staticmethod
    def _download_name(source: Path, label: Any) -> str:
        value = str(label or source.name).strip()
        if (
            not value
            or value in {".", ".."}
            or len(value.encode("utf-8")) > 255
            or Path(value).name != value
            or "\x00" in value
        ):
            raise FileDeliveryError("download_name must be one safe filename")
        return value

    def _remove(self, delivery_id: str) -> None:
        record: _Delivery | None
        with self._lock:
            record = self._deliveries.pop(delivery_id, None)
        if record is None:
            return
        if record.timer is not None and record.timer is not threading.current_thread():
            record.timer.cancel()
        shutil.rmtree(record.path.parent, ignore_errors=True)
        try:
            record.path.parent.parent.rmdir()
        except OSError:
            pass

    def _expire(self, delivery_id: str) -> None:
        self._remove(delivery_id)

    def stage(self, session_id: str, source_path: Any, download_name: Any = None) -> dict[str, Any]:
        raw_path = str(source_path or "").strip()
        if not raw_path or len(raw_path) > 4096:
            raise FileDeliveryError("path must be a non-empty host path")
        try:
            source = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileDeliveryError("delivery source does not exist") from exc
        if not self._allowed_source(source):
            raise FileDeliveryError("delivery source is outside the configured roots")
        name = self._download_name(source, download_name)

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            source_fd = os.open(source, flags)
        except OSError as exc:
            raise FileDeliveryError("delivery source could not be opened safely") from exc
        delivery_id = secrets.token_urlsafe(18)
        session_key = self._session_key(session_id)
        delivery_directory = self.directory / session_key / delivery_id
        target = delivery_directory / "payload"
        digest = hashlib.sha256()
        copied = 0
        try:
            info = os.fstat(source_fd)
            if not stat.S_ISREG(info.st_mode):
                raise FileDeliveryError("delivery source must be a regular file")
            if info.st_size > self.max_bytes:
                raise FileDeliveryError(
                    f"delivery source exceeds {self.max_bytes} bytes"
                )
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.directory.chmod(0o700)
            session_directory = delivery_directory.parent
            session_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            session_directory.chmod(0o700)
            delivery_directory.mkdir(mode=0o700)
            delivery_directory.chmod(0o700)
            with os.fdopen(source_fd, "rb", closefd=True) as source_stream:
                source_fd = -1
                with target.open("xb") as target_stream:
                    while True:
                        chunk = source_stream.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > self.max_bytes:
                            raise FileDeliveryError(
                                f"delivery source exceeds {self.max_bytes} bytes"
                            )
                        digest.update(chunk)
                        target_stream.write(chunk)
            target.chmod(0o400)
        except Exception:
            shutil.rmtree(delivery_directory, ignore_errors=True)
            raise
        finally:
            if source_fd >= 0:
                os.close(source_fd)

        expires_at_epoch = time.time() + self.ttl_s
        expires_at = datetime.fromtimestamp(
            expires_at_epoch, tz=timezone.utc
        ).isoformat(timespec="seconds")
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        record = _Delivery(
            delivery_id=delivery_id,
            session_key=session_key,
            path=target,
            download_name=name,
            media_type=media_type,
            size=copied,
            sha256=digest.hexdigest(),
            expires_monotonic=time.monotonic() + self.ttl_s,
            expires_at=expires_at,
        )
        timer = threading.Timer(self.ttl_s, self._expire, args=(delivery_id,))
        timer.daemon = True
        record.timer = timer
        with self._lock:
            self._deliveries[delivery_id] = record
        timer.start()
        return {
            "delivery_id": delivery_id,
            "download_url": f"/api/files/{delivery_id}",
            "download_name": name,
            "media_type": media_type,
            "bytes": copied,
            "sha256": record.sha256,
            "expires_at": expires_at,
            "scope": "browser_session",
            "staged_copy": True,
        }

    def resolve(self, session_id: str, delivery_id: str) -> dict[str, Any]:
        if not delivery_id or len(delivery_id) > 128:
            raise FileDeliveryError("file delivery was not found")
        with self._lock:
            record = self._deliveries.get(delivery_id)
            expired = bool(
                record is not None
                and time.monotonic() >= record.expires_monotonic
            )
            matches = bool(
                record is not None
                and record.session_key == self._session_key(session_id)
            )
        if expired:
            self._remove(delivery_id)
            record = None
        if record is None or not matches or not record.path.is_file():
            raise FileDeliveryError("file delivery was not found")
        return {
            "path": record.path,
            "download_name": record.download_name,
            "media_type": record.media_type,
            "bytes": record.size,
            "sha256": record.sha256,
        }

    def inspect(self, session_id: str, delivery_id: str) -> dict[str, Any]:
        resolved = self.resolve(session_id, delivery_id)
        return {
            "delivery_id": delivery_id,
            "download_url": f"/api/files/{delivery_id}",
            "download_name": resolved["download_name"],
            "media_type": resolved["media_type"],
            "bytes": resolved["bytes"],
            "sha256": resolved["sha256"],
            "scope": "browser_session",
            "available": True,
        }

    def clear(self, session_id: str) -> None:
        session_key = self._session_key(session_id)
        with self._lock:
            delivery_ids = [
                delivery_id
                for delivery_id, record in self._deliveries.items()
                if record.session_key == session_key
            ]
        for delivery_id in delivery_ids:
            self._remove(delivery_id)
        shutil.rmtree(self.directory / session_key, ignore_errors=True)

    def stats(self, session_id: str) -> dict[str, int]:
        session_key = self._session_key(session_id)
        with self._lock:
            records = [
                record
                for record in self._deliveries.values()
                if record.session_key == session_key
                and time.monotonic() < record.expires_monotonic
            ]
        return {
            "files": len(records),
            "bytes": sum(record.size for record in records),
        }
