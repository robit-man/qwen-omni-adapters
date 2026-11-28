"""Bounded rolling OCR using the Omnius Tesseract evidence strategy."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class PrintedText:
    text: str
    confidence: float


def _text_rows(tsv: str) -> PrintedText | None:
    lines: dict[tuple[str, ...], list[str]] = defaultdict(list)
    confidence = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        value = str(row.get("text") or "").strip()
        try:
            score = float(row.get("conf", "-1"))
        except (TypeError, ValueError):
            continue
        if not value or score < 42:
            continue
        key = tuple(row[field] for field in ("page_num", "block_num", "par_num", "line_num"))
        lines[key].append(value)
        confidence.append(score)
    text = "\n".join(" ".join(words) for words in lines.values()).strip()[:1000]
    letters = sum(char.isalnum() for char in text)
    if len(text) < 3 or letters < 3 or letters < .45 * len(text) or not confidence:
        return None
    mean = sum(confidence) / len(confidence)
    return PrintedText(text, mean) if mean >= 50 else None


class RollingOCR:
    def __init__(self, language: str = "eng", timeout: float = 1.6):
        self.language = language if re.fullmatch(r"[a-z0-9_]+(?:\+[a-z0-9_]+)*", language) else "eng"
        self.timeout = timeout
        self.binary = shutil.which("tesseract")

    def _read(self, image: Image.Image, psm: int, deadline: float) -> PrintedText | None:
        remaining = deadline - time.monotonic()
        if not self.binary or remaining < .05:
            return None
        output = io.BytesIO()
        image.save(output, "PNG")
        try:
            process = subprocess.run([self.binary, "stdin", "stdout", "-l", self.language,
                                      "--psm", str(psm), "tsv"], input=output.getvalue(),
                                     capture_output=True, timeout=remaining, check=False,
                                     env={"PATH": "/usr/local/bin:/usr/bin:/bin", "OMP_THREAD_LIMIT": "1"})
        except (OSError, subprocess.TimeoutExpired):
            return None
        if process.returncode or len(process.stdout) > 131_072:
            return None
        return _text_rows(process.stdout.decode("utf-8", "replace"))

    def recognize(self, frame: bytes) -> PrintedText | None:
        if not self.binary:
            return None
        deadline = time.monotonic() + self.timeout
        try:
            with Image.open(io.BytesIO(frame)) as source:
                gray = ImageOps.exif_transpose(source).convert("L")
                gray.thumbnail((1280, 900), Image.Resampling.LANCZOS)
        except (OSError, ValueError):
            return None
        first = self._read(gray, 11, deadline)
        if first and first.confidence >= 70:
            return first
        contrast = ImageOps.autocontrast(gray, cutoff=1).filter(ImageFilter.SHARPEN)
        second = self._read(contrast, 6, deadline)
        candidates = [result for result in (first, second) if result]
        return max(candidates, key=lambda result: (result.confidence, len(result.text)), default=None)
