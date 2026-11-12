"""Local, bounded PDF evidence extraction; no OCR, inference, or URL fetching."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_TEXT_CHARS = 12_000
MAX_OUTPUT_BYTES = 512 * 1024
MAX_TABLES = 4
MAX_COLUMNS = 16
MAX_LINKS = 20
MAX_RESULT_CHARS = 18_000


class PdfStructureError(ValueError):
    """A PDF could not be safely parsed within the extraction budget."""


def parse_pdf(source: Path, start_page: int, max_pages: int, max_rows: int) -> dict[str, Any]:
    import pdfplumber

    start_page = max(1, int(start_page))
    max_pages = max(1, min(5, int(max_pages)))
    max_rows = max(1, min(200, int(max_rows)))
    remaining = MAX_TEXT_CHARS
    truncated = False

    def text(value: str, limit: int = 6_000) -> tuple[str, bool]:
        nonlocal remaining, truncated
        kept = value[:min(limit, remaining)]
        remaining -= len(kept)
        cut = len(kept) < len(value)
        truncated |= cut
        return kept, cut

    with pdfplumber.open(source) as pdf:
        total = len(pdf.pages)
        if start_page > total:
            raise PdfStructureError(f"start_page exceeds the PDF's {total} pages")
        end = min(total, start_page + max_pages - 1)
        pages = []
        for page in pdf.pages[start_page - 1:end]:
            # Keep cell boundaries and null merged cells; never guess headers,
            # spans, chart values, or an absent borderless table.
            detected = page.find_tables()
            tables = []
            page_truncated = len(detected) > MAX_TABLES
            for ordinal, table in enumerate(detected[:MAX_TABLES]):
                rows = table.extract()
                cells = []
                table_truncated = len(rows) > max_rows
                for row in rows[:max_rows]:
                    table_truncated |= len(row) > MAX_COLUMNS
                    retained = []
                    for cell in row[:MAX_COLUMNS]:
                        if cell is None:
                            retained.append(None)
                        else:
                            value, cut = text(cell, 500)
                            retained.append(value)
                            table_truncated |= cut
                    cells.append(retained)
                tables.append({
                    "id": f"page-{page.page_number}-table-{ordinal + 1}",
                    "bbox": list(table.bbox), "rows": cells,
                    "row_count": len(rows),
                    "column_count": max((len(row) for row in rows), default=0),
                    "truncated": table_truncated, "structure_verified": False,
                })
                page_truncated |= table_truncated

            # Remove only returned, complete tables from prose. If a table was
            # omitted or truncated, its source text remains eligible for reading.
            boxes = [table.bbox for table, result in zip(detected, tables, strict=False)
                     if not result["truncated"]]

            def outside_tables(obj: dict[str, Any], regions=boxes) -> bool:
                if obj.get("object_type") != "char":
                    return True
                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2
                return not any(x0 <= x <= x1 and top <= y <= bottom
                               for x0, top, x1, bottom in regions)

            prose = page.filter(outside_tables).extract_text(layout=True) or ""
            prose = "\n".join(line.rstrip() for line in prose.splitlines()).strip()
            content, cut = text(prose)
            page_truncated |= cut
            links = []
            annotations = page.hyperlinks
            for link in annotations:
                uri = str(link.get("uri") or "")
                try:
                    parsed = urlsplit(uri)
                    valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                except ValueError:
                    valid = False
                if not valid or len(uri) > 2_048 or len(links) >= MAX_LINKS:
                    continue
                links.append({"url": uri, "source": "pdf_link_annotation",
                              "link_status": "not_fetched",
                              "bbox": [link.get(key) for key in ("x0", "top", "x1", "bottom")]})
            image_count = len(page.images)
            has_text = bool(page.chars)
            pages.append({
                "page": page.page_number, "width": page.width, "height": page.height,
                "text": content, "tables": tables, "links": links,
                "links_omitted": len(annotations) - len(links),
                "image_count": image_count,
                "image_regions_unread": image_count > 0,
                "text_layer_present": has_text,
                "ocr_required": not has_text and image_count > 0,
                "visual_review_required": image_count > 0 or not has_text,
                "truncated": page_truncated,
            })
            truncated |= page_truncated
            page.close()
        result = {
            "parser": {"name": "pdfplumber", "version": pdfplumber.__version__,
                       "method": "embedded_text_geometry_and_ruled_tables", "ocr_performed": False},
            "coverage": {"total_pages": total, "start_page": start_page, "end_page": end,
                         "next_start_page": end + 1 if end < total else None,
                         "all_pages_returned": start_page == 1 and end == total,
                         "truncated": truncated},
            "quality": {"reading_order_verified": False, "table_structure_verified": False,
                        "chart_values_verified": False, "visual_elements_interpreted": False,
                        "table_detection": "ruled_tables_only; undetected tables may remain in text"},
            "pages": pages,
        }
        _bound_result(result)
        return result


def _bound_result(result: dict[str, Any]) -> None:
    """Keep valid JSON and coverage even for dense tables or escaped text."""

    def size() -> int:
        return len(json.dumps(result, ensure_ascii=False))

    if size() <= MAX_RESULT_CHARS:
        return
    result["coverage"]["truncated"] = True
    for page in reversed(result["pages"]):
        page["truncated"] = True
        excess = max(0, size() - MAX_RESULT_CHARS)
        if excess:
            page["text"] = page["text"][:-excess] if excess < len(page["text"]) else ""
        for table in reversed(page["tables"]):
            while table["rows"] and size() > MAX_RESULT_CHARS:
                table["rows"].pop()
                table["truncated"] = True
        while page["links"] and size() > MAX_RESULT_CHARS:
            page["links"].pop()
            page["links_omitted"] += 1
        if size() <= MAX_RESULT_CHARS:
            return
    raise PdfStructureError("PDF metadata exceeds the bounded result limit")


def read_pdf_structure(raw: bytes, start_page: int, max_pages: int, max_rows: int) -> dict[str, Any]:
    """Bound parser lifetime and output independently of the request process."""

    if not raw.startswith(b"%PDF-"):
        raise PdfStructureError("PDF MIME type does not match its file signature")
    with tempfile.TemporaryDirectory(prefix="robit-omni-pdf-structure-") as directory:
        source, output = Path(directory) / "input.pdf", Path(directory) / "output.json"
        source.write_bytes(raw)
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(source), str(output),
                 str(start_page), str(max_pages), str(max_rows)],
                check=False, capture_output=True,
                timeout=min(60.0, max(1.0, float(os.environ.get("OMNI_PDF_TIMEOUT_S", "30")))),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PdfStructureError(f"PDF structure extraction failed: {type(exc).__name__}") from exc
        if completed.returncode != 0 or not output.is_file():
            detail = completed.stderr.decode("utf-8", errors="replace")[-300:]
            raise PdfStructureError(f"PDF structure extraction failed: {detail}")
        if output.stat().st_size > MAX_OUTPUT_BYTES:
            raise PdfStructureError("PDF structure exceeds the bounded output limit")
        return json.loads(output.read_text("utf-8"))


def main() -> int:
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    try:
        result = parse_pdf(Path(sys.argv[1]), *(int(arg) for arg in sys.argv[3:6]))
        Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - parser failures must not kill the portal
        print(f"{type(exc).__name__}: {str(exc)[:250]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
