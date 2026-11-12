from __future__ import annotations

import base64
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from portal.app import DEFAULT_MODEL, PortalConfig, create_app
from portal.documents import DocumentError, SessionDocumentStore
from portal.pdf_structure import (
    MAX_RESULT_CHARS,
    PdfStructureError,
    _bound_result,
    read_pdf_structure,
)
from portal.tools import PortalToolHarness

FIXTURE = Path(__file__).parent / "fixtures" / "structured-document.pdf"


def _upload(name="evidence.pdf", raw=None, mime="application/pdf"):
    return {"name": name, "mime_type": mime, "encoding": "base64",
            "data": base64.b64encode(FIXTURE.read_bytes() if raw is None else raw).decode()}


def _store(name="evidence.pdf"):
    store = SessionDocumentStore()
    _context, accepted = store.prepare("owner", [_upload(name)], "source evidence")
    return store, accepted[0]["id"]


def test_pdf_structure_preserves_tables_annotations_and_missing_text_layers():
    result = read_pdf_structure(FIXTURE.read_bytes(), 1, 2, 50)
    assert result["coverage"] == {
        "total_pages": 3, "start_page": 1, "end_page": 2,
        "next_start_page": 3, "all_pages_returned": False, "truncated": False,
    }
    first, scan = result["pages"]
    assert "Left column: Acacia" in first["text"]
    assert "Right column: Birch" in first["text"]
    assert first["tables"][0]["rows"] == [["Metric", "Value"], ["Acacia", "7"], ["Birch", "13"]]
    assert len(first["tables"][0]["bbox"]) == 4
    assert first["tables"][0]["structure_verified"] is False
    assert first["links"][0]["url"] == "https://example.org/source?ref=pdf"
    assert first["links"][0]["source"] == "pdf_link_annotation"
    assert first["links"][0]["link_status"] == "not_fetched"
    assert scan["ocr_required"] and scan["image_regions_unread"]
    assert scan["text"] == "" and not scan["text_layer_present"]
    assert result["parser"]["ocr_performed"] is False
    assert result["quality"]["chart_values_verified"] is False
    assert result["quality"]["reading_order_verified"] is False


@pytest.mark.parametrize("name", ["evidence.pdf", "mime-only-attachment"])
def test_pdf_structured_tool_reads_by_id_filename_and_path_with_provenance(name):
    store, document_id = _store(name)
    harness = PortalToolHarness(store)
    by_name = harness.execute("owner", "structured_read", {"document_id": name})
    assert "error" not in by_name
    assert by_name["provenance"]["source_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert by_name["provenance"]["current_visual_input"] is False
    assert "ocr_pdf" in by_name["alternative_tools"]
    by_id = harness.execute("owner", "structured_read", {
        "document_id": document_id, "start_page": 3, "max_pages": 1, "path": "pages[0].text",
    })
    assert "Final page" in by_id["data"]
    assert by_id["coverage"]["next_start_page"] is None
    assert by_id["coverage"]["all_pages_returned"] is False
    all_pages = store.structured_read("owner", document_id, max_pages=5)
    assert all_pages["coverage"]["all_pages_returned"] is True
    assert all_pages["quality"]["visual_elements_interpreted"] is False


def test_pdf_table_row_limit_and_global_budget_report_truncation():
    result = read_pdf_structure(FIXTURE.read_bytes(), 1, 1, 1)
    table = result["pages"][0]["tables"][0]
    assert table["rows"] == [["Metric", "Value"]]
    assert table["row_count"] == 3 and table["truncated"]
    assert result["coverage"]["truncated"]
    # A truncated table keeps its unreturned source cells available in prose.
    assert "Birch" in result["pages"][0]["text"]
    dense = copy.deepcopy(result)
    dense["pages"][0]["text"] = "\x00" * 20_000
    dense["pages"][0]["tables"][0]["rows"] = [["cell" * 500] * 16] * 200
    _bound_result(dense)
    assert len(json.dumps(dense, ensure_ascii=False)) <= MAX_RESULT_CHARS
    assert dense["coverage"]["total_pages"] == 3
    assert dense["coverage"]["truncated"]


def test_pdf_structure_rejects_invalid_pages_signatures_and_corrupt_files():
    with pytest.raises(PdfStructureError, match="signature"):
        read_pdf_structure(b"not a PDF", 1, 1, 50)
    with pytest.raises(PdfStructureError, match="start_page"):
        read_pdf_structure(FIXTURE.read_bytes(), 9, 1, 50)
    with pytest.raises(PdfStructureError, match="extraction failed"):
        read_pdf_structure(b"%PDF-1.7\nbroken", 1, 1, 50)


def test_pdf_parser_timeout_is_an_explicit_tool_failure(monkeypatch):
    store, document_id = _store()

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("portal.pdf_structure.subprocess.run", timeout)
    result = PortalToolHarness(store).execute("owner", "structured_read", {"document_id": document_id})
    assert result["error"] == "DocumentError"
    assert "TimeoutExpired" in result["message"]


def test_pdf_lookup_keeps_missing_format_and_other_session_errors_distinct():
    store, document_id = _store()
    harness = PortalToolHarness(store)
    assert harness.execute("stranger", "structured_read", {"document_id": document_id})["error"] == "DocumentError"
    missing = harness.execute("owner", "structured_read", {"document_id": "absent.pdf"})
    assert "not found" in missing["message"]
    store.prepare("owner", [_upload("notes.txt", b"Plain text evidence", "text/plain")], "notes")
    mismatch = harness.execute("owner", "structured_read", {"document_id": "notes.txt"})
    assert mismatch["error"] == "unsupported_document_format"
    assert mismatch["alternative_tools"] == ["document_search"]
    assert mismatch["task_blocked"] is False
    store.clear("owner")
    assert "error" in harness.execute("owner", "structured_read", {"document_id": document_id})


def test_trash_during_pdf_parse_cannot_return_retired_session_evidence(monkeypatch):
    store, document_id = _store()
    parse = read_pdf_structure

    def clear_then_parse(*args):
        store.clear("owner")
        return parse(*args)

    monkeypatch.setattr("portal.documents.read_pdf_structure", clear_then_parse)
    with pytest.raises(DocumentError, match="session expired"):
        store.structured_read("owner", document_id)


def test_pdf_session_expiry_prevents_parsing(monkeypatch):
    store, document_id = _store()
    index = store._sessions[store._key("owner")]
    monkeypatch.setattr("portal.documents.time.monotonic", lambda: index.last_seen + store.ttl_s + 1)
    with pytest.raises(DocumentError, match="no attached documents"):
        store.structured_read("owner", document_id)


@pytest.mark.parametrize(("name", "raw", "expected"), [
    ("data.json", b'{"n":7}', {"n": 7}),
    ("data.jsonl", b'{"n":7}\n{"n":13}\n', [{"n": 7}, {"n": 13}]),
    ("data.csv", b"n,label\n7,Acacia\n", [{"n": "7", "label": "Acacia"}]),
    ("data.tsv", b"n\tlabel\n7\tAcacia\n", [{"n": "7", "label": "Acacia"}]),
    ("data.yaml", b"n: 7\n", {"n": 7}),
])
def test_existing_structured_formats_and_implicit_selection_remain_usable(name, raw, expected):
    store, _ = _store()
    store.prepare("owner", [_upload(name, raw, "text/plain")], "n")
    assert store.structured_read("owner", name)["data"] == expected
    # A PDF and structured file require an explicit choice, not a guessed attachment.
    with pytest.raises(DocumentError, match="document_id is required"):
        store.structured_read("owner")


@pytest.mark.skipif(not all(shutil.which(name) for name in ("pdftoppm", "tesseract")),
                    reason="real local OCR binaries unavailable")
def test_mixed_pdf_ocr_is_explicit_and_search_keeps_unverified_attribution():
    store, document_id = _store()
    harness = PortalToolHarness(store)
    before = harness.execute("owner", "ocr_pdf", {"document_id": document_id})
    assert before["already_extractable"] is True
    assert "structured_read" in before["alternative_tools"]
    assert "does not prove" in before["claim_limits"]
    result = harness.execute("owner", "ocr_pdf", {"document_id": document_id, "force": True, "max_pages": 2})
    assert "error" not in result
    assert "cedar-nine" in result["preview"]
    assert result["text_source"] == "ocr_unverified"
    found = harness.execute("owner", "document_search", {"query": "cedar-nine"})
    assert "cedar-nine" in found["results"][0]["content"]
    assert found["results"][0]["text_source"] == "ocr_unverified"
    assert not harness.execute("stranger", "document_search", {"query": "cedar-nine"})["results"]


def test_portal_document_discovery_reads_pdf_and_pages_before_final_answer():
    token = "structured-pdf-test-token-with-24-characters"
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        names = {tool["function"]["name"] for tool in body.get("tools", [])}
        if len(seen) == 1:
            name, arguments = "tool_search", {"family": "documents"}
        elif len(seen) == 2:
            assert {"structured_read", "ocr_pdf", "document_search"} <= names
            name, arguments = "structured_read", {"document_id": "evidence.pdf"}
        elif len(seen) == 3:
            result = json.loads(body["messages"][-1]["content"])
            assert result["data"]["pages"][0]["tables"][0]["rows"][1] == ["Acacia", "7"]
            assert result["data"]["pages"][1]["ocr_required"]
            assert "ocr_pdf" in names
            name, arguments = "structured_read", {"document_id": "evidence.pdf", "start_page": result["coverage"]["next_start_page"]}
        else:
            result = json.loads(body["messages"][-1]["content"])
            assert "Final page" in result["data"]["pages"][0]["text"]
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "Acacia is 7; page 2 requires OCR."}})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"call-{len(seen)}", "function": {"name": name, "arguments": arguments}}]}})

    config = PortalConfig(adapter_url="http://adapter/api/chat", adapter_health_url="http://adapter/healthz",
                          comprehension_health_url="http://comprehension/health", tts_health_url="http://tts/healthz",
                          ollama_health_url="http://ollama/api/tags", model=DEFAULT_MODEL,
                          access_token=token, timeout_s=30, max_body_bytes=1024 * 1024)
    app = create_app(config, httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    client.get("/")
    response = client.post("/api/chat", headers={"Authorization": f"Bearer {token}"}, json={
        "model": DEFAULT_MODEL, "stream": False, "think": False,
        "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
        "response_modalities": ["text"], "speech_mode": "never", "portal_auto_tools": True,
        "messages": [{"role": "user", "content": "Read the attachment.", "documents": [_upload()]}],
    })
    assert response.status_code == 200
    assert len(seen) == 4
    trace = response.json["portal"]["safe_tools_executed"]
    assert [item["name"] for item in trace] == ["tool_search", "structured_read", "structured_read"]
    assert all(item["ok"] for item in trace)
