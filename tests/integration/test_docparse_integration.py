# Copyright 2025 Databend Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from databend_udf.client import UDFClient

from tests.integration.conftest import build_stage_mapping
from databend_aiserver.stages.operator import get_operator, resolve_stage_subpath


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PDF_SRC = DATA_DIR / "2206.01062.pdf"
DOCX_SRC = DATA_DIR / "lorem_ipsum.docx"


def _call_docparse(client: UDFClient, path: str, memory_stage):
    result = client.call_function(
        "ai_parse_document",
        path,
        stage_locations=[build_stage_mapping(memory_stage)],
    )
    assert len(result) == 1
    payload_raw = result[0]
    if isinstance(payload_raw, (bytes, bytearray)):
        payload_raw = payload_raw.decode("utf-8")
    if isinstance(payload_raw, str):
        payload = json.loads(payload_raw)
    else:
        payload = payload_raw
    return payload


def _normalize_payload(payload):
    pages = [
        {"index": p["index"], "content": "<PAGE_CONTENT>"}
        for p in (payload.get("pages") or [])
    ]
    return {
        "pages": pages,
        "metadata": {"pageCount": "<PAGECOUNT>"},
        "errorInformation": payload.get("errorInformation") or {},
    }


def test_docparse_pdf_structure(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "2206.01062.pdf", memory_stage)
    
    assert "pages" in payload
    assert isinstance(payload["pages"], list)
    assert "metadata" in payload
    assert "pageCount" in payload["metadata"]
    assert "errorInformation" in payload


def test_docparse_pdf_content(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "2206.01062.pdf", memory_stage)
    
    normalized = _normalize_payload(payload)
    page_count = len(normalized["pages"])
    expected = {
        "pages": [{"index": i, "content": "<PAGE_CONTENT>"} for i in range(page_count)],
        "metadata": {"pageCount": "<PAGECOUNT>"},
        "errorInformation": {},
    }
    assert json.dumps(normalized, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_docparse_docx_structure(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "lorem_ipsum.docx", memory_stage)
    
    assert "pages" in payload
    assert isinstance(payload["pages"], list)
    assert "metadata" in payload
    assert "pageCount" in payload["metadata"]
    assert "errorInformation" in payload


def test_docparse_docx_content(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "lorem_ipsum.docx", memory_stage)
    
    normalized = _normalize_payload(payload)
    page_count = len(normalized["pages"])
    expected = {
        "pages": [{"index": i, "content": "<PAGE_CONTENT>"} for i in range(page_count)],
        "metadata": {"pageCount": "<PAGECOUNT>"},
        "errorInformation": {},
    }
    assert json.dumps(normalized, sort_keys=True) == json.dumps(expected, sort_keys=True)
