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
        stage_locations=[build_stage_mapping(memory_stage, param_name="stage_location")],
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
    chunks = payload.get("chunks") or []
    metadata = payload.get("metadata") or {}
    return {
        "chunk_count": metadata.get("chunk_count"),
        "chunk_len": len(chunks),
        "has_path": "path" in metadata,
        "has_filename": "filename" in metadata,
        "has_file_size": "file_size" in metadata,
        "has_duration": "duration_ms" in metadata,
        "has_errors": "error_information" in payload,
    }


def test_docparse_pdf_structure(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "2206.01062.pdf", memory_stage)
    
    norm = _normalize_payload(payload)
    assert norm["chunk_count"] == norm["chunk_len"]
    assert norm["has_path"]
    assert norm["has_filename"]
    assert norm["has_file_size"]
    assert norm["has_duration"]
    assert not norm["has_errors"]


def test_docparse_pdf_content(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "2206.01062.pdf", memory_stage)
    
    assert "chunks" in payload and isinstance(payload["chunks"], list)
    assert payload["metadata"]["chunk_count"] == len(payload["chunks"])


def test_docparse_docx_structure(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "lorem_ipsum.docx", memory_stage)
    
    norm = _normalize_payload(payload)
    assert norm["chunk_count"] == norm["chunk_len"]
    assert norm["has_path"]
    assert norm["has_filename"]
    assert norm["has_file_size"]
    assert norm["has_duration"]
    assert not norm["has_errors"]


def test_docparse_docx_content(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    payload = _call_docparse(client, "lorem_ipsum.docx", memory_stage)
    
    assert "chunks" in payload and isinstance(payload["chunks"], list)
    assert payload["metadata"]["chunk_count"] == len(payload["chunks"])
