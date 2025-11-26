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

from databend_udf.client import UDFClient

from tests.integration.conftest import build_stage_mapping


def test_read_pdf_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_read_pdf",
        "sample.pdf",
        stage_locations=[build_stage_mapping(memory_stage)],
    )

    assert len(result) == 1
    payload = result[0]
    assert isinstance(payload, str)
    assert "dummy" in payload.replace(" ", "").lower()


def test_read_docx_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_read_docx",
        "sample.docx",
        stage_locations=[build_stage_mapping(memory_stage)],
    )

    assert len(result) == 1
    payload = result[0]
    assert isinstance(payload, str)
    assert "This is a short paragraph" in payload


def test_parse_document_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_parse_document",
        "sample.pdf",
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
    assert isinstance(payload, dict)
    normalized = dict(payload)
    normalized["content"] = "<CONTENT>"
    metadata = dict(normalized.get("metadata") or {})
    metadata["pageCount"] = "<PAGECOUNT>"
    normalized["metadata"] = metadata

    expected = {
        "content": "<CONTENT>",
        "metadata": {
            "pageCount": "<PAGECOUNT>",
            "mode": "LAYOUT",
            "tablesFormat": "markdown",
            "imagesMode": "placeholder",
            "generator": "docling",
            "unused_options": [],
        },
        "errorInformation": None,
    }

    actual_str = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    expected_str = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    assert actual_str == expected_str


def test_parse_document_docx_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_parse_document",
        "sample.docx",
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
    assert isinstance(payload, dict)
    normalized = dict(payload)
    normalized["content"] = "<CONTENT>"
    metadata = dict(normalized.get("metadata") or {})
    metadata["pageCount"] = "<PAGECOUNT>"
    normalized["metadata"] = metadata

    expected = {
        "content": "<CONTENT>",
        "metadata": {
            "pageCount": "<PAGECOUNT>",
            "mode": "LAYOUT",
            "tablesFormat": "markdown",
            "imagesMode": "placeholder",
            "generator": "docling",
            "unused_options": [],
        },
        "errorInformation": None,
    }

    actual_str = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    expected_str = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    assert actual_str == expected_str
