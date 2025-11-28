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

import pytest
from databend_udf.client import UDFClient
from typing import List, Dict, Any

from tests.integration.conftest import build_stage_mapping, StageLocation


def _get_listing(
    server_port: int, stage: StageLocation, pattern: str = None, max_files: int = 0
) -> List[Dict[str, Any]]:
    client = UDFClient(host="127.0.0.1", port=server_port)
    
    # ai_list_files(stage_location, pattern, max_files)
    # UDFClient.call_function accepts *args, not RecordBatch
    return client.call_function(
        "ai_list_files",
        pattern,
        max_files,
        stage_locations=[build_stage_mapping(stage, "stage_location")],
    )


def test_list_stage_files_content(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    assert len(rows) >= 3
    paths = {row["path"] for row in rows}
    # Paths returned by opendal scan are relative to the root, so they include the stage prefix "data/"
    expected_paths = {
        "data/2206.01062.pdf",
        "data/lorem_ipsum.docx",
        "data/subdir/note.txt",
    }
    assert expected_paths.issubset(paths)


def test_list_stage_files_metadata(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    assert {row["stage_name"] for row in rows} == {memory_stage.stage_name}
    # Memory stage uri might be just the path if no bucket/root
    assert all("uri" in row for row in rows)
    assert all(row["uri"].endswith(row["path"]) for row in rows)
    # Check that last_modified key exists (value might be None for memory backend)
    assert all("last_modified" in row for row in rows)


def test_list_stage_files_schema(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    for row in rows:
        assert "path" in row
        assert "uri" in row
        assert "size" in row
        assert "last_modified" in row
        assert "etag" in row  # May be None
        assert "content_type" in row  # May be None

        # Verify order implicitly by checking keys list if needed,
        # but for now just existence is enough as dicts are ordered in Python 3.7+
        keys = list(row.keys())
        # Expected keys: stage_name, path, uri, size, last_modified, etag, content_type
        # Note: stage_name is added by _get_listing or the UDF logic, let's check the core ones
        assert keys.index("path") < keys.index("uri")
        assert keys.index("last_modified") < keys.index("etag")


def test_list_stage_files_truncation(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage, max_files=1)
    assert len(rows) == 1
    assert "last_modified" in rows[0]


def test_list_stage_files_pattern(running_server, memory_stage):
    # Test pattern matching - patterns match against full path (e.g., "data/file.pdf")
    rows = _get_listing(running_server, memory_stage, pattern="data/*.pdf")
    assert len(rows) == 1
    assert rows[0]["path"].endswith(".pdf")

    rows = _get_listing(running_server, memory_stage, pattern="data/*.docx")
    assert len(rows) == 1
    assert rows[0]["path"].endswith(".docx")

    rows = _get_listing(running_server, memory_stage, pattern="data/subdir/*")
    # Matches data/subdir/ and data/subdir/note.txt
    assert len(rows) == 2
    paths = {r["path"] for r in rows}
    assert "data/subdir/note.txt" in paths
    assert "data/subdir/" in paths
