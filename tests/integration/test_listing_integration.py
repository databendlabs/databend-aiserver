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

from databend_udf.client import UDFClient

from tests.integration.conftest import build_stage_mapping


def _get_listing(running_server, memory_stage, limit=0):
    client = UDFClient(host="127.0.0.1", port=running_server)
    return client.call_function(
        "ai_list_files",
        limit,
        stage_locations=[build_stage_mapping(memory_stage)],
    )


def test_list_stage_files_content(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    assert len(rows) >= 3
    paths = {row["path"] for row in rows}
    assert {"2206.01062.pdf", "lorem_ipsum.docx", "subdir/note.txt"}.issubset(paths)


def test_list_stage_files_metadata(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    assert {row["stage"] for row in rows} == {memory_stage.stage_name}
    assert {row["relative_path"] for row in rows} == {memory_stage.relative_path}


def test_list_stage_files_schema(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage)
    for row in rows:
        assert "is_dir" in row
        assert row["size"] is not None
        assert row["mode"] is not None
        assert row["content_type"] is not None
        assert row["etag"] is not None
        assert row["truncated"] is False


def test_list_stage_files_truncation(running_server, memory_stage):
    rows = _get_listing(running_server, memory_stage, limit=1)
    assert len(rows) == 1
    assert rows[0]["truncated"] is True


# Tests with fs storage to expose real opendal API behavior
def test_list_fs_stage_files_content(running_server, fs_stage):
    """Test listing files with fs storage (same API as S3)."""
    client = UDFClient(host="127.0.0.1", port=running_server)
    rows = client.call_function(
        "ai_list_files",
        0,
        stage_locations=[build_stage_mapping(fs_stage, "stage_location")],
    )
    assert len(rows) >= 3
    paths = {row["path"] for row in rows}
    assert {"data/2206.01062.pdf", "data/lorem_ipsum.docx", "data/subdir/note.txt"}.issubset(paths)


def test_list_fs_stage_metadata_fields(running_server, fs_stage):
    """Test that all metadata fields are correctly populated with fs storage."""
    client = UDFClient(host="127.0.0.1", port=running_server)
    rows = client.call_function(
        "ai_list_files",
        0,
        stage_locations=[build_stage_mapping(fs_stage, "stage_location")],
    )
    
    for row in rows:
        # Verify all required fields exist
        assert "stage_name" in row
        assert "relative_path" in row
        assert "path" in row
        assert "is_dir" in row
        assert "size" in row
        assert "mode" in row
        assert "content_type" in row
        assert "etag" in row
        assert "truncated" in row
        
        # Verify field types and values
        assert isinstance(row["is_dir"], bool)
        assert row["stage_name"] == fs_stage.stage_name
        assert row["relative_path"] == fs_stage.relative_path
        
        # Files should have size, directories might not
        if not row["is_dir"]:
            assert row["size"] is not None
            assert row["size"] > 0

