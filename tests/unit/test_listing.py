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

from databend_aiserver.stages.operator import resolve_stage_subpath
from databend_aiserver.udfs.stage import _list_stage_files


def test_resolve_stage_subpath(memory_stage):
    assert resolve_stage_subpath(memory_stage) == "data"
    assert resolve_stage_subpath(memory_stage, "nested/file.txt") == "data/nested/file.txt"


def test_list_stage_files(memory_stage):
    result = _list_stage_files(memory_stage, None)

    paths = {item["path"] for item in result["files"]}
    assert "sample.pdf" in paths
    assert "sample.docx" in paths
    assert "subdir/note.txt" in paths

    sample_entry = next(item for item in result["files"] if item["path"] == "sample.pdf")
    assert sample_entry.get("size", 0) > 0


def test_list_stage_files_limited(memory_stage):
    result = _list_stage_files(memory_stage, 2)
    assert result["count"] == 2
    assert result["truncated"] is True
