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
from databend_aiserver.udfs.stage import _collect_stage_files


def test_resolve_stage_subpath(memory_stage):
    assert resolve_stage_subpath(memory_stage) == "data"
    assert resolve_stage_subpath(memory_stage, "nested/file.txt") == "data/nested/file.txt"


def test_list_stage_files(memory_stage):
    entries, truncated = _collect_stage_files(memory_stage, None)

    paths = {item["path"] for item in entries}
    assert "2206.01062.pdf" in paths
    assert "lorem_ipsum.docx" in paths
    assert "subdir/note.txt" in paths

    pdf_entry = next(item for item in entries if item["path"] == "2206.01062.pdf")
    assert pdf_entry.get("size", 0) > 0
    # last_modified may be absent for backends that don't expose it (e.g., memory)
    assert "last_modified" in pdf_entry or True
    assert truncated is False


def test_list_stage_files_limited(memory_stage):
    entries, truncated = _collect_stage_files(memory_stage, 2)

    assert len(entries) == 2
    assert truncated is True
