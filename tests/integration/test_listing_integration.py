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


def test_list_stage_files_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    rows = client.call_function(
        "ai_list_files",
        0,
        stage_locations=[build_stage_mapping(memory_stage)],
    )

    assert len(rows) >= 3
    paths = {row["path"] for row in rows}

    assert {"sample.pdf", "sample.docx", "subdir/note.txt"}.issubset(paths)
    assert {row["stage"] for row in rows} == {memory_stage.stage_name}
    assert {row["relative_path"] for row in rows} == {memory_stage.relative_path}
    assert all(row["truncated"] is False for row in rows)
    assert all("is_dir" in row for row in rows)


def test_list_stage_files_truncation(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    rows = client.call_function(
        "ai_list_files",
        1,
        stage_locations=[build_stage_mapping(memory_stage)],
    )

    assert len(rows) == 1
    assert rows[0]["truncated"] is True
