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
from typing import Any, Dict

from databend_udf.client import UDFClient

from tests.integration.conftest import build_stage_mapping


def _decode_variant(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unexpected VARIANT representation: {type(raw)!r}")


def test_list_stage_files_round_trip(running_server, memory_stage):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "aiserver_list_stage_files",
        0,
        stage_locations=[build_stage_mapping(memory_stage)],
    )

    assert len(result) == 1
    payload = _decode_variant(result[0])
    assert payload["count"] >= 3
    paths = {entry["path"] for entry in payload["files"]}
    assert {"sample.pdf", "sample.docx", "subdir/note.txt"}.issubset(paths)
