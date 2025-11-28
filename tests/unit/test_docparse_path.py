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

from databend_aiserver.udfs.docparse import ai_parse_document
import json


def test_docparse_metadata_path_uses_root(memory_stage_with_root):
    raw = ai_parse_document(memory_stage_with_root, "2206.01062.pdf")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    meta = payload.get("metadata", {})
    assert meta["uri"] == "s3://wizardbend/dataset/data/2206.01062.pdf"
    assert meta["filename"] == "2206.01062.pdf"
