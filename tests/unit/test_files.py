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

from databend_aiserver.udfs.files import _read_docx, _read_pdf
from databend_aiserver.udfs.docparse import ai_parse_document


def test_read_pdf(memory_stage):
    result = _read_pdf(memory_stage, "sample.pdf")

    assert isinstance(result, str)
    assert "dummy" in result.replace(" ", "").lower()


def test_read_docx(memory_stage):
    result = _read_docx(memory_stage, "sample.docx")

    assert isinstance(result, str)
    assert "This is a short paragraph" in result


def test_parse_document(memory_stage):
    result = ai_parse_document(memory_stage, "sample.pdf")

    assert isinstance(result, dict)
    result_str = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert isinstance(result_str, str)
    assert '"content":"' in result_str
    assert '"errorInformation":null' in result_str
    assert '"generator":"docling"' in result_str
    assert '"mode":"LAYOUT"' in result_str
    assert '"tablesFormat":"markdown"' in result_str


def test_parse_document_docx(memory_stage):
    result = ai_parse_document(memory_stage, "sample.docx")

    assert isinstance(result, dict)
    result_str = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert isinstance(result_str, str)
    assert '"content":"' in result_str
    assert '"errorInformation":null' in result_str
    assert '"generator":"docling"' in result_str
    assert '"mode":"LAYOUT"' in result_str
    assert '"tablesFormat":"markdown"' in result_str
