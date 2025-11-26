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

from databend_aiserver.udfs.files import _read_docx, _read_pdf


def test_read_pdf(memory_stage):
    result = _read_pdf(memory_stage, "sample.pdf")

    assert isinstance(result, str)
    assert "dummy" in result.replace(" ", "").lower()


def test_read_docx(memory_stage):
    result = _read_docx(memory_stage, "sample.docx")

    assert isinstance(result, str)
    assert "This is a short paragraph" in result
