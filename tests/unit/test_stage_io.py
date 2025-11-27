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

from databend_aiserver.stages.operator import load_stage_file, stage_file_suffix


def test_load_stage_file_reads_bytes(memory_stage):
    data = load_stage_file(memory_stage, "2206.01062.pdf")
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0


def test_load_stage_file_missing(memory_stage):
    with pytest.raises(FileNotFoundError):
        load_stage_file(memory_stage, "missing.pdf")


def test_stage_file_suffix_defaults_and_parses():
    assert stage_file_suffix("foo/bar.txt") == ".txt"
    assert stage_file_suffix("noext") == ".bin"
