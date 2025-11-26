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

from __future__ import annotations

from pathlib import Path

import pytest
from databend_udf import StageLocation

from databend_aiserver.stages.operator import clear_operator_cache, get_operator

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RELATIVE_PATH = "data"
PDF_SRC = DATA_DIR / "2206.01062.pdf"
DOCX_SRC = DATA_DIR / "lorem_ipsum.docx"


@pytest.fixture(autouse=True)
def _clear_operator_cache() -> None:
    clear_operator_cache()
    yield
    clear_operator_cache()


@pytest.fixture
def memory_stage() -> StageLocation:
    stage = StageLocation(
        name="stage",
        stage_name="memory_stage",
        stage_type="External",
        storage={"type": "memory"},
        relative_path=RELATIVE_PATH,
        raw_info={},
    )

    operator = get_operator(stage)
    operator.create_dir(f"{RELATIVE_PATH}/")
    operator.create_dir(f"{RELATIVE_PATH}/subdir/")
    operator.write(f"{RELATIVE_PATH}/2206.01062.pdf", PDF_SRC.read_bytes())
    operator.write(f"{RELATIVE_PATH}/lorem_ipsum.docx", DOCX_SRC.read_bytes())
    operator.write(f"{RELATIVE_PATH}/subdir/note.txt", b"hello from memory")
    return stage
