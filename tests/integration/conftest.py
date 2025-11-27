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

import socket
import threading
import time
from pathlib import Path
from typing import Dict

import pytest
from databend_udf import StageLocation
from prometheus_client import REGISTRY

from databend_aiserver.server import create_server
from databend_aiserver.stages.operator import clear_operator_cache, get_operator

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RELATIVE_PATH = "data"
PDF_SRC = DATA_DIR / "2206.01062.pdf"
DOCX_SRC = DATA_DIR / "lorem_ipsum.docx"


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_stage_mapping(stage: StageLocation, param_name: str = "stage") -> Dict[str, Dict[str, Dict[str, object]]]:
    return {
        "param_name": param_name,
        "relative_path": stage.relative_path,
        "stage_info": {
            "stage_name": stage.stage_name,
            "stage_type": stage.stage_type,
            "stage_params": {
                "storage": stage.storage,
            },
        },
    }


@pytest.fixture
def memory_stage() -> StageLocation:
    clear_operator_cache()
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
    operator.write(f"{RELATIVE_PATH}/subdir/note.txt", b"hello from integration")
    yield stage
    clear_operator_cache()


@pytest.fixture
def fs_stage(tmp_path) -> StageLocation:
    """Create a stage using fs storage to test real opendal API behavior."""
    clear_operator_cache()
    
    # Create test directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    subdir = data_dir / "subdir"
    subdir.mkdir()
    
    # Write test files
    (data_dir / "2206.01062.pdf").write_bytes(PDF_SRC.read_bytes())
    (data_dir / "lorem_ipsum.docx").write_bytes(DOCX_SRC.read_bytes())
    (subdir / "note.txt").write_bytes(b"hello from integration")
    
    stage = StageLocation(
        name="stage_location",
        stage_name="fs_stage",
        stage_type="External",
        storage={"type": "fs", "root": str(tmp_path)},
        relative_path="data",
        raw_info={},
    )
    yield stage
    clear_operator_cache()


@pytest.fixture
def running_server(memory_stage: StageLocation):
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

    port = _allocate_port()
    server = create_server(host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()

    time.sleep(0.2)

    yield port

    server.shutdown()
    thread.join(timeout=5)
