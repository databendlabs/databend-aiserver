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

"""Document parsing UDF powered by Docling."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from databend_udf import StageLocation, udf
from docling.document_converter import DocumentConverter
from docling.models.base import ConversionResult
from opendal import exceptions as opendal_exceptions

from databend_aiserver.stages.operator import (
    StageConfigurationError,
    get_operator,
    resolve_stage_subpath,
)


def _load_stage_file(stage: StageLocation, path: str) -> bytes:
    try:
        operator = get_operator(stage)
    except StageConfigurationError as exc:
        raise ValueError(str(exc)) from exc

    resolved = resolve_stage_subpath(stage, path)
    if not resolved:
        raise ValueError("ai_parse_document requires a non-empty file path")
    try:
        return operator.read(resolved)
    except opendal_exceptions.NotFound as exc:
        raise FileNotFoundError(f"Stage object '{resolved}' not found") from exc
    except opendal_exceptions.Error as exc:
        raise RuntimeError(f"Failed to read '{resolved}' from stage") from exc


def _convert_to_markdown(data: bytes, suffix: str) -> ConversionResult:
    converter = DocumentConverter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"doc{suffix}"
        tmp_path.write_bytes(data)
        return converter.convert(tmp_path)


@udf(
    name="ai_parse_document",
    stage_refs=["stage"],
    input_types=["STRING"],
    result_type="VARIANT",
    io_threads=4,
)
def ai_parse_document(stage: StageLocation, path: str) -> Dict[str, Any]:
    """Parse a document and return Snowflake-compatible layout output.

    Simplified semantics:
    - Always processes the full document.
    - Always returns Markdown layout in ``content``.
    - No options parameter (defaults are chosen for ease-of-use).
    """
    try:
        raw = _load_stage_file(stage, path)
        suffix = Path(path).suffix or ".bin"
        result = _convert_to_markdown(raw, suffix)
        doc = result.document
        markdown = doc.export_to_markdown()
        page_count = len(getattr(doc, "pages", [])) if hasattr(doc, "pages") else None

        return {
            "content": markdown,
            "metadata": {
                "pageCount": page_count,
                "mode": "LAYOUT",
                "tablesFormat": "markdown",
                "imagesMode": "placeholder",
                "generator": "docling",
                "unused_options": [],
            },
            "errorInformation": None,
        }
    except Exception as exc:  # pragma: no cover - defensive for unexpected docling errors
        return {
            "content": "",
            "metadata": {},
            "errorInformation": {"message": str(exc), "type": exc.__class__.__name__},
        }
