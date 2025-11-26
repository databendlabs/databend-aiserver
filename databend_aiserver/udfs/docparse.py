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

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from databend_udf import StageLocation, udf
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from opendal import exceptions as opendal_exceptions
from pypdf import PdfReader
from docx import Document as DocxDocument
from docx.oxml.ns import qn

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


def _extract_pdf_pages(data: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(data))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def _extract_docx_pages(data: bytes) -> List[str]:
    doc = DocxDocument(io.BytesIO(data))
    pages: List[str] = []
    current: List[str] = []

    def flush():
        if current:
            pages.append("\n".join(current).strip())
            current.clear()

    for para in doc.paragraphs:
        runs = para.runs or []
        has_page_break = any(
            run._element.tag.endswith("}br")
            and run._element.get(qn("w:type")) == "page"
            for run in runs
        )
        text = para.text.strip()
        if text:
            current.append(text)
        if has_page_break:
            flush()
    flush()
    if not pages and current:
        pages.append("\n".join(current))
    return pages or [doc.core_properties.title or ""]


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
    - Includes ``pages`` array with per-page content when possible.
    """
    try:
        raw = _load_stage_file(stage, path)
        suffix = Path(path).suffix or ".bin"
        result = _convert_to_markdown(raw, suffix)
        doc = result.document
        markdown = doc.export_to_markdown()

        # Build pages array (Snowflake-like pagination)
        pages: List[str]
        lowered = suffix.lower()
        if lowered == ".pdf":
            pages = _extract_pdf_pages(raw)
        elif lowered == ".docx":
            pages = _extract_docx_pages(raw)
        else:
            pages = [markdown] if markdown else []

        page_count = len(pages) if pages else None

        return {
            "content": markdown,
            "pages": [
                {"index": idx, "content": content} for idx, content in enumerate(pages)
            ],
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
