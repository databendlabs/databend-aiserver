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

"""Functions for reading stage files such as PDFs and Word documents."""

from __future__ import annotations

import io
import logging
from typing import List, Optional

from databend_udf import StageLocation, udf
from opendal import exceptions as opendal_exceptions
from pypdf import PdfReader
from docx import Document

from databend_aiserver.stages.operator import StageConfigurationError, get_operator, resolve_stage_subpath

logger = logging.getLogger(__name__)

_DEFAULT_JOIN = "\n\n"


def _read_stage_bytes(stage: StageLocation, path: str) -> bytes:
    try:
        operator = get_operator(stage)
    except StageConfigurationError as exc:
        raise ValueError(str(exc)) from exc

    resolved = resolve_stage_subpath(stage, path)
    if not resolved:
        raise ValueError("A file path must be provided")
    try:
        return operator.read(resolved)
    except opendal_exceptions.NotFound as exc:
        raise FileNotFoundError(f"Stage object '{resolved}' not found") from exc
    except opendal_exceptions.Error as exc:
        raise RuntimeError(f"Failed to read '{resolved}' from stage") from exc


def _read_pdf(stage: StageLocation, path: str) -> str:
    if not path:
        raise ValueError("PDF reader requires a non-empty path")

    data = _read_stage_bytes(stage, path)
    reader = PdfReader(io.BytesIO(data), strict=False)

    texts: List[str] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive in pdf parsing
            logger.warning("Failed to extract text from page %s: %s", index, exc)
            text = ""
        texts.append(text.strip())

    return _DEFAULT_JOIN.join(segment for segment in texts if segment)


def _read_docx(stage: StageLocation, path: str) -> str:
    if not path:
        raise ValueError("DOCX reader requires a non-empty path")

    data = _read_stage_bytes(stage, path)

    document = Document(io.BytesIO(data))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text]
    return _DEFAULT_JOIN.join(paragraphs)

