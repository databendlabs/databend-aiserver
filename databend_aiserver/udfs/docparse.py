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
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from databend_udf import StageLocation, udf
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
try:  # pragma: no cover - optional in-memory path
    from docling.datamodel.document import DocumentStream
except Exception:  # pragma: no cover
    DocumentStream = None  # type: ignore
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from opendal import exceptions as opendal_exceptions

from databend_aiserver.stages.operator import (
    StageConfigurationError,
    get_operator,
    resolve_stage_subpath,
)
from databend_aiserver.config import DEFAULT_EMBED_MODEL, DEFAULT_CHUNK_SIZE


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


def _guess_mime_from_suffix(suffix: str) -> str:
    mime, _ = mimetypes.guess_type(f"file{suffix}")
    return mime or "application/octet-stream"


def _convert_to_markdown(data: bytes, suffix: str) -> ConversionResult:
    converter = DocumentConverter()
    # Prefer in-memory stream when supported to avoid temp files
    if DocumentStream is not None:
        try:
            stream = DocumentStream(
                stream=data,
                name=f"doc{suffix}",
                mime_type=_guess_mime_from_suffix(suffix),
            )
            return converter.convert(stream)
        except Exception:
            pass  # fallback to temp file

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"doc{suffix}"
        tmp_path.write_bytes(data)
        return converter.convert(tmp_path)


_TOKENIZER_CACHE: Dict[str, HuggingFaceTokenizer] = {}


def _get_hf_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    if model_name not in _TOKENIZER_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = HuggingFaceTokenizer(tokenizer=tok)
    return _TOKENIZER_CACHE[model_name]


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

        # Docling chunking: chunk_size controls max_tokens; tokenizer aligned with embedding model
        tokenizer = _get_hf_tokenizer(DEFAULT_EMBED_MODEL)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=DEFAULT_CHUNK_SIZE)

        chunks = list(chunker.chunk(dl_doc=doc))
        pages: List[Dict[str, Any]] = [
            {"index": idx, "content": chunker.contextualize(chunk)}
            for idx, chunk in enumerate(chunks)
        ]
        if not pages:
            pages = [{"index": 0, "content": markdown}]

        page_count = len(pages)

        # Snowflake-compatible (page_split=true) shape:
        # { "pages": [...], "metadata": {"pageCount": N}, "errorInformation": null }
        return {
            "pages": pages,
            "metadata": {"pageCount": page_count},
            "errorInformation": None,
        }
    except Exception as exc:  # pragma: no cover - defensive for unexpected docling errors
        return {
            "content": "",
            "metadata": {},
            "errorInformation": {"message": str(exc), "type": exc.__class__.__name__},
        }
