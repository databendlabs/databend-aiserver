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

import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, Dict, List, Optional, Protocol, Tuple, Sequence

from databend_udf import StageLocation, udf
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
try:
    from docling.datamodel.document import DocumentStream
except Exception:
    DocumentStream = None  # type: ignore
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from databend_aiserver.runtime import DeviceRequest, choose_device, get_runtime
from databend_aiserver.stages.operator import (
    load_stage_file,
    stage_file_suffix,
    resolve_stage_subpath,
    resolve_full_path,
)
from databend_aiserver.config import DEFAULT_EMBED_MODEL, DEFAULT_CHUNK_SIZE

try:
    # Preferred: docling's public accelerator API (propagates through pipeline options).
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
except Exception:
    try:
        # Fallback for older installs.
        from docling_core.types import AcceleratorOptions, AcceleratorDevice
    except Exception:
        AcceleratorOptions = None  # type: ignore
        AcceleratorDevice = None  # type: ignore
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

logger = logging.getLogger(__name__)


class _ParserBackend(Protocol):
    name: str
    def convert(self, stage_location: StageLocation, path: str) -> tuple[ConversionResult, int]:
        ...


class _DoclingBackend:
    name = "docling"

    def __init__(self) -> None:
        self.accel = self._build_accelerator()

    def _build_accelerator(self):
        override = os.getenv("AISERVER_DOCLING_DEVICE")
        req = DeviceRequest(task="docling", allow_gpu=True, allow_mps=True, explicit=override)
        choice = choose_device(req)
        logger.info(
            "Docling device selected device=%s precision=%s reason=%s override=%s",
            choice.device,
            choice.precision,
            choice.reason,
            override,
        )

        if AcceleratorOptions is None or AcceleratorDevice is None:
            return None
        if choice.device.startswith("cuda"):
            return AcceleratorOptions(device=AcceleratorDevice.CUDA)
        if choice.device == "mps":
            return AcceleratorOptions(device=AcceleratorDevice.MPS)
        return AcceleratorOptions(device=AcceleratorDevice.CPU)

    def _build_converter(self):
        format_options: Dict[InputFormat, Any] = {}
        if self.accel is not None:
            pdf_opts = ThreadedPdfPipelineOptions()
            pdf_opts.accelerator_options = self.accel
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pdf_opts)

        try:
            return DocumentConverter(format_options=format_options if format_options else None)
        except TypeError:
            logger.warning("Installed docling version does not support format_options; using defaults")
            return DocumentConverter()

    def convert(self, stage_location: StageLocation, path: str) -> tuple[ConversionResult, int]:
        t_start = perf_counter()
        raw = load_stage_file(stage_location, path)
        suffix = stage_file_suffix(path)
        converter = self._build_converter()
        
        # Try processing from memory stream first
        if DocumentStream is not None:
            try:
                stream = DocumentStream(
                    stream=raw,
                    name=f"doc{suffix}",
                    mime_type=mimetypes.guess_type(f"file{suffix}")[0] or "application/octet-stream",
                )
                result = converter.convert(stream)
                logger.info("Docling convert path=%s stream=memory bytes=%s duration=%.3fs", 
                          path, len(raw), perf_counter() - t_start)
                return result, len(raw)
            except Exception:
                pass

        # Fallback to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"doc{suffix}"
            tmp_path.write_bytes(raw)
            result = converter.convert(tmp_path)
            logger.info("Docling convert path=%s stream=tempfile bytes=%s duration=%.3fs", 
                      path, len(raw), perf_counter() - t_start)
            return result, len(raw)


def _get_doc_parser_backend() -> _ParserBackend:
    backend = os.getenv("AISERVER_DOC_BACKEND", "docling").lower()
    if backend == "docling":
        logger.info("Doc parser backend selected: docling")
        return _DoclingBackend()
    raise ValueError(f"Unknown document parser backend '{backend}'")


_TOKENIZER_CACHE: Dict[str, HuggingFaceTokenizer] = {}


def _get_hf_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    if model_name not in _TOKENIZER_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = HuggingFaceTokenizer(
            tokenizer=tok, max_tokens=DEFAULT_CHUNK_SIZE
        )
    return _TOKENIZER_CACHE[model_name]





def _chunk_document(doc: Any) -> Tuple[List[Dict[str, Any]], int]:
    """Chunk the document and return pages/chunks and total tokens."""
    tokenizer = _get_hf_tokenizer(DEFAULT_EMBED_MODEL)
    chunker = HybridChunker(tokenizer=tokenizer)

    chunks = list(chunker.chunk(dl_doc=doc))
    if not chunks:
        raise ValueError("HybridChunker returned no chunks")
        
    logger.info(
        "HybridChunker produced %d chunks. Merging to fit %d tokens...",
        len(chunks),
        DEFAULT_CHUNK_SIZE,
    )

    merged_chunks = []
    current_chunk_text = ""
    current_tokens = 0
    total_tokens = 0
    delimiter = "\n\n"
    delimiter_tokens = tokenizer.count_tokens(delimiter)

    for chunk in chunks:
        text = chunker.contextualize(chunk)
        text_tokens = tokenizer.count_tokens(text)

        if current_tokens + delimiter_tokens + text_tokens > DEFAULT_CHUNK_SIZE and current_chunk_text:
            merged_chunks.append({
                "index": len(merged_chunks),
                "content": current_chunk_text,
                "tokens": current_tokens,
            })
            total_tokens += current_tokens
            current_chunk_text = text
            current_tokens = text_tokens
        else:
            if current_chunk_text:
                current_chunk_text += delimiter + text
                current_tokens += delimiter_tokens + text_tokens
            else:
                current_chunk_text = text
                current_tokens = text_tokens
    
    if current_chunk_text:
        merged_chunks.append({
            "index": len(merged_chunks),
            "content": current_chunk_text,
            "tokens": current_tokens,
        })
        total_tokens += current_tokens

    logger.info(
        "Merged chunks: %d -> %d",
        len(chunks),
        len(merged_chunks),
    )

    return merged_chunks, total_tokens


def _format_response(
    file_path: str,
    uri: str,
    pages: List[Dict[str, Any]],
    file_size: int,
    timings: Dict[str, float],
    num_tokens: int,
    error: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Format the response with a fixed template."""
    metadata = {
        "chunk_count": len(pages),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "duration_ms": timings.get("total", 0.0),
        "file_size": file_size,
        "filename": Path(file_path).name,
        "num_tokens": num_tokens,
        "uri": uri,
        "timings_ms": timings,
        "version": 1,
    }

    response = {
        "metadata": metadata,
        "chunks": pages,
    }

    if error:
        response["error_information"] = [error]

    return response


@udf(
    name="ai_parse_document",
    stage_refs=["stage_location"],
    input_types=["STRING"],
    result_type="VARIANT",
    io_threads=4,
)
def ai_parse_document(stage_location: StageLocation, file_path: str) -> Dict[str, Any]:
    """Parse a document and return Snowflake-compatible layout output."""
    try:
        t_total_ns = perf_counter_ns()
        runtime = get_runtime()
        logger.info(
            "ai_parse_document start path=%s runtime_device=%s kind=%s",
            file_path,
            runtime.capabilities.preferred_device,
            runtime.capabilities.device_kind,
        )
        
        backend = _get_doc_parser_backend()
        t_convert_start_ns = perf_counter_ns()
        result, file_size = backend.convert(stage_location, file_path)
        t_convert_end_ns = perf_counter_ns()

        pages, num_tokens = _chunk_document(result.document)
        t_chunk_end_ns = perf_counter_ns()

        uri = resolve_full_path(stage_location, file_path)
        
        timings = {
            "convert": (t_convert_end_ns - t_convert_start_ns) / 1_000_000.0,
            "chunk": (t_chunk_end_ns - t_convert_end_ns) / 1_000_000.0,
            "total": (t_chunk_end_ns - t_total_ns) / 1_000_000.0,
        }

        payload = _format_response(
            file_path, uri, pages, file_size, timings, num_tokens
        )
        
        logger.info(
            "ai_parse_document path=%s backend=%s chunks=%s duration_ms=%.1f",
            file_path,
            getattr(backend, "name", "unknown"),
            len(pages),
            timings["total"],
        )
        return payload
        
    except Exception as exc:  # pragma: no cover
        return {
            "metadata": {
                "file_path": file_path,
                "filename": Path(file_path).name,
            },
            "chunks": [],
            "error_information": [{"message": str(exc), "type": exc.__class__.__name__}],
        }

