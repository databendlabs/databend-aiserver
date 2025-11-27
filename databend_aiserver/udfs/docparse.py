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
from typing import Any, Dict, List, Optional, Protocol, Tuple

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


def _resolve_full_path(stage_location: StageLocation, path: str) -> str:
    resolved_path = resolve_stage_subpath(stage_location, path)
    storage = stage_location.storage or {}
    storage_root = str(storage.get("root", "") or "")
    bucket = storage.get("bucket") or storage.get("name")

    if storage_root.startswith("s3://"):
        base = storage_root.rstrip("/")
        return f"{base}/{resolved_path}"
    elif bucket:
        base = f"s3://{bucket}"
        if storage_root:
            base = f"{base}/{storage_root.strip('/')}"
        return f"{base}/{resolved_path}"
    
    return resolved_path or path


def _chunk_document(doc: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """Chunk the document and return pages/chunks and a fallback flag."""
    markdown = doc.export_to_markdown()
    tokenizer = _get_hf_tokenizer(DEFAULT_EMBED_MODEL)
    chunker = HybridChunker(tokenizer=tokenizer)

    try:
        chunks = list(chunker.chunk(dl_doc=doc))
        if not chunks:
            return [{"index": 0, "content": markdown}], True
            
        return [
            {"index": idx, "content": chunker.contextualize(chunk)}
            for idx, chunk in enumerate(chunks)
        ], False
    except Exception:
        return [{"index": 0, "content": markdown}], True


def _format_response(
    path: str,
    full_path: str,
    pages: List[Dict[str, Any]],
    file_size: int,
    timings: Dict[str, float],
    fallback: bool
) -> Dict[str, Any]:
    duration_ms = timings["total"]
    payload: Dict[str, Any] = {
        "metadata": {
            "chunk_count": len(pages),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "duration_ms": duration_ms,
            "file_size": file_size,
            "filename": Path(path).name,
            "path": full_path,
            "timings_ms": timings,
            "version": 1,
        },
        "chunks": pages,
    }
    
    if fallback:
        payload["error_information"] = [
            {
                "type": "ChunkingFallback",
                "message": "chunker failed or returned empty; returned full markdown instead",
            }
        ]
    return payload


@udf(
    name="ai_parse_document",
    stage_refs=["stage_location"],
    input_types=["STRING"],
    result_type="VARIANT",
    io_threads=4,
)
def ai_parse_document(stage_location: StageLocation, path: str) -> Dict[str, Any]:
    """Parse a document and return Snowflake-compatible layout output."""
    try:
        t_total_ns = perf_counter_ns()
        runtime = get_runtime()
        logger.info(
            "ai_parse_document start path=%s runtime_device=%s kind=%s",
            path,
            runtime.capabilities.preferred_device,
            runtime.capabilities.device_kind,
        )
        
        backend = _get_doc_parser_backend()
        t_convert_start_ns = perf_counter_ns()
        result, file_size = backend.convert(stage_location, path)
        t_convert_end_ns = perf_counter_ns()

        pages, fallback = _chunk_document(result.document)
        t_chunk_end_ns = perf_counter_ns()

        full_path = _resolve_full_path(stage_location, path)
        
        timings = {
            "convert": (t_convert_end_ns - t_convert_start_ns) / 1_000_000.0,
            "chunk": (t_chunk_end_ns - t_convert_end_ns) / 1_000_000.0,
            "total": (t_chunk_end_ns - t_total_ns) / 1_000_000.0,
        }

        payload = _format_response(path, full_path, pages, file_size, timings, fallback)
        
        logger.info(
            "ai_parse_document path=%s backend=%s chunks=%s fallback=%s duration_ms=%.1f",
            path,
            getattr(backend, "name", "unknown"),
            len(pages),
            fallback,
            timings["total"],
        )
        return payload
        
    except Exception as exc:  # pragma: no cover
        return {
            "metadata": {
                "path": path,
                "filename": Path(path).name,
            },
            "chunks": [],
            "error_information": [{"message": str(exc), "type": exc.__class__.__name__}],
        }

