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
from typing import Any, Dict, List, Optional, Protocol

from databend_udf import StageLocation, udf
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
try:
    from docling.datamodel.document import DocumentStream
except Exception:
    DocumentStream = None  # type: ignore
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from opendal import exceptions as opendal_exceptions
from time import perf_counter

from databend_aiserver.runtime import DeviceRequest, choose_device, get_runtime
from databend_aiserver.stages.operator import load_stage_file, stage_file_suffix
from databend_aiserver.config import DEFAULT_EMBED_MODEL, DEFAULT_CHUNK_SIZE

try:
    from docling_core.types import AcceleratorOptions, AcceleratorDevice
except Exception:
    AcceleratorOptions = None  # type: ignore
    AcceleratorDevice = None  # type: ignore

logger = logging.getLogger(__name__)


class _ParserBackend(Protocol):
    name: str
    def convert(self, stage: StageLocation, path: str) -> ConversionResult:
        ...


class _DoclingBackend:
    name = "docling"

    def __init__(self) -> None:
        self.choice = self._choose_device()
        self.accel = self._build_accelerator(self.choice)
        self.ocr_provider = self._select_ocr_provider()

    def _choose_device(self):
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
        return choice

    def _build_accelerator(self, choice):
        if AcceleratorOptions is None or AcceleratorDevice is None:
            return None
        if choice.device.startswith("cuda"):
            return AcceleratorOptions(device=AcceleratorDevice.CUDA)
        if choice.device == "mps":
            return AcceleratorOptions(device=AcceleratorDevice.MPS)
        return AcceleratorOptions(device=AcceleratorDevice.CPU)

    def _select_ocr_provider(self) -> Optional[str]:
        runtime = get_runtime()
        providers = runtime.capabilities.onnx_providers
        if runtime.capabilities.device_kind == "cuda" and "CUDAExecutionProvider" in providers:
            choice = "CUDAExecutionProvider"
        else:
            choice = "CPUExecutionProvider"
        logger.info("Docling OCR provider: %s (available=%s)", choice, providers)
        return choice

    def _build_converter(self):
        kwargs: Dict[str, Any] = {}
        if self.accel is not None:
            kwargs["accelerator"] = self.accel
        if self.ocr_provider:
            kwargs["ocr_provider"] = self.ocr_provider
        try:
            return DocumentConverter(**kwargs)
        except TypeError:
            logger.warning(
                "Installed docling version does not support accelerator/ocr_provider kwargs; using defaults"
            )
            return DocumentConverter()

    def convert(self, stage: StageLocation, path: str) -> ConversionResult:
        t_start = perf_counter()
        raw = load_stage_file(stage, path)
        suffix = stage_file_suffix(path)
        converter = self._build_converter()
        if DocumentStream is not None:
            try:
                stream = DocumentStream(
                    stream=raw,
                    name=f"doc{suffix}",
                    mime_type=mimetypes.guess_type(f"file{suffix}")[0] or "application/octet-stream",
                )
                result = converter.convert(stream)
                logger.info(
                    "Docling convert path=%s stream=memory bytes=%s duration=%.3fs",
                    path,
                    len(raw),
                    perf_counter() - t_start,
                )
                return result
            except Exception:
                pass
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"doc{suffix}"
            tmp_path.write_bytes(raw)
            result = converter.convert(tmp_path)
            logger.info(
                "Docling convert path=%s stream=tempfile bytes=%s duration=%.3fs",
                path,
                len(raw),
                perf_counter() - t_start,
            )
            return result


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
        t_total = perf_counter()
        runtime = get_runtime()
        logger.info(
            "ai_parse_document start path=%s runtime_device=%s kind=%s",
            path,
            runtime.capabilities.preferred_device,
            runtime.capabilities.device_kind,
        )
        backend = _get_doc_parser_backend()
        result = backend.convert(stage, path)
        doc = result.document
        markdown = doc.export_to_markdown()

        # Docling chunking: tokenizer aligned with embedding model.
        tokenizer = _get_hf_tokenizer(DEFAULT_EMBED_MODEL)
        chunker = HybridChunker(tokenizer=tokenizer)

        fallback = False
        try:
            chunks = list(chunker.chunk(dl_doc=doc))
            pages: List[Dict[str, Any]] = [
                {"index": idx, "content": chunker.contextualize(chunk)}
                for idx, chunk in enumerate(chunks)
            ]
        except Exception:
            pages = [{"index": 0, "content": markdown}]
            fallback = True
        if not pages:
            pages = [{"index": 0, "content": markdown}]
            fallback = True

        page_count = len(pages)

        # Snowflake-compatible (page_split=true) shape:
        # { "pages": [...], "metadata": {"pageCount": N}, "errorInformation": null }
        payload = {
            "pages": pages,
            "metadata": {
                "pageCount": page_count,
                "chunkingFallback": fallback,
            },
            "errorInformation": (
                {} if not fallback else {"type": "ChunkingFallback", "message": "chunker failed or returned empty; returned full markdown instead"}
            ),
        }
        logger.info(
            "ai_parse_document path=%s backend=%s pages=%s fallback=%s duration=%.3fs",
            path,
            getattr(backend, "name", "unknown"),
            page_count,
            fallback,
            perf_counter() - t_total,
        )
        return payload
    except Exception as exc:  # pragma: no cover - defensive for unexpected docling errors
        return {
            "content": "",
            "metadata": {},
            "errorInformation": {"message": str(exc), "type": exc.__class__.__name__},
        }
