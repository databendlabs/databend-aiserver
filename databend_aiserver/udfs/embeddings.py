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

"""Vector embedding UDFs backed by Hugging Face models."""

from __future__ import annotations

import os
import threading
from typing import Dict, Iterable, List, Optional, Tuple, Sequence
import logging
from pathlib import Path

from databend_udf import udf
from databend_aiserver.config import DEFAULT_EMBED_MODEL
from databend_aiserver.runtime import DeviceRequest, choose_device

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - torch must be installed by the user
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - transformers must be installed
    AutoModel = AutoTokenizer = None  # type: ignore[assignment]

# Supported embedding families mapped to expected output dimension.
SUPPORTED_MODELS = [
    ("qwen", DEFAULT_EMBED_MODEL, 1024),
]

_ALIAS_TO_ENTRY = {alias.lower(): (model_id, dim) for alias, model_id, dim in SUPPORTED_MODELS}
_MODEL_TO_ENTRY = {model_id.lower(): (model_id, dim) for _, model_id, dim in SUPPORTED_MODELS}
DEFAULT_EMBEDDING_MODEL = SUPPORTED_MODELS[0][1]
EXPECTED_DIMENSION = SUPPORTED_MODELS[0][2]

# Cache directory under the repository for downloaded models to avoid polluting
# the user's global cache and to make behaviour deterministic across runs.
EMBED_CACHE_DIR = Path(__file__).resolve().parents[2] / ".hf-cache"
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(EMBED_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(EMBED_CACHE_DIR))

_BACKEND_CACHE: Dict[Tuple[str, str], "_EmbeddingBackend"] = {}
_BACKEND_LOCK = threading.Lock()


class EmbeddingBackendError(RuntimeError):
    """Raised when the embedding backend cannot produce a vector."""


def _ensure_cache_directory() -> str:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(EMBED_CACHE_DIR)


class _EmbeddingBackend:
    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._logger = logging.getLogger(__name__)

    @classmethod
    def from_pretrained(cls, model_name: str, device: str, torch_dtype) -> "_EmbeddingBackend":
        if torch is None or AutoModel is None or AutoTokenizer is None:
            raise EmbeddingBackendError(
                "Both torch and transformers must be installed to use embedding UDFs"
            )

        cache_dir = _ensure_cache_directory()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=False,  # fast tokenizer for Qwen3 embedding can be incompatible with tokenizers version
        )
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model = model.eval()
        model.to(device)
        logging.getLogger(__name__).info(
            "vector_embed_text_1024 loaded '%s' on device '%s' (dtype=%s)",
            model_name,
            device,
            torch_dtype,
        )
        return cls(tokenizer, model, device)

    def embed(self, text: str) -> List[float]:
        if torch is None:
            raise EmbeddingBackendError("torch is not available in the runtime")

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        return [float(value) for value in embeddings[0].tolist()]


def _get_backend(model_name: str) -> _EmbeddingBackend:
    choice = choose_device(DeviceRequest(task="embedding"))
    cache_key = (model_name, choice.device)
    with _BACKEND_LOCK:
        backend = _BACKEND_CACHE.get(cache_key)
        if backend is None:
            backend = _EmbeddingBackend.from_pretrained(model_name, choice.device, choice.dtype)
            logging.getLogger(__name__).info(
                "Embedding backend created device=%s precision=%s reason=%s",
                choice.device,
                choice.precision,
                choice.reason,
            )
            _BACKEND_CACHE[cache_key] = backend
    return backend


def _resolve_model(model: str) -> Tuple[str, int]:
    lookup_key = model.strip().lower()
    entry = _ALIAS_TO_ENTRY.get(lookup_key)
    if entry is None:
        entry = _MODEL_TO_ENTRY.get(lookup_key)
    if entry is None:
        supported = ", ".join(
            f"alias '{alias}' (model id '{model_id}')" for alias, model_id, _ in SUPPORTED_MODELS
        )
        raise EmbeddingBackendError(
            f"Model '{model}' is not supported. Supported values: {supported}"
        )
    return entry


@udf(
    input_types=["STRING"],
    result_type="ARRAY(NULLABLE(FLOAT))",
    name="ai_embed_1024",
    io_threads=4,
    batch_mode=True,
)
def ai_embed_1024(text: Sequence[str] | str) -> List[List[float]]:
    """SQL definition:

    ```sql
    CREATE FUNCTION ai_embed_1024(text STRING)
        RETURNS ARRAY(FLOAT NULL);
    ```

    Uses the default embedding model (alias 'qwen' → Qwen/Qwen3-Embedding-0.6B).
    Accepts a single string or a list of strings for batch embedding.
    """

    if isinstance(text, list):
        texts = list(text)
    else:
        texts = [text]

    model_name, expected_dimension = _resolve_model(SUPPORTED_MODELS[0][0])
    backend = _get_backend(model_name)

    vectors: List[List[float]] = []
    for text_item in texts:
        if not text_item:
            vectors.append([])
            continue
        vector = backend.embed(text_item)
        if vector and len(vector) != expected_dimension:
            raise EmbeddingBackendError(
                f"Model '{model_name}' returned {len(vector)}-dimensional vector; "
                f"expected {expected_dimension}"
            )
        vectors.append(vector)

    return vectors
