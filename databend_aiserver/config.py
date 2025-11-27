# Centralized configuration for default models and chunking.

import os
from pathlib import Path

# Default embedding model used across UDFs.
DEFAULT_EMBED_MODEL = os.getenv("AISERVER_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Tokenizer must stay aligned with the embedding model.
DEFAULT_TOKENIZER_MODEL = DEFAULT_EMBED_MODEL

# Default chunk size (tokens) for document parsing when not specified.
# Aligned with embedding model context length; can be overridden via env.
def _default_chunk_size(embed_model: str) -> int:
    model_lower = embed_model.lower()
    if "qwen3" in model_lower or "qwen" in model_lower:
        return 32000  # Qwen3-Embedding context length
    return 2048


DEFAULT_CHUNK_SIZE = int(
    os.getenv("AISERVER_CHUNK_SIZE", str(_default_chunk_size(DEFAULT_EMBED_MODEL)))
)

# Shared cache root for all downloaded model artifacts (embeddings, etc.).
# Can be overridden via AISERVER_CACHE_DIR; defaults to repo/.cache
AISERVER_CACHE_DIR = Path(
    os.getenv("AISERVER_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache")
)
