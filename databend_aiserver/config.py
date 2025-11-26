# Centralized configuration for default models and chunking.

import os

# Default embedding model used across UDFs.
DEFAULT_EMBED_MODEL = os.getenv("AISERVER_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Tokenizer must stay aligned with the embedding model.
DEFAULT_TOKENIZER_MODEL = DEFAULT_EMBED_MODEL

# Default chunk size (tokens) for document parsing when not specified.
DEFAULT_CHUNK_SIZE = int(os.getenv("AISERVER_CHUNK_SIZE", "2048"))
