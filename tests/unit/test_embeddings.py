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

import pytest

from databend_aiserver.udfs.embeddings import (
    EXPECTED_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_MODELS,
    aiserver_vector_embed_text_1024,
    EmbeddingBackendError,
)


@pytest.mark.slow
def test_vector_embed_text_real_backend_alias():
    result = aiserver_vector_embed_text_1024("qwen", "Databend vector embedding test.")

    assert len(result) == 1
    assert len(result[0]) == EXPECTED_DIMENSION
    assert all(isinstance(v, float) for v in result[0])


@pytest.mark.slow
def test_vector_embed_text_explicit_model():
    result = aiserver_vector_embed_text_1024(
        DEFAULT_EMBEDDING_MODEL, "Databend explicit model embedding test."
    )

    assert len(result) == 1
    assert len(result[0]) == EXPECTED_DIMENSION
    assert all(isinstance(v, float) for v in result[0])


def test_vector_embed_text_rejects_unknown_model():
    with pytest.raises(EmbeddingBackendError) as exc:
        aiserver_vector_embed_text_1024("unknown-model", "text")

    assert str(exc.value) == (
        "Model 'unknown-model' is not supported. Supported values: "
        "alias 'qwen' (model id 'Qwen/Qwen3-Embedding-0.6B')"
    )


@pytest.mark.slow
def test_vector_embed_text_batch_inputs():
    texts = ["Databend batch vector embedding test.", "Another embedding row."]
    result = aiserver_vector_embed_text_1024("qwen", texts)

    assert len(result) == 2
    for vector in result:
        assert len(vector) == EXPECTED_DIMENSION
        assert all(isinstance(v, float) for v in vector)
