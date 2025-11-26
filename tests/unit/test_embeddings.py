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
    ai_embed_1024,
    EmbeddingBackendError,
)


@pytest.mark.slow
def test_vector_embed_text_real_backend_alias():
    result = ai_embed_1024("Databend vector embedding test.")

    assert len(result) == 1
    assert len(result[0]) == EXPECTED_DIMENSION
    assert all(isinstance(v, float) for v in result[0])


@pytest.mark.slow
def test_vector_embed_text_explicit_model():
    # DEFAULT_EMBEDDING_MODEL kept for backward compatibility but function ignores it
    result = ai_embed_1024("Databend explicit model embedding test.")

    assert len(result) == 1
    assert len(result[0]) == EXPECTED_DIMENSION
    assert all(isinstance(v, float) for v in result[0])


def test_vector_embed_text_rejects_unknown_model():
    # Model selection is fixed; unknown model input is not accepted by signature
    with pytest.raises(TypeError):
        ai_embed_1024("text", "extra")


@pytest.mark.slow
def test_vector_embed_text_batch_inputs():
    texts = ["Databend batch vector embedding test.", "Another embedding row."]
    result = ai_embed_1024(texts)

    assert len(result) == 2
    for vector in result:
        assert len(vector) == EXPECTED_DIMENSION
        assert all(isinstance(v, float) for v in vector)
