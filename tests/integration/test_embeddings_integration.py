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
from databend_udf.client import UDFClient
from databend_aiserver.udfs.embeddings import SUPPORTED_MODELS


@pytest.mark.slow
def test_embedding_dimension(running_server):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_embed_1024",
        "embedded text",
    )
    assert len(result) == 1
    assert len(result[0]) == 1024


@pytest.mark.slow
def test_embedding_data_type(running_server):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function(
        "ai_embed_1024",
        "embedded text",
    )
    payload = result[0]
    assert isinstance(payload, list)
    assert all(isinstance(x, float) for x in payload)


@pytest.mark.slow
def test_vector_embedding_batch_round_trip(running_server):
    client = UDFClient(host="127.0.0.1", port=running_server)
    result = client.call_function_batch(
        "ai_embed_1024",
        text=["embedded text", "second row"],
    )

    assert len(result) == 2
    for payload in result:
        assert isinstance(payload, list)
        assert len(payload) == 1024
        assert all(isinstance(x, float) for x in payload)
