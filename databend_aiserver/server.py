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

"""Entrypoint for the Databend AI UDF server."""

from __future__ import annotations

from typing import Optional

from databend_udf import UDFServer

from databend_aiserver.udfs import (
    ai_list_files,
    ai_embed_1024,
    ai_parse_document,
)


def create_server(
    host: str = "0.0.0.0", port: int = 8815, metric_port: Optional[int] = None
) -> UDFServer:
    """
    Create a configured UDF server instance.

    Parameters
    ----------
    host:
        Bind address for the Flight server.
    port:
        Bind port for the Flight server.
    metric_port:
        Optional metrics port for Prometheus exporter. When provided the
        databend-udf server will expose metrics via Prometheus.
    """
    location = f"{host}:{port}"
    metric_location = (
        f"{host}:{metric_port}" if metric_port is not None else None
    )
    server = UDFServer(location, metric_location=metric_location)
    server.add_function(ai_list_files)
    server.add_function(ai_embed_1024)
    server.add_function(ai_parse_document)
    return server
