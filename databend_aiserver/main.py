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

"""Command line entrypoint for the Databend AI UDF server."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from contextlib import suppress
from typing import Optional

from prometheus_client import start_http_server as start_prometheus_server

from databend_aiserver.server import create_server

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Databend AI UDF server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the gRPC server")
    parser.add_argument("--port", type=int, default=8815, help="Port for the gRPC server")
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Port for Prometheus metrics exporter (disabled when omitted).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    if args.metrics_port is not None:
        start_prometheus_server(args.metrics_port)
        logger.info("Prometheus metrics server started on port %s", args.metrics_port)

    server = create_server(host=args.host, port=args.port, metric_port=args.metrics_port)
    logger.info("Starting Databend AI UDF server on %s:%s", args.host, args.port)

    # Handle shutdown gracefully to ensure we stop serving when receiving termination signals.
    stop_event = getattr(server, "stopped", None)

    def _handle_signal(signum, frame):  # noqa: ANN001 - signature dictated by signal library
        logger.info("Received signal %s, shutting down.", signum)
        with suppress(Exception):
            server.shutdown()
        if stop_event is not None:
            stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
