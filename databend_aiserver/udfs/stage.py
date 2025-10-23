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

"""Functions for enumerating files inside Databend stages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from databend_udf import StageLocation, udf
from opendal import exceptions as opendal_exceptions

from databend_aiserver.stages.operator import (
    StageConfigurationError,
    get_operator,
    resolve_stage_subpath,
)


def _prepare_listing_response(
    stage: StageLocation, files: List[Dict[str, Any]], truncated: bool
) -> Dict[str, Any]:
    return {
        "stage": stage.stage_name,
        "relative_path": stage.relative_path,
        "files": files,
        "count": len(files),
        "truncated": truncated,
    }


def _list_stage_files(stage: StageLocation, limit: Optional[int]) -> Dict[str, Any]:
    """List objects stored under a Databend stage."""

    try:
        operator = get_operator(stage)
    except StageConfigurationError as exc:
        raise ValueError(str(exc)) from exc

    base_prefix = resolve_stage_subpath(stage)
    max_entries = limit if limit and limit > 0 else None

    entries: List[Dict[str, Any]] = []
    truncated = False

    iterator = operator.scan(base_prefix)

    prefix_to_strip = f"{base_prefix.rstrip('/')}/" if base_prefix else ""

    for entry in iterator:
        path = entry.path
        is_dir = path.endswith("/")
        relative_path = path
        if prefix_to_strip and path.startswith(prefix_to_strip):
            relative_path = path[len(prefix_to_strip) :]
        if not relative_path and is_dir:
            continue

        file_info: Dict[str, Any] = {"path": relative_path or path, "is_dir": is_dir}

        try:
            metadata = operator.stat(path)
        except opendal_exceptions.Error:
            metadata = None
        if metadata:
            if metadata.content_length is not None and metadata.content_length >= 0:
                file_info["size"] = int(metadata.content_length)
            if metadata.mode is not None:
                file_info["mode"] = str(metadata.mode)
            if metadata.content_type:
                file_info["content_type"] = metadata.content_type
            if metadata.etag:
                file_info["etag"] = metadata.etag

        entries.append(file_info)
        if max_entries is not None and len(entries) >= max_entries:
            truncated = True
            break

    return _prepare_listing_response(stage, entries, truncated)


@udf(stage_refs=["stage"], input_types=["INT"], result_type="VARIANT", io_threads=8)
def aiserver_list_stage_files(stage: StageLocation, limit: Optional[int]) -> Dict[str, Any]:
    """SQL definition:

    ```sql
    CREATE FUNCTION aiserver_list_stage_files(stage STAGE_LOCATION, limit INT)
        RETURNS VARIANT;
    ```
    """
    return _list_stage_files(stage, limit)
