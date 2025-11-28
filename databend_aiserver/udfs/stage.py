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

import fnmatch
import logging
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from databend_udf import StageLocation, udf
from databend_aiserver.stages.operator import (
    StageConfigurationError,
    get_operator,
    resolve_stage_subpath,
    resolve_storage_uri,
)
from opendal import exceptions as opendal_exceptions


def _format_last_modified(value: Any) -> Optional[str]:
    if value is None:
        return None
    # OpenDAL returns datetime objects; fall back to string otherwise.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _collect_stage_files(
    stage_location: StageLocation, max_files: Optional[int]
) -> tuple[List[Dict[str, Any]], bool]:
    """List objects stored under a Databend stage."""

    t_start = perf_counter()

    try:
        operator = get_operator(stage_location)
    except StageConfigurationError as exc:
        raise ValueError(str(exc)) from exc

    base_prefix = resolve_stage_subpath(stage_location)
    max_entries = max_files if max_files and max_files > 0 else None

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

        file_info: Dict[str, Any] = {
            "path": relative_path or path,
            "is_dir": is_dir,
            "size": 0,
            "mode": "",
            "content_type": "",
            "etag": "",
            "last_modified": None,
        }

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
            if hasattr(metadata, "last_modified"):
                file_info["last_modified"] = _format_last_modified(metadata.last_modified)

        entries.append(file_info)
        if max_entries is not None and len(entries) >= max_entries:
            truncated = True
            break

    duration = perf_counter() - t_start
    logging.getLogger(__name__).info(
        "ai_list_files scanned entries=%s truncated=%s stage=%s base=%s duration=%.3fs",
        len(entries),
        truncated,
        stage_location.stage_name,
        base_prefix,
        duration,
    )

    return entries, truncated


@udf(
    stage_refs=["stage_location"],
    input_types=["VARCHAR", "INT"],
    result_type=[
        ("stage_name", "VARCHAR"),
        ("path", "VARCHAR"),
        ("uri", "VARCHAR"),
        ("size", "UINT64"),
        ("last_modified", "VARCHAR"),
        ("etag", "VARCHAR"),
        ("content_type", "VARCHAR"),
    ],
    name="ai_list_files",
)
def ai_list_files(
    stage_location: StageLocation, pattern: Optional[str], max_files: Optional[int]
) -> Iterable[Dict[str, Any]]:
    """List objects in a stage."""

    logging.getLogger(__name__).info(
        "ai_list_files start stage=%s relative=%s pattern=%s max_files=%s",
        stage_location.stage_name,
        stage_location.relative_path,
        pattern,
        max_files,
    )

    if max_files is None or max_files <= 0:
        max_files = 0

    op = get_operator(stage_location)
    prefix = resolve_stage_subpath(stage_location)
    truncated = False

    try:
        # Use scan() to recursively list all files
        # Iterate lazily to support large datasets and early stopping
        scanner = op.scan(prefix)
        
        count = 0
        for entry in scanner:
            if pattern and not fnmatch.fnmatch(entry.path, pattern):
                continue

            if max_files > 0 and count >= max_files:
                truncated = True
                break
                
            count += 1
            if count % 1000 == 0:
                logging.getLogger(__name__).info(
                    "ai_list_files scanning... found %d files so far (max_files=%s)", 
                    count, max_files
                )

            metadata = op.stat(entry.path)
            # Check if directory using mode (opendal.Metadata doesn't have is_dir())
            # Mode for directories typically has specific bits set, or path ends with /
            
            # Convert mode to string if it exists, otherwise None
            
            yield {
                "stage_name": stage_location.stage_name,
                "path": entry.path,
                "uri": resolve_storage_uri(stage_location, entry.path),
                "size": metadata.content_length,
                "last_modified": _format_last_modified(
                    getattr(metadata, "last_modified", None)
                ),
                "etag": metadata.etag,
                "content_type": metadata.content_type,
            }
            
    except Exception as e:
        logging.getLogger(__name__).error("Error listing files: %s", e)
        # Stop yielding
