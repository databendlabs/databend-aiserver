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

"""Helpers to translate Databend stage metadata into OpenDAL operators."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Dict, Mapping, Tuple

from databend_udf import StageLocation
from opendal import Operator, exceptions as opendal_exceptions

logger = logging.getLogger(__name__)

_OPERATOR_CACHE: Dict[str, Operator] = {}
_CACHE_LOCK = threading.Lock()


class StageConfigurationError(RuntimeError):
    """Raised when an unsupported or invalid stage configuration is encountered."""


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def _first_present(storage: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in storage:
            value = storage[key]
            if value not in (None, "", {}):
                return value
    return None


def _build_s3_options(storage: Mapping[str, Any]) -> Dict[str, Any]:
    bucket = _first_present(storage, "bucket", "name")
    if not bucket:
        raise StageConfigurationError("S3 stage is missing bucket configuration")

    region = _first_present(storage, "region")
    endpoint = _first_present(storage, "endpoint", "endpoint_url")
    access_key = _first_present(storage, "access_key_id", "aws_key_id")
    secret_key = _first_present(storage, "secret_access_key", "aws_secret_key")
    security_token = _first_present(storage, "security_token", "session_token", "aws_token")
    master_key = _first_present(storage, "master_key")
    root = _first_present(storage, "root")
    role_arn = _first_present(storage, "role_arn", "aws_role_arn")
    external_id = _first_present(storage, "external_id", "aws_external_id")
    virtual_host_style = storage.get("enable_virtual_host_style")
    disable_loader = storage.get("disable_credential_loader")

    options: Dict[str, Any] = {"bucket": bucket}

    if region:
        options["region"] = region
    else:
        # Databend stages may skip region when working with S3 compatible endpoints.
        # OpenDAL requires a region, default to us-east-1 if not provided.
        options["region"] = "us-east-1"

    if endpoint:
        options["endpoint"] = endpoint
    if access_key:
        options["access_key_id"] = access_key
    if secret_key:
        options["secret_access_key"] = secret_key
    if security_token:
        options["security_token"] = security_token
    if master_key:
        options["master_key"] = master_key
    if root:
        options["root"] = root
    if role_arn:
        options["role_arn"] = role_arn
    if external_id:
        options["external_id"] = external_id
    if virtual_host_style is not None:
        options["enable_virtual_host_style"] = _normalize_bool(virtual_host_style)
    if disable_loader is not None:
        options["disable_credential_loader"] = _normalize_bool(disable_loader)

    return options


def _build_memory_options(_: Mapping[str, Any]) -> Dict[str, Any]:
    # Useful for local testing; not a Databend production configuration.
    return {}


_STORAGE_BUILDERS: Dict[str, Any] = {"s3": _build_s3_options, "memory": _build_memory_options}


def _cache_key(stage: StageLocation) -> str:
    payload = {
        "stage_name": stage.stage_name,
        "stage_type": stage.stage_type,
        "storage": stage.storage,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_operator(stage: StageLocation) -> Operator:
    storage = stage.storage or {}
    storage_type = str(storage.get("type", "")).lower()

    if storage_type not in _STORAGE_BUILDERS:
        raise StageConfigurationError(
            f"Unsupported stage storage type '{storage_type or 'unknown'}'"
        )

    builder = _STORAGE_BUILDERS[storage_type]
    options = builder(storage)
    logger.debug(
        "Creating OpenDAL operator for stage '%s' with backend '%s'",
        stage.stage_name,
        storage_type,
    )
    try:
        return Operator(storage_type, **options)
    except opendal_exceptions.Error as exc:
        raise StageConfigurationError(
            f"Failed to construct operator for stage '{stage.stage_name}': {exc}"
        ) from exc


def get_operator(stage: StageLocation) -> Operator:
    """Return a cached OpenDAL operator for the given stage."""

    cache_key = _cache_key(stage)
    with _CACHE_LOCK:
        operator = _OPERATOR_CACHE.get(cache_key)
        if operator is None:
            operator = _build_operator(stage)
            _OPERATOR_CACHE[cache_key] = operator
    return operator


def clear_operator_cache() -> None:
    """Utility that clears cached operators, primarily for testing."""

    with _CACHE_LOCK:
        _OPERATOR_CACHE.clear()


def resolve_stage_subpath(stage: StageLocation, path: str | None = None) -> str:
    """
    Combine the stage's relative path with a user-provided path.

    The resulting string is relative to the OpenDAL operator's configured root.
    """

    def _normalize(component: str | None) -> Tuple[str, ...]:
        if not component:
            return ()
        parts = []
        for part in component.split("/"):
            chunk = part.strip()
            if not chunk or chunk == ".":
                continue
            if chunk == "..":
                raise ValueError("Stage paths must not contain '..'")
            parts.append(chunk)
        return tuple(parts)

    base_parts = _normalize(stage.relative_path)
    extra_parts = _normalize(path)
    full_parts = base_parts + extra_parts
    if not full_parts:
        return ""
    return "/".join(full_parts)


def as_directory_path(path: str) -> str:
    """Ensure the provided path represents a directory for list operations."""
    if not path:
        return ""
    return path if path.endswith("/") else f"{path}/"
