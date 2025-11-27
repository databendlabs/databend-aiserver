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

"""Runtime capability detection shared by all UDFs.

This module runs a one-time probe at process start and exposes a read-only
``RuntimeCapabilities`` that UDFs can consult to pick devices/dtypes. The
runtime module deliberately avoids making policy decisions for specific UDFs;
it only reports what is available and preferred based on environment hints.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore


DeviceKind = Literal["cpu", "cuda", "mps", "rocm"]


@dataclass(frozen=True)
class DeviceRequest:
    """Parameters for choosing a device per UDF.

    - task: semantic hint for logging only.
    - allow_gpu / allow_mps: gates whether GPU/MPS are considered.
    - prefer_fp16 / prefer_bf16: toggles precision preference when non-CPU.
    - explicit: highest-priority device spec ("cuda:1", "cpu", "mps").
    - fallback: device string to use when nothing else fits.
    """

    task: Literal["embedding", "docling", "llm", "custom"]
    allow_gpu: bool = True
    allow_mps: bool = True
    prefer_fp16: bool = True
    prefer_bf16: bool = True
    explicit: Optional[str] = None
    fallback: str = "cpu"


@dataclass(frozen=True)
class DeviceChoice:
    device: str
    dtype: Optional["torch.dtype"]  # type: ignore[name-defined]
    precision: Literal["fp16", "bf16", "fp32", "none"]
    reason: str


@dataclass(frozen=True)
class RuntimeCapabilities:
    device_kind: DeviceKind
    preferred_device: str
    visible_devices: list[str]
    memory_mb: Optional[int]
    torch_available: bool
    supports_fp16: bool
    supports_bf16: bool
    onnx_providers: list[str]
    timestamp: datetime


_RUNTIME: Optional[RuntimeCapabilities] = None
_LOCK = threading.Lock()


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _parse_force_device(env_value: Optional[str]) -> Optional[str]:
    if not env_value:
        return None
    value = env_value.strip().lower()
    if value in {"cpu", "cuda", "mps", "rocm"}:
        return value
    # allow cuda:0 / rocm:0 etc.
    if any(value.startswith(prefix) for prefix in ("cuda:", "rocm:")):
        return value
    return None


def _detect_torch_device(force_device: Optional[str], disable_gpu: bool) -> tuple[DeviceKind, str, list[str], Optional[int], bool, bool, bool]:
    """Return (device_kind, preferred_device, visible_devices, memory_mb, torch_available, fp16, bf16)."""

    if torch is None:
        return "cpu", "cpu", ["cpu"], None, False, False, False

    # Respect disable flag first.
    if disable_gpu:
        return "cpu", "cpu", ["cpu"], None, True, False, False

    # Environment override (e.g. AISERVER_DEVICE=cuda:1 or cpu)
    if force_device:
        if force_device.startswith("cuda") and torch.cuda.is_available():
            idx = force_device.split(":")[-1] if ":" in force_device else "0"
            device = f"cuda:{idx}"
            mem = _get_cuda_total_memory(int(idx))
            fp16 = True
            bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda *_: False)(int(idx)))
            return "cuda", device, _visible_cuda_devices(), mem, True, fp16, bf16
        if force_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "mps", ["mps"], None, True, True, False
        if force_device.startswith("rocm") and torch.cuda.is_available():  # ROCm shows as cuda in torch
            return "rocm", force_device, _visible_cuda_devices(), None, True, True, False
        return "cpu", "cpu", ["cpu"], None, True, False, False

    # Automatic detection order: CUDA -> MPS -> ROCm -> CPU
    if torch.cuda.is_available():  # pragma: no cover - GPU seldom in CI
        device = "cuda:0"
        mem = _get_cuda_total_memory(0)
        fp16 = True
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda *_: False)(0))
        return "cuda", device, _visible_cuda_devices(), mem, True, fp16, bf16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return "mps", "mps", ["mps"], None, True, True, False

    # Torch on ROCm typically reports cuda devices; fallback handled above.
    return "cpu", "cpu", ["cpu"], None, True, False, False


def _get_cuda_total_memory(index: int) -> Optional[int]:
    try:
        props = torch.cuda.get_device_properties(index)
        return int(props.total_memory // (1024 * 1024))
    except Exception:
        return None


def _visible_cuda_devices() -> list[str]:
    try:
        count = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(count)] if count else ["cuda:0"]
    except Exception:
        return ["cuda:0"]


def _detect_onnx_providers() -> list[str]:
    if ort is None:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def detect_runtime(force_device: str | None = None, disable_gpu: bool = False) -> RuntimeCapabilities:
    """Probe runtime capabilities once and cache the result.

    Environment overrides:
    - AISERVER_DISABLE_GPU=1        → force CPU
    - AISERVER_DEVICE=cuda:0|cpu... → preferred device
    """

    global _RUNTIME
    with _LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        env_force = _parse_force_device(force_device) or _parse_force_device(os.getenv("AISERVER_DEVICE"))
        disable = disable_gpu or _env_bool("AISERVER_DISABLE_GPU")

        device_kind, preferred_device, visible_devices, memory_mb, torch_available, fp16, bf16 = _detect_torch_device(
            env_force, disable
        )
        onnx_providers = _detect_onnx_providers()

        _RUNTIME = RuntimeCapabilities(
            device_kind=device_kind,
            preferred_device=preferred_device,
            visible_devices=visible_devices,
            memory_mb=memory_mb,
            torch_available=torch_available,
            supports_fp16=fp16,
            supports_bf16=bf16,
            onnx_providers=onnx_providers,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "Runtime detected: device_kind=%s preferred=%s visible=%s memory_mb=%s torch=%s fp16=%s bf16=%s onnx_providers=%s",
            device_kind,
            preferred_device,
            ",".join(visible_devices),
            memory_mb,
            torch_available,
            fp16,
            bf16,
            onnx_providers,
        )

        return _RUNTIME


def get_runtime() -> RuntimeCapabilities:
    if _RUNTIME is None:
        raise RuntimeError("detect_runtime() must be called before accessing runtime capabilities")
    return _RUNTIME


def _device_available(device: str, runtime: RuntimeCapabilities, req: DeviceRequest) -> bool:
    if device.startswith("cuda"):
        return req.allow_gpu and runtime.device_kind == "cuda"
    if device == "mps":
        return req.allow_mps and runtime.device_kind == "mps"
    if device.startswith("rocm"):
        return req.allow_gpu and runtime.device_kind == "rocm"
    return True  # cpu


def _pick_precision(device: str, runtime: RuntimeCapabilities, req: DeviceRequest):
    if torch is None:
        return None, "none"
    if device == "cpu":
        return torch.float32, "fp32"
    if req.prefer_fp16 and runtime.supports_fp16:
        return torch.float16, "fp16"
    if req.prefer_bf16 and runtime.supports_bf16:
        return torch.bfloat16, "bf16"
    return torch.float32, "fp32"


def choose_device(req: DeviceRequest, runtime: RuntimeCapabilities | None = None) -> DeviceChoice:
    """Compute the best device for a UDF based on runtime capabilities and request hints."""

    rt = runtime or get_runtime()

    # 1) explicit override
    if req.explicit:
        if _device_available(req.explicit, rt, req):
            dtype, prec = _pick_precision(req.explicit, rt, req)
            return DeviceChoice(req.explicit, dtype, prec, f"explicit={req.explicit}")
        reason = f"explicit={req.explicit} unavailable; falling back"
    else:
        reason = "auto"

    # 2) runtime preferred GPU
    if rt.device_kind == "cuda" and req.allow_gpu:
        dtype, prec = _pick_precision(rt.preferred_device, rt, req)
        return DeviceChoice(rt.preferred_device, dtype, prec, reason + " -> cuda")

    # 3) runtime preferred MPS
    if rt.device_kind == "mps" and req.allow_mps:
        dtype, prec = _pick_precision("mps", rt, req)
        return DeviceChoice("mps", dtype, prec, reason + " -> mps")

    # 4) fallback
    dtype, prec = _pick_precision(req.fallback, rt, req)
    return DeviceChoice(req.fallback, dtype, prec, reason + f" -> fallback={req.fallback}")
