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

from types import SimpleNamespace

import pytest

import databend_aiserver.runtime as runtime


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "_RUNTIME", None, raising=False)


def test_detect_runtime_cpu_when_torch_missing(monkeypatch):
    monkeypatch.setattr(runtime, "torch", None, raising=False)
    result = runtime.detect_runtime()

    assert result.device_kind == "cpu"
    assert result.preferred_device == "cpu"
    assert result.torch_available is False
    assert result.supports_fp16 is False
    assert result.supports_bf16 is False


def test_detect_runtime_disable_gpu_even_if_cuda_available(monkeypatch):
    class _Props:
        total_memory = 8 * 1024 * 1024 * 1024

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(index):
            return _Props()

        @staticmethod
        def is_bf16_supported(index):
            return True

    fake_torch = SimpleNamespace(
        cuda=_Cuda,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float16="f16",
        bfloat16="bf16",
        float32="f32",
    )

    monkeypatch.setattr(runtime, "torch", fake_torch, raising=False)

    result = runtime.detect_runtime(disable_gpu=True)

    assert result.device_kind == "cpu"
    assert result.preferred_device == "cpu"
    assert result.torch_available is True
    assert result.supports_fp16 is False
    assert result.supports_bf16 is False


def test_choose_device_explicit_cpu_overrides_gpu(monkeypatch):
    fake_torch = SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32")
    monkeypatch.setattr(runtime, "torch", fake_torch, raising=False)

    rt = runtime.RuntimeCapabilities(
        device_kind="cuda",
        preferred_device="cuda:0",
        visible_devices=["cuda:0"],
        memory_mb=8192,
        torch_available=True,
        supports_fp16=True,
        supports_bf16=True,
        onnx_providers=[],
        timestamp=runtime.datetime.utcnow(),
    )

    choice = runtime.choose_device(runtime.DeviceRequest(task="embedding", explicit="cpu"), runtime=rt)

    assert choice.device == "cpu"
    assert choice.precision == "fp32"
    assert "explicit" in choice.reason


def test_choose_device_prefers_cuda_with_fp16(monkeypatch):
    fake_torch = SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32")
    monkeypatch.setattr(runtime, "torch", fake_torch, raising=False)

    rt = runtime.RuntimeCapabilities(
        device_kind="cuda",
        preferred_device="cuda:0",
        visible_devices=["cuda:0"],
        memory_mb=8192,
        torch_available=True,
        supports_fp16=True,
        supports_bf16=False,
        onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        timestamp=runtime.datetime.utcnow(),
    )

    choice = runtime.choose_device(runtime.DeviceRequest(task="docling"), runtime=rt)

    assert choice.device == "cuda:0"
    assert choice.precision == "fp16"
    assert "cuda" in choice.reason
