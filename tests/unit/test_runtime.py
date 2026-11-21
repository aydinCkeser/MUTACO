from __future__ import annotations

import pytest
import torch

from mutaco.runtime import DeviceUnavailableError, RuntimeRequest, resolve_runtime


def test_auto_visibly_falls_back_to_torch_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    decision = resolve_runtime(RuntimeRequest())
    assert decision.requested_device == "auto"
    assert decision.effective_device == "cpu"
    assert decision.torch_device == torch.device("cpu")
    assert decision.torch_dtype == torch.float64
    assert decision.fallback_reason == "cuda_unavailable"
    assert decision.provenance()["fallback_reason"] == "cuda_unavailable"


def test_explicit_unavailable_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="explicit CUDA"):
        resolve_runtime(RuntimeRequest(device="cuda"))


def test_cpu_is_explicit_and_has_no_fallback_reason() -> None:
    decision = resolve_runtime(RuntimeRequest(device="cpu", dtype="float32"))
    assert decision.effective_device == "cpu"
    assert decision.torch_dtype == torch.float32
    assert decision.fallback_reason is None
    assert decision.automatic_selection is False


def test_runtime_request_is_strict() -> None:
    with pytest.raises(ValueError, match="unsupported runtime device"):
        RuntimeRequest(device="mps")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        RuntimeRequest(cuda_index=-1)
    with pytest.raises(ValueError, match="unsupported runtime dtype"):
        RuntimeRequest(dtype="float16")  # type: ignore[arg-type]


@pytest.mark.cuda
def test_auto_selects_cuda_when_real_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("compatible Torch CUDA runtime is unavailable")
    decision = resolve_runtime(RuntimeRequest(device="auto"))
    assert decision.torch_device.type == "cuda"
    assert decision.fallback_reason is None
    assert decision.cuda_device_name
