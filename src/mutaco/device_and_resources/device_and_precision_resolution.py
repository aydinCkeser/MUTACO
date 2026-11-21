"""One authoritative Torch device and dtype decision."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Literal

import torch


class DeviceUnavailableError(RuntimeError):
    """The explicitly requested device cannot be used."""


@dataclass(frozen=True)
class RuntimeRequest:
    """User runtime request before hardware resolution."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    cuda_index: int = 0
    dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported runtime device: {self.device!r}")
        if isinstance(self.cuda_index, bool) or not isinstance(self.cuda_index, int):
            raise TypeError("cuda_index must be an integer")
        if self.cuda_index < 0:
            raise ValueError("cuda_index must be nonnegative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"unsupported runtime dtype: {self.dtype!r}")


@dataclass(frozen=True)
class RuntimeDecision:
    """Resolved placement shared by MUTACO and the selected simulator."""

    requested_device: str
    effective_device: str
    dtype_name: str
    automatic_selection: bool
    fallback_reason: str | None
    torch_version: str
    torch_cuda_build: str | None
    torch_cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    platform: str

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.effective_device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype_name]

    def provenance(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "effective_device": self.effective_device,
            "dtype": self.dtype_name,
            "automatic_selection": self.automatic_selection,
            "fallback_reason": self.fallback_reason,
            "torch_version": self.torch_version,
            "torch_cuda_build": self.torch_cuda_build,
            "torch_cuda_available": self.torch_cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "platform": self.platform,
        }


def resolve_runtime(request: RuntimeRequest) -> RuntimeDecision:
    """Resolve `auto`, `cpu`, or strict `cuda` without mutating global state."""

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    fallback_reason: str | None = None
    if request.device == "cpu":
        effective = "cpu"
    elif request.device == "auto" and not available:
        effective = "cpu"
        fallback_reason = "cuda_unavailable"
    else:
        if not available:
            raise DeviceUnavailableError(
                "explicit CUDA was requested, but torch.cuda.is_available() is False"
            )
        if request.cuda_index >= count:
            raise DeviceUnavailableError(
                f"CUDA index {request.cuda_index} was requested, but only {count} device(s) are visible"
            )
        effective = f"cuda:{request.cuda_index}"

    name = torch.cuda.get_device_name(request.cuda_index) if effective.startswith("cuda") else None
    return RuntimeDecision(
        requested_device=request.device,
        effective_device=effective,
        dtype_name=request.dtype,
        automatic_selection=request.device == "auto",
        fallback_reason=fallback_reason,
        torch_version=str(torch.__version__),
        torch_cuda_build=torch.version.cuda,
        torch_cuda_available=available,
        cuda_device_count=count,
        cuda_device_name=name,
        platform=platform.platform(),
    )
