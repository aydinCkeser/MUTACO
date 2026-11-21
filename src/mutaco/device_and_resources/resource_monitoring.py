"""Bounded monitoring that does not touch scientific RNG state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from mutaco.runtime import RuntimeDecision


class MonitoringLevel(str, Enum):
    LIGHTWEIGHT = "lightweight"
    DIAGNOSTIC_PROFILE = "diagnostic_profile"


@dataclass(frozen=True)
class MonitoringConfig:
    level: MonitoringLevel = MonitoringLevel.LIGHTWEIGHT
    progress_every_evaluations: int = 1

    def __post_init__(self) -> None:
        if self.progress_every_evaluations < 1:
            raise ValueError("progress_every_evaluations must be positive")


@dataclass(frozen=True)
class ResourceSnapshot:
    monitoring_level: str
    device: str
    torch_allocated_bytes: int | None
    torch_reserved_bytes: int | None
    torch_peak_allocated_bytes: int | None
    cpu_memory_bytes: int | None
    cpu_memory_source: str


def capture_resources(
    runtime: RuntimeDecision,
    *,
    level: MonitoringLevel = MonitoringLevel.LIGHTWEIGHT,
) -> ResourceSnapshot:
    """Capture bounded allocator state without synchronizing or clearing caches."""

    if runtime.torch_device.type == "cuda":
        index = runtime.torch_device.index or 0
        allocated = int(torch.cuda.memory_allocated(index))
        reserved = int(torch.cuda.memory_reserved(index))
        peak = int(torch.cuda.max_memory_allocated(index))
    else:
        allocated = reserved = peak = None
    return ResourceSnapshot(
        monitoring_level=level.value,
        device=str(runtime.torch_device),
        torch_allocated_bytes=allocated,
        torch_reserved_bytes=reserved,
        torch_peak_allocated_bytes=peak,
        cpu_memory_bytes=None,
        cpu_memory_source="not_collected_in_foundation",
    )
