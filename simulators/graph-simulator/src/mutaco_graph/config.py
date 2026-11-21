"""Validated graph simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .profiles import SCIENTIFIC_IDENTITY, SCIENTIFIC_M20, parameter_profile


@dataclass(frozen=True)
class GraphSimulationConfig:
    model_identity: str = SCIENTIFIC_IDENTITY
    parameter_profile_id: str = "p4-paper-baseline"
    response_profile_id: str = SCIENTIFIC_M20.identifier
    node_count: int = 8
    batch_size: int = 4
    repeat_size: int = 2
    temperature: float = 0.15
    dt: float = 0.1
    burn_in_steps: int = 100
    sample_steps: int = 900
    sample_stride: int = 1
    device: str = "auto"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        parameter_profile(self.parameter_profile_id)
        if self.response_profile_id != SCIENTIFIC_M20.identifier:
            raise ValueError("scientific configuration requires the canonical M=20 profile")
        for name in ("node_count", "batch_size", "repeat_size", "sample_steps", "sample_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.burn_in_steps, bool) or self.burn_in_steps < 0:
            raise ValueError("burn_in_steps must be a nonnegative integer")
        if self.dtype is not torch.float32:
            raise ValueError("the approved scientific model requires torch.float32")
        if self.dt <= 0 or self.temperature < 0:
            raise ValueError("dt must be positive and temperature nonnegative")

    @property
    def total_steps(self) -> int:
        return self.burn_in_steps + self.sample_steps

    @property
    def sampled_step_count(self) -> int:
        return (self.sample_steps - 1) // self.sample_stride + 1

    def identity_document(self) -> dict[str, object]:
        return {
            "model_identity": self.model_identity,
            "parameter_profile": self.parameter_profile_id,
            "response_profile": self.response_profile_id,
            "node_count": self.node_count,
            "batch_size": self.batch_size,
            "repeat_size": self.repeat_size,
            "temperature": self.temperature,
            "dt": self.dt,
            "burn_in_steps": self.burn_in_steps,
            "sample_steps": self.sample_steps,
            "sample_stride": self.sample_stride,
            "total_steps": self.total_steps,
            "dtype": "float32",
        }
