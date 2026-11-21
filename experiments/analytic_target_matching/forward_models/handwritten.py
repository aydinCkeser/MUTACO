"""Faithful Torch port of the historical transparent analytic map."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch

from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
)
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec


NoiseMode = Literal["deterministic", "task_diagonal_gaussian"]
THETA_STAR = (0.35, -0.40)
DEFAULT_NOISE_STD = (0.04, 0.06, 0.05)
SIMULATOR_VERSION = "2"


def analytic_response(
    theta: torch.Tensor,
    *,
    constant_output: float | None = None,
) -> torch.Tensor:
    """Evaluate a transparent nonlinear reference map in declared response order.

    For theta=(a,b), the three channels are (a+b^2/2, sin(pi*a)+b,
    a*b+a^2/4). Accept [2] or [H,2] and return [3] or [H,3] in the input dtype/device;
    optional constant_output appends a fourth channel to exercise scale floors.
    This function adds no noise and does not itself enforce simulator box bounds.
    """

    if not isinstance(theta, torch.Tensor) or theta.ndim not in {1, 2}:
        raise TensorContractError("analytic theta must be a Torch tensor with shape [2] or [H,2]")
    if theta.shape[-1] != 2:
        raise TensorContractError("analytic theta has parameter dimension 2")
    if not theta.is_floating_point() or not bool(torch.isfinite(theta).all()):
        raise TensorContractError("analytic theta must be finite and floating point")
    theta_1, theta_2 = theta[..., 0], theta[..., 1]
    response = torch.stack(
        (
            theta_1 + 0.5 * theta_2.square(),
            torch.sin(torch.pi * theta_1) + theta_2,
            theta_1 * theta_2 + 0.25 * theta_1.square(),
        ),
        dim=-1,
    )
    if constant_output is not None:
        if not math.isfinite(constant_output):
            raise ValueError("constant_output must be finite")
        constant = torch.full_like(response[..., :1], float(constant_output))
        response = torch.cat((response, constant), dim=-1)
    return response.contiguous()


@dataclass(frozen=True)
class HandwrittenAnalyticSimulator:
    """Two-parameter reference map on [-1,1]^2 with optional task-diagonal noise.

    Response order is f_1, f_2, f_3, then the optional constant channel. In noisy
    mode only the first three channels receive independent Gaussian noise from
    a context-owned generator; the constant channel remains exactly constant.
    Evaluation preserves configured float32/float64 and device without advancing
    the ambient RNG. This controlled analytic map is not a graph-physics model.
    """

    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    noise_mode: NoiseMode = "deterministic"
    noise_std: tuple[float, float, float] = DEFAULT_NOISE_STD
    constant_output: float | None = None
    _noise_scale: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.noise_mode not in {"deterministic", "task_diagonal_gaussian"}:
            raise ValueError(f"unsupported analytic noise mode: {self.noise_mode!r}")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("analytic simulator requires float32 or float64")
        if len(self.noise_std) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.noise_std
        ):
            raise ValueError("noise_std must contain three finite positive values")
        if self.constant_output is not None and not math.isfinite(self.constant_output):
            raise ValueError("constant_output must be finite")
        object.__setattr__(
            self,
            "_noise_scale",
            torch.tensor(self.noise_std, dtype=self.dtype, device=self.device),
        )

    @property
    def capabilities(self) -> SimulatorCapabilities:
        parameters = ParameterSpec(
            (
                ParameterAxis("theta_1", -1.0, 1.0),
                ParameterAxis("theta_2", -1.0, 1.0),
            )
        )
        responses = (
            ResponseAxis("f_1"),
            ResponseAxis("f_2"),
            ResponseAxis("f_3"),
        ) + (
            (ResponseAxis("f_4_constant"),) if self.constant_output is not None else ()
        )
        identity = f"analytic-handwritten.v{SIMULATOR_VERSION}.{self.noise_mode}"
        if self.constant_output is not None:
            identity += f".constant-{self.constant_output!r}"
        return SimulatorCapabilities(
            parameter_spec=parameters,
            response_spec=ResponseSpec(responses),
            simulator_identity=identity,
            deterministic=self.noise_mode == "deterministic",
            supports_batch=True,
            supports_context=True,
            definition_metadata={
                "simulator_name": "analytic_handwritten",
                "simulator_version": SIMULATOR_VERSION,
                "noise_mode": self.noise_mode,
                "noise_std": list(self.noise_std),
                "constant_output": self.constant_output,
            },
        )

    def evaluate(
        self,
        theta: torch.Tensor,
        context: SimulationContext | None = None,
    ) -> SimulationResult:
        # The simulator keeps its original placement when the current GPU changes.
        device = self._noise_scale.device if self.device == torch.device("cuda") else self.device
        if theta.device != device or theta.dtype != self.dtype:
            raise TensorContractError("analytic input does not match simulator device and dtype")
        latent = analytic_response(theta, constant_output=self.constant_output)
        seed: int | None = None
        if self.noise_mode == "task_diagonal_gaussian":
            if context is None:
                raise ValueError("noisy analytic evaluation requires a SimulationContext")
            seed = context.evaluation_seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn(
                latent[..., :3].shape,
                dtype=self.dtype,
                device=device,
                generator=generator,
            )
            response = torch.cat(
                (latent[..., :3] + noise * self._noise_scale, latent[..., 3:]),
                dim=-1,
            )
        else:
            response = latent.clone()
        return SimulationResult(
            response.contiguous(),
            {
                "simulator_name": "analytic_handwritten",
                "simulator_version": SIMULATOR_VERSION,
                "noise_mode": self.noise_mode,
                "evaluation_seed": seed,
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
                "transport_call_count": 1,
            },
        )
