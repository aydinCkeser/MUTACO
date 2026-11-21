"""Second minimal adapter used to prove simulator independence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
from mutaco.simulator import (
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    TensorContractError,
)
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec

from ..contracts import AdapterTargetBank


@dataclass(frozen=True)
class LinearDiagnosticSimulator:
    device: torch.device
    dtype: torch.dtype

    @property
    def capabilities(self) -> SimulatorCapabilities:
        return SimulatorCapabilities(
            ParameterSpec(
                (
                    ParameterAxis("a", -2.0, 2.0),
                    ParameterAxis("b", -1.0, 1.0),
                    ParameterAxis("c", 0.0, 3.0),
                )
            ),
            ResponseSpec((ResponseAxis("sum"), ResponseAxis("contrast"))),
            "linear-diagnostic.v1",
            deterministic=True,
            supports_batch=True,
            definition_metadata={"simulator_name": "linear_diagnostic", "version": 1},
        )

    def evaluate(
        self, theta: torch.Tensor, context: SimulationContext | None = None
    ) -> SimulationResult:
        if theta.device != self.device or theta.dtype != self.dtype:
            raise TensorContractError("linear input device/dtype mismatch")
        values = torch.stack(
            (
                theta[..., 0] + theta[..., 1] + theta[..., 2],
                theta[..., 0] - theta[..., 2],
            ),
            dim=-1,
        ).contiguous()
        return SimulationResult(values, {"simulator_name": "linear_diagnostic"})


class LinearDiagnosticAdapter:
    identifier = "linear_diagnostic"

    def target_bank(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterTargetBank:
        campaign = config["campaign"]
        count = int(campaign["target_count"])
        science = ScientificIdentity.create(
            {
                "schema": config["schema_version"],
                "campaign": campaign,
                "simulator": config["simulator"],
                "runner": config["runner"],
                "dtype": config["dtype"],
            }
        )
        seed = SeedDeriver(science, int(campaign["campaign_seed"])).seed_for(
            SeedCoordinate("target_bank", ("scrambled_sobol",))
        )
        unit = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(
            count, dtype=dtype
        )
        lower = torch.tensor([-2.0, -1.0, 0.0], dtype=dtype)
        upper = torch.tensor([2.0, 1.0, 3.0], dtype=dtype)
        theta = (lower + unit * (upper - lower)).to(device).contiguous()
        return AdapterTargetBank(
            theta,
            seed,
            science,
            ("a", "b", "c"),
            ((-2.0, -1.0, 0.0), (2.0, 1.0, 3.0)),
        )

    def build_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LinearDiagnosticSimulator:
        return LinearDiagnosticSimulator(device, dtype)

    def latent_response(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                theta[..., 0] + theta[..., 1] + theta[..., 2],
                theta[..., 0] - theta[..., 2],
            ),
            dim=-1,
        ).contiguous()

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.benchmarking.simulators.linear")
