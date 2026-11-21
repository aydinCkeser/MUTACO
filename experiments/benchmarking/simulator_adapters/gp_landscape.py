"""Generic benchmark adapter for the declared frozen analytic GP landscape."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, SimulationContext
from mutaco_analytic import (
    FrozenGPLandscapeConfig,
    FrozenGPLandscapeSimulator,
    FrozenGPLandscapeState,
)

from ..contracts import AdapterTargetBank


class FrozenGPLandscapeAdapter:
    """Expose a hash-pinned frozen landscape without changing its mathematics."""

    identifier = "analytic_gp_landscape"

    def _state(self, config: Mapping[str, Any]) -> FrozenGPLandscapeState:
        simulator = config.get("simulator")
        if not isinstance(simulator, Mapping) or simulator.get("adapter") != self.identifier:
            raise ValueError("GP-landscape campaign uses the wrong simulator adapter")
        frozen = simulator.get("frozen_state")
        if not isinstance(frozen, Mapping):
            raise ValueError("GP-landscape adapter requires an explicit frozen_state")
        if frozen.get("construction") != "deterministic_random_features_v1":
            raise ValueError("unsupported GP-landscape frozen-state construction")
        state = FrozenGPLandscapeState.generate(
            FrozenGPLandscapeConfig(
                parameter_dim=int(frozen["parameter_dim"]),
                response_dim=int(frozen["response_dim"]),
                feature_count=int(frozen["feature_count"]),
                latent_rank=int(frozen["latent_rank"]),
                lower=tuple(float(value) for value in frozen["lower"]),
                upper=tuple(float(value) for value in frozen["upper"]),
                lengthscale=float(frozen["lengthscale"]),
                amplitude=float(frozen["amplitude"]),
            ),
            landscape_seed=int(frozen["landscape_seed"]),
        )
        if state.fingerprint != frozen.get("expected_state_hash"):
            raise ValueError("GP-landscape frozen-state hash mismatch")
        self._active_state = state
        return state

    def target_bank(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterTargetBank:
        state = self._state(config)
        campaign = config["campaign"]
        count = campaign["target_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("target_count must be a positive integer")
        if campaign.get("target_bank_policy") != "scrambled_sobol":
            raise ValueError("GP-landscape target bank requires scrambled_sobol")
        science = ScientificIdentity.create(
            {
                "schema": "frozen-gp-landscape-target-family-v1",
                "state_hash": state.fingerprint,
                "target_bank_policy": "scrambled_sobol",
                "campaign_seed": campaign["campaign_seed"],
                "dtype": str(dtype).removeprefix("torch."),
            }
        )
        seed = SeedDeriver(science, int(campaign["campaign_seed"])).seed_for(
            SeedCoordinate("target_bank", ("scrambled_sobol",))
        )
        engine = torch.quasirandom.SobolEngine(
            state.config.parameter_dim, scramble=True, seed=seed
        )
        unit = engine.draw(count, dtype=dtype).to(device)
        lower = torch.tensor(state.config.lower, dtype=dtype, device=device)
        upper = torch.tensor(state.config.upper, dtype=dtype, device=device)
        theta = (lower + (upper - lower) * unit).contiguous()
        return AdapterTargetBank(
            theta=theta,
            seed=seed,
            science=science,
            parameter_names=tuple(
                f"theta_{index + 1}" for index in range(state.config.parameter_dim)
            ),
            bounds=(state.config.lower, state.config.upper),
        )

    def build_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FrozenGPLandscapeSimulator:
        simulator = config["simulator"]
        noise = simulator.get("observation_noise_std")
        return FrozenGPLandscapeSimulator(
            self._state(config),
            device=device,
            dtype=dtype,
            observation_noise_std=(
                tuple(float(value) for value in noise) if noise is not None else None
            ),
        )

    def latent_response(self, theta: torch.Tensor) -> torch.Tensor:
        state = getattr(self, "_active_state", None)
        if state is None:
            raise RuntimeError("GP-landscape state has not been resolved")
        simulator = FrozenGPLandscapeSimulator(
            state, device=theta.device, dtype=theta.dtype
        )
        return simulator.evaluate(
            theta,
            SimulationContext("gp-landscape-latent", "latent", 0, 0, 0),
        ).values

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.benchmarking.simulators.gp_landscape")
