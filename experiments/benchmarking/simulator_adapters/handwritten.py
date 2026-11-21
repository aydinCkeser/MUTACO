"""Handwritten analytic adapter preserving the accepted Prompt 5A science."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
from mutaco_analytic import HandwrittenAnalyticSimulator, analytic_response

from ..contracts import AdapterTargetBank


class HandwrittenAnalyticAdapter:
    identifier = "handwritten_analytic"

    def target_bank(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterTargetBank:
        campaign = config["campaign"]
        count = campaign["target_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("target_count must be a positive integer")
        if campaign["target_bank_policy"] != "scrambled_sobol":
            raise ValueError("unsupported target-bank policy")
        science = ScientificIdentity.create(
            {
                "schema": config["schema_version"],
                "campaign": campaign,
                "simulator": config["simulator"],
                "runner": config["runner"],
                "dtype": config["dtype"],
            }
        )
        family = ScientificIdentity.create(
            {
                "schema": "noisy-cohort-target-bank-family-v1",
                "policy": campaign["target_bank_policy"],
                "campaign_seed": campaign["campaign_seed"],
                "bounds": [[-1.0, -1.0], [1.0, 1.0]],
                "dtype": config["dtype"],
            }
        )
        seed = SeedDeriver(family, campaign["campaign_seed"]).seed_for(
            SeedCoordinate("target_bank", ("scrambled_sobol",))
        )
        engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
        theta = (-1.0 + 2.0 * engine.draw(count, dtype=dtype)).to(device).contiguous()
        if torch.unique(theta, dim=0).shape[0] != count:
            raise RuntimeError("target bank contains duplicates")
        return AdapterTargetBank(
            theta,
            seed,
            science,
            ("theta_1", "theta_2"),
            ((-1.0, -1.0), (1.0, 1.0)),
        )

    def build_simulator(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HandwrittenAnalyticSimulator:
        simulator = config["simulator"]
        return HandwrittenAnalyticSimulator(
            device=device,
            dtype=dtype,
            noise_mode=simulator["noise_mode"],
            noise_std=tuple(simulator["noise_std"]),
        )

    def latent_response(self, theta: torch.Tensor) -> torch.Tensor:
        return analytic_response(theta)

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.benchmarking.simulators.handwritten")
