"""One-sequence shared-prefix Sobol construction for the primary benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mutaco import (
    InitialDesignResult,
    SeedCoordinate,
    SeedDeriver,
    generate_initial_design,
    stable_hash,
)
from mutaco.types import ParameterSpec


@dataclass(frozen=True)
class PrimarySobolDesign:
    full: InitialDesignResult
    shared_prefix: torch.Tensor
    shared_prefix_identity: str
    campaign_id: str
    target_id: str
    budget: int

    def __post_init__(self) -> None:
        if self.full.theta.shape[0] != self.budget:
            raise ValueError("primary Sobol design length must equal budget")
        if self.shared_prefix.shape != self.full.theta[:16].shape or not torch.equal(
            self.shared_prefix, self.full.theta[:16]
        ):
            raise ValueError("primary shared prefix must be the exact first 16 Sobol rows")

    def document(self) -> dict[str, object]:
        return {
            "policy_id": "single_scrambled_sobol_sequence_per_target_budget_v1",
            "campaign_id": self.campaign_id,
            "target_id": self.target_id,
            "budget": self.budget,
            "full_design_id": self.full.design_id,
            "shared_prefix_identity": self.shared_prefix_identity,
            "shared_prefix_length": 16,
            "full_design": self.full.provenance(),
        }


def generate_primary_sobol_design(
    *,
    campaign_id: str,
    target_id: str,
    budget: int,
    parameter_spec: ParameterSpec,
    seed_deriver: SeedDeriver,
    dtype: torch.dtype,
    device: torch.device,
) -> PrimarySobolDesign:
    if not campaign_id.strip() or not target_id.strip():
        raise ValueError("primary Sobol campaign and target identities must be nonempty")
    if budget not in {40, 100}:
        raise ValueError("primary Sobol budget must be 40 or 100")
    coordinate = SeedCoordinate(
        "sobol_design", (campaign_id, target_id, budget, "sobol_design")
    )
    full = generate_initial_design(
        parameter_spec=parameter_spec,
        point_count=budget,
        seed_deriver=seed_deriver,
        seed_coordinate=coordinate,
        dtype=dtype,
        device=device,
        policy="scrambled_sobol",
    )
    prefix = full.theta[:16].clone().contiguous()
    prefix_identity = stable_hash(
        "primary-shared-sobol-prefix-v1",
        {
            "full_design_id": full.design_id,
            "length": 16,
            "dtype": str(dtype),
            "theta": prefix.detach().cpu().tolist(),
        },
    )
    return PrimarySobolDesign(
        full=full,
        shared_prefix=prefix,
        shared_prefix_identity=prefix_identity,
        campaign_id=campaign_id,
        target_id=target_id,
        budget=budget,
    )
