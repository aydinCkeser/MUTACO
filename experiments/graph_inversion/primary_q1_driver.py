"""Canonical graph binding for the primary NEI Q1 trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import ScientificIdentity, SeedDeriver, stable_hash
from experiments.benchmarking.artifacts import read_json
from experiments.benchmarking.methods.primary_search import run_primary_nei_search
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark.design import generate_primary_sobol_design
from experiments.paper.primary_benchmark.numerical_qualification import (
    evaluate_qualification_snapshot,
)
from experiments.paper.primary_benchmark.qualification import (
    NEIQualificationDeclaration,
)


class GraphPrimaryQ1Driver:
    """Bind generic capped Q1 orchestration to the canonical graph GP surface."""

    def __init__(
        self,
        *,
        declaration: NEIQualificationDeclaration,
        configuration_path: Path,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if device.type != "cpu" or dtype is not torch.float32:
            raise PaperContractError("graph Q1 requires CPU float32")
        self.declaration = declaration
        self.configuration_path = configuration_path.resolve()
        self.configuration = read_json(self.configuration_path)
        simulator = GraphScientificD4Adapter().build_target_simulator(
            self.configuration, device=device, dtype=dtype
        )
        self.parameter_spec = simulator.capabilities.parameter_spec
        self.response_spec = simulator.capabilities.response_spec
        self.device = device
        self.dtype = dtype
        runner = self.configuration.get("runner")
        if not isinstance(runner, Mapping) or not isinstance(runner.get("gp"), Mapping):
            raise PaperContractError("graph Q1 configuration lacks canonical GP settings")
        self.gp_settings = dict(runner["gp"])

    def run_trajectory(
        self,
        target: Mapping[str, Any],
        budget: int,
        tier_id: str,
        snapshots: tuple[int, ...],
        evaluate: Callable[[Sequence[float], int], Sequence[float]],
    ) -> Mapping[str, Any]:
        target_id = str(target.get("target_record_id", ""))
        if target_id not in self.declaration.target_record_identities:
            raise PaperContractError("graph Q1 target is absent from declaration")
        if tier_id != self.declaration.trajectory_tier_id:
            raise PaperContractError("graph Q1 trajectory tier differs from declaration")
        tier = next(item for item in self.declaration.tiers if item.tier_id == tier_id)
        target_physical = torch.tensor(
            target["target_response"], dtype=self.dtype, device=self.device
        ).contiguous()
        if target_physical.shape != (self.response_spec.dimension,):
            raise PaperContractError("graph Q1 target response dimension mismatch")
        design_family = stable_hash("primary-q1-design-family-v1", {
            "campaign_id": self.declaration.campaign_id,
            "target_record_identity": target_id,
            "budget": budget,
            "declaration_identity": self.declaration.document()["declaration_identity"],
        })
        design = generate_primary_sobol_design(
            campaign_id=self.declaration.campaign_id,
            target_id=target_id,
            budget=budget,
            parameter_spec=self.parameter_spec,
            seed_deriver=SeedDeriver(
                ScientificIdentity(design_family), self.declaration.design_campaign_seed
            ),
            dtype=self.dtype,
            device=self.device,
        )
        inference_family = stable_hash("primary-q1-inference-family-v1", {
            "campaign_id": self.declaration.campaign_id,
            "target_record_identity": target_id,
            "budget": budget,
            "method": "nei_bo",
        })
        reference_family = stable_hash("primary-q1-reference-family-v1", {
            "campaign_id": self.declaration.campaign_id,
            "target_record_identity": target_id,
            "budget": budget,
        })
        terminal_family = stable_hash("primary-q1-terminal-family-v1", {
            "campaign_id": self.declaration.campaign_id,
            "target_record_identity": target_id,
            "budget": budget,
        })
        reference_deriver = SeedDeriver(
            ScientificIdentity(reference_family), self.declaration.reference_campaign_seed
        )

        def snapshot_callback(position, surrogate, baseline, responses, transform):
            initial_errors = torch.abs(responses[:16] - target_physical).mean(dim=-1)
            run_scale = float(initial_errors.median().detach().cpu())
            return evaluate_qualification_snapshot(
                declaration=self.declaration,
                surrogate=surrogate,
                baseline=baseline,
                transform=transform,
                target_physical=target_physical,
                parameter_spec=self.parameter_spec,
                seed_deriver=reference_deriver,
                target_record_identity=target_id,
                budget=budget,
                snapshot_position=position,
                run_scale=run_scale,
            )

        result = run_primary_nei_search(
            initial_design=design.shared_prefix,
            budget=budget,
            target_physical=target_physical,
            target_id=target_id,
            target_index=int(target["target_index"]),
            parameter_spec=self.parameter_spec,
            response_spec=self.response_spec,
            evaluate=evaluate,
            inference_seed_deriver=SeedDeriver(
                ScientificIdentity(inference_family),
                self.declaration.inference_campaign_seed,
            ),
            terminal_seed_deriver=SeedDeriver(
                ScientificIdentity(terminal_family), self.declaration.terminal_campaign_seed
            ),
            coordinate_prefix=(
                self.declaration.campaign_id,
                target_id,
                budget,
                "nei_bo",
            ),
            gp_settings=self.gp_settings,
            nei_settings={
                "n_mc": tier.n_mc,
                "raw_samples": tier.raw_candidates,
                "restarts": tier.local_starts,
                "max_iterations": tier.optimization_steps,
                "learning_rate": tier.learning_rate,
            },
            snapshot_positions=snapshots,
            snapshot_callback=snapshot_callback,
        )
        tier_observations = [
            observation
            for snapshot in result.qualification_snapshots
            for observation in snapshot["tier_observations"]
        ]
        return {
            "status": result.status,
            "search_calls": result.successful_calls,
            "theta": result.theta.detach().cpu().tolist(),
            "responses_physical": result.responses_physical.detach().cpu().tolist(),
            "fit_diagnostics": list(result.fit_diagnostics),
            "proposal_diagnostics": list(result.proposal_diagnostics),
            "snapshots": list(result.qualification_snapshots),
            "tier_observations": tier_observations,
            "terminal_recommendation": result.recommendation,
            "failure_stage": result.failure_stage,
            "failure_type": result.failure_type,
            "failure_explanation": result.failure_explanation,
        }
