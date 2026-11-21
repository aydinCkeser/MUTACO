"""Graph binding for the prequalification three-method primary-path smoke."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import (
    RunnerStatus,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    stable_hash,
)
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods.mutaco_reference import run_mutaco_target
from experiments.benchmarking.methods.primary_search import (
    PrimarySearchResult,
    primary_decision_timing,
    run_primary_nei_search,
    run_primary_scalar_qlognei_search,
    run_primary_sobol_search,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.paper_v3_callbacks import _CallbackGraphAdapter
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    PrimaryDecisionOutcome,
    PrimaryMethodDeclaration,
    PrimarySobolDesign,
    PrimaryTrajectoryEntry,
)


SMOKE_NEI_TIER_ID = "tier_high_provisional_engineering_only"
SMOKE_NEI_SETTINGS: dict[str, int | float] = {
    "n_mc": 256,
    "raw_samples": 256,
    "restarts": 12,
    "max_iterations": 24,
    "learning_rate": 0.03,
}


def scalar_primary_method_identities(
    configuration_path: Path,
    *,
    budget: int,
    qlognei_settings: Mapping[str, Any],
) -> dict[str, str]:
    """Return identities for the scalar-qLogNEI comparison family."""

    configuration = read_json(configuration_path)
    return {
        "mutaco": stable_hash(
            "scalar-primary-mutaco-configuration-v1",
            {
                "configuration_sha256": file_hash(configuration_path),
                "budget": budget,
                "profile_id": configuration["graph_paper_policy"]["profile_id"],
            },
        ),
        "scalar_qlognei": stable_hash(
            "scalar-primary-qlognei-configuration-v1",
            {
                "settings": dict(qlognei_settings),
                "model": "botorch.SingleTaskGP",
                "objective": "negative_observed_physical_response_l1_mae",
                "acquisition": "botorch.qLogNoisyExpectedImprovement",
                "batch_q": 1,
                "dtype": "float32",
                "precision_fallback": "forbidden",
                "budget": budget,
            },
        ),
        "sobol": stable_hash(
            "scalar-primary-sobol-configuration-v1",
            {
                "design": "single_scrambled_sobol_sequence_per_target_budget_v1",
                "terminal": "evaluated_best_observed_physical_mae_v1",
                "budget": budget,
            },
        ),
    }


def primary_smoke_method_identities(configuration_path: Path) -> dict[str, str]:
    """Return target-independent method identities for the declared smoke."""

    configuration = read_json(configuration_path)
    gp = dict(configuration["runner"]["gp"])
    return {
        "mutaco": stable_hash(
            "primary-smoke-mutaco-configuration-v1",
            {
                "configuration_sha256": file_hash(configuration_path),
                "budget": 40,
                "profile_id": configuration["graph_paper_policy"]["profile_id"],
            },
        ),
        "nei_bo": stable_hash(
            "primary-smoke-nei-configuration-v1",
            {
                "gp": gp,
                "tier_id": SMOKE_NEI_TIER_ID,
                "settings": SMOKE_NEI_SETTINGS,
                "baseline_factorization": "promoted_float64_cholesky_zero_jitter_v1",
                "budget": 40,
            },
        ),
        "sobol": stable_hash(
            "primary-smoke-sobol-configuration-v1",
            {
                "design": "single_scrambled_sobol_sequence_per_target_budget_v1",
                "terminal": "evaluated_best_observed_physical_mae_v1",
                "budget": 40,
            },
        ),
    }


class GraphPrimaryIntegrationSmokeDriver:
    """Execute all three methods through the generic primary target boundary."""

    def __init__(
        self,
        *,
        declaration: Any,
        configuration_path: Path,
        qlognei_settings: Mapping[str, Any] | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if device.type != "cpu" or dtype is not torch.float32:
            raise PaperContractError("primary graph smoke requires CPU float32")
        if declaration.budget not in {40, 100} or not declaration.target_record_identities:
            raise PaperContractError("primary graph driver requires B40/B100 targets")
        self.declaration = declaration
        self.configuration_path = configuration_path.resolve()
        self.configuration = read_json(self.configuration_path)
        if int(self.configuration.get("simulation_budget_per_target", -1)) != declaration.budget:
            raise PaperContractError("primary graph driver config budget mismatch")
        simulator = GraphScientificD4Adapter().build_target_simulator(
            self.configuration, device=device, dtype=dtype
        )
        self.capabilities = simulator.capabilities
        if (
            declaration.comparison.simulator_definition_identity
            != simulator.capabilities.simulator_identity
            or declaration.comparison.simulator_configuration_identity
            != file_hash(self.configuration_path)
        ):
            raise PaperContractError("primary graph driver simulator identity mismatch")
        self.parameter_spec = simulator.capabilities.parameter_spec
        self.response_spec = simulator.capabilities.response_spec
        self.device = device
        self.dtype = dtype
        declared_methods = tuple(item.method for item in declaration.methods)
        self.qlognei_settings = (
            None if qlognei_settings is None else dict(qlognei_settings)
        )
        if declared_methods == ("mutaco", "nei_bo", "sobol"):
            identities = primary_smoke_method_identities(self.configuration_path)
        elif declared_methods == ("mutaco", "scalar_qlognei", "sobol"):
            if self.qlognei_settings is None:
                raise PaperContractError("scalar primary driver requires qLogNEI settings")
            identities = scalar_primary_method_identities(
                self.configuration_path,
                budget=declaration.budget,
                qlognei_settings=self.qlognei_settings,
            )
        else:
            raise PaperContractError("primary graph driver method family mismatch")
        if {
            item.method: item.configuration_identity for item in declaration.methods
        } != identities:
            raise PaperContractError("primary graph driver method identities mismatch")

    def _terminal_deriver(self, target_id: str) -> SeedDeriver:
        family = stable_hash(
            "primary-terminal-family-v1",
            {
                "protocol_id": self.declaration.comparison.protocol_id,
                "campaign_id": self.declaration.campaign_id,
                "study_identity": self.declaration.study_identity,
                "target_id": target_id,
                "budget": self.declaration.budget,
                "comparison_identity": self.declaration.comparison.identity,
            },
        )
        return SeedDeriver(
            ScientificIdentity(family), self.declaration.comparison.terminal_campaign_seed
        )

    def _simulator_context(
        self,
        method: str,
        target_id: str,
        inference_context: Mapping[str, Any],
        evaluation_index: int,
    ) -> dict[str, Any]:
        coordinate = SeedCoordinate(
            "simulation",
            (
                self.declaration.campaign_id,
                target_id,
                self.declaration.budget,
                method,
                evaluation_index,
                "simulation",
            ),
        )
        deriver = SeedDeriver(
            ScientificIdentity(str(inference_context["family_identity"])),
            int(inference_context["root_seed"]),
        )
        return {
            "family_identity": inference_context["family_identity"],
            "purpose": "primary_search_simulation",
            "logical_index": evaluation_index - 1,
            "evaluation_index": evaluation_index,
            "evaluation_seed": deriver.seed_for(coordinate),
            "coordinate": {
                "namespace": coordinate.namespace,
                "coordinates": list(coordinate.coordinates),
            },
        }

    def _trajectory(
        self,
        *,
        method: str,
        target_id: str,
        inference_context: Mapping[str, Any],
        theta: torch.Tensor,
        responses: torch.Tensor,
        target_response: torch.Tensor,
        evaluation_seconds: Sequence[float],
        fits: Sequence[Mapping[str, Any]],
        proposals: Sequence[Mapping[str, Any]],
    ) -> tuple[PrimaryTrajectoryEntry, ...]:
        best = float("inf")
        rows = []
        for offset, (theta_row, response) in enumerate(zip(theta, responses)):
            index = offset + 1
            error = float((response - target_response).abs().mean().detach().cpu())
            best = min(best, error)
            fit = None if index < 16 or not fits else fits[index - 16]
            proposal = None if index <= 16 or not proposals else proposals[index - 17]
            acquisition_value = None
            acquisition_seconds = None
            candidate_source = None
            if proposal is not None:
                raw_value = proposal.get("value")
                acquisition_value = (
                    float(raw_value) if isinstance(raw_value, (int, float)) else None
                )
                raw_seconds = proposal.get("elapsed_seconds")
                if raw_seconds is None and isinstance(proposal.get("timings"), Mapping):
                    raw_seconds = proposal["timings"].get("total_seconds")
                acquisition_seconds = (
                    float(raw_seconds)
                    if isinstance(raw_seconds, (int, float))
                    else None
                )
                candidate_source = str(proposal.get("candidate_source"))
            rows.append(
                PrimaryTrajectoryEntry(
                    evaluation_index=index,
                    theta=tuple(float(value) for value in theta_row.detach().cpu()),
                    response_physical=tuple(
                        float(value) for value in response.detach().cpu()
                    ),
                    observed_physical_mae=error,
                    best_observed_mae_so_far=best,
                    simulator_context=self._simulator_context(
                        method, target_id, inference_context, index
                    ),
                    simulator_wall_seconds=float(evaluation_seconds[offset]),
                    gp_fit_status=(None if fit is None else str(fit.get("status"))),
                    gp_fit_seconds=(
                        None if fit is None else float(fit.get("elapsed_seconds", 0.0))
                    ),
                    acquisition_value=acquisition_value,
                    acquisition_seconds=acquisition_seconds,
                    candidate_source=candidate_source,
                    acquisition_diagnostics=proposal,
                )
            )
        return tuple(rows)

    def _outcome_from_primary_search(
        self,
        *,
        method: PrimaryMethodDeclaration,
        target_id: str,
        inference_context: Mapping[str, Any],
        target_response: torch.Tensor,
        result: PrimarySearchResult,
        evaluation_seconds: Sequence[float],
    ) -> PrimaryDecisionOutcome:
        trajectory = self._trajectory(
            method=method.method,
            target_id=target_id,
            inference_context=inference_context,
            theta=result.theta,
            responses=result.responses_physical,
            target_response=target_response,
            evaluation_seconds=evaluation_seconds,
            fits=result.fit_diagnostics,
            proposals=result.proposal_diagnostics,
        )
        recommended = (
            None
            if result.recommendation is None
            else tuple(float(value) for value in result.recommendation["selected_theta"])
        )
        return PrimaryDecisionOutcome(
            method=method.method,
            status=result.status,
            configuration_identity=method.configuration_identity,
            search_policy_identifier=method.search_policy_identifier,
            terminal_policy_identifier=method.terminal_policy_identifier,
            budget=self.declaration.budget,
            attempted_search_calls=result.successful_calls,
            successful_search_calls=result.successful_calls,
            trajectory=trajectory,
            recommended_theta=recommended,
            terminal_metadata=result.recommendation,
            timing=dict(result.timing),
            failure_stage=result.failure_stage,
            failure_type=result.failure_type,
            failure_explanation=result.failure_explanation,
        )

    def run_method(
        self,
        target: Mapping[str, Any],
        method: PrimaryMethodDeclaration,
        inference_context: Mapping[str, Any],
        design: PrimarySobolDesign,
        evaluate: Callable[[Sequence[float], int], Sequence[float]],
    ) -> PrimaryDecisionOutcome:
        target_id = str(target["target_record_id"])
        target_response = torch.tensor(
            target["target_response"], device=self.device, dtype=self.dtype
        ).contiguous()
        evaluation_seconds: list[float] = []

        def timed_evaluate(theta: Sequence[float], index: int) -> Sequence[float]:
            started = perf_counter()
            try:
                return evaluate(theta, index)
            finally:
                evaluation_seconds.append(perf_counter() - started)

        terminal_deriver = self._terminal_deriver(target_id)
        if method.method == "nei_bo":
            result = run_primary_nei_search(
                initial_design=design.shared_prefix,
                budget=self.declaration.budget,
                target_physical=target_response,
                target_id=target_id,
                target_index=int(target["target_index"]),
                parameter_spec=self.parameter_spec,
                response_spec=self.response_spec,
                evaluate=timed_evaluate,
                inference_seed_deriver=SeedDeriver(
                    ScientificIdentity(str(inference_context["family_identity"])),
                    int(inference_context["root_seed"]),
                ),
                terminal_seed_deriver=terminal_deriver,
                coordinate_prefix=(
                    self.declaration.campaign_id,
                    target_id,
                    self.declaration.budget,
                    "nei_bo",
                ),
                gp_settings=dict(self.configuration["runner"]["gp"]),
                nei_settings=SMOKE_NEI_SETTINGS,
            )
            return self._outcome_from_primary_search(
                method=method,
                target_id=target_id,
                inference_context=inference_context,
                target_response=target_response,
                result=result,
                evaluation_seconds=evaluation_seconds,
            )
        if method.method == "scalar_qlognei":
            assert self.qlognei_settings is not None
            result = run_primary_scalar_qlognei_search(
                initial_design=design.shared_prefix,
                budget=self.declaration.budget,
                target_physical=target_response,
                target_id=target_id,
                target_index=int(target["target_index"]),
                parameter_spec=self.parameter_spec,
                response_spec=self.response_spec,
                evaluate=timed_evaluate,
                inference_seed_deriver=SeedDeriver(
                    ScientificIdentity(str(inference_context["family_identity"])),
                    int(inference_context["root_seed"]),
                ),
                terminal_seed_deriver=terminal_deriver,
                coordinate_prefix=(
                    self.declaration.campaign_id,
                    target_id,
                    self.declaration.budget,
                    "scalar_qlognei",
                ),
                qlognei_settings=self.qlognei_settings,
            )
            return self._outcome_from_primary_search(
                method=method,
                target_id=target_id,
                inference_context=inference_context,
                target_response=target_response,
                result=result,
                evaluation_seconds=evaluation_seconds,
            )
        if method.method == "sobol":
            result = run_primary_sobol_search(
                design=design.full.theta,
                target_physical=target_response,
                response_spec=self.response_spec,
                evaluate=timed_evaluate,
            )
            return self._outcome_from_primary_search(
                method=method,
                target_id=target_id,
                inference_context=inference_context,
                target_response=target_response,
                result=result,
                evaluation_seconds=evaluation_seconds,
            )
        if method.method != "mutaco":
            raise PaperContractError("primary graph smoke received an unknown method")

        counter = 0
        captured = []

        def callback(theta: Sequence[float], context: Mapping[str, Any]) -> Sequence[float]:
            nonlocal counter
            del context
            counter += 1
            return timed_evaluate(theta, counter)

        adapter = _CallbackGraphAdapter(
            self.capabilities,
            callback,
            device=self.device,
            dtype=self.dtype,
        )
        run_seed = SeedDeriver(
            ScientificIdentity(str(inference_context["family_identity"])),
            int(inference_context["root_seed"]),
        ).seed_for(
            SeedCoordinate(
                "mutaco_run",
                (
                    self.declaration.campaign_id,
                    target_id,
                    self.declaration.budget,
                    "mutaco",
                    "mutaco_run",
                ),
            )
        )
        row = run_mutaco_target(
            adapter=adapter,
            index=int(target["target_index"]),
            theta_star=torch.tensor(
                target["theta_star"], device=self.device, dtype=self.dtype
            ).contiguous(),
            config=self.configuration,
            campaign_science=ScientificIdentity(
                str(target.get("target_family_id", target_id))
            ),
            device=self.device,
            dtype=self.dtype,
            supplied_target_response=target_response,
            supplied_target_seed=int(target["target_generation_seed"]),
            supplied_target_identity=target_id,
            supplied_initial_points=design.shared_prefix,
            perform_held_out_assessment=False,
            diagnostic_result_callback=captured.append,
            supplied_run_seed=run_seed,
            supplied_terminal_seed_deriver=terminal_deriver,
        )
        if len(captured) != 1:
            raise PaperContractError("MUTACO smoke did not expose exactly one run result")
        run = captured[0]
        fits = tuple(item.provenance() for item in run.fits)
        proposals = tuple(item.provenance() for item in run.proposals)
        theta = torch.stack([item.theta for item in run.observations]).contiguous()
        responses = torch.stack(
            [item.response_physical for item in run.observations]
        ).contiguous()
        trajectory = self._trajectory(
            method="mutaco",
            target_id=target_id,
            inference_context=inference_context,
            theta=theta,
            responses=responses,
            target_response=target_response,
            evaluation_seconds=evaluation_seconds,
            fits=fits,
            proposals=proposals,
        )
        recommended = (
            tuple(float(value) for value in row["recommended_theta"])
            if run.status is RunnerStatus.COMPLETED
            else None
        )
        terminal = (
            dict(row["graph_terminal_recommendation"])
            if run.status is RunnerStatus.COMPLETED
            else None
        )
        terminal_seconds = (
            0.0 if terminal is None else float(terminal.get("elapsed_seconds", 0.0))
        )
        timing = primary_decision_timing(
            total=float(row["runtime_seconds"]),
            simulator=float(sum(evaluation_seconds)),
            gp_fit=sum(float(item.get("elapsed_seconds", 0.0)) for item in fits),
            acquisition=sum(
                float(item.get("timings", {}).get("total_seconds", 0.0))
                for item in proposals
            ),
            terminal=terminal_seconds,
        )
        return PrimaryDecisionOutcome(
            method="mutaco",
            status=("recommended" if run.status is RunnerStatus.COMPLETED else "failed"),
            configuration_identity=method.configuration_identity,
            search_policy_identifier=method.search_policy_identifier,
            terminal_policy_identifier=method.terminal_policy_identifier,
            budget=self.declaration.budget,
            attempted_search_calls=len(evaluation_seconds),
            successful_search_calls=len(run.observations),
            trajectory=trajectory,
            recommended_theta=recommended,
            terminal_metadata=terminal,
            timing=timing,
            failure_stage=(None if recommended is not None else "method_execution"),
            failure_type=(None if recommended is not None else "MutacoSmokeFailure"),
            failure_explanation=run.failure_explanation,
        )
