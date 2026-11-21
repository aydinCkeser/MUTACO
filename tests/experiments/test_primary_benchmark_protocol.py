from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from mutaco import ParameterAxis, ParameterSpec, ScientificIdentity, SeedDeriver
from experiments.paper.primary_benchmark import (
    NEIQualificationDeclaration,
    NEINumericalTier,
    PRIMARY_METHODS,
    PrimaryComparisonConfig,
    PrimaryDecisionOutcome,
    PrimaryMethodDeclaration,
    PrimaryPopulationDeclaration,
    PrimaryTrajectoryEntry,
    execute_primary_target,
    endpoint_results,
    generate_primary_sobol_design,
    load_nei_qualification_declaration,
    load_primary_population_declaration,
    nei_qualification_call_preview,
    primary_population_call_preview,
    paired_endpoint_differences,
    primary_assessment_context,
    runtime_results,
    trajectory_results,
)


def _methods() -> tuple[PrimaryMethodDeclaration, ...]:
    return tuple(
        PrimaryMethodDeclaration(
            method=method,
            configuration_identity=f"{method}-configuration",
            search_policy_identifier={
                "mutaco": "canonical_mutaco",
                "nei_bo": "composite_mc_nei_physical_mae_v1",
                "sobol": "single_scrambled_sobol_sequence_per_target_budget_v1",
            }[method],
            terminal_policy_identifier=(
                "evaluated_best_observed_physical_mae_v1"
                if method == "sobol"
                else "continuous_posterior_mean_physical_l1_mae"
            ),
        )
        for method in PRIMARY_METHODS
    )


def _declaration(budget: int = 40) -> PrimaryPopulationDeclaration:
    return PrimaryPopulationDeclaration(
        campaign_id=f"primary-b{budget}-test",
        comparison=PrimaryComparisonConfig(
            simulator_definition_identity="sim-definition",
            simulator_configuration_identity="sim-configuration",
            assessment_fidelity="sparse-m20",
            design_campaign_seed=101,
            inference_campaign_seed=102,
            terminal_campaign_seed=103,
            assessment_campaign_seed=104,
        ),
        study_identity="paper-study-spec-v1-test",
        target_ledger_hash="ledger-hash",
        target_record_identities=("target-a",),
        budget=budget,
        methods=_methods(),
        output_root=f"outputs/b{budget}",
    )


def test_primary_declaration_roundtrip_and_exact_call_previews() -> None:
    for budget, per_target in ((40, 123), (100, 303)):
        declaration = _declaration(budget)
        document = declaration.document()
        assert load_primary_population_declaration(document) == declaration
        preview = primary_population_call_preview(120, budget)
        assert preview["maximum_total_calls"] == 120 * per_target
        assert preview["terminal_calls"] == 0
        assert preview["scientific_calls_executed"] == 0


def test_primary_assessment_context_is_a_complete_simulator_address() -> None:
    context = primary_assessment_context(_declaration(), "target-a", replicate=2)
    assert context["purpose"] == "paired_held_out_assessment"
    assert context["logical_index"] == 2
    assert context["replicate"] == 2


def test_primary_sobol_uses_one_sequence_and_exact_shared_prefix() -> None:
    parameters = ParameterSpec(
        (ParameterAxis("x", 0.0, 1.0), ParameterAxis("y", -1.0, 1.0))
    )
    deriver = SeedDeriver(ScientificIdentity("primary-design-test"), 91)
    b40 = generate_primary_sobol_design(
        campaign_id="campaign",
        target_id="target-a",
        budget=40,
        parameter_spec=parameters,
        seed_deriver=deriver,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    repeat = generate_primary_sobol_design(
        campaign_id="campaign",
        target_id="target-a",
        budget=40,
        parameter_spec=parameters,
        seed_deriver=deriver,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    b100 = generate_primary_sobol_design(
        campaign_id="campaign",
        target_id="target-a",
        budget=100,
        parameter_spec=parameters,
        seed_deriver=deriver,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert torch.equal(b40.shared_prefix, b40.full.theta[:16])
    assert torch.equal(b40.full.theta, repeat.full.theta)
    assert b40.shared_prefix_identity == repeat.shared_prefix_identity
    assert not torch.equal(b40.shared_prefix, b100.shared_prefix)


def test_primary_target_execution_enforces_pairing_freeze_and_accounting() -> None:
    declaration = _declaration(40)
    parameters = ParameterSpec((ParameterAxis("x", 0.0, 1.0),))
    design = generate_primary_sobol_design(
        campaign_id=declaration.campaign_id,
        target_id="target-a",
        budget=40,
        parameter_spec=parameters,
        seed_deriver=SeedDeriver(
            ScientificIdentity("primary-design-execution-test"),
            declaration.comparison.design_campaign_seed,
        ),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    target = (0.5, 0.5)
    search_events: dict[str, list[tuple[tuple[float, ...], Mapping[str, Any]]]] = {
        method: [] for method in PRIMARY_METHODS
    }
    persisted: list[Mapping[str, Any]] = []
    assessment_contexts: list[Mapping[str, Any]] = []

    def evaluate_search(
        method: str, theta: Sequence[float], context: Mapping[str, Any]
    ) -> Sequence[float]:
        values = tuple(float(item) for item in theta)
        search_events[method].append((values, context))
        return (values[0], 1.0 - values[0])

    def run_method(
        method: PrimaryMethodDeclaration,
        context: Mapping[str, Any],
        shared_design,
        evaluate,
    ) -> PrimaryDecisionOutcome:
        del context
        trajectory = []
        best = float("inf")
        for index in range(1, 41):
            theta = shared_design.full.theta[index - 1]
            response = tuple(float(item) for item in evaluate(theta.tolist(), index))
            error = sum(abs(a - b) for a, b in zip(response, target)) / len(target)
            best = min(best, error)
            trajectory.append(
                PrimaryTrajectoryEntry(
                    evaluation_index=index,
                    theta=tuple(float(item) for item in theta.tolist()),
                    response_physical=response,
                    observed_physical_mae=error,
                    best_observed_mae_so_far=best,
                    simulator_context=search_events[method.method][-1][1],
                    simulator_wall_seconds=0.001,
                    gp_fit_status=(None if method.method == "sobol" else "success"),
                    acquisition_value=(None if index <= 16 or method.method == "sobol" else 0.1),
                    candidate_source=(None if index <= 16 else method.method),
                )
            )
        return PrimaryDecisionOutcome(
            method=method.method,
            status="recommended",
            configuration_identity=method.configuration_identity,
            search_policy_identifier=method.search_policy_identifier,
            terminal_policy_identifier=method.terminal_policy_identifier,
            budget=40,
            attempted_search_calls=40,
            successful_search_calls=40,
            trajectory=tuple(trajectory),
            recommended_theta=tuple(float(item) for item in shared_design.full.theta[0].tolist()),
            terminal_metadata={"simulator_calls": 0, "gp_fits": 0},
            timing={
                "simulator": 0.04,
                "gp_fit": 0.0,
                "acquisition": 0.0,
                "terminal": 0.01,
                "other": 0.95,
                "total": 1.0,
            },
        )

    def persist_freeze(document: Mapping[str, Any]) -> None:
        assert document["assessment_started"] is False
        persisted.append(document)

    def assess(theta: Sequence[float], context: Mapping[str, Any]) -> Sequence[float]:
        assert persisted
        assessment_contexts.append(context)
        return (float(theta[0]), 1.0 - float(theta[0]))

    result = execute_primary_target(
        declaration=declaration,
        target_id="target-a",
        target_response=target,
        design=design,
        run_method=run_method,
        evaluate_search=evaluate_search,
        persist_freeze=persist_freeze,
        evaluate_assessment=assess,
    )
    assert result.physical_calls["attempted"] == 123
    assert result.physical_calls["successful"] == 123
    assert len(result.records) == 3
    assert len(persisted) == 1
    assert all(context == assessment_contexts[0] for context in assessment_contexts)
    assert assessment_contexts[0]["purpose"] == "paired_held_out_assessment"
    assert assessment_contexts[0]["logical_index"] == 0
    search_seeds = {
        method: search_events[method][0][1]["evaluation_seed"]
        for method in PRIMARY_METHODS
    }
    assert len(set(search_seeds.values())) == 3
    initial_locations = {
        method: tuple(event[0] for event in search_events[method][:16])
        for method in PRIMARY_METHODS
    }
    assert initial_locations["mutaco"] == initial_locations["nei_bo"]
    assert initial_locations["mutaco"] == initial_locations["sobol"]
    assert all(record["heldout_physical_mae"] is not None for record in result.records)
    endpoints = endpoint_results(result.records)
    trajectories = trajectory_results(result.records)
    differences = paired_endpoint_differences(result.records)
    runtimes = runtime_results(result.records)
    assert len(endpoints) == len(runtimes) == 3
    assert len(trajectories) == 120
    assert differences[0]["MUTACO_minus_NEI"] == 0.0
    assert differences[0]["MUTACO_minus_Sobol"] == 0.0


def test_nei_qualification_is_zero_assessment_and_roundtrips() -> None:
    declaration = NEIQualificationDeclaration(
        campaign_id="nei-qualification",
        study_identity="study",
        target_ledger_hash="ledger",
        target_record_identities=("a", "b", "c", "d", "e"),
        tiers=(
            NEINumericalTier("low", 64, 32, 2, 4, 0.05),
            NEINumericalTier("high", 128, 64, 4, 8, 0.03),
        ),
        snapshot_positions_b40=(16, 28, 40),
        snapshot_positions_b100=(16, 40, 70, 100),
        output_root="outputs/qualification",
    )
    document = declaration.document()
    assert load_nei_qualification_declaration(document) == declaration
    assert nei_qualification_call_preview() == {
        "target_construction_calls": 0,
        "maximum_search_calls": 700,
        "maximum_assessment_calls": 0,
        "maximum_total_calls": 700,
        "scientific_calls_executed": 0,
    }
