from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.benchmarking.methods.direct_posterior_decision import (
    POSTERIOR_SAMPLE_MAP_POLICY,
    PosteriorSampleMapDecisionError,
    posterior_sample_map_decision,
)
from experiments.benchmarking.methods.direct_posterior_inference import (
    validate_sequential_round_allocation,
)
from experiments.benchmarking.methods.direct_posterior_diagnostics import (
    posterior_diagnostics,
)
from experiments.benchmarking.timing import summarize_sbi_timing
from experiments.paper.common.method_append import (
    append_method_record,
    load_method_record,
)
from experiments.paper.common.protocol_v3 import (
    DecisionOutcome,
    PaperComparisonConfigV3,
    PhysicalCallCounts,
    assess_frozen_decisions,
    freeze_decisions,
    validate_v3_record,
)
from experiments.paper.common.timing_report import summarize_persisted_timings
from experiments.paper.common.validation import PaperContractError


class TwoModePosterior:
    def __init__(self) -> None:
        self.seen: torch.Tensor | None = None
        self.normalization_requests: list[bool] = []

    def log_prob(
        self, theta: torch.Tensor, *, norm_posterior: bool = True
    ) -> torch.Tensor:
        self.seen = theta.detach().clone()
        self.normalization_requests.append(norm_posterior)
        if norm_posterior:
            torch.rand(1)
        left = ((theta - torch.tensor([-1.0, 0.0])) ** 2).sum(dim=1)
        right = ((theta - torch.tensor([1.0, 0.0])) ** 2).sum(dim=1)
        return -torch.minimum(left, right)


class NonfinitePosterior:
    def log_prob(
        self, theta: torch.Tensor, *, norm_posterior: bool = True
    ) -> torch.Tensor:
        del norm_posterior
        return torch.full((theta.shape[0],), float("nan"), dtype=theta.dtype)


def _select(
    posterior,
    samples: torch.Tensor,
    *,
    method: str = "sbi_npe",
):
    return posterior_sample_map_decision(
        method=method,
        posterior=posterior,
        samples=samples,
        lower=torch.tensor([-2.0, -2.0]),
        upper=torch.tensor([2.0, 2.0]),
        method_configuration_identity=f"config-{method}",
        inference_budget=40,
        inference_calls=PhysicalCallCounts(40, 40),
        target_index=7,
    )


def _config() -> PaperComparisonConfigV3:
    return PaperComparisonConfigV3(
        simulator_definition_identity="synthetic-definition-v1",
        simulator_configuration_identity="synthetic-config-v1",
        assessment_fidelity="synthetic-m2-v1",
        assessment_campaign_seed=20,
        inference_campaign_seed=21,
    )


def test_multimodal_sample_map_differs_from_mean_and_uses_physical_candidates():
    posterior = TwoModePosterior()
    samples = torch.tensor(
        [[-1.0, 0.0], [-0.8, 0.0], [0.8, 0.0], [1.0, 0.0]]
    )
    result = _select(posterior, samples)
    assert result.decision.status == "recommended"
    assert result.decision.recommended_theta == (-1.0, 0.0)
    assert result.decision.recommended_theta != tuple(samples.mean(dim=0).tolist())
    assert torch.equal(posterior.seen, samples)
    assert posterior.normalization_requests == [False]
    metadata = result.decision.decision_metadata
    assert metadata is not None
    assert metadata["selected_candidate_index"] == 0
    assert metadata["selected_score"] == 0.0
    assert metadata["transform_jacobian_policy"] == "none"
    assert metadata["refinement"] == "none"
    assert result.timing_event["task"] == "posterior_sample_map_scoring_selection"
    assert result.timing_event["work_count"] == 4
    assert result.timing_event["simulator_calls"] == 0
    assert result.timing_event["budget_classification"] == (
        "zero_call_terminal_recommendation"
    )
    assert result.timing_event["inclusive_seconds"] >= 0.0
    assert result.timing_event["process_cpu_inclusive_seconds"] >= 0.0


def test_map_and_diagnostics_omit_leakage_normalization_and_preserve_ambient_rng():
    posterior = TwoModePosterior()
    samples = torch.tensor(
        [[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]]
    )
    lower = torch.tensor([-2.0, -2.0])
    upper = torch.tensor([2.0, 2.0])
    torch.manual_seed(9127)
    before = torch.random.get_rng_state().clone()

    result = _select(posterior, samples)
    posterior_diagnostics(
        samples,
        lower=lower,
        upper=upper,
        posterior=posterior,
        target_response=torch.zeros(2),
    )

    assert result.decision.status == "recommended"
    assert posterior.normalization_requests == [False, False, False]
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("method", ["sbi_npe", "sbi_snpe"])
def test_npe_and_snpe_share_the_earliest_exact_tie_rule(method: str):
    samples = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.2, 0.0]])
    result = _select(TwoModePosterior(), samples, method=method)
    assert result.decision.recommended_theta == (1.0, 0.0)
    assert result.decision.decision_metadata["selected_candidate_index"] == 0


def test_support_and_nonfinite_candidates_are_removed_without_reordering():
    posterior = TwoModePosterior()
    samples = torch.tensor(
        [[float("nan"), 0.0], [9.0, 0.0], [0.8, 0.0], [1.0, 0.0]]
    )
    result = _select(posterior, samples)
    assert result.decision.recommended_theta == (1.0, 0.0)
    assert result.decision.decision_metadata["selected_candidate_index"] == 3
    assert torch.equal(posterior.seen, samples[2:])


def test_no_finite_score_is_a_typed_zero_call_decision_failure():
    result = _select(NonfinitePosterior(), torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert result.decision.status == "failed"
    assert result.decision.recommended_theta is None
    assert result.decision.failure_type == PosteriorSampleMapDecisionError.__name__
    assert "no finite" in result.decision.failure_explanation
    assert result.decision.decision_metadata["selected_candidate_index"] is None
    assert result.timing_event["status"] == "failed"
    assert result.timing_event["simulator_calls"] == 0

    outside = _select(TwoModePosterior(), torch.tensor([[9.0, 0.0]]))
    assert outside.decision.status == "failed"
    assert outside.decision.failure_type == PosteriorSampleMapDecisionError.__name__
    assert "in-support" in outside.decision.failure_explanation


def test_unapproved_sbi_family_is_rejected_before_policy_execution():
    with pytest.raises(ValueError, match="only for NPE and SNPE"):
        _select(TwoModePosterior(), torch.tensor([[0.0, 0.0]]), method="sbi_nle")


def test_explicit_snpe_round_allocation_is_exact_and_immutable():
    source = [14, 13, 13]
    assert validate_sequential_round_allocation(source, budget=40) == (14, 13, 13)
    source[0] = 1
    assert validate_sequential_round_allocation((14, 13, 13), budget=40) == (
        14,
        13,
        13,
    )


@pytest.mark.parametrize(
    "allocation, message",
    [
        ((), "positive integers"),
        ((14, 13, 12), "sum to the inference budget"),
        ((14, 13, 0, 13), "positive integers"),
        ((14, True, 25), "positive integers"),
    ],
)
def test_invalid_snpe_round_allocations_are_rejected(allocation, message):
    with pytest.raises(ValueError, match=message):
        validate_sequential_round_allocation(allocation, budget=40)


def test_map_metadata_survives_freeze_assessment_reload_validation_and_is_causal(
    tmp_path: Path,
):
    result = _select(TwoModePosterior(), torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    persisted = []
    batch = freeze_decisions(
        _config(), "target-map", [result.decision], persist=persisted.append
    )
    assert persisted[0]["decisions"][0]["decision_metadata"] == (
        result.decision.decision_metadata
    )
    first = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(0.0, 0.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda theta, context: (100.0, -100.0),
    )[0]
    second = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(0.0, 0.0),
        theta_star=(0.0, 0.0),
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda theta, context: (-4.0, 7.0),
        replicate=1,
    )[0]
    assert first["decision"] == second["decision"]
    assert first["recommended_theta"] == second["recommended_theta"] == [-1.0, 0.0]
    assert first["call_ledger"]["terminal_recommendation"]["attempted_calls"] == 0
    validate_v3_record(first)
    path = tmp_path / "sample-map-v3.json"
    append_method_record(path, first)
    loaded = load_method_record(path)
    assert loaded is not None
    assert loaded["decision"]["decision_metadata"] == result.decision.decision_metadata

    malformed = copy.deepcopy(first)
    malformed["decision"]["decision_metadata"]["tie_rule"] = "last"
    with pytest.raises(PaperContractError, match="metadata policy mismatch"):
        validate_v3_record(malformed)
    missing = copy.deepcopy(first)
    missing["decision"]["decision_metadata"] = None
    with pytest.raises(PaperContractError, match="requires decision metadata"):
        validate_v3_record(missing)


def test_failed_map_decision_freezes_and_is_never_assessed():
    result = _select(NonfinitePosterior(), torch.tensor([[0.0, 0.0]]))
    batch = freeze_decisions(
        _config(), "target-failed-map", [result.decision], persist=lambda document: None
    )
    calls = []
    record = assess_frozen_decisions(
        _config(),
        batch,
        target_response=(0.0, 0.0),
        theta_star=None,
        target_construction_counts=PhysicalCallCounts(0, 0),
        target_disposition="reused",
        evaluate=lambda theta, context: calls.append(theta),
    )[0]
    assert calls == []
    assert record["failure_stage"] == "final_decision"
    validate_v3_record(record)


def test_map_timing_is_separate_from_sampling_and_report_maps_common_phases():
    map_result = _select(TwoModePosterior(), torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    sample_event = {
        **map_result.timing_event,
        "event_id": "sample",
        "task": "posterior_sampling",
        "work_count": 2,
        "inclusive_seconds": 2.0,
        "process_cpu_seconds": 1.5,
        "process_cpu_inclusive_seconds": 1.5,
    }
    map_event = {
        **map_result.timing_event,
        "event_id": "map",
        "inclusive_seconds": 0.25,
        "process_cpu_seconds": 0.2,
        "process_cpu_inclusive_seconds": 0.2,
    }
    summary = summarize_sbi_timing(
        [sample_event, map_event],
        method_total_wall_seconds=2.5,
        method_total_process_cpu_seconds=2.0,
        status="completed",
        failure_phase=None,
    )
    report = summarize_persisted_timings(
        [
            {"method": "sbi_npe", "timings": {"total": 2.5, "sbi": summary}},
            {
                "method": "mutaco",
                "timings": {
                    "simulator": 3.0,
                    "training_or_fitting": 4.0,
                    "proposal": 5.0,
                    "recommendation": 0.5,
                    "assessment": 1.0,
                    "total": 13.5,
                },
            },
            {"method": "sbi_historical", "timings": {"total": 55.0}},
        ]
    )
    assert report["schema_version"] == "mutaco-paper-persisted-timing-report-v2"
    npe = report["methods"]["sbi_npe"]
    assert npe["common_phase_times"]["final_decision"]["wall_seconds"] == 0.25
    assert npe["common_phase_times"]["final_decision"]["process_cpu_seconds"] == 0.2
    assert npe["method_specific_phase_times"]["sbi_posterior_sampling"][
        "wall_seconds"
    ] == 2.0
    mutaco = report["methods"]["mutaco"]
    assert mutaco["common_phase_times"]["final_decision"]["wall_seconds"] == 0.5
    assert mutaco["common_phase_times"]["final_decision"][
        "process_cpu_seconds"
    ] is None
    assert mutaco["method_specific_phase_times"]["mutaco_proposal_acquisition"][
        "wall_seconds"
    ] == 5.0
    assert mutaco["phase_decomposition_available"]
    historical = report["methods"]["sbi_historical"]
    assert historical["timing_completeness"] == {"historical_total_only": 1}
    assert historical["common_phase_times"]["final_decision"]["wall_seconds"] is None


def test_legacy_posterior_mean_runner_source_is_unchanged():
    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/benchmarking/methods/sbi_suite.py"
    ).read_text(encoding="utf-8")
    assert "theta_rec = samples.mean(dim=0).contiguous()" in source
    assert POSTERIOR_SAMPLE_MAP_POLICY not in source
