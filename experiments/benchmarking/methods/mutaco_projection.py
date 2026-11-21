"""Timing-free canonical projection of a MUTACO sequential result."""

from __future__ import annotations

from typing import Any

from mutaco import SequentialRunResult, strict_metadata_document


def scientific_simulator_metadata(metadata: Any) -> dict[str, Any]:
    """Remove observational timings from replay-critical scientific content."""

    document = strict_metadata_document(metadata)
    return {
        key: value
        for key, value in document.items()
        if not key.endswith("_seconds")
    }


def scientific_projection(result: SequentialRunResult) -> dict[str, Any]:
    assert result.initial_design is not None
    assert result.transform is not None
    assert result.standardized_target is not None
    assert result.recommendation is not None
    return {
        "status": result.status.value,
        "scientific_identity": result.scientific_identity.value,
        "initial_design": {
            "design_id": result.initial_design.design_id,
            "observation_ids": result.initial_design.observation_ids,
            "theta": result.initial_design.theta.detach().cpu().tolist(),
        },
        "transform": result.transform.metadata.provenance_document(),
        "standardized_target": result.standardized_target.values.detach().cpu().tolist(),
        "observations": tuple(
            {
                "observation_id": item.observation_id,
                "phase": item.phase.value,
                "logical_index": item.logical_index,
                "theta": item.theta.detach().cpu().tolist(),
                "response_physical": item.response_physical.detach().cpu().tolist(),
                "response_standardized": item.response_standardized.detach().cpu().tolist(),
                "simulator_metadata": scientific_simulator_metadata(
                    item.simulator_metadata
                ),
                "evaluation_seed": item.context.evaluation_seed,
            }
            for item in result.observations
        ),
        "fits": tuple(
            {
                "training_version": item.training_version,
                "fit_id": item.diagnostic.fit_id,
                "training_data_id": item.diagnostic.training_data_id,
                "model_id": item.diagnostic.model_id,
                "configuration": item.diagnostic.configuration,
            }
            for item in result.fits
        ),
        "proposals": tuple(
            {
                "candidate_id": item.candidate_id,
                "candidate": item.candidate.detach().cpu().tolist(),
                "candidate_source": item.candidate_source.value,
                "branch": item.branch.value,
                "branch_reason": item.branch_reason.value,
                "pool_id": item.pool.pool_id,
                "seed_value": item.pool.seed_value,
                "raw_rankings": tuple(
                    {
                        "rank": rank.rank,
                        "raw_index": rank.raw_index,
                        "score": rank.score,
                        "margin": rank.margin,
                    }
                    for rank in item.raw_rankings
                ),
                "posterior_calls": item.posterior_calls.provenance(),
                "duplicate": item.duplicate.provenance() if item.duplicate else None,
            }
            for item in result.proposals
        ),
        "recommendation": {
            "selected_index": getattr(
                result.recommendation,
                "selected_index",
                getattr(result.recommendation, "selected_evaluation_index", None),
            ),
            "selected_observation_id": result.recommendation.selected_observation_id,
            "selected_theta": result.recommendation.selected_theta.detach().cpu().tolist(),
            "score": float(result.recommendation.score.detach().cpu()),
            "posterior_model_id": getattr(
                result.recommendation,
                "posterior_model_id",
                getattr(result.recommendation, "model_id", None),
            ),
        },
        "logical_accounting": result.logical_accounting,
        "physical_count_accounting": {
            key: value
            for key, value in result.physical_accounting.items()
            if not key.endswith("_seconds")
        },
    }
