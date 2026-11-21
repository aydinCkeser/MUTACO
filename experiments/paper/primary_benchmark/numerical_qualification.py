"""Deterministic zero-call numerical comparison at frozen NEI GP snapshots."""

from __future__ import annotations

from typing import Any

import torch

from mutaco import (
    FrozenResponseTransform,
    NEIConfig,
    ParameterSpec,
    SeedCoordinate,
    SeedDeriver,
    evaluate_composite_mc_nei_candidates,
    propose_composite_mc_nei,
    stable_hash,
)
from experiments.benchmarking.artifacts import canonicalize_evidence
from .qualification import NEIQualificationDeclaration, NEINumericalTier


def _canonical_numerical_document(value: Any) -> Any:
    """Compatibility alias for the shared evidence-boundary canonicalizer."""

    return canonicalize_evidence(value)


def _config(
    tier: NEINumericalTier,
    deriver: SeedDeriver,
    prefix: tuple[str | int, ...],
    *,
    mc_suffix: tuple[str | int, ...] = ("optimization", "nei_mc"),
) -> NEIConfig:
    return NEIConfig(
        n_mc=tier.n_mc,
        raw_samples=tier.raw_candidates,
        restarts=tier.local_starts,
        max_iterations=tier.optimization_steps,
        learning_rate=tier.learning_rate,
        seed_deriver=deriver,
        raw_coordinate=SeedCoordinate("nei_raw", (*prefix, "nei_raw")),
        mc_coordinate=SeedCoordinate("nei_mc", (*prefix, *mc_suffix)),
        optimizer_coordinate=SeedCoordinate(
            "nei_optimizer", (*prefix, "nei_optimizer")
        ),
    )


def _snapshot_state(surrogate: Any) -> dict[str, Any]:
    state = surrogate.warm_start_state()
    parameters = {
        name: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in sorted(state.parameter_state.items())
    }
    body = _canonical_numerical_document({
        "source_fit_id": state.source_fit_id,
        "source_model_id": state.source_model_id,
        "compatibility_id": state.compatibility_id,
        "consecutive_fixed_conditioning_updates": (
            state.consecutive_fixed_conditioning_updates
        ),
        "source_attempt_kind": getattr(
            state.source_attempt_kind, "value", state.source_attempt_kind
        ),
        "parameter_state": parameters,
        "nonfinite_float_encoding": "explicit_signed_infinity_v1",
    })
    return {
        **body,
        "snapshot_state_identity": stable_hash("nei-q1-gp-snapshot-state-v1", body),
    }


def evaluate_qualification_snapshot(
    *,
    declaration: NEIQualificationDeclaration,
    surrogate: Any,
    baseline: torch.Tensor,
    transform: FrozenResponseTransform,
    target_physical: torch.Tensor,
    parameter_spec: ParameterSpec,
    seed_deriver: SeedDeriver,
    target_record_identity: str,
    budget: int,
    snapshot_position: int,
    run_scale: float,
) -> dict[str, Any]:
    """Produce tier candidates and common-base high-precision reference values."""

    prefix = (
        declaration.campaign_id,
        target_record_identity,
        budget,
        snapshot_position,
    )
    proposals = []
    candidates = []
    for tier in declaration.tiers:
        proposal = propose_composite_mc_nei(
            surrogate,
            baseline=baseline,
            transform=transform,
            target_physical=target_physical,
            parameter_spec=parameter_spec,
            config=_config(tier, seed_deriver, (*prefix, tier.tier_id)),
        )
        proposals.append(proposal)
        candidates.append(proposal.candidate)
    reference = propose_composite_mc_nei(
        surrogate,
        baseline=baseline,
        transform=transform,
        target_physical=target_physical,
        parameter_spec=parameter_spec,
        config=_config(
            declaration.reference_optimizer,
            seed_deriver,
            (*prefix, declaration.reference_optimizer.tier_id),
        ),
    )
    candidates.append(reference.candidate)
    matrix = torch.stack(candidates).contiguous()
    batches = []
    batch_values = []
    for batch in range(4):
        reference_tier = NEINumericalTier(
            f"reference_batch_{batch}", 4096, 1, 1, 1, 0.01
        )
        result = evaluate_composite_mc_nei_candidates(
            surrogate,
            baseline=baseline,
            candidates=matrix,
            transform=transform,
            target_physical=target_physical,
            config=_config(
                reference_tier,
                seed_deriver,
                (*prefix, "reference_evaluation", batch),
                mc_suffix=("reference_batch", batch, "nei_mc"),
            ),
        )
        batches.append(result.provenance())
        batch_values.append(result.values)
    values = torch.stack(batch_values).mean(dim=0)
    maximum = float(values.max().detach().cpu())
    denominator = max(maximum, 0.01 * max(float(run_scale), 1e-12))
    lower = torch.tensor(parameter_spec.lower, dtype=baseline.dtype, device=baseline.device)
    upper = torch.tensor(parameter_spec.upper, dtype=baseline.dtype, device=baseline.device)
    observations = []
    for index, (tier, proposal) in enumerate(zip(declaration.tiers, proposals)):
        candidate = proposal.candidate
        flags = []
        if any(torch.equal(candidate, row) for row in baseline):
            flags.append("exact_duplicate")
        tolerance = 32.0 * torch.finfo(candidate.dtype).eps
        if bool(((candidate - lower).abs() <= tolerance).any()) or bool(
            ((candidate - upper).abs() <= tolerance).any()
        ):
            flags.append("boundary_candidate")
        observations.append({
            "target_record_identity": target_record_identity,
            "budget": budget,
            "snapshot_position": snapshot_position,
            "tier_id": tier.tier_id,
            "candidate": candidate.detach().cpu().tolist(),
            "candidate_source": proposal.candidate_source,
            "optimizer_success": proposal.candidate_source == "local_refinement",
            "reference_nei": float(values[index].detach().cpu()),
            "reference_max_nei": maximum,
            "run_scale": float(run_scale),
            "normalized_regret": max(maximum - float(values[index]), 0.0)
            / denominator,
            "failure_flags": flags,
            "proposal": proposal.provenance(),
        })
    body = _canonical_numerical_document({
        "schema_version": "mutaco-nei-q1-snapshot-v1",
        "declaration_identity": declaration.document()["declaration_identity"],
        "target_record_identity": target_record_identity,
        "budget": budget,
        "snapshot_position": snapshot_position,
        "baseline_size": int(baseline.shape[0]),
        "gp_snapshot": _snapshot_state(surrogate),
        "tier_observations": observations,
        "reference_candidate": reference.candidate.detach().cpu().tolist(),
        "reference_candidate_proposal": reference.provenance(),
        "reference_batches": batches,
        "simulator_calls": 0,
        "nonfinite_float_encoding": "explicit_signed_infinity_v1",
    })
    return {
        **body,
        "snapshot_identity": stable_hash("nei-q1-snapshot-v1", body),
    }
