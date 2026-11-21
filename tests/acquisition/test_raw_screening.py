"""Characterize executed screening with real objectives and controlled restarts.

These are deterministic test posteriors, never historical GP reconstructions.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from mutaco import (LatentPosteriorSummary, ParameterAxis, ParameterSpec,
                    ProposalConfig, ScientificIdentity, SeedCoordinate,
                    SeedDeriver, StandardizedTarget)
from mutaco.acquisition import proposal as p

CASES = (
    "barrier_mixed", "barrier_ties", "exploration_operator", "exploration_trace",
    "targeted_operator", "targeted_trace", "threshold_equal", "threshold_below",
    "feasible_boundary", "trace_radius_barrier", "zero_radius",
    "zero_radius_targeted", "overflow_mixed_targeted", "overflow_all_targeted",
)


@pytest.fixture(autouse=True)
def no_physical_simulator(monkeypatch):
    from mutaco_graph import OverdampedScientificSimulator
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("physical simulation forbidden in screening tests")
    for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
        monkeypatch.setattr(OverdampedScientificSimulator, name, forbidden)
    yield
    assert not attempts


def _rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _assert_rng(before):
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1])
    assert after[2:] == before[1][2:]
    assert torch.equal(torch.random.get_rng_state(), before[2])


def _tensor(x):
    return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
            "bytes": x.detach().contiguous().cpu().numpy().tobytes().hex()}


def _capture(name, dtype, value):
    if location := os.environ.get("MUTACO_SCREENING_CAPTURE_DIR"):
        output = Path(location).resolve()
        assert output.is_absolute() and not output.is_relative_to(Path(__file__).resolve().parents[2])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}-{dtype}.json").write_text(json.dumps(
            {"origin": "controlled characterization fixture, not historical GP state", **value},
            indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fixture(case, dtype):
    distance = torch.full((4,), 8.0, dtype=dtype)
    largest = torch.tensor([1.0, 4.0, 4.0, 2.0], dtype=dtype)
    second = torch.tensor([0.5, 1.0, 2.0, 1.0], dtype=dtype)
    settings = {"beta": 1.0, "alpha": 0.5, "tau_interior": 0.0,
                "raw_samples": 4, "restarts": 3, "max_iterations": 1}
    if case == "barrier_mixed": distance = torch.arange(4, dtype=dtype)
    if case == "barrier_ties": distance.zero_()
    if case in {"threshold_equal", "threshold_below"}: distance.fill_(1.0)
    if case in {"feasible_boundary", "trace_radius_barrier"}: distance.fill_(2.0)
    if case in {"barrier_mixed", "barrier_ties", "threshold_equal", "threshold_below",
                "feasible_boundary", "trace_radius_barrier"}:
        largest.fill_(4.0); second.fill_(1.0)
    if case == "threshold_equal": settings["tau_interior"] = 0.75
    if case == "threshold_below":
        settings["tau_interior"] = float(torch.nextafter(
            torch.tensor(0.75, dtype=dtype), torch.tensor(-torch.inf, dtype=dtype)))
    if case == "exploration_trace": settings["exploration_policy"] = "trace_marginal_variance"
    if case in {"targeted_trace", "trace_radius_barrier"}:
        settings["confidence_radius_policy"] = "trace_effective_dimension_radius_diagnostic_v1"
    targeted = "targeted" in case
    if targeted: settings["infeasible_fallback_policy"] = "minimum_normalized_target_miss_diagnostic_v1"
    if case.startswith("zero_radius"):
        distance.zero_(); largest.zero_(); second.zero_()
    if case.startswith("overflow"):
        distance[:1 if "mixed" in case else 4] = torch.finfo(dtype).max / 2
    mean = torch.stack((distance, torch.zeros_like(distance)), dim=-1).requires_grad_()
    covariance = torch.diag_embed(torch.stack((largest, second), dim=-1)).requires_grad_()
    posterior = LatentPosteriorSummary(mean, covariance, "screen-transform", "screen-model")
    target = StandardizedTarget(torch.zeros(2, dtype=dtype), ("a", "b"), "screen-transform")
    config = ProposalConfig.from_seed_deriver(
        SeedDeriver(ScientificIdentity("screening-characterization"), 193),
        SeedCoordinate("raw_candidates", ("round", 0)), **settings)
    # Closed-form operands for these four-point fixtures; no general optimizer.
    d = distance.clone()
    if case.startswith("overflow"): d[:1 if "mixed" in case else 4] = torch.inf
    radius = largest + second if case in {"targeted_trace", "trace_radius_barrier"} else largest
    margin = ((radius - d.square()) / radius if not case.startswith("zero_radius")
              else torch.zeros(4, dtype=dtype))
    valid = margin > settings["tau_interior"]
    barrier_scores = torch.where(valid, d - 0.5 * torch.log(torch.where(
        valid, margin, torch.ones_like(margin))), torch.full_like(d, torch.inf))
    if bool(valid.any()):
        scores, order, branch = barrier_scores, ([0, 1] if case == "barrier_mixed" else [0, 1, 2, 3]), p.AcquisitionBranch.BARRIER
    elif targeted:
        scores = d.square() / radius
        order = ([] if case in {"zero_radius_targeted", "overflow_all_targeted"}
                 else [1, 2, 3] if case == "overflow_mixed_targeted"
                 else [2, 1, 3, 0] if case == "targeted_trace" else [1, 2, 3, 0])
        branch = p.AcquisitionBranch.EXPLORATION
    else:
        scores = largest + second if case == "exploration_trace" else largest
        order = [2, 1, 3, 0] if case == "exploration_trace" else [1, 2, 3, 0] if case == "exploration_operator" else [0, 1, 2, 3]
        branch = p.AcquisitionBranch.EXPLORATION
    return posterior, target, config, (d, margin, valid, barrier_scores, scores, order, branch, targeted)


def _spies(monkeypatch, posterior, *, failure=None):
    seen = {"calls": [], "orders": [], "starts": []}
    class Provider:
        def __init__(self):
            self.model_id, self.transform_id = posterior.model_id, posterior.transform_id
            self.parameter_dim, self.dtype, self.device = 2, posterior.mean.dtype, posterior.mean.device
        def posterior(self, theta):
            seen["calls"].append("posterior"); seen["theta"] = theta
            assert theta.shape == (4, 2)
            if failure == "posterior": raise RuntimeError("controlled posterior failure")
            return posterior
    barrier = p.MatchedBarrierObjective.evaluate
    ordered, ranking, exploration = p._ordered_finite_indices, p._ranking, p.evaluate_exploration
    def evaluate(self, *args, **kwargs):
        seen["calls"].append("barrier")
        if failure == "barrier": raise FloatingPointError("controlled barrier failure")
        result = barrier(self, *args, **kwargs); seen["barrier"] = result; return result
    def order(values, *, minimize):
        seen["calls"].append("order")
        result = ordered(values, minimize=minimize)
        seen["orders"].append((values, minimize, result)); return result
    def explore(*args, **kwargs):
        seen["calls"].append("exploration")
        if failure == "exploration": raise ArithmeticError("controlled exploration failure")
        result = exploration(*args, **kwargs); seen["exploration"] = result; return result
    def rank(indices, values, margin):
        seen["calls"].append("ranking"); seen["ranking_inputs"] = indices, values, margin
        if failure == "ranking": raise MemoryError("controlled ranking failure")
        result = ranking(indices, values, margin); seen["rankings"] = result; return result
    def restart(provider, start, *, raw_index, restart_index, initial_score, **kwargs):
        seen["calls"].append("restart"); seen["starts"].append((raw_index, start, initial_score))
        diagnostic = p.RestartDiagnostic(restart_index, raw_index, p.RestartStatus.FAILED,
            p.RestartStopReason.POSTERIOR_ERROR, 0, 0, 0, 0, float(initial_score.detach()),
            None, "controlled downstream failure")
        return p._RestartOutcome(None, None, diagnostic)
    monkeypatch.setattr(p.MatchedBarrierObjective, "evaluate", evaluate)
    monkeypatch.setattr(p, "_ordered_finite_indices", order)
    monkeypatch.setattr(p, "evaluate_exploration", explore)
    monkeypatch.setattr(p, "_ranking", rank)
    monkeypatch.setattr(p, "_barrier_restart", restart)
    monkeypatch.setattr(p, "_exploration_restart", restart)
    return Provider(), seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", CASES)
def test_raw_screening_values_order_ownership_and_downstream_inputs(monkeypatch, dtype, case):
    posterior, target, cfg, expected = _fixture(case, dtype)
    provider, seen = _spies(monkeypatch, posterior)
    spec = ParameterSpec((ParameterAxis("x", 0., 1.), ParameterAxis("y", 0., 1.)))
    evaluated = torch.zeros((1, 2), dtype=dtype)
    tensors = [posterior.mean, posterior.covariance, posterior._eigenvalues, target.values, evaluated]
    snapshots = [(x.clone().detach(), x._version) for x in tensors]
    document = deepcopy(cfg.scientific_document()); before = _rng()
    result = p.propose_candidate(provider, parameter_spec=spec, target=target, evaluated_theta=evaluated, config=cfg)
    _assert_rng(before)
    d, margin, valid, barrier_values, scores, indices, branch, targeted = expected
    objective = seen["barrier"]
    for actual, wanted in ((objective.exploitation, d), (objective.margin, margin), (objective.values, barrier_values)):
        torch.testing.assert_close(actual, wanted, atol=0, rtol=0, equal_nan=True)
        assert actual.shape == (4,) and actual.dtype is dtype and actual.device.type == "cpu"
    assert torch.equal(objective.valid_domain, valid)
    actual_indices, actual_scores, actual_margin = seen["ranking_inputs"]
    torch.testing.assert_close(actual_scores, scores, atol=0, rtol=0, equal_nan=True)
    assert actual_indices.tolist() == indices and actual_indices.dtype is torch.int64
    assert actual_scores.requires_grad and actual_scores.shape == (4,)
    if branch is p.AcquisitionBranch.BARRIER:
        assert actual_scores is objective.values and actual_margin is objective.margin
        assert len(seen["orders"]) == 1 and "exploration" not in seen["calls"]
    else:
        assert actual_margin is None and len(seen["orders"]) == 2
        if not targeted: assert actual_scores is seen["exploration"].values
    assert seen["orders"][-1][0] is actual_scores and seen["orders"][-1][1] is (branch is p.AcquisitionBranch.BARRIER or targeted)
    assert result.branch is branch
    assert result.branch_direction == ("minimize" if branch is p.AcquisitionBranch.BARRIER or targeted else "maximize")
    assert result.raw_counts == {"candidate_count": 4, "strict_interior_finite": int(valid.sum()), "exploration_finite": None if bool(valid.any()) else len(indices)}
    assert result.raw_rankings is seen["rankings"]
    assert [r.raw_index for r in result.raw_rankings] == indices
    for i, ranking in enumerate(result.raw_rankings):
        assert ranking.rank == i and ranking.score.hex() == float(scores[indices[i]]).hex()
        assert ranking.margin == (float(margin[indices[i]]) if bool(valid.any()) else None)
    assert [x[0] for x in seen["starts"]] == indices[:3]
    for index, start, initial_score in seen["starts"]:
        assert start.untyped_storage().data_ptr() == result.pool.theta.untyped_storage().data_ptr()
        assert start.storage_offset() == index * 2
        assert initial_score.untyped_storage().data_ptr() == actual_scores.untyped_storage().data_ptr()
        torch.testing.assert_close(initial_score, scores[index], atol=0, rtol=0)
    if indices:
        assert result.status is p.ProposalStatus.SUCCESS and result.fallback_reason is p.FallbackReason.ALL_RESTARTS_FAILED
        assert result.candidate_source is p.ReturnedCandidateSource.RAW_FALLBACK
        assert torch.equal(result.candidate, result.pool.theta[indices[0]]) and result.candidate._base is None
        assert result.candidate.untyped_storage().data_ptr() != result.pool.theta.untyped_storage().data_ptr()
    else:
        assert result.status is p.ProposalStatus.FAILED and result.failure_reason is p.ProposalFailureReason.NO_FINITE_EXPLORATION_CANDIDATE
        assert result.failure_explanation == "the exploration branch contained no finite raw score"
        assert result.candidate is None and not result.restarts
    assert result.posterior_calls.total == result.posterior_calls.raw_pool == 1
    assert result.posterior_calls.per_restart == (0,) * min(3, len(indices))
    assert result.simulator_calls == 0 and cfg.scientific_document() == document
    assert seen["theta"] is result.pool.theta
    regenerated = p.generate_candidate_pool(parameter_spec=spec, config=cfg, dtype=dtype, device=torch.device("cpu"))
    assert regenerated.pool_id == result.pool.pool_id and torch.equal(regenerated.theta, result.pool.theta)
    assert result.pool.seed_value == cfg.candidate_seed and result.pool.seed_coordinate == cfg.seed_coordinate
    for actual, (wanted, version) in zip(tensors, snapshots):
        assert torch.equal(actual, wanted) and actual._version == version
    if case == "barrier_mixed":
        gradient, = torch.autograd.grad(actual_scores[:2].sum(), posterior.mean)
        assert torch.isfinite(gradient).all() and bool(gradient[1, 0] > 1)
    _assert_rng(before)
    _capture(case, str(dtype), {"barrier_values": _tensor(objective.values), "margin": _tensor(objective.margin),
        "scores": _tensor(actual_scores), "indices": indices, "branch": result.branch.value,
        "direction": result.branch_direction, "rankings": [{"rank":r.rank,"raw_index":r.raw_index,"score":r.score,"margin":r.margin} for r in result.raw_rankings],
        "starts": [{"index": i, "theta": _tensor(x), "score": _tensor(v)} for i,x,v in seen["starts"]],
        "calls": seen["calls"], "pool_id": result.pool.pool_id, "candidate_id": result.candidate_id,
        "status": result.status.value, "rng_preserved": True, "input_versions_preserved": True})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("failure", ["posterior", "barrier", "exploration", "ranking"])
def test_raw_screening_failure_precedence_and_accounting(monkeypatch, dtype, failure):
    posterior, target, cfg, _ = _fixture("exploration_operator", dtype)
    provider, seen = _spies(monkeypatch, posterior, failure=failure)
    before = _rng()
    result = p.propose_candidate(provider, parameter_spec=ParameterSpec((ParameterAxis("x",0.,1.),ParameterAxis("y",0.,1.))),
        target=target, evaluated_theta=torch.zeros((1,2),dtype=dtype), config=cfg)
    _assert_rng(before)
    assert result.status is p.ProposalStatus.FAILED and result.branch is p.AcquisitionBranch.NONE
    assert result.candidate is result.candidate_id is None and result.raw_rankings == result.restarts == ()
    assert result.posterior_calls.raw_pool == result.posterior_calls.total == 1 and result.posterior_calls.per_restart == ()
    assert result.raw_counts == {"candidate_count":4,"strict_interior_finite":None,"exploration_finite":None}
    assert result.simulator_calls == 0 and not seen["starts"]
    expected = {"posterior": ("RuntimeError", ["posterior"]),
                "barrier": ("FloatingPointError", ["posterior","barrier"]),
                "exploration": ("ArithmeticError", ["posterior","barrier","order","exploration"]),
                "ranking": ("MemoryError", ["posterior","barrier","order","exploration","order","ranking"])}
    assert result.failure_explanation == f"{expected[failure][0]}: controlled {failure} failure"
    assert seen["calls"] == expected[failure][1]
    assert result.failure_reason is (p.ProposalFailureReason.RAW_POSTERIOR_ERROR if failure=="posterior" else p.ProposalFailureReason.RAW_POSTERIOR_SCORING_ERROR)
    _capture("failure-"+failure,str(dtype),{"status":result.status.value,"reason":result.failure_reason.value,
        "message":result.failure_explanation,"calls":seen["calls"],"pool_id":result.pool.pool_id,"rng_preserved":True})
