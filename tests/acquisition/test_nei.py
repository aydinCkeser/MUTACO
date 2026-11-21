from __future__ import annotations

import pytest
import torch

from mutaco import (
    FrozenResponseTransform,
    GPFitConfig,
    GPFitStatus,
    JointLatentPosteriorSummary,
    NEIConfig,
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SurrogateTrainingData,
    fit_exact_multitask_gp,
    evaluate_composite_mc_nei_candidates,
    propose_composite_mc_nei,
)
from mutaco.acquisition import nei


class SmoothJointProvider:
    model_id = "smooth-joint-provider"

    def __init__(self, transform_id: str) -> None:
        self.transform_id = transform_id

    def joint_posterior(self, theta: torch.Tensor) -> JointLatentPosteriorSummary:
        x = theta[:, :1]
        mean = torch.cat((x, 1.0 - x), dim=-1).contiguous()
        distance = x - x.T
        point = torch.exp(-distance.square() / 0.3) + 0.05 * torch.eye(
            theta.shape[0], dtype=theta.dtype, device=theta.device
        )
        task = torch.tensor(
            [[0.4, 0.12], [0.12, 0.3]],
            dtype=theta.dtype,
            device=theta.device,
        )
        covariance = torch.kron(point, task).contiguous()
        covariance = (0.5 * (covariance + covariance.T)).contiguous()
        return JointLatentPosteriorSummary(
            mean, covariance, self.transform_id, self.model_id
        )


def _setup():
    dtype = torch.float64
    responses = torch.tensor(
        [[0.1, 0.9], [0.35, 0.65], [0.75, 0.25]], dtype=dtype
    ).contiguous()
    transform = FrozenResponseTransform.fit(
        responses,
        response_spec=ResponseSpec((ResponseAxis("a"), ResponseAxis("b"))),
        observation_ids=("a", "b", "c"),
    )
    baseline = torch.tensor([[0.1], [0.35], [0.75]], dtype=dtype).contiguous()
    target = torch.tensor([0.5, 0.5], dtype=dtype).contiguous()
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    deriver = SeedDeriver(ScientificIdentity("nei-test-family"), 7719)
    config = NEIConfig(
        n_mc=32,
        raw_samples=8,
        restarts=2,
        max_iterations=2,
        learning_rate=0.03,
        seed_deriver=deriver,
        raw_coordinate=SeedCoordinate("raw", ("target", 17)),
        mc_coordinate=SeedCoordinate("mc", ("target", 17)),
        optimizer_coordinate=SeedCoordinate("optimizer", ("target", 17)),
    )
    return transform, baseline, target, parameters, config


def test_composite_nei_is_seeded_rng_neutral_and_reproducible() -> None:
    transform, baseline, target, parameters, config = _setup()
    provider = SmoothJointProvider(transform.metadata.transform_id)
    before = torch.random.get_rng_state().clone()
    first = propose_composite_mc_nei(
        provider,
        baseline=baseline,
        transform=transform,
        target_physical=target,
        parameter_spec=parameters,
        config=config,
    )
    after = torch.random.get_rng_state()
    second = propose_composite_mc_nei(
        provider,
        baseline=baseline,
        transform=transform,
        target_physical=target,
        parameter_spec=parameters,
        config=config,
    )
    assert torch.equal(before, after)
    assert torch.equal(first.candidate, second.candidate)
    assert torch.equal(first.raw_values, second.raw_values)
    assert first.base_sample_identity == second.base_sample_identity
    assert first.base_sample_shape == (32, 8)
    assert first.baseline_size == 3
    assert 0.0 <= float(first.candidate[0]) <= 1.0
    assert first.provenance()["simulator_calls"] == 0
    assert first.provenance()["baseline_factorization"] == {
        "policy_id": "promoted_float64_cholesky_zero_jitter_v1",
        "input_dtype": "torch.float64",
        "factor_dtype": "torch.float64",
        "dimension": 6,
        "promotion_occurred": False,
        "jitter_added": 0.0,
        "relative_jitter": 0.0,
        "cholesky_attempts": 1,
    }
    assert first.candidate_source in {"local_refinement", "raw_recovery"}


def _rank_four_baseline_covariance() -> torch.Tensor:
    # Gram matrix of (1, x, x**2, x**3) at x = 0, 1/4, 1/2, 3/4, 1.
    return torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.066650390625, 1.142578125, 1.229248046875, 1.328125],
            [1.0, 1.142578125, 1.328125, 1.568359375, 1.875],
            [1.0, 1.229248046875, 1.568359375, 2.056884765625, 2.734375],
            [1.0, 1.328125, 1.875, 2.734375, 4.0],
        ],
        dtype=torch.float32,
    ).contiguous()


def test_baseline_factorization_promotes_without_jitter() -> None:
    covariance = _rank_four_baseline_covariance()
    # One exactly representable input variance removes the sole null direction
    # (1, -4, 6, -4, 1), making this fixture strictly positive definite.
    # This defines the test covariance; production adds no jitter or repair.
    covariance[0, 0] = 1.0 + 2.0**-23
    assert bool(torch.linalg.eigvalsh(covariance.double())[0] > 0)
    _, native_info = torch.linalg.cholesky_ex(covariance, check_errors=False)
    assert int(native_info) != 0

    factor, diagnostic = nei._promoted_baseline_cholesky(covariance)

    assert factor.dtype is torch.float64
    assert torch.allclose(
        factor @ factor.T, covariance.double(), atol=1e-14, rtol=1e-14
    )
    assert diagnostic.document() == {
        "policy_id": "promoted_float64_cholesky_zero_jitter_v1",
        "input_dtype": "torch.float32",
        "factor_dtype": "torch.float64",
        "dimension": 5,
        "promotion_occurred": True,
        "jitter_added": 0.0,
        "relative_jitter": 0.0,
        "cholesky_attempts": 1,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_baseline_factorization_rejects_exactly_singular_covariance(dtype) -> None:
    # Duplicate unit rows give an exact zero pivot in either precision, without
    # relying on the backend-dependent roundoff of the old rank-four fixture.
    covariance = torch.ones((2, 2), dtype=dtype)
    null = torch.tensor([1.0, -1.0], dtype=dtype)
    assert torch.equal(covariance @ null, torch.zeros(2, dtype=dtype))
    with pytest.raises(
        nei.NEIAcquisitionError,
        match="baseline latent covariance is not positive definite",
    ):
        nei._promoted_baseline_cholesky(covariance)


def test_baseline_factorization_does_not_mask_indefinite_covariance() -> None:
    covariance = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32
    ).contiguous()
    with pytest.raises(
        nei.NEIAcquisitionError,
        match="baseline latent covariance is not positive definite",
    ):
        nei._promoted_baseline_cholesky(covariance)


def test_composite_nei_uses_joint_cross_point_covariance() -> None:
    transform, baseline, target, parameters, config = _setup()
    provider = SmoothJointProvider(transform.metadata.transform_id)
    result = propose_composite_mc_nei(
        provider,
        baseline=baseline,
        transform=transform,
        target_physical=target,
        parameter_spec=parameters,
        config=config,
    )
    posterior = provider.joint_posterior(
        torch.cat((baseline, result.candidate.unsqueeze(0)), dim=0).contiguous()
    )
    assert posterior.covariance.shape == (8, 8)
    assert bool((posterior.covariance[:6, 6:].abs() > 0).any())
    assert result.posterior_calls >= 1 + config.raw_samples


def test_candidate_set_evaluation_uses_one_common_base_sample_and_is_rng_neutral() -> None:
    transform, baseline, target, _, config = _setup()
    provider = SmoothJointProvider(transform.metadata.transform_id)
    candidates = torch.tensor([[0.2], [0.5], [0.9]], dtype=baseline.dtype)
    before = torch.random.get_rng_state().clone()
    first = evaluate_composite_mc_nei_candidates(
        provider,
        baseline=baseline,
        candidates=candidates,
        transform=transform,
        target_physical=target,
        config=config,
    )
    second = evaluate_composite_mc_nei_candidates(
        provider,
        baseline=baseline,
        candidates=candidates,
        transform=transform,
        target_physical=target,
        config=config,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(first.values, second.values)
    assert first.base_sample_identity == second.base_sample_identity
    assert first.values.shape == (3,)
    assert first.posterior_calls == 4


def test_composite_nei_refines_against_exact_multitask_surrogate() -> None:
    dtype = torch.float64
    theta = torch.linspace(0.05, 0.95, 6, dtype=dtype).unsqueeze(-1).contiguous()
    physical = torch.cat((torch.sin(theta), torch.cos(theta)), dim=-1).contiguous()
    response_spec = ResponseSpec((ResponseAxis("sin"), ResponseAxis("cos")))
    observation_ids = tuple(f"observation-{index}" for index in range(6))
    transform = FrozenResponseTransform.fit(
        physical,
        response_spec=response_spec,
        observation_ids=observation_ids,
    )
    standardized = transform.to_standardized(
        physical, response_names=transform.metadata.response_names
    )
    parameters = ParameterSpec((ParameterAxis("theta", 0.0, 1.0),))
    fit = fit_exact_multitask_gp(
        SurrogateTrainingData(
            theta=theta,
            response_standardized=standardized,
            parameter_spec=parameters,
            observation_ids=observation_ids,
            transform_id=transform.metadata.transform_id,
        ),
        config=GPFitConfig(
            fit_seed=17,
            fit_seed_coordinate=SeedCoordinate("gp_fit", (0,)),
            max_iterations=4,
        ),
    )
    assert fit.diagnostic.status is GPFitStatus.SUCCESS
    assert fit.surrogate is not None
    deriver = SeedDeriver(ScientificIdentity("exact-nei-test"), 22)
    result = propose_composite_mc_nei(
        fit.surrogate,
        baseline=theta,
        transform=transform,
        target_physical=torch.tensor([0.45, 0.85], dtype=dtype).contiguous(),
        parameter_spec=parameters,
        config=NEIConfig(
            n_mc=8,
            raw_samples=4,
            restarts=1,
            max_iterations=2,
            learning_rate=0.02,
            seed_deriver=deriver,
            raw_coordinate=SeedCoordinate("raw", (0,)),
            mc_coordinate=SeedCoordinate("mc", (0,)),
            optimizer_coordinate=SeedCoordinate("optimizer", (0,)),
        ),
    )
    assert result.candidate.shape == (1,)
    assert bool(torch.isfinite(result.value))
