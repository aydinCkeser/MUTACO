from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.optim.core import OptimizationStatus
from linear_operator.utils.errors import NotPSDError

from mutaco import (
    GPFitConfig,
    GPFitFailureKind,
    GPFitStatus,
    InvalidPosteriorError,
    JointLatentPosteriorSummary,
    LatentPosteriorSummary,
    LikelihoodPolicy,
    POSTERIOR_NUMERICS_POLICIES,
    ParameterAxis,
    ParameterSpec,
    PolicyClassification,
    PosteriorNumericsPolicy,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    SurrogateTrainingData,
    fit_exact_multitask_gp,
    likelihood_support_markdown,
    posterior_numerics_support_markdown,
)


DTYPE = torch.float64


def training_data(parameter_dim: int = 2) -> SurrogateTrainingData:
    count = 6
    theta = torch.linspace(0.05, 0.95, count, dtype=DTYPE).unsqueeze(-1).repeat(
        1, parameter_dim
    )
    weights = torch.arange(1, parameter_dim + 1, dtype=DTYPE)
    scalar = (theta * weights).sum(dim=-1)
    response = torch.stack((torch.sin(scalar), torch.cos(scalar)), dim=-1).contiguous()
    parameters = ParameterSpec(
        tuple(ParameterAxis(f"theta_{index}", 0.0, 1.0) for index in range(parameter_dim))
    )
    return SurrogateTrainingData(
        theta.contiguous(),
        response,
        parameters,
        tuple(f"observation-{index}" for index in range(count)),
        "transform-test",
    )


def fitted_float32():
    data64 = training_data()
    data = SurrogateTrainingData(
        theta=data64.theta.float().contiguous(),
        response_standardized=data64.response_standardized.float().contiguous(),
        parameter_spec=data64.parameter_spec,
        observation_ids=data64.observation_ids,
        transform_id=data64.transform_id,
    )
    result = fit_exact_multitask_gp(
        data,
        config=GPFitConfig(
            fit_seed=771,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("prompt8k",)),
            max_iterations=8,
            symmetry_tolerance=2e-6,
            psd_tolerance=1e-6,
            covariance_validation_policy=(
                "structural_symmetric_materialization_v1"
            ),
            posterior_numerics_policy="float32_fit_float64_covariance",
        ),
    )
    assert result.surrogate is not None, result.diagnostic
    return result.surrogate


def fitted(
    likelihood: LikelihoodPolicy = LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN,
    parameter_dim: int = 2,
    covariance_validation_policy: str = "absolute_v1",
):
    coordinate = SeedCoordinate("gp_fit", ("round", 0))
    result = fit_exact_multitask_gp(
        training_data(parameter_dim),
        config=GPFitConfig.from_seed_deriver(
            SeedDeriver(ScientificIdentity.create({"experiment": "gp-test"}), 91),
            coordinate,
            likelihood=likelihood,
            max_iterations=8,
            covariance_validation_policy=covariance_validation_policy,
        ),
    )
    assert result.diagnostic.status is GPFitStatus.SUCCESS, result.diagnostic
    assert result.surrogate is not None
    return result.surrogate


def test_latent_posterior_public_boundary_scalar_batch_and_gradient() -> None:
    surrogate = fitted()
    scalar = surrogate.posterior(torch.tensor([0.3, 0.3], dtype=DTYPE))
    batch_theta = torch.tensor([[0.3, 0.3], [0.7, 0.7]], dtype=DTYPE)
    batch = surrogate.posterior(batch_theta)
    assert isinstance(scalar, LatentPosteriorSummary)
    assert scalar.mean.shape == (2,)
    assert scalar.covariance.shape == (2, 2)
    assert batch.mean.shape == (2, 2)
    assert batch.covariance.shape == (2, 2, 2)
    assert torch.allclose(scalar.mean, batch.mean[0])
    assert scalar.mean.dtype == DTYPE
    assert not hasattr(surrogate, "model")

    differentiable = torch.tensor([0.4, 0.4], dtype=DTYPE, requires_grad=True)
    value = surrogate.posterior(differentiable).mean.sum()
    (gradient,) = torch.autograd.grad(value, differentiable)
    assert bool(torch.isfinite(gradient).all())


def test_joint_latent_posterior_retains_cross_point_covariance_and_gradient() -> None:
    surrogate = fitted()
    theta = torch.tensor(
        [[0.25, 0.25], [0.65, 0.65]], dtype=DTYPE, requires_grad=True
    )
    joint = surrogate.joint_posterior(theta)
    marginal = surrogate.posterior(theta)
    assert isinstance(joint, JointLatentPosteriorSummary)
    assert joint.mean.shape == (2, 2)
    assert joint.covariance.shape == (4, 4)
    assert torch.equal(joint.covariance, joint.covariance.T)
    assert torch.allclose(joint.mean, marginal.mean)
    assert torch.allclose(joint.covariance[:2, :2], marginal.covariance[0])
    assert torch.allclose(joint.covariance[2:, 2:], marginal.covariance[1])
    assert bool((joint.covariance[:2, 2:].abs() > 0).any())
    gradient = torch.autograd.grad(joint.mean.sum(), theta)[0]
    assert bool(torch.isfinite(gradient).all())


def test_joint_interface_does_not_change_existing_posterior_or_model_identity() -> None:
    surrogate = fitted()
    theta = torch.tensor([[0.2, 0.2], [0.8, 0.8]], dtype=DTYPE).contiguous()
    before = surrogate.posterior(theta)
    model_id = surrogate.model_id
    surrogate.joint_posterior(theta)
    after = surrogate.posterior(theta)
    assert surrogate.model_id == model_id
    assert torch.equal(before.mean, after.mean)
    assert torch.equal(before.covariance, after.covariance)


def test_structural_covariance_policy_is_applied_at_gp_boundary() -> None:
    surrogate = fitted(
        covariance_validation_policy="structural_symmetric_materialization_v1"
    )
    posterior = surrogate.posterior(torch.tensor([0.3, 0.3], dtype=DTYPE))
    diagnostics = posterior.validation_diagnostics["structural_materialization"]
    assert diagnostics["source_stage"] == (
        "botorch_latent_posterior_dense_materialization"
    )
    assert diagnostics["exact_output_symmetry"] is True
    assert torch.equal(posterior.covariance, posterior.covariance.T)


@pytest.mark.parametrize(
    "likelihood",
    [
        LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN,
        LikelihoodPolicy.SHARED_ISOTROPIC_GAUSSIAN,
    ],
)
def test_likelihood_policies_keep_observation_noise_out_of_latent_geometry(
    likelihood: LikelihoodPolicy,
) -> None:
    surrogate = fitted(likelihood)
    theta = torch.tensor([0.5, 0.5], dtype=DTYPE)
    latent = surrogate.posterior(theta)
    observed = surrogate.observation_posterior(theta)
    increment = observed.covariance - latent.covariance
    assert bool((torch.diagonal(increment) > 0).all())
    assert torch.allclose(increment, torch.diag(torch.diagonal(increment)), atol=1e-8)
    assert surrogate.provenance()["likelihood_policy"] == likelihood.value
    assert surrogate.provenance()["posterior_covariance"] == "latent_standardized"


def test_fit_is_seeded_rng_neutral_identified_and_dimension_generic() -> None:
    before = torch.random.get_rng_state().clone()
    first = fitted(parameter_dim=6)
    after = torch.random.get_rng_state()
    second = fitted(parameter_dim=6)
    assert torch.equal(before, after)
    assert first.model_id == second.model_id
    assert first.parameter_dim == 6
    assert isinstance(first.provenance()["fit_seed"], int)
    assert first.provenance()["fit_seed_coordinate"]["namespace"] == "gp_fit"
    assert first.provenance()["training_data_id"].startswith("surrogate-training-v1-")


def test_unknown_likelihood_and_invalid_training_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        GPFitConfig(
            fit_seed=1,
            fit_seed_coordinate=SeedCoordinate("gp_fit"),
            likelihood="invented",  # type: ignore[arg-type]
        )
    data = training_data()
    with pytest.raises(ValueError, match="at least two"):
        SurrogateTrainingData(
            data.theta[:1],
            data.response_standardized[:1],
            data.parameter_spec,
            ("one",),
            data.transform_id,
        )


def test_numerical_fit_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    from mutaco.surrogate import exact_multitask

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deliberate fit failure")

    monkeypatch.setattr(exact_multitask, "_FIT_MLL", fail)
    result = fit_exact_multitask_gp(
        training_data(),
        config=GPFitConfig(
            fit_seed=4,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("failure-test",)),
            max_iterations=2,
        ),
    )
    assert result.surrogate is None
    assert result.diagnostic.status is GPFitStatus.FAILED
    assert result.diagnostic.failure_type == "RuntimeError"
    assert result.diagnostic.failure_message == "deliberate fit failure"
    assert result.diagnostic.failure_kind is GPFitFailureKind.OPTIMIZER_FAILURE
    assert result.diagnostic.failure_stage == "optimizer"
    assert result.diagnostic.exception_chain[-1]["type"] == "RuntimeError"
    assert result.diagnostic.recovery_attempts[0]["failure_category"] == "optimization"


def test_underlying_optimizer_covariance_failure_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.surrogate import exact_multitask

    observed = SimpleNamespace(
        status=OptimizationStatus.RUNNING,
        step=7,
        fval=1.25,
        runtime=0.5,
        message="iteration in progress",
    )

    def fail_scipy(*args: object, **kwargs: object) -> None:
        callback = kwargs["callback"]
        callback({}, observed)
        raise NotPSDError("training matrix not positive definite")

    monkeypatch.setattr(exact_multitask, "fit_gpytorch_mll_scipy", fail_scipy)
    result = fit_exact_multitask_gp(
        training_data(),
        config=GPFitConfig(
            fit_seed=5,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("not-psd",)),
            max_iterations=12,
        ),
    )
    diagnostic = result.diagnostic
    assert diagnostic.status is GPFitStatus.FAILED
    assert diagnostic.failure_type == "ModelFittingError"
    assert diagnostic.failure_kind is GPFitFailureKind.COVARIANCE_REJECTED
    assert diagnostic.optimizer_result is not None
    assert diagnostic.optimizer_result["actual_iterations"] == 7
    assert diagnostic.optimizer_result["final_objective"] == 1.25
    assert diagnostic.exception_chain[0]["type"] == "NotPSDError"
    assert diagnostic.exception_chain[-1]["type"] == "ModelFittingError"


def test_iteration_exhaustion_and_nonfinite_objective_are_distinguished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mutaco.surrogate import exact_multitask

    def outer_failure(mll: object, **kwargs: object) -> None:
        kwargs["optimizer"](
            mll,
            options=kwargs["optimizer_kwargs"]["options"],
        )
        raise ModelFittingError("diagnostic wrapper rejection")

    monkeypatch.setattr(exact_multitask, "_FIT_MLL", outer_failure)

    def stopped(*args: object, **kwargs: object) -> object:
        result = SimpleNamespace(
            status=OptimizationStatus.STOPPED,
            step=12,
            fval=0.5,
            runtime=0.2,
            message="TOTAL NO. OF ITERATIONS REACHED LIMIT",
        )
        kwargs["callback"]({}, result)
        return result

    monkeypatch.setattr(exact_multitask, "fit_gpytorch_mll_scipy", stopped)
    exhausted = fit_exact_multitask_gp(
        training_data(),
        config=GPFitConfig(
            fit_seed=6,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("iteration-limit",)),
            max_iterations=12,
        ),
    )
    assert exhausted.diagnostic.failure_kind is GPFitFailureKind.ITERATION_EXHAUSTED

    def nonfinite(*args: object, **kwargs: object) -> object:
        result = SimpleNamespace(
            status=OptimizationStatus.FAILURE,
            step=3,
            fval=float("nan"),
            runtime=0.1,
            message="nonfinite objective",
        )
        kwargs["callback"]({}, result)
        return result

    monkeypatch.setattr(exact_multitask, "fit_gpytorch_mll_scipy", nonfinite)
    invalid = fit_exact_multitask_gp(
        training_data(),
        config=GPFitConfig(
            fit_seed=7,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("nonfinite",)),
            max_iterations=12,
        ),
    )
    assert invalid.diagnostic.failure_kind is GPFitFailureKind.NONFINITE_OBJECTIVE


def test_likelihood_support_document_is_generated() -> None:
    path = (
        Path(__file__).parents[2]
        / "scientific_contracts"
        / "mutaco_d_contract"
        / "LIKELIHOOD_SUPPORT.md"
    )
    assert path.read_text(encoding="utf-8") == likelihood_support_markdown()


def test_prompt8k_mixed_precision_registry_and_generated_support() -> None:
    assert POSTERIOR_NUMERICS_POLICIES[
        PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE
    ] is PolicyClassification.DEFAULT
    assert POSTERIOR_NUMERICS_POLICIES[
        PosteriorNumericsPolicy.NATIVE_FLOAT32
    ] is PolicyClassification.DIAGNOSTIC_ONLY
    with pytest.raises(ValueError, match="unknown posterior numerics"):
        GPFitConfig(
            1,
            SeedCoordinate("gp_fit"),
            posterior_numerics_policy="guess_precision",
        )
    path = (
        Path(__file__).parents[2]
        / "scientific_contracts"
        / "mutaco_d_contract"
        / "POSTERIOR_NUMERICS_SUPPORT.md"
    )
    assert path.read_text(encoding="utf-8") == posterior_numerics_support_markdown()


def test_prompt8k_mixed_covariance_is_promoted_but_mean_remains_native() -> None:
    surrogate = fitted_float32()
    theta = torch.tensor([[0.3, 0.3], [0.7, 0.7]], dtype=torch.float32)
    posterior = surrogate.posterior(theta)
    numerics = posterior.validation_diagnostics["posterior_numerics"]
    assert numerics["policy_id"] == "float32_fit_float64_covariance"
    assert numerics["fit_dtype"] == "torch.float32"
    assert numerics["covariance_compute_dtype"] == "torch.float64"
    assert numerics["returned_dtype"] == "torch.float32"
    assert numerics["promotion_occurred"] is True
    assert numerics["additional_fit_attempts"] == 0
    assert posterior.mean.dtype is torch.float32
    assert posterior.covariance.dtype is torch.float32
    assert torch.equal(posterior.covariance, posterior.covariance.transpose(-1, -2))

    reversed_posterior = surrogate.posterior(theta.flip(0).contiguous())
    assert torch.allclose(posterior.mean, reversed_posterior.mean.flip(0), atol=1e-6)
    assert torch.allclose(
        posterior.covariance,
        reversed_posterior.covariance.flip(0),
        atol=1e-6,
    )
    scalar = surrogate.posterior(theta[0].contiguous())
    assert torch.allclose(scalar.mean, posterior.mean[0], atol=1e-6)
    assert torch.allclose(scalar.covariance, posterior.covariance[0], atol=1e-6)


def test_prompt8k_mean_only_path_never_materializes_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate = fitted_float32()

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("covariance path must not run")

    monkeypatch.setattr(surrogate, "_mixed_precision_covariance", forbidden)
    result = surrogate.posterior_mean(
        torch.tensor([[0.3, 0.3], [0.7, 0.7]], dtype=torch.float32)
    )
    assert result.mean.shape == (2, 2)
    assert bool(torch.isfinite(result.mean).all())


@pytest.mark.parametrize(
    "invalid_covariance",
    (
        torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], dtype=torch.float64),
    ),
)
def test_prompt8k_promoted_path_rejects_material_covariance_defects(
    monkeypatch: pytest.MonkeyPatch, invalid_covariance: torch.Tensor
) -> None:
    surrogate = fitted_float32()

    class FakePosterior:
        covariance_matrix = invalid_covariance

    class FakeModel:
        def posterior(self, *args: object, **kwargs: object) -> FakePosterior:
            return FakePosterior()

    monkeypatch.setattr(surrogate, "_promoted_covariance_model", lambda: FakeModel())
    with pytest.raises(InvalidPosteriorError):
        surrogate.posterior(torch.tensor([0.3, 0.3], dtype=torch.float32))


def test_full_float64_diagnostic_rejects_float32_fitted_model() -> None:
    surrogate = fitted_float32()
    surrogate._config = replace(
        surrogate._config,
        posterior_numerics_policy=PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC,
    )
    with pytest.raises(ValueError, match="requires a float64-fitted"):
        surrogate.posterior(torch.tensor([0.3, 0.3], dtype=torch.float32))


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compatible Torch CUDA is unavailable"
)
def test_prompt8k_mixed_precision_covariance_remains_on_cuda() -> None:
    cpu = fitted_float32()
    data = SurrogateTrainingData(
        theta=cpu._training_data.theta.cuda().contiguous(),
        response_standardized=(
            cpu._training_data.response_standardized.cuda().contiguous()
        ),
        parameter_spec=cpu._training_data.parameter_spec,
        observation_ids=cpu._training_data.observation_ids,
        transform_id=cpu._training_data.transform_id,
    )
    result = fit_exact_multitask_gp(
        data,
        config=replace(cpu._config, fit_seed=772),
    )
    assert result.surrogate is not None, result.diagnostic
    posterior = result.surrogate.posterior(data.theta[:2].contiguous())
    assert posterior.mean.device.type == "cuda"
    assert posterior.covariance.device.type == "cuda"
    assert posterior.validation_diagnostics["posterior_numerics"][
        "covariance_compute_dtype"
    ] == "torch.float64"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compatible Torch CUDA is unavailable")
def test_surrogate_fit_and_posterior_remain_on_cuda() -> None:
    cpu = training_data()
    data = SurrogateTrainingData(
        cpu.theta.cuda(),
        cpu.response_standardized.cuda(),
        cpu.parameter_spec,
        cpu.observation_ids,
        cpu.transform_id,
    )
    result = fit_exact_multitask_gp(
        data,
        config=GPFitConfig(
            fit_seed=2,
            fit_seed_coordinate=SeedCoordinate("gp_fit", ("cuda",)),
            max_iterations=5,
        ),
    )
    assert result.surrogate is not None, result.diagnostic
    posterior = result.surrogate.posterior(
        torch.tensor([0.5, 0.5], device="cuda", dtype=DTYPE)
    )
    assert posterior.mean.device.type == "cuda"
    assert posterior.covariance.device.type == "cuda"
