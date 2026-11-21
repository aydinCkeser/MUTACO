"""Manuscript Step 2: correlated exact GP for frozen standardized responses.

Physical parameter rows become normalized, task-augmented training inputs.
The pinned BoTorch model supplies the correlated kernel and registered priors;
fitting, bounded recovery and latent posterior validation are kept explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from time import perf_counter
from typing import Any, Callable
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.multitask import MIN_INFERRED_NOISE_LEVEL
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.optim.closures import get_loss_closure
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.settings import cholesky_jitter

from mutaco.identity import stable_hash
from mutaco.policies import PolicyClassification, surface_record
from mutaco.posterior import (
    CovarianceValidationPolicy,
    JointLatentPosteriorSummary,
    LatentPosteriorMeanSummary,
    LatentPosteriorSummary,
    structurally_symmetric_materialize,
)
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.runtime import CpuParallelismProvenance
from mutaco.types import ParameterSpec


class LikelihoodPolicy(str, Enum):
    TASK_DIAGONAL_GAUSSIAN = "task_diagonal_gaussian"
    SHARED_ISOTROPIC_GAUSSIAN = "shared_isotropic_gaussian"


class TaskVarianceFloorPolicy(str, Enum):
    NONE = "none"
    STANDARDIZED_ABSOLUTE_V1 = "standardized_absolute_v1"


class GPFitRecoveryPolicy(str, Enum):
    COLD_ONLY = "cold_only"
    GRAPH_M20_SEQUENTIAL_RECOVERY_V1 = "graph_m20_sequential_recovery_v1"


class PosteriorNumericsPolicy(str, Enum):
    """Precision boundary for posterior covariance construction."""

    NATIVE_FLOAT32 = "native_float32"
    FLOAT32_FIT_FLOAT64_COVARIANCE = "float32_fit_float64_covariance"
    FULL_FLOAT64_DIAGNOSTIC = "full_float64_diagnostic"


POSTERIOR_NUMERICS_POLICIES: dict[PosteriorNumericsPolicy, PolicyClassification] = {
    PosteriorNumericsPolicy.NATIVE_FLOAT32: PolicyClassification.DIAGNOSTIC_ONLY,
    PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE: (
        PolicyClassification.DEFAULT
    ),
    PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC: (
        PolicyClassification.DIAGNOSTIC_ONLY
    ),
}


GP_FIT_RECOVERY_POLICIES: dict[GPFitRecoveryPolicy, PolicyClassification] = {
    GPFitRecoveryPolicy.COLD_ONLY: PolicyClassification.DEFAULT,
    GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1: (
        PolicyClassification.APPROVED_OPTIONAL
    ),
}

GRAPH_M20_TASK_VARIANCE_FLOOR = 5.991468697175151e-06
GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE = 6.282510280609131
GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER = 8.0
GRAPH_M20_RESTART_COUNT = 4


LIKELIHOOD_POLICIES: dict[LikelihoodPolicy, PolicyClassification] = {
    LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN: PolicyClassification.DEFAULT,
    LikelihoodPolicy.SHARED_ISOTROPIC_GAUSSIAN: PolicyClassification.APPROVED_OPTIONAL,
}


class _FloorPreservingPositive(Positive):
    """Positive transform with an optimizer-visible floor and unchanged interior."""

    def __init__(self, floor: float) -> None:
        super().__init__()
        self._floor = float(floor)
        self.lower_bound.fill_(floor)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        transformed = super().transform(tensor)
        # Use the immutable scalar here because this constraint is registered after
        # MultiTaskGP has already followed its training data onto the target device.
        # A newly registered constraint buffer can therefore still reside on CPU.
        return transformed.clamp_min(self._floor) if self.enforced else transformed


class GPFitStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class GPFitFailureKind(str, Enum):
    ITERATION_EXHAUSTED = "iteration_exhausted"
    NONFINITE_OBJECTIVE = "nonfinite_objective"
    INVALID_PARAMETERS = "invalid_parameters"
    COVARIANCE_REJECTED = "covariance_rejected"
    OPTIMIZER_FAILURE = "optimizer_failure"
    INITIAL_GP_FIT_FAILURE = "initial_gp_fit_failure"
    RECOVERY_EXHAUSTED = "gp_fit_recovery_exhausted"
    CONSECUTIVE_FIXED_CONDITIONING_PROHIBITED = (
        "consecutive_fixed_conditioning_prohibited"
    )
    OTHER = "other"


@dataclass(frozen=True)
class SurrogateTrainingData:
    """Physical theta [N,P] paired with frozen-standardized responses [N,M].

    Rows/observation IDs share evaluation order; columns share response order.
    Tensors must be contiguous finite float32/float64 on the same device. This
    container validates but does not clone them; do not mutate retained training
    tensors after fitting. Normalization/task expansion is owned by _build_model.
    """

    theta: torch.Tensor
    response_standardized: torch.Tensor
    parameter_spec: ParameterSpec
    observation_ids: tuple[str, ...]
    transform_id: str

    def __post_init__(self) -> None:
        if not self.transform_id:
            raise ValueError("surrogate training data requires a transform_id")
        if not isinstance(self.theta, torch.Tensor) or not isinstance(
            self.response_standardized, torch.Tensor
        ):
            raise TypeError("surrogate training values must be Torch tensors")
        if self.theta.ndim != 2 or self.theta.shape[1] != self.parameter_spec.dimension:
            raise ValueError("training theta must have shape [N,P]")
        if (
            self.response_standardized.ndim != 2
            or self.response_standardized.shape[0] != self.theta.shape[0]
            or self.response_standardized.shape[1] < 1
        ):
            raise ValueError("standardized training responses must have shape [N,M]")
        if self.theta.shape[0] < 2:
            raise ValueError("surrogate training requires at least two observations")
        if self.theta.dtype not in {torch.float32, torch.float64}:
            raise ValueError("surrogate training requires float32 or float64")
        if (
            self.response_standardized.dtype != self.theta.dtype
            or self.response_standardized.device != self.theta.device
        ):
            raise ValueError("training theta and responses must share dtype and device")
        if not self.theta.is_contiguous() or not self.response_standardized.is_contiguous():
            raise ValueError("surrogate training tensors must be contiguous")
        if not bool(
            torch.isfinite(self.theta).all()
            and torch.isfinite(self.response_standardized).all()
        ):
            raise ValueError("surrogate training tensors must be finite")
        if len(self.observation_ids) != self.theta.shape[0]:
            raise ValueError("observation_ids must identify every training row")
        if len(set(self.observation_ids)) != len(self.observation_ids) or any(
            not item for item in self.observation_ids
        ):
            raise ValueError("observation_ids must be unique and nonempty")
        if any(axis.transformation != "identity" for axis in self.parameter_spec.axes):
            raise ValueError("Prompt 4A GP supports identity parameter transformations only")
        lower = torch.tensor(
            self.parameter_spec.lower, dtype=self.theta.dtype, device=self.theta.device
        )
        upper = torch.tensor(
            self.parameter_spec.upper, dtype=self.theta.dtype, device=self.theta.device
        )
        if bool(((self.theta < lower) | (self.theta > upper)).any()):
            raise ValueError("training theta lies outside declared parameter bounds")

    @property
    def parameter_dim(self) -> int:
        return self.theta.shape[1]

    @property
    def response_dim(self) -> int:
        return self.response_standardized.shape[1]

    def identity(self) -> str:
        semantic = {
            "parameter_names": tuple(axis.name for axis in self.parameter_spec.axes),
            "lower": self.parameter_spec.lower,
            "upper": self.parameter_spec.upper,
            "observation_ids": self.observation_ids,
            "transform_id": self.transform_id,
            "dtype": str(self.theta.dtype).removeprefix("torch."),
            "theta": self.theta.detach().cpu().tolist(),
            "response_standardized": self.response_standardized.detach().cpu().tolist(),
        }
        return stable_hash("surrogate-training-v1", semantic)


@dataclass(frozen=True)
class GPFitConfig:
    """Declared fit/validation policies, not a paper-specific numerical profile.

    The generic max_iterations default is 100; the accepted B40/B100 scientific
    profiles explicitly supply 12. Neither value is inferred from the data.
    Posterior precision/recovery choices are separate from optimizer settings.
    """

    fit_seed: int
    fit_seed_coordinate: SeedCoordinate
    likelihood: LikelihoodPolicy = LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN
    task_rank: int = 1
    normalize_parameters: bool = True
    max_iterations: int = 100
    jitter: float = 1e-8
    symmetry_tolerance: float = 1e-10
    psd_tolerance: float = 1e-10
    covariance_validation_policy: CovarianceValidationPolicy | str = (
        CovarianceValidationPolicy.ABSOLUTE_V1
    )
    posterior_numerics_policy: PosteriorNumericsPolicy | str = (
        PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE
    )
    task_variance_floor_policy: TaskVarianceFloorPolicy | str = (
        TaskVarianceFloorPolicy.NONE
    )
    task_variance_floor: float = 0.0
    task_variance_floor_reference_scale: float | None = None
    task_variance_floor_epsilon_multiplier: float | None = None
    recovery_policy: GPFitRecoveryPolicy | str = GPFitRecoveryPolicy.COLD_ONLY
    max_warm_start_states: int = 0

    @classmethod
    def from_seed_deriver(
        cls,
        seed_deriver: SeedDeriver,
        fit_seed_coordinate: SeedCoordinate,
        **settings: Any,
    ) -> "GPFitConfig":
        return cls(
            fit_seed=seed_deriver.seed_for(fit_seed_coordinate),
            fit_seed_coordinate=fit_seed_coordinate,
            **settings,
        )

    def __post_init__(self) -> None:
        if isinstance(self.fit_seed, bool) or not isinstance(self.fit_seed, int):
            raise TypeError("fit_seed must be an integer")
        if self.fit_seed < 0:
            raise ValueError("fit_seed must be nonnegative")
        if not isinstance(self.fit_seed_coordinate, SeedCoordinate):
            raise TypeError("fit_seed_coordinate must be a SeedCoordinate")
        try:
            object.__setattr__(self, "likelihood", LikelihoodPolicy(self.likelihood))
        except ValueError as exc:
            raise ValueError(f"unknown likelihood policy: {self.likelihood!r}") from exc
        try:
            object.__setattr__(
                self,
                "covariance_validation_policy",
                CovarianceValidationPolicy(self.covariance_validation_policy),
            )
        except ValueError as exc:
            raise ValueError(
                "unknown covariance validation policy: "
                f"{self.covariance_validation_policy!r}"
            ) from exc
        try:
            object.__setattr__(
                self,
                "posterior_numerics_policy",
                PosteriorNumericsPolicy(self.posterior_numerics_policy),
            )
        except ValueError as exc:
            raise ValueError(
                "unknown posterior numerics policy: "
                f"{self.posterior_numerics_policy!r}"
            ) from exc
        try:
            object.__setattr__(
                self,
                "task_variance_floor_policy",
                TaskVarianceFloorPolicy(self.task_variance_floor_policy),
            )
        except ValueError as exc:
            raise ValueError(
                "unknown task variance floor policy: "
                f"{self.task_variance_floor_policy!r}"
            ) from exc
        try:
            object.__setattr__(
                self, "recovery_policy", GPFitRecoveryPolicy(self.recovery_policy)
            )
        except ValueError as exc:
            raise ValueError(f"unknown GP fit recovery policy: {self.recovery_policy!r}") from exc
        if isinstance(self.task_rank, bool) or self.task_rank < 1:
            raise ValueError("task_rank must be a positive integer")
        if isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        for name, value, positive in (
            ("jitter", self.jitter, True),
            ("symmetry_tolerance", self.symmetry_tolerance, False),
            ("psd_tolerance", self.psd_tolerance, False),
        ):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                relation = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {relation}")
        if not math.isfinite(self.task_variance_floor) or self.task_variance_floor < 0:
            raise ValueError("task_variance_floor must be finite and nonnegative")
        if self.task_variance_floor_policy is TaskVarianceFloorPolicy.NONE:
            if self.task_variance_floor != 0:
                raise ValueError("task_variance_floor must be zero under policy 'none'")
            if (
                self.task_variance_floor_reference_scale is not None
                or self.task_variance_floor_epsilon_multiplier is not None
            ):
                raise ValueError("floor scale provenance requires an active floor policy")
        else:
            if self.task_variance_floor <= 0:
                raise ValueError("an active task variance floor must be positive")
            if (
                self.task_variance_floor_reference_scale is None
                or not math.isfinite(self.task_variance_floor_reference_scale)
                or self.task_variance_floor_reference_scale <= 0
            ):
                raise ValueError("active floor requires a positive finite reference scale")
            if (
                self.task_variance_floor_epsilon_multiplier is None
                or not math.isfinite(self.task_variance_floor_epsilon_multiplier)
                or self.task_variance_floor_epsilon_multiplier <= 0
            ):
                raise ValueError("active floor requires a positive finite epsilon multiplier")
        if isinstance(self.max_warm_start_states, bool) or self.max_warm_start_states < 0:
            raise ValueError("max_warm_start_states must be a nonnegative integer")
        if (
            self.recovery_policy is GPFitRecoveryPolicy.COLD_ONLY
            and self.max_warm_start_states != 0
        ):
            raise ValueError("cold-only recovery does not accept a warm-state limit")
        if (
            self.recovery_policy
            is GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1
            and self.max_warm_start_states != 2
        ):
            raise ValueError("graph-M20 recovery requires exactly two warm-start states")
        if self.recovery_policy is GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1:
            if (
                self.task_variance_floor_policy
                is not TaskVarianceFloorPolicy.STANDARDIZED_ABSOLUTE_V1
                or self.task_variance_floor != GRAPH_M20_TASK_VARIANCE_FLOOR
                or self.task_variance_floor_reference_scale
                != GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE
                or self.task_variance_floor_epsilon_multiplier
                != GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER
            ):
                raise ValueError(
                    "graph-M20 recovery requires its registered fixed task-variance floor"
                )

    def provenance(self) -> dict[str, object]:
        result: dict[str, object] = {
            "fit_seed": self.fit_seed,
            "fit_seed_coordinate": {
                "namespace": self.fit_seed_coordinate.namespace,
                "coordinates": self.fit_seed_coordinate.coordinates,
            },
            "likelihood_policy": self.likelihood.value,
            "likelihood_classification": LIKELIHOOD_POLICIES[self.likelihood].value,
            "task_rank": self.task_rank,
            "normalize_parameters": self.normalize_parameters,
            "optimizer": "botorch.fit_gpytorch_mll_scipy",
            "max_iterations": self.max_iterations,
            "jitter": self.jitter,
            "symmetry_tolerance": self.symmetry_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "covariance_validation_policy": self.covariance_validation_policy.value,
            "posterior_numerics_policy": self.posterior_numerics_policy.value,
            "posterior_numerics_classification": (
                POSTERIOR_NUMERICS_POLICIES[self.posterior_numerics_policy].value
            ),
            "recovery_policy": self.recovery_policy.value,
            "recovery_policy_classification": (
                GP_FIT_RECOVERY_POLICIES[self.recovery_policy].value
            ),
            "max_warm_start_states": self.max_warm_start_states,
        }
        if self.task_variance_floor_policy is not TaskVarianceFloorPolicy.NONE:
            result.update(
                {
                    "task_variance_floor_policy": self.task_variance_floor_policy.value,
                    "task_variance_floor": self.task_variance_floor,
                    "task_variance_floor_reference_scale": (
                        self.task_variance_floor_reference_scale
                    ),
                    "task_variance_floor_epsilon_multiplier": (
                        self.task_variance_floor_epsilon_multiplier
                    ),
                }
            )
        if self.recovery_policy is not GPFitRecoveryPolicy.COLD_ONLY:
            result.update(
                {
                    "restart_count": GRAPH_M20_RESTART_COUNT,
                    "fixed_conditioning_policy": (
                        "condition_previous_gp_on_current_data_v1"
                    ),
                    "maximum_consecutive_fixed_conditioning_updates": 1,
                }
            )
        return result


@dataclass(frozen=True)
class GPFitDiagnostic:
    status: GPFitStatus
    fit_id: str
    training_data_id: str
    model_id: str | None
    elapsed_seconds: float
    warnings: tuple[str, ...]
    failure_type: str | None
    failure_message: str | None
    validation_posterior_calls: int
    posterior_validation_diagnostics: dict[str, object] | None
    optimizer_result: dict[str, object] | None
    failure_kind: GPFitFailureKind | None
    failure_stage: str | None
    exception_chain: tuple[dict[str, str], ...]
    configuration: dict[str, object]
    execution: dict[str, object]
    recovery_attempts: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class GPWarmStartState:
    """Fitted parameter snapshot and provenance for compatible later fits.

    parameter_state is copied by warm_start_state(), not by this record. It carries
    model/likelihood state rather than past training observations; recovery decides
    how it may initialize or condition a model on the current dataset. The counter
    identifies an already fixed-conditioned state for the recovery guard.
    """
    source_fit_id: str
    source_model_id: str
    compatibility_id: str
    parameter_state: dict[str, torch.Tensor]
    consecutive_fixed_conditioning_updates: int = 0
    source_attempt_kind: str = "optimized"

    def __post_init__(self) -> None:
        if not self.source_fit_id or not self.source_model_id or not self.compatibility_id:
            raise ValueError("warm-start identities must be nonempty")
        if not self.parameter_state:
            raise ValueError("warm-start parameter state must be nonempty")
        if self.consecutive_fixed_conditioning_updates not in {0, 1}:
            raise ValueError("consecutive fixed-conditioning count must be zero or one")
        if self.source_attempt_kind not in {
            "optimized",
            "condition_previous_gp_on_current_data_v1",
        }:
            raise ValueError("unknown warm-start source attempt kind")


@dataclass(frozen=True)
class ObservationPosteriorSummary:
    """Native-precision standardized moments including likelihood noise.

    Mean [M] or [H,M] and covariance [M,M] or [H,M,M] preserve task order.
    This lightweight record is separate from the validated latent moments used
    by compatibility geometry; it does not perform their covariance validation.
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    transform_id: str
    model_id: str


class ExactMultiTaskGPSurrogate:
    """Physical-coordinate queries into a fitted standardized-response GP.

    Queries [P] or [H,P] must match the training dtype/device and bounds. Public
    latent moments preserve response order and transform identity; observation
    noise is included only by the explicitly named observation-posterior API.
    The fitted BoTorch model and any promoted covariance cache remain private.
    """

    def __init__(
        self,
        model: MultiTaskGP,
        *,
        training_data: SurrogateTrainingData,
        model_id: str,
        config: GPFitConfig,
        fit_id: str,
        consecutive_fixed_conditioning_updates: int = 0,
        source_attempt_kind: str = "optimized",
    ) -> None:
        self._model = model
        self._training_data = training_data
        self.model_id = model_id
        self.transform_id = training_data.transform_id
        self.parameter_dim = training_data.parameter_dim
        self.response_dim = training_data.response_dim
        self.dtype = training_data.theta.dtype
        self.device = training_data.theta.device
        self._config = config
        self._fit_id = fit_id
        self._consecutive_fixed_conditioning_updates = (
            consecutive_fixed_conditioning_updates
        )
        self._source_attempt_kind = source_attempt_kind
        self._float64_covariance_model: MultiTaskGP | None = None
        self._float64_covariance_model_build_seconds = 0.0

    def _validate_query(self, theta: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not isinstance(theta, torch.Tensor):
            raise TypeError("posterior theta must be a torch.Tensor")
        scalar = theta.ndim == 1
        if scalar:
            theta = theta.unsqueeze(0)
        if theta.ndim != 2 or theta.shape[1] != self.parameter_dim:
            raise ValueError("posterior theta must have shape [P] or [H,P]")
        if theta.dtype != self.dtype or theta.device != self.device:
            raise ValueError("posterior theta must match surrogate dtype and device")
        if not theta.is_contiguous() or not bool(torch.isfinite(theta).all()):
            raise ValueError("posterior theta must be contiguous and finite")
        lower = torch.tensor(
            self._training_data.parameter_spec.lower, dtype=self.dtype, device=self.device
        )
        upper = torch.tensor(
            self._training_data.parameter_spec.upper, dtype=self.dtype, device=self.device
        )
        if bool(((theta < lower) | (theta > upper)).any()):
            raise ValueError("posterior theta lies outside declared parameter bounds")
        normalized = (theta - lower) / (upper - lower)
        return (normalized if self._config.normalize_parameters else theta), scalar

    def _native_moments(
        self, theta: torch.Tensor, *, observation_noise: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, scalar = self._validate_query(theta)
        self._model.eval()
        posterior = self._model.posterior(
            query.unsqueeze(-2),
            output_indices=list(range(self.response_dim)),
            observation_noise=observation_noise,
        )
        mean = posterior.mean.squeeze(-2)
        covariance = posterior.covariance_matrix
        if scalar:
            mean = mean.squeeze(0)
            covariance = covariance.squeeze(0)
        return mean.contiguous(), covariance.contiguous()

    def _mean(self, theta: torch.Tensor, *, observation_noise: bool) -> torch.Tensor:
        query, scalar = self._validate_query(theta)
        self._model.eval()
        posterior = self._model.posterior(
            query.unsqueeze(-2),
            output_indices=list(range(self.response_dim)),
            observation_noise=observation_noise,
        )
        mean = posterior.mean.squeeze(-2)
        return (mean.squeeze(0) if scalar else mean).contiguous()

    def _joint_native_moments(
        self, theta: torch.Tensor, *, observation_noise: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, _ = self._validate_query(theta)
        self._model.eval()
        posterior = self._model.posterior(
            query,
            output_indices=list(range(self.response_dim)),
            observation_noise=observation_noise,
        )
        covariance = self._point_major_joint_covariance(
            posterior.covariance_matrix,
            point_count=query.shape[0],
            interleaved=bool(
                getattr(posterior.distribution, "_interleaved", True)
            ),
        )
        return posterior.mean.contiguous(), covariance

    def _point_major_joint_covariance(
        self,
        covariance: torch.Tensor,
        *,
        point_count: int,
        interleaved: bool,
    ) -> torch.Tensor:
        """Order full covariance [H*M,H*M] by point first, then response task.

        BoTorch may store all points of one task together. Reindex both axes when
        needed so joint draws align with mean.reshape(-1); retain cross-point blocks.
        An already interleaved contiguous matrix need not acquire new storage.
        """
        if interleaved:
            return covariance.contiguous()
        permutation = (
            torch.arange(
                point_count * self.response_dim,
                device=covariance.device,
            )
            .reshape(self.response_dim, point_count)
            .transpose(0, 1)
            .reshape(-1)
        )
        return covariance.index_select(0, permutation).index_select(
            1, permutation
        ).contiguous()

    def _joint_mixed_precision_covariance(
        self, theta: torch.Tensor, *, observation_noise: bool
    ) -> tuple[torch.Tensor, dict[str, object], dict[str, object]]:
        started = perf_counter()
        model_existed = self._float64_covariance_model is not None
        query, _ = self._validate_query(theta)
        promoted = self._promoted_covariance_model()
        posterior = promoted.posterior(
            query.double(),
            output_indices=list(range(self.response_dim)),
            observation_noise=observation_noise,
        )
        raw64 = self._point_major_joint_covariance(
            posterior.covariance_matrix,
            point_count=query.shape[0],
            interleaved=bool(
                getattr(posterior.distribution, "_interleaved", True)
            ),
        )
        symmetric64, promoted_diagnostics = structurally_symmetric_materialize(raw64)
        returned = symmetric64.to(dtype=self.dtype).contiguous()
        returned, returned_diagnostics = structurally_symmetric_materialize(returned)
        numerics = {
            "policy_id": self._config.posterior_numerics_policy.value,
            "interface": "joint_point_task_latent_covariance_v1",
            "point_major_flattening": True,
            "fit_dtype": str(self.dtype),
            "covariance_compute_dtype": str(raw64.dtype),
            "returned_dtype": str(returned.dtype),
            "promotion_occurred": True,
            "additional_fit_attempts": 0,
            "covariance_path_seconds": perf_counter() - started,
            "float64_model_cache_hit": model_existed,
            "float64_model_build_seconds": (
                0.0 if model_existed else self._float64_covariance_model_build_seconds
            ),
        }
        return returned, returned_diagnostics, numerics

    def _promoted_covariance_model(self) -> MultiTaskGP:
        """Cache a float64 copy of fitted state for covariance solves, without refit.

        Copy both training data and learned state, then freeze copied parameters;
        constructing this cache does not optimize new hyperparameters.
        """
        if self._float64_covariance_model is None:
            started = perf_counter()
            promoted_data = replace(
                self._training_data,
                theta=self._training_data.theta.double().contiguous(),
                response_standardized=(
                    self._training_data.response_standardized.double().contiguous()
                ),
            )
            promoted = _build_model(promoted_data, self._config)
            promoted_state = promoted.state_dict()
            promoted.load_state_dict(
                {
                    name: value.detach().to(dtype=promoted_state[name].dtype)
                    for name, value in self._model.state_dict().items()
                },
                strict=True,
            )
            for parameter in promoted.parameters():
                parameter.requires_grad_(False)
            promoted.eval()
            self._float64_covariance_model = promoted
            self._float64_covariance_model_build_seconds = perf_counter() - started
        return self._float64_covariance_model

    def _mixed_precision_covariance(
        self, theta: torch.Tensor, *, observation_noise: bool
    ) -> tuple[torch.Tensor, dict[str, object], dict[str, object]]:
        started = perf_counter()
        model_existed = self._float64_covariance_model is not None
        query, scalar = self._validate_query(theta)
        promoted = self._promoted_covariance_model()
        posterior = promoted.posterior(
            query.double().unsqueeze(-2),
            output_indices=list(range(self.response_dim)),
            observation_noise=observation_noise,
        )
        raw64 = posterior.covariance_matrix
        if scalar:
            raw64 = raw64.squeeze(0)
        symmetric64, promoted_diagnostics = structurally_symmetric_materialize(
            raw64.contiguous()
        )
        returned = symmetric64.to(dtype=self.dtype).contiguous()
        returned, returned_diagnostics = structurally_symmetric_materialize(returned)
        numerics = {
            "policy_id": self._config.posterior_numerics_policy.value,
            "fit_dtype": str(self.dtype),
            "covariance_compute_dtype": str(raw64.dtype),
            "returned_dtype": str(returned.dtype),
            "promotion_occurred": True,
            "posterior_mean_compute_dtype": str(self.dtype),
            "promoted_validation": promoted_diagnostics,
            "returned_validation": returned_diagnostics,
            "additional_fit_attempts": 0,
            "covariance_path_seconds": perf_counter() - started,
            "float64_model_cache_hit": model_existed,
            "float64_model_build_seconds": (
                0.0
                if model_existed
                else self._float64_covariance_model_build_seconds
            ),
            "tensor_conversions": {
                "training_theta_to_float64_on_cache_build": 0 if model_existed else 1,
                "training_response_to_float64_on_cache_build": 0 if model_existed else 1,
                "fitted_state_to_float64_on_cache_build": 0 if model_existed else 1,
                "query_to_float64": 1,
                "covariance_to_return_dtype": 1,
                "host_device_transfers": 0,
            },
        }
        return returned, returned_diagnostics, numerics

    def posterior_mean(self, theta: torch.Tensor) -> LatentPosteriorMeanSummary:
        """Return standardized latent means [M] or [H,M], without observation noise."""
        return LatentPosteriorMeanSummary(
            self._mean(theta, observation_noise=False),
            self.transform_id,
            self.model_id,
        )

    def posterior(self, theta: torch.Tensor) -> LatentPosteriorSummary:
        """Return latent means and within-point task covariances, in response order.

        Query [P] gives [M] and [M,M]; [H,P] gives [H,M] and [H,M,M], not a
        cross-point covariance. Under the mixed-precision policy, means use fit
        precision, covariance solves use copied float64 state, and covariance is
        returned in fit dtype/device. Validate symmetry/PSD under the configured
        policy; promotion performs no new fit and adds no observation noise.
        """
        mean = self._mean(theta, observation_noise=False)
        materialization_diagnostics = None
        numerics_diagnostics = {
            "policy_id": self._config.posterior_numerics_policy.value,
            "fit_dtype": str(self.dtype),
            "covariance_compute_dtype": str(self.dtype),
            "returned_dtype": str(self.dtype),
            "promotion_occurred": False,
            "posterior_mean_compute_dtype": str(self.dtype),
            "additional_fit_attempts": 0,
        }
        if (
            self._config.posterior_numerics_policy
            is PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE
            and self.dtype is torch.float32
        ):
            covariance, materialization_diagnostics, numerics_diagnostics = (
                self._mixed_precision_covariance(theta, observation_noise=False)
            )
        else:
            if (
                self._config.posterior_numerics_policy
                is PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC
                and self.dtype is not torch.float64
            ):
                raise ValueError(
                    "full_float64_diagnostic requires a float64-fitted surrogate"
                )
            _, covariance = self._native_moments(theta, observation_noise=False)
        if (
            materialization_diagnostics is None
            and self._config.covariance_validation_policy
            is CovarianceValidationPolicy.STRUCTURAL_SYMMETRIC_MATERIALIZATION_V1
        ):
            covariance, materialization_diagnostics = (
                structurally_symmetric_materialize(covariance)
            )
        return LatentPosteriorSummary(
            mean,
            covariance,
            self.transform_id,
            self.model_id,
            symmetry_tolerance=self._config.symmetry_tolerance,
            psd_tolerance=self._config.psd_tolerance,
            covariance_validation_policy=(
                self._config.covariance_validation_policy
            ),
            structural_materialization_diagnostics=materialization_diagnostics,
            posterior_numerics_diagnostics=numerics_diagnostics,
        )

    def joint_posterior(self, theta: torch.Tensor) -> JointLatentPosteriorSummary:
        """Return mean [H,M] and full latent covariance [H*M,H*M] for theta [H,P].

        Queries are physical parameters in training dtype/device and response columns
        retain training task order. Covariance is point-major and excludes likelihood
        noise. The float32-fit/float64-covariance policy uses the fixed-parameter promoted
        cache, validates materialization and returns the training dtype; it does not
        refit hyperparameters. A query may lazily create that cache.
        """

        if not isinstance(theta, torch.Tensor) or theta.ndim != 2:
            raise ValueError("joint posterior theta must have shape [H,P]")
        mean, native_covariance = self._joint_native_moments(
            theta, observation_noise=False
        )
        materialization_diagnostics: dict[str, object] | None = None
        if (
            self._config.posterior_numerics_policy
            is PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE
            and self.dtype is torch.float32
        ):
            covariance, materialization_diagnostics, _ = (
                self._joint_mixed_precision_covariance(
                    theta, observation_noise=False
                )
            )
        else:
            covariance, materialization_diagnostics = structurally_symmetric_materialize(
                native_covariance
            )
        return JointLatentPosteriorSummary(
            mean=mean,
            covariance=covariance,
            transform_id=self.transform_id,
            model_id=self.model_id,
            materialization_diagnostics=materialization_diagnostics,
        )

    def observation_posterior(self, theta: torch.Tensor) -> ObservationPosteriorSummary:
        """Return per-point predictive moments including fitted observation noise.

        Physical theta [P] or [H,P] yields standardized mean [M] or [H,M] and covariance
        [M,M] or [H,M,M]. This native-precision path does not use the promoted latent-
        covariance cache or its validation policy. Compatibility uses posterior()
        instead, so observation noise is not counted as latent model uncertainty.
        """
        mean, covariance = self._native_moments(theta, observation_noise=True)
        return ObservationPosteriorSummary(mean, covariance, self.transform_id, self.model_id)

    def provenance(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_family": "botorch.MultiTaskGP",
            "kernel_family": "RBF x positive-index task covariance",
            "posterior_covariance": "latent_standardized",
            "posterior_mean_interface": "mean_only_without_covariance_materialization",
            "joint_latent_interface": "point_major_full_point_task_covariance_v1",
            "observation_posterior_covariance": "observation_inclusive_standardized",
            "parameter_dim": self.parameter_dim,
            "response_dim": self.response_dim,
            "training_data_id": self._training_data.identity(),
            "transform_id": self.transform_id,
            "dtype": str(self.dtype).removeprefix("torch."),
            "device": str(self.device),
            **self._config.provenance(),
        }

    def warm_start_state(self) -> GPWarmStartState:
        """Detach and clone fitted state for later recovery without sharing storage.

        The compatibility identity binds reusable parameters to the model construction
        settings; the runner supplies the next dataset separately. Capturing this
        snapshot performs no optimization and does not advance random streams.
        """
        return GPWarmStartState(
            source_fit_id=self._fit_id,
            source_model_id=self.model_id,
            compatibility_id=_warm_start_compatibility_id(
                self._training_data, self._config
            ),
            parameter_state={
                name: value.detach().clone()
                for name, value in self._model.state_dict().items()
            },
            consecutive_fixed_conditioning_updates=(
                self._consecutive_fixed_conditioning_updates
            ),
            source_attempt_kind=self._source_attempt_kind,
        )


@dataclass(frozen=True)
class GPFitResult:
    surrogate: ExactMultiTaskGPSurrogate | None
    diagnostic: GPFitDiagnostic


def _likelihood(policy: LikelihoodPolicy, *, response_dim: int, task_feature: int):
    """Build task-diagonal or shared Gaussian noise with the same registered prior.

    Returns HadamardGaussianLikelihood or GaussianLikelihood. The LogNormal
    noise prior and inferred-noise lower bound are fitting assumptions, distinct
    from numerical Cholesky jitter used during optimization.
    """
    prior = LogNormalPrior(loc=-4.0, scale=1.0)
    constraint = GreaterThan(
        MIN_INFERRED_NOISE_LEVEL, transform=None, initial_value=prior.mode
    )
    if policy is LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN:
        return HadamardGaussianLikelihood(
            num_tasks=response_dim,
            noise_prior=prior,
            noise_constraint=constraint,
            task_feature_index=task_feature,
        )
    return GaussianLikelihood(noise_prior=prior, noise_constraint=constraint)


def _build_model(data: SurrogateTrainingData, config: GPFitConfig) -> MultiTaskGP:
    """Expand ordered [N,P]/[N,M] data into [N*M,P+1]/[N*M,1] model inputs.

    After optional affine parameter normalization, row i*M+p contains observation
    i with task index p; flatten responses in exactly that same order. Preserve
    training dtype/device. outcome_transform=None prevents a second response
    standardization on top of the frozen transform.

    In the pinned BoTorch implementation, MultiTaskGP supplies task-specific
    constant means, an ARD RBF input kernel times a PositiveIndexKernel task
    covariance, and registered length-scale/task priors. Task rank is capped by
    M; the task kernel constrains correlations to nonnegative values. Add the
    declared Gaussian likelihood and optional task-variance floor unchanged.
    """
    lower = torch.tensor(
        data.parameter_spec.lower, dtype=data.theta.dtype, device=data.theta.device
    )
    upper = torch.tensor(
        data.parameter_spec.upper, dtype=data.theta.dtype, device=data.theta.device
    )
    model_theta = (
        (data.theta - lower) / (upper - lower)
        if config.normalize_parameters
        else data.theta
    )
    design = model_theta.repeat_interleave(data.response_dim, dim=0)
    tasks = (
        torch.arange(data.response_dim, dtype=data.theta.dtype, device=data.theta.device)
        .repeat(data.theta.shape[0])
        .unsqueeze(-1)
    )
    augmented_theta = torch.cat((design, tasks), dim=-1)
    augmented_response = data.response_standardized.reshape(-1, 1)
    likelihood = _likelihood(
        config.likelihood,
        response_dim=data.response_dim,
        task_feature=augmented_theta.shape[-1] - 1,
    )
    model = MultiTaskGP(
        augmented_theta,
        augmented_response,
        task_feature=-1,
        all_tasks=list(range(data.response_dim)),
        output_tasks=list(range(data.response_dim)),
        rank=min(config.task_rank, data.response_dim),
        outcome_transform=None,
        likelihood=likelihood,
    )
    if config.task_variance_floor_policy is not TaskVarianceFloorPolicy.NONE:
        task_kernel = model.covar_module.kernels[1]
        task_kernel.register_constraint(
            "raw_var", _FloorPreservingPositive(config.task_variance_floor)
        )
    return model


def _warm_start_compatibility_id(
    data: SurrogateTrainingData, config: GPFitConfig
) -> str:
    return stable_hash(
        "gp-warm-start-compatibility-v1",
        {
            "model_family": "botorch.MultiTaskGP",
            "kernel_family": "RBF x positive-index task covariance",
            "likelihood": config.likelihood.value,
            "task_rank": config.task_rank,
            "normalize_parameters": config.normalize_parameters,
            "parameter_dim": data.parameter_dim,
            "response_dim": data.response_dim,
            "dtype": str(data.theta.dtype),
            "device": str(data.theta.device),
            "transform_id": data.transform_id,
            "task_variance_floor_policy": config.task_variance_floor_policy.value,
            "task_variance_floor": config.task_variance_floor,
            "task_variance_floor_reference_scale": (
                config.task_variance_floor_reference_scale
            ),
            "task_variance_floor_epsilon_multiplier": (
                config.task_variance_floor_epsilon_multiplier
            ),
            "jitter": config.jitter,
            "symmetry_tolerance": config.symmetry_tolerance,
            "psd_tolerance": config.psd_tolerance,
            "covariance_validation_policy": (
                config.covariance_validation_policy.value
            ),
            "posterior_numerics_policy": config.posterior_numerics_policy.value,
        },
    )


def _model_identity(
    model: MultiTaskGP, data: SurrogateTrainingData, config: GPFitConfig
) -> str:
    parameters = {
        name: parameter.detach().cpu().tolist()
        for name, parameter in sorted(model.named_parameters())
    }
    return stable_hash(
        "exact-multitask-gp-v1",
        {
            "training_data_id": data.identity(),
            "configuration": config.provenance(),
            "parameters": parameters,
        },
    )


def _execution_provenance(data: SurrogateTrainingData) -> dict[str, object]:
    return {
        "device": str(data.theta.device),
        "dtype": str(data.theta.dtype).removeprefix("torch."),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(data.theta.device)
            if data.theta.device.type == "cuda"
            else None
        ),
        "cpu_parallelism": CpuParallelismProvenance.inspect().provenance(),
    }


_FIT_MLL: Callable[..., object] = fit_gpytorch_mll


def _optimizer_result_provenance(result: object | None) -> dict[str, object] | None:
    if result is None:
        return None
    status = getattr(result, "status", None)
    status_name = getattr(status, "name", str(status))
    fval = float(getattr(result, "fval", float("nan")))
    return {
        "status": status_name.lower(),
        "success": status_name == "SUCCESS",
        "stopped": status_name == "STOPPED",
        "actual_iterations": int(getattr(result, "step", 0)),
        "final_objective": fval,
        "objective_finite": math.isfinite(fval),
        "gradient_norm": None,
        "gradient_norm_unavailable_reason": (
            "botorch_optimization_result_does_not_expose_final_gradient"
        ),
        "runtime_seconds": (
            None
            if getattr(result, "runtime", None) is None
            else float(getattr(result, "runtime"))
        ),
        "message": (
            None
            if getattr(result, "message", None) is None
            else str(getattr(result, "message"))
        ),
    }


def _exception_provenance(exc: BaseException | None) -> tuple[dict[str, str], ...]:
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _failure_kind(
    *,
    exc: BaseException,
    optimizer_exc: BaseException | None,
    optimizer_result: dict[str, object] | None,
    stage: str,
) -> GPFitFailureKind:
    underlying = optimizer_exc or exc
    combined = " ".join(
        (type(underlying).__name__, str(underlying), str(optimizer_result or {}))
    ).lower()
    if optimizer_result is not None and not bool(
        optimizer_result.get("objective_finite", True)
    ):
        return GPFitFailureKind.NONFINITE_OBJECTIVE
    if "iteration" in combined and "limit" in combined:
        return GPFitFailureKind.ITERATION_EXHAUSTED
    if "notpsd" in combined or "positive definite" in combined:
        return GPFitFailureKind.COVARIANCE_REJECTED
    if stage == "parameter_validation" or "nonfinite parameter" in combined:
        return GPFitFailureKind.INVALID_PARAMETERS
    if stage == "posterior_validation":
        return GPFitFailureKind.COVARIANCE_REJECTED
    if stage == "optimizer":
        return GPFitFailureKind.OPTIMIZER_FAILURE
    return GPFitFailureKind.OTHER


def _training_covariance_diagnostics(
    model: MultiTaskGP, data: SurrogateTrainingData
) -> dict[str, object]:
    """Inspect covariance structure without changing model parameters."""

    def matrix_metrics(value: torch.Tensor) -> dict[str, object]:
        dense = value.detach()
        symmetric = 0.5 * (dense + dense.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(symmetric.to(dtype=torch.float64))
        maximum = float(eigenvalues[-1])
        minimum = float(eigenvalues[0])
        positive = eigenvalues[eigenvalues > 0]
        condition = (
            float("inf")
            if positive.numel() == 0
            else maximum / float(positive[0])
        )
        return {
            "shape": list(dense.shape),
            "scale": float(dense.abs().max()),
            "symmetry_residual": float(
                (dense - dense.transpose(-1, -2)).abs().max()
            ),
            "minimum_symmetric_eigenvalue": minimum,
            "maximum_symmetric_eigenvalue": maximum,
            "positive_spectrum_condition_estimate": condition,
            "numerical_rank_float64": int(
                torch.linalg.matrix_rank(symmetric.to(dtype=torch.float64))
            ),
            "finite": bool(torch.isfinite(dense).all()),
        }

    document: dict[str, object] = {
        "dtype": str(data.theta.dtype),
        "dtype_epsilon": torch.finfo(data.theta.dtype).eps,
    }
    try:
        task_kernel = model.covar_module.kernels[1]
        task_covariance = task_kernel.covar_matrix.to_dense()
        document["task_covariance"] = matrix_metrics(task_covariance)
        document["task_diagonal_variances"] = (
            task_kernel.var.detach().cpu().tolist()
        )
        unnormalized = (
            task_kernel.covar_factor @ task_kernel.covar_factor.transpose(-1, -2)
            + torch.diag_embed(task_kernel.var)
        )
        document["task_target_normalization_denominator"] = float(
            unnormalized[
                task_kernel.target_task_index, task_kernel.target_task_index
            ].detach()
        )
        document["task_raw_var_minimum"] = float(task_kernel.raw_var.detach().min())
        document["task_raw_var_maximum"] = float(task_kernel.raw_var.detach().max())
        document["task_raw_factor_minimum"] = float(
            task_kernel.raw_covar_factor.detach().min()
        )
        document["task_raw_factor_maximum"] = float(
            task_kernel.raw_covar_factor.detach().max()
        )
        document["task_factor_norm"] = float(
            torch.linalg.vector_norm(task_kernel.covar_factor.detach())
        )
    except Exception as exc:
        document["task_covariance_error"] = f"{type(exc).__name__}: {exc}"
    try:
        lower = torch.tensor(
            data.parameter_spec.lower, dtype=data.theta.dtype, device=data.theta.device
        )
        upper = torch.tensor(
            data.parameter_spec.upper, dtype=data.theta.dtype, device=data.theta.device
        )
        inputs = (
            (data.theta - lower) / (upper - lower)
            if model._original_train_inputs is None
            else model._original_train_inputs[0][:: data.response_dim, :-1]
        )
        input_covariance = model.covar_module.kernels[0](inputs).to_dense()
        document["input_covariance"] = matrix_metrics(input_covariance)
    except Exception as exc:
        document["input_covariance_error"] = f"{type(exc).__name__}: {exc}"
    try:
        model_was_training = model.training
        likelihood_was_training = model.likelihood.training
        model.train()
        model.likelihood.train()
        with torch.no_grad():
            distribution = model(*model.train_inputs)
            full_covariance = model.likelihood(
                distribution, model.train_inputs[0]
            ).covariance_matrix
        document["training_covariance"] = matrix_metrics(full_covariance)
        model.train(model_was_training)
        model.likelihood.train(likelihood_was_training)
    except Exception as exc:
        document["training_covariance_error"] = f"{type(exc).__name__}: {exc}"
    try:
        document["likelihood_noise"] = (
            model.likelihood.noise.detach().cpu().reshape(-1).tolist()
        )
    except Exception as exc:
        document["likelihood_noise_error"] = f"{type(exc).__name__}: {exc}"
    return document


def _summarize_fit_optimizer_result(
    optimizer_observations: list[object],
    *,
    optimize: bool,
    initial_objective: float | None,
    config: GPFitConfig,
    last_valid_covariance: dict[str, object] | None,
    failure_covariance: dict[str, object] | None,
) -> dict[str, object] | None:
    """Describe the returned optimizer or fixed-conditioning attempt unchanged."""
    optimizer_document = _optimizer_result_provenance(
        optimizer_observations[-1] if optimizer_observations else None
    )
    if optimizer_document is None and not optimize and initial_objective is not None:
        optimizer_document = {
            "status": "not_run_fixed_conditioning",
            "success": True,
            "stopped": False,
            "actual_iterations": 0,
            "final_objective": initial_objective,
            "objective_finite": math.isfinite(initial_objective),
            "gradient_norm": None,
            "gradient_norm_unavailable_reason": "optimizer_not_run",
            "runtime_seconds": None,
            "message": "parameters fixed; posterior recomputed on current data",
        }
    if optimizer_document is not None:
        optimizer_document["configured_max_iterations"] = config.max_iterations
        optimizer_document["initial_objective"] = initial_objective
        final_objective = optimizer_document.get("final_objective")
        optimizer_document["objective_improvement"] = (
            None
            if initial_objective is None
            or not isinstance(final_objective, float)
            or not math.isfinite(final_objective)
            else initial_objective - final_objective
        )
        optimizer_document["last_valid_covariance"] = last_valid_covariance
        optimizer_document["failure_covariance"] = failure_covariance
    return optimizer_document


def fit_exact_multitask_gp(
    data: SurrogateTrainingData,
    *,
    config: GPFitConfig,
    warm_start_state: GPWarmStartState | None = None,
    initialization_kind: str | None = None,
    attempt_seed: int | None = None,
    optimize: bool = True,
) -> GPFitResult:
    """Initialize, fit once, and validate one exact correlated GP attempt.

    SurrogateTrainingData holds physical [N,P] inputs and standardized [N,M]
    responses. Within a restored Torch RNG scope, build the model with the
    effective attempt seed, then load a compatible warm state if supplied.
    The loss is negative ExactMarginalLogLikelihood including registered log
    priors (MAP-type, scaled per training scalar), not prior-free MML. This is a
    manuscript-wording issue for researcher review, not an alternative objective.

    Record the initial objective, run the configured SciPy fit with one library
    attempt and max_iterations, then check finite parameters and a midpoint
    latent posterior. optimize=False skips optimization but still conditions on
    current data and validates the resulting model. Success indicates these
    checks passed, not necessarily optimizer convergence. Exceptions become a
    failed diagnostic with no returned surrogate; the wrapper owns recovery.
    Training precision/device and posterior precision policy remain distinct.
    """
    training_id = data.identity()
    effective_seed = config.fit_seed if attempt_seed is None else attempt_seed
    attempt_kind = initialization_kind or (
        "deterministic_cold" if warm_start_state is None else "warm_optimized"
    )
    fit_semantic: dict[str, object] = {
        "training_data_id": training_id,
        "configuration": config.provenance(),
    }
    if warm_start_state is not None:
        fit_semantic["initialization"] = {
            "policy": "compatible_saved_parameter_state_v1",
            "source_fit_id": warm_start_state.source_fit_id,
            "source_model_id": warm_start_state.source_model_id,
        }
    fit_semantic["attempt"] = {
        "kind": attempt_kind,
        "seed": effective_seed,
        "optimize": optimize,
    }
    fit_id = stable_hash(
        "gp-fit-v1",
        fit_semantic,
    )
    captured: list[str] = []
    validation_posterior_calls = 0
    posterior_validation_diagnostics: dict[str, object] | None = None
    optimizer_observations: list[object] = []
    optimizer_exception: BaseException | None = None
    initial_objective: float | None = None
    failure_stage: str | None = None
    last_valid_state: dict[str, torch.Tensor] | None = None
    last_valid_covariance: dict[str, object] | None = None
    failure_covariance: dict[str, object] | None = None
    initial_parameters: dict[str, object] | None = None
    final_parameters: dict[str, object] | None = None
    start = perf_counter()
    cuda_devices = (
        [data.theta.device.index if data.theta.device.index is not None else 0]
        if data.theta.device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(effective_seed)
            if data.theta.device.type == "cuda":
                torch.cuda.manual_seed(effective_seed)
            with warnings.catch_warnings(record=True) as setup_warnings:
                warnings.simplefilter("always")
                model = _build_model(data, config)
            captured.extend(
                f"{item.category.__name__}: {item.message}"
                for item in setup_warnings
            )
            if warm_start_state is not None:
                expected = _warm_start_compatibility_id(data, config)
                if warm_start_state.compatibility_id != expected:
                    raise ValueError("incompatible GP warm-start state")
                model.load_state_dict(warm_start_state.parameter_state, strict=True)
            initial_parameters = {
                name: value.detach().cpu().tolist()
                for name, value in sorted(model.named_parameters())
            }
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with torch.no_grad():
                initial_value = get_loss_closure(mll)().sum()
                initial_objective = float(initial_value.detach())
                if not math.isfinite(initial_objective):
                    raise ValueError("initial GP objective is nonfinite")

            def recording_optimizer(*args: object, **kwargs: object) -> object:
                nonlocal optimizer_exception
                caller_callback = kwargs.pop("callback", None)

                def record_callback(parameters: object, result: object) -> None:
                    nonlocal last_valid_state, last_valid_covariance
                    optimizer_observations.append(result)
                    last_valid_state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                    if caller_callback is not None:
                        caller_callback(parameters, result)

                try:
                    result = fit_gpytorch_mll_scipy(
                        *args, **kwargs, callback=record_callback
                    )
                except BaseException as exc:
                    optimizer_exception = exc
                    nonlocal failure_covariance
                    failure_covariance = _training_covariance_diagnostics(model, data)
                    if last_valid_state is not None:
                        with warnings.catch_warnings(record=True) as diagnostic_warnings:
                            warnings.simplefilter("always")
                            last_valid_model = _build_model(data, config)
                        captured.extend(
                            f"{item.category.__name__}: {item.message}"
                            for item in diagnostic_warnings
                        )
                        last_valid_model.load_state_dict(last_valid_state, strict=True)
                        last_valid_covariance = _training_covariance_diagnostics(
                            last_valid_model, data
                        )
                    raise
                optimizer_observations.append(result)
                return result

            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                try:
                    if optimize:
                        failure_stage = "optimizer"
                        with cholesky_jitter(
                            float_value=config.jitter, double_value=config.jitter
                        ):
                            _FIT_MLL(
                                mll,
                                optimizer=recording_optimizer,
                                optimizer_kwargs={
                                    "options": {"maxiter": config.max_iterations}
                                },
                                max_attempts=1,
                            )
                finally:
                    captured.extend(
                        f"{item.category.__name__}: {item.message}" for item in emitted
                    )
            failure_stage = "parameter_validation"
            if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
                raise ValueError("fitted model contains nonfinite parameters")
            model.eval()
            final_parameters = {
                name: value.detach().cpu().tolist()
                for name, value in sorted(model.named_parameters())
            }
            last_valid_covariance = _training_covariance_diagnostics(model, data)
            model_id = _model_identity(model, data, config)
            surrogate = ExactMultiTaskGPSurrogate(
                model,
                training_data=data,
                model_id=model_id,
                config=config,
                fit_id=fit_id,
                consecutive_fixed_conditioning_updates=(
                    1
                    if attempt_kind
                    == "condition_previous_gp_on_current_data_v1"
                    else 0
                ),
                source_attempt_kind=(
                    "condition_previous_gp_on_current_data_v1"
                    if attempt_kind
                    == "condition_previous_gp_on_current_data_v1"
                    else "optimized"
                ),
            )
            midpoint = torch.tensor(
                [
                    (axis.lower + axis.upper) / 2
                    for axis in data.parameter_spec.axes
                ],
                dtype=data.theta.dtype,
                device=data.theta.device,
            )
            validation_posterior_calls += 1
            failure_stage = "posterior_validation"
            validation_summary = surrogate.posterior(midpoint)
            posterior_validation_diagnostics = (
                validation_summary.validation_diagnostics
            )
        status = GPFitStatus.SUCCESS
        failure_type = failure_message = None
        failure_kind = None
        failure_stage = None
        exception_chain: tuple[dict[str, str], ...] = ()
    except Exception as exc:
        surrogate = None
        model_id = None
        status = GPFitStatus.FAILED
        failure_type = type(exc).__name__
        failure_message = str(exc)
        optimizer_document = _optimizer_result_provenance(
            optimizer_observations[-1] if optimizer_observations else None
        )
        failure_kind = _failure_kind(
            exc=exc,
            optimizer_exc=optimizer_exception,
            optimizer_result=optimizer_document,
            stage=failure_stage or "other",
        )
        exception_chain = _exception_provenance(optimizer_exception)
        if not exception_chain or optimizer_exception is not exc:
            exception_chain = (*exception_chain, *_exception_provenance(exc))
    optimizer_document = _summarize_fit_optimizer_result(
        optimizer_observations,
        optimize=optimize,
        initial_objective=initial_objective,
        config=config,
        last_valid_covariance=last_valid_covariance,
        failure_covariance=failure_covariance,
    )
    initialization = {
        "kind": attempt_kind,
        "attempt_seed": effective_seed,
        "source_fit_id": (
            None if warm_start_state is None else warm_start_state.source_fit_id
        ),
        "source_model_id": (
            None if warm_start_state is None else warm_start_state.source_model_id
        ),
    }
    attempt_document: dict[str, object] = {
        "fit_executed": True,
        "initialization": initialization,
        "training_observation_count": int(data.theta.shape[0]),
        "training_covariance_dimension": int(
            data.theta.shape[0] * data.response_dim
        ),
        "status": status.value,
        "fit_id": fit_id,
        "model_id": model_id,
        "failure_kind": None if failure_kind is None else failure_kind.value,
        "failure_stage": failure_stage,
        "failure_category": (
            None
            if status is GPFitStatus.SUCCESS
            else (
                "posterior_materialization"
                if failure_stage == "posterior_validation"
                else "parameter_validation"
                if failure_stage == "parameter_validation"
                else "optimization"
                if failure_stage == "optimizer"
                else "conditioning"
                if not optimize
                else "other"
            )
        ),
        "failure_type": failure_type,
        "failure_message": failure_message,
        "optimizer_result": optimizer_document,
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "marginal_likelihood": (
            None
            if optimizer_document is None
            or not isinstance(optimizer_document.get("final_objective"), float)
            or not math.isfinite(optimizer_document["final_objective"])
            else -optimizer_document["final_objective"]
        ),
        "elapsed_seconds": perf_counter() - start,
    }
    diagnostic = GPFitDiagnostic(
        status=status,
        fit_id=fit_id,
        training_data_id=training_id,
        model_id=model_id,
        elapsed_seconds=perf_counter() - start,
        warnings=tuple(captured),
        failure_type=failure_type,
        failure_message=failure_message,
        validation_posterior_calls=validation_posterior_calls,
        posterior_validation_diagnostics=posterior_validation_diagnostics,
        optimizer_result=optimizer_document,
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        exception_chain=exception_chain,
        configuration=config.provenance(),
        execution=_execution_provenance(data),
        recovery_attempts=(attempt_document,),
    )
    return GPFitResult(surrogate=surrogate, diagnostic=diagnostic)


def fit_exact_multitask_gp_with_recovery(
    data: SurrogateTrainingData,
    *,
    config: GPFitConfig,
    warm_start_states: tuple[GPWarmStartState, ...] = (),
    training_version: int | None = None,
) -> GPFitResult:
    """Apply the declared recovery order and aggregate its actual fit diagnostics.

    cold_only delegates directly to one cold attempt. The optional graph policy
    tries compatible saved states in caller order (newest first in the runner),
    then a deterministic cold fit. If these fail, run four coordinate-seeded
    restarts and choose the usable finite objective maximum, keeping the earliest
    restart on an exact tie. The recorded 'marginal_likelihood' includes priors.

    At version zero, exhaustion is an initial-fit failure: no previous GP can
    be conditioned. Later, only the first compatible saved state may supply
    fixed-parameter conditioning on current observations, and never for two
    consecutive updates. If unavailable/invalid, return a failed result instead
    of a surrogate. Report every attempted/rejected stage and its work; do not
    discard already committed runner observations or silently relax policies.
    """
    if config.recovery_policy is GPFitRecoveryPolicy.COLD_ONLY:
        return fit_exact_multitask_gp(data, config=config)

    if training_version is None:
        training_version = 0 if not warm_start_states else 1
    if isinstance(training_version, bool) or training_version < 0:
        raise ValueError("training_version must be a nonnegative integer")

    expected = _warm_start_compatibility_id(data, config)
    documents: list[dict[str, object]] = []
    executed: list[GPFitResult] = []

    def append_result(result: GPFitResult, **annotations: object) -> None:
        executed.append(result)
        for item in result.diagnostic.recovery_attempts:
            document = dict(item)
            document.update(annotations)
            documents.append(document)

    def aggregate(selected: GPFitResult) -> GPFitResult:
        selected_fit_id = selected.diagnostic.fit_id
        selected_marked = False
        marked_documents: list[dict[str, object]] = []
        for document in documents:
            is_selected = bool(
                not selected_marked
                and document.get("fit_id") == selected_fit_id
                and document.get("status") == GPFitStatus.SUCCESS.value
            )
            selected_marked = selected_marked or is_selected
            marked_documents.append({**document, "selected": is_selected})
        marked = tuple(marked_documents)
        return GPFitResult(
            surrogate=selected.surrogate,
            diagnostic=replace(
                selected.diagnostic,
                elapsed_seconds=sum(item.diagnostic.elapsed_seconds for item in executed),
                warnings=tuple(
                    warning
                    for item in executed
                    for warning in item.diagnostic.warnings
                ),
                validation_posterior_calls=sum(
                    item.diagnostic.validation_posterior_calls for item in executed
                ),
                recovery_attempts=marked,
            ),
        )

    compatible_states: list[GPWarmStartState] = []
    if training_version > 0:
        for position, state in enumerate(
            warm_start_states[: config.max_warm_start_states]
        ):
            if state.compatibility_id != expected:
                documents.append(
                    {
                        "fit_executed": False,
                        "initialization": {
                            "kind": "warm_optimized",
                            "warm_position": position,
                            "source_fit_id": state.source_fit_id,
                            "source_model_id": state.source_model_id,
                        },
                        "status": "rejected_incompatible",
                        "reason": "warm_start_compatibility_id_mismatch",
                        "selected": False,
                    }
                )
                continue
            compatible_states.append(state)
            kind = "most_recent_warm_optimized" if position == 0 else (
                "penultimate_warm_optimized"
            )
            result = fit_exact_multitask_gp(
                data,
                config=config,
                warm_start_state=state,
                initialization_kind=kind,
            )
            append_result(result, recovery_stage=kind, warm_position=position)
            if result.surrogate is not None:
                return aggregate(result)

    cold = fit_exact_multitask_gp(
        data, config=config, initialization_kind="deterministic_cold"
    )
    append_result(cold, recovery_stage="deterministic_cold")
    if cold.surrogate is not None:
        return aggregate(cold)

    restart_results: list[tuple[int, GPFitResult, float]] = []
    for restart_index in range(GRAPH_M20_RESTART_COUNT):
        coordinate = {
            "namespace": "gp_fit_recovery",
            "coordinates": (
                *config.fit_seed_coordinate.coordinates,
                "training_version",
                training_version,
                config.recovery_policy.value,
                "restart",
                restart_index,
            ),
        }
        seed_hex = stable_hash(
            "gp-coordinate-restart-seed-v1",
            {
                "base_fit_seed": config.fit_seed,
                "base_coordinate_namespace": config.fit_seed_coordinate.namespace,
                **coordinate,
            },
        )
        restart_seed = int(seed_hex.rsplit("-", 1)[-1][:16], 16) & ((1 << 63) - 1)
        result = fit_exact_multitask_gp(
            data,
            config=config,
            initialization_kind="coordinate_seeded_restart",
            attempt_seed=restart_seed,
        )
        append_result(
            result,
            recovery_stage="coordinate_seeded_restart",
            restart_index=restart_index,
            restart_seed_coordinate=coordinate,
            restart_seed=restart_seed,
        )
        attempt = result.diagnostic.recovery_attempts[0]
        marginal_likelihood = attempt.get("marginal_likelihood")
        if (
            result.surrogate is not None
            and isinstance(marginal_likelihood, float)
            and math.isfinite(marginal_likelihood)
        ):
            restart_results.append((restart_index, result, marginal_likelihood))

    if restart_results:
        _, selected, _ = max(restart_results, key=lambda item: (item[2], -item[0]))
        return aggregate(selected)

    if training_version == 0:
        terminal = executed[-1]
        return GPFitResult(
            surrogate=None,
            diagnostic=replace(
                aggregate(terminal).diagnostic,
                status=GPFitStatus.FAILED,
                model_id=None,
                failure_kind=GPFitFailureKind.INITIAL_GP_FIT_FAILURE,
                failure_stage="initial_gp_fit_recovery",
                failure_type="InitialGPFitFailure",
                failure_message=(
                    "initial_gp_fit_failure: deterministic cold initialization and "
                    "all four coordinate-seeded restarts failed; no prior GP exists "
                    "for fixed conditioning"
                ),
                warnings=(
                    *aggregate(terminal).diagnostic.warnings,
                    "campaign_quality_warning_required_if_initial_failures_recur",
                ),
            ),
        )

    if not compatible_states:
        terminal = executed[-1]
        return GPFitResult(
            surrogate=None,
            diagnostic=replace(
                aggregate(terminal).diagnostic,
                failure_kind=GPFitFailureKind.RECOVERY_EXHAUSTED,
                failure_stage="fixed_conditioning_unavailable",
                failure_type="GPFitRecoveryExhausted",
                failure_message=(
                    "optimized recovery exhausted and no compatible prior GP exists "
                    "for fixed conditioning"
                ),
            ),
        )

    conditioning_source = compatible_states[0]
    if conditioning_source.consecutive_fixed_conditioning_updates >= 1:
        documents.append(
            {
                "fit_executed": False,
                "initialization": {
                    "kind": "condition_previous_gp_on_current_data_v1",
                    "source_fit_id": conditioning_source.source_fit_id,
                    "source_model_id": conditioning_source.source_model_id,
                },
                "status": "prohibited",
                "reason": "second_consecutive_fixed_conditioning_update",
                "selected": False,
            }
        )
        terminal = executed[-1]
        aggregated = aggregate(terminal)
        return GPFitResult(
            surrogate=None,
            diagnostic=replace(
                aggregated.diagnostic,
                failure_kind=(
                    GPFitFailureKind.CONSECUTIVE_FIXED_CONDITIONING_PROHIBITED
                ),
                failure_stage="fixed_conditioning_guard",
                failure_type="ConsecutiveFixedConditioningProhibited",
                failure_message=(
                    "a second consecutive fixed-hyperparameter conditioning update "
                    "is prohibited"
                ),
            ),
        )

    conditioned = fit_exact_multitask_gp(
        data,
        config=config,
        warm_start_state=conditioning_source,
        initialization_kind="condition_previous_gp_on_current_data_v1",
        optimize=False,
    )
    append_result(
        conditioned,
        recovery_stage="condition_previous_gp_on_current_data_v1",
    )
    if conditioned.surrogate is not None:
        return aggregate(conditioned)

    aggregated = aggregate(conditioned)
    return GPFitResult(
        surrogate=None,
        diagnostic=replace(
            aggregated.diagnostic,
            failure_kind=GPFitFailureKind.RECOVERY_EXHAUSTED,
            failure_stage="fixed_conditioning_validation",
            failure_type="GPFitRecoveryExhausted",
            failure_message=(
                "all optimized attempts and fixed-hyperparameter conditioning failed"
            ),
        ),
    )


def likelihood_support_markdown() -> str:
    descriptions = {
        LikelihoodPolicy.TASK_DIAGONAL_GAUSSIAN: (
            "One learned standardized Gaussian noise level per response"
        ),
        LikelihoodPolicy.SHARED_ISOTROPIC_GAUSSIAN: (
            "One learned standardized Gaussian noise level shared by all responses"
        ),
    }
    lines = [
        "# Prompt 4A likelihood support",
        "",
        "This table is generated from `mutaco.surrogate`; do not edit it independently.",
        "",
        "| Identifier | Classification | Surface | Behavior | Confidence covariance |",
        "|---|---|---|---|---|",
    ]
    for policy, classification in LIKELIHOOD_POLICIES.items():
        lines.append(
            f"| `{policy.value}` | {classification.value} | "
            f"`{surface_record(policy.value).surface.value}` | {descriptions[policy]} | "
            "latent only |"
        )
    return "\n".join(lines) + "\n"


def gp_fit_recovery_support_markdown() -> str:
    descriptions = {
        GPFitRecoveryPolicy.COLD_ONLY: (
            "One deterministic cold optimized fit; no warm state, restart, or "
            "fixed-conditioning recovery"
        ),
        GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1: (
            "Recent warm, penultimate warm, deterministic cold, four seeded "
            "restarts, then at most one fixed-conditioning update"
        ),
    }
    lines = [
        "# GP-fit recovery support",
        "",
        "| Identifier | Classification | Surface | Behavior | Simulator calls |",
        "|---|---|---|---|---:|",
    ]
    for policy, classification in GP_FIT_RECOVERY_POLICIES.items():
        lines.append(
            f"| `{policy.value}` | {classification.value} | "
            f"`{surface_record(policy.value).surface.value}` | "
            f"{descriptions[policy]} | 0 |"
        )
    return "\n".join(lines) + "\n"


def posterior_numerics_support_markdown() -> str:
    descriptions = {
        PosteriorNumericsPolicy.NATIVE_FLOAT32: (
            "Native fitted-model dtype for posterior mean and covariance"
        ),
        PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE: (
            "Native float32 fit and mean; cached float64 covariance construction, "
            "strict validation, then validated float32 return"
        ),
        PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC: (
            "Float64-fitted diagnostic model with native float64 posterior"
        ),
    }
    lines = [
        "# Posterior numerics support",
        "",
        "| Identifier | Classification | Surface | Behavior | Required fit dtype | Additional fit attempts | Simulator calls |",
        "|---|---|---|---|---|---:|---:|",
    ]
    required = {
        PosteriorNumericsPolicy.NATIVE_FLOAT32: "float32 or float64",
        PosteriorNumericsPolicy.FLOAT32_FIT_FLOAT64_COVARIANCE: "float32",
        PosteriorNumericsPolicy.FULL_FLOAT64_DIAGNOSTIC: "float64",
    }
    for policy, classification in POSTERIOR_NUMERICS_POLICIES.items():
        lines.append(
            f"| `{policy.value}` | {classification.value} | "
            f"`{surface_record(policy.value).surface.value}` | "
            f"{descriptions[policy]} | {required[policy]} | 0 | 0 |"
        )
    return "\n".join(lines) + "\n"
