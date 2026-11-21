"""Conventional scalar-loss qLogNEI backed by the pinned BoTorch implementation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Mapping
import warnings

import botorch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

from mutaco import ParameterSpec, SeedCoordinate, SeedDeriver, stable_hash


SCALAR_QLOGNEI_POLICY = "scalar_qlognei_negative_physical_mae_v1"
SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL = (
    "continuous_scalar_posterior_mean_physical_mae_v1"
)
SCALAR_QLOGNEI_SETTINGS_SCHEMA = "mutaco-scalar-qlognei-primary-settings-v1"
FLOAT32_CHOLESKY_INITIAL_JITTER = 1.0e-6
CHOLESKY_MAX_TRIES = 3
_JITTER_PATTERN = re.compile(r"added jitter of ([0-9.eE+-]+)")


class ScalarQLogNEIError(RuntimeError):
    """Typed failure of the conventional scalar BO baseline."""


def scalar_qlognei_settings_from_document(
    document: Mapping[str, Any],
) -> dict[str, int]:
    """Validate and flatten the frozen production-settings document."""

    body = dict(document)
    identity = body.pop("settings_identity", None)
    if identity != stable_hash("scalar-qlognei-primary-settings-v1", body):
        raise ValueError("scalar qLogNEI settings identity mismatch")
    expected = {
        "schema_version": SCALAR_QLOGNEI_SETTINGS_SCHEMA,
        "status": "frozen_zero_call_validated",
        "policy_id": SCALAR_QLOGNEI_POLICY,
        "model": "botorch.SingleTaskGP",
        "acquisition": "botorch.qLogNoisyExpectedImprovement",
        "objective": "negative_observed_physical_response_l1_mae",
        "batch_q": 1,
        "dtype": "float32",
        "precision_fallback": "forbidden",
        "automatic_retry": False,
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise ValueError(f"scalar qLogNEI settings {key} mismatch")
    if set(body) != {*expected, "cholesky", "search", "terminal"}:
        raise ValueError("scalar qLogNEI settings fields mismatch")
    cholesky = body.get("cholesky")
    if cholesky != {
        "implementation": "gpytorch.settings.cholesky_jitter",
        "initial_float32_jitter": FLOAT32_CHOLESKY_INITIAL_JITTER,
        "maximum_attempts": CHOLESKY_MAX_TRIES,
        "unrecovered_action": "typed_failure_before_next_simulator_call",
    }:
        raise ValueError("scalar qLogNEI Cholesky policy mismatch")
    search = body.get("search")
    terminal = body.get("terminal")
    if not isinstance(search, Mapping) or not isinstance(terminal, Mapping):
        raise ValueError("scalar qLogNEI settings sections are missing")
    if set(search) != {
        "n_mc", "raw_samples", "restarts", "max_iterations", "fit_max_iterations"
    } or set(terminal) != {
        "policy_id", "raw_samples", "restarts", "max_iterations",
        "simulator_calls", "gp_fits",
    }:
        raise ValueError("scalar qLogNEI numerical settings fields mismatch")
    if terminal.get("policy_id") != SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL:
        raise ValueError("scalar qLogNEI terminal policy mismatch")
    if terminal.get("simulator_calls") != 0 or terminal.get("gp_fits") != 0:
        raise ValueError("scalar qLogNEI terminal must remain zero-call and zero-fit")
    flattened = {
        "n_mc": search.get("n_mc"),
        "raw_samples": search.get("raw_samples"),
        "restarts": search.get("restarts"),
        "max_iterations": search.get("max_iterations"),
        "fit_max_iterations": search.get("fit_max_iterations"),
        "terminal_raw_samples": terminal.get("raw_samples"),
        "terminal_restarts": terminal.get("restarts"),
        "terminal_max_iterations": terminal.get("max_iterations"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in flattened.values()
    ):
        raise ValueError("scalar qLogNEI numerical settings must be positive integers")
    if flattened["restarts"] > flattened["raw_samples"]:
        raise ValueError("scalar qLogNEI search restarts exceed raw samples")
    if flattened["terminal_restarts"] > flattened["terminal_raw_samples"]:
        raise ValueError("scalar qLogNEI terminal restarts exceed raw samples")
    return flattened


@dataclass(frozen=True)
class ScalarQLogNEIConfig:
    n_mc: int
    raw_samples: int
    restarts: int
    max_iterations: int
    fit_max_iterations: int
    seed_deriver: SeedDeriver
    fit_coordinate: SeedCoordinate
    sampler_coordinate: SeedCoordinate
    optimizer_coordinate: SeedCoordinate

    def __post_init__(self) -> None:
        for name in (
            "n_mc",
            "raw_samples",
            "restarts",
            "max_iterations",
            "fit_max_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.restarts > self.raw_samples:
            raise ValueError("qLogNEI restarts cannot exceed raw samples")

    def document(self) -> dict[str, Any]:
        return {
            "policy_id": SCALAR_QLOGNEI_POLICY,
            "batch_q": 1,
            "n_mc": self.n_mc,
            "raw_samples": self.raw_samples,
            "restarts": self.restarts,
            "max_iterations": self.max_iterations,
            "fit_max_iterations": self.fit_max_iterations,
            "fit_seed": self.seed_deriver.seed_for(self.fit_coordinate),
            "sampler_seed": self.seed_deriver.seed_for(self.sampler_coordinate),
            "optimizer_seed": self.seed_deriver.seed_for(self.optimizer_coordinate),
            "dtype_policy": "float32_end_to_end_v1",
            "precision_fallback": "forbidden",
            "cholesky_policy": {
                "implementation": "gpytorch.settings.cholesky_jitter",
                "initial_float32_jitter": FLOAT32_CHOLESKY_INITIAL_JITTER,
                "maximum_attempts": CHOLESKY_MAX_TRIES,
                "unrecovered_action": "typed_failure_before_next_simulator_call",
            },
            "objective": "negative_observed_physical_response_l1_mae",
            "model": "botorch.SingleTaskGP",
            "acquisition": "botorch.qLogNoisyExpectedImprovement",
            "botorch_version": str(botorch.__version__),
            "gpytorch_version": str(gpytorch.__version__),
        }


@dataclass(frozen=True)
class ScalarLossGP:
    model: SingleTaskGP
    model_id: str
    fit_diagnostic: Mapping[str, Any]

    def posterior_mean_loss(self, theta: torch.Tensor) -> torch.Tensor:
        posterior = self.model.posterior(theta.contiguous())
        return (-posterior.mean.squeeze(-1)).contiguous()


@dataclass(frozen=True)
class ScalarQLogNEIProposal:
    candidate: torch.Tensor
    log_acquisition_value: torch.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ScalarLossRecommendation:
    selected_theta: torch.Tensor
    posterior_mean_loss: torch.Tensor
    provenance: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "policy_id": SCALAR_POSTERIOR_MEAN_LOSS_TERMINAL,
            "candidate_domain": "continuous_bounded_theta",
            "score_formula": "posterior_mean_observed_physical_response_l1_mae",
            "selected_theta": self.selected_theta.detach().cpu().tolist(),
            "score": float(self.posterior_mean_loss.detach().cpu()),
            "simulator_calls": 0,
            "gp_fits": 0,
            **dict(self.provenance),
        }


def observed_physical_mae(
    responses_physical: torch.Tensor, target_physical: torch.Tensor
) -> torch.Tensor:
    """Return one scalar physical-coordinate MAE per simulator response."""

    if responses_physical.ndim != 2:
        raise ValueError("scalar BO responses must have shape [N,M]")
    if target_physical.shape != (responses_physical.shape[1],):
        raise ValueError("scalar BO target dimension mismatch")
    if (
        responses_physical.dtype != target_physical.dtype
        or responses_physical.device != target_physical.device
    ):
        raise ValueError("scalar BO responses and target must share dtype/device")
    if not bool(torch.isfinite(responses_physical).all()) or not bool(
        torch.isfinite(target_physical).all()
    ):
        raise ValueError("scalar BO physical inputs must be finite")
    return (responses_physical - target_physical).abs().mean(dim=-1).contiguous()


def _bounds(parameter_spec: ParameterSpec, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [parameter_spec.lower, parameter_spec.upper],
        dtype=reference.dtype,
        device=reference.device,
    ).contiguous()


def _fork_devices(reference: torch.Tensor) -> list[int]:
    if reference.device.type != "cuda":
        return []
    return [reference.device.index or 0]


def _warning_documents(
    emitted: list[warnings.WarningMessage],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for item in emitted:
        message = str(item.message)
        match = _JITTER_PATTERN.search(message)
        document: dict[str, Any] = {
            "category": item.category.__name__,
            "message": message,
        }
        if match is not None:
            document["cholesky_jitter_added"] = float(match.group(1))
        documents.append(document)
    return documents


def _numerical_contexts() -> tuple[Any, Any]:
    """Freeze the official bounded float32 Cholesky retry policy."""

    return (
        gpytorch.settings.cholesky_jitter(
            float_value=FLOAT32_CHOLESKY_INITIAL_JITTER
        ),
        gpytorch.settings.cholesky_max_tries(CHOLESKY_MAX_TRIES),
    )


def fit_scalar_loss_gp(
    *,
    theta: torch.Tensor,
    responses_physical: torch.Tensor,
    target_physical: torch.Tensor,
    parameter_spec: ParameterSpec,
    config: ScalarQLogNEIConfig,
) -> ScalarLossGP:
    """Fit the pinned official single-output GP without changing float32 dtype."""

    started = perf_counter()
    if theta.ndim != 2 or theta.shape[0] != responses_physical.shape[0]:
        raise ValueError("scalar BO training rows are inconsistent")
    if theta.shape[1] != parameter_spec.dimension:
        raise ValueError("scalar BO parameter dimension mismatch")
    if theta.dtype is not torch.float32 or responses_physical.dtype is not torch.float32:
        raise ValueError("scalar qLogNEI primary policy requires float32 tensors")
    if theta.device != responses_physical.device:
        raise ValueError("scalar BO training tensors must share device")
    losses = observed_physical_mae(responses_physical, target_physical)
    train_y = (-losses).unsqueeze(-1).contiguous()
    bounds = _bounds(parameter_spec, theta)
    fit_seed = config.seed_deriver.seed_for(config.fit_coordinate)
    jitter_context, tries_context = _numerical_contexts()
    try:
        with warnings.catch_warnings(record=True) as emitted, torch.random.fork_rng(
            devices=_fork_devices(theta)
        ), jitter_context, tries_context:
            warnings.simplefilter("always")
            torch.manual_seed(fit_seed)
            model = SingleTaskGP(
                theta,
                train_y,
                input_transform=Normalize(d=parameter_spec.dimension, bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": config.fit_max_iterations}},
            )
    except (RuntimeError, FloatingPointError) as exc:
        warning_detail = _warning_documents(emitted)
        raise ScalarQLogNEIError(
            "scalar GP fit failed without precision fallback; "
            f"warnings={warning_detail}: {type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    body = {
        "policy_id": SCALAR_QLOGNEI_POLICY,
        "training_theta": theta.detach().cpu().tolist(),
        "training_negative_mae": train_y.detach().cpu().tolist(),
        "parameter_bounds": bounds.detach().cpu().tolist(),
        "configuration": config.document(),
        "fit_seed": fit_seed,
        "dtype": str(theta.dtype),
        "device": str(theta.device),
    }
    model_id = stable_hash("scalar-loss-single-task-gp-v1", body)
    diagnostic = {
        "status": "success",
        "model_id": model_id,
        "training_observation_count": int(theta.shape[0]),
        "training_covariance_dimension": int(theta.shape[0]),
        "dtype": str(theta.dtype),
        "device": str(theta.device),
        "fit_seed": fit_seed,
        "elapsed_seconds": perf_counter() - started,
        "warnings": _warning_documents(emitted),
        "precision_fallback_used": False,
        "botorch_version": str(botorch.__version__),
        "gpytorch_version": str(gpytorch.__version__),
    }
    return ScalarLossGP(model=model, model_id=model_id, fit_diagnostic=diagnostic)


def _exact_duplicate(candidate: torch.Tensor, baseline: torch.Tensor) -> bool:
    return any(torch.equal(candidate, row) for row in baseline)


def _nonfinite_kind(value: torch.Tensor) -> str | None:
    scalar = float(value.detach().cpu())
    if scalar != scalar:
        return "nan"
    if scalar == float("inf"):
        return "+inf"
    if scalar == float("-inf"):
        return "-inf"
    return None


def _invalid_optimizer_result(
    candidate: torch.Tensor,
    value: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[dict[str, Any], bool]:
    candidate_finite = bool(torch.isfinite(candidate).all())
    value_finite = bool(torch.isfinite(value))
    duplicate_indices = [
        index for index, row in enumerate(baseline) if torch.equal(candidate, row)
    ]
    reasons = []
    if not candidate_finite:
        reasons.append("nonfinite_candidate")
    if not value_finite:
        reasons.append("nonfinite_acquisition_value")
    if duplicate_indices:
        reasons.append("exact_baseline_duplicate")
    nonfinite_candidate = []
    for index, item in enumerate(candidate.reshape(-1)):
        kind = _nonfinite_kind(item)
        if kind is not None:
            nonfinite_candidate.append({"flat_index": index, "kind": kind})
    document = {
        "stage": "optimized_candidate_validation",
        "optimizer_returned": True,
        "exception_type": "ScalarQLogNEIError",
        "exception_message": "optimized qLogNEI candidate is invalid",
        "trigger_reasons": reasons,
        "rejected_candidate": (
            candidate.detach().cpu().tolist() if candidate_finite else None
        ),
        "rejected_candidate_finite": candidate_finite,
        "rejected_candidate_nonfinite": nonfinite_candidate,
        "rejected_log_acquisition_value": (
            float(value.detach().cpu()) if value_finite else None
        ),
        "rejected_log_acquisition_value_finite": value_finite,
        "rejected_log_acquisition_value_nonfinite_kind": (
            None if value_finite else _nonfinite_kind(value)
        ),
        "exact_duplicate_indices_zero_based": duplicate_indices,
    }
    return document, bool(reasons)


def _fallback_raw_candidate(
    *,
    acquisition: qLogNoisyExpectedImprovement,
    baseline: torch.Tensor,
    bounds: torch.Tensor,
    raw_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = torch.quasirandom.SobolEngine(
        baseline.shape[1], scramble=True, seed=seed
    )
    unit = engine.draw(raw_samples, dtype=baseline.dtype).to(baseline.device)
    raw = (bounds[0] + (bounds[1] - bounds[0]) * unit).contiguous()
    values = acquisition(raw.unsqueeze(-2)).detach()
    valid = torch.isfinite(values) & torch.tensor(
        [not _exact_duplicate(row, baseline) for row in raw],
        dtype=torch.bool,
        device=raw.device,
    )
    if not bool(valid.any()):
        raise ScalarQLogNEIError("no finite unevaluated qLogNEI raw candidate")
    ranked = torch.argsort(
        torch.where(valid, values, torch.full_like(values, -torch.inf)),
        descending=True,
        stable=True,
    )
    index = int(ranked[0].item())
    return raw[index], values[index]


def propose_scalar_qlognei(
    *,
    fitted: ScalarLossGP,
    baseline: torch.Tensor,
    parameter_spec: ParameterSpec,
    config: ScalarQLogNEIConfig,
) -> ScalarQLogNEIProposal:
    """Use official qLogNEI with q=1 and deterministic named seeds."""

    started = perf_counter()
    if baseline.dtype is not torch.float32:
        raise ValueError("scalar qLogNEI primary policy requires float32 baseline")
    sampler_seed = config.seed_deriver.seed_for(config.sampler_coordinate)
    optimizer_seed = config.seed_deriver.seed_for(config.optimizer_coordinate)
    bounds = _bounds(parameter_spec, baseline)
    source = "botorch_optimize_acqf"
    fallback_trigger: dict[str, Any] | None = None
    emitted: list[warnings.WarningMessage] = []
    try:
        jitter_context, tries_context = _numerical_contexts()
        with warnings.catch_warnings(record=True) as caught, torch.random.fork_rng(
            devices=_fork_devices(baseline)
        ), jitter_context, tries_context:
            warnings.simplefilter("always")
            torch.manual_seed(optimizer_seed)
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([config.n_mc]), seed=sampler_seed
            )
            acquisition = qLogNoisyExpectedImprovement(
                model=fitted.model,
                X_baseline=baseline,
                sampler=sampler,
                prune_baseline=True,
                cache_root=True,
            )
            candidate, value = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=config.restarts,
                raw_samples=config.raw_samples,
                options={"maxiter": config.max_iterations, "seed": optimizer_seed},
            )
            emitted.extend(caught)
        candidate = candidate[0].detach().contiguous()
        value = value.detach().reshape(()).contiguous()
        fallback_trigger, invalid = _invalid_optimizer_result(
            candidate, value, baseline
        )
        if invalid:
            raise ScalarQLogNEIError("optimized qLogNEI candidate is invalid")
        fallback_trigger = None
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        if "acquisition" not in locals():
            raise ScalarQLogNEIError(
                f"qLogNEI construction failed in float32: {type(exc).__name__}: {exc}"
            ) from exc
        if fallback_trigger is None:
            fallback_trigger = {
                "stage": "continuous_optimizer",
                "optimizer_returned": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "trigger_reasons": ["optimizer_exception"],
                "rejected_candidate": None,
                "rejected_candidate_finite": None,
                "rejected_candidate_nonfinite": [],
                "rejected_log_acquisition_value": None,
                "rejected_log_acquisition_value_finite": None,
                "rejected_log_acquisition_value_nonfinite_kind": None,
                "exact_duplicate_indices_zero_based": [],
            }
        try:
            candidate, value = _fallback_raw_candidate(
                acquisition=acquisition,
                baseline=baseline,
                bounds=bounds,
                raw_samples=config.raw_samples,
                seed=optimizer_seed,
            )
        except (RuntimeError, ValueError, FloatingPointError) as fallback_exc:
            raise ScalarQLogNEIError(
                "qLogNEI failed without precision fallback; "
                f"optimizer_trigger={fallback_trigger}; "
                f"fallback={type(fallback_exc).__name__}: {fallback_exc}; "
                f"warnings={_warning_documents(emitted)}"
            ) from fallback_exc
        source = f"deterministic_raw_fallback_after_{type(exc).__name__}"
    provenance = {
        "policy_id": SCALAR_QLOGNEI_POLICY,
        "batch_q": 1,
        "candidate": candidate.detach().cpu().tolist(),
        "log_acquisition_value": float(value.detach().cpu()),
        "candidate_source": source,
        "fallback_trigger": fallback_trigger,
        "baseline_size": int(baseline.shape[0]),
        "baseline_covariance_dimension": int(baseline.shape[0]),
        "sampler_seed": sampler_seed,
        "optimizer_seed": optimizer_seed,
        "configuration": config.document(),
        "dtype": str(candidate.dtype),
        "device": str(candidate.device),
        "warnings": _warning_documents(emitted),
        "precision_fallback_used": False,
        "elapsed_seconds": perf_counter() - started,
        "simulator_calls": 0,
    }
    return ScalarQLogNEIProposal(candidate, value, provenance)


def recommend_scalar_posterior_mean_loss(
    *,
    fitted: ScalarLossGP,
    parameter_spec: ParameterSpec,
    reference: torch.Tensor,
    seed_deriver: SeedDeriver,
    coordinate: SeedCoordinate,
    raw_samples: int,
    restarts: int,
    max_iterations: int,
) -> ScalarLossRecommendation:
    """Continuously minimize the scalar GP posterior mean loss with zero calls."""

    started = perf_counter()
    seed = seed_deriver.seed_for(coordinate)
    bounds = _bounds(parameter_spec, reference)
    jitter_context, tries_context = _numerical_contexts()
    try:
        with warnings.catch_warnings(record=True) as emitted, torch.random.fork_rng(
            devices=_fork_devices(reference)
        ), jitter_context, tries_context:
            warnings.simplefilter("always")
            torch.manual_seed(seed)
            acquisition = PosteriorMean(fitted.model, maximize=True)
            candidate, value = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=restarts,
                raw_samples=raw_samples,
                options={"maxiter": max_iterations, "seed": seed},
            )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ScalarQLogNEIError(
            "scalar terminal optimization failed without precision fallback; "
            f"warnings={_warning_documents(emitted)}: {type(exc).__name__}: {exc}"
        ) from exc
    selected = candidate[0].detach().contiguous()
    mean_loss = (-value.detach().reshape(())).contiguous()
    if not bool(torch.isfinite(selected).all()) or not bool(torch.isfinite(mean_loss)):
        raise ScalarQLogNEIError("scalar posterior-mean terminal optimization failed")
    return ScalarLossRecommendation(
        selected_theta=selected,
        posterior_mean_loss=mean_loss,
        provenance={
            "model_id": fitted.model_id,
            "batch_q": 1,
            "raw_samples": raw_samples,
            "restarts": restarts,
            "max_iterations": max_iterations,
            "seed_namespace": coordinate.namespace,
            "seed_coordinates": list(coordinate.coordinates),
            "derived_seed": seed,
            "dtype": str(selected.dtype),
            "device": str(selected.device),
            "warnings": _warning_documents(emitted),
            "precision_fallback_used": False,
            "elapsed_seconds": perf_counter() - started,
        },
    )
