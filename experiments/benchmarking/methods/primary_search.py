"""Primary-benchmark NEI and surrogate-free Sobol search implementations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import (
    EvaluatedDesigns,
    FrozenResponseTransform,
    GPFitConfig,
    GPFitStatus,
    NEIConfig,
    ParameterSpec,
    ResponseSpec,
    SeedCoordinate,
    SeedDeriver,
    SurrogateTrainingData,
    fit_exact_multitask_gp_with_recovery,
    propose_composite_mc_nei,
    recommend_best_observed_physical_mae,
    recommend_continuous_physical_mae,
)
from experiments.benchmarking.methods.scalar_qlognei import (
    ScalarQLogNEIConfig,
    fit_scalar_loss_gp,
    propose_scalar_qlognei,
    recommend_scalar_posterior_mean_loss,
)


EvaluateSearch = Callable[[Sequence[float], int], Sequence[float]]
SnapshotCallback = Callable[
    [int, Any, torch.Tensor, torch.Tensor, FrozenResponseTransform], Mapping[str, Any]
]


@dataclass(frozen=True)
class PrimarySearchResult:
    method: str
    status: str
    theta: torch.Tensor
    responses_physical: torch.Tensor
    transform: FrozenResponseTransform | None
    fit_diagnostics: tuple[Mapping[str, Any], ...]
    proposal_diagnostics: tuple[Mapping[str, Any], ...]
    recommendation: Mapping[str, Any] | None
    timing: Mapping[str, float]
    elapsed_seconds: float
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_explanation: str | None = None
    qualification_snapshots: tuple[Mapping[str, Any], ...] = ()

    @property
    def successful_calls(self) -> int:
        return int(self.theta.shape[0])


PRIMARY_TIMING_KEYS = (
    "simulator",
    "gp_fit",
    "acquisition",
    "terminal",
    "other",
    "total",
)


def primary_decision_timing(
    *,
    total: float,
    simulator: float,
    gp_fit: float,
    acquisition: float,
    terminal: float,
) -> dict[str, float]:
    """Return one non-overlapping, method-independent timing record."""

    values = {
        "simulator": float(simulator),
        "gp_fit": float(gp_fit),
        "acquisition": float(acquisition),
        "terminal": float(terminal),
        "total": float(total),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("primary timing inputs must be finite and nonnegative")
    accounted = sum(values[key] for key in PRIMARY_TIMING_KEYS[:4])
    tolerance = 1e-9 * max(1.0, values["total"], accounted)
    if accounted > values["total"] + tolerance:
        raise ValueError("primary timing components exceed total method time")
    values["other"] = max(0.0, values["total"] - accounted)
    return {key: values[key] for key in PRIMARY_TIMING_KEYS}


def _physical_tensor(
    values: Sequence[float],
    *,
    response_spec: ResponseSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    result = torch.tensor(values, dtype=dtype, device=device).contiguous()
    if result.shape != (response_spec.dimension,) or not bool(torch.isfinite(result).all()):
        raise ValueError("primary search simulator returned an invalid response")
    return result


def run_primary_sobol_search(
    *,
    design: torch.Tensor,
    target_physical: torch.Tensor,
    response_spec: ResponseSpec,
    evaluate: EvaluateSearch,
) -> PrimarySearchResult:
    """Evaluate a predetermined Sobol sequence and choose its observed best."""

    started = perf_counter()
    simulator_seconds = 0.0
    responses: list[torch.Tensor] = []
    for index, theta in enumerate(design, start=1):
        simulation_started = perf_counter()
        try:
            values = evaluate(theta.detach().cpu().tolist(), index)
        finally:
            simulator_seconds += perf_counter() - simulation_started
        responses.append(
            _physical_tensor(
                values,
                response_spec=response_spec,
                dtype=design.dtype,
                device=design.device,
            )
        )
    matrix = torch.stack(responses).contiguous()
    terminal_started = perf_counter()
    try:
        recommendation = recommend_best_observed_physical_mae(
            EvaluatedDesigns(
                design,
                tuple(
                    f"sobol-evaluation-{index}"
                    for index in range(1, design.shape[0] + 1)
                ),
            ),
            matrix,
            target_physical,
        )
    finally:
        terminal_seconds = perf_counter() - terminal_started
    total_seconds = perf_counter() - started
    timing = primary_decision_timing(
        total=total_seconds,
        simulator=simulator_seconds,
        gp_fit=0.0,
        acquisition=0.0,
        terminal=terminal_seconds,
    )
    recommendation_document = {
        **recommendation.document(),
        "elapsed_seconds": terminal_seconds,
    }
    return PrimarySearchResult(
        method="sobol",
        status="recommended",
        theta=design.detach().clone().contiguous(),
        responses_physical=matrix,
        transform=None,
        fit_diagnostics=(),
        proposal_diagnostics=(),
        recommendation=recommendation_document,
        timing=timing,
        elapsed_seconds=total_seconds,
    )


def run_primary_nei_search(
    *,
    initial_design: torch.Tensor,
    budget: int,
    target_physical: torch.Tensor,
    target_id: str,
    target_index: int,
    parameter_spec: ParameterSpec,
    response_spec: ResponseSpec,
    evaluate: EvaluateSearch,
    inference_seed_deriver: SeedDeriver,
    terminal_seed_deriver: SeedDeriver,
    coordinate_prefix: tuple[str | int, ...],
    gp_settings: Mapping[str, Any],
    nei_settings: Mapping[str, Any],
    snapshot_positions: tuple[int, ...] = (),
    snapshot_callback: SnapshotCallback | None = None,
    scale_floor: float = 1e-12,
) -> PrimarySearchResult:
    """Run sequential q=1 NEI with the canonical exact multitask GP class."""

    started = perf_counter()
    simulator_seconds = 0.0
    gp_fit_seconds = 0.0
    acquisition_seconds = 0.0
    terminal_seconds = 0.0
    if initial_design.shape != (16, parameter_spec.dimension):
        raise ValueError("primary NEI requires the exact shared 16-point design")
    if budget not in {40, 100}:
        raise ValueError("primary NEI budget must be 40 or 100")
    if target_physical.shape != (response_spec.dimension,):
        raise ValueError("primary NEI target has the wrong response dimension")
    theta_rows: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    fits: list[Mapping[str, Any]] = []
    proposals: list[Mapping[str, Any]] = []
    warm_states = []
    transform: FrozenResponseTransform | None = None
    snapshots: list[Mapping[str, Any]] = []
    if tuple(sorted(set(snapshot_positions))) != snapshot_positions or any(
        position < 16 or position > budget for position in snapshot_positions
    ):
        raise ValueError("primary NEI snapshot positions are invalid")
    if bool(snapshot_positions) != (snapshot_callback is not None):
        raise ValueError("primary NEI snapshots require positions and a callback")

    def failure(stage: str, exc: BaseException) -> PrimarySearchResult:
        total_seconds = perf_counter() - started
        timing = primary_decision_timing(
            total=total_seconds,
            simulator=simulator_seconds,
            gp_fit=gp_fit_seconds,
            acquisition=acquisition_seconds,
            terminal=terminal_seconds,
        )
        theta = (
            torch.stack(theta_rows).contiguous()
            if theta_rows
            else torch.empty(
                (0, parameter_spec.dimension),
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        )
        response = (
            torch.stack(responses).contiguous()
            if responses
            else torch.empty(
                (0, response_spec.dimension),
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        )
        return PrimarySearchResult(
            method="nei_bo",
            status="failed",
            theta=theta,
            responses_physical=response,
            transform=transform,
            fit_diagnostics=tuple(fits),
            proposal_diagnostics=tuple(proposals),
            recommendation=None,
            timing=timing,
            elapsed_seconds=total_seconds,
            failure_stage=stage,
            failure_type=type(exc).__name__,
            failure_explanation=str(exc),
            qualification_snapshots=tuple(snapshots),
        )

    try:
        for index, theta in enumerate(initial_design, start=1):
            simulation_started = perf_counter()
            try:
                response = _physical_tensor(
                    evaluate(theta.detach().cpu().tolist(), index),
                    response_spec=response_spec,
                    dtype=initial_design.dtype,
                    device=initial_design.device,
                )
            finally:
                simulator_seconds += perf_counter() - simulation_started
            theta_rows.append(theta.detach().clone().contiguous())
            responses.append(response)
        initial_ids = tuple(f"nei-evaluation-{index}" for index in range(1, 17))
        transform = FrozenResponseTransform.fit(
            torch.stack(responses).contiguous(),
            response_spec=response_spec,
            observation_ids=initial_ids,
            scale_floor=scale_floor,
        )
    except Exception as exc:
        return failure("initial_design_or_transform", exc)

    response_names = transform.metadata.response_names

    def fit_version(version: int):
        nonlocal gp_fit_seconds
        fit_started = perf_counter()
        try:
            return fit_version_inner(version)
        finally:
            gp_fit_seconds += perf_counter() - fit_started

    def fit_version_inner(version: int):
        standardized = transform.to_standardized(
            torch.stack(responses).contiguous(), response_names=response_names
        )
        data = SurrogateTrainingData(
            theta=torch.stack(theta_rows).contiguous(),
            response_standardized=standardized,
            parameter_spec=parameter_spec,
            observation_ids=tuple(
                f"nei-evaluation-{index}" for index in range(1, len(theta_rows) + 1)
            ),
            transform_id=transform.metadata.transform_id,
        )
        config = GPFitConfig.from_seed_deriver(
            inference_seed_deriver,
            SeedCoordinate(
                "gp_fit", (*coordinate_prefix, version, 0, "gp_fit")
            ),
            **dict(gp_settings),
        )
        result = fit_exact_multitask_gp_with_recovery(
            data,
            config=config,
            warm_start_states=tuple(reversed(warm_states)),
            training_version=version,
        )
        fits.append(
            {
                "training_version": version,
                "status": result.diagnostic.status.value,
                "fit_id": result.diagnostic.fit_id,
                "model_id": result.diagnostic.model_id,
                "training_data_id": result.diagnostic.training_data_id,
                "elapsed_seconds": result.diagnostic.elapsed_seconds,
                "failure_kind": (
                    None
                    if result.diagnostic.failure_kind is None
                    else result.diagnostic.failure_kind.value
                ),
                "failure_message": result.diagnostic.failure_message,
                "recovery_attempts": result.diagnostic.recovery_attempts,
            }
        )
        if result.diagnostic.status is not GPFitStatus.SUCCESS or result.surrogate is None:
            raise RuntimeError(
                result.diagnostic.failure_message or "NEI GP fit failed"
            )
        if config.max_warm_start_states > 0:
            warm_states.append(result.surrogate.warm_start_state())
            del warm_states[:-config.max_warm_start_states]
        return result.surrogate

    try:
        surrogate = fit_version(0)
        if 16 in snapshot_positions:
            snapshots.append(dict(snapshot_callback(
                16,
                surrogate,
                torch.stack(theta_rows).contiguous(),
                torch.stack(responses).contiguous(),
                transform,
            )))
    except Exception as exc:
        return failure("initial_gp_fit", exc)

    for round_index in range(budget - 16):
        proposal_started = perf_counter()
        try:
            config = NEIConfig(
                n_mc=int(nei_settings["n_mc"]),
                raw_samples=int(nei_settings["raw_samples"]),
                restarts=int(nei_settings["restarts"]),
                max_iterations=int(nei_settings["max_iterations"]),
                learning_rate=float(nei_settings["learning_rate"]),
                seed_deriver=inference_seed_deriver,
                raw_coordinate=SeedCoordinate(
                    "nei_raw", (*coordinate_prefix, round_index, "nei_raw")
                ),
                mc_coordinate=SeedCoordinate(
                    "nei_mc", (*coordinate_prefix, round_index, "optimization", "nei_mc")
                ),
                optimizer_coordinate=SeedCoordinate(
                    "nei_optimizer", (*coordinate_prefix, round_index, "nei_optimizer")
                ),
            )
            proposal = propose_composite_mc_nei(
                surrogate,
                baseline=torch.stack(theta_rows).contiguous(),
                transform=transform,
                target_physical=target_physical,
                parameter_spec=parameter_spec,
                config=config,
            )
            proposals.append(proposal.provenance())
        except Exception as exc:
            acquisition_seconds += perf_counter() - proposal_started
            return failure("acquisition", exc)
        acquisition_seconds += perf_counter() - proposal_started
        evaluation_index = len(theta_rows) + 1
        simulation_started = perf_counter()
        try:
            response = _physical_tensor(
                evaluate(proposal.candidate.detach().cpu().tolist(), evaluation_index),
                response_spec=response_spec,
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        except Exception as exc:
            simulator_seconds += perf_counter() - simulation_started
            return failure("simulation", exc)
        simulator_seconds += perf_counter() - simulation_started
        theta_rows.append(proposal.candidate.detach().clone().contiguous())
        responses.append(response)
        try:
            surrogate = fit_version(round_index + 1)
            if len(theta_rows) in snapshot_positions:
                assert snapshot_callback is not None
                snapshots.append(dict(snapshot_callback(
                    len(theta_rows),
                    surrogate,
                    torch.stack(theta_rows).contiguous(),
                    torch.stack(responses).contiguous(),
                    transform,
                )))
        except Exception as exc:
            return failure("sequential_gp_fit", exc)

    terminal_started = perf_counter()
    try:
        recommendation = recommend_continuous_physical_mae(
            surrogate=surrogate,
            transform=transform,
            target_physical=target_physical,
            target_id=target_id,
            parameter_spec=parameter_spec,
            seed_deriver=terminal_seed_deriver,
            target_index=target_index,
        )
    except Exception as exc:
        terminal_seconds += perf_counter() - terminal_started
        return failure("terminal_recommendation", exc)
    terminal_seconds += perf_counter() - terminal_started
    total_seconds = perf_counter() - started
    timing = primary_decision_timing(
        total=total_seconds,
        simulator=simulator_seconds,
        gp_fit=gp_fit_seconds,
        acquisition=acquisition_seconds,
        terminal=terminal_seconds,
    )
    return PrimarySearchResult(
        method="nei_bo",
        status="recommended",
        theta=torch.stack(theta_rows).contiguous(),
        responses_physical=torch.stack(responses).contiguous(),
        transform=transform,
        fit_diagnostics=tuple(fits),
        proposal_diagnostics=tuple(proposals),
        recommendation=recommendation.document(),
        timing=timing,
        elapsed_seconds=total_seconds,
        qualification_snapshots=tuple(snapshots),
    )


def run_primary_scalar_qlognei_search(
    *,
    initial_design: torch.Tensor,
    budget: int,
    target_physical: torch.Tensor,
    target_id: str,
    target_index: int,
    parameter_spec: ParameterSpec,
    response_spec: ResponseSpec,
    evaluate: EvaluateSearch,
    inference_seed_deriver: SeedDeriver,
    terminal_seed_deriver: SeedDeriver,
    coordinate_prefix: tuple[str | int, ...],
    qlognei_settings: Mapping[str, Any],
) -> PrimarySearchResult:
    """Run conventional sequential q=1 scalar-loss qLogNEI in float32."""

    started = perf_counter()
    simulator_seconds = 0.0
    gp_fit_seconds = 0.0
    acquisition_seconds = 0.0
    terminal_seconds = 0.0
    if initial_design.shape != (16, parameter_spec.dimension):
        raise ValueError("primary scalar qLogNEI requires the shared 16-point design")
    if initial_design.dtype is not torch.float32:
        raise ValueError("primary scalar qLogNEI requires float32 design tensors")
    if budget not in {40, 100}:
        raise ValueError("primary scalar qLogNEI budget must be 40 or 100")
    if target_physical.shape != (response_spec.dimension,):
        raise ValueError("primary scalar qLogNEI target has the wrong response dimension")
    theta_rows: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    fits: list[Mapping[str, Any]] = []
    proposals: list[Mapping[str, Any]] = []

    def failure(stage: str, exc: BaseException) -> PrimarySearchResult:
        total_seconds = perf_counter() - started
        timing = primary_decision_timing(
            total=total_seconds,
            simulator=simulator_seconds,
            gp_fit=gp_fit_seconds,
            acquisition=acquisition_seconds,
            terminal=terminal_seconds,
        )
        theta = (
            torch.stack(theta_rows).contiguous()
            if theta_rows
            else torch.empty(
                (0, parameter_spec.dimension),
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        )
        response = (
            torch.stack(responses).contiguous()
            if responses
            else torch.empty(
                (0, response_spec.dimension),
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        )
        return PrimarySearchResult(
            method="scalar_qlognei",
            status="failed",
            theta=theta,
            responses_physical=response,
            transform=None,
            fit_diagnostics=tuple(fits),
            proposal_diagnostics=tuple(proposals),
            recommendation=None,
            timing=timing,
            elapsed_seconds=total_seconds,
            failure_stage=stage,
            failure_type=type(exc).__name__,
            failure_explanation=str(exc),
        )

    def config(version: int, round_index: int) -> ScalarQLogNEIConfig:
        return ScalarQLogNEIConfig(
            n_mc=int(qlognei_settings["n_mc"]),
            raw_samples=int(qlognei_settings["raw_samples"]),
            restarts=int(qlognei_settings["restarts"]),
            max_iterations=int(qlognei_settings["max_iterations"]),
            fit_max_iterations=int(qlognei_settings["fit_max_iterations"]),
            seed_deriver=inference_seed_deriver,
            fit_coordinate=SeedCoordinate(
                "scalar_qlognei_gp_fit", (*coordinate_prefix, version, "gp_fit")
            ),
            sampler_coordinate=SeedCoordinate(
                "scalar_qlognei_mc", (*coordinate_prefix, round_index, "qlognei_mc")
            ),
            optimizer_coordinate=SeedCoordinate(
                "scalar_qlognei_optimizer",
                (*coordinate_prefix, round_index, "qlognei_optimizer"),
            ),
        )

    def fit_version(version: int, round_index: int):
        nonlocal gp_fit_seconds
        fit_started = perf_counter()
        try:
            fitted = fit_scalar_loss_gp(
                theta=torch.stack(theta_rows).contiguous(),
                responses_physical=torch.stack(responses).contiguous(),
                target_physical=target_physical,
                parameter_spec=parameter_spec,
                config=config(version, round_index),
            )
            fits.append(dict(fitted.fit_diagnostic))
            return fitted
        finally:
            gp_fit_seconds += perf_counter() - fit_started

    try:
        for index, theta in enumerate(initial_design, start=1):
            simulation_started = perf_counter()
            try:
                response = _physical_tensor(
                    evaluate(theta.detach().cpu().tolist(), index),
                    response_spec=response_spec,
                    dtype=initial_design.dtype,
                    device=initial_design.device,
                )
            finally:
                simulator_seconds += perf_counter() - simulation_started
            theta_rows.append(theta.detach().clone().contiguous())
            responses.append(response)
        fitted = fit_version(0, 0)
    except Exception as exc:
        return failure("initial_design_or_gp_fit", exc)

    for round_index in range(budget - 16):
        proposal_started = perf_counter()
        try:
            proposal = propose_scalar_qlognei(
                fitted=fitted,
                baseline=torch.stack(theta_rows).contiguous(),
                parameter_spec=parameter_spec,
                config=config(round_index, round_index),
            )
            proposals.append(dict(proposal.provenance))
        except Exception as exc:
            acquisition_seconds += perf_counter() - proposal_started
            return failure("acquisition", exc)
        acquisition_seconds += perf_counter() - proposal_started
        evaluation_index = len(theta_rows) + 1
        simulation_started = perf_counter()
        try:
            response = _physical_tensor(
                evaluate(proposal.candidate.detach().cpu().tolist(), evaluation_index),
                response_spec=response_spec,
                dtype=initial_design.dtype,
                device=initial_design.device,
            )
        except Exception as exc:
            simulator_seconds += perf_counter() - simulation_started
            return failure("simulation", exc)
        simulator_seconds += perf_counter() - simulation_started
        theta_rows.append(proposal.candidate.detach().clone().contiguous())
        responses.append(response)
        try:
            fitted = fit_version(round_index + 1, round_index + 1)
        except Exception as exc:
            return failure("sequential_gp_fit", exc)

    terminal_started = perf_counter()
    try:
        recommendation = recommend_scalar_posterior_mean_loss(
            fitted=fitted,
            parameter_spec=parameter_spec,
            reference=initial_design,
            seed_deriver=terminal_seed_deriver,
            coordinate=SeedCoordinate(
                "scalar_qlognei_terminal", (target_index, target_id, budget)
            ),
            raw_samples=int(qlognei_settings["terminal_raw_samples"]),
            restarts=int(qlognei_settings["terminal_restarts"]),
            max_iterations=int(qlognei_settings["terminal_max_iterations"]),
        )
    except Exception as exc:
        terminal_seconds += perf_counter() - terminal_started
        return failure("terminal_recommendation", exc)
    terminal_seconds += perf_counter() - terminal_started
    total_seconds = perf_counter() - started
    timing = primary_decision_timing(
        total=total_seconds,
        simulator=simulator_seconds,
        gp_fit=gp_fit_seconds,
        acquisition=acquisition_seconds,
        terminal=terminal_seconds,
    )
    return PrimarySearchResult(
        method="scalar_qlognei",
        status="recommended",
        theta=torch.stack(theta_rows).contiguous(),
        responses_physical=torch.stack(responses).contiguous(),
        transform=None,
        fit_diagnostics=tuple(fits),
        proposal_diagnostics=tuple(proposals),
        recommendation=recommendation.document(),
        timing=timing,
        elapsed_seconds=total_seconds,
    )
