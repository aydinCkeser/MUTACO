"""Observational diagnostics for NPE/SNPE; never changes recommendations."""

from __future__ import annotations

from math import log, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mutaco import stable_hash

from ..artifacts import file_hash

QUANTILES = (0.025, 0.25, 0.5, 0.75, 0.975)
SBI_LOSS_SUMMARY_SCHEMA = "sbi_training_summary_loss_v0261"
SBI_LEGACY_LOG_PROB_SUMMARY_SCHEMA = "sbi_training_summary_log_prob_legacy"
SBI_TENSORBOARD_AUDIT_SCHEMA = "sbi_tensorboard_training_loss_audit_v1"


class TrainingDiagnosticsError(ValueError):
    """Typed observational failure; never changes the trained estimator."""


def _summary(inference: Any) -> Mapping[str, Any]:
    summary = getattr(inference, "summary", None)
    if not isinstance(summary, Mapping):
        summary = getattr(inference, "_summary", None)
    return summary if isinstance(summary, Mapping) else {}


def observed_training_summary_keys(inference: Any) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in _summary(inference)))


def _finite_floats(values: Any, *, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TrainingDiagnosticsError(f"{name} must be a numeric sequence") from exc
    if not bool(torch.isfinite(torch.tensor(result, dtype=torch.float64)).all()):
        raise TrainingDiagnosticsError(f"{name} must contain only finite values")
    return result


def _positive_counts(values: Any) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 1 for value in result):
        raise TrainingDiagnosticsError("epochs_trained must contain positive counts")
    if any(float(raw) != value for raw, value in zip(values, result)):
        raise TrainingDiagnosticsError("epochs_trained values must be integers")
    return result


def _loss_summary(
    summary: Mapping[str, Any], *, round_index: int, max_num_epochs: int,
    stop_after_epochs: int | None, tolerance: float,
) -> dict[str, Any]:
    required = {
        "training_loss", "validation_loss", "epochs_trained",
        "best_validation_loss", "epoch_durations_sec",
    }
    if not required.issubset(summary):
        missing = sorted(required - set(summary))
        raise TrainingDiagnosticsError(f"SBI loss summary is incomplete: {missing}")
    counts = _positive_counts(summary["epochs_trained"])
    if round_index < 0 or round_index >= len(counts):
        raise TrainingDiagnosticsError("round_index is not present in epochs_trained")
    train_all = _finite_floats(summary["training_loss"], name="training_loss")
    valid_all = _finite_floats(summary["validation_loss"], name="validation_loss")
    duration_all = _finite_floats(
        summary["epoch_durations_sec"], name="epoch_durations_sec"
    )
    reported = _finite_floats(
        summary["best_validation_loss"], name="best_validation_loss"
    )
    if len(reported) != len(counts):
        raise TrainingDiagnosticsError("best-validation and epoch-count lengths differ")
    expected = sum(counts)
    if not (len(train_all) == len(valid_all) == len(duration_all) == expected):
        raise TrainingDiagnosticsError("cumulative SBI history lengths are inconsistent")
    if any(value < 0 for value in duration_all):
        raise TrainingDiagnosticsError("epoch durations must be nonnegative")
    start = sum(counts[:round_index])
    stop = start + counts[round_index]
    train, valid, durations = (
        values[start:stop] for values in (train_all, valid_all, duration_all)
    )
    best = min(range(len(valid)), key=valid.__getitem__)
    calculated = valid[best]
    if abs(calculated - reported[round_index]) > tolerance:
        raise TrainingDiagnosticsError(
            "reported and calculated best validation losses disagree"
        )
    epochs = counts[round_index]
    return {
        "diagnostic_schema": SBI_LOSS_SUMMARY_SCHEMA,
        "diagnostic_status": "complete",
        "source_type": "in_memory_sbi_summary",
        "source_keys": sorted(str(key) for key in summary),
        "round_index": round_index,
        "actual_epochs": epochs,
        "training_loss": train,
        "validation_loss": valid,
        "epoch_durations_seconds": durations,
        "best_epoch": best,
        "best_validation_loss": calculated,
        "reported_best_validation_loss": reported[round_index],
        "optimization_direction": "minimize",
        "stopping_reason": (
            "maximum_epochs" if epochs >= max_num_epochs else "early_stopping"
        ),
        "max_num_epochs": max_num_epochs,
        "stop_after_epochs": stop_after_epochs,
    }


def _legacy_summary(
    summary: Mapping[str, Any], *, round_index: int, max_num_epochs: int,
    stop_after_epochs: int | None,
) -> dict[str, Any]:
    if round_index != 0:
        raise TrainingDiagnosticsError(
            "legacy log-probability summary cannot identify sequential round slices"
        )
    train = _finite_floats(summary["training_log_probs"], name="training_log_probs")
    valid = _finite_floats(
        summary["validation_log_probs"], name="validation_log_probs"
    )
    if len(train) != len(valid) or not train:
        raise TrainingDiagnosticsError("legacy training histories are inconsistent")
    best = max(range(len(valid)), key=valid.__getitem__)
    epochs = len(train)
    return {
        "diagnostic_schema": SBI_LEGACY_LOG_PROB_SUMMARY_SCHEMA,
        "diagnostic_status": "complete",
        "source_type": "in_memory_sbi_summary",
        "source_keys": sorted(str(key) for key in summary),
        "round_index": round_index,
        "actual_epochs": epochs,
        "training_log_prob": train,
        "validation_log_prob": valid,
        "best_epoch": best,
        "best_validation_log_prob": valid[best],
        "optimization_direction": "maximize",
        "stopping_reason": (
            "maximum_epochs" if epochs >= max_num_epochs else "early_stopping"
        ),
        "max_num_epochs": max_num_epochs,
        "stop_after_epochs": stop_after_epochs,
    }


def training_diagnostics(
    inference: Any, estimator: torch.nn.Module, *,
    max_num_epochs: int, stop_after_epochs: int | None = None,
    round_index: int = 0, sbi_version: str | None = None,
    captured_warnings: tuple[str, ...] = (), tolerance: float = 1e-6,
) -> dict[str, Any]:
    summary = _summary(inference)
    loss_keys = {
        "training_loss", "validation_loss", "epochs_trained",
        "best_validation_loss", "epoch_durations_sec",
    }
    legacy_keys = {"training_log_probs", "validation_log_probs"}
    if loss_keys.issubset(summary):
        result = _loss_summary(
            summary, round_index=round_index, max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs, tolerance=tolerance,
        )
    elif legacy_keys.issubset(summary):
        result = _legacy_summary(
            summary, round_index=round_index, max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
        )
    else:
        raise TrainingDiagnosticsError(
            "unknown SBI training summary schema; observed keys="
            f"{sorted(str(key) for key in summary)}"
        )
    state = estimator.state_dict()
    result.update({
        "sbi_version": sbi_version,
        "warnings": captured_warnings,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "parameter_count": sum(p.numel() for p in estimator.parameters()),
        "state_structure_identity": stable_hash(
            "sbi-state-structure-v1",
            {key: tuple(value.shape) for key, value in state.items()},
        ),
    })
    return result


def safe_training_diagnostics(
    inference: Any, estimator: torch.nn.Module, **arguments: Any,
) -> dict[str, Any]:
    """Observe training without allowing diagnostics to change the method result."""

    try:
        return training_diagnostics(inference, estimator, **arguments)
    except Exception as exc:
        return {
            "diagnostic_schema": None,
            "diagnostic_status": "training_diagnostics_unavailable",
            "source_type": "in_memory_sbi_summary",
            "source_keys": list(observed_training_summary_keys(inference)),
            "diagnostic_failure": f"{type(exc).__name__}: {exc}",
            "round_index": int(arguments.get("round_index", 0)),
            "sbi_version": arguments.get("sbi_version"),
            "warnings": tuple(arguments.get("captured_warnings", ())),
            "diagnostic_simulator_calls": 0,
            "diagnostic_training_calls": 0,
        }


def tensorboard_training_diagnostics(
    event_path: Any, *, max_num_epochs: int, stop_after_epochs: int | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Read and strictly validate persisted SBI loss histories."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", ()))
    required = {
        "epochs_trained", "best_validation_loss", "validation_loss",
        "training_loss", "epoch_durations_sec",
    }
    if not required.issubset(tags):
        raise TrainingDiagnosticsError(
            f"TensorBoard training tags are incomplete: {sorted(required-tags)}"
        )
    summary = {
        key: [item.value for item in accumulator.Scalars(key)] for key in required
    }
    counts = _positive_counts(summary["epochs_trained"])
    rounds = [
        _loss_summary(
            summary, round_index=index, max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs, tolerance=tolerance,
        )
        for index in range(len(counts))
    ]
    for item in rounds:
        item["diagnostic_schema"] = SBI_TENSORBOARD_AUDIT_SCHEMA
        item["source_type"] = "tensorboard_event_scalars"
    return {
        "diagnostic_schema": SBI_TENSORBOARD_AUDIT_SCHEMA,
        "diagnostic_status": "complete",
        "source_type": "tensorboard_event_scalars",
        "source_keys": sorted(tags),
        "rounds": rounds,
        "diagnostic_simulator_calls": 0,
        "diagnostic_training_calls": 0,
    }


def validate_live_tensorboard_parity(
    live: Mapping[str, Any], audit_round: Mapping[str, Any], *,
    loss_tolerance: float = 1e-6, duration_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Fail closed when live and persisted training histories disagree."""

    exact_fields = ("actual_epochs", "best_epoch", "stopping_reason")
    for field in exact_fields:
        if live.get(field) != audit_round.get(field):
            raise TrainingDiagnosticsError(f"live/TensorBoard mismatch: {field}")
    for field, tolerance in (
        ("training_loss", loss_tolerance),
        ("validation_loss", loss_tolerance),
        ("epoch_durations_seconds", duration_tolerance),
    ):
        left = tuple(float(value) for value in live.get(field, ()))
        right = tuple(float(value) for value in audit_round.get(field, ()))
        if len(left) != len(right) or any(
            abs(a - b) > tolerance for a, b in zip(left, right)
        ):
            raise TrainingDiagnosticsError(f"live/TensorBoard mismatch: {field}")
    if abs(
        float(live["best_validation_loss"])
        - float(audit_round["best_validation_loss"])
    ) > loss_tolerance:
        raise TrainingDiagnosticsError(
            "live/TensorBoard mismatch: best_validation_loss"
        )
    return {
        "status": "exact_within_declared_tolerances",
        "loss_tolerance": loss_tolerance,
        "duration_tolerance": duration_tolerance,
        "diagnostic_simulator_calls": 0,
        "diagnostic_training_calls": 0,
    }


def audit_training_artifacts(
    artifact_directory: Path,
    live_history: Sequence[Mapping[str, Any]],
    *,
    max_num_epochs: int,
    stop_after_epochs: int | None = None,
) -> dict[str, Any]:
    """Audit one completed SBI worker without changing its scientific result."""

    events = sorted(artifact_directory.glob("events.out.tfevents*"))
    if len(events) != 1:
        raise TrainingDiagnosticsError(
            f"expected exactly one TensorBoard event file, found {len(events)}"
        )
    audit = tensorboard_training_diagnostics(
        events[0], max_num_epochs=max_num_epochs,
        stop_after_epochs=stop_after_epochs,
    )
    rounds = audit["rounds"]
    if len(live_history) != len(rounds):
        raise TrainingDiagnosticsError(
            "live/TensorBoard mismatch: training round count"
        )
    parity = [
        validate_live_tensorboard_parity(live, persisted)
        for live, persisted in zip(live_history, rounds)
    ]
    return {
        **audit,
        "event_path": events[0].as_posix(),
        "event_sha256": file_hash(events[0]),
        "parity": parity,
        "diagnostic_acceptance": "passed",
    }


def safe_audit_training_artifacts(
    artifact_directory: Path,
    live_history: Sequence[Mapping[str, Any]],
    **arguments: Any,
) -> dict[str, Any]:
    """Record an audit failure without changing completed SBI inference."""

    try:
        return audit_training_artifacts(
            artifact_directory, live_history, **arguments
        )
    except Exception as exc:
        return {
            "diagnostic_schema": SBI_TENSORBOARD_AUDIT_SCHEMA,
            "diagnostic_status": "training_diagnostics_audit_failed",
            "diagnostic_acceptance": "failed",
            "diagnostic_failure": f"{type(exc).__name__}: {exc}",
            "diagnostic_simulator_calls": 0,
            "diagnostic_training_calls": 0,
        }


def _log_gaussian(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(cov)
    delta = x - mean
    solved = torch.cholesky_solve(delta.unsqueeze(-1), chol).squeeze(-1)
    return -0.5 * (
        x.shape[1] * log(2.0 * pi)
        + 2.0 * torch.log(torch.diag(chol)).sum()
        + (delta * solved).sum(dim=1)
    )


def _multimodality(x: torch.Tensor) -> dict[str, Any]:
    n, p = x.shape
    covariance = torch.cov(x.T) + 1e-6 * torch.eye(p, dtype=x.dtype, device=x.device)
    one_ll = _log_gaussian(x, x.mean(0), covariance).sum()
    one_params = p + p * (p + 1) / 2
    one_bic = -2.0 * one_ll + one_params * log(n)
    best: tuple[torch.Tensor, ...] | None = None
    for start in range(10):
        order = torch.argsort(x[:, start % p])
        means = torch.stack((x[order[n // 4]], x[order[(3 * n) // 4]]))
        covs = covariance.repeat(2, 1, 1)
        weights = torch.full((2,), 0.5, dtype=x.dtype, device=x.device)
        previous = None
        converged = False
        for iteration in range(200):
            scores = torch.stack(
                [weights[k].log() + _log_gaussian(x, means[k], covs[k]) for k in range(2)],
                dim=1,
            )
            normalizer = torch.logsumexp(scores, dim=1)
            responsibility = torch.softmax(scores, dim=1)
            mass = responsibility.sum(0).clamp_min(1e-12)
            weights = mass / n
            means = responsibility.T @ x / mass[:, None]
            for k in range(2):
                delta = x - means[k]
                covs[k] = (
                    (responsibility[:, k, None] * delta).T @ delta / mass[k]
                    + 1e-6 * torch.eye(p, dtype=x.dtype, device=x.device)
                )
            ll = normalizer.sum()
            if previous is not None and abs(float(ll - previous)) <= 1e-7:
                converged = True
                break
            previous = ll
        candidate = (ll, weights.clone(), means.clone(), covs.clone(), torch.tensor(converged))
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    ll, weights, means, covs, converged = best
    two_params = 1 + 2 * p + 2 * p * (p + 1) / 2
    two_bic = -2.0 * ll + two_params * log(n)
    pooled = (covs[0] + covs[1]) / 2
    delta = means[0] - means[1]
    separation = torch.sqrt(delta @ torch.linalg.solve(pooled, delta))
    improvement = one_bic - two_bic
    warning = improvement > 10 and weights.min() >= 0.1 and separation > 2
    return {
        "bic_improvement": float(improvement),
        "component_weights": weights.cpu().tolist(),
        "mahalanobis_separation": float(separation),
        "converged": bool(converged),
        "warning": bool(warning),
    }


def posterior_diagnostics(
    samples: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor,
    posterior: Any, target_response: torch.Tensor,
) -> dict[str, Any]:
    finite = torch.isfinite(samples).all(1)
    support = ((samples >= lower) & (samples <= upper)).all(1)
    kept = samples[finite & support]
    if kept.shape[0] < 3:
        raise ValueError("insufficient finite in-support posterior samples")
    mean = kept.mean(0)
    covariance = torch.cov(kept.T)
    boundary = (((kept-lower) <= .01*(upper-lower)) | ((upper-kept) <= .01*(upper-lower))).any(1)
    density_failure = None
    low_support = None
    try:
        logp = posterior.log_prob(kept, norm_posterior=False)
        mean_logp = posterior.log_prob(
            mean.unsqueeze(0), norm_posterior=False
        )[0]
        finite_logp = logp[torch.isfinite(logp)]
        low_support = bool(mean_logp < torch.quantile(finite_logp, .05))
    except Exception as exc:
        density_failure = f"{type(exc).__name__}: {exc}"
    return {
        "requested_sample_count": samples.shape[0], "retained_sample_count": kept.shape[0],
        "finite_fraction": float(finite.float().mean()), "in_support_fraction": float(support.float().mean()),
        "mean": mean.cpu().tolist(), "covariance": covariance.cpu().tolist(),
        "standard_deviation": torch.sqrt(torch.diag(covariance)).cpu().tolist(),
        "quantile_probabilities": QUANTILES,
        "quantiles": torch.quantile(kept, torch.tensor(QUANTILES, device=kept.device, dtype=kept.dtype), dim=0).cpu().tolist(),
        "boundary_mass": float(boundary.float().mean()), "boundary_mass_warning": bool(boundary.float().mean() > .1),
        "low_support_mean_warning": low_support, "density_diagnostic_failure": density_failure,
        "multimodality": _multimodality(kept), "diagnostic_simulator_calls": 0,
    }
