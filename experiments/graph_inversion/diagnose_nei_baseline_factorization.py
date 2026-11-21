"""Zero-call replay of archived Q1 NEI baseline covariance failures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from mutaco import (
    FrozenResponseTransform,
    GPFitConfig,
    SurrogateTrainingData,
    stable_hash,
)
from mutaco.atomic_io import atomic_write_bytes
from mutaco.acquisition.nei import _promoted_baseline_cholesky
from mutaco.posterior import structurally_symmetric_materialize
from mutaco.randomness import SeedCoordinate
from mutaco.surrogate.exact_multitask import (
    ExactMultiTaskGPSurrogate,
    _build_model,
)
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)


SCHEMA = "mutaco-nei-baseline-factorization-diagnostic-v1"
ATTEMPTS = (
    (0, 58),
    (1, 35),
    (4, 39),
)


def _load_attempt(root: Path, worker: int) -> Mapping[str, Any]:
    path = (
        root
        / f"worker-{worker:02d}"
        / "trajectory_attempts"
        / f"target-{worker:02d}-b100.json"
    )
    return read_json(path)


def _copy_named_parameters(
    model: torch.nn.Module, values: Mapping[str, Any]
) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(values):
        missing = sorted(set(parameters) - set(values))
        extra = sorted(set(values) - set(parameters))
        raise ValueError(
            f"archived GP parameter mismatch (missing={missing}, extra={extra})"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(
                torch.tensor(values[name], dtype=parameter.dtype, device=parameter.device)
            )


def _surrogate_from_attempt(
    attempt: Mapping[str, Any], configuration: Mapping[str, Any]
) -> tuple[ExactMultiTaskGPSurrogate, torch.Tensor]:
    result = attempt["result"]
    theta = torch.tensor(result["theta"], dtype=torch.float32).contiguous()
    responses = torch.tensor(
        result["responses_physical"], dtype=torch.float32
    ).contiguous()
    simulator = GraphScientificD4Adapter().build_target_simulator(
        configuration, device=torch.device("cpu"), dtype=torch.float32
    )
    response_spec = simulator.capabilities.response_spec
    transform = FrozenResponseTransform.fit(
        responses[:16],
        response_spec=response_spec,
        observation_ids=tuple(f"nei-evaluation-{index}" for index in range(1, 17)),
        scale_floor=1e-12,
    )
    standardized = transform.to_standardized(
        responses, response_names=transform.metadata.response_names
    )
    data = SurrogateTrainingData(
        theta=theta,
        response_standardized=standardized,
        parameter_spec=simulator.capabilities.parameter_spec,
        observation_ids=tuple(
            f"nei-evaluation-{index}" for index in range(1, theta.shape[0] + 1)
        ),
        transform_id=transform.metadata.transform_id,
    )
    final_fit = result["fit_diagnostics"][-1]
    recovery = next(
        item for item in final_fit["recovery_attempts"] if bool(item.get("selected"))
    )
    gp_settings = dict(configuration["runner"]["gp"])
    config = GPFitConfig(
        fit_seed=int(recovery["initialization"]["attempt_seed"]),
        fit_seed_coordinate=SeedCoordinate(
            "gp_fit", ("archived-q1-factorization-replay", int(theta.shape[0]))
        ),
        **gp_settings,
    )
    model = _build_model(data, config)
    _copy_named_parameters(model, recovery["final_parameters"])
    model.eval()
    surrogate = ExactMultiTaskGPSurrogate(
        model,
        training_data=data,
        model_id=str(final_fit["model_id"]),
        config=config,
        fit_id=str(final_fit["fit_id"]),
    )
    return surrogate, theta


def _strict_cholesky(covariance: torch.Tensor) -> bool:
    _, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
    return int(info.detach().cpu()) == 0


def _metrics(covariance: torch.Tensor) -> dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(covariance.double())
    maximum = float(eigenvalues[-1].detach().cpu())
    minimum = float(eigenvalues[0].detach().cpu())
    positive = eigenvalues[eigenvalues > 0]
    condition = (
        math.inf if positive.numel() == 0 else maximum / float(positive[0].detach().cpu())
    )
    return {
        "dtype": str(covariance.dtype),
        "dimension": int(covariance.shape[-1]),
        "minimum_eigenvalue_float64": minimum,
        "maximum_eigenvalue_float64": maximum,
        "positive_spectrum_condition_estimate": condition,
        "strict_cholesky": _strict_cholesky(covariance),
    }


def _promoted_source_covariance(
    surrogate: ExactMultiTaskGPSurrogate, theta: torch.Tensor
) -> torch.Tensor:
    query, _ = surrogate._validate_query(theta)
    promoted = surrogate._promoted_covariance_model()
    posterior = promoted.posterior(
        query.double(),
        output_indices=list(range(surrogate.response_dim)),
        observation_noise=False,
    )
    covariance = surrogate._point_major_joint_covariance(
        posterior.covariance_matrix,
        point_count=query.shape[0],
        interleaved=bool(getattr(posterior.distribution, "_interleaved", True)),
    )
    symmetric, _ = structurally_symmetric_materialize(covariance.contiguous())
    return symmetric


def _jitter_trials(covariance: torch.Tensor) -> list[dict[str, Any]]:
    promoted = covariance.double()
    scale = float(torch.diagonal(promoted).abs().amax().clamp_min(1.0).cpu())
    eps = torch.finfo(promoted.dtype).eps
    identity = torch.eye(
        promoted.shape[-1], dtype=promoted.dtype, device=promoted.device
    )
    trials = []
    for multiplier in (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        jitter = multiplier * promoted.shape[-1] * eps * scale
        accepted = _strict_cholesky(promoted + jitter * identity)
        trials.append(
            {
                "dimension_epsilon_multiplier": multiplier,
                "jitter": jitter,
                "relative_to_diagonal_scale": jitter / scale,
                "strict_cholesky": accepted,
            }
        )
    return trials


def diagnose(
    archive_root: Path, configuration_path: Path, *, source_commit: str
) -> dict[str, Any]:
    configuration = read_json(configuration_path)
    cases = []
    for worker, expected_size in ATTEMPTS:
        attempt = _load_attempt(archive_root, worker)
        surrogate, theta = _surrogate_from_attempt(attempt, configuration)
        if theta.shape[0] != expected_size:
            raise ValueError(
                f"worker {worker:02d} baseline size changed: {theta.shape[0]}"
            )
        returned = surrogate.joint_posterior(theta).covariance.detach()
        promoted_after_return = returned.double().contiguous()
        promoted_source = _promoted_source_covariance(surrogate, theta).detach()
        production_factor, production_diagnostic = _promoted_baseline_cholesky(
            returned
        )
        reconstruction_error = float(
            (
                production_factor @ production_factor.T
                - promoted_after_return
            )
            .abs()
            .amax()
            .cpu()
        )
        cases.append(
            {
                "worker": worker,
                "baseline_size": int(theta.shape[0]),
                "flattened_dimension": int(returned.shape[-1]),
                "archived_failure": attempt["failure"],
                "archived_fit_id": attempt["result"]["fit_diagnostics"][-1]["fit_id"],
                "archived_model_id": attempt["result"]["fit_diagnostics"][-1]["model_id"],
                "returned_float32": _metrics(returned),
                "promoted_after_float32_return": _metrics(promoted_after_return),
                "promoted_at_surrogate_source": _metrics(promoted_source),
                "selected_production_policy": {
                    **production_diagnostic.document(),
                    "maximum_absolute_reconstruction_error": reconstruction_error,
                    "passed": True,
                },
                "bounded_jitter_trials_on_returned_covariance": _jitter_trials(returned),
            }
        )
    body = {
        "schema_version": SCHEMA,
        "status": "completed_zero_call",
        "source_commit": source_commit,
        "scientific_calls": 0,
        "simulator_calls": 0,
        "truba_jobs_submitted": 0,
        "archive_root": archive_root.as_posix(),
        "configuration": configuration_path.as_posix(),
        "cases": cases,
    }
    return {
        **body,
        "diagnostic_identity": stable_hash(
            "nei-baseline-factorization-diagnostic-v1", body
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = diagnose(
        args.archive_root,
        args.configuration,
        source_commit=args.source_commit,
    )
    if args.output is None:
        print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    else:
        output = args.output.resolve()
        if output.exists() or output.with_suffix(".sha256").exists():
            raise FileExistsError(f"diagnostic output must be new: {output}")
        write_pretty_json(output, document)
        atomic_write_bytes(
            output.with_suffix(".sha256"),
            (file_hash(output) + "\n").encode("ascii"),
        )


if __name__ == "__main__":
    main()
