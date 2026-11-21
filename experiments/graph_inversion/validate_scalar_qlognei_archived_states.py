"""Zero-call scalar-qLogNEI validation on archived composite-NEI states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import botorch
import gpytorch
import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.methods.scalar_qlognei import (
    ScalarQLogNEIConfig,
    fit_scalar_loss_gp,
    propose_scalar_qlognei,
    recommend_scalar_posterior_mean_loss,
    scalar_qlognei_settings_from_document,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)


ARCHIVED_CASES = (
    (0, 16),
    (0, 40),
    (0, 60),
    (1, 34),
    (3, 66),
)


def _verified(path: Path) -> Mapping[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"archived state/hash pair is missing: {path}")
    if sidecar.read_text(encoding="ascii").strip() != file_hash(path):
        raise ValueError(f"archived state hash mismatch: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"archived state is not an object: {path}")
    return value


def validate_archived_states(
    *,
    archive_root: Path,
    target_root: Path,
    configuration_path: Path,
    settings_path: Path,
) -> dict[str, Any]:
    configuration = read_json(configuration_path)
    settings_document = _verified(settings_path)
    settings = scalar_qlognei_settings_from_document(settings_document)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        configuration, device=torch.device("cpu"), dtype=torch.float32
    )
    parameter_spec = simulator.capabilities.parameter_spec
    cases: list[dict[str, Any]] = []
    rng_before = torch.random.get_rng_state().clone()

    for rank, count in ARCHIVED_CASES:
        state_path = (
            archive_root
            / f"worker-{rank:02d}"
            / "trajectory_attempts"
            / f"target-{rank:02d}-b100.json"
        )
        target_path = target_root / f"target_{rank:03d}.json"
        state = _verified(state_path)
        target = _verified(target_path)
        result = state.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"archived trajectory result is missing: {state_path}")
        theta = torch.tensor(
            result.get("theta", [])[:count], dtype=torch.float32
        ).contiguous()
        responses = torch.tensor(
            result.get("responses_physical", [])[:count], dtype=torch.float32
        ).contiguous()
        target_response = torch.tensor(
            target.get("target_response", []), dtype=torch.float32
        ).contiguous()
        if theta.shape != (count, parameter_spec.dimension):
            raise ValueError(f"archived theta prefix has wrong shape: {state_path}")
        if responses.shape[0] != count or responses.shape[1] != target_response.shape[0]:
            raise ValueError(f"archived response prefix has wrong shape: {state_path}")
        case_identity = stable_hash(
            "scalar-qlognei-archived-validation-case-v1",
            {
                "state_sha256": file_hash(state_path),
                "target_sha256": file_hash(target_path),
                "prefix_count": count,
                "settings_identity": settings_document["settings_identity"],
            },
        )
        deriver = SeedDeriver(ScientificIdentity(case_identity), 92021011)
        config = ScalarQLogNEIConfig(
            n_mc=settings["n_mc"],
            raw_samples=settings["raw_samples"],
            restarts=settings["restarts"],
            max_iterations=settings["max_iterations"],
            fit_max_iterations=settings["fit_max_iterations"],
            seed_deriver=deriver,
            fit_coordinate=SeedCoordinate("archived_fit", (rank, count)),
            sampler_coordinate=SeedCoordinate("archived_mc", (rank, count)),
            optimizer_coordinate=SeedCoordinate("archived_optimizer", (rank, count)),
        )
        fitted = fit_scalar_loss_gp(
            theta=theta,
            responses_physical=responses,
            target_physical=target_response,
            parameter_spec=parameter_spec,
            config=config,
        )
        proposal = propose_scalar_qlognei(
            fitted=fitted,
            baseline=theta,
            parameter_spec=parameter_spec,
            config=config,
        )
        terminal = recommend_scalar_posterior_mean_loss(
            fitted=fitted,
            parameter_spec=parameter_spec,
            reference=theta,
            seed_deriver=deriver,
            coordinate=SeedCoordinate("archived_terminal", (rank, count)),
            raw_samples=settings["terminal_raw_samples"],
            restarts=settings["terminal_restarts"],
            max_iterations=settings["terminal_max_iterations"],
        )
        if torch.any(torch.all(proposal.candidate == theta, dim=1)):
            raise ValueError("archived validation returned an exact duplicate")
        if proposal.candidate.dtype is not torch.float32:
            raise ValueError("archived validation changed qLogNEI dtype")
        cases.append(
            {
                "case_identity": case_identity,
                "rank": rank,
                "baseline_size": count,
                "state_path": state_path.as_posix(),
                "state_sha256": file_hash(state_path),
                "target_path": target_path.as_posix(),
                "target_sha256": file_hash(target_path),
                "fit": dict(fitted.fit_diagnostic),
                "proposal": dict(proposal.provenance),
                "terminal": terminal.document(),
                "status": "passed",
                "simulator_calls": 0,
            }
        )
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise ValueError("archived validation changed ambient Torch RNG state")
    body = {
        "schema_version": "mutaco-scalar-qlognei-archived-validation-v1",
        "status": "passed",
        "classification": "zero_call_numerical_prequalification_not_performance_evidence",
        "settings": settings_path.as_posix(),
        "settings_identity": settings_document["settings_identity"],
        "configuration": configuration_path.as_posix(),
        "configuration_sha256": file_hash(configuration_path),
        "cases": cases,
        "baseline_sizes": [item[1] for item in ARCHIVED_CASES],
        "dtype": "torch.float32",
        "precision_fallback_used": False,
        "ambient_rng_isolated": True,
        "botorch_version": str(botorch.__version__),
        "gpytorch_version": str(gpytorch.__version__),
        "scientific_calls": 0,
        "jobs_submitted": 0,
    }
    return {
        **body,
        "validation_identity": stable_hash(
            "scalar-qlognei-archived-validation-v1", body
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    document = validate_archived_states(
        archive_root=args.archive_root,
        target_root=args.target_root,
        configuration_path=args.configuration,
        settings_path=args.settings,
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    if not args.write:
        return
    if not args.confirm_approved or args.output is None:
        raise SystemExit("validation writing requires approval and an output path")
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        raise SystemExit("validation output is append-only")
    write_pretty_json(args.output, document)
    args.output.with_suffix(".sha256").write_text(
        file_hash(args.output), encoding="ascii"
    )


if __name__ == "__main__":
    main()
