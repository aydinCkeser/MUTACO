"""Replay archived scalar-qLogNEI fallback states without simulator calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import botorch
import gpytorch
import torch

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods import scalar_qlognei
from experiments.benchmarking.methods.scalar_qlognei import (
    ScalarQLogNEIConfig,
    fit_scalar_loss_gp,
    propose_scalar_qlognei,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)


class _FixedSeedDeriver:
    def __init__(self, seeds: Mapping[object, int]) -> None:
        self._seeds = dict(seeds)

    def seed_for(self, coordinate: object) -> int:
        return self._seeds[coordinate]


def _config(document: Mapping[str, Any]) -> ScalarQLogNEIConfig:
    fit_coordinate = object()
    sampler_coordinate = object()
    optimizer_coordinate = object()
    deriver = _FixedSeedDeriver(
        {
            fit_coordinate: int(document["fit_seed"]),
            sampler_coordinate: int(document["sampler_seed"]),
            optimizer_coordinate: int(document["optimizer_seed"]),
        }
    )
    return ScalarQLogNEIConfig(
        n_mc=int(document["n_mc"]),
        raw_samples=int(document["raw_samples"]),
        restarts=int(document["restarts"]),
        max_iterations=int(document["max_iterations"]),
        fit_max_iterations=int(document["fit_max_iterations"]),
        seed_deriver=deriver,  # type: ignore[arg-type]
        fit_coordinate=fit_coordinate,  # type: ignore[arg-type]
        sampler_coordinate=sampler_coordinate,  # type: ignore[arg-type]
        optimizer_coordinate=optimizer_coordinate,  # type: ignore[arg-type]
    )


def _replay_once(
    *,
    baseline: torch.Tensor,
    responses: torch.Tensor,
    target: torch.Tensor,
    parameter_spec: Any,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    config = _config(configuration)
    captured: dict[str, torch.Tensor] = {}
    original_optimize = scalar_qlognei.optimize_acqf

    def capture_optimize(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        candidate, value = original_optimize(*args, **kwargs)
        captured["candidate"] = candidate.detach().clone().contiguous()
        captured["value"] = value.detach().clone().contiguous()
        return candidate, value

    rng_before = torch.random.get_rng_state().clone()
    scalar_qlognei.optimize_acqf = capture_optimize
    try:
        fitted = fit_scalar_loss_gp(
            theta=baseline,
            responses_physical=responses,
            target_physical=target,
            parameter_spec=parameter_spec,
            config=config,
        )
        proposal = propose_scalar_qlognei(
            fitted=fitted,
            baseline=baseline,
            parameter_spec=parameter_spec,
            config=config,
        )
    finally:
        scalar_qlognei.optimize_acqf = original_optimize
    if "candidate" not in captured or "value" not in captured:
        raise RuntimeError("official optimizer returned no capturable candidate")
    optimized = captured["candidate"].reshape(-1, baseline.shape[1])[0]
    optimized_value = captured["value"].reshape(())
    duplicate_indices = [
        index
        for index, row in enumerate(baseline)
        if torch.equal(optimized, row)
    ]
    trigger_components = []
    if not bool(torch.isfinite(optimized).all()):
        trigger_components.append("nonfinite_candidate")
    if not bool(torch.isfinite(optimized_value)):
        trigger_components.append("nonfinite_acquisition_value")
    if duplicate_indices:
        trigger_components.append("exact_baseline_duplicate")
    return {
        "model_id": fitted.model_id,
        "optimized_candidate": optimized.detach().cpu().tolist(),
        "optimized_log_acquisition_value": float(optimized_value.detach().cpu()),
        "optimized_candidate_finite": bool(torch.isfinite(optimized).all()),
        "optimized_value_finite": bool(torch.isfinite(optimized_value)),
        "exact_duplicate_indices_zero_based": duplicate_indices,
        "trigger_components": trigger_components,
        "returned_candidate": proposal.candidate.detach().cpu().tolist(),
        "returned_log_acquisition_value": float(
            proposal.log_acquisition_value.detach().cpu()
        ),
        "returned_candidate_source": proposal.provenance["candidate_source"],
        "returned_candidate_finite": bool(torch.isfinite(proposal.candidate).all()),
        "returned_candidate_is_exact_duplicate": any(
            torch.equal(proposal.candidate, row) for row in baseline
        ),
        "precision_fallback_used": proposal.provenance["precision_fallback_used"],
        "ambient_torch_rng_preserved": torch.equal(
            rng_before, torch.random.get_rng_state()
        ),
        "simulator_calls": proposal.provenance["simulator_calls"],
    }


def diagnose_record(
    *,
    label: str,
    record_path: Path,
    target: torch.Tensor,
    parameter_spec: Any,
) -> dict[str, Any]:
    record = read_json(record_path)
    trajectory = record["decision"]["trajectory"]
    results = []
    for item in trajectory:
        diagnostics = item.get("acquisition_diagnostics")
        if not isinstance(diagnostics, Mapping) or not str(
            diagnostics.get("candidate_source", "")
        ).startswith("deterministic_raw_fallback_after_"):
            continue
        baseline_size = int(diagnostics["baseline_size"])
        baseline = torch.tensor(
            [row["theta"] for row in trajectory[:baseline_size]],
            dtype=torch.float32,
        )
        responses = torch.tensor(
            [row["response_physical"] for row in trajectory[:baseline_size]],
            dtype=torch.float32,
        )
        first = _replay_once(
            baseline=baseline,
            responses=responses,
            target=target,
            parameter_spec=parameter_spec,
            configuration=diagnostics["configuration"],
        )
        second = _replay_once(
            baseline=baseline,
            responses=responses,
            target=target,
            parameter_spec=parameter_spec,
            configuration=diagnostics["configuration"],
        )
        archived_candidate = [float(value) for value in item["theta"]]
        archived_source = diagnostics["candidate_source"]
        source_matches = first["returned_candidate_source"] == archived_source
        candidate_matches = first["returned_candidate"] == archived_candidate
        if source_matches and candidate_matches:
            replay_classification = "exact_archived_fallback_reproduced"
        else:
            replay_classification = "platform_divergent_archived_trigger_unresolved"
        results.append(
            {
                "evaluation_index_one_based": int(item["evaluation_index"]),
                "baseline_size": baseline_size,
                "archived_candidate_source": archived_source,
                "archived_trigger_constraint": (
                    "optimized_candidate_returned_then_failed_compound_validity_check"
                ),
                "archived_unpersisted_trigger_subtypes": [
                    "nonfinite_candidate",
                    "nonfinite_acquisition_value",
                    "exact_baseline_duplicate",
                ],
                "archived_returned_candidate": archived_candidate,
                "archived_returned_log_acquisition_value": diagnostics[
                    "log_acquisition_value"
                ],
                "replay": first,
                "second_replay_identical": first == second,
                "returned_candidate_matches_archive_exactly": candidate_matches,
                "candidate_source_matches_archive": source_matches,
                "replay_classification": replay_classification,
            }
        )
    exact = [
        item
        for item in results
        if item["replay_classification"] == "exact_archived_fallback_reproduced"
    ]
    return {
        "label": label,
        "budget": int(record["budget"]),
        "record_identity": record["record_identity"],
        "record_sha256": file_hash(record_path),
        "fallback_count": len(results),
        "exact_archived_fallbacks_reproduced": len(exact),
        "platform_divergent_states": len(results) - len(exact),
        "reproduced_exact_duplicate_triggers": sum(
            item["replay"]["trigger_components"] == ["exact_baseline_duplicate"]
            for item in exact
        ),
        "states": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b40-record", type=Path, required=True)
    parser.add_argument("--b100-record", type=Path, required=True)
    parser.add_argument("--target-record", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    args = parser.parse_args()
    target_document = read_json(args.target_record)
    target = torch.tensor(target_document["target_response"], dtype=torch.float32)
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(args.configuration), device=torch.device("cpu"), dtype=torch.float32
    )
    body = {
        "schema_version": "mutaco-scalar-qlognei-fallback-forensics-v1",
        "classification": "zero_call_archived_state_replay",
        "target_record_identity": target_document["target_record_id"],
        "target_record_sha256": file_hash(args.target_record),
        "configuration_sha256": file_hash(args.configuration),
        "replay_environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "botorch": str(botorch.__version__),
            "gpytorch": str(gpytorch.__version__),
            "device": "cpu",
            "dtype": "float32",
        },
        "production_environment_relation": (
            "library_versions_match_but_operating_system_differs_from_linux_jobs"
        ),
        "records": [
            diagnose_record(
                label="job_6248175_b40",
                record_path=args.b40_record,
                target=target,
                parameter_spec=simulator.capabilities.parameter_spec,
            ),
            diagnose_record(
                label="job_6248220_b100",
                record_path=args.b100_record,
                target=target,
                parameter_spec=simulator.capabilities.parameter_spec,
            ),
        ],
        "scientific_calls": 0,
        "jobs_submitted": 0,
    }
    print(json.dumps(body, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
