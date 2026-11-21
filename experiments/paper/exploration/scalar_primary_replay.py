"""Strict historical MUTACO replay, with no physical simulator backend.

This is a regression adapter around run_mutaco_target, not another optimizer.
GP fits and their diagnostics are reconstructed in the current environment;
the compact archives contain observations, not historical GP parameter states.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash
from mutaco_graph import OverdampedScientificSimulator
from experiments.benchmarking.methods.mutaco_reference import run_mutaco_target
from experiments.benchmarking.methods.scalar_qlognei import scalar_qlognei_settings_from_document
from experiments.graph_inversion.paper_v3_callbacks import _CallbackGraphAdapter
from experiments.graph_inversion.primary_integration_smoke_driver import GraphPrimaryIntegrationSmokeDriver
from experiments.paper.primary_benchmark import (
    generate_primary_sobol_design,
    load_scalar_primary_population_declaration,
)
from experiments.paper.primary_benchmark.execution import primary_method_context


ROOT = Path(__file__).resolve().parents[3]
ARCHIVES = ROOT / "results/paper/scalar_primary_trajectory_archives_v3r2"
REFERENCES = {
    (40, 16): (48, "2f86646f7c21914552d10e9e1fcb209d29880cee194a3e79f8f8fae211b6659c"),
    (100, 0): (0, "e30befedd72b85bd0eda8fe1d7e33ed6ef10acd0c880b1bd6aca60865e4a49c7"),
}


def _sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_archive(budget: int) -> dict[str, Any]:
    """Verify transport, manifest and every member without extracting files."""
    key = (40, 16) if budget == 40 else (100, 0)
    if budget not in {40, 100}:
        raise ValueError("only the two declared historical budgets are supported")
    prefix = f"scalar-primary-b{budget}-v3r2"
    directory = ARCHIVES / f"b{budget}-v3r2"
    archive = directory / (prefix + ".tar.gz")
    manifest_path = directory / (prefix + ".manifest.json")
    with archive.open("rb") as stream:
        actual = _sha256(stream)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not (actual == REFERENCES[key][1] == manifest["archive_sha256"]
            == (directory / (prefix + ".tar.sha256")).read_text().strip()):
        raise ValueError("compressed archive SHA-256 mismatch")
    if hashlib.sha256(manifest_bytes).hexdigest() != (
        directory / (prefix + ".manifest.sha256")
    ).read_text().strip():
        raise ValueError("export manifest SHA-256 mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) != 4 or {m.name for m in members} != set(expected):
            raise ValueError("archive member inventory mismatch")
        for member in members:
            if not member.isfile() or member.size != expected[member.name]["size_bytes"]:
                raise ValueError(f"archive member type or size mismatch: {member.name}")
            with source.extractfile(member) as stream:
                if _sha256(stream) != expected[member.name]["sha256"]:
                    raise ValueError(f"archive member SHA-256 mismatch: {member.name}")
        for stem in ("collection", "compact_export"):
            with source.extractfile(f"collection/{stem}.sha256") as stream:
                sidecar = stream.read().decode("ascii").strip()
            if sidecar != expected[f"collection/{stem}.json"]["sha256"]:
                raise ValueError(f"internal JSON sidecar mismatch: {stem}")
    return {"budget": budget, "archive_sha256": actual, "members_verified": 4}


def load_reference(budget: int, target: int) -> dict[str, Any]:
    record_index, _ = REFERENCES[(budget, target)]
    verify_archive(budget)
    archive = ARCHIVES / f"b{budget}-v3r2/scalar-primary-b{budget}-v3r2.tar.gz"
    with tarfile.open(archive, "r:gz") as source:
        with source.extractfile("collection/compact_export.json") as stream:
            document = json.load(stream)
    if len(document["records"]) != 360 or len(document["trajectory_results"]) != 360 * budget:
        raise ValueError("compact export campaign coverage mismatch")
    record = document["records"][record_index]
    target_path = ROOT / f"experiments/graph_inversion/frozen_targets/paper120_v3/target_{target:03d}.json"
    target_bytes = target_path.read_bytes()
    if hashlib.sha256(target_bytes).hexdigest() != target_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("frozen target SHA-256 mismatch")
    target_document = json.loads(target_bytes)
    if (record["method"] != "mutaco" or record["budget"] != budget
            or record["target_id"] != target_document["target_record_id"]
            or record["status"] != "completed"):
        raise ValueError("historical record identity mismatch")
    if [r["evaluation_index"] for r in record["decision"]["trajectory"]] != list(range(1, budget + 1)):
        raise ValueError("historical trajectory order mismatch")
    return record


class ReplayMismatch(AssertionError):
    def __init__(self, detail: Mapping[str, Any]):
        self.detail = dict(detail)
        super().__init__(json.dumps(self.detail, sort_keys=True))


def json_metadata_equal(actual: Any, expected: Any) -> bool:
    """Compare archive JSON metadata without inventing a tuple/list distinction.

    Numeric values are unchanged; candidate/context checks remain separate.
    Archives cannot encode Python tuples, whereas provenance methods use them.
    """
    return json.loads(json.dumps(actual, allow_nan=False)) == json.loads(json.dumps(expected, allow_nan=False))


class ArchivedResponses:
    """Return an observation only after exact vector, order and context checks."""

    def __init__(self, trajectory: Sequence[Mapping[str, Any]]):
        self.trajectory = trajectory
        self.consumed = 0
        self.first_divergence: dict[str, Any] | None = None

    def reject(self, kind: str, **values: Any) -> None:
        detail = {"kind": kind, "evaluation_index": self.consumed + 1, **values}
        if self.first_divergence is None:
            self.first_divergence = detail
        raise ReplayMismatch(detail)

    def __call__(self, theta: Sequence[float], context: Mapping[str, Any]) -> list[float]:
        if self.first_divergence is not None:
            raise ReplayMismatch(self.first_divergence)
        if self.consumed >= len(self.trajectory):
            self.reject("observation_budget_exceeded")
        expected = self.trajectory[self.consumed]
        if dict(context) != expected["simulator_context"]:
            self.reject("effective_simulation_context", expected=expected["simulator_context"], actual=dict(context))
        if list(theta) != expected["theta"]:
            error = max((abs(a - b) for a, b in zip(theta, expected["theta"])), default=0.0)
            self.reject("candidate_vector", expected=expected["theta"], actual=list(theta), max_abs_error=error)
        self.consumed += 1
        return list(expected["response_physical"])


def replay_reference(budget: int, target: int) -> dict[str, Any]:
    """Invoke the production runner with the production primary seed binding."""
    rng_before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    record = load_reference(budget, target)
    archived = ArchivedResponses(record["decision"]["trajectory"])
    config_root = ROOT / "experiments/graph_inversion/configs/paper"
    declaration = load_scalar_primary_population_declaration(json.loads(
        (config_root / f"scalar_primary_population_b{budget}_v3r2.json").read_text()
    ))
    config_name = "policy_upgrade_100_targets_v1.json" if budget == 40 else "paper120_b100_mutaco_v1.json"
    settings = scalar_qlognei_settings_from_document(json.loads(
        (config_root / "scalar_qlognei_primary_settings_v1.json").read_text()
    ))
    driver = GraphPrimaryIntegrationSmokeDriver(
        declaration=declaration, configuration_path=config_root / config_name,
        qlognei_settings=settings,
    )
    target_document = json.loads((ROOT / f"experiments/graph_inversion/frozen_targets/paper120_v3/target_{target:03d}.json").read_text())
    target_id = target_document["target_record_id"]
    family = stable_hash("primary-design-family-v1", {
        "protocol_id": declaration.comparison.protocol_id,
        "campaign_id": declaration.campaign_id, "study_identity": declaration.study_identity,
        "target_id": target_id, "budget": budget,
        "comparison_identity": declaration.comparison.identity,
    })
    design = generate_primary_sobol_design(
        campaign_id=declaration.campaign_id, target_id=target_id, budget=budget,
        parameter_spec=driver.parameter_spec,
        seed_deriver=SeedDeriver(ScientificIdentity(family), declaration.comparison.design_campaign_seed),
        dtype=torch.float32, device=torch.device("cpu"),
    )
    inference = primary_method_context(declaration, target_id, "mutaco")
    run_seed = SeedDeriver(ScientificIdentity(inference["family_identity"]), inference["root_seed"]).seed_for(
        SeedCoordinate("mutaco_run", (declaration.campaign_id, target_id, budget, "mutaco", "mutaco_run"))
    )
    latest_proposal: dict[str, Any] = {}
    transform_check: dict[str, Any] = {
        "historical": record["decision"]["trajectory"][16]["acquisition_diagnostics"]["transform_id"],
        "reconstructed": None,
    }

    def progress(event: Any) -> None:
        if event.kind.value == "transform_fitted":
            transform_check["reconstructed"] = event.details["transform_id"]
        elif event.kind.value == "proposal_finished":
            latest_proposal.clear()
            latest_proposal.update(event.details)

    def evaluate(theta: Sequence[float], context: Mapping[str, Any]) -> list[float]:
        index = archived.consumed + 1
        runner_context = context["simulation_context"]
        if runner_context["logical_index"] != index - 1:
            archived.reject("runner_observation_order", actual=runner_context)
        effective = driver._simulator_context("mutaco", target_id, inference, index)
        # Check the actual proposed vector and physical context before reading a response.
        expected = archived.trajectory[archived.consumed]
        if list(theta) == expected["theta"] and index > 16:
            if transform_check["reconstructed"] != transform_check["historical"]:
                archived.reject("frozen_transform_identity", **transform_check)
            historical = expected["acquisition_diagnostics"]
            for key in ("branch", "branch_reason", "candidate_source", "candidate_id"):
                if latest_proposal.get(key) != historical[key]:
                    archived.reject("acquisition_decision", field=key, expected=historical[key], actual=latest_proposal.get(key))
        return archived(theta, effective)

    captured: list[Any] = []
    physical_attempts = 0

    def forbidden_physical_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal physical_attempts
        physical_attempts += 1
        raise RuntimeError("physical simulator boundary reached during zero-call replay")

    report: dict[str, Any] = {
        "budget": budget, "target": target, "record_pointer": f"/records/{REFERENCES[(budget, target)][0]}",
        "gp_diagnostics_origin": "reconstructed_current_environment",
        "historical_gp_parameter_snapshots_available": False,
        "candidate_comparison": "exact; no tolerance or candidate substitution",
        "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
    }
    try:
        if design.shared_prefix_identity != record["shared_prefix_identity"]:
            archived.reject("shared_sobol_prefix_identity", expected=record["shared_prefix_identity"], actual=design.shared_prefix_identity)
        with ExitStack() as stack:
            for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
                stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden_physical_call))
            row = run_mutaco_target(
                adapter=_CallbackGraphAdapter(driver.capabilities, evaluate, device=driver.device, dtype=driver.dtype),
                index=target, theta_star=torch.tensor(target_document["theta_star"], dtype=torch.float32),
                config=driver.configuration,
                campaign_science=ScientificIdentity(target_document["target_family_id"]),
                device=driver.device, dtype=driver.dtype,
                supplied_target_response=torch.tensor(target_document["target_response"], dtype=torch.float32),
                supplied_target_seed=target_document["target_generation_seed"], supplied_target_identity=target_id,
                supplied_initial_points=design.shared_prefix, perform_held_out_assessment=False,
                supplied_run_seed=run_seed, supplied_terminal_seed_deriver=driver._terminal_deriver(target_id),
                diagnostic_result_callback=captured.append, verbose_callback=progress,
            )
        if archived.first_divergence is not None:
            raise ReplayMismatch(archived.first_divergence)
        if row["status"] != "completed":
            archived.reject("production_runner_failure", status=row["status"], explanation=row["failure_explanation"])
        if archived.consumed != budget or row["gp_fits"] != budget - 16 + 1:
            archived.reject("logical_accounting", observations=archived.consumed, gp_fits=row["gp_fits"])
        for offset, proposal in enumerate(captured[0].proposals, start=16):
            actual = proposal.provenance()
            expected = archived.trajectory[offset]["acquisition_diagnostics"]
            for key in ("branch", "branch_direction", "branch_reason", "candidate_source",
                        "duplicate", "duplicate_recovery", "raw_rankings", "configuration",
                        "status", "failure_reason", "fallback_reason"):
                if not json_metadata_equal(actual[key], expected[key]):
                    archived.reject("acquisition_diagnostic", proposal_evaluation_index=offset + 1,
                                    field=key, expected=expected[key], actual=actual[key])
        actual_terminal = row["graph_terminal_recommendation"]
        expected_terminal = record["decision"]["terminal_metadata"]
        for key in ("selected_theta", "score", "derived_seed", "seed_namespace", "seed_coordinates",
                    "policy_id", "simulator_calls", "gp_fits", "settings", "raw_rankings",
                    "selected_restart_index", "returned_candidate_source", "target_id",
                    "transform_id", "model_id", "posterior_calls"):
            if not json_metadata_equal(actual_terminal[key], expected_terminal[key]):
                archived.reject("terminal_recommendation", field=key, expected=expected_terminal[key], actual=actual_terminal[key])
        report["status"] = "passed"
    except ReplayMismatch as error:
        report.update(status="diverged", first_divergence=error.detail)
    except Exception as error:
        report.update(status="execution_error", error=f"{type(error).__name__}: {error}")
    finally:
        rng_after = np.random.get_state()
        report.update(
            archived_observations_consumed=archived.consumed,
            physical_simulator_calls=physical_attempts,
            ambient_rng_preserved=(random.getstate() == rng_before[0]
                and all(np.array_equal(a, b) for a, b in zip(rng_before[1], rng_after))
                and torch.equal(rng_before[2], torch.random.get_rng_state())),
            latest_proposal_reconstructed=latest_proposal,
            frozen_transform_identity=transform_check,
        )
        if (transform_check["reconstructed"] is not None
                and transform_check["reconstructed"] != transform_check["historical"]):
            report["first_identity_divergence"] = {
                "stage": "frozen_response_transform", "after_observations": 16,
                **transform_check,
            }
        if captured:
            run = captured[0]
            report["production_status"] = run.status.value
            report["logical_replay_accounting"] = run.logical_accounting
            report["reconstructed_response_transform"] = (
                None if run.transform is None else run.transform.metadata.provenance_document()
            )
            report["reconstructed_gp_fits"] = [fit.provenance() for fit in run.fits]
            report["reconstructed_proposals"] = [proposal.provenance() for proposal in run.proposals]
        if physical_attempts or not report["ambient_rng_preserved"]:
            report["status"] = "invariant_failure"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=(40, 100), required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-intraop-threads", type=int, default=20)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    args = parser.parse_args()
    if (args.budget, args.target) not in REFERENCES:
        parser.error("only B40 target 16 and B100 target 0 are declared references")
    if args.output.exists():
        parser.error("replay reports are append-only; choose a new output path")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    report = replay_reference(args.budget, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("reconstructed_")}, sort_keys=True))
    sys.exit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
