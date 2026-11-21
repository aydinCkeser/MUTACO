"""Zero-physical-call, same-environment refactor references.

This harness imports the scientific implementation from an explicit isolated
source tree. It calls existing production operations, never a second MUTACO
optimizer. Historical closed-loop replay and conditional mathematical inputs
are separate artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3"
TEST_PATHS = [
    "tests/mathematics", "tests/surrogate", "tests/acquisition",
    "tests/orchestration", "tests/unit", "tests/contracts/test_policy_surface.py",
    "tests/infrastructure/test_local_runtime_hygiene.py",
    "tests/experiments/test_scalar_primary_protocol.py",
    "tests/experiments/test_scalar_primary_b100_archive.py",
    "tests/experiments/test_scalar_primary_b100_preparation_audit.py",
]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def canonical(value):
    """Exclude wall time and execution provenance; encode nonfinite diagnostics."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()
                if k not in {"timings", "execution"} and not k.endswith("_seconds")}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": str(value)}
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numerical_payload(document):
    first = document["first_adaptive"]["capture"]
    reconstructed = canonical(first["reconstructed"])
    reconstructed.pop("transform_investigation", None)
    if "model_builds" in reconstructed:
        reconstructed["model_builds"] = [b for b in reconstructed["model_builds"] if b["role"] == "fit_model_initialization"]
    return {"first": reconstructed,
            "transform": first["replay_outcome"].get("reconstructed_response_transform"),
            "fits": canonical(first["replay_outcome"].get("reconstructed_gp_fits", [])),
            "logical_accounting": first["replay_outcome"].get("logical_replay_accounting"),
            "runner_oracle_accounting": canonical(first["replay_outcome"].get("runner_oracle_accounting", {})),
            "checkpoints": document["mathematical_checkpoints"]}


def environment_signature(environment, *, legacy_ordered=False):
    document = {k: environment[k] for k in (
        "platform", "architecture", "cpu_model", "python", "packages", "torch_build",
        "numpy_build", "scipy_build", "torch_parallel_info", "thread_environment", "runtime_threadpools")}
    if not legacy_ordered:
        # Runtime loader enumeration has no defined order. Preserve every field,
        # association, thread count and duplicate record; canonicalize order only.
        document["runtime_threadpools"] = sorted(document["runtime_threadpools"],
            key=lambda item:json.dumps(item,sort_keys=True,separators=(",", ":")))
    return digest(document)


def fit_document(version, diagnostic):
    from mutaco.orchestration.runner import FitRecord
    return FitRecord(version, diagnostic).provenance()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_gzip(path: Path, value) -> None:
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode())


def require_compute() -> None:
    if not os.environ.get("SLURM_JOB_ID") or "ui" in platform.node().split(".")[0]:
        raise RuntimeError("numerical reference work requires a Slurm compute allocation, never a login node")


def bootstrap(science: Path, harness: Path):
    if git(science, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("scientific source must be clean")
    from importlib import metadata
    lock = json.loads((science/"infrastructure/environments/mutaco-sbi-lock.json").read_text())
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("this reference qualification requires Python 3.11; patch differences are recorded")
    for name in ("torch", "botorch", "gpytorch", "numpy", "scipy", "pytest", "psutil"):
        if metadata.version(name) != lock["packages"][name]:
            raise RuntimeError("reference numerical dependency differs from its pinned version: "+name)
    sys.path[:0] = [str(science), str(science/"src"),
                   str(science/"experiments/analytic_target_matching/src"), str(science/"simulators/graph-simulator/src")]
    import mutaco
    if not Path(mutaco.__file__).resolve().is_relative_to(science):
        raise RuntimeError("scientific implementation imported outside the selected source")
    package = importlib.import_module("experiments.paper.exploration")
    loaded = []
    for name in ("scalar_primary_replay", "replay_divergence"):
        full = "experiments.paper.exploration."+name
        spec = importlib.util.spec_from_file_location(full, harness/f"experiments/paper/exploration/{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        setattr(package, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    replay, diagnostics = loaded
    replay.ROOT = science
    replay.ARCHIVES = science/"results/paper/scalar_primary_trajectory_archives_v3r2"
    binding = importlib.import_module("experiments.benchmarking.methods.mutaco_reference")
    if not Path(binding.__file__).resolve().is_relative_to(science):
        raise RuntimeError("production method binding imported from the harness rather than scientific source")
    diagnostics.source_document = lambda: {
        "scientific_commit": git(science, "rev-parse", "HEAD"),
        "harness_commit": git(harness, "rev-parse", "HEAD"),
        "scientific_import": str(mutaco.__file__), "binding_import": str(binding.__file__),
        "core_files_sha256": {str(p.relative_to(science)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((science/"src/mutaco").rglob("*.py"))},
        "diagnostic_files_sha256": {name: hashlib.sha256((harness/f"experiments/paper/exploration/{name}.py").read_bytes()).hexdigest()
                                    for name in ("scalar_primary_replay", "replay_divergence")},
    }
    return replay, diagnostics, binding


def simulator_guard(stack: ExitStack):
    from mutaco_graph import OverdampedScientificSimulator
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append("blocked_physical_backend_attempt")
        raise RuntimeError("physical simulator is forbidden in refactor-reference qualification")
    for name in ("evaluate", "evaluate_batch", "evaluate_condition_grid"):
        stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
    return attempts


def duplicate_checkpoint(record, run, config, deriver):
    """Replay archived duplicate-recovery operands, not a fabricated GP state."""
    from dataclasses import replace
    import torch
    from mutaco import SeedCoordinate
    from mutaco.acquisition.proposal import (
        AcquisitionBranch, DuplicateDiagnostic, ProposalConfig, RawRanking,
        ReturnedCandidateSource, generate_candidate_pool, recover_best_ranked_unevaluated_raw,
    )
    trajectory = record["decision"]["trajectory"]
    selected = next((r for r in trajectory[16:] if r["acquisition_diagnostics"].get("duplicate_recovery")), None)
    if selected is None:
        return {"status": "not_present_in_selected_historical_record"}
    a = selected["acquisition_diagnostics"]
    recovery = a["duplicate_recovery"]
    index = selected["evaluation_index"]-1
    cfg = ProposalConfig.from_seed_deriver(deriver, SeedCoordinate("raw_candidates", ("sequential_round", index-16)),
                                          **config.intent.acquisition_settings)
    pool = generate_candidate_pool(parameter_spec=config.intent.parameter_spec, config=cfg,
                                   dtype=torch.float32, device=torch.device("cpu"))
    if pool.pool_id != a["pool"]["pool_id"]:
        raise RuntimeError("archived duplicate-recovery pool identity mismatch")
    theta = torch.tensor([r["theta"] for r in trajectory[:index]], dtype=torch.float32)
    original_indices = tuple(recovery["original_duplicate_indices"])
    fixture = replace(run.proposals[0], candidate=theta[original_indices[0]].clone(),
                      candidate_id=recovery["original_candidate_id"],
                      candidate_source=ReturnedCandidateSource(recovery["original_candidate_source"]),
                      branch=AcquisitionBranch(a["branch"]), pool=pool,
                      model_id=a["model_id"], target_id=a["target_id"], transform_id=a["transform_id"],
                      raw_rankings=tuple(RawRanking(**row) for row in a["raw_rankings"]),
                      duplicate=DuplicateDiagnostic(True, original_indices, original_indices[0], 0.0))
    result = recover_best_ranked_unevaluated_raw(fixture, evaluated_theta=theta,
                parameter_spec=config.intent.parameter_spec, config=cfg, policy_id=recovery["policy_id"])
    if result is None or result.candidate.tolist() != selected["theta"]:
        raise RuntimeError("archived duplicate operands did not recover their recorded raw candidate")
    return {"origin": "reconstructed_archived_recovery_operands_no_GP_state_or_observation_substitution",
            "evaluation_index": index+1, "candidate": result.candidate.tolist(),
            "candidate_id": result.candidate_id, "recovery": result.duplicate_recovery.provenance()}


def mathematical_checkpoints(record, run, config, diagnostics):
    """Independent final-data cold-fit fixture; never advanced as a trajectory."""
    import random
    import numpy as np
    import torch
    from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
    from mutaco.identity.sequential import sequential_observation_identity
    from mutaco.surrogate import GPFitConfig, SurrogateTrainingData, fit_exact_multitask_gp_with_recovery
    from mutaco.recommendation import recommend_continuous_physical_mae
    from mutaco.policies import RecommendationPolicy
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    intent = config.intent
    science = intent.identity()
    deriver = SeedDeriver(ScientificIdentity(intent.diagnostic_internal_seed_family_id)
                         if intent.diagnostic_internal_seed_family_id else science, intent.root_seed)
    rows = record["decision"]["trajectory"]
    theta = torch.tensor([r["theta"] for r in rows], dtype=torch.float32)
    physical = torch.tensor([r["response_physical"] for r in rows], dtype=torch.float32)
    ids = [o.observation_id for o in run.observations[:16]]
    ids += [sequential_observation_identity(scientific_identity=science, global_logical_index=i,
             sequential_round_index=i-16, candidate_id=r["acquisition_diagnostics"]["candidate_id"], theta=theta[i].contiguous())
            for i, r in enumerate(rows) if i >= 16]
    data = SurrogateTrainingData(theta=theta,
        response_standardized=run.transform.to_standardized(physical, response_names=run.transform.metadata.response_names),
        parameter_spec=intent.parameter_spec, observation_ids=tuple(ids), transform_id=run.transform.metadata.transform_id)
    version = len(rows)-16
    fit_config = GPFitConfig.from_seed_deriver(deriver, SeedCoordinate("gp_fit", ("training_version", version)), **intent.gp_settings)
    fit = fit_exact_multitask_gp_with_recovery(data, config=fit_config, warm_start_states=(), training_version=version)
    if fit.surrogate is None:
        raise RuntimeError("final archived-data cold-fit reference failed")
    terminal = config.terminal_recommendation_context
    if terminal is None or terminal.policy_id != RecommendationPolicy.CONTINUOUS_POSTERIOR_MEAN_PHYSICAL_L1_MAE.value:
        raise RuntimeError("unsupported terminal checkpoint; do not substitute a numerical policy")
    recommendation = recommend_continuous_physical_mae(surrogate=fit.surrogate, transform=run.transform,
        target_physical=torch.tensor(intent.physical_target, dtype=torch.float32), target_id=terminal.target_id,
        parameter_spec=intent.parameter_spec, seed_deriver=terminal.seed_deriver, target_index=terminal.target_index)
    points = torch.stack((theta[0], theta[-1], (theta.min(0).values+theta.max(0).values)/2)).contiguous()
    posterior = fit.surrogate.posterior(points)
    duplicate = duplicate_checkpoint(record, run, config, deriver)
    after = np.random.get_state()
    preserved = (before[0] == random.getstate() and all(np.array_equal(a,b) for a,b in zip(before[1],after))
                 and torch.equal(before[2],torch.random.get_rng_state()))
    if not preserved or recommendation.simulator_calls or recommendation.gp_fits:
        raise RuntimeError("mathematical checkpoint violated RNG or zero-call boundary")
    return canonical({
        "origin": "reconstructed_conditional_mathematical_checkpoints_not_historical_terminal_or_closed_loop",
        "final_fit_initialization": "production_deterministic_cold_on_archived_final_data_no_historical_warm_starts",
        "archived_rows": len(rows), "theta": diagnostics.tensor_document(theta),
        "standardized_responses": diagnostics.tensor_document(data.response_standardized),
        "observation_ids": ids, "fit": fit_document(version, fit.diagnostic),
        "fitted_parameters": diagnostics.parameters(fit.surrogate._model),
        "fitted_noise": diagnostics.tensor_document(fit.surrogate._model.likelihood.noise),
        "task_covariance": diagnostics.tensor_document(fit.surrogate._model.covar_module.kernels[1].covar_matrix.to_dense()),
        "posterior_points": diagnostics.tensor_document(points),
        "posterior_mean": diagnostics.tensor_document(posterior.mean),
        "posterior_covariance": diagnostics.tensor_document(posterior.covariance),
        "terminal": recommendation.document(),
        "duplicate_recovery": duplicate,
        "ambient_rng_preserved": preserved, "physical_simulator_calls": 0,
    })


def capture(args):
    require_compute()
    replay, diagnostics, binding = bootstrap(args.baseline_root, args.source_root)
    import torch
    torch.set_num_threads(20)
    torch.set_num_interop_threads(1)
    objects = {}
    original = binding.run_sequential
    def traced(simulator, *, config, **kwargs):
        objects["config"] = config
        objects["run"] = original(simulator, config=config, **kwargs)
        return objects["run"]
    with ExitStack() as stack:
        attempts = simulator_guard(stack)
        stack.enter_context(patch.object(binding, "run_sequential", traced))
        first = diagnostics.capture_first_adaptive(args.budget, args.target)
        if "capture_failure" in first["reconstructed"] or first["replay_outcome"]["physical_simulator_calls"]:
            raise RuntimeError("first production fit/proposal failed; preserve existing diagnostics")
        first["replay_outcome"]["runner_oracle_accounting"] = objects["run"].physical_accounting
        if (first["replay_outcome"]["archived_observations_consumed"] != 16
                or not first["reconstructed"]["complete_diagnostic_ambient_rng_preserved"]
                or not first["reconstructed"]["initial_theta_comparison"]["exact"]
                or not first["reconstructed"]["initial_response_comparison"]["exact"]):
            raise RuntimeError("first-reference input, RNG or evaluation-accounting invariant failed")
        packet = diagnostics.reference_packet(first)
        write_gzip(Path(str(args.output)+".first.json.gz"), packet)
        extra = mathematical_checkpoints(replay.load_reference(args.budget,args.target), objects["run"], objects["config"], diagnostics)
    if attempts:
        raise RuntimeError("physical backend was reached")
    environment = first["environment"]
    signature = environment_signature(environment)
    result = {"schema_version": "mutaco-truba-refactor-reference-v1", "baseline": BASELINE,
              "source": first["source"], "environment": environment, "environment_signature": signature,
              "environment_signature_schema": "runtime-library-set-v2",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "compute_host": platform.node(),
              "first_adaptive": packet, "mathematical_checkpoints": extra,
              "physical_backend_attempts": len(attempts)}
    result["scientific_hash"] = digest(numerical_payload(result))
    write_gzip(args.output, result)
    print(json.dumps({"output":str(args.output),"scientific_hash":result["scientific_hash"],
                      "historical_first_proposal_status":first["replay_outcome"]["status"],"physical_calls":0}))


def compare_files(first: Path, second: Path):
    def read(path):
        with gzip.open(path,"rt") as stream:return json.load(stream)
    a,b = read(first),read(second)
    valid = all(d.get("environment_signature_schema","ordered-v1") in {"ordered-v1","runtime-library-set-v2"}
                and digest(numerical_payload(d)) == d["scientific_hash"]
                and environment_signature(d["environment"], legacy_ordered=(d.get("environment_signature_schema","ordered-v1")=="ordered-v1"))
                    == d["environment_signature"] for d in (a,b))
    normalized = [environment_signature(d["environment"]) for d in (a,b)]
    return {"artifacts_self_consistent":valid,
            "environment_comparison_schema": "runtime-library-set-v2",
            "same_environment":normalized[0]==normalized[1],
            "runtime_enumeration_order_only_difference": (a["environment_signature"]!=b["environment_signature"] and normalized[0]==normalized[1]),
            "scientific_exact_equal":valid and a["scientific_hash"]==b["scientific_hash"],
            "physical_backend_attempts":[a["physical_backend_attempts"],b["physical_backend_attempts"]],
            "tolerances": "none; exact operation-preserving reference",
            "baseline_scientific_hash":a["scientific_hash"],"comparison_scientific_hash":b["scientific_hash"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("validate","tests","capture","historical","compare"))
    parser.add_argument("--source-root",type=Path,default=ROOT)
    parser.add_argument("--baseline-root",type=Path)
    parser.add_argument("--baseline-commit",default=BASELINE)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--budget",type=int,default=40,choices=(40,100))
    parser.add_argument("--target",type=int,default=16)
    parser.add_argument("--fit-events",type=Path,help="optional separate observed progress-event sidecar")
    parser.add_argument("--first",type=Path)
    parser.add_argument("--second",type=Path)
    args=parser.parse_args()
    if args.action=="validate":
        assert args.baseline_commit==BASELINE
        assert git(args.source_root,"cat-file","-t",BASELINE)=="commit"
        assert not args.output_root.exists()
        print(json.dumps({"scientific_calls":0,"nodes":1,"tasks":1,"cpus":20,"maximum_minutes":15}))
        print(json.dumps({"scientific_calls_executed":0,"job_submitted":False,"physical_simulator_calls_authorized":0,"baseline":BASELINE}))
        return
    if args.action=="compare":
        report=compare_files(args.first,args.second)
        write_json(args.output,report)
        print(json.dumps(report))
        if not report["artifacts_self_consistent"] or not report["same_environment"] or not report["scientific_exact_equal"] or any(report["physical_backend_attempts"]):raise SystemExit(2)
        return
    require_compute()
    if args.action=="capture":capture(args);return
    replay,diagnostics,binding=bootstrap(args.baseline_root,args.source_root)
    import torch
    torch.set_num_threads(20);torch.set_num_interop_threads(1)
    with ExitStack() as stack:
        attempts=simulator_guard(stack)
        if args.action=="historical":
            events=[]
            if args.fit_events is not None:
                original=binding.run_sequential
                def observed(simulator,*,config,**kwargs):
                    callback=kwargs.pop("progress_callback",None)
                    def progress(event):
                        events.append(event.provenance())
                        if callback is not None:callback(event)
                    return original(simulator,config=config,progress_callback=progress,**kwargs)
                stack.enter_context(patch.object(binding,"run_sequential",observed))
            report=replay.replay_reference(args.budget,args.target)
            if args.fit_events is not None:
                write_json(args.fit_events,{"origin":"observed candidate events during strict replay", "events":events})
            write_gzip(args.output,report)
            print(json.dumps({k:report[k] for k in ("status","archived_observations_consumed","physical_simulator_calls")}))
            if report["status"] not in {"passed","diverged"} or report["physical_simulator_calls"] or attempts:raise SystemExit(2)
        else:
            import pytest
            args.output_root.mkdir(parents=True,exist_ok=False)
            os.chdir(args.baseline_root)
            status=pytest.main(["-q",*TEST_PATHS,"--basetemp",str(args.output_root/"temp"),
                                "--junitxml",str(args.output_root/"junit.xml")])
            write_json(args.output_root/"status.json",{"pytest_exit":int(status),"physical_backend_attempts":len(attempts),
                "scientific_source":git(args.baseline_root,"rev-parse","HEAD"),"runtime":sys.executable})
            raise SystemExit(int(status) if not attempts else 2)


if __name__=="__main__":main()
