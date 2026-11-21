"""Capture the first production fit/proposal without changing their inputs.

Verbose numerical states belong outside the checkout. Archived observations
and acquisition summaries are identified separately from reconstructed states.
The alternative reduction below is an offline identity investigation only;
it is never supplied to the production replay or accepted as a passing gate.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, redirect_stdout
import gzip
import hashlib
from importlib import import_module, metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import scipy
from threadpoolctl import threadpool_info
import torch

from mutaco import stable_hash
from . import scalar_primary_replay as replay


HISTORICAL_EXECUTION_SOURCES = {
    40: "bd83de6ec01f0f7e2241ee498960a4d493d601e6",
    100: "649235bcb7e8773f5f4b0425ad111314d7d3b1f9",
}


def tensor_document(value: torch.Tensor) -> dict[str, Any]:
    stored = value.detach().cpu().contiguous()
    array = stored.numpy()
    raw = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
    return {"dtype": str(value.dtype), "device": str(value.device),
            "shape": list(value.shape), "stride": list(value.stride()),
            "numpy_dtype": array.dtype.newbyteorder("<").str, "byte_order": "little",
            "data_base64": base64.b64encode(raw).decode("ascii"),
            "sha256_contiguous_bytes": hashlib.sha256(raw).hexdigest(),
            "values": stored.tolist()}


def parameters(model: Any) -> dict[str, Any]:
    return {name: tensor_document(value) for name, value in model.named_parameters()}


def environment_document() -> dict[str, Any]:
    def installed_version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
    numpy_build = io.StringIO()
    scipy_build = io.StringIO()
    with redirect_stdout(numpy_build):
        np.show_config()
    with redirect_stdout(scipy_build):
        scipy.show_config()
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    elif platform.system() == "Darwin":
        cpu_model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                   capture_output=True, check=False).stdout.strip() or cpu_model
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "cpu_model": cpu_model or "unavailable", "processor_description": platform.processor(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "packages": {p: installed_version(p) for p in (
                "torch", "botorch", "gpytorch", "numpy", "scipy", "sbi", "pytest", "linear_operator", "threadpoolctl")},
            "torch_build": torch.__config__.show(), "numpy_build": numpy_build.getvalue(),
            "scipy_build": scipy_build.getvalue(),
            "runtime_threadpools": threadpool_info(),
            "torch_parallel_info": torch.__config__.parallel_info(),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def source_document() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(replay.ROOT), *args], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    files = [Path(__file__), Path(replay.__file__), *sorted((replay.ROOT / "src/mutaco").rglob("*.py"))]
    return {"git_head": git("rev-parse", "HEAD"), "git_status_short": git("status", "--short"),
            "file_sha256": {str(p.relative_to(replay.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files}}


def reference_packet(capture: dict[str, Any]) -> dict[str, Any]:
    """Serialize this same capture without timers or duplicate decimal tensors.

    Tensor payloads are exact little-endian bytes, including infinities and
    signed zero. Source/environment remain outside the numerical-state hash.
    No values from this representation are fed back to the algorithm.
    """
    def clean(value: Any, *, numerical: bool = False) -> Any:
        if isinstance(value, dict):
            tensor = "data_base64" in value and "sha256_contiguous_bytes" in value
            return {key: clean(item, numerical=numerical) for key, item in value.items()
                    if key != "timings" and not key.endswith("_seconds")
                    and not (tensor and key in {"values", "stride"})
                    and not (numerical and key == "execution")}
        if isinstance(value, (list, tuple)):
            return [clean(item, numerical=numerical) for item in value]
        return value

    numerical_state = clean({
        "reconstructed": capture["reconstructed"],
        "frozen_transform": capture["replay_outcome"].get("reconstructed_response_transform"),
        "gp_fit_diagnostics": capture["replay_outcome"].get("reconstructed_gp_fits", []),
    }, numerical=True)
    numerical_state["reconstructed"].pop("transform_investigation", None)
    if "model_builds" in numerical_state["reconstructed"]:
        numerical_state["reconstructed"]["model_builds"] = [
            build for build in numerical_state["reconstructed"]["model_builds"]
            if build["role"] == "fit_model_initialization"]
    encoded = json.dumps(numerical_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema_version": "mutaco-first-adaptive-reference-v1",
            "tensor_origin": "reconstructed_reference_values_not_historical_gp_state",
            "tensor_encoding": "base64_little_endian_numpy_dtype_C_order",
            "numerical_state_sha256": hashlib.sha256(encoded).hexdigest(),
            "capture": clean(capture)}


def write_reference_packet(output: Path, capture: dict[str, Any]) -> dict[str, Any]:
    packet = reference_packet(capture)
    raw = (json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    # Fixed mtime and no filename keep the compressed bytes reproducible.
    with output.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(raw)
    return packet


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"same_shape_and_dtype": False, "exact": False}
    delta = (left - right).abs()
    return {"same_shape_and_dtype": True, "exact": torch.equal(left, right),
            "different_elements": int((left != right).sum()),
            "max_abs_difference": float(delta.max())}


def transform_investigation(
    physical: torch.Tensor, metadata_document: dict[str, Any], historical_id: str,
) -> dict[str, Any]:
    """Test a named reduction order against the entire historical identity."""
    semantic = {k: v for k, v in metadata_document.items() if k not in {"fit_device", "transform_id"}}
    native_mean = physical.mean(dim=0)
    alternate_mean = physical.T.contiguous().mean(dim=1)
    scale = torch.tensor(semantic["effective_scale"], dtype=physical.dtype)
    # Only the mean field changes. The complete hash also covers every scale,
    # observation identifier, component name, dtype, floor, and ddof setting.
    alternate = {**semantic, "mean": alternate_mean.tolist()}
    alternate_id = stable_hash("response-transform-v1", alternate)
    native_id = stable_hash("response-transform-v1", semantic)
    native_match = native_id == metadata_document["transform_id"] == historical_id
    native_z = (physical - native_mean) / scale
    alternate_z = (physical - alternate_mean) / scale
    changed = torch.nonzero(native_mean != alternate_mean).flatten().tolist()
    columns = [{"index": i, "response": semantic["response_names"][i],
                "native_mean": float(native_mean[i]), "alternate_mean": float(alternate_mean[i]),
                "absolute_difference": float(abs(native_mean[i] - alternate_mean[i]))}
               for i in changed]
    flat = int((native_z - alternate_z).abs().argmax())
    row, column = divmod(flat, physical.shape[1])
    return {
        "historical_values_origin": ("native_execution_hash_verified_against_history" if native_match else
                                     "hash_verified_reconstruction" if alternate_id == historical_id else "unresolved"),
        "historical_id": historical_id, "native_id": metadata_document["transform_id"],
        "native_hash_self_consistent": native_id == metadata_document["transform_id"],
        "native_full_semantic_hash_matches_history": native_match,
        "native_mean_matches_executed_transform": native_mean.tolist() == list(semantic["mean"]),
        "alternate_algorithm": "responses.T.contiguous().mean(dim=1), float32",
        "alternate_id": alternate_id, "alternate_full_semantic_hash_matches_history": alternate_id == historical_id,
        "changed_semantic_fields": ["mean"] if changed else [],
        "mean_differences": columns,
        "scale_floor": semantic["scale_floor"], "degrees_of_freedom": semantic["degrees_of_freedom"],
        "floor_activated_count": sum(semantic["floor_activated"]),
        "standardized_data_difference": difference(native_z, alternate_z),
        "largest_standardized_difference_example": {
            "row": row, "component": column, "physical_response": float(physical[row, column]),
            "native_mean": float(native_mean[column]), "alternate_mean": float(alternate_mean[column]),
            "effective_scale": float(scale[column]), "native_standardized": float(native_z[row, column]),
            "alternate_standardized": float(alternate_z[row, column]),
        },
        "alternate_semantic_document": alternate,
        "production_replay_inputs_changed": False,
    }


def capture_first_adaptive(budget: int, target: int) -> dict[str, Any]:
    rng_before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    gp = import_module("mutaco.surrogate.exact_multitask")
    proposal = import_module("mutaco.acquisition.proposal")
    runner = import_module("mutaco.orchestration.runner")
    record = replay.load_reference(budget, target)
    captured: dict[str, Any] = {"origin": "reconstructed_current_execution", "model_builds": [], "restarts": [], "score_calls": []}
    objects: dict[str, Any] = {}
    build_model, propose, posterior = gp._build_model, runner.propose_candidate, gp.ExactMultiTaskGPSurrogate.posterior
    barrier_restart, exploration_restart, score = proposal._barrier_restart, proposal._exploration_restart, proposal._score
    evaluate_barrier = proposal.MatchedBarrierObjective.evaluate
    run_target = replay.run_mutaco_target
    adapters: list[Any] = []

    def finish(outcome: dict[str, Any]) -> dict[str, Any]:
        rng_after = np.random.get_state()
        captured["complete_diagnostic_ambient_rng_preserved"] = (
            random.getstate() == rng_before[0]
            and all(np.array_equal(a, b) for a, b in zip(rng_before[1], rng_after))
            and torch.equal(rng_before[2], torch.random.get_rng_state()))
        manifest_path = replay.ROOT / f"results/paper/graph_scalar_primary_b{budget}_v1/manifest.json"
        historical_source = json.loads(manifest_path.read_text())["scientific_source"]
        if historical_source["commit"] != HISTORICAL_EXECUTION_SOURCES[budget]:
            raise ValueError("historical execution-source identity differs from declared diagnostic reference")
        return {"schema_version": "mutaco-replay-divergence-capture-v1", "budget": budget, "target": target,
                "diagnostic_scope": "first_16_observations_one_fit_one_proposal_no_observation_17",
                "replay_outcome": outcome,
                "archived_first_proposal": record["decision"]["trajectory"][16]["acquisition_diagnostics"],
                "reference": {
                    "record_pointer": f"/records/{replay.REFERENCES[(budget, target)][0]}",
                    "archive_sha256": replay.REFERENCES[(budget, target)][1],
                    "historical_execution_source_commit": HISTORICAL_EXECUTION_SOURCES[budget],
                    "historical_execution_source_origin": f"benchmarks/paper/graph_scalar_primary_b{budget}_v1/manifest.json",
                    "historical_scientific_source": historical_source,
                    "campaign_id": record.get("campaign_id"), "target_id": record.get("target_id"),
                    "shared_prefix_identity": record.get("shared_prefix_identity"),
                    "archived_candidate_17": record["decision"]["trajectory"][16].get("theta"),
                },
                "reconstructed": captured, "environment": environment_document(), "source": source_document()}

    class CaptureComplete(RuntimeError):
        """The first proposal matched, so no observation 17 is needed."""

    class FirstAdaptiveResponses(replay.ArchivedResponses):
        def __init__(self, trajectory: Any):
            super().__init__(trajectory)
            self.capture_complete = False
            adapters.append(self)

        def __call__(self, theta: Any, context: Any) -> list[float]:
            if self.consumed == 16 and self.first_divergence is None:
                expected = self.trajectory[16]
                if list(theta) == expected["theta"] and dict(context) == expected["simulator_context"]:
                    self.capture_complete = True
                    raise CaptureComplete("bounded diagnostic: stopped before returning observation 17")
            return super().__call__(theta, context)

    def capture_build(data: Any, config: Any) -> Any:
        model = build_model(data, config)
        captured["model_builds"].append({
            "role": ("fit_model_initialization" if data.theta.dtype == torch.float32
                     else "fixed_parameter_float64_covariance_mirror"),
            "configuration": config.provenance(), "initial_parameters": parameters(model),
            "augmented_training_inputs": tensor_document(model.train_inputs[0]),
            "flattened_training_targets": tensor_document(model.train_targets),
        })
        if "training_theta" not in captured:
            captured.update(training_theta=tensor_document(data.theta),
                            standardized_training_responses=tensor_document(data.response_standardized),
                            training_observation_ids=list(data.observation_ids))
        return model

    def capture_proposal(surrogate: Any, **kwargs: Any) -> Any:
        objects["surrogate"] = surrogate
        captured["standardized_target"] = tensor_document(kwargs["target"].values)
        result = propose(surrogate, **kwargs)
        captured["raw_candidate_pool"] = tensor_document(result.pool.theta)
        captured["production_proposal"] = result.provenance()
        objects["candidate"] = result.candidate
        return result

    def capture_posterior(surrogate: Any, theta: torch.Tensor) -> Any:
        result = posterior(surrogate, theta)
        if surrogate is objects.get("surrogate") and theta.ndim == 2 and "raw_posterior" not in captured:
            captured["raw_posterior"] = {"mean": tensor_document(result.mean), "covariance": tensor_document(result.covariance)}
        return result

    def capture_score(*args: Any, **kwargs: Any) -> Any:
        result, valid = score(*args, **kwargs)
        number = float(result.detach())
        captured["score_calls"].append({"point": tensor_document(args[1]),
                                       "value": number if math.isfinite(number) else None,
                                       "nonfinite_kind": None if math.isfinite(number) else str(number),
                                       "valid": valid})
        return result, valid

    def capture_barrier(objective: Any, posterior: Any, target: Any) -> Any:
        result = evaluate_barrier(objective, posterior, target)
        if result.values.shape == (64,) and "raw_screening" not in captured:
            values = tensor_document(result.values)
            # Infeasible points legitimately have +inf barrier scores. Keep
            # strict JSON and explicitly identify their original values.
            values["nonfinite"] = [{"index": i, "kind": str(v)}
                                   for i, v in enumerate(values["values"]) if not math.isfinite(v)]
            values["values"] = [v if math.isfinite(v) else None for v in values["values"]]
            captured["raw_screening"] = {
                "values": values, "valid_domain": tensor_document(result.valid_domain),
                "exploitation": tensor_document(result.exploitation),
                "margin": tensor_document(result.margin),
                "feasibility_score": tensor_document(result.feasibility_score),
            }
        return result

    def capture_restart(original: Any, *args: Any, **kwargs: Any) -> Any:
        first = len(captured["score_calls"])
        result = original(*args, **kwargs)
        captured["restarts"].append({"raw_index": kwargs["raw_index"],
                                     "start": tensor_document(args[1]), "selected": None if result.candidate is None else tensor_document(result.candidate),
                                     "score_call_range": [first, len(captured["score_calls"])],
                                     "diagnostic": result.diagnostic.provenance()})
        return result

    def capture_run(**kwargs: Any) -> Any:
        callback = kwargs["diagnostic_result_callback"]
        def finished(run: Any) -> None:
            objects["run"] = run
            callback(run)
        return run_target(**{**kwargs, "diagnostic_result_callback": finished})

    with ExitStack() as stack:
        stack.enter_context(patch.object(gp, "_build_model", capture_build))
        stack.enter_context(patch.object(runner, "propose_candidate", capture_proposal))
        stack.enter_context(patch.object(gp.ExactMultiTaskGPSurrogate, "posterior", capture_posterior))
        stack.enter_context(patch.object(proposal, "_score", capture_score))
        stack.enter_context(patch.object(proposal.MatchedBarrierObjective, "evaluate", capture_barrier))
        stack.enter_context(patch.object(proposal, "_barrier_restart", lambda *a, **k: capture_restart(barrier_restart, *a, **k)))
        stack.enter_context(patch.object(proposal, "_exploration_restart", lambda *a, **k: capture_restart(exploration_restart, *a, **k)))
        stack.enter_context(patch.object(replay, "run_mutaco_target", capture_run))
        stack.enter_context(patch.object(replay, "ArchivedResponses", FirstAdaptiveResponses))
        outcome = replay.replay_reference(budget, target)

    # A matching first proposal also stops here on a reference platform. This
    # command cannot accidentally continue through a full historical budget.
    bounded_stop = bool(adapters and adapters[0].capture_complete)
    if bounded_stop:
        outcome.pop("first_divergence", None)
        outcome.update(status="bounded_diagnostic_complete",
                       diagnostic_stop="before_returning_observation_17")
    outcome.update(full_historical_replay_executed=False,
                   golden_gate_verdict="not_evaluated_by_bounded_capture")

    run = objects.get("run")
    if (run is None or run.transform is None or "surrogate" not in objects
            or objects.get("candidate") is None):
        captured["capture_failure"] = "production execution did not reach a successful first proposal; inspect replay_outcome"
        return finish(outcome)
    initial = run.observations[:16]
    physical = torch.stack([item.response_physical for item in initial])
    theta = torch.stack([item.theta for item in initial])
    archived_initial = record["decision"]["trajectory"][:16]
    archived_y = torch.tensor([item["response_physical"] for item in archived_initial], dtype=torch.float32)
    archived_theta = torch.tensor([item["theta"] for item in archived_initial], dtype=torch.float32)
    captured.update(physical_training_responses=tensor_document(physical),
                    initial_theta_comparison=difference(theta, archived_theta),
                    initial_response_comparison=difference(physical, archived_y),
                    all_archived_initial_numbers_exactly_float32=(
                        archived_y.double().tolist() == [item["response_physical"] for item in archived_initial]
                        and archived_theta.double().tolist() == [item["theta"] for item in archived_initial]))
    historical = record["decision"]["trajectory"][16]["acquisition_diagnostics"]
    transform = run.transform.metadata.provenance_document()
    captured["transform_investigation"] = transform_investigation(physical, transform, historical["transform_id"])
    captured["executed_standardized_data_matches_direct_calculation"] = torch.equal(
        torch.stack([item.response_standardized for item in initial]),
        (physical - run.transform.mean) / run.transform.scale)
    surrogate = objects["surrogate"]
    model = surrogate._model
    captured["fitted_parameters"] = parameters(model)
    captured["effective_candidate_17"] = tensor_document(objects["candidate"])
    captured["fitted_likelihood_noise"] = tensor_document(model.likelihood.noise)
    captured["fitted_task_covariance"] = tensor_document(model.covar_module.kernels[1].covar_matrix.to_dense())
    diagnostic_points = torch.stack((archived_theta[0], torch.tensor([1.25, 1.25, 1.25, 0.5]),
                                     torch.tensor(record["decision"]["trajectory"][16]["theta"]), objects["candidate"]))
    # These extra queries happen only after the actual replay has finished.
    with torch.no_grad():
        diagnostic = posterior(surrogate, diagnostic_points)
    captured["post_replay_diagnostic_posterior"] = {
        "labels": ["initial_point_1", "fixed_interior_point", "archived_candidate_17", "replayed_candidate_17"],
        "theta": tensor_document(diagnostic_points), "mean": tensor_document(diagnostic.mean),
        "covariance": tensor_document(diagnostic.covariance), "included_in_production_call_accounting": False,
    }
    actual_rank = captured["production_proposal"]["raw_rankings"]
    archived_rank = historical["raw_rankings"]
    actual_by_index = {row["raw_index"]: row for row in actual_rank}
    archived_by_index = {row["raw_index"]: row for row in archived_rank}
    shared = sorted(actual_by_index.keys() & archived_by_index.keys())
    captured["historical_raw_ranking_comparison"] = {
        "same_raw_index_order": [r["raw_index"] for r in actual_rank] == [r["raw_index"] for r in archived_rank],
        "only_replayed_indices": sorted(actual_by_index.keys() - archived_by_index.keys()),
        "only_archived_indices": sorted(archived_by_index.keys() - actual_by_index.keys()),
        "rows": [{"raw_index": i, "archived_score": archived_by_index[i]["score"],
                  "replayed_score": actual_by_index[i]["score"],
                  "score_absolute_difference": abs(actual_by_index[i]["score"] - archived_by_index[i]["score"]),
                  "archived_margin": archived_by_index[i]["margin"],
                  "replayed_margin": actual_by_index[i]["margin"]} for i in shared],
    }
    return finish(outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True, choices=(40, 100))
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("verbose", "reference"), default="verbose",
                        help="reference writes deterministic compact JSON in gzip with exact tensor bytes")
    parser.add_argument("--torch-intraop-threads", type=int, default=20)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    args = parser.parse_args()
    if (args.budget, args.target) not in replay.REFERENCES:
        parser.error("only B40 target 16 and B100 target 0 are declared references")
    output = args.output.resolve()
    if output.is_relative_to(replay.ROOT) or output.exists():
        parser.error("choose a new output file outside the repository")
    if args.format == "reference" and not output.name.endswith(".json.gz"):
        parser.error("reference output must end in .json.gz")
    torch.set_num_threads(args.torch_intraop_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    result = capture_first_adaptive(args.budget, args.target)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "reference":
        packet = write_reference_packet(output, result)
        print(json.dumps({"numerical_state_sha256": packet["numerical_state_sha256"],
                          "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
    else:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    investigation = result["reconstructed"].get("transform_investigation", {})
    print(json.dumps({"output": str(output), "replay_status": result["replay_outcome"]["status"],
                      "initial_responses_exact": result["reconstructed"].get("initial_response_comparison", {}).get("exact"),
                      "historical_transform_recovered": (investigation.get("native_full_semantic_hash_matches_history", False)
                                                          or investigation.get("alternate_full_semantic_hash_matches_history", False)),
                      "different_mean_components": len(investigation.get("mean_differences", []))}))
    if (result["replay_outcome"]["status"] not in {"diverged", "bounded_diagnostic_complete"}
            or not result["reconstructed"]["complete_diagnostic_ambient_rng_preserved"]
            or "capture_failure" in result["reconstructed"]
            or result["replay_outcome"]["physical_simulator_calls"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
