"""Bounded Prompt 5C calibration orchestration and readable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

import psutil

from mutaco import RuntimeRequest, resolve_runtime, stable_hash

from .api import load_campaign
from .artifacts import file_hash, read_json, write_csv, write_pretty_json
from .contracts import TargetSelection
from .methods.supervisor import run_bounded_command
from .simulator_adapters import get_simulator_adapter


CALIBRATION_SBI_METHODS = (
    "sbi_npe", "sbi_nle", "sbi_snle", "sbi_nre", "sbi_snre",
)
CALIBRATION_METHODS = ("mutaco", *CALIBRATION_SBI_METHODS)


@dataclass(frozen=True)
class CalibrationPolicy:
    global_deadline_seconds: float
    target_deadline_seconds: float
    poll_interval_seconds: float
    progress_interval_seconds: float
    termination_grace_seconds: float
    target_indices: tuple[int, ...]
    method_orders: Mapping[int, tuple[str, ...]]
    projection_target_counts: tuple[int, ...]
    projection_concurrency_levels: tuple[int, ...]


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return float(value)


def _positive_int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value)
    ):
        raise ValueError(f"{field} must be a nonempty list of positive integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} values must be unique")
    return tuple(value)


def validate_calibration_document(document: Mapping[str, Any]) -> CalibrationPolicy:
    """Validate the complete non-negotiable Prompt 5C declaration."""

    if document.get("campaign_classification") != "final_setting_calibration":
        raise ValueError("Prompt 5C requires final_setting_calibration classification")
    if document.get("dtype") != "float32":
        raise ValueError("Prompt 5C requires common float32")
    if document.get("simulation_budget_per_target") != 14:
        raise ValueError("Prompt 5C requires exactly 14 inference calls")
    if document.get("runner") != {
        "initial_points": 8,
        "sequential_rounds": 6,
        "transform": {"ddof": 1, "scale_floor": 1e-12},
        "gp": {"likelihood": "task_diagonal_gaussian", "max_iterations": 12},
        "acquisition": {
            "raw_policy": "scrambled_sobol",
            "raw_samples": 64,
            "restarts": 4,
            "max_iterations": 8,
            "learning_rate": 0.05,
            "alpha": 0.1,
            "tau_interior": 1e-8,
        },
        "recommendation_policy": "final_posterior_standardized_l1",
    }:
        raise ValueError("Prompt 5C requires the accepted Prompt 5A MUTACO runner")
    selection = document.get("comparison_set")
    if selection != {"policy": "explicit_indices", "indices": [0, 49, 99]}:
        raise ValueError("Prompt 5C target selection must be [0, 49, 99]")
    if tuple(document.get("sbi_methods", ())) != CALIBRATION_SBI_METHODS:
        raise ValueError("Prompt 5C requires the registered five-method SBI suite")
    if document.get("views") != ["cold_start_per_target"]:
        raise ValueError("Prompt 5C primary view must be cold_start_per_target")
    if document.get("regimes") != ["equal_simulator_calls"]:
        raise ValueError("Prompt 5C primary regime must be equal_simulator_calls")
    if document.get("process_policy") != "fresh_method_process":
        raise ValueError("Prompt 5C requires fresh_method_process")
    campaign = document.get("campaign", {})
    if campaign.get("target_count") != 100 or campaign.get("campaign_seed") != 60101:
        raise ValueError("Prompt 5C requires the frozen 100-target bank and seed 60101")
    simulator = document.get("simulator", {})
    if (
        simulator.get("adapter") != "handwritten_analytic"
        or document.get("simulator_profile") != "noisy_default"
    ):
        raise ValueError("Prompt 5C requires handwritten_analytic/noisy_default")
    sbi = document.get("sbi")
    if not isinstance(sbi, Mapping):
        raise ValueError("Prompt 5C requires materialized sbi settings")
    required_sbi = {
        "density_estimator",
        "classifier",
        "training_batch_size",
        "learning_rate",
        "validation_fraction",
        "snre_validation_fraction",
        "stop_after_epochs",
        "max_num_epochs",
        "posterior_samples",
        "show_progress_bars",
        "mcmc",
    }
    if set(sbi) != required_sbi:
        raise ValueError("Prompt 5C sbi settings must be fully materialized")
    if not isinstance(sbi["density_estimator"], str) or not sbi["density_estimator"]:
        raise ValueError("density_estimator must be a nonempty identifier")
    if not isinstance(sbi["classifier"], str) or not sbi["classifier"]:
        raise ValueError("classifier must be a nonempty identifier")
    for field in (
        "training_batch_size",
        "stop_after_epochs",
        "max_num_epochs",
    ):
        value = sbi[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    _positive_number(sbi["learning_rate"], "learning_rate")
    for field in ("validation_fraction", "snre_validation_fraction"):
        value = sbi[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"{field} must lie strictly between zero and one")
    if not isinstance(sbi["show_progress_bars"], bool):
        raise ValueError("show_progress_bars must be boolean")
    if sbi["posterior_samples"] != 10_000:
        raise ValueError("Prompt 5C requires exactly 10,000 posterior samples")
    mcmc = sbi.get("mcmc")
    if not isinstance(mcmc, Mapping) or dict(mcmc) != {
        "method": "slice_np_vectorized",
        "num_chains": 20,
        "warmup_steps": 100,
        "thin": 1,
        "init_strategy": "proposal",
        "fallback_policy": "none",
    }:
        raise ValueError("Prompt 5C requires the accepted explicit no-fallback MCMC policy")
    calibration = document.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Prompt 5C requires calibration settings")
    global_seconds = _positive_number(
        calibration.get("global_deadline_seconds"), "global_deadline_seconds"
    )
    target_seconds = _positive_number(
        calibration.get("target_deadline_seconds"), "target_deadline_seconds"
    )
    if global_seconds > 1800.0 or target_seconds > 600.0:
        raise ValueError("Prompt 5C deadlines may not exceed 30/10 minutes")
    if global_seconds != 3.0 * target_seconds:
        raise ValueError("global deadline must equal three target blocks")
    if calibration.get("amortized_reuse_policy") != (
        "reuse_completed_cold_simulations_only"
    ):
        raise ValueError("unknown amortized reuse policy")
    if calibration.get("extra_simulator_calls_for_amortized") != 0:
        raise ValueError("Prompt 5C prohibits extra amortized simulator calls")
    raw_orders = calibration.get("target_method_orders")
    if not isinstance(raw_orders, Mapping) or set(raw_orders) != {"0", "49", "99"}:
        raise ValueError("Prompt 5C requires method orders for targets 0, 49, and 99")
    method_orders: dict[int, tuple[str, ...]] = {}
    for raw_index, raw_order in raw_orders.items():
        order = tuple(raw_order) if isinstance(raw_order, list) else ()
        if len(order) != len(CALIBRATION_METHODS) or set(order) != set(
            CALIBRATION_METHODS
        ):
            raise ValueError(f"target {raw_index} method order must contain every method once")
        if order[0] != "mutaco":
            raise ValueError("MUTACO must construct each shared target response first")
        method_orders[int(raw_index)] = order
    grace_seconds = float(calibration.get("termination_grace_seconds", -1.0))
    if grace_seconds < 0.0:
        raise ValueError("termination_grace_seconds must be nonnegative")
    return CalibrationPolicy(
        global_deadline_seconds=global_seconds,
        target_deadline_seconds=target_seconds,
        poll_interval_seconds=_positive_number(
            calibration.get("poll_interval_seconds"), "poll_interval_seconds"
        ),
        progress_interval_seconds=_positive_number(
            calibration.get("progress_interval_seconds"),
            "progress_interval_seconds",
        ),
        termination_grace_seconds=grace_seconds,
        target_indices=(0, 49, 99),
        method_orders=method_orders,
        projection_target_counts=_positive_int_tuple(
            calibration.get("projection_target_counts"), "projection_target_counts"
        ),
        projection_concurrency_levels=_positive_int_tuple(
            calibration.get("projection_concurrency_levels"),
            "projection_concurrency_levels",
        ),
    )


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "sbi", "botorch", "gpytorch", "numpy", "psutil"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "packages": packages,
    }


def validate_calibration_environment(
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail before artifact creation unless the pinned SBI runtime is active."""

    observed = dict(_environment() if environment is None else environment)
    packages = observed.get("packages")
    if not isinstance(packages, Mapping):
        raise RuntimeError("Prompt 5C environment provenance is incomplete")
    if packages.get("sbi") != "0.26.1":
        raise RuntimeError(
            "Prompt 5C requires pinned sbi 0.26.1 in the active interpreter; "
            f"observed sbi={packages.get('sbi')!r}, "
            f"executable={observed.get('executable')!r}. Select the mutaco-sbi "
            "interpreter in PyCharm before running."
        )
    torch_version = packages.get("torch")
    if not isinstance(torch_version, str) or not torch_version.startswith("2.12.1"):
        raise RuntimeError(
            "Prompt 5C requires the pinned Torch 2.12.1 family; "
            f"observed torch={torch_version!r}"
        )
    return observed


def _placeholder(
    *,
    campaign_id: str,
    target_index: int,
    method: str,
    status: str,
    failure_type: str,
    explanation: str,
) -> dict[str, Any]:
    return {
        "schema_version": "mutaco-calibration-record-v1",
        "campaign_id": campaign_id,
        "target_index": target_index,
        "method": method,
        "status": status,
        "failure_type": failure_type,
        "failure_explanation": explanation,
        "target_construction_calls": 0,
        "inference_calls": 0,
        "assessment_calls": 0,
        "gp_fits": 0,
        "runtime_seconds": 0.0,
    }


def deadline_exhaustion(
    now: float, *, target_deadline: float, global_deadline: float
) -> str | None:
    """Return the exact exhausted parent deadline, preferring the global cap."""

    if now >= global_deadline:
        return "global"
    if now >= target_deadline:
        return "target"
    return None


def _run_method(
    *,
    root: Path,
    repository_root: Path,
    config: Mapping[str, Any],
    campaign_id: str,
    target_index: int,
    method: str,
    target_response: Sequence[float] | None,
    effective_device: str,
    allowed_seconds: float,
    deadline_scope: str,
    policy: CalibrationPolicy,
    campaign_science_document: Mapping[str, Any] | None = None,
    target_seed: int | None = None,
    progress_label: str = "Prompt 5C",
) -> dict[str, Any]:
    process_root = root / "processes" / f"target_{target_index:03d}" / method
    request_path = process_root / "request.json"
    result_path = process_root / "result.json"
    request = {
        "schema_version": "mutaco-calibration-worker-request-v1",
        "campaign_id": campaign_id,
        "method": method,
        "simulator_adapter": "handwritten_analytic",
        "target_index": target_index,
        "target_response": list(target_response) if target_response is not None else None,
        "configuration": dict(config),
        "device": effective_device,
        "artifact_output": str(root / "training" / f"t{target_index}_{method}"),
        "progress_output": str(process_root / "progress.json"),
        "campaign_science_document": (
            dict(campaign_science_document)
            if campaign_science_document is not None else None
        ),
        "target_seed": target_seed,
    }
    write_pretty_json(request_path, request)
    command = [
        sys.executable,
        "-m",
        "experiments.benchmarking.methods.calibration_worker",
        "--input",
        str(request_path),
        "--output",
        str(result_path),
    ]
    process_evidence = run_bounded_command(
        command,
        cwd=repository_root,
        stdout_path=process_root / "stdout.log",
        stderr_path=process_root / "stderr.log",
        deadline_seconds=allowed_seconds,
        poll_interval_seconds=policy.poll_interval_seconds,
        termination_grace_seconds=policy.termination_grace_seconds,
        progress_interval_seconds=policy.progress_interval_seconds,
        progress_callback=lambda event: print(
            f"[{progress_label} | target {target_index} | {method}] running; "
            f"elapsed={event['elapsed_seconds']:.1f}s "
            f"remaining_method={event['remaining_seconds']:.1f}s"
        ),
    )
    if process_evidence["timed_out"]:
        progress_path = process_root / "progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else {}
        result = _placeholder(
            campaign_id=campaign_id,
            target_index=target_index,
            method=method,
            status="timeout",
            failure_type=f"{deadline_scope}_deadline_exceeded",
            explanation=(
                f"parent {deadline_scope} hard deadline expired after "
                f"{allowed_seconds:.6f}s"
            ),
        )
        result.update(
            {
                "target_construction_calls": progress.get(
                    "target_construction_calls_completed", 0
                ),
                "target_construction_calls_started": progress.get(
                    "target_construction_calls_started", 0
                ),
                "inference_calls": progress.get("inference_calls_completed", 0),
                "inference_calls_started": progress.get(
                    "inference_calls_started", 0
                ),
                "assessment_calls": progress.get(
                    "assessment_calls_completed", 0
                ),
                "assessment_calls_started": progress.get(
                    "assessment_calls_started", 0
                ),
                "interrupted_phase": progress.get("phase"),
                "call_accounting_status": (
                    "exact_started_and_completed_counts_from_atomic_progress"
                    if progress
                    else "no_scientific_call_progress_observed"
                ),
            }
        )
    elif process_evidence["exit_code"] != 0 or not result_path.is_file():
        result = _placeholder(
            campaign_id=campaign_id,
            target_index=target_index,
            method=method,
            status="failed",
            failure_type="worker_process_failure",
            explanation=(
                f"worker exit={process_evidence['exit_code']}; "
                f"result_exists={result_path.is_file()}"
            ),
        )
    else:
        result = read_json(result_path)
        result.setdefault(
            "failure_type",
            None if result.get("status") == "completed" else "method_failure",
        )
    result["process_evidence"] = process_evidence
    result["cold_end_to_end_seconds"] = process_evidence["wall_seconds"]
    result["parent_deadline_seconds"] = allowed_seconds
    result["parent_deadline_scope"] = deadline_scope
    return result


def _artifact_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _projections(
    rows: Sequence[Mapping[str, Any]],
    policy: CalibrationPolicy,
) -> list[dict[str, Any]]:
    observed = [
        float(row["cold_end_to_end_seconds"])
        for row in rows
        if "cold_end_to_end_seconds" in row
    ]
    seconds_per_target = sum(observed) / len(policy.target_indices)
    censored = sum(row.get("status") != "completed" for row in rows)
    projections: list[dict[str, Any]] = []
    for count in policy.projection_target_counts:
        for concurrency in policy.projection_concurrency_levels:
            projections.append(
                {
                    "target_count": count,
                    "declared_concurrency": concurrency,
                    "projected_wall_seconds": (
                        seconds_per_target * count / concurrency
                    ),
                    "basis": "observed_cold_process_wall_time",
                    "observed_target_count": len(policy.target_indices),
                    "censored_or_failed_record_count": censored,
                    "uncertainty": (
                        "lower-bound-prone because timed-out and not-started work "
                        "is right-censored"
                        if censored
                        else "three-target calibration; scheduling and hardware variance remain"
                    ),
                }
            )
    return projections


def _report(
    *,
    rows: Sequence[Mapping[str, Any]],
    projections: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    elapsed_seconds: float,
    artifact_bytes: int,
) -> str:
    completed = sum(row.get("status") == "completed" for row in rows)
    timeout = sum(row.get("status") == "timeout" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    not_started_target = sum(
        row.get("status") == "not_started_due_to_target_deadline" for row in rows
    )
    not_started_global = sum(
        row.get("status") == "not_started_due_to_global_deadline" for row in rows
    )
    lines = [
        "# Prompt 5C calibration report",
        "",
        "This is resource and failure calibration, not a scientific comparison.",
        "Observed accuracy must not be used for method tuning or selection.",
        "",
        "## Run status",
        "",
        f"- elapsed calculation time: `{elapsed_seconds:.6f}` seconds",
        f"- completed records: `{completed}`",
        f"- timed-out records: `{timeout}`",
        f"- failed records: `{failed}`",
        f"- not started because a target deadline expired: `{not_started_target}`",
        f"- not started because the global deadline expired: `{not_started_global}`",
        f"- effective device: `{runtime.get('effective_device')}`",
        f"- automatic fallback: `{runtime.get('fallback_reason')}`",
        f"- artifact bytes before manifest: `{artifact_bytes}`",
        "",
        "## Projection table",
        "",
        "| targets | concurrency | projected seconds | qualification |",
        "|---:|---:|---:|---|",
    ]
    for item in projections:
        lines.append(
            f"| {item['target_count']} | {item['declared_concurrency']} | "
            f"{item['projected_wall_seconds']:.3f} | {item['uncertainty']} |"
        )
    lines.extend(
        [
            "",
            "Amortized shared training was not run unless completed cold-process "
            "simulation payloads were reusable with zero additional simulator calls. "
            "The status is recorded in `calibration_summary.json`.",
            "",
            "Prompt 5D remains unapproved.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_calibration_artifacts(root: str | Path) -> dict[str, Any]:
    """Validate the immutable Prompt 5C manifest and every inventoried file."""

    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    sidecar = directory / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise ValueError("calibration manifest or manifest hash is missing")
    if file_hash(manifest_path) != sidecar.read_text(encoding="ascii").strip():
        raise ValueError("calibration manifest hash mismatch")
    manifest = read_json(manifest_path)
    for relative, expected in manifest["artifacts"].items():
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"calibration artifact is missing: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"calibration artifact size mismatch: {relative}")
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"calibration artifact hash mismatch: {relative}")
    return manifest


def run_calibration(
    config_path: str | Path,
    *,
    output: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the fixed Prompt 5C calculation under hard parent deadlines."""

    wall_started, cpu_started = perf_counter(), process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    campaign = load_campaign(config_path, device=device)
    config = campaign.document
    policy = validate_calibration_document(config)
    environment = validate_calibration_environment()
    selection = TargetSelection.resolve(
        config["comparison_set"], int(config["campaign"]["target_count"])
    )
    if selection.indices != policy.target_indices:
        raise ValueError("resolved calibration targets differ from policy")
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"calibration output must be new or empty for immutable evidence: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    write_pretty_json(root / "effective_configuration.json", config)
    runtime = resolve_runtime(
        RuntimeRequest(device=campaign.requested_device, dtype=config["dtype"])
    )
    adapter = get_simulator_adapter(campaign.simulator_identifier)
    bank = adapter.target_bank(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    repository_root = Path(__file__).resolve().parents[2]
    global_deadline = wall_started + policy.global_deadline_seconds
    rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    for target_position, target_index in enumerate(policy.target_indices, start=1):
        target_started = perf_counter()
        target_deadline = min(
            target_started + policy.target_deadline_seconds, global_deadline
        )
        target_response: Sequence[float] | None = None
        print(
            f"[Prompt 5C | target {target_position}/3 | index {target_index}] "
            f"started; target_limit={policy.target_deadline_seconds:.0f}s"
        )
        for method_position, method in enumerate(
            policy.method_orders[target_index], start=1
        ):
            now = perf_counter()
            exhausted = deadline_exhaustion(
                now,
                target_deadline=target_deadline,
                global_deadline=global_deadline,
            )
            remaining = min(target_deadline, global_deadline) - now
            if exhausted is not None:
                row = _placeholder(
                    campaign_id=bank.science.value,
                    target_index=target_index,
                    method=method,
                    status=f"not_started_due_to_{exhausted}_deadline",
                    failure_type=f"{exhausted}_deadline_exhausted_before_start",
                    explanation=f"no {exhausted} wall-clock budget remained",
                )
                rows.append(row)
                print(f"[Prompt 5C | target {target_index} | {method}] not started")
                continue
            if method != "mutaco" and target_response is None:
                row = _placeholder(
                    campaign_id=bank.science.value,
                    target_index=target_index,
                    method=method,
                    status="failed",
                    failure_type="shared_target_unavailable",
                    explanation="MUTACO target-construction process did not return a target",
                )
                rows.append(row)
                continue
            elapsed = perf_counter() - wall_started
            eta = max(policy.global_deadline_seconds - elapsed, 0.0)
            print(
                f"[Prompt 5C | target {target_index} | method "
                f"{method_position}/6 | {method}] starting; "
                f"remaining_target={remaining:.1f}s remaining_global={eta:.1f}s"
            )
            row = _run_method(
                root=root,
                repository_root=repository_root,
                config=config,
                campaign_id=bank.science.value,
                target_index=target_index,
                method=method,
                target_response=target_response,
                effective_device=runtime.effective_device,
                allowed_seconds=remaining,
                deadline_scope=(
                    "global" if global_deadline <= target_deadline else "target"
                ),
                policy=policy,
            )
            row["method_order_position"] = method_position
            rows.append(row)
            if method == "mutaco" and row.get("status") == "completed":
                target_response = row.get("target_response")
            print(
                f"[Prompt 5C | target {target_index} | {method}] "
                f"{row['status']}; elapsed={row.get('cold_end_to_end_seconds', 0.0):.1f}s"
            )
        target_summaries.append(
            {
                "target_index": target_index,
                "wall_seconds": perf_counter() - target_started,
                "deadline_seconds": policy.target_deadline_seconds,
                "statuses": {
                    method: next(
                        row["status"]
                        for row in rows
                        if row["target_index"] == target_index
                        and row["method"] == method
                    )
                    for method in CALIBRATION_METHODS
                },
            }
        )
    elapsed_seconds = perf_counter() - wall_started
    for row in rows:
        record = (
            root
            / "records"
            / f"target_{int(row['target_index']):03d}"
            / f"{row['method']}.json"
        )
        write_pretty_json(record, row)
        record.with_suffix(".sha256").write_text(file_hash(record), encoding="ascii")
    write_csv(
        root / "records.csv",
        rows,
        (
            "target_index",
            "method",
            "method_order_position",
            "status",
            "failure_type",
            "failure_explanation",
            "inference_calls",
            "assessment_calls",
            "gp_fits",
            "target_error_l1",
            "target_error_l2",
            "design_error_l2",
            "runtime_seconds",
            "cold_end_to_end_seconds",
            "parent_deadline_seconds",
            "parent_deadline_scope",
        ),
    )
    projections = _projections(rows, policy)
    all_processes_clean = all(
        row.get("process_evidence", {})
        .get("termination", {})
        .get("all_descendants_terminated", True)
        for row in rows
    )
    call_ledger = [
        {
            "target_index": row["target_index"],
            "method": row["method"],
            "status": row["status"],
            "target_construction_calls": row.get("target_construction_calls", 0),
            "inference_calls": row.get("inference_calls", 0),
            "assessment_calls": row.get("assessment_calls", 0),
            "gp_fits": row.get("gp_fits", 0),
            "sequential_round_allocation": row.get(
                "sequential_round_allocation"
            ),
        }
        for row in rows
    ]
    summary = {
        "schema_version": "mutaco-prompt5c-calibration-summary-v1",
        "classification": "resource_calibration_not_scientific_comparison",
        "target_summaries": target_summaries,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in (
                "completed",
                "timeout",
                "failed",
                "not_started_due_to_target_deadline",
                "not_started_due_to_global_deadline",
            )
        },
        "call_ledger": call_ledger,
        "process_cleanup": {
            "all_timed_process_trees_clean": all_processes_clean,
        },
        "amortized_measurement": {
            "status": "not_completed_no_reusable_simulation_payload",
            "extra_simulator_calls": 0,
            "explanation": (
                "fresh cold-process records do not expose reusable training tensors; "
                "the zero-extra-call rule therefore prevented shared retraining"
            ),
        },
        "projections": projections,
    }
    write_pretty_json(root / "calibration_summary.json", summary)
    write_pretty_json(root / "environment.json", environment)
    write_pretty_json(root / "runtime.json", runtime.provenance())
    artifact_bytes = _artifact_size(root)
    (root / "CALIBRATION_REPORT.md").write_text(
        _report(
            rows=rows,
            projections=projections,
            runtime=runtime.provenance(),
            elapsed_seconds=elapsed_seconds,
            artifact_bytes=artifact_bytes,
        ),
        encoding="utf-8",
    )
    inventory = {
        path.relative_to(root).as_posix(): {
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "mutaco-prompt5c-calibration-manifest-v1",
        "classification": "resource_calibration_not_scientific_comparison",
        "campaign_id": bank.science.value,
        "configuration_hash": stable_hash("prompt5c-calibration-config-v1", config),
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed_seconds,
        "parent_process_cpu_seconds": process_time() - cpu_started,
        "parent_pid": os.getpid(),
        "runtime": runtime.provenance(),
        "environment": environment,
        "target_indices": policy.target_indices,
        "method_orders": {
            str(index): order for index, order in policy.method_orders.items()
        },
        "global_deadline_seconds": policy.global_deadline_seconds,
        "target_deadline_seconds": policy.target_deadline_seconds,
        "all_timed_process_trees_clean": all_processes_clean,
        "storage_bytes_before_manifest": _artifact_size(root),
        "summary": summary,
        "artifacts": inventory,
    }
    manifest_path = root / "manifest.json"
    write_pretty_json(manifest_path, manifest)
    (root / "manifest.sha256").write_text(file_hash(manifest_path), encoding="ascii")
    validate_calibration_artifacts(root)
    print(
        f"[Prompt 5C] calculation complete in {elapsed_seconds:.1f}s; "
        f"output={root}"
    )
    return manifest
