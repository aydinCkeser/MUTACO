"""Typed declaration/execution split for simulator-general paper campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter, process_time
import traceback
from typing import Any, Callable, Mapping
from hashlib import sha256

import torch

from mutaco import RuntimeRequest, resolve_runtime, stable_hash

from experiments.benchmarking.artifacts import file_hash
from experiments.benchmarking.contracts import TargetSelection
from experiments.benchmarking.engine.target_ledger import load_or_create_target_ledger
from experiments.benchmarking.simulators import get_simulator_adapter
from experiments.benchmarking.timing import (
    TimingValidationError,
    attach_paper_dispatch_timing,
)

from .method_append import append_method_record, load_method_record, record_path
from .integrity import (
    AttemptJournal,
    ConsoleProgress,
    ProgressCallback,
    append_execution_receipt,
    load_execution_receipts,
    new_execution_identity,
    utc_now,
    validate_attempt_journals,
    write_mutaco_evidence,
)
from .validation import (
    FROZEN_PAPER100_CONFIGURATION,
    FROZEN_PAPER100_SHA256,
    PaperContractError,
    read_json_object,
    repository_path,
    require_unexecuted,
    validate_method_ids,
)


PAPER_SCHEMA = "mutaco-paper-campaign-declaration-v1"
SELECTION_SCHEMA = "mutaco-paper-execution-selection-v1"
RECORD_SCHEMA = "mutaco-paper-target-method-record-v2"


@dataclass(frozen=True)
class PaperCampaignDeclaration:
    source_path: Path
    repository_root: Path
    document: dict[str, Any]
    benchmark_configuration: dict[str, Any]
    benchmark_configuration_path: Path
    declared_methods: tuple[str, ...]
    campaign_identity: str
    configuration_warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionSelection:
    source_path: Path
    campaign_id: str
    methods_to_execute: tuple[str, ...]
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PaperExecution:
    output: Path
    executed_records: int
    reused_records: int
    failed_records: int
    execution_receipt: Path


def _read_object(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _method_configuration(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    common = {
        "simulation_budget_per_target": config["simulation_budget_per_target"],
        "dtype": config["dtype"],
        "method_seed_namespace": config.get("randomness", {}).get("method_seed_namespace"),
    }
    if method == "mutaco":
        return {
            **common,
            "runner": config["runner"],
            "proposal_effort": config.get("proposal_effort"),
            "terminal_recommendation": config.get("terminal_recommendation"),
        }
    return {**common, "sbi": config["sbi"], "method": method}


def _effective_method_config(
    declaration: PaperCampaignDeclaration, method: str
) -> dict[str, Any]:
    config = json.loads(json.dumps(declaration.benchmark_configuration))
    overrides = declaration.document.get("method_configuration_overrides", {}).get(method, {})
    if not isinstance(overrides, Mapping):
        raise PaperContractError(f"method override must be a mapping: {method}")
    for section, values in overrides.items():
        if isinstance(values, Mapping) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


def load_paper_campaign(path: str | Path) -> PaperCampaignDeclaration:
    source = Path(path).resolve()
    document = _read_object(source)
    if document.get("schema_version") != PAPER_SCHEMA:
        raise PaperContractError("unsupported paper campaign declaration schema")
    require_unexecuted(document)
    repository_root = Path(__file__).resolve().parents[3]
    benchmark_path = repository_path(
        repository_root, str(document["benchmark_configuration"]), field="benchmark_configuration"
    )
    benchmark = _read_object(benchmark_path)
    if benchmark.get("schema_version") != "mutaco-benchmark-campaign-v1":
        raise PaperContractError("paper campaign requires a benchmark campaign configuration")
    methods = validate_method_ids(document.get("scientifically_comparable_methods", ()))
    if not methods:
        raise PaperContractError("paper campaign must declare comparable methods")
    identity_document = {
        "schema_version": PAPER_SCHEMA,
        "campaign_id": document["campaign_id"],
        "simulator_family": document["simulator_family"],
        "benchmark_configuration_hash": stable_hash("paper-benchmark-config-v1", benchmark),
        "target_cohort": document["target_cohort"],
        "budget_interpretation": document["budget_interpretation"],
        "evaluation_randomness_policy": document["evaluation_randomness_policy"],
        "assessment_policy": document["assessment_policy"],
        "software_schema": RECORD_SCHEMA,
    }
    warnings = ()
    relative_benchmark = benchmark_path.relative_to(repository_root).as_posix()
    if relative_benchmark == FROZEN_PAPER100_CONFIGURATION:
        warnings = (
            "frozen_paper100_identical_duplicate_keys_accepted_at_exact_sha256:"
            + FROZEN_PAPER100_SHA256,
        )
    return PaperCampaignDeclaration(
        source, repository_root, document, benchmark, benchmark_path, methods,
        stable_hash("paper-campaign-science-v1", identity_document),
        warnings,
    )


def load_execution_selection(
    path: str | Path, declaration: PaperCampaignDeclaration
) -> ExecutionSelection:
    source = Path(path).resolve()
    document = _read_object(source)
    if document.get("schema_version") != SELECTION_SCHEMA:
        raise PaperContractError("unsupported paper execution-selection schema")
    require_unexecuted(document)
    if document.get("campaign_id") != declaration.document["campaign_id"]:
        raise PaperContractError("execution selection names a different campaign")
    methods = validate_method_ids(
        document.get("methods_to_execute", ()), declared=declaration.declared_methods
    )
    if not methods:
        raise PaperContractError("methods_to_execute must not be empty")
    raw_indices = document.get("target_indices")
    target_indices: tuple[int, ...] | None = None
    if raw_indices is not None:
        if not isinstance(raw_indices, list) or not raw_indices:
            raise PaperContractError("target_indices must be a nonempty list when present")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_indices):
            raise PaperContractError("target_indices must contain integers")
        target_indices = tuple(raw_indices)
        if len(set(target_indices)) != len(target_indices):
            raise PaperContractError("target_indices must be unique")
        if any(item < 0 for item in target_indices):
            raise PaperContractError("target_indices must be nonnegative")
    return ExecutionSelection(source, str(document["campaign_id"]), methods, target_indices)


def _output_path(declaration: PaperCampaignDeclaration) -> Path:
    output = repository_path(
        declaration.repository_root,
        str(declaration.document["output_directory"]),
        field="output_directory",
    )
    paper_root = (declaration.repository_root / "runs" / "paper").resolve()
    try:
        output.relative_to(paper_root)
    except ValueError as exc:
        raise PaperContractError("paper outputs must resolve under runs/paper") from exc
    return output


def _expected_identity(
    declaration: PaperCampaignDeclaration,
    *,
    adapter: Any,
    target_bank: Any,
    target: Mapping[str, Any],
    method: str,
    runtime: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    simulator = adapter.build_simulator(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    method_config = _method_configuration(config, method)
    software_compatibility = {
        "paper_record_schema": RECORD_SCHEMA,
        "benchmark_method_schema": "benchmark-method-run-v1",
        "torch_version": torch.__version__,
    }
    ledger_hash = declaration.document["target_cohort"].get("hidden_target_ledger_sha256")
    if ledger_hash is None:
        ledger_hash = stable_hash(
            "paper-target-bank-ledger-v1",
            {"target_bank_identity": target_bank.science.value, "indices": config["comparison_set"]},
        )
    return {
        "campaign_identity": declaration.campaign_identity,
        "simulator_identity": simulator.capabilities.simulator_identity,
        "simulator_configuration_hash": stable_hash("paper-simulator-config-v1", config["simulator"]),
        "target_bank_identity": target_bank.science.value,
        "target_record_identity": target["target_record_id"],
        "hidden_target_ledger_hash": ledger_hash,
        "target_response_identity": target["content_hash"],
        "target_response_seed": target["target_seed"],
        "method": method,
        "method_configuration_hash": stable_hash("paper-method-config-v1", method_config),
        "effective_benchmark_configuration_hash": stable_hash(
            "paper-effective-benchmark-config-v1", config
        ),
        "inference_budget": config["simulation_budget_per_target"],
        "budget_interpretation": declaration.document["budget_interpretation"],
        "evaluation_randomness_policy": declaration.document["evaluation_randomness_policy"],
        "assessment_policy": declaration.document["assessment_policy"],
        "assessment_identity": stable_hash(
            "paper-assessment-policy-v1",
            {
                "policy": declaration.document["assessment_policy"],
                "target_record_identity": target["target_record_id"],
                "method": method,
            },
        ),
        "parameter_dimension": len(simulator.capabilities.parameter_spec.axes),
        "response_dimension": len(simulator.capabilities.response_spec.axes),
        "dtype_policy": config["dtype"],
        "device_policy": config.get("device", "auto"),
        "resolved_device": runtime.effective_device,
        "software_schema": RECORD_SCHEMA,
        "software_compatibility_hash": stable_hash(
            "paper-software-compatibility-v1", software_compatibility
        ),
    }


def _execute_method(
    method: str,
    *,
    adapter: Any,
    index: int,
    theta_star: torch.Tensor,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
    campaign_science: Any,
    runtime: Any,
    artifact_output: Path,
    accounting_callback: Callable[[Mapping[str, Any]], None] | None = None,
    method_progress_callback: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    target_response = torch.tensor(
        target["target_response"], device=runtime.torch_device, dtype=runtime.torch_dtype
    ).contiguous()
    if method == "mutaco":
        from experiments.benchmarking.methods.mutaco_reference import run_mutaco_target

        return run_mutaco_target(
            adapter=adapter, index=index, theta_star=theta_star, config=config,
            campaign_science=campaign_science, device=runtime.torch_device,
            dtype=runtime.torch_dtype, supplied_target_response=target_response,
            supplied_target_seed=int(target["target_seed"]),
            supplied_target_identity=str(target["target_record_id"]),
            accounting_callback=accounting_callback,
            verbose_callback=method_progress_callback,
        )
    from experiments.benchmarking.methods.sbi_suite import run_sbi_target

    return run_sbi_target(
        method=method, adapter=adapter, index=index, theta_star=theta_star,
        target_response=target_response, config=config, campaign_science=campaign_science,
        device=runtime.torch_device, dtype=runtime.torch_dtype,
        artifact_output=artifact_output, target_record_id=str(target["target_record_id"]),
        progress_callback=accounting_callback,
    )


def execute_paper_campaign(
    declaration: PaperCampaignDeclaration,
    selection: ExecutionSelection,
    *,
    method_executor: Callable[..., dict[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PaperExecution:
    """Execute only the selected methods; compatible completed records are reused."""

    config = declaration.benchmark_configuration
    runtime = resolve_runtime(RuntimeRequest(device=str(config.get("device", "auto")), dtype=str(config["dtype"])))
    adapter = get_simulator_adapter(str(config["simulator"]["adapter"]))
    bank = adapter.target_bank(config, device=runtime.torch_device, dtype=runtime.torch_dtype)
    declared_targets = TargetSelection.resolve(
        config["comparison_set"], bank.theta.shape[0]
    ).indices
    if selection.target_indices is None:
        selected_targets = declared_targets
    else:
        undeclared = set(selection.target_indices) - set(declared_targets)
        if undeclared:
            raise PaperContractError(
                "execution selection contains targets outside the scientific declaration: "
                + ", ".join(str(index) for index in sorted(undeclared))
            )
        selected_targets = selection.target_indices
    output = _output_path(declaration)
    output.mkdir(parents=True, exist_ok=True)
    prior_receipts = load_execution_receipts(output)
    validate_attempt_journals(output, prior_receipts)
    executor = method_executor or _execute_method
    reporter = progress_callback or ConsoleProgress()
    executed = reused = failed = 0
    started = perf_counter()
    started_at = utc_now()
    execution_id = new_execution_identity(
        declaration.campaign_identity, selection.methods_to_execute
    )
    receipt_records: list[dict[str, Any]] = []
    task_count = len(selection.methods_to_execute) * len(selected_targets)
    completed_tasks = 0
    target_accounting = {
        "target_calls_started": 0,
        "target_calls_completed": 0,
        "target_records_reused": 0,
    }

    def target_progress(event: Mapping[str, Any]) -> None:
        for key in target_accounting:
            if key in event:
                target_accounting[key] = int(event[key])
        try:
            reporter(
                {
                    "event": "target_ledger_progress",
                    "campaign": declaration.document["campaign_id"],
                    "method": "target_ledger",
                    "target_position": int(event.get("target_index", 0)) + 1,
                    "target_count": len(selected_targets),
                    "phase": event.get("phase"),
                    "inference_calls_completed": target_accounting[
                        "target_calls_completed"
                    ],
                    "expected_inference_calls": len(selected_targets),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
        except Exception:
            pass

    try:
        targets, target_manifest = load_or_create_target_ledger(
            adapter=adapter,
            config=config,
            theta_bank=bank.theta,
            target_indices=selected_targets,
            target_family=bank.science,
            output=output,
            device=runtime.torch_device,
            dtype=runtime.torch_dtype,
            restart=True,
            progress_callback=target_progress,
        )
    except Exception as exc:
        receipt_path = append_execution_receipt(
            output,
            {
                "schema_version": "mutaco-paper-execution-receipt-v1",
                "execution_receipt_id": execution_id,
                "campaign_identity": declaration.campaign_identity,
                "declaration": declaration.source_path.relative_to(
                    declaration.repository_root
                ).as_posix(),
                "selection": selection.source_path.relative_to(
                    declaration.repository_root
                ).as_posix(),
                "methods_executed": selection.methods_to_execute,
                "target_indices_selected": selected_targets,
                "configuration_warnings": declaration.configuration_warnings,
                "started_at": started_at,
                "finished_at": utc_now(),
                "environment": runtime.provenance(),
                "status": "target_ledger_failed",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "exception_chain": traceback.format_exc(),
                },
                "target_accounting": target_accounting,
                "records": (),
                "executed_records": 0,
                "reused_records": 0,
                "failed_records": 0,
                "elapsed_seconds": perf_counter() - started,
            },
        )
        raise PaperContractError(
            f"target ledger failed; execution receipt preserved at {receipt_path}"
        ) from exc

    def report(
        *,
        event: str,
        method: str,
        target_position: int,
        phase: str,
        journal: Mapping[str, Any] | None = None,
    ) -> None:
        accounting = {} if journal is None else journal.get("accounting", {})
        elapsed = perf_counter() - started
        eta = (
            elapsed / completed_tasks * (task_count - completed_tasks)
            if completed_tasks > 0
            else None
        )
        document = {
            "event": event,
            "campaign": declaration.document["campaign_id"],
            "method": method,
            "target_position": target_position,
            "target_count": len(selected_targets),
            "phase": phase,
            "inference_calls_completed": accounting.get(
                "inference_calls_completed", 0
            ),
            "expected_inference_calls": config["simulation_budget_per_target"],
            "gp_fits_completed": accounting.get("gp_fits_completed", 0),
            "assessment_calls_completed": accounting.get(
                "assessment_calls_completed", 0
            ),
            "last_completed_operation": (
                None if journal is None else journal.get("last_completed_operation")
            ),
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": eta,
        }
        try:
            reporter(document)
        except Exception:
            pass

    for method in selection.methods_to_execute:
        method_config = _effective_method_config(declaration, method)
        for target_position, index in enumerate(selected_targets, start=1):
            target = targets[index]
            expected = _expected_identity(
                declaration, adapter=adapter, target_bank=bank, target=target,
                method=method, runtime=runtime, config=method_config,
            )
            destination = record_path(output, str(target["target_record_id"]), method)
            existing = load_method_record(destination, expected)
            if existing is not None:
                completed_tasks += 1
                if existing["status"] == "completed":
                    reused += 1
                    event = "method_reused"
                else:
                    failed += 1
                    event = "method_failed"
                receipt_records.append(
                    {
                        "target_record_identity": target["target_record_id"],
                        "method": method,
                        "status": existing["status"],
                        "disposition": "reused" if existing["status"] == "completed" else "existing_failure_preserved",
                        "record": destination.relative_to(output).as_posix(),
                        "record_sha256": file_hash(destination),
                        "record_accounting": existing.get("accounting"),
                        "execution_accounting": {
                            "inference_calls_started": 0,
                            "inference_calls_completed": 0,
                            "gp_fits_started": 0,
                            "gp_fits_completed": 0,
                            "assessment_calls_started": 0,
                            "assessment_calls_completed": 0,
                        },
                    }
                )
                report(
                    event=event,
                    method=method,
                    target_position=target_position,
                    phase="record_reuse",
                )
                continue
            method_started = perf_counter()
            method_cpu_started = process_time()
            artifact_root = (
                output / "method_artifacts" / method / f"target_{index:03d}"
            )
            journal = AttemptJournal(
                output
                / "attempts"
                / ("execution-" + sha256(execution_id.encode("utf-8")).hexdigest()[:16])
                / destination.parent.name
                / f"{method}.json",
                declaration.campaign_identity,
                str(target["target_record_id"]),
                method,
                int(method_config["simulation_budget_per_target"]),
            )
            report(
                event="method_started",
                method=method,
                target_position=target_position,
                phase="method_dispatch",
                journal=journal.state,
            )

            def account(event: Mapping[str, Any]) -> None:
                journal.update(event)
                report(
                    event="method_progress",
                    method=method,
                    target_position=target_position,
                    phase=str(journal.state["phase"]),
                    journal=journal.state,
                )

            caught: BaseException | None = None
            try:
                raw = executor(
                    method, adapter=adapter, index=index, theta_star=bank.theta[index],
                    target=target, config=method_config, campaign_science=bank.science,
                    runtime=runtime, artifact_output=artifact_root,
                    accounting_callback=account,
                    method_progress_callback=None,
                )
            except Exception as exc:
                caught = exc
                raw = {
                    "status": "failed", "failure_stage": journal.state["phase"],
                    "failure_explanation": f"{type(exc).__name__}: {exc}",
                    "failure_type": type(exc).__name__,
                    "failure_traceback": traceback.format_exc(),
                }
            status = "completed" if raw.get("status") == "completed" else "failed"
            journal_document = journal.finalize(
                status=status,
                raw=raw,
                exception=caught,
                opaque_failure=(caught is not None and method_executor is not None),
            )
            if method == "mutaco":
                write_mutaco_evidence(artifact_root, raw)
            runner_provenance = raw.get("runner_provenance")
            runner_physical = (
                runner_provenance.get("physical_accounting", {})
                if isinstance(runner_provenance, Mapping)
                else {}
            )
            runner_timings = (
                runner_provenance.get("timings", {})
                if isinstance(runner_provenance, Mapping)
                else {}
            )
            artifact_inventory = [
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(artifact_root.rglob("*"))
                if path.is_file()
            ]
            sbi_timing = None
            if method.startswith("sbi_"):
                dispatch_wall = perf_counter() - method_started
                dispatch_cpu = process_time() - method_cpu_started
                if isinstance(raw.get("timing_summary"), Mapping):
                    try:
                        sbi_timing = attach_paper_dispatch_timing(
                            raw["timing_summary"],
                            paper_dispatch_wall_seconds=dispatch_wall,
                            paper_dispatch_process_cpu_seconds=dispatch_cpu,
                        )
                    except TimingValidationError as timing_error:
                        sbi_timing = dict(raw["timing_summary"])
                        sbi_timing["completeness"] = "partial_unknown"
                        sbi_timing["paper_dispatch_validation_error"] = (
                            f"{type(timing_error).__name__}: {timing_error}"
                        )
                else:
                    sbi_timing = {
                        "schema_version": "mutaco-paper-sbi-timing-v1",
                        "scope": "in_process_target_method",
                        "completeness": "partial_unknown",
                        "termination_status": status,
                        "failure_phase": raw.get("failure_stage"),
                        "raw_events": raw.get("timing_events"),
                        "method_total": None,
                        "paper_dispatch_total": {
                            "wall_seconds": dispatch_wall,
                            "process_cpu_seconds": dispatch_cpu,
                        },
                        "unavailable_reason": (
                            "SBI executor did not return a validated timing summary"
                        ),
                    }
            record = {
                "schema_version": RECORD_SCHEMA, **expected,
                "target_index": index, "status": status,
                "failure_stage": raw.get("failure_stage"),
                "failure_explanation": raw.get("failure_explanation"),
                "failure_type": raw.get("failure_type"),
                "failure_traceback": raw.get("failure_traceback"),
                "termination_reason": raw.get("status"),
                "warnings": raw.get("warnings", []),
                "recommended_theta": raw.get("recommended_theta"),
                "assessment_response": raw.get("assessment_response"),
                "target_error_l1": raw.get("target_error_l1"),
                "target_error_l2": raw.get("target_error_l2"),
                "design_error_l2": raw.get("design_error_l2"),
                "design_error_l1": raw.get("design_error_l1"),
                "latent_error_l1": raw.get("latent_error_l1"),
                "latent_error_l2": raw.get("latent_error_l2"),
                "accounting": {
                    **journal_document["accounting"],
                    "completeness": journal_document["accounting_completeness"],
                    "inference_simulator_successes": int(
                        raw.get("inference_calls", 0)
                    ),
                    "assessment_simulator_successes": int(
                        raw.get("assessment_calls", 0)
                    ),
                    "gp_fit_successes": int(raw.get("gp_fits", 0)),
                    "training_epochs": raw.get("training_epochs"),
                    "posterior_samples": raw.get("posterior_samples"),
                },
                "diagnostics": {
                    "training": raw.get("training_diagnostics"),
                    "posterior": raw.get("posterior_diagnostics"),
                },
                "timings": {
                    "simulator": raw.get(
                        "simulator_seconds", runner_physical.get("simulator_seconds")
                    ),
                    "training_or_fitting": raw.get(
                        "training_seconds",
                        raw.get("gp_fit_seconds", runner_physical.get("gp_fit_seconds")),
                    ),
                    "proposal": raw.get(
                        "proposal_seconds", runner_physical.get("proposal_seconds")
                    ),
                    "recommendation": raw.get(
                        "recommendation_seconds",
                        runner_timings.get("recommendation_seconds"),
                    ),
                    "assessment": raw.get("assessment_seconds"),
                    "total": float(raw.get("runtime_seconds", perf_counter() - method_started)),
                    "sbi": sbi_timing,
                },
                "environment": runtime.provenance(),
                "seeds": raw.get(
                    "seed_ledger",
                    {
                        "target": raw.get("target_seed", target["target_seed"]),
                        "method_root": raw.get("mutaco_root_seed"),
                        "assessment": raw.get("assessment_seed"),
                    },
                ),
                "attempt_journal": {
                    "path": journal.path.relative_to(output).as_posix(),
                    "sha256": file_hash(journal.path),
                },
                "artifact_inventory": artifact_inventory,
                "artifact_hash_policy": "record_content_hash_plus_sha256_sidecar_v1",
            }
            append_method_record(destination, record)
            executed += 1
            failed += int(status == "failed")
            completed_tasks += 1
            receipt_records.append(
                {
                    "target_record_identity": target["target_record_id"],
                    "method": method,
                    "status": status,
                    "disposition": "executed",
                    "record": destination.relative_to(output).as_posix(),
                    "record_sha256": file_hash(destination),
                    "attempt_journal": journal.path.relative_to(output).as_posix(),
                    "attempt_journal_sha256": file_hash(journal.path),
                    "record_accounting": record["accounting"],
                    "execution_accounting": journal_document["accounting"],
                    "accounting_completeness": journal_document[
                        "accounting_completeness"
                    ],
                }
            )
            report(
                event="method_completed" if status == "completed" else "method_failed",
                method=method,
                target_position=target_position,
                phase=status,
                journal=journal_document,
            )
    execution_accounting = {
        key: sum(
            int(item.get("execution_accounting", {}).get(key, 0))
            for item in receipt_records
        )
        for key in (
            "inference_calls_started",
            "inference_calls_completed",
            "gp_fits_started",
            "gp_fits_completed",
            "proposals_started",
            "proposals_completed",
            "recommendations_started",
            "recommendations_completed",
            "terminal_recommendations_started",
            "terminal_recommendations_completed",
            "assessment_calls_started",
            "assessment_calls_completed",
        )
    }
    execution_accounting.update(target_accounting)
    receipt = {
        "schema_version": "mutaco-paper-execution-receipt-v1",
        "execution_receipt_id": execution_id,
        "campaign_identity": declaration.campaign_identity,
        "declaration": declaration.source_path.relative_to(declaration.repository_root).as_posix(),
        "selection": selection.source_path.relative_to(declaration.repository_root).as_posix(),
        "methods_executed": selection.methods_to_execute,
        "target_indices_selected": selected_targets,
        "target_ledger": target_manifest,
        "target_accounting": target_accounting,
        "configuration_warnings": declaration.configuration_warnings,
        "started_at": started_at,
        "finished_at": utc_now(),
        "environment": runtime.provenance(),
        "records": receipt_records,
        "execution_accounting": execution_accounting,
        "status": "completed_with_failures" if failed else "completed",
        "executed_records": executed, "reused_records": reused, "failed_records": failed,
        "elapsed_seconds": perf_counter() - started,
    }
    receipt_path = append_execution_receipt(output, receipt)
    return PaperExecution(output, executed, reused, failed, receipt_path)

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.campaign")
