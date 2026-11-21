"""Shallow public API for running and reusing benchmark campaigns."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch
import psutil

from mutaco import RuntimeRequest, resolve_runtime, stable_hash

from .artifacts import (
    file_hash,
    read_json,
    write_canonical_json,
    write_csv,
    write_pretty_json,
)
from .contracts import (
    Campaign,
    CampaignExecution,
    MutacoBudget,
    TargetSelection,
    validate_method_selection,
)
from .engine.reference import replay_reference, run_reference_cohort
from .engine.target_ledger import (
    load_or_create_target_ledger,
    uses_independent_target_ledger,
)
from .methods import (
    AMORTIZED_METHODS,
    run_amortized_bank,
    run_sbi_target_process,
)
from .methods.sbi_suite import run_sbi_target
from .results import (
    ROW_FIELDS, BenchmarkResultSet, export_tables, load_results, plot_results, summarize_results,
)
from .simulator_adapters import get_simulator_adapter
from .statistics import aggregate_rows


def _compact_public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep shallow analysis fields; full method records remain separately hashed."""

    omitted = {
        "runner_provenance",
        "scientific_projection",
        "timing_events",
    }
    return {key: value for key, value in row.items() if key not in omitted}


def load_campaign(path: str | Path, *, device: str | None = None) -> Campaign:
    source = Path(path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    schema = document.get("schema_version")
    if schema == "mutaco-noisy-cohort-v1":
        simulator = str(document["simulator"]["identifier"])
        reference = "mutaco"
        sbi_methods: tuple[str, ...] = ()
    elif schema == "mutaco-benchmark-campaign-v1":
        simulator = str(document["simulator"]["adapter"])
        reference = str(document.get("reference_method", "mutaco"))
        sbi_methods = validate_method_selection(
            reference, document.get("sbi_methods", [])
        )
        MutacoBudget.resolve(document)
    else:
        raise ValueError(f"unsupported benchmark campaign schema: {schema!r}")
    if simulator == "graph_scientific_d4" and (
        "proposal_effort" in document or "proposal_controls" in document
    ):
        from experiments.graph_inversion.paper_policies import (
            validate_graph_paper_configuration,
        )

        validate_graph_paper_configuration(document)
    requested_device = device or str(document.get("device", "auto"))
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    validate_method_selection(reference, sbi_methods)
    if sbi_methods:
        if document.get("dtype") != "float32":
            raise ValueError(
                "pinned sbi 0.26.1 requires float32; all compared methods must "
                "use the same declared dtype"
            )
        sbi_settings = document.get("sbi")
        if not isinstance(sbi_settings, dict):
            raise ValueError("SBI selection requires typed sbi settings")
        posterior_samples = sbi_settings.get("posterior_samples")
        if (
            isinstance(posterior_samples, bool)
            or not isinstance(posterior_samples, int)
            or posterior_samples < 1
        ):
            raise ValueError("posterior_samples must be a positive integer")
        if (
            document.get("campaign_classification") != "mechanical_smoke"
            and posterior_samples != 10_000
        ):
            raise ValueError(
                "scientific SBI campaigns require exactly 10,000 posterior samples"
            )
        views = tuple(document.get("views", ("cold_start_per_target",)))
        unknown_views = set(views) - {"cold_start_per_target", "amortized_bank"}
        if unknown_views:
            raise ValueError(f"unknown benchmark views: {sorted(unknown_views)}")
        regimes = tuple(document.get("regimes", ("equal_simulator_calls",)))
        unknown_regimes = set(regimes) - {
            "equal_simulator_calls",
            "equal_wall_clock",
            "joint_cap",
        }
        if unknown_regimes:
            raise ValueError(f"unknown budget regimes: {sorted(unknown_regimes)}")
        process_policy = document.get("process_policy")
        if process_policy not in {"fresh_method_process", "in_process"}:
            raise ValueError(
                "process_policy must be fresh_method_process or in_process"
            )
        if (
            "cold_start_per_target" in views
            and process_policy != "fresh_method_process"
        ):
            raise ValueError(
                "cold_start_per_target requires process_policy=fresh_method_process"
            )
    if document.get("identity_policy") == "target_method_execution_v1":
        limits = document.get("runtime_limits")
        if not isinstance(limits, dict):
            raise ValueError("target/method identity campaigns require runtime_limits")
        for name in ("global_deadline_seconds", "artifact_cap_bytes"):
            value = limits.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"runtime_limits.{name} must be a positive integer")
        if limits.get("resume_policy") != "completed_target_or_method_only":
            raise ValueError("unsupported target/method recovery policy")
    return Campaign(source, document, simulator, reference, sbi_methods, requested_device)


def run_campaign(
    campaign: Campaign,
    *,
    output: str | Path,
    restart: bool = False,
    verbose_events: bool = False,
) -> CampaignExecution:
    """Run the declared campaign; reference-only mode imports no SBI."""

    config = campaign.document
    benchmark_config_hash = stable_hash("benchmark-config-v1", config)
    runtime = resolve_runtime(
        RuntimeRequest(device=campaign.requested_device, dtype=str(config["dtype"]))
    )
    adapter = get_simulator_adapter(campaign.simulator_identifier)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    runtime_limits = config.get("runtime_limits", {})
    global_deadline_seconds = runtime_limits.get("global_deadline_seconds")

    def remaining_global_seconds() -> float | None:
        if global_deadline_seconds is None:
            return None
        return max(0.0, float(global_deadline_seconds) - (perf_counter() - started))
    bank = adapter.target_bank(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    selection_document = config.get("comparison_set", {"policy": "full_bank"})
    selection = TargetSelection.resolve(selection_document, bank.theta.shape[0])
    target_records = None
    target_ledger_manifest = None
    if uses_independent_target_ledger(config):
        target_records, target_ledger_manifest = load_or_create_target_ledger(
            adapter=adapter,
            config=config,
            theta_bank=bank.theta,
            target_indices=selection.indices,
            target_family=bank.science,
            output=root,
            device=runtime.torch_device,
            dtype=runtime.torch_dtype,
            restart=restart,
        )
    bank_document = {
        "campaign_id": bank.science.value,
        "target_family_id": bank.science.value,
        "policy": config["campaign"]["target_bank_policy"],
        "seed": bank.seed,
        "target_count": bank.theta.shape[0],
        "selected_indices": selection.indices,
        "parameter_names": bank.parameter_names,
        "bounds": bank.bounds,
        "dtype": config["dtype"],
        "theta": bank.theta.detach().cpu().tolist(),
    }
    existing_bank = root / "target_bank.json"
    if restart and existing_bank.exists():
        if read_json(existing_bank) != json.loads(json.dumps(bank_document)):
            raise ValueError("existing target bank does not match requested campaign")
    else:
        write_pretty_json(existing_bank, bank_document)
    reference_rows, execution = run_reference_cohort(
        adapter=adapter,
        config=config,
        theta_bank=bank.theta,
        target_indices=selection.indices,
        campaign_science=bank.science,
        output=root,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
        restart=restart,
        verbose_events=verbose_events,
        target_records=target_records,
    )
    replay = replay_reference(
        adapter=adapter,
        config=config,
        theta_bank=bank.theta,
        target_indices=selection.indices,
        campaign_science=bank.science,
        rows=reference_rows,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
        target_records=target_records,
    )
    replay_passed = all(item["equal"] for item in replay["results"])
    if not replay_passed and not uses_independent_target_ledger(config):
        raise RuntimeError("campaign replay audit failed")
    rows = list(reference_rows)
    sbi_executed_count = 0
    sbi_recovered_count = 0
    if campaign.sbi_methods:
        views = tuple(config.get("views", ("cold_start_per_target",)))
        regimes = tuple(config.get("regimes", ("equal_simulator_calls",)))
        unknown_views = set(views) - {"cold_start_per_target", "amortized_bank"}
        unknown_regimes = set(regimes) - {
            "equal_simulator_calls",
            "equal_wall_clock",
            "joint_cap",
        }
        if unknown_views:
            raise ValueError(f"unknown benchmark views: {sorted(unknown_views)}")
        if unknown_regimes:
            raise ValueError(f"unknown budget regimes: {sorted(unknown_regimes)}")
        reference_by_index = {row["target_index"]: row for row in reference_rows}
        target_responses = {
            index: torch.tensor(
                (
                    target_records[index]["target_response"]
                    if target_records is not None
                    else row["target_response"]
                ),
                device=runtime.torch_device,
                dtype=runtime.torch_dtype,
            ).contiguous()
            for index, row in reference_by_index.items()
        }
        rows = []
        for view in views:
            for regime in regimes:
                for reference in reference_rows:
                    copied = dict(reference)
                    copied["view"] = view
                    copied["regime"] = regime
                    rows.append(copied)
                for method in campaign.sbi_methods:
                    if view == "amortized_bank" and method in AMORTIZED_METHODS:
                        recovered_rows = [
                            _recover_method_record(
                                root,
                                view=view,
                                regime=regime,
                                method=method,
                                target_index=index,
                                campaign_id=bank.science.value,
                                configuration_hash=benchmark_config_hash,
                            )
                            for index in selection.indices
                        ] if restart else []
                        if recovered_rows and all(
                            row is not None for row in recovered_rows
                        ):
                            method_rows = [
                                row for row in recovered_rows if row is not None
                            ]
                            sbi_recovered_count += len(method_rows)
                        else:
                            method_rows = run_amortized_bank(
                                method=method,
                                adapter=adapter,
                                target_indices=selection.indices,
                                theta_bank=bank.theta,
                                target_responses=target_responses,
                                config=config,
                                campaign_science=bank.science,
                                device=runtime.torch_device,
                                dtype=runtime.torch_dtype,
                                artifact_output=(
                                    root / "method_logs" / view / regime / method
                                ),
                            )
                            sbi_executed_count += len(method_rows)
                        for row in method_rows:
                            row["view"] = view
                            row["regime"] = regime
                            row["benchmark_configuration_hash"] = benchmark_config_hash
                            row["amortization_classification"] = "shared_training"
                            rows.append(row)
                        continue
                    for position, index in enumerate(selection.indices):
                        print(
                            f"[{view} | {regime} | {method} | "
                            f"target {position + 1:03d}/{len(selection.indices):03d}] starting"
                        )
                        recovered = (
                            _recover_method_record(
                                root,
                                view=view,
                                regime=regime,
                                method=method,
                                target_index=index,
                                campaign_id=bank.science.value,
                                configuration_hash=benchmark_config_hash,
                            )
                            if restart
                            else None
                        )
                        if recovered is not None:
                            rows.append(recovered)
                            sbi_recovered_count += 1
                            continue
                        deadline = None
                        method_limit = runtime_limits.get("npe_method_deadline_seconds")
                        if method_limit is not None:
                            deadline = float(method_limit)
                        if regime in {"equal_wall_clock", "joint_cap"}:
                            deadline = float(
                                config.get(
                                    "diagnostic_wall_clock_seconds",
                                    reference_by_index[index]["runtime_seconds"],
                                )
                            )
                        remaining = remaining_global_seconds()
                        if remaining is not None:
                            deadline = remaining if deadline is None else min(deadline, remaining)
                        if deadline is not None and deadline <= 0:
                            row = {
                                "schema_version": "mutaco-sbi-target-v1",
                                "campaign_id": bank.science.value,
                                "target_family_id": bank.science.value,
                                "target_record_id": (
                                    str(target_records[index]["target_record_id"])
                                    if target_records is not None
                                    else None
                                ),
                                "target_index": index,
                                "method": method,
                                "status": "failed",
                                "failure_explanation": "BenchmarkDeadlineExceeded: no positive method time remains",
                                "failure_type": "timeout",
                                "failure_stage": "before_method_start",
                                "inference_calls": 0,
                                "assessment_calls": 0,
                                "gp_fits": 0,
                                "runtime_seconds": 0.0,
                            }
                            row["view"] = view
                            row["regime"] = regime
                            row["benchmark_configuration_hash"] = benchmark_config_hash
                            row["amortization_classification"] = "cold_start"
                            rows.append(row)
                            continue
                        if remaining is not None and remaining <= 0:
                            row = {
                                "schema_version": "mutaco-sbi-target-v1",
                                "campaign_id": bank.science.value,
                                "target_family_id": bank.science.value,
                                "target_record_id": (
                                    str(target_records[index]["target_record_id"])
                                    if target_records is not None
                                    else None
                                ),
                                "target_index": index,
                                "method": method,
                                "status": "failed",
                                "failure_explanation": "GlobalCampaignDeadlineExceeded",
                                "failure_type": "timeout",
                                "failure_stage": "before_method_start",
                                "inference_calls": 0,
                                "assessment_calls": 0,
                                "gp_fits": 0,
                                "runtime_seconds": 0.0,
                            }
                            row["view"] = view
                            row["regime"] = regime
                            row["benchmark_configuration_hash"] = benchmark_config_hash
                            row["amortization_classification"] = "cold_start"
                            rows.append(row)
                            continue
                        if (
                            view == "cold_start_per_target"
                            and config.get("process_policy")
                            == "fresh_method_process"
                        ):
                            row = run_sbi_target_process(
                                method=method,
                                simulator_adapter=adapter.identifier,
                                target_index=index,
                                target_response=reference_by_index[index][
                                    "target_response"
                                ],
                                target_record_id=(
                                    str(target_records[index]["target_record_id"])
                                    if target_records is not None
                                    else None
                                ),
                                config=config,
                                device=runtime.effective_device,
                                deadline_seconds=deadline,
                                hard_timeout_seconds=(
                                    deadline
                                    if remaining is None
                                    else min(float(deadline), remaining)
                                ),
                                output=(
                                    root / "processes" / view / regime
                                ),
                                repository_root=Path(__file__).resolve().parents[2],
                            )
                        else:
                            row = run_sbi_target(
                                method=method,
                                adapter=adapter,
                                index=index,
                                theta_star=bank.theta[index],
                                target_response=target_responses[index],
                                config=config,
                                campaign_science=bank.science,
                                device=runtime.torch_device,
                                dtype=runtime.torch_dtype,
                                deadline_seconds=deadline,
                                artifact_output=(
                                    root
                                    / "method_logs"
                                    / view
                                    / regime
                                    / method
                                    / f"target_{index:03d}"
                                ),
                                target_record_id=(
                                    str(target_records[index]["target_record_id"])
                                    if target_records is not None
                                    else None
                                ),
                            )
                        row.setdefault("campaign_id", bank.science.value)
                        row.setdefault("target_family_id", bank.science.value)
                        row.setdefault(
                            "target_record_id",
                            (
                                str(target_records[index]["target_record_id"])
                                if target_records is not None
                                else None
                            ),
                        )
                        row["view"] = view
                        row["regime"] = regime
                        row["benchmark_configuration_hash"] = benchmark_config_hash
                        row["amortization_classification"] = (
                            "target_specific_non_amortized"
                            if view == "amortized_bank"
                            else "cold_start"
                        )
                        rows.append(row)
                        if config.get("artifact_policy") == "compact_paper_records_v1":
                            process_root = (
                                root / "processes" / view / regime / method
                                / f"target_{index:03d}" / "training"
                            )
                            if row.get("training_diagnostics") is not None:
                                for event_file in process_root.glob("events.out.tfevents.*"):
                                    event_file.unlink()
                            used = sum(
                                path.stat().st_size
                                for path in root.rglob("*")
                                if path.is_file()
                            )
                            if used >= int(runtime_limits["artifact_cap_bytes"]):
                                raise RuntimeError(
                                    "ArtifactCapReached: campaign stopped before another method"
                                )
                        sbi_executed_count += 1
                        print(
                            f"[{method} | target {position + 1:03d}/"
                            f"{len(selection.indices):03d}] {row['status']} "
                            f"elapsed={row['runtime_seconds']:.1f}s"
                        )
    compact_paper = config.get("artifact_policy") == "compact_paper_records_v1"
    for row in rows:
        row.setdefault("benchmark_configuration_hash", benchmark_config_hash)
        if compact_paper and row.get("method") == "mutaco":
            # The hash-paired target result is the authoritative full MUTACO
            # record; do not store a second byte-for-byte semantic duplicate.
            continue
        record_path = (
            root
            / "records"
            / str(row.get("view", "reference_only"))
            / str(row.get("regime", "reference_only"))
            / str(row["method"])
            / f"target_{int(row['target_index']):03d}.json"
        )
        write_pretty_json(record_path, row)
        record_path.with_suffix(".sha256").write_text(
            file_hash(record_path), encoding="ascii"
        )
    summary = aggregate_rows(rows)
    public_rows = [_compact_public_row(row) for row in rows] if compact_paper else rows
    write_canonical_json(
        root / "evidence" / "rows.canonical.json", {"rows": public_rows}
    )
    write_pretty_json(root / "effective_configuration.json", config)
    write_pretty_json(root / "metrics.json", {"rows": public_rows})
    write_csv(root / "metrics.csv", public_rows, ROW_FIELDS)
    write_pretty_json(root / "summary.json", summary)
    write_pretty_json(root / "replay_evidence.json", replay)
    report = _markdown_report(campaign, runtime.provenance(), summary)
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    (root / "FIELDS.md").write_text(_field_dictionary(), encoding="utf-8")
    manifest = {
        "schema_version": "mutaco-benchmark-manifest-v1",
        "campaign_id": bank.science.value,
        "configuration": config,
        "configuration_path": campaign.source_path.as_posix(),
        "configuration_hash": stable_hash("cohort-config-v1", config),
        "benchmark_configuration_hash": benchmark_config_hash,
        "runtime": runtime.provenance(),
        "environment": _environment_provenance(campaign.sbi_methods),
        "simulator_adapter": adapter.identifier,
        "reference_method": "mutaco",
        "sbi_methods": campaign.sbi_methods,
        "target_selection": {
            "policy": selection.policy,
            "indices": selection.indices,
        },
        "target_ledger": target_ledger_manifest,
        "reference_replay_passed": replay_passed,
        "summary": summary,
        "accounting": {
            "target_generation_calls": (
                int(target_ledger_manifest["created_records"])
                if target_ledger_manifest is not None
                else sum(
                    row.get("target_construction_calls", 0)
                    for row in reference_rows
                )
            ),
            "inference_calls": (
                sum(row.get("inference_calls", 0) for row in reference_rows)
                + sum(
                    row.get("inference_calls", 0)
                    for row in rows
                    if row.get("method") != "mutaco"
                )
            ),
            "assessment_calls": (
                sum(row.get("assessment_calls", 0) for row in reference_rows)
                + sum(
                    row.get("assessment_calls", 0)
                    for row in rows
                    if row.get("method") != "mutaco"
                )
            ),
            "gp_fits": sum(row.get("gp_fits", 0) for row in reference_rows),
            "replay_indices": replay["indices"],
        },
        "total_runtime_seconds": perf_counter() - started,
        "execution_mode": "restart" if restart else "fresh",
        "cohort_execution": execution,
        "method_execution": {
            "sbi_executed_record_count": sbi_executed_count,
            "sbi_recovered_record_count": sbi_recovered_count,
        },
        "timing_events": tuple(
            event for row in rows for event in row.get("timing_events", ())
        ),
    }
    manifest["artifacts"] = {
        path.relative_to(root).as_posix(): {
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".tmp")
        and path.name not in {"manifest.json", "console.stdout.log", "console.stderr.log"}
    }
    artifact_bytes = sum(
        int(item["bytes"]) for item in manifest["artifacts"].values()
    )
    artifact_cap = runtime_limits.get("artifact_cap_bytes")
    manifest["resource_limits"] = {
        "global_deadline_seconds": global_deadline_seconds,
        "global_deadline_exceeded": (
            global_deadline_seconds is not None
            and perf_counter() - started > float(global_deadline_seconds)
        ),
        "artifact_cap_bytes": artifact_cap,
        "artifact_bytes_before_manifest": artifact_bytes,
        "artifact_cap_exceeded": (
            artifact_cap is not None and artifact_bytes > int(artifact_cap)
        ),
        "resume_policy": runtime_limits.get("resume_policy"),
    }
    write_pretty_json(root / "manifest.json", manifest)
    return CampaignExecution(root, manifest, tuple(rows))


def _markdown_report(
    campaign: Campaign, runtime: Mapping[str, Any], summary: Mapping[str, Any]
) -> str:
    lines = [
        "# Benchmark campaign report",
        "",
        f"- Configuration: `{campaign.source_path.as_posix()}`",
        f"- Simulator adapter: `{campaign.simulator_identifier}`",
        f"- Device requested/effective: `{campaign.requested_device}` / `{runtime['effective_device']}`",
        f"- CPU fallback reason: `{runtime.get('fallback_reason')}`",
        f"- SBI methods: `{list(campaign.sbi_methods)}`",
        "",
        "## Summary",
        "",
        "The complete numerical records are in `metrics.csv` and `metrics.json`.",
    ]
    for group in summary["groups"]:
        lines.extend(
            (
                "",
                f"- Group: `{group['group']}`",
                f"- Completed: {group['completed_count']}/{group['target_count']}",
                f"- Failed: {group['failed_count']}",
            )
        )
    return "\n".join(lines) + "\n"


def _field_dictionary() -> str:
    return """# Persisted result fields

- `target_index`: stable source target-bank index.
- `method`: registered method identifier.
- `status`: typed terminal method status.
- `target_error_l1`, `target_error_l2`: held-out physical-response errors.
- `design_error_l2`: parameter-space error when hidden truth is available.
- `latent_error_l1`, `latent_error_l2`: optional adapter-provided latent errors.
- `runtime_seconds`: recorded per-target method wall time.
- `failure_explanation`: retained typed failure detail.
"""


def _environment_provenance(sbi_methods: Sequence[str]) -> dict[str, Any]:
    packages = (
        "torch",
        "botorch",
        "gpytorch",
        "numpy",
        "scipy",
        "matplotlib",
        "sbi",
        "nflows",
        "zuko",
        "skorch",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "packages": versions,
        "sbi_required": bool(sbi_methods),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }


def _recover_method_record(
    root: Path,
    *,
    view: str,
    regime: str,
    method: str,
    target_index: int,
    campaign_id: str,
    configuration_hash: str,
) -> dict[str, Any] | None:
    path = (
        root
        / "records"
        / view
        / regime
        / method
        / f"target_{target_index:03d}.json"
    )
    sidecar = path.with_suffix(".sha256")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"incomplete benchmark method artifact: {path}")
    if file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise ValueError(f"benchmark method artifact hash mismatch: {path}")
    row = read_json(path)
    if row.get("campaign_id") != campaign_id:
        raise ValueError(f"benchmark method campaign mismatch: {path}")
    if row.get("benchmark_configuration_hash") != configuration_hash:
        raise ValueError(f"benchmark method configuration mismatch: {path}")
    return row
