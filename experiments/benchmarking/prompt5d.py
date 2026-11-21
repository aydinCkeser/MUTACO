"""Prompt 5D master-target preparation and user-launched campaign orchestration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import json
import random
import shutil
import subprocess

from mutaco import (
    RuntimeRequest, ScientificIdentity, SeedCoordinate, SeedDeriver,
    SimulationContext, checked_evaluate, resolve_runtime,
)

from .api import ROW_FIELDS, load_campaign
from .artifacts import file_hash, read_json, write_csv, write_pretty_json
from .calibration import CalibrationPolicy, _run_method, validate_calibration_environment
from .simulator_adapters import get_simulator_adapter
from .methods.supervisor import run_bounded_command
from .statistics import aggregate_rows
from .plotting import generate_distribution_plots
import sys


METHODS = ("mutaco", "sbi_npe", "sbi_snpe")
PROMPT5D_ROW_FIELDS = ROW_FIELDS + (
    "regime",
    "budget",
    "source_regime",
    "master_target_identity",
    "target_construction_calls",
    "inference_calls",
    "assessment_calls",
    "gp_fits",
    "posterior_samples",
    "derived_zero_call_view",
)
COHORTS = {
    "extreme_few_shot_cold_start_14": (tuple(range(1, 25)), 14),
    "cold_start_equal_calls_40": (tuple(range(1, 13)), 40),
    "cold_start_equal_calls_105": (tuple(range(1, 7)), 105),
}


def _repository_provenance(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "working_tree_clean": not status,
        "status_lines": status,
    }


def validate_prompt5d_document(document: Mapping[str, Any]) -> None:
    if document.get("campaign_classification") != "prompt5d_scientific_campaign":
        raise ValueError("invalid Prompt 5D classification")
    if document.get("dtype") != "float32":
        raise ValueError("Prompt 5D requires float32")
    if tuple(document.get("sbi_methods", ())) != ("sbi_npe", "sbi_snpe"):
        raise ValueError("Prompt 5D methods must be NPE and SNPE")
    p5d = document.get("prompt5d", {})
    if p5d.get("calibration_exclusions") != [0, 49, 99]:
        raise ValueError("Prompt 5D calibration exclusions drifted")
    if p5d.get("global_deadline_seconds") != 10800.0:
        raise ValueError("Prompt 5D global deadline must be three hours")
    if document["sbi"].get("posterior_samples") != 10_000:
        raise ValueError("Prompt 5D requires 10,000 posterior samples")
    if "mcmc" in document["sbi"]:
        raise ValueError("Prompt 5D direct-posterior configuration cannot contain MCMC")
    if document.get("comparison_set") != {
        "policy": "explicit_indices", "indices": list(range(1, 25))
    }:
        raise ValueError("Prompt 5D master cohort must be ordered indices 1..24")
    if document.get("views") != ["cold_start_per_target", "amortized_bank"]:
        raise ValueError("Prompt 5D view declaration drifted")
    if document.get("regimes") != ["equal_simulator_calls", "equal_wall_clock"]:
        raise ValueError("Prompt 5D regime declaration drifted")
    if document.get("process_policy") != "fresh_method_process":
        raise ValueError("Prompt 5D requires fresh method processes")
    expected_prompt5d = {
        "calibration_exclusions": [0, 49, 99],
        "cohorts": {
            "final_24": [1, 24],
            "nested_12": [1, 12],
            "nested_6": [1, 6],
        },
        "budgets": [14, 40, 105],
        "sequential_allocations": {
            "14": [5, 5, 4],
            "40": [14, 13, 13],
            "105": [35, 35, 35],
        },
        "method_deadline_seconds": 180.0,
        "amortized_npe_deadline_seconds": 1800.0,
        "global_deadline_seconds": 10800.0,
        "artifact_cap_bytes": 500000000,
        "master_target_policy": "construct_once_reuse_immutable",
    }
    if p5d != expected_prompt5d:
        raise ValueError("Prompt 5D execution and reuse declaration drifted")
    expected_sbi = {
        "density_estimator": "maf",
        "training_batch_size": 64,
        "learning_rate": 0.0005,
        "validation_fraction": 0.2,
        "stop_after_epochs": 20,
        "max_num_epochs": 200,
        "posterior_samples": 10000,
        "show_progress_bars": False,
        "posterior_sampling": "direct",
        "fallback_policy": "none",
    }
    if document["sbi"] != expected_sbi:
        raise ValueError("Prompt 5D SBI declaration drifted")
    runner = document["runner"]
    if "duplicate_policy" in runner:
        if runner["duplicate_policy"] != "best_ranked_unevaluated_raw_v1":
            raise ValueError("Prompt 5D corrected duplicate policy drifted")
        if (
            runner["gp"].get("jitter") != 1e-6
            or
            runner["gp"].get("symmetry_tolerance") != 1e-6
            or runner["gp"].get("psd_tolerance") != 1e-6
        ):
            raise ValueError("Prompt 5D corrected float32 GP controls drifted")


def _master_identity(config: Mapping[str, Any]) -> tuple[ScientificIdentity, dict[str, Any]]:
    document = {
        "schema": "mutaco-prompt5d-master-targets-v1",
        "campaign_seed": config["campaign"]["campaign_seed"],
        "target_bank_policy": config["campaign"]["target_bank_policy"],
        "simulator": config["simulator"],
        "dtype": config["dtype"],
        "indices": list(range(1, 25)),
    }
    return ScientificIdentity.create(document), document


def prepare_master_targets(
    config_path: str | Path, *, output: str | Path, device: str | None = None,
    master_target_source: str | Path | None = None,
) -> dict[str, Any]:
    """Construct each noisy hidden target once and persist immutable records."""

    campaign = load_campaign(config_path, device=device)
    config = campaign.document
    validate_prompt5d_document(config)
    validate_calibration_environment()
    runtime = resolve_runtime(RuntimeRequest(
        device=campaign.requested_device, dtype=config["dtype"]
    ))
    adapter = get_simulator_adapter(campaign.simulator_identifier)
    bank = adapter.target_bank(config, device=runtime.torch_device, dtype=runtime.torch_dtype)
    science, science_document = _master_identity(config)
    deriver = SeedDeriver(science, int(config["campaign"]["campaign_seed"]))
    simulator = adapter.build_simulator(
        config, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    root = Path(output).resolve()
    target_root = root / "master_targets"
    target_root.mkdir(parents=True, exist_ok=True)
    reused_source = (
        Path(master_target_source).resolve().as_posix()
        if master_target_source is not None else None
    )
    if master_target_source is not None:
        source_root = Path(master_target_source).resolve()
        if source_root == target_root:
            raise ValueError("master-target source and destination must differ")
        source_manifest_path = source_root / "manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("master-target source manifest is missing")
        source_manifest = read_json(source_manifest_path)
        if (
            source_manifest.get("identity") != science.value
            or source_manifest.get("identity_document") != science_document
            or source_manifest.get("indices") != list(range(1, 25))
        ):
            raise ValueError("master-target source identity mismatch")
        copy_source = not any(target_root.iterdir())
        source_records = source_manifest.get("records", {})
        for filename, expected_hash in source_records.items():
            source_record = source_root / filename
            source_sidecar = source_record.with_suffix(".sha256")
            if (
                not source_record.is_file()
                or not source_sidecar.is_file()
                or file_hash(source_record) != expected_hash
                or source_sidecar.read_text(encoding="ascii").strip() != expected_hash
            ):
                raise ValueError(f"invalid source master-target record: {filename}")
            if copy_source:
                shutil.copy2(source_record, target_root / filename)
                shutil.copy2(
                    source_sidecar, (target_root / filename).with_suffix(".sha256")
                )
        if copy_source:
            shutil.copy2(source_manifest_path, target_root / "manifest.json")
    manifest_path = target_root / "manifest.json"
    existing_manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if existing_manifest is not None and (
        existing_manifest.get("identity") != science.value
        or existing_manifest.get("identity_document") != science_document
        or existing_manifest.get("indices") != list(range(1, 25))
    ):
        raise ValueError("master-target manifest identity mismatch")
    records = []
    created_now = 0
    for index in range(1, 25):
        seed = deriver.seed_for(SeedCoordinate("target_response_noise", (index,)))
        path = target_root / f"target_{index:03d}.json"
        expected_prefix = {
            "schema_version": "mutaco-prompt5d-master-target-v1",
            "master_target_identity": science.value,
            "target_index": index,
            "target_seed": seed,
            "theta_star": bank.theta[index].detach().cpu().tolist(),
        }
        if path.is_file():
            sidecar = path.with_suffix(".sha256")
            if not sidecar.is_file() or file_hash(path) != sidecar.read_text(
                encoding="ascii"
            ).strip():
                raise ValueError(f"master target hash mismatch at index {index}")
            record = read_json(path)
            if any(record.get(k) != v for k, v in expected_prefix.items()):
                raise ValueError(f"master target mismatch at index {index}")
        else:
            if existing_manifest is not None:
                raise ValueError(
                    f"immutable master-target ledger is missing index {index}"
                )
            response = checked_evaluate(
                simulator, bank.theta[index], device=runtime.torch_device,
                dtype=runtime.torch_dtype,
                context=SimulationContext(science.value, "target", index, 0, seed),
            ).values
            record = {
                **expected_prefix,
                "target_response": response.detach().cpu().tolist(),
                "target_construction_calls": 1,
            }
            write_pretty_json(path, record)
            path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")
            created_now += 1
        records.append(record)
    manifest = {
        "schema_version": "mutaco-prompt5d-master-target-manifest-v1",
        "identity": science.value,
        "identity_document": science_document,
        "indices": list(range(1, 25)),
        "target_construction_calls": 24,
        "records": {f"target_{r['target_index']:03d}.json": file_hash(
            target_root / f"target_{r['target_index']:03d}.json"
        ) for r in records},
    }
    if existing_manifest is None:
        write_pretty_json(manifest_path, manifest)
    elif existing_manifest != manifest:
        raise ValueError("master-target manifest content mismatch")
    return {
        "manifest": manifest,
        "records": records,
        "science_document": science_document,
        "created_now": created_now,
        "recovered_now": 24 - created_now,
        "runtime": runtime.provenance(),
        "reused_source": reused_source,
    }


def _bootstrap_paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create deterministic paired mean-difference intervals from persisted rows."""

    metrics = ("target_error_l1", "target_error_l2", "design_error_l2")
    completed = [row for row in rows if row.get("status") == "completed"]
    groups: list[dict[str, Any]] = []
    for regime in sorted({str(row.get("regime")) for row in completed}):
        regime_rows = [row for row in completed if row.get("regime") == regime]
        methods = sorted({str(row.get("method")) for row in regime_rows})
        indexed = {
            (str(row["method"]), int(row["target_index"])): row
            for row in regime_rows
        }
        for left_position, left in enumerate(methods):
            for right in methods[left_position + 1:]:
                common = sorted(
                    {
                        int(row["target_index"])
                        for row in regime_rows
                        if row.get("method") == left
                    }
                    & {
                        int(row["target_index"])
                        for row in regime_rows
                        if row.get("method") == right
                    }
                )
                for metric in metrics:
                    differences = [
                        float(indexed[(left, index)][metric])
                        - float(indexed[(right, index)][metric])
                        for index in common
                        if indexed[(left, index)].get(metric) is not None
                        and indexed[(right, index)].get(metric) is not None
                    ]
                    if not differences:
                        continue
                    observed = sum(differences) / len(differences)
                    generator = random.Random(
                        f"prompt5d-bootstrap-v1:{regime}:{left}:{right}:{metric}"
                    )
                    replicates = sorted(
                        sum(
                            differences[generator.randrange(len(differences))]
                            for _ in differences
                        ) / len(differences)
                        for _ in range(2_000)
                    )
                    groups.append({
                        "regime": regime,
                        "left_method": left,
                        "right_method": right,
                        "metric": metric,
                        "difference_definition": "left_minus_right",
                        "paired_target_indices": common,
                        "valid_pair_count": len(differences),
                        "mean_difference": observed,
                        "bootstrap_replicates": 2_000,
                        "bootstrap_seed_policy": "coordinate_addressed_python_random_v1",
                        "confidence_level": 0.95,
                        "ci_low": replicates[49],
                        "ci_high": replicates[1949],
                    })
    return {
        "schema_version": "mutaco-prompt5d-paired-summary-v1",
        "groups": groups,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _regime_config(base: Mapping[str, Any], name: str, budget: int) -> dict[str, Any]:
    config = deepcopy(dict(base))
    indices, _ = COHORTS[name]
    config["comparison_set"] = {"policy": "explicit_indices", "indices": list(indices)}
    config["simulation_budget_per_target"] = budget
    config["runner"]["sequential_rounds"] = budget - int(config["runner"]["initial_points"])
    return config


def run_prompt5d(
    config_path: str | Path, *, output: str | Path, device: str | None = None,
    smoke: bool = False, restart: bool = True,
    master_target_source: str | Path | None = None,
) -> dict[str, Any]:
    """Run the prepared campaign; ``smoke`` is mechanics-only and separately labelled."""

    started = perf_counter()
    root = Path(output).resolve()
    campaign = load_campaign(config_path, device=device)
    validate_prompt5d_document(campaign.document)
    master = prepare_master_targets(
        config_path, output=root, device=device,
        master_target_source=master_target_source,
    )
    runtime = resolve_runtime(RuntimeRequest(
        device=campaign.requested_device, dtype=campaign.document["dtype"]
    ))
    policy = CalibrationPolicy(
        target_indices=(1,), method_orders={1: METHODS},
        global_deadline_seconds=60.0 if smoke else 10800.0,
        target_deadline_seconds=60.0 if smoke else 180.0,
        poll_interval_seconds=.1, progress_interval_seconds=10.0,
        termination_grace_seconds=2.0, projection_target_counts=(),
        projection_concurrency_levels=(),
    )
    repository_root = Path(__file__).resolve().parents[2]
    repository = _repository_provenance(repository_root)
    if not smoke and not repository["working_tree_clean"]:
        raise RuntimeError(
            "Prompt 5D scientific campaign requires a clean committed repository"
        )
    rows = []
    regimes = (("extreme_few_shot_cold_start_14", (1,), 14),) if smoke else (
        (name, indices, budget) for name, (indices, budget) in COHORTS.items()
    )
    global_deadline = started + policy.global_deadline_seconds
    by_index = {r["target_index"]: r for r in master["records"]}
    for regime, indices, budget in regimes:
        config = _regime_config(campaign.document, regime, budget)
        regime_science_document = {
            "schema": "mutaco-prompt5d-regime-v1",
            "master_target_identity": master["manifest"]["identity"],
            "regime": regime, "budget": budget, "configuration": config,
        }
        for position, index in enumerate(indices):
            order = METHODS[position % 3:] + METHODS[:position % 3]
            for method in order:
                record_path = (
                    root / "records" / regime / method / f"target_{index:03d}.json"
                )
                sidecar = record_path.with_suffix(".sha256")
                if restart and record_path.is_file() and sidecar.is_file():
                    if file_hash(record_path) != sidecar.read_text(encoding="ascii").strip():
                        raise ValueError(f"record hash mismatch: {record_path}")
                    recovered = read_json(record_path)
                    if (
                        recovered.get("master_target_identity")
                        != master["manifest"]["identity"]
                        or recovered.get("regime") != regime
                        or recovered.get("budget") != budget
                    ):
                        raise ValueError(f"recovered record identity mismatch: {record_path}")
                    rows.append(recovered)
                    print(f"[Prompt 5D | {regime} | {method} | target {index}] recovered")
                    continue
                remaining = min(180.0, global_deadline - perf_counter())
                if remaining <= 0:
                    raise TimeoutError("Prompt 5D global deadline exhausted")
                record = by_index[index]
                row = _run_method(
                    root=root / regime, repository_root=repository_root, config=config,
                    campaign_id=ScientificIdentity.create(regime_science_document).value,
                    target_index=index, method=method,
                    target_response=record["target_response"],
                    effective_device=runtime.effective_device,
                    allowed_seconds=remaining, deadline_scope="method", policy=policy,
                    campaign_science_document=regime_science_document,
                    target_seed=record["target_seed"], progress_label="Prompt 5D smoke" if smoke else "Prompt 5D",
                )
                row["regime"] = regime
                row["budget"] = budget
                row["master_target_identity"] = master["manifest"]["identity"]
                rows.append(row)
                write_pretty_json(record_path, row)
                sidecar.write_text(file_hash(record_path), encoding="ascii")
    amortized_status = "excluded_from_smoke"
    if not smoke:
        amortized_root = root / "amortized_shared_bank_336"
        request_path = amortized_root / "request.json"
        result_path = amortized_root / "result.json"
        amortized_science = {
            "schema": "mutaco-prompt5d-amortized-v1",
            "master_target_identity": master["manifest"]["identity"],
            "method": "sbi_npe", "budget": 336,
        }
        amortized_config = _regime_config(
            campaign.document, "extreme_few_shot_cold_start_14", 14
        )
        write_pretty_json(request_path, {
            "configuration": amortized_config,
            "device": runtime.effective_device,
            "simulator_adapter": campaign.simulator_identifier,
            "campaign_science_document": amortized_science,
            "target_records": master["records"],
            "artifact_output": str(amortized_root / "training"),
        })
        evidence = run_bounded_command(
            [sys.executable, "-m",
             "experiments.benchmarking.methods.prompt5d_amortized_worker",
             "--input", str(request_path), "--output", str(result_path)],
            cwd=repository_root, stdout_path=amortized_root / "stdout.log",
            stderr_path=amortized_root / "stderr.log",
            deadline_seconds=min(1800.0, global_deadline - perf_counter()),
            poll_interval_seconds=.1, termination_grace_seconds=2.0,
            progress_interval_seconds=10.0,
            progress_callback=lambda event: print(
                f"[Prompt 5D | amortized NPE] elapsed={event['elapsed_seconds']:.1f}s "
                f"remaining={event['remaining_seconds']:.1f}s"
            ),
        )
        if evidence["timed_out"] or not result_path.is_file():
            amortized_status = "timeout" if evidence["timed_out"] else "failed"
        else:
            amortized_rows = read_json(result_path)["rows"]
            for row in amortized_rows:
                row["regime"] = "amortized_shared_bank_336"
                row["budget"] = 336
                row["master_target_identity"] = master["manifest"]["identity"]
            rows.extend(amortized_rows)
            amortized_status = "completed"
    if not smoke:
        source = list(rows)
        for row in source:
            if (
                row.get("regime") == "cold_start_equal_calls_40"
                and row.get("method") == "mutaco"
                and row["target_index"] <= 6
            ) or (
                row.get("regime") == "cold_start_equal_calls_105"
                and row.get("method") in {"sbi_npe", "sbi_snpe"}
            ):
                derived = deepcopy(row)
                derived["source_regime"] = row["regime"]
                derived["regime"] = "equal_wall_clock_hard_cap_180"
                derived["inference_calls"] = 0
                derived["assessment_calls"] = 0
                derived["derived_zero_call_view"] = True
                rows.append(derived)
        for row in source:
            if (
                row.get("regime") == "extreme_few_shot_cold_start_14"
                and row.get("method") == "mutaco"
            ):
                derived = deepcopy(row)
                derived["source_regime"] = row["regime"]
                derived["regime"] = "amortized_shared_bank_336"
                derived["inference_calls"] = 0
                derived["assessment_calls"] = 0
                derived["derived_zero_call_view"] = True
                rows.append(derived)
    summary = aggregate_rows(rows, group_by=("regime", "method"))
    paired_summary = _bootstrap_paired_summary(rows)
    write_pretty_json(root / "metrics.json", {"rows": rows})
    write_csv(root / "metrics.csv", rows, PROMPT5D_ROW_FIELDS)
    write_pretty_json(root / "summary.json", summary)
    write_pretty_json(root / "paired_summary.json", paired_summary)
    _write_jsonl(root / "training_histories.jsonl", [
        {
            "regime": row.get("regime"),
            "method": row.get("method"),
            "target_index": row.get("target_index"),
            "training_history": row.get("training_history"),
            "training_diagnostics": row.get("training_diagnostics"),
        }
        for row in rows
        if row.get("training_history") is not None
        or row.get("training_diagnostics") is not None
    ])
    write_pretty_json(root / "posterior_diagnostics.json", {
        "records": [
            {
                "regime": row.get("regime"),
                "method": row.get("method"),
                "target_index": row.get("target_index"),
                "posterior_diagnostics": row["posterior_diagnostics"],
            }
            for row in rows if row.get("posterior_diagnostics") is not None
        ]
    })
    write_pretty_json(root / "seed_ledger.json", {
        "master_target_identity": master["manifest"]["identity"],
        "records": [
            {
                "regime": row.get("regime"),
                "method": row.get("method"),
                "target_index": row.get("target_index"),
                "target_seed": row.get("target_seed"),
                "assessment_seed": row.get("assessment_seed"),
                "recommendation_seed": row.get("recommendation_seed"),
                "seed_ledger": row.get("seed_ledger", []),
            }
            for row in rows if not row.get("derived_zero_call_view")
        ],
    })
    write_pretty_json(root / "runtime_and_device.json", {
        "runtime": master["runtime"],
        "master_targets_created_this_invocation": master["created_now"],
        "master_targets_recovered_this_invocation": master["recovered_now"],
        "master_target_source": master["reused_source"],
        "elapsed_seconds_at_export": perf_counter() - started,
        "global_deadline_seconds": policy.global_deadline_seconds,
        "method_deadline_seconds": policy.target_deadline_seconds,
    })
    plot_inventory = []
    if not smoke:
        style = repository_root / "experiments/analytic_target_matching/noisy_cohort/styles/noisy_cohort_default.mplstyle"
        for group in summary["groups"]:
            selection = group["group"]
            selected = [
                row for row in rows
                if row.get("regime") == selection["regime"]
                and row.get("method") == selection["method"]
            ]
            one = aggregate_rows(selected, group_by=())["groups"][0]
            plot_inventory.append(generate_distribution_plots(
                summary=one, rows=selected,
                output=root / "plots" / str(selection["regime"]) / str(selection["method"]),
                style=style,
            ))
    result = {
        "schema_version": "mutaco-prompt5d-execution-v1",
        "classification": "bounded_no_claim_smoke" if smoke else "scientific_campaign",
        "master_target_identity": master["manifest"]["identity"],
        "rows": rows, "elapsed_seconds": perf_counter() - started,
        "amortized_status": amortized_status,
        "equal_time_status": "derived_after_source_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "restart_enabled": restart,
        "repository": repository,
        "summary": summary,
        "paired_summary": paired_summary,
        "plots": plot_inventory,
    }
    write_pretty_json(root / ("SMOKE_RESULT.json" if smoke else "campaign_result.json"), result)
    if smoke and any(row.get("status") != "completed" for row in rows):
        failed = [
            f"{row.get('method')}:{row.get('failure_explanation')}"
            for row in rows if row.get("status") != "completed"
        ]
        raise RuntimeError(f"Prompt 5D no-claim smoke failed: {failed}")
    if not smoke:
        (root / "REPORT.md").write_text(
            "# Prompt 5D analytic campaign\n\n"
            "Complete per-target numerical data are in `metrics.csv` and "
            "`metrics.json`; aggregate statistics are in `summary.json`.\n\n"
            f"- Master target identity: `{master['manifest']['identity']}`\n"
            f"- Amortized NPE status: `{amortized_status}`\n"
            f"- Runtime: `{result['elapsed_seconds']:.3f}` seconds\n"
            "- Equal-call, derived equal-time, and amortized rows remain "
            "separately labelled.\n"
            "- Paired differences and deterministic bootstrap intervals are "
            "in `paired_summary.json`.\n"
            "- Training histories, posterior diagnostics, seed provenance, "
            "and runtime/device evidence are exported separately.\n",
            encoding="utf-8",
        )
        artifacts = {
            path.relative_to(root).as_posix(): {
                "sha256": file_hash(path), "bytes": path.stat().st_size,
            }
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        stored = sum(item["bytes"] for item in artifacts.values())
        if stored > int(campaign.document["prompt5d"]["artifact_cap_bytes"]):
            raise RuntimeError("Prompt 5D artifact cap exceeded")
        manifest = {
            "schema_version": "mutaco-prompt5d-manifest-v1",
            "classification": "scientific_campaign",
            "configuration": campaign.document,
            "master_target_identity": master["manifest"]["identity"],
            "master_target_source": master["reused_source"],
            "runtime": master["runtime"],
            "repository": repository,
            "accounting": {
                "target_construction_calls": 24,
                "target_construction_calls_this_invocation": master["created_now"],
                "inference_calls": sum(
                    int(row.get("inference_calls", 0)) for row in rows
                    if not row.get("derived_zero_call_view")
                ),
                "assessment_calls": sum(
                    int(row.get("assessment_calls", 0)) for row in rows
                    if not row.get("derived_zero_call_view")
                ),
            },
            "summary": summary, "artifacts": artifacts,
            "stored_artifact_bytes": stored,
        }
        write_pretty_json(root / "manifest.json", manifest)
    return result
