"""Deterministic zero-call audit for scalar-primary B100 population preparation."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.graph_inversion.prepare_scalar_primary_population import materialize_scalar_population
from experiments.paper.common.cross_experiment_contract import validate_primary_budget_pair
from experiments.paper.common.evidence_analysis import analysis_specification, canonical_json_text
from experiments.paper.protocol.validation import PaperContractError
from experiments.paper.primary_benchmark import load_scalar_primary_population_declaration, primary_worker_preview


OUTPUT = "results/paper/graph_scalar_primary_b100_population_preparation_v3r2"
DECLARATION = "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v3r2.json"
PLAN = "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_plan_v3r2.json"
PROFILE = "infrastructure/truba/scalar_primary_population_b100_v3r2.env"
COHORT = "data/cohorts/paper120_v3"
STUDY = "experiments/paper/studies/graph_paper120/studies/mutaco_scalar_qlognei_sobol_primary_v1.json"
CONFIG = "experiments/graph_inversion/configs/paper/paper120_b100_mutaco_v1.json"
SETTINGS = "experiments/graph_inversion/configs/paper/scalar_qlognei_primary_settings_v1.json"
B40_DECLARATION = "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v3r2.json"
B40_CONFIG = "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json"
OUTPUT_ROOT = "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v3r2/workers"


SOURCE_RECEIPT = (
    "experiments/paper/exploration/outputs/scalar_primary_smoke/"
    "population_b100_v3r2_source_validation_receipt.json"
)
SOURCE_RECEIPT_SHA256 = "91213dba23c23d12c9a1e9fd8f96b29f13aa3302fd88cb9c1173fb26f3960a50"


def verified_historical_profile(root: Path, current: bytes) -> bytes:
    """Read the profile used by the B100 workers and verify the sole scheduler relocation."""
    receipt_path = root / SOURCE_RECEIPT
    if file_hash(receipt_path) != SOURCE_RECEIPT_SHA256:
        raise PaperContractError("B100 source validation receipt SHA-256 differs")
    receipt = read_json(receipt_path)
    if receipt["profile"] != PROFILE:
        raise PaperContractError("B100 source validation receipt references a different profile")
    try:
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{receipt['source_commit']}:{PROFILE}"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise PaperContractError("B100 original scheduler profile requires the recorded Git commit") from exc
    if sha256(original).hexdigest() != receipt["profile_sha256"]:
        raise PaperContractError("B100 original scheduler profile SHA-256 differs from its receipt")
    old = b'export MUTACO_SCALAR_PRIMARY_TARGET_LEDGER="data/cohorts/paper120_v3"\n'
    new = b'export MUTACO_SCALAR_PRIMARY_TARGET_LEDGER="experiments/graph_inversion/frozen_targets/paper120_v3"\n'
    if original.count(old) != 1 or current != original.replace(old, new):
        raise PaperContractError("B100 current scheduler profile differs beyond the approved cohort path")
    return original


def _content(root: Path, *, historical: bool) -> dict[str, str]:
    declaration_path = root / DECLARATION
    plan_path = root / PLAN
    profile_path = root / PROFILE
    cohort_path = resolve_recorded_path(root, COHORT) / "manifest.json"
    declaration = read_json(declaration_path)
    plan = read_json(plan_path)
    expected_declaration, expected_plan = materialize_scalar_population(
        cohort_root=resolve_recorded_path(root, COHORT),
        study=root / STUDY,
        configuration=root / CONFIG,
        settings=root / SETTINGS,
        output_root=OUTPUT_ROOT,
        budget=100,
        execution_revision=4,
    )
    if declaration != expected_declaration or plan != expected_plan:
        raise PaperContractError("B100 declaration or plan differs from deterministic materialization")
    for path in (declaration_path, plan_path):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(path):
            raise PaperContractError("B100 declaration hash pair is invalid")
    loaded = load_scalar_primary_population_declaration(declaration)
    previews = [primary_worker_preview(loaded, rank=rank, workers=120, maximum_calls=303) for rank in range(120)]
    if (
        [preview["target_positions"] for preview in previews] != [[rank] for rank in range(120)]
        or any(preview["planned_calls"] != 303 for preview in previews)
        or any(preview["scientific_calls_executed"] != 0 for preview in previews)
    ):
        raise PaperContractError("B100 logical worker coverage or call cap is invalid")
    contract_audit = validate_primary_budget_pair(
        (
            load_scalar_primary_population_declaration(read_json(root / B40_DECLARATION)),
            loaded,
        ),
        cohort_manifest=read_json(cohort_path),
        cohort_manifest_sha256=file_hash(cohort_path),
        simulator_configurations={40: read_json(root / B40_CONFIG), 100: read_json(root / CONFIG)},
        simulator_configuration_sha256={40: file_hash(root / B40_CONFIG), 100: file_hash(root / CONFIG)},
    )
    profile_bytes = profile_path.read_bytes()
    if historical:
        profile_bytes = verified_historical_profile(root, profile_bytes)
    profile = profile_bytes.decode("utf-8")
    required_profile_fragments = (
        'MUTACO_SCALAR_PRIMARY_ARRAY_TASKS="0-59"',
        "MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY=60",
        "MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS=60",
        "MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK=2",
        "MUTACO_SCALAR_PRIMARY_WORKERS=120",
        "MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER=303",
        "MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS=36360",
        "MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK=20",
        'MUTACO_SCALAR_PRIMARY_BEGIN_DELAY="now+10minutes"',
        "MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED=0",
    )
    if any(fragment not in profile for fragment in required_profile_fragments):
        raise PaperContractError("B100 TRUBA profile differs from frozen 60-node geometry")
    scheduler = plan["scheduler"]
    if (
        scheduler.get("physical_task_count") != 60
        or scheduler.get("logical_workers_per_task") != 2
        or scheduler.get("logical_rank_mapping") != ["task_id", "task_id+60"]
        or scheduler.get("submitted_job_count") != 61
        or scheduler.get("transactional_delayed_start") is not True
        or scheduler.get("rollback_before_begin") is not True
    ):
        raise PaperContractError("B100 scheduler transaction is invalid")
    analysis_spec = analysis_specification(
        analysis_id="scalar-primary-b100-v1",
        bootstrap_seed=92024100,
        bootstrap_resamples=10_000,
    )
    body: dict[str, Any] = {
        "schema_version": "mutaco-scalar-primary-b100-preparation-audit-v1",
        "status": "complete_zero_call_unsubmitted",
        "study_identity": declaration["study_identity"],
        "declaration": DECLARATION,
        "declaration_sha256": file_hash(declaration_path),
        "declaration_identity": declaration["declaration_identity"],
        "plan": PLAN,
        "plan_sha256": file_hash(plan_path),
        "plan_identity": plan["plan_identity"],
        "profile": PROFILE,
        "profile_sha256": sha256(profile_bytes).hexdigest(),
        "cohort_manifest": f"{COHORT}/manifest.json",
        "cohort_manifest_sha256": file_hash(cohort_path),
        "cross_experiment_contract_audit": contract_audit,
        "methods": ["mutaco", "scalar_qlognei", "sobol"],
        "budget": 100,
        "logical_workers": 120,
        "physical_array_tasks": 60,
        "logical_workers_per_task": 2,
        "maximum_calls_per_logical_worker": 303,
        "maximum_search_calls": 36_000,
        "maximum_paired_assessment_calls": 360,
        "maximum_total_calls": 36_360,
        "scheduler_policy": {
            "layout": "60_exclusive_nodes_two_sequential_logical_workers_each",
            "array_indices": "0-59",
            "array_concurrency": 60,
            "cpus_per_task": 20,
            "worker_walltime": "04:00:00",
            "transaction": "delayed_start_then_afterany_collector_with_partial_submit_rollback",
            "automatic_retry": False,
            "slurm_requeue": False,
        },
        "timing_fields": ["simulator", "gp_fit", "acquisition", "terminal", "other", "total"],
        "collection": {
            "collector": "infrastructure/truba/collect_primary_benchmark.py",
            "collector_sha256": file_hash(root / "infrastructure/truba/collect_primary_benchmark.py"),
            "append_only_worker_evidence": True,
            "collector_dependency": "afterany_array",
            "compact_export_schema": "mutaco-scalar-primary-compact-export-v1",
        },
        "analysis": {
            "adapter": "adapt_scalar_primary_export",
            "adapter_source": "experiments/paper/common/evidence_analysis.py",
            "adapter_source_sha256": file_hash(root / "experiments/paper/common/evidence_analysis.py"),
            "analysis_spec_identity": analysis_spec["analysis_spec_identity"],
            "primary_metric": analysis_spec["primary_metric"],
            "bootstrap_seed": analysis_spec["bootstrap"]["seed"],
            "bootstrap_resamples": analysis_spec["bootstrap"]["resamples"],
        },
        "supersedes_scheduler_only": {
            "declaration": "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v2.json",
            "reason": "two-array 100-plus-20 layout is incompatible with the demonstrated effective submit-job limit",
            "scientific_policy_changed": False,
        },
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
        "source_transfers": 0,
        "execution_authorized": False,
        "historical_evidence_changed": False,
    }
    manifest = {**body, "package_identity": stable_hash("scalar-primary-b100-preparation-audit-v1", body)}
    return {
        "analysis_spec.json": canonical_json_text(analysis_spec),
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    output = root / OUTPUT
    contents = _content(root, historical=check)
    if check:
        for name, content in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise PaperContractError(f"tracked B100 preparation artifact differs: {name}")
            sidecar = output / f"{name}.sha256"
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked B100 preparation sidecar differs: {name}")
        status = "reproduced"
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite an existing B100 preparation package: {output}")
        output.mkdir(parents=True, exist_ok=False)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(file_hash(path) + "\n", encoding="ascii", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(canonical_json_text(build(repository_root=args.repository_root, check=args.check)), end="")


from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.scalar_primary_b100_preparation_audit_cli")


if __name__ == "__main__":
    main()
