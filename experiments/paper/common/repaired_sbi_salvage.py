"""Zero-call design and inventory for a repaired MUTACO--SBI comparison."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_experiment_contract import PAPER120_COHORT_SHA256, contract_document
from .validation import PaperContractError


DESIGN_SCHEMA = "mutaco-repaired-sbi-salvage-design-v1"
AUDIT_SCHEMA = "mutaco-repaired-sbi-salvage-inventory-v1"
DESIGN_DOMAIN = "repaired-sbi-salvage-design-v1"
AUDIT_DOMAIN = "repaired-sbi-salvage-inventory-v1"

B40_COMMIT = "9235d2fefefa6af0d35cc3af2dfaa01a2d56a3a9"
B100_QUALIFICATION_COMMIT = "2b067c33720ed7c6b30ee91ae6087081d5ad0776"
B100_POPULATION_COMMIT = "e7baa235fd7282ea478acc83f9457cfd95b76185"
RUNNER_PATH = "src/mutaco/orchestration/runner.py"
RUNNER_BLOB = "99cc8ef1606a41dd221cf583ca225222178ca15c"
TERMINAL_PATH = "src/mutaco/recommendation/_continuous.py"
DEFECTIVE_TERMINAL_BLOB = "84fee3d4c0199aafa3d5455b794b557953a3ee33"
CORRECTED_TERMINAL_BLOB = "76569b61e31fb8d10575d97f4b4ad136eb481507"
CORRECTION_COMMIT = "9603305a7f7f7b6c6f93586aa5728c0dc2393237"

B40_RAW = "runs/paper/graph_oscillator/graph_paper120_b40_mutaco_npe_v3_v1"
B40_COMPACT = "benchmarks/graph/paper120_mutaco_npe_snpe_v1/paired_records.jsonl"
B40_COMPACT_MANIFEST = "benchmarks/graph/paper120_mutaco_npe_snpe_v1/manifest.json"
B100_COMPACT = "benchmarks/paper/graph_paper120/b100_v1/paired_records.jsonl"
B100_COMPACT_MANIFEST = "benchmarks/paper/graph_paper120/b100_v1/manifest.json"
B40_FORENSIC = "benchmarks/paper/historical_terminal_lineage_forensics/v1/manifest.json"
RECOVERY_MANIFEST = "benchmarks/paper/repaired_mutaco_sbi_recovery_qualification/v1/manifest.json"
RECOVERY_EVIDENCE = "benchmarks/paper/repaired_mutaco_sbi_recovery_qualification/v1/b100_raw_recovery.json"
REPLAY_QUALIFICATION = "benchmarks/paper/repaired_mutaco_sbi_recovery_qualification/v1/exact_state_qualification.json"


def salvage_design() -> dict[str, Any]:
    body = {
        "schema_version": DESIGN_SCHEMA,
        "design_id": "paper120_repaired_mutaco_npe_snpe_b40_b100_salvage_v1",
        "cross_experiment_contract_identity": contract_document()["contract_identity"],
        "cohort_sha256": PAPER120_COHORT_SHA256,
        "targets": 120,
        "budgets": [40, 100],
        "methods": ["mutaco", "sbi_npe", "sbi_snpe"],
        "causal_boundary": {
            "conclusion": "terminal_recommendation_is_strictly_downstream_of_all_search_evaluations_and_final_gp_fit",
            "runner_path": RUNNER_PATH,
            "runner_git_blob_sha1": RUNNER_BLOB,
            "identical_across_historical_commits": True,
            "sequential_loop_lines": [1171, 1315],
            "recommendation_boundary_lines": [1317, 1422],
            "terminal_simulator_calls": 0,
        },
        "minimal_source_delta": {
            "path": TERMINAL_PATH,
            "before_git_blob_sha1": DEFECTIVE_TERMINAL_BLOB,
            "after_git_blob_sha1": CORRECTED_TERMINAL_BLOB,
            "correction_commit": CORRECTION_COMMIT,
            "allowed_change": "post_update_scoring_convergence_and_best_iterate_retention_only",
            "all_other_source_must_match_each_budget_native_commit": True,
        },
        "reuse_policy": {
            "selection_rule": "target_outcome_blind_evidence_completeness_and_exact_replay_only",
            "mutaco_search": "reuse_only_after_complete_raw_lineage_and_exact_final_model_replay",
            "mutaco_terminal": "always_new_corrected_refined_terminal_evidence",
            "sbi_search_and_recommendations": "reuse_immutable_native_recommendations",
            "historical_heldout_responses": "context_evidence_only_not_primary_endpoint",
            "partial_pooling": "forbidden",
        },
        "exact_replay_gate": {
            "simulator_calls": 0,
            "required_equalities": [
                "ordered observation ids theta and physical responses",
                "response transform identity",
                "final training data identity",
                "final GP model identity",
                "terminal seed context and settings",
            ],
            "failure_action": "rerun_mutaco_search_for_that_target_under_native_source_plus_minimal_terminal_delta",
            "refit_classification": "prospective_reconstruction_validation_never_historical_evidence",
        },
        "assessment_policy": {
            "identifier": "new_budget_specific_three_method_paired_heldout_assessment_v1",
            "freeze_all_recommendations_before_assessment": True,
            "calls_per_valid_recommendation": 1,
            "b40_assessment_campaign_seed": 92021540,
            "b100_assessment_campaign_seed": 92021600,
            "method_independent_within_target_budget": True,
            "distinct_across_budgets": True,
            "reason": "uniform new endpoint evidence avoids mixed assessment epochs and repairs partial B40 context provenance",
        },
        "call_formula": {
            "n40_rerun": "number of B40 MUTACO targets failing raw-lineage or exact-model replay",
            "n100_rerun": "number of B100 MUTACO targets failing raw-lineage or exact-model replay",
            "search_calls": "40*n40_rerun + 100*n100_rerun",
            "assessment_calls_maximum": 720,
            "total_calls_maximum_formula": "720 + 40*n40_rerun + 100*n100_rerun",
            "best_case": 720,
            "worst_case": 17520,
        },
        "evidence_policy": {
            "new_versioned_records": True,
            "historical_records_immutable": True,
            "error_vocabulary": ["E_obs", "E_raw_GP", "E_term_GP", "E_heldout"],
            "analysis_pipeline": "cross_experiment_evidence_analysis_v1",
            "provenance_requires": [
                "native source record identity",
                "raw trajectory SHA256 when reused",
                "exact replay identities",
                "terminal implementation blob",
                "new recommendation identity",
                "new assessment context and response",
            ],
        },
        "authorization": {
            "raw_archive_recovery": False,
            "exact_replay": False,
            "terminal_generation": False,
            "simulator_execution": False,
            "truba_submission": False,
        },
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "design_identity": stable_hash(DESIGN_DOMAIN, body)}


def validate_salvage_design(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = salvage_design()
    if dict(document) != expected:
        raise PaperContractError("repaired SBI salvage design is not canonical")
    return expected


def _git_blob(root: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _cohort_ids(root: Path) -> tuple[str, ...]:
    path = root / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
    if file_hash(path) != PAPER120_COHORT_SHA256:
        raise PaperContractError("salvage cohort hash mismatch")
    records = read_json(path).get("records", {})
    values = tuple(str(records[str(index)]["target_record_id"]) for index in range(120))
    if len(set(values)) != 120:
        raise PaperContractError("salvage cohort identities are invalid")
    return values


def _check_compact(
    source_address: str,
    cohort_ids: tuple[str, ...],
    repository_root: Path,
    *,
    budget: int,
    expected_sha256: str,
) -> dict[str, Any]:
    path = resolve_recorded_path(repository_root, source_address)
    actual_sha256 = file_hash(path)
    if actual_sha256 != expected_sha256:
        raise PaperContractError(f"compact salvage source hash mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 120:
        raise PaperContractError(f"compact salvage source is incomplete: {path}")
    for position, row in enumerate(rows):
        if row.get("target_position") != position or row.get("target_record_identity") != cohort_ids[position]:
            raise PaperContractError(f"compact salvage cohort mismatch: {path}")
        methods = row.get("methods", {})
        if set(methods) != {"mutaco", "sbi_npe", "sbi_snpe"}:
            raise PaperContractError(f"compact salvage method grid mismatch: {path}")
        if any(methods[name].get("status") != "completed" for name in methods):
            raise PaperContractError(f"compact salvage method is incomplete: {path}")
        for method in methods.values():
            theta = method.get("recommended_theta")
            inference = method.get("call_ledger", {}).get("method_inference_or_training", {})
            if inference.get("successful_calls") != budget or not isinstance(theta, list) or len(theta) != 4:
                raise PaperContractError(f"compact salvage endpoint is invalid: {path}")
    return {
        "path": source_address,
        "sha256": actual_sha256,
        "rows": len(rows),
    }


def inventory_salvage(repository_root: Path, design: Mapping[str, Any]) -> dict[str, Any]:
    validate_salvage_design(design)
    root = repository_root.resolve()
    cohort_ids = _cohort_ids(root)
    commits = (B40_COMMIT, B100_QUALIFICATION_COMMIT, B100_POPULATION_COMMIT)
    for commit in commits:
        if _git_blob(root, commit, RUNNER_PATH) != RUNNER_BLOB:
            raise PaperContractError("historical runner blob mismatch")
        if _git_blob(root, commit, TERMINAL_PATH) != DEFECTIVE_TERMINAL_BLOB:
            raise PaperContractError("historical terminal blob mismatch")
    if _git_blob(root, CORRECTION_COMMIT, TERMINAL_PATH) != CORRECTED_TERMINAL_BLOB:
        raise PaperContractError("corrected terminal blob mismatch")

    raw_paths = sorted((root / B40_RAW).glob("worker-*/raw/target-*/mutaco.json"))
    if len(raw_paths) != 120:
        raise PaperContractError("B40 raw salvage inventory is incomplete")
    seen: set[int] = set()
    runtime_counts: dict[str, int] = {}
    for path in raw_paths:
        actual_hash = file_hash(path)
        sidecar = path.with_suffix(".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual_hash:
            raise PaperContractError(f"B40 raw salvage hash mismatch: {path}")
        row = read_json(path)
        position = row.get("target_index")
        if not isinstance(position, int) or position not in range(120) or position in seen:
            raise PaperContractError("B40 raw salvage target coverage is invalid")
        seen.add(position)
        if row.get("target_record_id") != cohort_ids[position] or row.get("status") != "completed":
            raise PaperContractError("B40 raw salvage target identity/status mismatch")
        projection = row.get("scientific_projection", {})
        observations = projection.get("observations", [])
        fits = projection.get("fits", [])
        if row.get("inference_calls") != 40 or len(observations) != 40 or len(fits) != 25:
            raise PaperContractError("B40 raw salvage trajectory is incomplete")
        observation_ids = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise PaperContractError("B40 raw salvage observation is invalid")
            observation_ids.append(observation.get("observation_id"))
            if len(observation.get("theta", ())) != 4 or len(observation.get("response_physical", ())) != 20:
                raise PaperContractError("B40 raw salvage observation shape is invalid")
        if any(not value for value in observation_ids) or len(set(observation_ids)) != 40:
            raise PaperContractError("B40 raw salvage observation identities are invalid")
        final_fit = fits[-1]
        if final_fit.get("training_version") != 24 or not final_fit.get("model_id") or not final_fit.get("training_data_id"):
            raise PaperContractError("B40 final GP identity lineage is incomplete")
        recommendation = row.get("graph_terminal_recommendation", {})
        if recommendation.get("transform_id") != projection.get("transform", {}).get("transform_id"):
            raise PaperContractError("B40 transform lineage mismatch")
        context = row.get("runner_provenance", {}).get("configuration", {}).get("execution", {}).get("terminal_recommendation_context", {})
        if context.get("target_id") != cohort_ids[position] or context.get("target_index") != position:
            raise PaperContractError("B40 terminal context lineage mismatch")
        execution = row.get("runner_provenance", {}).get("fits", [{}])[-1].get("execution", {})
        runtime_key = f"{execution.get('torch_version')}|{execution.get('device')}|{execution.get('dtype')}"
        runtime_counts[runtime_key] = runtime_counts.get(runtime_key, 0) + 1

    b40_manifest = read_json(resolve_recorded_path(root, B40_COMPACT_MANIFEST))
    b100_manifest = read_json(resolve_recorded_path(root, B100_COMPACT_MANIFEST))
    b40_compact = _check_compact(
        B40_COMPACT,
        cohort_ids,
        root,
        budget=40,
        expected_sha256=str(b40_manifest.get("paired_records_sha256", "")),
    )
    b100_compact = _check_compact(
        B100_COMPACT,
        cohort_ids,
        root,
        budget=100,
        expected_sha256=str(b100_manifest.get("paired_records_sha256", "")),
    )
    forensic_manifest = read_json(resolve_recorded_path(root, B40_FORENSIC))
    if forensic_manifest.get("mismatch_count") != 0 or forensic_manifest.get("b40_raw_files_verified") != 120:
        raise PaperContractError("terminal forensic prerequisite is not clean")
    recovery_manifest = read_json(resolve_recorded_path(root, RECOVERY_MANIFEST))
    recovery_evidence = read_json(resolve_recorded_path(root, RECOVERY_EVIDENCE))
    replay_qualification = read_json(resolve_recorded_path(root, REPLAY_QUALIFICATION))
    if (
        recovery_manifest.get("b100_raw_trajectories_verified") != 120
        or recovery_evidence.get("coverage", {}).get("missing") != 0
        or recovery_evidence.get("coverage", {}).get("sha256_mismatches") != 0
        or replay_qualification.get("cases_passed") != 240
        or replay_qualification.get("cases_failed") != 0
    ):
        raise PaperContractError("raw recovery or exact-state qualification prerequisite failed")

    body = {
        "schema_version": AUDIT_SCHEMA,
        "design_identity": design["design_identity"],
        "causal_boundary": {
            "status": "verified_exact_source",
            "runner_blob": RUNNER_BLOB,
            "search_effect_of_terminal_defect": "none",
            "recommendation_effect": "demonstrated",
        },
        "b40": {
            "raw_trajectory_files_local_verified": 120,
            "complete_observation_trajectories": 120,
            "complete_fit_lineages": 120,
            "runtime_counts": dict(sorted(runtime_counts.items())),
            "compact_source": b40_compact,
            "current_classification": "population_exact_state_replay_passed_corrected_terminal_pending",
            "exact_model_replays_completed": 120,
            "best_case_new_calls": 360,
            "search_rerun_calls_required": 0,
        },
        "b100": {
            "raw_trajectory_files_local_verified": 120,
            "compact_source": b100_compact,
            "registered_remote_remaining117_output_root": "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/b100_remaining117/e7baa235/workers",
            "remote_existence_verified_by_this_gate": True,
            "qualification3_raw_recovery_path_registered": True,
            "current_classification": "population_exact_state_replay_passed_corrected_terminal_pending",
            "exact_model_replays_completed": 120,
            "assessment_calls": 360,
            "search_rerun_calls_required": 0,
        },
        "combined_call_preview": {
            "formula": "720 + 40*n40_rerun + 100*n100_rerun",
            "best_case": 720,
            "worst_case": 17520,
            "resolved_search_rerun_calls": 0,
            "resolved_remaining_scientific_calls": 720,
        },
        "readiness": {
            "b40_scientific_execution_ready": False,
            "b100_scientific_execution_ready": False,
            "next_zero_call_gate": "corrected-terminal exact-source preparation and deterministic recommendation-generation declaration",
            "reason": "all historical searches are now salvageable; corrected recommendations must be generated and frozen before assessment",
        },
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "inventory_identity": stable_hash(AUDIT_DOMAIN, body)}
