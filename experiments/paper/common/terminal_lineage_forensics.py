"""Zero-call terminal-lineage forensics for historical Paper-120 MUTACO."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_experiment_contract import PAPER120_COHORT_SHA256, contract_document
from .evidence_analysis import canonical_json_text
from .validation import PaperContractError


SPEC_SCHEMA = "mutaco-historical-terminal-lineage-forensic-spec-v1"
ROW_SCHEMA = "mutaco-historical-terminal-lineage-forensic-row-v1"
REPORT_SCHEMA = "mutaco-historical-terminal-lineage-forensic-report-v1"
SPEC_DOMAIN = "historical-terminal-lineage-forensic-spec-v1"
ROW_DOMAIN = "historical-terminal-lineage-forensic-row-v1"
REPORT_DOMAIN = "historical-terminal-lineage-forensic-report-v1"

CONTINUOUS_PATH = "src/mutaco/recommendation/_continuous.py"
PHYSICAL_MAE_PATH = "src/mutaco/recommendation/physical_mae.py"
CONTINUOUS_BLOB = "84fee3d4c0199aafa3d5455b794b557953a3ee33"
PHYSICAL_MAE_BLOB = "a5bfa2793742db9f6b195fcdfd6f4e7615e786c5"
B40_COMMIT = "9235d2fefefa6af0d35cc3af2dfaa01a2d56a3a9"
B100_QUALIFICATION_COMMIT = "2b067c33720ed7c6b30ee91ae6087081d5ad0776"
B100_POPULATION_COMMIT = "e7baa235fd7282ea478acc83f9457cfd95b76185"


def forensic_specification() -> dict[str, Any]:
    body = {
        "schema_version": SPEC_SCHEMA,
        "forensic_id": "paper120_historical_mutaco_b40_b100_terminal_lineage_v1",
        "contract_identity": contract_document()["contract_identity"],
        "cohort_sha256": PAPER120_COHORT_SHA256,
        "budgets": [40, 100],
        "method": "mutaco",
        "historical_sources": {
            "b40": {"source_commits": [B40_COMMIT], "raw_terminal_metadata": "complete_120_of_120"},
            "b100": {
                "source_commits": [B100_QUALIFICATION_COMMIT, B100_POPULATION_COMMIT],
                "qualification_target_positions": [0, 49, 99],
                "raw_terminal_metadata": "not_retained_in_local_canonical_bundle",
            },
        },
        "exact_source": {
            "continuous_optimizer_path": CONTINUOUS_PATH,
            "continuous_optimizer_git_blob_sha1": CONTINUOUS_BLOB,
            "physical_mae_path": PHYSICAL_MAE_PATH,
            "physical_mae_git_blob_sha1": PHYSICAL_MAE_BLOB,
            "identical_across_all_source_commits": True,
            "historical_line_evidence": {
                "initial_and_previous_score": [102, 103],
                "pre_update_score_and_gradient": [109, 121],
                "optimizer_update": [122, 124],
                "incorrect_current_assignment": [125, 125],
                "convergence_test": [127, 132],
                "post_loop_score_and_acceptance": [133, 145],
            },
        },
        "historical_terminal_settings": {
            "raw_samples": 1024,
            "restarts": 10,
            "max_iterations": 64,
            "learning_rate": 0.05,
            "optimizer": "projected_adam",
            "convergence_tolerance": 1e-7,
        },
        "fact_strengths": [
            "direct_persisted",
            "direct_recomputed_from_persisted",
            "exact_source_inferred",
            "unavailable",
        ],
        "error_vocabulary": ["E_obs", "E_raw_GP", "E_term_GP", "E_heldout"],
        "missing_policy": "unavailable_never_reconstructed_or_imputed",
        "scientific_calls": 0,
    }
    return {**body, "forensic_spec_identity": stable_hash(SPEC_DOMAIN, body)}


def validate_specification(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = forensic_specification()
    if dict(document) != expected:
        raise PaperContractError("terminal-lineage forensic specification is not canonical")
    return expected


def verify_exact_source_git(repository_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    """Verify frozen Git objects without checking out or modifying a source tree."""

    validate_specification(specification)
    root = repository_root.resolve()
    commits = tuple(specification["historical_sources"][key]["source_commits"] for key in ("b40", "b100"))
    flattened = tuple(item for group in commits for item in group)
    observed: dict[str, dict[str, str]] = {}
    for commit in flattened:
        observed[commit] = {}
        for path, expected in ((CONTINUOUS_PATH, CONTINUOUS_BLOB), (PHYSICAL_MAE_PATH, PHYSICAL_MAE_BLOB)):
            completed = subprocess.run(
                ["git", "rev-parse", f"{commit}:{path}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            actual = completed.stdout.strip()
            if actual != expected:
                raise PaperContractError(f"exact-source Git blob mismatch at {commit}:{path}")
            observed[commit][path] = actual
    return {"status": "verified", "commits": observed, "working_tree_modified": False}


def _cohort_ids(cohort: Mapping[str, Any]) -> tuple[str, ...]:
    records = cohort.get("records")
    if not isinstance(records, Mapping) or set(records) != {str(index) for index in range(120)}:
        raise PaperContractError("forensic cohort is incomplete")
    ids = tuple(str(records[str(index)].get("target_record_id", "")) for index in range(120))
    if any(not value for value in ids) or len(set(ids)) != 120:
        raise PaperContractError("forensic cohort identities are invalid")
    return ids


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperContractError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PaperContractError(f"{path} must be finite")
    return result


def _metric(value: float | None, *, strength: str, source: str) -> dict[str, Any]:
    return {
        "available": value is not None,
        "component_mean_mae": value,
        "response_l1_sum": None if value is None else 20.0 * value,
        "fact_strength": strength,
        "source": source,
    }


def _mae(response: Sequence[Any], target: Sequence[Any], path: str) -> float:
    left = tuple(_number(value, path) for value in response)
    right = tuple(_number(value, path) for value in target)
    if len(left) != 20 or len(right) != 20:
        raise PaperContractError(f"{path} must contain 20 responses")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 20.0


def _hashed_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise PaperContractError(f"forensic source is missing: {path}")
    actual = file_hash(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual:
        raise PaperContractError(f"forensic source hash mismatch: {path}")
    return read_json(path), actual


def _forensic_row(body: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": ROW_SCHEMA, **deepcopy(dict(body))}
    value["row_identity"] = stable_hash(ROW_DOMAIN, value)
    return value


def _compact_mutaco_by_position(path: Path, cohort_ids: Sequence[str]) -> tuple[dict[int, Mapping[str, Any]], str]:
    actual_hash = file_hash(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 120:
        raise PaperContractError("forensic compact evidence must contain 120 targets")
    result = {}
    for row in rows:
        position = row.get("target_position")
        if isinstance(position, bool) or not isinstance(position, int) or position not in range(120):
            raise PaperContractError("forensic compact target position is invalid")
        if row.get("target_record_identity") != cohort_ids[position] or position in result:
            raise PaperContractError("forensic compact target order is invalid")
        method = row.get("methods", {}).get("mutaco")
        if not isinstance(method, Mapping) or method.get("status") != "completed":
            raise PaperContractError("forensic MUTACO compact record is incomplete")
        if method.get("decision_policy_identifier") != "continuous_posterior_mean_physical_l1_mae":
            raise PaperContractError("forensic MUTACO compact policy is unexpected")
        result[position] = method
    return result, actual_hash


def build_b40_rows(
    *, repository_root: Path, raw_root: Path, compact_path: Path, cohort: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    repository_root = repository_root.resolve()
    cohort_ids = _cohort_ids(cohort)
    compact, compact_hash = _compact_mutaco_by_position(compact_path, cohort_ids)
    raw_paths = sorted(raw_root.glob("worker-*/raw/target-*/mutaco.json"))
    if len(raw_paths) != 120:
        raise PaperContractError("B40 raw MUTACO evidence is not complete")
    rows = []
    seen = set()
    settings = forensic_specification()["historical_terminal_settings"]
    for path in raw_paths:
        raw, raw_hash = _hashed_json(path)
        position = raw.get("target_index")
        target_id = str(raw.get("target_record_id", ""))
        if isinstance(position, bool) or not isinstance(position, int) or position not in range(120):
            raise PaperContractError("B40 raw target position is invalid")
        if position in seen or cohort_ids[position] != target_id:
            raise PaperContractError("B40 raw target identity or coverage is invalid")
        seen.add(position)
        native = compact[position]
        recommendation = raw.get("graph_terminal_recommendation")
        if not isinstance(recommendation, Mapping):
            raise PaperContractError("B40 terminal recommendation metadata is missing")
        if recommendation.get("policy_id") != native.get("decision_policy_identifier"):
            raise PaperContractError("B40 terminal policy differs between raw and compact evidence")
        theta = tuple(_number(value, "recommended_theta") for value in raw.get("recommended_theta", ()))
        selected_theta = tuple(_number(value, "selected_theta") for value in recommendation.get("selected_theta", ()))
        compact_theta = tuple(_number(value, "compact.recommended_theta") for value in native.get("recommended_theta", ()))
        if len(theta) != 4 or theta != selected_theta or theta != compact_theta:
            raise PaperContractError("B40 recommendation theta lineage mismatch")
        if recommendation.get("settings") != settings:
            raise PaperContractError("B40 historical terminal settings mismatch")
        histories = recommendation.get("restart_histories")
        if not isinstance(histories, list) or len(histories) != 10:
            raise PaperContractError("B40 terminal restart histories are incomplete")
        iterations = []
        initial_scores = []
        terminations = Counter()
        failures = 0
        for restart_index, history in enumerate(histories):
            if history.get("restart_index") != restart_index or history.get("raw_rank") != restart_index:
                raise PaperContractError("B40 terminal restart order is invalid")
            iteration_count = history.get("iterations")
            if isinstance(iteration_count, bool) or not isinstance(iteration_count, int) or iteration_count < 0:
                raise PaperContractError("B40 terminal iteration count is invalid")
            iterations.append(iteration_count)
            initial_scores.append(_number(history.get("initial_score"), "restart.initial_score"))
            terminations[str(history.get("termination"))] += 1
            failures += history.get("failure") is not None
        best_raw = min(initial_scores)
        returned = _number(recommendation.get("score"), "terminal.score")
        source = str(recommendation.get("returned_candidate_source", ""))
        selected_restart = recommendation.get("selected_restart_index")
        selected_start = None
        selected_improvement = None
        if source == "refined":
            if isinstance(selected_restart, bool) or not isinstance(selected_restart, int) or selected_restart not in range(10):
                raise PaperContractError("B40 refined recommendation lacks a selected restart")
            selected_start = initial_scores[selected_restart]
            selected_improvement = selected_start - returned
            if selected_improvement < -2e-6:
                raise PaperContractError("B40 selected refinement worsened its own restart")
        elif source == "raw_fallback":
            if selected_restart is not None or not math.isclose(returned, best_raw, rel_tol=2e-6, abs_tol=2e-6):
                raise PaperContractError("B40 raw fallback lineage is inconsistent")
        else:
            raise PaperContractError("B40 terminal candidate source is unsupported")
        observations = raw.get("scientific_projection", {}).get("observations")
        target = raw.get("target_response")
        if not isinstance(observations, list) or len(observations) != 40:
            raise PaperContractError("B40 observation lineage is incomplete")
        observed_error = min(_mae(item.get("response_physical", ()), target, "B40 observation") for item in observations)
        heldout = _number(native.get("response_error_mae"), "B40 heldout MAE")
        heldout_l1 = _number(native.get("response_error_l1_sum"), "B40 heldout L1")
        if not math.isclose(20.0 * heldout, heldout_l1, rel_tol=1e-12, abs_tol=1e-12):
            raise PaperContractError("B40 heldout MAE/L1 mismatch")
        all_one = all(value == 1 for value in iterations)
        path_classification = (
            "historical_one_step_terminal"
            if source == "refined" and all_one
            else "raw_fallback_after_one_step_attempts"
            if source == "raw_fallback" and all_one
            else "direct_nonuniform_historical_path"
        )
        rows.append(_forensic_row({
            "budget": 40,
            "target_position": position,
            "target_id": target_id,
            "source_commit": B40_COMMIT,
            "source_commit_strength": "direct_persisted",
            "native_record_identity": str(native.get("record_content_hash", "")),
            "raw_artifact": {
                "path": path.resolve().relative_to(repository_root).as_posix(),
                "sha256": raw_hash,
            },
            "compact_artifact_sha256": compact_hash,
            "recommended_theta": list(theta),
            "native_terminal_policy": str(recommendation.get("policy_id")),
            "implementation_classification": "defective_preupdate_convergence_comparison",
            "path_classification": path_classification,
            "path_fact_strength": "direct_persisted",
            "returned_candidate_source": source,
            "terminal_settings": deepcopy(recommendation.get("settings")),
            "restart_count": len(histories),
            "restart_iterations": iterations,
            "termination_counts": dict(sorted(terminations.items())),
            "restart_failure_count": failures,
            "posterior_calls": recommendation.get("posterior_calls"),
            "terminal_gp_fits": recommendation.get("gp_fits"),
            "terminal_simulator_calls": recommendation.get("simulator_calls"),
            "selected_restart_index": selected_restart,
            "best_raw_surrogate_mae": best_raw,
            "returned_surrogate_mae": returned,
            "returned_minus_best_raw_surrogate_mae": returned - best_raw,
            "selected_start_surrogate_mae": selected_start,
            "selected_start_minus_returned_surrogate_mae": selected_improvement,
            "errors": {
                "E_obs": _metric(observed_error, strength="direct_recomputed_from_persisted", source="40 stored physical search responses"),
                "E_raw_GP": _metric(best_raw, strength="direct_persisted", source="minimum persisted terminal restart initial_score"),
                "E_term_GP": _metric(returned, strength="direct_persisted", source="persisted terminal score at returned theta"),
                "E_heldout": _metric(heldout, strength="direct_persisted", source="paired fresh heldout physical assessment"),
            },
            "current_policy_conformance": "protocol_noncompliant_for_current_corrected_refinement_claim",
        }))
    return tuple(sorted(rows, key=lambda row: row["target_position"]))


def build_b100_rows(*, compact_path: Path, cohort: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    cohort_ids = _cohort_ids(cohort)
    compact, compact_hash = _compact_mutaco_by_position(compact_path, cohort_ids)
    qualification = {0, 49, 99}
    rows = []
    for position in range(120):
        native = compact[position]
        source_commit = B100_QUALIFICATION_COMMIT if position in qualification else B100_POPULATION_COMMIT
        theta = tuple(_number(value, "B100 recommended_theta") for value in native.get("recommended_theta", ()))
        if len(theta) != 4:
            raise PaperContractError("B100 recommendation theta dimension mismatch")
        heldout = _number(native.get("response_error_mae"), "B100 heldout MAE")
        if not math.isclose(20.0 * heldout, _number(native.get("response_error_l1_sum"), "B100 heldout L1"), rel_tol=1e-12, abs_tol=1e-12):
            raise PaperContractError("B100 heldout MAE/L1 mismatch")
        rows.append(_forensic_row({
            "budget": 100,
            "target_position": position,
            "target_id": cohort_ids[position],
            "source_commit": source_commit,
            "source_commit_strength": "direct_persisted",
            "native_record_identity": str(native.get("record_content_hash", "")),
            "raw_artifact": None,
            "compact_artifact_sha256": compact_hash,
            "recommended_theta": list(theta),
            "native_terminal_policy": str(native.get("decision_policy_identifier", "")),
            "implementation_classification": "defective_preupdate_convergence_comparison_exact_source_inferred",
            "path_classification": "indeterminate_from_retained_evidence",
            "path_fact_strength": "unavailable",
            "returned_candidate_source": None,
            "terminal_settings": deepcopy(forensic_specification()["historical_terminal_settings"]),
            "restart_count": None,
            "restart_iterations": None,
            "termination_counts": None,
            "restart_failure_count": None,
            "posterior_calls": None,
            "terminal_gp_fits": None,
            "terminal_simulator_calls": 0,
            "selected_restart_index": None,
            "best_raw_surrogate_mae": None,
            "returned_surrogate_mae": None,
            "returned_minus_best_raw_surrogate_mae": None,
            "selected_start_surrogate_mae": None,
            "selected_start_minus_returned_surrogate_mae": None,
            "errors": {
                "E_obs": _metric(None, strength="unavailable", source="search responses omitted from canonical B100 bundle"),
                "E_raw_GP": _metric(None, strength="unavailable", source="raw terminal diagnostics omitted from canonical B100 bundle"),
                "E_term_GP": _metric(None, strength="unavailable", source="terminal surrogate score omitted from canonical B100 bundle"),
                "E_heldout": _metric(heldout, strength="direct_persisted", source="paired fresh heldout physical assessment"),
            },
            "current_policy_conformance": "protocol_noncompliant_for_current_corrected_refinement_claim",
        }))
    return tuple(rows)


def validate_forensic_rows(rows: Sequence[Mapping[str, Any]], cohort: Mapping[str, Any]) -> dict[str, Any]:
    if len(rows) != 240:
        raise PaperContractError("terminal-lineage forensic grid must contain 240 rows")
    cohort_ids = _cohort_ids(cohort)
    seen = set()
    for original in rows:
        row = dict(original)
        if row.pop("row_identity", None) != stable_hash(ROW_DOMAIN, row):
            raise PaperContractError("terminal-lineage forensic row identity mismatch")
        if row.get("schema_version") != ROW_SCHEMA:
            raise PaperContractError("terminal-lineage forensic row schema mismatch")
        key = (row.get("budget"), row.get("target_position"))
        if key in seen or key[0] not in {40, 100} or key[1] not in range(120):
            raise PaperContractError("terminal-lineage forensic grid is duplicated or invalid")
        seen.add(key)
        if row.get("target_id") != cohort_ids[key[1]]:
            raise PaperContractError("terminal-lineage forensic cohort order mismatch")
        if tuple(row.get("errors", {})) != ("E_obs", "E_raw_GP", "E_term_GP", "E_heldout"):
            raise PaperContractError("terminal-lineage forensic error vocabulary mismatch")
    if seen != {(budget, position) for budget in (40, 100) for position in range(120)}:
        raise PaperContractError("terminal-lineage forensic grid is incomplete")
    return {"status": "valid", "rows": 240, "budgets": [40, 100], "targets_per_budget": 120}


def _summary(values: Iterable[float]) -> dict[str, Any]:
    items = tuple(float(value) for value in values)
    return {
        "n": len(items),
        "mean": None if not items else mean(items),
        "median": None if not items else median(items),
        "minimum": None if not items else min(items),
        "maximum": None if not items else max(items),
    }


def forensic_report(rows: Sequence[Mapping[str, Any]], specification: Mapping[str, Any], cohort: Mapping[str, Any]) -> dict[str, Any]:
    validate_specification(specification)
    validate_forensic_rows(rows, cohort)
    b40 = tuple(row for row in rows if row["budget"] == 40)
    b100 = tuple(row for row in rows if row["budget"] == 100)
    deltas = tuple(float(row["returned_minus_best_raw_surrogate_mae"]) for row in b40)
    selected_improvements = tuple(float(row["selected_start_minus_returned_surrogate_mae"]) for row in b40 if row["selected_start_minus_returned_surrogate_mae"] is not None)
    body = {
        "schema_version": REPORT_SCHEMA,
        "forensic_spec_identity": specification["forensic_spec_identity"],
        "row_count": len(rows),
        "b40": {
            "fact_strength": "direct_persisted_and_direct_recomputed",
            "targets": len(b40),
            "path_classifications": dict(sorted(Counter(row["path_classification"] for row in b40).items())),
            "returned_candidate_sources": dict(sorted(Counter(row["returned_candidate_source"] for row in b40).items())),
            "restart_histories": sum(int(row["restart_count"]) for row in b40),
            "one_iteration_restart_histories": sum(sum(value == 1 for value in row["restart_iterations"]) for row in b40),
            "posterior_calls": dict(sorted(Counter(str(row["posterior_calls"]) for row in b40).items())),
            "restart_failures": sum(int(row["restart_failure_count"]) for row in b40),
            "returned_vs_best_raw_surrogate_mae": {
                "summary": _summary(deltas),
                "returned_better": sum(value < -1e-12 for value in deltas),
                "equal_within_1e_12": sum(abs(value) <= 1e-12 for value in deltas),
                "returned_worse": sum(value > 1e-12 for value in deltas),
                "interpretation": "surrogate-only comparison; not heldout physical improvement",
            },
            "selected_restart_improvement_surrogate_mae": _summary(selected_improvements),
            "error_availability": {key: sum(row["errors"][key]["available"] for row in b40) for key in ("E_obs", "E_raw_GP", "E_term_GP", "E_heldout")},
        },
        "b100": {
            "fact_strength": "direct_endpoints_plus_exact_source_inference",
            "targets": len(b100),
            "path_classifications": dict(sorted(Counter(row["path_classification"] for row in b100).items())),
            "source_commits": dict(sorted(Counter(row["source_commit"] for row in b100).items())),
            "error_availability": {key: sum(row["errors"][key]["available"] for row in b100) for key in ("E_obs", "E_raw_GP", "E_term_GP", "E_heldout")},
            "target_specific_raw_vs_refined_classification": "unavailable_without_raw_terminal_metadata",
        },
        "exact_source_conclusion": {
            "continuous_optimizer_blob_identical_across_b40_and_b100": True,
            "physical_mae_blob_identical_across_b40_and_b100": True,
            "defect": "convergence compares the raw/pre-update score with the same pre-update score after optimizer.step",
            "mathematical_trace": [
                "s0 = L(mu(theta0), y_star); previous = s0",
                "compute gradient of s(theta0), then theta1 = project(Adam(theta0, grad s(theta0)))",
                "historical current = s(theta0), not s(theta1)",
                "abs(previous-current) = abs(s0-s0) = 0 <= tolerance",
                "loop terminates after one update; only afterward is s(theta1) evaluated",
            ],
        },
        "scientific_disposition": {
            "valid_claims": [
                "historical searches and recommendations are valid evidence for the policies actually executed",
                "heldout physical assessments remain observations of the immutable returned recommendations",
                "B40 directly executed 1024-point screening followed by one update attempt per restart",
            ],
            "invalid_current_policy_claims": [
                "historical MUTACO endpoints do not estimate performance of the corrected multi-step terminal optimizer",
                "raw, terminal-surrogate, and heldout errors cannot be substituted for one another",
                "B100 target-specific raw-versus-one-step return branches cannot be claimed from compact evidence",
            ],
            "requires_new_evidence": "a repaired MUTACO-vs-SBI comparison is required for current-policy performance claims",
        },
        "historical_evidence_modified": False,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "forensic_report_identity": stable_hash(REPORT_DOMAIN, body)}


def forensic_audit(rows: Sequence[Mapping[str, Any]], source_verification: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = []
    if source_verification.get("status") != "verified":
        mismatches.append("exact_source_not_verified")
    for row in rows:
        if row["budget"] == 40 and row["path_classification"] == "direct_nonuniform_historical_path":
            mismatches.append(f"nonuniform_b40_path:{row['target_position']}")
    body = {
        "schema_version": "mutaco-historical-terminal-lineage-forensic-audit-v1",
        "checks": [
            "source_git_blobs",
            "raw_file_sha256_sidecars",
            "cohort_identity_and_order",
            "raw_compact_recommendation_theta_equality",
            "native_policy_equality",
            "restart_history_structure",
            "heldout_mae_l1_consistency",
        ],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "historical_evidence_modified": False,
        "scientific_calls": 0,
    }
    return {**body, "forensic_audit_identity": stable_hash("historical-terminal-lineage-forensic-audit-v1", body)}
