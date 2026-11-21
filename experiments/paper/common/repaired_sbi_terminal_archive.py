"""Deterministic compact archive for corrected MUTACO recommendations."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mutaco import stable_hash
from mutaco.identity import ScientificIdentity
from mutaco.randomness import SeedCoordinate, SeedDeriver

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_experiment_contract import PAPER120_COHORT_SHA256
from .evidence_analysis import canonical_json_text
from .exact_terminal_generation import _without_timings
from .repaired_sbi_terminal_preparation import OUTPUT as PREPARATION_OUTPUT
from .validation import PaperContractError


RECORD_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-record-v1"
SUMMARY_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-summary-v1"
PACKAGE_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-package-v1"
RECORD_DOMAIN = "repaired-sbi-corrected-terminal-record-v1"
SUMMARY_DOMAIN = "repaired-sbi-corrected-terminal-summary-v1"
PACKAGE_DOMAIN = "repaired-sbi-corrected-terminal-package-v1"

ATTEMPT_ROOT = (
    "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/"
    "corrected_terminal_generation_v1"
)
OUTPUT = "results/paper/repaired_mutaco_sbi_corrected_terminal/v1"
EXPECTED_PREPARATION_PACKAGE = (
    "repaired-sbi-corrected-terminal-preparation-package-v1-"
    "123306d9650ee714d0601a253149c609bdfa08faabc762d3300cc5bb425181e6"
)
EXPECTED_SETTINGS = {
    "convergence_tolerance": 1.0e-7,
    "learning_rate": 0.05,
    "max_iterations": 64,
    "optimizer": "projected_adam",
    "raw_samples": 1024,
    "restarts": 10,
}
ALLOWED_RETURN_SOURCES = {"refined", "raw_fallback"}


def _cohort(root: Path) -> dict[int, Mapping[str, Any]]:
    cohort_root = root / "experiments/graph_inversion/frozen_targets/paper120_v3"
    if file_hash(cohort_root / "manifest.json") != PAPER120_COHORT_SHA256:
        raise PaperContractError("terminal archive cohort hash mismatch")
    records: dict[int, Mapping[str, Any]] = {}
    for position in range(120):
        row = read_json(cohort_root / f"target_{position:03d}.json")
        if row.get("target_index") != position:
            raise PaperContractError("terminal archive cohort ordering mismatch")
        records[position] = row
    return records


def _semantic_identity(document: Mapping[str, Any]) -> str:
    body = {
        key: document[key]
        for key in (
            "source_commit",
            "composite_source_identity",
            "raw_sha256",
            "exact_replay_identity",
            "budget",
            "target_position",
            "target_record_id",
            "final_model_id",
            "recommendation",
        )
    }
    return stable_hash("repaired-sbi-corrected-terminal-recommendation-v1", body)


def _validate_recommendation(
    document: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    target: Mapping[str, Any],
) -> Mapping[str, Any]:
    if document.get("status") != "completed":
        raise PaperContractError("corrected-terminal attempt is not completed")
    expected_fields = {
        "target_record_id": case["target_record_id"],
        "source_commit": case["native_source_commit"],
        "composite_source_identity": case["composite_source_identity"],
        "raw_sha256": case["raw_sha256"],
        "exact_replay_identity": case["exact_replay_identity"],
        "final_model_id": case["final_model_id"],
    }
    if any(document.get(key) != value for key, value in expected_fields.items()):
        raise PaperContractError("corrected-terminal attempt provenance mismatch")
    if _semantic_identity(document) != document.get("recommendation_identity"):
        raise PaperContractError("corrected-terminal semantic identity mismatch")
    recommendation = document.get("recommendation")
    timed = document.get("recommendation_with_timings")
    if not isinstance(recommendation, Mapping) or not isinstance(timed, Mapping):
        raise PaperContractError("corrected-terminal recommendation payload is missing")
    if _without_timings(timed) != recommendation:
        raise PaperContractError("timed and semantic terminal payloads differ")
    if (
        recommendation.get("policy_id")
        != "continuous_posterior_mean_physical_l1_mae"
        or recommendation.get("candidate_domain") != "continuous_bounded_theta"
        or recommendation.get("model_id") != case["final_model_id"]
        or recommendation.get("target_id") != case["target_record_id"]
        or recommendation.get("settings") != EXPECTED_SETTINGS
        or recommendation.get("returned_candidate_source") not in ALLOWED_RETURN_SOURCES
    ):
        raise PaperContractError("corrected-terminal frozen policy mismatch")
    context = case["terminal_context"]
    expected_seed = SeedDeriver(
        ScientificIdentity(str(context["seed_family_id"])), int(context["root_seed"])
    ).seed_for(
        SeedCoordinate(
            "terminal_recommendation",
            (int(context["target_index"]), str(context["policy_id"])),
        )
    )
    if (
        recommendation.get("derived_seed") != expected_seed
        or recommendation.get("seed_namespace") != "terminal_recommendation"
        or recommendation.get("seed_coordinates")
        != [int(context["target_index"]), str(context["policy_id"])]
    ):
        raise PaperContractError("corrected-terminal seed lineage mismatch")
    theta = recommendation.get("selected_theta")
    predicted = recommendation.get("predicted_physical_response")
    bounds = recommendation.get("physical_bounds")
    score = recommendation.get("score")
    if (
        not isinstance(theta, list)
        or len(theta) != 4
        or not isinstance(predicted, list)
        or len(predicted) != 20
        or not isinstance(bounds, list)
        or len(bounds) != 2
        or any(len(items) != 4 for items in bounds)
        or not isinstance(score, (int, float))
        or not all(math.isfinite(float(value)) for value in [*theta, *predicted, score])
        or any(not (bounds[0][axis] <= theta[axis] <= bounds[1][axis]) for axis in range(4))
    ):
        raise PaperContractError("corrected-terminal recommendation is nonfinite or unbounded")
    target_response = target.get("target_response")
    if not isinstance(target_response, list) or len(target_response) != 20:
        raise PaperContractError("corrected-terminal target response is invalid")
    recomputed = sum(abs(float(value) - float(goal)) for value, goal in zip(predicted, target_response)) / 20
    if not math.isclose(float(score), recomputed, rel_tol=0.0, abs_tol=2.0e-8):
        raise PaperContractError("corrected-terminal physical-MAE score mismatch")
    ledger = document.get("call_accounting", {})
    if ledger != {
        "simulator_constructed": False,
        "simulator_calls": 0,
        "search_calls": 0,
        "gp_fit_attempts": 0,
        "terminal_optimizations": 1,
    }:
        raise PaperContractError("corrected-terminal call accounting mismatch")
    return recommendation


def collect_terminal_archive(repository_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = repository_root.resolve()
    preparation_manifest = read_json(resolve_recorded_path(root, PREPARATION_OUTPUT) / "manifest.json")
    preparation = read_json(resolve_recorded_path(root, PREPARATION_OUTPUT) / "preparation_inventory.json")
    if preparation_manifest.get("package_identity") != EXPECTED_PREPARATION_PACKAGE:
        raise PaperContractError("corrected-terminal preparation package mismatch")
    cases = {
        (int(case["budget"]), int(case["target_position"])): case
        for case in preparation["cases"]
    }
    if set(cases) != {(budget, position) for budget in (40, 100) for position in range(120)}:
        raise PaperContractError("corrected-terminal prepared case matrix is incomplete")
    cohort = _cohort(root)
    attempt_root = root / ATTEMPT_ROOT
    paths = sorted(attempt_root.glob("b*/target-*.json"))
    if len(paths) != 240:
        raise PaperContractError("corrected-terminal attempt archive is incomplete")

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    recommendation_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    return_counts: Counter[str] = Counter()
    attempt_bytes = 0
    for path in paths:
        document = read_json(path)
        key = (int(document.get("budget", -1)), int(document.get("target_position", -1)))
        if key not in cases or key in seen:
            raise PaperContractError("corrected-terminal attempt coverage is invalid")
        seen.add(key)
        case = cases[key]
        target = cohort[key[1]]
        if target.get("target_record_id") != case["target_record_id"]:
            raise PaperContractError("corrected-terminal cohort identity mismatch")
        recommendation = _validate_recommendation(document, case=case, target=target)
        recommendation_id = str(document["recommendation_identity"])
        if recommendation_id in recommendation_ids:
            raise PaperContractError("corrected-terminal recommendation identity is duplicated")
        recommendation_ids.add(recommendation_id)
        relative = path.relative_to(root).as_posix()
        digest = file_hash(path)
        size = path.stat().st_size
        attempt_bytes += size
        source_counts[str(document["source_commit"])] += 1
        return_counts[str(recommendation["returned_candidate_source"])] += 1
        body = {
            "schema_version": RECORD_SCHEMA,
            "budget": key[0],
            "target_position": key[1],
            "target_record_id": document["target_record_id"],
            "native_source_commit": document["source_commit"],
            "composite_source_identity": document["composite_source_identity"],
            "full_attempt_artifact": relative,
            "full_attempt_sha256": digest,
            "full_attempt_bytes": size,
            "raw_sha256": document["raw_sha256"],
            "exact_replay_identity": document["exact_replay_identity"],
            "final_model_id": document["final_model_id"],
            "recommendation_identity": recommendation_id,
            "policy_id": recommendation["policy_id"],
            "selected_theta": recommendation["selected_theta"],
            "predicted_physical_response": recommendation["predicted_physical_response"],
            "posterior_mean_physical_mae": recommendation["score"],
            "returned_candidate_source": recommendation["returned_candidate_source"],
            "posterior_calls": recommendation["posterior_calls"],
            "derived_seed": recommendation["derived_seed"],
            "transform_id": recommendation["transform_id"],
            "simulator_calls": 0,
            "gp_fit_attempts": 0,
            "search_calls": 0,
            "terminal_optimizations": 1,
        }
        records.append({**body, "record_identity": stable_hash(RECORD_DOMAIN, body)})
    records.sort(key=lambda item: (item["budget"], item["target_position"]))
    if seen != set(cases):
        raise PaperContractError("corrected-terminal archive is missing prepared cases")
    expected_sources = {
        "9235d2fefefa6af0d35cc3af2dfaa01a2d56a3a9": 120,
        "2b067c33720ed7c6b30ee91ae6087081d5ad0776": 3,
        "e7baa235fd7282ea478acc83f9457cfd95b76185": 117,
    }
    if dict(source_counts) != expected_sources:
        raise PaperContractError("corrected-terminal native source split mismatch")
    summary_body = {
        "schema_version": SUMMARY_SCHEMA,
        "preparation_package_identity": EXPECTED_PREPARATION_PACKAGE,
        "preparation_identity": preparation["preparation_identity"],
        "record_count": len(records),
        "b40_records": 120,
        "b100_records": 120,
        "unique_recommendation_identities": len(recommendation_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "return_source_counts": dict(sorted(return_counts.items())),
        "full_attempt_bytes": attempt_bytes,
        "records_identity": stable_hash(
            "repaired-sbi-corrected-terminal-record-list-v1",
            [record["record_identity"] for record in records],
        ),
        "terminal_optimizations": 240,
        "simulator_calls": 0,
        "gp_fit_attempts": 0,
        "search_calls": 0,
        "heldout_assessment_calls": 0,
        "failed_recommendations": 0,
        "automatic_retries": 0,
        "historical_evidence_modified": False,
    }
    summary = {
        **summary_body,
        "summary_identity": stable_hash(SUMMARY_DOMAIN, summary_body),
    }
    return records, summary


def _contents(root: Path) -> dict[str, str]:
    records, summary = collect_terminal_archive(root)
    records_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    summary_text = canonical_json_text(summary)
    body = {
        "schema_version": PACKAGE_SCHEMA,
        "summary_identity": summary["summary_identity"],
        "records_identity": summary["records_identity"],
        "records_content_identity": stable_hash(
            "repaired-sbi-corrected-terminal-records-file-v1", records_text
        ),
        "summary_content_identity": stable_hash(
            "repaired-sbi-corrected-terminal-summary-file-v1", summary_text
        ),
        "record_count": 240,
        "terminal_recommendations_frozen": 240,
        "simulator_calls": 0,
        "heldout_assessment_calls": 0,
        "assessment_authorized": False,
        "truba_jobs_submitted": 0,
    }
    manifest = {**body, "package_identity": stable_hash(PACKAGE_DOMAIN, body)}
    return {
        "recommendations.jsonl": records_text,
        "summary.json": summary_text,
        "manifest.json": canonical_json_text(manifest),
    }


def build(*, repository_root: Path, check: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    output = root / OUTPUT
    contents = _contents(root)
    if check:
        for name, expected in contents.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise PaperContractError(f"tracked terminal archive differs: {name}")
            sidecar = output / f"{name}.sha256"
            if sidecar.read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked terminal archive sidecar differs: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(
                file_hash(path) + "\n", encoding="ascii", newline="\n"
            )
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "terminal_recommendations_frozen": 240,
        "simulator_calls": 0,
        "heldout_assessment_calls": 0,
        "truba_jobs_submitted": 0,
    }
