"""Zero-call preparation for the repaired MUTACO--SBI assessment."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from experiments.paper.common.cross_experiment_contract import PAPER120_COHORT_SHA256, contract_document
from experiments.paper.common.evidence_analysis import canonical_json_text
from experiments.paper.protocol.validation import PaperContractError


DECLARATION_SCHEMA = "mutaco-repaired-sbi-assessment-declaration-v1"
TASK_SCHEMA = "mutaco-repaired-sbi-assessment-task-v1"
PREPARATION_SCHEMA = "mutaco-repaired-sbi-assessment-preparation-v1"
PACKAGE_SCHEMA = "mutaco-repaired-sbi-assessment-preparation-package-v1"
DECLARATION_DOMAIN = "repaired-sbi-assessment-declaration-v1"
TASK_DOMAIN = "repaired-sbi-assessment-task-v1"
PREPARATION_DOMAIN = "repaired-sbi-assessment-preparation-v1"
PACKAGE_DOMAIN = "repaired-sbi-assessment-preparation-package-v1"

OUTPUT = "results/paper/repaired_mutaco_sbi_assessment_preparation/v1"
DECLARATION_PATH = (
    "experiments/paper/studies/graph_paper120/analysis/"
    "repaired_mutaco_sbi_assessment_v1.json"
)
COHORT = "data/cohorts/paper120_v3/manifest.json"
TARGET_CONFIG = "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json"
TARGET_CONFIG_SHA256 = "2614cdae3f2eb40ff1a0f0ee473056f05c8284d3ab59e2f159969ed1cc704a8a"
MUTACO_RECORDS = (
    "benchmarks/paper/repaired_mutaco_sbi_corrected_terminal/v1/"
    "recommendations.jsonl"
)
MUTACO_RECORDS_SHA256 = "a5f1ecca16c560378c57836dca64853a80eb3f3a144cfafd52a1f4eb0fb7ab81"
MUTACO_PACKAGE_IDENTITY = (
    "repaired-sbi-corrected-terminal-package-v1-"
    "6988fdc19056535377b54b28ba6d6b03c9ee46ddba5f793d03e536d053d490a2"
)
SBI_SOURCES = {
    40: (
        "benchmarks/graph/paper120_mutaco_npe_snpe_v1/paired_records.jsonl",
        "cd0352310c8e570bdc56fe6e4ef587bd58e1c26f1f0865d44878ffc7dd2c0c72",
    ),
    100: (
        "benchmarks/paper/graph_paper120/b100_v1/paired_records.jsonl",
        "29f6af90cdec9581dd3b850636c05c6dd740dac2aa22164c7f4bc68b3ab61f67",
    ),
}
SALVAGE_DESIGN_IDENTITY = (
    "repaired-sbi-salvage-design-v1-"
    "5283c87e6f5fedfc8d611682769ca0a8b1bbdd184b26110678f2fd74ceaa2a9b"
)
METHODS = ("mutaco", "sbi_npe", "sbi_snpe")
ASSESSMENT_SEEDS = {40: 92021540, 100: 92021600}
PROTOCOL_ID = "paper120_repaired_mutaco_npe_snpe_assessment_v1"
SIMULATOR_DEFINITION_IDENTITY = (
    "graph-oscillator.overdamped-scientific.v1.p4-paper-baseline."
    "m20-five-force-scientific.float32"
)
SIMULATOR_CONFIGURATION_IDENTITY = (
    "graph-paper-v3-simulator-configuration-v1-"
    "ff70b08584d86189e7464b20602142d9faeca01f588c0d691de6cdf2c7568a08"
)
ASSESSMENT_FIDELITY = "canonical_sparse_m20_one_evaluation_v1"
POLICY_ID = "new_budget_specific_three_method_paired_heldout_assessment_v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def assessment_declaration() -> dict[str, Any]:
    body = {
        "schema_version": DECLARATION_SCHEMA,
        "campaign_id": "paper120_repaired_mutaco_npe_snpe_b40_b100_assessment_v1",
        "cross_experiment_contract_identity": contract_document()["contract_identity"],
        "salvage_design_identity": SALVAGE_DESIGN_IDENTITY,
        "corrected_mutaco_package_identity": MUTACO_PACKAGE_IDENTITY,
        "cohort": {"path": COHORT, "sha256": PAPER120_COHORT_SHA256, "targets": 120},
        "budgets": [40, 100],
        "methods": list(METHODS),
        "recommendation_sources": {
            "mutaco": {"path": MUTACO_RECORDS, "sha256": MUTACO_RECORDS_SHA256},
            "sbi_b40": {"path": SBI_SOURCES[40][0], "sha256": SBI_SOURCES[40][1]},
            "sbi_b100": {"path": SBI_SOURCES[100][0], "sha256": SBI_SOURCES[100][1]},
        },
        "simulator": {
            "configuration_path": TARGET_CONFIG,
            "configuration_sha256": TARGET_CONFIG_SHA256,
            "definition_identity": SIMULATOR_DEFINITION_IDENTITY,
            "configuration_identity": SIMULATOR_CONFIGURATION_IDENTITY,
            "fidelity": ASSESSMENT_FIDELITY,
            "dtype": "float32",
            "device": "cpu",
        },
        "assessment_policy": {
            "identifier": POLICY_ID,
            "protocol_id": PROTOCOL_ID,
            "campaign_seeds": {"40": ASSESSMENT_SEEDS[40], "100": ASSESSMENT_SEEDS[100]},
            "randomness_namespace": "paired_heldout_assessment",
            "randomness_coordinates": ["budget", "target_record_identity", "replicate"],
            "method_in_random_address": False,
            "same_context_for_all_methods_within_target_budget": True,
            "distinct_context_across_budgets": True,
            "replicate": 0,
            "calls_per_recommendation": 1,
            "automatic_retry": False,
        },
        "endpoint": {
            "symbol": "E_heldout",
            "formula": "mean_m_abs_y_assess_m_theta_hat_minus_y_star_m",
            "response_source": "fresh_heldout_simulator_response",
            "response_dimension": 20,
            "selection_domain": "frozen_recommendation_only",
        },
        "call_preview": {
            "targets": 120,
            "budgets": 2,
            "methods": 3,
            "maximum_assessment_calls": 720,
            "search_calls": 0,
            "gp_fit_attempts": 0,
            "terminal_optimizations": 0,
            "scientific_calls_executed": 0,
        },
        "evidence_policy": {
            "append_only_attempts": True,
            "preserve_failures": True,
            "historical_evidence_immutable": True,
            "historical_heldout_responses_excluded": True,
            "partial_pooling_forbidden": True,
            "canonical_json_allow_nan": False,
        },
        "authorization": {
            "assessment_execution": False,
            "simulator_execution": False,
            "truba_preparation": False,
            "truba_submission": False,
        },
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "declaration_identity": stable_hash(DECLARATION_DOMAIN, body)}


def validate_assessment_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = assessment_declaration()
    if dict(document) != expected:
        raise PaperContractError("repaired SBI assessment declaration is not canonical")
    return expected


def assessment_context(*, budget: int, target_position: int, target_id: str) -> dict[str, Any]:
    if budget not in ASSESSMENT_SEEDS or not 0 <= target_position < 120 or not target_id:
        raise PaperContractError("repaired SBI assessment address is invalid")
    campaign_seed = ASSESSMENT_SEEDS[budget]
    family_definition = {
        "protocol_id": PROTOCOL_ID,
        "budget": budget,
        "target_record_identity": target_id,
        "simulator_definition_identity": SIMULATOR_DEFINITION_IDENTITY,
        "simulator_configuration_identity": SIMULATOR_CONFIGURATION_IDENTITY,
        "assessment_fidelity": ASSESSMENT_FIDELITY,
        "assessment_campaign_seed": campaign_seed,
    }
    family = ScientificIdentity.create(family_definition)
    coordinate = SeedCoordinate("paired_heldout_assessment", (budget, target_id, 0))
    return {
        "policy_identifier": POLICY_ID,
        "family_identity": family.value,
        "family_definition": family_definition,
        "coordinate": {
            "namespace": coordinate.namespace,
            "coordinates": list(coordinate.coordinates),
        },
        "target_record_identity": target_id,
        "budget": budget,
        "replicate": 0,
        "purpose": "paired_heldout_assessment",
        "logical_index": target_position,
        "evaluation_seed": SeedDeriver(family, campaign_seed).seed_for(coordinate),
    }


def _cohort(root: Path) -> dict[int, str]:
    path = resolve_recorded_path(root, COHORT)
    if file_hash(path) != PAPER120_COHORT_SHA256:
        raise PaperContractError("repaired SBI assessment cohort hash mismatch")
    records = read_json(path).get("records", {})
    result = {position: str(records[str(position)]["target_record_id"]) for position in range(120)}
    if len(set(result.values())) != 120:
        raise PaperContractError("repaired SBI assessment cohort identities are invalid")
    return result


def _theta(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise PaperContractError(f"{label} recommendation is invalid")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise PaperContractError(f"{label} recommendation is nonfinite")
    return result


def _recommendations(root: Path, cohort: Mapping[int, str]) -> dict[tuple[int, int, str], dict[str, Any]]:
    mutaco_path = resolve_recorded_path(root, MUTACO_RECORDS)
    if file_hash(mutaco_path) != MUTACO_RECORDS_SHA256:
        raise PaperContractError("corrected MUTACO recommendation source hash mismatch")
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in _jsonl(mutaco_path):
        budget, position = int(row["budget"]), int(row["target_position"])
        if row.get("target_record_id") != cohort[position]:
            raise PaperContractError("corrected MUTACO recommendation cohort mismatch")
        result[(budget, position, "mutaco")] = {
            "recommended_theta": _theta(row.get("selected_theta"), "MUTACO"),
            "recommendation_identity": row["recommendation_identity"],
            "source_record_identity": row["record_identity"],
            "source_path": MUTACO_RECORDS,
            "source_sha256": MUTACO_RECORDS_SHA256,
            "recommendation_provenance": "corrected_refined_terminal",
        }
    for budget, (source, expected_hash) in SBI_SOURCES.items():
        path = resolve_recorded_path(root, source)
        if file_hash(path) != expected_hash:
            raise PaperContractError(f"immutable SBI B{budget} source hash mismatch")
        rows = _jsonl(path)
        if len(rows) != 120:
            raise PaperContractError(f"immutable SBI B{budget} source is incomplete")
        for position, row in enumerate(rows):
            if (
                row.get("target_position") != position
                or row.get("target_record_identity") != cohort[position]
            ):
                raise PaperContractError(f"immutable SBI B{budget} cohort mismatch")
            for method in ("sbi_npe", "sbi_snpe"):
                record = row.get("methods", {}).get(method, {})
                if record.get("status") != "completed" or not record.get("record_content_hash"):
                    raise PaperContractError(f"immutable {method} B{budget} record is invalid")
                result[(budget, position, method)] = {
                    "recommended_theta": _theta(record.get("recommended_theta"), method),
                    "recommendation_identity": record["record_content_hash"],
                    "source_record_identity": record["record_content_hash"],
                    "source_path": source,
                    "source_sha256": expected_hash,
                    "recommendation_provenance": "immutable_native_sbi_recommendation",
                }
    expected = {
        (budget, position, method)
        for budget in (40, 100)
        for position in range(120)
        for method in METHODS
    }
    if set(result) != expected:
        raise PaperContractError("repaired SBI assessment recommendation matrix is incomplete")
    return result


def preparation_inventory(repository_root: Path, declaration: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = validate_assessment_declaration(declaration)
    root = repository_root.resolve()
    if file_hash(root / TARGET_CONFIG) != TARGET_CONFIG_SHA256:
        raise PaperContractError("repaired SBI assessment simulator configuration mismatch")
    cohort = _cohort(root)
    recommendations = _recommendations(root, cohort)
    tasks: list[dict[str, Any]] = []
    for budget in (40, 100):
        for position in range(120):
            context = assessment_context(
                budget=budget, target_position=position, target_id=cohort[position]
            )
            for method in METHODS:
                recommendation = recommendations[(budget, position, method)]
                body = {
                    "schema_version": TASK_SCHEMA,
                    "budget": budget,
                    "target_position": position,
                    "target_record_id": cohort[position],
                    "method": method,
                    **recommendation,
                    "assessment_context": context,
                    "nominal_assessment_calls": 1,
                    "automatic_retry": False,
                }
                tasks.append({**body, "task_identity": stable_hash(TASK_DOMAIN, body)})
    if len(tasks) != 720 or len({task["task_identity"] for task in tasks}) != 720:
        raise PaperContractError("repaired SBI assessment task identities are invalid")
    contexts = {
        (task["budget"], task["target_position"]): task["assessment_context"]
        for task in tasks
    }
    if len(contexts) != 240 or len({value["family_identity"] for value in contexts.values()}) != 240:
        raise PaperContractError("repaired SBI assessment context matrix is invalid")
    for budget in (40, 100):
        for position in range(120):
            group = [
                task for task in tasks
                if task["budget"] == budget and task["target_position"] == position
            ]
            if len(group) != 3 or len({canonical_json_text(task["assessment_context"]) for task in group}) != 1:
                raise PaperContractError("repaired SBI assessment methods are not paired")
    summary_body = {
        "schema_version": PREPARATION_SCHEMA,
        "declaration_identity": expected["declaration_identity"],
        "task_count": 720,
        "target_budget_contexts": 240,
        "budget_counts": dict(Counter(str(task["budget"]) for task in tasks)),
        "method_counts": dict(Counter(task["method"] for task in tasks)),
        "unique_task_identities": 720,
        "unique_context_family_identities": 240,
        "same_context_for_all_methods_within_target_budget": True,
        "cross_budget_context_overlap": 0,
        "maximum_assessment_calls": 720,
        "simulator_calls": 0,
        "search_calls": 0,
        "gp_fit_attempts": 0,
        "terminal_optimizations": 0,
        "truba_jobs_submitted": 0,
        "execution_authorized": False,
        "historical_evidence_modified": False,
        "tasks_identity": stable_hash(
            "repaired-sbi-assessment-task-list-v1",
            [task["task_identity"] for task in tasks],
        ),
    }
    return tasks, {
        **summary_body,
        "preparation_identity": stable_hash(PREPARATION_DOMAIN, summary_body),
    }


def _contents(root: Path) -> dict[str, str]:
    declaration = assessment_declaration()
    tasks, preparation = preparation_inventory(root, declaration)
    task_text = "".join(
        json.dumps(task, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for task in tasks
    )
    declaration_text = canonical_json_text(declaration)
    preparation_text = canonical_json_text(preparation)
    body = {
        "schema_version": PACKAGE_SCHEMA,
        "declaration_identity": declaration["declaration_identity"],
        "preparation_identity": preparation["preparation_identity"],
        "tasks_identity": preparation["tasks_identity"],
        "declaration_content_identity": stable_hash(
            "repaired-sbi-assessment-declaration-file-v1", declaration_text
        ),
        "tasks_content_identity": stable_hash(
            "repaired-sbi-assessment-tasks-file-v1", task_text
        ),
        "preparation_content_identity": stable_hash(
            "repaired-sbi-assessment-preparation-file-v1", preparation_text
        ),
        "task_count": 720,
        "maximum_future_assessment_calls": 720,
        "scientific_calls": 0,
        "assessment_execution_authorized": False,
        "truba_preparation_authorized": False,
        "truba_submission_authorized": False,
    }
    manifest = {**body, "package_identity": stable_hash(PACKAGE_DOMAIN, body)}
    return {
        "declaration.json": declaration_text,
        "assessment_tasks.jsonl": task_text,
        "preparation.json": preparation_text,
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
                raise PaperContractError(f"tracked assessment preparation differs: {name}")
            if (output / f"{name}.sha256").read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"assessment preparation sidecar differs: {name}")
        status = "reproduced"
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            path = output / name
            path.write_text(content, encoding="utf-8", newline="\n")
            (output / f"{name}.sha256").write_text(
                file_hash(path) + "\n", encoding="ascii", newline="\n"
            )
        declaration_path = root / DECLARATION_PATH
        declaration_path.parent.mkdir(parents=True, exist_ok=True)
        declaration_path.write_text(contents["declaration.json"], encoding="utf-8", newline="\n")
        status = "materialized"
    manifest = json.loads(contents["manifest.json"])
    return {
        "status": status,
        "output": OUTPUT,
        "package_identity": manifest["package_identity"],
        "task_count": 720,
        "maximum_future_assessment_calls": 720,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.common.repaired_sbi_assessment_preparation")
