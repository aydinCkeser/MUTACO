"""Zero-call declaration for corrected MUTACO terminal generation."""

from __future__ import annotations

from reproducibility.recorded_paths import resolve_recorded_path

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json

from .cross_experiment_contract import PAPER120_COHORT_SHA256
from .evidence_analysis import canonical_json_text
from .repaired_sbi_recovery import QUALIFICATION_POSITIONS
from .repaired_sbi_salvage import (
    B40_COMMIT,
    B40_COMPACT,
    B40_RAW,
    B100_COMPACT,
    B100_POPULATION_COMMIT,
    B100_QUALIFICATION_COMMIT,
    CORRECTED_TERMINAL_BLOB,
    CORRECTION_COMMIT,
    DEFECTIVE_TERMINAL_BLOB,
    RECOVERY_EVIDENCE,
    REPLAY_QUALIFICATION,
    TERMINAL_PATH,
)
from .validation import PaperContractError


DECLARATION_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-declaration-v1"
INVENTORY_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-preparation-v1"
PACKAGE_SCHEMA = "mutaco-repaired-sbi-corrected-terminal-preparation-package-v1"
DECLARATION_DOMAIN = "repaired-sbi-corrected-terminal-declaration-v1"
INVENTORY_DOMAIN = "repaired-sbi-corrected-terminal-preparation-v1"
PACKAGE_DOMAIN = "repaired-sbi-corrected-terminal-preparation-package-v1"

OUTPUT = "results/paper/repaired_mutaco_sbi_terminal_preparation/v1"
DECLARATION_PATH = (
    "experiments/paper/studies/graph_paper120/analysis/"
    "repaired_mutaco_sbi_corrected_terminal_v1.json"
)
RECOVERY_MANIFEST = (
    "benchmarks/paper/repaired_mutaco_sbi_recovery_qualification/v1/manifest.json"
)
GENERATOR_PATH = "experiments/paper/common/exact_terminal_generation.py"
BOOTSTRAP_PATH = "experiments/paper/common/exact_terminal_bootstrap.py"
PHYSICAL_MAE_PATH = "src/mutaco/recommendation/physical_mae.py"
PHYSICAL_MAE_BLOB = "a5bfa2793742db9f6b195fcdfd6f4e7615e786c5"
B100_RECOVERY_ROOT = (
    "experiments/paper/exploration/outputs/repaired_mutaco_sbi_salvage/"
    "b100_raw_recovery"
)

RECOVERY_PACKAGE_IDENTITY = (
    "repaired-sbi-recovery-qualification-package-v1-"
    "59b60902132c611f80695f52cddc061c53570a26de2fb41f09f0759897c55890"
)
QUALIFICATION_IDENTITY = (
    "repaired-sbi-exact-state-qualification-v1-"
    "843aeb4d8cae6bb30d79ce7b660056386f2f8f43443ef67953607870f1c9cb6d"
)
SALVAGE_DESIGN_IDENTITY = (
    "repaired-sbi-salvage-design-v1-"
    "5283c87e6f5fedfc8d611682769ca0a8b1bbdd184b26110678f2fd74ceaa2a9b"
)


def _git_blob(root: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _composite_source_identity(native_commit: str) -> str:
    return stable_hash(
        "repaired-sbi-native-plus-terminal-overlay-v1",
        {
            "native_commit": native_commit,
            "overlay_path": TERMINAL_PATH,
            "replaced_blob": DEFECTIVE_TERMINAL_BLOB,
            "corrected_blob": CORRECTED_TERMINAL_BLOB,
            "unchanged_physical_mae_blob": PHYSICAL_MAE_BLOB,
        },
    )


def corrected_terminal_declaration() -> dict[str, Any]:
    source_commits = {
        "b40_all": B40_COMMIT,
        "b100_qualification_positions": B100_QUALIFICATION_COMMIT,
        "b100_remaining_positions": B100_POPULATION_COMMIT,
    }
    body = {
        "schema_version": DECLARATION_SCHEMA,
        "declaration_id": "paper120_repaired_mutaco_b40_b100_corrected_terminal_v1",
        "salvage_design_identity": SALVAGE_DESIGN_IDENTITY,
        "recovery_package_identity": RECOVERY_PACKAGE_IDENTITY,
        "exact_state_qualification_identity": QUALIFICATION_IDENTITY,
        "cohort_sha256": PAPER120_COHORT_SHA256,
        "targets": 120,
        "budgets": [40, 100],
        "terminal_cases": 240,
        "source_policy": {
            "kind": "clean_native_source_plus_single_in_memory_terminal_overlay_v1",
            "native_commits": source_commits,
            "b100_qualification_positions": list(QUALIFICATION_POSITIONS),
            "overlay_path": TERMINAL_PATH,
            "defective_blob": DEFECTIVE_TERMINAL_BLOB,
            "corrected_blob": CORRECTED_TERMINAL_BLOB,
            "correction_commit": CORRECTION_COMMIT,
            "physical_mae_path": PHYSICAL_MAE_PATH,
            "unchanged_physical_mae_blob": PHYSICAL_MAE_BLOB,
            "all_other_files": "native_commit_unchanged",
            "composite_identities": {
                name: _composite_source_identity(commit)
                for name, commit in source_commits.items()
            },
        },
        "model_policy": {
            "method": "exact_source_model_construction_plus_archived_selected_parameter_state",
            "optimizer_rerun": False,
            "required_prior_gate": "240_of_240_exact_state_replays_passed",
            "required_equalities": [
                "response_transform_identity",
                "standardized_observations_bitwise",
                "final_fit_configuration",
                "final_training_data_identity",
                "materialized_final_model_identity",
            ],
            "additional_gp_fit_attempts": 0,
        },
        "terminal_policy": {
            "policy_id": "continuous_posterior_mean_physical_l1_mae",
            "candidate_domain": "continuous_bounded_theta",
            "objective": "mean_m_abs_mu_physical_m_theta_minus_target_physical_m",
            "dtype": "float32",
            "device": "cpu",
            "raw_samples": 1024,
            "restarts": 10,
            "max_iterations": 64,
            "learning_rate": 0.05,
            "optimizer": "projected_adam",
            "convergence_tolerance": 1.0e-7,
            "randomness": {
                "seed_policy": "mutaco-coordinate-sha256-v1",
                "namespace": "terminal_recommendation",
                "coordinates": ["target_index", "policy_id"],
                "context": "reuse_immutable_historical_terminal_context",
                "ambient_rng_isolation": True,
            },
            "corrected_behavior": "score_post_update_state_retain_best_accepted_iterate",
            "simulator_calls": 0,
        },
        "immutable_sbi_recommendations": {
            "methods": ["sbi_npe", "sbi_snpe"],
            "b40_source": B40_COMPACT,
            "b40_source_sha256": "cd0352310c8e570bdc56fe6e4ef587bd58e1c26f1f0865d44878ffc7dd2c0c72",
            "b100_source": B100_COMPACT,
            "b100_source_sha256": "29f6af90cdec9581dd3b850636c05c6dd740dac2aa22164c7f4bc68b3ab61f67",
            "regeneration": False,
        },
        "evidence_policy": {
            "append_only_attempts": True,
            "one_successful_recommendation_per_target_budget": True,
            "canonical_json_allow_nan": False,
            "semantic_identity_excludes_elapsed_timings": True,
            "failure_attempts_preserved": True,
            "freeze_all_240_before_heldout_assessment": True,
        },
        "call_accounting": {
            "simulator_constructed": False,
            "simulator_calls": 0,
            "search_calls": 0,
            "gp_fit_attempts": 0,
            "terminal_optimizations_planned": 240,
            "heldout_assessment_calls": 0,
        },
        "authorization": {
            "terminal_generation": False,
            "heldout_assessment": False,
            "simulator_execution": False,
            "truba_submission": False,
        },
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
    return {**body, "declaration_identity": stable_hash(DECLARATION_DOMAIN, body)}


def validate_corrected_terminal_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = corrected_terminal_declaration()
    if dict(document) != expected:
        raise PaperContractError("corrected-terminal declaration is not canonical")
    return expected


def _raw_paths(root: Path, budget: int) -> dict[int, Path]:
    if budget == 40:
        candidates = (root / B40_RAW).glob("worker-*/raw/target-*/mutaco.json")
    else:
        candidates = (root / B100_RECOVERY_ROOT).glob("*/worker-*/raw/target-*/mutaco.json")
    result: dict[int, Path] = {}
    for path in candidates:
        row = read_json(path)
        position = row.get("target_index")
        if not isinstance(position, int) or position in result:
            raise PaperContractError(f"terminal raw archive coverage is invalid for B{budget}")
        result[position] = path
    if set(result) != set(range(120)):
        raise PaperContractError(f"terminal raw archive is incomplete for B{budget}")
    return result


def preparation_inventory(repository_root: Path, declaration: Mapping[str, Any]) -> dict[str, Any]:
    expected = validate_corrected_terminal_declaration(declaration)
    root = repository_root.resolve()
    recovery_manifest = read_json(resolve_recorded_path(root, RECOVERY_MANIFEST))
    qualification = read_json(resolve_recorded_path(root, REPLAY_QUALIFICATION))
    recovery = read_json(resolve_recorded_path(root, RECOVERY_EVIDENCE))
    if recovery_manifest.get("package_identity") != RECOVERY_PACKAGE_IDENTITY:
        raise PaperContractError("corrected-terminal recovery package identity mismatch")
    if qualification.get("qualification_identity") != QUALIFICATION_IDENTITY:
        raise PaperContractError("corrected-terminal replay qualification identity mismatch")
    if qualification.get("cases_passed") != 240 or qualification.get("cases_failed") != 0:
        raise PaperContractError("corrected-terminal replay matrix is not complete")

    for commit in (B40_COMMIT, B100_QUALIFICATION_COMMIT, B100_POPULATION_COMMIT):
        if _git_blob(root, commit, TERMINAL_PATH) != DEFECTIVE_TERMINAL_BLOB:
            raise PaperContractError("native terminal blob mismatch")
        if _git_blob(root, commit, PHYSICAL_MAE_PATH) != PHYSICAL_MAE_BLOB:
            raise PaperContractError("native physical-MAE policy blob mismatch")
    if _git_blob(root, CORRECTION_COMMIT, TERMINAL_PATH) != CORRECTED_TERMINAL_BLOB:
        raise PaperContractError("corrected terminal overlay blob mismatch")

    compact_sources = expected["immutable_sbi_recommendations"]
    for budget in (40, 100):
        prefix = f"b{budget}"
        path = resolve_recorded_path(root, str(compact_sources[f"{prefix}_source"]))
        if file_hash(path) != compact_sources[f"{prefix}_source_sha256"]:
            raise PaperContractError(f"immutable SBI B{budget} source hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 120 or any(
            set(row.get("methods", {})) != {"mutaco", "sbi_npe", "sbi_snpe"}
            or any(len(row["methods"][method].get("recommended_theta", ())) != 4 for method in ("sbi_npe", "sbi_snpe"))
            for row in rows
        ):
            raise PaperContractError(f"immutable SBI B{budget} recommendation matrix is invalid")

    replay_cases = {
        (int(case["budget"]), int(case["target_position"])): case
        for case in qualification["cases"]
    }
    recovery_hashes = {
        int(entry["target_position"]): str(entry["raw_sha256"])
        for entry in recovery["entries"]
    }
    cases: list[dict[str, Any]] = []
    for budget in (40, 100):
        paths = _raw_paths(root, budget)
        for position in range(120):
            replay = replay_cases[(budget, position)]
            raw_path = paths[position]
            digest = file_hash(raw_path)
            if budget == 100 and recovery_hashes.get(position) != digest:
                raise PaperContractError("B100 terminal raw hash differs from recovery evidence")
            raw = read_json(raw_path)
            context = raw["runner_provenance"]["configuration"]["execution"]["terminal_recommendation_context"]
            source_commit = str(replay["source_commit"])
            if (
                raw.get("target_record_id") != replay["target_record_id"]
                or context.get("target_id") != replay["target_record_id"]
                or context.get("target_index") != position
                or context.get("policy_id") != expected["terminal_policy"]["policy_id"]
                or raw.get("inference_calls") != budget
            ):
                raise PaperContractError("corrected-terminal raw context mismatch")
            cases.append(
                {
                    "budget": budget,
                    "target_position": position,
                    "target_record_id": replay["target_record_id"],
                    "native_source_commit": source_commit,
                    "composite_source_identity": _composite_source_identity(source_commit),
                    "raw_artifact": raw_path.relative_to(root).as_posix(),
                    "raw_sha256": digest,
                    "exact_replay_identity": replay["replay_identity"],
                    "final_training_data_id": replay["final_training_data_id"],
                    "final_model_id": replay["final_model_id"],
                    "terminal_context": {
                        key: context[key]
                        for key in (
                            "policy_id",
                            "root_seed",
                            "seed_family_id",
                            "seed_policy",
                            "target_id",
                            "target_index",
                        )
                    },
                }
            )

    body = {
        "schema_version": INVENTORY_SCHEMA,
        "declaration_identity": expected["declaration_identity"],
        "recovery_package_identity": RECOVERY_PACKAGE_IDENTITY,
        "exact_state_qualification_identity": QUALIFICATION_IDENTITY,
        "cases": cases,
        "case_count": len(cases),
        "b40_cases": 120,
        "b100_cases": 120,
        "exact_state_cases_passed": 240,
        "immutable_sbi_recommendation_rows": 480,
        "terminal_recommendations_generated": 0,
        "heldout_assessments_executed": 0,
        "simulator_calls": 0,
        "truba_jobs_submitted": 0,
        "generation_ready": True,
    }
    return {**body, "preparation_identity": stable_hash(INVENTORY_DOMAIN, body)}


def _contents(root: Path) -> dict[str, str]:
    declaration = corrected_terminal_declaration()
    inventory = preparation_inventory(root, declaration)
    declaration_text = canonical_json_text(declaration)
    inventory_text = canonical_json_text(inventory)
    body = {
        "schema_version": PACKAGE_SCHEMA,
        "declaration_identity": declaration["declaration_identity"],
        "preparation_identity": inventory["preparation_identity"],
        "declaration_content_identity": stable_hash(
            "repaired-sbi-corrected-terminal-declaration-file-v1", declaration_text
        ),
        "preparation_content_identity": stable_hash(
            "repaired-sbi-corrected-terminal-preparation-file-v1", inventory_text
        ),
        "generator_path": GENERATOR_PATH,
        "generator_sha256": file_hash(root / GENERATOR_PATH),
        "bootstrap_path": BOOTSTRAP_PATH,
        "bootstrap_sha256": file_hash(root / BOOTSTRAP_PATH),
        "terminal_cases_ready": 240,
        "terminal_recommendations_generated": 0,
        "simulator_calls": 0,
        "truba_jobs_submitted": 0,
        "generation_authorized": False,
    }
    manifest = {**body, "package_identity": stable_hash(PACKAGE_DOMAIN, body)}
    return {
        "declaration.json": declaration_text,
        "preparation_inventory.json": inventory_text,
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
                raise PaperContractError(f"tracked terminal preparation differs: {name}")
            if (output / f"{name}.sha256").read_text(encoding="ascii") != file_hash(path) + "\n":
                raise PaperContractError(f"tracked terminal preparation sidecar differs: {name}")
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
        "terminal_cases_ready": 240,
        "terminal_recommendations_generated": 0,
        "scientific_calls": 0,
        "truba_jobs_submitted": 0,
    }
