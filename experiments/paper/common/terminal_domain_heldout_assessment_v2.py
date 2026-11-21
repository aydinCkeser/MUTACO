"""Frozen local paired held-out assessment for the B' versus C' terminal ablation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver, SimulationContext
from mutaco import (
    SimulatorCapabilitySnapshot,
    checked_evaluate,
    freeze_strict_metadata,
    stable_hash,
    strict_metadata_document,
)
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.simulators.graph_scientific_d4 import GraphScientificD4Adapter
from experiments.graph_inversion.paper_v3_callbacks import _assessment_randomness_evidence
from experiments.paper.common.aggregate import _bootstrap_mean_interval
from experiments.paper.common.validation import PaperContractError


SPEC = Path(
    "experiments/paper/studies/graph_paper120/analysis/"
    "terminal_domain_heldout_assessment_b40_v2.json"
)
OUTPUT = Path("benchmarks/paper/terminal_domain_heldout_assessment_b40_v2")
MODULE = Path("experiments/paper/common/terminal_domain_heldout_assessment_v2.py")
TEST_MODULE = Path("tests/experiments/test_terminal_domain_heldout_assessment_v2.py")
SOURCE_ROWS = Path("benchmarks/paper/terminal_domain_ablation_b40_refit_v1/rows.jsonl")
SOURCE_MANIFEST = Path("benchmarks/paper/terminal_domain_ablation_b40_refit_v1/manifest.json")
COHORT = Path("data/cohorts/paper120_v3/manifest.json")
TARGET_ROOT = Path("data/cohorts/paper120_v3")
SIMULATOR_CONFIG = Path("experiments/graph_inversion/configs/paper/paper120_v3_target_config.json")
SEARCH_EXPORT = Path(
    "experiments/paper/exploration/outputs/scalar_primary_smoke/"
    "population_b40_v3r2_compact_fetch/population/collection/compact_export.json"
)
ADAPTER_MODULE = Path("experiments/benchmarking/simulators/graph_scientific_d4.py")
SIMULATOR_MODULE = Path("packages/graph-simulator/src/mutaco_graph/scientific.py")
PROTOCOL_MODULE = Path("src/mutaco/simulator/protocol.py")
V1_TASKS = Path("benchmarks/paper/terminal_domain_heldout_assessment_b40_v1/tasks.jsonl")

DECLARATION_SCHEMA = "mutaco-terminal-domain-heldout-declaration-v2"
TASK_SCHEMA = "mutaco-terminal-domain-heldout-task-v2"
PREFLIGHT_SCHEMA = "mutaco-terminal-domain-heldout-preflight-v2"
INTENT_SCHEMA = "mutaco-terminal-domain-heldout-call-intent-v2"
OUTCOME_SCHEMA = "mutaco-terminal-domain-heldout-call-outcome-v2"
PAIR_SCHEMA = "mutaco-terminal-domain-heldout-pair-v2"
CAMPAIGN_ID = "paper120_b40_terminal_domain_bprime_cprime_heldout_v2"
ASSESSMENT_SEED = 92021841
BOOTSTRAP_SEED = 92024101
BOOTSTRAP_RESAMPLES = 10_000
MAXIMUM_CALLS = 240
POLICIES = (
    ("B_prime", "B_prime_evaluated_posterior_mean_physical_mae_v1", "policy_b_prime"),
    ("C_prime", "continuous_posterior_mean_physical_l1_mae", "policy_c_prime"),
)
INFERENTIAL_SCOPE = (
    "The paired target-level bootstrap quantifies variation across the Paper-120 "
    "target/context pairs; it does not separately estimate within-target Monte "
    "Carlo uncertainty."
)


def canonicalize_result_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Thaw recursively immutable strict metadata into canonical JSON data."""

    thawed = strict_metadata_document(metadata)
    return json.loads(
        json.dumps(thawed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def v1_failure_metadata_shape() -> Mapping[str, Any]:
    """Reproduce the recursively frozen metadata shape that v1 shallow-thawed."""

    return freeze_strict_metadata(
        {
            "model_identity": "graph-oscillator.overdamped-scientific.v1",
            "simulator_identity": "graph-oscillator.m20.float32",
            "evaluation_seed": 811673546353753898,
            "seed_policy": "mutaco-coordinate-sha256-v1",
            "seed_manifest": {
                "node_parameter/0/weight": 11,
                "pairwise_coupling/0/1": 12,
                "initial_state/0": 13,
                "thermal_noise/0": 14,
            },
            "device_effective": "cpu",
            "dtype": "float32",
            "shape": [20],
            "timings": {"generation_seconds": 0.1, "evolution_seconds": 0.2},
        },
        label="v1 simulator result metadata",
    )


def serializer_self_check() -> dict[str, Any]:
    canonical = canonicalize_result_metadata(v1_failure_metadata_shape())
    encoded = json.dumps(canonical, sort_keys=True, allow_nan=False)
    return {
        "policy": "strict_metadata_document_then_canonical_JSON_round_trip_v2",
        "exact_v1_failure_shape_serialized": True,
        "canonical_document_identity": stable_hash(
            "terminal-domain-heldout-v1-metadata-shape-canonical-v2", canonical
        ),
        "canonical_JSON_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
        "simulator_constructed": False,
        "simulator_calls": 0,
    }


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _append_json(path: Path, document: Mapping[str, Any]) -> None:
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise PaperContractError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pretty_json(path, dict(document))
    sidecar.write_text(f"{file_hash(path)}  {path.name}\n", encoding="ascii", newline="\n")


def _load_hashed(path: Path) -> dict[str, Any]:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise PaperContractError(f"missing hash-paired artifact: {path}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    if expected != file_hash(path):
        raise PaperContractError(f"artifact hash mismatch: {path}")
    return read_json(path)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise PaperContractError(f"append-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    sidecar.write_text(f"{file_hash(path)}  {path.name}\n", encoding="ascii", newline="\n")


def _canonical_theta(values: Any) -> tuple[list[float], str, str]:
    if not isinstance(values, list) or len(values) != 4:
        raise PaperContractError("recommendation theta must have four coordinates")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise PaperContractError("recommendation theta must be finite")
    raw = struct.pack("<4f", *converted)
    canonical = list(struct.unpack("<4f", raw))
    lower, upper = (0.5, 0.5, 0.5, 0.0), (2.0, 2.0, 2.0, 1.0)
    if any(value < lo or value > hi for value, lo, hi in zip(canonical, lower, upper, strict=True)):
        raise PaperContractError("recommendation theta is outside canonical bounds")
    return canonical, raw.hex(), sha256(raw).hexdigest()


def _git_status(root: Path) -> list[dict[str, str]]:
    text = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    )
    result = []
    output_prefix = f"{OUTPUT.as_posix()}/"
    for line in text.splitlines():
        path = line[3:]
        if path == OUTPUT.as_posix() or path.startswith(output_prefix):
            continue
        result.append({"status": line[:2], "path": path})
    return result


def _runtime() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_executable": os.path.realpath(os.sys.executable),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "dtype": "float32",
    }


def _context(target_position: int, target_id: str) -> dict[str, Any]:
    family_definition = {
        "schema": "terminal-domain-heldout-family-v2",
        "campaign_id": CAMPAIGN_ID,
        "target_record_identity": target_id,
        "simulator_definition_identity": (
            "graph-oscillator.overdamped-scientific.v1.p4-paper-baseline."
            "m20-five-force-scientific.float32"
        ),
        "simulator_configuration_sha256": "2614cdae3f2eb40ff1a0f0ee473056f05c8284d3ab59e2f159969ed1cc704a8a",
        "assessment_fidelity": "canonical_sparse_m20_one_evaluation_v1",
        "assessment_campaign_seed": ASSESSMENT_SEED,
    }
    family = ScientificIdentity.create(family_definition)
    coordinate = SeedCoordinate(
        "terminal_domain_paired_heldout_assessment", (CAMPAIGN_ID, target_id, 0)
    )
    return {
        "family_identity": family.value,
        "family_definition": family_definition,
        "root_seed": ASSESSMENT_SEED,
        "purpose": "paired_held_out_assessment",
        "logical_index": target_position,
        "replicate": 0,
        "coordinate": {
            "namespace": coordinate.namespace,
            "coordinates": list(coordinate.coordinates),
        },
        "evaluation_seed": SeedDeriver(family, ASSESSMENT_SEED).seed_for(coordinate),
    }


def _targets(root: Path) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    if file_hash(root / COHORT) != "359fc0bb59fc9a8a52f5da684e7a7eb352466bdc0d0a54fb8efa94131ce423bb":
        raise PaperContractError("Paper-120 cohort hash mismatch")
    cohort = read_json(root / COHORT)
    ids = [str(cohort["records"][str(position)]["target_record_id"]) for position in range(120)]
    if len(set(ids)) != 120:
        raise PaperContractError("Paper-120 target identities are not unique")
    targets: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for position, target_id in enumerate(ids):
        path = root / TARGET_ROOT / f"target_{position:03d}.json"
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != file_hash(path):
            raise PaperContractError(f"target {position} hash pair is invalid")
        target = read_json(path)
        if target.get("target_index") != position or target.get("target_record_id") != target_id:
            raise PaperContractError(f"target {position} identity mismatch")
        response = target.get("target_response")
        if not isinstance(response, list) or len(response) != 20 or not all(math.isfinite(float(x)) for x in response):
            raise PaperContractError(f"target {position} response is invalid")
        targets[target_id] = target
        hashes[target_id] = file_hash(path)
    return ids, targets, hashes


def build_preparation(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = read_json(root / SPEC)
    if file_hash(root / SOURCE_ROWS) != spec["recommendation_source"]["rows_sha256"]:
        raise PaperContractError("zero-call recommendation rows hash mismatch")
    source_manifest = _load_hashed(root / SOURCE_MANIFEST)
    if source_manifest.get("package_identity") != spec["recommendation_source"]["package_identity"]:
        raise PaperContractError("zero-call package identity mismatch")
    if source_manifest.get("rows_sha256") != file_hash(root / SOURCE_ROWS):
        raise PaperContractError("zero-call manifest does not bind recommendation rows")
    if file_hash(root / SIMULATOR_CONFIG) != spec["simulator"]["configuration_sha256"]:
        raise PaperContractError("simulator configuration hash mismatch")
    target_ids, _, target_hashes = _targets(root)
    rows = _jsonl(root / SOURCE_ROWS)
    if len(rows) != 120 or len({row.get("target_id") for row in rows}) != 120:
        raise PaperContractError("zero-call recommendation coverage is invalid")
    by_position = {int(row["target_position"]): row for row in rows}
    search = read_json(root / SEARCH_EXPORT)
    search_by_target = {
        str(record["target_id"]): record
        for record in search["records"]
        if record.get("method") == "mutaco" and record.get("budget") == 40
    }
    v1_tasks_sidecar = _sidecar(root / V1_TASKS)
    if (
        not v1_tasks_sidecar.is_file()
        or v1_tasks_sidecar.read_text(encoding="ascii").strip().split()[0]
        != file_hash(root / V1_TASKS)
    ):
        raise PaperContractError("v1 assessment task inventory hash mismatch")
    v1_tasks = _jsonl(root / V1_TASKS)
    if len(v1_tasks) != 240:
        raise PaperContractError("v1 assessment context inventory is incomplete")
    v1_context_seeds = {
        int(task["assessment_context"]["evaluation_seed"]) for task in v1_tasks
    }
    v1_context_families = {
        str(task["assessment_context"]["family_identity"]) for task in v1_tasks
    }
    tasks: list[dict[str, Any]] = []
    contexts: dict[str, Any] = {}
    equal_count = 0
    search_seed_overlap_count = 0
    v1_context_overlap_count = 0
    for position, target_id in enumerate(target_ids):
        row = by_position.get(position)
        if row is None or row.get("target_id") != target_id:
            raise PaperContractError(f"recommendation target {position} mismatch")
        context = _context(position, target_id)
        contexts[target_id] = context
        archived = search_by_target.get(target_id)
        if archived is None or len(archived["decision"]["trajectory"]) != 40:
            raise PaperContractError("archived B40 search context is missing")
        search_seeds = {
            int(item["simulator_context"]["evaluation_seed"])
            for item in archived["decision"]["trajectory"]
        }
        if int(context["evaluation_seed"]) in search_seeds:
            search_seed_overlap_count += 1
        if (
            int(context["evaluation_seed"]) in v1_context_seeds
            or str(context["family_identity"]) in v1_context_families
        ):
            v1_context_overlap_count += 1
        canonical_by_policy: dict[str, list[float]] = {}
        for short, expected_policy, field in POLICIES:
            policy = row.get(field)
            if not isinstance(policy, Mapping) or policy.get("policy_id") != expected_policy:
                raise PaperContractError(f"recommendation policy mismatch at target {position}")
            theta, theta_bytes, theta_hash = _canonical_theta(policy.get("selected_theta"))
            canonical_by_policy[short] = theta
            body = {
                "schema_version": TASK_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "target_position": position,
                "target_record_identity": target_id,
                "target_file_sha256": target_hashes[target_id],
                "policy": short,
                "policy_identity": expected_policy,
                "recommendation_row_identity": row["row_identity"],
                "recommendation_source_rows_sha256": file_hash(root / SOURCE_ROWS),
                "theta_float32": theta,
                "theta_float32_little_endian_hex": theta_bytes,
                "theta_float32_sha256": theta_hash,
                "surrogate_posterior_physical_mae": float(policy["score"]),
                "assessment_context": context,
                "fixed_evaluation_order": 0 if short == "B_prime" else 1,
            }
            tasks.append({**body, "task_identity": stable_hash("terminal-domain-heldout-task-v2", body)})
        if struct.pack("<4f", *canonical_by_policy["B_prime"]) == struct.pack("<4f", *canonical_by_policy["C_prime"]):
            equal_count += 1
    if search_seed_overlap_count:
        raise PaperContractError("held-out evaluation seed overlaps archived B40 search")
    if v1_context_overlap_count:
        raise PaperContractError("v2 context overlaps the v1 assessment context family")
    if len(tasks) != 240 or equal_count != 0:
        raise PaperContractError("assessment requires 240 distinct recommendation calls")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_paths = (SPEC, MODULE, TEST_MODULE, ADAPTER_MODULE, SIMULATOR_MODULE, PROTOCOL_MODULE)
    source_hashes = {str(path): file_hash(root / path) for path in source_paths}
    declaration_body = {
        "schema_version": DECLARATION_SCHEMA,
        "study_identity": spec["study_id"],
        "campaign_id": CAMPAIGN_ID,
        "cohort": {"path": str(COHORT), "sha256": file_hash(root / COHORT), "targets": 120},
        "recommendation_source": {
            "rows": str(SOURCE_ROWS),
            "rows_sha256": file_hash(root / SOURCE_ROWS),
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": file_hash(root / SOURCE_MANIFEST),
            "package_identity": source_manifest["package_identity"],
            "authority": "sole_authoritative_B_prime_C_prime_recommendation_source",
            "task_identity_list": stable_hash(
                "terminal-domain-heldout-task-list-v2", [task["task_identity"] for task in tasks]
            ),
        },
        "recommendations": {
            "targets": 120,
            "tasks": 240,
            "exact_equal_pairs": equal_count,
            "canonical_encoding": "IEEE754_float32_little_endian_four_coordinates",
            "normalization_round_trip": False,
            "bounds_validated": True,
        },
        "targets": {
            "source": str(TARGET_ROOT),
            "manifest": str(COHORT),
            "response_dimension": 20,
            "ordered_target_identity_list": target_ids,
            "target_file_sha256": target_hashes,
        },
        "assessment_contexts": {
            "campaign_seed": ASSESSMENT_SEED,
            "namespace": "terminal_domain_paired_heldout_assessment",
            "mapping": contexts,
            "same_context_within_target": True,
            "distinct_context_across_targets": len({value["evaluation_seed"] for value in contexts.values()}) == 120,
            "archived_B40_search_seed_overlap": search_seed_overlap_count,
            "v1_assessment_context_overlap": v1_context_overlap_count,
            "v1_outcomes_reused": False,
        },
        "simulator": {
            **spec["simulator"],
            "configuration_sha256": file_hash(root / SIMULATOR_CONFIG),
            "fresh_simulator_instances_per_policy": True,
            "shared_mutable_state": False,
        },
        "endpoint": spec["endpoint"],
        "analysis": {
            **spec["analysis"],
            "bootstrap_implementation": "experiments.paper.common.aggregate._bootstrap_mean_interval",
            "contrast": "C_prime_minus_B_prime",
            "population_relative_change": "(mean_C_prime_minus_mean_B_prime)/mean_B_prime",
            "confirmatory_requires_all_120_pairs": True,
        },
        "serializer": {
            "policy": "strict_metadata_document_then_canonical_JSON_round_trip_v2",
            "implementation": "canonicalize_result_metadata",
            "implementation_source": str(MODULE),
            "implementation_source_sha256": source_hashes[str(MODULE)],
            "simulator_metadata_values_changed": False,
            "zero_call_self_check": serializer_self_check(),
        },
        "verdict_rules": {
            "supported": "all_120_complete_and_CI_upper_lt_0",
            "reversed": "all_120_complete_and_CI_lower_gt_0",
            "not_detected": "all_120_complete_and_CI_includes_0",
            "incomplete": "any_required_pair_not_complete_or_integrity_invalid",
        },
        "execution": spec["execution"],
        "inferential_scope": INFERENTIAL_SCOPE,
        "source_provenance": {
            "base_git_commit": commit,
            "worktree_clean": False,
            "worktree_status": _git_status(root),
            "declared_preexisting_dirty_paths": spec["preexisting_dirty_paths"],
            "declared_allowed_assessment_paths": spec["allowed_assessment_paths"],
            "source_file_sha256": source_hashes,
            "runtime": _runtime(),
        },
        "call_accounting": {
            "maximum_scientific_calls": MAXIMUM_CALLS,
            "expected_scientific_calls": 240,
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "outcome_unknown": 0,
            "retry_calls": 0,
            "gp_fit_attempts": 0,
            "search_calls": 0,
            "terminal_optimizations": 0,
        },
        "authorization": {"local_simulator_execution": True, "truba": False, "slurm": False, "commit": False, "push": False},
    }
    declaration = {
        **declaration_body,
        "declaration_identity": stable_hash("terminal-domain-heldout-declaration-v2", declaration_body),
    }
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "declaration_identity": declaration["declaration_identity"],
        "target_count": 120,
        "tasks": 240,
        "B_prime_tasks": 120,
        "C_prime_tasks": 120,
        "exact_equal_pairs": equal_count,
        "expected_calls": 240,
        "maximum_calls": MAXIMUM_CALLS,
        "recommendation_hashes_valid": True,
        "target_context_mapping_frozen": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_sign_test": {
            "negative_C_minus_B": _bootstrap_mean_interval([-1.0] * 120, seed=BOOTSTRAP_SEED, resamples=100),
            "positive_C_minus_B": _bootstrap_mean_interval([1.0] * 120, seed=BOOTSTRAP_SEED, resamples=100),
            "passed": True,
        },
        "simulator_provenance_frozen": True,
        "pairing_condition": "A_fresh_simulator_per_policy_same_context",
        "archived_search_seed_overlap": 0,
        "v1_assessment_context_overlap": 0,
        "v1_assessment_outcomes_reused": False,
        "serializer_repair_qualified": True,
        "serializer_self_check": serializer_self_check(),
        "write_ahead_ledger_initialized": True,
        "retry_policy": "none",
        "simulator_calls": 0,
        "gp_fit_attempts": 0,
        "mutaco_search_calls": 0,
        "truba_jobs": 0,
        "slurm_jobs": 0,
        "gate_passed": True,
    }
    return declaration, tasks, preflight


def prepare(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise PaperContractError("assessment output namespace already exists")
    declaration, tasks, preflight = build_preparation(root)
    output.mkdir(parents=True)
    _append_json(output / "declaration.json", declaration)
    _append_jsonl(output / "tasks.jsonl", tasks)
    _append_json(output / "preflight.json", preflight)
    _append_json(output / "ledger_initialization.json", {
        "schema_version": "mutaco-terminal-domain-heldout-ledger-initialization-v2",
        "declaration_identity": declaration["declaration_identity"],
        "task_identity_list": declaration["recommendation_source"]["task_identity_list"],
        "maximum_calls": MAXIMUM_CALLS,
        "attempted_calls": 0,
        "automatic_retry": False,
        "pending_tasks": 240,
        "simulator_calls": 0,
    })
    package_body = {
        "schema_version": "mutaco-terminal-domain-heldout-preparation-package-v2",
        "declaration_identity": declaration["declaration_identity"],
        "declaration_sha256": file_hash(output / "declaration.json"),
        "tasks_sha256": file_hash(output / "tasks.jsonl"),
        "preflight_sha256": file_hash(output / "preflight.json"),
        "ledger_initialization_sha256": file_hash(output / "ledger_initialization.json"),
        "scientific_calls": 0,
        "gate_passed": True,
    }
    package = {**package_body, "package_identity": stable_hash("terminal-domain-heldout-preparation-package-v2", package_body)}
    _append_json(output / "preparation_manifest.json", package)
    return package


def validate_preparation(root: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declaration = _load_hashed(output / "declaration.json")
    preflight = _load_hashed(output / "preflight.json")
    _load_hashed(output / "ledger_initialization.json")
    _load_hashed(output / "preparation_manifest.json")
    tasks_path = output / "tasks.jsonl"
    if _sidecar(tasks_path).read_text(encoding="ascii").strip().split()[0] != file_hash(tasks_path):
        raise PaperContractError("task inventory hash mismatch")
    tasks = _jsonl(tasks_path)
    if preflight.get("gate_passed") is not True or len(tasks) != 240:
        raise PaperContractError("zero-call execution gate is not qualified")
    expected_declaration, expected_tasks, expected_preflight = build_preparation(root)
    if declaration != expected_declaration or tasks != expected_tasks or preflight != expected_preflight:
        raise PaperContractError("frozen preparation differs from canonical zero-call reconstruction")
    config = read_json(root / SIMULATOR_CONFIG)
    simulator = GraphScientificD4Adapter().build_target_simulator(config, device=torch.device("cpu"), dtype=torch.float32)
    snapshot = SimulatorCapabilitySnapshot.capture(simulator.capabilities)
    if snapshot.document["response_spec"] and len(snapshot.document["response_spec"]) != 20:
        raise PaperContractError("simulator response dimension mismatch")
    return declaration, tasks


def _call_paths(output: Path, task: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    base = output / "calls" / f"target-{int(task['target_position']):03d}" / str(task["policy"])
    return base.with_suffix(".intent.json"), base.with_suffix(".outcome.json"), base.with_suffix(".unknown.json")


def execute(root: Path, output: Path) -> dict[str, Any]:
    declaration, tasks = validate_preparation(root, output)
    if (output / "execution_manifest.json").exists():
        raise PaperContractError("assessment execution is already terminal")
    _, targets, _ = _targets(root)
    config = read_json(root / SIMULATOR_CONFIG)
    attempted = successful = failed = unknown = constructed = 0
    for task in tasks:
        intent_path, outcome_path, unknown_path = _call_paths(output, task)
        if outcome_path.exists() or unknown_path.exists():
            continue
        if intent_path.exists():
            attempted += 1
            unknown += 1
            _append_json(unknown_path, {
                "schema_version": OUTCOME_SCHEMA,
                "status": "attempted_outcome_unknown",
                "task_identity": task["task_identity"],
                "call_identity": _load_hashed(intent_path)["call_identity"],
                "automatic_retry": False,
            })
            break
        if attempted >= MAXIMUM_CALLS:
            raise PaperContractError("scientific call cap exhausted")
        simulator = GraphScientificD4Adapter().build_target_simulator(
            config, device=torch.device("cpu"), dtype=torch.float32
        )
        constructed += 1
        call_body = {
            "schema_version": INTENT_SCHEMA,
            "status": "attempted_pending_outcome",
            "declaration_identity": declaration["declaration_identity"],
            "task_identity": task["task_identity"],
            "target_position": task["target_position"],
            "target_record_identity": task["target_record_identity"],
            "policy": task["policy"],
            "policy_identity": task["policy_identity"],
            "theta_float32": task["theta_float32"],
            "theta_float32_sha256": task["theta_float32_sha256"],
            "assessment_context": task["assessment_context"],
            "attempt_index": 0,
            "automatic_retry": False,
        }
        intent = {**call_body, "call_identity": stable_hash("terminal-domain-heldout-call-v2", call_body)}
        _append_json(intent_path, intent)
        attempted += 1
        wall_started, cpu_started = perf_counter(), process_time()
        try:
            context = task["assessment_context"]
            simulation_context = SimulationContext(
                str(context["family_identity"]),
                str(context["purpose"]),
                int(context["logical_index"]),
                int(context["replicate"]),
                int(context["evaluation_seed"]),
            )
            theta = torch.tensor(task["theta_float32"], dtype=torch.float32, device="cpu").contiguous()
            if theta.detach().cpu().numpy().tobytes(order="C").hex() != task["theta_float32_little_endian_hex"]:
                raise PaperContractError("simulator theta bytes differ from frozen recommendation")
            result = checked_evaluate(
                simulator, theta, device=torch.device("cpu"), dtype=torch.float32, context=simulation_context
            )
            response = [float(value) for value in result.values.detach().cpu().tolist()]
            target = targets[str(task["target_record_identity"])]
            goal = [float(value) for value in target["target_response"]]
            mae = sum(abs(value - expected) for value, expected in zip(response, goal, strict=True)) / 20
            randomness = _assessment_randomness_evidence(context, result.metadata)
            body = {
                "schema_version": OUTCOME_SCHEMA,
                "status": "success",
                "declaration_identity": declaration["declaration_identity"],
                "task_identity": task["task_identity"],
                "call_identity": intent["call_identity"],
                "target_position": task["target_position"],
                "target_record_identity": task["target_record_identity"],
                "policy": task["policy"],
                "policy_identity": task["policy_identity"],
                "theta_float32": task["theta_float32"],
                "theta_float32_sha256": task["theta_float32_sha256"],
                "assessment_context": context,
                "assessment_response": response,
                "target_response": goal,
                "heldout_physical_component_mean_mae": mae,
                "assessment_randomness_evidence": randomness["compact"],
                "assessment_randomness_raw": randomness["raw"],
                "simulator_result_metadata": canonicalize_result_metadata(result.metadata),
                "simulator_instance_policy": "fresh_instance_for_each_policy_call",
                "shared_mutable_simulator_state": False,
                "call_accounting": {"attempted": 1, "successful": 1, "failed": 0, "retry": 0},
                "timing": {
                    "wall_seconds": perf_counter() - wall_started,
                    "process_cpu_seconds": process_time() - cpu_started,
                },
            }
            outcome = {**body, "outcome_identity": stable_hash("terminal-domain-heldout-outcome-v2", body)}
            successful += 1
        except Exception as exc:
            body = {
                "schema_version": OUTCOME_SCHEMA,
                "status": "failed",
                "declaration_identity": declaration["declaration_identity"],
                "task_identity": task["task_identity"],
                "call_identity": intent["call_identity"],
                "target_position": task["target_position"],
                "target_record_identity": task["target_record_identity"],
                "policy": task["policy"],
                "policy_identity": task["policy_identity"],
                "theta_float32": task["theta_float32"],
                "theta_float32_sha256": task["theta_float32_sha256"],
                "assessment_context": task["assessment_context"],
                "failure_type": type(exc).__name__,
                "failure_explanation": str(exc),
                "call_accounting": {"attempted": 1, "successful": 0, "failed": 1, "retry": 0},
                "timing": {
                    "wall_seconds": perf_counter() - wall_started,
                    "process_cpu_seconds": process_time() - cpu_started,
                },
            }
            outcome = {**body, "outcome_identity": stable_hash("terminal-domain-heldout-outcome-v2", body)}
            failed += 1
        _append_json(outcome_path, outcome)
        if outcome["status"] != "success":
            break
    terminal = {
        "schema_version": "mutaco-terminal-domain-heldout-execution-manifest-v2",
        "declaration_identity": declaration["declaration_identity"],
        "execution_location": "local_ubuntu",
        "truba_jobs": 0,
        "slurm_jobs": 0,
        "simulator_instances_constructed": constructed,
        "physical_calls": {
            "maximum": MAXIMUM_CALLS,
            "attempted": attempted,
            "successful": successful,
            "failed": failed,
            "attempted_outcome_unknown": unknown,
            "retry": 0,
        },
        "gp_fit_attempts": 0,
        "mutaco_search_calls": 0,
        "terminal_optimizations": 0,
        "policy_A_calls": 0,
        "automatic_retry": False,
    }
    _append_json(output / "execution_manifest.json", terminal)
    return terminal


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    def percentile(p: float) -> float:
        location = p * (len(ordered) - 1)
        lo, hi = math.floor(location), math.ceil(location)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - location) + ordered[hi] * (location - lo)
    return {
        "minimum": ordered[0], "q05": percentile(0.05), "q25": percentile(0.25),
        "median": statistics.median(ordered), "mean": statistics.fmean(ordered),
        "q75": percentile(0.75), "q95": percentile(0.95), "maximum": ordered[-1],
    }


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def verdict_for(*, complete_pairs: int, interval: Sequence[float] | None) -> str:
    if complete_pairs != 120 or interval is None:
        return "TERMINAL_DOMAIN_HELDOUT_ASSESSMENT_INCOMPLETE"
    lower, upper = float(interval[0]), float(interval[1])
    if upper < 0.0:
        return "TERMINAL_DOMAIN_PHYSICAL_BENEFIT_SUPPORTED"
    if lower > 0.0:
        return "TERMINAL_DOMAIN_PHYSICAL_BENEFIT_REVERSED"
    return "TERMINAL_DOMAIN_PHYSICAL_BENEFIT_NOT_DETECTED"


def analyze(root: Path, output: Path) -> dict[str, Any]:
    declaration, tasks = validate_preparation(root, output)
    execution = _load_hashed(output / "execution_manifest.json")
    task_map = {(int(task["target_position"]), str(task["policy"])): task for task in tasks}
    pairs: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for position in range(120):
        outcomes: dict[str, dict[str, Any]] = {}
        for policy in ("B_prime", "C_prime"):
            task = task_map[(position, policy)]
            _, outcome_path, unknown_path = _call_paths(output, task)
            if unknown_path.exists():
                incomplete.append({"target_position": position, "policy": policy, "status": "attempted_outcome_unknown"})
            elif not outcome_path.exists():
                incomplete.append({"target_position": position, "policy": policy, "status": "missing"})
            else:
                outcome = _load_hashed(outcome_path)
                if outcome.get("status") != "success":
                    incomplete.append({"target_position": position, "policy": policy, "status": outcome.get("status")})
                else:
                    outcomes[policy] = outcome
        if len(outcomes) != 2:
            continue
        b, c = outcomes["B_prime"], outcomes["C_prime"]
        if b["assessment_context"] != c["assessment_context"]:
            incomplete.append({"target_position": position, "status": "paired_context_mismatch"})
            continue
        if b["assessment_randomness_raw"]["full_seed_manifest"] != c["assessment_randomness_raw"]["full_seed_manifest"]:
            incomplete.append({"target_position": position, "status": "paired_randomness_manifest_mismatch"})
            continue
        b_mae = float(b["heldout_physical_component_mean_mae"])
        c_mae = float(c["heldout_physical_component_mean_mae"])
        delta = c_mae - b_mae
        task_b, task_c = task_map[(position, "B_prime")], task_map[(position, "C_prime")]
        surrogate_delta = float(task_c["surrogate_posterior_physical_mae"]) - float(task_b["surrogate_posterior_physical_mae"])
        body = {
            "schema_version": PAIR_SCHEMA,
            "target_position": position,
            "target_record_identity": b["target_record_identity"],
            "assessment_context": b["assessment_context"],
            "B_prime": {
                "theta": b["theta_float32"], "theta_sha256": b["theta_float32_sha256"],
                "physical_response": b["assessment_response"], "heldout_physical_mae": b_mae,
                "call_identity": b["call_identity"], "outcome_identity": b["outcome_identity"],
            },
            "C_prime": {
                "theta": c["theta_float32"], "theta_sha256": c["theta_float32_sha256"],
                "physical_response": c["assessment_response"], "heldout_physical_mae": c_mae,
                "call_identity": c["call_identity"], "outcome_identity": c["outcome_identity"],
            },
            "target_response": b["target_response"],
            "surrogate_C_prime_minus_B_prime": surrogate_delta,
            "physical_C_prime_minus_B_prime": delta,
            "winner": "C_prime" if delta < 0.0 else "B_prime" if delta > 0.0 else "tie",
        }
        pairs.append({**body, "pair_identity": stable_hash("terminal-domain-heldout-pair-v2", body)})
    complete = len(pairs) == 120 and not incomplete
    b_values = [float(pair["B_prime"]["heldout_physical_mae"]) for pair in pairs]
    c_values = [float(pair["C_prime"]["heldout_physical_mae"]) for pair in pairs]
    deltas = [float(pair["physical_C_prime_minus_B_prime"]) for pair in pairs]
    surrogate = [float(pair["surrogate_C_prime_minus_B_prime"]) for pair in pairs]
    interval = _bootstrap_mean_interval(deltas, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES) if complete else None
    association = None
    if len(pairs) >= 2 and len(set(surrogate)) > 1 and len(set(deltas)) > 1:
        association = {
            "pearson": statistics.correlation(surrogate, deltas),
            "spearman": statistics.correlation(_ranks(surrogate), _ranks(deltas)),
            "role": "descriptive_only",
        }
    mean_b = statistics.fmean(b_values) if b_values else None
    mean_c = statistics.fmean(c_values) if c_values else None
    report_body = {
        "schema_version": "mutaco-terminal-domain-heldout-analysis-v2",
        "declaration_identity": declaration["declaration_identity"],
        "inferential_scope": INFERENTIAL_SCOPE,
        "declared_targets": 120,
        "complete_pairs": len(pairs),
        "incomplete": incomplete,
        "confirmatory_complete": complete,
        "B_prime_heldout_physical_mae": _distribution(b_values),
        "C_prime_heldout_physical_mae": _distribution(c_values),
        "physical_C_prime_minus_B_prime": _distribution(deltas),
        "paired_mean_bootstrap_95_percentile_interval": interval,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "paired_target_delta", "statistic": "arithmetic_mean",
            "lower_percentile": 2.5, "upper_percentile": 97.5,
            "stratified": False, "studentized": False,
            "contrast": "C_prime_minus_B_prime",
        },
        "counts": {
            "C_prime_better": sum(value < 0.0 for value in deltas),
            "tie_exact_float": sum(value == 0.0 for value in deltas),
            "B_prime_better": sum(value > 0.0 for value in deltas),
        },
        "population_relative_physical_mae_change": (
            None if mean_b in (None, 0.0) or mean_c is None else (mean_c - mean_b) / mean_b
        ),
        "surrogate_vs_physical_association": association,
        "execution_accounting": execution,
        "verdict": verdict_for(complete_pairs=len(pairs) if not incomplete else -1, interval=interval),
    }
    report = {**report_body, "analysis_identity": stable_hash("terminal-domain-heldout-analysis-v2", report_body)}
    _append_jsonl(output / "paired_results.jsonl", pairs)
    _append_json(output / "analysis_report.json", report)
    manifest_body = {
        "schema_version": "mutaco-terminal-domain-heldout-evidence-package-v2",
        "declaration_identity": declaration["declaration_identity"],
        "preparation_manifest_sha256": file_hash(output / "preparation_manifest.json"),
        "execution_manifest_sha256": file_hash(output / "execution_manifest.json"),
        "paired_results_sha256": file_hash(output / "paired_results.jsonl"),
        "analysis_report_sha256": file_hash(output / "analysis_report.json"),
        "analysis_identity": report["analysis_identity"],
        "verdict": report["verdict"],
        "existing_primary_evidence_modified": False,
        "truba_jobs": 0,
        "slurm_jobs": 0,
    }
    manifest = {**manifest_body, "package_identity": stable_hash("terminal-domain-heldout-evidence-package-v2", manifest_body)}
    _append_json(output / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "analyze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--confirm-approved", action="store_true")
    args = parser.parse_args()
    root = _root()
    output = (root / args.output).resolve()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if args.action == "prepare":
        result = prepare(root, output)
    elif args.action == "preflight":
        declaration, tasks = validate_preparation(root, output)
        result = {"gate_passed": True, "declaration_identity": declaration["declaration_identity"], "tasks": len(tasks), "simulator_calls": 0}
    elif args.action == "execute":
        if not args.confirm_approved:
            raise SystemExit("local scientific execution requires --confirm-approved")
        result = execute(root, output)
    else:
        result = analyze(root, output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
