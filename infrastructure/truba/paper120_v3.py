"""Canonical Paper-120 declaration and zero-call collection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.graph_inversion.paper_v3_callbacks import method_configuration_identity
from experiments.paper.common.population_v3 import (
    MethodDeclarationV3,
    PopulationDeclarationV3,
    shard_indices,
)
from experiments.paper.common.protocol_v3 import PaperComparisonConfigV3
from experiments.paper.common.validation import PaperContractError


TARGET_COUNT = 120
WORKER_COUNT = 40
TARGETS_PER_WORKER = 3
CALLS_PER_TARGET = 82
CALLS_PER_WORKER = TARGETS_PER_WORKER * CALLS_PER_TARGET
GLOBAL_CALL_CAP = TARGET_COUNT * CALLS_PER_TARGET
BOOTSTRAP_SEED = 92020120


def repository_paths(root: Path) -> dict[str, Path]:
    return {
        "target_ledger": root / "experiments/graph_inversion/frozen_targets/paper120_v3",
        "mutaco_config": root / "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json",
        "npe_config": root / "experiments/graph_inversion/configs/d4_100_targets.json",
    }


def canonical_declaration(root: Path) -> dict[str, Any]:
    """Derive the exact declaration from the frozen ledger and accepted configs."""

    paths = repository_paths(root)
    manifest_path = paths["target_ledger"] / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("completed_count") != TARGET_COUNT or manifest.get("missing_count") != 0:
        raise PaperContractError("Paper-120 target ledger is not complete")
    if manifest.get("selected_indices") != list(range(TARGET_COUNT)):
        raise PaperContractError("Paper-120 target ledger indices are not canonical")
    records = manifest.get("records")
    if not isinstance(records, Mapping) or set(records) != {str(i) for i in range(TARGET_COUNT)}:
        raise PaperContractError("Paper-120 target ledger record map is incomplete")
    target_identities = tuple(str(records[str(i)]["target_record_id"]) for i in range(TARGET_COUNT))
    comparison = PaperComparisonConfigV3(
        simulator_definition_identity=(
            "graph-oscillator.overdamped-scientific.v1.p4-paper-baseline."
            "m20-five-force-scientific.float32"
        ),
        simulator_configuration_identity=(
            "graph-paper-v3-simulator-configuration-v1-"
            "ff70b08584d86189e7464b20602142d9faeca01f588c0d691de6cdf2c7568a08"
        ),
        assessment_fidelity="canonical_sparse_m20_one_evaluation_v1",
        assessment_campaign_seed=92020402,
        inference_campaign_seed=92020401,
    )
    methods = (
        MethodDeclarationV3(
            "mutaco",
            method_configuration_identity("mutaco", paths["mutaco_config"]),
            "continuous_posterior_mean_physical_l1_mae",
            40,
        ),
        MethodDeclarationV3(
            "sbi_npe",
            method_configuration_identity("sbi_npe", paths["npe_config"]),
            "posterior_sample_map_physical_v1",
            40,
        ),
    )
    return PopulationDeclarationV3(
        campaign_id="graph_paper120_b40_mutaco_npe_v3_v1",
        comparison=comparison,
        target_ledger_hash=file_hash(manifest_path),
        target_record_identities=target_identities,
        methods=methods,
        output_root="runs/paper/graph_oscillator/graph_paper120_b40_mutaco_npe_v3_v1",
    ).document()


def canonical_shards() -> tuple[tuple[int, ...], ...]:
    shards = tuple(
        shard_indices(TARGET_COUNT, rank=rank, workers=WORKER_COUNT)
        for rank in range(WORKER_COUNT)
    )
    if any(len(shard) != TARGETS_PER_WORKER for shard in shards):
        raise PaperContractError("Paper-120 requires exactly three targets per worker")
    if sorted(index for shard in shards for index in shard) != list(range(TARGET_COUNT)):
        raise PaperContractError("Paper-120 shards do not cover the cohort exactly once")
    return shards
