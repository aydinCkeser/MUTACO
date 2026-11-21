from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import file_hash
from experiments.paper.protocol.method_append import record_path
from experiments.paper.common.population_v3 import (
    load_population_declaration_v3,
    population_call_preview,
    shard_indices,
)
from experiments.paper.common.validation import PaperContractError
from infrastructure.truba.collect_paper120_v3 import _coverage_errors, collect
import infrastructure.truba.collect_paper120_v3 as collector_module
from infrastructure.truba.paper120_v3 import (
    CALLS_PER_WORKER,
    GLOBAL_CALL_CAP,
    TARGET_COUNT,
    WORKER_COUNT,
    canonical_declaration,
    canonical_shards,
)


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/paper120_b40_mutaco_npe_v3_v1.json"
REMAINING_SNPE_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/paper120_b40_snpe_remaining117_v1.json"


def test_paper120_declaration_is_exactly_derived_from_frozen_inputs():
    document = json.loads(DECLARATION.read_text(encoding="utf-8"))
    assert document == canonical_declaration(ROOT)
    declaration = load_population_declaration_v3(document)
    assert len(declaration.target_record_identities) == TARGET_COUNT == 120
    assert declaration.comparison.inference_campaign_seed == 92020401
    assert declaration.comparison.assessment_campaign_seed == 92020402
    assert [method.inference_budget for method in declaration.methods] == [40, 40]
    assert document["call_preview"]["maximum_total_calls"] == GLOBAL_CALL_CAP == 9840
    assert DECLARATION.with_suffix(".sha256").read_text().strip() == file_hash(DECLARATION)


def test_paper120_shards_cover_once_at_exact_caps():
    shards = canonical_shards()
    assert len(shards) == WORKER_COUNT == 40
    assert all(len(shard) == 3 for shard in shards)
    assert sorted(index for shard in shards for index in shard) == list(range(TARGET_COUNT))
    assert CALLS_PER_WORKER == 246
    assert WORKER_COUNT * CALLS_PER_WORKER == GLOBAL_CALL_CAP


def test_remaining_snpe_declaration_excludes_qualified_targets_at_exact_caps():
    baseline = load_population_declaration_v3(json.loads(DECLARATION.read_text()))
    remaining = load_population_declaration_v3(
        json.loads(REMAINING_SNPE_DECLARATION.read_text())
    )
    excluded = {baseline.target_record_identities[index] for index in (0, 49, 99)}
    assert len(remaining.target_record_identities) == 117
    assert set(remaining.target_record_identities) == set(baseline.target_record_identities) - excluded
    assert [method.method for method in remaining.methods] == ["sbi_snpe"]
    assert remaining.document()["call_preview"]["maximum_total_calls"] == 4797
    shards = tuple(shard_indices(117, rank=rank, workers=39) for rank in range(39))
    assert all(len(shard) == 3 for shard in shards)
    assert population_call_preview(3, remaining.methods)["maximum_total_calls"] == 123


def test_global_coverage_validation_rejects_missing_and_duplicate_slots():
    targets, methods = ("a", "b"), ("mutaco", "sbi_npe")
    complete = [
        {"target_record_identity": target, "method": method}
        for target in targets for method in methods
    ]
    assert _coverage_errors(complete, targets, methods) == []
    assert _coverage_errors(complete[:-1], targets, methods)[0]["failure_type"] == "PopulationCoverageError"
    assert _coverage_errors(complete + complete[:1], targets, methods)[0]["failure_type"] == "PopulationCoverageError"


def test_empty_collection_preserves_failure_and_never_writes_summary(tmp_path):
    output = tmp_path / "collection"
    with pytest.raises(PaperContractError, match="preserved population_collection"):
        collect(
            root=ROOT,
            declaration_path=DECLARATION,
            workers_root=tmp_path / "workers",
            output=output,
            expected_commit="a" * 40,
        )
    failure = json.loads((output / "population_collection.json").read_text())
    assert failure["status"] == "failed"
    assert failure["collected_slot_count"] == 0
    assert failure["scientific_calls_added_by_collection"] == 0
    assert not (output / "population_summary.json").exists()


@pytest.mark.parametrize(
    ("declaration_path", "worker_count", "expected_slots", "expected_calls"),
    (
        (DECLARATION, 40, 240, 9840),
        (REMAINING_SNPE_DECLARATION, 39, 117, 4797),
    ),
)
def test_global_collector_accepts_exact_complete_worker_population(
    tmp_path, monkeypatch, declaration_path, worker_count, expected_slots, expected_calls
):
    declaration = load_population_declaration_v3(json.loads(declaration_path.read_text()))
    declaration_sha = file_hash(declaration_path)
    commit = "b" * 40
    methods = tuple(method.method for method in declaration.methods)
    shards = tuple(
        shard_indices(len(declaration.target_record_identities), rank=rank, workers=worker_count)
        for rank in range(worker_count)
    )
    record_lookup = {}
    for rank, positions in enumerate(shards):
        worker = tmp_path / "workers" / f"worker-{rank:02d}"
        for position in positions:
            target = declaration.target_record_identities[position]
            for method in methods:
                record_lookup[record_path(worker, target, method)] = {
                    "target_record_identity": target,
                    "method": method,
                    "status": "completed",
                }

    def fake_hashed(path):
        rank = int(path.parent.name.removeprefix("worker-"))
        positions = shards[rank]
        worker_calls = population_call_preview(len(positions), declaration.methods)[
            "maximum_total_calls"
        ]
        calls = {
            "maximum_calls": worker_calls,
            "attempted_calls": worker_calls,
            "successful_calls": worker_calls,
            "failed_calls": 0,
        }
        if path.name == "worker_start.json":
            return {
                "source_commit": commit,
                "declaration_sha256": declaration_sha,
                "preview": {
                    "rank": rank,
                    "workers": worker_count,
                    "target_positions": list(positions),
                    "maximum_calls_argument": worker_calls,
                },
            }
        if path.name == "worker_terminal.json":
            return {
                "source_commit": commit,
                "declaration_sha256": declaration_sha,
                "status": "completed",
                "physical_calls": calls,
            }
        if path.name == "population_execution_attempt.json":
            return {"status": "completed", "physical_calls": calls}
        if path.name == "population_collection.json":
            return {
                "complete": True,
                "declared_slot_count": len(positions) * len(methods),
            }
        raise AssertionError(path)

    monkeypatch.setattr(collector_module, "_load_hashed", fake_hashed)
    monkeypatch.setattr(collector_module, "load_method_record", lambda path: record_lookup.get(path))
    monkeypatch.setattr(
        collector_module,
        "summarize_population_v3",
        lambda *args, **kwargs: {"schema_version": "test-summary", "scientific_calls": 0},
    )
    output = tmp_path / "collection"
    result = collect(
        root=ROOT,
        declaration_path=declaration_path,
        workers_root=tmp_path / "workers",
        output=output,
        expected_commit=commit,
        workers=worker_count,
    )
    assert result["status"] == "completed"
    assert result["collected_slot_count"] == expected_slots
    assert result["physical_calls"] == {
        "maximum_calls": expected_calls,
        "attempted_calls": expected_calls,
        "successful_calls": expected_calls,
        "failed_calls": 0,
    }
    assert (output / "population_summary.json").is_file()


def test_submission_surface_is_validation_first():
    campaign = (ROOT / "infrastructure/truba/submit_paper120_v3_population.sh").read_text()
    sbatch = (ROOT / "infrastructure/truba/paper_v3_population_template.sbatch").read_text()
    prepared = (ROOT / "infrastructure/truba/paper120_b40_snpe_remaining117_v1.env").read_text()
    smoke = (ROOT / "infrastructure/truba/paper120_b40_all_method_smoke_v1.env").read_text()
    b100 = (ROOT / "infrastructure/truba/paper120_b100_three_method_qualification_v1.env").read_text()
    b100_remaining = (ROOT / "infrastructure/truba/paper120_b100_mutaco_npe_snpe_remaining117_v1.env").read_text()
    assert '${1:---validate-only}' in campaign
    assert '--workers "${MUTACO_V3_WORKERS}"' in campaign
    assert '--maximum-calls "${MUTACO_V3_MAXIMUM_CALLS}"' in campaign
    assert '--torch-intraop-threads "${sbatch_cpus_per_task}"' in campaign
    assert '--method-specs "${MUTACO_V3_METHOD_SPECS}"' in campaign
    assert "--mutaco-config" not in campaign and "--npe-config" not in campaign
    assert "#SBATCH --nodes=8" in sbatch and "#SBATCH --ntasks=40" in sbatch
    assert "#SBATCH --exclude=barbun[112-119]" in sbatch
    assert "#SBATCH --ntasks-per-node=5" in sbatch and "#SBATCH --cpus-per-task=8" in sbatch
    assert "#SBATCH --mem=64G" in sbatch
    assert "--kill-on-bad-exit=0" in sbatch
    assert 'srun --ntasks="${MUTACO_V3_WORKERS}"' in sbatch
    assert '--cpus-per-task="${MUTACO_V3_SBATCH_CPUS_PER_TASK}"' in sbatch
    assert '--workers "${MUTACO_V3_WORKERS}"' in sbatch
    assert 'MUTACO_V3_TARGET_LEDGER="experiments/graph_inversion/frozen_targets/paper120_v3"' in prepared
    assert '--nodes="${sbatch_nodes}"' in campaign
    assert '--ntasks="${sbatch_ntasks}"' in campaign
    assert '--ntasks-per-node="${sbatch_ntasks_per_node}"' in campaign
    assert '--cpus-per-task="${sbatch_cpus_per_task}"' in campaign
    assert '--exclude="${sbatch_exclude}"' in campaign
    assert 'sbatch_ntasks != sbatch_nodes * sbatch_ntasks_per_node' in campaign
    assert 'sbatch_ntasks_per_node * sbatch_cpus_per_task != 20' in campaign
    assert 'sbatch_ntasks < MUTACO_V3_WORKERS' in campaign
    assert 'MUTACO_V3_WORKERS=1' in smoke
    assert 'MUTACO_V3_MAXIMUM_CALLS=123' in smoke
    assert 'MUTACO_V3_SBATCH_NODES=1' in smoke
    assert 'MUTACO_V3_SBATCH_NTASKS=1' in smoke
    assert 'MUTACO_V3_SBATCH_CPUS_PER_TASK=20' in smoke
    assert 'MUTACO_V3_SBATCH_EXCLUDE="barbun70,barbun[112-119]"' in smoke
    assert 'MUTACO_V3_WORKERS=3' in b100
    assert 'MUTACO_V3_MAXIMUM_CALLS=303' in b100
    assert 'MUTACO_V3_SBATCH_NODES=3' in b100
    assert 'MUTACO_V3_SBATCH_NTASKS=3' in b100
    assert 'MUTACO_V3_SBATCH_NTASKS_PER_NODE=1' in b100
    assert 'MUTACO_V3_SBATCH_CPUS_PER_TASK=20' in b100
    assert 'MUTACO_V3_SBATCH_EXCLUDE="barbun70,barbun[112-119]"' in b100
    assert 'MUTACO_V3_WORKERS=39' in b100_remaining
    assert 'MUTACO_V3_MAXIMUM_CALLS=909' in b100_remaining
    assert 'MUTACO_V3_SBATCH_NODES=10' in b100_remaining
    assert 'MUTACO_V3_SBATCH_NTASKS=40' in b100_remaining
    assert 'MUTACO_V3_SBATCH_NTASKS_PER_NODE=4' in b100_remaining
    assert 'MUTACO_V3_SBATCH_CPUS_PER_TASK=5' in b100_remaining
    assert 10 * 4 == 40
    assert 4 * 5 == 20
    assert 'MUTACO_V3_SBATCH_EXCLUDE="barbun70,barbun[112-119]"' in b100_remaining
