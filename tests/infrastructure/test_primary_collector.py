from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmarking.artifacts import read_json
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import append_hashed_json
from infrastructure.truba.collect_primary_benchmark import (
    collect_primary,
    load_primary_collection_declaration,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / (
    "experiments/graph_inversion/configs/paper/"
    "primary_prequalification_b40_integration_smoke_v1.json"
)
SCALAR = ROOT / (
    "experiments/graph_inversion/configs/paper/"
    "scalar_primary_prequalification_b40_v1.json"
)
B100_SCALAR = ROOT / (
    "experiments/graph_inversion/configs/paper/"
    "scalar_primary_prequalification_b100_v1.json"
)
SOURCE_COMMIT = "2" * 40


def _fake_worker(root: Path, declaration) -> None:
    identity = declaration.document()["declaration_identity"]
    target_id = declaration.target_record_identities[0]
    worker = root / "worker-00"
    budget = declaration.budget
    maximum_calls = declaration.document()["call_preview"]["maximum_total_calls"]
    append_hashed_json(worker / "worker_start.json", {
        "source_commit": SOURCE_COMMIT,
        "declaration_identity": identity,
        "preview": {"target_positions": [0]},
    })
    append_hashed_json(worker / "worker_terminal.json", {
        "source_commit": SOURCE_COMMIT,
        "declaration_identity": identity,
        "status": "completed",
        "physical_calls": {
            "maximum_calls": maximum_calls,
            "attempted_calls": maximum_calls,
            "successful_calls": maximum_calls,
            "failed_calls": 0,
        },
    })
    records = []
    for index, method in enumerate(item.method for item in declaration.methods):
        records.append({
            "schema_version": declaration.comparison.record_schema,
            "target_id": target_id,
            "budget": budget,
            "method": method,
            "decision": {
                "recommended_theta": [0.1, 0.2, 0.3, 0.4],
                "trajectory": [],
                "timing": {
                    "simulator": 1.0,
                    "gp_fit": 0.0,
                    "acquisition": 0.0,
                    "terminal": 0.1,
                    "other": 0.0,
                    "total": 1.1,
                },
            },
            "heldout_physical_mae": 0.1 + index,
            "status": "completed",
            "record_identity": f"record-{method}",
        })
    append_hashed_json(
        worker / "targets" / "target-000" / "target_execution.json",
        {"target_id": target_id, "records": records},
    )


@pytest.mark.parametrize(
    ("path", "collection_schema", "compact_schema", "difference_key", "expected_calls"),
    (
        (
            HISTORICAL,
            "mutaco-primary-global-collection-v1",
            "mutaco-primary-compact-export-v1",
            "MUTACO_minus_NEI",
            123,
        ),
        (
            SCALAR,
            "mutaco-scalar-primary-global-collection-v1",
            "mutaco-scalar-primary-compact-export-v1",
            "MUTACO_minus_scalar_qLogNEI",
            123,
        ),
        (
            B100_SCALAR,
            "mutaco-scalar-primary-global-collection-v1",
            "mutaco-scalar-primary-compact-export-v1",
            "MUTACO_minus_scalar_qLogNEI",
            303,
        ),
    ),
)
def test_primary_collector_dispatches_historical_and_scalar_schemas(
    tmp_path: Path,
    path: Path,
    collection_schema: str,
    compact_schema: str,
    difference_key: str,
    expected_calls: int,
) -> None:
    declaration = load_primary_collection_declaration(read_json(path))
    workers = tmp_path / path.stem / "workers"
    output = tmp_path / path.stem / "collection"
    _fake_worker(workers, declaration)
    result = collect_primary(
        declaration=declaration,
        workers_root=workers,
        output=output,
        workers=1,
        expected_commit=SOURCE_COMMIT,
    )
    compact = read_json(output / "compact_export.json")
    assert result["schema_version"] == collection_schema
    assert result["status"] == "completed"
    assert result["record_count"] == 3
    assert result["physical_calls"]["attempted_calls"] == expected_calls
    assert compact["schema_version"] == compact_schema
    assert difference_key in compact["paired_endpoint_differences"][0]
    assert compact["scientific_calls_added_by_export"] == 0


def test_primary_collector_rejects_unknown_declaration_schema() -> None:
    with pytest.raises(PaperContractError, match="schema"):
        load_primary_collection_declaration({"schema_version": "unknown"})
