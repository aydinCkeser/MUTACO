from __future__ import annotations

from pathlib import Path
from typing import Any

from mutaco import stable_hash

from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.paper.primary_benchmark import (
    PRIMARY_METHODS,
    PrimaryComparisonConfig,
    PrimaryMethodDeclaration,
    PrimaryPopulationDeclaration,
    execute_primary_assessment_rescue,
    load_hashed_json,
    load_primary_assessment_rescue_declaration,
    primary_assessment_rescue_preview,
)
from infrastructure.truba.collect_primary_benchmark import collect_assessment_rescue


ROOT = Path(__file__).resolve().parents[2]
TRACKED = ROOT / "experiments/graph_inversion/configs/paper/primary_prequalification_b40_assessment_rescue_v1.json"


def _primary() -> PrimaryPopulationDeclaration:
    methods = tuple(
        PrimaryMethodDeclaration(
            method=method,
            configuration_identity=f"{method}-configuration",
            search_policy_identifier=f"{method}-search",
            terminal_policy_identifier=f"{method}-terminal",
        )
        for method in PRIMARY_METHODS
    )
    return PrimaryPopulationDeclaration(
        campaign_id="assessment-rescue-test",
        comparison=PrimaryComparisonConfig(
            simulator_definition_identity="sim-definition",
            simulator_configuration_identity="sim-configuration",
            assessment_fidelity="sparse-m2",
            design_campaign_seed=11,
            inference_campaign_seed=12,
            terminal_campaign_seed=13,
            assessment_campaign_seed=14,
        ),
        study_identity="study-test",
        target_ledger_hash="ledger-test",
        target_record_identities=("target-a",),
        budget=40,
        methods=methods,
        output_root="unused-primary-output",
    )


def _freeze(primary: PrimaryPopulationDeclaration) -> dict[str, Any]:
    decisions = [
        {
            "method": method,
            "status": "recommended",
            "budget": 40,
            "search_calls": {"attempted": 40, "successful": 40},
            "recommended_theta": [float(index), 0.5],
        }
        for index, method in enumerate(PRIMARY_METHODS)
    ]
    body = {
        "schema_version": "mutaco-primary-all-method-decision-freeze-v1",
        "campaign_id": primary.campaign_id,
        "target_id": "target-a",
        "budget": 40,
        "assessment_started": False,
        "decisions": decisions,
    }
    return {
        **body,
        "freeze_identity": stable_hash("primary-all-method-decision-freeze-v1", body),
    }


def _rescue(primary: PrimaryPopulationDeclaration, freeze: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": "mutaco-primary-assessment-rescue-declaration-v1",
        "state": "declared_unexecuted_submission_unauthorized",
        "classification": "prequalification_engineering_rescue_not_q1_or_primary_performance_evidence",
        "campaign_id": primary.campaign_id,
        "primary_declaration_identity": primary.document()["declaration_identity"],
        "source_attempt_job_id": 6247524,
        "source_attempt_commit": "a" * 40,
        "decision_freeze_identity": freeze["freeze_identity"],
        "decision_freeze_sha256": "freeze-file-sha",
        "target_id": "target-a",
        "budget": 40,
        "assessment_replicate": 0,
        "methods": [
            {
                "method": decision["method"],
                "recommended_theta": decision["recommended_theta"],
            }
            for decision in freeze["decisions"]
        ],
        "configuration_sha256": "configuration-sha",
        "output_root": "fresh-rescue-output",
        "search_calls": 0,
        "maximum_assessment_calls": 3,
        "automatic_retry": False,
        "scientific_submission_authorized": False,
        "primary_benchmark_evidence_eligible": False,
    }
    return load_primary_assessment_rescue_declaration({
        **body,
        "declaration_identity": stable_hash(
            "primary-assessment-rescue-declaration-v1", body
        ),
    })


def test_tracked_rescue_declaration_is_hash_paired_and_unauthorized() -> None:
    declaration = load_primary_assessment_rescue_declaration(read_json(TRACKED))
    assert TRACKED.with_suffix(".sha256").read_text().strip() == file_hash(TRACKED)
    assert declaration["maximum_assessment_calls"] == 3
    assert declaration["search_calls"] == 0
    assert declaration["scientific_submission_authorized"] is False


def test_assessment_rescue_executes_only_three_paired_callbacks_and_collects(
    tmp_path: Path,
) -> None:
    primary = _primary()
    freeze = _freeze(primary)
    rescue = _rescue(primary, freeze)
    preview = primary_assessment_rescue_preview(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256="freeze-file-sha",
    )
    assert preview["scientific_calls_executed"] == 0
    assert preview["search_calls"] == 0
    contexts = []

    def assess(theta, context):
        contexts.append(dict(context))
        return [float(theta[0]), float(theta[1])]

    worker = tmp_path / "workers" / "worker-00"
    result = execute_primary_assessment_rescue(
        rescue=rescue,
        primary=primary,
        freeze=freeze,
        freeze_sha256="freeze-file-sha",
        target_response=[0.0, 0.0],
        evaluate_assessment=assess,
        output=worker,
        source_commit="b" * 40,
    )
    assert result["physical_calls"] == {
        "maximum_calls": 3,
        "attempted_calls": 3,
        "successful_calls": 3,
        "failed_calls": 0,
    }
    assert result["search_rerun"] is False
    assert len(contexts) == 3 and contexts[0] == contexts[1] == contexts[2]
    assert contexts[0]["purpose"] == "paired_held_out_assessment"
    collection = collect_assessment_rescue(
        declaration=rescue,
        workers_root=tmp_path / "workers",
        output=tmp_path / "workers" / "collection",
        expected_commit="b" * 40,
    )
    assert collection["record_count"] == 3
    assert collection["physical_calls"]["successful_calls"] == 3
    assert load_hashed_json(worker / "rescue_terminal.json")["status"] == "completed"


def test_assessment_rescue_preserves_partial_failure_without_exceeding_cap(
    tmp_path: Path,
) -> None:
    primary = _primary()
    freeze = _freeze(primary)
    rescue = _rescue(primary, freeze)
    calls = []

    def fail_second(theta, context):
        calls.append((theta, context))
        if len(calls) == 2:
            raise RuntimeError("synthetic assessment failure")
        return [0.0, 0.0]

    from experiments.paper.common.validation import PaperContractError
    import pytest

    with pytest.raises(PaperContractError, match="evidence was preserved"):
        execute_primary_assessment_rescue(
            rescue=rescue,
            primary=primary,
            freeze=freeze,
            freeze_sha256="freeze-file-sha",
            target_response=[0.0, 0.0],
            evaluate_assessment=fail_second,
            output=tmp_path / "worker-00",
            source_commit="c" * 40,
        )
    terminal = load_hashed_json(tmp_path / "worker-00" / "rescue_terminal.json")
    assert terminal["physical_calls"] == {
        "maximum_calls": 3,
        "attempted_calls": 2,
        "successful_calls": 1,
        "failed_calls": 1,
    }
    assert len(calls) == 2
