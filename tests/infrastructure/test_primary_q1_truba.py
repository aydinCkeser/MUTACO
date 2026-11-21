from __future__ import annotations

import hashlib
import json
from pathlib import Path

from infrastructure.truba.primary_q1 import validate_preparation
from mutaco import stable_hash


ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_preparation_v1.json"
DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_v1.json"
RETRY_DECLARATION = (
    ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_retry_v1.json"
)
EVIDENCE_REPAIR_DECLARATION = (
    ROOT
    / "experiments/graph_inversion/configs/paper/primary_nei_q1_evidence_repair_v1.json"
)
EVIDENCE_REPAIR_ATTEMPT = (
    ROOT
    / "experiments/graph_inversion/configs/paper/primary_nei_q1_evidence_repair_attempt_v1.json"
)
POST_SMOKE_DECLARATION = (
    ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_post_smoke_v1.json"
)
POST_SMOKE_ATTEMPT = (
    ROOT
    / "experiments/graph_inversion/configs/paper/primary_nei_q1_post_smoke_attempt_v1.json"
)


def test_q1_truba_preview_is_zero_call_and_submission_unauthorized() -> None:
    preview = validate_preparation(PREPARATION, workers=5, maximum_calls=140)
    assert preview["global_search_call_cap"] == 700
    assert preview["target_construction_calls_required"] == 5
    assert preview["target_construction_authorized"] is False
    assert preview["qualification_authorized"] is False
    assert preview["job_submitted"] is False
    assert preview["scientific_calls_executed"] == 0


def test_q1_remote_scripts_are_lf_only_and_validate_first() -> None:
    submit = (ROOT / "infrastructure/truba/submit_primary_nei_q1.sh").read_bytes()
    validate = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_bytes()
    assert b"\r" not in submit + validate
    text = submit.decode()
    assert '${1:---validate-only}' in text
    assert "MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED" in text
    assert "separately constructed frozen target responses" in text
    assert "exec sbatch" in text
    assert text.index("MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED") < text.index("exec sbatch")
    assert text.index("run_primary_q1_worker") < text.index(
        'if [[ "${mode}" == "--validate-only" ]]'
    )
    assert text.index("run_primary_q1_worker") < text.index(
        "MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED"
    )
    assert 'export PYTHONPATH="${root}/src:' in text
    assert "${PYTHONPATH:+:${PYTHONPATH}}" in text
    assert "submit_primary_nei_q1.sh --validate-only" in validate.decode()


def test_q1_target_construction_surface_is_exactly_five_calls_and_gated() -> None:
    submit = (
        ROOT / "infrastructure/truba/submit_primary_nei_q1_target_construction.sh"
    ).read_text()
    batch = (
        ROOT / "infrastructure/truba/primary_nei_q1_target_construction.sbatch"
    ).read_text()
    profile = (
        ROOT / "infrastructure/truba/primary_nei_q1_target_construction_v1.env"
    ).read_text()
    assert '${1:---validate-only}' in submit
    assert 'MUTACO_Q1_TARGET_MAXIMUM_CALLS}" == "5"' in submit
    assert "MUTACO_Q1_TARGET_CONSTRUCTION_AUTHORIZED" in submit
    assert submit.index("MUTACO_Q1_TARGET_CONSTRUCTION_AUTHORIZED") < submit.index("exec sbatch")
    assert 'export PYTHONPATH="${root}/src:' in submit
    assert "${PYTHONPATH:+:${PYTHONPATH}}" in submit
    assert "--maximum-target-calls 5" in submit
    assert "--execute --confirm-approved" in batch
    assert "MUTACO_Q1_TARGET_MAXIMUM_CALLS=5" in profile
    assert "MUTACO_Q1_TARGET_CONSTRUCTION_AUTHORIZED=0" in profile


def test_q1_postconstruction_declaration_and_profile_are_frozen() -> None:
    document = json.loads(DECLARATION.read_text(encoding="utf-8"))
    retry = json.loads(RETRY_DECLARATION.read_text(encoding="utf-8"))
    sidecar = DECLARATION.with_suffix(".sha256").read_text(encoding="ascii").strip()
    retry_sidecar = RETRY_DECLARATION.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()
    profile = (ROOT / "infrastructure/truba/primary_nei_q1_v1.env").read_text()
    assert document["declaration_identity"] == (
        "nei-numerical-qualification-declaration-v1-"
        "12c8822d0e09859eafc77401febad714785b4bcd772c77b378e904954a4a1d3c"
    )
    assert document["call_preview"] == {
        "maximum_assessment_calls": 0,
        "maximum_search_calls": 700,
        "maximum_total_calls": 700,
        "scientific_calls_executed": 0,
        "target_construction_calls": 0,
    }
    assert len(document["target_record_identities"]) == 5
    assert sidecar == "2cd37c1e7efb5c429aaa0fe5ba4561fa2b816dfda9301ff0ef614eae8a2bd594"
    assert hashlib.sha256(DECLARATION.read_bytes()).hexdigest() == sidecar
    assert retry["declaration_identity"] == (
        "nei-numerical-qualification-declaration-v1-"
        "e2b36a74389d85f96c1fe5926a5cebff9979db310ae7bd8d83433d27a8aac9e2"
    )
    assert retry["output_root"] == (
        "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/primary_nei_q1/qualification-retry-v1/workers"
    )
    original_science = {
        key: value
        for key, value in document.items()
        if key not in {"declaration_identity", "output_root"}
    }
    retry_science = {
        key: value
        for key, value in retry.items()
        if key not in {"declaration_identity", "output_root"}
    }
    assert retry_science == original_science
    assert retry_sidecar == (
        "dca5aec54ab81e0662db9596ede74c3a50f51e5a5d4ea3236606b0cda0569416"
    )
    assert hashlib.sha256(RETRY_DECLARATION.read_bytes()).hexdigest() == retry_sidecar
    assert (
        'MUTACO_PRIMARY_Q1_TARGET_RESPONSES="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/'
        'primary_nei_q1/67537cc2610a/target-construction/workers/target_ledger"'
    ) in profile
    assert (
        'MUTACO_PRIMARY_Q1_DECLARATION="experiments/graph_inversion/configs/'
        'paper/primary_nei_q1_retry_v1.json"'
    ) in profile
    assert "MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED=0" in profile


def test_post_repair_q1_declaration_changes_only_attempt_identity_and_output() -> None:
    original = json.loads(DECLARATION.read_text(encoding="utf-8"))
    repaired = json.loads(EVIDENCE_REPAIR_DECLARATION.read_text(encoding="utf-8"))
    sidecar = EVIDENCE_REPAIR_DECLARATION.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()
    assert sidecar == "4e17f07fc767ae5718f59b6915de6090005d53fe9cd3655a4dd30bb9c8c1d0ff"
    assert hashlib.sha256(EVIDENCE_REPAIR_DECLARATION.read_bytes()).hexdigest() == sidecar
    assert repaired["declaration_identity"] == (
        "nei-numerical-qualification-declaration-v1-"
        "18244241c5af5f5abcd5171c39a2ea2a9dd6db04ac5da32ec944384ba0b56af1"
    )
    assert repaired["output_root"] == (
        "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/primary_nei_q1/"
        "qualification-evidence-repair-v1/workers"
    )
    excluded = {"declaration_identity", "output_root"}
    assert {key: value for key, value in repaired.items() if key not in excluded} == {
        key: value for key, value in original.items() if key not in excluded
    }


def test_post_repair_run_declaration_excludes_historical_partial_evidence() -> None:
    attempt = json.loads(EVIDENCE_REPAIR_ATTEMPT.read_text(encoding="utf-8"))
    sidecar = EVIDENCE_REPAIR_ATTEMPT.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()
    assert hashlib.sha256(EVIDENCE_REPAIR_ATTEMPT.read_bytes()).hexdigest() == sidecar
    identity = attempt.pop("run_declaration_identity")
    assert identity == stable_hash("mutaco-nei-q1-run-declaration-v1", attempt)
    assert attempt["evidence_repair_ancestor"] == (
        "ae8dfdf04360e7869b6ca4d6c82a843e6cb6c59f"
    )
    assert attempt["tier_selection_evidence_policy"] == (
        "complete_matrix_from_this_qualification_declaration_only"
    )
    assert [row["job_id"] for row in attempt["historical_attempts"]] == [
        6244497,
        6244676,
    ]
    assert all(
        row["eligible_for_tier_selection"] is False
        for row in attempt["historical_attempts"]
    )
    assert attempt["scientific_retry_authorized"] is False
    assert attempt["job_submitted"] is False


def test_post_repair_q1_profile_is_zero_call_and_non_submitting() -> None:
    profile = (
        ROOT / "infrastructure/truba/primary_nei_q1_evidence_repair_v1.env"
    ).read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_nei_q1"' in profile
    assert "primary_nei_q1_evidence_repair_v1.json" in profile
    assert "MUTACO_PRIMARY_Q1_WORKERS=5" in profile
    assert "MUTACO_PRIMARY_Q1_MAXIMUM_CALLS=140" in profile
    assert "MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED=0" in profile


def test_post_smoke_q1_declaration_preserves_science_with_fresh_output() -> None:
    repaired = json.loads(EVIDENCE_REPAIR_DECLARATION.read_text(encoding="utf-8"))
    fresh = json.loads(POST_SMOKE_DECLARATION.read_text(encoding="utf-8"))
    sidecar = POST_SMOKE_DECLARATION.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()
    assert hashlib.sha256(POST_SMOKE_DECLARATION.read_bytes()).hexdigest() == sidecar
    assert fresh["output_root"] == (
        "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/primary_nei_q1/qualification-post-smoke-v1/workers"
    )
    excluded = {"declaration_identity", "output_root"}
    assert {key: value for key, value in fresh.items() if key not in excluded} == {
        key: value for key, value in repaired.items() if key not in excluded
    }


def test_post_smoke_run_declaration_excludes_every_historical_q1_attempt() -> None:
    attempt = json.loads(POST_SMOKE_ATTEMPT.read_text(encoding="utf-8"))
    sidecar = POST_SMOKE_ATTEMPT.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()
    assert hashlib.sha256(POST_SMOKE_ATTEMPT.read_bytes()).hexdigest() == sidecar
    identity = attempt.pop("run_declaration_identity")
    assert identity == stable_hash("mutaco-nei-q1-run-declaration-v1", attempt)
    assert [row["job_id"] for row in attempt["historical_attempts"]] == [
        6244497,
        6244676,
        6246600,
    ]
    assert all(
        row["eligible_for_tier_selection"] is False
        for row in attempt["historical_attempts"]
    )
    assert attempt["engineering_prequalification"]["eligible_for_tier_selection"] is False
    assert attempt["scientific_calls_executed"] == 0
    assert attempt["scientific_retry_authorized"] is False
    assert attempt["job_submitted"] is False


def test_post_smoke_q1_profile_is_zero_call_and_non_submitting() -> None:
    profile = (
        ROOT / "infrastructure/truba/primary_nei_q1_post_smoke_v1.env"
    ).read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_nei_q1"' in profile
    assert "primary_nei_q1_post_smoke_v1.json" in profile
    assert "MUTACO_PRIMARY_Q1_WORKERS=5" in profile
    assert "MUTACO_PRIMARY_Q1_MAXIMUM_CALLS=140" in profile
    assert "MUTACO_PRIMARY_Q1_EXECUTION_AUTHORIZED=0" in profile


def test_q1_compute_smoke_is_single_job_zero_simulator_and_separately_gated() -> None:
    submit = (
        ROOT / "infrastructure/truba/submit_primary_nei_q1_compute_smoke.sh"
    ).read_bytes()
    batch = (
        ROOT / "infrastructure/truba/primary_nei_q1_compute_smoke.sbatch"
    ).read_bytes()
    profile = (
        ROOT / "infrastructure/truba/primary_nei_q1_compute_smoke_v1.env"
    ).read_text()
    validate = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_text()
    assert b"\r" not in submit + batch
    text = submit.decode()
    assert '${1:---validate-only}' in text
    assert "MUTACO_PRIMARY_Q1_COMPUTE_SMOKE_AUTHORIZED" in text
    assert text.index("MUTACO_PRIMARY_Q1_COMPUTE_SMOKE_AUTHORIZED") < text.index(
        "exec sbatch"
    )
    assert "MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NODES != 1" in text
    assert "MUTACO_PRIMARY_Q1_SMOKE_SBATCH_NTASKS != 1" in text
    assert "infrastructure.truba.scheduler_layout" in text
    assert "--execute --confirm-approved" in batch.decode()
    assert "run_primary_q1_compute_smoke" in batch.decode()
    assert "MUTACO_PRIMARY_Q1_COMPUTE_SMOKE_AUTHORIZED=0" in profile
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_nei_q1_compute_smoke"' in profile
    assert "primary_nei_q1_compute_smoke)" in validate
