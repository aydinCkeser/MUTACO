from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRUBA = ROOT / "infrastructure/truba"


def test_repaired_sbi_profile_and_submission_geometry() -> None:
    profile = (TRUBA / "repaired_sbi_assessment_v1.env").read_text()
    submit = (TRUBA / "submit_repaired_sbi_assessment.sh").read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="repaired_sbi_assessment"' in profile
    assert "MUTACO_REPAIRED_SBI_WORKERS=120" in profile
    assert "MUTACO_REPAIRED_SBI_MAXIMUM_CALLS_PER_WORKER=6" in profile
    assert "MUTACO_REPAIRED_SBI_SBATCH_NODES=30" in profile
    assert "MUTACO_REPAIRED_SBI_SBATCH_NTASKS_PER_NODE=4" in profile
    assert "MUTACO_REPAIRED_SBI_SBATCH_CPUS_PER_TASK=5" in profile
    assert 'mode="${1:---validate-only}"' in submit
    assert 'MUTACO_REPAIRED_SBI_EXECUTION_AUTHORIZED:-0' in submit
    assert "exec sbatch" in submit


def test_repaired_sbi_batch_collects_even_after_worker_failure() -> None:
    batch = (TRUBA / "repaired_sbi_assessment_template.sbatch").read_text()
    assert "--nodes=30" in batch and "--ntasks=120" in batch
    assert "--ntasks-per-node=4" in batch and "--cpus-per-task=5" in batch
    assert "--no-requeue" in batch
    assert "--kill-on-bad-exit=0" in batch
    assert "|| worker_status=$?" in batch
    assert batch.index("run_repaired_sbi_assessment_worker.sh") < batch.index(
        "collect_repaired_sbi_assessment"
    )


def test_exact_source_validation_knows_repaired_sbi_surface() -> None:
    text = (TRUBA / "validate_exact_source.sh").read_text()
    assert "repaired_sbi_assessment)" in text
    assert "submit_repaired_sbi_assessment.sh --validate-only" in text
