from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRUBA = ROOT / "infrastructure/truba"


def test_assessment_rescue_profile_is_three_call_and_unauthorized() -> None:
    profile = (TRUBA / "primary_prequalification_b40_assessment_rescue_v1.env").read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_assessment_rescue"' in profile
    assert "MUTACO_PRIMARY_RESCUE_MAXIMUM_CALLS=3" in profile
    assert "MUTACO_PRIMARY_RESCUE_EXECUTION_AUTHORIZED=0" in profile
    assert "prequalification-b40-v1/workers/worker-00/targets/target-000/decision_freeze.json" in profile


def test_assessment_rescue_submission_validates_before_authorization() -> None:
    submit = (TRUBA / "submit_primary_assessment_rescue.sh").read_text()
    preview = submit.index("run_primary_assessment_rescue")
    validation_exit = submit.index('if [[ "${mode}" == "--validate-only" ]]')
    authorization = submit.index("MUTACO_PRIMARY_RESCUE_EXECUTION_AUTHORIZED")
    submission = submit.index("exec sbatch")
    assert preview < validation_exit < authorization < submission
    assert "exact three-call cap" in submit


def test_assessment_rescue_batch_has_no_search_and_collects() -> None:
    runner = (TRUBA / "run_primary_assessment_rescue.sh").read_text()
    batch = (TRUBA / "primary_assessment_rescue.sbatch").read_text()
    validation = (TRUBA / "validate_exact_source.sh").read_text()
    assert "run_primary_assessment_rescue" in runner
    assert "primary_integration_smoke_driver" not in runner
    assert "collect_primary_benchmark assessment-rescue" in batch
    assert "primary_assessment_rescue)" in validation
