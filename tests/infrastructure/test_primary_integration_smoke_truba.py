from pathlib import Path

from infrastructure.truba.scalar_primary_population_task import (
    execute_task,
    logical_ranks,
)


ROOT = Path(__file__).resolve().parents[2]
TRUBA = ROOT / "infrastructure/truba"


def test_primary_smoke_profile_is_disabled_and_exactly_capped() -> None:
    profile = (TRUBA / "primary_prequalification_b40_integration_smoke_v1.env").read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_integration_smoke"' in profile
    assert "MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS=123" in profile
    assert "MUTACO_PRIMARY_SMOKE_WORKERS=1" in profile
    assert "MUTACO_PRIMARY_SMOKE_EXECUTION_AUTHORIZED=0" in profile

    b100 = (TRUBA / "scalar_primary_prequalification_b100_v1.env").read_text()
    assert 'MUTACO_TRUBA_VALIDATION_SURFACE="primary_integration_smoke"' in b100
    assert "scalar_primary_prequalification_b100_v1.json" in b100
    assert "paper120_b100_mutaco_v1.json" in b100
    assert "MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS=303" in b100
    assert "MUTACO_PRIMARY_SMOKE_WORKERS=1" in b100
    assert "MUTACO_PRIMARY_SMOKE_EXECUTION_AUTHORIZED=0" in b100


def test_primary_smoke_validation_precedes_authorization_and_submission() -> None:
    submit = (TRUBA / "submit_primary_integration_smoke.sh").read_text()
    preview = submit.index("run_primary_integration_smoke_worker")
    validate_exit = submit.index('if [[ "${mode}" == "--validate-only" ]]')
    authorization = submit.index("MUTACO_PRIMARY_SMOKE_EXECUTION_AUTHORIZED")
    submission = submit.index("exec sbatch")
    assert preview < validate_exit < authorization < submission
    assert "--execute" not in submit[:validate_exit]
    assert '"${MUTACO_PRIMARY_SMOKE_MAXIMUM_CALLS}"' in submit[:validate_exit]


def test_primary_smoke_batch_runs_worker_before_collection() -> None:
    batch = (TRUBA / "primary_integration_smoke_template.sbatch").read_text()
    assert batch.index("run_primary_integration_smoke_worker") < batch.index(
        "collect_primary_benchmark primary"
    )
    lifecycle = (TRUBA / "validate_exact_source.sh").read_text()
    assert "primary_integration_smoke" in lifecycle


def test_scalar_population_profiles_are_disabled_one_target_arrays() -> None:
    for budget, worker_cap, global_cap in ((40, 123, 14760), (100, 303, 36360)):
        profile = (TRUBA / f"scalar_primary_population_b{budget}_v1.env").read_text()
        assert 'MUTACO_TRUBA_VALIDATION_SURFACE="scalar_primary_population"' in profile
        assert "MUTACO_SCALAR_PRIMARY_WORKERS=120" in profile
        assert f"MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER={worker_cap}" in profile
        assert f"MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS={global_cap}" in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY=120" in profile
        assert "MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK=20" in profile
        assert "MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED=0" in profile


def test_scalar_population_validation_precedes_separate_array_and_collector_jobs() -> None:
    submit = (TRUBA / "submit_scalar_primary_population.sh").read_text()
    preview = submit.index("--validate-all-shards")
    validate_exit = submit.index('if [[ "${mode}" == "--validate-only" ]]')
    authorization = submit.index("MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED")
    array_submission = submit.index('array_job="$(sbatch')
    collector_submission = submit.index('collector_job="$(sbatch')
    assert (
        preview
        < validate_exit
        < authorization
        < array_submission
        < collector_submission
    )
    assert "--execute" not in submit[:validate_exit]
    assert '--array="${MUTACO_SCALAR_PRIMARY_ARRAY_TASKS}%' in submit
    assert '--dependency="afterany:${array_job_id}"' in submit
    assert 'sbatch --parsable --begin="${MUTACO_SCALAR_PRIMARY_BEGIN_DELAY}"' in submit
    assert "scontrol release" not in submit
    assert 'MUTACO_SCALAR_PRIMARY_BEGIN_DELAY:-}" != "now+10minutes"' in submit
    assert "rollback_partial_submission" in submit
    assert "--exclusive" in submit
    assert "--no-requeue" in submit
    lifecycle = (TRUBA / "validate_exact_source.sh").read_text()
    assert "scalar_primary_population" in lifecycle


def test_scalar_population_worker_uses_sequential_task_runner_and_collector_is_zero_call() -> None:
    worker = (TRUBA / "run_scalar_primary_population_worker.sh").read_text()
    collector = (TRUBA / "scalar_primary_population_collector.sbatch").read_text()
    assert "infrastructure.truba.scalar_primary_population_task" in worker
    assert "collect_primary_benchmark primary" in collector
    assert "run_primary_integration_smoke_worker" not in collector


def test_scalar_population_v2_profiles_freeze_two_disjoint_physical_arrays() -> None:
    for budget, worker_cap, global_cap in ((40, 123, 14760), (100, 303, 36360)):
        profile = (TRUBA / f"scalar_primary_population_b{budget}_v2.env").read_text()
        assert f"scalar_primary_population_b{budget}_v2.json" in profile
        assert f"scalar_primary_population_b{budget}_plan_v2.json" in profile
        assert 'MUTACO_SCALAR_PRIMARY_ARRAY_A_TASKS="0-99"' in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_A_RANK_OFFSET=0" in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_A_CONCURRENCY=100" in profile
        assert 'MUTACO_SCALAR_PRIMARY_ARRAY_B_TASKS="0-19"' in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_B_RANK_OFFSET=100" in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_B_CONCURRENCY=20" in profile
        assert f"MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER={worker_cap}" in profile
        assert f"MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS={global_cap}" in profile
        assert "MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED=0" in profile


def test_scalar_population_v3_profile_freezes_60_sequential_pairs() -> None:
    for suffix in ("v3", "v3r1"):
        profile = (TRUBA / f"scalar_primary_population_b40_{suffix}.env").read_text()
        assert f"scalar_primary_population_b40_{suffix}.json" in profile
        assert f"scalar_primary_population_b40_plan_{suffix}.json" in profile
        assert 'MUTACO_SCALAR_PRIMARY_ARRAY_TASKS="0-59"' in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY=60" in profile
        assert "MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS=60" in profile
        assert "MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK=2" in profile
        assert "MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER=123" in profile
        assert "MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS=14760" in profile
        assert "MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED=0" in profile
    collector = (TRUBA / "scalar_primary_population_collector.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=20" in collector


def test_scalar_population_v3r2_profile_freezes_delayed_start_transaction() -> None:
    for budget, worker_cap, global_cap in ((40, 123, 14760), (100, 303, 36360)):
        profile = (TRUBA / f"scalar_primary_population_b{budget}_v3r2.env").read_text()
        assert f"scalar_primary_population_b{budget}_v3r2.json" in profile
        assert f"scalar_primary_population_b{budget}_plan_v3r2.json" in profile
        assert 'MUTACO_SCALAR_PRIMARY_ARRAY_TASKS="0-59"' in profile
        assert "MUTACO_SCALAR_PRIMARY_ARRAY_CONCURRENCY=60" in profile
        assert "MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS=60" in profile
        assert "MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK=2" in profile
        assert f"MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER={worker_cap}" in profile
        assert f"MUTACO_SCALAR_PRIMARY_GLOBAL_MAXIMUM_CALLS={global_cap}" in profile
        assert 'MUTACO_SCALAR_PRIMARY_BEGIN_DELAY="now+10minutes"' in profile
        assert "MUTACO_SCALAR_PRIMARY_EXECUTION_AUTHORIZED=0" in profile


def _task_environment() -> dict[str, str]:
    return {
        "SLURM_ARRAY_TASK_ID": "7",
        "MUTACO_SCALAR_PRIMARY_PHYSICAL_TASKS": "60",
        "MUTACO_SCALAR_PRIMARY_LOGICAL_WORKERS_PER_TASK": "2",
        "MUTACO_SCALAR_PRIMARY_WORKERS": "120",
        "MUTACO_V3_PYTHON": "/pinned/python",
        "MUTACO_SCALAR_PRIMARY_DECLARATION": "declaration.json",
        "MUTACO_SCALAR_PRIMARY_PLAN": "plan.json",
        "MUTACO_SCALAR_PRIMARY_TARGET_LEDGER": "targets",
        "MUTACO_SCALAR_PRIMARY_CONFIG": "config.json",
        "MUTACO_SCALAR_PRIMARY_QLOGNEI_SETTINGS": "settings.json",
        "MUTACO_SCALAR_PRIMARY_OUTPUT_ROOT": "/append-only/workers",
        "MUTACO_SCALAR_PRIMARY_MAXIMUM_CALLS_PER_WORKER": "123",
        "MUTACO_EXPECTED_COMMIT": "a" * 40,
        "MUTACO_SCALAR_PRIMARY_CPUS_PER_TASK": "20",
    }


def test_scalar_population_v3_rank_mapping_covers_all_workers_once() -> None:
    mapped = [
        rank
        for task_id in range(60)
        for rank in logical_ranks(task_id, physical_tasks=60, logical_workers=120)
    ]
    assert sorted(mapped) == list(range(120))
    assert len(set(mapped)) == 120


def test_scalar_population_v3_runs_second_rank_after_first_failure() -> None:
    commands: list[tuple[str, ...]] = []

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def runner(command: tuple[str, ...], *, check: bool) -> Result:
        assert check is False
        commands.append(command)
        return Result(9 if len(commands) == 1 else 0)

    returncode, terminal = execute_task(_task_environment(), runner=runner)
    assert returncode == 1
    assert [command[command.index("--rank") + 1] for command in commands] == ["7", "67"]
    assert terminal["logical_worker_results"] == [
        {"rank": 7, "returncode": 9},
        {"rank": 67, "returncode": 0},
    ]
    assert terminal["all_logical_workers_succeeded"] is False
