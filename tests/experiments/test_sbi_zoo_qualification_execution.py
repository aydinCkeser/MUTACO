from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.benchmarking.methods.direct_posterior_decision import (
    posterior_sample_map_decision,
)
from experiments.benchmarking.methods.sbi_zoo_contract import (
    BUDGETS,
    METHODS,
    POSTERIOR_SAMPLE_COUNT,
    PRIOR_LAW,
    SBI_ZOO_PHYSICAL_SCORE_API,
    SBI_ZOO_SCORE_IDENTITIES,
    dataset_seed_deriver,
)
from experiments.benchmarking.methods.sbi_zoo_failures import (
    PhysicalSimulatorTripwireTriggered,
    ScientificBudgetViolation,
)
from experiments.benchmarking.methods.sbi_zoo_inference import (
    SbiZooRecommendation,
)
from experiments.graph_inversion import sbi_zoo_qualification as qualification
from experiments.paper.common.protocol_v3 import PhysicalCallCounts
from infrastructure.truba.lifecycle import _parse_json_documents
from infrastructure.truba.lifecycle import _write_receipt


ROOT = Path(__file__).resolve().parents[2]
TARGET = (
    ROOT
    / "experiments/graph_inversion/configs/paper/"
    "sbi_zoo_qualification_target_v1.json"
)
MANIFEST = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3/manifest.json"
CONFIG = ROOT / "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json"


@pytest.fixture(autouse=True)
def physical_simulator_tripwire(monkeypatch: pytest.MonkeyPatch):
    from mutaco_graph.scientific import OverdampedScientificSimulator
    from experiments.benchmarking.simulators.graph_scientific_d4 import (
        GraphScientificD4Adapter,
    )
    from experiments.graph_inversion.paper_v3_callbacks import (
        _CallbackGraphAdapter,
        _CallbackSimulator,
    )

    state = {"triggered": False, "expected": False}

    def guard(*args, **kwargs):
        del args, kwargs
        state["triggered"] = True
        raise PhysicalSimulatorTripwireTriggered("E0 physical simulator tripwire")

    for cls, name in (
        (OverdampedScientificSimulator, "__post_init__"),
        (OverdampedScientificSimulator, "evaluate"),
        (OverdampedScientificSimulator, "evaluate_batch"),
        (GraphScientificD4Adapter, "build_simulator"),
        (GraphScientificD4Adapter, "build_target_simulator"),
        (_CallbackGraphAdapter, "build_simulator"),
        (_CallbackSimulator, "evaluate"),
        (_CallbackSimulator, "evaluate_batch"),
    ):
        monkeypatch.setattr(cls, name, guard)
    yield state
    assert state["triggered"] is state["expected"]


def _receipt(tmp_path: Path, *, source: str = "a" * 40) -> Path:
    path = tmp_path / "lifecycle" / "validation.json"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_receipt(
        path,
        {
            "schema_version": "mutaco-truba-validation-receipt-v1",
            "status": "validated_zero_call",
            "source_commit": source,
            "source_materialization": {
                "schema_version": "mutaco-truba-source-materialization-v1",
                "status": "materialized_clean",
                "source_commit": source,
                "source_root": "/arf/scratch/example/source-exact",
            },
            "worker_preview": {
                "schema_version": "mutaco-sbi-zoo-qualification-lifecycle-preflight-v1",
                "qualification_execution_source": source,
                "scientific_calls_executed": 0,
                "job_submitted": False,
                "declaration_published": False,
                "lifecycle_receipt_binding_required": True,
            },
            "scientific_calls": 0,
            "job_submitted": False,
            "submission_authorized": False,
        },
    )
    return path


def _declaration(tmp_path: Path) -> dict:
    return qualification.build_declaration(
        target_record_path=TARGET,
        paper120_manifest_path=MANIFEST,
        simulator_configuration_path=CONFIG,
        output_root=tmp_path / "workers",
        qualification_execution_source="a" * 40,
        qualification_execution_tree="b" * 40,
        lifecycle_validation_receipt_path=_receipt(tmp_path),
    )


def test_target_declaration_randomness_and_zero_call_preview_are_deterministic(tmp_path):
    first = _declaration(tmp_path)
    second = _declaration(tmp_path)
    assert first == second
    assert first["target"]["target_record_id"] == qualification.TARGET_RECORD_ID
    assert first["target"]["paper120_membership"] is False
    assert first["datasets"]["declared_slots"] == {"40": 40, "100": 100}
    assert first["datasets"]["hard_attempt_cap"] == 140
    assert first["heldout_firewall"]["repaired_family_roots_forbidden"] == [
        92021540,
        92021600,
    ]
    preview = qualification.preview(first)
    assert preview["scientific_calls_executed"] == 0
    assert preview["required_recommendations"] == 4
    deriver = dataset_seed_deriver(
        first["qualification_declaration_identity"], first["randomness"]["root_seed"]
    )
    b40, _ = PRIOR_LAW.sample(
        40, deriver=deriver, target_record_id=qualification.TARGET_RECORD_ID, budget=40
    )
    b100, _ = PRIOR_LAW.sample(
        100, deriver=deriver, target_record_id=qualification.TARGET_RECORD_ID, budget=100
    )
    assert not torch.equal(b40, b100[:40])


def test_declaration_roundtrip_is_append_only_and_external(tmp_path):
    document = _declaration(tmp_path)
    path = tmp_path / "declaration" / "declaration.json"
    qualification.write_declaration(path, document)
    assert qualification.load_declaration(path) == document
    with pytest.raises(qualification.QualificationExecutionError, match="append-only"):
        qualification.write_declaration(path, document)
    with pytest.raises(qualification.QualificationExecutionError, match="outside"):
        qualification.build_declaration(
            target_record_path=TARGET,
            paper120_manifest_path=MANIFEST,
            simulator_configuration_path=CONFIG,
            output_root=ROOT / "forbidden-output",
            qualification_execution_source="a" * 40,
            qualification_execution_tree="b" * 40,
            lifecycle_validation_receipt_path=_receipt(tmp_path),
        )


def test_lifecycle_receipt_is_required_bound_hashed_and_validated(tmp_path):
    receipt_path = _receipt(tmp_path)
    document = _declaration(tmp_path)
    binding = document["lifecycle_validation_receipt"]
    assert binding["receipt_identity"] == binding["receipt"]["receipt_identity"]
    assert binding["receipt_sha256"] == qualification.file_hash(receipt_path)
    assert binding["receipt_identity"].startswith(
        "mutaco-truba-validation-receipt-v1-"
    )

    missing = tmp_path / "missing-receipt.json"
    with pytest.raises(qualification.QualificationExecutionError, match="required"):
        qualification.build_declaration(
            target_record_path=TARGET,
            paper120_manifest_path=MANIFEST,
            simulator_configuration_path=CONFIG,
            output_root=tmp_path / "other-workers",
            qualification_execution_source="a" * 40,
            qualification_execution_tree="b" * 40,
            lifecycle_validation_receipt_path=missing,
        )

    mismatched_path = _receipt(tmp_path / "mismatched", source="c" * 40)
    with pytest.raises(qualification.QualificationExecutionError, match="mismatched"):
        qualification.build_declaration(
            target_record_path=TARGET,
            paper120_manifest_path=MANIFEST,
            simulator_configuration_path=CONFIG,
            output_root=tmp_path / "mismatched-workers",
            qualification_execution_source="a" * 40,
            qualification_execution_tree="b" * 40,
            lifecycle_validation_receipt_path=mismatched_path,
        )

    blocked_q0_style = tmp_path / "blocked-q0-validation.json"
    blocked_q0_style.parent.mkdir(parents=True, exist_ok=True)
    _write_receipt(
        blocked_q0_style,
        {
            "schema_version": "mutaco-truba-validation-receipt-v1",
            "status": "validated_zero_call",
            "source_commit": "a" * 40,
            "source_materialization": {
                "status": "materialized_clean",
                "source_commit": "a" * 40,
            },
            "worker_preview": {
                "schema_version": "mutaco-sbi-zoo-qualification-preview-v1",
                "qualification_execution_source": "a" * 40,
                "scientific_calls_executed": 0,
                "job_submitted": False,
            },
            "scientific_calls": 0,
            "job_submitted": False,
            "submission_authorized": False,
        },
    )
    with pytest.raises(qualification.QualificationExecutionError, match="mismatched"):
        qualification.build_declaration(
            target_record_path=TARGET,
            paper120_manifest_path=MANIFEST,
            simulator_configuration_path=CONFIG,
            output_root=tmp_path / "blocked-q0-workers",
            qualification_execution_source="a" * 40,
            qualification_execution_tree="b" * 40,
            lifecycle_validation_receipt_path=blocked_q0_style,
        )

    tampered = dict(document)
    tampered_binding = dict(binding)
    tampered_receipt = dict(binding["receipt"])
    tampered_receipt["receipt_identity"] = (
        "mutaco-truba-validation-receipt-v1-" + "0" * 64
    )
    tampered_binding["receipt"] = tampered_receipt
    tampered_binding["receipt_identity"] = tampered_receipt["receipt_identity"]
    tampered["lifecycle_validation_receipt"] = tampered_binding
    body = {
        key: value
        for key, value in tampered.items()
        if key
        not in {
            "qualification_declaration_identity",
            "dataset_generation_declaration_identities",
        }
    }
    tampered["qualification_declaration_identity"] = qualification.stable_hash(
        "sbi-zoo-physical-qualification-declaration-v1", body
    )
    with pytest.raises(qualification.QualificationExecutionError, match="mismatched"):
        qualification.validate_declaration(tampered)


def test_lifecycle_preflight_is_zero_call_and_not_a_declaration(tmp_path):
    result = qualification.build_zero_call_preflight(
        target_record_path=TARGET,
        paper120_manifest_path=MANIFEST,
        simulator_configuration_path=CONFIG,
        output_root=tmp_path / "workers",
        qualification_execution_source="a" * 40,
        qualification_execution_tree="b" * 40,
    )
    assert result["scientific_calls_executed"] == 0
    assert result["job_submitted"] is False
    assert result["declaration_published"] is False
    assert result["lifecycle_receipt_binding_required"] is True
    assert "qualification_declaration_identity" not in result


def test_negative_tripwire_probe_fails_closed(physical_simulator_tripwire):
    from experiments.benchmarking.simulators.graph_scientific_d4 import (
        GraphScientificD4Adapter,
    )

    physical_simulator_tripwire["expected"] = True
    with pytest.raises(PhysicalSimulatorTripwireTriggered):
        GraphScientificD4Adapter().build_target_simulator(
            {}, device=torch.device("cpu"), dtype=torch.float32
        )


def test_full_mock_execution_shared_artifacts_restart_and_collection(
    tmp_path, monkeypatch
):
    from experiments.benchmarking.simulators.graph_scientific_d4 import (
        GraphScientificD4Adapter,
    )

    document = _declaration(tmp_path)
    calls = []
    response_spec = SimpleNamespace(
        axes=tuple(SimpleNamespace(name=f"response_{index:02d}") for index in range(20))
    )
    fake_simulator = SimpleNamespace(
        capabilities=SimpleNamespace(
            simulator_identity=qualification.SIMULATOR_ID,
            response_spec=response_spec,
        )
    )
    monkeypatch.setattr(qualification, "_require_exact_clean_source", lambda document: None)
    monkeypatch.setattr(
        GraphScientificD4Adapter,
        "build_target_simulator",
        lambda self, config, device, dtype: fake_simulator,
    )

    def fake_checked(simulator, theta, *, device, dtype, context):
        del simulator, device, dtype
        calls.append((context.scientific_run_id, context.logical_index, context.evaluation_seed))
        base = torch.arange(20, dtype=torch.float32)
        return SimpleNamespace(values=(base + theta.sum()).contiguous())

    monkeypatch.setattr(qualification, "checked_evaluate", fake_checked)
    for budget in BUDGETS:
        result = qualification.execute_dataset(document, budget=budget, target_record_path=TARGET)
        assert result["accounting"]["successful_calls"] == budget
    assert len(calls) == 140
    assert len({item[2] for item in calls}) == 140
    before = len(calls)
    with pytest.raises(ScientificBudgetViolation):
        qualification.execute_dataset(document, budget=40, target_record_path=TARGET)
    assert len(calls) == before

    class FakeEstimator:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    def fake_train(dataset, *, method, training_seed, training_log_dir):
        del training_log_dir
        return SimpleNamespace(
            method=method,
            budget=int(dataset.metadata["budget"]),
            estimator=FakeEstimator(),
            dataset_receipt=dataset.consumer_receipt(method),
            effective_configuration={"method": method, "budget": int(dataset.metadata["budget"])},
            training_seed=training_seed,
        )

    samples = torch.empty((POSTERIOR_SAMPLE_COUNT, 4), dtype=torch.float32)
    samples[:] = torch.tensor([0.5, 0.5, 0.5, 0.0])
    samples[-1] = torch.tensor([1.0, 1.0, 1.0, 0.5])

    def fake_recommend(trained, posterior, *, recommendation_seed, method_configuration_identity):
        del posterior
        decision = posterior_sample_map_decision(
            method=trained.method,
            posterior=SimpleNamespace(),
            samples=samples,
            lower=PRIOR_LAW.bounds()[0],
            upper=PRIOR_LAW.bounds()[1],
            method_configuration_identity=method_configuration_identity,
            inference_budget=trained.budget,
            inference_calls=PhysicalCallCounts(trained.budget, trained.budget),
            physical_score=lambda theta: theta[:, 0],
            physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[trained.method],
            physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
        )
        return SbiZooRecommendation(
            method=trained.method,
            samples=samples,
            sample_map=decision,
            recommendation_seed=recommendation_seed,
            mcmc_configuration={"requested_total_samples": POSTERIOR_SAMPLE_COUNT},
        )

    monkeypatch.setattr(qualification, "train_single_round", fake_train)
    monkeypatch.setattr(qualification, "build_mcmc_posterior", lambda trained, target: object())
    monkeypatch.setattr(qualification, "recommend_sample_map", fake_recommend)
    for budget in BUDGETS:
        for method in METHODS:
            result = qualification.execute_method(
                document, method=method, budget=budget, target_record_path=TARGET
            )
            assert result["recommendation_identity"].startswith(
                "sbi-zoo-qualification-recommendation-v1-"
            )
    collection = qualification.collect_qualification(document, write=True)
    assert collection["overall"] == "PASS"
    assert collection["scientific_calls"] == 140
    assert collection["heldout_calls"] == 0
    assert collection["paper120_training_calls"] == 0
    assert collection["lifecycle_validation_receipt_identity"] == document[
        "lifecycle_validation_receipt"
    ]["receipt_identity"]
    assert collection["lifecycle_validation_receipt_sha256"] == document[
        "lifecycle_validation_receipt"
    ]["receipt_sha256"]
    assert collection["components"] == {
        "SHARED_DATASET_GENERATION": "PASS",
        "NLE_B40": "PASS",
        "NRE_B40": "PASS",
        "NLE_B100": "PASS",
        "NRE_B100": "PASS",
    }


def test_empty_collection_is_blocked_and_does_not_create_output(tmp_path):
    document = _declaration(tmp_path)
    result = qualification.collect_qualification(document)
    assert result["overall"] == "BLOCKED"
    assert result["scientific_calls"] == 0
    assert not (tmp_path / "workers").exists()


def test_truba_surface_is_zero_call_validate_only_and_no_requeue():
    validate = (ROOT / "infrastructure/truba/validate_exact_source.sh").read_text()
    submit = (ROOT / "infrastructure/truba/submit_sbi_zoo_qualification.sh").read_text()
    sbatch = (ROOT / "infrastructure/truba/sbi_zoo_qualification.sbatch").read_text()
    profile = (ROOT / "infrastructure/truba/sbi_zoo_qualification_v1.env").read_text()
    assert "sbi_zoo_qualification)" in validate
    assert "submit_sbi_zoo_qualification.sh --validate-only" in validate
    assert "--validate-only" in submit and "--submit" in submit
    assert "sbi_zoo_qualification preflight" in submit
    assert "sbi_zoo_qualification declare" not in submit
    assert "MUTACO_SBI_ZOO_EXECUTION_AUTHORIZED" in submit
    assert "#SBATCH --no-requeue" in sbatch
    assert "#SBATCH --exclusive" in sbatch
    assert "dataset \"${common[@]}\" --budget 40" in sbatch
    assert "dataset \"${common[@]}\" --budget 100" in sbatch
    assert "92021540" not in submit + sbatch + profile
    assert "92021600" not in submit + sbatch + profile


def test_local_lifecycle_validation_emits_receipt_pair_and_no_output(tmp_path):
    # This is a shell/preflight test, with Git metadata doubled below. Use a
    # supported source path rather than weakening the production path guard.
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    for relative in (
        "infrastructure/truba/validate_exact_source.sh",
        "infrastructure/truba/submit_sbi_zoo_qualification.sh",
        "infrastructure/truba/sbi_zoo_qualification_v1.env",
        str(TARGET.relative_to(ROOT)), str(TARGET.with_suffix(".sha256").relative_to(ROOT)),
        str(MANIFEST.relative_to(ROOT)), str(MANIFEST.with_suffix(".sha256").relative_to(ROOT)),
        str(CONFIG.relative_to(ROOT)),
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *'rev-parse --show-toplevel'*) printf '%s\\n' \"$FAKE_REPO_ROOT\" ;;\n"
        "  *'rev-parse HEAD^{tree}'*) printf '%s\\n' \"$FAKE_TREE\" ;;\n"
        "  *'rev-parse HEAD'*) printf '%s\\n' \"$MUTACO_EXPECTED_COMMIT\" ;;\n"
        "  *'status --porcelain --untracked-files=all'*) : ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    bootstrap = tmp_path / "preflight.py"
    bootstrap.write_text('''
from contextlib import ExitStack
import runpy, sys
from unittest.mock import patch
from mutaco.surrogate import exact_multitask as gp
from mutaco_graph import OverdampedScientificSimulator
def forbidden(*a, **k):
    raise AssertionError("scientific execution forbidden in lifecycle preflight")
assert sys.argv[1] == "-m"
sys.argv = sys.argv[2:]
assert sys.argv[0] in {"infrastructure.truba.scheduler_layout", "experiments.graph_inversion.sbi_zoo_qualification"}
if sys.argv[0].startswith("experiments."):
    assert sys.argv[1] == "preflight"
with ExitStack() as stack:
    for name in ("_build_model", "fit_exact_multitask_gp", "fit_exact_multitask_gp_with_recovery"):
        stack.enter_context(patch.object(gp, name, forbidden))
    for name in ("__post_init__", "evaluate", "evaluate_batch", "evaluate_condition_grid"):
        stack.enter_context(patch.object(OverdampedScientificSimulator, name, forbidden))
    runpy.run_module(sys.argv[0], run_name="__main__")
''', encoding="utf-8")
    runtime = fake_bin / "python"
    runtime.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable)
                       + " " + shlex.quote(str(bootstrap)) + ' "$@"\n', encoding="utf-8")
    runtime.chmod(0o755)
    forbidden_job = tmp_path / "job-attempted"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\ntouch " + shlex.quote(str(forbidden_job))
                      + "\nexit 99\n", encoding="utf-8")
    sbatch.chmod(0o755)
    output = tmp_path / "workers"
    logs = tmp_path / "logs"
    declaration_root = tmp_path / "declaration"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_REPO_ROOT": str(source),
        "FAKE_TREE": "b" * 40,
        "MUTACO_PAPER120_ROOT": str(source),
        "MUTACO_V3_PYTHON": str(runtime),
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "MUTACO_EXPECTED_COMMIT": "a" * 40,
        "MUTACO_SBI_ZOO_DECLARATION": str(declaration_root / "declaration.json"),
        "MUTACO_SBI_ZOO_TARGET_RECORD": str(TARGET),
        "MUTACO_SBI_ZOO_PAPER120_MANIFEST": str(MANIFEST),
        "MUTACO_SBI_ZOO_CONFIG": str(CONFIG),
        "MUTACO_SBI_ZOO_OUTPUT_ROOT": str(output),
        "MUTACO_SBI_ZOO_JOB_LOG_DIR": str(logs),
        "MUTACO_SBI_ZOO_MAXIMUM_CALLS": "140",
        "MUTACO_SBI_ZOO_SBATCH_NODES": "1",
        "MUTACO_SBI_ZOO_SBATCH_NTASKS": "1",
        "MUTACO_SBI_ZOO_SBATCH_NTASKS_PER_NODE": "1",
        "MUTACO_SBI_ZOO_SBATCH_CPUS_PER_TASK": "20",
    }
    completed = subprocess.run(
        [
            "bash",
            "infrastructure/truba/validate_exact_source.sh",
            str(source),
            str(output),
            str(logs),
            str(runtime),
            "a" * 40,
            "infrastructure/truba/sbi_zoo_qualification_v1.env",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    documents = _parse_json_documents(completed.stdout)
    assert len(documents) == 2
    assert documents[0]["scientific_calls"] == 0
    assert documents[1]["scientific_calls_executed"] == 0
    assert documents[1]["job_submitted"] is False
    assert not output.exists()
    assert not declaration_root.exists()
    assert not forbidden_job.exists()


@pytest.mark.parametrize("argument, value", (
    (0, "source with spaces"), (0, "source;unsafe"), (0, "source/.."),
    (1, "output with spaces"), (2, "logs\nunsafe"), (3, "runtime with spaces"),
))
def test_lifecycle_rejects_unsafe_paths_before_external_execution(tmp_path, argument, value):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "external-attempted"
    for name in ("git", "sbatch", "srun"):
        executable = fake_bin / name
        executable.write_text("#!/usr/bin/env bash\ntouch " + shlex.quote(str(marker)) + "\nexit 99\n")
        executable.chmod(0o755)
    arguments = [str(tmp_path / name) for name in ("source", "output", "logs", "python")]
    arguments += ["a" * 40, "infrastructure/truba/sbi_zoo_qualification_v1.env"]
    arguments[argument] = str(tmp_path / value)
    result = subprocess.run(["bash", str(ROOT / "infrastructure/truba/validate_exact_source.sh"), *arguments],
                            cwd=tmp_path, env={**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
                            text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert result.stdout == result.stderr == ""
    assert not (tmp_path / "output").exists()
    assert not marker.exists()


def test_cli_declare_requires_lifecycle_receipt():
    parser = qualification._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "declare",
                "--target-record",
                str(TARGET),
                "--paper120-manifest",
                str(MANIFEST),
                "--config",
                str(CONFIG),
                "--output-root",
                "/tmp/mutaco-sbi-zoo-test-workers",
                "--execution-commit",
                "a" * 40,
                "--execution-tree",
                "b" * 40,
            ]
        )
