from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mutaco import ScientificIdentity
from experiments.benchmarking.methods.direct_posterior_decision import (
    HISTORICAL_DIRECT_POSTERIOR_SCORE_API,
    SBI_ZOO_PHYSICAL_SCORE_API,
    SBI_ZOO_SCORE_IDENTITIES,
    posterior_sample_map_decision,
)
from experiments.benchmarking.methods.sbi_zoo_accounting import (
    WriteAheadScientificCallLedger,
)
from experiments.benchmarking.methods.sbi_zoo_contract import (
    METHODS,
    POSTERIOR_SAMPLE_COUNT,
    PRIOR_LAW,
    complete_implementation_contract,
    dataset_seed_deriver,
    estimator_seed,
    estimator_contract,
    mcmc_contract,
    recommendation_seed,
)
from experiments.benchmarking.methods.sbi_zoo_dataset import (
    DatasetIntegrityError,
    SLOT_IDENTITY_FIELDS,
    canonical_vector_sha256,
    load_slot_outcome_artifact,
    load_shared_training_dataset,
    response_ordering_identity,
    write_slot_outcome_artifact,
    write_shared_training_dataset,
)
from experiments.benchmarking.methods.sbi_zoo_dataset_generation import (
    _assemble_slot_responses,
    generate_shared_training_dataset,
)
from experiments.benchmarking.methods.sbi_zoo_failures import (
    DatasetSerializationError,
    InsufficientPosteriorSamples,
    MCMCInitializationFailure,
    PhysicalSimulatorTripwireTriggered,
    SbiZooScoringContractError,
    ScientificBudgetViolation,
)
from experiments.benchmarking.methods.sbi_zoo_inference import (
    TrainedSbiZooEstimator,
    _builders,
    authoritative_prior,
    build_inference,
    build_mcmc_posterior,
    physical_posterior_score,
    sample_exact_posterior_candidates,
    train_single_round,
)
from experiments.paper.common.protocol_v3 import (
    PhysicalCallCounts,
    _validate_posterior_sample_map_decision,
)


RESPONSE_NAMES = tuple(f"response_{index:02d}" for index in range(20))


@pytest.fixture(autouse=True)
def fail_closed_physical_simulator_tripwire(monkeypatch):
    """Any canonical graph construction/evaluation during these tests is fatal."""

    from mutaco_graph.scientific import OverdampedScientificSimulator
    from experiments.benchmarking.simulators.graph_scientific_d4 import (
        GraphScientificD4Adapter,
    )
    from experiments.graph_inversion.paper_v3_callbacks import (
        _CallbackGraphAdapter,
        _CallbackSimulator,
    )

    state = {"triggered": False, "expected": False}

    def forbidden(*args, **kwargs):
        del args, kwargs
        state["triggered"] = True
        raise PhysicalSimulatorTripwireTriggered(
            "physical graph simulator is forbidden in the SBI-zoo engineering gate"
        )

    monkeypatch.setattr(OverdampedScientificSimulator, "__post_init__", forbidden)
    monkeypatch.setattr(OverdampedScientificSimulator, "evaluate", forbidden)
    monkeypatch.setattr(OverdampedScientificSimulator, "evaluate_batch", forbidden)
    monkeypatch.setattr(GraphScientificD4Adapter, "build_simulator", forbidden)
    monkeypatch.setattr(GraphScientificD4Adapter, "build_target_simulator", forbidden)
    monkeypatch.setattr(_CallbackGraphAdapter, "build_simulator", forbidden)
    monkeypatch.setattr(_CallbackSimulator, "evaluate", forbidden)
    monkeypatch.setattr(_CallbackSimulator, "evaluate_batch", forbidden)
    yield state
    assert state["triggered"] is state["expected"]


@pytest.mark.parametrize(
    "owner_name,method_name",
    [
        ("physical", "__post_init__"),
        ("physical", "evaluate"),
        ("physical", "evaluate_batch"),
        ("adapter", "build_simulator"),
        ("adapter", "build_target_simulator"),
        ("callback_adapter", "build_simulator"),
        ("callback", "evaluate"),
        ("callback", "evaluate_batch"),
    ],
)
def test_physical_simulator_tripwire_fails_closed(
    fail_closed_physical_simulator_tripwire, owner_name, method_name
):
    from mutaco_graph.scientific import OverdampedScientificSimulator
    from experiments.benchmarking.simulators.graph_scientific_d4 import (
        GraphScientificD4Adapter,
    )
    from experiments.graph_inversion.paper_v3_callbacks import (
        _CallbackGraphAdapter,
        _CallbackSimulator,
    )

    owners = {
        "physical": OverdampedScientificSimulator,
        "adapter": GraphScientificD4Adapter,
        "callback_adapter": _CallbackGraphAdapter,
        "callback": _CallbackSimulator,
    }
    fail_closed_physical_simulator_tripwire["expected"] = True
    with pytest.raises(PhysicalSimulatorTripwireTriggered):
        getattr(owners[owner_name], method_name)(object())


def _metadata(budget: int, *, suffix: str = "a") -> dict:
    return {
        "target_record_id": f"synthetic-target-{suffix}",
        "budget": budget,
        "parameter_names": list(PRIOR_LAW.parameter_names),
        "response_names": list(RESPONSE_NAMES),
        "parameter_bounds": {
            "lower": list(PRIOR_LAW.lower),
            "upper": list(PRIOR_LAW.upper),
        },
        "simulator_identity": "synthetic-never-physical-v1",
        "simulator_configuration_identity": "synthetic-config-v1",
        "row_context_identities": [f"synthetic-context-{suffix}-{i:03d}" for i in range(budget)],
        "dataset_generation_declaration_identity": "synthetic-declaration-v1",
        "scientific_call_ledger_identity": "synthetic-ledger-v1",
        "theta_law": PRIOR_LAW.document(),
        "theta_seed": 17,
    }


def _arrays(budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    deriver = dataset_seed_deriver("synthetic-dataset-science-v1", 9142)
    theta, _ = PRIOR_LAW.sample(
        budget,
        deriver=deriver,
        target_record_id="synthetic-target",
        budget=budget,
    )
    row = torch.arange(budget * 20, dtype=torch.float32).reshape(budget, 20)
    response = (row / float(max(1, budget * 20))).contiguous()
    return theta, response


@pytest.mark.parametrize("budget", [40, 100])
def test_shared_dataset_is_byte_canonical_and_cross_budget_independent(tmp_path, budget):
    theta, response = _arrays(budget)
    artifact = write_shared_training_dataset(
        tmp_path / f"b{budget}", theta=theta, response=response, metadata=_metadata(budget)
    )
    reloaded = load_shared_training_dataset(artifact.root)
    assert torch.equal(reloaded.theta, theta)
    assert torch.equal(reloaded.response, response)
    assert reloaded.theta.dtype is torch.float32 and reloaded.theta.is_contiguous()
    assert reloaded.response.dtype is torch.float32 and reloaded.response.is_contiguous()
    receipts = [reloaded.consumer_receipt(method) for method in METHODS]
    assert receipts[0]["canonical_content_sha256"] == receipts[1]["canonical_content_sha256"]
    assert receipts[0]["dataset_identity"] == receipts[1]["dataset_identity"]
    assert (artifact.root / "theta.f32le").read_bytes() == theta.numpy().astype("<f4").tobytes(order="C")


def test_dataset_hash_detects_byte_mutation_and_nonfinite_input(tmp_path):
    theta, response = _arrays(40)
    artifact = write_shared_training_dataset(
        tmp_path / "dataset", theta=theta, response=response, metadata=_metadata(40)
    )
    response_path = artifact.root / "response.f32le"
    payload = bytearray(response_path.read_bytes())
    payload[-1] ^= 1
    response_path.write_bytes(payload)
    with pytest.raises(DatasetIntegrityError, match="identity mismatch"):
        load_shared_training_dataset(artifact.root)
    bad = response.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(DatasetIntegrityError, match="forbids"):
        write_shared_training_dataset(
            tmp_path / "bad", theta=theta, response=bad, metadata=_metadata(40, suffix="bad")
        )


def test_prior_law_is_deterministic_and_b40_b100_are_not_nested():
    deriver = dataset_seed_deriver("synthetic-prior-v1", 1001)
    first, first_seed = PRIOR_LAW.sample(40, deriver=deriver, target_record_id="t", budget=40)
    replay, replay_seed = PRIOR_LAW.sample(40, deriver=deriver, target_record_id="t", budget=40)
    b100, b100_seed = PRIOR_LAW.sample(100, deriver=deriver, target_record_id="t", budget=100)
    assert torch.equal(first, replay) and first_seed == replay_seed
    assert first_seed != b100_seed
    assert not torch.equal(first, b100[:40])
    lower, upper = PRIOR_LAW.bounds()
    assert bool(((first >= lower) & (first <= upper)).all())
    training_nle = estimator_seed(
        deriver, method=METHODS[0], target_record_id="t", budget=40
    )
    training_nre = estimator_seed(
        deriver, method=METHODS[1], target_record_id="t", budget=40
    )
    posterior_nle = recommendation_seed(
        deriver, method=METHODS[0], target_record_id="t", budget=40
    )
    assert len({first_seed, b100_seed, training_nle, training_nre, posterior_nle}) == 5


def test_mock_dataset_generation_uses_write_ahead_once_per_row(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    seen = []

    def fake(theta, context):
        seen.append((theta.clone(), dict(context)))
        return torch.linspace(0.0, 1.0, 20, dtype=torch.float32) + len(seen)

    artifact = generate_shared_training_dataset(
        tmp_path / "dataset",
        target_record_id="synthetic-target",
        budget=40,
        deriver=dataset_seed_deriver("synthetic-generation-v1", 77),
        response_names=RESPONSE_NAMES,
        simulator_identity="synthetic-fake-v1",
        simulator_configuration_identity="synthetic-fake-config-v1",
        dataset_generation_declaration_identity="declaration",
        ledger=ledger,
        evaluate=fake,
    )
    assert len(seen) == 40
    assert ledger.accounting() == {
        "maximum_calls": 40,
        "attempted_calls": 40,
        "successful_calls": 40,
        "failed_calls": 0,
        "attempted_outcome_unknown": 0,
        "retry_calls": 0,
    }
    assert artifact.metadata["scientific_call_ledger_identity"] == ledger.identity
    with pytest.raises(ScientificBudgetViolation):
        ledger.persist_intent("row-000", {})


def _completed_generation(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)

    def fake(theta, context):
        return torch.arange(20, dtype=torch.float32) + float(context["row_index"])

    artifact = generate_shared_training_dataset(
        tmp_path / "dataset",
        target_record_id="synthetic-target",
        budget=40,
        deriver=dataset_seed_deriver("synthetic-generation-v1", 77),
        response_names=RESPONSE_NAMES,
        simulator_identity="synthetic-fake-v1",
        simulator_configuration_identity="synthetic-fake-config-v1",
        dataset_generation_declaration_identity="declaration",
        ledger=ledger,
        evaluate=fake,
    )
    return ledger, artifact


def _reassemble(ledger, artifact):
    return _assemble_slot_responses(
        ledger=ledger,
        target_record_id=artifact.metadata["target_record_id"],
        budget=artifact.metadata["budget"],
        theta=artifact.theta,
        context_identities=artifact.metadata["row_context_identities"],
        response_ordering_identity_value=response_ordering_identity(RESPONSE_NAMES),
        dataset_generation_declaration_identity=artifact.metadata[
            "dataset_generation_declaration_identity"
        ],
        simulator_identity=artifact.metadata["simulator_identity"],
        simulator_configuration_identity=artifact.metadata[
            "simulator_configuration_identity"
        ],
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "swapped"])
def test_dataset_assembly_rejects_missing_extra_and_swapped_slots(tmp_path, mutation):
    ledger, artifact = _completed_generation(tmp_path)
    outcomes = ledger.root / "outcomes"
    if mutation == "missing":
        shutil.rmtree(outcomes / "row-039")
    elif mutation == "extra":
        shutil.copytree(outcomes / "row-000", outcomes / "row-999")
    else:
        (outcomes / "row-000").rename(outcomes / "swap")
        (outcomes / "row-001").rename(outcomes / "row-000")
        (outcomes / "swap").rename(outcomes / "row-001")
    with pytest.raises(DatasetIntegrityError):
        _reassemble(ledger, artifact)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("target_record_id", "wrong-target"),
        ("budget", 100),
        ("dataset_generation_declaration_identity", "wrong-declaration"),
        ("call_ledger_identity", "wrong-ledger"),
        ("theta_canonical_sha256", "0" * 64),
        ("stochastic_context_identity", "wrong-context"),
        ("response_ordering_identity", "wrong-order"),
        ("simulator_identity", "wrong-simulator"),
        ("simulator_configuration_identity", "wrong-config"),
        ("row_index", 1),
        ("scientific_slot_identity", "duplicate-or-wrong-slot"),
    ],
)
def test_dataset_assembly_rejects_every_call_identity_mismatch(
    tmp_path, field, replacement
):
    ledger, artifact = _completed_generation(tmp_path)
    root = ledger.root / "outcomes" / "row-000"
    original = load_slot_outcome_artifact(root)
    identity = {name: original.metadata[name] for name in SLOT_IDENTITY_FIELDS}
    identity[field] = replacement
    shutil.rmtree(root)
    write_slot_outcome_artifact(
        root, response=original.response, call_identity=identity
    )
    with pytest.raises(DatasetIntegrityError):
        _reassemble(ledger, artifact)


@pytest.mark.parametrize("terminal", ["attempted_outcome_unknown", "failed"])
def test_dataset_assembly_rejects_unknown_and_failed_slots(tmp_path, terminal):
    ledger, artifact = _completed_generation(tmp_path)
    success = ledger.root / "slots" / "row-000" / "event-02-success.json"
    success.unlink()
    if terminal == "failed":
        import experiments.benchmarking.methods.sbi_zoo_accounting as accounting_module

        accounting_module._append_event(
            accounting_module._event_path(ledger.root, "row-000", 2, "failed"),
            {
                "schema_version": accounting_module.LEDGER_SCHEMA,
                "declaration_identity": ledger.declaration_identity,
                "ledger_identity": ledger.identity,
                "slot": "row-000",
                "status": "failed",
                "payload": {"failure_stage": "synthetic-persisted-failure"},
                "retry_calls": 0,
            },
        )
    assert ledger.slot_status("row-000") == terminal
    with pytest.raises(DatasetIntegrityError):
        _reassemble(ledger, artifact)


def test_attempted_unknown_is_not_retryable_and_cap_counts_attempts(tmp_path):
    root = tmp_path / "ledger"
    ledger = WriteAheadScientificCallLedger(root, "declaration", 40)
    ledger.persist_intent("row-000", {"theta": "hash"})
    ledger.mark_attempted("row-000")
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"
    with pytest.raises(ScientificBudgetViolation):
        ledger.mark_attempted("row-000")
    with pytest.raises(ScientificBudgetViolation):
        ledger.persist_intent("row-000", {})
    replay = WriteAheadScientificCallLedger(root, "declaration", 40)
    assert replay.slot_status("row-000") == "attempted_outcome_unknown"
    with pytest.raises(ScientificBudgetViolation):
        replay.resolve("row-000", "success", {})


def _slot_call_identity(ledger, *, row_index=0, target="synthetic-target"):
    theta = torch.tensor([0.5, 0.75, 1.0, 0.25], dtype=torch.float32)
    return {
        "dataset_generation_declaration_identity": "synthetic-declaration-v1",
        "call_ledger_identity": ledger.identity,
        "target_record_id": target,
        "budget": 40,
        "scientific_slot_identity": f"synthetic-scientific-slot-{row_index:03d}",
        "row_index": row_index,
        "theta_canonical_sha256": canonical_vector_sha256(
            theta, length=4, field="test theta"
        ),
        "stochastic_context_identity": f"synthetic-context-{row_index:03d}",
        "response_ordering_identity": response_ordering_identity(RESPONSE_NAMES),
        "simulator_identity": "synthetic-never-physical-v1",
        "simulator_configuration_identity": "synthetic-config-v1",
    }


def test_invocation_failure_is_failed_and_consumes_budget(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)

    def invoke():
        raise RuntimeError("synthetic invocation failure")

    with pytest.raises(RuntimeError, match="invocation failure"):
        ledger.invoke_once(
            "row-000", intent={"kind": "test"}, invoke=invoke, persist_outcome=lambda _: {}
        )
    assert ledger.slot_status("row-000") == "failed"
    assert ledger.attempted_count() == 1
    assert ledger.terminal_payload("row-000")["failure_stage"] == "simulator_invocation"


@pytest.mark.parametrize(
    "failure_label",
    ["response serialization", "metadata serialization"],
)
def test_post_invocation_serialization_failure_is_explicit_unknown(
    tmp_path, failure_label
):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)

    def persist(_):
        raise TypeError(failure_label)

    with pytest.raises(TypeError, match=failure_label):
        ledger.invoke_once(
            "row-000",
            intent={"kind": "test"},
            invoke=lambda: torch.zeros(20),
            persist_outcome=persist,
        )
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"
    assert ledger.attempted_count() == 1
    with pytest.raises(ScientificBudgetViolation):
        ledger.persist_intent("row-000", {})
    with pytest.raises(ScientificBudgetViolation):
        ledger.resolve("row-000", "success", {})


def test_atomic_outcome_write_failure_is_unknown(tmp_path, monkeypatch):
    import experiments.benchmarking.methods.sbi_zoo_dataset as dataset_module

    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    identity = _slot_call_identity(ledger)
    original = dataset_module.atomic_write_bytes

    def fail_response(path, payload):
        if Path(path).name == "response.f32le":
            raise OSError("synthetic atomic response failure")
        return original(path, payload)

    monkeypatch.setattr(dataset_module, "atomic_write_bytes", fail_response)
    with pytest.raises(DatasetSerializationError, match="synthetic atomic response failure"):
        ledger.invoke_once(
            "row-000",
            intent=identity,
            invoke=lambda: torch.arange(20, dtype=torch.float32),
            persist_outcome=lambda response: write_slot_outcome_artifact(
                ledger.root / "outcomes" / "row-000",
                response=response,
                call_identity=identity,
            ).receipt(),
        )
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"
    assert ledger.attempted_count() == 1


def test_reload_verification_failure_retains_unknown_artifact(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    identity = _slot_call_identity(ledger)
    outcome_root = ledger.root / "outcomes" / "row-000"

    def persist(response):
        write_slot_outcome_artifact(
            outcome_root, response=response, call_identity=identity
        )
        response_path = outcome_root / "response.f32le"
        corrupted = bytearray(response_path.read_bytes())
        corrupted[-1] ^= 1
        response_path.write_bytes(corrupted)
        return load_slot_outcome_artifact(
            outcome_root, expected_call_identity=identity
        ).receipt()

    with pytest.raises(DatasetIntegrityError):
        ledger.invoke_once(
            "row-000",
            intent=identity,
            invoke=lambda: torch.arange(20, dtype=torch.float32),
            persist_outcome=persist,
        )
    assert outcome_root.exists()
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"


def test_success_event_failure_retains_orphan_and_is_unknown(tmp_path, monkeypatch):
    import experiments.benchmarking.methods.sbi_zoo_accounting as accounting_module

    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    identity = _slot_call_identity(ledger)
    outcome_root = ledger.root / "outcomes" / "row-000"
    original = accounting_module._append_event

    def fail_success(path, document):
        if Path(path).name.endswith("-success.json"):
            raise OSError("synthetic success-ledger failure")
        return original(path, document)

    monkeypatch.setattr(accounting_module, "_append_event", fail_success)
    with pytest.raises(OSError, match="success-ledger failure"):
        ledger.invoke_once(
            "row-000",
            intent=identity,
            invoke=lambda: torch.arange(20, dtype=torch.float32),
            persist_outcome=lambda response: write_slot_outcome_artifact(
                outcome_root, response=response, call_identity=identity
            ).receipt(),
        )
    assert outcome_root.exists()
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"
    with pytest.raises(ScientificBudgetViolation):
        ledger.persist_intent("row-000", {})


def test_unknown_event_write_failure_replays_attempted_only_as_unknown(
    tmp_path, monkeypatch
):
    import experiments.benchmarking.methods.sbi_zoo_accounting as accounting_module

    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    original = accounting_module._append_event

    def fail_unknown(path, document):
        if Path(path).name.endswith("-attempted_outcome_unknown.json"):
            raise OSError("synthetic unknown-ledger failure")
        return original(path, document)

    monkeypatch.setattr(accounting_module, "_append_event", fail_unknown)
    with pytest.raises(TypeError, match="post-invocation"):
        ledger.invoke_once(
            "row-000",
            intent={"kind": "test"},
            invoke=lambda: torch.zeros(20),
            persist_outcome=lambda _: (_ for _ in ()).throw(TypeError("post-invocation")),
        )
    assert ledger.slot_status("row-000") == "attempted_outcome_unknown"
    assert ledger.terminal_payload("row-000") is None
    assert ledger.attempted_count() == 1
    with pytest.raises(ScientificBudgetViolation):
        ledger.persist_intent("row-000", {})


def test_normal_outcome_is_persisted_reloaded_then_success(tmp_path):
    ledger = WriteAheadScientificCallLedger(tmp_path / "ledger", "declaration", 40)
    identity = _slot_call_identity(ledger)
    outcome_root = ledger.root / "outcomes" / "row-000"
    response = torch.arange(20, dtype=torch.float32)
    ledger.invoke_once(
        "row-000",
        intent=identity,
        invoke=lambda: response,
        persist_outcome=lambda value: write_slot_outcome_artifact(
            outcome_root, response=value, call_identity=identity
        ).receipt(),
    )
    loaded = load_slot_outcome_artifact(
        outcome_root, expected_call_identity=identity
    )
    assert torch.equal(loaded.response, response)
    assert ledger.slot_status("row-000") == "success"
    assert dict(ledger.terminal_payload("row-000")) == loaded.receipt()
    with pytest.raises(DatasetSerializationError, match="already exists"):
        write_slot_outcome_artifact(
            outcome_root, response=response, call_identity=identity
        )
    (outcome_root / "unexpected.txt").write_text("forbidden", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="contains extras"):
        load_slot_outcome_artifact(outcome_root, expected_call_identity=identity)


def test_resolved_contract_has_no_implicit_estimator_or_mcmc_controls():
    contract = complete_implementation_contract()
    assert contract["runtime"]["python_package_sbi"] == "0.26.1"
    for budget in (40, 100):
        for method in METHODS:
            config = estimator_contract(method, budget)
            assert config["training"]["scientific_maximum_epochs"] == 200
            assert config["training"]["sbi_api_max_num_epochs_argument"] == 199
            assert config["training"]["optimizer"]["class"] == "torch.optim.Adam"
    mcmc = mcmc_contract()
    assert mcmc["requested_total_samples"] == 10_000
    assert mcmc["samples_per_chain"] == 500
    assert mcmc["aggregation"] == "sbi_chain_major_C_order_reshape_then_first_10000"
    assert mcmc["init_strategy"] == "proposal"
    assert mcmc["initialization_candidate_count"] is None
    assert contract["durable_outcomes"]["post_invocation_failure"].startswith(
        "attempted_outcome_unknown"
    )
    machine = json.loads(
        Path(
            "experiments/paper/studies/graph_paper120/analysis/"
            "sbi_zoo_extension_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        machine["status"]
        == "physical_qualification_passed_p1_execution_package_zero_call_validated"
    )
    assert machine["implementation"]["physical_qualification_complete"] is True
    assert machine["implementation"]["p1_execution"]["scientific_calls"] == 0
    assert machine["training"]["sbi_api_max_num_epochs_argument"] == 199
    assert machine["posterior_sampling"]["retained_order"].startswith("chain_major")
    assert machine["training_data"]["canonical_artifact"]["byte_order"] == "little_endian"


def test_pinned_sbi_training_source_proves_epoch_split_loader_and_optimizer_semantics():
    from sbi.inference import NLE

    loop_source = inspect.getsource(NLE._run_training_loop)
    loader_source = inspect.getsource(NLE.get_dataloaders)
    assert "while self.epoch <= train_config.max_num_epochs" in loop_source
    assert "self.optimizer = Adam(" in loop_source
    assert "lr=train_config.learning_rate" in loop_source
    assert "int((1 - validation_fraction) * num_examples)" in loader_source
    assert "torch.randperm(num_examples)" in loader_source
    assert "SubsetRandomSampler(self.train_indices.tolist())" in loader_source
    assert '"drop_last": True' in loader_source


def test_pinned_sbi_source_establishes_initialization_order_and_chain_major_flattening():
    from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
    from sbi.samplers.mcmc.init_strategy import proposal_init

    sample_source = inspect.getsource(MCMCPosterior.sample)
    slice_source = inspect.getsource(MCMCPosterior._slice_np_mcmc)
    init_source = inspect.getsource(MCMCPosterior._get_initial_params)
    proposal_source = inspect.getsource(proposal_init)
    potential_source = inspect.getsource(MCMCPosterior.potential)
    prepared_source = inspect.getsource(MCMCPosterior._prepare_potential)
    assert "num_samples = torch.Size(sample_shape).numel()" in sample_source
    assert "ceil((num_samples * thin) / num_chains)" in slice_source
    assert "samples.reshape(-1, dim_samples)[:num_samples]" in slice_source
    assert "for _ in" in init_source and "range(num_chains)" in init_source
    assert "proposal.sample((1,))" in proposal_source
    assert "self.potential_fn(" in potential_source
    assert "transformed_potential" not in potential_source
    assert "transformed_potential" in prepared_source


@pytest.mark.parametrize("method", METHODS)
def test_pinned_inference_api_construction_uses_explicit_builder(method, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path / method))
    inference, prior = build_inference(method, summary_writer=writer)
    writer.close()
    assert inference._prior is prior
    expected_class = "NLE_A" if method == METHODS[0] else "NRE_B"
    assert type(inference).__name__ == expected_class
    _, key, builder = _builders(method)
    assert key == ("density_estimator" if method == METHODS[0] else "classifier")
    closure = inspect.getclosurevars(builder).nonlocals
    assert closure["builder_kwargs"]["hidden_features"] == 50
    assert closure["builder_kwargs"]["z_score_x"] == "independent"
    assert closure["builder_kwargs"]["z_score_y"] == "independent"
    if method == METHODS[0]:
        assert closure["model"] == "maf"
        assert closure["builder_kwargs"]["num_transforms"] == 5
        assert closure["builder_kwargs"]["num_blocks"] == 2
    else:
        assert closure["model"] == "mlp"
        assert closure["builder_kwargs"]["norm_layer"] is torch.nn.LayerNorm


def test_production_training_wrapper_consumes_exact_dataset_and_pinned_kwargs(
    tmp_path, monkeypatch
):
    theta, response = _arrays(40)
    dataset = write_shared_training_dataset(
        tmp_path / "dataset", theta=theta, response=response, metadata=_metadata(40)
    )

    class FakeInference:
        def append_simulations(self, seen_theta, seen_response):
            assert torch.equal(seen_theta, dataset.theta)
            assert torch.equal(seen_response, dataset.response)
            return self

        def train(self, **kwargs):
            self.kwargs = kwargs
            return "estimator"

    fake = FakeInference()
    monkeypatch.setattr(
        "experiments.benchmarking.methods.sbi_zoo_inference.build_inference",
        lambda method, summary_writer: (fake, "prior"),
    )
    trained = train_single_round(
        dataset,
        method=METHODS[1],
        training_seed=991,
        training_log_dir=tmp_path / "training-log",
    )
    assert trained.estimator == "estimator"
    assert trained.dataset_receipt["canonical_content_sha256"] == dataset.content_sha256
    assert fake.kwargs["num_atoms"] == 2
    assert fake.kwargs["max_num_epochs"] == 199
    assert fake.kwargs["dataloader_kwargs"]["num_workers"] == 0
    assert fake.kwargs["dataloader_kwargs"]["in_order"] is True


class _FakeMCMCPosterior:
    def __init__(self, samples: torch.Tensor, *, invalid_init: bool = False):
        self.samples = samples
        self.invalid_init = invalid_init

    def _get_initial_params(self, *args, **kwargs):
        del args, kwargs
        result = torch.zeros((20, 4), dtype=torch.float32)
        if self.invalid_init:
            result[0, 0] = float("nan")
        return result

    def sample(self, shape, show_progress_bars=False):
        del show_progress_bars
        self._get_initial_params("proposal", 20, 1, False)
        assert shape == (10_000,)
        return self.samples.clone()


class _RandomFakeMCMCPosterior(_FakeMCMCPosterior):
    def sample(self, shape, show_progress_bars=False):
        del show_progress_bars
        self._get_initial_params("proposal", 20, 1, False)
        assert shape == (10_000,)
        unit = np.random.rand(POSTERIOR_SAMPLE_COUNT, 4).astype("float32")
        lower, upper = PRIOR_LAW.bounds()
        return lower + torch.from_numpy(unit) * (upper - lower)


def test_exact_10000_sample_order_is_preserved_and_initialization_is_guarded():
    samples = torch.empty((POSTERIOR_SAMPLE_COUNT, 4), dtype=torch.float32)
    for index in range(POSTERIOR_SAMPLE_COUNT):
        samples[index] = torch.tensor([0.5 + index / 20_000, 0.5, 0.5, 0.25])
    posterior = _FakeMCMCPosterior(samples)
    actual = sample_exact_posterior_candidates(posterior, recommendation_seed=123)
    assert torch.equal(actual, samples)
    assert actual[500, 0] == samples[500, 0]
    with pytest.raises(MCMCInitializationFailure):
        sample_exact_posterior_candidates(
            _FakeMCMCPosterior(samples, invalid_init=True), recommendation_seed=123
        )
    with pytest.raises(InsufficientPosteriorSamples):
        sample_exact_posterior_candidates(
            _FakeMCMCPosterior(samples[:-1]), recommendation_seed=123
        )


def test_mcmc_sampling_is_deterministic_and_isolates_torch_and_numpy_rng():
    torch.manual_seed(13)
    np.random.seed(14)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    first = sample_exact_posterior_candidates(
        _RandomFakeMCMCPosterior(torch.empty(0)), recommendation_seed=987654321
    )
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    second = sample_exact_posterior_candidates(
        _RandomFakeMCMCPosterior(torch.empty(0)), recommendation_seed=987654321
    )
    assert torch.equal(first, second)


class _PhysicalPotentialPosterior:
    def potential(self, theta, track_gradients=False):
        assert track_gradients is False
        return theta[:, 0]


def test_final_map_uses_physical_potential_not_transformed_jacobian():
    posterior = _PhysicalPotentialPosterior()
    samples = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [1.9, 0.5, 0.5, 0.5]], dtype=torch.float32
    )
    # A deliberately opposite transformed-space score demonstrates that the
    # final selector receives only the physical potential.
    transformed_with_jacobian = torch.tensor([100.0, -100.0])
    assert int(torch.argmax(transformed_with_jacobian)) == 0
    result = posterior_sample_map_decision(
        method=METHODS[0],
        posterior=posterior,
        samples=samples,
        lower=PRIOR_LAW.bounds()[0],
        upper=PRIOR_LAW.bounds()[1],
        method_configuration_identity="config",
        inference_budget=40,
        inference_calls=PhysicalCallCounts(40, 40),
        physical_score=lambda theta: physical_posterior_score(posterior, theta),
        physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[METHODS[0]],
        physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
    )
    assert result.decision.recommended_theta == tuple(samples[1].tolist())
    assert result.decision.decision_metadata["selected_candidate_index"] == 1
    assert result.decision.decision_metadata["transform_jacobian_policy"] == "none"
    # The canonical paper-record validator accepts the new method-specific
    # physical-potential score API without weakening historical NPE/SNPE rules.
    expanded = []
    for index in range(5_000):
        expanded.extend(samples)
    full_samples = torch.stack(expanded)
    full_result = posterior_sample_map_decision(
        method=METHODS[0],
        posterior=posterior,
        samples=full_samples,
        lower=PRIOR_LAW.bounds()[0],
        upper=PRIOR_LAW.bounds()[1],
        method_configuration_identity="config",
        inference_budget=40,
        inference_calls=PhysicalCallCounts(40, 40),
        physical_score=lambda theta: physical_posterior_score(posterior, theta),
        physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[METHODS[0]],
        physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
    )
    _validate_posterior_sample_map_decision(asdict(full_result.decision))


def test_earliest_tie_support_mask_and_nonfinite_score_for_new_methods():
    samples = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [0.6, 0.5, 0.5, 0.5], [9.0, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
    )
    for method in METHODS:
        result = posterior_sample_map_decision(
            method=method,
            posterior=SimpleNamespace(),
            samples=samples,
            lower=PRIOR_LAW.bounds()[0],
            upper=PRIOR_LAW.bounds()[1],
            method_configuration_identity="config",
            inference_budget=40,
            inference_calls=PhysicalCallCounts(40, 40),
            physical_score=lambda theta: torch.zeros(theta.shape[0]),
            physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[method],
            physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
        )
        assert result.decision.decision_metadata["selected_candidate_index"] == 0


@pytest.mark.parametrize("method", ["sbi_npe", "sbi_snpe"])
def test_historical_direct_posterior_rejects_custom_score_and_wrong_api(method):
    base = {
        "method": method,
        "posterior": _PhysicalPotentialPosterior(),
        "samples": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        "lower": PRIOR_LAW.bounds()[0],
        "upper": PRIOR_LAW.bounds()[1],
        "method_configuration_identity": "historical-config",
        "inference_budget": 40,
        "inference_calls": PhysicalCallCounts(40, 40),
    }
    with pytest.raises(SbiZooScoringContractError):
        posterior_sample_map_decision(
            **base,
            physical_score=lambda theta: theta[:, 0],
            physical_score_identity="disguised-custom-score",
        )
    with pytest.raises(SbiZooScoringContractError):
        posterior_sample_map_decision(**base, physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API)


@pytest.mark.parametrize("method", METHODS)
def test_zoo_score_contract_rejects_missing_wrong_api_and_wrong_identity(method):
    base = {
        "method": method,
        "posterior": _PhysicalPotentialPosterior(),
        "samples": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        "lower": PRIOR_LAW.bounds()[0],
        "upper": PRIOR_LAW.bounds()[1],
        "method_configuration_identity": "zoo-config",
        "inference_budget": 40,
        "inference_calls": PhysicalCallCounts(40, 40),
    }
    with pytest.raises(SbiZooScoringContractError):
        posterior_sample_map_decision(**base)
    with pytest.raises(SbiZooScoringContractError):
        posterior_sample_map_decision(
            **base,
            physical_score=lambda theta: theta[:, 0],
            physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[method],
            physical_score_api=HISTORICAL_DIRECT_POSTERIOR_SCORE_API,
        )
    other = METHODS[1] if method == METHODS[0] else METHODS[0]
    for invalid_identity in ("", "arbitrary", SBI_ZOO_SCORE_IDENTITIES[other]):
        with pytest.raises(SbiZooScoringContractError):
            posterior_sample_map_decision(
                **base,
                physical_score=lambda theta: theta[:, 0],
                physical_score_identity=invalid_identity,
                physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
            )


@pytest.mark.parametrize("method", METHODS)
def test_protocol_rejects_zoo_score_api_and_identity_mismatches(method):
    samples = torch.full((10_000, 4), 0.75, dtype=torch.float32)
    result = posterior_sample_map_decision(
        method=method,
        posterior=_PhysicalPotentialPosterior(),
        samples=samples,
        lower=PRIOR_LAW.bounds()[0],
        upper=PRIOR_LAW.bounds()[1],
        method_configuration_identity="zoo-config",
        inference_budget=40,
        inference_calls=PhysicalCallCounts(40, 40),
        physical_score=lambda theta: theta[:, 0],
        physical_score_identity=SBI_ZOO_SCORE_IDENTITIES[method],
        physical_score_api=SBI_ZOO_PHYSICAL_SCORE_API,
    )
    valid = asdict(result.decision)
    _validate_posterior_sample_map_decision(valid)
    wrong_api = json.loads(json.dumps(valid))
    wrong_api["decision_metadata"]["score_api"] = HISTORICAL_DIRECT_POSTERIOR_SCORE_API
    with pytest.raises(SbiZooScoringContractError):
        _validate_posterior_sample_map_decision(wrong_api)
    wrong_identity = json.loads(json.dumps(valid))
    other = METHODS[1] if method == METHODS[0] else METHODS[0]
    wrong_identity["decision_metadata"]["score_identity"] = SBI_ZOO_SCORE_IDENTITIES[other]
    with pytest.raises(SbiZooScoringContractError):
        _validate_posterior_sample_map_decision(wrong_identity)


def test_installed_ratio_potential_orientation_is_log_ratio_plus_prior():
    from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential

    class Ratio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, theta, x):
            del x
            return (3.0 * theta[:, 0] + self.anchor).reshape(-1, 1)

    prior = authoritative_prior()
    potential = RatioBasedPotential(
        Ratio(), prior, x_o=torch.zeros((1, 20), dtype=torch.float32), device="cpu"
    )
    theta = torch.tensor([[0.75, 1.0, 1.0, 0.2], [1.25, 1.0, 1.0, 0.2]])
    scores = potential(theta, track_gradients=False)
    expected = 3.0 * theta[:, 0] + prior.log_prob(theta)
    assert torch.allclose(scores, expected)
    assert scores[1] > scores[0]


def test_installed_likelihood_potential_is_log_likelihood_plus_prior():
    from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential

    class Likelihood(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def log_prob(self, x, condition):
            del x
            return (2.0 * condition[:, 0] + self.anchor).unsqueeze(0)

    prior = authoritative_prior()
    potential = LikelihoodBasedPotential(
        Likelihood(), prior, x_o=torch.zeros((1, 20), dtype=torch.float32), device="cpu"
    )
    theta = torch.tensor([[0.75, 1.0, 1.0, 0.2], [1.25, 1.0, 1.0, 0.2]])
    scores = potential(theta, track_gradients=False)
    expected = 2.0 * theta[:, 0] + prior.log_prob(theta)
    assert torch.allclose(scores, expected)


@pytest.mark.parametrize("method", METHODS)
def test_minimal_real_sbi_training_posterior_and_mcmc_api_smoke(method, tmp_path):
    """One synthetic epoch and four MCMC draws; never a scientific fit."""

    theta, response = _arrays(40)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path / method))
    inference, prior = build_inference(method, summary_writer=writer)
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(441)
            np.random.seed(442)
            inference.append_simulations(theta, response)
            kwargs = {
                "training_batch_size": 32,
                "learning_rate": 5e-4,
                "validation_fraction": 0.2,
                "stop_after_epochs": 1,
                # Pinned sbi requires a positive value and uses an inclusive loop;
                # one is therefore its smallest accepted smoke setting (two epochs).
                "max_num_epochs": 1,
                "clip_max_norm": 5.0,
                "show_train_summary": False,
                "dataloader_kwargs": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "timeout": 0,
                    "persistent_workers": False,
                    "collate_fn": None,
                    "worker_init_fn": None,
                    "multiprocessing_context": None,
                    "generator": None,
                    "prefetch_factor": None,
                    "pin_memory_device": "",
                    "in_order": True,
                },
            }
            if method == METHODS[1]:
                kwargs["num_atoms"] = 2
            estimator = inference.train(**kwargs)
            trained = TrainedSbiZooEstimator(
                method=method,
                budget=40,
                inference=inference,
                estimator=estimator,
                prior=prior,
                dataset_receipt={"synthetic": True},
                effective_configuration=estimator_contract(method, 40),
                training_seed=441,
            )
            posterior = build_mcmc_posterior(trained, response[0])
            samples = posterior.sample(
                (4,),
                warmup_steps=1,
                num_chains=2,
                init_strategy="proposal",
                init_strategy_parameters={},
                show_progress_bars=False,
            )
    finally:
        np.random.set_state(numpy_state)
    writer.close()
    assert samples.shape == (4, 4)
    assert bool(torch.isfinite(samples).all())
    assert bool(torch.isfinite(physical_posterior_score(posterior, samples)).all())
