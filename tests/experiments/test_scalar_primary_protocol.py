from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mutaco import ScientificIdentity, SeedCoordinate, SeedDeriver
from experiments.benchmarking.artifacts import file_hash, read_json
from experiments.benchmarking.methods.scalar_qlognei import (
    ScalarQLogNEIConfig,
    fit_scalar_loss_gp,
    propose_scalar_qlognei,
    recommend_scalar_posterior_mean_loss,
    scalar_qlognei_settings_from_document,
)
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.graph_inversion.prepare_scalar_primary_integration_smoke import (
    materialize_scalar_smoke,
)
from experiments.graph_inversion.prepare_scalar_primary_population import (
    materialize_scalar_population,
)
from experiments.graph_inversion.primary_integration_smoke_driver import (
    GraphPrimaryIntegrationSmokeDriver,
    scalar_primary_method_identities,
)
from experiments.graph_inversion.run_primary_integration_smoke_worker import (
    _load_population_targets,
    _validate_plan,
)
from experiments.paper.common.validation import PaperContractError
from experiments.paper.primary_benchmark import (
    load_primary_population_declaration,
    load_scalar_primary_population_declaration,
    scalar_primary_call_preview,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json"
B100_CONFIG = ROOT / "experiments/graph_inversion/configs/paper/paper120_b100_mutaco_v1.json"
SETTINGS = ROOT / "experiments/graph_inversion/configs/paper/scalar_qlognei_primary_settings_v1.json"
STUDY = ROOT / "experiments/paper/studies/graph_paper120/studies/mutaco_scalar_qlognei_sobol_primary_v1.json"
Q1 = ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_evidence_repair_v1.json"
DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_prequalification_b40_v1.json"
PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_prequalification_b40_plan_v1.json"
B100_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_prequalification_b100_v1.json"
B100_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_prequalification_b100_plan_v1.json"
COHORT = ROOT / "experiments/graph_inversion/frozen_targets/paper120_v3"
POPULATION_B40_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v1.json"
POPULATION_B40_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_plan_v1.json"
POPULATION_B100_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v1.json"
POPULATION_B100_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_plan_v1.json"
POPULATION_B40_V2_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v2.json"
POPULATION_B40_V2_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_plan_v2.json"
POPULATION_B100_V2_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_v2.json"
POPULATION_B100_V2_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b100_plan_v2.json"
POPULATION_B40_V3_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v3.json"
POPULATION_B40_V3_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_plan_v3.json"
POPULATION_B40_V3R1_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v3r1.json"
POPULATION_B40_V3R1_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_plan_v3r1.json"
POPULATION_B40_V3R2_DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_v3r2.json"
POPULATION_B40_V3R2_PLAN = ROOT / "experiments/graph_inversion/configs/paper/scalar_primary_population_b40_plan_v3r2.json"


def test_scalar_protocol_call_caps_and_historical_loader_are_disjoint() -> None:
    assert scalar_primary_call_preview(1, 40)["maximum_total_calls"] == 123
    assert scalar_primary_call_preview(1, 100)["maximum_total_calls"] == 303
    historical = read_json(
        ROOT / "experiments/graph_inversion/configs/paper/primary_prequalification_b40_integration_smoke_v1.json"
    )
    assert load_primary_population_declaration(historical).document() == historical
    with pytest.raises(PaperContractError, match="schema"):
        load_primary_population_declaration(read_json(DECLARATION))


def test_scalar_smoke_declaration_plan_and_settings_are_canonical() -> None:
    settings_document = read_json(SETTINGS)
    settings = scalar_qlognei_settings_from_document(settings_document)
    declaration, plan = materialize_scalar_smoke(
        q1_declaration=Q1,
        study=STUDY,
        configuration=CONFIG,
        settings=SETTINGS,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_smoke/b40-v1/workers",
    )
    assert read_json(DECLARATION) == declaration
    assert read_json(PLAN) == plan
    assert DECLARATION.with_suffix(".sha256").read_text().strip() == file_hash(DECLARATION)
    assert PLAN.with_suffix(".sha256").read_text().strip() == file_hash(PLAN)
    loaded = load_scalar_primary_population_declaration(declaration)
    assert loaded.document()["call_preview"]["maximum_total_calls"] == 123
    _validate_plan(plan, declaration["declaration_identity"])
    assert {
        item.method: item.configuration_identity for item in loaded.methods
    } == scalar_primary_method_identities(CONFIG, budget=40, qlognei_settings=settings)
    GraphPrimaryIntegrationSmokeDriver(
        declaration=loaded,
        configuration_path=CONFIG,
        qlognei_settings=settings,
    )
    tampered = dict(settings_document)
    tampered["batch_q"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        scalar_qlognei_settings_from_document(tampered)


def test_b100_scalar_driver_surface_accepts_matching_configuration() -> None:
    settings = scalar_qlognei_settings_from_document(read_json(SETTINGS))
    declaration, plan = materialize_scalar_smoke(
        q1_declaration=Q1,
        study=STUDY,
        configuration=B100_CONFIG,
        settings=SETTINGS,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_smoke/b100-v1/workers",
        budget=100,
    )
    assert read_json(B100_DECLARATION) == declaration
    assert read_json(B100_PLAN) == plan
    assert B100_DECLARATION.with_suffix(".sha256").read_text().strip() == file_hash(
        B100_DECLARATION
    )
    assert B100_PLAN.with_suffix(".sha256").read_text().strip() == file_hash(B100_PLAN)
    loaded = load_scalar_primary_population_declaration(declaration)
    assert loaded.document()["call_preview"]["maximum_total_calls"] == 303
    _validate_plan(plan, declaration["declaration_identity"])
    assert {
        item.method: item.configuration_identity for item in loaded.methods
    } == scalar_primary_method_identities(
        B100_CONFIG, budget=100, qlognei_settings=settings
    )
    GraphPrimaryIntegrationSmokeDriver(
        declaration=loaded,
        configuration_path=B100_CONFIG,
        qlognei_settings=settings,
    )
    tampered = dict(plan)
    tampered["maximum_total_calls"] = 123
    with pytest.raises(PaperContractError, match="scalar primary smoke plan"):
        _validate_plan(tampered, declaration["declaration_identity"])


def test_scalar_qlognei_exact_b100_capacity_is_float32_zero_call() -> None:
    settings = scalar_qlognei_settings_from_document(read_json(SETTINGS))
    simulator = GraphScientificD4Adapter().build_target_simulator(
        read_json(CONFIG), device=torch.device("cpu"), dtype=torch.float32
    )
    spec = simulator.capabilities.parameter_spec
    unit = torch.quasirandom.SobolEngine(spec.dimension, scramble=True, seed=20260816).draw(100)
    lower = torch.tensor(spec.lower, dtype=torch.float32)
    upper = torch.tensor(spec.upper, dtype=torch.float32)
    theta = lower + unit * (upper - lower)
    phase = theta.sum(dim=1, keepdim=True)
    axes = torch.linspace(0.2, 1.2, 20, dtype=torch.float32).unsqueeze(0)
    responses = torch.sin(phase * axes) + 0.05 * theta[:, :1]
    target = torch.zeros(20, dtype=torch.float32)
    deriver = SeedDeriver(ScientificIdentity("scalar-b100-capacity-v1"), 92021131)
    config = ScalarQLogNEIConfig(
        n_mc=settings["n_mc"],
        raw_samples=settings["raw_samples"],
        restarts=settings["restarts"],
        max_iterations=settings["max_iterations"],
        fit_max_iterations=settings["fit_max_iterations"],
        seed_deriver=deriver,
        fit_coordinate=SeedCoordinate("fit", (100,)),
        sampler_coordinate=SeedCoordinate("mc", (100,)),
        optimizer_coordinate=SeedCoordinate("optimizer", (100,)),
    )
    fitted = fit_scalar_loss_gp(
        theta=theta,
        responses_physical=responses,
        target_physical=target,
        parameter_spec=spec,
        config=config,
    )
    proposal = propose_scalar_qlognei(
        fitted=fitted, baseline=theta, parameter_spec=spec, config=config
    )
    terminal = recommend_scalar_posterior_mean_loss(
        fitted=fitted,
        parameter_spec=spec,
        reference=theta,
        seed_deriver=deriver,
        coordinate=SeedCoordinate("terminal", (100,)),
        raw_samples=settings["terminal_raw_samples"],
        restarts=settings["terminal_restarts"],
        max_iterations=settings["terminal_max_iterations"],
    )
    assert fitted.model.train_inputs[0].shape[-2] == 100
    assert proposal.provenance["baseline_covariance_dimension"] == 100
    assert proposal.candidate.dtype is torch.float32
    assert proposal.provenance["precision_fallback_used"] is False
    assert terminal.document()["simulator_calls"] == 0
    assert terminal.document()["precision_fallback_used"] is False


@pytest.mark.parametrize(
    ("budget", "configuration", "output_root", "declaration_path", "plan_path", "cap"),
    (
        (
            40,
            CONFIG,
            "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b40-v1/workers",
            POPULATION_B40_DECLARATION,
            POPULATION_B40_PLAN,
            14760,
        ),
        (
            100,
            B100_CONFIG,
            "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v1/workers",
            POPULATION_B100_DECLARATION,
            POPULATION_B100_PLAN,
            36360,
        ),
    ),
)
def test_scalar_population_declaration_and_one_target_shards_are_canonical(
    budget: int,
    configuration: Path,
    output_root: str,
    declaration_path: Path,
    plan_path: Path,
    cap: int,
) -> None:
    declaration, plan = materialize_scalar_population(
        cohort_root=COHORT,
        study=STUDY,
        configuration=configuration,
        settings=SETTINGS,
        output_root=output_root,
        budget=budget,
    )
    assert read_json(declaration_path) == declaration
    assert read_json(plan_path) == plan
    assert declaration_path.with_suffix(".sha256").read_text().strip() == file_hash(
        declaration_path
    )
    assert plan_path.with_suffix(".sha256").read_text().strip() == file_hash(plan_path)
    loaded = load_scalar_primary_population_declaration(declaration)
    assert len(loaded.target_record_identities) == 120
    assert loaded.document()["call_preview"]["maximum_total_calls"] == cap
    assert plan["worker_count"] == 120
    assert plan["maximum_calls_per_worker"] == 3 * (budget + 1)
    assert plan["scheduler"]["worker_packing"] is False
    _validate_plan(plan, declaration["declaration_identity"])
    targets = _load_population_targets(
        COHORT,
        target_ids=loaded.target_record_identities,
        target_ledger_hash=loaded.target_ledger_hash,
    )
    assert len(targets) == 120
    GraphPrimaryIntegrationSmokeDriver(
        declaration=loaded,
        configuration_path=configuration,
        qlognei_settings=scalar_qlognei_settings_from_document(read_json(SETTINGS)),
    )


def test_scalar_population_budgets_are_independent_and_policy_is_identical() -> None:
    b40 = read_json(POPULATION_B40_DECLARATION)
    b100 = read_json(POPULATION_B100_DECLARATION)
    assert b40["campaign_id"] != b100["campaign_id"]
    assert b40["comparison"]["design_campaign_seed"] != b100["comparison"]["design_campaign_seed"]
    assert b40["comparison"]["assessment_campaign_seed"] != b100["comparison"]["assessment_campaign_seed"]
    assert b40["target_record_identities"] == b100["target_record_identities"]
    assert [item["method"] for item in b40["methods"]] == [
        "mutaco",
        "scalar_qlognei",
        "sobol",
    ]
    assert [item["search_policy_identifier"] for item in b40["methods"]] == [
        item["search_policy_identifier"] for item in b100["methods"]
    ]


@pytest.mark.parametrize(
    ("budget", "configuration", "output_root", "declaration_path", "plan_path"),
    (
        (
            40,
            CONFIG,
            "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b40-v2/workers",
            POPULATION_B40_V2_DECLARATION,
            POPULATION_B40_V2_PLAN,
        ),
        (
            100,
            B100_CONFIG,
            "/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b100-v2/workers",
            POPULATION_B100_V2_DECLARATION,
            POPULATION_B100_V2_PLAN,
        ),
    ),
)
def test_scalar_population_v2_respects_truba_array_limit_without_policy_change(
    budget: int,
    configuration: Path,
    output_root: str,
    declaration_path: Path,
    plan_path: Path,
) -> None:
    declaration, plan = materialize_scalar_population(
        cohort_root=COHORT,
        study=STUDY,
        configuration=configuration,
        settings=SETTINGS,
        output_root=output_root,
        budget=budget,
        execution_revision=2,
    )
    assert read_json(declaration_path) == declaration
    assert read_json(plan_path) == plan
    _validate_plan(plan, declaration["declaration_identity"])
    physical = plan["scheduler"]["physical_arrays"]
    mapped = list(range(100)) + [index + 100 for index in range(20)]
    assert mapped == list(range(120))
    assert [item["task_indices"] for item in physical] == ["0-99", "0-19"]
    assert [item["rank_offset"] for item in physical] == [0, 100]
    v1 = read_json(
        POPULATION_B40_DECLARATION if budget == 40 else POPULATION_B100_DECLARATION
    )
    for key in ("comparison", "methods", "target_record_identities", "budget"):
        assert declaration[key] == v1[key]
    assert declaration["output_root"] != v1["output_root"]


def test_scalar_population_v3_packs_two_logical_workers_sequentially_without_policy_change() -> None:
    declaration, plan = materialize_scalar_population(
        cohort_root=COHORT,
        study=STUDY,
        configuration=CONFIG,
        settings=SETTINGS,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b40-v3/workers",
        budget=40,
        execution_revision=3,
    )
    assert read_json(POPULATION_B40_V3_DECLARATION) == declaration
    assert read_json(POPULATION_B40_V3_PLAN) == plan
    _validate_plan(plan, declaration["declaration_identity"])
    scheduler = plan["scheduler"]
    mapped = list(range(60)) + [index + 60 for index in range(60)]
    assert sorted(mapped) == list(range(120))
    assert scheduler["array_indices"] == "0-59"
    assert scheduler["logical_rank_mapping"] == ["task_id", "task_id+60"]
    assert scheduler["submitted_job_count"] == 61
    assert scheduler["worker_walltime"] == "04:00:00"
    assert scheduler["transactional_hold_release"] is True
    v2 = read_json(POPULATION_B40_V2_DECLARATION)
    for key in ("comparison", "methods", "target_record_identities", "budget"):
        assert declaration[key] == v2[key]
    assert declaration["output_root"] != v2["output_root"]


def test_scalar_population_v3r1_is_fresh_and_canonical() -> None:
    declaration, plan = materialize_scalar_population(
        cohort_root=COHORT,
        study=STUDY,
        configuration=CONFIG,
        settings=SETTINGS,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b40-v3r1/workers",
        budget=40,
        execution_revision=3,
    )
    assert read_json(POPULATION_B40_V3R1_DECLARATION) == declaration
    assert read_json(POPULATION_B40_V3R1_PLAN) == plan
    _validate_plan(plan, declaration["declaration_identity"])
    v3 = read_json(POPULATION_B40_V3_DECLARATION)
    for key in ("comparison", "methods", "target_record_identities", "budget"):
        assert declaration[key] == v3[key]
    assert declaration["output_root"] != v3["output_root"]


def test_scalar_population_v3r2_uses_delayed_start_without_policy_change() -> None:
    declaration, plan = materialize_scalar_population(
        cohort_root=COHORT,
        study=STUDY,
        configuration=CONFIG,
        settings=SETTINGS,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/scalar_primary_population/b40-v3r2/workers",
        budget=40,
        execution_revision=4,
    )
    assert read_json(POPULATION_B40_V3R2_DECLARATION) == declaration
    assert read_json(POPULATION_B40_V3R2_PLAN) == plan
    _validate_plan(plan, declaration["declaration_identity"])
    scheduler = plan["scheduler"]
    assert scheduler["array_begin_delay"] == "now+10minutes"
    assert scheduler["transactional_delayed_start"] is True
    assert scheduler["rollback_before_begin"] is True
    assert scheduler["worker_walltime"] == "04:00:00"
    v3r1 = read_json(POPULATION_B40_V3R1_DECLARATION)
    for key in ("comparison", "methods", "target_record_identities", "budget"):
        assert declaration[key] == v3r1[key]
    assert declaration["output_root"] != v3r1["output_root"]
