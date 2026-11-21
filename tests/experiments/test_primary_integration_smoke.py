from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mutaco import ScientificIdentity, SeedDeriver, SimulationContext

from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.graph_inversion.prepare_primary_integration_smoke import materialize
from experiments.graph_inversion.primary_integration_smoke_driver import (
    GraphPrimaryIntegrationSmokeDriver,
    primary_smoke_method_identities,
)
from experiments.graph_inversion.run_primary_integration_smoke_worker import (
    _load_smoke_target,
    _simulation_context,
    _validate_plan,
)
from experiments.paper.primary_benchmark import (
    generate_primary_sobol_design,
    load_primary_population_declaration,
    primary_assessment_context,
    primary_method_context,
)


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "experiments/graph_inversion/configs/paper/primary_prequalification_b40_integration_smoke_v1.json"
PLAN = ROOT / "experiments/graph_inversion/configs/paper/primary_prequalification_b40_integration_smoke_plan_v1.json"
CONFIG = ROOT / "experiments/graph_inversion/configs/paper/policy_upgrade_100_targets_v1.json"
Q1 = ROOT / "experiments/graph_inversion/configs/paper/primary_nei_q1_evidence_repair_v1.json"


def test_smoke_declaration_and_plan_are_canonical_and_hash_paired() -> None:
    declaration, plan = materialize(
        q1_declaration=Q1,
        configuration=CONFIG,
        output_root="/arf/scratch/REPLACE_WITH_TRUBA_USER/mutaco/primary_three_method_smoke/prequalification-b40-v1/workers",
    )
    assert read_json(DECLARATION) == declaration
    assert read_json(PLAN) == plan
    assert DECLARATION.with_suffix(".sha256").read_text().strip() == file_hash(DECLARATION)
    assert PLAN.with_suffix(".sha256").read_text().strip() == file_hash(PLAN)
    loaded = load_primary_population_declaration(declaration)
    assert loaded.document()["call_preview"]["maximum_total_calls"] == 123
    _validate_plan(plan, loaded.document()["declaration_identity"])


def test_smoke_driver_identities_and_sobol_path_are_zero_model_call() -> None:
    declaration = load_primary_population_declaration(read_json(DECLARATION))
    assert {
        item.method: item.configuration_identity for item in declaration.methods
    } == primary_smoke_method_identities(CONFIG)
    driver = GraphPrimaryIntegrationSmokeDriver(
        declaration=declaration, configuration_path=CONFIG
    )
    target_id = declaration.target_record_identities[0]
    family = "primary-smoke-test-design-family"
    design = generate_primary_sobol_design(
        campaign_id=declaration.campaign_id,
        target_id=target_id,
        budget=40,
        parameter_spec=driver.parameter_spec,
        seed_deriver=SeedDeriver(
            ScientificIdentity(family), declaration.comparison.design_campaign_seed
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    target = {
        "target_record_id": target_id,
        "target_index": 0,
        "target_response": [0.0] * driver.response_spec.dimension,
        "theta_star": [0.0] * driver.parameter_spec.dimension,
        "target_generation_seed": 1,
    }
    calls = []

    def evaluate(theta, index):
        calls.append((tuple(theta), index))
        return [float(sum(theta))] * driver.response_spec.dimension

    method = declaration.methods[2]
    outcome = driver.run_method(
        target,
        method,
        primary_method_context(declaration, target_id, "sobol"),
        design,
        evaluate,
    )
    assert outcome.status == "recommended"
    assert outcome.successful_search_calls == len(calls) == 40
    assert all(item.gp_fit_status is None for item in outcome.trajectory)
    assert all(item.acquisition_diagnostics is None for item in outcome.trajectory)
    assert outcome.terminal_metadata["simulator_calls"] == 0
    assert set(outcome.timing) == {
        "simulator", "gp_fit", "acquisition", "terminal", "other", "total"
    }
    assert outcome.timing["gp_fit"] == outcome.timing["acquisition"] == 0.0
    assert outcome.timing["terminal"] > 0.0


def test_smoke_target_loader_selects_one_hash_verified_q1_target(
    tmp_path: Path,
) -> None:
    declaration = load_primary_population_declaration(read_json(DECLARATION))
    target_id = declaration.target_record_identities[0]
    for index, identity in enumerate((target_id, "unselected-target")):
        path = tmp_path / f"target_{index:03d}.json"
        write_pretty_json(path, {
            "target_record_id": identity,
            "declaration_hash": declaration.target_ledger_hash,
        })
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")
    targets = _load_smoke_target(
        tmp_path,
        target_id=target_id,
        target_ledger_hash=declaration.target_ledger_hash,
    )
    assert tuple(targets) == (target_id,)


def test_production_smoke_bridge_accepts_paired_assessment_context() -> None:
    declaration = load_primary_population_declaration(read_json(DECLARATION))
    context = primary_assessment_context(
        declaration, declaration.target_record_identities[0]
    )
    restored = _simulation_context(context)
    assert restored == SimulationContext(
        context["family_identity"],
        "paired_held_out_assessment",
        0,
        0,
        context["evaluation_seed"],
    )


def test_production_smoke_bridge_reproduces_archived_incomplete_context() -> None:
    with pytest.raises(KeyError, match="purpose"):
        _simulation_context({
            "family_identity": "archived-assessment-family",
            "evaluation_seed": 17,
            "replicate": 0,
        })
