from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mutaco import (
    InitialDesignPolicy,
    ParameterAxis,
    ParameterSpec,
    ScientificIdentity,
    SeedCoordinate,
    SeedDeriver,
    generate_initial_design,
    initial_design_support_markdown,
    validate_caller_design,
)


def spec(dimension: int) -> ParameterSpec:
    return ParameterSpec(
        tuple(ParameterAxis(f"theta_{index}", -1.0, 2.0) for index in range(dimension))
    )


def deriver() -> SeedDeriver:
    return SeedDeriver(ScientificIdentity.create({"experiment": "design-test"}), 1729)


@pytest.mark.parametrize("dimension", [2, 6])
def test_scrambled_sobol_is_deterministic_bounded_and_dimension_generic(
    dimension: int,
) -> None:
    coordinate = SeedCoordinate("initial_design", ("scientific-repeat", 0))
    first = generate_initial_design(
        parameter_spec=spec(dimension),
        point_count=8,
        seed_deriver=deriver(),
        seed_coordinate=coordinate,
    )
    second = generate_initial_design(
        parameter_spec=spec(dimension),
        point_count=8,
        seed_deriver=deriver(),
        seed_coordinate=coordinate,
    )
    assert torch.equal(first.theta, second.theta)
    assert first.design_id == second.design_id
    assert first.theta.shape == (8, dimension)
    assert first.theta.dtype == torch.float64
    assert bool(((first.theta >= -1.0) & (first.theta <= 2.0)).all())
    assert first.policy_id == "scrambled_sobol"
    assert first.classification == "D default"


def test_design_seeds_and_policies_are_discriminating_and_rng_neutral() -> None:
    before = torch.random.get_rng_state().clone()
    sobol = generate_initial_design(
        parameter_spec=spec(2),
        point_count=8,
        seed_deriver=deriver(),
        seed_coordinate=SeedCoordinate("initial_design", (0,)),
    )
    different = generate_initial_design(
        parameter_spec=spec(2),
        point_count=8,
        seed_deriver=deriver(),
        seed_coordinate=SeedCoordinate("initial_design", (1,)),
    )
    iid = generate_initial_design(
        parameter_spec=spec(2),
        point_count=8,
        seed_deriver=deriver(),
        seed_coordinate=SeedCoordinate("initial_design", (0,)),
        policy=InitialDesignPolicy.IID_UNIFORM,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(sobol.theta, different.theta)
    assert not torch.equal(sobol.theta, iid.theta)
    assert iid.classification == "diagnostic-only"
    with pytest.raises(ValueError, match="unknown"):
        generate_initial_design(
            parameter_spec=spec(2),
            point_count=8,
            seed_deriver=deriver(),
            seed_coordinate=SeedCoordinate("initial_design"),
            policy="latin_hypercube",
        )


def test_caller_design_preserves_order_and_is_strict() -> None:
    theta = torch.tensor([[1.0, 0.0], [-0.5, 1.5]], dtype=torch.float64)
    result = validate_caller_design(
        theta, parameter_spec=spec(2), dtype=torch.float64, device=torch.device("cpu")
    )
    assert torch.equal(result.theta, theta)
    assert result.policy_id == "caller_supplied"
    assert result.seed_value is None
    with pytest.raises(ValueError, match="dtype and device"):
        validate_caller_design(
            theta.float(),
            parameter_spec=spec(2),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="at least two"):
        validate_caller_design(
            theta[:1],
            parameter_spec=spec(2),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )


def test_initial_design_support_document_is_generated() -> None:
    path = (
        Path(__file__).parents[2]
        / "scientific_contracts"
        / "mutaco_d_contract"
        / "INITIAL_DESIGN_SUPPORT.md"
    )
    assert path.read_text(encoding="utf-8") == initial_design_support_markdown()
