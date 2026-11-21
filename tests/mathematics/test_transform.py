from __future__ import annotations

import pytest
import torch

from mutaco import FrozenResponseTransform, ResponseAxis, ResponseSpec


DTYPE = torch.float64
NAMES = ("frequency", "amplitude")
SPEC = ResponseSpec(tuple(ResponseAxis(name) for name in NAMES))


def fitted_transform() -> FrozenResponseTransform:
    responses = torch.tensor([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]], dtype=DTYPE)
    return FrozenResponseTransform.fit(
        responses,
        response_spec=SPEC,
        observation_ids=("initial-0", "initial-1", "initial-2"),
        scale_floor=0.25,
    )


def test_transform_uses_sample_std_floor_and_frozen_metadata() -> None:
    transform = fitted_transform()
    assert torch.equal(transform.mean, torch.tensor([3.0, 4.0], dtype=DTYPE))
    assert torch.equal(transform.empirical_scale, torch.tensor([2.0, 0.0], dtype=DTYPE))
    assert torch.equal(transform.scale, torch.tensor([2.0, 0.25], dtype=DTYPE))
    assert transform.metadata.degrees_of_freedom == 1
    assert transform.metadata.floor_activated == (False, True)
    before = transform.metadata.provenance_document()

    standardized = transform.to_standardized(
        torch.tensor([[7.0, 4.5]], dtype=DTYPE), response_names=NAMES
    )
    assert torch.equal(standardized, torch.tensor([[2.0, 2.0]], dtype=DTYPE))
    assert transform.metadata.provenance_document() == before


def test_response_and_covariance_round_trip() -> None:
    transform = fitted_transform()
    response = torch.tensor([7.0, 4.5], dtype=DTYPE)
    assert torch.equal(
        transform.to_physical(
            transform.to_standardized(response, response_names=NAMES),
            response_names=NAMES,
        ),
        response,
    )
    covariance = torch.tensor([[4.0, 0.5], [0.5, 0.25]], dtype=DTYPE)
    standardized = transform.covariance_to_standardized(covariance, response_names=NAMES)
    assert torch.equal(
        transform.covariance_to_physical(standardized, response_names=NAMES),
        covariance,
    )


def test_transform_is_strict_about_fit_and_component_identity() -> None:
    with pytest.raises(ValueError, match="at least two"):
        FrozenResponseTransform.fit(
            torch.ones((1, 2), dtype=DTYPE),
            response_spec=SPEC,
            observation_ids=("only",),
        )
    with pytest.raises(ValueError, match="unique"):
        FrozenResponseTransform.fit(
            torch.ones((2, 2), dtype=DTYPE),
            response_spec=SPEC,
            observation_ids=("same", "same"),
        )
    with pytest.raises(ValueError, match="names/order"):
        fitted_transform().to_standardized(
            torch.ones(2, dtype=DTYPE), response_names=tuple(reversed(NAMES))
        )


def test_transform_hash_is_semantic_while_device_is_execution_provenance() -> None:
    transform = fitted_transform()
    semantic = transform.metadata.semantic_document()
    provenance = transform.metadata.provenance_document()
    assert "fit_device" not in semantic
    assert provenance["fit_device"] == "cpu"
    assert provenance["transform_id"] == transform.metadata.transform_id


def test_identity_transform_omits_centering_and_scaling() -> None:
    responses = torch.tensor([[1.0, 4.0], [3.0, 9.0], [5.0, 16.0]], dtype=DTYPE)
    transform = FrozenResponseTransform.identity(
        responses,
        response_spec=SPEC,
        observation_ids=("initial-0", "initial-1", "initial-2"),
    )
    target = torch.tensor([7.0, 25.0], dtype=DTYPE)
    assert transform.metadata.transform_policy == "identity_physical_v1"
    assert transform.metadata.degrees_of_freedom == 0
    assert torch.equal(transform.to_standardized(responses, response_names=NAMES), responses)
    assert torch.equal(transform.standardize_target(target, response_names=NAMES).values, target)
    assert torch.equal(transform.to_physical(target, response_names=NAMES), target)
    covariance = torch.tensor([[4.0, 0.5], [0.5, 0.25]], dtype=DTYPE)
    assert torch.equal(
        transform.covariance_to_standardized(covariance, response_names=NAMES),
        covariance,
    )


def test_historical_fitted_transform_semantic_document_is_unchanged() -> None:
    transform = fitted_transform()
    assert transform.metadata.transform_policy == "initial_design_zscore_v1"
    assert "transform_policy" not in transform.metadata.semantic_document()
