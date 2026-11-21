from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from gpytorch.constraints import Positive

import mutaco.surrogate.exact_multitask as gp
from mutaco import (
    GPFitConfig,
    GPFitDiagnostic,
    GPFitFailureKind,
    GPFitRecoveryPolicy,
    GPFitResult,
    GPFitStatus,
    GPWarmStartState,
    GP_FIT_RECOVERY_POLICIES,
    GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER,
    GRAPH_M20_TASK_VARIANCE_FLOOR,
    GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE,
    ParameterAxis,
    ParameterSpec,
    PolicyClassification,
    SeedCoordinate,
    SurrogateTrainingData,
    TaskVarianceFloorPolicy,
    fit_exact_multitask_gp,
    fit_exact_multitask_gp_with_recovery,
    gp_fit_recovery_support_markdown,
)


def _data(count: int = 6) -> SurrogateTrainingData:
    theta = torch.linspace(0.05, 0.95, count, dtype=torch.float32).unsqueeze(-1)
    theta = torch.cat((theta, 1.0 - theta), dim=1).contiguous()
    response = torch.stack(
        (torch.sin(theta[:, 0]), torch.cos(theta[:, 1])), dim=-1
    ).contiguous()
    return SurrogateTrainingData(
        theta=theta,
        response_standardized=response,
        parameter_spec=ParameterSpec(
            (ParameterAxis("a", 0.0, 1.0), ParameterAxis("b", 0.0, 1.0))
        ),
        observation_ids=tuple(f"observation-{index}" for index in range(count)),
        transform_id="fixed-transform",
    )


def _config(**changes: object) -> GPFitConfig:
    base = GPFitConfig(
        fit_seed=177,
        fit_seed_coordinate=SeedCoordinate("gp_fit", ("training_version", 3)),
        max_iterations=2,
        jitter=1e-6,
        symmetry_tolerance=2e-6,
        psd_tolerance=1e-6,
        covariance_validation_policy="structural_symmetric_materialization_v1",
        task_variance_floor_policy=TaskVarianceFloorPolicy.STANDARDIZED_ABSOLUTE_V1,
        task_variance_floor=GRAPH_M20_TASK_VARIANCE_FLOOR,
        task_variance_floor_reference_scale=(
            GRAPH_M20_TASK_VARIANCE_REFERENCE_SCALE
        ),
        task_variance_floor_epsilon_multiplier=(
            GRAPH_M20_TASK_VARIANCE_EPSILON_MULTIPLIER
        ),
        recovery_policy=GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1,
        max_warm_start_states=2,
    )
    return replace(base, **changes)


def _state(
    data: SurrogateTrainingData,
    config: GPFitConfig,
    name: str,
    *,
    consecutive: int = 0,
) -> GPWarmStartState:
    model = gp._build_model(data, config)
    return GPWarmStartState(
        source_fit_id=f"fit-{name}",
        source_model_id=f"model-{name}",
        compatibility_id=gp._warm_start_compatibility_id(data, config),
        parameter_state={key: value.clone() for key, value in model.state_dict().items()},
        consecutive_fixed_conditioning_updates=consecutive,
        source_attempt_kind=(
            "condition_previous_gp_on_current_data_v1"
            if consecutive
            else "optimized"
        ),
    )


def _fake_result(
    *,
    kind: str,
    success: bool,
    marginal_likelihood: float | None = None,
    surrogate: object | None = None,
) -> GPFitResult:
    fit_id = f"fit-{kind}-{marginal_likelihood}"
    attempt = {
        "fit_executed": True,
        "initialization": {"kind": kind},
        "status": "success" if success else "failed",
        "fit_id": fit_id,
        "model_id": f"model-{kind}" if success else None,
        "marginal_likelihood": marginal_likelihood,
    }
    diagnostic = GPFitDiagnostic(
        status=GPFitStatus.SUCCESS if success else GPFitStatus.FAILED,
        fit_id=fit_id,
        training_data_id="training",
        model_id=f"model-{kind}" if success else None,
        elapsed_seconds=0.01,
        warnings=(),
        failure_type=None if success else "NotPSDError",
        failure_message=None if success else "not positive definite",
        validation_posterior_calls=1 if success else 0,
        posterior_validation_diagnostics={} if success else None,
        optimizer_result={"final_objective": -marginal_likelihood}
        if marginal_likelihood is not None
        else None,
        failure_kind=None if success else GPFitFailureKind.COVARIANCE_REJECTED,
        failure_stage=None if success else "optimizer",
        exception_chain=(),
        configuration={},
        execution={},
        recovery_attempts=(attempt,),
    )
    return GPFitResult(
        surrogate=(surrogate if success else None),  # type: ignore[arg-type]
        diagnostic=diagnostic,
    )


def test_recovery_registry_is_typed_and_graph_policy_requires_exact_floor() -> None:
    assert GP_FIT_RECOVERY_POLICIES == {
        GPFitRecoveryPolicy.COLD_ONLY: PolicyClassification.DEFAULT,
        GPFitRecoveryPolicy.GRAPH_M20_SEQUENTIAL_RECOVERY_V1: (
            PolicyClassification.APPROVED_OPTIONAL
        ),
    }
    assert GPFitConfig(1, SeedCoordinate("gp_fit")).recovery_policy is (
        GPFitRecoveryPolicy.COLD_ONLY
    )
    with pytest.raises(ValueError, match="unknown GP fit recovery"):
        GPFitConfig(1, SeedCoordinate("gp_fit"), recovery_policy="unknown")
    with pytest.raises(ValueError, match="registered fixed task-variance floor"):
        _config(task_variance_floor=GRAPH_M20_TASK_VARIANCE_FLOOR * 2)
    support = gp_fit_recovery_support_markdown()
    assert "`cold_only` | D default" in support
    assert "`graph_m20_sequential_recovery_v1` | approved optional" in support
    assert support.count("| 0 |") == 2


def test_floor_is_optimizer_visible_and_preserves_positive_interior() -> None:
    model = gp._build_model(_data(), _config())
    constraint = model.covar_module.kernels[1].raw_var_constraint
    assert float(constraint.lower_bound) == GRAPH_M20_TASK_VARIANCE_FLOOR
    raw = torch.tensor([0.5], dtype=torch.float32)
    assert torch.equal(constraint.transform(raw), Positive().transform(raw))
    assert float(constraint.transform(torch.tensor([-100.0]))) == pytest.approx(
        GRAPH_M20_TASK_VARIANCE_FLOOR
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_floor_transform_follows_cuda_training_data() -> None:
    cpu = _data()
    data = replace(
        cpu,
        theta=cpu.theta.to("cuda"),
        response_standardized=cpu.response_standardized.to("cuda"),
    )
    model = gp._build_model(data, _config())
    constraint = model.covar_module.kernels[1].raw_var_constraint
    transformed = constraint.transform(
        torch.full((2,), -100.0, device="cuda", dtype=torch.float32)
    )
    assert transformed.device.type == "cuda"
    assert torch.all(
        transformed
        == torch.tensor(
            GRAPH_M20_TASK_VARIANCE_FLOOR,
            device="cuda",
            dtype=torch.float32,
        )
    )


def test_initial_failure_runs_cold_and_all_four_restarts_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None]] = []

    def fail(*args: object, **kwargs: object) -> GPFitResult:
        kind = str(kwargs.get("initialization_kind"))
        calls.append((kind, kwargs.get("attempt_seed")))  # type: ignore[arg-type]
        return _fake_result(kind=kind, success=False)

    monkeypatch.setattr(gp, "fit_exact_multitask_gp", fail)
    torch.manual_seed(1)
    first = fit_exact_multitask_gp_with_recovery(
        _data(), config=_config(), training_version=0
    )
    first_calls = tuple(calls)
    calls.clear()
    torch.manual_seed(99999)
    second = fit_exact_multitask_gp_with_recovery(
        _data(), config=_config(), training_version=0
    )

    assert first.diagnostic.failure_kind is GPFitFailureKind.INITIAL_GP_FIT_FAILURE
    assert first.surrogate is None
    assert first_calls == tuple(calls)
    assert [kind for kind, _ in calls] == [
        "deterministic_cold",
        *(["coordinate_seeded_restart"] * 4),
    ]
    seeds = [seed for _, seed in calls[1:]]
    assert len(set(seeds)) == 4
    assert second.diagnostic.warnings[-1] == (
        "campaign_quality_warning_required_if_initial_failures_recur"
    )


def test_sequential_order_runs_all_restarts_and_selects_best_with_earliest_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    restart_scores = iter((1.0, 3.0, 3.0, 2.0))

    def controlled(*args: object, **kwargs: object) -> GPFitResult:
        kind = str(kwargs.get("initialization_kind"))
        calls.append(kind)
        if kind == "coordinate_seeded_restart":
            score = next(restart_scores)
            return _fake_result(
                kind=kind,
                success=True,
                marginal_likelihood=score,
                surrogate=SimpleNamespace(label=f"restart-{len(calls) - 4}"),
            )
        return _fake_result(kind=kind, success=False)

    monkeypatch.setattr(gp, "fit_exact_multitask_gp", controlled)
    data = _data()
    config = _config()
    result = fit_exact_multitask_gp_with_recovery(
        data,
        config=config,
        warm_start_states=(_state(data, config, "recent"), _state(data, config, "old")),
        training_version=3,
    )
    assert calls == [
        "most_recent_warm_optimized",
        "penultimate_warm_optimized",
        "deterministic_cold",
        *(["coordinate_seeded_restart"] * 4),
    ]
    assert result.surrogate.label == "restart-1"  # type: ignore[union-attr]
    restart_documents = [
        item
        for item in result.diagnostic.recovery_attempts
        if item.get("recovery_stage") == "coordinate_seeded_restart"
    ]
    assert len(restart_documents) == 4
    assert [item["selected"] for item in restart_documents] == [False, True, False, False]


def test_fixed_conditioning_uses_all_current_data_and_marks_consecutive_state() -> None:
    previous = _data(6)
    current = _data(7)
    config = _config()
    state = _state(previous, config, "previous")
    result = fit_exact_multitask_gp(
        current,
        config=config,
        warm_start_state=state,
        initialization_kind="condition_previous_gp_on_current_data_v1",
        optimize=False,
    )
    assert result.diagnostic.status is GPFitStatus.SUCCESS
    assert result.surrogate is not None
    assert any("InputDataWarning" in item for item in result.diagnostic.warnings)
    attempt = result.diagnostic.recovery_attempts[0]
    assert attempt["training_observation_count"] == 7
    assert attempt["training_covariance_dimension"] == 14
    conditioned_state = result.surrogate.warm_start_state()
    assert conditioned_state.consecutive_fixed_conditioning_updates == 1
    assert conditioned_state.source_attempt_kind == (
        "condition_previous_gp_on_current_data_v1"
    )


def test_second_consecutive_fixed_conditioning_is_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail(*args: object, **kwargs: object) -> GPFitResult:
        kind = str(kwargs.get("initialization_kind"))
        calls.append(kind)
        return _fake_result(kind=kind, success=False)

    monkeypatch.setattr(gp, "fit_exact_multitask_gp", fail)
    data = _data()
    config = _config()
    result = fit_exact_multitask_gp_with_recovery(
        data,
        config=config,
        warm_start_states=(
            _state(data, config, "conditioned", consecutive=1),
            _state(data, config, "older"),
        ),
        training_version=4,
    )
    assert result.surrogate is None
    assert result.diagnostic.failure_kind is (
        GPFitFailureKind.CONSECUTIVE_FIXED_CONDITIONING_PROHIBITED
    )
    assert len(calls) == 7
    assert result.diagnostic.recovery_attempts[-1]["reason"] == (
        "second_consecutive_fixed_conditioning_update"
    )


def test_fixed_conditioning_is_final_fallback_after_all_optimized_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def controlled(*args: object, **kwargs: object) -> GPFitResult:
        kind = str(kwargs.get("initialization_kind"))
        optimize = bool(kwargs.get("optimize", True))
        calls.append((kind, optimize))
        return _fake_result(
            kind=kind,
            success=not optimize,
            marginal_likelihood=-2.0 if not optimize else None,
            surrogate=SimpleNamespace(label="conditioned") if not optimize else None,
        )

    monkeypatch.setattr(gp, "fit_exact_multitask_gp", controlled)
    data = _data()
    config = _config()
    result = fit_exact_multitask_gp_with_recovery(
        data,
        config=config,
        warm_start_states=(_state(data, config, "recent"), _state(data, config, "old")),
        training_version=5,
    )
    assert calls[-1] == ("condition_previous_gp_on_current_data_v1", False)
    assert len(calls) == 8
    assert result.surrogate.label == "conditioned"  # type: ignore[union-attr]
    assert result.diagnostic.recovery_attempts[-1]["selected"] is True


def test_incompatible_warm_state_is_rejected_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    config = _config()
    incompatible = replace(
        _state(data, config, "wrong"), compatibility_id="incompatible"
    )
    valid = _state(data, config, "valid")
    calls: list[str] = []

    def controlled(*args: object, **kwargs: object) -> GPFitResult:
        kind = str(kwargs.get("initialization_kind"))
        calls.append(kind)
        return _fake_result(
            kind=kind,
            success=True,
            marginal_likelihood=1.0,
            surrogate=SimpleNamespace(label="valid-warm"),
        )

    monkeypatch.setattr(gp, "fit_exact_multitask_gp", controlled)
    result = fit_exact_multitask_gp_with_recovery(
        data,
        config=config,
        warm_start_states=(incompatible, valid),
        training_version=2,
    )
    assert calls == ["penultimate_warm_optimized"]
    assert result.diagnostic.recovery_attempts[0]["status"] == (
        "rejected_incompatible"
    )
    assert result.surrogate.label == "valid-warm"  # type: ignore[union-attr]
