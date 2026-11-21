from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mutaco import RunnerStatus
from experiments.paper.exploration import plot_paper120_b40_convergence as utility


def test_scientific_replay_is_disabled_by_default() -> None:
    with pytest.raises(RuntimeError, match="scientific replay is disabled"):
        utility.require_scientific_replay_authorization()


def test_explicit_execution_gate_requires_exact_confirmation() -> None:
    with pytest.raises(RuntimeError, match="scientific replay is disabled"):
        utility.require_scientific_replay_authorization(True, "yes")
    utility.require_scientific_replay_authorization(
        True,
        utility.REQUIRED_CONFIRMATION,
    )


def test_loads_all_canonical_records_without_calls() -> None:
    records = utility.load_canonical_records()
    assert set(records) == set(range(120))
    assert records[98]["target_position"] == 98


def test_convergence_statistics_use_only_bo_iterations() -> None:
    first_bo = [float(24 - index) for index in range(24)]
    second_bo = [value + 2.0 for value in first_bo]
    trajectories, mean, std = utility.convergence_statistics(
        [
            {"ordered_target_error_l1": [0.01] * 16 + first_bo},
            {"ordered_target_error_l1": [0.02] * 16 + second_bo},
        ]
    )
    assert trajectories.shape == (2, 24)
    assert torch.equal(mean, torch.tensor(first_bo, dtype=torch.float64) + 1.0)
    assert torch.allclose(std, torch.full((24,), 2.0**0.5, dtype=torch.float64))


def test_plot_writes_without_scientific_calls(tmp_path: Path, monkeypatch) -> None:
    records = [
        {
            "target_index": 0,
            "ordered_target_error_l1": [10.0] * 16
            + [1.0 / (index + 1) for index in range(24)],
        }
    ]
    monkeypatch.setattr(utility, "FORMATS", ("png",))
    figure = utility.plot_convergence(records, tmp_path)
    assert (tmp_path / "paper120_b40_convergence.png").is_file()
    axis = figure.axes[0]
    assert axis.get_xlabel() == r"BO iteration $t$"
    assert tuple(axis.get_xlim()) == (1.0, 24.0)
    assert len(axis.lines) == 1
    utility.plt.close(figure)


def test_writer_rejects_noncanonical_replay(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replay recommendation differs"):
        utility.write_diagnostic(
            {
                "target_index": 98,
                "canonical_recommendation_within_tolerance": False,
            },
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


class _FakeTransform:
    dtype = torch.float32
    device = torch.device("cpu")
    metadata = SimpleNamespace(response_names=tuple(f"y{index}" for index in range(20)))

    @staticmethod
    def covariance_to_physical(
        covariance: torch.Tensor,
        *,
        response_names: tuple[str, ...],
    ) -> torch.Tensor:
        assert len(response_names) == 20
        scales = torch.arange(1, 21, dtype=covariance.dtype)
        return covariance * scales.reshape(1, 20, 1) * scales.reshape(1, 1, 20)


class _FakeSurrogate:
    @staticmethod
    def posterior(theta: torch.Tensor) -> SimpleNamespace:
        assert theta.shape == (1, 4)
        return SimpleNamespace(
            mean=torch.zeros((1, 20), dtype=torch.float32),
            covariance=torch.eye(20, dtype=torch.float32).unsqueeze(0),
        )


def test_diagnostic_document_extracts_physical_gp_standard_deviations() -> None:
    observations = []
    for index in range(40):
        observations.append(
            SimpleNamespace(
                response_physical=torch.full((20,), float(index), dtype=torch.float32),
                phase=SimpleNamespace(value="initial" if index < 16 else "sequential"),
                logical_index=index,
                observation_id=f"observation-{index}",
                theta=torch.full((4,), float(index), dtype=torch.float32),
            )
        )
    recommendation = SimpleNamespace(selected_theta=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    run_result = SimpleNamespace(
        status=RunnerStatus.COMPLETED,
        terminal_surrogate=_FakeSurrogate(),
        transform=_FakeTransform(),
        recommendation=recommendation,
        observations=tuple(observations),
        logical_accounting={"committed_observations": 40},
        physical_accounting={"simulator_successes": 40},
    )
    target = {
        "target_index": 98,
        "target_record_id": "target-98",
        "target_response": [0.0] * 20,
    }
    canonical = {
        "methods": {"mutaco": {"recommended_theta": [1.0, 2.0, 3.0, 4.0]}}
    }
    document = utility.diagnostic_document(
        target=target,
        canonical_record=canonical,
        run_result=run_result,
    )
    assert document["gp_std_physical"] == pytest.approx(list(range(1, 21)))
    assert document["gp_std_omega_gauge0"] == pytest.approx([1, 3, 5, 7, 9])
    assert document["gp_std_M_gauge0"] == pytest.approx([11, 13, 15, 17, 19])
    assert document["best_so_far_target_error_l1"] == [0.0] * 40
    assert document["canonical_recommendation_within_tolerance"] is True
