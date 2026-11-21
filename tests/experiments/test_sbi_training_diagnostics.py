from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.benchmarking.methods.direct_posterior_diagnostics import (
    SBI_LEGACY_LOG_PROB_SUMMARY_SCHEMA,
    SBI_LOSS_SUMMARY_SCHEMA,
    TrainingDiagnosticsError,
    audit_training_artifacts,
    safe_audit_training_artifacts,
    safe_training_diagnostics,
    tensorboard_training_diagnostics,
    training_diagnostics,
    validate_live_tensorboard_parity,
)


class _Inference:
    def __init__(self, summary):
        self.summary = summary


def _estimator() -> torch.nn.Module:
    return torch.nn.Linear(2, 1)


def _loss_summary() -> dict:
    return {
        "training_loss": [3.0, 2.0, 1.0, 5.0, 4.0],
        "validation_loss": [4.0, 1.5, 2.0, 6.0, 3.0],
        "epoch_durations_sec": [0.1, 0.2, 0.3, 0.4, 0.5],
        "epochs_trained": [3, 2],
        "best_validation_loss": [1.5, 3.0],
    }


def test_sbi_0261_loss_summary_uses_current_round_and_minimizes() -> None:
    result = training_diagnostics(
        _Inference(_loss_summary()), _estimator(), max_num_epochs=200,
        stop_after_epochs=20, round_index=1, sbi_version="0.26.1",
    )
    assert result["diagnostic_schema"] == SBI_LOSS_SUMMARY_SCHEMA
    assert result["actual_epochs"] == 2
    assert result["training_loss"] == (5.0, 4.0)
    assert result["validation_loss"] == (6.0, 3.0)
    assert result["best_epoch"] == 1
    assert result["best_validation_loss"] == 3.0
    assert result["optimization_direction"] == "minimize"
    assert result["stopping_reason"] == "early_stopping"


def test_legacy_summary_is_explicit_and_maximizes() -> None:
    result = training_diagnostics(
        _Inference({
            "training_log_probs": [-3.0, -2.0],
            "validation_log_probs": [-4.0, -1.0],
        }),
        _estimator(), max_num_epochs=2, round_index=0,
    )
    assert result["diagnostic_schema"] == SBI_LEGACY_LOG_PROB_SUMMARY_SCHEMA
    assert result["best_epoch"] == 1
    assert result["best_validation_log_prob"] == -1.0
    assert result["optimization_direction"] == "maximize"


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda x: x.update(validation_loss=[1.0]), "lengths"),
        (lambda x: x.update(epoch_durations_sec=[0.1, -0.2, 0.3, 0.4, 0.5]), "nonnegative"),
        (lambda x: x.update(best_validation_loss=[9.0, 3.0]), "disagree"),
    ],
)
def test_loss_summary_rejects_inconsistent_evidence(change, match: str) -> None:
    summary = _loss_summary()
    change(summary)
    with pytest.raises(TrainingDiagnosticsError, match=match):
        training_diagnostics(
            _Inference(summary), _estimator(), max_num_epochs=200,
            stop_after_epochs=20, round_index=0,
        )


def test_unknown_schema_is_typed_but_safe_wrapper_is_observational() -> None:
    inference = _Inference({"new_unknown_key": [1.0]})
    with pytest.raises(TrainingDiagnosticsError, match="unknown SBI"):
        training_diagnostics(inference, _estimator(), max_num_epochs=200)
    result = safe_training_diagnostics(
        inference, _estimator(), max_num_epochs=200, round_index=0,
        sbi_version="future",
    )
    assert result["diagnostic_status"] == "training_diagnostics_unavailable"
    assert result["source_keys"] == ["new_unknown_key"]
    assert result["diagnostic_simulator_calls"] == 0
    assert result["diagnostic_training_calls"] == 0


def test_tensorboard_audit_and_live_parity(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path))
    for step, (train, valid, duration) in enumerate(
        zip((3.0, 2.0, 1.0), (4.0, 1.5, 2.0), (0.1, 0.2, 0.3))
    ):
        writer.add_scalar("training_loss", train, step)
        writer.add_scalar("validation_loss", valid, step)
        writer.add_scalar("epoch_durations_sec", duration, step)
    writer.add_scalar("epochs_trained", 3, 0)
    writer.add_scalar("best_validation_loss", 1.5, 0)
    writer.close()
    event = next(tmp_path.glob("events.out.tfevents*"))
    audit = tensorboard_training_diagnostics(
        event, max_num_epochs=200, stop_after_epochs=20
    )
    live = training_diagnostics(
        _Inference({
            "training_loss": [3.0, 2.0, 1.0],
            "validation_loss": [4.0, 1.5, 2.0],
            "epoch_durations_sec": [0.1, 0.2, 0.3],
            "epochs_trained": [3],
            "best_validation_loss": [1.5],
        }),
        _estimator(), max_num_epochs=200, stop_after_epochs=20,
    )
    parity = validate_live_tensorboard_parity(
        live, audit["rounds"][0], duration_tolerance=1e-5
    )
    assert parity["status"] == "exact_within_declared_tolerances"
    combined = audit_training_artifacts(
        tmp_path, [live], max_num_epochs=200, stop_after_epochs=20
    )
    assert combined["diagnostic_acceptance"] == "passed"
    assert combined["event_sha256"]
    assert combined["diagnostic_simulator_calls"] == 0
    assert combined["diagnostic_training_calls"] == 0


def test_training_artifact_audit_failure_is_observational(tmp_path: Path) -> None:
    result = safe_audit_training_artifacts(
        tmp_path, [], max_num_epochs=200, stop_after_epochs=20
    )
    assert result["diagnostic_status"] == "training_diagnostics_audit_failed"
    assert result["diagnostic_acceptance"] == "failed"
    assert result["diagnostic_simulator_calls"] == 0
    assert result["diagnostic_training_calls"] == 0
