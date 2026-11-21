from __future__ import annotations

from pathlib import Path

import pytest

from experiments.paper.exploration import plot_paper120_target as utility


def test_loads_canonical_target_and_methods() -> None:
    target, methods = utility.load_target_and_methods(0)
    assert target["target_index"] == 0
    assert set(("mutaco", "sbi_npe")) <= methods.keys()


def test_reconstruction_uses_one_shared_context_without_real_calls() -> None:
    target, methods = utility.load_target_and_methods(0)
    target_response = tuple(target["target_response"])
    calls = []

    # Make stored errors agree with the substituted response while testing only
    # selection and pairing; the real simulator is never constructed here.
    methods = {name: dict(row, response_error_l1_sum=0.0) for name, row in methods.items()}

    def fake(method, theta, context):
        calls.append((method, tuple(theta), context))
        return target_response

    responses = utility.reconstruct(target, methods, evaluate=fake)
    assert set(responses) == {"target", "mutaco", "sbi_npe"}
    assert [call[0] for call in calls] == ["mutaco", "sbi_npe"]
    assert calls[0][2] == calls[1][2]
    assert calls[0][2]["coordinate"]["namespace"] == "paired_held_out_assessment"


def test_historical_thread_layout_comes_from_frozen_target(monkeypatch) -> None:
    target, _ = utility.load_target_and_methods(0)
    observed = []
    monkeypatch.setattr(
        utility.torch,
        "set_num_threads",
        lambda value: observed.append(("intra", value)),
    )
    monkeypatch.setattr(
        utility.torch,
        "set_num_interop_threads",
        lambda value: observed.append(("interop", value)),
    )
    utility._configure_historical_threads(target)
    assert observed == [("intra", 8), ("interop", 1)]
    assert utility.os.environ["OMP_NUM_THREADS"] == "8"
    assert utility.os.environ["MKL_NUM_THREADS"] == "8"
    assert utility.os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_reconstruction_accepts_small_float32_drift() -> None:
    target, methods = utility.load_target_and_methods(0)
    response = list(target["target_response"])
    response[0] += utility.REPLAY_L1_TOLERANCE / 2
    methods = {name: dict(row, response_error_l1_sum=0.0) for name, row in methods.items()}
    reconstructed = utility.reconstruct(
        target, methods, evaluate=lambda method, theta, context: response
    )
    assert reconstructed["mutaco"][0] == response[0]


def test_reconstruction_rejects_material_drift() -> None:
    target, methods = utility.load_target_and_methods(0)
    response = list(target["target_response"])
    response[0] += utility.REPLAY_L1_TOLERANCE * 2
    methods = {name: dict(row, response_error_l1_sum=0.0) for name, row in methods.items()}
    with pytest.raises(ValueError, match="replay differs"):
        utility.reconstruct(
            target, methods, evaluate=lambda method, theta, context: response
        )


def test_plotting_writes_requested_formats_without_real_calls(tmp_path: Path, monkeypatch) -> None:
    target, methods = utility.load_target_and_methods(0)
    responses = {name: tuple(target["target_response"]) for name in ("target", "mutaco", "sbi_npe")}
    monkeypatch.setattr(utility, "FORMATS", ("png",))
    utility.plot_target(target, 0, tmp_path)
    utility.plot_reconstruction(target, methods, responses, 0, tmp_path)
    assert (tmp_path / "forward.png").is_file()
    assert (tmp_path / "reconstruction.png").is_file()


@pytest.mark.parametrize("target_index", (-1, 120))
def test_rejects_out_of_range_target(target_index: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 119"):
        utility.load_target_and_methods(target_index)
