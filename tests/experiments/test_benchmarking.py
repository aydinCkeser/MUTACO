from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mutaco import (
    ParameterAxis,
    ParameterSpec,
    ResponseAxis,
    ResponseSpec,
    RuntimeRequest,
    SimulationContext,
    SimulationResult,
    SimulatorCapabilities,
    canonical_json_bytes,
    resolve_runtime,
)
from experiments.benchmarking import (
    export_tables,
    load_campaign,
    load_results,
    plot_results,
    run_campaign,
    summarize_results,
)
from experiments.benchmarking.contracts import TargetSelection, validate_method_selection
from experiments.benchmarking.methods.sbi_suite import _simulate_batch, sequential_allocation
from experiments.benchmarking.methods import run_mutaco_target
from experiments.benchmarking.simulators import get_simulator_adapter
from experiments.benchmarking.statistics import aggregate_rows
from experiments.analytic_target_matching.noisy_cohort._campaign import (
    run_target as legacy_run_target,
    target_bank as legacy_target_bank,
)


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SMOKE = ROOT / "experiments/analytic_target_matching/configs/noisy_cohort_smoke.json"
LEGACY_FULL = ROOT / "experiments/analytic_target_matching/configs/noisy_cohort_100.json"
SBI_SMOKE = ROOT / "experiments/analytic_target_matching/configs/benchmark_campaigns/prompt5b_sbi_smoke.json"
LINEAR_SMOKE = (
    ROOT / "experiments/analytic_target_matching/configs/benchmark_campaigns/prompt5b_linear_reference_smoke.json"
)


def test_target_selectors_and_method_registry_fail_closed() -> None:
    assert TargetSelection.resolve({"policy": "full_bank"}, 4).indices == (0, 1, 2, 3)
    assert TargetSelection.resolve({"policy": "prefix", "count": 2}, 4).indices == (0, 1)
    assert TargetSelection.resolve(
        {"policy": "explicit_indices", "indices": [3, 0]}, 4
    ).indices == (3, 0)
    with pytest.raises(ValueError, match="unique"):
        TargetSelection.resolve(
            {"policy": "explicit_indices", "indices": [1, 1]}, 4
        )
    with pytest.raises(ValueError, match="unknown"):
        TargetSelection.resolve({"policy": "observed_best"}, 4)
    assert validate_method_selection("mutaco", ()) == ()
    with pytest.raises(ValueError, match="unknown SBI"):
        validate_method_selection("mutaco", ("sbi_magic",))


class _SeedResponseSimulator:
    def __init__(self) -> None:
        self.capabilities = SimulatorCapabilities(
            ParameterSpec((ParameterAxis("theta", 0.0, 1.0),)),
            ResponseSpec((ResponseAxis("seed_response"),)),
            "seed-response-simulator",
            deterministic=False,
            supports_context=True,
        )

    def evaluate(
        self, theta: torch.Tensor, context: SimulationContext | None = None
    ) -> SimulationResult:
        assert context is not None
        return SimulationResult(
            torch.tensor(
                [float(context.evaluation_seed % 1_000_003)], dtype=theta.dtype
            )
        )


def test_sbi_per_logical_row_contexts_are_unique_and_chunk_invariant() -> None:
    theta = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    simulator = _SeedResponseSimulator()
    ledger: list[dict[str, object]] = []
    ambient_before = torch.random.get_rng_state().clone()
    whole = _simulate_batch(
        simulator,
        theta,
        run_id="scientific-v1-row-context-test",
        target_index=3,
        round_index=0,
        seed=91,
        device=torch.device("cpu"),
        dtype=torch.float32,
        context_policy="per_logical_simulation_index_v1",
        logical_start_index=0,
        context_ledger=ledger,
    )
    first = _simulate_batch(
        simulator,
        theta[:2].contiguous(),
        run_id="scientific-v1-row-context-test",
        target_index=3,
        round_index=0,
        seed=91,
        device=torch.device("cpu"),
        dtype=torch.float32,
        context_policy="per_logical_simulation_index_v1",
        logical_start_index=0,
    )
    second = _simulate_batch(
        simulator,
        theta[2:].contiguous(),
        run_id="scientific-v1-row-context-test",
        target_index=3,
        round_index=0,
        seed=91,
        device=torch.device("cpu"),
        dtype=torch.float32,
        context_policy="per_logical_simulation_index_v1",
        logical_start_index=2,
    )
    assert torch.equal(whole, torch.cat((first, second)))
    assert torch.equal(ambient_before, torch.random.get_rng_state())
    assert len(ledger) == 4
    assert len({int(row["seed"]) for row in ledger}) == 4
    assert [row["logical_index"] for row in ledger] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="unknown SBI batch context"):
        _simulate_batch(
            simulator,
            theta[:1].contiguous(),
            run_id="scientific-v1-row-context-test",
            target_index=3,
            round_index=0,
            seed=91,
            device=torch.device("cpu"),
            dtype=torch.float32,
            context_policy="guess",
        )


def test_sbi_precision_and_scientific_posterior_count_fail_closed(
    tmp_path: Path,
) -> None:
    document = json.loads(SBI_SMOKE.read_text(encoding="utf-8"))
    document["dtype"] = "float64"
    path = tmp_path / "float64.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="requires float32"):
        load_campaign(path)
    document["dtype"] = "float32"
    document["campaign_classification"] = "scientific"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="10,000"):
        load_campaign(path)
    document["campaign_classification"] = "mechanical_smoke"
    document["process_policy"] = "in_process"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="cold_start_per_target requires"):
        load_campaign(path)


def test_exact_sequential_remainder_allocation() -> None:
    assert sequential_allocation(14, 3) == (5, 5, 4)
    assert sum(sequential_allocation(14, 3)) == 14
    with pytest.raises(ValueError):
        sequential_allocation(2, 3)


def test_reference_only_load_does_not_import_or_require_sbi() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from experiments.benchmarking import load_campaign; "
                f"c=load_campaign(r'{LEGACY_SMOKE}'); "
                "assert c.sbi_methods == (); "
                "assert 'sbi' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_readable_benchmark_entry_points_resolve_outside_repository(
    tmp_path: Path,
) -> None:
    directory = ROOT / "experiments/analytic_target_matching/sbi_benchmark"
    for filename in ("run.py", "summarize.py", "plot.py"):
        completed = subprocess.run(
            [sys.executable, str(directory / filename), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_second_adapter_has_independent_dimensions() -> None:
    adapter = get_simulator_adapter("linear_diagnostic")
    simulator = adapter.build_simulator(
        {}, device=torch.device("cpu"), dtype=torch.float64
    )
    assert simulator.capabilities.parameter_spec.dimension == 3
    assert simulator.capabilities.response_spec.dimension == 2


def test_second_adapter_reference_smoke_completes(tmp_path: Path) -> None:
    execution = run_campaign(
        load_campaign(LINEAR_SMOKE, device="cpu"), output=tmp_path / "linear"
    )
    assert len(execution.rows) == 2
    assert all(row["status"] == "completed" for row in execution.rows)
    assert all(len(row["theta_star"]) == 3 for row in execution.rows)
    assert all(len(row["target_response"]) == 2 for row in execution.rows)


def test_generic_handwritten_method_exactly_matches_prompt5a_projection() -> None:
    document = json.loads(LEGACY_SMOKE.read_text(encoding="utf-8"))
    runtime = resolve_runtime(RuntimeRequest(device="cpu", dtype=document["dtype"]))
    legacy_bank, _, science = legacy_target_bank(
        document, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    adapter = get_simulator_adapter("handwritten_analytic")
    generic_bank = adapter.target_bank(
        document, device=runtime.torch_device, dtype=runtime.torch_dtype
    )
    assert torch.equal(legacy_bank, generic_bank.theta)
    legacy = legacy_run_target(
        index=0,
        theta_star=legacy_bank[0],
        config=document,
        campaign_science=science,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
    )
    generic = run_mutaco_target(
        adapter=adapter,
        index=0,
        theta_star=generic_bank.theta[0],
        config=document,
        campaign_science=generic_bank.science,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
    )
    assert canonical_json_bytes(legacy["scientific_projection"]) == canonical_json_bytes(
        generic["scientific_projection"]
    )
    assert legacy["assessment_response"] == generic["assessment_response"]
    assert legacy["recommended_theta"] == generic["recommended_theta"]


def test_public_result_reload_select_resummarize_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = load_campaign(LEGACY_SMOKE, device="cpu")
    document = deepcopy(campaign.document)
    document["campaign"]["target_count"] = 1
    document["runner"]["initial_points"] = 4
    document["runner"]["sequential_rounds"] = 1
    document["runner"]["gp"]["max_iterations"] = 2
    document["runner"]["acquisition"].update(
        {"raw_samples": 8, "restarts": 1, "max_iterations": 1}
    )
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    execution = run_campaign(
        load_campaign(config_path, device="cpu"), output=tmp_path / "output"
    )
    assert len(execution.rows) == 1
    from mutaco_analytic import HandwrittenAnalyticSimulator

    def forbidden_evaluate(*args: object, **kwargs: object) -> object:
        raise AssertionError("post-processing attempted a simulator call")

    monkeypatch.setattr(HandwrittenAnalyticSimulator, "evaluate", forbidden_evaluate)
    results = load_results(execution.output)
    selected = results.select(methods=("mutaco",), statuses=("completed",))
    assert len(selected.rows) == 1
    summary = summarize_results(selected)
    assert summary["groups"][0]["completed_count"] == 1
    written = export_tables(selected, summary, output=execution.output / "tables")
    assert all(path.is_file() for path in written)
    plots = plot_results(
        selected,
        summary,
        output=execution.output / "plots",
        style=ROOT / "experiments/analytic_target_matching/noisy_cohort/styles/noisy_cohort_default.mplstyle",
    )
    assert len(plots["files"]) == 4
    assert (execution.output / "REPORT.md").is_file()
    assert (execution.output / "FIELDS.md").is_file()
    assert "\n  " in (execution.output / "summary.json").read_text(encoding="utf-8")


def test_recurrent_initial_gp_failures_invalidate_aggregate_claims() -> None:
    summary = aggregate_rows(
        (
            {
                "method": "mutaco",
                "status": "failed",
                "runner_provenance": {"status": "initial_gp_fit_failure"},
            },
            {
                "method": "mutaco",
                "status": "failed",
                "failure_explanation": "initial_gp_fit_failure",
            },
            {"method": "sbi_npe", "status": "completed"},
        )
    )
    assert summary["campaign_quality"] == {
        "initial_gp_fit_failure_count": 2,
        "recurrent_initial_gp_fit_failure": True,
        "aggregate_mutaco_claims_valid": False,
        "warning": (
            "recurrent_initial_gp_fit_failure_invalidates_aggregate_mutaco_claims"
        ),
    }
    mutaco = next(
        group for group in summary["groups"] if group["group"]["method"] == "mutaco"
    )
    assert mutaco["aggregate_claims_valid"] is False


@pytest.mark.skipif(importlib.util.find_spec("sbi") is None, reason="requires mutaco-sbi")
def test_all_five_sbi_methods_complete_declared_mechanical_smoke(tmp_path: Path) -> None:
    campaign = load_campaign(SBI_SMOKE, device="cpu")
    output = tmp_path / "sbi"
    execution = run_campaign(campaign, output=output)
    rows = {row["method"]: row for row in execution.rows}
    assert set(rows) == {
        "mutaco",
        "sbi_npe",
        "sbi_nle",
        "sbi_snle",
        "sbi_nre",
        "sbi_snre",
    }
    assert all(row["status"] == "completed" for row in rows.values())
    assert tuple(rows["sbi_snle"]["sequential_round_allocation"]) == (5, 5, 4)
    assert tuple(rows["sbi_snre"]["sequential_round_allocation"]) == (5, 5, 4)
    assert all(rows[name]["inference_calls"] == 14 for name in rows)
    assert all(
        rows[name]["cold_process_timing"]["exit_code"] == 0
        for name in rows
        if name != "mutaco"
    )
    restarted = run_campaign(campaign, output=output, restart=True)
    assert restarted.manifest["method_execution"] == {
        "sbi_executed_record_count": 0,
        "sbi_recovered_record_count": 5,
    }


@pytest.mark.skipif(importlib.util.find_spec("sbi") is None, reason="requires mutaco-sbi")
def test_amortized_shared_training_and_typed_deadline_failure(tmp_path: Path) -> None:
    document = json.loads(SBI_SMOKE.read_text(encoding="utf-8"))
    document["campaign"]["target_count"] = 2
    document["sbi_methods"] = ["sbi_npe"]
    document["views"] = ["amortized_bank"]
    config_path = tmp_path / "amortized.json"
    config_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    execution = run_campaign(
        load_campaign(config_path, device="cpu"), output=tmp_path / "amortized"
    )
    sbi_rows = [row for row in execution.rows if row["method"] == "sbi_npe"]
    assert len(sbi_rows) == 2
    assert len({row["shared_training_id"] for row in sbi_rows}) == 1
    assert sum(row["inference_calls"] for row in sbi_rows) == 28
    assert sum(bool(row["shared_cost_charged_here"]) for row in sbi_rows) == 1

    document["campaign"]["target_count"] = 1
    document["views"] = ["cold_start_per_target"]
    document["regimes"] = ["equal_wall_clock"]
    document["diagnostic_wall_clock_seconds"] = 0.0
    deadline_path = tmp_path / "deadline.json"
    deadline_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    deadline = run_campaign(
        load_campaign(deadline_path, device="cpu"), output=tmp_path / "deadline"
    )
    failed = [row for row in deadline.rows if row["method"] == "sbi_npe"]
    assert failed[0]["status"] == "failed"
    assert "BenchmarkDeadlineExceeded" in failed[0]["failure_explanation"]
    assert failed[0]["inference_calls"] == 0
