"""Run the configurable noisy handwritten MUTACO target cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from time import perf_counter

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mutaco
import mutaco_analytic
import psutil
from mutaco import stable_hash

from experiments.benchmarking import load_campaign, run_campaign

from experiments.analytic_target_matching.noisy_cohort._artifacts import (
    file_hash,
    read_json,
    write_csv,
    write_json,
)
from experiments.analytic_target_matching.noisy_cohort._metrics import aggregate
from experiments.analytic_target_matching.noisy_cohort._plotting import generate_plots


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/analytic_target_matching/configs/noisy_cohort_100.json"
DEFAULT_OUTPUT = ROOT / "runs/analytic_target_matching/noisy_cohort_100"
DEFAULT_STYLE = ROOT / "experiments/analytic_target_matching/noisy_cohort/styles/noisy_cohort_default.mplstyle"


def _git_state() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "entries": status}


def execute(
    *, config_path: Path, output: Path, device_request: str = "auto",
    restart: bool = False, verbose_events: bool = False,
) -> dict:
    """Run Prompt 5A through the public generic reference-only interface."""

    campaign = load_campaign(config_path, device=device_request)
    config = campaign.document
    started = perf_counter()
    execution = run_campaign(
        campaign,
        output=output,
        restart=restart,
        verbose_events=verbose_events,
    )
    rows = list(execution.rows)
    generic_manifest = execution.manifest
    summary = aggregate(rows, len(rows))
    write_json(output / "metrics.json", {"rows": rows})
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", summary)
    replay = read_json(output / "replay_evidence.json")
    plots = generate_plots(
        summary=summary, rows=rows, output=output / "plots",
        style=DEFAULT_STYLE,
    )
    write_json(output / "plots" / "plot_manifest.json", plots)
    artifacts = {
        path.relative_to(output).as_posix(): {
            "sha256": file_hash(path), "bytes": path.stat().st_size
        }
        for path in output.rglob("*")
        if (
            path.is_file()
            and not path.name.endswith(".tmp")
            and path.name
            not in {"console.stdout.log", "console.stderr.log", "manifest.json"}
        )
    }
    manifest = {
        "schema_version": "mutaco-noisy-cohort-manifest-v1",
        "campaign_id": generic_manifest["campaign_id"],
        "configuration": config,
        "configuration_path": config_path.as_posix(),
        "configuration_hash": stable_hash("cohort-config-v1", config),
        "runtime": generic_manifest["runtime"],
        "package_origins": {
            "mutaco": str(Path(mutaco.__file__).resolve()),
            "mutaco_analytic": str(Path(mutaco_analytic.__file__).resolve()),
        },
        "source": _git_state(),
        "summary": summary,
        "accounting": {
            "target_generation_calls": sum(row.get("target_construction_calls", 0) for row in rows),
            "inference_calls": sum(row.get("inference_calls", 0) for row in rows),
            "assessment_calls": sum(row.get("assessment_calls", 0) for row in rows),
            "gp_fits": sum(row.get("gp_fits", 0) for row in rows),
            "replay_indices": tuple(replay["indices"]),
        },
        "total_runtime_seconds": perf_counter() - started,
        "execution_mode": "restart" if restart else "fresh",
        "cohort_execution": generic_manifest["cohort_execution"],
        "memory_summary": {
            "process_rss_bytes_at_manifest": psutil.Process().memory_info().rss,
            "system_available_bytes_at_manifest": psutil.virtual_memory().available,
        },
        "plotting": plots,
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    print(f"[cohort] completed {summary['completed_count']}/{len(rows)} targets; output={output}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--verbose-events", action="store_true")
    args = parser.parse_args()
    execute(
        config_path=args.config, output=args.output, device_request=args.device,
        restart=args.restart, verbose_events=args.verbose_events,
    )


if __name__ == "__main__":
    main()
