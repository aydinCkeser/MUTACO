"""Run the complete human-readable analytic target-matching suite.

The protocol intentionally reads from top to bottom: load two declared
configurations, resolve CPU/CUDA placement, prepare two hidden-target mock
experiments, run MUTACO plus a deterministic replay, assess the existing-point
recommendation, and write readable outputs.

This suite preserves the named Prompt 5 acceptance milestone while giving the
scientific experiment a durable, descriptive name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import mutaco
import mutaco_analytic
from mutaco import RuntimeRequest, resolve_runtime

if __package__:
    from ._support import (
        build_manifest,
        load_config,
        prepare_profiles,
        run_and_assess_profile,
        write_json,
    )
else:
    from _support import (
        build_manifest,
        load_config,
        prepare_profiles,
        run_and_assess_profile,
        write_json,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDWRITTEN_CONFIG = (
    REPOSITORY_ROOT
    / "experiments"
    / "analytic_target_matching"
    / "configs"
    / "handwritten.json"
)
DEFAULT_GP_LANDSCAPE_CONFIG = (
    REPOSITORY_ROOT
    / "experiments"
    / "analytic_target_matching"
    / "configs"
    / "gp_landscape.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs" / "analytic_target_matching" / "suite"


def execute(
    *,
    handwritten_config_path: Path,
    gp_landscape_config_path: Path,
    output: Path,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Run both declared mock-forward experiments and their exact replays."""

    # 1. Load the two declared scientific protocols and require shared settings.
    handwritten_config = load_config(handwritten_config_path)
    gp_landscape_config = load_config(gp_landscape_config_path)
    if set(handwritten_config["profiles"]) != {"handwritten"}:
        raise ValueError("suite handwritten configuration has the wrong profile")
    if set(gp_landscape_config["profiles"]) != {"gp_landscape"}:
        raise ValueError("suite GP-landscape configuration has the wrong profile")
    if handwritten_config["schema_version"] != gp_landscape_config["schema_version"]:
        raise ValueError("suite configurations must use the same schema")
    if handwritten_config["dtype"] != gp_landscape_config["dtype"]:
        raise ValueError("suite configurations must use the same dtype")
    if handwritten_config["runner"] != gp_landscape_config["runner"]:
        raise ValueError("suite configurations must use the same runner settings")
    config = {
        "schema_version": handwritten_config["schema_version"],
        "dtype": handwritten_config["dtype"],
        "runner": handwritten_config["runner"],
        "profiles": {
            **handwritten_config["profiles"],
            **gp_landscape_config["profiles"],
        },
    }
    return _execute_loaded(
        config=config,
        config_paths=(handwritten_config_path, gp_landscape_config_path),
        output=output,
        requested_device=requested_device,
    )


def execute_combined_config(
    *,
    config_path: Path,
    output: Path,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Run the legacy consolidated Prompt 5 configuration."""

    return _execute_loaded(
        config=load_config(config_path),
        config_paths=(config_path,),
        output=output,
        requested_device=requested_device,
    )


def _execute_loaded(
    *,
    config: dict[str, Any],
    config_paths: tuple[Path, ...],
    output: Path,
    requested_device: str,
) -> dict[str, Any]:
    runtime = resolve_runtime(
        RuntimeRequest(device=requested_device, dtype=str(config["dtype"]))
    )

    # 2. Construct the hidden targets and simulator-independent run intents.
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    profiles = prepare_profiles(
        config,
        device=runtime.torch_device,
        dtype=runtime.torch_dtype,
    )

    # 3. Run MUTACO, replay it, assess its evaluated-point recommendation,
    #    and write the per-profile histories and plots.
    results = tuple(
        run_and_assess_profile(
            profile,
            output=output,
            device=runtime.torch_device,
        )
        for profile in profiles
    )

    # 4. Record execution provenance separately from scientific replay state.
    manifest = build_manifest(
        config=config,
        config_paths=config_paths,
        runtime=runtime,
        results=results,
        total_runtime_seconds=perf_counter() - started,
        package_origins={
            "mutaco": str(Path(mutaco.__file__).resolve()),
            "mutaco_analytic": str(Path(mutaco_analytic.__file__).resolve()),
        },
        cuda_command=(
            "python -m experiments.analytic_target_matching.run_suite "
            "--output <output> --device cuda"
        ),
    )
    write_json(output / "manifest.json", manifest)
    print(
        "[analytic target matching] completed both forwards on "
        f"{runtime.effective_device} in {manifest['total_runtime_seconds']:.3f}s; "
        f"outputs: {output}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handwritten-config",
        type=Path,
        default=DEFAULT_HANDWRITTEN_CONFIG,
    )
    parser.add_argument(
        "--gp-landscape-config",
        type=Path,
        default=DEFAULT_GP_LANDSCAPE_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    arguments = parser.parse_args()
    execute(
        handwritten_config_path=arguments.handwritten_config,
        gp_landscape_config_path=arguments.gp_landscape_config,
        output=arguments.output,
        requested_device=arguments.device,
    )


if __name__ == "__main__":
    main()
