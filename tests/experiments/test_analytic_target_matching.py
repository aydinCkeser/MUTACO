from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from experiments.analytic.prompt5_acceptance import execute as legacy_execute
from experiments.analytic.prompt5_support import SCHEMA as LEGACY_SCHEMA
from experiments.analytic_target_matching._support import (
    SCHEMA,
    load_config,
    write_json,
)
from experiments.analytic_target_matching.run_gp_landscape import (
    execute as execute_gp_landscape,
)
from experiments.analytic_target_matching.run_handwritten import (
    execute as execute_handwritten,
)
from experiments.analytic_target_matching.run_suite import execute


CONFIG_ROOT = Path("experiments/analytic_target_matching/configs")
HANDWRITTEN_CONFIG = CONFIG_ROOT / "handwritten.json"
GP_LANDSCAPE_CONFIG = CONFIG_ROOT / "gp_landscape.json"
LEGACY_CONFIG = Path("experiments/analytic_target_matching/configs/prompt5_vertical_slice.json")
HISTORICAL_CHECKOUT_NAME = "mutaco-historical"


def _make_quick(config: dict[str, object]) -> dict[str, object]:
    quick = deepcopy(config)
    runner = quick["runner"]
    assert isinstance(runner, dict)
    gp = runner["gp"]
    acquisition = runner["acquisition"]
    assert isinstance(gp, dict)
    assert isinstance(acquisition, dict)
    gp["max_iterations"] = 2
    acquisition.update({"raw_samples": 8, "restarts": 1, "max_iterations": 1})
    profiles = quick["profiles"]
    assert isinstance(profiles, dict)
    for profile in profiles.values():
        assert isinstance(profile, dict)
        profile["initial_points"] = 4
        profile["sequential_rounds"] = 1
    if "gp_landscape" in profiles:
        profiles["gp_landscape"]["feature_count"] = 8
    return quick


def test_descriptive_configurations_preserve_prompt5_science() -> None:
    handwritten = load_config(HANDWRITTEN_CONFIG)
    gp_landscape = load_config(GP_LANDSCAPE_CONFIG)
    assert handwritten["schema_version"] == SCHEMA
    assert gp_landscape["schema_version"] == SCHEMA
    assert handwritten["dtype"] == gp_landscape["dtype"] == "float64"
    assert handwritten["runner"] == gp_landscape["runner"]
    assert set(handwritten["profiles"]) == {"handwritten"}
    assert set(gp_landscape["profiles"]) == {"gp_landscape"}
    assert handwritten["profiles"]["handwritten"]["initial_points"] == 8
    assert handwritten["profiles"]["handwritten"]["sequential_rounds"] == 6
    assert gp_landscape["profiles"]["gp_landscape"]["parameter_dim"] == 6
    assert gp_landscape["profiles"]["gp_landscape"]["response_dim"] == 4
    assert gp_landscape["profiles"]["gp_landscape"]["initial_points"] == 12
    assert gp_landscape["profiles"]["gp_landscape"]["sequential_rounds"] == 6
    assert handwritten["runner"]["acquisition"]["raw_samples"] == 64
    assert handwritten["runner"]["acquisition"]["restarts"] == 4
    assert set(load_config(LEGACY_CONFIG)["profiles"]) == {
        "handwritten",
        "gp_landscape",
    }
    assert LEGACY_SCHEMA == "mutaco-prompt5-acceptance-v1"
    assert callable(legacy_execute)


def test_analytic_suite_executes_both_forwards_and_writes_replay_evidence(
    tmp_path: Path,
) -> None:
    handwritten = _make_quick(load_config(HANDWRITTEN_CONFIG))
    gp_landscape = _make_quick(load_config(GP_LANDSCAPE_CONFIG))
    handwritten_path = tmp_path / "handwritten.json"
    gp_landscape_path = tmp_path / "gp_landscape.json"
    output = tmp_path / "output"
    write_json(handwritten_path, handwritten)
    write_json(gp_landscape_path, gp_landscape)

    manifest = execute(
        handwritten_config_path=handwritten_path,
        gp_landscape_config_path=gp_landscape_path,
        output=output,
        requested_device="auto",
    )

    assert manifest["milestone"] == "Prompt 5"
    assert manifest["experiment"] == "analytic_target_matching"
    assert {item["name"] for item in manifest["profiles"]} == {
        "handwritten",
        "gp_landscape",
    }
    assert all(item["status"] == "completed" for item in manifest["profiles"])
    assert all(item["replay_equal"] for item in manifest["profiles"])
    assert manifest["same_cuda_capable_configuration"][
        "scientific_configuration_unchanged"
    ]
    assert HISTORICAL_CHECKOUT_NAME not in manifest["package_origins"]["mutaco"]
    assert (
        HISTORICAL_CHECKOUT_NAME
        not in manifest["package_origins"]["mutaco_analytic"]
    )
    for profile in ("handwritten", "gp_landscape"):
        result_path = output / profile / "result.json"
        assert result_path.is_file()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["response_error"]["scope"] == "in_sample_evaluated_observation"
        assert result["response_error"]["held_out_simulator_calls"] == 0
        assert result["target_construction"]["simulator_calls"] == 1
        assert (
            result["target_construction"]["budget_domain"]
            == "target_generation_outside_mutaco_inference"
        )
        assert (output / profile / "scientific_history.json").is_file()
        assert (output / profile / "replay_evidence.json").is_file()
        for filename in (
            "parameter_trajectory.svg",
            "error_trajectory.svg",
            "acquisition_branches.svg",
        ):
            ET.parse(output / profile / "plots" / filename)


def test_individual_entry_points_write_profile_specific_cuda_commands(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "handwritten",
            HANDWRITTEN_CONFIG,
            execute_handwritten,
            "experiments.analytic_target_matching.run_handwritten",
        ),
        (
            "gp_landscape",
            GP_LANDSCAPE_CONFIG,
            execute_gp_landscape,
            "experiments.analytic_target_matching.run_gp_landscape",
        ),
    )
    for name, source_path, entry_point, module_name in cases:
        config = _make_quick(load_config(source_path))
        config_path = tmp_path / f"{name}.json"
        output = tmp_path / name
        write_json(config_path, config)
        manifest = entry_point(
            config_path=config_path,
            output=output,
            requested_device="auto",
        )
        assert [item["name"] for item in manifest["profiles"]] == [name]
        assert module_name in manifest["same_cuda_capable_configuration"]["command"]
        assert (output / name / "result.json").is_file()


def test_legacy_prompt5_entry_point_remains_executable(tmp_path: Path) -> None:
    config = _make_quick(load_config(LEGACY_CONFIG))
    config_path = tmp_path / "prompt5.json"
    output = tmp_path / "legacy"
    write_json(config_path, config)
    manifest = legacy_execute(
        config_path=config_path,
        output=output,
        requested_device="auto",
    )
    assert manifest["milestone"] == "Prompt 5"
    assert {item["name"] for item in manifest["profiles"]} == {
        "handwritten",
        "gp_landscape",
    }


def test_documented_module_entrypoint_resolves_required_clean_api() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from mutaco import RunnerProgressEvent; "
                "from mutaco_analytic import HandwrittenAnalyticSimulator; "
                "import mutaco, mutaco_analytic; "
                "print(mutaco.__file__); print(mutaco_analytic.__file__)"
            ),
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert HISTORICAL_CHECKOUT_NAME not in completed.stdout


def test_readable_entry_points_support_direct_file_execution(
    tmp_path: Path,
) -> None:
    experiment_dir = Path("experiments/analytic_target_matching").resolve()
    for filename in ("run_handwritten.py", "run_gp_landscape.py", "run_suite.py"):
        completed = subprocess.run(
            [sys.executable, str(experiment_dir / filename), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--device {auto,cpu,cuda}" in completed.stdout
