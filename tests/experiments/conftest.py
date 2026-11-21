"""External, byte-verified inputs for offline paper-analysis checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest


def _original_trajectory_archive(directory: Path, budget: int) -> Path:
    archive = directory / f"scalar-primary-b{budget}-v3r2.tar.gz"
    if not archive.is_file():
        pytest.skip("requires the byte-exact trajectory packet from the private historical archive")
    return archive


@pytest.fixture(scope="session")
def offline_analysis_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Restore the missing expanded export from its committed archive only.

    Expected reports remain independent archived files. No producer is run,
    and no result/configuration or expanded input is written into the checkout.
    """
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path_factory.mktemp("offline-analysis-inputs")
    # Small fixed-input copies avoid requiring symlink privileges on Windows.
    for relative in (
        "experiments/graph_inversion/frozen_targets/paper120_v3",
        "results/paper/cross_experiment_analysis/b40_v1",
        "results/paper/graph_scalar_primary_b40_v1",
        "results/paper/graph_scalar_primary_b100_v1",
        "results/paper/graph_scalar_primary_b100_population_preparation_v3r2",
        "results/paper/scalar_primary_b40_b100_synthesis_v1",
        "experiments/paper/studies/graph_paper120/analysis",
    ):
        target = root / relative
        shutil.copytree(repository / relative, target)
    directory = repository / "results/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2"
    archive = _original_trajectory_archive(directory, 40)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "2f86646f7c21914552d10e9e1fcb209d29880cee194a3e79f8f8fae211b6659c"
    )
    destination = root / "experiments/paper/exploration/outputs/scalar_primary_smoke/population_b40_v3r2_compact_fetch/population/collection/compact_export.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        content = source.extractfile("collection/compact_export.json").read()
    expected = json.loads((repository / "results/paper/graph_scalar_primary_b40_v1/manifest.json").read_text())
    assert hashlib.sha256(content).hexdigest() == expected["integrity"]["compact_export_sha256"]
    destination.write_bytes(content)
    return root


def prepare_offline_figure_inputs(root: Path) -> Path:
    """Copy only renderer inputs; restore primary exports from sealed archives."""
    repository = Path(__file__).resolve().parents[2]
    cohort = "experiments/graph_inversion/frozen_targets/paper120_v3"
    shutil.copytree(repository / cohort, root / cohort)
    directories = (
        "scalar_primary_b40_b100_synthesis_v1", "graph_scalar_primary_b40_v1",
        "graph_scalar_primary_b100_v1", "repaired_mutaco_sbi_assessment_6251303/v1",
        "repaired_mutaco_sbi_assessment_preparation/v1", "sbi_zoo_comparison_v1",
        "sbi_zoo_p3_6265116/v1", "terminal_domain_ablation_b40_refit_v1",
        "terminal_domain_heldout_assessment_b40_v2",
    )
    for relative in directories:
        source = repository / "results/paper" / relative
        destination = root / "results/paper" / relative
        destination.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    data = "experiments/paper/oscillator_physics_multi_target_sensitivity/outputs/paper120_maximin12_job6353452/plotting_data.json"
    (root / data).parent.mkdir(parents=True)
    shutil.copy2(repository / data, root / data)
    archives = repository / "results/paper/scalar_primary_trajectory_archives_v3r2"
    for budget, digest in (
        (40, "2f86646f7c21914552d10e9e1fcb209d29880cee194a3e79f8f8fae211b6659c"),
        (100, "e30befedd72b85bd0eda8fe1d7e33ed6ef10acd0c880b1bd6aca60865e4a49c7"),
    ):
        archive = _original_trajectory_archive(archives / f"b{budget}-v3r2", budget)
        with archive.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
        destination = root / f"experiments/paper/exploration/outputs/scalar_primary_smoke/population_b{budget}_v3r2_compact_fetch/population/collection/compact_export.json"
        destination.parent.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as source, destination.open("wb") as output:
            shutil.copyfileobj(source.extractfile("collection/compact_export.json"), output)
        manifest = json.loads((repository / f"results/paper/graph_scalar_primary_b{budget}_v1/manifest.json").read_text())
        with destination.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["integrity"]["compact_export_sha256"]
    return root


@pytest.fixture(scope="session")
def offline_figure_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return prepare_offline_figure_inputs(tmp_path_factory.mktemp("offline-figure-inputs"))


@pytest.fixture(scope="session")
def native_terminal_preparation(historical_checkout, offline_analysis_root):
    """Read both original paired inventories using their checkout's code, data and tests."""
    from functools import lru_cache
    import subprocess
    import sys

    # First committed tree containing every source byte in both declarations.
    legacy_checkout = historical_checkout("3c74fba15e5d6a22ba8e202df032ba3180510f0a")

    @lru_cache(maxsize=2)
    def prepare(name):
        assert name in {"terminal_domain_heldout_assessment", "terminal_domain_heldout_assessment_v2"}
        # Restore the omitted member only in the disposable historical checkout.
        # Both algorithms and their recorded paths are used without monkeypatching.
        program = """
import importlib, json, sys
import shutil, subprocess
from hashlib import sha256
from pathlib import Path
root, external, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root), str(root/'src'), str(root/'packages/graph-simulator/src'),
                str(root/'packages/analytic-simulator/src')]
module = importlib.import_module('experiments.paper.common.' + name)
assert Path(module.__file__).resolve().is_relative_to(root)
destination = root / module.SEARCH_EXPORT
assert destination.resolve().is_relative_to(root)
assert not subprocess.check_output(["git", "-C", str(root), "ls-files", "--", str(module.SEARCH_EXPORT)])
source = external / module.SEARCH_EXPORT
manifest = json.loads((root/"benchmarks/paper/graph_scalar_primary_b40_v1/manifest.json").read_bytes())
expected_hash = manifest["integrity"]["compact_export_sha256"]
assert sha256(source.read_bytes()).hexdigest() == expected_hash
if not destination.exists():
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file)
assert sha256(destination.read_bytes()).hexdigest() == expected_hash
declaration, tasks, preflight = module.build_preparation(root)
archived_tasks = (root/module.OUTPUT/'tasks.jsonl').read_text().splitlines()
assert len(tasks) == 240 and tasks == [json.loads(line) for line in archived_tasks]
original = json.loads((root/module.OUTPUT/'declaration.json').read_bytes())
for key, value in original['source_provenance']['source_file_sha256'].items():
    assert sha256((root/key).read_bytes()).hexdigest() == value
print(json.dumps([declaration, tasks, preflight]))
"""
        output = subprocess.check_output(
            [sys.executable, "-I", "-B", "-c", program, str(legacy_checkout),
             str(offline_analysis_root), name], cwd=legacy_checkout,
        )
        return json.loads(output)
    return prepare
