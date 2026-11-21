"""Synthetic, zero-simulator checks for the target-98 release arrays."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paper.exploration.target098_response_archive import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    ArchiveValidationError,
    load_archive,
    render_figures,
    validate_arrays,
    write_archive,
)


REPO = Path(__file__).resolve().parents[2]


def _synthetic_arrays() -> dict[str, np.ndarray]:
    forces = np.linspace(0, 5, 200, dtype=np.float32)
    graph_offsets = np.arange(80, dtype=np.float32)[None, :] / np.float32(800)
    target_omega_graphs = forces[:, None] / np.float32(5) + graph_offsets
    target_M_graphs = np.float32(0.4) + forces[:, None] / np.float32(20) + graph_offsets
    sparse_forces = np.array([0, 1.25, 2.5, 3.75, 5], dtype=np.float32)
    return {
        "forces": forces,
        "sparse_forces": sparse_forces,
        "target_omega_graphs": target_omega_graphs,
        "target_M_graphs": target_M_graphs,
        "target_omega_mean": target_omega_graphs.mean(axis=1),
        "target_omega_std": target_omega_graphs.std(axis=1, ddof=1),
        "target_M_mean": target_M_graphs.mean(axis=1),
        "target_M_std": target_M_graphs.std(axis=1, ddof=1),
        "mutaco_omega": forces / np.float32(5),
        "mutaco_M": np.full(200, np.float32(0.5)),
        "true_sparse_omega": sparse_forces / np.float32(5),
        "true_sparse_M": np.full(5, np.float32(0.5)),
        "mutaco_sparse_omega": sparse_forces / np.float32(5),
        "mutaco_sparse_M": np.full(5, np.float32(0.5)),
    }


def _provenance() -> dict[str, object]:
    return {
        "target_position": 98,
        "target_seed": 5172277108155418169,
        "assessment_seed": 7596823460410727628,
        "target_record_id": "synthetic-target-98",
        "simulator_configuration": {
            "node_count": 10, "batch_size": 80, "repeat_size": 2,
            "temperature": 0.15, "dt": 0.1, "burn_in_steps": 100,
            "sample_steps": 900, "sample_stride": 1, "total_steps": 1000,
            "dtype": "float32",
        },
        "execution_device": "cpu",
        "max_conditions_per_chunk": 10,
        "evaluation_condition_counts": [200, 200, 10],
        "dense_grid": {
            "identifier": "paper120_dense_force_gauge_zero",
            "condition_count": 200, "gauges": [0.0],
        },
        "reductions": {"target_std": "unbiased across graphs"},
    }


def test_archive_round_trip_and_renderer_never_imports_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _synthetic_arrays()
    archive_dir = tmp_path / "archive"
    archive, manifest_path = write_archive(archive_dir, arrays, _provenance(), REPO)
    loaded, manifest = load_archive(archive, manifest_path)
    assert set(loaded) == set(arrays)
    for name, original in arrays.items():
        assert loaded[name].dtype == np.float32
        np.testing.assert_array_equal(loaded[name], original)
    assert manifest["simulator_evaluations_added_by_archival"] == 0
    assert manifest["source"]["git_head"]
    assert manifest["arrays"]["target_omega_graphs"]["shape"] == [200, 80]

    monkeypatch.setenv("MPLBACKEND", "Agg")
    original_import = builtins.__import__

    def no_simulator_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("mutaco_graph"):
            raise AssertionError("archive renderer imported the forward simulator")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_simulator_import)
    outputs = render_figures(archive, manifest_path, tmp_path / "figures", use_tex=False)
    assert len(outputs) == 4
    assert all(path.stat().st_size > 0 for path in outputs.values())


def test_archive_rejects_missing_corrupt_and_invalid_arrays(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ArchiveValidationError, match="missing"):
        load_archive(missing / ARCHIVE_NAME, missing / MANIFEST_NAME)

    arrays = _synthetic_arrays()
    invalid = dict(arrays)
    invalid["target_M_graphs"] = invalid["target_M_graphs"].astype(np.float64)
    with pytest.raises(ArchiveValidationError, match="float32"):
        validate_arrays(invalid)
    invalid = dict(arrays)
    invalid["mutaco_omega"] = invalid["mutaco_omega"][:-1]
    with pytest.raises(ArchiveValidationError, match="shape"):
        validate_arrays(invalid)
    invalid = dict(arrays)
    invalid["target_omega_mean"] = invalid["target_omega_mean"].copy()
    invalid["target_omega_mean"][0] += np.float32(0.1)
    with pytest.raises(ArchiveValidationError, match="mean/std"):
        validate_arrays(invalid)

    archive, manifest_path = write_archive(tmp_path / "valid", arrays, _provenance(), REPO)
    with pytest.raises(FileExistsError, match="already exist"):
        write_archive(tmp_path / "valid", arrays, _provenance(), REPO)
    archive.write_bytes(b"corrupt NPZ")
    with pytest.raises(ArchiveValidationError, match="SHA-256"):
        load_archive(archive, manifest_path)
    assert json.loads(manifest_path.read_text())["archive_sha256"]
