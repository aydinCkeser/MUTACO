"""Frozen target-98 plot data and simulator-free figure rendering.

The production plot script supplies arrays after its three existing simulator
evaluations. Loading and rendering this archive never imports the simulator.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
from zipfile import BadZipFile

import numpy as np


SCHEMA = "mutaco-paper120-target098-response-arrays-v1"
ARCHIVE_NAME = "paper120_target098_response_arrays_v1.npz"
MANIFEST_NAME = "paper120_target098_response_arrays_v1.manifest.json"

# All arrays are the actual float32 values passed to the two plot builders.
ARRAYS: dict[str, tuple[tuple[int, ...], str]] = {
    "forces": ((200,), "Dense F grid at Phi=0"),
    "sparse_forces": ((5,), "Canonical Phi=0 force grid"),
    "target_omega_graphs": ((200, 80), "Target mean phase rotation per graph, averaged over repeats"),
    "target_M_graphs": ((200, 80), "Target synchronization per graph, averaged over repeats"),
    "target_omega_mean": ((200,), "Mean of target_omega_graphs across graphs"),
    "target_omega_std": ((200,), "Unbiased standard deviation of target_omega_graphs across graphs"),
    "target_M_mean": ((200,), "Mean of target_M_graphs across graphs"),
    "target_M_std": ((200,), "Unbiased standard deviation of target_M_graphs across graphs"),
    "mutaco_omega": ((200,), "Dense MUTACO mean phase rotation across graphs and repeats"),
    "mutaco_M": ((200,), "Dense MUTACO mean synchronization across graphs and repeats"),
    "true_sparse_omega": ((5,), "Frozen target response at canonical Phi=0 forces"),
    "true_sparse_M": ((5,), "Frozen target synchronization at canonical Phi=0 forces"),
    "mutaco_sparse_omega": ((5,), "MUTACO response at canonical Phi=0 forces"),
    "mutaco_sparse_M": ((5,), "MUTACO synchronization at canonical Phi=0 forces"),
}


class ArchiveValidationError(ValueError):
    """The frozen target-98 data are missing or inconsistent."""


def _validate_scientific_provenance(scientific: Mapping[str, Any]) -> None:
    if scientific.get("target_position") != 98:
        raise ArchiveValidationError("target position must be 98")
    if scientific.get("target_seed") != 5172277108155418169:
        raise ArchiveValidationError("target seed differs from the frozen calculation")
    if scientific.get("assessment_seed") != 7596823460410727628:
        raise ArchiveValidationError("assessment seed differs from the frozen calculation")
    config = scientific.get("simulator_configuration")
    expected_config = {
        "node_count": 10, "batch_size": 80, "repeat_size": 2,
        "temperature": 0.15, "dt": 0.1, "burn_in_steps": 100,
        "sample_steps": 900, "sample_stride": 1, "total_steps": 1000,
        "dtype": "float32",
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected_config.items()):
        raise ArchiveValidationError("simulator configuration differs from the frozen calculation")
    if scientific.get("execution_device") != "cpu":
        raise ArchiveValidationError("target-98 archival calculation must use CPU")
    if scientific.get("max_conditions_per_chunk") != 10:
        raise ArchiveValidationError("target-98 chunk size differs from 10")
    if scientific.get("evaluation_condition_counts") != [200, 200, 10]:
        raise ArchiveValidationError("target-98 condition counts differ")
    dense = scientific.get("dense_grid")
    if not isinstance(dense, dict) or dense.get("identifier") != "paper120_dense_force_gauge_zero" \
            or dense.get("condition_count") != 200 or dense.get("gauges") != [0.0]:
        raise ArchiveValidationError("target-98 dense grid identity differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != set(ARRAYS):
        raise ArchiveValidationError(
            f"array names differ: missing={sorted(set(ARRAYS) - set(arrays))}, "
            f"extra={sorted(set(arrays) - set(ARRAYS))}"
        )
    checked: dict[str, np.ndarray] = {}
    for name, (shape, _) in ARRAYS.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != np.dtype("float32"):
            raise ArchiveValidationError(
                f"{name}: expected shape {shape} and float32, got {value.shape} and {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ArchiveValidationError(f"{name}: nonfinite values")
        checked[name] = np.ascontiguousarray(value)
    if checked["forces"][0] != 0 or checked["forces"][-1] != 5:
        raise ArchiveValidationError("dense force endpoints differ from 0 and 5")
    if not np.all(np.diff(checked["forces"]) > 0):
        raise ArchiveValidationError("dense forces are not strictly increasing")
    if not np.array_equal(
        checked["sparse_forces"], np.array([0, 1.25, 2.5, 3.75, 5], dtype=np.float32)
    ):
        raise ArchiveValidationError("sparse force grid differs from the canonical Phi=0 grid")
    for observable in ("omega", "M"):
        graphs = checked[f"target_{observable}_graphs"]
        if not np.allclose(
            checked[f"target_{observable}_mean"], graphs.mean(axis=1), rtol=3e-6, atol=3e-6
        ) or not np.allclose(
            checked[f"target_{observable}_std"], graphs.std(axis=1, ddof=1),
            rtol=3e-6, atol=3e-6
        ):
            raise ArchiveValidationError(f"target {observable} mean/std disagree with graph curves")
    return checked


def _source_provenance(repo: Path) -> dict[str, Any]:
    source_paths = [
        *sorted((repo / "src/mutaco").rglob("*.py")),
        *sorted((repo / "simulators/graph-simulator/src/mutaco_graph").rglob("*.py")),
        repo / "experiments/paper/common/population_v3.py",
        repo / "experiments/paper/common/protocol_v3.py",
        repo / "experiments/paper/exploration/plot_paper120_target098_mutaco_only_v1.py",
        repo / "experiments/paper/exploration/target098_response_archive.py",
        repo / "experiments/paper/exploration/render_paper120_target098_archive_v1.py",
        repo / "experiments/graph_inversion/configs/paper/paper120_v3_target_config.json",
        repo / "experiments/graph_inversion/frozen_targets/paper120_v3/target_098.json",
        repo / "results/paper/graph_paper120/paper120/paired_records.jsonl",
        repo / "results/paper/graph_paper120/paper120/executed_declaration.json",
        repo / "pyproject.toml",
    ]
    files = {path.relative_to(repo).as_posix(): sha256_file(path) for path in source_paths}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *files],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "matplotlib", "sbi", "botorch", "gpytorch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"git_head": revision, "source_status": status, "file_sha256": files, "versions": versions}


def write_archive(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    scientific_provenance: Mapping[str, Any],
    repo: Path,
) -> tuple[Path, Path]:
    """Write one append-only archive after the existing simulator calls finish."""
    checked = validate_arrays(arrays)
    _validate_scientific_provenance(scientific_provenance)
    archive_path = output_dir / ARCHIVE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if archive_path.exists() or manifest_path.exists():
        raise FileExistsError("target-98 archive outputs already exist")
    source = _source_provenance(repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    with archive_path.open("xb") as stream:
        np.savez_compressed(stream, **checked)
    manifest = {
        "schema_version": SCHEMA,
        "archive_name": ARCHIVE_NAME,
        "archive_sha256": sha256_file(archive_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "meaning": ARRAYS[name][1],
                "sha256_c_order": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            for name, value in checked.items()
        },
        "source": source,
        "scientific": dict(scientific_provenance),
        "simulator_evaluations_added_by_archival": 0,
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return archive_path, manifest_path


def load_archive(archive_path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate checksums and array contracts without simulator imports."""
    if not archive_path.is_file() or not manifest_path.is_file():
        raise ArchiveValidationError("target-98 archive or manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA or manifest.get("archive_name") != archive_path.name:
            raise ArchiveValidationError("target-98 manifest schema or archive name mismatch")
        if manifest.get("archive_sha256") != sha256_file(archive_path):
            raise ArchiveValidationError("target-98 NPZ SHA-256 mismatch")
        if set(manifest.get("arrays", {})) != set(ARRAYS):
            raise ArchiveValidationError("target-98 manifest array inventory mismatch")
        with np.load(archive_path, allow_pickle=False) as source:
            if set(source.files) != set(ARRAYS):
                raise ArchiveValidationError("target-98 NPZ array inventory mismatch")
            arrays = validate_arrays({name: source[name] for name in ARRAYS})
        for name, value in arrays.items():
            record = manifest["arrays"][name]
            if (
                record.get("shape") != list(value.shape)
                or record.get("dtype") != str(value.dtype)
                or record.get("meaning") != ARRAYS[name][1]
                or record.get("sha256_c_order") != hashlib.sha256(value.tobytes(order="C")).hexdigest()
            ):
                raise ArchiveValidationError(f"{name}: manifest does not match array")
        if manifest.get("simulator_evaluations_added_by_archival") != 0:
            raise ArchiveValidationError("manifest claims simulator calls during archival")
        scientific = manifest.get("scientific", {})
        _validate_scientific_provenance(scientific)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BadZipFile, EOFError) as exc:
        if isinstance(exc, ArchiveValidationError):
            raise
        raise ArchiveValidationError(f"invalid target-98 archive: {exc}") from exc
    return arrays, manifest


def render_figures(
    archive_path: Path, manifest_path: Path, output_dir: Path, *, use_tex: bool = True
) -> dict[str, Path]:
    """Render the original two-panel plots from archived values only."""
    arrays, _ = load_archive(archive_path, manifest_path)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage[T1]{fontenc}",
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })
    outputs = {
        "forward_pdf": output_dir / "paper120_target_098_forward.pdf",
        "forward_png": output_dir / "paper120_target_098_forward.png",
        "reconstruction_pdf": output_dir / "paper120_target_098_mutaco_reconstruction.pdf",
        "reconstruction_png": output_dir / "paper120_target_098_mutaco_reconstruction.png",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("target-98 figure output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    forces = arrays["forces"]
    sparse_forces = arrays["sparse_forces"]
    fig_forward, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, values, mean, std, ylabel in (
        (axes[0], arrays["target_omega_graphs"], arrays["target_omega_mean"],
         arrays["target_omega_std"], r"$\langle\bar{\omega}\rangle$"),
        (axes[1], arrays["target_M_graphs"], arrays["target_M_mean"],
         arrays["target_M_std"], r"$\langle M\rangle$"),
    ):
        for graph_index in range(values.shape[1]):
            axis.plot(forces, values[:, graph_index], color="gray", alpha=0.15, linewidth=0.6)
        axis.fill_between(forces, mean - std, mean + std, color="#56B4E9", alpha=0.25,
                          label=r"$\pm 1\sigma$ across graphs")
        axis.plot(forces, mean, color="#0072B2", linewidth=2, label="Ensemble mean")
        axis.set_xlabel(r"$F$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    fig_reconstruction, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    panels = (
        (axes[0], arrays["target_omega_mean"], arrays["target_omega_std"],
         arrays["true_sparse_omega"], arrays["mutaco_omega"], arrays["mutaco_sparse_omega"],
         r"$\langle\bar{\omega}\rangle$"),
        (axes[1], arrays["target_M_mean"], arrays["target_M_std"],
         arrays["true_sparse_M"], arrays["mutaco_M"], arrays["mutaco_sparse_M"],
         r"$\langle M\rangle$"),
    )
    for axis, target_mean, target_std, true_sparse, mutaco_dense, mutaco_sparse, ylabel in panels:
        axis.fill_between(forces, target_mean - target_std, target_mean + target_std,
                          color="lightgray", alpha=0.7, label=r"True mean $\pm 1\sigma$")
        axis.plot(forces, target_mean, color="black", linewidth=2)
        axis.plot(sparse_forces, true_sparse, linestyle="none", marker="o", markersize=10,
                  markerfacecolor="white", markeredgecolor="black", markeredgewidth=2,
                  label="True")
        axis.plot(forces, mutaco_dense, color="#0072B2", linestyle=":", linewidth=2)
        axis.plot(sparse_forces, mutaco_sparse, linestyle="none", marker="s", markersize=7,
                  markerfacecolor="#0072B2", markeredgecolor="#0072B2", label="MUTACO")
        axis.set_xlabel(r"$F$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    fixed_pdf_metadata = {
        "Creator": "MUTACO visualization replay", "Producer": "Matplotlib",
        "CreationDate": None, "ModDate": None,
    }
    fixed_png_metadata = {"Software": "MUTACO visualization replay"}
    fig_forward.savefig(outputs["forward_pdf"], bbox_inches="tight", metadata=fixed_pdf_metadata)
    fig_forward.savefig(outputs["forward_png"], bbox_inches="tight", dpi=300,
                        metadata=fixed_png_metadata)
    fig_reconstruction.savefig(outputs["reconstruction_pdf"], bbox_inches="tight",
                               metadata=fixed_pdf_metadata)
    fig_reconstruction.savefig(outputs["reconstruction_png"], bbox_inches="tight", dpi=300,
                               metadata=fixed_png_metadata)
    plt.close(fig_forward)
    plt.close(fig_reconstruction)
    return outputs
