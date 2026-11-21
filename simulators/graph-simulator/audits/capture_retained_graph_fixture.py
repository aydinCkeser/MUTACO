"""Capture a small exact fixture from the frozen historical NumPy/Numba graph path."""

from __future__ import annotations

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
from typing import Any

import numpy as np


SCHEMA_VERSION = "retained-graph-fixture-v1"
FIXTURE_IDENTITY = "retained-cpu-numpy-numba.small-p2.v1"


def _source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tensor_record(value: Any, axes: tuple[str, ...]) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    raw = array.tobytes(order="C")
    return {
        "axes": list(axes),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256(raw).hexdigest(),
        "data_base64": b64encode(raw).decode("ascii"),
        "values": array.tolist(),
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *args),
        text=True,
    ).strip()


def capture(historical: Path) -> dict[str, Any]:
    graph_modules = historical / "graph_modules"
    sys.path.insert(0, str(graph_modules))
    sys.path.insert(0, str(historical))
    # Parameter generation imports Seaborn only for optional plotting methods.
    # The frozen numerical path never calls those methods, so avoid turning an
    # unused historical visualization dependency into an oracle prerequisite.
    sys.modules.setdefault("seaborn", ModuleType("seaborn"))
    sys.modules.setdefault("pandas", ModuleType("pandas"))
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    patches = ModuleType("matplotlib.patches")
    patches.Rectangle = object
    matplotlib.pyplot = pyplot
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)
    sys.modules.setdefault("matplotlib.patches", patches)
    sklearn = ModuleType("sklearn")
    isotonic = ModuleType("sklearn.isotonic")
    isotonic.IsotonicRegression = object
    sklearn.isotonic = isotonic
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.isotonic", isotonic)
    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_: iterable
    sys.modules.setdefault("tqdm", tqdm_module)

    import graph_parameters
    import graph_systems
    import graph_utils
    import numba
    import scipy
    import simulator as simulator_module
    from graph_parameters import GraphParameterGenerator, HyperBatchParameters
    from graph_systems import GraphOscillatorSystem, compute_coupling_term_numba
    from graph_wrapper_for_bandits import run_graph_simulation_from_theta
    from simulator import SimulationParameters, Simulator

    for module in (graph_utils, graph_systems, simulator_module):
        module.xp = np

    theta = np.asarray(((1.0, 1.25, 1.0, 0.30),), dtype=np.float64)
    fixed = np.asarray((0.25, 0.25, 0.25), dtype=np.float64)
    node_count = 3
    batch_size = 2
    repeat_size = 2
    forces = np.asarray((0.0, 5.0), dtype=np.float64)
    gauges = np.asarray((0.0, np.pi), dtype=np.float64)
    steps = 3
    dt = 0.1
    temperature = 0.15
    seed = 73007

    weight_mean, coupling_mean, bias_mean, coupling_cv = theta[0]
    weight_cv, inverse_viscosity_cv, bias_cv = fixed
    hyper = HyperBatchParameters(
        weight_mean=weight_mean,
        inverse_viscosity_mean=1.0,
        self_coupling_mean=0.0,
        pairwise_coupling_mean=coupling_mean,
        bias_scale_mean=bias_mean,
        weight_variance=(weight_cv * weight_mean) ** 2,
        inverse_viscosity_variance=inverse_viscosity_cv**2,
        self_coupling_variance=0.0,
        pairwise_coupling_variance=(coupling_cv * coupling_mean) ** 2,
        bias_scale_variance=(bias_cv * bias_mean) ** 2,
        mass_mean=1e-3,
        mass_variance=0.0,
        correlations={(i, j): 0.0 for i in range(7) for j in range(i + 1, 7)},
    )
    sim_parameters = SimulationParameters(
        d_t=dt,
        n_t=steps,
        seed=seed,
        size_parameters={
            "hyper_batch_size": 1,
            "batch_size": batch_size,
            "repeat_size": repeat_size,
        },
    )
    parameters = GraphParameterGenerator([hyper], rng=sim_parameters.rng).generate_parameters(
        N_hyper_batch=1,
        N_batch=batch_size,
        N=node_count,
    )

    retained = Simulator(
        parameters=sim_parameters,
        normalized_temperature=temperature,
        system_class=GraphOscillatorSystem,
        record_time_series=False,
        verbose=False,
    )
    retained.create_dc_bias(driving_forces=forces, applied_gauges=gauges)
    retained.create_system(system_size={"N": node_count}, parameter_dict=parameters)
    system = retained.system
    assert isinstance(system, GraphOscillatorSystem)

    tensors: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, axes: tuple[str, ...]) -> None:
        tensors[name] = _tensor_record(value, axes)

    add("canonical_theta", theta[:, (1, 3)], ("H", "P"))
    add("expanded_wrapper_theta", theta, ("H", "wrapper_parameter"))
    add("fixed_system_parameters", fixed, ("fixed_parameter",))
    for name in ("weight", "inverse_viscosity", "self_coupling", "bias_scale", "mass"):
        add(f"realized_{name}", parameters[name], ("H", "B", "N"))
    add(
        "realized_pairwise_coupling_before_diagonal",
        parameters["pairwise_coupling_matrix"],
        ("H", "B", "N", "N"),
    )
    add("realized_total_coupling", system.K_total, ("H", "B", "N", "N"))
    add("initial_phase", system.current_state_phi, ("H", "V", "B", "R", "N"))
    add("overdamped_mask", system.overdamped_mask, ("H", "singleton_V", "B", "singleton_R", "N"))

    noises: list[np.ndarray] = []
    for step in range(steps):
        noise = retained._draw_noise()
        noises.append(np.array(noise, copy=True))
        if step == 0:
            phi = system.current_state_phi
            velocity = system.current_state_dphi
            A = system.A[:, None, :, None, :]
            W = system.W[:, None, :, None, :]
            alpha = system.alpha[:, None, :, None, :]
            mass = system.mass[:, None, :, None, :]
            gauge = system.gauge_column[None, :, None, None, None]
            force = system.force_column[None, :, None, None, None]
            restoring = -W * np.sin(phi - A * gauge)
            coupling = compute_coupling_term_numba(phi, system.K_total)
            bias = A * force
            noise_term = noise * np.sqrt(alpha)
            alpha_force = alpha * (restoring + coupling + bias) + noise_term
            underdamped_acceleration = (alpha_force - velocity) / mass
            add("first_step_restoring", restoring, ("H", "V", "B", "R", "N"))
            add("first_step_coupling", coupling, ("H", "V", "B", "R", "N"))
            add("first_step_bias", bias, ("H", "V", "B", "R", "N"))
            add("first_step_noise_term", noise_term, ("H", "V", "B", "R", "N"))
            add("first_step_alpha_force", alpha_force, ("H", "V", "B", "R", "N"))
            add(
                "first_step_underdamped_acceleration",
                underdamped_acceleration,
                ("H", "V", "B", "R", "N"),
            )
        new_velocity, new_acceleration = system.equation_of_motion(noise)
        if step == 0:
            add("first_step_returned_velocity", new_velocity, ("H", "V", "B", "R", "N"))
            add("first_step_returned_acceleration", new_acceleration, ("H", "V", "B", "R", "N"))
        system.update_state(new_velocity, new_acceleration, dt)
        if step == 0:
            add("first_step_phase_after_update", system.current_state_phi, ("H", "V", "B", "R", "N"))
            add("first_step_velocity_after_update", system.current_state_dphi, ("H", "V", "B", "R", "N"))
            add("first_step_Z_accumulator", system.Z_accum, ("H", "V", "B", "R"))
            add("first_step_phase_increment_accumulator", system.phi_diff_accum, ("H", "V", "B", "R"))
    add("thermal_noise", np.stack(noises), ("T", "H", "V", "B", "R", "N"))
    add("final_phase", system.current_state_phi, ("H", "V", "B", "R", "N"))
    add("final_velocity", system.current_state_dphi, ("H", "V", "B", "R", "N"))
    add("final_Z_accumulator", system.Z_accum, ("H", "V", "B", "R"))
    add("final_phase_increment_accumulator", system.phi_diff_accum, ("H", "V", "B", "R"))

    raw_output = system.output(dt, steps)
    angular = raw_output["angular_velocity"].reshape(1, len(forces), len(gauges), batch_size, repeat_size)
    magnitude = raw_output["Z_magnitude"].reshape(1, len(forces), len(gauges), batch_size, repeat_size)
    response = np.concatenate(
        (
            np.mean(angular, axis=(-1, -2)).reshape(1, -1),
            np.mean(magnitude, axis=(-1, -2)).reshape(1, -1),
        ),
        axis=-1,
    )
    add("raw_angular_velocity", raw_output["angular_velocity"], ("H", "V", "B", "R"))
    add("raw_Z_magnitude", raw_output["Z_magnitude"], ("H", "V", "B", "R"))
    add("final_response", response, ("H", "M"))

    wrapper_output, _ = run_graph_simulation_from_theta(
        theta,
        fixed,
        {"N": node_count},
        {
            "d_t": dt,
            "n_t": steps,
            "seed": seed,
            "batch_size": batch_size,
            "repeat_size": repeat_size,
        },
        len(forces),
        temperature,
        plots_enable=False,
        verbose=False,
    )
    if not np.array_equal(response, wrapper_output):
        raise RuntimeError("instrumented retained path does not exactly match wrapper output")

    source_files = (
        historical / "graph_wrapper_for_bandits.py",
        graph_modules / "graph_parameters.py",
        graph_modules / "graph_systems.py",
        graph_modules / "simulator.py",
        graph_modules / "graph_utils.py",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_identity": FIXTURE_IDENTITY,
        "classification": "frozen retained behavior; not scientific endorsement",
        "provenance": {
            "historical_repository": str(historical),
            "historical_commit": _git(historical, "rev-parse", "HEAD"),
            "historical_status": _git(historical, "status", "--short").splitlines(),
            "source_sha256": {
                path.relative_to(historical).as_posix(): _source_hash(path)
                for path in source_files
            },
            "backend": "cpu_numpy_numba",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "numba_version": numba.__version__,
        },
        "configuration": {
            "canonical_theta": [1.25, 0.30],
            "expanded_wrapper_theta": theta[0].tolist(),
            "fixed_system_parameters": fixed.tolist(),
            "node_count": node_count,
            "batch_size": batch_size,
            "repeat_size": repeat_size,
            "forces": forces.tolist(),
            "gauges": gauges.tolist(),
            "steps": steps,
            "dt": dt,
            "temperature": temperature,
            "seed": seed,
            "response_names": [
                f"{quantity}(F={force:.1f}, gauge={gauge_name})"
                for quantity in ("omega", "Z_magnitude")
                for force in forces
                for gauge_name in ("0", "pi")
            ],
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing fixture: {output}")
    document = capture(args.historical_repository.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
