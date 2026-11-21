"""Offline exact-source final-GP replay from immutable raw observations.

This script deliberately imports only the selected source tree's ``mutaco``
package.  It never constructs or calls a simulator and stops before terminal
recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch

from mutaco import stable_hash
from mutaco.identity import ScientificIdentity
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.response import FrozenResponseTransform
from mutaco.surrogate import GPFitConfig, SurrogateTrainingData
from mutaco.surrogate.exact_multitask import _build_model, _model_identity
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec


SCHEMA = "mutaco-exact-source-final-gp-replay-v1"
DOMAIN = "exact-source-final-gp-replay-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _specifications(intent: Mapping[str, Any]) -> tuple[ParameterSpec, ResponseSpec]:
    specifications = intent["specifications"]
    parameter_spec = ParameterSpec(
        tuple(ParameterAxis(**dict(axis)) for axis in specifications["parameters"])
    )
    response_spec = ResponseSpec(
        tuple(ResponseAxis(**dict(axis)) for axis in specifications["responses"])
    )
    return parameter_spec, response_spec


def _selected_parameters(fit_provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = [
        item for item in fit_provenance["recovery_attempts"] if item.get("selected")
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("final_parameters"), Mapping):
        raise ValueError("final fit does not have one selected persisted parameter state")
    return selected[0]["final_parameters"]


def replay(
    raw_path: Path,
    *,
    raw_artifact: str,
    expected_budget: int,
    source_commit: str,
) -> dict[str, Any]:
    started = perf_counter()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    projection = raw["scientific_projection"]
    provenance = raw["runner_provenance"]
    intent = provenance["configuration"]["scientific_intent"]
    observations = projection["observations"]
    expected_fits = projection["fits"]
    initial_count = int(raw["resolved_mutaco_budget"]["initial_points"])
    sequential_count = int(raw["resolved_mutaco_budget"]["sequential_rounds"])
    if raw.get("inference_calls") != expected_budget or len(observations) != expected_budget:
        raise ValueError("raw trajectory does not match the replay budget")
    if initial_count + sequential_count != expected_budget or len(expected_fits) != sequential_count + 1:
        raise ValueError("raw trajectory fit schedule is incomplete")
    if intent.get("dtype") != "float32":
        raise ValueError("historical replay requires the persisted float32 policy")

    dtype = torch.float32
    device = torch.device("cpu")
    parameter_spec, response_spec = _specifications(intent)
    theta = torch.tensor([item["theta"] for item in observations], dtype=dtype, device=device).contiguous()
    response_physical = torch.tensor(
        [item["response_physical"] for item in observations], dtype=dtype, device=device
    ).contiguous()
    observation_ids = tuple(str(item["observation_id"]) for item in observations)
    transform = FrozenResponseTransform.fit(
        response_physical[:initial_count],
        response_spec=response_spec,
        observation_ids=observation_ids[:initial_count],
        scale_floor=float(intent["transform_settings"]["scale_floor"]),
    )
    expected_transform_id = str(projection["transform"]["transform_id"])
    transform_equal = transform.metadata.transform_id == expected_transform_id
    standardized = transform.to_standardized(
        response_physical,
        response_names=tuple(axis.name for axis in response_spec.axes),
    ).contiguous()
    persisted_standardized = torch.tensor(
        [item["response_standardized"] for item in observations], dtype=dtype, device=device
    ).contiguous()
    standardized_equal = bool(torch.equal(standardized, persisted_standardized))

    science = ScientificIdentity(str(projection["scientific_identity"]))
    seed_deriver = SeedDeriver(science, int(intent["root_seed"]))
    gp_settings = dict(intent["gp_settings"])
    final_version = sequential_count
    final_expected = expected_fits[-1]
    final_runner_fit = provenance["fits"][-1]
    if final_runner_fit.get("training_version") != final_version:
        raise ValueError("runner final-fit provenance has the wrong training version")
    if any(
        final_runner_fit.get(key) != final_expected.get(key)
        for key in ("fit_id", "training_data_id", "model_id")
    ):
        raise ValueError("runner and scientific-projection final-fit identities differ")

    data = SurrogateTrainingData(
        theta=theta.clone().contiguous(),
        response_standardized=standardized.clone().contiguous(),
        parameter_spec=parameter_spec,
        observation_ids=observation_ids,
        transform_id=transform.metadata.transform_id,
    )
    fit_config = GPFitConfig.from_seed_deriver(
        seed_deriver,
        SeedCoordinate("gp_fit", ("training_version", final_version)),
        **gp_settings,
    )
    configuration_equal = _canonical_json(fit_config.provenance()) == _canonical_json(
        final_expected["configuration"]
    )
    training_data_id = data.identity()
    training_data_equal = training_data_id == final_expected["training_data_id"]

    model = _build_model(data, fit_config)
    named_parameters = dict(model.named_parameters())
    persisted_parameters = _selected_parameters(final_runner_fit)
    if set(named_parameters) != set(persisted_parameters):
        raise ValueError("persisted final parameter names do not match exact-source model")
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            value = torch.tensor(
                persisted_parameters[name], dtype=parameter.dtype, device=parameter.device
            )
            if value.shape != parameter.shape:
                raise ValueError(f"persisted final parameter shape mismatch: {name}")
            parameter.copy_(value)
    model.eval()
    materialized_model_id = _model_identity(model, data, fit_config)
    model_equal = materialized_model_id == final_expected["model_id"]

    terminal_context = provenance["configuration"]["execution"]["terminal_recommendation_context"]
    body = {
        "schema_version": SCHEMA,
        "source_commit": source_commit,
        "raw_artifact": raw_artifact,
        "target_position": int(raw["target_index"]),
        "target_record_id": str(raw["target_record_id"]),
        "budget": expected_budget,
        "observation_count": len(observations),
        "fit_count": len(expected_fits),
        "transform_identity_equal": transform_equal,
        "standardized_observations_bitwise_equal": standardized_equal,
        "final_fit_configuration_equal": configuration_equal,
        "final_training_data_identity_equal": training_data_equal,
        "expected_final_training_data_id": final_expected["training_data_id"],
        "materialized_final_training_data_id": training_data_id,
        "final_model_identity_equal": model_equal,
        "expected_final_model_id": final_expected["model_id"],
        "materialized_final_model_id": materialized_model_id,
        "final_parameter_state_loaded_from_archived_provenance": True,
        "optimizer_rerun": False,
        "qualification_passed": all(
            (
                transform_equal,
                standardized_equal,
                configuration_equal,
                training_data_equal,
                model_equal,
            )
        ),
        "terminal_context_verified_not_executed": {
            "policy_id": terminal_context["policy_id"],
            "target_id": terminal_context["target_id"],
            "target_index": terminal_context["target_index"],
            "root_seed": terminal_context["root_seed"],
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "dtype": "float32",
            "device": "cpu",
        },
        "elapsed_seconds": perf_counter() - started,
        "simulator_constructed": False,
        "simulator_calls": 0,
        "terminal_recommendation_executed": False,
    }
    identity_body = dict(body)
    identity_body.pop("elapsed_seconds")
    return {
        **body,
        "replay_identity": stable_hash(DOMAIN, identity_body),
    }


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only replay output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-artifact", required=True)
    parser.add_argument("--budget", type=int, choices=(40, 100), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        document = replay(
            arguments.raw.resolve(),
            raw_artifact=arguments.raw_artifact,
            expected_budget=arguments.budget,
            source_commit=arguments.source_commit,
        )
    except Exception as exc:
        document = {
            "schema_version": SCHEMA,
            "source_commit": arguments.source_commit,
            "raw_artifact": arguments.raw_artifact,
            "budget": arguments.budget,
            "qualification_passed": False,
            "failure_type": type(exc).__name__,
            "failure_explanation": str(exc),
            "simulator_constructed": False,
            "simulator_calls": 0,
            "terminal_recommendation_executed": False,
        }
    _write_once(arguments.output.resolve(), document)
    print(json.dumps(document, sort_keys=True, allow_nan=False))
    if not document.get("qualification_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
