"""Generate one corrected terminal recommendation without simulator access."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch

from mutaco import stable_hash
from mutaco.identity import ScientificIdentity
from mutaco.randomness import SeedCoordinate, SeedDeriver
from mutaco.recommendation.physical_mae import recommend_continuous_physical_mae
from mutaco.response import FrozenResponseTransform
from mutaco.surrogate import GPFitConfig, SurrogateTrainingData
from mutaco.surrogate.exact_multitask import (
    ExactMultiTaskGPSurrogate,
    _build_model,
    _model_identity,
)
from mutaco.types import ParameterAxis, ParameterSpec, ResponseAxis, ResponseSpec


SCHEMA = "mutaco-corrected-terminal-recommendation-attempt-v1"
IDENTITY_DOMAIN = "repaired-sbi-corrected-terminal-recommendation-v1"


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _specifications(intent: Mapping[str, Any]) -> tuple[ParameterSpec, ResponseSpec]:
    specifications = intent["specifications"]
    return (
        ParameterSpec(tuple(ParameterAxis(**dict(axis)) for axis in specifications["parameters"])),
        ResponseSpec(tuple(ResponseAxis(**dict(axis)) for axis in specifications["responses"])),
    )


def _selected_parameters(fit_provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = [item for item in fit_provenance["recovery_attempts"] if item.get("selected")]
    if len(selected) != 1 or not isinstance(selected[0].get("final_parameters"), Mapping):
        raise ValueError("final fit does not have one selected archived parameter state")
    return selected[0]["final_parameters"]


def _without_timings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_timings(item)
            for key, item in value.items()
            if not key.endswith("_seconds") and key != "component_timings"
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def generate(
    raw_path: Path,
    *,
    raw_artifact: str,
    raw_sha256: str,
    expected_budget: int,
    source_commit: str,
    composite_source_identity: str,
    exact_replay_identity: str,
    expected_model_id: str,
) -> dict[str, Any]:
    started = perf_counter()
    if _file_hash(raw_path) != raw_sha256:
        raise ValueError("terminal raw artifact hash mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    projection = raw["scientific_projection"]
    provenance = raw["runner_provenance"]
    intent = provenance["configuration"]["scientific_intent"]
    observations = projection["observations"]
    fits = projection["fits"]
    if raw.get("inference_calls") != expected_budget or len(observations) != expected_budget:
        raise ValueError("terminal raw trajectory budget mismatch")
    initial_count = int(raw["resolved_mutaco_budget"]["initial_points"])
    final_fit = fits[-1]
    runner_fit = provenance["fits"][-1]
    if final_fit.get("model_id") != expected_model_id:
        raise ValueError("terminal expected model identity mismatch")

    dtype = torch.float32
    device = torch.device("cpu")
    parameter_spec, response_spec = _specifications(intent)
    theta = torch.tensor([item["theta"] for item in observations], dtype=dtype, device=device).contiguous()
    response = torch.tensor(
        [item["response_physical"] for item in observations], dtype=dtype, device=device
    ).contiguous()
    observation_ids = tuple(str(item["observation_id"]) for item in observations)
    transform = FrozenResponseTransform.fit(
        response[:initial_count],
        response_spec=response_spec,
        observation_ids=observation_ids[:initial_count],
        scale_floor=float(intent["transform_settings"]["scale_floor"]),
    )
    standardized = transform.to_standardized(
        response, response_names=tuple(axis.name for axis in response_spec.axes)
    ).contiguous()
    persisted = torch.tensor(
        [item["response_standardized"] for item in observations], dtype=dtype, device=device
    ).contiguous()
    if not torch.equal(standardized, persisted):
        raise ValueError("terminal standardized observations do not replay bitwise")

    science = ScientificIdentity(str(projection["scientific_identity"]))
    fit_seed_deriver = SeedDeriver(science, int(intent["root_seed"]))
    final_version = len(fits) - 1
    fit_config = GPFitConfig.from_seed_deriver(
        fit_seed_deriver,
        SeedCoordinate("gp_fit", ("training_version", final_version)),
        **dict(intent["gp_settings"]),
    )
    data = SurrogateTrainingData(
        theta=theta,
        response_standardized=standardized,
        parameter_spec=parameter_spec,
        observation_ids=observation_ids,
        transform_id=transform.metadata.transform_id,
    )
    if data.identity() != final_fit["training_data_id"]:
        raise ValueError("terminal final training-data identity mismatch")
    model = _build_model(data, fit_config)
    named_parameters = dict(model.named_parameters())
    archived_parameters = _selected_parameters(runner_fit)
    if set(named_parameters) != set(archived_parameters):
        raise ValueError("terminal archived parameter names mismatch")
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            value = torch.tensor(
                archived_parameters[name], dtype=parameter.dtype, device=parameter.device
            )
            if value.shape != parameter.shape:
                raise ValueError(f"terminal archived parameter shape mismatch: {name}")
            parameter.copy_(value)
    model.eval()
    materialized_model_id = _model_identity(model, data, fit_config)
    if materialized_model_id != expected_model_id:
        raise ValueError("terminal materialized final-model identity mismatch")
    surrogate = ExactMultiTaskGPSurrogate(
        model,
        training_data=data,
        model_id=materialized_model_id,
        config=fit_config,
        fit_id=str(final_fit["fit_id"]),
    )
    context = provenance["configuration"]["execution"]["terminal_recommendation_context"]
    terminal_seed_deriver = SeedDeriver(
        ScientificIdentity(str(context["seed_family_id"])), int(context["root_seed"])
    )
    target = torch.tensor(intent["physical_target"], dtype=dtype, device=device).contiguous()
    recommendation = recommend_continuous_physical_mae(
        surrogate=surrogate,
        transform=transform,
        target_physical=target,
        target_id=str(context["target_id"]),
        parameter_spec=parameter_spec,
        seed_deriver=terminal_seed_deriver,
        target_index=int(context["target_index"]),
    ).document()
    if recommendation.get("simulator_calls") != 0 or recommendation.get("gp_fits") != 0:
        raise ValueError("terminal recommendation violated zero-call boundary")
    semantic = {
        "source_commit": source_commit,
        "composite_source_identity": composite_source_identity,
        "raw_sha256": raw_sha256,
        "exact_replay_identity": exact_replay_identity,
        "budget": expected_budget,
        "target_position": int(raw["target_index"]),
        "target_record_id": str(raw["target_record_id"]),
        "final_model_id": materialized_model_id,
        "recommendation": _without_timings(recommendation),
    }
    return {
        "schema_version": SCHEMA,
        **semantic,
        "recommendation_identity": stable_hash(IDENTITY_DOMAIN, semantic),
        "recommendation_with_timings": recommendation,
        "call_accounting": {
            "simulator_constructed": False,
            "simulator_calls": 0,
            "search_calls": 0,
            "gp_fit_attempts": 0,
            "terminal_optimizations": 1,
        },
        "elapsed_seconds": perf_counter() - started,
        "status": "completed",
        "raw_artifact": raw_artifact,
    }


def _write_once(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only terminal output exists: {path}")
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
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--budget", type=int, choices=(40, 100), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--composite-source-identity", required=True)
    parser.add_argument("--exact-replay-identity", required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        document = generate(
            arguments.raw.resolve(),
            raw_artifact=arguments.raw_artifact,
            raw_sha256=arguments.raw_sha256,
            expected_budget=arguments.budget,
            source_commit=arguments.source_commit,
            composite_source_identity=arguments.composite_source_identity,
            exact_replay_identity=arguments.exact_replay_identity,
            expected_model_id=arguments.expected_model_id,
        )
    except Exception as exc:
        document = {
            "schema_version": SCHEMA,
            "source_commit": arguments.source_commit,
            "raw_artifact": arguments.raw_artifact,
            "raw_sha256": arguments.raw_sha256,
            "budget": arguments.budget,
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_explanation": str(exc),
            "call_accounting": {
                "simulator_constructed": False,
                "simulator_calls": 0,
                "search_calls": 0,
                "gp_fit_attempts": 0,
                "terminal_optimizations": 0,
            },
        }
    _write_once(arguments.output.resolve(), document)
    print(_canonical(document))
    if document.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
