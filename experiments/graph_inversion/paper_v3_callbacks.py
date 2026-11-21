"""Graph-owned bindings for the generic v3 paper population runner.

The method implementations remain the accepted MUTACO and NPE adapters.  This
module supplies only the experiment boundary that forces every physical call
through :func:`execute_population_v3`'s capped callback.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import (
    ScientificIdentity,
    SimulationContext,
    SimulationResult,
    checked_evaluate,
    freeze_strict_metadata,
    stable_hash,
    strict_metadata_document,
)
from experiments.benchmarking.artifacts import file_hash, read_json, write_pretty_json
from experiments.benchmarking.methods.direct_posterior_inference import (
    run_direct_posterior_sample_map_target,
    run_npe_sample_map_target,
)
from experiments.benchmarking.methods.mutaco_reference import run_mutaco_target
from experiments.benchmarking.simulators.graph_scientific_d4 import (
    GraphScientificD4Adapter,
)
from experiments.paper.common.population_v3 import MethodDeclarationV3
from experiments.paper.common.protocol_v3 import (
    AssessmentEvaluationV3,
    DecisionOutcome,
    PhysicalCallCounts,
)
from experiments.paper.common.validation import PaperContractError


MUTACO_METHODS = ("mutaco", "mutaco_standardized", "mutaco_unstandardized")
SUPPORTED_METHODS = (*MUTACO_METHODS, "sbi_npe", "sbi_snpe")


class GraphRawEvidencePersistenceError(PaperContractError):
    failure_stage = "post_decision_raw_evidence_persistence"


def method_configuration_identity(method: str, configuration_path: Path) -> str:
    """Return the stable, target-independent identity of one method config."""

    if method not in SUPPORTED_METHODS:
        raise PaperContractError(f"unsupported graph v3 method {method!r}")
    from mutaco import stable_hash

    return stable_hash(
        "graph-paper-v3-method-configuration-v1",
        {"method": method, "configuration_sha256": file_hash(configuration_path)},
    )


def _context_document(context: SimulationContext) -> dict[str, Any]:
    return {
        "scientific_run_id": context.scientific_run_id,
        "purpose": context.purpose,
        "logical_index": context.logical_index,
        "repeat_index": context.repeat_index,
        "evaluation_seed": context.evaluation_seed,
        "attempt_index": context.attempt_index,
    }


def _restore_context(document: Mapping[str, Any]) -> SimulationContext:
    return SimulationContext(
        str(document["scientific_run_id"]),
        str(document["purpose"]),
        int(document["logical_index"]),
        int(document["repeat_index"]),
        int(document["evaluation_seed"]),
        int(document.get("attempt_index", 0)),
    )


def _assessment_randomness_evidence(
    assessment_context: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    raw_manifest = metadata.get("seed_manifest")
    if not isinstance(raw_manifest, Mapping) or not raw_manifest:
        raise PaperContractError("graph assessment did not return its seed manifest")
    manifests: dict[str, dict[str, int]] = {
        "quenched_disorder": {},
        "initial_state": {},
        "thermal_noise": {},
    }
    for key, value in raw_manifest.items():
        name = str(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PaperContractError("graph assessment seed manifest is malformed")
        if name.startswith(("node_parameter/", "pairwise_coupling/")):
            namespace = "quenched_disorder"
        elif name.startswith("initial_state/"):
            namespace = "initial_state"
        elif name.startswith(("thermal_noise/", "thermal_noise_dense_force_v1/")):
            namespace = "thermal_noise"
        else:
            raise PaperContractError(f"unclassified graph assessment seed: {name}")
        manifests[namespace][name] = value
    if any(not manifest for manifest in manifests.values()):
        raise PaperContractError("graph assessment omitted a required randomness namespace")
    family = str(assessment_context["family_identity"])
    root_seed = int(assessment_context["evaluation_seed"])
    identities = {
        namespace: stable_hash(
            "paper-assessment-randomness-coordinate-family-v1",
            {
                "assessment_family_identity": family,
                "root_evaluation_seed": root_seed,
                "namespace": namespace,
            },
        )
        for namespace in manifests
    }
    hashes = {
        namespace: stable_hash(
            "paper-assessment-seed-manifest-namespace-v1",
            {"namespace": namespace, "seeds": manifest},
        )
        for namespace, manifest in manifests.items()
    }
    full = {key: value for manifest in manifests.values() for key, value in manifest.items()}
    compact = {
        "schema_version": "mutaco-paper-assessment-randomness-evidence-v1",
        "root_evaluation_seed": root_seed,
        "coordinate_family_identities": identities,
        "namespace_seed_manifest_hashes": hashes,
        "full_seed_manifest_hash": stable_hash(
            "paper-assessment-seed-manifest-v1", full
        ),
    }
    raw = {
        "schema_version": "mutaco-paper-assessment-randomness-raw-v1",
        "root_evaluation_seed": root_seed,
        "namespace_seed_manifests": manifests,
        "full_seed_manifest": full,
    }
    return {"compact": compact, "raw": raw}


class _CallbackSimulator:
    """Capability-only simulator facade with no uncapped physical backend."""

    def __init__(
        self,
        capabilities: Any,
        evaluate: Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._capabilities = capabilities
        self._evaluate = evaluate
        self._device = device
        self._dtype = dtype

    @property
    def capabilities(self) -> Any:
        return self._capabilities

    def evaluate(
        self, theta: torch.Tensor, context: SimulationContext | None = None
    ) -> SimulationResult:
        if context is None:
            raise PaperContractError("graph v3 physical calls require named context")
        values = self._evaluate(
            theta.detach().cpu().tolist(),
            {"simulation_context": _context_document(context)},
        )
        return SimulationResult(
            torch.tensor(values, device=self._device, dtype=self._dtype).contiguous(),
            {"physical_backend": "population_v3_capped_callback"},
        )

    def evaluate_batch(
        self, theta: torch.Tensor, contexts: Sequence[SimulationContext]
    ) -> SimulationResult:
        if len(contexts) != int(theta.shape[0]):
            raise PaperContractError("one named context is required per graph batch row")
        rows = [
            self.evaluate(theta[index].contiguous(), context).values
            for index, context in enumerate(contexts)
        ]
        return SimulationResult(
            torch.stack(rows).contiguous(),
            {"physical_backend": "population_v3_capped_callback", "row_count": len(rows)},
        )


class _CallbackGraphAdapter:
    identifier = GraphScientificD4Adapter.identifier

    def __init__(
        self,
        capabilities: Any,
        evaluate: Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._capabilities = capabilities
        self._evaluate = evaluate
        self._device = device
        self._dtype = dtype

    def build_simulator(self, *args: Any, **kwargs: Any) -> _CallbackSimulator:
        del args, kwargs
        return _CallbackSimulator(
            self._capabilities,
            self._evaluate,
            device=self._device,
            dtype=self._dtype,
        )

    def latent_response(self, theta: torch.Tensor) -> None:
        del theta
        return None


class GraphPopulationCallbacks:
    """Exact graph callbacks consumed by the v3 population orchestrator."""

    def __init__(
        self,
        *,
        mutaco_configuration: Path | None = None,
        npe_configuration: Path | None = None,
        snpe_configuration: Path | None = None,
        method_configurations: Mapping[str, Path] | None = None,
        output: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cpu" or dtype is not torch.float32:
            raise PaperContractError("graph Paper-v3 callbacks require CPU float32")
        paths = dict(method_configurations or {})
        for method, path in (
            ("mutaco", mutaco_configuration),
            ("sbi_npe", npe_configuration),
            ("sbi_snpe", snpe_configuration),
        ):
            if path is not None:
                if method in paths:
                    raise PaperContractError(f"duplicate graph v3 configuration for {method}")
                paths[method] = path
        if not paths or any(method not in SUPPORTED_METHODS for method in paths):
            raise PaperContractError("graph Paper-v3 callbacks require supported method configs")
        self.configuration_paths = {
            method: path.resolve() for method, path in paths.items()
        }
        self.configurations = {
            method: read_json(path) for method, path in self.configuration_paths.items()
        }
        self.output = output
        self.device = device
        self.dtype = dtype
        base = GraphScientificD4Adapter()
        simulators = {
            method: base.build_simulator(config, device=device, dtype=dtype)
            for method, config in self.configurations.items()
        }
        documents = {
            method: simulator.capabilities.scientific_document()
            for method, simulator in simulators.items()
        }
        reference_method = next(iter(documents))
        if any(document != documents[reference_method] for document in documents.values()):
            raise PaperContractError("graph v3 method simulator configurations differ")
        self._physical_simulator = simulators[reference_method]
        self.capabilities = self._physical_simulator.capabilities

    def _effective_configuration(
        self, method: str, inference_context: Mapping[str, Any]
    ) -> dict[str, Any]:
        # The v3 context is additional orchestration provenance.  It must not
        # move the accepted adapter's internal coordinate-addressed streams.
        if not isinstance(inference_context.get("root_seed"), int):
            raise PaperContractError("graph v3 inference root seed is invalid")
        return deepcopy(self.configurations[method])

    def _raw_path(self, target_record_id: str, method: str) -> Path:
        token = sha256(target_record_id.encode("utf-8")).hexdigest()[:16]
        return self.output / "raw" / f"target-{token}" / f"{method}.json"

    def _persist_raw(
        self, target_record_id: str, method: str, row: Mapping[str, Any]
    ) -> None:
        path = self._raw_path(target_record_id, method)
        if path.exists() or path.with_suffix(".sha256").exists():
            raise PaperContractError("append-only raw graph method evidence already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        projected = strict_metadata_document(row)
        document = strict_metadata_document(
            freeze_strict_metadata(projected, label="graph v3 raw method evidence")
        )
        write_pretty_json(path, document)
        path.with_suffix(".sha256").write_text(file_hash(path), encoding="ascii")

    def _persist_raw_staged(
        self, target_record_id: str, method: str, row: Mapping[str, Any]
    ) -> None:
        try:
            self._persist_raw(target_record_id, method, row)
        except Exception as exc:
            raise GraphRawEvidencePersistenceError(str(exc)) from exc

    def run_method(
        self,
        declaration: MethodDeclarationV3,
        target: Mapping[str, Any],
        inference_context: Mapping[str, Any],
        evaluate: Callable[[Sequence[float], Mapping[str, Any]], Sequence[float]],
    ) -> DecisionOutcome:
        method = declaration.method
        if method not in SUPPORTED_METHODS:
            raise PaperContractError(f"unsupported graph v3 method {method!r}")
        expected_identity = method_configuration_identity(
            method, self.configuration_paths[method]
        )
        if declaration.configuration_identity != expected_identity:
            raise PaperContractError("graph v3 method configuration identity mismatch")
        coordinate = inference_context.get("coordinate")
        if not isinstance(coordinate, Mapping) or coordinate.get("namespace") != "method_inference":
            raise PaperContractError("graph v3 callback requires method-addressed inference context")
        if list(coordinate.get("coordinates", ())) != [target["target_record_id"], method]:
            raise PaperContractError("graph v3 inference coordinate differs from target/method")
        config = self._effective_configuration(method, inference_context)
        if int(config.get("simulation_budget_per_target", -1)) != declaration.inference_budget:
            raise PaperContractError("graph method config budget differs from declaration")
        target_index = int(target["target_index"])
        target_record_id = str(target["target_record_id"])
        target_seed = int(target.get("target_generation_seed", target.get("target_seed")))
        theta_star = torch.tensor(target["theta_star"], device=self.device, dtype=self.dtype)
        target_response = torch.tensor(
            target["target_response"], device=self.device, dtype=self.dtype
        )
        adapter = _CallbackGraphAdapter(
            self.capabilities, evaluate, device=self.device, dtype=self.dtype
        )
        campaign_science = ScientificIdentity(str(target["target_family_id"]))
        if method in MUTACO_METHODS:
            row = run_mutaco_target(
                adapter=adapter,
                index=target_index,
                theta_star=theta_star,
                config=config,
                campaign_science=campaign_science,
                device=self.device,
                dtype=self.dtype,
                supplied_target_response=target_response,
                supplied_target_seed=target_seed,
                supplied_target_identity=target_record_id,
                perform_held_out_assessment=False,
            )
            calls = int(row.get("inference_calls", 0))
            self._persist_raw_staged(target_record_id, method, row)
            if row.get("status") != "completed":
                return DecisionOutcome(
                    method=method,
                    status="failed",
                    decision_policy_identifier=declaration.decision_policy_identifier,
                    method_configuration_identity=declaration.configuration_identity,
                    inference_budget=declaration.inference_budget,
                    inference_calls=PhysicalCallCounts(calls, calls),
                    failure_stage="method_execution",
                    failure_type="MutacoPopulationFailure",
                    failure_explanation=str(row.get("failure_explanation")),
                )
            metadata = dict(row["graph_terminal_recommendation"])
            return DecisionOutcome(
                method=method,
                status="recommended",
                decision_policy_identifier=declaration.decision_policy_identifier,
                method_configuration_identity=declaration.configuration_identity,
                inference_budget=declaration.inference_budget,
                inference_calls=PhysicalCallCounts(calls, calls),
                recommended_theta=tuple(float(value) for value in row["recommended_theta"]),
                decision_metadata=metadata,
            )

        token = sha256(target_record_id.encode("utf-8")).hexdigest()[:16]
        runner = (
            run_npe_sample_map_target
            if method == "sbi_npe"
            else run_direct_posterior_sample_map_target
        )
        runner_kwargs = {} if method == "sbi_npe" else {"method": method}
        row = runner(
            adapter=adapter,
            index=target_index,
            theta_star=theta_star,
            target_response=target_response,
            config=config,
            campaign_science=campaign_science,
            device=self.device,
            dtype=self.dtype,
            artifact_output=self.output / "method_logs" / f"target-{token}" / method,
            target_record_id=target_record_id,
            **runner_kwargs,
        )
        outcome = row.pop("decision_outcome", None)
        self._persist_raw_staged(target_record_id, method, row)
        calls = int(row.get("inference_calls", 0))
        if not isinstance(outcome, DecisionOutcome) or outcome.status != "recommended":
            return DecisionOutcome(
                method=method,
                status="failed",
                decision_policy_identifier=declaration.decision_policy_identifier,
                method_configuration_identity=declaration.configuration_identity,
                inference_budget=declaration.inference_budget,
                inference_calls=PhysicalCallCounts(calls, calls),
                failure_stage="final_decision",
                failure_type=str(row.get("failure_type", "DirectPosteriorPopulationFailure")),
                failure_explanation=str(row.get("failure_explanation")),
                decision_metadata=row.get("decision_metadata"),
            )
        metadata = dict(outcome.decision_metadata or {})
        return DecisionOutcome(
            method=method,
            status="recommended",
            decision_policy_identifier=declaration.decision_policy_identifier,
            method_configuration_identity=declaration.configuration_identity,
            inference_budget=declaration.inference_budget,
            inference_calls=PhysicalCallCounts(calls, calls),
            recommended_theta=outcome.recommended_theta,
            decision_metadata=metadata,
        )

    def evaluate_inference(
        self,
        method: str,
        target: Mapping[str, Any],
        theta: Sequence[float],
        simulator_context: Mapping[str, Any],
    ) -> Sequence[float]:
        if method not in SUPPORTED_METHODS:
            raise PaperContractError(f"unsupported graph v3 method {method!r}")
        del target
        context_document = simulator_context.get("simulation_context")
        if not isinstance(context_document, Mapping):
            raise PaperContractError("graph inference callback lost simulation context")
        result = checked_evaluate(
            self._physical_simulator,
            torch.tensor(theta, device=self.device, dtype=self.dtype).contiguous(),
            device=self.device,
            dtype=self.dtype,
            context=_restore_context(context_document),
        )
        return result.values.detach().cpu().tolist()

    def evaluate_assessment(
        self,
        target: Mapping[str, Any],
        theta: Sequence[float],
        assessment_context: Mapping[str, Any],
    ) -> AssessmentEvaluationV3:
        context = SimulationContext(
            str(assessment_context["family_identity"]),
            "paired_held_out_assessment",
            int(target["target_index"]),
            int(assessment_context["replicate"]),
            int(assessment_context["evaluation_seed"]),
        )
        result = checked_evaluate(
            self._physical_simulator,
            torch.tensor(theta, device=self.device, dtype=self.dtype).contiguous(),
            device=self.device,
            dtype=self.dtype,
            context=context,
        )
        evidence = _assessment_randomness_evidence(assessment_context, result.metadata)
        return AssessmentEvaluationV3(
            tuple(float(value) for value in result.values.detach().cpu().tolist()),
            evidence["compact"],
            evidence["raw"],
        )
