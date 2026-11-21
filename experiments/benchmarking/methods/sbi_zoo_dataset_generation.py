"""Future capped shared-dataset generation with no simulator dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from mutaco import SeedCoordinate, SeedDeriver, stable_hash

from .sbi_zoo_accounting import WriteAheadScientificCallLedger
from .sbi_zoo_contract import PRIOR_LAW, RESPONSE_DIMENSION
from .sbi_zoo_dataset import (
    SharedTrainingDataset,
    canonical_vector_sha256,
    load_slot_outcome_artifact,
    response_ordering_identity,
    write_shared_training_dataset,
    write_slot_outcome_artifact,
)
from .sbi_zoo_failures import DatasetIntegrityError


def generate_shared_training_dataset(
    output: Path,
    *,
    target_record_id: str,
    budget: int,
    deriver: SeedDeriver,
    response_names: Sequence[str],
    simulator_identity: str,
    simulator_configuration_identity: str,
    dataset_generation_declaration_identity: str,
    ledger: WriteAheadScientificCallLedger,
    evaluate: Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor],
    resume_unattempted: bool = False,
) -> SharedTrainingDataset:
    """Generate one dataset through an injected evaluator under write-ahead control.

    This module neither imports nor constructs the physical graph simulator. The
    separately authorized worker will inject that evaluator. Tests inject fakes.
    """

    if ledger.maximum_calls != budget:
        raise DatasetIntegrityError("dataset ledger cap must equal its scientific budget")
    theta, theta_seed = PRIOR_LAW.sample(
        budget,
        deriver=deriver,
        target_record_id=target_record_id,
        budget=budget,
    )
    context_identities: list[str] = []
    ordering_identity = response_ordering_identity(response_names)
    for row_index in range(budget):
        evaluation_seed = deriver.seed_for(
            SeedCoordinate(
                "sbi_zoo_shared_dataset_simulation_row",
                (target_record_id, budget, row_index),
            )
        )
        context_body = {
            "schema_version": "mutaco-sbi-zoo-training-context-v1",
            "target_record_id": target_record_id,
            "budget": budget,
            "row_index": row_index,
            "evaluation_seed": evaluation_seed,
            "purpose": "sbi_zoo_shared_prior_predictive_training",
        }
        context_identity = stable_hash("sbi-zoo-training-context-v1", context_body)
        context = {**context_body, "context_identity": context_identity}
        context_identities.append(context_identity)
        row = theta[row_index].contiguous()
        row_sha = canonical_vector_sha256(row, length=4, field="dataset theta row")
        slot = f"row-{row_index:03d}"
        slot_body = {
            "dataset_generation_declaration_identity": dataset_generation_declaration_identity,
            "call_ledger_identity": ledger.identity,
            "target_record_id": target_record_id,
            "budget": budget,
            "slot": slot,
            "row_index": row_index,
            "theta_canonical_sha256": row_sha,
            "stochastic_context_identity": context_identity,
        }
        scientific_slot_identity = stable_hash("sbi-zoo-scientific-slot-v1", slot_body)
        call_identity = {
            "dataset_generation_declaration_identity": dataset_generation_declaration_identity,
            "call_ledger_identity": ledger.identity,
            "target_record_id": target_record_id,
            "budget": budget,
            "scientific_slot_identity": scientific_slot_identity,
            "row_index": row_index,
            "theta_canonical_sha256": row_sha,
            "stochastic_context_identity": context_identity,
            "response_ordering_identity": ordering_identity,
            "simulator_identity": simulator_identity,
            "simulator_configuration_identity": simulator_configuration_identity,
        }

        def invoke(row: torch.Tensor = row, context: Mapping[str, Any] = context) -> torch.Tensor:
            return evaluate(row, context)

        def persist_outcome(
            value: Any,
            call_identity: Mapping[str, Any] = call_identity,
            slot: str = slot,
        ) -> Mapping[str, Any]:
            if not isinstance(value, torch.Tensor):
                raise DatasetIntegrityError("dataset evaluator must return a tensor")
            checked = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if checked.shape != (RESPONSE_DIMENSION,) or not bool(torch.isfinite(checked).all()):
                raise DatasetIntegrityError("dataset response must be one finite 20-vector")
            if dict(ledger.intent_payload(slot)) != dict(call_identity):
                raise DatasetIntegrityError(
                    "slot outcome call identity disagrees with write-ahead intent"
                )
            artifact = write_slot_outcome_artifact(
                ledger.root / "outcomes" / slot,
                response=checked,
                call_identity=call_identity,
            )
            return artifact.receipt()

        status = ledger.slot_status(slot)
        if resume_unattempted and status == "success":
            if dict(ledger.intent_payload(slot)) != call_identity:
                raise DatasetIntegrityError("successful restart slot intent mismatch")
            continue
        invocation = (
            ledger.invoke_unattempted_once if resume_unattempted else ledger.invoke_once
        )
        response = invocation(
            slot, intent=call_identity, invoke=invoke, persist_outcome=persist_outcome
        )
        del response
    response_matrix = _assemble_slot_responses(
        ledger=ledger,
        target_record_id=target_record_id,
        budget=budget,
        theta=theta,
        context_identities=context_identities,
        response_ordering_identity_value=ordering_identity,
        dataset_generation_declaration_identity=dataset_generation_declaration_identity,
        simulator_identity=simulator_identity,
        simulator_configuration_identity=simulator_configuration_identity,
    )
    lower, upper = PRIOR_LAW.bounds()
    metadata = {
        "target_record_id": target_record_id,
        "budget": budget,
        "parameter_names": list(PRIOR_LAW.parameter_names),
        "response_names": list(response_names),
        "parameter_bounds": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "simulator_identity": simulator_identity,
        "simulator_configuration_identity": simulator_configuration_identity,
        "row_context_identities": context_identities,
        "dataset_generation_declaration_identity": dataset_generation_declaration_identity,
        "scientific_call_ledger_identity": ledger.identity,
        "theta_law": PRIOR_LAW.document(),
        "theta_seed": theta_seed,
    }
    return write_shared_training_dataset(
        output,
        theta=theta,
        response=response_matrix,
        metadata=metadata,
    )


def _assemble_slot_responses(
    *,
    ledger: WriteAheadScientificCallLedger,
    target_record_id: str,
    budget: int,
    theta: torch.Tensor,
    context_identities: Sequence[str],
    response_ordering_identity_value: str,
    dataset_generation_declaration_identity: str,
    simulator_identity: str,
    simulator_configuration_identity: str,
) -> torch.Tensor:
    """Reconstruct row order only from validated successful scientific slots."""

    outcome_root = ledger.root / "outcomes"
    expected_slots = {f"row-{index:03d}" for index in range(budget)}
    actual_slots = (
        {item.name for item in outcome_root.iterdir() if item.is_dir()}
        if outcome_root.is_dir()
        else set()
    )
    if actual_slots != expected_slots:
        raise DatasetIntegrityError("slot outcome set has missing, duplicate, or extra rows")
    responses: list[torch.Tensor] = []
    seen_scientific_slots: set[str] = set()
    for row_index in range(budget):
        slot = f"row-{row_index:03d}"
        if ledger.slot_status(slot) != "success":
            raise DatasetIntegrityError("dataset assembly requires durable success for every row")
        row_sha = canonical_vector_sha256(
            theta[row_index].contiguous(), length=4, field="dataset theta row"
        )
        slot_body = {
            "dataset_generation_declaration_identity": dataset_generation_declaration_identity,
            "call_ledger_identity": ledger.identity,
            "target_record_id": target_record_id,
            "budget": budget,
            "slot": slot,
            "row_index": row_index,
            "theta_canonical_sha256": row_sha,
            "stochastic_context_identity": context_identities[row_index],
        }
        scientific_slot_identity = stable_hash("sbi-zoo-scientific-slot-v1", slot_body)
        if scientific_slot_identity in seen_scientific_slots:
            raise DatasetIntegrityError("duplicate scientific slot identity")
        seen_scientific_slots.add(scientific_slot_identity)
        call_identity = {
            "dataset_generation_declaration_identity": dataset_generation_declaration_identity,
            "call_ledger_identity": ledger.identity,
            "target_record_id": target_record_id,
            "budget": budget,
            "scientific_slot_identity": scientific_slot_identity,
            "row_index": row_index,
            "theta_canonical_sha256": row_sha,
            "stochastic_context_identity": context_identities[row_index],
            "response_ordering_identity": response_ordering_identity_value,
            "simulator_identity": simulator_identity,
            "simulator_configuration_identity": simulator_configuration_identity,
        }
        artifact = load_slot_outcome_artifact(
            outcome_root / slot, expected_call_identity=call_identity
        )
        if dict(ledger.intent_payload(slot)) != call_identity:
            raise DatasetIntegrityError("slot outcome does not match its write-ahead intent")
        if dict(ledger.terminal_payload(slot) or {}) != artifact.receipt():
            raise DatasetIntegrityError("ledger success does not bind the slot outcome artifact")
        responses.append(artifact.response)
    return torch.stack(responses).contiguous()
