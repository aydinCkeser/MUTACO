"""Independent pre-migration contracts: no inference or physical execution."""
from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import pickle

import pytest

# This was a supported import order before the circular-import repair.
from experiments.benchmarking.methods import direct_posterior_decision
from experiments.benchmarking.methods import sbi_zoo_failures
from experiments.paper.common import protocol_v3 as protocol
from experiments.paper.common.validation import PaperContractError


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/shared_contracts_before_2d.json"


def capture_contracts() -> dict:
    counts = protocol.PhysicalCallCounts(5, 3, 2, 1)
    decision = protocol.DecisionOutcome(
        method="mutaco", status="recommended", decision_policy_identifier="fixed-policy",
        method_configuration_identity="fixed-method", inference_budget=5,
        inference_calls=counts, recommended_theta=(0.25, -0.0),
        decision_metadata={"ordered": [3, 1]},
    )
    config = protocol.PaperComparisonConfigV3(
        simulator_definition_identity="fixed-definition", simulator_configuration_identity="fixed-configuration",
        assessment_fidelity="fixed-fidelity", assessment_campaign_seed=7, inference_campaign_seed=11,
    )
    persisted = []
    batch = protocol.freeze_decisions(config, "fixed-target", [decision], persist=persisted.append)
    values = {
        "counts_class": protocol.PhysicalCallCounts, "error_class": PaperContractError,
        "counts": counts, "decision": decision, "batch": batch,
        "error": PaperContractError("fixed validation failure"),
        "scoring_error": sbi_zoo_failures.SbiZooScoringContractError("fixed scoring failure"),
    }
    errors = []
    for args in ((True, 0), (-1, 0), (2, 1), (1, 1, 0, 2), (1.0, 1)):
        try:
            protocol.PhysicalCallCounts(*args)
        except Exception as error:
            errors.append({"args": args, "type": type(error).__name__, "message": str(error)})
    return json.loads(json.dumps({
        "counts": counts.document(), "decision": asdict(decision),
        "config": config.document(), "freeze": persisted, "freeze_identity": batch.freeze_identity,
        "assessment_context": protocol.paired_assessment_context(config, "fixed-target"),
        "inference_context": protocol.method_inference_context(config, "fixed-target", "mutaco", "fixed-method"),
        "errors": errors,
        "mro": [[cls.__module__, cls.__qualname__] for cls in sbi_zoo_failures.SbiZooScoringContractError.__mro__],
        "pickles": {name: {str(p): pickle.dumps(value, protocol=p).hex() for p in (2, 4, 5)}
                    for name, value in values.items()},
        "exports": {name: list(importlib.import_module(name).__all__)
                    for name in ("experiments.paper", "experiments.paper.common")},
    }))


def test_exact_pre_migration_schemas_values_errors_and_pickle_bytes() -> None:
    actual = capture_contracts()
    expected = json.loads(FIXTURE.read_text())["snapshot"]
    from infrastructure.truba.b100_regression import first_difference
    assert first_difference(expected, actual) is None


@pytest.mark.parametrize("protocol_number", (2, 4, 5))
def test_independent_historical_pickles_resolve_to_identical_classes(protocol_number) -> None:
    saved = json.loads(FIXTURE.read_text())["snapshot"]["pickles"]
    restored = {name: pickle.loads(bytes.fromhex(versions[str(protocol_number)]))
                for name, versions in saved.items()}
    assert restored["counts_class"] is protocol.PhysicalCallCounts
    assert restored["error_class"] is PaperContractError
    assert type(restored["counts"]) is protocol.PhysicalCallCounts
    assert type(restored["decision"]) is protocol.DecisionOutcome
    assert type(restored["error"]) is PaperContractError
    assert type(restored["scoring_error"]) is sbi_zoo_failures.SbiZooScoringContractError
    assert restored["batch"].document()["freeze_identity"] == json.loads(FIXTURE.read_text())["snapshot"]["freeze_identity"]


def test_metadata_is_owned_and_counts_are_frozen() -> None:
    metadata = {"nested": [1]}
    decision = protocol.DecisionOutcome("mutaco", "recommended", "policy", "method", 0,
                                        protocol.ZERO_CALLS, (1.0,), decision_metadata=metadata)
    metadata["nested"].append(2)
    assert decision.decision_metadata == {"nested": [1]}
    with pytest.raises(AttributeError):
        decision.inference_calls.attempted = 1
    assert direct_posterior_decision.PhysicalCallCounts is protocol.PhysicalCallCounts
    assert direct_posterior_decision.DecisionOutcome is protocol.DecisionOutcome


def test_shared_definition_statements_match_pre_migration_AST() -> None:
    saved = json.loads(FIXTURE.read_text())["definition_AST_sha256"]
    # The defining location may change; the class statements must not.
    candidates = [ROOT / "experiments/paper/common/protocol_v3.py",
                  ROOT / "experiments/paper/common/validation.py"]
    shared = ROOT / "experiments/contracts/__init__.py"
    if shared.exists():
        candidates.append(shared)
    actual = {}
    for path in candidates:
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name in saved:
                assert node.name not in actual
                actual[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert actual == saved
