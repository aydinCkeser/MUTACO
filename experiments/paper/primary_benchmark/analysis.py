"""Deterministic analysis-table reconstruction from primary records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.paper.protocol.validation import PaperContractError
from .protocol import PRIMARY_METHODS, PRIMARY_RECORD_SCHEMA
from .scalar_protocol import SCALAR_PRIMARY_METHODS, SCALAR_PRIMARY_RECORD_SCHEMA


def _methods(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    schemas = {row.get("schema_version") for row in records}
    if schemas == {PRIMARY_RECORD_SCHEMA}:
        return PRIMARY_METHODS
    if schemas == {SCALAR_PRIMARY_RECORD_SCHEMA}:
        return SCALAR_PRIMARY_METHODS
    raise PaperContractError("primary analysis record schema mismatch")


def _ordered(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(records)
    identities = tuple(str(row.get("record_identity", "")) for row in rows)
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise PaperContractError("primary analysis records require unique identities")
    _methods(rows)
    return tuple(
        sorted(rows, key=lambda row: (str(row["target_id"]), int(row["budget"]), str(row["method"])))
    )


def endpoint_results(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result = []
    for row in _ordered(records):
        decision = row["decision"]
        timing = decision["timing"]
        result.append(
            {
                "target_id": row["target_id"],
                "budget": row["budget"],
                "method": row["method"],
                "theta_hat": decision["recommended_theta"],
                "heldout_mae": row["heldout_physical_mae"],
                "search_wall_time": timing.get("total"),
                "simulator_wall_time": timing.get("simulator"),
                "non_simulator_overhead": (
                    None
                    if timing.get("total") is None or timing.get("simulator") is None
                    else float(timing["total"]) - float(timing["simulator"])
                ),
                "status": row["status"],
            }
        )
    return tuple(result)

def trajectory_results(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result = []
    for row in _ordered(records):
        for item in row["decision"]["trajectory"]:
            result.append(
                {
                    "target_id": row["target_id"],
                    "budget": row["budget"],
                    "method": row["method"],
                    "call": item["evaluation_index"],
                    "observed_mae": item["observed_physical_mae"],
                    "best_observed_mae_so_far": item[
                        "best_observed_mae_so_far"
                    ],
                }
            )
    return tuple(result)


def paired_endpoint_differences(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    method_order = _methods(records)
    groups: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in _ordered(records):
        groups.setdefault((str(row["target_id"]), int(row["budget"])), {})[
            str(row["method"])
        ] = row
    result = []
    for (target_id, budget), methods in sorted(groups.items()):
        if tuple(sorted(methods)) != tuple(sorted(method_order)):
            raise PaperContractError("paired endpoint group is method-incomplete")
        errors = {
            method: methods[method]["heldout_physical_mae"] for method in method_order
        }
        row = {
            "target_id": target_id,
            "budget": budget,
            "MUTACO_minus_Sobol": (
                None
                if errors["mutaco"] is None or errors["sobol"] is None
                else errors["mutaco"] - errors["sobol"]
            ),
            "completion": {
                method: methods[method]["status"] for method in method_order
            },
        }
        if method_order == PRIMARY_METHODS:
            row.update({
                "MUTACO_minus_NEI": (
                    None
                    if errors["mutaco"] is None or errors["nei_bo"] is None
                    else errors["mutaco"] - errors["nei_bo"]
                ),
                "NEI_minus_Sobol": (
                    None
                    if errors["nei_bo"] is None or errors["sobol"] is None
                    else errors["nei_bo"] - errors["sobol"]
                ),
            })
        else:
            row.update({
                "MUTACO_minus_scalar_qLogNEI": (
                    None
                    if errors["mutaco"] is None
                    or errors["scalar_qlognei"] is None
                    else errors["mutaco"] - errors["scalar_qlognei"]
                ),
                "scalar_qLogNEI_minus_Sobol": (
                    None
                    if errors["scalar_qlognei"] is None
                    or errors["sobol"] is None
                    else errors["scalar_qlognei"] - errors["sobol"]
                ),
            })
        result.append(row)
    return tuple(result)


def runtime_results(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result = []
    for row in _ordered(records):
        timing = row["decision"]["timing"]
        result.append(
            {
                "target_id": row["target_id"],
                "budget": row["budget"],
                "method": row["method"],
                "simulator_time": timing.get("simulator"),
                "gp_fit_time": timing.get("gp_fit"),
                "acquisition_time": timing.get("acquisition"),
                "terminal_optimization_time": timing.get("terminal"),
                "other_decision_time": timing.get("other"),
                "total_time": timing.get("total"),
            }
        )
    return tuple(result)
