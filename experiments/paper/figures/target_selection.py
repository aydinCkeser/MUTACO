"""Outcome-independent representative-target selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


POLICY_ID = "historical_reference_nearest_v1"


def select_historical_reference_nearest(
    targets: Sequence[Mapping[str, Any]],
    *,
    parameter_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    historical_reference: Mapping[str, float],
) -> dict[str, Any]:
    """Select solely by normalized theta distance to a historical reference."""

    names = tuple(parameter_names)
    lo, hi = tuple(map(float, lower)), tuple(map(float, upper))
    if not targets or not (len(names) == len(lo) == len(hi)):
        raise ValueError("target selection requires compatible nonempty inputs")
    if set(historical_reference) != set(names):
        raise ValueError("historical reference names do not match modern profile")
    widths = tuple(b - a for a, b in zip(lo, hi))
    if any(width <= 0.0 for width in widths):
        raise ValueError("parameter bounds must have positive width")
    reference = tuple(float(historical_reference[name]) for name in names)
    if any(value < a or value > b for value, a, b in zip(reference, lo, hi)):
        raise ValueError("historical reference lies outside modern bounds")
    normalized_reference = tuple(
        (value - a) / width for value, a, width in zip(reference, lo, widths)
    )
    candidates = []
    for target in targets:
        theta = tuple(float(value) for value in target["theta_star"])
        if len(theta) != len(names):
            raise ValueError("target theta dimension mismatch")
        normalized = tuple(
            (value - a) / width for value, a, width in zip(theta, lo, widths)
        )
        distance = sum(
            (value - reference_value) ** 2
            for value, reference_value in zip(normalized, normalized_reference)
        )
        candidates.append(
            {
                "target_index": int(target["target_index"]),
                "target_record_id": str(target["target_record_id"]),
                "theta_star": list(theta),
                "normalized_theta": list(normalized),
                "normalized_squared_distance": distance,
            }
        )
    candidates.sort(
        key=lambda row: (row["normalized_squared_distance"], row["target_index"])
    )
    selected = candidates[0]
    ties = [
        row["target_index"]
        for row in candidates
        if row["normalized_squared_distance"]
        == selected["normalized_squared_distance"]
    ]
    return {
        "schema_version": "mutaco-paper-representative-target-selection-v1",
        "policy_id": POLICY_ID,
        "selection_uses_method_results": False,
        "formula": "argmin_i ||normalize(theta_i)-normalize(theta_historical)||_2^2",
        "tie_rule": "lowest_target_index_under_exact_tie",
        "parameter_names": list(names),
        "lower": list(lo),
        "upper": list(hi),
        "historical_reference": dict(zip(names, reference)),
        "normalized_historical_reference": list(normalized_reference),
        "selected": selected,
        "exact_tied_target_indices": ties,
        "candidates": candidates,
    }

from experiments._legacy_pickle import preserve_pickle_addresses as _preserve_pickle_addresses

_preserve_pickle_addresses(globals(), "experiments.paper.figures.common.target_selection")
