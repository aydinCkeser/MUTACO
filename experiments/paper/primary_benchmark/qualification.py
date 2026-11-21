"""Approval-gated NEI numerical qualification declarations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Mapping, Sequence

from mutaco import stable_hash
from experiments.paper.protocol.validation import PaperContractError


NEI_QUALIFICATION_SCHEMA = "mutaco-nei-numerical-qualification-declaration-v1"
NEI_QUALIFICATION_RESULT_SCHEMA = "mutaco-nei-numerical-qualification-result-v1"


@dataclass(frozen=True)
class NEINumericalTier:
    tier_id: str
    n_mc: int
    raw_candidates: int
    local_starts: int
    optimization_steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        if not self.tier_id.strip():
            raise PaperContractError("NEI tier id must be nonempty")
        for name in ("n_mc", "raw_candidates", "local_starts", "optimization_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PaperContractError(f"NEI tier {name} must be positive")
        if self.local_starts > self.raw_candidates:
            raise PaperContractError("NEI tier starts exceed raw candidates")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise PaperContractError("NEI tier learning rate must be positive")

    def document(self) -> dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "n_mc": self.n_mc,
            "raw_candidates": self.raw_candidates,
            "local_starts": self.local_starts,
            "optimization_steps": self.optimization_steps,
            "learning_rate": self.learning_rate,
        }


def nei_qualification_call_preview(target_count: int = 5) -> dict[str, int]:
    if target_count != 5:
        raise PaperContractError("NEI qualification requires exactly five targets")
    return {
        "target_construction_calls": 0,
        "maximum_search_calls": target_count * (40 + 100),
        "maximum_assessment_calls": 0,
        "maximum_total_calls": 700,
        "scientific_calls_executed": 0,
    }


@dataclass(frozen=True)
class NEIQualificationDeclaration:
    campaign_id: str
    study_identity: str
    target_ledger_hash: str
    target_record_identities: tuple[str, ...]
    tiers: tuple[NEINumericalTier, ...]
    snapshot_positions_b40: tuple[int, ...]
    snapshot_positions_b100: tuple[int, ...]
    output_root: str
    trajectory_tier_id: str | None = None
    reference_optimizer: NEINumericalTier = NEINumericalTier(
        "reference", 1024, 512, 16, 48, 0.025
    )
    design_campaign_seed: int = 92020821
    inference_campaign_seed: int = 92020822
    terminal_campaign_seed: int = 92020823
    reference_campaign_seed: int = 92020824

    def __post_init__(self) -> None:
        if any(not str(getattr(self, name)).strip() for name in (
            "campaign_id", "study_identity", "target_ledger_hash", "output_root"
        )):
            raise PaperContractError("NEI qualification identities must be nonempty")
        if len(self.target_record_identities) != 5 or len(set(self.target_record_identities)) != 5:
            raise PaperContractError("NEI qualification requires five unique targets")
        if len(self.tiers) < 2 or len({item.tier_id for item in self.tiers}) != len(self.tiers):
            raise PaperContractError("NEI qualification requires unique increasing tiers")
        trajectory = self.trajectory_tier_id or self.tiers[-1].tier_id
        if trajectory not in {item.tier_id for item in self.tiers}:
            raise PaperContractError("NEI qualification trajectory tier is undeclared")
        object.__setattr__(self, "trajectory_tier_id", trajectory)
        if self.reference_optimizer.tier_id in {item.tier_id for item in self.tiers}:
            raise PaperContractError("NEI reference optimizer id overlaps a candidate tier")
        for name in (
            "design_campaign_seed",
            "inference_campaign_seed",
            "terminal_campaign_seed",
            "reference_campaign_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PaperContractError(f"NEI qualification {name} is invalid")
        for positions, budget in (
            (self.snapshot_positions_b40, 40),
            (self.snapshot_positions_b100, 100),
        ):
            if (
                tuple(sorted(set(positions))) != positions
                or any(value < 16 or value > budget for value in positions)
            ):
                raise PaperContractError("NEI snapshot positions are invalid")

    def document(self) -> dict[str, Any]:
        body = {
            "schema_version": NEI_QUALIFICATION_SCHEMA,
            "state": "unexecuted",
            "execution_requires_approval": True,
            "campaign_id": self.campaign_id,
            "study_identity": self.study_identity,
            "target_ledger_hash": self.target_ledger_hash,
            "target_record_identities": list(self.target_record_identities),
            "budgets": [40, 100],
            "methods": ["nei_bo"],
            "heldout_assessment": False,
            "tiers": [item.document() for item in self.tiers],
            "trajectory_tier_id": self.trajectory_tier_id,
            "reference_optimizer": self.reference_optimizer.document(),
            "campaign_seeds": {
                "design": self.design_campaign_seed,
                "inference": self.inference_campaign_seed,
                "terminal": self.terminal_campaign_seed,
                "reference": self.reference_campaign_seed,
            },
            "snapshot_positions": {
                "40": list(self.snapshot_positions_b40),
                "100": list(self.snapshot_positions_b100),
            },
            "reference_evaluation": {
                "batches": 4,
                "samples_per_batch": 4096,
                "common_base_samples_within_batch": True,
                "minimum_optimizer_success_rate": 0.95,
                "maximum_median_normalized_regret": 0.01,
                "maximum_p90_normalized_regret": 0.05,
            },
            "output_root": self.output_root,
            "call_preview": nei_qualification_call_preview(),
        }
        return {
            **body,
            "declaration_identity": stable_hash(
                "nei-numerical-qualification-declaration-v1", body
            ),
        }


def load_nei_qualification_declaration(
    document: Mapping[str, Any],
) -> NEIQualificationDeclaration:
    if document.get("schema_version") != NEI_QUALIFICATION_SCHEMA:
        raise PaperContractError("unsupported NEI qualification schema")
    if document.get("state") != "unexecuted" or document.get("execution_requires_approval") is not True:
        raise PaperContractError("NEI qualification must remain approval-gated")
    raw_tiers = document.get("tiers")
    snapshots = document.get("snapshot_positions")
    reference = document.get("reference_optimizer")
    if (
        not isinstance(raw_tiers, list)
        or not isinstance(snapshots, Mapping)
        or not isinstance(reference, Mapping)
    ):
        raise PaperContractError("NEI qualification tiers or snapshots are missing")
    seeds = document.get("campaign_seeds")
    if not isinstance(seeds, Mapping):
        raise PaperContractError("NEI qualification campaign seeds are missing")
    declaration = NEIQualificationDeclaration(
        campaign_id=str(document.get("campaign_id", "")),
        study_identity=str(document.get("study_identity", "")),
        target_ledger_hash=str(document.get("target_ledger_hash", "")),
        target_record_identities=tuple(str(item) for item in document.get("target_record_identities", ())),
        tiers=tuple(
            NEINumericalTier(
                tier_id=str(item.get("tier_id", "")),
                n_mc=item.get("n_mc"),
                raw_candidates=item.get("raw_candidates"),
                local_starts=item.get("local_starts"),
                optimization_steps=item.get("optimization_steps"),
                learning_rate=item.get("learning_rate"),
            )
            for item in raw_tiers
            if isinstance(item, Mapping)
        ),
        snapshot_positions_b40=tuple(snapshots.get("40", ())),
        snapshot_positions_b100=tuple(snapshots.get("100", ())),
        output_root=str(document.get("output_root", "")),
        trajectory_tier_id=str(document.get("trajectory_tier_id", "")),
        reference_optimizer=NEINumericalTier(
            tier_id=str(reference.get("tier_id", "")),
            n_mc=reference.get("n_mc"),
            raw_candidates=reference.get("raw_candidates"),
            local_starts=reference.get("local_starts"),
            optimization_steps=reference.get("optimization_steps"),
            learning_rate=reference.get("learning_rate"),
        ),
        design_campaign_seed=seeds.get("design"),
        inference_campaign_seed=seeds.get("inference"),
        terminal_campaign_seed=seeds.get("terminal"),
        reference_campaign_seed=seeds.get("reference"),
    )
    if declaration.document() != dict(document):
        raise PaperContractError("NEI qualification declaration is not canonical")
    return declaration


def select_nei_numerical_tier(
    declaration: NEIQualificationDeclaration,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the lowest declared tier satisfying the frozen Q1 rule."""

    expected_snapshots = len(declaration.target_record_identities) * (
        len(declaration.snapshot_positions_b40)
        + len(declaration.snapshot_positions_b100)
    )
    by_tier: dict[str, list[Mapping[str, Any]]] = {
        tier.tier_id: [] for tier in declaration.tiers
    }
    seen: set[tuple[str, int, int, str]] = set()
    for row in observations:
        tier_id = str(row.get("tier_id", ""))
        if tier_id not in by_tier:
            raise PaperContractError("qualification observation uses undeclared tier")
        key = (
            str(row.get("target_record_identity", "")),
            int(row.get("budget", -1)),
            int(row.get("snapshot_position", -1)),
            tier_id,
        )
        positions = (
            declaration.snapshot_positions_b40
            if key[1] == 40
            else declaration.snapshot_positions_b100
            if key[1] == 100
            else ()
        )
        if key[0] not in declaration.target_record_identities or key[2] not in positions:
            raise PaperContractError("qualification observation slot is undeclared")
        if key in seen:
            raise PaperContractError("qualification observation slot is duplicated")
        regret = row.get("normalized_regret")
        if (
            isinstance(regret, bool)
            or not isinstance(regret, (int, float))
            or not math.isfinite(float(regret))
            or float(regret) < 0
        ):
            raise PaperContractError("qualification regret must be finite and nonnegative")
        if not isinstance(row.get("optimizer_success"), bool):
            raise PaperContractError("qualification optimizer status must be boolean")
        flags = row.get("failure_flags", [])
        if not isinstance(flags, list) or any(not isinstance(item, str) for item in flags):
            raise PaperContractError("qualification failure flags must be strings")
        seen.add(key)
        by_tier[tier_id].append(row)
    if any(len(rows) != expected_snapshots for rows in by_tier.values()):
        raise PaperContractError("qualification observations do not cover every tier/snapshot")

    summaries = []
    selected: str | None = None
    for tier in declaration.tiers:
        rows = by_tier[tier.tier_id]
        regrets = sorted(float(row["normalized_regret"]) for row in rows)
        p90_index = max(0, math.ceil(0.9 * len(regrets)) - 1)
        success = sum(bool(row["optimizer_success"]) for row in rows) / len(rows)
        systematic_failure = any(bool(row.get("failure_flags")) for row in rows)
        qualifies = (
            success >= 0.95
            and median(regrets) <= 0.01
            and regrets[p90_index] <= 0.05
            and not systematic_failure
        )
        summaries.append({
            "tier_id": tier.tier_id,
            "observation_count": len(rows),
            "optimizer_success_rate": success,
            "median_normalized_regret": median(regrets),
            "p90_normalized_regret": regrets[p90_index],
            "systematic_failure": systematic_failure,
            "qualifies": qualifies,
        })
        if qualifies and selected is None:
            selected = tier.tier_id
    body = {
        "schema_version": NEI_QUALIFICATION_RESULT_SCHEMA,
        "declaration_identity": declaration.document()["declaration_identity"],
        "status": "qualified" if selected is not None else "no_tier_qualified",
        "selected_tier_id": selected,
        "tier_summaries": summaries,
        "scientific_calls_added_by_selection": 0,
    }
    return {
        **body,
        "result_identity": stable_hash("nei-numerical-qualification-result-v1", body),
    }
