# Paper comparison protocol

Start with [campaign.py](campaign.py) to read a campaign declaration and its
separately approved execution selection. The declaration fixes the target
family, method choices and budgets. Worker and simulator implementations live
with their experiments. The protocol records when a decision becomes frozen,
which calls belong to its inference budget and how its held-out assessment is
paired with another method.

| Stage | Implementation | What the stage owns |
| --- | --- | --- |
| Specify and freeze the cohort | [cohort_v2.py](cohort_v2.py), [target_freeze_v2.py](target_freeze_v2.py), [supplied_targets.py](supplied_targets.py) | Ordered target identities and declarations, previews, optional target evaluations and append-only saved target records. |
| Declare methods and evaluate a population | [method_specs.py](method_specs.py), [population_v3](../common/population_v3.py) | Method-specific configuration paths and seeds, target selection, worker shard bounds, callback budgets and collection of outcomes. |
| Freeze decisions and pair assessments | [protocol_v3](../common/protocol_v3.py) | Method-addressed inference contexts, frozen decisions, common assessment contexts, typed failures and call counts. |
| Commit and check receipts | [method_append.py](method_append.py), [compact_append.py](compact_append.py), [integrity.py](integrity.py), [validation.py](validation.py) | Hash and schema checks, append-only method records, precise failure and recovery status. |

The installed core has no dependency on these paper protocols.
[experiments/contracts](../../contracts/README.md) defines the shared counting
value and validation exception. The cross-experiment vocabulary still lives
in [cross_experiment_contract.py](../common/cross_experiment_contract.py).
Both module addresses for the population and v3 decision protocol share their
original implementations. The target-98 response-archive builder hashes
`common/population_v3.py` and `common/protocol_v3.py` by literal path.
The
validation source lives here and loads under its historical module name
through a relative file link. This preserves the exact module-name object on
`PaperContractError` and its historical pickles. Other historical
`experiments.paper.common` imports also remain supported. Merely
importing the protocol never starts a campaign; execution through campaign,
freeze or population workers can use real evaluations.
