# Historical replay, recovery and held-out assessment

These workflows interpret specific recorded versions of the graph comparison.
Their inputs include source commits, native Git blobs, archived fit records,
frozen targets, decisions and signed-off run declarations. Outputs range from
a saved-data inventory to a new evaluation. Choose a workflow by its evidence
and computation requirements.

| Task | Entry point | Scientific boundary |
| --- | --- | --- |
| Inspect terminal lineage | [terminal_lineage_forensics_cli.py](terminal_lineage_forensics_cli.py) | Checks the declared historical sources and reports which target, terminal policy and assessment generated a stored error. |
| Inventory and recover saved evidence | [repaired_sbi_salvage.py](repaired_sbi_salvage.py), [repaired_sbi_recovery.py](repaired_sbi_recovery.py) | Pair each target and budget with retained fit state and manifest information. A recovery manifest qualifies only the fields it verified. |
| Prepare corrected recommendations | [repaired_sbi_terminal_preparation.py](repaired_sbi_terminal_preparation.py) | Binds the native and corrected sources. Its manifest hashes the [bootstrap](../common/exact_terminal_bootstrap.py) and [generator](../common/exact_terminal_generation.py) at those exact paths. Executing the generator runs numerical optimization. |
| Reconstruct a historical GP | [exact_source_bootstrap.py](exact_source_bootstrap.py), [single-record script](../common/exact_model_replay.py), [batch script](../common/exact_model_replay_batch.py) | Isolates imports to a selected worktree. The batch script imports `exact_model_replay` as a standalone sibling. These two source files stay together at their native paths; no checkout-prepending shim may be interposed in that historical interpreter. |
| Compare terminal policies | [terminal_domain_ablation.py](terminal_domain_ablation.py), [v1](../common/terminal_domain_heldout_assessment.py), [v2](../common/terminal_domain_heldout_assessment_v2.py) | A refit or assessment can perform GP computation or real physical evaluations. Each held-out declaration records its own source path and hash; both original versions are kept. |
| Assess repaired SBI results | [preparation](repaired_sbi_assessment_preparation.py), [execution](repaired_sbi_assessment_execution.py), [archiving](repaired_sbi_assessment_archive_cli.py) | Preparation freezes assessment identity, execution can invoke physics, and archiving summarizes saved outcomes. This preparation also serves later comparisons. |
| Verify a candidate sequence | [strict replay](../exploration/scalar_primary_replay.py), [divergence capture](../exploration/replay_divergence.py) | Fits GPs and checks proposed candidates before a saved response may be returned. The qualified harness loads these two files by their original filesystem paths and overrides their source root. |

The salvage, B100 recovery, terminal preparation, corrected-terminal archive
and terminal-lineage producers retain their native source files under `common/`.
Reproducing their entire packages requires the original B40 worker records,
B100 recovery captures or private corrected-terminal attempts. The assessment
archive CLI similarly consumes its raw worker compact export, and the scalar
B100 archive CLI validates an original worker export and inventory. The new
addresses expose those same producers; their stored reports and known hashes
can be inspected without retrieving private records.

The [legacy notebook](plot_paper120_target_v2.py) retains its original,
machine-specific execution assumptions and can evaluate the physical model.
The [Linux capture notes](../../../docs/history/linux_reference_capture.md)
record the environment of a past investigation. Inspect [stored results](../../../results/)
or [offline verification](../../../reproducibility/) before choosing a worker.
The former `experiments.paper.common` module addresses remain usable by
historical commands and serialized records.


## Verify the original terminal-domain declarations

The v1 and v2 generators hash their own source, tests, simulator and protocol,
and store their native paths in the declaration. The original jobs recorded a
dirty worktree. Commit `3c74fba15e5d6a22ba8e202df032ba3180510f0a`
contains every source byte listed in both declarations. Run this offline check
from a full Git checkout with the qualified environment:

```sh
infrastructure/local/run_pytest.sh -q tests/experiments/test_experiment_owner_compatibility.py -k original_240_paired_tasks
```

The test opens a detached clone outside the current checkout, restores the B40
compact export from its SHA-256-verified archive and runs the unpatched
preparation in a separate Python process. It checks every recorded source
hash and compares all 240 ordered tasks per version against the sealed JSONL.
No GP fit or physical assessment runs. Other historical worker commands require
their original private inputs and their own authorization.
