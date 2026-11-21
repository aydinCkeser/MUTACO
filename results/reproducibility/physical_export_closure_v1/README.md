# Qualified supplementary physical-evidence capture

**Milestone 3E's bounded evidence-export task is closed through this supplement.**
The declared 236-leaf tensor contract is complete, with explicit mixed provenance:
14 leaves recovered from retained original records and 222 newly computed leaves.
The original failed dump and its unexported arrays were not recovered or relabelled.

Original job **6925432 remains FAILED, 1:0**. Its successful scientific computations,
[original record](../physical_integration_qualification_v1/README.md), 55-file
private seal and [offline repair/partial recovery](../physical_export_recovery_v1/README.md)
remain unchanged. This supplement follows explicit approval of the previously
recorded 33-kernel plan. It is not a repeat of the end-to-end experiment.

New job **6925576 COMPLETED, 0:0** on qualified TRUBA Linux x86-64 CPU:

- 21 exporter tests passed before any kernel ran, with zero failures/skips or
  forbidden scientific calls.
- Exactly 33 forward runs: 16 current graph Torch, 16 independent NumPy-reference
  and one retained Torch. Zero GP fits, simulator API calls, end-to-end repeats,
  unplanned attempts or automatic resubmissions.
- All 16 fixed graph parity cases and the retained fixture passed unchanged
  comparison limits. Every original per-case/retained metric matched exactly,
  including signed-zero-sensitive float identities. All three previously recorded
  API responses and their seed manifests matched exactly.
- The repaired exporter wrote and read back 226 new parity tensor leaves. Independent
  offline verification checked shapes/dtypes, existing numerical limits, recorded
  metrics and all four overlaps with retained values. The complete 236-leaf supplement
  then passed bitwise serialization/readback verification. No kernel or GP was
  rerun during postprocessing.

The current graph tensors remain float32. The legacy retained candidate remains
float64 against its float32 historical fixture: the source explicitly returns
that dtype. An initial offline check's float32-only assumption was corrected to
those existing contracts, without casting data; both verifier versions/logs are
preserved. Exact equality with unavailable original arrays is not claimed merely
because their comparison metrics match.

Source checkout `ea18d988d2af3c0d289808825e885c4b568435de` has the unchanged
qualified scientific core `35ef82dba2160dfb3f709854b0e11c3956744c31`.
Python 3.11.17, Torch 2.12.1+cpu, BoTorch 0.18.1, GPyTorch 1.15.2 and all other
qualified numerical dependencies were unchanged; Torch threads were 20/1.
The single allocation requested 20 CPUs, 4 GiB and five minutes, no GPU. Elapsed time
was 65 seconds; Slurm reported batch peak RSS 138720 KiB (~0.132 GiB).

The [manifest](manifest.json) binds the repaired exporter, qualified source,
input-plan identity, capture hashes and private seals. New raw packets, composite
artifact, per-tensor shape/dtype/value-hash/provenance inventory and verifier logs
are separately sealed outside Git (61 files). The original 55 files and previous
partial-recovery/check seals were independently verified unchanged. Historical
source hashes, 71 existing portable checksum entries, eight archive members and
21 figure/array files remain intact. [SHA256SUMS](SHA256SUMS) seals these new
portable summaries; it does not replace original private-byte evidence.

Two original config-repr strings were not recreated; original typed settings and
full GP diagnostics/events remain recorded. Full scientific-suite/B40/B100 replay
was unnecessary and was not rerun. SBI campaigns, statistical physics cohorts,
reconstruction studies, other-platform/GPU equivalence and historical TeX rendering
remain outside this bounded qualification. The repository is ready for reviewed
private technical sharing, subject to a separately authorized push. This is not
complete paper reproducibility or external submission approval.
