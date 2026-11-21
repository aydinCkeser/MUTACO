# Qualified documented scientific core

Scientific source `35ef82dba2160dfb3f709854b0e11c3956744c31` passed the established TRUBA
Linux x86-64 CPU workflow. The following evidence-only commit does not change
that tested scientific source.

- All 531 accepted test identities: **523 passed, eight unchanged skips, zero
  failures/errors/deselections** under the corrected-development v2 test gate.
- Strict B40 target 16 (`/records/48`): **40/40** observations, 25 fit/update
  records, 24 proposals and one terminal recommendation.
- Strict B100 target 0 (`/records/0`): **100/100** observations, 85 fit/update
  records, 84 proposals and one terminal recommendation.
- Full histories, scientific checkpoints, environment fingerprints, seeds/RNG,
  contexts, duplicate recovery, terminal outputs and accounting met the unchanged
  exact contract. All 50/170 fit-progress events agreed. No candidate substitution,
  tolerance relaxation, baseline rerun or reference regeneration occurred.
- Job 6925290: **COMPLETED**, exit **0:0**; one 20-CPU, 20-GiB,
  20-minute allocation, 416 seconds elapsed, about
  11.63 GiB reported peak RSS, no GPUs.
- **Zero physical simulator attempts/calls.** No numerical dependencies changed
  and no numerical workload ran on the login node.

The [manifest](manifest.json) binds all 47 current core-source hashes to the
qualified commit. The [3C documentation record](../scientific_core_documentation_v1/manifest.json),
previous source `4f56343` and its [original qualification](../final_qualified_validation_v1/README.md)
remain unchanged. Independent AST/compiled-operation and 194-symbol interface
comparisons reconfirmed that the eight byte changes are documentation only.
Original hashes remain attached to their original source; no old evidence was
relabelled as this run.

All 63 pre-existing portable checksum entries, eight archive members and 21
original figure/array files were verified. Private raw captures and source delta
were sealed outside Git and independently compared after transfer; the manifest
records their inventory checksum. [SHA256SUMS](SHA256SUMS) seals these portable
derivatives, not the original raw packets. Test-identity hashes reuse the
[unchanged accepted inventory](../final_qualified_validation_v1/test-selection.json).

This qualifies the documented core for reviewed private technical sharing.
It does not authorize a push or external release. SBI workflows, physical-kernel
correctness, GPU/other-platform equivalence, independent Linux/Windows installs
and historical TeX rendering remain outside this gate. GP reference states and
event expectations are reconstructed rather than original historical iterates.
Existing manuscript/MAP wording, terminal-policy and GP diagnostic questions
remain for separate researcher review.
