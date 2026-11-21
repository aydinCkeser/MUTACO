# Physical evidence export repair and partial recovery

**The serializer is repaired offline. Milestone 3E is not fully closed:**
222 original tensor leaves were never preserved and cannot be recovered from
summary statistics. No additional physics, GP fitting or Slurm job ran.

Original job **6925432 remains FAILED, exit 1:0**. Its completed scientific
checks and [original record](../physical_integration_qualification_v1/README.md)
are unchanged. At `physical_worker.py:176`, `torch.save` called
`torch.serialization._save` / `pickler.dump(obj)` and rejected
`raw.graph_full_reference.metadata.seed_manifest`, a nested `mappingproxy`.
The outer `dict(metadata)` was only a shallow copy. Line 179 attempted the same
payload as fallback and failed before writing `verdict.json`. Both incomplete
ZIP files contain neither `data.pkl` nor tensor storage entries.

The new [offline exporter](../../../tools/reproducibility/tensor_export.py)
recursively converts mapping containers while retaining tensor objects, values,
dtypes/devices, list/tuple structure and ordering. It requires declared tensor
paths, verifies tensor bits/storage contracts after readback and publishes only
a complete verified file without replacing existing evidence. It raises on
missing evidence or failed serialization/readback. The original sealed worker,
scientific code and read-only verification CLI were not modified.

| Original tensor fields | Recovery |
| --- | --- |
| Analytic noisy/constant outputs, three repeated graph candidate responses, full-resolution graph reference/candidate, retained reference, two GP training datasets and frozen mean/scales | 14 leaves restored from exact retained JSON/fixture values, with shape/dtype/bit checks and both original training identities verified. The separately sealed output is explicitly partial and derived. Original memory/storage identities are not claimed. |
| Both implementations' first-step arrays for 16 cases | 192 leaves unavailable |
| 16 NumPy final responses, 13 other Torch final responses, one computed retained response | 30 leaves unavailable |

The JSON responses, GP diagnostics, contexts, fitting/proposal/recommendation
provenance, seed manifests, metrics and event/call logs survive. Distances and
quantiles do not uniquely determine missing arrays. Recorded reference values
were never substituted for missing computed responses. A request to export the
full 236-leaf contract correctly fails; the verified partial artifact contains
only the 14 available leaves. Thirteen unrecorded parity seed manifests and two
unrecorded GP-config repr strings are also distinguished from retained metadata.

**75 offline tests passed**, including 21 new exporter cases, with zero skips,
failures or scientific-execution attempts. All 55 original private evidence files,
68 existing portable checksum entries, 47 current and 47 historical source hashes,
eight archive members and 21 figure/array files passed integrity checks. New
partial recovery and test records are separately sealed outside Git. See the
[manifest](manifest.json) and [portable checksums](SHA256SUMS).

The [minimum recapture plan](recapture-plan.json) is awaiting explicit approval:
**33 forward kernel evaluations** (16 current Torch, 16 independent NumPy,
one retained Torch), zero simulator API calls, zero GP fits and no repeat of
analytic/full-resolution/integration runs. It uses the original 16 fixed
vectors, named contexts/seeds, N=3/B=1/R=1, two burn-in plus four sampled steps,
and the original three-step retained fixture. Proposed limit: one 20-CPU,
4-GiB, five-minute qualified TRUBA CPU job, preserving 20/1 Torch threads.
The original 33 kernel runs totaled ~0.59 seconds, excluding import, realization
and I/O overhead. No job is submitted without approval. Existing parity limits
and exact comparison to original saved metrics must pass; newly captured arrays
must be labeled new computations rather than recovered original tensors.
