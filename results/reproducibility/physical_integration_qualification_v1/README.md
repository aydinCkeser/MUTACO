# Bounded real-simulator and integration evidence

Tested checkout `1af825f93ab06a36a679ca08f83fb8a524e3b10b` contains the unchanged
qualified scientific core `35ef82dba2160dfb3f709854b0e11c3956744c31`.
The [predeclared plan](plan.json) and [manifest](manifest.json) record the limits,
fixed inputs, existing comparison rules and actual outcomes.

**All bounded scientific checks passed. The Slurm job failed during subsequent
binary evidence export.** Job 6925432 ended **FAILED, 1:0**, after 108 seconds.
The private capture helper could not pickle nested immutable metadata. Both
incomplete tensor files and the missing final worker verdict are recorded; no
job or physical evaluation was repeated. This is not an entirely successful
execution or a complete raw tensor capture.

The original JSON responses, full GP diagnostics/integration provenance,
seed manifests, events, call records, comparison metrics and error logs survived.
Independent offline verification matched all backend response bits and contexts
to committed observations and reconstructed both exact GP training identities.
Derived training tensors are explicitly labeled as reconstructed from JSON.
Unexported first-step parity tensors cannot be recovered from these files.

| Check | Result under existing reference rules |
| --- | --- |
| Handwritten analytic testbed | Four deterministic vectors including bounds; max absolute difference 2.22e-16 (atol 1e-15). Three noisy calls verify seed replay/isolation; one exact constant channel. This polynomial/trigonometric map is not oscillator physics. |
| Current overdamped graph kernel | All 16 existing fixed small cases passed independent NumPy/Torch evolution and first-step comparisons; largest final/intermediate absolute difference 4.77e-7. Both implementations use the same realized inputs, so this is not independent sampling validation. |
| Current graph API | Three repeated center/boundary cases reproduce kernel responses and seed manifests exactly. |
| Full-resolution frozen graph target 16 | P=4, M=20, N=10, B=80, R=2, dt=0.1, 100 burn-in + 900 sampled steps. Max absolute difference 2.68817e-5, relative L2 4.82331e-6; seed manifest exact. Passes existing final rtol 5e-5, atol 5e-6, relative-L2 limit 5e-5; it is not bitwise historical equality. |
| Retained historical graph oracle | One separate frozen legacy case passed; max absolute difference 4.77e-7. Never treated as the current scientific model. |
| Invalid inputs | Six wrong-shape/nonfinite/out-of-bounds requests rejected before a backend call. |
| Real end-to-end MUTACO | 16 scrambled-Sobol initial + 1 adaptive observation, two 12-iteration GP fits, one proposal and one continuous physical posterior-mean MAE recommendation. Exact internal training/context/response/accounting checks passed; ambient RNG preserved. |

The integration uses the full graph resolution and unchanged transform, GP,
acquisition and duplicate controls from the locked paper configuration. Only the
observation budget is deliberately 17, with its own scientific identity. It is a
supported direct-core diagnostic, not a registered B40/B100 campaign. Its target
is read from the frozen cohort; no extra target-generation/held-out evaluation
occurs. The terminal recommendation adds no GP fit or simulator evaluation.
Both optimizers reached the accepted 12-iteration limit and were accepted by the
existing fit policy; convergence is not claimed.

Planned and actual counts agree: **29 simulator API evaluations** (eight analytic,
four direct graph, 17 integration graph) and **62 forward computations** including
independent references. The latter comprises eight analytic, 37 current graph
Torch kernels, 16 NumPy references and one retained kernel. Graph API calls are
already included in those 37 kernels. No backend failures, unplanned calls or
retries occurred. The allocation was 20 CPUs/20 GiB/20 minutes, CPU only; reported
peak RSS 432268 KiB (~0.412 GiB). The real integration took 39.65 seconds, including
19.82 seconds of simulator work. No dependencies changed or login-node numerical
work occurred.

All 47 versioned core-source entries, eight documentation hashes, 65 pre-existing
portable checksum entries, eight immutable archive members, 21 figure/array files
and all 2542 original tracked files remained intact during execution. Historical hashes and the
[3D full-suite/B40/B100 evidence](../documented_core_qualification_v1/README.md)
retain their original meaning. This evidence-only checkpoint updates the state/index
and adds this record. No full suite or historical replay was rerun here. Newly
calculated GP diagnostics are not archived historical optimizer states.
Private raw evidence, executed worker and independent verifier are sealed outside
Git; their hashes are in the manifest. [SHA256SUMS](SHA256SUMS) seals these derived
shareable records rather than claiming byte identity with raw evidence.

Coverage remains bounded: SBI training, statistical graph cohorts, physical
reconstruction studies, GPU/other-platform equality and historical TeX/figure
regeneration were not tested. One full-resolution response averaged ~1.17 seconds
here; future campaigns need separate budgets including GP/training costs. The
existing current-model statistical parity cohort alone entails 256 forward kernel
runs; its runtime was not measured. SBI requires a separately qualified environment,
and reconstruction requires explicit grid/seed budgets. TeX needs the historical
rendering toolchain. This record supports private technical review with the export
limitation visible; it does not establish complete paper reproducibility or external
submission approval and authorizes no further execution or push.
