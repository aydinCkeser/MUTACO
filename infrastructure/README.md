# Execution infrastructure

- [local](local/README.md): pinned Windows/macOS launchers, external pytest
  temporary storage, environment qualification and the guarded scientific/import
  selection. Start here for local checks.
- [truba](truba/README.md): exact-source preparation, transport and zero-call
  validation through the tracked lifecycle command. It does not submit a job.
  Scheduler submission and simulator-bearing work require separate authorization.
- [reproducibility](../reproducibility/README.md): read-only checks of
  saved scientific/reference evidence, with no fitting, replay or submission.
  The checksum-recorded implementation bodies remain at their legacy TRUBA
  paths; the restricted API/CLI exposes only verification.

The [environment records](environments/) capture package versions used by specific scientific or optional-SBI workflows. Use the [results registry](../results/experiment_registry.toml) to identify the environment and source associated with a published quantity.

Many `.env`, batch and submission files preserve a specific past campaign's
source/resource contract. Their presence is not permission to rerun it. Use
[documentation authority](../docs/DOCUMENTATION_AUTHORITY.md), the relevant declaration and current
cluster policy; historical access paths are not portable setup recipes.
Numerical cluster work belongs on scheduled compute nodes. Job scripts,
environment profiles, resource limits and historical producer files remain here.
