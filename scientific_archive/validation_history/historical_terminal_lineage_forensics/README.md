# Historical MUTACO terminal-lineage forensics

**Classification:** zero-call forensic evidence; no historical artifact is
modified or reinterpreted as current-policy output.

The audit covers the immutable Paper-120 MUTACO arms at B40 and B100. It keeps
four quantities distinct:

\[
E_{\rm obs},\qquad E_{\rm raw}^{\rm GP},\qquad
E_{\rm term}^{\rm GP},\qquad E_{\rm heldout}.
\]

All 120 B40 raw records retain the 40 physical search responses, terminal
screen scores, restart histories, returned surrogate score, and held-out
endpoint. Consequently all four quantities and the target-specific terminal
path can be audited directly. The B100 canonical bundle retains recommendations
and held-out endpoints but not raw trajectories or terminal metadata. Its
exact-source implementation can be established, but its target-specific raw
versus refined branch is unavailable.

## Historical defect

For a restart beginning at \(\theta_0\), the historical optimizer sets
\(s_0=L(\mu(\theta_0),y^\star)\) and `previous = s0`. It evaluates the gradient
at \(\theta_0\), applies one Adam update to produce \(\theta_1\), but assigns
`current` from the already computed pre-update score. Thus

\[
|\text{previous}-\text{current}|=|s_0-s_0|=0,
\]

so the convergence test fires after the first update. Only after leaving the
loop is \(L(\mu(\theta_1),y^\star)\) evaluated. This is not raw-screen-only,
but it is also not the declared 64-step refinement.

The exact optimizer and physical-MAE Git blobs are identical across the B40
source commit and both B100 source commits. Exact line references and blob
identities are frozen in the machine forensic specification.

Historical held-out assessments remain valid observations of the immutable
recommendations that actually ran. They do not estimate performance of the
corrected multi-step terminal optimizer. A repaired MUTACO--SBI comparison is
therefore required for a current-policy comparison.

The canonical target-level rows, aggregate report, zero-discrepancy audit, and
source-blob verification are indexed by
`benchmarks/paper/historical_terminal_lineage_forensics/v1/manifest.json`.
