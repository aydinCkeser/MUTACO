# Shareable engineering milestone evidence

These files derive from a verified private backup of original branch head
d7b5b1267419ca4a95001c2784cfb83b0a020e7b. The original branch, manifests, raw
logs, captures and source history remain unchanged outside this checkout.
These derivatives omit account names, absolute machine paths, editable local
installation URLs, login/compute hostnames, raw logs and embedded source bundles.

The manifest maps each derived file to its original raw-file SHA-256 values
and its new derived-file SHA-256. The raw-evidence index inventories private
raw-file identities without their contents or backup location. Derived files
are not claimed to be byte-identical to the raw inputs.

The two mathematical files retain the original exact scientific payloads.
The first-fit/conditional checkpoint hash is
12251e746d4c39f2f028651f766c96a6723dc39cb63d5cd484ad661b9cf3001c.
The complete replay mathematical hash is
687433605adebeb928b081c616e7d603267d71faebb3cfa061a6c730865ec782.
Candidates, tensor bytes, fitted parameters, seed identities, decisions and
accounting within these payloads are unchanged. They describe reconstructed
GP states, not original historical GP snapshots. The separately labelled
final-data cold-fit checkpoint is not the historical final warm-start model.

The recorded B40 baseline and refactor replayed all 40 observations with
25 fits, 24 proposals and one terminal recommendation. Baseline tests:
169 passed, 8 skipped, one known NEI fixture failure. Accepted refactor tests:
175 passed, 8 skipped, the same failure. No new relevant failures or physical
simulator calls occurred. The original raw numerical/environment checks passed.
B100 has not been captured.

## Verification boundaries

From the repository root, use the approved local launcher:

    infrastructure/local/run_mutaco.sh -m infrastructure.truba.shareable_evidence verify \
      --directory benchmarks/reproducibility/shareable_milestone_v1

This checks derived integrity and scientific hashes, with zero new numerical
calculations. To verify original evidence independently, add
--raw-source "$PRIVATE_RESTORED_SOURCE" pointing to a separately verified
private restore of the original branch. Original comparator, checksums and
environment fingerprint must all pass. The tool also compares each derived
projection to its raw source. Original checks are not weakened.

Exact environment acceptance cannot use sanitized metadata alone: the original
fingerprint includes library paths. A derived backend summary is provenance,
not a replacement environment identity. Raw acceptance packets remain private
and recoverable. Never feed these derivatives to the original raw comparator.

The candidate batch script requires MUTACO_PRIVATE_REFERENCE_DIR, pointing
to the original raw reference directory outside the checkout. All original
checksum and mathematical checks remain. A restored private capture-harness
checkout supplies the recorded historical harness. No execution is authorized.

Verify all derived file bytes from this directory with
shasum -a 256 -c SHA256SUMS. Preparing these shareable files launches no cluster
job or physical simulation.
