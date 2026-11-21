# Linux x86-64 first-adaptive reference capture

This procedure reuses `replay_divergence.py`; there is no second optimizer or
replay engine. The Python diagnostic already works without macOS APIs. The
`reference` output format adds deterministic compression, exact tensor bytes
and explicit historical-source metadata. Production code is unchanged.

Begin with B40 target 16, `/records/48`. The diagnostic regenerates the original
initial design, checks and returns only its 16 archived observations, freezes
the response transform, fits the production multitask GP and proposes candidate
17. It never substitutes a historical candidate. The archive adapter cannot
return observation 17, including when the proposal matches history. All three
physical entry points (`evaluate`, `evaluate_batch`, `evaluate_condition_grid`)
are replaced by raising tripwires. An attempted physical call is an invariant
failure and gives a nonzero exit. No scheduler command is used.

This is the preserved historical capture procedure. Current TRUBA validation
follows scientific/refactor_validation/README.md. Original transfer bundles
and raw packets remain private. Recover original source identities from the
private backup; sanitized summaries never replace acceptance packets.

## Source transfer into an existing canonical clone

The prepared transfer directory contains an incremental Git bundle based on
canonical baseline `aa7f4feb58302ecf6abfa6dee2d81071eabb7fe3`, a
`SOURCE_REVISION` file identifying its exact diagnostic revision, and the
matching macOS B40 reference packet. The diagnostic's parent milestone is
`002f3141d4d173ba5754c0314c7a41154c804822`; use the revision in
`SOURCE_REVISION`, which also includes this capture-format preparation.

On an ordinary Linux x86-64 computer, use Bash and replace the three paths:

```bash
set -euo pipefail
MUTACO_TRANSFER=/absolute/path/to/received-transfer-directory
MUTACO_SOURCE=/absolute/path/to/existing-canonical-MUTACO-clone
MUTACO_CHECKOUT=/absolute/path/to/new-MUTACO-reference-checkout

(cd "$MUTACO_TRANSFER" && sha256sum -c TRANSFER_SHA256SUMS)
git -C "$MUTACO_SOURCE" bundle verify "$MUTACO_TRANSFER/MUTACO-reference.bundle"
git -C "$MUTACO_SOURCE" fetch "$MUTACO_TRANSFER/MUTACO-reference.bundle" ege/repro-refactor
test "$(git -C "$MUTACO_SOURCE" rev-parse FETCH_HEAD)" = "$(cat "$MUTACO_TRANSFER/SOURCE_REVISION")"
git -C "$MUTACO_SOURCE" worktree add --detach "$MUTACO_CHECKOUT" FETCH_HEAD
cd "$MUTACO_CHECKOUT"
test -z "$(git status --porcelain)"
```

This fetch reads a local bundle. It does not contact GitHub or modify `main`
or an existing development branch. Bundle verification must pass; stop if the
canonical baseline/prerequisite objects are missing. The checkout path must be
new. A researcher who already has the exact `SOURCE_REVISION` can instead use
an isolated clean checkout of that revision.

## Environment and immutable inputs

Prefer the original scientific environment if it is available. Set the exact
Python executable explicitly; its existing prefix may be `mutaco-sbi-0261`.
The direct Linux command below performs a pinned preflight rather than using
the macOS launcher's environment-name gate.

```bash
MUTACO_REFERENCE_PYTHON=/absolute/path/to/scientific-environment/bin/python
export PYTHONPATH="$PWD/src:$PWD/packages/analytic-simulator/src:$PWD/packages/graph-simulator/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
MUTACO_CAPTURE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mutaco-reference-b40.XXXXXX")"
```

Required Python is **3.11.15**, CPU execution, with every package version in
`configs/environments/mutaco-sbi-lock.json`. This includes Torch **2.12.1+cpu**,
BoTorch **0.18.1**, GPyTorch **1.15.2**, SBI **0.26.1**, NumPy **2.4.6**, SciPy
**1.17.1**, and pytest **9.1.1**. Preserve existing BLAS/thread environment
variables; the packet records them. The diagnostic explicitly sets Torch
intra/inter-op threads to **20/1**, matching the historical worker profile.
Do not silently substitute versions or change numerical policies if setup fails.

If a new environment is necessary, first provision Python 3.11.15, for example
with `conda create --prefix /absolute/new/path/mutaco-sbi python=3.11.15 pip --yes`.
Point `MUTACO_REFERENCE_PYTHON` to it, then resolve the existing reference pins:

```bash
"$MUTACO_REFERENCE_PYTHON" - <<'PY' > "$MUTACO_CAPTURE_DIR/requirements.reference.txt"
import json
from pathlib import Path
lock = json.loads(Path('configs/environments/mutaco-sbi-lock.json').read_text())
print('\n'.join(f'{name}=={version}' for name, version in lock['packages'].items() if name != 'torch'))
PY
"$MUTACO_REFERENCE_PYTHON" -m pip install 'torch==2.12.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$MUTACO_REFERENCE_PYTHON" -m pip install -r "$MUTACO_CAPTURE_DIR/requirements.reference.txt"
"$MUTACO_REFERENCE_PYTHON" -m pip check
```

New Linux installation was not executed during the macOS preparation. Download
availability and transitive dependency resolution must be verified there;
record the resulting freeze. Matching top-level versions does not reconstruct
an undocumented historical BLAS build or guarantee numerical equivalence.

Run this preflight before the capture; it fails before any fit on mismatch:

```bash
"$MUTACO_REFERENCE_PYTHON" - <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import platform
import torch
import mutaco
lock = json.loads(Path('configs/environments/mutaco-sbi-lock.json').read_text())
assert platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64')
assert platform.python_version() == lock['python'], platform.python_version()
differences = {name: (pin, version(name)) for name, pin in lock['packages'].items() if version(name) != pin}
assert not differences, differences
assert str(torch.get_default_device()) == 'cpu'
assert Path(mutaco.__file__).resolve().is_relative_to(Path.cwd().resolve() / 'src')
print('Linux x86-64 pinned CPU preflight passed')
PY
```

Keep these B40 files in their committed locations; do not extract or edit them:

```text
benchmarks/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2/scalar-primary-b40-v3r2.tar.gz
benchmarks/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2/scalar-primary-b40-v3r2.manifest.json
benchmarks/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2/scalar-primary-b40-v3r2.manifest.sha256
benchmarks/paper/scalar_primary_trajectory_archives_v3r2/b40-v3r2/scalar-primary-b40-v3r2.tar.sha256
data/cohorts/paper120_v3/target_016.json
data/cohorts/paper120_v3/target_016.sha256
```

The archive SHA-256 must be
`2f86646f7c21914552d10e9e1fcb209d29880cee194a3e79f8f8fae211b6659c`.
The tool checks this, its manifest and every member/sidecar before using data.
Keep the rest of the source checkout, production configurations, profile and
cohort manifest present. Historical execution source is
`bd83de6ec01f0f7e2241ee498960a4d493d601e6`, read from the committed scientific
manifest and recorded separately from the diagnostic's Git revision.

## Exact capture and return commands

Run two fresh processes with identical settings:

```bash
for MUTACO_REPEAT in 1 2; do
  /usr/bin/time -v "$MUTACO_REFERENCE_PYTHON" -m experiments.paper.exploration.replay_divergence \
    --budget 40 --target 16 --format reference \
    --torch-intraop-threads 20 --torch-interop-threads 1 \
    --output "$MUTACO_CAPTURE_DIR/b40.reference.$MUTACO_REPEAT.json.gz" \
    > "$MUTACO_CAPTURE_DIR/b40.reference.$MUTACO_REPEAT.log" 2>&1
done
"$MUTACO_REFERENCE_PYTHON" -m pip freeze --all > "$MUTACO_CAPTURE_DIR/environment.freeze.txt"
git rev-parse HEAD > "$MUTACO_CAPTURE_DIR/source.txt"
git status --porcelain >> "$MUTACO_CAPTURE_DIR/source.txt"
if cmp -s "$MUTACO_CAPTURE_DIR/b40.reference.1.json.gz" "$MUTACO_CAPTURE_DIR/b40.reference.2.json.gz"; then
  printf 'Repeated reference packets are byte-identical\n' > "$MUTACO_CAPTURE_DIR/repeat-comparison.txt"
else
  printf 'Repeated packets differ; preserve both for investigation\n' > "$MUTACO_CAPTURE_DIR/repeat-comparison.txt"
fi
(cd "$MUTACO_CAPTURE_DIR" && sha256sum b40.reference.*.json.gz b40.reference.*.log environment.freeze.txt source.txt repeat-comparison.txt > SHA256SUMS)
(cd "$MUTACO_CAPTURE_DIR" && sha256sum -c SHA256SUMS)
tar -czf "${MUTACO_CAPTURE_DIR}.tar.gz" -C "$MUTACO_CAPTURE_DIR" .
sha256sum "${MUTACO_CAPTURE_DIR}.tar.gz"
printf 'Return %s and its SHA-256\n' "${MUTACO_CAPTURE_DIR}.tar.gz"
```

`/usr/bin/time -v` is GNU time, used only for resource measurement; if it is
unavailable, omit that prefix and record that resources were not measured.
An exit of zero means a complete safe diagnostic packet, including a genuine
historical divergence, was captured. It does not mean golden replay passed.
On nonzero exit, stop and return the partial packet/log; do not change seeds,
inputs, fitting policy or tolerance to obtain a successful command.

Return the small output archive and its SHA-256 through the researcher's chosen
file-transfer channel. No automatic sending, GitHub update or remote job is
part of this procedure. Preserve both repeats if they differ. B100 is supported
by the same tool using `--budget 100 --target 0`, its corresponding committed
archive and target files; begin comparison with B40 before extending it.

## Packet contents and comparison

The gzip file contains sorted compact JSON, with fixed gzip metadata and no
timers. Provenance retains Git/source hashes, archived execution source,
OS/architecture/CPU model, Python/packages, Torch/NumPy/SciPy backend builds,
runtime threadpool reports, thread settings and CPU dtype/device. An unavailable
CPU-model query is labelled explicitly. All newly calculated tensors are labelled
as reconstructed reference values.

The capture includes initial observation IDs and physical arrays; complete
frozen means/scales/floor/transform identity; standardized data; GP cold
initialization/configuration/seeds, fitted parameters/noise/task covariance,
objective/attempts/termination/recovery diagnostics; raw and four-point selected
posteriors; 64 candidates and screening scores; branch, restart starts/traces,
rankings and the effective candidate 17. Nonfinite screening scores retain their
original bits and explicit tags. The four-point diagnostic posterior is one
extra batched query after replay, outside production accounting.

Each tensor stores shape, Torch dtype, a NumPy byte dtype, little-endian base64
bytes and SHA-256. For example, decode the actual proposed candidate:

```python
import base64, gzip, json
import numpy as np
with gzip.open('b40.reference.1.json.gz', 'rt') as stream:
    packet = json.load(stream)
capture = packet['capture']
t = capture['reconstructed']['effective_candidate_17']
candidate = np.frombuffer(base64.b64decode(t['data_base64']), dtype=t['numpy_dtype']).reshape(t['shape'])
print(candidate, capture['replay_outcome']['physical_simulator_calls'])
```

The `numerical_state_sha256` excludes source/environment metadata, timers,
the offline alternative-mean probe and unused float64-mirror initialization.
Individual tensor hashes locate differences. A differing whole-file checksum
between computers is expected because their provenance differs; compare the
numerical states themselves. First compare physical inputs and transform,
then cold parameters/seeds, fitted state, posterior/screening, and refinement.
No numerical tolerance or historical replacement is introduced by this format.

The original historical GP initialization, fitted tensors, optimizer iterates
and exact runtime backend were not preserved in the compact archives. A new
x86 capture is a reconstruction. If it also differs from history, the two
modern environments alone cannot identify which absent historical intermediate
first differed. An original checkpoint/log would still be required. Core
refactoring remains gated until the researcher reviews the reference evidence.

## Local preparation checks and resources

On the qualified macOS environment, the focused diagnostic/archive tests give
**27 passed, 2 skipped**. The skips are the separate opt-in exact golden gates.
The suite includes production capture invariance for both targets, exact tensor
byte round-trips, deterministic gzip output and blocked-entry tests for every
physical simulator method. The existing scientific source and policies are
unchanged.

The new B40 format preserves all 111 tensor signatures and 1,096 other compared
scientific fields from the prior macOS capture. Fresh reconstructions have the
same numerical-state hash and candidate `[0.9671909809112549,
0.9396123290061951, 0.5113476514816284, 0.882079005241394]`; they still differ
from history at evaluation 17. The transfer packet carries a clean-source
capture and its checksum. No Linux capture or cross-platform equivalence is
claimed by this preparation.

Measured macOS B40 output is approximately **141 KiB compressed / 277 KiB
uncompressed**, with a roughly **8-second** command wall time and **561 MiB**
peak resident memory in a separate resource wrapper. The source tracks about
**173 MiB** of files, including both compressed archives (about 33 MiB); local
Git storage is about 86 MiB, and the installed Mac environment about 1 GiB.
Linux wheel/install sizes and peak memory have not been measured. Allow a few
GiB of free disk for source/environment/install caches and provision at least
2 GiB RAM for this bounded comparison, then record actual GNU-time results.
These are setup estimates, not a memory guarantee or campaign benchmark.
