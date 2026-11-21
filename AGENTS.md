# MUTACO agent instructions

Before work, read Git branch/HEAD/status and `docs/DOCUMENTATION_AUTHORITY.md`.
Read only the contracts and registry entries relevant to the active request.
Normative behavior is indexed by `scientific_contracts/scientific_registry.toml`.
Research evidence is indexed by `results/experiment_registry.toml`. The reviewed
project state and handoff notes in `scientific_archive/project_records/` are historical.

Present a bounded execution prompt before changing the repository. Execute it
only after explicit approval. Never push or launch a scientific run without
explicit authorization. Keep the historical repository and
`docs/latex/overleaf/bandits_v4.tex` untouched.

Keep the core simulator-independent and Torch-native. Preserve named
coordinate-addressed randomness, ambient-RNG isolation, option identities,
dtype/device, timing, call accounting, and accepted scientific contracts.
Profiles select core options; do not create method forks for paper choices.

On Windows, run Python through `infrastructure/local/run_mutaco.ps1`; Paper,
SBI, TRUBA, and all pytest workflows must use its pinned `mutaco-sbi`
environment. Run tests through `infrastructure/local/run_pytest.ps1`. Never
pass a repository-relative `--basetemp` or create dot-prefixed test-output
directories in this Dropbox checkout. `mutaco-dev` is allowed only through the
explicit non-SBI launcher gate.

Use the tracked TRUBA lifecycle command for exact-source preparation and
zero-call validation. Do not compose nested PowerShell/SSH/Bash commands.
Scientific submission remains a separate explicit authorization gate.

Record approved implementation changes in the migration map, scientific behavior
in its normative contract and completed research in its versioned result record.
Keep the reviewed historical state and source identities intact. Do not create prompt histories or routine
handoff documents. Create a handoff only when the user explicitly requests a
new task or agent transfer. Leave committed work clean.
