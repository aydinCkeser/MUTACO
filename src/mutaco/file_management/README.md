# Run artifacts and file writes

[artifact_documents_and_manifests.py](artifact_documents_and_manifests.py)
defines the version markers for run configuration, observations, recommendation,
accounting and other recorded artifacts. `make_artifact_document` builds a
document for a registered filename. `validate_artifact_document` checks that its
type and version agree with that filename and that the result is canonically
serializable. `build_manifest` takes a mapping of filenames to their **final
bytes**. It sorts paths, records each byte count and SHA-256 and rejects a
manifest that includes its own hash. None of these operations writes a file.

[atomic_file_writes.py](atomic_file_writes.py) contains the filesystem
operation. It writes supplied bytes to a unique temporary sibling and
replaces the destination. Transient Windows sharing locks can retry the
replacement without repeating the calculation that supplied the bytes.
`atomic_write_text` encodes the text once and uses the same byte path.
Neither module selects a scientific output directory or runs a simulation.

Existing `mutaco.artifacts`, `mutaco.artifacts.schemas` and `mutaco.atomic_io`
imports continue to expose the original callables and patchable module globals.
