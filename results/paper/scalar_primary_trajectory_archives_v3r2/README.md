# Public B40 and B100 trajectory archives

The two v3r2-public archives provide the original scalar-primary B40 and B100
numerical trajectory exports with one disclosure change. Each absolute Python
interpreter path in the acquisition execution metadata was replaced with
`<redacted>`. Across the two archives, reversing that substitution reproduced
the original compact JSON payloads byte for byte. All 360 method records per
budget, their order, seeds, and numerical observations are preserved. B40 and
B100 were independent campaigns with 14,400 and 36,000 trajectory steps.

| Budget | Archive | Files |
| --- | --- | --- |
| B40 | [scalar-primary-b40-v3r2-public.tar.gz](b40-v3r2/scalar-primary-b40-v3r2-public.tar.gz) | [Manifest](b40-v3r2/scalar-primary-b40-v3r2-public.manifest.json), [archive SHA-256](b40-v3r2/scalar-primary-b40-v3r2-public.tar.sha256) |
| B100 | [scalar-primary-b100-v3r2-public.tar.gz](b100-v3r2/scalar-primary-b100-v3r2-public.tar.gz) | [Manifest](b100-v3r2/scalar-primary-b100-v3r2-public.manifest.json), [archive SHA-256](b100-v3r2/scalar-primary-b100-v3r2-public.tar.sha256) |

From the repository root, run `python -m reproducibility.public_trajectories 40`
and `python -m reproducibility.public_trajectories 100`. Each check verifies
the manifest, compressed archive, member inventory, and member checksums.
The manifest gives the source archive and compact-export hashes for private
provenance. The unchanged original archives are held in the private historical
repository and are required for byte-exact historical validation and replay.
These data do not include complete GP model/optimizer states or scheduler logs.

See [data availability](../../../DATA_AVAILABILITY.md) and
[dataset licensing](../../../DATA_LICENSE.md).
