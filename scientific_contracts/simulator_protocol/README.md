# Simulator protocol

The minimum simulator maps `[P] -> [M]`. A declared batched capability may map
`[H,P] -> [H,M]` without changing the meaning or accounting of one parameter
evaluation.

Inputs and outputs are finite, contiguous Torch tensors on the resolved device
and dtype. Inputs are caller-owned and immutable. Responses are newly owned by
the simulator boundary and may not alias inputs. Component ordering, physical
units, intrinsic aggregation, and simulator stochasticity belong to the
simulator specification.

MUTACO requests no simulator gradients and contains no graph-specific axes.

## Sequential definition, metadata, and retry contract

Simulator capabilities expose immutable pre-run definition metadata. Both
definition metadata and per-evaluation result metadata use strict finite
JSON-compatible values: null, booleans, integers, finite floats, strings,
arrays, and string-keyed objects. Tensors, paths, dataclasses, enums, tuples,
sets, bytes, arbitrary objects, non-string keys, and non-finite floats fail
with a typed metadata-contract error.

Definition metadata participates in the parent run identity. Returned
metadata never enters that parent identity or seed derivation. It is validated
and recorded only with its evaluation.

The sequential runner captures one stable capability snapshot and rejects
drift before every physical attempt. Stochastic simulators require context
support. Deterministic
simulators may be context-free. Sequential D supports only identity parameter
transformations.

Infrastructure retries are opt-in through `RetryableSimulationError` and its
stable reason enum. Arbitrary runtime errors and tensor/metadata contract
failures are not implicitly retryable. Per-logical and total physical attempt
budgets are explicit execution controls. Retried attempts reuse the logical
evaluation seed; their attempt index is physical provenance rather than a new
scientific realization.
