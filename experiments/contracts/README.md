# Shared research contracts

[__init__.py](__init__.py) owns two small, policy-neutral definitions:

- `PhysicalCallCounts`: an immutable attempted/successful/failed/retry value,
  with the existing integer and consistency checks. Constructing it performs
  no evaluation, retry, timeout handling or ledger update.
- `PaperContractError`: the existing validation exception, retaining its
  historical name and inheritance for compatibility. No new error hierarchy
  or experimental policy is introduced.

These contracts require only the Python standard library. Old imports from
`experiments.paper.common.protocol_v3` and `experiments.paper.common.validation`
refer to the same classes. Their canonical `__module__` names deliberately
remain historical so existing pickles and newly written equivalents retain
their original interfaces; the definitions themselves live here.

[benchmarking/contracts.py](../benchmarking/contracts.py) retains benchmark
adapters, result containers, method selection and budget-resolution rules.
[paper/common/protocol_v3.py](../paper/common/protocol_v3.py) retains
`DecisionOutcome`, frozen decisions, registered-method checks, seed identities,
paired assessment and record schemas. Those are paper methodology, not generic
contracts. SBI-zoo failures remain with the [method implementation](../benchmarking/methods/sbi_zoo_failures.py).

The [paper package](../paper/README.md) exposes its existing public names lazily;
requesting an execution function still loads its actual implementation and does
not authorize running it. Offline result analysis belongs in
[paper/analysis](../paper/analysis), separately from experiment execution.
