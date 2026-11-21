# Prompt 7 forward-only graph experiment

This experiment runs no MUTACO inversion, SBI, target-bank campaign, or paper
reconstruction.

From the repository root:

```text
python -m pip install -e simulators/graph-simulator
python -m experiments.graph_forward.run --device auto --output runs/graph_forward/prompt7_cpu
python -m experiments.graph_forward.plot --input runs/graph_forward/prompt7_cpu
python -m experiments.graph_forward.run_parity --output runs/graph_forward/prompt7_parity
python -m experiments.graph_forward.profile --output runs/graph_forward/prompt7_profile
```

The public `load_forward_artifacts()` API validates schema, hashes, response
dimension, and finite numerical data before loading or replotting.

The prepared real-GPU command is:

```text
python -m experiments.graph_forward.validate_cuda --output <output> --device cuda
```

It fails explicitly without CUDA and compares CPU and CUDA with the same
captured realized inputs. No CPU-only run establishes CUDA parity.
