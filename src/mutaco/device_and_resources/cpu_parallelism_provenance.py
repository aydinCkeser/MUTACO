"""Read-only CPU and Torch parallelism provenance."""

from __future__ import annotations

from dataclasses import dataclass
import os

import psutil
import torch


_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CpuParallelismProvenance:
    physical_cores: int | None
    logical_processors: int | None
    torch_intraop_threads: int
    torch_interop_threads: int
    environment_overrides: dict[str, str]
    parallel_info: str

    @classmethod
    def inspect(cls) -> "CpuParallelismProvenance":
        """Inspect settings without calling any thread-setting API."""

        return cls(
            physical_cores=psutil.cpu_count(logical=False),
            logical_processors=psutil.cpu_count(logical=True),
            torch_intraop_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            environment_overrides={
                name: os.environ[name]
                for name in _THREAD_ENVIRONMENT_NAMES
                if name in os.environ
            },
            parallel_info=torch.__config__.parallel_info(),
        )

    def provenance(self) -> dict[str, object]:
        return {
            "physical_cores": self.physical_cores,
            "logical_processors": self.logical_processors,
            "torch_intraop_threads": self.torch_intraop_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "environment_overrides": self.environment_overrides,
            "parallel_info": self.parallel_info,
            "selection_policy": "environment_resolved_preserved",
        }
