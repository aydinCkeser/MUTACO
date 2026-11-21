"""Computation-free validation of the proposed v3 Barbun shard geometry."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.paper.common.population_v3 import shard_indices


@dataclass(frozen=True)
class BarbunV3Layout:
    nodes: int = 8
    tasks_per_node: int = 5
    cpus_per_task: int = 8

    @property
    def workers(self) -> int:
        return self.nodes * self.tasks_per_node

    def validate(self, target_count: int) -> dict[str, object]:
        shards = tuple(
            shard_indices(target_count, rank=rank, workers=self.workers)
            for rank in range(self.workers)
        )
        flattened = tuple(index for shard in shards for index in shard)
        if sorted(flattened) != list(range(target_count)) or len(flattened) != len(set(flattened)):
            raise ValueError("rank-strided shards do not cover targets exactly once")
        return {
            "nodes": self.nodes,
            "tasks_per_node": self.tasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "workers": self.workers,
            "target_count": target_count,
            "waves": (target_count + self.workers - 1) // self.workers,
            "shards": [list(shard) for shard in shards],
            "scheduler_queried": False,
            "job_submitted": False,
            "scientific_calls": 0,
        }
