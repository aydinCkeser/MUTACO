"""Typed nested wall/CPU timing events."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter, process_time
from typing import Any, Iterator


class TimingLedger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._stack: list[str] = []

    @contextmanager
    def event(
        self,
        task: str,
        *,
        method: str,
        target_index: int | None = None,
        round_index: int | None = None,
        work_count: int = 0,
        simulator_calls: int = 0,
        budget_classification: str = "algorithm",
        device: str,
    ) -> Iterator[dict[str, Any]]:
        event_id = f"timing-{len(self.events):06d}"
        parent = self._stack[-1] if self._stack else None
        started_utc = datetime.now(timezone.utc).isoformat()
        wall_start, cpu_start = perf_counter(), process_time()
        record: dict[str, Any] = {
            "event_id": event_id,
            "parent_event_id": parent,
            "task": task,
            "method": method,
            "target_index": target_index,
            "round_index": round_index,
            "device": device,
            "status": "running",
            "work_count": work_count,
            "simulator_calls": simulator_calls,
            "budget_classification": budget_classification,
            "started_utc": started_utc,
        }
        self._stack.append(event_id)
        try:
            yield record
        except Exception as exc:
            record["status"] = "failed"
            record["failure_type"] = type(exc).__name__
            record["failure_message"] = str(exc)
            raise
        else:
            record["status"] = "completed"
        finally:
            self._stack.pop()
            record["ended_utc"] = datetime.now(timezone.utc).isoformat()
            wall_seconds = perf_counter() - wall_start
            cpu_seconds = process_time() - cpu_start
            child_wall = sum(
                float(event["inclusive_seconds"])
                for event in self.events
                if event.get("parent_event_id") == event_id
            )
            child_cpu = sum(
                float(event["process_cpu_inclusive_seconds"])
                for event in self.events
                if event.get("parent_event_id") == event_id
            )
            record["wall_seconds"] = wall_seconds
            record["process_cpu_seconds"] = cpu_seconds
            record["inclusive_seconds"] = wall_seconds
            record["exclusive_seconds"] = max(0.0, wall_seconds - child_wall)
            record["process_cpu_inclusive_seconds"] = cpu_seconds
            record["process_cpu_exclusive_seconds"] = max(
                0.0, cpu_seconds - child_cpu
            )
            self.events.append(record)
