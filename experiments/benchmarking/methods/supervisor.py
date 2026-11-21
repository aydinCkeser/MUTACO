"""Hard-deadline process supervision with descendant cleanup evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
from time import perf_counter, process_time, sleep
from typing import Any, Callable, Mapping, Sequence

import psutil


def _live_processes(processes: Sequence[psutil.Process]) -> list[psutil.Process]:
    live: list[psutil.Process] = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return live


def _process_tree(root_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return []
    try:
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    return [*children, root]


def terminate_process_tree(root_pid: int, *, grace_seconds: float) -> dict[str, Any]:
    """Terminate then kill one complete process tree and prove no survivor."""

    processes = _process_tree(root_pid)
    observed_pids = tuple(process.pid for process in processes)
    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(processes, timeout=max(grace_seconds, 0.0))
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=max(grace_seconds, 0.0))
    survivors = tuple(process.pid for process in _live_processes(processes))
    return {
        "observed_process_tree_pids": observed_pids,
        "surviving_pids": survivors,
        "all_descendants_terminated": not survivors,
    }


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    deadline_seconds: float,
    poll_interval_seconds: float,
    termination_grace_seconds: float,
    progress_interval_seconds: float | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a command under a parent-enforced wall deadline."""

    if deadline_seconds <= 0.0:
        raise ValueError("deadline_seconds must be positive")
    if poll_interval_seconds <= 0.0:
        raise ValueError("poll_interval_seconds must be positive")
    if termination_grace_seconds < 0.0:
        raise ValueError("termination_grace_seconds must be nonnegative")
    if progress_interval_seconds is not None and progress_interval_seconds <= 0.0:
        raise ValueError("progress_interval_seconds must be positive")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started, cpu_started = perf_counter(), process_time()
    peak_rss_bytes = 0
    sampled_child_cpu_seconds = 0.0
    timed_out = False
    observed_processes: dict[int, psutil.Process] = {}
    termination = {
        "observed_process_tree_pids": (),
        "surviving_pids": (),
        "all_descendants_terminated": True,
    }
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
        )
        root_pid = process.pid
        next_progress = progress_interval_seconds
        while process.poll() is None:
            elapsed = perf_counter() - wall_started
            if elapsed >= deadline_seconds:
                timed_out = True
                termination = terminate_process_tree(
                    root_pid, grace_seconds=termination_grace_seconds
                )
                break
            tree = _process_tree(root_pid)
            observed_processes.update({member.pid: member for member in tree})
            rss = 0
            cpu = 0.0
            for member in tree:
                try:
                    rss += int(member.memory_info().rss)
                    times = member.cpu_times()
                    cpu += float(times.user + times.system)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            peak_rss_bytes = max(peak_rss_bytes, rss)
            sampled_child_cpu_seconds = max(sampled_child_cpu_seconds, cpu)
            if (
                progress_callback is not None
                and next_progress is not None
                and elapsed >= next_progress
            ):
                try:
                    progress_callback(
                        {
                            "elapsed_seconds": elapsed,
                            "remaining_seconds": max(
                                deadline_seconds - elapsed, 0.0
                            ),
                            "peak_process_tree_rss_bytes": peak_rss_bytes,
                            "sampled_child_cpu_seconds": sampled_child_cpu_seconds,
                        }
                    )
                except Exception:
                    pass
                next_progress += progress_interval_seconds
            sleep(min(poll_interval_seconds, max(deadline_seconds - elapsed, 0.0)))
        if timed_out:
            try:
                process.wait(timeout=max(termination_grace_seconds, 0.1))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()
            lingering = _live_processes(tuple(observed_processes.values()))
            cleanup_records = [
                terminate_process_tree(
                    member.pid, grace_seconds=termination_grace_seconds
                )
                for member in lingering
            ]
            survivors = tuple(
                sorted(
                    {
                        pid
                        for record in cleanup_records
                        for pid in record["surviving_pids"]
                    }
                )
            )
            termination = {
                "observed_process_tree_pids": tuple(sorted(observed_processes)),
                "surviving_pids": survivors,
                "all_descendants_terminated": not survivors,
                "post_exit_lingering_pids": tuple(
                    member.pid for member in lingering
                ),
            }
    wall_seconds = perf_counter() - wall_started
    return {
        "schema_version": "mutaco-bounded-process-v1",
        "command": list(command),
        "cwd": str(cwd),
        "root_pid": root_pid,
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_seconds": deadline_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "progress_interval_seconds": progress_interval_seconds,
        "wall_seconds": wall_seconds,
        "parent_process_cpu_seconds": process_time() - cpu_started,
        "sampled_child_cpu_seconds": sampled_child_cpu_seconds,
        "peak_process_tree_rss_bytes": peak_rss_bytes,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "termination": termination,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
