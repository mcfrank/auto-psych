"""One core per process: thread counts within a Slurm allocation.

Research Computing flagged job 47042590 (promote, 20 CPUs, 2026-10-08): every
fit process held 140-160 threads. JAX, numpyro and OpenBLAS size their thread
pools by the cores a process can see (about 7 per core), and every process saw
all 20; the single-thread ``XLA_FLAGS`` stop XLA *using* the threads, not
creating them (`docs/auto_rsa/HANDOFF_threads.md`). The checker counts threads,
idle ones too.

So every process that computes sees one core:

- the harness's main thread pins itself at start (`pin_main_thread`), before
  JAX starts its pools; threads it starts later inherit that core;
- a fit child or an agent is started from a thread pinned, for the moment of
  the start, to a core of its own (`one_core`): a child inherits the affinity
  of the thread that starts it, from its first instruction, so even the
  numpy it imports before any of our code runs sees one core.

Affinity is per thread on Linux, so pinning a thread around a start changes
nothing else in the process. The cores are the ones the process could see on
first import of this module (`job_cpus`): the job's allocation.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, List

_CAN_PIN = hasattr(os, "sched_setaffinity")
_JOB_CPUS: List[int] = sorted(os.sched_getaffinity(0)) if _CAN_PIN else list(range(os.cpu_count() or 1))


def job_cpus() -> List[int]:
    """The cores of the job (what the process saw before anything was pinned)."""
    return list(_JOB_CPUS)


class _Cores:
    """Hands out the job's cores, least used first, so concurrent processes
    spread over the allocation (two share a core only when there are more
    processes than cores)."""

    def __init__(self, cores: List[int]) -> None:
        self._use = {c: 0 for c in cores}
        self._lock = threading.Lock()

    def take(self) -> int:
        with self._lock:
            core = min(self._use, key=lambda c: (self._use[c], c))
            self._use[core] += 1
            return core

    def give_back(self, core: int) -> None:
        with self._lock:
            self._use[core] -= 1


CORES = _Cores(_JOB_CPUS)


@contextmanager
def pinned_thread(core: int) -> Iterator[int]:
    """Run the block with the calling thread on ``core`` alone (processes it
    starts inherit that), then restore the thread's affinity."""
    if not _CAN_PIN:
        yield core
        return
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {core})
    try:
        yield core
    finally:
        os.sched_setaffinity(0, before)


@contextmanager
def one_core() -> Iterator[int]:
    """A core of the job for a process started in the block, held until the
    block ends: start the process inside, wait for it inside."""
    core = CORES.take()
    try:
        with pinned_thread(core):
            yield core
    finally:
        CORES.give_back(core)


def pin_main_thread() -> None:
    """Pin the calling thread (an entry point's main thread, before it starts
    JAX) to one core of the job for good. The core counts as in use."""
    if _CAN_PIN:
        os.sched_setaffinity(0, {CORES.take()})


def thread_count(pid: int | str = "self") -> int:
    """Threads of a process (Linux), as Research Computing's checker counts them."""
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise RuntimeError(f"/proc/{pid}/status has no Threads line")
