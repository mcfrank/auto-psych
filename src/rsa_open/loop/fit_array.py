"""Fits as Slurm array tasks: the step design's fit backend (`DISTRIBUTED_DESIGN.md` v3, §0a-§0b).

In step mode (``RSA_OPEN_FIT_BACKEND=slurm``) a loop never fits in its own job. A fit that isn't done
is *pending*: `ArrayBackend.attempt` records it and raises `FitPending`. The loop collects every
pending fit of a phase (`collect_pending`), and the step submits them as one array with its own
continuation (`src.rsa_open.loop.slurm`) and exits. Each array task runs one fit exactly as the
local backend's child does (`run_task`: the same child, its time limit, its peak memory) and
writes a result file beside the fit: ``ok`` (the fit is in the cache), ``model`` (the theory's own
failure), ``timeout`` (past its time limit), ``infrastructure``, or ``killed`` (by the out-of-memory
killer, with its peak). The
continuation's `ArrayBackend.attempt` reads that result and returns exactly what the local
backend would have, so admission, the out-of-memory retry with twice the room, the time limits and
failure classification are unchanged.

Memory is one size per array (`FIT_GB`, 18 GB: peaks on chain 1 were 11.9 GB median and 14.2 GB at
most), and `RETRY_GB` (36) for a fit killed out of memory. A fit submitted `MAX_SUBMISSIONS` times
without a result (a task that died with its wrapper) is an infrastructure failure, so a step can
never resubmit forever.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FitTimeLimitExceeded
from src.rsa_open.loop.fitting import FitBackend, FitJob, _Killed, machine

FIT_GB = 18
RETRY_GB = 36
MAX_SUBMISSIONS = 2
GiB = 2**30


class FitPending(Exception):
    """A fit that must run as an array task before this step can go on."""

    def __init__(self, job: FitJob, mem_gb: int) -> None:
        super().__init__(f"{job.name}: pending ({mem_gb} GB)")
        self.job, self.mem_gb = job, mem_gb


class FitsPending(Exception):
    """Every fit a phase needs that isn't done: the step submits them and exits."""

    def __init__(self, pending: Sequence[FitPending]) -> None:
        unique: Dict[str, FitPending] = {}
        for p in pending:
            unique.setdefault(f"{p.job.nc_path}|{p.mem_gb}", p)
        self.pending = list(unique.values())
        super().__init__(f"{len(self.pending)} fit(s) pending")


def mem_gb_for(at_least: float) -> int:
    """The memory size of an attempt that needs ``at_least`` bytes (0: a first attempt)."""
    if at_least <= FIT_GB * GiB:
        return FIT_GB
    if at_least <= RETRY_GB * GiB:
        return RETRY_GB
    raise RuntimeError(f"a fit needs {at_least / GiB:.0f} GB, more than the {RETRY_GB} GB a retry gets: an "
                       "infrastructure failure, not the theory's")


def result_path(job: FitJob, mem_gb: int) -> Path:
    return job.nc_path.with_name(f"{job.nc_path.name}.mem{mem_gb}.result.json")


def submissions_path(job: FitJob, mem_gb: int) -> Path:
    return job.nc_path.with_name(f"{job.nc_path.name}.mem{mem_gb}.submitted")


class ArrayBackend(FitBackend):
    """An attempt is an array task's result, read back; a missing one is pending."""

    name = "slurm"

    def attempt(self, job: FitJob, at_least: float = 0.0) -> tuple:
        gb = mem_gb_for(at_least)
        result = result_path(job, gb)
        if result.exists():
            r = json.loads(result.read_text())
            if r["kind"] == "killed":
                raise _Killed(r.get("exitcode"), int(r.get("peak") or 0), reserved=gb * GiB)
            if r["kind"] == "timeout":  # as the local child raises it: too slow, the theory's failure
                raise FitTimeLimitExceeded(r["message"])
            if r["kind"] == "ok" and not (job.nc_path.exists() and job.meta_path.exists()):
                raise RuntimeError(f"{job.name}: its array task reported a fit at {job.nc_path} that isn't there")
            return r["kind"], r.get("message"), r.get("traceback")
        sub = submissions_path(job, gb)
        count = int(sub.read_text()) if sub.exists() else 0
        if count >= MAX_SUBMISSIONS:
            raise RuntimeError(f"{job.name}: submitted {count} times as an array task without a result "
                               f"({result.name}): an infrastructure failure, not the theory's")
        raise FitPending(job, gb)


# ---------- collecting a phase's pending fits ----------

def collect_pending(calls) -> List[Any]:
    """Run each zero-argument call; gather `FitPending`/`FitsPending` instead of stopping at the first.
    Returns the calls' results (None for a pending one) and raises `FitsPending` with all of them."""
    pending: List[FitPending] = []
    results = []
    for call in calls:
        try:
            results.append(call())
        except FitPending as p:
            pending.append(p)
            results.append(None)
        except FitsPending as ps:
            pending.extend(ps.pending)
            results.append(None)
    if pending:
        raise FitsPending(pending)
    return results


# ---------- the array: its manifest and one task ----------

def job_record(p: FitPending) -> Dict[str, Any]:
    j = p.job
    return {"theory_path": str(j.theory_path), "name": j.name, "corpus_path": str(j.corpus_path),
            "settings": asdict(j.settings), "nc_path": str(j.nc_path), "meta_path": str(j.meta_path),
            "time_limit_sec": j.time_limit_sec, "init": j.init, "mem_gb": p.mem_gb}


def job_from_record(r: Mapping[str, Any]) -> FitJob:
    return FitJob(Path(r["theory_path"]), r["name"], Path(r["corpus_path"]), FitSettings(**r["settings"]),
                  Path(r["nc_path"]), Path(r["meta_path"]), float(r["time_limit_sec"]), r.get("init"))


def write_manifest(path: Path, pending: Sequence[FitPending]) -> Path:
    """The array's task list (one record per task, all of one memory size), and each fit's count of
    submissions (`MAX_SUBMISSIONS`)."""
    sizes = {p.mem_gb for p in pending}
    if len(sizes) != 1:
        raise ValueError(f"one memory size per array, not {sorted(sizes)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"mem_gb": sizes.pop(), "tasks": [job_record(p) for p in pending]}, indent=1))
    os.replace(tmp, path)
    for p in pending:
        sub = submissions_path(p.job, p.mem_gb)
        sub.parent.mkdir(parents=True, exist_ok=True)  # a fit's first: its cache directory may not exist yet
        count = int(sub.read_text()) if sub.exists() else 0
        sub.write_text(str(count + 1))
    return path


def _write_result(job: FitJob, mem_gb: int, record: Mapping[str, Any]) -> Dict[str, Any]:
    out = result_path(job, mem_gb)
    out.parent.mkdir(parents=True, exist_ok=True)
    full = {**record, "machine": machine()}
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(full))
    os.replace(tmp, out)
    return full


def run_task(manifest: Path, index: int) -> Dict[str, Any]:
    """One array task: the fit, in a child of this task as the local backend runs it (its time limit,
    its peak), and its result file. A theory's own failure is a result, not a failed task."""
    import multiprocessing as mp

    from src.rsa_open.loop.fitting import _child, _run_child

    spec = json.loads(Path(manifest).read_text())
    job, gb = job_from_record(spec["tasks"][index]), int(spec["mem_gb"])
    if result_path(job, gb).exists():  # a requeued task whose result is in: nothing to do
        return json.loads(result_path(job, gb).read_text())
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(str(job.theory_path), job.name, str(job.corpus_path), job.settings,
                                            str(job.nc_path), str(job.meta_path), queue, job.init))
    try:
        kind, message, tb = _run_child(proc, queue, job.name, job.nc_path, job.time_limit_sec)
        record = {"kind": kind, "message": message, "traceback": tb}
    except _Killed as killed:
        record = {"kind": "killed", "exitcode": killed.exitcode, "peak": killed.peak}
    except Exception as exc:  # the time limit, or the task's own trouble: reported as the local backend would
        if isinstance(exc, FitTimeLimitExceeded):
            record = {"kind": "timeout", "message": str(exc)}
        else:
            record = {"kind": "infrastructure", "message": f"{type(exc).__name__}: {exc}",
                      "traceback": "".join(traceback.format_exception(exc))[-4000:]}
    return _write_result(job, gb, record)
