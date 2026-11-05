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

Memory is one size per array, one of `MEM_TIERS` (18, 36, 72 GB). A theory fitted before starts at
the tier that holds its recorded peak with margin (its folds after its full fit: two of chain 1's
experiment-3 folds reached 28-32 GB, review V3 §11), any other at 18 GB (peaks on chain 1 were
11.9 GB median); a fit killed out of memory is retried at the next tier. A fit submitted
`MAX_SUBMISSIONS` times without a result (a task that died with its wrapper) is an infrastructure
failure, so a step can never resubmit forever.

**Folds after their full fit** (`FoldAfter`): a theory whose out-of-fold scores will certainly be
needed (a starting theory) or are needed if it is admitted (a repair) has its cross-validation folds
submitted with its full fit, as an array that starts when the full fits' arrays end (``afterany``).
Each such task reads the full fit from the cache, starts every fold where the loop would (the full
fit's posterior medians, the same time limit) and fits it as the loop would, near-miss refit
included. It saves a step that would only submit folds (rule 10's 10-minute minimum; review V3
§11). A task whose full fit is missing, failed, or (for a repair) did not converge, fits nothing, and
a fold that fails is left to the next step: it can only save work, never change a result.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FitTimeLimitExceeded
from src.rsa_open.loop.fitting import FitBackend, FitJob, _Killed, machine

MEM_TIERS = (18, 36, 72)
FIT_GB, RETRY_GB = MEM_TIERS[0], MEM_TIERS[1]
MAX_SUBMISSIONS = 2
GiB = 2**30


class FitPending(Exception):
    """A fit that must run as an array task before this step can go on."""

    def __init__(self, job: FitJob, mem_gb: int) -> None:
        super().__init__(f"{job.name}: pending ({mem_gb} GB)")
        self.job, self.mem_gb = job, mem_gb


@dataclass(frozen=True)
class FoldAfter:
    """One cross-validation fold to fit as soon as its theory's full fit is in the cache."""
    theory_path: Path
    name: str
    full_corpus: Path
    full_settings: FitSettings
    cache_dir: Path
    fold_corpus: Path
    fold_settings: FitSettings
    limit: Dict[str, Any]
    """`fold_limit`'s rule: {"fixed": seconds or None} or {"in_play", "fallback", "min", "factor"}."""
    only_if_converged: bool
    mem_gb: int

    @property
    def key(self) -> str:
        return f"after|{self.cache_dir}|{self.name}|{self.fold_corpus}"


def fold_limit(rule: Mapping[str, Any], timing: Optional[Mapping[str, float]]) -> Optional[float]:
    """A fold's time limit from its full fit's timing (the loop's `OpenLoop.fold_time_limit`)."""
    if "fixed" in rule:
        return rule["fixed"]
    full = timing["warmup_sec"] + timing["sampling_sec"] if timing else rule["fallback"]
    return min(rule["in_play"], max(rule["min"], rule["factor"] * full))


class FitsPending(Exception):
    """Every fit a phase needs that isn't done: the step submits them and exits. ``after``: folds
    to fit once these fits are done (`FoldAfter`)."""

    def __init__(self, pending: Sequence[FitPending], after: Sequence[FoldAfter] = ()) -> None:
        unique: Dict[str, FitPending] = {}
        for p in pending:
            unique.setdefault(f"{p.job.nc_path}|{p.mem_gb}", p)
        self.pending = list(unique.values())
        later: Dict[str, FoldAfter] = {}
        for a in after:
            later.setdefault(a.key, a)
        self.after = list(later.values())
        super().__init__(f"{len(self.pending)} fit(s) pending" + (f", {len(self.after)} fold(s) after them"
                                                                   if self.after else ""))


def mem_gb_for(at_least: float) -> int:
    """The memory tier of an attempt that needs ``at_least`` bytes (0: nothing known)."""
    for gb in MEM_TIERS:
        if at_least <= gb * GiB:
            return gb
    raise RuntimeError(f"a fit needs {at_least / GiB:.0f} GB, more than the largest tier ({MEM_TIERS[-1]} GB): an "
                       "infrastructure failure, not the theory's")


def result_path(job: FitJob, mem_gb: int) -> Path:
    return job.nc_path.with_name(f"{job.nc_path.name}.mem{mem_gb}.result.json")


def submissions_path(job: FitJob, mem_gb: int) -> Path:
    return job.nc_path.with_name(f"{job.nc_path.name}.mem{mem_gb}.submitted")


class ArrayBackend(FitBackend):
    """An attempt is an array task's result, read back; a missing one is pending."""

    name = "slurm"

    def attempt(self, job: FitJob, at_least: float = 0.0) -> tuple:
        from src.rsa_open.loop.fitting import scheduler

        gb = mem_gb_for(max(at_least, scheduler().known(job.name) or 0.0))
        # A result at a lower tier stands unless the fit was killed there (what is known of a theory's
        # peak can grow between steps, which raises the tier asked for).
        for tier in [t for t in MEM_TIERS if t < gb]:
            low = result_path(job, tier)
            if low.exists() and json.loads(low.read_text())["kind"] != "killed":
                gb = tier
                break
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
    after: List[FoldAfter] = []
    results = []
    for call in calls:
        try:
            results.append(call())
        except FitPending as p:
            pending.append(p)
            results.append(None)
        except FitsPending as ps:
            pending.extend(ps.pending)
            after.extend(ps.after)
            results.append(None)
    if pending or after:
        raise FitsPending(pending, after)
    return results


# ---------- the array: its manifest and one task ----------

def job_record(p: FitPending) -> Dict[str, Any]:
    j = p.job
    return {"theory_path": str(j.theory_path), "name": j.name, "corpus_path": str(j.corpus_path),
            "settings": asdict(j.settings), "nc_path": str(j.nc_path), "meta_path": str(j.meta_path),
            "time_limit_sec": j.time_limit_sec, "mem_gb": p.mem_gb,
            # a fold's warm start (numpy scalars from the full fit's draws); float() is exact, so the
            # task starts where the local child would have, under the same fingerprint
            "init": None if j.init is None else {k: float(v) for k, v in j.init.items()}}


def after_record(a: FoldAfter) -> Dict[str, Any]:
    return {"kind": "fold_after", "theory_path": str(a.theory_path), "name": a.name, "full_corpus": str(a.full_corpus),
            "full_settings": asdict(a.full_settings), "cache_dir": str(a.cache_dir), "fold_corpus": str(a.fold_corpus),
            "fold_settings": asdict(a.fold_settings), "limit": dict(a.limit), "only_if_converged": a.only_if_converged,
            "mem_gb": a.mem_gb}


def write_after_manifest(path: Path, after: Sequence[FoldAfter]) -> Path:
    """An array of `FoldAfter` tasks, all of one memory size (no submission counts: such a task can
    only save work, and the fold is counted when the loop asks for it)."""
    sizes = {a.mem_gb for a in after}
    if len(sizes) != 1:
        raise ValueError(f"one memory size per array, not {sorted(sizes)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"mem_gb": sizes.pop(), "tasks": [after_record(a) for a in after]}, indent=1))
    os.replace(tmp, path)
    return path


def run_fold_after(r: Mapping[str, Any]) -> Dict[str, Any]:
    """A `FoldAfter` task: the fold as the loop would fit it, if its full fit is there to start from."""
    import numpy as np

    from src.rsa.loop.fitting import ModelFailure
    from src.rsa_open.loop.fitting import cached_loop_fit, loop_fit

    theory, name, cache = Path(r["theory_path"]), r["name"], Path(r["cache_dir"])
    full = cached_loop_fit(theory, name, Path(r["full_corpus"]), FitSettings(**r["full_settings"]), cache,
                           log_likelihood=False)
    if full is None:
        return {"kind": "skipped", "message": "its full fit is not in the cache (it failed or was killed)"}
    if r["only_if_converged"] and not full.converged:
        return {"kind": "skipped", "message": "its full fit did not converge, so it is not admitted"}
    start = {k: float(np.median(v)) for k, v in full.flat().items()}
    try:
        loop_fit(theory, name, Path(r["fold_corpus"]), FitSettings(**r["fold_settings"]), cache,
                 time_limit_sec=fold_limit(r["limit"], full.timing), init=start, log_likelihood=False)
    except ModelFailure as exc:
        return {"kind": "skipped", "message": f"the fold failed ({exc}); the loop fits it again when it needs it"}
    return {"kind": "ok", "message": None}


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
    if spec["tasks"][index].get("kind") == "fold_after":
        try:
            return {**run_fold_after(spec["tasks"][index]), "machine": machine()}
        except Exception as exc:  # the task's own trouble; the loop fits the fold when it needs it
            return {"kind": "infrastructure", "message": f"{type(exc).__name__}: {exc}",
                    "traceback": "".join(traceback.format_exception(exc))[-4000:]}
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


if __name__ == "__main__":  # one array task: scripts/rsa/slurm/fit_array.sbatch
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Run one fit of a step's array and write its result.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    out = run_task(a.manifest, a.index)
    print(f"[fit task] {a.index}: {out['kind']}" + (f": {out.get('message')}" if out.get("message") else ""), flush=True)
    sys.exit(1 if out["kind"] == "infrastructure" else 0)  # a theory's failure is a result, not a failed task
