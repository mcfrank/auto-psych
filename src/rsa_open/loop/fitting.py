"""Cached, time-limited corpus fits for the open loop (the closed loop's `src.rsa.loop.fitting`,
for theories fitted to many paradigms).

A corpus is written once as a JSON file (`write_corpus`): every block's link (name and frozen
options) and trials (scene, response, unit, source). A fit is content-addressed by (theory file
sha, corpus file sha, sampler settings) and stored as ``<cache>/<name>.<fingerprint>.nc`` with a
sidecar ``.json``, written under a temporary name and ``os.replace``d into place; a cached fit is
reused, never recomputed. With a time limit the fit runs in a spawned child (one core) that is
killed at the limit; the child records its warmup's sampler report beside the fit
(``<fit>.warmup.json``), so a fit killed while sampling is rejected with what its geometry was
(`src.rsa_open.sampler`). A fit that fails the convergence gate is refitted once at smaller
steps only when it is a near miss (main's rule, `src.models.mcmc_defaults` NEAR_MISS_*): a fit
far from converging has a geometry problem that a refit repeats at twice the cost. Failures are classified as the closed loop's: the theory's own (its code
raised, memo refused it, it broke the contract, it gives an observed response probability
zero) is a `ModelFailure`, remembered for the process; anything else is infrastructure and
raises.

**Memory.** Every fit child is started through `FitScheduler`: it runs only when the job's
memory holds it beside the children already running, at the most a fit of that theory has
used (its peak, recorded with the fit as ``peak_hwm_mb``) or, for a theory never fitted, at
`DEFAULT_FIT_BYTES`. A child the system kills while running (the job's out-of-memory killer) is
refitted once with twice the room; killed again, it raises as before. Open chain 1's experiment
2 lost its loop twice to the job's memory limit (2026-11-01 at 96 GB, 2026-11-02 at 180 GB):
the scoring step ran 20 fold fits at once, with nothing bounding their memory (PI 2026-11-02).
A loaded fit may leave out its per-trial log-likelihood (``log_likelihood=False``): the loop and
the design never read it, and on a whole corpus it is about 0.9 GB a fit (4 chains x 1,000
draws x 58,069 trials in float32), held in the loop's own process for every live theory.

**Where a fit ran** (`machine`): every fit's sidecar records its host, CPU model, and Slurm job
and partition. Time limits decide admission, so a fit's speed must be comparable across chains;
the record shows which hardware each fit had (`DISTRIBUTED_DESIGN.md` v2, §0).

**Backends** (`fit_backend`): a time-limited fit's run is one attempt of a backend, chosen by
``RSA_OPEN_FIT_BACKEND``. ``local`` (the default, and the only one built) fits in a child of this
job, as above. A cluster backend (Slurm jobs, or a worker queue) plugs in at `FitBackend.attempt`
once SRCC has said which is allowed; everything around it (the cache, time limits, the
out-of-memory retry, failure classification) stays as it is.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from src.rsa.context import Context
from src.rsa.cpus import CORES, job_cpus
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import (SINGLE_THREAD_XLA_FLAGS, FitTimeLimitExceeded, ModelFailure, _start_with_cache,
                                  cache_paths, is_model_failure, refit_settings)
from src.rsa_open.corpus import Block, CorpusFit, check_blocks

CORPUS_FORMAT = 1
FIT_FORMAT = 1


class ZeroProbabilityResponse(RuntimeError):
    """A theory gives an observed response probability zero (a model failure)."""


# --------------------------------------------------------------------------- memory

MEMORY_FRACTION = 0.9
"""Of the job's memory, what the loop's process and its fit children may use together."""
DEFAULT_FIT_BYTES = 10 * 2**30
"""The room reserved for a theory never fitted in this process (chain 1's fits used 5-9 GB)."""
PEAK_MARGIN = 1.25
"""A theory fitted before is given its largest recorded peak times this."""
POLL_SEC = 30.0


def _kib_to_bytes(maxrss: int) -> int:
    return maxrss if sys.platform == "darwin" else maxrss * 1024  # ru_maxrss: bytes on macOS, KiB on Linux


def _proc_status_bytes(pid: Any, key: str) -> int:
    """A /proc/<pid>/status memory line (VmRSS, VmHWM) in bytes; 0 where there is no /proc."""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _own_peak() -> int:
    """This process's own peak memory. Not ``ru_maxrss``: a spawned child's starts at the size of the
    process that spawned it, so chain 1's fits all recorded the loop's 15.8 GB, the scheduler reserved
    about 20 GB for fits of 7-10 GB, and 7 ran with 96 GB and 17 cores free (peak_rss_mb, 2026-11-02;
    that key is ignored). ``VmHWM`` starts afresh at exec."""
    return _proc_status_bytes("self", "VmHWM") or _kib_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def own_rss() -> int:
    """This process's resident memory now (the loop's own: its fits, CV results, JAX)."""
    return _proc_status_bytes("self", "VmRSS")


def job_memory_bytes() -> int:
    """The job's memory: Slurm's allocation, else the cgroup's limit, else the machine's."""
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return int(os.environ["SLURM_MEM_PER_NODE"]) * 2**20
    if os.environ.get("SLURM_MEM_PER_CPU"):
        return int(os.environ["SLURM_MEM_PER_CPU"]) * 2**20 * len(job_cpus())
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2**60:
            return int(text)
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class FitScheduler:
    """Starts fit children only while the job's memory holds them.

    A child is admitted when fewer than ``slots`` run and the room reserved for the running ones,
    its own and this process's resident memory fit in ``usable`` bytes; one child always may run
    (a theory needing more than the job runs alone). The room for a theory is its recorded peak
    (`observe`) times `PEAK_MARGIN`, or `DEFAULT_FIT_BYTES` before its first fit."""

    def __init__(self, usable_bytes: float, slots: int, rss: Callable[[], int] = own_rss) -> None:
        self.usable = float(usable_bytes)
        self.slots = max(1, int(slots))
        self._rss = rss
        self._cond = threading.Condition()
        self._running: Dict[int, float] = {}
        self._peaks: Dict[str, float] = {}
        self._next = 0

    def estimate(self, name: str) -> float:
        with self._cond:
            peak = self._peaks.get(name)
        return DEFAULT_FIT_BYTES if peak is None else PEAK_MARGIN * peak

    def observe(self, name: str, peak_bytes: float) -> None:
        if peak_bytes and peak_bytes > 0:
            with self._cond:
                self._peaks[name] = max(self._peaks.get(name, 0.0), float(peak_bytes))

    def running(self) -> int:
        with self._cond:
            return len(self._running)

    @contextmanager
    def room(self, name: str, at_least: float = 0.0) -> Iterator[float]:
        """Wait for room for one of ``name``'s fits (at least ``at_least`` bytes); hold it in the block."""
        need = max(self.estimate(name), float(at_least))
        with self._cond:
            while self._running and (len(self._running) >= self.slots
                                     or sum(self._running.values()) + need + self._rss() > self.usable):
                self._cond.wait(timeout=POLL_SEC)  # also re-reads this process's memory as it changes
            token, self._next = self._next, self._next + 1
            self._running[token] = need
        try:
            yield need
        finally:
            with self._cond:
                del self._running[token]
                self._cond.notify_all()


_SCHEDULER: Optional[FitScheduler] = None
_SCHEDULER_LOCK = threading.Lock()


def scheduler() -> FitScheduler:
    """The process's scheduler: `MEMORY_FRACTION` of the job's memory, one child per core of the job."""
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = FitScheduler(MEMORY_FRACTION * job_memory_bytes(), len(job_cpus()))
        return _SCHEDULER


# --------------------------------------------------------------------------- the corpus file

def _context_json(ctx: Context) -> Dict[str, Any]:
    out = {}
    for f in fields(Context):
        v = getattr(ctx, f.name)
        out[f.name] = [list(r) for r in v] if f.name == "objects" else (list(v) if isinstance(v, tuple) else v)
    return out


def write_corpus(path: Path, blocks: Sequence[Block]) -> Path:
    """The blocks as one JSON file, written once (an existing file must hold the same corpus)."""
    check_blocks(blocks)
    doc = {"format": CORPUS_FORMAT, "blocks": [{
        "name": b.name, "link": {"name": b.link.name, **b.link.options},
        **({"fixed": b.fixed_params} if b.fixed else {}),
        "trials": [{"scene": _context_json(c), "response": dict(r), "unit": u,
                    "source": b.sources[i] if b.sources else None}
                   for i, (c, r, u) in enumerate(zip(b.contexts, b.responses, b.units))]} for b in blocks]}
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"{path} holds another corpus; a corpus file is written once")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def read_corpus(path: Path) -> List[Block]:
    from src.rsa_open.links import likelihood

    doc = json.loads(Path(path).read_text())
    if doc.get("format") != CORPUS_FORMAT:
        raise ValueError(f"{path}: corpus format {doc.get('format')}, not {CORPUS_FORMAT}")
    blocks = []
    for b in doc["blocks"]:
        link_spec = dict(b["link"])
        link = likelihood(link_spec.pop("name"), **link_spec)
        trials = b["trials"]
        sources = tuple(t["source"] for t in trials) if trials and trials[0]["source"] is not None else ()
        blocks.append(Block(b["name"], link, tuple(Context(**t["scene"]) for t in trials),
                            tuple(t["response"] for t in trials), tuple(t["unit"] for t in trials), sources,
                            tuple(sorted(b.get("fixed", {}).items()))))
    return blocks


# --------------------------------------------------------------------------- cached fits

def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(theory_path: Path, corpus_path: Path, settings: FitSettings,
                init: Optional[Dict[str, float]] = None) -> str:
    h = hashlib.sha256()
    h.update(f"open-format{FIT_FORMAT}".encode())
    h.update(_sha_file(theory_path).encode())
    h.update(_sha_file(corpus_path).encode())
    h.update(json.dumps(asdict(settings), sort_keys=True).encode())
    if init:  # a warm-started fit is keyed by where it started (a cold fit's key is unchanged)
        h.update(json.dumps({k: round(float(v), 6) for k, v in sorted(init.items())}).encode())
    return h.hexdigest()[:20]


def _model_failure_types() -> tuple:
    from memo.core import MemoError

    from src.rsa.model_file import ModelContractViolation

    return (ModelContractViolation, ZeroProbabilityResponse, MemoError, SyntaxError)


def _is_model_failure(exc: BaseException, theory_path: Path) -> bool:
    if isinstance(exc, (OSError, MemoryError)):
        return False
    return isinstance(exc, _model_failure_types()) or is_model_failure(exc, theory_path)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine() or "unknown"


def machine() -> Dict[str, Optional[str]]:
    """Where this process runs: host, CPU model, Slurm job and partition (None outside Slurm)."""
    import socket

    return {"host": socket.gethostname(), "cpu": _cpu_model(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION")}


def _write(fitted: CorpusFit, nc_path: Path, meta_path: Path, record_peak: bool = False) -> None:
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = nc_path.with_suffix(f".tmp{os.getpid()}.nc")
    fitted.idata.to_netcdf(str(tmp))
    os.replace(tmp, nc_path)
    meta = dict(theory_name=fitted.theory_name, blocks=fitted.blocks, param_names=fitted.param_names,
                convergence_problems=fitted.convergence_problems,
                offsets={k: list(v) for k, v in fitted.offsets.items()}, timing=fitted.timing, machine=machine())
    if record_peak:  # a fit child's peak memory, its write included (an in-process fit's would be the loop's)
        meta["peak_hwm_mb"] = round(_own_peak() / 2**20)
    tmp_meta = meta_path.with_suffix(f".tmp{os.getpid()}.json")
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_meta, meta_path)


def _without_log_likelihood(idata: Any) -> Any:
    import arviz as az

    return az.InferenceData(**{g: getattr(idata, g) for g in idata.groups() if g != "log_likelihood"})


def _read(nc_path: Path, meta_path: Path, log_likelihood: bool = True) -> CorpusFit:
    """A cached fit; without its per-trial log-likelihood when ``log_likelihood`` is False. A
    recorded peak informs the scheduler's room for the theory's next fits (its CV folds)."""
    import arviz as az

    meta = json.loads(meta_path.read_text())
    idata = az.from_netcdf(str(nc_path))
    if not log_likelihood:
        idata = _without_log_likelihood(idata)
    idata.load()
    if meta.get("peak_hwm_mb"):
        scheduler().observe(meta["theory_name"], meta["peak_hwm_mb"] * 2**20)
    return CorpusFit(meta["theory_name"], meta["blocks"], idata, meta["param_names"], meta["convergence_problems"],
                     {k: tuple(v) for k, v in meta["offsets"].items()}, meta.get("timing", {}))


def warmup_path(nc_path: Path) -> Path:
    return Path(nc_path).with_suffix(".warmup.json")


def _fit_now(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings,
             init: Optional[Dict[str, float]] = None, progress: Optional[Path] = None) -> CorpusFit:
    from src.rsa_open.corpus import fit_corpus
    from src.rsa_open.fit import TheoryModel

    def on_warmup(record: Dict[str, Any]) -> None:
        progress.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(record))
        os.replace(tmp, progress)

    fitted = fit_corpus(TheoryModel(theory_path, name=name), read_corpus(corpus_path), settings, init=init,
                        on_warmup=on_warmup if progress is not None else None)
    ll = np.asarray(fitted.idata.log_likelihood["response"])
    if not np.all(np.isfinite(ll)):
        bad = int(np.sum(~np.all(np.isfinite(ll.reshape(-1, ll.shape[-1])), axis=0)))
        raise ZeroProbabilityResponse(f"{name} gives {bad} observed responses probability zero (or non-finite)")
    return fitted


CHILD_OOM_SCORE_ADJ = "500"
"""A fit child asks to be the out-of-memory killer's first choice (raising its own score needs no
privilege): a killed child is fitted again, while the loop's own process holds the run."""


def _child(theory_path, name, corpus_path, settings, nc_path, meta_path, queue, init=None) -> None:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + SINGLE_THREAD_XLA_FLAGS).strip()
    try:
        Path("/proc/self/oom_score_adj").write_text(CHILD_OOM_SCORE_ADJ)
    except OSError:
        pass  # no /proc (macOS)
    try:
        _write(_fit_now(Path(theory_path), name, Path(corpus_path), settings, init, warmup_path(Path(nc_path))),
               Path(nc_path), Path(meta_path), record_peak=True)
        queue.put(("ok", None, None))
    except BaseException as exc:
        queue.put(("model" if _is_model_failure(exc, Path(theory_path)) else "infrastructure",
                   f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc))[-4000:]))


_FAILURES: Dict[str, ModelFailure] = {}
_FAILURES_LOCK = threading.Lock()


def fit_cached(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings, cache_dir: Path, *,
               time_limit_sec: Optional[float] = None, init: Optional[Dict[str, float]] = None,
               log_likelihood: bool = True) -> CorpusFit:
    """The theory's fit to the corpus, from the cache or fitted now (in a time-limited child with
    ``time_limit_sec``). ``init``: start every chain there (the constrained values of the sampled
    sites; a cross-validation fold starts at its full fit's posterior). ``log_likelihood`` False:
    the fit without its per-trial log-likelihood (the cached file keeps it). Raises `ModelFailure`
    for the theory's failure (remembered), and RuntimeError for any other."""
    fp = fingerprint(theory_path, corpus_path, settings, init)
    nc_path, meta_path = cache_paths(cache_dir, name, fp)
    with _FAILURES_LOCK:
        failed = _FAILURES.get(str(nc_path))
    if failed is not None:
        raise type(failed)(str(failed))
    try:
        if nc_path.exists() and meta_path.exists():
            return _read(nc_path, meta_path, log_likelihood)
        if time_limit_sec is None:
            try:
                fitted = _fit_now(theory_path, name, corpus_path, settings, init)
            except Exception as exc:
                if _is_model_failure(exc, theory_path):
                    raise ModelFailure(f"{type(exc).__name__}: {exc}") from exc
                raise
            _write(fitted, nc_path, meta_path)
            if not log_likelihood:
                fitted.idata = _without_log_likelihood(fitted.idata)
            return fitted
        return _fit_in_child(theory_path, name, corpus_path, settings, nc_path, meta_path, time_limit_sec, init,
                             log_likelihood)
    except ModelFailure as exc:
        with _FAILURES_LOCK:
            _FAILURES[str(nc_path)] = exc
        raise


class _Killed(Exception):
    """A fit child ended by a signal without reporting (the job's out-of-memory killer, as a rule)."""

    def __init__(self, exitcode: Optional[int], peak: int, reserved: float = 0.0) -> None:
        super().__init__(exitcode)
        self.exitcode, self.peak, self.reserved = exitcode, peak, reserved


def _run_child(proc, queue, name: str, nc_path: Path, time_limit_sec: float) -> tuple:
    """Start ``proc`` on a core of its own, wait for it (killed at the limit) and return its report."""
    core = CORES.take()
    peak = 0
    try:
        with tempfile.TemporaryDirectory(prefix="rsa-open-fit-cache-") as own_cache:
            _start_with_cache(proc, own_cache, core)
            deadline = None if math.isinf(time_limit_sec) else time.monotonic() + time_limit_sec
            while proc.is_alive():
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    proc.kill()
                    proc.join()
                    raise FitTimeLimitExceeded(f"{name}: the fit did not finish within {time_limit_sec / 60:.0f} "
                                               f"minutes" + _warmup_note(Path(nc_path)))
                proc.join(POLL_SEC if left is None else min(POLL_SEC, left))
                peak = max(peak, _proc_status_bytes(proc.pid, "VmHWM"))
            if queue.empty():
                if proc.exitcode == -signal.SIGKILL:
                    raise _Killed(proc.exitcode, peak)
                raise RuntimeError(f"{name}: the fit process exited with code {proc.exitcode} without reporting: "
                                   f"an infrastructure failure, not the theory's")
            return queue.get()
    finally:
        CORES.give_back(core)


@dataclass(frozen=True)
class FitJob:
    """One time-limited fit, as a backend runs it: everything the fit's child needs."""
    theory_path: Path
    name: str
    corpus_path: Path
    settings: FitSettings
    nc_path: Path
    meta_path: Path
    time_limit_sec: float
    init: Optional[Dict[str, float]] = None


class FitBackend:
    """Runs one attempt of a fit and returns its child's report, ``(kind, message, traceback)``
    with ``kind`` "ok" (the fit is in the cache), "model" or "infrastructure"; raises `_Killed`
    when the attempt was killed without reporting (out of memory, as a rule), with the room it had."""

    name = "abstract"

    def attempt(self, job: FitJob, at_least: float = 0.0) -> tuple:
        raise NotImplementedError


class LocalBackend(FitBackend):
    """A spawned child of this job, on a core of its own, started when the job's memory holds it
    (`FitScheduler`) and killed at the fit's time limit."""

    name = "local"

    def attempt(self, job: FitJob, at_least: float = 0.0) -> tuple:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_child, args=(str(job.theory_path), job.name, str(job.corpus_path), job.settings,
                                                str(job.nc_path), str(job.meta_path), queue, job.init))
        with scheduler().room(job.name, at_least) as reserved:
            try:
                return _run_child(proc, queue, job.name, job.nc_path, job.time_limit_sec)
            except _Killed as killed:
                killed.reserved = reserved
                raise


FIT_BACKEND_ENV = "RSA_OPEN_FIT_BACKEND"
PLANNED_BACKENDS = ("slurm", "queue")


def fit_backend() -> FitBackend:
    """The backend ``RSA_OPEN_FIT_BACKEND`` names (default ``local``). A planned one that isn't
    built yet, or an unknown name, raises: a run never falls back to another backend silently."""
    name = os.environ.get(FIT_BACKEND_ENV, "local") or "local"
    if name == "local":
        return LocalBackend()
    if name in PLANNED_BACKENDS:
        raise NotImplementedError(f"{FIT_BACKEND_ENV}={name}: that backend is planned, not built "
                                  "(docs/auto_open/DISTRIBUTED_DESIGN.md); unset it to fit locally")
    raise ValueError(f"{FIT_BACKEND_ENV}={name!r}: not a fit backend (local; planned: {', '.join(PLANNED_BACKENDS)})")


def _fit_in_child(theory_path, name, corpus_path, settings, nc_path, meta_path, time_limit_sec,
                  init=None, log_likelihood: bool = True) -> CorpusFit:
    backend, at_least = fit_backend(), 0.0
    job = FitJob(Path(theory_path), name, Path(corpus_path), settings, Path(nc_path), Path(meta_path),
                 time_limit_sec, init)
    for attempt in (1, 2):
        try:
            kind, message, tb = backend.attempt(job, at_least)
            break
        except _Killed as killed:
            scheduler().observe(name, killed.peak)
            if attempt == 2:
                raise RuntimeError(
                    f"{name}: the fit process was killed (exit code {killed.exitcode}) twice without reporting, "
                    f"the second time with {killed.reserved / 2**30:.0f} GB reserved for it (out of memory?): an "
                    f"infrastructure failure, not the theory's") from None
            at_least = max(2 * killed.reserved, PEAK_MARGIN * killed.peak)
            print(f"[fit] {name}: the fit process was killed at {killed.peak / 2**30:.1f} GB (out of memory?) "
                  f"with {killed.reserved / 2**30:.0f} GB reserved; fitting it again with "
                  f"{at_least / 2**30:.0f} GB", flush=True)
    if kind == "ok":
        return _read(nc_path, meta_path, log_likelihood)
    if kind == "model":
        raise ModelFailure(message)
    raise RuntimeError(f"{name}: infrastructure failure while fitting: {message}\n{tb}")


def _warmup_note(nc_path: Path) -> str:
    """What a killed fit's warmup said (`src.rsa_open.sampler`), or that warmup never ended."""
    from src.rsa_open.sampler import SamplerReport

    path = warmup_path(nc_path)
    if not path.exists():
        return " (it was still in warmup, adapting its step size: the posterior is hard to explore)"
    record = json.loads(path.read_text())
    report = SamplerReport.from_json(record["report"])
    return (f" (warmup took {record['timing']['warmup_sec'] / 60:.0f} minutes and it was killed while sampling. "
            f"{report.text()})")


def cached_loop_fit(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings,
                    cache_dir: Path, log_likelihood: bool = True) -> Optional[CorpusFit]:
    """What `loop_fit` would return, if the cache already holds it; never fits (None otherwise)."""
    nc, meta = cache_paths(cache_dir, name, fingerprint(theory_path, corpus_path, settings))
    if not (nc.exists() and meta.exists()):
        return None
    first = _read(nc, meta, log_likelihood)
    if first.converged or not first.near_miss:
        return first
    nc, meta = cache_paths(cache_dir, name, fingerprint(theory_path, corpus_path, refit_settings(settings)))
    return _read(nc, meta, log_likelihood) if nc.exists() and meta.exists() else None


def loop_fit(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings, cache_dir: Path, *,
             time_limit_sec: Optional[float] = None, init: Optional[Dict[str, float]] = None,
             log_likelihood: bool = True) -> CorpusFit:
    """The fit the loop uses: at ``settings``, or, when that fails the convergence gate as a near
    miss (`CorpusFit.near_miss`), the one refit (`src.rsa.loop.fitting.refit_settings`), from
    ``init`` if given. A fit far from converging is returned as it is."""
    fitted = fit_cached(theory_path, name, corpus_path, settings, cache_dir, time_limit_sec=time_limit_sec, init=init,
                        log_likelihood=log_likelihood)
    if fitted.converged or not fitted.near_miss:
        return fitted
    return fit_cached(theory_path, name, corpus_path, refit_settings(settings), cache_dir,
                      time_limit_sec=time_limit_sec, init=init, log_likelihood=log_likelihood)
