"""Cached, time-limited fits of memo model files for the RSA loop.

A fit is content-addressed by (model file sha, responses file sha, sampler
settings) and stored as ``<cache>/<name>.<fingerprint>.nc`` with a sidecar
``.json`` (parameter names, convergence problems). A cached fit is reused,
never recomputed; files are written under a temporary name and
``os.replace``d into place.

A fit can run in a spawned child process with a time limit (agent-written
candidates: a pathological model must not stall the loop). Failures are
classified as in the PyMC domain: a failure of the model itself (its own
code raised, memo refused to compile it, it broke the contract, it gives an
observed choice probability zero) is a `ModelFailure` the caller turns into a
rejection; anything else (a killed worker, an unreadable file, OSError,
MemoryError) is infrastructure and raises.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rsa.cpus import CORES, pinned_thread
from src.rsa.fit import FitSettings, RSAFit, ZeroProbabilityChoice

FIT_TIME_LIMIT_SEC = 30 * 60  # as the PyMC domain's CANDIDATE_FIT_TIME_LIMIT_SEC
# A fit child runs single-threaded: measured on the combined data (40k trials,
# 2026-10-07), one fit takes ~55 s on one thread and ~53 s on four, and four
# single-threaded fits at once on four cores ~75 s each. So the loop runs one
# fit per CPU (`src.rsa.loop.orchestrator.default_fit_workers`).
SINGLE_THREAD_XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Part of every fingerprint: bump when what a cached fit holds changes (2:
# pattern-level log-likelihood, src.rsa.fit), so an old file is never read
# as a new one.
FIT_FORMAT = 2


class ModelFailure(RuntimeError):
    """The model failed (a rejection), as opposed to the infrastructure."""


class FitTimeLimitExceeded(ModelFailure):
    """The fit did not finish within the time limit (the model is too slow)."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(model_path: Path, responses_path: Path, settings: FitSettings) -> str:
    h = hashlib.sha256()
    h.update(f"format{FIT_FORMAT}".encode())
    h.update(_sha_file(model_path).encode())
    h.update(_sha_file(responses_path).encode())
    key = asdict(settings)
    if not key["dense_mass"]:
        del key["dense_mass"]  # the default: cache keys from before the setting existed stay valid
    h.update(json.dumps(key, sort_keys=True).encode())
    return h.hexdigest()[:20]


def cache_paths(cache_dir: Path, name: str, fp: str) -> tuple[Path, Path]:
    # Not Path.with_suffix: it would replace ".<fp>" and key the cache by name
    # alone (until 2026-10-07 it did: a refit read back the fit it replaces).
    base = f"{name}.{fp}"
    return Path(cache_dir) / f"{base}.nc", Path(cache_dir) / f"{base}.json"


def _model_failure_types() -> tuple:
    from memo.core import MemoError

    from src.rsa.model_file import ModelContractViolation

    return (ModelContractViolation, ZeroProbabilityChoice, MemoError, SyntaxError)


def is_model_failure(exc: BaseException, model_path: Path) -> bool:
    """A failure of the model: a known model-failure type, or raised in its file."""
    if isinstance(exc, (OSError, MemoryError)):
        return False
    if isinstance(exc, _model_failure_types()):
        return True
    model_path = str(Path(model_path).resolve())
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and any(str(Path(f.filename).resolve()) == model_path for f in frames)


def _write(fitted: RSAFit, nc_path: Path, meta_path: Path) -> None:
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = nc_path.with_suffix(f".tmp{os.getpid()}.nc")
    fitted.idata.to_netcdf(str(tmp))
    os.replace(tmp, nc_path)
    meta = dict(
        model_name=fitted.model_name,
        param_names=fitted.param_names,
        convergence_problems=fitted.convergence_problems,
    )
    tmp_meta = meta_path.with_suffix(f".tmp{os.getpid()}.json")
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_meta, meta_path)


def _read(nc_path: Path, meta_path: Path) -> RSAFit:
    import arviz as az

    meta = json.loads(meta_path.read_text())
    idata = az.from_netcdf(str(nc_path))
    idata.load()
    return RSAFit(
        model_name=meta["model_name"],
        idata=idata,
        param_names=meta["param_names"],
        convergence_problems=meta["convergence_problems"],
    )


def _fit_now(model_path: Path, name: str, responses_path: Path, settings: FitSettings) -> RSAFit:
    from src.rsa.dataset import load_forced_choice
    from src.rsa.fit import fit
    from src.rsa.model_file import RSAModel

    trials = load_forced_choice(responses_path)
    return fit(RSAModel(model_path, name=name), trials.contexts, trials.choices, settings)


def _child(model_path, name, responses_path, settings, nc_path, meta_path, queue) -> None:
    # Before JAX's first computation, which reads the flags.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + SINGLE_THREAD_XLA_FLAGS).strip()
    try:
        fitted = _fit_now(Path(model_path), name, Path(responses_path), settings)
        _write(fitted, Path(nc_path), Path(meta_path))
        queue.put(("ok", None, None))
    except BaseException as exc:  # reported to the parent, which classifies it
        queue.put(
            (
                "model" if is_model_failure(exc, Path(model_path)) else "infrastructure",
                f"{type(exc).__name__}: {exc}",
                "".join(traceback.format_exception(exc))[-4000:],
            )
        )


# A model's failure, by the fit it was for, for the life of the process: the
# loop fits a round's candidates concurrently before admitting them one by
# one, and a failed (or timed-out) fit must not be run a second time at
# admission. Infrastructure failures are never remembered.
_FAILURES: Dict[str, ModelFailure] = {}
_FAILURES_LOCK = threading.Lock()


def fit_cached(
    model_path: Path,
    name: str,
    responses_path: Path,
    settings: FitSettings,
    cache_dir: Path,
    *,
    time_limit_sec: Optional[float] = None,
) -> RSAFit:
    """The model's fit to the responses, from the cache or fitted now.

    With ``time_limit_sec`` the fit runs in a spawned child process that is
    killed at the limit (`FitTimeLimitExceeded`). Raises `ModelFailure` for a
    failure of the model (remembered: the same fit raises it again without
    running), and RuntimeError for any other failure.
    """
    fp = fingerprint(model_path, responses_path, settings)
    key = str(cache_paths(cache_dir, name, fp)[0])
    with _FAILURES_LOCK:
        failed = _FAILURES.get(key)
    if failed is not None:
        raise type(failed)(str(failed))
    try:
        return _fit_cached(model_path, name, responses_path, settings, cache_dir, time_limit_sec=time_limit_sec)
    except ModelFailure as exc:
        with _FAILURES_LOCK:
            _FAILURES[key] = exc
        raise


def _fit_cached(
    model_path: Path,
    name: str,
    responses_path: Path,
    settings: FitSettings,
    cache_dir: Path,
    *,
    time_limit_sec: Optional[float] = None,
) -> RSAFit:
    fp = fingerprint(model_path, responses_path, settings)
    nc_path, meta_path = cache_paths(cache_dir, name, fp)
    if nc_path.exists() and meta_path.exists():
        return _read(nc_path, meta_path)
    if time_limit_sec is None:
        try:
            fitted = _fit_now(model_path, name, responses_path, settings)
        except Exception as exc:
            if is_model_failure(exc, model_path):
                raise ModelFailure(f"{type(exc).__name__}: {exc}") from exc
            raise
        _write(fitted, nc_path, meta_path)
        return fitted

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_child,
        args=(str(model_path), name, str(responses_path), settings, str(nc_path), str(meta_path), queue),
    )
    core = CORES.take()
    try:
        with tempfile.TemporaryDirectory(prefix="rsa-fit-cache-") as own_cache:
            _start_with_cache(proc, own_cache, core)
            return _finish_child(proc, queue, name, nc_path, meta_path, time_limit_sec)
    finally:
        CORES.give_back(core)


# The environment a spawned child starts with is the parent's at that moment.
_SPAWN_LOCK = threading.Lock()


def _start_with_cache(proc, cache_dir: str, core: Optional[int] = None) -> None:
    """Start a fit child with ``XDG_CACHE_HOME`` of its own, on one core.

    The child sees ``core`` alone from its first instruction (`src.rsa.cpus`):
    started from a thread pinned to it, its JAX and OpenBLAS pools are one
    core's, about 7 threads instead of 7 per core of the job (Research
    Computing's warning on job 47042590, 2026-10-08).

    On import, arviz writes a once-a-day marker, ``<XDG cache>/arviz/
    daily_warning``, through a fixed-name ``daily_warning.tmp``. Fit children
    that import it together after midnight collide on that file and die
    (three of Sherlock run 2's cells, 2026-10-08; the SR harness's
    ``_fit_process_caches`` fixed the same bug). A child imports arviz before
    any code of ours runs in it (spawn re-imports the main module), so the
    directory has to be in the environment it starts with.
    """
    with _SPAWN_LOCK:
        before = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = cache_dir
        try:
            if core is None:
                proc.start()
            else:
                with pinned_thread(core):
                    proc.start()
        finally:
            if before is None:
                del os.environ["XDG_CACHE_HOME"]
            else:
                os.environ["XDG_CACHE_HOME"] = before


def _finish_child(proc, queue, name: str, nc_path: Path, meta_path: Path, time_limit_sec: float) -> RSAFit:
    proc.join(time_limit_sec)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise FitTimeLimitExceeded(
            f"{name}: the fit did not finish within {time_limit_sec / 60:.0f} minutes"
        )
    if queue.empty():
        raise RuntimeError(
            f"{name}: the fit process exited with code {proc.exitcode} without reporting "
            f"(killed? out of memory?) — an infrastructure failure, not the model's"
        )
    kind, message, tb = queue.get()
    if kind == "ok":
        return _read(nc_path, meta_path)
    if kind == "model":
        raise ModelFailure(message)
    raise RuntimeError(f"{name}: infrastructure failure while fitting: {message}\n{tb}")


REFIT_TARGET_ACCEPT = 0.95


def refit_settings(settings: FitSettings) -> FitSettings:
    """The one refit of a fit that fails the convergence gate: target_accept
    0.95 and a seed of its own."""
    return replace(settings, target_accept=max(settings.target_accept, REFIT_TARGET_ACCEPT), seed=settings.seed + 1)


def loop_fit(model_path: Path, name: str, responses_path: Path, settings: FitSettings, cache_dir: Path,
             *, time_limit_sec: Optional[float] = None) -> RSAFit:
    """The fit the loop uses for a model: the fit at ``settings``, or, when it
    fails the convergence gate, the refit (`refit_settings`). Everything that
    scores the loop's models (admission, held-out, recovery) goes through
    this, so all of them read the same cached fit."""
    fitted = fit_cached(model_path, name, responses_path, settings, cache_dir, time_limit_sec=time_limit_sec)
    if fitted.converged:
        return fitted
    return fit_cached(model_path, name, responses_path, refit_settings(settings), cache_dir,
                      time_limit_sec=time_limit_sec)


def cached_fits(cache_dir: Path) -> List[Path]:
    return sorted(Path(cache_dir).glob("*.nc"))


def describe(fitted: RSAFit) -> dict[str, Any]:
    return dict(name=fitted.model_name, converged=fitted.converged, problems=fitted.convergence_problems)
