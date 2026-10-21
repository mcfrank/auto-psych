"""Cached, time-limited corpus fits for the open loop (the closed loop's `src.rsa.loop.fitting`,
for theories fitted to many paradigms).

A corpus is written once as a JSON file (`write_corpus`): every block's link (name and frozen
options) and trials (scene, response, unit, source). A fit is content-addressed by (theory file
sha, corpus file sha, sampler settings) and stored as ``<cache>/<name>.<fingerprint>.nc`` with a
sidecar ``.json``, written under a temporary name and ``os.replace``d into place; a cached fit is
reused, never recomputed. With a time limit the fit runs in a spawned child (one core) that is
killed at the limit. Failures are classified as the closed loop's: the theory's own (its code
raised, memo refused it, it broke the contract, it gives an observed response probability
zero) is a `ModelFailure`, remembered for the process; anything else is infrastructure and
raises.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import threading
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rsa.context import Context
from src.rsa.cpus import CORES
from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import (SINGLE_THREAD_XLA_FLAGS, FitTimeLimitExceeded, ModelFailure, _start_with_cache,
                                  cache_paths, is_model_failure, refit_settings)
from src.rsa_open.corpus import Block, CorpusFit, check_blocks

CORPUS_FORMAT = 1
FIT_FORMAT = 1


class ZeroProbabilityResponse(RuntimeError):
    """A theory gives an observed response probability zero (a model failure)."""


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
                            tuple(t["response"] for t in trials), tuple(t["unit"] for t in trials), sources))
    return blocks


# --------------------------------------------------------------------------- cached fits

def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(theory_path: Path, corpus_path: Path, settings: FitSettings) -> str:
    h = hashlib.sha256()
    h.update(f"open-format{FIT_FORMAT}".encode())
    h.update(_sha_file(theory_path).encode())
    h.update(_sha_file(corpus_path).encode())
    h.update(json.dumps(asdict(settings), sort_keys=True).encode())
    return h.hexdigest()[:20]


def _model_failure_types() -> tuple:
    from memo.core import MemoError

    from src.rsa.model_file import ModelContractViolation

    return (ModelContractViolation, ZeroProbabilityResponse, MemoError, SyntaxError)


def _is_model_failure(exc: BaseException, theory_path: Path) -> bool:
    if isinstance(exc, (OSError, MemoryError)):
        return False
    return isinstance(exc, _model_failure_types()) or is_model_failure(exc, theory_path)


def _write(fitted: CorpusFit, nc_path: Path, meta_path: Path) -> None:
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = nc_path.with_suffix(f".tmp{os.getpid()}.nc")
    fitted.idata.to_netcdf(str(tmp))
    os.replace(tmp, nc_path)
    meta = dict(theory_name=fitted.theory_name, blocks=fitted.blocks, param_names=fitted.param_names,
                convergence_problems=fitted.convergence_problems,
                offsets={k: list(v) for k, v in fitted.offsets.items()})
    tmp_meta = meta_path.with_suffix(f".tmp{os.getpid()}.json")
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_meta, meta_path)


def _read(nc_path: Path, meta_path: Path) -> CorpusFit:
    import arviz as az

    meta = json.loads(meta_path.read_text())
    idata = az.from_netcdf(str(nc_path))
    idata.load()
    return CorpusFit(meta["theory_name"], meta["blocks"], idata, meta["param_names"], meta["convergence_problems"],
                     {k: tuple(v) for k, v in meta["offsets"].items()})


def _fit_now(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings) -> CorpusFit:
    from src.rsa_open.corpus import fit_corpus
    from src.rsa_open.fit import TheoryModel

    fitted = fit_corpus(TheoryModel(theory_path, name=name), read_corpus(corpus_path), settings)
    ll = np.asarray(fitted.idata.log_likelihood["response"])
    if not np.all(np.isfinite(ll)):
        bad = int(np.sum(~np.all(np.isfinite(ll.reshape(-1, ll.shape[-1])), axis=0)))
        raise ZeroProbabilityResponse(f"{name} gives {bad} observed responses probability zero (or non-finite)")
    return fitted


def _child(theory_path, name, corpus_path, settings, nc_path, meta_path, queue) -> None:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + SINGLE_THREAD_XLA_FLAGS).strip()
    try:
        _write(_fit_now(Path(theory_path), name, Path(corpus_path), settings), Path(nc_path), Path(meta_path))
        queue.put(("ok", None, None))
    except BaseException as exc:
        queue.put(("model" if _is_model_failure(exc, Path(theory_path)) else "infrastructure",
                   f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc))[-4000:]))


_FAILURES: Dict[str, ModelFailure] = {}
_FAILURES_LOCK = threading.Lock()


def fit_cached(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings, cache_dir: Path, *,
               time_limit_sec: Optional[float] = None) -> CorpusFit:
    """The theory's fit to the corpus, from the cache or fitted now (in a time-limited child with
    ``time_limit_sec``). Raises `ModelFailure` for the theory's failure (remembered), and
    RuntimeError for any other."""
    fp = fingerprint(theory_path, corpus_path, settings)
    nc_path, meta_path = cache_paths(cache_dir, name, fp)
    with _FAILURES_LOCK:
        failed = _FAILURES.get(str(nc_path))
    if failed is not None:
        raise type(failed)(str(failed))
    try:
        if nc_path.exists() and meta_path.exists():
            return _read(nc_path, meta_path)
        if time_limit_sec is None:
            try:
                fitted = _fit_now(theory_path, name, corpus_path, settings)
            except Exception as exc:
                if _is_model_failure(exc, theory_path):
                    raise ModelFailure(f"{type(exc).__name__}: {exc}") from exc
                raise
            _write(fitted, nc_path, meta_path)
            return fitted
        return _fit_in_child(theory_path, name, corpus_path, settings, nc_path, meta_path, time_limit_sec)
    except ModelFailure as exc:
        with _FAILURES_LOCK:
            _FAILURES[str(nc_path)] = exc
        raise


def _fit_in_child(theory_path, name, corpus_path, settings, nc_path, meta_path, time_limit_sec) -> CorpusFit:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(str(theory_path), name, str(corpus_path), settings, str(nc_path),
                                            str(meta_path), queue))
    core = CORES.take()
    try:
        with tempfile.TemporaryDirectory(prefix="rsa-open-fit-cache-") as own_cache:
            _start_with_cache(proc, own_cache, core)
            proc.join(None if math.isinf(time_limit_sec) else time_limit_sec)
            if proc.is_alive():
                proc.kill()
                proc.join()
                raise FitTimeLimitExceeded(f"{name}: the fit did not finish within {time_limit_sec / 60:.0f} minutes")
            if queue.empty():
                raise RuntimeError(f"{name}: the fit process exited with code {proc.exitcode} without reporting "
                                   f"(killed? out of memory?): an infrastructure failure, not the theory's")
            kind, message, tb = queue.get()
    finally:
        CORES.give_back(core)
    if kind == "ok":
        return _read(nc_path, meta_path)
    if kind == "model":
        raise ModelFailure(message)
    raise RuntimeError(f"{name}: infrastructure failure while fitting: {message}\n{tb}")


def loop_fit(theory_path: Path, name: str, corpus_path: Path, settings: FitSettings, cache_dir: Path, *,
             time_limit_sec: Optional[float] = None) -> CorpusFit:
    """The fit the loop uses: at ``settings``, or the one refit when that fails the convergence
    gate (`src.rsa.loop.fitting.refit_settings`)."""
    fitted = fit_cached(theory_path, name, corpus_path, settings, cache_dir, time_limit_sec=time_limit_sec)
    if fitted.converged:
        return fitted
    return fit_cached(theory_path, name, corpus_path, refit_settings(settings), cache_dir,
                      time_limit_sec=time_limit_sec)
