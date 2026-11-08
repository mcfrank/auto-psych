"""The open-paradigm campaign (docs/auto_open/PLAN.md): paradigm-free theories, the links that
map them to each response type, and the port of the closed campaign's models to theory + link.

Importing it gives XLA's CPU pool at least two threads (`XLA_POOL_THREADS`). The corpus fit's
sampler calls back into a separately compiled gradient (`corpus._callback_potential`): XLA runs the
callback on a thread of its pool, and the gradient's program needs another. XLA sizes the pool by
the cores the process sees, and every process that computes sees one (`src/rsa/cpus.py`, SRCC's
thread rule), so on Sherlock every fit hung (review V3 §14, 2026-11-08). XLA's CPU client reads the
size from ``PJRT_NPROC`` when it starts, so it is set here, before any of this package computes;
the cores a process may use are unchanged. A fit refuses to start where the pool is smaller
(`xla_pool_problem`) rather than hang.
"""

import os as _os

XLA_POOL_THREADS = 2

try:
    from jax._src.xla_bridge import backends_are_initialized as _started
except ImportError:  # pragma: no cover - jax is pinned; a rename should fail the guard, not the import
    def _started() -> bool:
        return True

_STARTED_BEFORE_IMPORT = _started()
_PRESET = _os.environ.get("PJRT_NPROC")
_os.environ.setdefault("PJRT_NPROC", str(XLA_POOL_THREADS))


def xla_pool_threads() -> int:
    """The threads of XLA's CPU pool in this process, as far as can be known: ``PJRT_NPROC`` when
    the pool was made (here, unless JAX had already started), else the cores the process sees."""
    value = _PRESET if _STARTED_BEFORE_IMPORT else _os.environ.get("PJRT_NPROC")
    if value:
        return int(value)
    return len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 1)


def xla_pool_problem() -> str:
    """Why a callback fit would deadlock here, or "" when it would not."""
    n = xla_pool_threads()
    if n >= XLA_POOL_THREADS:
        return ""
    return (f"XLA's CPU pool has {n} thread(s); the corpus fit's callback needs {XLA_POOL_THREADS} or it "
            f"deadlocks. Set PJRT_NPROC={XLA_POOL_THREADS} before JAX starts (importing src.rsa_open "
            f"before any JAX computation does this)")
