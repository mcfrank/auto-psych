"""On one core the corpus fit's callback sampler deadlocked (review V3 §14, 2026-11-08): XLA runs
the callback on a thread of its CPU pool, the callback's gradient needs another, and XLA sizes the
pool by the cores a process sees, which is one for every process that computes on Sherlock
(src/rsa/cpus.py). Importing src.rsa_open sets the pool's size (PJRT_NPROC) before JAX starts, and
a fit refuses to start where the pool would be too small."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_rsa_open_corpus import blocks, theory  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_package_gives_xla_a_pool_of_two_threads():
    import src.rsa_open as rsa_open

    assert int(os.environ["PJRT_NPROC"]) >= rsa_open.XLA_POOL_THREADS == 2
    assert rsa_open.xla_pool_problem() == ""


def test_a_fit_refuses_to_start_where_the_pool_would_deadlock(monkeypatch, theory, blocks):  # noqa: F811
    import src.rsa_open as rsa_open
    from src.rsa.fit import FitSettings
    from src.rsa_open.corpus import fit_corpus

    monkeypatch.setattr(rsa_open, "xla_pool_threads", lambda: 1)
    assert "PJRT_NPROC=2" in rsa_open.xla_pool_problem()
    with pytest.raises(RuntimeError, match="deadlocks"):
        fit_corpus(theory, blocks, FitSettings(num_warmup=5, num_samples=5, num_chains=2))


ONE_CORE_FIT = """
import tempfile
from pathlib import Path
from src.rsa.fit import FitSettings
from src.rsa_open.corpus import fit_corpus
from src.rsa_open.fit import TheoryModel
from src.rsa_open.links import likelihood
from src.rsa_open.port import port
import tests.test_rsa_open_corpus as T
path = Path(tempfile.mkdtemp()) / "rsa_l1_salience.py"
path.write_text(port((T.SEEDS / "rsa_l1_salience.py").read_text(), name="rsa_l1_salience.py"))
theory = TheoryModel(path)
blocks = [T.simulated_block(theory, "E", likelihood("choice"), 400, 1, {"temperature": 1.0, "lapse": 0.05}),
          T.simulated_block(theory, "exp1", likelihood("rating", scale_points=5), 400, 2,
                            {"temperature": 2.5, "lapse": 0.05, "center": 0.0, "spread": 0.8})]
fit = fit_corpus(theory, blocks, FitSettings(num_warmup=10, num_samples=10, num_chains=2, dense_mass=True))
print("finished", fit.timing["sampling_steps"] > 0)
"""


@pytest.mark.slow  # NUTS, on one core
@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="pins the process to one core (Linux)")
def test_a_callback_fit_finishes_in_a_process_that_sees_one_core():
    """Two blocks, as in the fit that hung (one block's gradient did not); PJRT_NPROC left unset."""
    env = {k: v for k, v in os.environ.items() if k != "PJRT_NPROC"}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))  # keep a pytest supplied there (V3 §15)
    core = min(os.sched_getaffinity(0))
    out = subprocess.run([sys.executable, "-c", ONE_CORE_FIT], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=1200, preexec_fn=lambda: os.sched_setaffinity(0, {core}))
    assert out.returncode == 0, out.stderr[-3000:]
    assert "finished True" in out.stdout
