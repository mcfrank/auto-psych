"""One core per process (Research Computing's warning on job 47042590, 2026-10-08).

JAX, numpyro and OpenBLAS size their thread pools by the cores a process can
see; every fit child saw the whole job and held 140-160 threads. A process the
harness starts must see one core from its first instruction.
"""

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import threading

import pytest

from src.rsa import cpus

pytestmark = pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="needs Linux and at least two CPUs",
)


def _report(queue):
    import jax
    import jax.numpy as jnp
    import numpy as np

    np.dot(np.ones((300, 300)), np.ones((300, 300)))
    jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((200, 200))).block_until_ready()
    queue.put((sorted(os.sched_getaffinity(0)), cpus.thread_count()))


def _start_like_a_fit(core):
    from src.rsa.loop.fitting import _start_with_cache

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_report, args=(queue,))
    with tempfile.TemporaryDirectory() as cache:
        _start_with_cache(proc, cache, core)
        proc.join(120)
    return queue.get(timeout=5)


def test_a_fit_child_sees_one_core_and_holds_one_cores_threads():
    seen = sorted(os.sched_getaffinity(0))
    core = seen[-1]
    affinity, threads = _start_like_a_fit(core)
    assert affinity == [core]
    assert threads <= 10, threads  # about 7 per visible core under JAX
    # The thread that started it is back on every core.
    assert sorted(os.sched_getaffinity(0)) == seen


def test_a_fit_takes_a_core_and_gives_it_back(monkeypatch, tmp_path):
    from src.rsa.fit import FitSettings
    from src.rsa.loop import fitting

    started = []

    def fake_start(proc, cache_dir, core=None):
        started.append(core)
        raise RuntimeError("stop here")

    monkeypatch.setattr(fitting, "_start_with_cache", fake_start)
    model = tmp_path / "m.py"
    model.write_text("x = 1\n")
    data = tmp_path / "r.csv"
    data.write_text("a\n1\n")
    use_before = dict(cpus.CORES._use)
    with pytest.raises(RuntimeError, match="stop here"):
        fitting._fit_cached(model, "m", data, FitSettings(), tmp_path / "cache", time_limit_sec=60)
    assert started and started[0] in cpus.job_cpus()
    assert cpus.CORES._use == use_before  # given back even though the start failed


def test_cores_are_handed_out_least_used_first():
    pool = cpus._Cores([3, 5, 7])
    taken = [pool.take() for _ in range(4)]
    assert taken[:3] == [3, 5, 7] and taken[3] == 3
    pool.give_back(5)
    assert pool.take() == 5


def test_an_agent_runs_on_a_core_of_its_own(monkeypatch, tmp_path):
    import src.runtime.coding_agent as coding_agent
    from src.rsa.loop.orchestrator import coding_agent_spawner

    seen = {}

    def fake_agent(prompt, **kw):
        seen["affinity"] = sorted(os.sched_getaffinity(0))
        return True, None

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_agent)
    spawn = coding_agent_spawner(models_dir=tmp_path, responses_path=tmp_path / "r.csv", timeout_sec=10,
                                 backend="opencode", model="m", agent_root=tmp_path, sandbox=False, network=True)
    before = sorted(os.sched_getaffinity(0))
    t = threading.Thread(target=spawn, args=(tmp_path, "hi"))
    t.start()
    t.join()
    assert len(seen["affinity"]) == 1 and seen["affinity"][0] in cpus.job_cpus()
    assert sorted(os.sched_getaffinity(0)) == before


def test_an_entry_point_pins_its_main_thread_but_still_counts_the_jobs_cores():
    code = (
        "import os\n"
        "from src.rsa.cpus import pin_main_thread, job_cpus\n"
        "from src.rsa.loop.orchestrator import default_fit_workers\n"
        "pin_main_thread()\n"
        "print(len(os.sched_getaffinity(0)), len(job_cpus()), default_fit_workers())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    n = len(os.sched_getaffinity(0))
    assert out == ["1", str(n), str(n)]
