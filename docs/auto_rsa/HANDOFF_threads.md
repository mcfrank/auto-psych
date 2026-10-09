# Handoff: thread counts on Sherlock (for the driver session)

From the local Sherlock session, 2026-10-09. **The rehearsal (`HANDOFF_sherlock_promote.md`
step 4) is on hold until this is fixed.** Step 3b (`chains.sbatch`) is running; it is a
single process, and I submitted it with `-c 1 --mem=7G` (below).

## 1. What happened

Research Computing emailed Mike on 2026-10-08 about **job 47042590**
(`promote.sbatch`, 20 CPUs on sh04-17n29, 1 h 45 min, finished before the email arrived).

- **The email's wording:** a "resource usage mismatch": the job was "starting more
  computing processes than the number of CPU cores it requested".
- **Its process-tree snapshot:**
  - the parent `python -m src.rsa.promote` had **164 threads**;
  - 17 spawned fit children (`python -c from multiprocessing...`) had **140-160 threads
    each**.
- **The warning:** further cases can get jobs cancelled automatically and job submission
  suspended.

Sherlock's agent rules (`/etc/agents/AGENTS.md`, `slurm.md`) also say to set each threaded
program's thread count to the allocation.

**The CPU use itself was within the allocation.** `sacct`: TotalCPU 30:39:40 over 1:44:48
of wall time = 17.6 cores on average out of 20; each child ran at ~100% CPU. **What the
checker flags is the thread count**, not CPU over-use. Idle threads count.

## 2. Cause, measured

A JAX (+ numpyro, numpy/OpenBLAS) process builds thread pools **sized to the CPUs it can
see** (`sched_getaffinity`), about **7 threads per visible core**.

| Probe (§5, one process, 4-chain numpyro NUTS) | Threads at the end |
|---|---|
| 1-CPU allocation | 7 |
| 4-CPU allocation | 28 |
| 4-CPU allocation, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` | **28** (unchanged) |
| 4-CPU allocation, process pinned with `taskset -c <one core>` | **7** |
| promote's children in the 20-CPU job (from the email) | 140-160 (≈ 7 × 20) |

By stage (4-CPU allocation):

| Stage | Threads |
|---|---|
| `import numpy` + one `np.dot` | 4 (OpenBLAS: one per core; `OMP_NUM_THREADS=1` alone did not stop it) |
| `import jax` | 4 |
| first `jit` | 21 |
| numpyro 4 chains | 28 |

**Conclusions:**

1. **The single-thread `XLA_FLAGS` do not reduce the thread count.** They stop XLA *using*
   parallel threads (CPU stays at ~1 core per process), but the pools are still *created*,
   sized to the visible cores.
   - Proof: promote's fit children already set them (`src/rsa/loop/fitting.py:136`,
     `_child`) and still held 140-160 threads each.
   - So 2f90c225 ("JAX on one CPU per process", `_env.sh:42`) is harmless but does not
     fix this.
2. **What reduces it is limiting the CPUs each process can see**: pin it to one core.
3. **Thread count scales with workers x cores.** A pool of N fit workers in an N-CPU job
   holds about 7·N² threads (20 workers x 140 ≈ 2,800 threads on 20 cores).
4. **Runs 1 and 2 had the same pattern, smaller:** 4-CPU cells, so each fit child and each
   agent self-check saw 4 cores, ~28 threads per process.

## 3. Where processes see more than one core

- **`src/rsa/loop/fitting.py` `_fit_cached` → `_start_with_cache` → spawned `_child`.**
  - Every fit with a time limit runs here: loop admission, the prefit, CV folds via
    `loop_fit`, promote, recovery.
  - The child inherits the parent's full affinity.
  - Note: a `spawn` child re-imports the main module, and so numpy and arviz, *before*
    `_child` runs. That is why `_start_with_cache` sets `XDG_CACHE_HOME` in the parent's
    environment around `proc.start()`.
- **Fits without a time limit run in the parent process** (`_fit_now` directly). The
  parent sees all the job's cores, and its own pools are sized to them. This is the
  parent's 164 threads in the email.
- **The pools that start those fits:**
  - `src/rsa/loop/orchestrator.py`: `default_fit_workers()` = `len(sched_getaffinity(0))`,
    ThreadPoolExecutors at :256 and :423;
  - `src/rsa/loop/cv.py:130`;
  - `src/rsa/promote.py:176,197`.

  The executor threads only wait on children, which is fine. Each child they start sees
  every core.
- **Agents' self-checks** (`src/rsa/loop/check_candidate.py`, run by each agent through
  `brief.CHECK_COMMAND` inside bwrap): a JAX fit in a process that sees all the job's
  cores. Up to 6 concurrent agents per cell, plus repairs, so 6 × (7 × N) threads.
- **Single-process jobs** (design, chains, simulate, held-out pages, recovery scoring) see
  however many CPUs Slurm gave the job, and **memory buys CPUs on `mcfrank`**:
  `MaxMemPerCPU=8000` MB.
  - So `--mem=32G -c 4` was allocated 5 CPUs (design 47043328, and chains on its first
    submission).
  - With `-c 1 --mem=7G`, `chains.sbatch` (job 47075879) runs as 1 process with 7
    threads, measured from inside the allocation.

## 4. Suggested fix

The aim is that every process holds the threads of one core: about 7 under JAX, 1 for
plain Python.

1. **Pin each spawned fit child to one core of the parent's affinity.**
   - Cores are handed out by a small allocator in the parent: a lock-protected free set,
     taken before `proc.start()` and returned in `_finish_child`'s `finally`.
   - Two ways to apply it:
     - `os.sched_setaffinity(proc.pid, {core})` from the parent right after `start()`;
     - simpler and race-free, pass `core` to `_child`, which calls
       `os.sched_setaffinity(0, {core})` as its first line.
   - Check which one is early enough. JAX/XLA sizes its pools at backend initialisation
     (the first computation), which happens inside `_child`, so pinning at the top of
     `_child` should catch XLA. OpenBLAS sizes its pool when numpy loads, which happens
     during spawn's re-import, *before* `_child`. So also set `OPENBLAS_NUM_THREADS=1`
     in the environment the child starts with (as `_start_with_cache` does for
     `XDG_CACHE_HOME`), or globally (item 3).
   - Verify with the probe: §5's script reports per-stage thread counts.
2. **Fits in the parent.** Either always run fits in a pinned child (simplest, if the
   no-time-limit path isn't hot), or pin the parent to one core when it only orchestrates.
   - Don't pin a parent that runs several in-process fits concurrently on threads: they
     would share one core.
3. **`_env.sh`: `export OPENBLAS_NUM_THREADS=1`.** Also `OMP_NUM_THREADS=1` and
   `MKL_NUM_THREADS=1` are already set by the SR `_env.sh`. Keep the `XLA_FLAGS` (they
   stop XLA's threads from competing for CPU).
4. **Agents' processes.** Pin each agent slot to a distinct core: slot *i* → core *i* mod
   the job's CPUs. This is easiest in the RSA spawner or `no_network.write_shell_wrapper`
   (`taskset -c <core> bash …`), so the agent's shell and its `check_candidate` inherit
   it.
   - opencode itself is light; pinning it too is harmless.
   - The self-check runs single-threaded either way.
   - Two agents sharing a core when slots outnumber CPUs is fine; it is what happens now,
     plus the threads.
5. **Job scripts for single-process jobs:** request `-c 1` and `--mem` ≤ 7.8G (or pin the
   process), so Slurm doesn't hand them extra CPUs through memory.
   - Affected: `chains.sbatch`, `design.sbatch`, the held-out pages, `simulate` in setup.
   - `chains.sbatch` asks for `-c 4 --mem=32GB`, which gets 5 CPUs; its process used 1.
   - The design job peaked at 1.8 GB.

### Tests

- **On Linux CI** (GitHub's runners have several cores):
  - start a fit child through `_fit_cached` with a tiny model, or a stub target through
    the same start path;
  - assert its `sched_getaffinity` is one core;
  - assert its `/proc/<pid>/status` `Threads` is small (≤ 10 after a jitted computation)
    while the parent sees more than one core.
  - Skip when the runner has one CPU.
- **On Sherlock**, before a long job: run §5's probe in a multi-CPU allocation through the
  real start path. During the job, check from the login node:
  `srun --jobid=<id> --overlap -n1 ps -u $USER -o pid,nlwp,pcpu,cmd --sort=-nlwp`.

## 5. The probe

```python
import os
def threads():
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("Threads:"))
print("affinity", len(os.sched_getaffinity(0)), "start", threads())
import numpy as np; np.dot(np.ones((300, 300)), np.ones((300, 300))); print("numpy", threads())
import jax, jax.numpy as jnp; print("import jax", threads())
jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((200, 200))).block_until_ready(); print("jit", threads())
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
def model(y):
    mu = numpyro.sample("mu", dist.Normal(0, 1)); numpyro.sample("y", dist.Normal(mu, 1), obs=y)
MCMC(NUTS(model), num_warmup=100, num_samples=100, num_chains=4, chain_method="sequential",
     progress_bar=False).run(jax.random.PRNGKey(0), y=jnp.ones(50))
print("numpyro 4 chains", threads())
```

Run on Sherlock, in a job:

```bash
srun -p mcfrank -c 4 --mem=4G -t 00:10:00 $VENV_PY probe.py
srun -p mcfrank -c 4 --mem=4G -t 00:10:00 taskset -c <a core of the allocation> $VENV_PY probe.py
```

## 6. Partitions for the campaign (the PI asked)

The PI wants long jobs spread across `mcfrank` and `owners`:

- **The rehearsal fits on `mcfrank` alone.** It is 2 tasks × 8 CPUs; 3 live chains × 8
  would fill the node's 24.
- **`-p owners` is preemptible.** A preempted job with `--requeue` restarts its script, and
  `outer_rehearsal.sbatch` resumes from the last scored step. But agent calls in flight
  are lost and paid for again, and NUTS fits in flight restart.
- **Suggest** `-p mcfrank,owners --requeue` only as overflow when the node is busy. The
  owners QOS caps at 48 h, the same as the rehearsal's `--time`.
- If the driver wants owners by default, make sure a requeued task finds its lock files
  and partial round in a state the resume handles: `.setup.lock` uses `flock`, which is
  released when the process dies.

## 7. Notes

- The `sherlock` skill on Mike's Mac (§9, failure modes) now records the thread-pool
  behaviour, the probe commands and `MaxMemPerCPU`.
- No commit on Sherlock. Step 3b's output will be committed from the Mac as the handoff
  says.
