# Handoff: fits in the step's own job, checked without agents (2026-11-09)

For the Sherlock session. Written by Claude Opus 5.5; not reviewed by a person.

**Why.** Review V3 §16: 316 of 325 fit tasks of the stepped rehearsal ran under rule 10's 10-minute
line (aggregate short tasks so a job's work lasts at least 10 minutes, ideally 30), and the steps' CPUs
were 84-93% idle. The PI decided (11-09) that steps run their fits themselves.

**What changed** (`auto-psych-open` at this handoff's commit or later):

- **`STEP_FITS=local`, the default** (`src.rsa_open.loop.slurm.LocalStepper`). A step writes the same task
  lists and result files as the arrays, runs each fit as `fit_array`'s task in its own clean process (no
  `.secrets`, no agent settings) on a core of the job, up to `local_workers()` at once, folds after every fit
  started before them, and then `run_chain` **continues in the same job** from disk with a fresh chain and
  loop, as a continuation job would (`StepContinue`). JAX's caches are cleared at each continuation. A launch
  now submits no job, except a continuation when its time runs short (the existing pause, exit 5).
  `STEP_FITS=array` keeps the arrays of 11-05.
- **Step size for local fits:** `STEP_CPUS=12`, `STEP_MEM=72G`, `STEP_TIME=4-00:00:00` by default (8 fits of
  about 6 GB beside the step's 24 GB; the lab node gives about 6 GB a CPU).
- **`LAUNCH_EXPERIMENTS=all`** (default `1`): a launch runs to the chain's end instead of stopping after one
  experiment, with 40 steps per experiment as its budget (`begin_launch` records `last_experiment`).
- **Fixes from §16:** `CLEAR_STOP=1` works with `open_rehearsal.sbatch`; the STOP hint names both launchers;
  the admission-limit log says which bound applied; the one-core test keeps a supplied `PYTHONPATH`.

## 1. Tests (no agents)

1. Fast: `uv run pytest -q -m "not slow" tests/test_rsa_open*.py`.
2. Slow, the new equivalence test (a loop whose steps run their own fits decides exactly as one run straight
   through: same ledger, history, export, each agent once):
   `uv run pytest -q tests/test_rsa_open_step_equivalence.py` (both tests), 12 CPUs, 32 GB. The new one
   passed in the cloud container (50 min on 4 cores).

## 2. The no-agent replay at smoke size, with fits in the step (this session submits it: no agents)

As `HANDOFF_stepmode.md` §4.2's smoke (1 starting theory plus the chance model, 1 round of 2;
`scripts/rsa/open_compute_checks/replay_step.sbatch`, whose header gives the setup), in a fresh work root,
with `STEP_FITS=local` and the step sized for it:

```bash
STEP_MODE=1 STEP_FITS=local CHAIN=0 ... \
  sbatch --job-name=rsa_open_replay_local_c0 --partition=mcfrank --cpus-per-task=12 --mem=72G --time=1-00:00:00 \
    --no-requeue --chdir=$HARNESS_REPO -o $WORK_ROOT/logs/%x_%j.out $WORK_ROOT/replay_step.sbatch
```

Report, briefly:
- that **one job** ran the experiment to its end: no continuation job, no array (`sacct --name`), and
  `steps.jsonl` holds only `"local": true` records;
- the workers per step (`workers` in each record), each record's `wait_sec`, and the job's wall time against
  §11's array version of the same smoke (about 6.5 h, mostly queue waits);
- the job's MaxRSS and its thread count while fits ran (`grep Threads /proc/<pid>/status` of the chain's
  process and of a fit), and CPU efficiency (`seff`);
- any fit task that exited nonzero, and anything in the log that looks like a hang.

If it works, the registered pair can use it; nothing else needs a rehearsal with agents, since the agents' side
did not change. If it doesn't, `STEP_FITS=array` is the fallback, as rehearsed.

## Result and follow-up (2026-11-09)

Review V3 §17: both tests pass, and two no-agent replays each ran an experiment to its end in one job. Its point 1
(a local fit's memory is not capped) is closed: each local fit raises its own `oom_score_adj` (inherited by its
sampler child), so if the step runs out of memory the kernel kills a fit, never the chain or an agent, and the
killed fit is retried at the next memory tier; and a fit whose tier says it is large reserves that much of the step's
memory before it starts (`MemoryBudget`, `fit_reserve_gb`; a fit larger than the budget runs alone). Point 2: the
standing no-agent source is the stepped rehearsal's experiment 1 (§17's replay 2), not chain 1's experiment 2, whose
theories the hand-set-number gate now refuses.
