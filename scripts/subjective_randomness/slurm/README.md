# Holdout-recovery test-retest (Sherlock)

Runs the holdout-recovery pipeline N times (default 5), each with a distinct
seed, then summarises **test-retest reliability** of the recovered fit across
repeats.

## Files

- `submit_holdout_test_retest.sh` — submits the 3-stage chained pipeline.
- `holdout_setup.sbatch` — stage 1: `uv sync` the env **once** (off `$HOME`)
  and create one git worktree per repeat at `$WORK_ROOT/worktrees/run<i>`.
- `holdout_recovery_array.sbatch` — stage 2: array of N repeats. Each task runs
  from its own worktree (the opencode agent works in its CWD, so per-task
  worktrees prevent collisions). Repeat *i* uses `--seed BASE_SEED+i` and writes
  output to `$WORK_ROOT/run<i>/`.
- `holdout_analysis.sbatch` — stage 3: `holdout_test_retest.py`, runs
  `afterok` the array. Writes `test_retest.{json,csv,png}` to `$WORK_ROOT`.
- `_env.sh` — shared env (modules, caches off `$HOME`, agent credentials).

## Prerequisite: coding-agent credentials

The config uses **opencode + `google/gemini-3.1-pro-preview`**, so a Gemini key
must be available. Put it in the repo's `.secrets` file:

```
GOOGLE_API_KEY=<your key>
```

`_env.sh` sources `.secrets` and also exports it as `GEMINI_API_KEY` /
`GOOGLE_GENERATIVE_AI_API_KEY` for opencode. If opencode needs an interactive
`opencode auth login`, run that first on a login shell.

## Before submitting: commit your code

The per-repeat worktrees are checked out at `HEAD`, so **commit any local edits
to `src/` or `holdout_recovery.py` before submitting** — uncommitted changes
won't reach the worktrees. (The Slurm scripts themselves run from your main
checkout, so they don't need committing.)

## Submit

```bash
bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
```

Override via env vars:

```bash
N_REPEATS=5 BASE_SEED=100 \
WORK_ROOT=$SCRATCH/auto-psych/tr_run2 \
CONFIG=scripts/subjective_randomness/configs/holdout_recovery.yaml \
  bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
```

Watch: `squeue --me`. A full 3-model repeat takes hours; the array requests
1 day on `normal` (raise `--time` / add `--qos=long` for up to 7 days if needed).

## Output

`$WORK_ROOT/` (default `$SCRATCH/auto-psych/holdout_test_retest/`):

- `run<i>/holdout.{json,csv,png}` — each repeat's trajectories.
- `test_retest.json` — ICC(2,1), mean pairwise across-run correlation, and
  per-ground-truth mean/sd/CV of the final Pearson r plus best-model agreement.
- `test_retest.csv` — one row per (gt_model, run) final-step metric.
- `test_retest.png` — final r per ground-truth model across repeats.

## Resume

Each repeat runs with `--resume`: a requeued task skips ground truths that
already have a `trajectory.json`. Re-running `submit_*.sh` with the same
`WORK_ROOT` continues where it stopped.

## Impossible-theory variant

The same test-retest design, but the ground truths are deliberately weird
generators (e.g. "more heads ⇒ more random") that humans could not plausibly
use. The loop keeps the **full normal seed pool** and is *expected to fail* to
recover them; 5 repeats per ground truth measure how *stably* it fails.

- `run_impossible_test_retest.sh` — pinned wrapper (5 repeats, `BASE_SEED=100`,
  the impossible config, a dedicated `..._full` work root). Start here.
- `submit_impossible_holdout_test_retest.sh` — submits the 3-stage chain.
- `impossible_holdout_setup.sbatch` — stage 1: build the venv **and** stage
  pristine, off-the-agent snapshots of the impossible recipe (the models dir and
  the config) on scratch.
- `impossible_holdout_recovery_array.sbatch` — stage 2: `R × G` array running
  `impossible_holdout_recovery.py`.
- stage 3 reuses `holdout_analysis.sbatch` / `holdout_test_retest.py` (both are
  ground-truth-agnostic).

**Key difference from the standard holdout — leak prevention.** The impossible
ground truth is *not* a seed model, so nothing is held out of the seed set.
Instead, what must stay off the agent's disk is the impossible **recipe**: both
the model files (`impossible_models/*.py`) and their answer-bearing **names** in
the config (`more_heads_more_random`, …). So each array task's repo copy
*excludes* `impossible_models/` and the config, and the parent loads the ground
truth from the pristine scratch snapshot via a per-task config whose
`gt_models_dir` is rewritten to that absolute path (opencode reads of those
paths are denied too).

```bash
# cheap pre-flight (one task, no inner loop, tiny MCMC):
SMOKE=1 bash scripts/subjective_randomness/slurm/run_impossible_test_retest.sh

# the real run:
bash scripts/subjective_randomness/slurm/run_impossible_test_retest.sh
```

Output lands in `$WORK_ROOT` (default
`$SCRATCH/auto-psych/impossible_holdout_test_retest_full/`) with the same
`run<i>/<gt>/holdout.{json,csv,png}` and `test_retest.{json,csv,png}` layout.

## Ablation: no inner loop

The inner-loop ablation runs the **same** pipeline with
`--inner-loop-iterations 0`, so the inner model loop does **zero**
candidate-conjecturing rounds. At 0 iterations the inner loop spawns no agents
at all — it only fits and ELPD-LOO-scores the carried model set (the
experiment-1 seed pool, carried forward verbatim into experiments ≥ 2; there is
no theorist agent) and records the single seed-scoring step the trajectory
needs. The design agent and synthetic collection still run in full, so the only
thing removed is the inner candidate/critique search:

```
full pipeline:  seed/carry-forward → design → collect → [inner loop: fit +
                critique + conjecture N rounds of new PyMC models] → score
ablation:       seed/carry-forward → design → collect → [fit + score the
                carried models] → score
```

Comparing the ablation against the full run isolates how much of the recovery
comes from the inner loop's agent-discovered models vs. the outer loop alone.

- `run_no_inner_loop_test_retest.sh` — standard holdout, inner loop removed.
  Dedicated work root `…/holdout_test_retest_no_inner_loop`.
- `run_impossible_no_inner_loop_test_retest.sh` — impossible variant, inner loop
  removed. Dedicated work root `…/impossible_holdout_test_retest_no_inner_loop`.

Both are thin wrappers that pin `INNER_LOOP_ITERATIONS=0` (forwarded by the
submit scripts to every array task) and otherwise reuse the full pipeline, so
they inherit every fix, isolation guard, and leak-prevention measure.

```bash
# cheap pre-flight first (one task, tiny MCMC):
SMOKE=1 bash scripts/subjective_randomness/slurm/run_no_inner_loop_test_retest.sh
SMOKE=1 bash scripts/subjective_randomness/slurm/run_impossible_no_inner_loop_test_retest.sh

# the real ablation runs:
bash scripts/subjective_randomness/slurm/run_no_inner_loop_test_retest.sh
bash scripts/subjective_randomness/slurm/run_impossible_no_inner_loop_test_retest.sh
```

Output lands in each ablation's own `_no_inner_loop` work root, with the same
`run<i>/<gt>/holdout.{json,csv,png}` and `test_retest.{json,csv,png}` layout —
so the standard and impossible ablations can be analyzed side by side with their
full-pipeline counterparts.

## Re-scoring everything on one exhaustive eval pool

The studies above are not all scored on the same held-out stimuli: the standard
holdout config (`holdout_recovery.yaml`) measures recovery on a **sampled** pool
(500 pairs at lengths 6 & 8) while the impossible config measures it on the
**exhaustive** pool (every distinct unordered pair over all sequences up to
length 8). Their RMSE / Pearson-r numbers are therefore not comparable across
studies.

`submit_reanalyze_exhaustive.sh` re-scores finished runs on **one common
exhaustive pool** (lengths 1..8), without re-running any agents or resampling
MCMC. For each `holdout.json` it rebuilds the held-out set as the exhaustive pair
space (still excluding that run's *trained* pairs), then re-correlates every
trajectory step and both seed baselines against the ground truth — every per-step
fit is a hit on the run's own `.nc` cache, so the only real cost is predicting
held-out `p_left` over the ~130k-pair pool (`predict_max_draws` thins the
posterior to keep that bounded). The impossible runs were already exhaustive, so
re-scoring them is ~idempotent; the sampled-pool runs are brought onto the same
pool. The ground-truth generator is located automatically (the seed dir for a
normal GT, `impossible_models/` for an impossible one).

It rewrites each run's `holdout.{json,csv,png}` **in place**, so the test-retest
/ combined collectors pick the exhaustive numbers up unchanged. **Run it only
once the studies' run arrays have finished** (a run still in flight has no
`holdout.json` yet, so it is silently skipped).

- `reanalyze_setup.sbatch` — stage 1: build the reanalysis venv once (same pins
  as the run venv, **plus** `h5netcdf`/`h5py` so the `.nc` caches can be read).
- `reanalyze_exhaustive_array.sbatch` — stage 2: one array task per
  `holdout.json` from the manifest, re-scored via
  `reanalyze_holdout_exhaustive.py`.

```bash
# preview the manifest without submitting:
DRY_RUN=1 bash scripts/subjective_randomness/slurm/submit_reanalyze_exhaustive.sh \
  $SCRATCH/auto-psych/holdout_test_retest_v2 \
  $SCRATCH/auto-psych/holdout_test_retest_no_inner_loop_v2 \
  $SCRATCH/auto-psych/impossible_holdout_test_retest_v2 \
  $SCRATCH/auto-psych/impossible_holdout_test_retest_no_inner_loop

# submit (chained setup → array). Pass the SAME study roots you re-ran:
bash scripts/subjective_randomness/slurm/submit_reanalyze_exhaustive.sh <roots...>
```

Work/logs land in `$SCRATCH/auto-psych/reanalyze_exhaustive/` (override with
`WORK_ROOT`). Re-run your collectors afterward to roll the exhaustive numbers up
into `data/results`. A single run can also be re-scored directly:

```bash
uv run python scripts/subjective_randomness/reanalyze_holdout_exhaustive.py \
  --result $SCRATCH/auto-psych/holdout_test_retest_v2/run1/window_typicality/holdout.json
```

### Automating it: re-score each study as it finishes

To kick reanalysis off automatically, gate it on the run pipeline with a Slurm
dependency. Because the manifest is built (by `find`) when the submit script
*runs*, the discovery must happen AFTER the runs finish — so wrap the submit in a
tiny launcher job gated on each study's stage-3 analysis job (`afterany`, so it
proceeds on whatever finished). To run several independent (per-study) launchers
without them racing to rebuild the venv, build it ONCE with `--setup-only` and
have each reuse it with `--skip-setup` + a shared `UV_PROJECT_ENVIRONMENT` (each
keeps its own `WORK_ROOT` for an isolated manifest/logs):

```bash
SLURM=scripts/subjective_randomness/slurm
RB=$SCRATCH/auto-psych/reanalyze_exhaustive

# build the shared venv once (returns a job id):
setup_id=$(WORK_ROOT=$RB UV_PROJECT_ENVIRONMENT=$RB/venv \
  bash $SLURM/submit_reanalyze_exhaustive.sh --setup-only | awk '/setup-only job/{print $4}')

# one launcher per study — gated on the venv (afterok) AND that study's analysis
# job (afterany). <analysis_jobid> is the stage-3 job subm_*.sh reported.
sbatch --dependency=afterok:$setup_id,afterany:<analysis_jobid> \
  --partition=normal --time=00:10:00 --mem=2G \
  --output=$RB/launch_<tag>_%j.out \
  --wrap="WORK_ROOT=$RB/<tag> UV_PROJECT_ENVIRONMENT=$RB/venv \
    bash $SLURM/submit_reanalyze_exhaustive.sh --skip-setup <study_root>"
```

The `afterok:$setup_id` term guarantees no array starts before the venv exists.
An already-finished study needs only `--dependency=afterok:$setup_id` (no
analysis job to wait on).
