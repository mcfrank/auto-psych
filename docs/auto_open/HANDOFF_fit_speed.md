# Handoff: fit speed after deduplication, on chain 1's real corpus (2026-11-08)

For the Sherlock session. Written by Claude Opus 5.5; not reviewed by a person.

**What changed.** The open campaign's fit (`src/rsa_open/fit.py`, `OpenData`) evaluated the theory once
per *trial*. It now evaluates it once per distinct display, and the link once per distinct (display,
response) pattern, weighted by its count, as the closed campaign's fit always did. Per-trial results are
exact: `tests/test_rsa_open_fit_patterns.py` holds every link to the old per-trial computation.

Measured in a 4-core cloud container on the committed CC-BY literature sources (27,949 trials, 173
displays, 402 patterns), at the loop's settings (4 chains, dense mass), for the depth-3 starting
theory `isolated_graded_costly_l3`:

| | per trial (old) | per pattern (new) |
|---|---|---|
| one gradient | 25.9 ms | 1.8 ms |
| NUTS 250 + 250 | 854 s | 77 s |
| NUTS 1,000 + 1,000 | about 55 min (extrapolated) | 119 s (about 42 s of it compiling) |

The posterior was the same (e.g. alpha 0.964 against 0.966), and so was the potential to float32
rounding. Separately, building the data took about 20 s per 17k trials, from thousands of small device
arrays; it now builds arrays for the distinct displays only.

**First, the slow tests** of the changed code, which a 4-core container can't run in reasonable time
(the fast ones, 131 across the fitting, corpus, gate, loop and critique modules, pass):
`uv run pytest -q tests/test_rsa_open_fit.py tests/test_rsa_open_fit_patterns.py tests/test_rsa_open_corpus.py
tests/test_rsa_open_loop_fitting.py tests/test_rsa_open_loop_orchestrator.py`.

**What to run** (no agents; fits in this process; aggregates only), with
`scripts/rsa/open_compute_checks/fit_speed.py`, on a clone of `auto-psych-open` at this commit and,
for comparison, one at `rsa-open-prereg-a2` (the code chain 1 runs):

1. `fit_speed.py data/rsa_open/seeds/bar/isolated_graded_costly_l3.py` with no corpus: should
   reproduce the table above on a Sherlock core (and tell us how Sherlock's cores compare).
2. With `--corpus` = chain 1's `experiment3/corpus.json` (copied, never modified):
   - three of experiment 3's slowest admitted theories (by their fit sidecars' `timing`) and the
     depth-3 seed, at the loop's settings (1,000 + 1,000);
   - the same theories at chain 1's recorded times, for comparison (from the sidecars; no refit);
   - trials, displays and patterns per block, as the script prints them.
3. Peak memory (`/usr/bin/time -v` or `sacct MaxRSS`), new code only: chain 1's fits peaked at 12-15 GB,
   and some folds at 28-32 GB.

Run them as single-core jobs on `owners` or `normal` (not the lab node while chain 1 is on it), 24 GB.

**Report** in `DISTRIBUTED_DESIGN_REVIEW_V3.md` §13 (or after the refit test's section): a table of
theory × (chain 1's recorded time, the new code's time, its warm-up and sampling split, steps per
iteration, peak memory), and whether each new fit converged.

**Why it matters.** If real fits drop from tens of minutes to a few, then step mode's arrays would
mostly be jobs under Slurm's sensible minimum (SRCC: 20 minutes), the fits could run inside the
experiment's own job again, and the 3-day scoping (`HANDOFF_refit_test.md`) can be revisited with
full-fidelity settings. The refit test still stands: it measures cheaper settings, not cheaper code.

## Round 2 (2026-11-08): the sampler compiles the density once, outside its own program

Written by Claude Opus 5.5; not reviewed by a person. Code: `auto-psych-open` at d2fcfb25 or later.

**What §13 found and why.** On experiment 3's corpus the big theories still took 52-60 min and 36-39 GB
a fit. Profiling one of chain 1's experiment-2 theories (463 lines) in a cloud container showed:

- numpyro compiled its whole sampler into one program with the theory's log-density and gradient
  inlined, once per display shape, and compiled **two** such programs per fit (warm-up and
  sampling; `fit_corpus` runs them as two calls). On the literature block alone that was ~88 s and
  6 GB per program, against 14 s and 1.3 GB for the gradient by itself; on three blocks it ran
  out of memory above 14 GB. That is §13's "warm-up costs less per step than sampling" puzzle: the
  sampling phase's time included its own compile.
- Before compiling, numpyro ran the model eagerly to set up: 1,904 one-operation compilations,
  112-131 s (fixed in 9664699f by jitting each block's likelihood).

**What changed (d2fcfb25).** `fit_corpus` compiles the value and gradient once and gives NUTS an
opaque callback into it (`_callback_potential`), so the sampler's programs are small. In the container:
1.8 GB on the literature block, 2.6 GB on three blocks (was out of memory), about 34 s to set up and
compile the gradient, ~27 s for the small sampler, ~0.06 s an iteration for 4 chains; a 1,000 + 1,000 fit
of that theory on the literature projects to about 3 min. Same density, same NUTS; the old path stays as
`inline_potential=True`.

**Also (this round): one evaluation per display shape across blocks.** The corpus likelihood
traced the theory once per (block, display shape): 18 copies where 11 shapes suffice on the
literature plus chain 1's first two experiments' blocks, 23 for 13 on experiment 3's corpus. It now
evaluates each (shape, listener or speaker side) once over every block's distinct displays and hands
each block its rows (`fit.corpus_log_lik`; same numbers, `tests/test_rsa_open_fit_patterns.py`). On
the three blocks, with the callback sampler: setup and gradient compile 68.7 s → 50.2 s, peak
2.6 → 2.2 GB.

**What to run**, as before (`fit_speed.py`, single-core jobs, the same CPU), with
`JAX_COMPILATION_CACHE_DIR` exported to an empty directory of its own (the fit compiles each of its
programs once; a second, identical fit should reuse them):

1. The slow test comparing the two samplers' posteriors:
   `uv run pytest -q tests/test_rsa_open_callback_sampler.py` (with the fast ones).
2. `fit_speed.py` on experiment 3's corpus for §13's three theories and the depth-3 seed, at 1,000 +
   1,000: total time, warm-up and sampling split, steps per iteration, peak memory, convergence. Then the
   first theory once more with the same cache directory, to see what a repeated fit costs.
3. Request 16 GB first; report any that run out.

**Report** beside §13: the table with this code's column added.

## Round 3 (2026-11-08): one core, and a cheaper start

Written by Claude Opus 5.5 after review V3 §14; not reviewed by a person. Code: `auto-psych-open` at the
commit that adds this section or later.

**What changed.**

1. **The one-core deadlock (§14, problem 1) is fixed.** XLA runs the sampler's callback on a thread of
   its CPU pool, and the callback's gradient needs a second; XLA sizes the pool by the cores a process
   sees, one for every process that computes. Importing `src.rsa_open` now sets `PJRT_NPROC=2` (XLA's
   own setting for the pool's size) before JAX starts, so the pool has two threads whatever the cores;
   the cores a process may use are unchanged. A fit whose pool would be smaller refuses to start
   (`xla_pool_problem`) instead of hanging. Reproduced and checked in the cloud: §14's test hung on one
   core (`taskset -c 0`) and now passes (207 s); a fit process on one core has **10 threads**, against
   6 for JAX alone on one core (SRCC's flag was 140-160). Turning off async dispatch, single-thread
   Eigen and XLA's concurrency scheduler did not fix it; the legacy runtime is gone from this jaxlib.
   `tests/test_rsa_open_one_core.py` runs a two-block fit in a process pinned to one core.
2. **The start costs less (§14's "setup", 135 s and a 7.7 GB peak of a cold fit).** Given all chains'
   keys at once, numpyro searches for the starts in a compiled loop over the chains, compiling the
   likelihood and gradient once more. `fit_corpus` now asks one key at a time, which numpyro tries
   eagerly: the **same starts** (checked against the batched ones; so the same draws), and on chain 1's
   experiment-2 corpus in the cloud setup took 42 s instead of 102 s, with a peak of 2.0 GB instead of
   5.1 GB.
3. **§14's failing recovery test** (`test_a_joint_fit_recovers_the_shared_theory_and_each_block_s_link`)
   failed on the data, not the sampler: `temperature@exp1`'s true 2.5 sits in the posterior's lowest
   percent for both samplers (1% quantile 2.44-2.59 across the callback and the inlined density at seeds
   0-1; 0.5% quantile 2.536). Its check is now "within 3 posterior SDs of the posterior mean".

**Open: a segmentation fault.** In the cloud container `tests/test_rsa_open_loop_orchestrator.py::
test_a_time_limited_round_scores_admitted_theories_while_repairs_run_and_decides_as_before` crashed twice with a
segmentation fault inside XLA's compiler, in the test's main process (compiling a gate's contract check, after
its `run` fixture had run a whole loop of callback fits in that process). Not yet known whether `PJRT_NPROC`,
the callback, or the 16 GB container causes it; under investigation. If the slow tests crash the same way on
Sherlock, say so with the fault's stack.

**What to run** (no agents; aggregates only):

1. The slow tests of round 1's five files plus `tests/test_rsa_open_callback_sampler.py` and
   `tests/test_rsa_open_one_core.py`, on 2 CPUs with **32 GB** (§14's test job peaked at 19.9 GB).
2. `fit_speed.py` for `figure_ground_crowd_topic_tom` on experiment 3's corpus, cold (an empty
   `JAX_COMPILATION_CACHE_DIR`), **on one core** (`--cpus-per-task=1`, or `taskset` to one core), 16 GB:
   total, warm-up and sampling split, peak memory, and the fit process's thread count during sampling
   (`grep Threads /proc/<pid>/status`). Compare with §14's 13.4 min and 9.8 GB on two cores.
3. If (2) is fine, the depth-3 seed the same way.

**Report** after §14.
