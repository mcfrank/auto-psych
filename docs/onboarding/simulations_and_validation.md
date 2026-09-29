# Simulations and validation

Before trusting what the loop "discovers" from people, we want to know two
things. **Can it find a true model when there is one** (recovery)? And **does it
avoid fitting nonsense** (the impossible controls)? Both questions are answered
by running the same loop, with the same agents, on simulated participants
whose generating model is known to us and hidden from the loop.

The library code is in `src/subjective_randomness/`, the entry scripts in
`scripts/subjective_randomness/`, and the cluster launchers in
`scripts/subjective_randomness/slurm/`. Its own runbooks are
`scripts/subjective_randomness/README.md` and `…/slurm/README.md`.

## The three checks

| check | question | agents? | entry point |
|---|---|---|---|
| **Model recovery** (closed) | If one of the known models generated the data, does fitting and comparing pick that model? | no | `scripts/subjective_randomness/model_recovery.py` |
| **Holdout recovery** | If the true model is *removed* from the starting set, can the agents rebuild something that predicts like it? | yes | `scripts/subjective_randomness/holdout_recovery.py` |
| **Impossible controls** | If the data come from a process no sensible person would use, does the loop *fail* to fit it well, as it should? | yes | `scripts/subjective_randomness/impossible_holdout_recovery.py` |

### Model recovery

Each of the four starting models generates data for 40 simulated participants.
All four are then fitted and compared by ELPD-LOO, with no agents. The result
is a confusion matrix: rows are the true model, columns the model picked. If
the loop cannot tell the literature models apart even when one of them is
true, nothing downstream can be trusted.

### Holdout recovery

In each run, one of the four starting models plays the "true" model and is
withheld:

- **Data.** It generates the responses: 40 simulated participants per
  experiment, 3 experiments, every participant answering every designed pair,
  with left/right randomised.
- **Starting set.** The loop starts from the other three models.
- **Keeping the answer hidden from the agents.** The agents work in a copy of
  the repository with the withheld model's file and manifest entry removed.
  The run directory is numbered (`cell_<n>`) rather than named after the
  withheld model. A scan (`slurm/scan_gt_name.sh`) stops the run if the model's
  name appears anywhere the agents can see. The data files carry only the five
  raw columns. `src/subjective_randomness/leakage_audit.py` also looks
  afterwards for copies of the hidden model; that check is a heuristic, not a
  proof.

**Scoring.** After every round, the current best model's predicted `p_left`
is compared with the hidden model's true `p_left`. The comparison uses a large
evaluation set: every same-length pair of lengths 1–8 that was *not* used in
training. The metrics (`src/subjective_randomness/recovery_metrics.py`) are:

- `pearson_r`: correlation between the two;
- `rmse`: root-mean-square difference;
- `kl_regret`: extra prediction error in bits compared with the truth;
- `bias` and `calib_slope` / `calib_intercept`: calibration.

A pair on which a model's `p_left` is undefined, or on which the model's own
code cannot compute its features (say, a feature that looks at the fourth flip
of a length-2 sequence), is left out of that step's metrics. The cell's log
and `eval_exclusions.jsonl` say so, and `holdout.csv` counts the excluded
pairs (`n_eval_excluded`).

The same metrics with the suffix `_bma` are computed for the average of all
models, weighted by the reported posterior.

**Baselines,** to tell whether the agents added anything:

- `baseline`: the three remaining starting models at default parameters, no
  fitting.
- `fitted_baseline`: the same three fitted on all the collected data. Its
  headline numbers are `elpd_best_r` and `elpd_best_rmse`. **The key question
  is whether the loop's final model beats the fitted baseline.** If not, the
  agents' new models added nothing that refitting the literature models could
  not.

A *cell* (a code term) is one repeat of one held-out model. A standard
sweep is 4 held-out models × 5 repeats = 20 cells. Repeats differ only in their
random seed, which measures how reproducible the outcome is.

### Impossible controls

The hidden model is one of four deliberately implausible rules
(`src/subjective_randomness/impossible_models/`), each with only a noise
parameter and a side bias:

| rule | "more random" means… |
|---|---|
| `more_heads_more_random` | more heads |
| `fewer_heads_more_random` | fewer heads (mirror image of the first) |
| `longer_runs_more_random` | a longer longest run (the opposite of the gambler's-fallacy intuition) |
| `more_imbalance_more_random` | a bigger heads/tails imbalance (the opposite of representativeness) |

The loop keeps all four normal starting models.

**Expected result: poor recovery**, for example a low held-out `pearson_r`.
The prompts steer agents towards psychologically plausible mechanisms, so a
high correlation here means one of two things. Either the loop is flexible
enough to fit anything, in which case its successes on real data say little.
Or information about the hidden rule reached the agents. `pearson_r` can be
undefined when a model's predictions are constant.

## How to read a sweep

- **Per cell.** In `trajectory.json`, or `holdout.csv` with one row per scoring
  step: the final step's `pearson_r` / `rmse` / `kl_regret` against
  `fitted_baseline.elpd_best_r` / `elpd_best_rmse`, and how the metrics change
  from step to step (`holdout.png`).
- **Per sweep.** `test_retest.json` / `.csv` / `.png` report how consistent the
  repeats are:
  - an intraclass correlation;
  - mean, SD and coefficient of variation of the final correlation per hidden
    model;
  - whether the repeats agree on the best model.
- **Did the loop actually change its mind?** Each scoring step records
  `incumbent_changed` (did the best model change?) and
  `incumbent_is_discovered` (is the best model a new, agent-written one rather
  than a starting model?). `scripts/subjective_randomness/incumbent_report.py`
  tabulates this over a sweep (`src/subjective_randomness/incumbent.py`). In
  the three complete cells of an earlier sweep that it examined, over 27 steps
  the best model never changed. That motivated the "improve the best" agent
  roles described in [how_the_loop_works.md](how_the_loop_works.md).
- **Was the run valid?**
  `WORK_ROOT=<sweep dir> bash scripts/subjective_randomness/slurm/verify_holdout_run.sh`
  writes `VERDICT.md`. It checks that the results exist, that the data files
  carried only raw columns, that the agents saw only their own copy of the
  code, the import allowlist, and whether critique ran. It only warns about an
  unchanged best model.

## Launching a sweep (cluster)

These launchers submit chains of Slurm jobs (setup → an array of cells →
analysis, plus automatic retries). Standard cells take about a day, with 16
CPUs and 64 GB; impossible cells about a day with 8 CPUs and 32 GB.

| command | what |
|---|---|
| `SMOKE=1 bash scripts/subjective_randomness/slurm/run_faithful_test_retest.sh` | cheap check that the whole chain runs (one task, tiny MCMC). Do this first. |
| `bash scripts/subjective_randomness/slurm/run_faithful_test_retest.sh` | holdout recovery: 4 held-out models × 5 repeats |
| `bash scripts/subjective_randomness/slurm/run_impossible_test_retest.sh` | the impossible controls |
| `run_no_inner_loop_test_retest.sh`, `run_impossible_no_inner_loop_test_retest.sh` | the same with the agents switched off (0 rounds): a no-discovery comparison |

## Why this matters before a live run

A live run cannot tell you whether it found the truth, because nobody knows
the truth. The simulations are the only evidence of what the loop can and cannot
find, how often, and how much its answer varies from one repeat to the next.
They use the same code and agents as a live run, apart from collection. They
are also the natural place to test changes to the loop before paying
participants. The live-run rehearsal in the runbook (R2) only checks that a run
completes; it says nothing about whether the answer is right.

## Caveats found while writing this page

- `data/results/holdout_test_retest/SUMMARY.md` and its siblings report sweeps
  over an **older set of hidden models** (`bayesian_diagnosticity`,
  `encoding_compressibility`, `prototype_similarity`, `window_typicality`), not
  the current four. There is no committed summary for the current set.
- The impossible-control config (`configs/impossible_holdout_recovery.yaml`)
  runs 2 rounds × 3 proposals. The standard holdout config
  (`holdout_recovery_faithful.yaml`) runs 5 × 6. The two sweeps are therefore
  not like-for-like; the main README says both use 5 × 6.
- `slurm/run_test_retest.sh` pins old hidden-model names and would abort at
  setup. Use `run_faithful_test_retest.sh`. The comments in
  `run_faithful_test_retest.sh` also give outdated settings (2×3 rounds,
  4000/3000 draws). The config file is authoritative.
- `slurm/README.md` describes the evaluation set as ~130k pairs including
  different-length pairs. The code uses same-length pairs only (about 43k at
  lengths 1–8).
- The impossible rules declare precomputed feature inputs rather than raw
  sequences. I did not check how they are fed at data-generation time.
