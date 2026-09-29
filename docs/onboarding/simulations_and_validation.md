# Simulations and validation

A live run cannot say whether it found the truth. Simulations can: the same
loop and agents run on simulated participants whose generating model is known
to us and hidden from the loop. Code: `src/subjective_randomness/`; scripts
and cluster launchers: `scripts/subjective_randomness/` (its `README.md` and
`slurm/README.md` are the detailed runbooks).

## The checks

| check | question | agents? | entry point |
|---|---|---|---|
| model recovery | if a starting model generated the data, does fitting and comparing pick it? | no | `model_recovery.py` |
| held-out recovery | with the true model removed from the start set, do the agents rebuild something that predicts like it? | yes | `holdout_recovery.py` |
| impossible controls (the paper's "alien" rules) | with an implausible true rule, does the loop fail, as it should? | yes | `impossible_holdout_recovery.py` |
| no-inner-loop variants | how much do the agents add? | no | `run_no_inner_loop_test_retest.sh`, `run_impossible_no_inner_loop_test_retest.sh` |

Settings (`configs/holdout_recovery_faithful.yaml`; the impossible config is
identical apart from the hidden models, and a test keeps it so): 40
participants, 3 experiments, 5 rounds × 6 proposals, 64 stimuli, 30-minute
agents, 1,000 draws × 4 chains at `target_accept` 0.8, Gemini agents. Each
code "cell" is one hidden model × one repeat; a standard sweep is 4 × 5 = 20
cells, repeats differing only in their random seed.

**Two conditions.** Sweeps started before 28 September 2026 (the running
Gemini sweeps included) never removed starting models; later ones can.
`holdout.json` records `starting_models_prunable` (absent = never removed).
Compare only like with like.

**Hiding the answer.** The agents work in a trimmed copy of the repository
without the hidden model, docs or tests, under a random folder name; a scan stops the cell if the hidden name appears anywhere the agents
can see (`slurm/scan_gt_name.sh`); the data carry only the raw columns.

**Impossible rules** (`src/subjective_randomness/impossible_models/`): more
heads, fewer heads, a longer longest run, or a bigger heads/tails imbalance
looks more random. A good fit to these would mean the loop can fit anything,
or that the answer leaked.

## Scoring

After every round the loop's best model is compared with the hidden model on
every same-length pair of lengths 1–8 not used in training (about 43,000;
`recovery_metrics.py`): `pearson_r`, `rmse`, `kl_regret` (extra prediction
error in bits), and calibration. The `_bma` versions score the average of all
models weighted by the reported posterior.

The comparison that matters is with **`fitted_baseline`**: the remaining
starting models refitted to the same data, taking the best by ELPD-LOO
(`elpd_best_r`, `elpd_best_rmse`). It uses the files the cell started with,
including those the loop removed. If the loop does not beat it, the agents
added nothing that refitting the literature could not.
`fitted_baseline_by_experiment` gives it at the end of every experiment.

## Reading a sweep

- Per cell: `holdout.csv` / `trajectory.json` (one row per scoring step) and
  `holdout.png`.
- Per sweep: `test_retest.json` gives the repeat-to-repeat consistency and
  `loop_vs_fitted_baseline`. Every summary lists unfinished and missing cells
  (also `MISSING_CELLS.txt`) and averages the loop and baselines over the same
  cells, aligned by experiment.
- Whether the best model changed: `incumbent_changed` and
  `incumbent_is_discovered` per step; `incumbent_report.py` tabulates them.
- Validity: `WORK_ROOT=<sweep dir> bash scripts/subjective_randomness/slurm/verify_holdout_run.sh`
  writes `VERDICT.md` (results present, raw columns only, agents confined to
  their copy, critique ran).

## Launching

`SMOKE=1 bash scripts/subjective_randomness/slurm/run_faithful_test_retest.sh`
first (one cheap cell), then without `SMOKE` for the sweep, and
`run_impossible_test_retest.sh` for the controls. Each chains setup, the cell
array (1 day, 16 CPUs / 64 GB, the same for impossible cells), up to two
rounds of automatic retries, and a summary. A sweep runs on one version of
the code, recorded at setup. Claude agents need `CLAUDE_AUTH=subscription`
or `api` and its key; agents that hit a usage limit wait for its reset (up
to 12 h), and a proposal's fit may take up to 30 minutes per sampling run.

## Results so far

See [BRIEF.md § 3](BRIEF.md#3-where-things-stand-28-september-2026). The
committed summaries in `data/results/holdout_test_retest/` are from the
**old** starting models and old code, not the current ones.

## Caveats

- `slurm/run_test_retest.sh` pins old model names and aborts; use
  `run_faithful_test_retest.sh` (its comments give outdated settings; the
  config is authoritative).
- The live presets do not use these settings (runbook § 3).
