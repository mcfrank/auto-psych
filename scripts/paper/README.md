# Paper figures 2-4

Three steps; each writes files the next reads.

| step | command | writes | cost |
|---|---|---|---|
| SR data | `uv run python -m scripts.paper.sr_figure_data` | `data/paper/figure_data/subjective_randomness.json` | hours: 15 MCMC fits at ~7,700 trials, then 7 on all runs pooled (~23,000) |
| RSA data | `uv run python -m scripts.paper.rsa_figure_data` | `data/paper/figure_data/rsa.json` | seconds: reads existing aggregates |
| figures | `uv run python -m scripts.paper.figures` | `data/paper/figures/fig{2,3,4}_*.{pdf,png}` | seconds |

What each figure shows:

- **Figure 2** (`fig2_elpd`): per domain, a column per run, in ELPD
  difference from the best starting model: the run's starting models
  (dashes), its other models live at the end (small dots) and its best
  discovered model (±2 SE of the paired difference, labelled).
  - Subjective randomness: PSIS-LOO on each run's final pooled data; the
    starting models are the run's seeds, refitted there by
    `sr_figure_data.py` (they were pruned in experiment 1) with the live
    run's sampler settings. The SE is stimulus-clustered, as in the loop's
    pruning. The other models live at the end are the loop's own ELPD-LOO on
    the same data and settings (no MCMC; its winner's agrees with the refit
    within 1 nat).
  - Reference games: log predictive density on the held-out pre-registered
    experiment (`data/rsa/prereg/run/prereg_analysis.json`). A chain's
    starting models are its share of the promoted seeds (run 2's best
    models, refitted; `data/rsa/live_seeds/chains/chains.json`); the zero is
    the best of all of them, the pre-registered bar, and a final's difference
    and SE are the pre-registered test's. The textbook RSA models are not
    drawn. The campaign's "chains" are its runs; no other model of a chain
    was scored on the pre-registered data.
- **Figure 3** (`fig3_conditions`): all runs' data pooled, each condition's
  observed proportion against the prediction of the best discovered model and
  of the best seed, both fitted on the pooled data (the figure data's
  `pooled`).
  - Subjective randomness: every run's winner and the seeds are fitted on the
    three runs' final data together (each run's participants kept apart; the
    collected runs hold no Prolific IDs, so a person in two runs counts as
    two), every model at the same settings: `--pooled-draws` (1,500) draws per
    chain, `--pooled-tune` (4,000), `--pooled-target-accept` (0.95),
    `--pooled-random-seed` (20261018) and `--pooled-fit-method`
    (`nuts_no_jitter`: every chain starts at the model's initial point; with
    PyMC's jittered starts some chain, a different one at each seed tried,
    started so far out that its step size fell to zero and it never moved). The
    winner and the seed with the highest ELPD-LOO there are drawn; any winner
    or seed whose fit did not converge, or whose PSIS-LOO is unreliable, stops
    the script (which model is best is then not known). Conditions are
    unordered sequence pairs.
  - RSA: the fits on the pooled data in `data/rsa/viz/panels.json`; the best
    model is the chain final with the highest held-out density, the best
    starting model the best promoted seed (panels.json
    records no pooled ELPD). Conditions are (display, object class) cells.
- **Figure 4** (`fig4_pseudocode`): hand-written annotated pseudocode of each
  domain's Figure 3 model, `pseudocode/<domain>.yaml`. Each spec records its
  model file's SHA-256, and the figure script refuses a spec whose model
  changed or that is not Figure 3's best model.

`figures.py --mock-domains <domain> ...` marks those domains' titles "(MOCK
DATA)", for drafting the figures' look from invented figure data.

The number game is drawn as "results pending" until its figure data exist
(`--pending-domains` controls this; any other missing domain raises).

Run the SR data step on Sherlock: `sbatch scripts/paper/sr_figure_data.sbatch`
from the checkout. Its fits take about 16 GB (kept on `$SCRATCH`). Each run's
result, and the pooled one, is saved as `scored.json` in the work directory, so
a rerun reuses it. Locally, `--discard-fits-after-run` deletes each step's fits
once its result is saved (peak ~6 GB plus swap); on macOS it also needs a
stand-in for Linux's `os.sched_getaffinity`.
