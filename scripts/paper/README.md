# Paper figures 2-4

Each step writes files the next reads. The three fitting steps run on
Sherlock, each as a chain of jobs: setup, an array of shards scoring the
models each run removed (Figure 2's "pruned during the run", about 50-80 per
run), then the assembly.

| step | on Sherlock | writes | cost |
|---|---|---|---|
| SR data | `bash scripts/paper/submit_figure_data.sh sr` | `data/paper/figure_data/subjective_randomness.json` | 176 removed models on each run's data and 254 other models on all runs pooled (86 shards), then 15 MCMC fits at ~7,700 trials and 7 on all runs pooled (~23,000) |
| number-game data | `bash scripts/paper/submit_figure_data.sh number_game` | `data/paper/figure_data/number_game.json` | 170 removed models on each run's data and 250 other models on all runs pooled, × 5 fits (84 shards), then 7-10 models × 5 per run and 6 × 5 pooled |
| RSA chain models | `bash scripts/paper/submit_rsa_chain_models.sh` | `data/rsa/prereg/chain_models_lpd.json` (sums only) | 224 numpyro fits on the pooled data (28 shards of 8) |
| RSA data | `uv run python -m scripts.paper.rsa_figure_data` (anywhere) | `data/paper/figure_data/rsa.json` | seconds: reads aggregates, the chain models' sums included (`--chain-models None` leaves them out) |
| figures | `uv run python -m scripts.paper.figures` (anywhere) | `data/paper/figures/fig{2,3,4}_*.{pdf,png}` | seconds |

Every array task writes one record per model and skips models whose record
is current, so resubmitting a submitter redoes only what is missing; the
assembly raises naming any model without a record. A removed model pruned at
the end of its run's final experiment (SR and number game) is not refitted:
that experiment's `history.json` records its ELPD on exactly the final data
at the run's settings (`removed_models.recorded_scores`; a refit reproduced
it to four decimals).
| figures | `uv run python -m scripts.paper.figures` | `data/paper/figures/fig{2,3,4}_*.{pdf,png}` | seconds |

What each figure shows:

- **Figure 2**, in two layouts of the same comparison: `fig2_elpd` (models
  across, ELPD up) and `fig2_elpd_horizontal` (a forest plot: one row per
  model, ELPD across). Per domain, each model's ELPD (a dashed line marks
  the best starting model's), the runs collapsed: the best discovered
  models and the starting models as points with ±2 SE of the paired
  difference (names: `model_names.yaml`; the paper each starting model
  implements, or null: `seed_papers.yaml`), and the non-winning models
  (the other models live at the end and those pruned during the runs) as
  one strip, each run's in the order it admitted them (`pruned_order`,
  `others_order`); every model the loop wrote is coloured by its run
  (`RUN_COLORS`). The vertical layout
  names the best discovered models at the top and, at the bottom, the
  starting models a paper proposed (by the paper) and the best starting
  model; the horizontal one names every row. A point whose paired SE
  is not computed yet is drawn hollow, a strip not scored yet "pending".
  Subjective randomness and the number game are drawn from all runs pooled
  (each run is scored on its own data); reference games from every chain on
  the pre-registered test. The per-run records below are still written.
  - Subjective randomness: PSIS-LOO on each run's final pooled data; the
    starting models are the run's seeds, refitted there by
    `sr_figure_data.py` (they were pruned in experiment 1) with the live
    run's sampler settings. The SE is stimulus-clustered, as in the loop's
    pruning. The other models live at the end are the loop's own ELPD-LOO on
    the same data and settings (no MCMC; its winner's agrees with the refit
    within 1 nat).
  - Reference games: log predictive density on the held-out pre-registered
    experiment (`data/rsa/prereg/run/prereg_analysis.json`). Each chain's
    other live models and removed models are `rsa_chain_models.py`'s,
    scored exactly like the finals and the bar (fitted on the same pooled
    data at the bar's settings; the job first checks it reproduces their
    pre-registered lpd) but after the fact: descriptive only, since the
    pre-registered data are spent. A chain's
    starting models are its share of the promoted seeds (run 2's best
    models, refitted; `data/rsa/live_seeds/chains/chains.json`); the zero is
    the best of all of them, the pre-registered bar, and a final's difference
    and SE are the pre-registered test's. The textbook RSA models are not
    drawn. The campaign's "chains" are its runs.
  - Number game: ELPD from grouped cross-validation (4 folds of whole
    example sets, the runs' own criterion and settings, from their
    `holdout.json`, which must agree) on each of the three dataset runs'
    final data, 7,560 responses each, and on all three pooled, 22,680
    (`data/results/number_game_dataset-oct26/`; each run drew its own items).
    The starting models are the three literature seeds (pruned in experiment
    1), the other models live at the end each run's other carried models;
    `number_game_figure_data.py` refits them all, per run and pooled. The SE
    is stimulus-clustered ((set, target) cells).
  - Every domain also draws the models pruned during the run (grey dots):
    every model the run admitted and later pruned, retired or dropped,
    scored on the same data as the rest of its column (`pruned_during_run`;
    one whose fit fails or does not converge is listed with the reason in
    `pruned_unscored` instead).
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
  - Number game: the three runs' final data pooled (22,680 responses; the
    runs hide participant ids, so nothing is renumbered); the best model is
    the winner and the best starting model the seed with the highest ELPD-CV
    there. Conditions are (example set, target) cells, 9 responses per run
    that drew the cell: 2,147 cells, 1,845 drawn by one run, 231 by two and
    71 by all three.
- **Figure 4** (`fig4_pseudocode`): hand-written annotated pseudocode of each
  domain's Figure 3 model, `pseudocode/<domain>.yaml`. Each spec records its
  model file's SHA-256, and the figure script refuses a spec whose model
  changed or that is not Figure 3's best model.

- **Ablation figure** (`fig_ablation`, `uv run python -m scripts.paper.ablation_figure`,
  seconds): the backward elimination of each domain's overall winner, one
  panel each: the subjective-randomness final best on all runs pooled
  (Figure 3's, `pooled.best_model`; its elimination in `docs/sr_ablation.md`)
  and the reference-game consolidation (`data/rsa/consolidate/eliminate/`).
  The number game was not ablated. Each panel is the path of the model kept
  after each round: its grouped-CV ELPD against the full model's, by its
  number of mechanisms, each step named by the mechanism it dropped; the
  path's last step is the removal that stopped the elimination, below the dashed
  grey line 2 clustered SEs (of that comparison) under the final model.

`figures.py --mock-domains <domain> ...` marks those domains' titles "(MOCK
DATA)", for drafting the figures' look from invented figure data.

The number game is drawn as "results pending" until its figure data exist
(`--pending-domains` controls this; any other missing domain raises).

Run the number-game data step on Sherlock too (both steps:
`bash scripts/paper/submit_figure_data.sh number_game`; fits on `$SCRATCH`,
per run in `<run>/` and pooled in `pooled/`; a rerun reuses them and each
`scored.json`).

Run the SR data step on Sherlock: `sbatch scripts/paper/sr_figure_data.sbatch`
from the checkout. Its fits take about 16 GB (kept on `$SCRATCH`). Each run's
result, and the pooled one, is saved as `scored.json` in the work directory, so
a rerun reuses it. Locally, `--discard-fits-after-run` deletes each step's fits
once its result is saved (peak ~6 GB plus swap); on macOS it also needs a
stand-in for Linux's `os.sched_getaffinity`.
