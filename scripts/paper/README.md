# Paper figures 2-4

Each step writes files the next reads. The three fitting steps run on
Sherlock, each as a chain of jobs: setup, an array of shards scoring the
models each run removed (Figure 2's "pruned during the run", about 50-80 per
run), then the assembly.

| step | on Sherlock | writes | cost |
|---|---|---|---|
| SR data | `bash scripts/paper/submit_figure_data.sh sr` | `data/paper/figure_data/subjective_randomness.json` | 176 removed models on each run's data and 254 other models on all runs pooled (86 shards), then 15 MCMC fits at ~7,700 trials and 7 on all runs pooled (~23,000) |
| number-game data | `bash scripts/paper/submit_figure_data.sh number_game` | `data/paper/figure_data/number_game.json` | the November sweep: 176 removed models on each run's data and 264 other models on all runs pooled, × 5 fits (88 shards), then 4-11 models × 5 per run and 6 × 5 pooled; work dir `$SCRATCH/auto-psych/paper/number_game_figure_data_nov26` |
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

- **Figure 2** (`fig2_elpd`, `figures/fig2.py`): per domain, each model's
  ΔELPD against the best starting model (dashed), the runs collapsed: the
  best discovered models and the starting models as points with ±2 SE of
  the paired difference (names: `model_names.yaml`; the paper each starting
  model implements, or null: `seed_papers.yaml`), and the non-winning models
  (the other models live at the end and those pruned during the runs) as
  one "search history" strip, each run's in the order it admitted them
  (`pruned_order`, `others_order`), under a step line tracing the loop's own
  incumbent (each `history.json` step's `best_model`, from the run
  directories `--histories` names). Each run has its own shade, by its best
  model (the best darkest), and so does its winner. The best discovered models are named
  at the top, the starting models at the bottom (by their paper, or their
  own name where seeds share one). Reference games are re-zeroed on the
  best textbook model (`best_textbook`), with the best promoted seed (the
  pre-registered bar) as a second line. The house style is Avenir where
  installed (`figures/style.py`; else the closest sans serif, said so).
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
    (the November sweep, `data/results/number_game_dataset-nov26/`, which
    started from the 2016 paper's seeds; each run drew its own items).
    The starting models are the three literature seeds (pruned in experiment
    1), the other models live at the end each run's other carried models;
    `number_game_figure_data.py` refits them all, per run and pooled. The SE
    is stimulus-clustered ((set, target) cells). Figure 2 draws instead the
    pooled models' log predictive density on held-out responses
    (`data/paper/number_game_unseen_items.json`, `number_game_unseen_items.py`):
    by default on every (set, target) item no run drew (`figures
    --ng-test-set unseen_example_sets` draws the stricter example sets no run
    drew: 11 sets, too few to separate the models), the winners and seeds from their pooled fits, the
    non-winning models from their all-data pooled fits at the runs' settings
    (scored in shards on Sherlock:
    `STRIP_SHARDS=22 sbatch --array=0-21 scripts/paper/number_game_unseen_items.sbatch`,
    then the assembly without `--array`), each against the best seed on that
    test (SE clustered by example set). Figures 3 and 4 keep the
    cross-validated comparison.
  - Every domain also draws the models pruned during the run (grey dots):
    every model the run admitted and later pruned, retired or dropped,
    scored on the same data as the rest of its column (`pruned_during_run`;
    one whose fit fails or does not converge is listed with the reason in
    `pruned_unscored` instead).
- **Figure 3** (`fig3_effects`): per domain, from the figure data's `conditions` (pooled best model and
  best seed): (a) observed against predicted, with R² and RMSE; (b) the classic effect as its original paper
  drew it; (c) a new effect only the winner captures. Reference games use the pre-registered experiment,
  their displays from `data/rsa/viz/panel_displays.json`. The number game's (b) and every (c) are pending.
- **Figure 4** (`fig4_ablation`): each domain winner's backward elimination on two axes: the path as
  ELPD-CV vs. the full model, each step ±1 SE of its difference from the previous round's model, and the
  kept column as each removal's ELPD-CV vs. the final model, ±1 SE of that difference (the SEs the
  stopping rule used); every mechanism's name and origin are in `mechanisms.yaml`.
- **Ablation figure** (`fig_ablation`, `uv run python -m scripts.paper.ablation_figure`,
  seconds): the backward elimination of each domain's overall winner, one
  panel each: the subjective-randomness final best on all runs pooled
  (Figure 3's, `pooled.best_model`; its elimination in `docs/sr_ablation.md`),
  chain 0's reference-game final, the best on the pre-registered test
  (`data/rsa/c0_ablation/eliminate/`, `docs/auto_rsa/ABLATION_C0.md`; it
  replaces the consolidation's superset elimination), and the number-game
  November sweep's final best on all three runs' data pooled
  (`data/results/number_game_dataset-nov26/analysis/ablation/`,
  `docs/ng_ablation.md`; Figure 3's best). That elimination is not run yet
  (its final is chosen by the pooled comparison), so until it is committed
  draw the panel as pending: `ablation_figure --pending-domains number_game`;
  without that a missing record raises. Each panel is the path of the model kept
  after each round: its grouped-CV ELPD against the full model's, by its
  number of mechanisms, each step named by the mechanism it dropped, with ±1
  clustered SE of the model's own ELPD-CV (`elpd_se`); the path's last step,
  grey and dashed, is the removal that stopped the elimination (≥ 2 SEs of
  its paired comparison). `elpd_se` is recorded since the elimination began
  recording it: an older `elimination.json` lacks it, and the figure stops
  until that ablation's plan step is rerun (it replays the saved CVs, no fits).

`figures --figures 2 3` draws only those figures (Figure 4 needs every
drawn domain's data). `figures --mock-domains <domain> ...` marks those domains' titles "(MOCK
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
