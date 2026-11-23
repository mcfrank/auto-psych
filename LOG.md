# LOG

Lessons from work on this project, written by agents (and Ben) at the end of each task. The
research hub (`~/Documents/research-hub`) reads this file during consolidation and promotes
recurring lessons into its `knowledge/` files. Never delete entries; add a new one that
supersedes an old entry instead.

Entry format (newest at the bottom):

## YYYY-MM-DD: <task, in a few words>
- Lesson: <abstract principle, not a narrative>
- Why: <what happened that shows it>
- Evidence: <commit, file, or branch>

## 2026-11-20: Figure 4 error bars matched to the elimination's stopping rule
- Lesson: An error bar should show the comparison the decision was made on. For a sequential procedure
  (backward elimination), plot each step's SE vs the previous step (the stored candidate `se`), not
  vs the starting model (`se_vs_start`). Draw quantities on very different scales on separate axes.
- Why: Fig 4 used ±SE vs the full model for every point, so SR's stopping step (2.34 SE vs the
  previous round) plotted as −51 ± 19. One shared y axis let the kept-removal costs (hundreds of nats)
  flatten the path (a few to ~50 nats).
- Lesson: Rotated point labels that hang below their points need vertical room in the axes. When the
  y range is fitted to the data, reserve that room explicitly (solve for the y limit), or the names
  run into the tick labels.
- Why: After the split, the number game's long step names forced its path into the top ~quarter of
  the panel at the fixed 4.9 in height.
- Evidence: branch `fig4-stopping-rule-se`, `scripts/paper/figures/fig4.py` (`ablation_points`),
  `tests/test_paper_figures.py`.
- Note: two test modules fail at collection on `paper-figures`, independent of this change:
  `tests/test_paper_ablation_figure.py` imports the removed `scripts.paper.figures.legacy`, and
  `tests/test_paper_recovery_figures.py` needs `plotnine`, which the uv env (`--group dev`) lacks.
- Follow-up (2026-11-20): Name the steps of a path with rotated x tick labels (two balanced lines at most),
  not annotations hanging below the points, so the y range fits only the data. Then place the rows from their
  measured tick-label and title extents (`stack_rows` in fig4.py), because the room the names need differs
  by domain. Evidence: the number-game path went from the top ~quarter of its panel to filling it.
- Follow-up (2026-11-20): Figure 4 no longer has a mechanism-count x axis. Each path tick names the mechanism
  that step dropped (the first reads "full model"), and the x title appears only on the bottom row. Full-model
  counts for the caption: subjective randomness 15, reference games 16, number game 14. Name labels are in ink;
  only the points carry the new/literature colour (light-blue 7 pt text reads poorly). "kept" heads the column,
  above it, so leaders to names hanging below short axes never cross it.
- Follow-up (2026-11-20): Encode the same thing with the same colour in every figure. Fig 4 now uses Figs 2-3's
  scheme: blue `BEST` (#2171b5) for what auto-psych discovered, seed grey `SEED` (#b3b3b3) for what came from the
  literature, both imported from fig3.py. The path line and all error bars are INK_2 and the full model is an
  INK square. Why: two shades of one blue on a near-black line made the light points read as glitches. Ben
  rejected filled/hollow (A) and blue/orange (B) variants.

- 2026-11-20 (hub): merged paper-figures + fig4-stopping-rule-se into main (9ee01d66); removed the
  obsolete `scripts/paper/ablation_figure.py` and its test (0c07c3c3), superseded by
  `scripts/paper/figures/fig4.py`. Lesson: when a branch deletes a module, grep for importers
  (`git grep <module>`) before merging; dead importers break test collection on main.
- 2026-11-20 (hub): resolved the pre-merge stash against merged main. Main's figure package already
  did what the stashed draft edits did (November number-game data, unseen-items test set, per-run
  colours, number-game row (a) in Fig 3); the drafts and fig3_conditions.* were deleted on main.
  Only the Tenenbaum 1999 citation in seed_papers.yaml was new; applied it and regenerated Fig 2.

- 2026-11-21 (fig3-new-effects): Figure 3 row (c) built from the figure data's `conditions`, not from a
  trial-level export. Lesson: a pattern defined on trials can usually be recomputed from condition-level
  tables (observed proportion, n, per-condition predictions) by weighting each condition by its n; do that
  rather than re-adding a trial export, and check that the totals (trials, displays, sets) match the audit
  that first defined the pattern before trusting the plot.
  - Why: the conditions are the figure's single source of truth, so every row of a column then uses the same
    best model and best seed; re-deriving from them also caught nothing new to tune, since the audit's
    definitions reproduced its numbers (SR 2,524 trials, 11.8/13.0/22.7%; RG 32 displays, 1,475 trials).
  - Evidence: branch `fig3-new-effects`, `scripts/paper/figures/fig3.py` (`streak_choices`, `match_roles`,
    `digit_confidence`), tests in `tests/test_paper_figures.py`.
  - Note: panel titles inside a row with a row title need extra vertical room; lift the row title and widen
    hspace rather than shrinking fonts. Pre-existing failures on main (macOS): `os.sched_getaffinity` in
    `src/models/pymc_inference.py` breaks 24 tests in information_criteria / rsa_chain_models /
    sr_figure_data, and `test_paper_recovery_figures.py` needs plotnine.

- 2026-11-21 (fig3-new-effects): Fig. 3's number-game column now shows run 3's winner
  (`fig3.COLUMN_MODEL`), from `figure_data/number_game_model_conditions.json`
  (`scripts/paper/number_game_model_conditions.py`).
  - Lesson: to reuse one model's cluster fit locally, compute its cache fingerprint with the repo's own
    `fit_fingerprint` (rebuild the same model directory, pooled responses and resolved settings), then copy
    only that `.nc` file and read it with `load_cached_fit`, which never samples. Why: the cache holds one
    file per training split (full data plus each CV fold), and file size cannot tell them apart. Evidence:
    the computed fingerprints matched the files on Sherlock for both run 2's winner (6345f7dc…) and run 3's
    (90b1b6b7…).
  - Lesson: PyTensor 2.38 adds `-ld64` on macOS ≥ 15, and the current Xcode linker rejects it ("library 'd64'
    not found"). Its Python backend (`cxx=`) runs far too slowly for posterior predictives (>30 min,
    unfinished). Fix it in the environment, not the code: set `PYTENSOR_FLAGS="cxx=<wrapper>,blas__ldflags="`,
    where the wrapper is a clang++ script that drops `-ld64` (46 s). Evidence: this task.

- 2026-11-21 (pre-submission review, round 4; `paper/pre_submission_review_2026-11-21.md`):
  - Compile a copy of the Overleaf clone, never the clone itself. It tracks `auto-psych.aux` and
    `supporting-info.aux`, and its `latexmkrc` spawns a nested `latexmk` on the other document
    without `-outdir`, so a build inside the clone dirties it. Building the SI before the main
    file resolves the `xr` cross-references.
  - Where the numbers live:
    - Fig. 2: `data/paper/figure_data/*.json` (`pooled`, `runs`). The number-game panel is
      `number_game_unseen_items.json`, using whichever test set
      `scripts/paper/figures/__init__.py` `ng_test_set` selects (default `unseen_items`).
    - Table S2: `data/paper/tables/information_criteria_table.tex` and
      `data/paper/information_criteria.json`.
    - Ablations: the three `elimination.json` paths in `scripts/paper/figures/fig4.py`
      `sources()`.
    - Reference-game parameters: `data/paper/parameter_summaries/rsa_finals/parameters.csv`.
    - Preregistration: `docs/auto_rsa/PREREG_RESULT_REPORT.md`.
  - Byte-compare `figures/*.pdf` in the clone with `data/paper/figures/` before reading numbers
    off a figure. Fig. 3 in the manuscript was the deleted `fig3_draft`, not main's
    `fig3_effects`.
  - When a figure's data switch (as the number-game test set did on 2026-11-20), check its
    caption and the SI table that describes it, which can be left behind.
  - Crossref rate-limits bursts (HTTP 429). Retry with backoff. arXiv's API (`export.arxiv.org`,
    https) is a quick check for arXiv-only references.

- 2026-11-21 (pre-submission review, round 5; `paper/pre_submission_review_2026-11-21b.md`):
  - A full compile of the copy needs the SI to finish: the SI's UTF-8 listing errors stop
    `latexmk`, which leaves the SI citations undefined. Run `bibtex supporting-info` and then
    `pdflatex` twice by hand.
  - The Overleaf figure files can change on Overleaf without changing the code: `fig4_ablation.pdf`
    was relabelled there this round. Re-run the byte comparison on every round, not just once.
  - When a footnote or bug note appears, find the bug in `paper/results_update.md` (M-items) or the
    `CAMPAIGN_C*_REPORT.md` files and check which domain and run it belongs to. This round's
    PSIS-LOO bug note (subjective randomness) was placed in the reference-games section.
  - Fig. 3 data sources: `scripts/paper/figures/fig3.py` `load()` (reference games use prereg
    rows only; the number game uses `number_game_model_conditions.json` for run 3's winner). Its
    helpers (`digit_confidence`, `match_roles`, `streak_choices`, `r2`) recompute every Fig. 3
    and number-game text number with `.venv/bin/python` from the repository root. Row (c) is
    exploratory by its own docstring.
  - The "best" number-game model differs by criterion: run 3 on unseen items, run 2 on pooled CV,
    AIC, BIC and BF. Check which one any new text means.
  - Fig. 4 mechanism labels map to `elimination.json` names in round order. For example, RG "has a
    word of its own" = `avoidable`.
- 2026-11-21 (recovery-figs-style): the SI recovery figures (`scripts/paper/recovery_figures.py`) now use
  Figures 2-4's matplotlib house style, not plotnine.
  - Lesson: to restyle a figure without touching its data, keep the frame-building functions byte-identical,
    dump their frames to CSV before and after (`cmp`), and tag each plotted line with a `gid`
    ("best", "seed", "run") so tests can compare every line's x/y against the frame. Why: it proves "same data"
    mechanically, and the gids also let a test pin the palette. Evidence: commit 446dae5d,
    `tests/test_paper_recovery_figures.py`.
  - Lesson: with `sharey=True`, set y limits once, after every panel is drawn. Calling
    `ax.set_ylim(bottom=0)` per panel freezes the top at the first panel's data and silently clips the
    others (the number game's third panel lost its top). Why: `set_ylim` turns off autoscaling for every
    shared sibling. A test that every plotted value lies inside its panel's y range catches it.
  - Note: `MaxNLocator(5, steps=[1, 2, 5, 10])` keeps free-scale tick labels short (no "0.025") so they stay
    clear of the panel to the left. The main figures' style needs Avenir (macOS ships it). The SI captions
    still say "dashed orange line" (now grey dashed). The combined figure still says "Number sequence
    generalization" where Figures 2-4 say "Number game". `test_paper_figures.py::test_figures_3_and_4_...`
    fails on `paper-figures` before this change too (a missing data file).
## 2026-11-21: Figure 3 rows (a) and (b) on Figure 2's split
- Lesson: For a binary outcome, a trial's PSIS-LOO predictive probability needs no posterior-predictive pass:
  P_loo(y=1) is exp(loo_i) when y=1 and 1 - exp(loo_i) when y=0, so the pointwise `loo_i` that gave the ELPD
  also give held-out per-condition predictions that sum back to exactly that ELPD.
- Why: Figure 3's subjective-randomness panels needed LOO predictions from Sherlock's cached pooled fits; this
  took 2 min per job and its summed log density equals Figure 2's ELPD-LOO to 1e-6 by construction.
- Lesson: When exporting held-out predictions to pair with an existing score, store each condition's summed log
  predictive density next to its mean prediction, and test that the conditions sum to the published score. That
  one test pins model, fit, split and row alignment together. Separately check row alignment against an
  independent quantity (here, in-sample P(left) from the log-likelihood vs the figure data's, within 0.001).
- Why: `tests/test_paper_figures.py` ties `*_heldout_conditions.json` to Figure 2's ELPD-LOO and to
  `number_game_unseen_items.json`'s lpd (reproduced to 1e-9 with the same draws and seed).
- Lesson: To run branch code on Sherlock without touching the shared checkout, rsync `src/`, `scripts/`,
  `pyproject.toml` and the needed figure-data files into `$SCRATCH/.../code` and symlink gitignored data
  (`data/number-game`) from `~/auto-psych`. Fit-cache fingerprints depend only on model source, responses
  bytes and sampler settings, not on the code version, so cached fits load from any copy.
- Lesson: On macOS the fast suite shows ~230 PyTensor `CompileError` failures unless
  `PYTENSOR_FLAGS="cxx=<clang++ wrapper dropping -ld64>,blas__ldflags="` is set (see the earlier 2026-11-21 entry on
  this branch). They come from the environment, not the code.
- Note: the number game's held-out set has 23,360 items, and a scatter at fixed alpha turns into a solid block.
  `fig3._scatter` now caps each point's alpha at 250 / n points, which leaves SR and RSA unchanged. Of the 100
  numbers, the four-example profile ("80,70,90,100") keeps only 55 (runs drew 45) and "100" keeps 90.
- Evidence: branch `fig3-heldout`, `scripts/paper/fig3_heldout_conditions.py` (+ `.sbatch`), Sherlock
  `$SCRATCH/auto-psych/fig3_heldout/` (jobs 53258041, 53258042).
## 2026-11-21: Figure 3 row (c) reference games: "word fits 3 objects" replaces "no word"
- Lesson: when a figure panel shows a group that a search picked, select the group with the search's own
  feature function (`fig3_patterns.FEATURES`, imported lazily inside the function because that module imports
  `fig3`), keyed by `ROW_C`. Don't copy the definition. Why: the panel then can't drift from what
  `check_row_c` validates. Evidence: `fig3.fits_three_roles`, branch `fig3c-rsa-fits3`.
- Lesson: the search's `trials` column sums `n` over class rows, and every class row carries its display's
  `n`. So it counts a display once per class: 11,200 for "fits 3" against 3,472 real trials over 75 displays.
  Report trials from unique displays.
- Lesson: in a group with a fixed number of fitting objects, the mean choice rate is fixed by construction
  (about 1/3). Plot how choices split across object roles instead. Count each identical copy as one object:
  rate = class share / copies, weighted by n x copies.
- Note: the earlier reference-game panels in Fig. 3 use binomial CIs. This panel clusters by display instead,
  using `cluster_dse` on the weighted residuals, because objects in one display share trials. Binomial CIs
  would be about half as wide for "fewest features" (0.017 vs 0.043).
## 2026-11-21: pre-submission review round 6 (paper 838c741, code 97778ec3)
- Lesson: before trusting a quoted participant-level summary (completion time, pay rate), trace it to a
  committed output file, not just a script. Why: the SI's old SR time (3.52 min) matched the pre-October
  `data/results/human_experiment/completion_times.csv` exactly, not sr-oct26, and the new 5.49 comes from
  `scripts/sr_completion_times.py`, which only prints. Evidence: `paper/pre_submission_review_2026-11-21c.md` #4.
- Lesson: `information_criteria.json` has three ESS fields per hierarchical SR record: `evidence.min_ess` (outer
  importance weights), `evidence.inner_min_ess` (worst person) and `marginal.min_ess` (the IS marginal behind
  SR AIC/BIC; 1.3 for run 2). `marginal_log_lik` maximizes a Laplace approximation, then importance-samples at
  the mode. Check which field and which step any paper sentence describes.
- Lesson: to check a Fig. 3 row-(c) claim held out, run the row's own helper (`fig3.digit_confidence`,
  `fig3.streak_choices`) on `*_heldout_conditions.json` with `PYTHONPATH=.` and `.venv/bin/python`. Don't name
  a scratch script `src.py`: it shadows the repo's `src` package.
- Note: a default `latexmk` SI build stops on the listings' UTF-8 errors before bibtex, so citations come out
  undefined; run `bibtex supporting-info` and pdflatex twice by hand in the scratch copy.
## 2026-11-23: where the N for Figs. 2-4 live (hub brief `briefs/2026-11-23-auto-psych-figure-ns.md`)
- Lesson: before counting N off a figure, check which commit drew the Overleaf copy (`git hash-object` against
  `git rev-parse <commit>:data/paper/figures/<f>.pdf`). Why: Overleaf's `fig3_effects.pdf` was `origin/main`
  221d8234's (Daniel, unmerged into local main); its NG (b) and (c) panels have different subsets and N from main's.
- Where the counts are: SR `figure_data/subjective_randomness.json` `pooled.n_trials/n_participants` (+ `len(conditions)`
  = 416 pairs, the SE clusters); RG prereg `data/rsa/prereg/run/prereg_analysis.json` `n_trials/n_displays/n_participants`;
  NG `data/paper/number_game_unseen_items.json` `<test set>.n_responses/n_items/n_example_sets`; Fig. 4 `elimination.json`
  `n_trials`. Fig. 3 subset counts are only computable by running the `fig3.py` helpers.
- Lesson: the NG pooled CV data (22,680 rows: Fig. 4, the CV comparison, Fig. 3 seen items) repeat responses that two or three
  runs drew from one item (19,554-19,741 distinct). Report rows and distinct responses separately.
- Note: Fig. 2 RG re-zeroes on the textbook seed but draws the winners' SE against the promoted bar (`vs_best_starting`
  holds the SEs for the textbook zero, about 3× wider). The RG ablation's CV-unit count needs `finals_pool/pooled.csv` (Sherlock).
