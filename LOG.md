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
