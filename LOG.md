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
