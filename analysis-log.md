# Analysis log: every exploratory analysis path tried, including ones that did not work.

## 2026-11-21: Figure 3 row (c), reference games: pick changed from "no word" to "word fits 3 objects" (exploratory)
- Change: `ROW_C["rsa"]` in `scripts/paper/figures/fig3.py` went from ("query", "no word") to
  ("objects the word fits", "3"). Ben made the change after viewing the search's candidates
  (`scripts/paper/fig3_patterns.py` -> `data/paper/figure_data/fig3_patterns.json`).
- Reason: no-word (prior) queries are an edge case of the task. Displays where the word fits exactly 3 objects
  are 27% of the search's condition rows (z = 3.5, effect 0.053).
- Both picks are still among the search's three candidates. The third is "objects the word fits" = "4"
  (z = 2.2, effect 0.081). Nothing in the search changed.
- Status: exploratory. The search used all the in-sample data, and the pick was made by hand after seeing it.
- Panel: per-object choice rate by role in those displays. People's 95% CIs are clustered by display.
  "All fitting alike" (7 displays, 321 trials, about 1/3 per object for everyone) is dropped.
