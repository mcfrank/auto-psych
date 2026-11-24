# Analysis log: every exploratory analysis path tried, including ones that did not work.

## 2026-11-24: Fig. 2 reference games: the existing-data loop's incumbents scored on the pre-registered experiment (exploratory)
- What: every incumbent (each `history.json` step's `best_model`) of Sherlock run 2's real_rep1-3, the loop whose
  models were promoted to seed the live chains, scored by lpd on the pre-registered experiment (14,404 trials), as
  `rsa_chain_models.py` scores the live chains' other models: fitted once on the pooled data at the bar's settings,
  posterior-mean class probabilities on the frozen design pool, `trial_lpd`. Output:
  `data/rsa/prereg/campaign_incumbents_lpd.json`; Fig. 2 draws one grey step line per replicate over the promoted seeds.
- Status: exploratory and descriptive. The pre-registered data are spent; this is run once and is not a test.
- 22 (replicate, model) pairs, 20 distinct names. 17 fitted (all converged, none unscored). 5 take an existing
  score because their source is byte for byte a bar model's (sha256): `rsa_l2` in each replicate, and real_rep1's
  `rsa_l2_singleton_distinct_feat_color_l0` and `rsa_l2_singleton_feat_color_valence_l0`. The scoring reproduced
  `prereg_analysis.json`'s lpd of the finals and the bar exactly (0 nats).
- Result (ΔELPD vs the best promoted seed, step 0 to 9). All three start at `rsa_l2`, -677.2.
  - real_rep1 ends at -1.0: it reaches the best promoted seed itself (0.0) at step 6, then the incumbent switches
    to a model 1 nat worse.
  - real_rep2 ends at -78.2. It rises at every incumbent change.
  - real_rep3 ends at -51.0. It falls twice when the incumbent changes: singleton_isolation_listener (-149.3)
    to singleton_isolation_l2 (-154.4) at step 2, and crowding_singleton_confusion_l2 (-43.8) to
    crowd_discrim_confusion_chromatic_l2 (-51.0) at step 8.
  - The loop selected on grouped CV over the existing data, so a step down here is a disagreement between that
    criterion and the new experiment, not an error.

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
