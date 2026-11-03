---
name: person-breadth-and-momentary-odds-nulls
description: exp3 iter_1 c5 repair: per-person range breadth sd≈0.07 and trial-level rule-odds noise sd≈0.23 both collapse to beta_share (pool RMSE <0.002)
metadata:
  type: project
---
exp3 iter_1 candidate_5 (2026-10-31): person_specific_range_breadth was rejected as a dup of beta_share (RMSE 0.00048). The novelty pool (novelty_pool.json) has only number_set and target, so every person-indexed model is scored at its population-mean person.
Repair tests (2×300 fits, script /tmp/nov7/nov.py):
- marginalising over persons with Gauss-Hermite nodes: share_logit_sd 0.07±0.05, so it is still a dup;
- momentary rule-vs-range odds noise (GH-averaged sigmoid): odds_sd 0.23±0.16, ll −3654.2, RMSE 0.0014 vs beta_share, so it is a dup.
**Why:** exp3 shows no heterogeneity in breadth or rule odds, and the gate needs RMSE ≥0.002.
**How to apply:** avoid person- or trial-variability refinements of beta_share. Any graft must add a structural difference that survives when its parameter goes null. Related: [[individual-rule-vs-range-tendency]], [[two-cluster-range-concepts]].
