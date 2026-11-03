---
name: range-first-satisficing-search
description: exp3 iter_2 c0 repair: rule odds × exp(-effort·per-example range fit); MAP effort≈0.1, RMSE 0.0036 vs beta_share; K-sample hypothesis inclusion is a λ reparam
metadata:
  type: project
---
exp3 iter_2 candidate_0_repair_1 (2026-10-31), submitted as range_first_satisficing_search on the beta_share base, after target_primed_rule_retrieval ([[target-primed-rule-retrieval]]) was rejected as a dup of beta_share (RMSE 0.0012).
numpy MAP: effort 0.10, nll 3646.88 vs base 3647.09; p RMSE 0.0036 vs base; effort ±0.5 costs ~3 nats.
Null: "consider K concepts sampled from the prior" (inclusion 1-(1-π)^K): interval priors are all tiny, so it just rescales λ → RMSE 0.0003.
Scripts /tmp/c0r1 (fit.py numpy base loader importing candidate's prepare_observed, sat.py).
**How to apply:** a set-dependent rule-odds shift (range fit per example) moves predictions enough to clear novelty; constant/prior-proportional reweightings get absorbed by λ.
