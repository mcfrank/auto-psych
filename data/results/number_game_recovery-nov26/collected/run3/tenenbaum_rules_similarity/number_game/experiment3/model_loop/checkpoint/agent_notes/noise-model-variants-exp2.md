---
name: noise-model-variants-exp2
description: experiment2 iter_0/candidate_4 tests on gamma_width_exception_family_concepts — inferred stray rate ≈0.89, one-sided ranges weight 0, at-most-one-stray far worse
metadata:
  type: project
---
experiment2 iter_0/candidate_4 (2026-10-30). Quick 2x300 NUTS LOO; incumbent gamma_width_exception_family_concepts -2278.3 (shape≈4.4, mean width≈49, lambda_0≈0.46). It is calibrated by set size, by p-bin, by distance outside/inside the span and by magnitude.
- Inferred per-example stray rate (alpha ~ Beta(9,1), likelihood grouped by (interval size, examples inside)): alpha = 0.890 ± 0.007, LOO -2278.6, p_yes RMSE vs incumbent 0.0085. Submitted as inferred_stray_rate_concepts; it may fail novelty. About 6x slower (147 s vs 25 s).
- One-sided ranges (1..b, a..100) as a third kind: weight 0.003, -2279.1, RMSE 0.005.
- "At most one stray per set" noise: -2586 (much worse). Independent per-example noise is essential, because the model predicts as little as 0.35 "yes" to shown examples (e.g. 83 after 83,38,58,68), and people agree.
**Why:** saves re-testing noise and range-kind variants.
**How to apply:** the fixed ALPHA 0.9 is already optimal. See [[incumbent-residual-diagnostics]] and [[concept-space-additions-map]].
