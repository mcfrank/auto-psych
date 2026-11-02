---
name: kind-uncertainty-shifted-criterion
description: exp3 iter_0 c4 repair: entropy over concept kinds (families + range) instead of concepts; shift +0.24±0.19, pool RMSE 0.0054 vs beta_share
metadata:
  type: project
---
exp3 iter_0/candidate_4_repair_1 (2026-10-31), submitted as kind_uncertainty_shifted_criterion: the incumbent uncertainty_shifted_criterion, but the criterion shift uses the normalised entropy of the posterior over the 8 rule families plus "range", not over the 145 concepts.
Quick 2×300 fit: shift +0.24±0.19 (the incumbent's concept-entropy shift is −0.19±0.14, the opposite sign), mean ll −3653.3 vs incumbent −3653.1. Novelty-pool RMSE 0.0077 vs incumbent, 0.0054 vs beta_share. Smoke check OK in 17 s.
The first attempt (entropy-scaled gain, [[uncertainty-damped-decisiveness]]) was rejected as a duplicate of beta_share (RMSE 0.0011).
Script: /tmp/nov5/nov.py <model.py ...> saves pool p_yes means as .npy. It is slow on this 1-core box.
**How to apply:** a readout whose parameter collapses to 0 duplicates beta_share. Coarse uncertainty measures keep enough variance to pass novelty.
