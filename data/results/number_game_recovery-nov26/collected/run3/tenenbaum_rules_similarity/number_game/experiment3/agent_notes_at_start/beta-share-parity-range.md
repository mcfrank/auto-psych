---
name: beta-share-parity-range
description: exp2 iter_1 cand5 repair: parity-within-range + beta-share width — parity kind weight 0.003, LOO −1.4 vs incumbent, RMSE 0.0064
metadata:
  type: project
---
experiment2 iter_1/candidate_5_repair_1 (2026-10-30), beta_share_parity_range_concepts = gamma_width_parity_range_concepts with the beta-share width prior. Quick 2x300 LOO -2274.9 vs incumbent -2273.6; kind weights rules 0.46, ranges 0.54, parity-ranges 0.003; mean_share 0.48, conc 5.4. Posterior-mean p_yes RMSE vs incumbent on observed cells 0.0064, so it should pass novelty. Fit ~20 s per chain.
The first attempt (beta_share_prior_leaking_judgment, see [[beta-share-prior-leaking]]) was rejected as a near-duplicate of beta_share_width_prior_leaking_judgment (RMSE 0.00034).
**Why:** on exp2, parity-within-range concepts earn almost no weight, unlike exp1 ([[incumbent-residual-diagnostics]]).
**How to apply:** don't add parity ranges to beta-share again. Before submitting a graft, check the RMSE against every live model, not only the incumbent.
