---
name: exp3-iter1-speed-repair
description: "Why exp3 iter_1 candidates timed out (1 core, load ~20, 170 ms/eval; dsum_decay ESS 28) and the width-grouped Chebyshev trick that made location_biased_range_prior ~9 min/chain"
metadata:
  node_type: memory
  type: project
  originSessionId: c6487a9a-2f52-4b68-a50e-1f19db019f7e
  modified: 2026-11-02T03:37:21.632Z
---

Exp3 iter_1 (2026-11-01): every candidate built on graded_prior_digit_sum_concepts was rejected as too slow. Admission runs warm-start NUTS at target_accept 0.99 with 3000 tune + 4000 draws per chain, under a 30-minute limit per sampling run, on 1 core at load ~20.
- The incumbent-style graph (Chebyshev in kappa over sets x 5050 ranges, plus a posterior @ membership matmul) costs ~170 ms per eval. The cumsum-over-(lo,hi)-grid trick was slower (gathers and cumsums on 154x10000 arrays are memory-bound).
- dsum_decay is barely identified (posterior sd 0.6 vs prior 1, ESS 28) and funnels against the digit-sum family weight. This is the geometry problem behind the graded models.
- Fix (candidate_1_repair_1): drop centre weighting and dsum_decay. Range log-likelihoods are then constant, so in prepare_observed precompute per set, and per observed (set, target) cell, sums over the ranges of each width of exp(ll - max) times Chebyshev coefficients of exp(slope*u) in the slope. In the graph that becomes cells x 100 x 8 contractions: 12 ms per eval, 16 steps, 0.08 s per iteration, about 9 min per chain. Verified exact against brute force.
- Location slope posterior 0.13 ± 0.36. Prediction RMSE vs centre_weighted_ranges_digit_sums (kappa posterior mean 0.039) ≈ 0.0037.
**How to apply:** for exp3 candidates, time the gradient (logp_dlogp_function) and a 600-iteration warm-start run before submitting; avoid parameters such as dsum_decay that the data barely constrain. See [[exp3-iter1-saturation-probes]].
