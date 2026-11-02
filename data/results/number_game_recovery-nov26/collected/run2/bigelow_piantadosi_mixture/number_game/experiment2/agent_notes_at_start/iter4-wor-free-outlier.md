---
name: iter4-wor-free-outlier
description: Iter-4 cand_3 — WOR incumbent + free outlier rate: MAP outlier 0.124, +1.1 nats in-sample, RMSE 0.010; Chebyshev-in-log(eps) interval tables at 2.4 ms/grad
metadata:
  type: project
---
Iter 4, candidate_3 (2026-10-30): submitted `without_replacement_free_outlier_rate`. It is `concepts_sampled_without_replacement` with a free outlier rate, Beta(2,10). MAP: outlier 0.124, lapse 0.021, gamma -0.10. Observed loglik -901.7 vs the incumbent's -902.9. RMSE of p_yes vs the incumbent is 0.010. Smoke loglik -903.9.

Implementation: rule likelihoods are computed exactly from a (sets, rules, 4) table of member rates. The interval log evidence, and the per-target interval logsumexps, are fitted as degree-20 Chebyshev series in log(eps) over [0.01, 0.45] (error about 1e-12). The "Maximum.make_node" FusionOptimizer ERROR also shows up for the incumbent and is harmless.

**Why:** the outlier rate gains the same ~1 nat under without-replacement sampling as with replacement ([[iter3-free-outlier]]), so the two mechanisms don't overlap.
**How to apply:** don't resubmit this combination. See [[number-game-ceiling]].
