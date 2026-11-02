---
name: iter3-free-outlier
description: Round 3 cand_3 — incumbent + free outlier rate gives MAP outlier≈0.12, +1.1 nats in-sample, RMSE 0.012 vs fixed 0.1; harness pitfall with p_yes compile
metadata:
  type: project
---
Round 3, candidate_3 (2026-10-30): submitted `breadth_prior_concepts_free_outlier_rate`, which is `rule_breadth_prior_concepts` with a free outlier (stray-example) rate, Beta(2,10). MAP: outlier 0.119, lapse 0.019, gamma -0.17. In-sample loglik is -902.5 against -903.6 with the rate fixed at 0.1. The p_yes RMSE between those two settings is 0.012, which clears the novelty gate. Intervals are grouped by (size, k inside), and per-row hits go through a CSR structured_dot. That runs at 6 ms per gradient and 11 s per 200 iterations, against 26 ms with dense gathers.

**Why:** a tiny gain, which confirms [[number-game-ceiling]].
**How to apply:** when evaluating p_yes at a point, compile `model.replace_rvs_by_values([p_yes])` and `.copy()` the output. Otherwise the RVs get sampled at random and the numbers look nondeterministic.
