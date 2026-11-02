---
name: kind-specific-reliability
description: exp2 iter_0 cand_0 (2026-11-01): separate reliability for rules vs ranges +1.5 MAP; one-stray cap ruled out by 908 nats; repair speed fixes
metadata:
  type: project
---
On experiment2 data (incumbent family_pooled_uncertain_reliability, MAP loglik -2012.3), I tested two changes:
- Cap of at most one stray per set (j >= n-1): MAP -2920, so it loses by 908 nats.
- Separate Beta mean reliability for intervals (alpha_int) vs rules (alpha), kappa shared: MAP -2010.8 (+1.5), alpha 0.915, alpha_int 0.943. Submitted as kind_specific_reliability_concepts.
The first attempt was rejected as too slow (811 s per 200-iteration smoke fit). The repair (candidate_0_repair_1):
- tau ~ LogNormal(log 1.5, 0.2) in place of HalfNormal, which removes the funnel at 0.
- The interval evidence is a dense pt.dot of prepared (n,k,s)-flattened counts with the flattened likelihood table, not a gather of lik_table[set_n_idx] and lik_table[cell_set].
That cut it to 120 s per smoke fit and 4 ms/grad, with the MAP unchanged.
Gotcha: with a dense dot, the masked invalid table entries get a gradient. exp(logsumexp(-1e4 ...)) then gives NaN gradients, so use a mask value of -50 and exp(...).sum().
**How to apply:** the funnel and gather fixes apply to every family-pooled reliability model. See [[family-pooled-reliability-graft]] and [[check-sibling-candidates-first]].
