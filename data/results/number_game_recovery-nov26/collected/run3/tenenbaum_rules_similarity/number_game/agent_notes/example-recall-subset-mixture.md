---
name: example-recall-subset-mixture
description: Round-4 memory-limit model (independent recall of each example, mixture over recalled subsets) on parity_within_range base; linear-space matvec trick makes it fast
metadata:
  type: project
---
Round 4 (2026-10-30), example_recall_limited_concepts: each example recalled w.p. r (Beta(4,1.5)), p_yes = mixture over non-empty recalled subsets of the parity_within_range_concepts learner.
First attempt (iter_4/candidate_2) REJECTED as too slow: 84 ms logp+grad; production is 4 chains × (3000 tune + 4000 draws) at target_accept 0.99 within 30 min.
Repair (candidate_2_repair_1): compute in linear space — per state exp(ll - max) likelihoods summed per interval size, and per (cell, subset slot) the same sums restricted to target-containing concepts; then evidence and "contains" are just matvecs with the prior vector. 2 ms logp+grad (logp identical to old). Also declared SAMPLER_SETTINGS draws/tune 1000, target_accept 0.95. Full 4-chain fit 92 s, 0 divergences, recall ≈ 0.987, novelty-pool RMSE vs recall→1 ≈ 0.005.
**Why:** the gradient of a large row gather (AdvancedIncSubtensor1) dominated the cost; the matvec form removes it.
**How to apply:** use the same linear-space matvec layout for any concept-mixture model with per-set states; see [[incumbent-residual-diagnostics]].
