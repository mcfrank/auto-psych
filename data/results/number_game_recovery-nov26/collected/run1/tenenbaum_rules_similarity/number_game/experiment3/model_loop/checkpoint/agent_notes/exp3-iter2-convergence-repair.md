---
name: exp3-iter2-convergence-repair
description: Dirichlet family weights pinned at 0 + dsum_decay sigma 1 caused R-hat 1.057; softmax Normal logits + dsum_decay sigma 0.5 fixes it (exp3 iter_2)
metadata:
  node_type: memory
  type: project
  originSessionId: d5f0974f-67d5-46cc-a478-8b612558e818
  modified: 2026-11-02T05:36:03.762Z
---

Exp3 iter_2 candidate_4 (graded_final_digit_digit_sum_concepts, uniform ranges, width-summed precompute) failed admission: R-hat 1.057, ESS 92.
Cause: contains-digit and exception family weights pinned at ~0 on the Dirichlet simplex, and dsum_decay (sd ~0.5) trading off against the digit-sum family weight.
Fix: family_logit ~ Normal(-2, 2.5) over 7 families with ranges fixed at logit 0, then softmax; dsum_decay ~ Normal(0, 0.5). 4 chains x 1000: R-hat 1.00, min ESS ~1170, 0 divergences, about 60 s.
Final-digit decay posterior was -0.023 ± 0.023, so it is close to the incumbent and may run into the novelty gate.

**Why:** the same Dirichlet geometry sits in every family-prior model in this loop.
**How to apply:** use softmax over Normal logits from the start in new family-prior candidates. See [[exp3-iter1-speed-repair]].
