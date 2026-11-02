---
name: size-weighted-within-kind-rules
description: exp2 iter_2 cand3 — within-kind rule prior ∝ size^γ on beta_share incumbent: γ≈-0.10±0.11, LOO −0.8, RMSE 0.006
metadata:
  type: project
---
experiment2 iter_2/candidate_3 (2026-10-30), aimed at the nested_multiple_specificity critique (obs 0.49 > null 0.39, q=0.28).
Within each family, rule prior ∝ (member count)^γ (centred log size, Normal(0,1)). 2x400 NUTS: γ=-0.10±0.11, LOO -2274.0 vs incumbent -2273.3; posterior-mean p_yes RMSE vs incumbent 0.006. Submitted as size_weighted_within_kind_rules.
**Why:** the nested-multiple gap is not closed by a prior over rule specificity; size principle already handles it.
**How to apply:** don't retry within-kind prior reweighting on exp2; see [[experiment2-null-refinements]], [[experiment2-iter2-critique-stats]].
