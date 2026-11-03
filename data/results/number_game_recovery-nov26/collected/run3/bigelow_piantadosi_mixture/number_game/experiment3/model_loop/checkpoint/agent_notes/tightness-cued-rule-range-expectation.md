---
name: tightness-cued-rule-range-expectation
description: exp3 iter_2 c3: cluster-tightness shift on the variable-lambda incumbent's mean: +1.63 MAP, pool RMSE .0094 vs incumbent, .0065 vs cluster_cued
metadata:
  type: project
---
exp3 iter_2/candidate_3 (2026-11-02) is tightness_cued_rule_range_expectation. It takes variable_rule_range_expectation and sets the per-set logit-lambda mean to mu - beta*tight, with tight = 1-span/99 (0 when n=1). lam_k becomes (sets, K).
- MAP ll: -3317.61, against -3319.24 for the incumbent and -3318.97 for cluster_cued.
- MAP values: beta .48 (the same as in cluster_cued), lambda_sd .71.
- Novelty-pool RMSE: .0094 vs the incumbent, .0065 vs cluster_cued.
- Speed: smoke sampling took 112 s, against 71-83 s for the incumbent at the same load (about 1.4x slower). Tightness is centred at a fixed 0.34 to decorrelate beta from lambda_mu.
- The two effects stack additively and do not substitute for each other.
- The MAP+pool script is in iter_2/candidate_3/scratch/mapeval.py; it takes model paths, and the first one is the reference.
Sibling candidate_4 also refines the incumbent and may take the same graft, see [[check-sibling-candidates-first]].
**How to apply:** see [[variable-rule-range-expectation]] and [[cluster-cued-range-salience]].
