---
name: seed-plus-digit-overshadowing
description: Seed-multiple model plus final-digit overshadowing gave MAP loglik -3174, vs -3197 for the incumbent and -3262 for the seed model (experiment2 iter_0)
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_0/candidate_5, seed_multiple_digit_overshadowing). Full-data MAP log-likelihoods:
- seed model alone: -3262
- final_digit_overshadowing_overlap (incumbent): -3197
- seed model with digit overshadowing grafted on: -3174 (retain about 0.002). Submitted.
- seed model where the seed relation itself overshadows the other properties: -3252. Weak.
The LOO gaps are much smaller than the MAP gaps (incumbent vs seed: 8 nats LOO, 65 at MAP). The digit-overshadowing gain likely rests on a few sets.
**How to apply:** see [[final-digit-overshadowing]] and [[seed-multiple-property]]. The two mechanisms stack.
