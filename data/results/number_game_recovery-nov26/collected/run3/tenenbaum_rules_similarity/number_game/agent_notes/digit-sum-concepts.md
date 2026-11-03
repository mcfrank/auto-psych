---
name: digit-sum-concepts
description: exp3 iter_4 c0: "digits add up to k" rule family on rule_family_doubt_criterion; pool RMSE 0.0093 vs incumbent, ll ≈ incumbent
metadata:
  type: project
---
exp3 iter_4/candidate_0 (2026-10-31), digit_sum_concepts: the incumbent rule_family_doubt_criterion plus 18 rules "digit sum = k" (k=1..18) as a 9th rule family (Dirichlet(2) weight); it also enters the family-doubt entropy. Smoke check OK (21 s, ll −3653.9). In 2×300 fits: posterior-mean ll −3648.4 vs incumbent −3648.1; shift 0.20. Novelty-pool RMSE vs incumbent 0.0093 (max abs diff 0.11).
Data barely test it: the digit-sum-consistent sets are 19,73 (ds 10, 75 trials), 92,56 (ds 11), and the multiples-of-9 sets (ds 9).
**How to apply:** this confirms [[digit-parity-family-doubt]]: a new family passes novelty but gives no likelihood gain. The arithmetic-progression sets in exp3 are 53,47,41, 63,23,43 and 18,45,72.
