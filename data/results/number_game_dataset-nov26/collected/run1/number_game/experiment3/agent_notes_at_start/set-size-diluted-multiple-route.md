---
name: set-size-diluted-multiple-route
description: exp2 iter1 c5 repair — multiples route weight × exp(-dilution·(n-1)) on multiple_route_exemplar, MLE -7 nats; smoke -2688.7
metadata:
  type: project
---
Experiment 2, iter_1 candidate_5_repair_1 (2026-11-02): `set_size_diluted_multiple_route` refines [[multiple-route-exemplar]]. The first attempt (the kinship model) was rejected because it was an exact duplicate of [[divisor-family-exemplar]] (RMSE 0). The change: the additive multiples-route weight is multiplied by exp(-dilution·(n-1)), which adds one HalfNormal parameter. In-sample MLE NLL was 2683.3, against 2690.4 for the base model (fitted dilution ≈ 0.98). The smoke fit was -2688.7.
Variants screened on multiple_route that gave no gain:
- kinship penalty
- route via set gcd: 2703
- digit route: 2711
- multiple-or-divisor: 2708
- size-principle weight 1/#multiples^tau: 0
Smaller gains: 1/n^tau gave 2684.6 and "confirmed by another example" gave 2685.7.
No trial_index drift was found in the background yes-rate.
**Why:** background_set_rate_sd residuals: sets with a small example ("50,2", "55,3", "3,33,99") are over-predicted, and lone examples ("5", "7", "100") are under-predicted.
**How to apply:** the screen scripts are in iter_1/candidate_5_repair_1/scratch/.
