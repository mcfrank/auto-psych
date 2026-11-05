---
name: recency-gated-averaging-route
description: exp3 iter0 c5: n^-kappa averaging grafted onto recency_gated_arithmetic_route; smoke -4105.5 vs incumbent -4115.1
metadata:
  type: project
---
Exp3 iter_0 candidate_5 (refinement slot): recency_gated_arithmetic_route + similarity averaging (log support − kappa·log n, kappa~Beta(2,2)).
Smoke log-lik (100/100, 1 chain): -4105.5; incumbent round_number_averaging_exemplar -4115.1; recency_gated -4150.5.
So recency gating and averaging stack (~+10 over incumbent). Relates to [[round-number-averaging-exemplar]], [[recency-gated-arithmetic-route]].
**Why:** recency gating and averaging were each tested only alone on round_number_kinship.
**How to apply:** if admitted, treat as the new base; other incumbent slots may build the same combo (check duplicates).
