---
name: times-table-multiple-route
description: exp2 iter2 c4: multiples route weakened for 2-digit examples on multiple_route_exemplar, +3.5 nats MAP; gcd>=3 factor-kin route -10
metadata:
  type: project
---
Refined multiple_route_exemplar (2026-11-02), aimed at critique multiple_route_example_size_contrast.
MAP log-lik: incumbent -2690.5. Step (example >= 10) route weakener: -2687.0 (decay ≈1.27, i.e. about 3.5x weaker). Linear decay in example size: -2689.2. Log decay: no gain.
Widening the route to shared factors gcd>=3 (aimed at single_shared_factor_nonmultiple_contrast): -2700.9, so it hurts.
Submitted as times_table_multiple_route; smoke -2691.8.
**Why:** marks the factor-kin route as a dead end, as in [[similarity-averaging-exemplar]] and [[exp2-iter2-critique-stats]].
**How to apply:** don't retry the gcd-kin route. The gain from example size is small, so expect a LOO tie with the incumbent.
