---
name: common-divisor-membership-route
description: exp2 iter1 c3 repair — additive route for multiples of set gcd>=3 (singles incl.) on incumbent, MAP +22 nats; per-set random breadth too slow
metadata:
  type: project
---
Exp2 iter_1 candidate_3_repair_1 (2026-11-02), `common_divisor_membership_route`. The first attempt was rejected as an exact duplicate of divisor_family_exemplar. The repair adds summed_sim += exp(log_w)*[t % g == 0, g = set gcd >= 3 incl. single examples, t not in set] to the incumbent. MAP log-lik -2692.4 vs incumbent -2714.4. multiple_route reached -2690.5, and the two differ by RMSE 0.036 on the data stimuli. Smoke fit -2696.6, 9 s.
Rejected along the way: a per-set non-centred random breadth (gain) scored MAP -2533, but NUTS hit tree depth 8 (82 s per 200 iterations with 211 slots; 4093 slots took >10 min). Production uses 7000 iterations × 4 chains at target_accept 0.99, so it can't fit in the 30-minute limit.
**Why:** per-set random effects are infeasible under admission's time limit; the biggest background residuals are sets whose members are multiples of the set divisor ("5", "7", "3,63", "64,4", "84,56").
**How to apply:** don't propose per-set random effects; see [[multiple-route-exemplar]] and [[exp2-iter1-critique-stats]].
