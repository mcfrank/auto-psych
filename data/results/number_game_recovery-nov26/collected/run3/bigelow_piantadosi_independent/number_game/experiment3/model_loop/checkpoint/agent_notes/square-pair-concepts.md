---
name: square-pair-concepts
description: exp3 iter4 cand0 — {n, n²} pair concepts +19.8 MAP loglik on forward_continuation incumbent, almost all from set 8,64; doubling chains null
metadata:
  type: project
---
2026-10-31, exp3 iter_4/candidate_0 (`square_pair_concepts`). Base forward_continuation_response_bias has MAP loglik -3411.6. Adding {b, b*b} for b=2..10 as cost-2 rules with one fitted bonus (about -1.2) gives -3391.8 (+19.8). Per set: 8,64 gives +18.3, 81,9 gives -2.4 and 14,98,91 gives -1.9. People say no to nearly everything after 8,64 (0-3/5 yes), while the base predicted 0.6-0.7. Adding {b, b³} pairs as well is worse (-3394.7). Doubling chains {s·2^k} are null (bonus goes to -6). Novelty-pool RMSE vs the base is 0.023.
Tools: scratch/pool.py computes pool p_yes at the MAP. Residual-by-pair z ranking found the 8,64 misfit quickly.
**Why:** the gain rests on a single set, so it is fragile in LOO and the novelty check.
**How to apply:** look for "relation between examples" concepts (n and f(n)). Don't re-screen doubling chains. Related: [[concept-space-saturated-exp3]], [[exp3-iter4-critique-stats]].
