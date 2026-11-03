---
name: exp2-iter3-consideration-set
description: Exp2 iter-3 cand_1 — rule-vs-range kind commitment pinned at κ=1 (dup); limited consideration set 1-(1-π)^N ties, RMSE 0.003 at N=300, submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_1 (2026-10-30). Base: digit_sum_open_ended_range_discount, MAP NLL 1924.02.
- Kind commitment: sigmoid(κ·rule-vs-range log odds), with or without the same κ on sums-vs-rules. κ goes to 1.006, so it is a near-duplicate and was dropped.
- Limited consideration set: each concept's effective weight is 1-(1-π)^N. The likelihood is flat for N from 1 to 1000 (-1924.01), costs 0.15 nats at N=1000 and 8.5 at N=1e4. Novelty RMSE against N=1 is 0.0008 at N=100, 0.0034 at N=300 and 0.0068 at N=1000. Submitted `limited_concept_consideration_set` with N ~ LogNormal(log 300, 1). Smoke loglik -1926.6, 29 s per 200 iterations.
- Scripts: /tmp/prof.py profiles the likelihood with pm.do, /tmp/nov.py computes novelty RMSE on the pool (not persistent).

**How to apply:** process slots on exp2 are pinned to normative Bayes. For novelty, choose a parameter whose likelihood is flat but whose predictions move ([[number-game-ceiling]]).
