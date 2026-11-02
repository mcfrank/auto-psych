---
name: exp3-iter4-set-size-decisiveness
description: Exp3 iter_4 cand_4 — set-size logit gain exp(g*(n-2.5)) on last_example_recency_weight +1.0 nats MAP (g 0.041), MAP RMSE 0.0071 vs the incumbent; submitted set_size_response_decisiveness
metadata:
  type: project
---
2026-11-02. Incumbent MAP LL -3514.11 (last_weight 0.98). Adding a one-parameter logit gain exp(g*(n-2.5)) on the belief gives -3513.12, with g = 0.041 and RMSE 0.0071 on the novelty pool, so it is well above the gate. The smoke check passes (LL -3516.7).
Scripts are in iter_4/candidate_4/scratch/p (nov.py, inc.py, gain.py).
**How to apply:** the gain is similar to the 3-parameter version in [[exp3-iter3-decision-rule-probes]]. One slope parameter captures all of it.
