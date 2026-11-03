---
name: exp3-iter4-set-size-trust
description: Exp3 iter_4 cand_3 — set-size-dependent example trust (alpha logit-linear in n) on last_example_recency_weight: slope -0.025 (dead), alpha 0.908, +0.9 MAP, RMSE 0.0056; submitted set_size_dependent_example_trust
metadata:
  type: project
---
2026-11-02, exp3 iter_4 candidate_3 (incumbent refinement slot). Incumbent MAP LL -3514.11.
- Example trust alpha = sigmoid(a0 + a1*(n-2.5)), computed in-model from per-set rule membership counts: LL -3513.19, a0 2.295 (alpha 0.908), a1 -0.025. MAP pool RMSE vs incumbent 0.0056. The gain is from alpha, not from the slope.
- Fast: nov.py MAP + pool run takes about 60 s on 1 CPU.
- The iter_4 critique on the incumbent had 0/8 significant statistics, so the incumbent is saturated.
**How to apply:** set-size dependence of trust is dead. See [[exp3-iter2-sequential-kind-probes]] (reliability pins at 0.9) and [[exp3-iter3-decision-rule-probes]] (set-size gain is tiny).
