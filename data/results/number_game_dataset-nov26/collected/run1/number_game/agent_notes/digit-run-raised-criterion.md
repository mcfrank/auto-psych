---
name: digit-run-raised-criterion
description: iter_4 c4 repair — digit-run criterion shift on shared-factor incumbent (attention version was rejected as a duplicate), -15.4 nats MLE; smoke -1543.4
metadata:
  type: project
---
iter_4 candidate_4_repair_1 (2026-11-02), `digit_run_raised_criterion_exemplar`. The first attempt (digit-run attention boost) was rejected at novelty because it duplicated [[digit-run-rule-attention]] (RMSE 0.001). The repair puts a criterion shift on the incumbent when >=3 examples share a final digit. MLE NLL 1539.1 vs base 1554.5 (shift -1.26). Other screens on the same base: size-principle rule evidence as the criterion, 1545.8; signed rule-membership evidence, 1552.8; the digit run plus that signed evidence, 1539.1. The novelty pool has 60 digit-run stimuli, 10 of them with digit-matching targets. Only those 10 tell a criterion shift apart from an attention boost.
**Why:** the novelty gate compares models on novelty_pool.json, not on the training stimuli. **How to apply:** before building a variant, count its distinguishing stimuli in the pool. See [[digit-run-strict-criterion]], [[iter4-critique-stats]].
