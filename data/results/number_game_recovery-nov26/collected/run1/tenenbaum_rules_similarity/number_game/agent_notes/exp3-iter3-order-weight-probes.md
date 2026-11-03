---
name: exp3-iter3-order-weight-probes
description: Exp3 iter_3 cand_0 — last-example weight 0.98, first-example weight 1.01 (multi-set only or incl. singletons) all pin at 1 on the no-centre equal-prior base; submitted last_example_recency_weight
metadata:
  type: project
---
Base: equal_prior_familiar_rules_digit_sums with centre weighting dropped (MAP LL -3514.87 with centre, 0.014). Every concept's example log-lik is precomputed per set, split into the weighted example and the rest. The fit is fast.
- Last example weight w (multi-example sets only): w 0.982, LL -3514.11, pool RMSE vs incumbent 0.0028 (partly from dropping centre). Submitted as last_example_recency_weight with a LogNormal(0, 0.5) prior.
- First example weight, multi-example sets only: 1.011, LL -3514.35. Including singletons: 1.013, LL -3514.25.
Probe: iter_3/candidate_0/scratch/p/*.py plus iter_2/candidate_5_repair_1/scratch/nov.py.
**How to apply:** order or position weights are dead on exp3 too (see [[exp2-iter4-order-probes]], [[exp3-iter2-sequential-kind-probes]]). Don't re-probe them. Novelty risk: MAP RMSE below 0.0035.
