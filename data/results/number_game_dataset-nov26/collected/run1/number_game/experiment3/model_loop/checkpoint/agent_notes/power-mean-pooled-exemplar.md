---
name: power-mean-pooled-exemplar
description: exp3 iter1 c4: power-mean pooling (r≈0.34) on whole_set_round_category, +3.9 nats MAP log-lik; smoke -4101.6
metadata:
  type: project
---
Refinement of incumbent whole_set_round_category (2026-11-02): replaced sum/n^kappa by power mean M_r (+ (1-kappa) log n, nests incumbent at r=1). Targets near_target_set_size_contrast critique. MAP: log_r=-1.08 (r≈0.34), kappa 0.60, response log-lik -4094.4 vs incumbent -4098.3 (+3.9 nats, 1 extra param). Smoke fit -4101.6, OK.
**Why:** follows [[exp3-iter1-critique-stats]]; base is [[whole-set-round-category]].
**How to apply:** gain is modest; pooling-form ideas likely near saturation (see [[similarity-averaging-exemplar]]).
