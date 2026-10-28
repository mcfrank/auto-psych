---
name: iter4-incumbent-refinement-screens
description: iter_4 MAP screens on uneven_rule_familiarity_ends — singleton rule-share/evidence nulls; breadth-criterion graft +1.9 nats
metadata:
  type: project
---
Experiment3 iter_4 candidate_4 (2026-10-21). Incumbent uneven_rule_familiarity_ends MAP loglik -3186.35.
- Singleton-only p_mix logit shift (targets the singleton_rule_far_generalization critique, q=0.30): +0.3 nats, shift -0.19. Null. (Sibling candidate_3 screened the same thing.)
- Singleton-only rule evidence multiplier: +0.24 nats, multiplier 1.05. Null.
- Residuals by (set size, far, shared-rule kind) all look like noise (|z| < 2.5).
- Grafting adaptive_breadth_criterion (criterion = bias + slope*log expected share of 1-100): +1.86 nats, p_yes RMSE 0.0099. The raw log share has mean -1.25 and sd 0.36, so centre at 0.29 and scale by 0.36. Submitted as uneven_familiarity_breadth_criterion, with softplus size_inflation.
Bench (target_accept 0.9, 200/250 draws, 1 chain): 15 steps per iteration, 0 divergences, min bulk ESS 154 (size_exponent), 128 s; check_candidate OK. The breadth slope is 0.08 ± 0.05.
**Why:** the singleton critique cannot be closed via rule share or evidence; set-level criterion calibration still gains on the new base.
**How to apply:** scripts are in iter_4/candidate_4/scratch (mapfit.py, resid.py, bench.py). See [[adaptive-breadth-criterion]], [[size-inflation-softplus]].
