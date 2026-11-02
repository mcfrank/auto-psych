---
name: tau-funnel-speed-fix
description: family-pooled models time out at admission (4 chains x 7000 iters, target 0.99); fixing tau=1.6 cuts NUTS from 521 to 31 steps/iter
metadata:
  type: project
---
Tested in experiment2 iter_0 candidate_4_repair_1 (2026-11-01). Under warm-start NUTS at target_accept 0.99, the family-pooled prior with an inferred tau (HalfNormal) gave tree depth about 9 (521 steps/iter, about 3 h for a production chain). The incumbent only passes because carried models are never time-limited. Fixing tau at 1.6, its posterior mode, gave depth 5 (31 steps/iter, 0.15 s/iter at 3.3 ms/grad), with no MAP loss.
Also: a stray generator built from equally weighted rules, q=(1-mix)/100+mix·mean_r 1[x∈r]/|r|, gave MAP ll -2010.3. That beats the version weighted by the learner's rule prior (-2011.0) and decouples q from rule_dev.
**Why:** admission gives each sampling run 30 min, and production is 7000 iters per chain.
**How to apply:** time with the scratch/ws.py pattern, which calls src.models.pymc_inference._warm_start_sample on 1 chain and 100 iters. Aim for ≤ about 40 steps/iter. See [[family-pooled-reliability-graft]], [[exp2-incumbent-refinements]].
