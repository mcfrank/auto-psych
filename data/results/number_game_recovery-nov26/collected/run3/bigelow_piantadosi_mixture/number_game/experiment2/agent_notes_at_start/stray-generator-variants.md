---
name: stray-generator-variants
description: Non-uniform stray distributions on the number-game incumbent (rule-generated, prior-predictive, interval-typicality) plus spike-reliability and round-interval near-dups
metadata:
  type: project
---
iter_4/candidate_4 (2026-10-30), refining uncertain_reliability_noise_concepts (MAP loglik -815.88):
- Spike-and-slab reliability (pi of a perfectly reliable program): -815.87, pool RMSE 0.0045, so a near-dup.
- Round-endpoint interval prior multiplier: rho≈0, RMSE 0.0003, so a dup.
- Strays from the full prior predictive q(x)=sum_h p(h)1[x∈h]/|h|: -817.7, RMSE 0.023.
- Strays from the interval-only predictive: -829.7, bad.
- Strays from rule-only q(x)=sum_r w_r 1[x∈r]/|r|: -816.28, pool RMSE 0.043. Submitted as rule_generated_stray_concepts; check OK, smoke loglik -831.5, ~80 s per 200 draws.
Exact trick: p(X|h)/prod q = sum_j s^-j e_j(1/q over in-h examples) c_j. Intervals contain a contiguous run of the sorted examples (10 runs + none), so counts go by (run, size).
The incumbent's interval prior is 1/100^2, not 1/5050, so lambda_0 is not comparable across parameterisations.
Harness: iter_4/candidate_4/scratch/cmp.py.
**How to apply:** see [[complement-stray-refinement]], [[interval-tempering-speed-trick]].
