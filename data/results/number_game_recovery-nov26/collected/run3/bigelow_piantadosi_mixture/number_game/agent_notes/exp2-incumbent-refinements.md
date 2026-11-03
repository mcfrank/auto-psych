---
name: exp2-incumbent-refinements
description: experiment2 iter_0 refinements of family_pooled_uncertain_reliability (MAP ll -2012.27) — what collapsed, what gained
metadata:
  type: project
---
I tested these in experiment2 iter_0/candidate_4 (2026-11-01), using scratch/cmp.py (prints MAP ll and novelty-pool RMSE against base.py):
- Response lapse (lapse plus a yes-guess rate): the lapse went to 0, a duplicate.
- Stray-aware readout (a non-member is endorsed in proportion to E[1-a|X,h]): omega went to 0.
- Rule-size term in the prior: gamma ≈ 0, a duplicate. A separate tau per family: -0.6 ll, RMSE 0.004.
- Student-t rule deviations: a near-duplicate. Laplace deviations: -1.3 ll.
- Nesting the normalisation within families: pool RMSE 0.0017, a near-duplicate.
- Without-replacement sampling: -1.5 ll, RMSE 0.008.
- Pure rule-generated strays: -2.6 ll, RMSE 0.036.
- Strays as a mix of uniform and rule-generated draws, q=(1-mix)/100 + mix·Σ w_r 1[x∈r]/|r|: **+1.3 ll**, mix≈0.35, tau falls to 1.56, pool RMSE 0.017. Submitted as mixed_rule_generated_strays.
Sibling candidate_3 in the same iter did the digit-slip stray graft.
**Why:** the residuals look like binomial noise, and the only change that gained anything was the shape of the stray distribution.
**How to apply:** don't retry the duplicates listed above. See [[stray-generator-variants]], [[check-sibling-candidates-first]].
Speed: the rule-generated-stray graph took 20-30 ms/grad. Almost all of it was the scatter-gradient of gathering the (cells, runs, 100 sizes) table. Since size enters only as size^-j, prepare_observed now pre-sums count·size^-j into (cells, runs, 5), and rules gather from a 16-subset e_j table. That brought it to 8.6 ms/grad, against the incumbent's 16 ms on the same loaded CPU.
