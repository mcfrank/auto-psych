---
name: size-discounted-anything-concept
description: exp3 iter_0 c1 — "any number" concept in the share, weight rho^(n-1), on span_interval_similarity; smoke loglik -4688.4
metadata:
  type: project
---
Tried 2026-11-02 (experiment3 iter_0/candidate_1, size_discounted_anything_concept). I added one always-hit concept to span_interval_similarity's share with salience s*rho^(n-1). It targets the single_vs_multi_yes_gap critique (exp3 yes rates for non-examples: n=1 0.58, n=2 0.41, n=3 0.40, n=4 0.37) and the flat single-example distance gradient. Smoke fit (1 chain, 100+100 draws) took 24 s, with log-lik -4688.4. I have no incumbent smoke number on exp3 to compare it with.
**Why:** this is a graded set-size version of n=1 leniency ([[single-example-leniency]], [[band-plus-leniency]]) that acts inside the share rather than as a logit shift.
**How to apply:** check its ELPD rank in the next round before trying more set-size leniency variants.
