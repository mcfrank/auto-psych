---
name: coincidence-gated-pattern-belief
description: Mixture "program is random (yes at ~0.78)" gated by coincidence strength (n-1)*log(100/|tightest shared category|) on gap_step incumbent: +19.8 loglik; noisy-OR combination -59
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_2/candidate_1, coincidence_gated_pattern_belief). Incumbent gap_step_progression_overlap scores -3145.7 (posterior-mean-p loglik, 2x300 NUTS).
- p_yes = pc*chance_yes + (1-pc)*sigmoid(overlap), with pc = sigmoid(c0 - k*coincidence): -3125.9 (k about 1.1, chance_yes about 0.78). Submitted. Smoke -3131.8, about 13 s.
- The same mixture gated only on n==1: -3127.9, so most of the gain comes from single-example sets.
- Noisy-OR (unnormalized evidence, with no penalty for missed properties) instead of the proportional share: -3204.7. The normalization by profile size is essential.
- Incumbent n=1 residuals are mostly a level shift (+0.04). Target-intrinsic properties (round, prime, and so on) show no residual on multi-example sets.
**Why:** an exploratory slot targeting the single_example_yes_gap critique with a different process family (randomness or coincidence detection).
**How to apply:** this may be flagged as a near-duplicate of [[single-example-leniency]]. See [[single-example-breadth]].
