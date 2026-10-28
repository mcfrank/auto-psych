---
name: digit-sum-concepts
description: exp3 iter_0 c1: digit-sum rules (k=1..18) as a 6th rule kind on last_example_marks_range_top; smoke LL -3095.3, ~200s/run
metadata:
  type: project
---
On 2026-10-22, exp3 iter_0 candidate_1 (`digit_sum_concepts`) copied the incumbent and added 18 digit-sum rules as their own kind (N_KINDS=6). The check passed: smoke LL -3095.3 (incumbent ELPD -3099.1), about 199 s per sampling run. Novelty risk: digit-sum 9 is nearly the same set as multiples of 9, so it may fail the RMSE<0.002 gate (see [[novelty-gate-tiny-tweaks]]).
Exp3 sets that touch digit sums: 36,45,54,81; 18,45,72; 81,9; 15,51.
**Why:** In exp2 additive concept families gained under 1 nat ([[concept-space-saturated]]).
**How to apply:** Check whether it was admitted before trying another additive concept family.
