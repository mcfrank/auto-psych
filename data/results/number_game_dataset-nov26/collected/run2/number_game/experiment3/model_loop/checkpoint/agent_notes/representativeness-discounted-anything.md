---
name: representativeness-discounted-anything
description: exp3 iter_1 c5 — round-example (mult of 5) discount on the anything concept: smoke -4687.7 vs base -4688.4, ~0 gain
metadata:
  type: project
---
Tried 2026-11-02 (exp3 iter_1/candidate_5, representativeness_discounted_anything). I refined [[size-discounted-anything-concept]] so that the anything weight is multiplied by phi^(number of examples that are multiples of 5). Smoke log-lik was -4687.7 against the base's -4688.4, so the gain is about 0.
The data fact behind it: n=1 sets with a multiple-of-5 example are endorsed at 0.40, against 0.60 for the other n=1 sets (225 vs 1755 rows). The opposite-parity targets in n=1 sets come almost entirely (135 of 144 rows) from those round examples, so the n1 parity gap is confounded with roundness.
**Why:** the share's base-ten family probably already absorbs the round-example effect.
**How to apply:** check its ELPD. If it is ~0, the n1_set_mean_variance gap needs a mechanism that varies from set to set and is not tied to roundness. Related: [[exp3-n1-data-facts]]
