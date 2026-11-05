---
name: round-example-random-doubt
description: exp3: n=1 multiple-of-5 examples get yes 0.41 vs 0.60; scaling doubt for round lone examples on the doubt incumbent gives smoke -4679.2 (~+3)
metadata:
  type: project
---
exp3 iter_1 c2 (2026-11-02): one-example sets whose example is a multiple of 5 are endorsed 0.41 vs 0.60 for the others, even for targets that are themselves multiples of 5. The n1 "parity gap" (+0.19) is almost entirely this confound: 135 of the 144 opposite-parity n=1 rows come from m5 examples.
Model `round_example_reduces_random_doubt`: on single_example_random_program_doubt, the doubt share is scaled by a Beta ratio for round lone examples. Smoke -4679.2 vs -4682.4, so only a small gain.

**Why:** the n1_set_mean_variance critique is mostly binomial noise plus this round-number effect.
**How to apply:** don't treat the n=1 parity gap as a parity effect. Related: [[exp3-n1-data-facts]], [[single-example-random-doubt]]
