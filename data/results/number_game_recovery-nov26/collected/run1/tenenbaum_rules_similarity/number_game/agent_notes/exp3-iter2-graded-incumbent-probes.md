---
name: exp3-iter2-graded-incumbent-probes
description: Exp3 iter_2 cand_4 MAP probes on a fast graded_prior_digit_sum_concepts (centre weight dropped) — the graded dsum gain is the single example 89; ends-in graded +0.65 (submitted); round final digits, edge-touching ranges, size-graded priors dead
metadata:
  type: project
---
2026-11-01, exp3 iter_2 candidate_4 (incumbent refinement slot). Harness in /tmp/c4i2 (base.py: numpy loglik on 1227 cells with width-grouped range sums; fit2.py, probe*.py, cmp.py; may not persist).
- Without centre weighting: flat dsum -3514.63, sum-graded -3513.15 (better than the incumbent's -3513.54 with kappa), so centre weighting does nothing on exp3.
- Of the graded-dsum gain, +1.56 nats comes from the single set (89,): it lowers 79/69/59 (observed 6/15 yes) through the {89,98} sum-17 concept. The mechanism is essentially a fit to one set.
- On the sum-graded base: ends-in prior exp(-h*d) +0.65 (h -0.026, submitted as graded_final_digit_digit_sum_concepts); 0/5 final-digit bonus +0.01; ranges touching 1/100 bonus +0.02; rule prior ∝ size^γ across families +0.24; digit-sum prior graded by concept size rather than sum is 0.6 worse than by sum.
- Worst-fitting sets are spread-out random 4-sets (e.g. 43,4,91,9 and 17,50,5,82 are under-predicted; 70,88,7,22 is over-predicted), with mixed signs.
**How to apply:** exp3 is at the noise ceiling; prefer speed and identifiability over new cues. See [[exp3-iter1-saturation-probes]], [[exp3-iter1-speed-repair]].
