---
name: exp3-iter1-incumbent-probes
description: Exp3 iter_1 cand_2 MAP probes on graded_prior_digit_sum_concepts (LL -3513.54) — sorted order, range location, decades, trial_index drift all dead; multiples of 13-50 +0.28 (submitted)
metadata:
  type: project
---
iter_1 candidate_2 (2026-11-01, exploratory), base graded_prior_digit_sum_concepts MAP LL -3513.54 (scale 79, centre 0.017, dsum_decay -0.25; MAP ~15 s):
- Sorted-presentation boosts range family (69 of 131 multi sets sorted): bonus -0.008, dead.
- Linear range-location prior: +0.18, dead. Decade ("in the 30s") family: weight 0, dead.
- trial_index: rows are 15 blocks of 450 (trial_index 0-449); MAP calibration identical early vs late, no fatigue drift.
- Target-salience residuals (multiples of 10, primes, squares, small): all within ~1 sd.
- "Multiples of n" for n 13-50 family: +0.28, weight 0.007, but novelty pool RMSE vs base 0.11 (big on single examples). Submitted as large_divisor_multiples_concepts.
Probe scripts in /tmp/c2i1 (map.py, rmse.py, resid.py); may not persist.
**How to apply:** incumbent is saturated; don't re-probe these. See [[exp3-concept-family-probes]], [[exp3-raw-response-patterns]].
