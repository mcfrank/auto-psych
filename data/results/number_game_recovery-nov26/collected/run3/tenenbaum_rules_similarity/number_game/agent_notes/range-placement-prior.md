---
name: range-placement-prior
description: exp2 iter_3 c3: beta prior over a range's centre position on beta-share incumbent; speed notes (per-interval 200ms/grad, size×20 position bins 42ms)
metadata:
  type: project
---
exp2 iter_3/candidate_3 (2026-10-30): range_placement_prior_concepts = incumbent + Beta(mean_position, position_concentration) over range centre, normalised within each size.
Incumbent quick 2x300 LOO -2273.0 (lambda_0 .46, mean_share .48, conc 5.4).
Speed: exact per-interval (5095 concepts) = 200 ms/grad (too slow, check timed out at 10 min); grouping intervals by (size, centre twentieth), 1090 groups = 42 ms/grad; incumbent 3 ms.
**Why:** avoid redoing location-prior and per-interval layouts.
**How to apply:** see result line below once fit ends; related [[round4-null-refinements]] (one-sided ranges weight 0).
Result: grouped version quick LOO -2273.4 vs incumbent -2273.0 (null); mean_position 0.44±0.05, position_concentration 3.3±1.4 (slight mid/low preference). p_yes RMSE vs incumbent 0.010 (max 0.07), so it should pass novelty. 2x300 fit 495 s; check_candidate OK.
