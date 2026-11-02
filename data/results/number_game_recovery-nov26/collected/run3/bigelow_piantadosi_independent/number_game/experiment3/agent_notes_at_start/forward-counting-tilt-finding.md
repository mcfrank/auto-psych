---
name: forward-counting-tilt-finding
description: upward-tilted interval sampling density on the incumbent: tilt ~0.46 at MAP, +2.8 log-post, 1.65x slower grad
metadata:
  type: project
---
Tried 2026-10-30 (iter_4 candidate_2 repair, `forward_counting_range_sampling`). Interval likelihood p(x|[a,b]) = (1+g*u)/|[a,b]|, u = 1-2(x-a)/(b-a), g = tanh(z), on top of the multiplicative incumbent.
- MAP g = 0.46 (examples read as low members of a range, so people generalise upward). Log posterior -845.0 vs -847.8 for the incumbent. p_yes RMSE vs incumbent 0.014, so it is novel.
- The gradient takes 46 ms vs 28 ms for the incumbent. It uses a sparse inc_subtensor over (set, interval) slots; the dense S x 4 x 5050 version took 80 ms. Softmax over S x 5100 dominates.
- At MAP the lapse goes to 0 in both models.

**Why:** the critique's upward_minus_downward_near (z=2.1) didn't survive FDR, and the gain here is small.
**How to apply:** treat directional range asymmetry as a weak signal. Related: [[contains-digit-cost-on-incumbent]].
