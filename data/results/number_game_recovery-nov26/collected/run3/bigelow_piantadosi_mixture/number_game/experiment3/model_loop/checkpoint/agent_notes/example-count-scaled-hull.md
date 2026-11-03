---
name: example-count-scaled-hull
description: exp3 iter_4 c4: min–max hull boost ×(n−1) on variable_min_max incumbent, +0.18 MAP with 0 params, pool RMSE .0036; hull-bounded rules, tightness-on-boost: near-dups
metadata:
  type: project
---
exp3 iter_4/candidate_4 (2026-11-02). Incumbent variable_min_max_bounded_range, MAP ll −3317.89. Scratch in iter_4/candidate_4/scratch (mapeval.py with absolute paths, resid*.py residual scans).
These MAP screens were run:
- Rule members between min and max as a boosted range-limited rule (shared or separate boost): −3317.96 / −3317.92, pool RMSE ≤ .0003. A dup.
- Tightness on the hull boost: slope .17, a tie, RMSE .0003. A dup.
- log boost + slope·(n−3): slope .48, −3317.62, RMSE .0032.
- boost·(n−1) with no parameter: −3317.71, RMSE .0036. **Submitted** as example_count_scaled_hull.
- Residuals by n × target position are calibrated within 1.4 z. There is no final-digit binding signal (tens=d .16 obs vs .16 pred).
**How to apply:** hull-prior levers are nearly exhausted. See [[variable-min-max-bounded-range]], [[min-max-bounded-range]].
