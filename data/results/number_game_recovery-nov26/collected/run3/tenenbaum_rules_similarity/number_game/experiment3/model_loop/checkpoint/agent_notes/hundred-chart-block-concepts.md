---
name: hundred-chart-block-concepts
description: exp2 iter_3 c0: rectangular blocks of the tens×units hundred chart as a third concept kind — weight 0.003, p_yes RMSE vs incumbent 0.0057
metadata:
  type: project
---
experiment2 iter_3/candidate_0 (2026-10-30), hundred_chart_block_concepts = beta_share_width_exception_concepts + kind of rectangles on the tens-digit × units-digit grid (≥2 rows, <10 columns; 2475 blocks, aggregated by size in prepare_observed; same beta-share size prior renormalised over block sizes). Quick 2x300: kind weights rules 0.46 / ranges 0.53 / blocks 0.003; incumbent LOO -2273.6; smoke ll -2269.0. Row-level posterior-mean RMSE vs incumbent 0.0057 (biggest diffs on 31,3,1,15). Fit ≈15 s/chain.
**Why:** a third concept-space kind gets near-zero weight on exp2 data, just like parity ranges ([[beta-share-parity-range]], [[concept-space-additions-map]]).
**How to apply:** only new stimuli (e.g. 22,62,26,82-like column clusters) can test it; don't add more kinds on exp2 data.
