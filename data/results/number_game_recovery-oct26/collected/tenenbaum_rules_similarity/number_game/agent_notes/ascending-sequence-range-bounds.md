---
name: ascending-sequence-range-bounds
description: exp3 iter_2 c0: incumbent + first-shown min marks range bottom (own bottom_odds); 4-way interval split; raw min-first below yes 0.60 vs 0.51
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_0 `ascending_sequence_range_bounds` is a copy of compressed_rule_extension with one change: for sets of 2 or more that open on their min, ranges whose bottom is the min get log1p(bottom_odds/draw_prob). The intervals are split 4 ways (bottom x top). It targets the critique below_asym_min_first, where the model over-predicts the min-first downward boost (obs 0.09 vs 0.17). In raw data, min-first sets get yes 0.60 within 10 below vs 0.51 for other sets.
Check OK, smoke LL -3092.7 (incumbent -3094.5), but the whole check took 10.4 min vs 225 s per run for the incumbent, so the 30-min speed limit is at risk.
**How to apply:** check LOO and the bottom_odds posterior. Related: [[last-example-marks-range-edge]], [[exp3-iter2-critique-stats]]
