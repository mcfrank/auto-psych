---
name: last-example-marks-range-edge
description: exp3 iter_0 c4: last example marks top if max, bottom if min; exp3 descending sets show below<above; smoke LL -3096.5
metadata:
  type: project
---
2026-10-22, exp3 iter_0 candidate_4 (`last_example_marks_range_edge`) is a copy of the incumbent last_example_marks_range_top with one change: when a set ends on its min, ranges whose bottom is the min get the same generative credit (shared top_odds). In exp3 raw data, sets ending on the min that also start on the max have yes 0.39 within 10 below vs 0.51 within 10 above, the mirror of max-last sets. The check passed: smoke LL -3096.5, 184 s per run.
**Why:** It is an order cue the incumbent ignores, and it changes min-last predictions, so it should clear novelty.
**How to apply:** Check its LOO. If it wins, try a separate odds for the bottom edge. Related: [[last-example-top-marking]], [[exp3-example-rejection-critique]]
