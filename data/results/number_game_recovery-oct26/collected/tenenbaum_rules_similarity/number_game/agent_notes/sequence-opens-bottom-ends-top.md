---
name: sequence-opens-bottom-ends-top
description: exp3 iter_2 c5: first-shown min marks range bottom (own odds) + last-shown max marks top; smoke LL -3093.3; check outcome
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_5 (`sequence_opens_bottom_ends_top`) refines last_example_marks_range_edge. The bottom edge is credited when a set of 2 or more examples opens on its min, with a separate bottom_odds. Plain ranges are split into 4 parts (at_top x at_bottom), and the credits add. It targets the below_asym_min_first critique (q=0.10). Check OK: smoke LL -3093.3, which beats the earlier smoke values of about -3094 to -3096, at roughly 240-260 s per run.
In raw data the yes-rate 1-10 below the min shows no clear min-first effect. The effect exists only relative to the model.
**Why:** It is an order cue no live model uses. **How to apply:** Check its LOO and its novelty result. Related: [[last-example-marks-range-edge]], [[exp3-iter2-critique-stats]]
