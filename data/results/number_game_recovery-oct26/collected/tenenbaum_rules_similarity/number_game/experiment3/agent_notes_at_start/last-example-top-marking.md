---
name: last-example-top-marking
description: exp2 iter_4 c3: top-marking credit only when max example is shown last (order cue); raw data hint at order asymmetry; check outcome
metadata:
  type: project
---
2026-10-19, exp2 iter_4 candidate_3 (`last_example_marks_range_top`): a copy of the incumbent top_marking_range_sampling with one change, top_share = 1 if the last example is the max, else 0, so there is no 1/n factor. Raw multi-example sets: when the max is shown last, yes-rate is 0.61 within 10 below the examples vs 0.40 within 10 above; when the max is first, 0.41 vs 0.39. This is confounded with set identity. The two kinds of set are balanced: ~1400 rows with the max last, ~2600 not. Check OK: smoke LL -2055.0, ~142 s per run.
The iter_4 critique was all null; session SDs were reproduced, which rules out the [[session-heterogeneity]] need.
**Why:** It is an order-based cue that no live model uses, and it should pass novelty.
**How to apply:** Check its LOO. If it wins, order-of-presentation cues are worth more. Related: [[generative-top-marking]]
