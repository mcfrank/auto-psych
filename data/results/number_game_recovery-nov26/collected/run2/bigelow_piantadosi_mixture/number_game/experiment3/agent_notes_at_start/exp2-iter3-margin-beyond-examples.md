---
name: exp2-iter3-margin-beyond-examples
description: Exp2 iter-3 cand_4 — range-prior screens on the digit_sum_open_ended incumbent; all within 1.2 nats; margin-beyond-examples prior submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_4 (2026-10-30), refining digit_sum_open_ended_range_discount. Its MAP NLL is 1923.10.
One-parameter range-prior screens, MAP NLL:
- overhang past the examples, (room below + room above)/100: 1921.95 (beta 0.74; gamma moves to -0.19)
- heading graft: 1921.98
- log overhang: 1922.05
- decade-aligned bounds: 1922.35
- min-side overhang: 1922.46
- lo location: 1922.73
- log relative width: 1922.86
- lo==1 split, round5 bounds, within-decade: all null

Submitted `margin_beyond_examples_range_prior`: Taylor moments over 199 (width, open) groups. The smoke check took 86 s and passed, with LL -1925.8. Its RMSE against the incumbent is 0.0097.
Screener: experiment2/model_loop/iter_3/candidate_4/scratch/screen.py. It takes set-dependent features with shape (sets, intervals).
**How to apply:** the range-prior slot on exp2 is saturated (about 1 nat per idea). See [[exp2-iter2-open-range-graft]], [[exp2-iter1-range-location-prior]].
