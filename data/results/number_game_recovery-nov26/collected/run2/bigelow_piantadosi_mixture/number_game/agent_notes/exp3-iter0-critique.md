---
name: exp3-iter0-critique
description: Exp3 iter-0 critique of heading_direction_range_extrapolation — data layout and 8 raw-slice stats submitted
metadata:
  type: project
---
Exp3 iter 0 (2026-10-31): the data has 6750 rows and 218 sets, mostly scattered 4-example sets. Each trial_index has 15 rows (about 15 people × 450 trials), and there is no participant_id. Observed patterns: the yes rate falls from 0.51 to 0.35 over the session; 70,10,88,40 gets only 0.13 yes despite being all even; final-digit rules with one exception get 0.89 yes; shown examples get 0.90 yes.
I submitted 8 stats: early-minus-late drift, near first vs near last example, 3-of-4 last-digit match, wide-span interior, single-example distance gradient, far extrapolation from 2-example sets, same-parity match, and target-is-example. Each takes about 0.02 s per call. Template: experiment3/model_loop/iter_0/critique/test_stats/. See [[critique-stat-no-imports]], [[exp2-iter4-critique]].
