---
name: exp2-iter2-open-range-graft
description: Exp2 iter-2 cand_3 — grafting open-ended range discount onto digit_sum_concepts gives +2.2 nats MAP (best single graft); submitted
metadata:
  type: project
---
Exp2 iter 2, candidate_3 (2026-10-30), refining digit_sum_concepts (MAP NLL 1925.26). One-parameter range-prior grafts, MAP NLL: open-ended (lo==1 or hi==100) 1923.10 (log_open≈-1.09); lo==1 only 1923.3; hi==100 only 1925.2; lo-location 1924.1; relative width 1924.07. Submitted `digit_sum_open_ended_range_discount`: groups intervals by (width, open flag), 199 groups. Screener: experiment2/model_loop/iter_2/candidate_3/scratch/screen.py (dense interval-level, ~1 min per model).
**How to apply:** the open-ended discount mostly comes from the bottom end (lo==1). See [[exp2-iter1-open-ended-range]], [[exp2-iter1-digit-sum-concepts]].
Speed fix: storing the within-group membership only for the observed (set, target) cells (895 cells instead of 203 sets × 100 targets), as (cells, groups) and gathered with cell_set, cut the smoke fit from 55 s to 9 s per 200 iterations. The smoke loglik was -1926.6 and the check said OK. Apply this to any dense (sets, groups, 100) model. See [[fit-speed-budget]].
