---
name: exp2-iter4-first-example-edge
description: Exp2 iter-4 cand_1 — order range-prior screens on margin incumbent all ≤0.35 nats; first-example-marks-edge margin (0 params) +0.3 MAP submitted
metadata:
  type: project
---
Exp2 iter 4, candidate_1 (2026-10-30), exploratory order slot. Margin-incumbent MAP NLL 1921.95.
Each screen adds one extra feature on top of the margin feature:
- next = last + step in range: 1921.94
- reach of the last step: 1921.70
- room on the last-shown side: 1921.76
- room on the first-shown side: 1921.60 (negative weight)

Submitted `first_example_marks_range_edge`: the margin counts no room past the first-shown end when that example is the set's min or max (only for sets of 2 or more). It adds 0 params, scores 1921.64, and its smoke LL is -1925.6.
Screener: experiment2/model_loop/iter_4/candidate_1/scratch/screen2.py (env M=0 drops the margin term).
**How to apply:** the exp2 order/range slot is saturated, worth ≤0.4 nats per idea. Novelty RMSE was not checked, so it may be a near-duplicate. See [[exp2-iter3-margin-beyond-examples]], [[exp2-iter2-heading-direction]].
