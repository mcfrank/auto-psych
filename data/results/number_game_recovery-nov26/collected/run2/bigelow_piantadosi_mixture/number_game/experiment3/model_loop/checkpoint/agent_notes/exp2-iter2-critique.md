---
name: exp2-iter2-critique
description: Exp2 iter-2 critique of digit_sum_concepts — MAP fit, residual z-scores, 8 raw-slice stats (no proxy) submitted
metadata:
  type: project
---
Exp2 iter 2 (2026-10-30): digit_sum_concepts MAP gives lambda 0.34, gamma ≈0, sum share 0.10, lapse→0, NLL 1928 (the iter-1 incumbent was 1930). The fit is at ceiling. The largest MAP-residual slices: out-of-range targets nearer the LAST-shown example z=+2.1 (recency); 1-5 below the min z=+1.5; single shared final digit z=-1.3; round targets z=-1.2; n=1 sets at distance 6-20 z=-0.95; p band 0.1-0.3 z=+1.3. Within-cell variance is only slightly above p(1-p) (0.142 vs 0.138). Digit-sum slices are tied.
Data: rows are ordered by participant (index//450). Participants 0-4 and 5-9 share a fixed trial order, so trial-index drift is confounded with the stimuli. This round I used raw slice yes-rates (no proxy needed). MAP script: experiment2/model_loop/iter_2/critique/scratch/fit.py, slicer sl.py. See [[exp2-iter1-critique]], [[critique-stat-no-imports]].
