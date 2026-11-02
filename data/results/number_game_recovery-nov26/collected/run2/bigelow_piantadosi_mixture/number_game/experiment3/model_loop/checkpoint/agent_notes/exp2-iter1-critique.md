---
name: exp2-iter1-critique
description: Exp2 iter-1 critique of narrow_range_specificity_bias — MAP params, residual hot spots, 8 MAP-proxy stats submitted
metadata:
  type: project
---
Exp2 iter 1 (2026-10-30): the incumbent's MAP is lambda 0.315, gamma -0.08, lapse→0, guess 0.64, NLL 1930. The iter-0 critique had 0/8 significant. Hot spots against MAP p: tight clusters (span ≤10) inside -0.045 / outside +0.029; targets 2-5 past the edge +0.033; below-min +0.024 vs above-max +0.002; final digit shared with 2-3 examples +0.04 vs one example -0.02; four-example ambiguous +0.022; within-cell variance above p(1-p) in the 0.1-0.3 and 0.7-0.9 bands.
The stats embed an exact MAP proxy of the incumbent (r=0.99999), template at experiment2/model_loop/iter_1/critique/scratch/proxy.txt; the MAP fit script is scratch/fit.py. See [[exp2-iter0-critique]], [[critique-stat-no-imports]].
