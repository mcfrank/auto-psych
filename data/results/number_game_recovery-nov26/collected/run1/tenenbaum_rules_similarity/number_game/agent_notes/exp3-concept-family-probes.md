---
name: exp3-concept-family-probes
description: exp3 (iter_0 cand_1, 2026-11-01) concept-family probes on centre_weighted_range_sampling; only digit sums gain, graded digit-sum prior +3 nats
metadata:
  type: project
---
Experiment 3 data: 6750 rows, 154 sets; many sets look designed around near-misses (17,50,5,82 = n^2+1; 8,24,48,63 = n^2-1; 33,17,5,9 = 2^k+1; 31,3,1,15 = 2^k-1), but few targets test those rules.
MAP LL probes, each family added alone to centre_weighted_range_sampling (MAP -3516.53, centre_weight 0.02, scale 73):
- near-miss "squares/cubes/powers of 2,3 plus or minus 1": weight 0 (dead).
- offset arithmetic progressions (step 2-12, all offsets): weight 0.
- "ends in d, or x" exception unions: +0.18, RMSE 0.0025.
- digit-structure (all-odd/all-even digits, ascending/descending, repdigits): weight 0.
- digit sum = s (1..18): -3515.03 (+1.5), RMSE 0.009; gain spread over many sets (89, 43,4,91,9; 17,50,5,82 gets worse).
- digit sums with prior over s ∝ exp(-decay*s): -3513.54 (+3.0), decay -0.25 (large sums favoured). Submitted as graded_prior_digit_sum_concepts.
Probe scripts: /tmp/c1e3 (probe.py = MAP LL + RMSE of two model files, perset.py = per-set LL gain); may not persist.
**How to apply:** concept space is still saturated on exp3 apart from digit sums; see [[exp2-map-probe-results]], [[exp2-iter3-saturation-probes]].
