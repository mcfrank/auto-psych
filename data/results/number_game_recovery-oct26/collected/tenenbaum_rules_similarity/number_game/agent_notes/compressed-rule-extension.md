---
name: compressed-rule-extension
description: exp3 iter_1 c0: rule size^power in size principle (ranges true size); MAP power 0.976, +0.4 nats; parity critique is really opposite-parity-just-outside-span over-predicted
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_0 (exploratory): `compressed_rule_extension`, which is the incumbent last_example_marks_range_top with rule size N**rule_size_power (Beta(4,2)). At MAP power 0.976, LL -3090.1 vs -3090.5, p_yes RMSE 0.0085 vs the incumbent. Check OK, smoke LL -3094.5, 225 s per run.
Parity critique diagnosis (MAP p_yes, scratch/par2.py): same-parity targets are fit well. Opposite-parity targets just outside the span (<=3 away) are observed at 0.44 vs 0.52 predicted; inside the span they are 0.49 vs 0.45, mostly in n=2 sets. People extrapolate by parity but interpolate by magnitude.
**Why:** rule-size compression doesn't close this gap. **How to apply:** next try an interpolation-vs-extrapolation mechanism. Related: [[exp3-iter1-critique-stats]], [[write-candidate-first]]
