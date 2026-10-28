---
name: examples-bracket-range-ends
description: exp3 iter_2 c2 (retry): order-free both-ends generative credit on exact-span range; parity misfit is n=2 same-parity sets; check outcome
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_2. The first attempt left no files. The retry is `examples_bracket_range_ends`: compressed_rule_extension plus bracket_odds. The range [min,max] gets log1p(top_odds*top_share/p + bracket_odds/p^2) as a third interval partition, for sets with 2+ distinct values.
Diagnosis (iter_2/candidate_2/scratch/par3.py, inside.py, using iter_1 p_inc.npy):
- The misfit is in same-parity sets: opposite-parity neighbours 1-3 outside are over-predicted (maxearly-above 0.52 vs 0.65; below 0.37 vs 0.43). Mixed-parity sets are fit well there.
- Interior opposite parity in n=2 sets is under-predicted (0.455 vs 0.35, 145 obs).
- Min-first below rates have small N.
Check OK: smoke LL -3094.3, about 272 s per sampling run (slow, at risk for admission).
**How to apply:** check the outcome and the bracket_odds posterior. Related: [[compressed-rule-extension]], [[parity-restricted-range-concepts]]
