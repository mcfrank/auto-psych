---
name: fives-duplicate-rejection
description: exp2 iter4 cand3 fives bonus rejected as exact duplicate of fives_fluent_skip_counting; repair used mult-3-only cost
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_3: fives bonus on prime_over_odd_reading was rejected (p_yes RMSE 0.00000) because `fives_fluent_skip_counting` already is that model. Repair `irregular_threes_times_table` puts a HalfNormal cost on "multiples of 3" only (row 16). That distinguishes it from hidden_divisibility_by_three (3/6/9/12, different base).

**Why:** the live set already holds models with "no comparison row". Check its sources before reusing an earlier screen's winner.
**How to apply:** grep models/ for the mechanism before you write it. Related: [[fives-coupling-and-speed]], [[hidden-triples-finding]].
