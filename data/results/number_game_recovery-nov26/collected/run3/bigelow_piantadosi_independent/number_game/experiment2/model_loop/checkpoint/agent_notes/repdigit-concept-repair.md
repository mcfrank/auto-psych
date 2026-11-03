---
name: repdigit-concept-repair
description: exp2 iter4 cand4 repair — fives bonus on prime_over_odd was rejected as a near-duplicate of fives_fluent_skip_counting (RMSE 0.00087); I switched to a repdigit-concept rule
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_4_repair_1 (`repdigit_pattern_chart_concepts`). The first attempt, a fives bonus on prime_over_odd_reading, was rejected because fives_fluent_skip_counting is already live with the same mechanism. The repair adds a repdigits rule (11..99): a 2-primitive cost plus a fitted bonus. It is appended as the last rule row and uses the grouped-interval implementation.
The novelty pool is in model_loop/novelty_pool.json (512 rows, 215 sets). Only 9 rows have ≥2 repdigit examples; 102 rows are in all-multiple-of-3 sets.
hidden_triples_concepts (a mult-3 cost on prime_over_odd) was dropped as too slow, not on its merits.

**Why:** before picking a refinement, check the live list for the same mechanism on the same base.
**How to apply:** grep attempted_hypotheses.jsonl and models/ first. Related: [[big-step-multiples-finding]], [[fives-coupling-and-speed]].
