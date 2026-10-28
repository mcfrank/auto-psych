---
name: position-graded-stray-example
description: exp3 iter_3 c4 repair: digit-pair c4 was exact dup of c3 (RMSE 0); replaced by any-position stray example with recency slope (5 readings)
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_4 (`paired_final_digit_rules`) was rejected as an exact duplicate of c3 `final_digit_pair_concepts` (RMSE 0.00000), because parallel slots chose the same idea. The repair `position_graded_stray_example` generalizes stray_closing_example: one example of n>=2 may be a stray, and its position weight is softmax(recency_slope * k/(n-1)). It uses 5 readings (full + drop each k) stacked on axis 1.
**Why:** Parallel slots in one round converge on the critique's obvious fix. **How to apply:** In a refinement slot, avoid the critique's most obvious mechanism, since another slot probably took it. Related: [[stray-closing-example]], [[final-digit-pair-concepts]], [[novelty-gate-tiny-tweaks]]
