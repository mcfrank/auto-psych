---
name: graded-top-edge-duplicate
description: exp2 iter_2 c3 graded top-edge bonus (gap 0-3, decay) rejected as near-duplicate of incumbent (RMSE 0.0015); repair grafted range-discount sharpness
metadata:
  type: project
---
2026-10-19, exp2 iter_2 candidate_3: `graded_upper_edge_marking_ranges` was rejected by novelty gate (p_yes RMSE 0.00152 < 0.002 vs upper_edge_marking_ranges) — nested extensions whose extra parameter can shrink to zero collapse onto the incumbent. Repair `upper_edge_discounted_range_size` = range_discounted_size_principle + top_bonus (k-resolved interval groups split by at_top).
**Why:** novelty gate threshold is tiny (0.002 RMSE); nested params that fit near null fail it.
**How to apply:** in refinement slots, graft a component known to move predictions (one that was itself admitted as distinct), not a nested generalization. See [[incumbent-near-ceiling-exp2]].

Repair check passed: smoke LL -2055.9 (best smoke seen in exp2 iter_2); 261-361 s per 100+100 sampling run, about 12 min for the whole check. Check its admission/LOO outcome before reusing the pattern.
