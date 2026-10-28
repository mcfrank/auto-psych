---
name: parity-representative-stray-example
description: exp3 iter_4 c5 repair: c5 was exact dup of c4 (same parity graft on incumbent); repair grafts parity penalty onto position_graded_stray_example
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_5 was rejected because it was an exact duplicate of c4 `parity_representative_sequence_edges` (RMSE 0). The repair `parity_representative_stray_example` copies the position_graded_stray_example source and adds the parity penalty per reading unit. Check OK: smoke LL -3093.2, 456 s and 372 s per smoke run (the base model is slow). parity_repeats is computed on the examples kept, so a stray of the other parity frees the rest to be parity-pure.
**Why:** Parallel refinement slots converge on the same graft again. **How to apply:** Graft onto a different live base than the one the other slot is likely to use. Check the outcome. Related: [[position-graded-stray-example]], [[parity-representative-sequence-edges]], [[novelty-gate-tiny-tweaks]]
