---
name: teacher-upper-edge-candidate
description: exp2 iter_3 cand_5 refinement: upper-edge top_bonus grafted onto pedagogical_teacher_sampling; smoke LL -2057.4, 86s/run
metadata:
  type: project
---
2026-10-19, exp2 iter_3 candidate_5 (refinement slot): `teacher_sampling_upper_edge_ranges` = pedagogical_teacher_sampling + top_bonus * range_at_top (set x hyp float32 mask built in prepare_observed, added to log_post). Check OK in 3.5 min, smoke LL -2057.4, 86 s per sampling run.

**Why:** Teacher sampling had no top-edge mechanism; the top-edge mechanism has already passed novelty ([[novelty-gate-tiny-tweaks]], [[compose-near-ties]]).
**How to apply:** Check the admission/LOO outcome before stacking top-edge onto more bases.
