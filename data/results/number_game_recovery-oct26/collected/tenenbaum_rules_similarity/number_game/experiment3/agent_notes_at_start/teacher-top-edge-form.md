---
name: teacher-top-edge-form
description: exp2 iter_3 c3 refinement: top-edge marking recast as a teacher-shows-top mixture (Beta rho), so the bonus grows with range width
metadata:
  type: project
---
Exp2 iter_3 candidate_3 (2026-10-19), `top_example_teacher_ranges`: a copy of upper_edge_marking_ranges with top_bonus replaced by top_shown ~ Beta(1,1). The likelihood is a mixture: random sampling over all intervals, plus a "teacher" row for at-top intervals with the max example's sampling factor dropped, which works out to a bonus of about log(width). The cost is the same as the incumbent (still 2 x N_GROUPS columns). It was chosen to avoid the novelty-gate collapse that nested tweaks hit ([[novelty-gate-tiny-tweaks]]).
Check passed: smoke LL -2056.7, 156-171 s per sampling run.
**Why:** a structural change of form, not a nested parameter. **How to apply:** check its admission/LOO outcome before reusing it.
