---
name: digit-run-strict-criterion
description: iter_3 c3 repair — criterion shift when >=3 examples share a final digit, -16 nats MLE on graded-parity incumbent; smoke -1548.7
metadata:
  type: project
---
iter_3 candidate_3_repair_1 (2026-11-02), `digit_run_strict_criterion_exemplar`: replaces the rejected per-set random criterion ([[set-specific-criterion-candidate]]). That model failed on the novelty pool because its per-set index went out of bounds on unseen sets, and it could not generalize to them anyway. Added a criterion shift whenever >=3 examples all share one final digit. MLE NLL 1545.0 vs base 1561.3 (fit shift -1.28). Other criterion features: H 1553, S/PH 1558.4, all-share with any n 1561, modal count variants ~1560.5, digits shared at any position or tens digit as a similarity dim ~0. check OK, smoke LL -1548.7.
**Why:** a model of set-level variation must generalize to unseen sets, so per-set random effects are out. **How to apply:** see [[iter3-critique-stats]], [[divisibility-kinship-screen]].
