---
name: two-cluster-range-concepts
description: exp3 iter_1 c1: union-of-two-disjoint-ranges kind on beta_share learner; smoke ll -3653.9 (≈incumbent); size-only likelihood trick
metadata:
  type: project
---
exp3 iter_1 candidate_1 (2026-10-31), two_cluster_range_concepts: beta_share_width_exception_concepts plus a magnitude kind "two disjoint, non-adjacent ranges, each holding ≥1 example", share Beta(1,9) of magnitude mass, total size on the same beta-share prior / N(S)=(S-1)·C(101-S,2).
Trick: only concepts containing all examples are used, so likelihood depends only on total size S; prepare_observed enumerates (a,b,c,d) per gap split with np.add.at on a difference array (checked against brute force). Smoke OK in ~2 min, ll -3653.9 (incumbent ~-3653). Posterior of the share was not inspected.
Exp3 shown-example targets: the model already predicts 0.655 vs observed 0.658, so a repeat-avoidance idea has nothing to fix.
**How to apply:** if pruned or a duplicate, the gap-split sets (13,91; 43,4,91,9; 20,32,92) carry little signal. Related: [[graded-neighbourhood-magnitude]], [[concept-space-additions-map]].
